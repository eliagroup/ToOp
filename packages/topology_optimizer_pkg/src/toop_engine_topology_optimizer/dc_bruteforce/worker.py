# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Standalone Kafka worker for DC bruteforce optimization."""

import time
from functools import partial
from uuid import uuid4

import jax
import structlog
from beartype.typing import Callable
from confluent_kafka import Producer
from fsspec import AbstractFileSystem
from pydantic import BaseModel
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import serialize_message
from toop_engine_topology_optimizer.dc.worker.idle_loop import idle_loop
from toop_engine_topology_optimizer.dc_bruteforce.optimizer import (
    OptimizerData,
    convert_topologies_to_messages,
    extract_topologies,
    get_num_branch_topologies_tried,
    initialize_optimization,
    is_exhausted,
    run_epoch,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import (
    Heartbeat,
    HeartbeatUnion,
    OptimizationStartedHeartbeat,
    OptimizationStatsHeartbeat,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
)

logger = structlog.get_logger(__name__)


class Args(BaseModel):
    """Launch arguments for the standalone bruteforce worker."""

    kafka_broker: str = "localhost:9092"
    """The Kafka broker to connect to."""

    optimizer_command_topic: str = "commands"
    """The Kafka topic to listen for commands on."""

    optimizer_results_topic: str = "results"
    """The topic to push results to."""

    optimizer_heartbeat_topic: str = "heartbeat"
    """The topic to push heartbeats to."""

    heartbeat_interval_ms: int = 1000
    """The interval in milliseconds to send heartbeats."""

    max_command_age_hours: float = 3.0
    """The maximum age of a command in hours before it is ignored."""


def push_topologies(optimizer_data: OptimizerData, epoch: int, send_result_fn: Callable[[ResultUnion], None]) -> int:
    """Push new bruteforce topologies to Kafka.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer state containing the latest improved topologies to emit.
    epoch : int
        The current epoch, used in the emitted result messages.
    send_result_fn : Callable[[ResultUnion], None]
        A function to call to queue results for the Kafka results topic.

    Returns
    -------
    int
        The number of topology messages queued for emission.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        push_results = convert_topologies_to_messages(extract_topologies(optimizer_data), epoch)
        for push_result in push_results:
            send_result_fn(push_result)
        return len(push_results)


def optimization_loop(
    dc_params: DCOptimizerParameters,
    grid_files: list[GridFile],
    send_result_fn: Callable[[ResultUnion], None],
    flush_result_fn: Callable[[], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
    processed_gridfile_fs: AbstractFileSystem,
) -> None:
    """Run the standalone bruteforce optimization loop.

    Parameters
    ----------
    dc_params : DCOptimizerParameters
        Parameters controlling the bruteforce optimization run.
    grid_files : list[GridFile]
        Grid files to load, where each file represents one timestep.
    send_result_fn : Callable[[ResultUnion], None]
        A function to queue results for the results topic. This callback is not expected to flush and may be called
        multiple times within a single epoch.
    flush_result_fn : Callable[[], None]
        A function to flush queued results to Kafka after one or more calls to ``send_result_fn``.
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        A function to call after every epoch to signal that the worker is still alive.
    optimization_id : str
        Identifier of the optimization run. This value is propagated into emitted results.
    processed_gridfile_fs : AbstractFileSystem
        Filesystem containing the preprocessed grid data for all timesteps.
    """
    logger.info(f"Initializing DC bruteforce optimization {optimization_id}")

    try:
        send_heartbeat_fn(OptimizationStartedHeartbeat(optimization_id=optimization_id))
        optimizer_data, stats, initial_strategy = initialize_optimization(
            params=dc_params,
            optimization_id=optimization_id,
            static_information_files=tuple(gf.static_information_file for gf in grid_files),
            processed_gridfile_fs=processed_gridfile_fs,
        )
        send_result_fn(OptimizationStartedResult(initial_topology=initial_strategy, initial_stats=stats))
        flush_result_fn()
    except Exception as exc:
        send_result_fn(OptimizationStoppedResult(reason="error", message=str(exc)))
        flush_result_fn()
        logger.error(f"Error during bruteforce initialization {optimization_id}: {exc}")
        return

    epoch = 1
    start_time = time.time()
    while True:
        try:
            optimizer_data = run_epoch(optimizer_data)
            n_pushes = push_topologies(optimizer_data, epoch, send_result_fn)
            if n_pushes > 0:
                flush_result_fn()
            logger.info(
                f"Sent {n_pushes} bruteforce strategies to results topic,"
                f" best fitness: {optimizer_data.runtime_state.best_fitness}, epoch: {epoch}"
                f" progress: {get_num_branch_topologies_tried(optimizer_data)} "
                f"/ {optimizer_data.runtime_state.total_workset_size}"
            )
        except Exception as exc:
            send_result_fn(OptimizationStoppedResult(reason="error", message=str(exc)))
            flush_result_fn()
            logger.error(f"Error during bruteforce optimization {optimization_id}, epoch {epoch}: {exc}")
            return

        epoch += 1
        send_heartbeat_fn(
            OptimizationStatsHeartbeat(
                optimization_id=optimization_id,
                wall_time=time.time() - start_time,
                iteration=epoch,
                num_branch_topologies_tried=get_num_branch_topologies_tried(optimizer_data),
                num_injection_topologies_tried=get_num_branch_topologies_tried(optimizer_data),
            )
        )

        if is_exhausted(optimizer_data):
            send_result_fn(OptimizationStoppedResult(epoch=epoch, reason="converged", message="workset exhausted"))
            flush_result_fn()
            return

        if time.time() - start_time > dc_params.ga_config.runtime_seconds:
            send_result_fn(OptimizationStoppedResult(epoch=epoch, reason="converged", message="runtime limit"))
            flush_result_fn()
            return


def main(
    args: Args,
    processed_gridfile_fs: AbstractFileSystem,
    producer: Producer,
    command_consumer: LongRunningKafkaConsumer,
) -> None:
    """Run the standalone bruteforce worker loop.

    Parameters
    ----------
    args : Args
        Worker launch arguments.
    processed_gridfile_fs : AbstractFileSystem
        Filesystem containing the preprocessed grid data.
    producer : Producer
        Kafka producer used to send heartbeats and results.
    command_consumer : LongRunningKafkaConsumer
        Kafka consumer used to receive optimization commands.
    """
    instance_id = str(uuid4())
    logger.info(f"Starting DC bruteforce worker {instance_id} with config {args}")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_logging_level", "INFO")

    def send_heartbeat(message: HeartbeatUnion, ping_consumer: bool) -> None:
        heartbeat = Heartbeat(
            optimizer_type=OptimizerType.DC,
            instance_id=instance_id,
            message=message,
        )
        producer.produce(
            args.optimizer_heartbeat_topic,
            value=serialize_message(heartbeat.model_dump_json()),
            key=heartbeat.instance_id.encode(),
        )
        producer.flush()
        if ping_consumer:
            command_consumer.heartbeat()

    def send_result(message: ResultUnion, optimization_id: str) -> None:
        result = Result(
            result=message,
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.DC,
            instance_id=instance_id,
        )
        producer.produce(
            args.optimizer_results_topic,
            value=serialize_message(result.model_dump_json()),
            key=optimization_id.encode(),
        )

    def send_result_and_flush(message: ResultUnion, optimization_id: str) -> None:
        send_result(message=message, optimization_id=optimization_id)
        producer.flush()

    def flush_results() -> None:
        producer.flush()

    while True:
        command = idle_loop(
            consumer=command_consumer,
            send_heartbeat_fn=partial(send_heartbeat, ping_consumer=False),
            send_result_fn=send_result_and_flush,
            heartbeat_interval_ms=args.heartbeat_interval_ms,
            max_command_age_hours=args.max_command_age_hours,
        )
        command_consumer.start_processing()
        optimization_loop(
            dc_params=command.dc_params,
            grid_files=command.grid_files,
            send_result_fn=partial(send_result, optimization_id=command.optimization_id),
            flush_result_fn=flush_results,
            send_heartbeat_fn=partial(send_heartbeat, ping_consumer=True),
            optimization_id=command.optimization_id,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        command_consumer.stop_processing()
