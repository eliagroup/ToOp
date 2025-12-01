"""Kafka worker for the genetic algorithm optimization."""

import time
from functools import partial
from pathlib import Path
from typing import Callable
from uuid import uuid4

import jax
import logbook
from confluent_kafka import Producer
from pydantic import BaseModel
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.dc.worker.optimizer import (
    OptimizerData,
    extract_results,
    initialize_optimization,
    run_epoch,
)
from toop_engine_topology_optimizer.interfaces.messages.commands import (
    Command,
    ShutdownCommand,
    StartOptimizationCommand,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import (
    Heartbeat,
    HeartbeatUnion,
    IdleHeartbeat,
    OptimizationStartedHeartbeat,
    OptimizationStatsHeartbeat,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
)

logger = logbook.Logger(__name__)


class Args(BaseModel):
    """Launch arguments for the worker, which can not be changed during the optimization run."""

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

    processed_gridfile_folder: Path = Path("gridfiles")
    """The parent folder where all the grid files are stored. In production this is a shared network
    filesystem between the backend and all workers. When a command is received, this will be pre-
    fixed to the static information file"""


def idle_loop(
    consumer: LongRunningKafkaConsumer,
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    heartbeat_interval_ms: int,
    parent_grid_folder: Path,
) -> StartOptimizationCommand:
    """Run idle loop of the worker.

    This will be running when the worker is currently not optimizing
    This will wait until a StartOptimizationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    consumer : LongRunningKafkaConsumer
        The initialized Kafka consumer to listen for commands on.
    send_heartbeat_fn : callable
        A function to call when there were no messages received for a while.
    heartbeat_interval_ms : int
        The time to wait for a new command in milliseconds. If no command has been received, a
        heartbeat will be sent and then the receiver will wait for commands again.
    parent_grid_folder : Path
        The folder where all the grid files are stored. This will be prefixed to the static
        information files in the StartOptimizationCommand.

    Returns
    -------
    StartOptimizationCommand
        The start optimization command to start the optimization run with

    Raises
    ------
    SystemExit
        If a ShutdownCommand is received
    """
    send_heartbeat_fn(IdleHeartbeat())
    logger.info("Entering idle loop")
    while True:
        message = consumer.poll(timeout=heartbeat_interval_ms / 1000)

        # Wait timeout exceeded
        if not message:
            send_heartbeat_fn(IdleHeartbeat())
            continue

        command = Command.model_validate_json(deserialize_message(message.value()))

        if isinstance(command.command, StartOptimizationCommand):
            # Prefix the gridfile folder to the static information files
            command.command.grid_files = [
                gf.model_copy(update={"grid_folder": str(parent_grid_folder / gf.grid_folder)})
                for gf in command.command.grid_files
            ]
            return command.command

        if isinstance(command.command, ShutdownCommand):
            logger.info("Shutting down due to ShutdownCommand")
            consumer.commit()
            consumer.consumer.close()
            raise SystemExit(command.command.exit_code)

        # If we are here, we received a command that we do not know
        logger.warning(f"Received unknown command, dropping: {command} / {message.value}")
        consumer.commit()


def optimization_loop(
    dc_params: DCOptimizerParameters,
    grid_files: list[GridFile],
    send_result_fn: Callable[[ResultUnion], None],
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    optimization_id: str,
) -> None:
    """Run an optimization until the optimization has converged

    Parameters
    ----------
    dc_params : DCOptimizerParameters
        The parameters for the optimization run, usually from the start command
    grid_files : list[GridFile]
        The grid files to load, where each gridfile represents one timestep.
    send_result_fn : Callable[[ResultUnion], None]
        A function to call to send results back to the results topic.
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        A function to call after every epoch to signal that the worker is still alive.
    optimization_id : str
        The id of the optimization run. This will be used to identify the optimization run in the
        results. Should stay the same for the whole optimization run and should be equal to the kafka
        event key.

    Raises
    ------
    SystemExit
        If a ShutdownCommand is received
    """
    logger.info(f"Initializing DC optimization {optimization_id}")
    try:
        send_heartbeat_fn(
            OptimizationStartedHeartbeat(
                optimization_id=optimization_id,
            )
        )
        optimizer_data, stats, initial_strategy = initialize_optimization(
            params=dc_params,
            optimization_id=optimization_id,
            static_information_files=[gf.static_information_file for gf in grid_files],
        )
        send_result_fn(
            OptimizationStartedResult(
                initial_topology=initial_strategy,
                initial_stats=stats,
            )
        )

    except Exception as e:
        send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))
        logger.error(f"Error during initialization of optimization {optimization_id}: {e}")
        return

    def push_topologies(optimizer_data: OptimizerData) -> None:
        """Push topologies to the results topic."""
        with jax.default_device(jax.devices("cpu")[0]):
            push_result = extract_results(optimizer_data)
            if len(push_result.strategies):
                send_result_fn(push_result)
                best_fitness = max(
                    timestep.metrics.fitness for strategy in push_result.strategies for timestep in strategy.timesteps
                )
                logger.info(
                    f"Sent {len(push_result.strategies)} strategies with best fitness {best_fitness} to results topic"
                )
            else:
                logger.warning("No strategies extracted, skipping push.")

    logger.info(f"Starting optimization {optimization_id}")
    epoch = 0
    running = True
    start_time = time.time()
    while running:
        try:
            optimizer_data = run_epoch(optimizer_data)
            push_topologies(optimizer_data)
        except Exception as e:
            # Send a stop message to the results
            send_result_fn(OptimizationStoppedResult(reason="error", message=str(e)))

            logger.error(f"Error during optimization {optimization_id}, epoch {epoch}: {e}")
            return
        epoch += 1

        if time.time() - start_time > dc_params.ga_config.runtime_seconds:
            logger.info(f"Stopping optimization {optimization_id} at epoch {epoch} due to runtime limit")
            send_result_fn(OptimizationStoppedResult(epoch=epoch, reason="converged", message="runtime limit"))
            running = False
            break

        send_heartbeat_fn(
            OptimizationStatsHeartbeat(
                optimization_id=optimization_id,
                wall_time=time.time() - start_time,
                iteration=epoch,
                num_branch_topologies_tried=optimizer_data.jax_data.emitter_state["total_branch_combis"].sum().item(),
                num_injection_topologies_tried=optimizer_data.jax_data.emitter_state["total_inj_combis"].sum().item(),
            )
        )


def main(
    args: Args,
    producer: Producer,
    command_consumer: LongRunningKafkaConsumer,
) -> None:
    """Start the worker and run the optimization loop."""
    instance_id = str(uuid4())
    logger.info(f"Starting DC worker {instance_id} with config {args}")
    if not args.processed_gridfile_folder.exists():
        raise FileNotFoundError(f"Processed gridfile folder {args.processed_gridfile_folder} does not exist. ")
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
        producer.flush()

    while True:
        command = idle_loop(
            consumer=command_consumer,
            send_heartbeat_fn=partial(send_heartbeat, ping_consumer=False),
            heartbeat_interval_ms=args.heartbeat_interval_ms,
            parent_grid_folder=args.processed_gridfile_folder,
        )
        command_consumer.start_processing()
        optimization_loop(
            dc_params=command.dc_params,
            grid_files=command.grid_files,
            send_result_fn=partial(send_result, optimization_id=command.optimization_id),
            send_heartbeat_fn=partial(send_heartbeat, ping_consumer=True),
            optimization_id=command.optimization_id,
        )
        command_consumer.stop_processing()
