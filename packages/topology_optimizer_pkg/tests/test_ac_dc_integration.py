# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

import logbook
import pytest
import ray
from confluent_kafka import Consumer, Producer
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.ac.worker import Args as ACArgs
from toop_engine_topology_optimizer.ac.worker import main as ac_main
from toop_engine_topology_optimizer.dc.worker.worker import Args as DCArgs
from toop_engine_topology_optimizer.dc.worker.worker import main as dc_main
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, ShutdownCommand, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import OptimizationStoppedResult, Result, TopologyPushResult

logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logging.INFO).push_application()
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


def dc_main_wrapper(args: DCArgs, processed_gridfile_fs: AbstractFileSystem) -> None:
    instance_id = str(uuid4())
    command_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_command_topic,
        group_id="dc_optimizer",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        }
    )

    dc_main(args, processed_gridfile_fs, producer, command_consumer)


def ac_main_wrapper(
    args: ACArgs,
    processed_gridfile_fs: AbstractFileSystem,
    loadflow_result_fs: AbstractFileSystem,
) -> None:
    instance_id = str(uuid4())
    command_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_command_topic,
        group_id="ac_optimizer",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    result_consumer = LongRunningKafkaConsumer(
        topic=args.optimizer_results_topic,
        group_id="ac_optimizer_results",
        bootstrap_servers=args.kafka_broker,
        client_id=instance_id,
    )
    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        }
    )
    ac_main(args, loadflow_result_fs, processed_gridfile_fs, producer, command_consumer, result_consumer)


@ray.remote
def launch_dc_worker(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_fs: AbstractFileSystem,
):
    logging.basicConfig(level=logging.INFO)
    try:
        dc_main_wrapper(
            DCArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                instance_id="dc_worker",
            ),
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except SystemExit:
        # This is expected when the worker receives a shutdown command
        logger.info("DC worker stopped")
        pass


@ray.remote
def launch_ac_worker(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_fs: AbstractFileSystem,
    loadflow_result_fs: AbstractFileSystem,
):
    logging.basicConfig(level=logging.INFO)
    print("Starting AC worker")
    try:
        ac_main_wrapper(
            ACArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                instance_id="ac_worker",
            ),
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except SystemExit:
        # This is expected when the worker receives a shutdown command
        logger.info("AC worker stopped")
        pass


# TODO: set to 200, once the xdist_group is run on a dedicated runner
@pytest.mark.skip(reason="This test is currently flaky, should be fixed and re-enabled")
@pytest.mark.timeout(400)
def test_ac_dc_integration(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    grid_folder: Path,
    loadflow_result_folder: Path,
) -> None:
    # Start the ray runtime
    ray.init(num_cpus=4)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    try:
        ac_future = launch_ac_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
        )
        dc_future = launch_dc_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            processed_gridfile_fs=processed_gridfile_fs,
        )

        grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
        ac_parameters = ACOptimizerParameters(
            ga_config=ACGAParameters(
                runtime_seconds=50,
                pull_prob=1.0,
                reconnect_prob=0.0,
                close_coupler_prob=0.0,
                seed=42,
                enable_ac_rejection=False,
            )
        )
        dc_parameters = DCOptimizerParameters(
            ga_config=BatchedMEParameters(iterations_per_epoch=2, runtime_seconds=30),
            loadflow_solver_config=LoadflowSolverParameters(
                batch_size=16,
            ),
        )
        start_command = Command(
            command=StartOptimizationCommand(
                ac_params=ac_parameters,
                dc_params=dc_parameters,
                grid_files=grid_files,
                optimization_id="test",
            )
        )

        producer = Producer({"bootstrap.servers": kafka_connection_str, "log_level": 2})
        producer.produce(kafka_command_topic, value=serialize_message(start_command.model_dump_json()))
        producer.flush()

        # This is the runtime of the AC worker
        time.sleep(50)

        consumer = Consumer(
            {
                "bootstrap.servers": kafka_connection_str,
                "group.id": "integration_test",
                "auto.offset.reset": "earliest",
                "log_level": 2,
            }
        )
        consumer.subscribe([kafka_results_topic])

        ac_converged = False
        dc_converged = False
        ac_topo_push = False
        dc_topo_push = False
        split_topo_push = False

        result_history = []
        while message := consumer.poll(timeout=10.0):
            result = Result.model_validate_json(deserialize_message(message.value()))
            result_history.append(result)
            if isinstance(result.result, OptimizationStoppedResult):
                assert result.result.reason == "converged", f"{result}"
                if result.optimizer_type == OptimizerType.AC:
                    ac_converged = True
                elif result.optimizer_type == OptimizerType.DC:
                    dc_converged = True
            elif isinstance(result.result, TopologyPushResult):
                if result.optimizer_type == OptimizerType.AC:
                    ac_topo_push = True
                elif result.optimizer_type == OptimizerType.DC:
                    dc_topo_push = True
                for strategy in result.result.strategies:
                    if len(strategy.timesteps[0].actions):
                        split_topo_push = True
                        break

            if ac_converged and dc_converged:
                break

        logger.info(f"{[type(result.result) for result in result_history]}")
        assert result_history
        assert ac_converged
        assert dc_converged
        assert dc_topo_push
        assert split_topo_push
        assert ac_topo_push

        shutdown_command = Command(command=ShutdownCommand())
        producer.produce(kafka_command_topic, value=serialize_message(shutdown_command.model_dump_json()))
        producer.flush()

        # Give everyone a chance to shutdown
        ray.get(ac_future)
        ray.get(dc_future)

    finally:
        ray.shutdown()
