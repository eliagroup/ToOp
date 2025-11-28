import logging
import sys
import time
from pathlib import Path

import logbook
import pytest
import ray
from confluent_kafka import Consumer, Producer
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


@ray.remote
def launch_dc_worker(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    grid_folder: Path,
):
    logging.basicConfig(level=logging.INFO)
    try:
        dc_main(
            DCArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                processed_gridfile_folder=grid_folder,
                instance_id="dc_worker",
            )
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
    grid_folder: Path,
    loadflow_result_folder: Path,
):
    logging.basicConfig(level=logging.INFO)
    print("Starting AC worker")
    try:
        ac_main(
            ACArgs(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                processed_gridfile_folder=grid_folder,
                instance_id="ac_worker",
                loadflow_result_folder=loadflow_result_folder,
            )
        )
    except SystemExit:
        # This is expected when the worker receives a shutdown command
        logger.info("AC worker stopped")
        pass


@pytest.mark.timeout(200)
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

    try:
        ac_future = launch_ac_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            grid_folder=grid_folder,
            loadflow_result_folder=loadflow_result_folder,
        )
        dc_future = launch_dc_worker.remote(
            kafka_command_topic=kafka_command_topic,
            kafka_heartbeat_topic=kafka_heartbeat_topic,
            kafka_results_topic=kafka_results_topic,
            kafka_connection_str=kafka_connection_str,
            grid_folder=grid_folder,
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
