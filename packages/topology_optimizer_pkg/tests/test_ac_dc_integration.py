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
from toop_engine_dc_solver.example_grids import three_node_pst_example_folder_powsybl
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.ac.worker import Args as ACArgs
from toop_engine_topology_optimizer.ac.worker import main as ac_main
from toop_engine_topology_optimizer.dc.worker.worker import Args as DCArgs
from toop_engine_topology_optimizer.dc.worker.worker import main as dc_main
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, ShutdownCommand, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef, Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    TopologyPushResult,
)

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


class FakeProducer:
    def __init__(self):
        self.messages = {}

    def produce(self, topic: str, value: bytes, key: str | None = None):
        if topic not in self.messages:
            self.messages[topic] = []
        self.messages[topic].append(value)

    def flush(self):
        pass


class FakeConsumerEmptyException(Exception):
    pass


class FakeMessage:
    """Mock Kafka Message for testing"""

    def __init__(self, value_bytes: bytes):
        self._value = value_bytes

    def value(self) -> bytes:
        return self._value


class FakeConsumer:
    def __init__(self, messages: dict[str, list[bytes]], kill_on_empty: bool = False):
        self.messages = messages
        self.offsets = {topic: 0 for topic in messages}
        self.kill_on_empty = kill_on_empty

    def _check_empty(self):
        if not self.kill_on_empty:
            return
        for topic, msgs in self.messages.items():
            offset = self.offsets[topic]
            if offset < len(msgs):
                return
        raise FakeConsumerEmptyException("No more messages to consume")

    def consume(self, timeout: float | int, num_messages: int) -> list[FakeMessage]:
        consumed_messages: list[FakeMessage] = []
        self._check_empty()
        for topic, msgs in self.messages.items():
            if len(consumed_messages) >= num_messages:
                break
            offset = self.offsets[topic]
            while offset < len(msgs) and len(consumed_messages) < num_messages:
                msg = FakeMessage(msgs[offset])
                consumed_messages.append(msg)
                offset += 1
            self.offsets[topic] = offset
            if len(consumed_messages) >= num_messages:
                break
        return consumed_messages

    def poll(self, timeout: float | int) -> FakeMessage | None:
        self._check_empty()
        for topic, msgs in self.messages.items():
            offset = self.offsets[topic]
            if offset < len(msgs):
                msg = FakeMessage(msgs[offset])
                self.offsets[topic] += 1
                return msg
        return None

    def commit(self):
        pass

    def start_processing(self):
        pass

    def heartbeat(self):
        pass

    def stop_processing(self):
        pass

    def close(self):
        pass


@pytest.mark.timeout(100)
def test_ac_dc_integration_sequential(grid_folder: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    ac_parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=20,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            enable_ac_rejection=False,
        )
    )
    dc_parameters = DCOptimizerParameters(
        ga_config=BatchedMEParameters(iterations_per_epoch=2, runtime_seconds=20),
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

    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )
    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        dc_main(
            DCArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0
    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"

    result_consumer = FakeConsumer(
        messages={
            "results": producer.messages["results"],
        }
    )
    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )

    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        ac_main(
            ACArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            loadflow_result_fs=DirFileSystem(str(tmp_path_factory.mktemp("loadflow_results"))),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
            result_consumer=result_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0

    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"


@pytest.mark.timeout(100)
def test_ac_dc_integration_psts(tmp_path_factory: pytest.TempPathFactory) -> None:
    grid_folder = tmp_path_factory.mktemp("grid_folder")
    (grid_folder / "threenode").mkdir()
    three_node_pst_example_folder_powsybl(grid_folder / "threenode")
    load_grid(
        data_folder_dirfs=DirFileSystem(str(grid_folder / "threenode")),
        parameters=PreprocessParameters(),
    )

    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="threenode")]
    ac_parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=20,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            enable_ac_rejection=False,
        )
    )
    dc_parameters = DCOptimizerParameters(
        ga_config=BatchedMEParameters(
            iterations_per_epoch=10,
            runtime_seconds=20,
            substation_split_prob=0,
            n_worst_contingencies=2,
            pst_mutation_sigma=3.0,
            target_metrics=(("overload_energy_n_1", 1.0),),
            observed_metrics=("overload_energy_n_1", "split_subs"),
            me_descriptors=(DescriptorDef(metric="split_subs", num_cells=2),),
        ),
        loadflow_solver_config=LoadflowSolverParameters(
            batch_size=16,
            max_num_splits=1,
            max_num_disconnections=0,
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

    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )
    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        dc_main(
            DCArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0
    # First one should be a OptimizationStartedResult
    # And we should have overloads in the grid
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)
    assert first_msg.result.initial_topology.timesteps[0].pst_setpoints is None
    assert first_msg.result.initial_topology.timesteps[0].metrics.fitness < 0

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0
    assert second_msg.result.strategies[0].timesteps[0].pst_setpoints is not None

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"

    result_consumer = FakeConsumer(
        messages={
            "results": producer.messages["results"],
        }
    )
    command_consumer = FakeConsumer(
        messages={"commands": [serialize_message(start_command.model_dump_json())]}, kill_on_empty=True
    )

    producer = FakeProducer()

    with pytest.raises(FakeConsumerEmptyException):
        ac_main(
            ACArgs(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
            ),
            loadflow_result_fs=DirFileSystem(str(tmp_path_factory.mktemp("loadflow_results"))),
            processed_gridfile_fs=DirFileSystem(str(grid_folder)),
            producer=producer,
            command_consumer=command_consumer,
            result_consumer=result_consumer,
        )

    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0

    # First one should be a OptimizationStartedResult
    first_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][0]))
    assert isinstance(first_msg, Result)
    assert isinstance(first_msg.result, OptimizationStartedResult)

    # There should be at least one TopologyPushResult before the OptimizationStoppedResult
    second_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][1]))
    assert isinstance(second_msg, Result)
    assert isinstance(second_msg.result, TopologyPushResult)
    assert len(second_msg.result.strategies) > 0
    assert len(second_msg.result.strategies[0].timesteps) > 0

    last_msg = Result.model_validate_json(deserialize_message(producer.messages["results"][-1]))
    assert isinstance(last_msg, Result)
    assert isinstance(last_msg.result, OptimizationStoppedResult)
    assert last_msg.result.reason == "converged"
