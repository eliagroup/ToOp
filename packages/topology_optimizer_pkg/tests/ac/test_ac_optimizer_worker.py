# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import sys
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
from beartype.typing import Callable
from confluent_kafka import Producer
from fsspec.implementations.dirfs import DirFileSystem
from sqlmodel import select
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_dc_solver.preprocess.network_data import load_network_data
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_interfaces.stored_action_set import load_action_set, random_actions
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, create_session
from toop_engine_topology_optimizer.ac.worker import Args, WorkerData, idle_loop, main, optimization_loop
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import (
    Command,
    ShutdownCommand,
    StartOptimizationCommand,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import HeartbeatUnion, OptimizationStartedHeartbeat
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
    Strategy,
    Topology,
    TopologyPushResult,
)

# Add parent directory to path to import fake_kafka
sys.path.insert(0, str(Path(__file__).parent.parent))
from fake_kafka import FakeConsumer, FakeConsumerEmptyException, FakeProducer

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


@pytest.mark.timeout(60)
def test_main_simple(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_folder: Path,
    loadflow_result_folder: Path,
    create_consumer: Callable,
    create_producer: Callable,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()
    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(processed_gridfile_folder))
    instance_id = str(uuid4())
    with pytest.raises(SystemExit):
        main(
            args=Args(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
            ),
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
            producer=create_producer(kafka_connection_str, instance_id, log_level=2),
            command_consumer=create_consumer(
                "LongRunningKafkaConsumer",
                topic=kafka_command_topic,
                group_id="ac_optimizer",
                bootstrap_servers=kafka_connection_str,
                client_id=instance_id,
            ),
            result_consumer=create_consumer(
                "LongRunningKafkaConsumer",
                topic=kafka_results_topic,
                group_id=f"ac_listener_{instance_id}_{uuid4()}",
                bootstrap_servers=kafka_connection_str,
                client_id=instance_id,
            ),
        )


@pytest.fixture
def topopushresult(grid_folder: Path, contingency_ids_case_57: list[str]) -> Result:
    # Generate random DC topologies to pull
    action_set = load_action_set(grid_folder / "case57" / PREPROCESSING_PATHS["action_set_file_path"])
    assert len(action_set.local_actions)
    network_data = load_network_data(grid_folder / "case57" / PREPROCESSING_PATHS["network_data_file_path"])
    assert network_data.branch_action_set is not None

    topos = []
    for _ in range(10):
        action = random_actions(action_set, np.random.default_rng(42), n_split_subs=2)

        # Create a random integer array for worst_k_contingency_cases
        worst_k_contingency_cases = np.random.choice(contingency_ids_case_57, size=5, replace=False).tolist()

        topology = Topology(
            actions=action,
            disconnections=[],
            pst_setpoints=None,
            metrics=Metrics(
                fitness=-42,
                extra_scores={
                    "overload_energy_n_1": 123.4,
                    "top_k_overloads_n_1": float(np.random.rand()),
                },
                worst_k_contingency_cases=worst_k_contingency_cases,
            ),
        )
        topos.append(topology)
    topopushresult = Result(
        result=TopologyPushResult(
            strategies=[Strategy(timesteps=[topo]) for topo in topos],
        ),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        instance_id="dc_optimizer",
    )
    return topopushresult


def test_idle_loop_no_message() -> None:
    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    # Make poll() return None on the first call, then a ShutdownCommand message
    shutdown_command = Command(command=ShutdownCommand())
    shutdown_message = Mock()
    shutdown_message.value.return_value = serialize_message(shutdown_command.model_dump_json())
    worker_data.command_consumer.poll.side_effect = [None, shutdown_message]
    worker_data.result_consumer.consume.return_value = []
    heartbeats = []
    with pytest.raises(SystemExit):
        idle_loop(
            worker_data=worker_data,
            send_heartbeat_fn=lambda hb: heartbeats.append(hb),
            heartbeat_interval_ms=100,
        )
    assert len(heartbeats) == 2
    assert worker_data.command_consumer.poll.call_count == 2
    assert worker_data.result_consumer.consume.call_count == 1
    assert worker_data.result_consumer.close.call_count == 1
    assert worker_data.command_consumer.close.call_count == 1


def test_idle_loop_optimization_started() -> None:
    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    # Make poll() return an OptimizationStartedCommand message
    start_command = Command(
        command=StartOptimizationCommand(
            ac_params=ACOptimizerParameters(),
            grid_files=[GridFile(framework=Framework.PYPOWSYBL, grid_folder="not/exist")],
            optimization_id="test",
        )
    )
    start_message = Mock()
    start_message.value.return_value = serialize_message(start_command.model_dump_json())
    worker_data.command_consumer.poll.return_value = start_message
    worker_data.result_consumer.consume.return_value = []
    heartbeats = []
    parsed_start_command = idle_loop(
        worker_data=worker_data,
        send_heartbeat_fn=lambda hb: heartbeats.append(hb),
        heartbeat_interval_ms=100,
    )
    assert parsed_start_command.grid_files[0].grid_folder == "not/exist"

    assert len(heartbeats) == 1
    assert worker_data.command_consumer.poll.call_count == 1
    assert worker_data.result_consumer.consume.call_count == 0
    assert worker_data.result_consumer.close.call_count == 0
    assert worker_data.command_consumer.close.call_count == 0


def test_optimization_loop(
    grid_folder: Path,
    topopushresult: Result,
    loadflow_result_folder: Path,
) -> None:
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=1,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            runner_processes=8,
            enable_ac_rejection=False,
        )
    )

    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    worker_data.result_consumer.consume = Mock(
        return_value=[Mock(value=Mock(return_value=serialize_message(topopushresult.model_dump_json())))]
    )
    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    optimization_loop(
        ac_params=parameters,
        grid_files=grid_files,
        worker_data=worker_data,
        send_result_fn=send_result_fn,
        send_heartbeat_fn=send_heartbeat_fn,
        optimization_id="test",
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    assert len(results) >= 3
    assert isinstance(results[0], OptimizationStartedResult)
    assert isinstance(results[1], TopologyPushResult)
    assert isinstance(results[-1], OptimizationStoppedResult)
    assert heartbeats
    assert isinstance(heartbeats[0], OptimizationStartedHeartbeat)

    assert len(worker_data.db.exec(select(ACOptimTopology)).all())


def test_optimization_loop_error_during_initialization(
    grid_folder: Path,
    loadflow_result_folder: Path,
) -> None:
    """Test error handling when initialization fails."""
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    parameters = ACOptimizerParameters()

    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    worker_data.result_consumer.consume = Mock(return_value=[])

    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    with patch("toop_engine_topology_optimizer.ac.worker.initialize_optimization") as init_mock:
        init_mock.side_effect = Exception("Test error")
        optimization_loop(
            ac_params=parameters,
            grid_files=grid_files,
            worker_data=worker_data,
            send_result_fn=send_result_fn,
            send_heartbeat_fn=send_heartbeat_fn,
            optimization_id="test_init_error",
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )

    assert len(results) == 1
    assert isinstance(results[0], OptimizationStoppedResult)
    assert results[0].reason == "error"


def test_optimization_loop_error_dc_timeout(
    grid_folder: Path,
    loadflow_result_folder: Path,
) -> None:
    """Test error handling when waiting for DC results times out."""
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    parameters = ACOptimizerParameters()

    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    worker_data.result_consumer.consume = Mock(return_value=[])

    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    with patch("toop_engine_topology_optimizer.ac.worker.wait_for_first_dc_results") as wait_mock:
        wait_mock.side_effect = TimeoutError("Test error")
        optimization_loop(
            ac_params=parameters,
            grid_files=grid_files,
            worker_data=worker_data,
            send_result_fn=send_result_fn,
            send_heartbeat_fn=send_heartbeat_fn,
            optimization_id="test_dc_timeout",
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )

    assert len(results) == 1
    assert isinstance(results[0], OptimizationStoppedResult)
    assert results[0].reason == "dc-not-started"


def test_optimization_loop_error_during_epoch(
    grid_folder: Path,
    loadflow_result_folder: Path,
) -> None:
    """Test error handling when run_epoch fails.

    This test verifies that errors during epoch execution are properly caught
    and result in an OptimizationStoppedResult. The wait_for_first_dc_results
    function must be patched to avoid the actual timeout behavior since no
    DC results are being sent through the mocked consumer.
    """
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    parameters = ACOptimizerParameters()

    worker_data = WorkerData(
        db=create_session(),
        command_consumer=Mock(spec=LongRunningKafkaConsumer),
        result_consumer=Mock(spec=LongRunningKafkaConsumer),
        producer=Mock(spec=Producer),
    )
    worker_data.result_consumer.consume = Mock(return_value=[])

    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Patch wait_for_first_dc_results to avoid timeout since no real DC results are sent
    with patch("toop_engine_topology_optimizer.ac.worker.wait_for_first_dc_results"):
        with patch("toop_engine_topology_optimizer.ac.worker.run_epoch") as run_mock:
            run_mock.side_effect = Exception("Test error")
            optimization_loop(
                ac_params=parameters,
                grid_files=grid_files,
                worker_data=worker_data,
                send_result_fn=send_result_fn,
                send_heartbeat_fn=send_heartbeat_fn,
                optimization_id="test_epoch_error",
                loadflow_result_fs=loadflow_result_fs,
                processed_gridfile_fs=processed_gridfile_fs,
            )

    assert len(results) == 2
    assert isinstance(results[0], OptimizationStartedResult)
    assert isinstance(results[1], OptimizationStoppedResult)
    assert results[1].reason == "error"


@pytest.mark.timeout(30)
def test_main(
    grid_folder: Path,
    topopushresult: Result,
    loadflow_result_folder: Path,
) -> None:
    """Test the main AC worker function using mocked Kafka.

    This test verifies that the AC worker can:
    1. Start and initialize
    2. Receive DC topologies
    3. Run optimization
    4. Produce results
    5. Shutdown gracefully
    """
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    parameters = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=1,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            runner_processes=8,
            enable_ac_rejection=False,
        )
    )
    start_command = Command(
        command=StartOptimizationCommand(
            ac_params=parameters,
            grid_files=grid_files,
            optimization_id="test",
        )
    )

    # Set up fake Kafka with commands and DC results
    # Note: No ShutdownCommand - let FakeConsumerEmptyException be raised when commands run out
    command_consumer = FakeConsumer(
        messages={
            "commands": [
                serialize_message(start_command.model_dump_json()),
            ]
        },
        kill_on_empty=True,
    )

    result_consumer = FakeConsumer(
        messages={
            "results": [serialize_message(topopushresult.model_dump_json())],
        },
        kill_on_empty=False,
    )

    producer = FakeProducer()

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))

    with pytest.raises(FakeConsumerEmptyException):
        main(
            args=Args(
                optimizer_command_topic="commands",
                optimizer_heartbeat_topic="heartbeats",
                optimizer_results_topic="results",
                heartbeat_interval_ms=100,
                kafka_broker="not_used",
            ),
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
            producer=producer,
            command_consumer=command_consumer,
            result_consumer=result_consumer,
        )

    # Verify that results were produced
    assert "results" in producer.messages
    assert len(producer.messages["results"]) > 0

    # Parse all results
    results = [Result.model_validate_json(deserialize_message(msg)) for msg in producer.messages["results"]]

    # Check for expected result types
    started_found = False
    stopped_found = False
    topo_push_found = False

    for result in results:
        if isinstance(result.result, OptimizationStartedResult):
            started_found = True
        elif isinstance(result.result, TopologyPushResult):
            topo_push_found = True
        elif isinstance(result.result, OptimizationStoppedResult):
            stopped_found = True
            assert result.result.reason == "converged"

    assert started_found, "OptimizationStartedResult not found in results"
    assert stopped_found, "OptimizationStoppedResult not found in results"
    assert topo_push_found, "TopologyPushResult not found in results"
