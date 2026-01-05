# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logging
from pathlib import Path
from typing import Literal, Union
from unittest.mock import patch

import pytest
from confluent_kafka import Consumer, Producer
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.dc.worker.worker import Args, idle_loop, main, optimization_loop
from toop_engine_topology_optimizer.interfaces.messages.commands import (
    Command,
    DCOptimizerParameters,
    ShutdownCommand,
    StartOptimizationCommand,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import HeartbeatUnion, OptimizationStartedHeartbeat
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    ResultUnion,
    TopologyPushResult,
)

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


def create_producer(kafka_broker: str, instance_id: str, log_level: int = 2) -> Producer:
    producer = Producer(
        {
            "bootstrap.servers": kafka_broker,
            "client.id": instance_id,
            "log_level": log_level,
        },
        logger=logging.getLogger(f"ac_worker_producer_{instance_id}"),
    )
    return producer


def create_consumer(
    type: Literal["LongRunningKafkaConsumer", "Consumer"], topic: str, group_id: str, bootstrap_servers: str, client_id: str
) -> Union[LongRunningKafkaConsumer, Consumer]:
    if type == "LongRunningKafkaConsumer":
        consumer = LongRunningKafkaConsumer(
            topic=topic,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
        )
    elif type == "Consumer":
        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
                "client.id": client_id,
            }
        )
    else:
        raise ValueError(f"Unknown consumer type: {type}")
    return consumer


@pytest.mark.timeout(60)
def test_idle_loop(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )

    command = Command(
        command=StartOptimizationCommand(
            dc_params=DCOptimizerParameters(
                summary_frequency=1,
                check_command_frequency=1,
            ),
            grid_files=[
                GridFile(
                    framework=Framework.PANDAPOWER,
                    grid_folder="child_folder",
                )
            ],
            optimization_id="test",
        )
    )

    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    consumer = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        bootstrap_servers=kafka_connection_str,
        group_id="test_idle_loop",
        client_id="test_idle_loop_client",
    )

    parsed = idle_loop(consumer, lambda _: None, 100)
    assert parsed.optimization_id == "test"
    assert tuple(gf.grid_folder for gf in parsed.grid_files) == ("child_folder",)
    assert consumer.last_msg is not None
    consumer.commit()

    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    with pytest.raises(SystemExit) as excinfo:
        idle_loop(consumer, lambda _: None, 100)
    assert excinfo.value.code == 0
    consumer.consumer.close()


@pytest.mark.timeout(60)
def test_main_simple(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    processed_gridfile_folder: Path,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    command = Command(command=ShutdownCommand(optimization_id="test"))
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    processed_gridfile_fs = DirFileSystem(str(processed_gridfile_folder))
    with pytest.raises(SystemExit):
        main(
            Args(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=1000,
                kafka_broker=kafka_connection_str,
            ),
            processed_gridfile_fs=processed_gridfile_fs,
            producer=create_producer(kafka_connection_str, "dc_worker"),
            command_consumer=create_consumer(
                "LongRunningKafkaConsumer",
                kafka_command_topic,
                "dc_optimizer",
                kafka_connection_str,
                "dc_worker",
            ),
        )


@pytest.mark.timeout(300)
def test_main(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    grid_folder: str,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=BatchedMEParameters(
                runtime_seconds=30,
            ),
        ),
        grid_files=[
            GridFile(
                framework=Framework.PANDAPOWER,
                grid_folder="oberrhein",
            )
        ],
        optimization_id="test",
    )

    # order exactly one epoch
    command = Command(command=start_opt_command)
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()), partition=0)
    producer.flush()

    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()), partition=0)
    producer.flush()

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # run the worker
    with pytest.raises(SystemExit):
        main(
            args=Args(
                optimizer_command_topic=kafka_command_topic,
                optimizer_heartbeat_topic=kafka_heartbeat_topic,
                optimizer_results_topic=kafka_results_topic,
                kafka_broker=kafka_connection_str,
            ),
            processed_gridfile_fs=processed_gridfile_fs,
            producer=create_producer(kafka_connection_str, "dc_worker"),
            command_consumer=create_consumer(
                "LongRunningKafkaConsumer",
                kafka_command_topic,
                "dc_optimizer",
                kafka_connection_str,
                "dc_worker",
            ),
        )

    # subscribe to the results topic
    consumer = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "auto.offset.reset": "earliest",
            "group.id": "test_main",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_results_topic])

    # first message should be the starting message
    message = consumer.poll(timeout=10.0)
    result = Result.model_validate_json(deserialize_message(message.value()))
    assert isinstance(result.result, OptimizationStartedResult)

    topo_push_found = False
    split_topo_push_found = False
    stopped_found = False
    while message := consumer.poll(timeout=1.0):
        result = Result.model_validate_json(deserialize_message(message.value()))
        if isinstance(result.result, TopologyPushResult):
            topo_push_found = True
            for strategy in result.result.strategies:
                if len(strategy.timesteps[0].actions):
                    split_topo_push_found = True
        elif isinstance(result.result, OptimizationStoppedResult):
            stopped_found = True
            assert result.result.reason == "converged"
            break

    assert topo_push_found, "Expected at least one TopologyPushResult"
    assert split_topo_push_found, "Expected at least one TopologyPushResult with split actions"
    assert stopped_found, "Expected an OptimizationStoppedResult with reason 'converged'"

    # there should be no more messages after the stop result
    message = consumer.poll(timeout=1.0)
    assert message is None, "Expected no more messages in the results topic"
    consumer.close()


# @pytest.mark.skip()
@pytest.mark.timeout(60)
@pytest.mark.parametrize("distributed", [False, True])
def test_optimization_loop(
    grid_folder: str,
    distributed: bool,
) -> None:
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=BatchedMEParameters(
                runtime_seconds=20,
            ),
            loadflow_solver_config=LoadflowSolverParameters(distributed=distributed),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="oberrhein")],
        optimization_id="test",
    )

    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    processed_gridfile_fs = DirFileSystem(str(grid_folder))

    optimization_loop(
        dc_params=start_opt_command.dc_params,
        grid_files=start_opt_command.grid_files,
        send_result_fn=send_result_fn,
        send_heartbeat_fn=send_heartbeat_fn,
        optimization_id=start_opt_command.optimization_id,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    assert isinstance(results[0], OptimizationStartedResult)
    assert isinstance(results[1], TopologyPushResult)
    assert isinstance(results[-1], OptimizationStoppedResult)
    assert results[-1].reason == "converged"

    assert isinstance(heartbeats[0], OptimizationStartedHeartbeat)


# @pytest.mark.skip()
@pytest.mark.timeout(60)
def test_optimization_loop_error_handling(
    grid_folder: str,
) -> None:
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=BatchedMEParameters(
                runtime_seconds=5,
            ),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="oberrhein")],
        optimization_id="test",
    )

    results = []

    def send_result_fn(result: ResultUnion) -> None:
        results.append(result)

    heartbeats = []

    def send_heartbeat_fn(heartbeat: HeartbeatUnion) -> None:
        heartbeats.append(heartbeat)

    processed_gridfile_fs = DirFileSystem(str(grid_folder))

    with patch("toop_engine_topology_optimizer.dc.worker.worker.initialize_optimization") as mock_initialize_optimization:
        mock_initialize_optimization.side_effect = Exception("error")
        optimization_loop(
            dc_params=start_opt_command.dc_params,
            grid_files=start_opt_command.grid_files,
            send_result_fn=send_result_fn,
            send_heartbeat_fn=send_heartbeat_fn,
            optimization_id=start_opt_command.optimization_id,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        assert mock_initialize_optimization.called

    assert len(results) == 1
    assert isinstance(results[0], OptimizationStoppedResult)
    assert results[0].reason == "error"

    results = []
    heartbeats = []

    with patch("toop_engine_topology_optimizer.dc.worker.worker.run_epoch") as mock_run_epoch:
        mock_run_epoch.side_effect = Exception("error")
        optimization_loop(
            dc_params=start_opt_command.dc_params,
            grid_files=start_opt_command.grid_files,
            send_result_fn=send_result_fn,
            send_heartbeat_fn=send_heartbeat_fn,
            optimization_id=start_opt_command.optimization_id,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        assert mock_run_epoch.called

    assert len(results) == 2
    assert isinstance(results[0], OptimizationStartedResult)
    assert isinstance(results[1], OptimizationStoppedResult)
    assert results[1].reason == "error"
