# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
from confluent_kafka import Consumer, Producer
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import serialize_message
from toop_engine_topology_optimizer.dc.worker.idle_loop import idle_loop
from toop_engine_topology_optimizer.interfaces.messages.commands import (
    Command,
    DCOptimizerParameters,
    ShutdownCommand,
    StartOptimizationCommand,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.results import OptimizationStoppedResult, ResultUnion

from packages.topology_optimizer_pkg.tests.fake_kafka import FakeMessage

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


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

    parsed = idle_loop(
        consumer, lambda _: None, lambda _result, _optim: None, heartbeat_interval_ms=100, max_command_age_hours=2.0
    )
    assert parsed.optimization_id == "test"
    assert tuple(gf.grid_folder for gf in parsed.grid_files) == ("child_folder",)
    assert consumer.last_msg is not None
    consumer.commit()

    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    with pytest.raises(SystemExit) as excinfo:
        idle_loop(
            consumer, lambda _: None, lambda _result, _optim: None, heartbeat_interval_ms=100, max_command_age_hours=2.0
        )
    assert excinfo.value.code == 0
    consumer.consumer.close()


def test_idle_loop_optimization_started_command_too_old() -> None:
    mock_consumer = Mock(spec=LongRunningKafkaConsumer)
    mock_consumer.consumer = Mock(spec=Consumer)
    shutdown_command = Command(command=ShutdownCommand())
    start_command = Command(
        command=StartOptimizationCommand(
            dc_params=DCOptimizerParameters(),
            grid_files=[GridFile(framework=Framework.PYPOWSYBL, grid_folder="not/exist")],
            optimization_id="test",
        ),
        timestamp=(datetime.now() - timedelta(hours=1)).isoformat(),
    )
    start_message = FakeMessage(
        value_bytes=serialize_message(start_command.model_dump_json()),
    )
    shutdown_message = FakeMessage(
        value_bytes=serialize_message(shutdown_command.model_dump_json()),
    )

    mock_consumer.poll.side_effect = [start_message, shutdown_message]
    results = []
    heartbeats = []

    def send_result_fn(result: ResultUnion, optimization_id: str) -> None:
        results.append((result, optimization_id))

    with pytest.raises(SystemExit):
        idle_loop(
            consumer=mock_consumer,
            send_heartbeat_fn=lambda hb: heartbeats.append(hb),
            send_result_fn=send_result_fn,
            heartbeat_interval_ms=100,
            max_command_age_hours=0.5,
        )
    assert len(results) == 1
    assert isinstance(results[0][0], OptimizationStoppedResult)
    assert results[0][1] == "test"
    assert results[0][0].reason == "command-too-old"
