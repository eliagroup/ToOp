# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from confluent_kafka import Consumer, Producer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


@pytest.mark.timeout(60)
def test_kafka(kafka_command_topic: str, kafka_connection_str: str) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )

    producer.produce(kafka_command_topic, value=b"Hello world")
    producer.flush()

    consumer = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "auto.offset.reset": "earliest",
            "group.id": "test",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_command_topic])
    message = consumer.poll(timeout=10.0)
    assert message.value() == b"Hello world"
    consumer.close()


@pytest.mark.timeout(60)
def test_serialization(kafka_command_topic: str, kafka_connection_str: str, static_information_file: str) -> None:
    command = Command(
        command=StartOptimizationCommand(
            optimization_id="test",
            grid_files=[
                GridFile(
                    framework=Framework.PANDAPOWER,
                    grid_folder="test",
                )
            ],
        )
    )
    data = command.model_dump_json(indent=2)

    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    producer.produce(kafka_command_topic, value=serialize_message(data))
    producer.flush()

    consumer = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "auto.offset.reset": "earliest",
            "group.id": "test_serialization",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_command_topic])
    message = consumer.poll(timeout=10.0)
    assert deserialize_message(message.value()) == data

    data_decoded = Command.model_validate_json(deserialize_message(message.value()))
    assert data_decoded.command.optimization_id == "test"
    consumer.close()
