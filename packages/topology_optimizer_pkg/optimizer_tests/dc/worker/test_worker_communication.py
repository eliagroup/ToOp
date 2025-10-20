import pytest
from confluent_kafka import Consumer, Producer
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile


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
    producer.produce(kafka_command_topic, value=data.encode())
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
    assert message.value().decode() == data

    data_decoded = Command.model_validate_json(message.value().decode())
    assert data_decoded.command.optimization_id == "test"
    consumer.close()
