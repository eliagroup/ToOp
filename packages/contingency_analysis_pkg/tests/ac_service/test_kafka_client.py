import time

import pytest
from confluent_kafka import Consumer, Producer
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


@pytest.mark.timeout(60)
def test_long_running_kafka_consumer(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    client = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_long_running_kafka_consumer",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client",
    )

    # Test producing a message
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    message = "Test message for long running consumer"
    producer.produce(kafka_command_topic, value=serialize_message(message))
    producer.flush()

    # Test consuming a message
    consumed_message = client.poll(timeout=30)
    assert consumed_message is not None
    assert deserialize_message(consumed_message.value()) == message

    # Produce a second message that shall not be consumed yet
    message = "Second test message for long running consumer"
    producer.produce(kafka_command_topic, value=serialize_message(message))
    producer.flush()

    # Test pausing and resuming the consumer
    client.start_processing()
    for _ in range(10):
        client.heartbeat()
        time.sleep(0.1)

    client.stop_processing()
    consumed_message = client.poll(timeout=5)
    assert deserialize_message(consumed_message.value()) == message, (
        "The second message should not have been consumed while processing was stopped."
    )

    with pytest.raises(RuntimeError):
        client.heartbeat()

    client.commit()

    for _ in range(10):
        producer.produce(kafka_command_topic, value=serialize_message(message))
    producer.flush()

    messages = client.consume(timeout=5, num_messages=100)
    assert len(messages) == 10

    producer.produce(kafka_command_topic, value=serialize_message(message))
    producer.flush()
    messages = client.consume(timeout=5, num_messages=100)
    assert len(messages) == 1
    client.consumer.close()


@pytest.mark.timeout(60)
def test_pause_directly_after_subscribe(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    # Test producing a message
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    message = "Test message for pause directly after subscribe"
    producer.produce(kafka_command_topic, value=serialize_message(message))
    producer.flush()

    client = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_pause_directly_after_subscribe",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client",
    )
    client.start_processing()

    # Do a few heartbeats to ensure the consumer was rebalanced
    for _ in range(30):
        client.heartbeat()
        time.sleep(1)

    client.stop_processing()

    # Attempt to consume a message
    consumed_message = client.poll(timeout=5)
    assert consumed_message is not None
    assert deserialize_message(consumed_message.value()) == message, (
        "The message should have been consumed after pausing and resuming the consumer."
    )
    client.consumer.close()


@pytest.mark.skip(reason="Very flaky for some reason, TODO fix")
@pytest.mark.timeout(100)
def test_two_consumers(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    # Create plenty of messages to ensure there are enough for every partition
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    for i in range(25):
        message = f"Test message {i} for two consumers"
        producer.produce(kafka_command_topic, value=serialize_message(message), partition=i % 2)
    producer.flush()

    # Start two consumers that will consume messages
    client1 = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_two_consumers",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client_1",
    )
    client2 = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_two_consumers",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client_2",
    )

    time.sleep(10)  # Give some time for the consumers to join the group and rebalance

    msg1 = client1.poll(timeout=50)
    msg2 = client2.poll(timeout=30)

    assert msg1 is not None, "The first consumer should have consumed a message."
    assert msg2 is not None, "The second consumer should have consumed a message."
    client1.consumer.close()
    client2.consumer.close()


@pytest.mark.timeout(100)
def test_pause_including_rebalance(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    # Create plenty of messages to ensure there are enough for every partition
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    for i in range(25):
        message = f"Test message {i} for pause after rebalance"
        producer.produce(kafka_command_topic, value=serialize_message(message), partition=i % 2)
    producer.flush()

    # Start one consumer that will sleep and one that will consume
    client1 = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_pause_after_rebalance",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client",
    )
    msg = client1.poll(timeout=50)
    assert msg is not None, "The first consumer should have consumed a message."
    client1.start_processing()

    # Create the second consumer to trigger a rebalance while consumer 1 is processing
    client2 = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_pause_after_rebalance",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client_2",
    )

    # Do the trigger-rebalance dance:
    assert client2.poll(timeout=5) is None
    client1.heartbeat()
    msg = client2.poll(timeout=15)
    assert msg is not None, "The second consumer should have consumed a message."
    client2.commit()

    client1.heartbeat()

    # Unpause the consumer
    client1.stop_processing()

    # Attempt to consume a message
    assert client1.poll(timeout=10) is not None
    assert client2.poll(timeout=10) is not None
    client1.consumer.close()
    client2.consumer.close()


@pytest.mark.timeout(100)
@pytest.mark.skip(reason="This test is a duplicate of test_two_consumers")
def test_two_consumers_direct(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    # Create plenty of messages to ensure there are enough for every partition
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    for i in range(25):
        message = f"Test message {i} for two consumers direct"
        producer.produce(kafka_command_topic, value=serialize_message(message), partition=i % 2)
    producer.flush()

    # Start two consumers that will consume messages
    client1 = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "group.id": "test_two_consumers_direct",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "client.id": "test_client_1",
            "log_level": 2,
        }
    )
    client1.subscribe([kafka_command_topic])
    client2 = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "group.id": "test_two_consumers_direct",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "client.id": "test_client_2",
            "log_level": 2,
        }
    )
    client2.subscribe([kafka_command_topic])

    time.sleep(10)

    msg1 = client1.poll(timeout=50)
    msg2 = client2.poll(timeout=30)

    assert msg1 is not None, "The first consumer should have consumed a message."
    assert msg2 is not None, "The second consumer should have consumed a message."
    client1.close()
    client2.close()


@pytest.mark.timeout(100)
@pytest.mark.skip(reason="This test is a duplicate of test_two_consumers")
def test_two_consumers_different_order_direct(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    # Create plenty of messages to ensure there are enough for every partition
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    for i in range(25):
        message = f"Test message {i} for two consumers different order direct"
        producer.produce(kafka_command_topic, value=serialize_message(message), partition=i % 2)
    producer.flush()

    # Start two consumers that will consume messages
    client1 = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "group.id": "test_two_consumers_different_order_direct",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "client.id": "test_client_1",
            "max.poll.interval.ms": 300000,
            "log_level": 2,
        }
    )
    client1.subscribe([kafka_command_topic])
    assert client1.poll(timeout=30) is not None

    client2 = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "group.id": "test_two_consumers_different_order_direct",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "client.id": "test_client_2",
            "max.poll.interval.ms": 300000,
            "log_level": 2,
        }
    )
    client2.subscribe([kafka_command_topic])
    assert client2.poll(timeout=3) is None
    assert client1.poll(timeout=10) is not None
    assert client2.poll(timeout=10) is not None
    client1.close()
    client2.close()
