# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides a wrapper around the confluent_kafka Consumer to allow long running processes.

There are fundamentally two ways how to deal with long running processes in kafka:
  - increase the max.poll.interval.ms to a very high value so the processing can happen in between.
  - pause the consumer while processing and resume it afterwards, regularly polling the paused consumer to reset the
  max.poll.interval.ms timeout. These polls will not consume any messages, but will reset the timeout.

The first method is not recommended as it inhibits rebalances during frozen time, does not detect frozen consumers and is
generally not what kafka was designed for. However, kafka does not provide a way to pause and resume a topic, only a
topic-partition. That means if a consumer is paused but then a rebalance happens, new topic-partitions will not be
paused and the consumer may receive messages for it. However, the polls we are doing are happening in the processing loop
and we are fundamentally unable to process anything there. Hence, this wrapper provides a way to pause and resume a consumer,
listening to the assignment changes and pausing or resuming the new TPs accordingly. As a drawback, this consumer then looses
the ability to consume multiple topics.
"""

from logging import getLogger

from beartype.typing import Optional
from confluent_kafka import Consumer, Message, TopicPartition
from logbook import Logger

logger = Logger(__name__)


class LongRunningKafkaConsumer:
    """A kafka consumer for long running processes that need to pause and resume the topic consumption."""

    def __init__(
        self,
        topic: str,
        group_id: str,
        bootstrap_servers: str,
        client_id: str,
        max_poll_interval_ms: int = 1_800_000,
        kafka_auth_config: dict | None = None,
    ) -> None:
        """Initialize the LongRunningKafkaConsumer.

        Parameters
        ----------
        topic : str
            The topic to subscribe to. This can only be a single topic as the consumer will pause and resume it.
        group_id : str
            The consumer group id
        bootstrap_servers : str
            The bootstrap servers to connect to, e.g. "localhost:9092
        client_id : str
            The client id to use for the consumer. This is used for logging and debugging purposes.
        max_poll_interval_ms : int, optional
            The maximum time in milliseconds between polls before the consumer is considered dead. Defaults to 1_800_000
            (30 minutes). Set this long enough so the process fits in with confidence.
        kafka_auth_config : dict | None, optional
            Additional kafka authentication configuration to pass to the consumer. Defaults to None.
        """
        self.topic = topic
        consumer_config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "client.id": client_id,
            "max.poll.interval.ms": max_poll_interval_ms,
            "log_level": 2,
        }
        if kafka_auth_config:
            consumer_config.update(kafka_auth_config)
        self.consumer = Consumer(
            consumer_config,
            logger=getLogger(f"consumer_{client_id}"),
        )
        self.client_id = client_id
        self.assignment: list[TopicPartition] = []
        self.consumer.subscribe(
            [self.topic],
            on_assign=lambda _consumer, assignment: self._update_assignment(new_tps=assignment, removed_tps=[]),
            on_revoke=lambda _consumer, assignment: self._update_assignment(new_tps=[], removed_tps=assignment),
            on_lost=lambda _consumer, assignment: self._update_assignment(new_tps=[], removed_tps=assignment),
        )
        self.last_msg: Optional[Message] = None
        self.is_paused = False

    def _update_assignment(self, new_tps: list[TopicPartition], removed_tps: list[TopicPartition]) -> None:
        """Update the consumers assignment and pause or resume based on the is_paused flag.

        This is an internal method, do not call it directly

        Parameters
        ----------
        new_tps : list[TopicPartition]
            The new topic partitions assigned to the consumer.
        removed_tps : list[TopicPartition]
            The topic partitions that were removed from the consumer's assignment.
        """
        if new_tps:
            self.assignment += new_tps
        if removed_tps:
            self.assignment = [tp for tp in self.assignment if tp not in removed_tps]

        if self.is_paused:
            self.consumer.pause(self.assignment)
        if not self.is_paused:
            self.consumer.resume(self.assignment)

    def consume(self, timeout: float | int, num_messages: int) -> list[Message]:
        """Consume a batch of messages from the kafka topic at once.

        This will commit all offsets directly after consuming the messages.

        Parameters
        ----------
        timeout : float | int
            The maximum time to wait for messages in seconds. If no messages are available, returns an empty list.
        num_messages : int
            The maximum number of messages to consume. If more messages are available, they will not be consumed.

        Returns
        -------
        list[Message]
            The consumed messages, or an empty list if no messages are available within the timeout.
        """
        if self.last_msg is not None:
            raise RuntimeError("Commit the last message either through commit or stop_processing before consuming again")

        messages = self.consumer.consume(num_messages=num_messages, timeout=float(timeout))
        if not messages:
            return []

        self.consumer.commit(message=messages[-1], asynchronous=True)
        return messages

    def poll(self, timeout: float | int) -> Optional[Message]:
        """Consume a single message from the Kafka topic.

        This will not commit the offset to the broker

        Parameters
        ----------
        timeout : float | int
            The maximum time to wait for a message in seconds. If no message is available, returns None.

        Returns
        -------
        Optional[Message]
            The consumed message, or None if no message is available within the timeout.
        """
        if self.last_msg is not None:
            raise RuntimeError("Commit the last message either through commit or stop_processing before consuming again")

        msg = self.consumer.poll(timeout=float(timeout))
        self.last_msg = msg
        return msg

    def commit(self) -> None:
        """Commit the last consumed message."""
        if self.last_msg is not None:
            self.consumer.commit(message=self.last_msg, asynchronous=False)
            self.last_msg = None
        else:
            raise RuntimeError("No message to commit")

    def start_processing(self) -> None:
        """Start a long running process to consume the message.

        This will internally pause the consumer. To not exceed the poll timeout, call heartbeat() periodically while
        processing, e.g. every epoch
        """
        self.is_paused = True
        self._update_assignment([], [])

    def heartbeat(self) -> None:
        """Send a heartbeat to the kafka topic while processing to reset the max poll interval timeout."""
        if not self.is_paused:
            raise RuntimeError("Cannot send heartbeat while not processing")
        self._update_assignment([], [])
        msg = self.consumer.poll(timeout=0)
        if msg is not None:
            raise RuntimeError("Heartbeat should not consume messages")

    def stop_processing(self) -> None:
        """Stop the long running process and commit the last message"""
        self.is_paused = False
        self._update_assignment([], [])
        if self.last_msg:
            self.consumer.commit(message=self.last_msg, asynchronous=False)
            self.last_msg = None

    def close(self) -> None:
        """Close the consumer and commit the last message if any."""
        if self.last_msg:
            self.consumer.commit(message=self.last_msg, asynchronous=False)
            self.last_msg = None
        self.consumer.close()
