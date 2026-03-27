# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
from confluent_kafka import Producer
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer


class FakeProducer(Producer):
    def __init__(self):
        self.messages = {}

    def produce(
        self, topic: str, value: bytes, key: str | bytes | None = None, headers: list[tuple[str, bytes]] | None = None
    ):
        if topic not in self.messages:
            self.messages[topic] = []
        self.messages[topic].append(value)

    def flush(self):
        pass


class FakeConsumerEmptyException(Exception):
    pass


class FakeMessage:
    """Mock Kafka Message for testing"""

    def __init__(self, value_bytes: bytes, headers: list[tuple[str, bytes]] | None = None):
        self._value = value_bytes
        self._headers: list[tuple[str, bytes]] | None = headers if headers is not None else [("FakeHeader", value_bytes)]

    def value(self) -> bytes:
        return self._value

    def headers(self) -> list[tuple[str, bytes]] | None:
        return self._headers


class FakeConsumer(LongRunningKafkaConsumer):
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
