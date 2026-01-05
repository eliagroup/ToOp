# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logging
import sys
from logging import getLogger
from multiprocessing import Process, set_start_method
from pathlib import Path
from typing import Literal, Union
from uuid import uuid4

import logbook
import pytest
from confluent_kafka import Consumer, Producer
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_importer.worker.worker import Args, idle_loop, main
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    Command,
    ShutdownCommand,
    StartPreprocessingCommand,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import PreprocessHeartbeat
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    PreprocessingStartedResult,
    PreprocessingSuccessResult,
    Result,
)
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message

# Ensure that tests using Kafka are not run in parallel with each other
pytestmark = pytest.mark.xdist_group("kafka")


logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logging.INFO).push_application()


def create_producer(kafka_broker: str, instance_id: str, log_level: int = 2) -> Producer:
    producer = Producer(
        {
            "bootstrap.servers": kafka_broker,
            "client.id": instance_id,
            "log_level": log_level,
        },
        logger=getLogger(f"ac_worker_producer_{instance_id}"),
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


@pytest.mark.timeout(100)
def test_kafka(kafka_command_topic: str, kafka_connection_str: str) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    producer.produce(kafka_command_topic, value=serialize_message("Hello world"))
    producer.flush()

    consumer = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "auto.offset.reset": "earliest",
            "group.id": "test_kafka",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_command_topic])
    message = consumer.poll(timeout=30.0)
    assert deserialize_message(message.value()) == "Hello world"


@pytest.mark.timeout(100)
def test_serialization(kafka_command_topic: str, kafka_connection_str: str) -> None:
    command = Command(
        command=StartPreprocessingCommand(
            preprocess_id="test",
            importer_parameters=UcteImporterParameters(
                grid_model_file="not/actually/a/ucte/file", data_folder="not/a/folder"
            ),
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
    message = consumer.poll(timeout=30.0)
    assert deserialize_message(message.value()) == data

    data_decoded = Command.model_validate_json(deserialize_message(message.value()))
    assert data_decoded.command.preprocess_id == "test"


@pytest.mark.timeout(100)
def test_idle_loop(
    kafka_command_topic: str,
    kafka_connection_str: str,
) -> None:
    producer = Producer({"bootstrap.servers": kafka_connection_str, "log_level": 2})

    command = Command(
        command=StartPreprocessingCommand(
            preprocess_id="test",
            importer_parameters=UcteImporterParameters(
                grid_model_file="not/actually/a/ucte/file", data_folder="not/a/folder"
            ),
        )
    )
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    consumer = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_idle_loop",
        bootstrap_servers=kafka_connection_str,
        client_id="test_idle_loop_client",
    )

    parsed = idle_loop(consumer, lambda: None, 1000)
    assert parsed.preprocess_id == "test"
    consumer.commit()

    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    with pytest.raises(SystemExit):
        idle_loop(consumer, lambda: None, 100)


@pytest.mark.timeout(30)
def test_main_simple(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
) -> None:
    producer = Producer({"bootstrap.servers": kafka_connection_str, "log_level": 2})
    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()

    unprocessed_gridfile_fs = LocalFileSystem()
    processed_gridfile_fs = LocalFileSystem()
    loadflow_result_fs = LocalFileSystem()
    with pytest.raises(SystemExit):
        instance_id = str(uuid4())
        main(
            args=Args(
                importer_command_topic=kafka_command_topic,
                importer_heartbeat_topic=kafka_heartbeat_topic,
                importer_results_topic=kafka_results_topic,
                kafka_broker=kafka_connection_str,
            ),
            unprocessed_gridfile_fs=unprocessed_gridfile_fs,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
            producer=create_producer(kafka_connection_str, instance_id),
            consumer=create_consumer(
                "LongRunningKafkaConsumer",
                kafka_command_topic,
                "importer_worker",
                kafka_connection_str,
                instance_id,
            ),
        )


@pytest.mark.timeout(130)
def test_main(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    ucte_file: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    output_path = tmp_path_factory.mktemp("output")
    loadflow_path = tmp_path_factory.mktemp("loadflow")

    producer = Producer({"bootstrap.servers": kafka_connection_str, "log_level": 2})

    command = Command(
        command=StartPreprocessingCommand(
            preprocess_id="test",
            importer_parameters=UcteImporterParameters(grid_model_file=ucte_file.name, data_folder=Path("some_timestep")),
        )
    )
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))

    # Shutdown the worker after successful preprocessing
    command = Command(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
    producer.flush()
    # Check that the messages are sent
    consumer = Consumer(
        {"bootstrap.servers": kafka_connection_str, "auto.offset.reset": "earliest", "group.id": "test_main", "log_level": 2}
    )
    consumer.subscribe([kafka_command_topic])
    assert consumer.poll(timeout=30.0) is not None
    assert consumer.poll(timeout=1.0) is not None
    # Subscribe to the results topic
    consumer.unsubscribe()
    consumer.subscribe([kafka_results_topic])
    unprocessed_gridfile_fs = DirFileSystem(ucte_file.parent)
    processed_gridfile_fs = DirFileSystem(output_path)
    loadflow_result_fs = DirFileSystem(loadflow_path)

    with pytest.raises(SystemExit):
        instance_id = str(uuid4())
        main(
            args=Args(
                importer_command_topic=kafka_command_topic,
                importer_heartbeat_topic=kafka_heartbeat_topic,
                importer_results_topic=kafka_results_topic,
                kafka_broker=kafka_connection_str,
            ),
            unprocessed_gridfile_fs=unprocessed_gridfile_fs,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
            producer=create_producer(kafka_connection_str, instance_id),
            consumer=create_consumer(
                "LongRunningKafkaConsumer",
                kafka_command_topic,
                "importer_worker",
                kafka_connection_str,
                instance_id,
            ),
        )

    message = consumer.poll(timeout=30.0)
    assert message is not None
    result = Result.model_validate_json(deserialize_message(message.value()))
    assert isinstance(result.result, PreprocessingStartedResult)

    message = consumer.poll(timeout=1.0)
    assert message is not None
    result = Result.model_validate_json(deserialize_message(message.value()))
    assert isinstance(result.result, PreprocessingSuccessResult)

    assert (output_path / "some_timestep" / PREPROCESSING_PATHS["static_information_file_path"]).exists()


def main_wrapper(
    args: Args,
    unprocessed_gridfile_fs,
    processed_gridfile_fs,
    loadflow_result_fs,
) -> None:
    instance_id = str(uuid4())

    consumer = LongRunningKafkaConsumer(
        topic=args.importer_command_topic,
        bootstrap_servers=args.kafka_broker,
        group_id="importer-worker",
        client_id=instance_id,
    )
    producer = Producer(
        {
            "bootstrap.servers": args.kafka_broker,
            "client.id": instance_id,
            "log_level": 2,
        },
        logger=getLogger("confluent_kafka.producer"),
    )

    main(
        args,
        producer=producer,
        consumer=consumer,
        unprocessed_gridfile_fs=unprocessed_gridfile_fs,
        processed_gridfile_fs=processed_gridfile_fs,
        loadflow_result_fs=loadflow_result_fs,
    )


def test_main_idle(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
) -> None:
    set_start_method("spawn")
    # Create an idling main process
    unprocessed_gridfile_fs = LocalFileSystem()
    processed_gridfile_fs = LocalFileSystem()
    loadflow_result_fs = LocalFileSystem()
    p = Process(
        target=main_wrapper,
        args=(
            Args(
                importer_command_topic=kafka_command_topic,
                importer_heartbeat_topic=kafka_heartbeat_topic,
                importer_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
            ),
            unprocessed_gridfile_fs,
            processed_gridfile_fs,
            loadflow_result_fs,
        ),
    )
    p.start()

    try:
        # Subscribe to the heartsbeat topic and expect a heartbeat
        consumer = Consumer(
            {
                "bootstrap.servers": kafka_connection_str,
                "auto.offset.reset": "earliest",
                "group.id": "test_main_idle",
                "log_level": 2,
            }
        )
        consumer.subscribe([kafka_heartbeat_topic])
        message = consumer.poll(timeout=30.0)
        assert message is not None
        heartbeat = PreprocessHeartbeat.model_validate_json(deserialize_message(message.value()))
        assert heartbeat.idle
        assert heartbeat.status_info is None

        # Send a shutdown command
        producer = Producer(
            {
                "bootstrap.servers": kafka_connection_str,
                "log_level": 2,
            }
        )
        command = Command(command=ShutdownCommand())
        producer.produce(kafka_command_topic, value=serialize_message(command.model_dump_json()))
        producer.flush()
        p.join(timeout=10)
    finally:
        # Ensure the process is terminated
        p.terminate()
        p.join()
