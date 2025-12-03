from multiprocessing import Process, set_start_method
from pathlib import Path

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

pytestmark = pytest.mark.xdist_group("kafka")


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
        main(
            Args(
                importer_command_topic=kafka_command_topic,
                importer_heartbeat_topic=kafka_heartbeat_topic,
                importer_results_topic=kafka_results_topic,
                kafka_broker=kafka_connection_str,
            ),  # type: ignore
            unprocessed_gridfile_fs=unprocessed_gridfile_fs,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
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
        main(
            Args(
                importer_command_topic=kafka_command_topic,
                importer_heartbeat_topic=kafka_heartbeat_topic,
                importer_results_topic=kafka_results_topic,
                kafka_broker=kafka_connection_str,
            ),  # type: ignore
            unprocessed_gridfile_fs=unprocessed_gridfile_fs,
            processed_gridfile_fs=processed_gridfile_fs,
            loadflow_result_fs=loadflow_result_fs,
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
        target=main,
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
