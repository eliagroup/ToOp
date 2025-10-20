from multiprocessing import Process, set_start_method
from pathlib import Path

import pypowsybl
import pytest
from confluent_kafka import Consumer, Producer
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_contingency_analysis.ac_loadflow_service.lf_worker import LoadflowWorkerArgs, idle_loop, main
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_commands import (
    Job,
    LoadflowServiceCommand,
    PowsyblGrid,
    ShutdownCommand,
    StartCalculationCommand,
)
from toop_engine_interfaces.messages.lf_service.loadflow_heartbeat import LoadflowHeartbeat
from toop_engine_interfaces.messages.lf_service.loadflow_results import (
    LoadflowBaseResult,
    LoadflowStartedResult,
    LoadflowSuccessResult,
)


@pytest.mark.timeout(30)
def test_serialization(kafka_command_topic: str, kafka_connection_str: str) -> None:
    command = LoadflowServiceCommand(
        command=StartCalculationCommand(
            loadflow_id="test", method="ac", grid_data=PowsyblGrid(grid_files=["grid.xiidm"]), jobs=[Job(id="test_job")]
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
            "group.id": "test",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_command_topic])
    message = consumer.poll(timeout=10)
    assert message is not None
    assert message.value().decode() == data

    data_decoded = LoadflowServiceCommand.model_validate_json(message.value().decode())
    assert data_decoded.command.loadflow_id == "test"
    consumer.close()


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

    command = LoadflowServiceCommand(
        command=StartCalculationCommand(
            loadflow_id="test", method="ac", grid_data=PowsyblGrid(grid_files=["grid.xiidm"]), jobs=[Job(id="test_job")]
        )
    )
    producer.produce(kafka_command_topic, value=command.model_dump_json().encode())
    producer.flush()

    consumer = LongRunningKafkaConsumer(
        topic=kafka_command_topic,
        group_id="test_idle_loop",
        bootstrap_servers=kafka_connection_str,
        client_id="test_client",
    )

    parsed = idle_loop(consumer, lambda: None, 100)
    assert parsed.loadflow_id == "test"
    consumer.commit()

    command = LoadflowServiceCommand(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=command.model_dump_json().encode())

    with pytest.raises(SystemExit):
        idle_loop(consumer, lambda: None, 100)


@pytest.mark.timeout(60)
def test_main_simple(
    kafka_command_topic: str, kafka_heartbeat_topic: str, kafka_results_topic: str, kafka_connection_str: str
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    command = LoadflowServiceCommand(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=command.model_dump_json().encode())
    producer.flush()

    with pytest.raises(SystemExit):
        main(
            LoadflowWorkerArgs(
                loadflow_command_topic=kafka_command_topic,
                loadflow_heartbeat_topic=kafka_heartbeat_topic,
                loadflow_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                processed_gridfile_folder=Path("/tmp"),
            )  # type: ignore
        )


@pytest.mark.timeout(200)
def test_main(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    powsybl_bus_breaker_net: pypowsybl.network.Network,
    tmp_path: Path,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": kafka_connection_str,
            "log_level": 2,
        }
    )
    powsybl_bus_breaker_net.save(tmp_path / "grid.xiidm")
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    # read json file
    command = LoadflowServiceCommand(
        command=StartCalculationCommand(
            loadflow_id="test",
            method="ac",
            grid_data=PowsyblGrid(grid_files=["grid.xiidm"], n_1_definition=nminus1_definition),
            jobs=[Job(id="test_job")],
        )
    )
    producer.produce(kafka_command_topic, value=command.model_dump_json().encode())
    producer.flush()

    # Shutdown the worker after successful preprocessing
    command = LoadflowServiceCommand(command=ShutdownCommand())
    producer.produce(kafka_command_topic, value=command.model_dump_json().encode())
    producer.flush()

    with pytest.raises(SystemExit):
        main(
            LoadflowWorkerArgs(
                loadflow_command_topic=kafka_command_topic,
                loadflow_heartbeat_topic=kafka_heartbeat_topic,
                loadflow_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                processed_gridfile_folder=tmp_path,
                loadflow_result_folder=tmp_path,
            )  # type: ignore
        )

    consumer = Consumer(
        {
            "bootstrap.servers": kafka_connection_str,
            "auto.offset.reset": "earliest",
            "group.id": "test_main",
            "log_level": 2,
        }
    )
    consumer.subscribe([kafka_results_topic])

    message = consumer.poll(timeout=10)
    assert message is not None

    result = LoadflowBaseResult.model_validate_json(message.value().decode())
    assert isinstance(result.result, LoadflowStartedResult)

    message = consumer.poll(timeout=1)
    assert message is not None
    result = LoadflowBaseResult.model_validate_json(message.value().decode())
    assert isinstance(result.result, LoadflowSuccessResult)
    assert isinstance(result.result.loadflow_reference.relative_path, str)

    dirfs = DirFileSystem(str(tmp_path))
    lf_result = load_loadflow_results_polars(dirfs, reference=result.result.loadflow_reference)
    assert isinstance(lf_result, LoadflowResultsPolars)
    assert lf_result.job_id == "test_job"
    assert len(lf_result.branch_results.collect()) > 0
    consumer.close()


@pytest.mark.timeout(100)
def test_main_idle(
    kafka_command_topic: str,
    kafka_heartbeat_topic: str,
    kafka_results_topic: str,
    kafka_connection_str: str,
    tmp_path: Path,
) -> None:
    set_start_method("spawn")
    # Create an idling main process
    p = Process(
        target=main,
        args=(
            LoadflowWorkerArgs(
                loadflow_command_topic=kafka_command_topic,
                loadflow_heartbeat_topic=kafka_heartbeat_topic,
                loadflow_results_topic=kafka_results_topic,
                heartbeat_interval_ms=100,
                kafka_broker=kafka_connection_str,
                processed_gridfile_folder=tmp_path,
                loadflow_result_folder=tmp_path,
            ),
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
        message = consumer.poll(timeout=60)
        assert message is not None
        heartbeat = LoadflowHeartbeat.model_validate_json(message.value().decode())
        assert heartbeat.idle
        assert heartbeat.status_info is None

        # Send a shutdown command
        producer = Producer(
            {
                "bootstrap.servers": kafka_connection_str,
                "log_level": 2,
            }
        )
        command = LoadflowServiceCommand(command=ShutdownCommand())
        producer.produce(kafka_command_topic, value=command.model_dump_json().encode())
        producer.flush()
        consumer.close()
        p.join(timeout=10)
    finally:
        # Ensure the process is terminated
        p.terminate()
        p.join()
