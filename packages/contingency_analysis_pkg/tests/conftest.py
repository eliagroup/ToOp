import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Generator

import docker
import pandapower
import pandas as pd
import pandera
import polars as pl
import pypowsybl
import pytest
import ray
from docker.client import DockerClient
from docker.models.containers import Container
from toop_engine_grid_helpers.pandapower.example_grids import pandapower_extended_oberrhein
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_busbar_b_in_ieee,
    powsybl_extended_case57,
)

# Setup pandera
config = pandera.config.PanderaConfig(
    validation_enabled=True, validation_depth=pandera.config.ValidationDepth.SCHEMA_AND_DATA
)
pandera.config.reset_config_context(config)
from toop_engine_interfaces.loadflow_results import BranchResultSchema
from toop_engine_interfaces.loadflow_results_polars import BranchResultSchemaPolars


@pytest.fixture(scope="session")
def init_ray() -> bool:
    ray.init(runtime_env={"working_dir": Path(__file__).parent}, ignore_reinit_error=True, include_dashboard=False)
    # Return a dummy bool so it can be used as a fixture on only those tests that need ray, autouse is a bit overkill
    return True


@pytest.fixture(scope="session")
def docker_client() -> DockerClient:
    return docker.from_env()


@pytest.fixture(scope="session")
def kafka_connection_str(kafka_container: Container) -> str:
    for _ in range(100):
        kafka_container.reload()
        if kafka_container.status == "running":
            return f"localhost:{kafka_container.ports['9092/tcp'][0]['HostPort']}"

        time.sleep(0.1)
    raise RuntimeError("Could not get Kafka port")


def make_topic(kafka_container: Container, topic: str) -> None:
    # Remove existing topic if it exists (due to previous tests)
    exit_code, output = kafka_container.exec_run(
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic {topic} --if-exists"
    )
    assert exit_code == 0, output.decode()

    # Wait for the topic to be deleted (max 3 seconds)
    for _ in range(30):
        exit_code, output = kafka_container.exec_run(
            f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic {topic}"
        )
        if exit_code != 0:
            break
        time.sleep(0.1)

    # Create new topic
    exit_code, output = kafka_container.exec_run(
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic {topic} --partitions 2 --replication-factor 1"
    )
    assert exit_code == 0, output.decode()


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_command_topic(kafka_container: Container) -> str:
    topic = f"command_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_results_topic(kafka_container: Container) -> str:
    topic = f"results_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_heartbeat_topic(kafka_container: Container) -> str:
    topic = f"heartbeat_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


def kill_all_containers_with_name(docker_client: DockerClient, target_name: str) -> None:
    """Kill all docker containers with the given name.

    There might be left-over containers from previous tests in case the test crashed and didn't clean up properly."""
    # Get all containers
    containers: list[Container] = docker_client.containers.list()

    containers_to_kill = []
    for container in containers:
        if container.name == target_name:
            containers_to_kill.append(container.id)

    for container_id in set(containers_to_kill):
        container = docker_client.containers.get(container_id)
        container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def kafka_container(docker_client: DockerClient) -> Generator[Container, None, None]:
    # Kill all containers that expose port 9092
    kill_all_containers_with_name(docker_client, "test_kafka")

    container = docker_client.containers.run(
        "apache/kafka",
        detach=True,
        name="test_kafka",
        auto_remove=True,
        ports={"9092/tcp": 9092},
        environment={
            "KAFKA_NODE_ID": "1",
            "KAFKA_PROCESS_ROLES": "broker,controller",
            "KAFKA_LISTENERS": "PLAINTEXT://:9092,CONTROLLER://:9093",
            "KAFKA_ADVERTISED_LISTENERS": "PLAINTEXT://localhost:9092",
            "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
            "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@localhost:9093",
            "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
            "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS": "0",
            "KAFKA_NUM_PARTITIONS": "1",
            "KAFKA_AUTO_CREATE_TOPICS_ENABLE": "false",
            "KAFKA_DELETE_TOPIC_ENABLE": "true",
        },
    )
    for log_line in container.logs(stream=True):
        if "Kafka Server started" in log_line.decode():
            break
    yield container
    container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def kafka_connection_str(kafka_container: Container) -> str:
    for _ in range(100):
        kafka_container.reload()
        if kafka_container.status == "running":
            return f"localhost:{kafka_container.ports['9092/tcp'][0]['HostPort']}"

        time.sleep(0.1)
    raise RuntimeError("Could not get Kafka port")


@pytest.fixture(scope="session")
def _pandapower_net() -> pandapower.pandapowerNet:
    net = pandapower_extended_oberrhein()
    pandapower.rundcpp(net)
    return net


@pytest.fixture(scope="function")
def pandapower_net(_pandapower_net: pandapower.pandapowerNet) -> pandapower.pandapowerNet:
    # Create a copy of the pandapower network for each test to avoid side effects
    return deepcopy(_pandapower_net)


@pytest.fixture(scope="session")
def _powsybl_networks() -> tuple[pypowsybl.network.Network, pypowsybl.network.Network]:
    node_breaker_grid = basic_node_breaker_network_powsybl()

    bus_breaker_grid = powsybl_extended_case57()
    create_busbar_b_in_ieee(bus_breaker_grid)

    return node_breaker_grid, bus_breaker_grid


@pytest.fixture(scope="function")
def powsybl_bus_breaker_net(
    _powsybl_networks: tuple[pypowsybl.network.Network, pypowsybl.network.Network],
) -> pypowsybl.network.Network:
    # Create a copy of the bus breaker network for each test to avoid side effects
    return deepcopy(_powsybl_networks[1])


@pytest.fixture(scope="function")
def powsybl_node_breaker_net(
    _powsybl_networks: tuple[pypowsybl.network.Network, pypowsybl.network.Network],
) -> pypowsybl.network.Network:
    # Create a copy of the node breaker network for each test to avoid side effects
    return deepcopy(_powsybl_networks[0])


@pytest.fixture(scope="function")
def branch_results_df_fast_failing() -> pd.DataFrame:
    # Create a MultiIndex for (timestep, element, contingency)
    idx = pd.MultiIndex.from_tuples(
        [
            (0, "cont1", "branch1", 1),
            (0, "cont1", "branch2", 1),
            (0, "cont2", "branch1", 1),
            (0, "cont2", "branch2", 1),
            (0, "cont3", "branch1", 1),
            (0, "cont3", "branch2", 1),
            (1, "cont1", "branch1", 1),
            (1, "cont1", "branch2", 1),
            (1, "cont2", "branch1", 1),
            (1, "cont2", "branch2", 1),
            (1, "cont3", "branch1", 1),
            (1, "cont3", "branch2", 1),
        ],
        names=["timestep", "contingency", "element", "side"],
    )
    # Data: purposely set overloads for cont2 and cont3 higher than cont1, for both timesteps
    data = {
        "p": [float(x) for x in [10, 20, 100, 200, 50, 60, 15, 25, 110, 210, 55, 65]],
        "i": [float(x) for x in [1, 2, 10, 20, 5, 6, 2, 3, 11, 21, 6, 7]],
        "loading": [float(x) for x in [10, 10, 50, 50, 25, 30, 12, 12, 52, 52, 27, 32]],
    }
    df = pd.DataFrame(data, index=idx)
    # Add required columns for BranchResultSchema, fill with dummy values if needed
    for col in BranchResultSchema.to_schema().columns:
        if col not in df.columns:
            dtype = BranchResultSchema.to_schema().columns[col].dtype.type.name
            if pd.api.types.is_float_dtype(dtype):
                df[col] = 0.0
            elif pd.api.types.is_integer_dtype(dtype):
                df[col] = 0
            elif pd.api.types.is_bool_dtype(dtype):
                df[col] = False
            else:
                df[col] = ""
    BranchResultSchema.validate(df)
    return df


@pytest.fixture(scope="function")
def branch_results_df_fast_failing_polars(branch_results_df_fast_failing) -> pl.LazyFrame:
    df = branch_results_df_fast_failing
    df_polars = pl.from_pandas(df, include_index=True).lazy()
    df_polars = BranchResultSchemaPolars.validate(df_polars)
    return df_polars
