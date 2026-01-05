# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Generator, Literal, Union

import chex
import docker
import jax
import numpy as np
import pandera
import pytest
from confluent_kafka import Consumer, Producer
from docker import DockerClient
from docker.models.containers import Container
from fsspec.implementations.dirfs import DirFileSystem
from jaxtyping import Int
from omegaconf import DictConfig
from sqlmodel import Session, SQLModel, create_engine, select
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_dc_solver.example_grids import (
    case14_pandapower,
    case57_data_powsybl,
    case57_non_converging,
    create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder,
    oberrhein_data,
)
from toop_engine_dc_solver.preprocess import load_grid
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, load_nminus1_definition
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import hash_topo_data

config = pandera.config.PanderaConfig(
    validation_enabled=True, validation_depth=pandera.config.ValidationDepth.SCHEMA_AND_DATA
)
pandera.config.reset_config_context(config)

jax.config.update("jax_enable_x64", True)
## Set up loggers
# JAX
jax.config.update("jax_logging_level", "WARNING")

# NUMBA
logging.getLogger("numba").setLevel(logging.WARNING)


def pytest_sessionstart(session):
    chex.set_n_cpu_devices(2)


@pytest.fixture(scope="session")
def docker_client() -> Generator[DockerClient, None, None]:
    client = docker.from_env()
    yield client
    kill_all_containers_with_name(client, "test_kafka")


def kill_all_containers_with_name(docker_client: DockerClient, target_name: str) -> None:
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
        "apache/kafka:4.0.0",
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


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_connection_str(kafka_container: Container) -> str:
    for _ in range(100):
        kafka_container.reload()
        if kafka_container.status == "running":
            return f"localhost:{kafka_container.ports['9092/tcp'][0]['HostPort']}"

        time.sleep(0.1)
    raise RuntimeError("Could not get Kafka port")


def delete_topic(kafka_container: Container, topic: str) -> None:
    exit_code, output = kafka_container.exec_run(
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic {topic} --if-exists"
    )
    assert exit_code == 0, output.decode()


def make_topic(kafka_container: Container, topic: str, partitions: int = 1) -> None:
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
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic {topic} --partitions {partitions} --replication-factor 1"
    )
    assert exit_code == 0, output.decode()
    # Wait for the topic to be created (max 3 seconds)
    for _ in range(30):
        exit_code, output = kafka_container.exec_run(
            f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic {topic}"
        )
        if exit_code == 0:
            break
        time.sleep(0.1)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_command_topic(kafka_container: Container) -> Generator[str, None, None]:
    topic = "command_topic" + str(os.urandom(4).hex())
    make_topic(kafka_container, topic)
    yield topic
    delete_topic(kafka_container, topic)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_results_topic(kafka_container: Container) -> Generator[str, None, None]:
    topic = "results_topic" + str(os.urandom(4).hex())
    make_topic(kafka_container, topic)
    yield topic
    delete_topic(kafka_container, topic)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_heartbeat_topic(kafka_container: Container) -> Generator[str, None, None]:
    topic = "heartbeat_topic" + str(os.urandom(4).hex())
    make_topic(kafka_container, topic)
    yield topic
    delete_topic(kafka_container, topic)


@pytest.fixture(scope="session")
def grid_folder() -> Path:
    """Grid data directory prepared once and shared across workers.

    Returns:
        Path: The path to the grid data directory.
    """

    def initialize_grid_dirs(target_path: Path) -> Path:
        target_path.mkdir(exist_ok=True)

        oberrhein_path = target_path / "oberrhein"
        if not oberrhein_path.exists():
            oberrhein_data(oberrhein_path)
            filesystem_dir = DirFileSystem(str(oberrhein_path))
            load_grid(filesystem_dir, pandapower=True)

        case14_path = target_path / "case14"
        if not case14_path.exists():
            case14_pandapower(case14_path)
            filesystem_dir = DirFileSystem(str(case14_path))
            load_grid(filesystem_dir, pandapower=True)

        case57_path = target_path / "case57"
        if not case57_path.exists():
            case57_data_powsybl(case57_path)
            filesystem_dir = DirFileSystem(str(case57_path))
            load_grid(filesystem_dir, pandapower=False)

        complex_grid_path = target_path / "complex_grid"
        if not complex_grid_path.exists():
            create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder(complex_grid_path)
            filesystem_dir = DirFileSystem(str(complex_grid_path))
            load_grid(filesystem_dir, pandapower=False)

        return target_path

    data_path = Path(__file__).parent / "data"

    target_path = initialize_grid_dirs(data_path)

    return target_path


@pytest.fixture(scope="session")
def case57_non_converging_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to the case57 non-converging grid file"""
    tmp_path = tmp_path_factory.mktemp("case57_non_converging")

    case57_non_converging(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    load_grid(filesystem_dir, pandapower=True)
    return tmp_path


@pytest.fixture(scope="session")
def static_information_file(grid_folder: Path) -> Path:
    return grid_folder / "oberrhein" / PREPROCESSING_PATHS["static_information_file_path"]


@pytest.fixture(scope="session")
def static_information_file_complex(grid_folder: Path) -> Path:
    return grid_folder / "complex_grid" / PREPROCESSING_PATHS["static_information_file_path"]


def build_dc_config(base_path: str, static_info_file: Path) -> DictConfig:
    """
    Builds a configuration dictionary for testing purposes.

    Args:
        base_path (str): The base directory path where results and output files will be stored.
        static_info_file (Path): Path to the static information file to include in the configuration.

    Returns
    -------
        DictConfig: The configuration dictionary for testing.
    """
    return DictConfig(
        {
            "task_name": "test",
            "fixed_files": [
                str(static_info_file),
            ],
            "double_precision": None,
            "tensorboard_dir": base_path + "/results/{task_name}",
            "stats_dir": base_path + "/results/{task_name}",
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "stdout": None,
            "double_limits": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "output_json": base_path + "/results/output.json",
            "lf_config": {"distributed": False},
            "ga_config": {
                "runtime_seconds": 30,
                "me_descriptors": [{"metric": "split_subs", "num_cells": 10}],
                "observed_metrics": ["overload_energy_n_1", "split_subs"],
            },
        }
    )


@pytest.fixture(scope="session", params=["oberrhein", "complex_grid"])
def dc_config(
    request, static_information_file: Path, static_information_file_complex: Path, tmp_path_factory: pytest.TempPathFactory
) -> DictConfig:
    """ToOp configuration to test the optimiser."""
    if request.param == "oberrhein":
        static_info_file = static_information_file
    else:
        static_info_file = static_information_file_complex

    base_path = str(tmp_path_factory.mktemp("base"))
    return build_dc_config(base_path, static_info_file)


def build_ac_config() -> DictConfig:
    """AC validation configuration to test the optimiser."""
    return DictConfig({"n_processes": 1, "k_best_topos": 5})


def build_pipeline_cfg(complex_grid_dst: Path, iteration_name: str, file_name: str) -> DictConfig:
    """Builds a pipeline configuration dictionary for testing purposes."""
    return DictConfig(
        {"root_path": complex_grid_dst, "iteration_name": iteration_name, "file_name": file_name, "grid_type": "powsybl"}
    )


@pytest.fixture(scope="session")
def pipeline_and_configs(
    tmp_path_factory: pytest.TempPathFactory, grid_folder: Path
) -> tuple[DictConfig, DictConfig, DictConfig]:
    """Configuration for the end-to-end pipeline tests"""
    tmp_grid_folder = tmp_path_factory.mktemp("pipeline_grid")
    # Copy complex grid data to temporary folder
    complex_grid_src = grid_folder / "complex_grid"
    complex_grid_dst = tmp_grid_folder / "complex_grid"
    os.makedirs(complex_grid_dst, exist_ok=True)
    shutil.copytree(str(complex_grid_src), str(complex_grid_dst), dirs_exist_ok=True)

    file_name = "grid.xiidm"
    iteration_name = ""
    pipeline_cfg = build_pipeline_cfg(complex_grid_dst, iteration_name, file_name)
    dc_cfg = build_dc_config(str(tmp_grid_folder), complex_grid_dst / PREPROCESSING_PATHS["static_information_file_path"])
    ac_cfg = build_ac_config()
    return pipeline_cfg, dc_cfg, ac_cfg


@pytest.fixture(scope="session")
def preprocessing_parameters() -> DictConfig:
    """Preprocessing parameters for the end-to-end pipeline tests"""
    return DictConfig({"action_set_clip": 2**10, "enable_bb_outage": True, "bb_outage_as_nminus1": False})


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine, tables=[ACOptimTopology.__table__])
    with Session(engine) as session:
        yield session


@pytest.fixture
def dc_repertoire_elements_per_sub() -> tuple[Int[np.ndarray, " n_relevant_subs"], Int[np.ndarray, " n_relevant_subs"]]:
    """How many branches and injections are in each relevant substation in the DC repertoire"""
    branches_per_sub = np.array([5, 6, 4, 9])
    injections_per_sub = np.array([2, 12, 3, 1])

    return branches_per_sub, injections_per_sub


@pytest.fixture
def dc_repertoire(session: Session) -> list[ACOptimTopology]:
    """Populate the database with three strategies of 10 random timesteps each"""
    for _strategy in range(3):
        assignments = []
        for _timestep in range(10):
            # Generate random assignments
            n_splits = np.random.randint(1, 6)
            n_disconnections = np.random.randint(0, 3)
            n_psts = 5

            assignments.append(
                (
                    np.random.choice(a=5000, size=n_splits, replace=False).tolist(),  # actions
                    np.random.choice(a=10, size=n_disconnections, replace=False).tolist(),  # disconnections
                    np.random.randint(low=0, high=60, size=n_psts).tolist(),  # pst_setpoints
                )
            )
        strategy_hash = hash_topo_data(assignments)
        for timestep, (actions, disconnections, pst_setpoints) in enumerate(assignments):
            session.add(
                ACOptimTopology(
                    actions=actions,
                    disconnections=disconnections,
                    pst_setpoints=pst_setpoints,
                    unsplit=False,
                    timestep=timestep,
                    strategy_hash=strategy_hash,
                    optimization_id="test",
                    optimizer_type=OptimizerType.DC,
                    fitness=np.random.rand() * -3000,
                    metrics={
                        "overload_energy_n_1": np.random.rand(),
                        "switching_distance": np.random.randint(0, 30),
                        "split_subs": np.random.randint(0, 5),
                        "disconnections": np.random.randint(0, 10),
                    },
                )
            )
    session.commit()
    return session.exec(select(ACOptimTopology)).all()


@pytest.fixture
def n_minus1_definitions_case_57(grid_folder: Path) -> list[Nminus1Definition]:
    """N-1 definitions for case57 data"""
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder=str(grid_folder / "case57"))]
    return [load_nminus1_definition(grid_file.nminus1_definition_file) for grid_file in grid_files]


@pytest.fixture
def contingency_ids_case_57(n_minus1_definitions_case_57: list[Nminus1Definition]) -> list[str]:
    """Contingnecy ids for case57 data"""
    contingencies = n_minus1_definitions_case_57[0].contingencies
    cont_ids = []
    for cont in contingencies:
        cont_ids.append(cont.id)
    return cont_ids


@pytest.fixture
def unsplit_ac_dc_repertoire(session: Session, contingency_ids_case_57: list[str]) -> tuple[list[ACOptimTopology], Session]:
    """Populate the database with three strategies of 10 random timesteps each"""
    for _strategy in range(3):
        assignments = []
        for _timestep in range(10):
            # Generate random assignments
            n_splits = np.random.randint(1, 6)
            n_disconnections = np.random.randint(0, 3)
            n_psts = 5

            assignments.append(
                (
                    np.random.choice(a=5000, size=n_splits, replace=False).tolist(),  # actions
                    np.random.choice(a=10, size=n_disconnections, replace=False).tolist(),  # disconnections
                    np.random.randint(low=0, high=60, size=n_psts).tolist(),  # pst_setpoints
                )
            )
        strategy_hash = hash_topo_data(assignments)
        for timestep, (actions, disconnections, pst_setpoints) in enumerate(assignments):
            case_ids = np.random.choice(contingency_ids_case_57, size=5, replace=False).tolist()
            session.add(
                ACOptimTopology(
                    actions=actions,
                    disconnections=disconnections,
                    pst_setpoints=pst_setpoints,
                    unsplit=False,
                    timestep=timestep,
                    strategy_hash=strategy_hash,
                    optimization_id="test",
                    optimizer_type=OptimizerType.DC,
                    fitness=np.random.rand(),
                    metrics={
                        "overload_energy_n_1": np.random.rand(),
                        "top_k_overloads_n_1": np.random.rand(),
                    },
                    worst_k_contingency_cases=case_ids,
                )
            )

    # Add unsplit topology AC topology
    unsplit_strategy_hash = hash_topo_data([([], [], [])])  # Unique hash for unsplit topology
    # Generate unique case_ids for the unsplit topology
    case_ids = np.random.choice(contingency_ids_case_57, size=5, replace=False).tolist()
    unsplit_topology = ACOptimTopology(
        actions=[],
        disconnections=[],
        pst_setpoints=[],
        unsplit=True,
        timestep=0,
        strategy_hash=unsplit_strategy_hash,
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=np.random.rand(),
        metrics={
            "overload_energy_n_1": np.random.rand(),
            "top_k_overloads_n_1": np.random.rand(),
        },
        worst_k_contingency_cases=case_ids,
    )
    session.add(unsplit_topology)
    session.commit()
    return session.exec(select(ACOptimTopology)).all(), session


@pytest.fixture
def processed_gridfile_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary folder for processed grid files"""
    return tmp_path_factory.mktemp("processed_gridfiles")


@pytest.fixture
def loadflow_result_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary folder for loadflow results"""
    return tmp_path_factory.mktemp("loadflow_results")


@pytest.fixture
def create_producer():
    def _create(kafka_broker: str, instance_id: str, log_level: int = 2) -> Producer:
        return Producer(
            {
                "bootstrap.servers": kafka_broker,
                "client.id": instance_id,
                "log_level": log_level,
            },
            logger=logging.getLogger(f"ac_worker_producer_{instance_id}"),
        )

    return _create


@pytest.fixture
def create_consumer():
    def _create(
        type: Literal["LongRunningKafkaConsumer", "Consumer"],
        topic: str,
        group_id: str,
        bootstrap_servers: str,
        client_id: str,
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

    return _create
