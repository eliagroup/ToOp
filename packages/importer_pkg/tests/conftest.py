# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# import os
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Generator

import docker
import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pypowsybl
import pytest
from docker import DockerClient
from docker.models.containers import Container
from toop_engine_grid_helpers.pandapower.example_grids import (
    example_multivoltage_cross_coupler,
)
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import get_topology
from toop_engine_importer.network_graph.data_classes import BranchSchema, NetworkGraphData, SubstationInformation
from toop_engine_importer.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_importer.network_graph.network_graph import generate_graph
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data, remove_helper_branches
from toop_engine_importer.network_graph.powsybl_station_to_graph import (
    get_node_breaker_topology_graph,
    node_breaker_topology_to_graph_data,
)
from toop_engine_importer.pandapower_import import add_substation_column_to_bus
from toop_engine_importer.pypowsybl_import import powsybl_masks, preprocessing
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    LimitAdjustmentParameters,
    UcteImporterParameters,
)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def docker_client() -> DockerClient:
    return docker.from_env()


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


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
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


@pytest.fixture(
    scope="session",
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
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic {topic} --partitions 1 --replication-factor 1"
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


@pytest.fixture(scope="session")
def ucte_file() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_ucte_powsybl_example.uct")
    return ucte_file


@pytest.fixture(scope="session")
def ucte_file_exporter_test() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_uct_exporter_uct_file.uct")
    return ucte_file


@pytest.fixture(scope="session")
def ucte_json_exporter_test() -> Path:
    ucte_json = Path(f"{os.path.dirname(__file__)}/files/test_uct_exporter_json_file.json")
    return ucte_json


@pytest.fixture(scope="session")
def output_uct_exporter_ref() -> Path:
    output_uct = Path(f"{os.path.dirname(__file__)}/files/test_uct_exporter_uct_file_output_ref.uct")
    return output_uct


@pytest.fixture(scope="session")
def ucte_file_with_border() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_ucte_powsybl_example_with_border.uct")
    return ucte_file


@pytest.fixture(scope="session")
def test_pypowsybl_cgmes_with_3w_trafo() -> Path:
    cgmes_file = Path(f"{os.path.dirname(__file__)}/files/CGMES_Full.zip")
    return cgmes_file


@pytest.fixture(scope="function")
def ucte_importer_parameters(tmp_path_factory: pytest.TempPathFactory, ucte_file):
    tmp_path = tmp_path_factory.mktemp("empty_folder")
    return UcteImporterParameters(
        grid_model_file=ucte_file,
        data_folder=tmp_path,
        white_list_file=None,
        black_list_file=None,
        area_settings=AreaSettings(
            cutoff_voltage=220,
            control_area=["D8"],
            view_area=["D2", "D4", "D7", "D8"],
            nminus1_area=["D2", "D4", "D7", "D8"],
            dso_trafo_factors=None,
            border_line_factors=None,
        ),
    )


@pytest.fixture(scope="function")
def cgmes_importer_parameters(tmp_path_factory: pytest.TempPathFactory, test_pypowsybl_cgmes_with_3w_trafo):
    tmp_path = tmp_path_factory.mktemp("empty_folder")
    return CgmesImporterParameters(
        grid_model_file=test_pypowsybl_cgmes_with_3w_trafo,
        data_folder=tmp_path,
        white_list_file=None,
        black_list_file=None,
        area_settings=AreaSettings(
            cutoff_voltage=220,
            control_area=["BE"],
            view_area=["BE", "NL"],
            nminus1_area=["BE", "NL"],
            dso_trafo_factors=None,
            border_line_factors=None,
        ),
    )


@pytest.fixture(scope="session")
def imported_ucte_file_data_folder(tmp_path_factory: pytest.TempPathFactory, ucte_file) -> Path:
    tmp_path = tmp_path_factory.mktemp("imported_ucte_file_data_folder")
    importer_parameters = UcteImporterParameters(
        grid_model_file=ucte_file,
        data_folder=tmp_path,
        white_list_file=None,
        black_list_file=None,
        area_settings=AreaSettings(
            cutoff_voltage=220,
            control_area=["D8"],
            view_area=["D2", "D4", "D7", "D8"],
            nminus1_area=["D2", "D4", "D7", "D8"],
            dso_trafo_factors=None,
            border_line_factors=None,
        ),
    )
    tmp_grid_file_path_pandapower = tmp_path / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    tmp_grid_file_path_pandapower.parent.mkdir(parents=True, exist_ok=True)
    temp_network_data_file_path = tmp_path / PREPROCESSING_PATHS["network_data_file_path"]
    temp_network_data_file_path.parent.mkdir(parents=True, exist_ok=True)
    # network, network_masks, statistics = network_analysis.convert_file(file, black_white_list_path, output_folder, white_list=True, black_list=True, cross_border_current=False)
    preprocessing.convert_file(importer_parameters=importer_parameters)
    return tmp_path


@pytest.fixture(scope="session")
def ucte_asset_topology(ucte_file: Path) -> Topology:
    network = pypowsybl.network.load(ucte_file)
    importer_parameters = UcteImporterParameters(
        grid_model_file=ucte_file,
        data_folder="files_path",
        area_settings=AreaSettings(
            cutoff_voltage=220,
            control_area=["D8"],
            view_area=["D2", "D4", "D7", "D8"],
            nminus1_area=["D2", "D4", "D7", "D8"],
        ),
    )

    network_masks = powsybl_masks.make_masks(network=network, importer_parameters=importer_parameters)
    topology_model = get_topology(
        network,
        relevant_stations=network_masks.relevant_subs,
        topology_id="test",
        grid_model_file=str(importer_parameters.grid_model_file),
    )
    return topology_model


@pytest.fixture(scope="session")
def limit_update_input() -> tuple[pd.DataFrame, LimitAdjustmentParameters, powsybl_masks.NetworkMasks]:
    shared_index = ["line1", "tie_line1", "trafo1"]
    branch_df = pd.DataFrame(
        index=shared_index,
        data={
            "type": ["LINE", "TIE_LINE", "TWO_WINDINGS_TRANSFORMER"],
            "i1": [10.0, 10.0, 10.0],
            "i2": [10.0, 10.0, 10.0],
            "n0_i1_max": [100.0, 100.0, 100.0],
            "n0_group_name_1": ["group_1", "group_1", "group_1"],
            "n0_i2_max": [100.0, 100.0, 100.0],
            "n0_group_name_2": ["group_2", "group_2", "group_2"],
            "n1_i1_max": [100.0, 100.0, 100.0],
            "n1_group_name_1": ["group_3", "group_3", "group_3"],
            "n1_i2_max": [100.0, 100.0, 100.0],
            "n1_group_name_2": ["group_4", "group_4", "group_4"],
            "dangling_line1_id": [np.nan, "d1", np.nan],
            "dangling_line2_id": [np.nan, "d2", np.nan],
        },
    )
    limit_parameters = LimitAdjustmentParameters(n_0_factor=1.1, n_0_min_increase=0.1, n_1_factor=1.1, n_1_min_increase=0.1)
    # Create network masks where all array have length one and are true
    true_masks = {key: np.array([True]) for key in powsybl_masks.NetworkMasks.__annotations__.keys()}
    network_masks = powsybl_masks.NetworkMasks(**true_masks)
    return branch_df, limit_parameters, network_masks


@pytest.fixture(scope="session")
def _pp_network_w_switches() -> pp.pandapowerNet:
    # https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb
    net = pp.networks.example_multivoltage()
    net.trafo["tap_dependent_impedance"] = False
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=True)
    return net


@pytest.fixture
def pp_network_w_switches(_pp_network_w_switches: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_pp_network_w_switches)


@pytest.fixture(scope="session")
def _pp_network_w_switches_open_coupler() -> pp.pandapowerNet:
    # https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb
    net = pp.networks.example_multivoltage()
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=True)
    net.switch.loc[14, "closed"] = False
    return net


@pytest.fixture
def pp_network_w_switches_open_coupler(_pp_network_w_switches_open_coupler: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_pp_network_w_switches_open_coupler)


@pytest.fixture(scope="session")
def _pp_network_w_switches_parallel_coupler() -> pp.pandapowerNet:
    # https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb
    net = pp.networks.example_multivoltage()
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=True)
    id_max = net.switch.index.max()
    second_busbar_coupler = net.switch.loc[[0, 1, 14]]
    second_busbar_coupler["index"] = [id_max + 1, id_max + 2, id_max + 3]
    second_busbar_coupler.set_index("index", inplace=True, drop=True)
    net.switch = pd.concat([net.switch, second_busbar_coupler])
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 57
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 58
    net.switch.loc[id_max + 1, "element"] = 57
    net.switch.loc[id_max + 2, "element"] = 58
    net.switch.loc[id_max + 3, "element"] = 58
    net.switch.loc[id_max + 3, "bus"] = 57
    return net


@pytest.fixture
def pp_network_w_switches_parallel_coupler(_pp_network_w_switches_parallel_coupler: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_pp_network_w_switches_parallel_coupler)


@pytest.fixture(scope="session")
def _net_multivoltage_cross_coupler() -> pp.pandapowerNet:
    net = example_multivoltage_cross_coupler()
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=False)
    net.bus["Busbar_id"] = ""
    net.bus["Busbar_name"] = ""
    net.bus.loc[net.bus["type"] == "b", "Busbar_id"] = net.bus["name"]
    net.bus.loc[net.bus["type"] == "b", "Busbar_name"] = net.bus["name"]
    return net


@pytest.fixture
def net_multivoltage_cross_coupler(_net_multivoltage_cross_coupler: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_net_multivoltage_cross_coupler)


@pytest.fixture(scope="session")
def get_graph_input_dicts() -> tuple[dict, dict, dict]:
    # data from basic_node_breaker_network_powsybl -> see in explorer for more details
    # fmt: off
    nodes_dict = {
        "connectable_id": {i: val for i, val in enumerate(["BBS3_1", "BBS3_2", "L3", "", "L6", "", "L7", "", "L9", "", "", "", "load2", ""])},
        "connectable_type": {i: val for i, val in enumerate(["BUSBAR_SECTION", "BUSBAR_SECTION", "LINE", "", "LINE", "", "LINE", "", "LINE", "", "", "", "LOAD", ""])},
        "foreign_id": {i: val for i, val in enumerate(["ab", "cd", "ed", "gh", "ij", "lm", "no", "pp", "oo", "ii", "zz", "tt", "sa", "as"])},
        "grid_model_id": {i: val for i, val in enumerate(["BBS3_1", "BBS3_2", "", "", "", "", "", "", "", "", "", "", "", ""])},
        "bus_id": {i: val for i, val in enumerate(["BBS3_1_bus_id", "BBS3_2_bus_id", "", "", "", "", "", "", "", "", "", "", "", ""])},
        "node_type": {i: val for i, val in enumerate(["busbar", "busbar", "node", "node", "node", "node", "node", "node", "node", "node", "node", "node", "node", "node"])},
        "substation_id": {i: "TODO" for i in range(14)},
        "system_operator": {i: "TODO" for i in range(14)},
        "voltage_level": {i: 380 for i in range(14)},
        "helper_node": {i: False for i in range(14)},
    }
    switches_dict = {
        "grid_model_id": {i: val for i, val in enumerate([
            "L32_BREAKER", "L32_DISCONNECTOR_3_0", "L32_DISCONNECTOR_3_1", "L62_BREAKER", "L62_DISCONNECTOR_5_0",
            "L62_DISCONNECTOR_5_1", "L72_BREAKER", "L72_DISCONNECTOR_7_0", "L72_DISCONNECTOR_7_1", "L91_BREAKER",
            "L91_DISCONNECTOR_9_0", "L91_DISCONNECTOR_9_1", "VL3_BREAKER", "VL3_DISCONNECTOR_10_0", "VL3_DISCONNECTOR_11_1",
            "load2_BREAKER", "load2_DISCONNECTOR_13_0", "load2_DISCONNECTOR_13_1"
        ])},
        "foreign_id": {i: "" for i in range(18)},
        "asset_type": {i: val for i, val in enumerate([
            "BREAKER", "DISCONNECTOR", "DISCONNECTOR", "BREAKER", "DISCONNECTOR", "DISCONNECTOR", "BREAKER", "DISCONNECTOR",
            "DISCONNECTOR", "BREAKER", "DISCONNECTOR", "DISCONNECTOR", "BREAKER", "DISCONNECTOR", "DISCONNECTOR",
            "BREAKER", "DISCONNECTOR", "DISCONNECTOR"
        ])},
        "open": {i: val for i, val in enumerate([
            False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, False, True, False
        ])},
        "retained": {i: val for i, val in enumerate([
            True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False
        ])},
        "from_node": {i: val for i, val in enumerate([2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13])},
        "to_node": {i: val for i, val in enumerate([3, 0, 1, 5, 0, 1, 7, 0, 1, 9, 0, 1, 11, 0, 1, 13, 0, 1])},
    }
    node_assets_dict = {
        "grid_model_id": {i: val for i, val in enumerate(["L3", "L6", "L7", "L9", "load2"])},
        "foreign_id": {i: "" for i in range(5)},
        "node": {i: val for i, val in enumerate([2, 4, 6, 8, 12])},
        "asset_type": {i: val for i, val in enumerate(["LINE", "LINE", "LINE", "LINE", "LOAD"])},
    }
    # fmt: on
    return nodes_dict, switches_dict, node_assets_dict


@pytest.fixture(scope="function")
def network_graph_data_test1(get_graph_input_dicts) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    switches_df = pd.DataFrame(switches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True

    network_graph = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df)
    return network_graph


@pytest.fixture(scope="function")
def basic_node_breaker_network_powsybl_network_graph():
    net = pypowsybl.network.create_empty()

    n_subs = 5
    n_vls = 5
    # substation_id : number of buses
    n_buses = {1: 3, 2: 3, 3: 2, 4: 2, 5: 1}
    n_busbar_coupler = {1: 2, 2: 3, 3: 2, 4: 2, 5: 1}

    stations = pd.DataFrame.from_records(
        index="id", data=[{"id": f"S{i + 1}", "country": "BE", "name": f"Station{i + 1}"} for i in range(n_subs)]
    )
    voltage_levels = pd.DataFrame.from_records(
        index="id",
        data=[
            {
                "substation_id": f"S{i + 1}",
                "id": f"VL{i + 1}",
                "topology_kind": "NODE_BREAKER",
                "nominal_v": 225,
                "name": f"VLevel{i + 1}",
            }
            for i in range(n_vls)
        ],
    )
    busbars = pd.DataFrame.from_records(
        index="id",
        data=[
            {"voltage_level_id": f"VL{sub_id}", "id": f"BBS{sub_id}_{bus_id}", "node": bus_id - 1, "name": f"bus{bus_id}"}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )
    busbarSectionPosition = pd.DataFrame.from_records(
        index="id",
        data=[
            {"id": f"BBS{sub_id}_{bus_id}", "section_index": 1, "busbar_index": bus_id}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )

    net.create_substations(stations)
    net.create_voltage_levels(voltage_levels)
    net.create_busbar_sections(busbars)
    net.create_extensions("busbarSectionPosition", busbarSectionPosition)

    lines = pd.DataFrame.from_records(
        data=[
            {"bus_or_busbar_section_id_1": "BBS1_1", "bus_or_busbar_section_id_2": "BBS2_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS2_2"},
            {"bus_or_busbar_section_id_1": "BBS1_3", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS1_3", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS4_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS2_2", "bus_or_busbar_section_id_2": "BBS3_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS3_1", "bus_or_busbar_section_id_2": "BBS5_1"},
        ]
    )
    lines["r"] = 0.1
    lines["x"] = 10
    lines["g1"] = 0
    lines["b1"] = 0
    lines["g2"] = 0
    lines["b2"] = 0
    lines["position_order_1"] = 1
    lines["position_order_2"] = 1
    for i, line in lines.iterrows():
        lines.loc[i, "id"] = f"L{i + 1}"
    lines = lines.set_index("id")
    pypowsybl.network.create_line_bays(net, lines)
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS1_1", "BBS1_2"], bus_or_busbar_section_id_2=["BBS1_2", "BBS1_3"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_1"], bus_or_busbar_section_id_2=["BBS2_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_2"], bus_or_busbar_section_id_2=["BBS2_3"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS3_1"], bus_or_busbar_section_id_2=["BBS3_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS4_1"], bus_or_busbar_section_id_2=["BBS4_2"]
    )
    pypowsybl.network.create_load_bay(net, id="load1", bus_or_busbar_section_id="BBS2_1", p0=100, q0=10, position_order=1)
    pypowsybl.network.create_load_bay(net, id="load2", bus_or_busbar_section_id="BBS3_2", p0=100, q0=10, position_order=2)
    pypowsybl.network.create_generator_bay(
        net,
        id="generator1",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=100,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS1_1",
        position_order=1,
    )
    pypowsybl.network.create_generator_bay(
        net,
        id="generator2",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=100,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS5_1",
        position_order=2,
    )

    return net


@pytest.fixture(scope="function")
def basic_node_breaker_network_powsyblV2():
    net = pypowsybl.network.create_empty()

    n_subs = 6
    n_vls = 6
    # substation_id : number of buses
    n_buses = {1: 2, 2: 3, 3: 4, 4: 4, 5: 3}
    n_busbar_coupler = {1: 2, 2: 3, 3: 2, 4: 2, 5: 1}

    stations = pd.DataFrame.from_records(
        index="id", data=[{"id": f"S{i + 1}", "country": "BE", "name": f"Station{i + 1}"} for i in range(n_subs)]
    )
    voltage_levels = pd.DataFrame.from_records(
        index="id",
        data=[
            {
                "substation_id": f"S{i + 1}",
                "id": f"VL{i + 1}",
                "topology_kind": "NODE_BREAKER",
                "nominal_v": 225,
                "name": f"VLevel{i + 1}",
            }
            for i in range(n_vls)
        ],
    )
    busbars = pd.DataFrame.from_records(
        index="id",
        data=[
            {"voltage_level_id": f"VL{sub_id}", "id": f"BBS{sub_id}_{bus_id}", "node": bus_id - 1, "name": f"bus{bus_id}"}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )
    busbarSectionPosition = pd.DataFrame.from_records(
        index="id",
        data=[
            {"id": f"BBS{sub_id}_{bus_id}", "section_index": 1, "busbar_index": bus_id}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )
    busbarSectionPosition.loc["BBS3_3", "section_index"] = 2
    busbarSectionPosition.loc["BBS3_4", "section_index"] = 2
    busbarSectionPosition.loc["BBS3_3", "busbar_index"] = 1
    busbarSectionPosition.loc["BBS3_4", "busbar_index"] = 2

    net.create_substations(stations)
    net.create_voltage_levels(voltage_levels)
    net.create_busbar_sections(busbars)

    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS5_1", "BBS5_1"], bus_or_busbar_section_id_2=["BBS5_2", "BBS5_3"]
    )

    net.create_extensions("busbarSectionPosition", busbarSectionPosition)

    lines = pd.DataFrame.from_records(
        data=[
            {"bus_or_busbar_section_id_1": "BBS1_1", "bus_or_busbar_section_id_2": "BBS2_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS2_2"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS4_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS2_2", "bus_or_busbar_section_id_2": "BBS3_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS3_1", "bus_or_busbar_section_id_2": "BBS5_1"},
            {"bus_or_busbar_section_id_1": "BBS5_1", "bus_or_busbar_section_id_2": "VL6_1_1"},
        ]
    )
    lines["r"] = 0.2
    lines["x"] = 10
    lines["g1"] = 0
    lines["b1"] = 0
    lines["g2"] = 0
    lines["b2"] = 0
    lines["position_order_1"] = 1
    lines["position_order_2"] = 1
    for i, line in lines.iterrows():
        lines.loc[i, "id"] = f"L{i + 1}"
    lines = lines.set_index("id")

    # display(lines)
    lines.loc["L3", "r"] = 1
    lines.loc["L3", "x"] = 20

    pypowsybl.network.create_voltage_level_topology(
        net, id="VL6", aligned_buses_or_busbar_count=2, switch_kinds="BREAKER, DISCONNECTOR"
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL6_1_2"], bus_or_busbar_section_id_2=["VL6_2_2"]
    )
    pypowsybl.network.create_load_bay(net, id="load6", bus_or_busbar_section_id="VL6_1_1", p0=100, q0=10, position_order=2)

    pypowsybl.network.create_line_bays(net, lines)
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS1_1"], bus_or_busbar_section_id_2=["BBS1_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_1"], bus_or_busbar_section_id_2=["BBS2_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_2"], bus_or_busbar_section_id_2=["BBS2_3"]
    )
    pypowsybl.network.create_coupling_device(
        net,
        bus_or_busbar_section_id_1=["BBS3_1", "BBS3_1", "BBS3_2"],
        bus_or_busbar_section_id_2=["BBS3_2", "BBS3_3", "BBS3_4"],
    )
    pypowsybl.network.create_coupling_device(
        net,
        bus_or_busbar_section_id_1=["BBS4_1", "BBS4_1", "BBS4_1"],
        bus_or_busbar_section_id_2=["BBS4_2", "BBS4_3", "BBS4_4"],
    )

    pypowsybl.network.create_load_bay(net, id="load1", bus_or_busbar_section_id="BBS2_1", p0=100, q0=10, position_order=1)
    pypowsybl.network.create_load_bay(net, id="load2", bus_or_busbar_section_id="BBS3_2", p0=100, q0=10, position_order=2)
    pypowsybl.network.create_generator_bay(
        net,
        id="generator1",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=50,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS1_1",
        position_order=1,
    )
    pypowsybl.network.create_generator_bay(
        net,
        id="generator2",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=50,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS1_2",
        position_order=1,
    )
    pypowsybl.network.create_generator_bay(
        net,
        id="generator3",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=100,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS5_3",
        position_order=2,
    )
    limits = pd.DataFrame.from_records(
        data=[
            {
                "element_id": "L1",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L2",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L3",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
        ],
        index="element_id",
    )
    net.create_operational_limits(limits)
    busbarSectionPosition
    net.get_switches()
    net.update_switches(id="VL5_BREAKER", open=True)
    return net


@pytest.fixture(scope="session")
def get_graph_input_dicts_helper_branches() -> tuple[dict, dict, dict, dict]:
    switches_dict = {
        "grid_model_id": {i: str(i) for i in range(35)},
        "foreign_id": {i: f"fid_{i}" for i in range(35)},
        "asset_type": {i: "BREAKER" if i < 9 else "DISCONNECTOR" for i in range(35)},
        "open": {i: i in [9, 13, 14, 16, 18, 19, 27, 31] for i in range(35)},
        "from_node": {i: 30 + 2 * i for i in range(35)},
        "to_node": {i: 31 + 2 * i for i in range(35)},
    }
    nodes_dict = {
        "connectable_id": {
            i: f"conid_{i}" if i in [28, 29, 100, 101, 102, 103, 104, 105, 106, 107] else "" for i in range(108)
        },
        "connectable_type": {i: "BUSBAR_SECTION" if i in [28, 29] else "LINE" if i >= 100 else "" for i in range(108)},
        "foreign_id": {i: f"fid_{i}" if i in [28, 29] else "" for i in range(108)},
        "grid_model_id": {i: f"gid_{i}" if i in [28, 29] else "" for i in range(108)},
        "bus_id": {i: f"bus_{i}" if i in [28, 29] else "" for i in range(108)},
        "node_type": {i: "busbar" if i in [28, 29] else "node" for i in range(108)},
        "substation_id": {i: "Test_station1" for i in range(108)},
        "system_operator": {i: "TSO" for i in range(108)},
        "voltage_level": {i: 150 for i in range(108)},
        "helper_node": {i: False if i >= 28 else True for i in range(108)},
    }
    # turn of ruff formatting for this dict
    # fmt: off
    helper_branches_dict = {
        "from_node": {i: i + 28 for i in range(80)},
        "to_node": {
            0: 10, 1: 5, 2: 27, 3: 18, 4: 2, 5: 25, 6: 14, 7: 16, 8: 11, 9: 1,
            10: 13, 11: 23, 12: 7, 13: 22, 14: 19, 15: 8, 16: 20, 17: 4, 18: 24, 19: 3,
            20: 5, 21: 18, 22: 17, 23: 14, 24: 10, 25: 4, 26: 5, 27: 16, 28: 5, 29: 23,
            30: 10, 31: 25, 32: 10, 33: 1, 34: 10, 35: 8, 36: 22, 37: 10, 38: 5, 39: 3,
            40: 5, 41: 1, 42: 21, 43: 24, 44: 0, 45: 27, 46: 2, 47: 6, 48: 10, 49: 3,
            50: 11, 51: 15, 52: 12, 53: 20, 54: 26, 55: 19, 56: 10, 57: 16, 58: 10, 59: 18,
            60: 10, 61: 23, 62: 5, 63: 8, 64: 5, 65: 4, 66: 5, 67: 25, 68: 13, 69: 9,
            70: 5, 71: 7, 72: 21, 73: 17, 74: 26, 75: 12, 76: 15, 77: 6, 78: 0, 79: 9,
        },
    }
    # fmt: on
    node_assets_dict = {
        "grid_model_id": {
            0: "conid_100",
            1: "conid_101",
            2: "conid_102",
            3: "conid_103",
            4: "conid_104",
            5: "conid_105",
            6: "conid_106",
            7: "conid_107",
        },
        "foreign_id": {0: "", 1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: ""},
        "node": {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107},
        "asset_type": {0: "LINE", 1: "LINE", 2: "LINE", 3: "LINE", 4: "LINE", 5: "LINE", 6: "LINE", 7: "LINE"},
    }
    return nodes_dict, switches_dict, node_assets_dict, helper_branches_dict


@pytest.fixture(scope="session")
def network_graph_data_test2_helper_branches(get_graph_input_dicts_helper_branches) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""

    network_graph_data = NetworkGraphData(
        nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df, helper_branches=helper_branches_df
    )
    return network_graph_data


@pytest.fixture(scope="session")
def network_graph_data_test2_helper_branches_removed(get_graph_input_dicts_helper_branches) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""
    branches_df = pd.DataFrame(columns=list(BranchSchema.to_schema().columns.keys()))

    remove_helper_branches(
        nodes_df=nodes_df,
        helper_branches_df=helper_branches_df,
        node_assets_df=nodes_asstets_df,
        switches_df=switches_df,
        branches_df=branches_df,
    )
    network_graph_data = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df)
    return network_graph_data


@pytest.fixture(scope="session")
def network_graph_for_asset_topo(get_graph_input_dicts_helper_branches) -> tuple[nx.Graph, NetworkGraphData]:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_asstets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""

    network_graph_data = NetworkGraphData(
        nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df, helper_branches=helper_branches_df
    )
    add_graph_specific_data(network_graph_data)
    graph = generate_graph(network_graph_data)
    run_default_filter_strategy(graph=graph)
    return graph, network_graph_data


@pytest.fixture(scope="function")
def network_graph_for_asset_topoV2_S1(basic_node_breaker_network_powsyblV2) -> tuple[nx.Graph, NetworkGraphData]:
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    network_graph_data = node_breaker_topology_to_graph_data(basic_node_breaker_network_powsyblV2, substation_information)
    graph = get_node_breaker_topology_graph(network_graph_data)
    return graph, network_graph_data


@pytest.fixture(scope="function")
def network_graph_for_asset_topoV2_S3(basic_node_breaker_network_powsyblV2) -> tuple[nx.Graph, NetworkGraphData]:
    substation_dict = {"name": "Station3", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL3"}
    substation_information = SubstationInformation(**substation_dict)
    network_graph_data = node_breaker_topology_to_graph_data(basic_node_breaker_network_powsyblV2, substation_information)
    graph = get_node_breaker_topology_graph(network_graph_data)
    return graph, network_graph_data


@pytest.fixture(scope="function")
def asset_topo_edge_cases_node_breaker_grid() -> pypowsybl.network.Network:
    net = pypowsybl.network.create_empty()
    n_subs = 2
    n_vls = 2

    stations = pd.DataFrame.from_records(
        index="id", data=[{"id": f"S{i + 1}", "country": "BE", "name": f"Station{i + 1}"} for i in range(n_subs)]
    )
    voltage_levels = pd.DataFrame.from_records(
        index="id",
        data=[
            {
                "substation_id": f"S{i + 1}",
                "id": f"VL{i + 1}",
                "topology_kind": "NODE_BREAKER",
                "nominal_v": 225,
                "name": f"VLevel{i + 1}",
            }
            for i in range(n_vls)
        ],
    )

    net.create_substations(stations)
    net.create_voltage_levels(voltage_levels)

    # ################# VL1 #################
    # VL1 -> 3 buses, fist one is out of service
    pypowsybl.network.create_voltage_level_topology(
        net, id="VL1", aligned_buses_or_busbar_count=3, switch_kinds="DISCONNECTOR, BREAKER"
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_1"], bus_or_busbar_section_id_2=["VL1_1_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_1"], bus_or_busbar_section_id_2=["VL1_1_3"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_2"], bus_or_busbar_section_id_2=["VL1_1_3"]
    )

    # set first Breaker VL1_BREAKER open, no conection on left side, busbar VL1_2_2 on right side
    # asset topo should open the breaker and randomly select one busbar on the left side, likely VL1_1_1 or VL1_2_1
    # -> error if breaker is not open -> connects out of service busbar with in service busbar if breaker is not opened
    net.update_switches(id="VL1_DISCONNECTOR_15_0", open=True)
    net.update_switches(id="VL1_DISCONNECTOR_16_3", open=True)
    net.update_switches(id="VL1_DISCONNECTOR_16_4", open=False)

    # set second breaker VL1_BREAKER#0
    net.remove_elements(["VL1_DISCONNECTOR_17_0", "VL1_DISCONNECTOR_18_7"])
    net.update_switches(id="VL1_DISCONNECTOR_18_6", open=True)
    net.update_switches(id="VL1_DISCONNECTOR_17_1", open=False)
    net.update_switches(id="VL1_DISCONNECTOR_18_8", open=False)

    # set third breaker VL1_BREAKER#1, no connection on left and side, but also no out of service busbar connectable
    # asset topo should open the breaker and randomly select one busbar on the left side, likely VL1_2_2 left and VL1_2_3 right
    net.remove_elements(["VL1_DISCONNECTOR_19_3", "VL1_DISCONNECTOR_20_6"])

    # ################# VL2 #################
    # wired, but realistic setup, see net.get_single_line_diagram('VL2')
    pypowsybl.network.create_voltage_level_topology(
        net,
        id="VL2",
        aligned_buses_or_busbar_count=2,
        switch_kinds="DISCONNECTOR, BREAKER, DISCONNECTOR, DISCONNECTOR, DISCONNECTOR, DISCONNECTOR, DISCONNECTOR",
    )

    net.remove_elements(
        [
            "VL2_DISCONNECTOR_0_2",
            "VL2_DISCONNECTOR_2_16",
            "VL2_BREAKER_1_2",
            "VL2_DISCONNECTOR_17_4",
            "VL2_DISCONNECTOR_4_6",
            "VL2_DISCONNECTOR_12_14",
            "VL2_DISCONNECTOR_13_15",
        ]
    )
    net.create_switches(id="BBS1_1-BBS1_4", voltage_level_id="VL2", node1=0, node2=5, kind="DISCONNECTOR", open=False)
    net.create_switches(id="BBS1_3-BBS1_5", voltage_level_id="VL2", node1=3, node2=6, kind="DISCONNECTOR", open=False)
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL2_1_5"], bus_or_busbar_section_id_2=["VL2_1_6"]
    )
    net.update_switches(id="VL2_DISCONNECTOR_20_8", open=True)
    net.update_switches(id="VL2_DISCONNECTOR_20_9", open=False)

    # set  VL2_BREAKER#0, force one connection to out of service busbar
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL2_1_7"], bus_or_busbar_section_id_2=["VL2_1_8"]
    )
    net.remove_elements(["VL2_DISCONNECTOR_23_14"])

    lines = pd.DataFrame.from_records(
        data=[
            {"bus_or_busbar_section_id_1": "VL2_1_1", "bus_or_busbar_section_id_2": "VL1_2_1"},
            {"bus_or_busbar_section_id_1": "VL2_2_1", "bus_or_busbar_section_id_2": "VL1_2_1"},
            {"bus_or_busbar_section_id_1": "VL2_1_4", "bus_or_busbar_section_id_2": "VL1_3_1"},
            {"bus_or_busbar_section_id_1": "VL2_1_4", "bus_or_busbar_section_id_2": "VL1_3_1"},
            {"bus_or_busbar_section_id_1": "VL2_2_4", "bus_or_busbar_section_id_2": "VL1_3_2"},
            {"bus_or_busbar_section_id_1": "VL2_2_1", "bus_or_busbar_section_id_2": "VL1_3_3"},
            {"bus_or_busbar_section_id_1": "VL2_1_7", "bus_or_busbar_section_id_2": "VL1_2_2"},
            {"bus_or_busbar_section_id_1": "VL2_1_7", "bus_or_busbar_section_id_2": "VL1_2_3"},
            {"bus_or_busbar_section_id_1": "VL2_2_7", "bus_or_busbar_section_id_2": "VL2_1_8"},
            {"bus_or_busbar_section_id_1": "VL2_1_8", "bus_or_busbar_section_id_2": "VL1_3_2"},
            {"bus_or_busbar_section_id_1": "VL2_1_4", "bus_or_busbar_section_id_2": "VL2_1_5"},  # used as empty bay
        ]
    )
    lines["r"] = 0.1
    lines["x"] = 10
    lines["g1"] = 0
    lines["b1"] = 0
    lines["g2"] = 0
    lines["b2"] = 0
    lines["position_order_1"] = 1
    lines["position_order_2"] = 1
    for i, _ in lines.iterrows():
        lines.loc[i, "id"] = f"L{i + 1}"
    lines = lines.set_index("id")
    pypowsybl.network.create_line_bays(net, lines)

    pypowsybl.network.create_generator_bay(
        net,
        id="generator1",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=100,
        target_q=150,
        target_v=225,
        bus_or_busbar_section_id="VL2_1_1",
        position_order=5,
    )

    pypowsybl.network.create_load_bay(net, id="load1", p0=150, q0=225, bus_or_busbar_section_id="VL1_2_1", position_order=5)
    net.create_internal_connections(voltage_level_id="VL1", node1=39, node2=80)
    net.create_loads(id="load1_2", voltage_level_id="VL1", node=80, p0=10, q0=3)

    # add an empty bay to VL2
    net.remove_elements(["L11"])
    # create an empty bay with not BREAKER
    net.remove_elements(["L112_BREAKER"])

    # add a random switch to left side of a busbar coupler
    net.create_switches(id="DS", voltage_level_id="VL1", node1=15, node2=70, kind="DISCONNECTOR", open=False)
    net.remove_elements(["VL1_BREAKER"])
    net.create_switches(id="VL1_BREAKER", voltage_level_id="VL1", node1=16, node2=70, kind="BREAKER", open=False)
    # add a random second bay to L10
    net.create_switches(
        id="L10_DISCONNECTOR_BUS7", voltage_level_id="VL1", node1=37, node2=7, kind="DISCONNECTOR", open=True
    )

    return net


@pytest.fixture(scope="function")
def basic_node_breaker_network_powsybl_not_disconnectable():
    network = basic_node_breaker_network_powsybl()
    lines = pd.DataFrame.from_records(
        data=[
            {
                "node1": 51,
                "node2": 51,
                "voltage_level2_id": "VL1",
                "id": "not_disconnectable_line",
                "voltage_level1_id": "VL2",
                "r": 0.1,
                "x": 10,
                "g1": 0,
                "b1": 0,
                "g2": 0,
                "b2": 0,
            },
        ]
    )
    lines = lines.set_index("id")
    network.create_lines(lines)
    return network
