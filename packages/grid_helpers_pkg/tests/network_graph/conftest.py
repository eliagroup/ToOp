# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import pandas as pd
import pypowsybl
import pytest
from pypowsybl.network import Network
from toop_engine_grid_helpers.network_graph.data_classes import BranchSchema, NetworkGraphData, SubstationInformation
from toop_engine_grid_helpers.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_grid_helpers.network_graph.network_graph import generate_graph
from toop_engine_grid_helpers.network_graph.network_graph_data import add_graph_specific_data, remove_helper_branches
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    basic_node_breaker_network_powsybl_v2,
)
from toop_engine_grid_helpers.powsybl.powsybl_station_to_graph import (
    get_node_breaker_topology_graph,
    node_breaker_topology_to_graph_data,
)


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
def network_graph_data_test1(get_graph_input_dicts: tuple[dict, dict, dict]) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    switches_df = pd.DataFrame(switches_dict)
    nodes_assets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_assets_df["in_service"] = True

    return NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_assets_df)


@pytest.fixture(scope="function")
def basic_node_breaker_network_powsybl_grid() -> Network:
    return basic_node_breaker_network_powsybl()


@pytest.fixture(scope="function")
def basic_node_breaker_network_powsybl_grid_v2() -> Network:
    return basic_node_breaker_network_powsybl_v2()


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
def network_graph_data_test2_helper_branches(
    get_graph_input_dicts_helper_branches: tuple[dict, dict, dict, dict],
) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_assets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_assets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""

    return NetworkGraphData(
        nodes=nodes_df,
        switches=switches_df,
        node_assets=nodes_assets_df,
        helper_branches=helper_branches_df,
    )


@pytest.fixture(scope="session")
def network_graph_data_test2_helper_branches_removed(
    get_graph_input_dicts_helper_branches: tuple[dict, dict, dict, dict],
) -> NetworkGraphData:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_assets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_assets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""
    branches_df = pd.DataFrame(columns=list(BranchSchema.to_schema().columns.keys()))

    remove_helper_branches(
        nodes_df=nodes_df,
        helper_branches_df=helper_branches_df,
        node_assets_df=nodes_assets_df,
        switches_df=switches_df,
        branches_df=branches_df,
    )
    return NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_assets_df)


@pytest.fixture(scope="session")
def network_graph_for_asset_topo(
    get_graph_input_dicts_helper_branches: tuple[dict, dict, dict, dict],
) -> tuple[nx.Graph, NetworkGraphData]:
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_assets_df = pd.DataFrame(node_assets_dict)
    nodes_assets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""

    network_graph_data = NetworkGraphData(
        nodes=nodes_df,
        switches=switches_df,
        node_assets=nodes_assets_df,
        helper_branches=helper_branches_df,
    )
    add_graph_specific_data(network_graph_data)
    graph = generate_graph(network_graph_data)
    run_default_filter_strategy(graph=graph)
    return graph, network_graph_data


@pytest.fixture(scope="function")
def network_graph_for_asset_topoV2_S1(
    basic_node_breaker_network_powsybl_grid_v2: Network,
) -> tuple[nx.Graph, NetworkGraphData]:
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    network_graph_data = node_breaker_topology_to_graph_data(
        basic_node_breaker_network_powsybl_grid_v2,
        substation_information,
    )
    graph = get_node_breaker_topology_graph(network_graph_data)
    return graph, network_graph_data


@pytest.fixture(scope="function")
def network_graph_for_asset_topoV2_S3(
    basic_node_breaker_network_powsybl_grid_v2: Network,
) -> tuple[nx.Graph, NetworkGraphData]:
    substation_dict = {"name": "Station3", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL3"}
    substation_information = SubstationInformation(**substation_dict)
    network_graph_data = node_breaker_topology_to_graph_data(
        basic_node_breaker_network_powsybl_grid_v2,
        substation_information,
    )
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
