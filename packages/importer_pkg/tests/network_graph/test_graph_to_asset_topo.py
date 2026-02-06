# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logbook
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from toop_engine_importer.network_graph.data_classes import NetworkGraphData, SubstationInformation
from toop_engine_importer.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_importer.network_graph.graph_to_asset_topo import (
    get_asset_bay,
    get_busbar_df,
    get_coupler_df,
    get_dv_switch,
    get_sl_switch,
    get_state_of_coupler_based_on_bay,
    get_station_connection_tables,
    get_switchable_asset,
    remove_double_connections,
    select_one_busbar_for_coupler_side,
)
from toop_engine_importer.network_graph.network_graph import (
    generate_graph,
    get_busbar_connection_info,
    get_edge_connection_info,
)
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data
from toop_engine_importer.network_graph.powsybl_station_to_graph import get_station
from toop_engine_interfaces.asset_topology import AssetBay


def test_remove_double_connections():
    with logbook.handlers.TestHandler() as caplog:
        # Test case 1: No double connections
        switching_table = np.array(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        result = remove_double_connections(switching_table)
        assert np.array_equal(result, switching_table), f"Expected {expected_result}, but got {result}"

        # Test case 2: Double connections
        switching_table = np.array(
            [[True, True, False, True, False], [False, True, True, True, False], [True, False, True, True, False]]
        )
        expected_result = np.array(
            [[True, True, False, True, False], [False, False, True, False, False], [False, False, False, False, False]]
        )
        result = remove_double_connections(switching_table)
        assert "Double connections in the switching table detected and removed" in "".join(caplog.formatted_records)
        assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

        # Test case 3: All false
        switching_table = np.array([[False, False, False], [False, False, False], [False, False, False]])
        expected_result = np.array([[False, False, False], [False, False, False], [False, False, False]])
        result = remove_double_connections(switching_table)
        assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

        # Test case 4: Mixed connections
        switching_table = np.array([[True, False, True], [True, True, False], [False, True, True]])
        expected_result = np.array([[True, False, True], [False, True, False], [False, False, False]])
        result = remove_double_connections(switching_table, substation_id="test")
        assert "Double connections in the switching table detected and removed. Station: test" in "".join(
            caplog.formatted_records
        )
        assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_get_busbar_df(network_graph_data_test1: NetworkGraphData):
    graph = generate_graph(network_graph_data_test1)
    nodes = network_graph_data_test1.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)
    expected = [
        {
            "grid_model_id": "BBS3_1",
            "type": "busbar",
            "name": "ab",
            "int_id": 0,
            "in_service": True,
            "bus_branch_bus_id": "BBS3_1_bus_id",
        },
        {
            "grid_model_id": "BBS3_2",
            "type": "busbar",
            "name": "cd",
            "int_id": 1,
            "in_service": True,
            "bus_branch_bus_id": "BBS3_2_bus_id",
        },
    ]
    assert busbar_df.to_dict(orient="records") == expected


def test_get_coupler_df_busbar_coupler(network_graph_for_asset_topo: tuple[nx.Graph, NetworkGraphData]):
    graph, network_graph_data = network_graph_for_asset_topo
    switches_df = network_graph_data.switches
    nodes = network_graph_data.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)
    expected = [
        {
            "grid_model_id": "5",
            "type": "BREAKER",
            "name": "fid_5",
            "in_service": True,
            "open": False,
            "busbar_from_id": 1,
            "busbar_to_id": 0,
        }
    ]
    res = get_coupler_df(switches_df, busbar_df, substation_id, graph=graph)
    assert res.to_dict(orient="records") == expected

    # test empty coupler_df
    switches_df = switches_df.iloc[0:1]
    res = get_coupler_df(switches_df, busbar_df, substation_id, graph=graph)
    assert res.empty


def test_get_switchable_asset(network_graph_for_asset_topoV2_S1: tuple[nx.Graph, NetworkGraphData]):
    graph, network_graph_data = network_graph_for_asset_topoV2_S1
    nodes_asset_df = network_graph_data.node_assets
    bus_info = get_busbar_connection_info(graph=graph)
    branches_df = network_graph_data.branches
    expected = [
        {"grid_model_id": "L1", "name": "", "type": "LINE", "in_service": True},
        {"grid_model_id": "L2", "name": "", "type": "LINE", "in_service": True},
        {"grid_model_id": "L3", "name": "", "type": "LINE", "in_service": True},
        {"grid_model_id": "L4", "name": "", "type": "LINE", "in_service": True},
        {"grid_model_id": "L5", "name": "", "type": "LINE", "in_service": True},
        {"grid_model_id": "generator1", "name": "", "type": "GENERATOR", "in_service": True},
        {"grid_model_id": "generator2", "name": "", "type": "GENERATOR", "in_service": True},
    ]
    res = get_switchable_asset(busbar_connection_info=bus_info, node_assets_df=nodes_asset_df, branches_df=branches_df)
    assert res.to_dict(orient="records") == expected


def test_switching_tables_V2(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    station_info = {"name": "Station5", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL5"}
    station_info = SubstationInformation(**station_info)
    station = get_station(net, "VL5_0", station_info)
    asset_connectivity = np.array([[True, True, True], [True, True, True], [True, True, True]])
    asset_switching_table = np.array([[True, True, False], [False, False, False], [False, False, True]])

    assert np.array_equal(station.asset_connectivity, asset_connectivity)
    assert np.array_equal(station.asset_switching_table, asset_switching_table)
    assert len(station.couplers) == 2
    assert station.couplers[0].type == "BREAKER"
    assert station.couplers[1].type == "BREAKER"

    station_info = {"name": "Station6", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL6"}
    station_info = SubstationInformation(**station_info)
    station = get_station(net, "VL6_0", station_info)

    asset_switching_table = np.array(
        [[True, True], [False, False], [False, False], [False, False], [False, False], [False, False]]
    )

    asset_connectivity = np.array(
        [[True, True], [False, False], [False, False], [True, True], [False, False], [False, False]]
    )

    assert np.array_equal(station.asset_connectivity, asset_connectivity)
    assert np.array_equal(station.asset_switching_table, asset_switching_table)


def test_station_coupler(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2

    station_info = {"name": "Station2", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL2"}
    station_info = SubstationInformation(**station_info)
    # Voltage level 2
    station = get_station(net, "VL2_0", station_info)

    assert len(station.couplers) == 2
    assert station.couplers[0].grid_model_id == "VL2_BREAKER"
    assert station.couplers[0].type == "BREAKER"
    assert station.couplers[0].name == "VL2_BREAKER"
    assert station.couplers[0].busbar_from_id == 0
    assert station.couplers[0].busbar_to_id == 1

    assert station.couplers[1].grid_model_id == "VL2_BREAKER#0"
    assert station.couplers[1].type == "BREAKER"
    assert station.couplers[1].name == "VL2_BREAKER#0"
    assert station.couplers[1].busbar_from_id == 1
    assert station.couplers[1].busbar_to_id == 2

    station_info = {"name": "Station6", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL6"}
    station_info = SubstationInformation(**station_info)
    # Voltage level 6
    station = get_station(net, "VL6_0", station_info)

    assert len(station.couplers) == 5

    assert station.couplers[0].grid_model_id == "VL6_BREAKER"
    assert station.couplers[0].type == "BREAKER"
    assert station.couplers[0].name == "VL6_BREAKER"
    assert station.couplers[0].busbar_from_id == 1
    assert station.couplers[0].busbar_to_id == 4
    assert not station.couplers[0].open

    assert station.couplers[1].grid_model_id == "VL6_BREAKER_1_1"
    assert station.couplers[1].type == "BREAKER"
    assert station.couplers[1].name == "VL6_BREAKER_1_1"
    assert station.couplers[1].busbar_from_id == 0
    assert station.couplers[1].busbar_to_id == 1
    assert not station.couplers[1].open
    assert station.couplers[1].in_service

    assert station.couplers[2].grid_model_id == "VL6_BREAKER_2_1"
    assert station.couplers[2].type == "BREAKER"
    assert station.couplers[2].name == "VL6_BREAKER_2_1"
    assert station.couplers[2].busbar_from_id == 3
    assert station.couplers[2].busbar_to_id == 4
    assert not station.couplers[2].open
    assert station.couplers[2].in_service

    assert station.couplers[3].grid_model_id == "VL6_DISCONNECTOR_2_4"
    assert station.couplers[3].type == "DISCONNECTOR"
    assert station.couplers[3].name == "VL6_DISCONNECTOR_2_4"
    assert station.couplers[3].busbar_from_id == 1
    assert station.couplers[3].busbar_to_id == 2

    assert station.couplers[4].grid_model_id == "VL6_DISCONNECTOR_3_5"
    assert station.couplers[4].type == "DISCONNECTOR"
    assert station.couplers[4].name == "VL6_DISCONNECTOR_3_5"
    assert station.couplers[4].busbar_from_id == 4
    assert station.couplers[4].busbar_to_id == 5

    # Voltage level 4
    station_info = {"name": "Station4", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL4"}
    station_info = SubstationInformation(**station_info)
    station = get_station(net, "VL4_0", station_info)
    assert len(station.couplers) == 3
    assert station.couplers[0].grid_model_id == "VL4_BREAKER"
    assert station.couplers[0].type == "BREAKER"
    assert station.couplers[0].name == "VL4_BREAKER"
    assert station.couplers[0].busbar_from_id == 0
    assert station.couplers[0].busbar_to_id == 1

    assert station.couplers[1].grid_model_id == "VL4_BREAKER#0"
    assert station.couplers[1].type == "BREAKER"
    assert station.couplers[1].name == "VL4_BREAKER#0"
    assert station.couplers[1].busbar_from_id == 0
    assert station.couplers[1].busbar_to_id == 2

    assert station.couplers[2].grid_model_id == "VL4_BREAKER#1"
    assert station.couplers[2].type == "BREAKER"
    assert station.couplers[2].name == "VL4_BREAKER#1"
    assert station.couplers[2].busbar_from_id == 0
    assert station.couplers[2].busbar_to_id == 3


@pytest.mark.xfail(reason="Failing edge case, there are no lines connected to the busbars in the middle")
def test_switching_tables_failing_edgecase(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    # net.get_single_line_diagram('VL5')

    station = get_station(net, "VL6_0", {"voltage_level_id": "VL6", "region": "BE", "nominal_v": 380, "name": "Station6"})
    assert station.couplers[0].model_dump() == {
        "grid_model_id": "VL6_BREAKER",
        "type": "busbar_coupler",
        "name": "VL6_BREAKER",
        "busbar_from_id": 4,
        "busbar_to_id": 1,
        "open": False,
        "in_service": True,
    }
    # see reason with: net.get_single_line_diagram('VL6')
    # not sure how to solve this edge case


def test_asset_bay(network_graph_for_asset_topoV2_S3: tuple[nx.Graph, NetworkGraphData]):
    graph, network_graph_data = network_graph_for_asset_topoV2_S3
    nodes = network_graph_data.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)

    nodes_asset_df = network_graph_data.node_assets
    switches_df = network_graph_data.switches
    bus_info = get_busbar_connection_info(graph=graph)
    edge_connection_info = get_edge_connection_info(graph=graph)
    branches_df = network_graph_data.branches
    switchable_assets_df = get_switchable_asset(
        busbar_connection_info=bus_info, node_assets_df=nodes_asset_df, branches_df=branches_df
    )
    expected = {
        "L3": AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="L32_BREAKER",
            sr_switch_grid_model_id={"BBS3_1": "L32_DISCONNECTOR_5_0", "BBS3_2": "L32_DISCONNECTOR_5_1"},
        ),
        "L6": AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="L62_BREAKER",
            sr_switch_grid_model_id={"BBS3_1": "L62_DISCONNECTOR_7_0", "BBS3_2": "L62_DISCONNECTOR_7_1"},
        ),
        "L7": AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="L72_BREAKER",
            sr_switch_grid_model_id={"BBS3_1": "L72_DISCONNECTOR_9_0", "BBS3_2": "L72_DISCONNECTOR_9_1"},
        ),
        "L9": AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="L91_BREAKER",
            sr_switch_grid_model_id={"BBS3_1": "L91_DISCONNECTOR_11_0", "BBS3_2": "L91_DISCONNECTOR_11_1"},
        ),
        "load2": AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="load2_BREAKER",
            sr_switch_grid_model_id={"BBS3_1": "load2_DISCONNECTOR_19_0", "BBS3_2": "load2_DISCONNECTOR_19_1"},
        ),
    }
    asset_bay_dict = {}
    station_logs = []
    for asset_grid_model_id in switchable_assets_df["grid_model_id"].to_list():
        asset_bay, logs = get_asset_bay(
            network_graph_data.switches,
            asset_grid_model_id=asset_grid_model_id,
            busbar_df=busbar_df,
            edge_connection_info=edge_connection_info,
        )
        asset_bay_dict[asset_grid_model_id] = asset_bay
        station_logs.extend(logs)
    assert asset_bay_dict == expected
    assert len(station_logs) == 0

    switches_df.loc[0, "asset_type"] = "NOT_VALID"
    with pytest.raises(ValueError, match="Expected 3 switches, but got"):
        get_asset_bay(
            switches_df=switches_df,
            asset_grid_model_id="L3",
            busbar_df=busbar_df,
            edge_connection_info=edge_connection_info,
        )
    switches_df.loc[0, "asset_type"] = "BREAKER"
    switches_df.loc[1, "asset_type"] = "BREAKER"
    switches_df.loc[2, "asset_type"] = "BREAKER"

    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        asset_grid_model_id="L3",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    expected = AssetBay(
        sl_switch_grid_model_id=None,
        dv_switch_grid_model_id="L32_BREAKER",
        sr_switch_grid_model_id={"BBS3_1": "L32_DISCONNECTOR_5_0", "BBS3_2": "L32_DISCONNECTOR_5_1"},
    )
    assert asset_grid_model_id == expected
    assert logs == [
        "Warning: There is a BREAKER directly connected to a busbar ['L32_DISCONNECTOR_5_0', 'L32_DISCONNECTOR_5_1'] Will be modelled as sr switch. grid_model_id: L32_DISCONNECTOR_5_0"
    ]

    switches_df.drop(1, inplace=True)
    switches_df.drop(2, inplace=True)
    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        asset_grid_model_id="L3",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    assert asset_grid_model_id is None
    assert logs == ["Warning: There should be at least one sr switch but got 0, AssetBay ignored for grid_model_id: L3"]

    edge_connection_info["L62_DISCONNECTOR_7_0"].direct_busbar_grid_model_id = ""
    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        asset_grid_model_id="L6",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    expected = AssetBay(
        sl_switch_grid_model_id="L62_DISCONNECTOR_7_0",
        dv_switch_grid_model_id="L62_BREAKER",
        sr_switch_grid_model_id={"BBS3_2": "L62_DISCONNECTOR_7_1"},
    )
    assert asset_grid_model_id == expected
    assert len(logs) == 0


def test_switching_table(network_graph_data_test1: NetworkGraphData):
    network_graph_data = network_graph_data_test1
    add_graph_specific_data(network_graph_data)
    graph = generate_graph(network_graph_data)
    run_default_filter_strategy(graph)

    nodes = network_graph_data_test1.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)

    nodes_asset_df = network_graph_data_test1.node_assets
    bus_info = get_busbar_connection_info(graph=graph)
    # bus_info = network_graph_test1.get_busbar_connection_info()
    branches_df = network_graph_data_test1.branches
    switchable_assets_df = get_switchable_asset(
        busbar_connection_info=bus_info, node_assets_df=nodes_asset_df, branches_df=branches_df
    )
    (switching_table_asset_physically, switching_table_asset, switching_table_busbar_physically, switching_table_busbar) = (
        get_station_connection_tables(bus_info, busbar_df=busbar_df, switchable_assets_df=switchable_assets_df)
    )
    assert switching_table_asset_physically.all().all()
    array_compare = np.ones((2, 2), dtype=bool)
    np.fill_diagonal(array_compare, False)
    assert np.array_equal(switching_table_busbar, array_compare)
    assert np.array_equal(switching_table_busbar_physically, array_compare)
    switching_compare = np.array([[True, True, False, True, False], [False, False, True, False, True]])
    assert np.array_equal(switching_table_asset, switching_compare)


def test_get_sl_switch():
    # Test case 1: No sl_switch found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["busbar1", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_sl_switch = get_sl_switch(asset_bays_df)
    assert result is None, f"Expected None, but got {result}"
    assert n_sl_switch == 0
    assert logs == []

    # Test case 2: One sl_switch found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_sl_switch = get_sl_switch(asset_bays_df)
    assert result == "id1", f"Expected 'switch1', but got {result}"
    assert n_sl_switch == 1
    assert logs == []

    # Test case 3: Multiple sl_switches found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, True],
        }
    )
    result, logs, n_sl_switch = get_sl_switch(asset_bays_df)
    assert result == "id2"  # Expectes the open switch
    assert n_sl_switch == 2
    assert "There should be maximum one sl_switch but got 2" in logs[0]

    # Test case 4: No DISCONNECTOR type
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["BREAKER", "BREAKER"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_sl_switch = get_sl_switch(asset_bays_df)
    assert result is None, f"Expected None, but got {result}"
    assert n_sl_switch == 0
    assert logs == []


def test_get_dv_switch(caplog):
    # Test case 1: No dv_switch found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["busbar1", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_dv_sw_found = get_dv_switch(asset_bays_df, "asset1")
    assert result == "", f"Expected '', but got {result}"
    assert logs == [
        "Warning:There should be exactly one dv switch but got '0', dv switch_id is left empty for grid_model_id: asset1, grid_model_id of first bay switch: id1"
    ]
    assert n_dv_sw_found == 0
    # Test case 2: One dv_switch found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["BREAKER", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_dv_sw_found = get_dv_switch(asset_bays_df, "asset2")
    assert result == "id1", f"Expected 'switch1', but got {result}"
    assert len(logs) == 0
    assert n_dv_sw_found == 1

    # Test case 3: Multiple dv_switches found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["BREAKER", "BREAKER"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_dv_sw_found = get_dv_switch(asset_bays_df, "asset3")
    assert result == "id1", f"Expected 'switch1', but got {result}"
    assert logs == [
        "Warning: There should be exactly one dv switch but got '2' with grid_model_id ['id1', 'id2']",
        "Selecting the first Switch. grid_model_id: id1",
    ]
    assert n_dv_sw_found == 2

    # Test case 3: Multiple dv_switches found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["BREAKER", "BREAKER"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, True],
        }
    )
    result, logs, n_dv_sw_found = get_dv_switch(asset_bays_df, "asset3")
    assert result == "id1", f"Expected 'switch1', but got {result}"
    assert logs == [
        "Warning: There should be exactly one dv switch but got '2' with grid_model_id ['id1', 'id2']",
        "Selecting the first open Switch. grid_model_id: id2",
    ]
    assert n_dv_sw_found == 2

    # Test case 4: No BREAKER type
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_dv_sw_found = get_dv_switch(asset_bays_df, "asset4")
    assert result == "", f"Expected '', but got {result}"
    assert logs == [
        "Warning:There should be exactly one dv switch but got '0', dv switch_id is left empty for grid_model_id: asset4, grid_model_id of first bay switch: id1"
    ]
    assert n_dv_sw_found == 0


def test_select_one_busbar_for_coupler_side():
    bay_df = pd.DataFrame.from_records(
        [
            {
                "grid_model_id": "VL2_BREAKER",
                "direct_busbar_grid_model_id": "",
                "open": False,
                "from_busbar_grid_model_ids": ["BBS2_1", "BBS2_2"],
                "from_coupler_ids": ["VL2_DISCONNECTOR_13_0", "VL2_DISCONNECTOR_13_1"],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_13_0",
                "direct_busbar_grid_model_id": "BBS2_1",
                "open": False,
                "from_busbar_grid_model_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_13_1",
                "direct_busbar_grid_model_id": "BBS2_2",
                "open": True,
                "from_busbar_grid_model_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_14_1",
                "direct_busbar_grid_model_id": "BBS2_2",
                "open": False,
                "from_busbar_grid_model_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_14_2",
                "direct_busbar_grid_model_id": "BBS2_3",
                "open": True,
                "from_busbar_grid_model_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_BREAKER",
                "direct_busbar_grid_model_id": "",
                "open": False,
                "from_busbar_grid_model_ids": ["BBS2_1", "BBS2_2", "BBS2_3"],
                "from_coupler_ids": ["VL2_DISCONNECTOR_13_0", "VL2_DISCONNECTOR_13_1", "VL2_DISCONNECTOR_14_2"],
            },
        ]
    )
    coupler_index = 0
    out_of_service_busbar_ids = []

    res = select_one_busbar_for_coupler_side(
        bay_df=bay_df,
        coupler_index=coupler_index,
        side="from",
        out_of_service_busbar_ids=out_of_service_busbar_ids,
    )
    assert res == "BBS2_1", f"Expected 'BBS2_1', but got {res}"

    # test ignore busbar id
    ignore_id = "BBS2_1"
    res = select_one_busbar_for_coupler_side(
        bay_df=bay_df,
        coupler_index=coupler_index,
        side="from",
        out_of_service_busbar_ids=out_of_service_busbar_ids,
        ignore_busbar_id=ignore_id,
    )
    assert res == "BBS2_2", f"Expected 'BBS2_2', but got {res}"

    # test out of service busbar id with second to ignore
    ignore_id = "BBS2_2"
    out_of_service_busbar_ids = ["BBS2_1"]
    res = select_one_busbar_for_coupler_side(
        bay_df=bay_df,
        coupler_index=coupler_index,
        side="from",
        out_of_service_busbar_ids=out_of_service_busbar_ids,
        ignore_busbar_id=ignore_id,
    )
    assert res == "BBS2_1", f"Expected 'BBS2_1', but got {res}"

    # test out of service busbar id with second to ignore
    coupler_index = len(bay_df) - 1
    ignore_id = "BBS2_1"
    out_of_service_busbar_ids = ["BBS2_1", "BBS2_2"]
    res = select_one_busbar_for_coupler_side(
        bay_df=bay_df,
        coupler_index=coupler_index,
        side="from",
        out_of_service_busbar_ids=out_of_service_busbar_ids,
        ignore_busbar_id=ignore_id,
    )
    assert res == "BBS2_3", f"Expected 'BBS2_1', but got {res}"

    with pytest.raises(ValueError, match="Coupler has no busbar grid model id"):
        select_one_busbar_for_coupler_side(
            bay_df=bay_df,
            coupler_index=1,
            side="from",
            out_of_service_busbar_ids=out_of_service_busbar_ids,
            ignore_busbar_id=ignore_id,
        )


def test_get_state_of_coupler_based_on_bay():
    # Case 1: All from_switches open
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw",
                "open": True,
                "from_coupler_ids": ["sw1", "sw2"],
                "to_coupler_ids": ["sw3", "sw4"],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw1",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
            {
                "grid_model_id": "sw2",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
            {
                "grid_model_id": "sw3",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS2",
            },
            {
                "grid_model_id": "sw4",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS2",
            },
        ]
    )
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 2: All to_switches open right, else closed
    bay_df["open"] = [False, False, False, True, True]
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 3: All to_switches open left, else closed
    bay_df["open"] = [False, True, True, False, False]
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 4: All to_switches open left, main witch open, else closed
    bay_df["open"] = [True, True, True, False, False]
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 4: A switch is open
    bay_df["open"] = [False, True, False, True, False]
    assert not get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 5: A switch is open + main switch open
    bay_df["open"] = [True, True, False, True, False]
    assert not get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 6: Empty from_switches and to_switches
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw1",
                "open": False,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
        ]
    )
    assert not get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 7: Empty one side
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw",
                "open": False,
                "from_coupler_ids": ["sw1"],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw1",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
        ]
    )
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 8: Empty one side
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": ["sw1"],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw1",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
        ]
    )
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 8: Empty one side
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw",
                "open": False,
                "from_coupler_ids": [],
                "to_coupler_ids": ["sw1"],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw1",
                "open": False,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
        ]
    )
    assert not get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 8: from_switches and to_switches refer to non-existent switches (should not fail)
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw1",
                "open": False,
                "from_coupler_ids": ["swX"],
                "to_coupler_ids": ["swY"],
                "direct_busbar_grid_model_id": "",
            },
        ]
    )
    assert not get_state_of_coupler_based_on_bay(0, bay_df)

    # Case 9: one side has an sl_switch
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "sw",
                "open": True,
                "from_coupler_ids": ["sw1", "sw2", "sw2_sl"],
                "to_coupler_ids": ["sw3", "sw4"],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw1",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
            {
                "grid_model_id": "sw2",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS1",
            },
            {
                "grid_model_id": "sw2_sl",
                "open": False,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sw3",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS2",
            },
            {
                "grid_model_id": "sw4",
                "open": True,
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "BBS2",
            },
        ]
    )
    assert get_state_of_coupler_based_on_bay(0, bay_df)

    bay_df["open"] = [False, True, True, False, False, False]
    assert get_state_of_coupler_based_on_bay(0, bay_df)
    bay_df["open"] = [False, False, False, True, False, False]
    assert get_state_of_coupler_based_on_bay(0, bay_df)
    bay_df["open"] = [False, False, True, False, True, False]
    assert not get_state_of_coupler_based_on_bay(0, bay_df)
