# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import structlog.testing
from toop_engine_grid_helpers.network_graph.data_classes import EdgeConnectionInfo, NetworkGraphData, SubstationInformation
from toop_engine_grid_helpers.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_grid_helpers.network_graph.graph_to_asset_topo import (
    _build_coupler_bay_payload,
    _get_coupler_side_switches,
    get_asset_bay,
    get_asset_disconnector,
    get_busbar_df,
    get_coupler_df,
    get_dv_switch,
    get_state_of_coupler_based_on_bay,
    get_station_connection_tables,
    get_switchable_asset,
    remove_double_connections,
    select_one_busbar_for_coupler_side,
)
from toop_engine_grid_helpers.network_graph.network_graph import (
    generate_graph,
    get_busbar_connection_info,
    get_edge_connection_info,
)
from toop_engine_grid_helpers.network_graph.network_graph_data import add_graph_specific_data
from toop_engine_grid_helpers.powsybl.powsybl_station_to_graph import (
    _build_master_bus_group_from_busbar_group,
    _build_station_connectivity_by_asset_type,
    _get_station_asset_connections,
    _get_station_busbar_view,
    _get_station_topology_frames,
    get_node_breaker_topology_graph,
    node_breaker_topology_to_graph_data,
)
from toop_engine_interfaces.asset_topology.assets import AssetBay, build_asset_bay_id


def build_station_from_bus_id(net, bus_id: str, station_info: SubstationInformation):
    """Build one canonical station view from a selected bus id in the graph-based helper tests."""
    graph_data = node_breaker_topology_to_graph_data(net, substation_info=station_info)
    graph = get_node_breaker_topology_graph(graph_data)
    _busbar_df, selected_busbar_ids, _busbar_connection_info = _get_station_busbar_view(
        graph=graph,
        graph_data=graph_data,
        bus_id=bus_id,
        substation_id=station_info.name,
    )
    station, _branch_assets, _injection_assets, _asset_bays = _build_master_bus_group_from_busbar_group(
        network=net,
        station_info=station_info,
        selected_busbar_ids=selected_busbar_ids,
        station_grid_model_id=bus_id,
    )
    busbar_df, _coupler_df, busbar_connection_info, switchable_assets_df, asset_bays_by_asset_id, _station_logs = (
        _get_station_topology_frames(
            network=net,
            station_info=station_info,
            selected_busbar_ids=selected_busbar_ids,
            station_grid_model_id=bus_id,
        )
    )
    _branch_assets, _injection_assets, _branch_connections, _injection_connections, branch_mask, _asset_bays = (
        _get_station_asset_connections(
            network=net,
            station_info=station_info,
            busbar_df=busbar_df,
            switchable_assets_df=switchable_assets_df,
            asset_bays_by_asset_id=asset_bays_by_asset_id,
        )
    )
    _branch_connectivity, _injection_connectivity = _build_station_connectivity_by_asset_type(
        busbar_connection_info=busbar_connection_info,
        busbar_df=busbar_df,
        switchable_assets_df=switchable_assets_df,
        branch_mask=branch_mask,
    )
    _asset_connectivity, asset_switching_table, _busbar_connectivity, _busbar_switching_table = (
        get_station_connection_tables(
            busbar_connection_info,
            busbar_df=busbar_df,
            switchable_assets_df=switchable_assets_df,
        )
    )
    asset_switching_table = remove_double_connections(asset_switching_table, substation_id=station_info.name)
    branch_switching_table = asset_switching_table[:, branch_mask]
    injection_switching_table = asset_switching_table[:, [not is_branch for is_branch in branch_mask]]
    return station, branch_switching_table, injection_switching_table


def test_remove_double_connections():
    with structlog.testing.capture_logs() as cap_logs:
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
        assert any("Double connections in the switching table detected and removed" in e["event"] for e in cap_logs)
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
        assert any(
            "Double connections in the switching table detected and removed. Station: test" in e["event"] for e in cap_logs
        )
        assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_build_coupler_bay_payload_collects_multiple_internal_switches() -> None:
    """Verify that multi-switch coupler paths keep all internal breakers and disconnectors."""
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "dv-a",
                "asset_type": "BREAKER",
                "from_busbar_ids": ["bb1", "bb2"],
                "to_busbar_ids": ["bb3", "bb4"],
                "from_coupler_ids": ["sel-bb1", "sel-bb2"],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "dv-b",
                "asset_type": "BREAKER",
                "from_busbar_ids": [],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "mid-d1",
                "asset_type": "DISCONNECTOR",
                "from_busbar_ids": [],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "mid-d2",
                "asset_type": "DISCONNECTOR",
                "from_busbar_ids": [],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "sel-bb1",
                "asset_type": "DISCONNECTOR",
                "from_busbar_ids": ["bb1"],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "bb1",
            },
            {
                "grid_model_id": "sel-bb2",
                "asset_type": "DISCONNECTOR",
                "from_busbar_ids": ["bb2"],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "bb2",
            },
        ]
    ).set_index("grid_model_id", drop=False)

    coupler_bay = _build_coupler_bay_payload(coupler_index="dv-a", bay_df=bay_df)

    assert coupler_bay["coupler_breaker_ids"] == ["dv-a", "dv-b"]
    assert coupler_bay["coupler_disconnector_ids"] == ["mid-d1", "mid-d2"]
    assert coupler_bay["from_busbar_ids"] == ["bb1", "bb2"]
    assert coupler_bay["to_busbar_ids"] == ["bb3", "bb4"]
    assert coupler_bay["from_busbar_disconnector_ids"] == {"bb1": "sel-bb1", "bb2": "sel-bb2"}
    assert coupler_bay["to_busbar_disconnector_ids"] == {}


def test_get_coupler_side_switches_keeps_single_selector_with_empty_placeholder() -> None:
    """Keep the selector for a single-busbar coupler side despite an empty path entry."""
    bay_df = pd.DataFrame(
        [
            {
                "grid_model_id": "breaker",
                "from_busbar_ids": ["from-busbar"],
                "to_busbar_ids": ["to-busbar"],
                "from_coupler_ids": ["from-selector"],
                "to_coupler_ids": ["to-selector", ""],
                "direct_busbar_grid_model_id": "",
            },
            {
                "grid_model_id": "to-selector",
                "from_busbar_ids": [],
                "to_busbar_ids": [],
                "from_coupler_ids": [],
                "to_coupler_ids": [],
                "direct_busbar_grid_model_id": "to-busbar",
            },
        ]
    ).set_index("grid_model_id", drop=False)

    assert _get_coupler_side_switches(coupler_index="breaker", bay_df=bay_df, side="to") == {"to-busbar": "to-selector"}


def test_get_busbar_df(network_graph_data_test1: NetworkGraphData):
    graph = generate_graph(network_graph_data_test1)
    nodes = network_graph_data_test1.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)
    expected = [
        {
            "grid_model_id": "BBS3_1",
            "busbar_type": "busbar",
            "name": "ab",
            "int_id": 0,
            "in_service": True,
            "bus_branch_bus_id": "BBS3_1_bus_id",
            "bus_breaker_bus_id": None,
        },
        {
            "grid_model_id": "BBS3_2",
            "busbar_type": "busbar",
            "name": "cd",
            "int_id": 1,
            "in_service": True,
            "bus_branch_bus_id": "BBS3_2_bus_id",
            "bus_breaker_bus_id": None,
        },
    ]
    assert busbar_df.to_dict(orient="records") == expected


def test_get_coupler_df_busbar_coupler(network_graph_for_asset_topo: tuple[nx.Graph, NetworkGraphData]):
    graph, network_graph_data = network_graph_for_asset_topo
    switches_df = network_graph_data.switches
    nodes = network_graph_data.nodes
    substation_id = graph.nodes[0]["substation_id"]
    busbar_df = get_busbar_df(nodes, substation_id)
    res = get_coupler_df(switches_df, busbar_df, substation_id, graph=graph)
    assert res[["grid_model_id", "coupler_type", "name", "in_service", "open", "busbar_from_id", "busbar_to_id"]].to_dict(
        orient="records"
    ) == [
        {
            "grid_model_id": "5",
            "coupler_type": "BREAKER",
            "name": "fid_5",
            "in_service": True,
            "open": False,
            "busbar_from_id": 1,
            "busbar_to_id": 0,
        }
    ]
    assert res.loc[0, "coupler_bay"]["coupler_breaker_ids"] == ["5"]
    assert sorted(res.loc[0, "coupler_bay"]["coupler_disconnector_ids"]) == ["17", "34"]
    assert res.loc[0, "coupler_bay"]["connection_kind"] == "coupler"
    expected_from_busbar_grid_model_id = busbar_df.loc[res.loc[0, "busbar_from_id"], "grid_model_id"]
    expected_to_busbar_grid_model_id = busbar_df.loc[res.loc[0, "busbar_to_id"], "grid_model_id"]
    assert res.loc[0, "coupler_bay"]["from_busbar_ids"] == [expected_from_busbar_grid_model_id]
    assert res.loc[0, "coupler_bay"]["to_busbar_ids"] == [expected_to_busbar_grid_model_id]

    # test empty coupler_df
    switches_df = switches_df.iloc[0:1]
    res = get_coupler_df(switches_df, busbar_df, substation_id, graph=graph)
    assert res.empty


def test_get_coupler_df_ignores_empty_bay_disconnector_path() -> None:
    graph = nx.Graph()
    graph.add_node(1, substation_id="station", node_type="busbar", grid_model_id="bb1")
    graph.add_node(2, substation_id="station", node_type="busbar", grid_model_id="bb2")
    graph.add_node(3, substation_id="station", node_type="node", grid_model_id="n1")
    graph.add_node(4, substation_id="station", node_type="node", grid_model_id="n2")

    switches_df = pd.DataFrame(
        [
            {
                "grid_model_id": "ds_left",
                "foreign_id": "ds_left",
                "asset_type": "DISCONNECTOR",
                "substation_id": "station",
                "from_node": 1,
                "to_node": 3,
                "in_service": True,
                "open": False,
            },
            {
                "grid_model_id": "ds_middle",
                "foreign_id": "ds_middle",
                "asset_type": "DISCONNECTOR",
                "substation_id": "station",
                "from_node": 3,
                "to_node": 4,
                "in_service": True,
                "open": False,
            },
            {
                "grid_model_id": "ds_right",
                "foreign_id": "ds_right",
                "asset_type": "DISCONNECTOR",
                "substation_id": "station",
                "from_node": 4,
                "to_node": 2,
                "in_service": True,
                "open": False,
            },
        ]
    )

    graph.add_edge(
        1,
        3,
        grid_model_id="ds_left",
        asset_type="DISCONNECTOR",
        bay_id="ds_middle",
        bay_weight=10.0,
        empty_bay=True,
        edge_connection_info=EdgeConnectionInfo(),
        foreign_id="ds_left",
        in_service=True,
        open=False,
    )
    graph.add_edge(
        3,
        4,
        grid_model_id="ds_middle",
        asset_type="DISCONNECTOR",
        bay_id="ds_middle",
        bay_weight=10.0,
        empty_bay=True,
        edge_connection_info=EdgeConnectionInfo(
            coupler_type="busbar_coupler",
            from_busbar_ids=["bb1"],
            to_busbar_ids=["bb2"],
        ),
        foreign_id="ds_middle",
        in_service=True,
        open=False,
    )
    graph.add_edge(
        4,
        2,
        grid_model_id="ds_right",
        asset_type="DISCONNECTOR",
        bay_id="ds_middle",
        bay_weight=10.0,
        empty_bay=True,
        edge_connection_info=EdgeConnectionInfo(),
        foreign_id="ds_right",
        in_service=True,
        open=False,
    )

    nodes_df = pd.DataFrame(
        [
            {
                "grid_model_id": "bb1",
                "foreign_id": "bb1",
                "substation_id": "station",
                "node_type": "busbar",
                "bus_id": "bb1_bus",
                "in_service": True,
            },
            {
                "grid_model_id": "bb2",
                "foreign_id": "bb2",
                "substation_id": "station",
                "node_type": "busbar",
                "bus_id": "bb2_bus",
                "in_service": True,
            },
        ],
        index=[1, 2],
    )
    busbar_df = get_busbar_df(nodes_df, "station")

    res = get_coupler_df(switches_df, busbar_df, "station", graph=graph)

    assert res.empty


def test_get_switchable_asset(network_graph_for_asset_topoV2_S1: tuple[nx.Graph, NetworkGraphData]):
    graph, network_graph_data = network_graph_for_asset_topoV2_S1
    nodes_asset_df = network_graph_data.node_assets
    bus_info = get_busbar_connection_info(graph=graph)
    branches_df = network_graph_data.branches
    expected = [
        {"grid_model_id": "L1", "name": "", "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L2", "name": "", "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L3", "name": "", "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L4", "name": "", "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L5", "name": "", "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "generator1", "name": "", "asset_type": "GENERATOR", "in_service": True},
        {"grid_model_id": "generator2", "name": "", "asset_type": "GENERATOR", "in_service": True},
    ]
    res = get_switchable_asset(busbar_connection_info=bus_info, node_assets_df=nodes_asset_df, branches_df=branches_df)
    assert res.to_dict(orient="records") == expected


def test_switching_tables_V2(basic_node_breaker_network_powsybl_grid_v2):
    net = basic_node_breaker_network_powsybl_grid_v2
    station_info = {"name": "Station5", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL5"}
    station_info = SubstationInformation(**station_info)
    station, branch_switching_table, injection_switching_table = build_station_from_bus_id(net, "VL5_0", station_info)
    asset_connectivity = np.array([[True, True, True], [True, True, True], [True, True, True]])
    asset_switching_table = np.array([[True, True, False], [False, False, False], [False, False, True]])

    assert len(station.branch_connections) == 2
    assert len(station.injection_connections) == 1
    assert np.array_equal(
        np.concatenate([station.branch_connectivity, station.injection_connectivity], axis=1), asset_connectivity
    )
    assert np.array_equal(np.concatenate([branch_switching_table, injection_switching_table], axis=1), asset_switching_table)
    assert len(station.couplers) == 2
    assert station.couplers[0].coupler_type == "BREAKER"
    assert station.couplers[1].coupler_type == "BREAKER"

    station_info = {"name": "Station6", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL6"}
    station_info = SubstationInformation(**station_info)
    station, branch_switching_table, injection_switching_table = build_station_from_bus_id(net, "VL6_0", station_info)

    asset_switching_table = np.array(
        [[True, True], [False, False], [False, False], [False, False], [False, False], [False, False]]
    )

    asset_connectivity = np.array(
        [[True, True], [False, False], [False, False], [True, True], [False, False], [False, False]]
    )

    assert len(station.branch_connections) == 1
    assert len(station.injection_connections) == 1
    assert np.array_equal(
        np.concatenate([station.branch_connectivity, station.injection_connectivity], axis=1), asset_connectivity
    )
    assert np.array_equal(np.concatenate([branch_switching_table, injection_switching_table], axis=1), asset_switching_table)


def test_station_coupler(basic_node_breaker_network_powsybl_grid_v2):
    net = basic_node_breaker_network_powsybl_grid_v2

    def assert_coupler_bay(
        coupler,
        from_busbars: list[str],
        to_busbars: list[str],
        from_switches: dict[str, str],
        to_switches: dict[str, str],
        connection_kind: str = "coupler",
    ) -> None:
        assert coupler.coupler_bay is not None
        assert coupler.coupler_bay.connection_kind == connection_kind
        assert coupler.coupler_bay.from_busbar_ids == from_busbars
        assert coupler.coupler_bay.to_busbar_ids == to_busbars
        assert coupler.coupler_bay.from_busbar_disconnector_ids == from_switches
        assert coupler.coupler_bay.to_busbar_disconnector_ids == to_switches

    station_info = {"name": "Station2", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL2"}
    station_info = SubstationInformation(**station_info)
    # Voltage level 2
    station, _branch_switching_table, _injection_switching_table = build_station_from_bus_id(net, "VL2_0", station_info)

    assert len(station.couplers) == 2
    assert station.couplers[0].grid_model_id == "VL2_BREAKER"
    assert station.couplers[0].coupler_type == "BREAKER"
    assert station.couplers[0].name == "VL2_BREAKER"
    assert_coupler_bay(
        station.couplers[0],
        from_busbars=["BBS2_1", "BBS2_2"],
        to_busbars=["BBS2_2", "BBS2_3"],
        from_switches={"BBS2_1": "VL2_DISCONNECTOR_13_0", "BBS2_2": "VL2_DISCONNECTOR_13_1"},
        to_switches={"BBS2_2": "VL2_DISCONNECTOR_14_1", "BBS2_3": "VL2_DISCONNECTOR_14_2"},
    )

    assert station.couplers[1].grid_model_id == "VL2_BREAKER#0"
    assert station.couplers[1].coupler_type == "BREAKER"
    assert station.couplers[1].name == "VL2_BREAKER#0"
    assert_coupler_bay(
        station.couplers[1],
        from_busbars=["BBS2_1", "BBS2_2"],
        to_busbars=["BBS2_2", "BBS2_3"],
        from_switches={"BBS2_1": "VL2_DISCONNECTOR_15_0", "BBS2_2": "VL2_DISCONNECTOR_15_1"},
        to_switches={"BBS2_2": "VL2_DISCONNECTOR_16_1", "BBS2_3": "VL2_DISCONNECTOR_16_2"},
    )

    station_info = {"name": "Station6", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL6"}
    station_info = SubstationInformation(**station_info)
    # Voltage level 6
    station, _branch_switching_table, _injection_switching_table = build_station_from_bus_id(net, "VL6_0", station_info)

    assert len(station.couplers) == 5

    assert station.couplers[0].grid_model_id == "VL6_BREAKER"
    assert station.couplers[0].coupler_type == "BREAKER"
    assert station.couplers[0].name == "VL6_BREAKER"
    assert_coupler_bay(
        station.couplers[0],
        from_busbars=["VL6_1_2"],
        to_busbars=["VL6_2_2"],
        from_switches={"VL6_1_2": "VL6_DISCONNECTOR_10_2"},
        to_switches={"VL6_2_2": "VL6_DISCONNECTOR_11_3"},
    )

    assert station.couplers[1].grid_model_id == "VL6_BREAKER_1_1"
    assert station.couplers[1].coupler_type == "BREAKER"
    assert station.couplers[1].name == "VL6_BREAKER_1_1"
    assert_coupler_bay(
        station.couplers[1],
        from_busbars=["VL6_1_1"],
        to_busbars=["VL6_1_2"],
        from_switches={"VL6_1_1": "VL6_DISCONNECTOR_0_6"},
        to_switches={"VL6_1_2": "VL6_DISCONNECTOR_7_2"},
    )

    assert station.couplers[2].grid_model_id == "VL6_BREAKER_2_1"
    assert station.couplers[2].coupler_type == "BREAKER"
    assert station.couplers[2].name == "VL6_BREAKER_2_1"
    assert_coupler_bay(
        station.couplers[2],
        from_busbars=["VL6_2_1"],
        to_busbars=["VL6_2_2"],
        from_switches={"VL6_2_1": "VL6_DISCONNECTOR_1_8"},
        to_switches={"VL6_2_2": "VL6_DISCONNECTOR_9_3"},
    )

    assert station.couplers[3].grid_model_id == "VL6_DISCONNECTOR_2_4"
    assert station.couplers[3].coupler_type == "DISCONNECTOR"
    assert station.couplers[3].name == "VL6_DISCONNECTOR_2_4"
    assert_coupler_bay(
        station.couplers[3],
        from_busbars=["VL6_1_2"],
        to_busbars=["VL6_1_3"],
        from_switches={},
        to_switches={},
        connection_kind="disconnector",
    )

    assert station.couplers[4].grid_model_id == "VL6_DISCONNECTOR_3_5"
    assert station.couplers[4].coupler_type == "DISCONNECTOR"
    assert station.couplers[4].name == "VL6_DISCONNECTOR_3_5"
    assert_coupler_bay(
        station.couplers[4],
        from_busbars=["VL6_2_2"],
        to_busbars=["VL6_2_3"],
        from_switches={},
        to_switches={},
        connection_kind="disconnector",
    )

    # Voltage level 4
    station_info = {"name": "Station4", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL4"}
    station_info = SubstationInformation(**station_info)
    station, _branch_switching_table, _injection_switching_table = build_station_from_bus_id(net, "VL4_0", station_info)
    assert len(station.couplers) == 3
    assert station.couplers[0].grid_model_id == "VL4_BREAKER"
    assert station.couplers[0].coupler_type == "BREAKER"
    assert station.couplers[0].name == "VL4_BREAKER"
    assert_coupler_bay(
        station.couplers[0],
        from_busbars=["BBS4_1", "BBS4_2", "BBS4_3"],
        to_busbars=["BBS4_2", "BBS4_3", "BBS4_4"],
        from_switches={
            "BBS4_1": "VL4_DISCONNECTOR_10_0",
            "BBS4_2": "VL4_DISCONNECTOR_10_1",
            "BBS4_3": "VL4_DISCONNECTOR_10_2",
        },
        to_switches={
            "BBS4_2": "VL4_DISCONNECTOR_11_1",
            "BBS4_3": "VL4_DISCONNECTOR_11_2",
            "BBS4_4": "VL4_DISCONNECTOR_11_3",
        },
    )

    assert station.couplers[1].grid_model_id == "VL4_BREAKER#0"
    assert station.couplers[1].coupler_type == "BREAKER"
    assert station.couplers[1].name == "VL4_BREAKER#0"
    assert_coupler_bay(
        station.couplers[1],
        from_busbars=["BBS4_1", "BBS4_2", "BBS4_3"],
        to_busbars=["BBS4_2", "BBS4_3", "BBS4_4"],
        from_switches={
            "BBS4_1": "VL4_DISCONNECTOR_12_0",
            "BBS4_2": "VL4_DISCONNECTOR_12_1",
            "BBS4_3": "VL4_DISCONNECTOR_12_2",
        },
        to_switches={
            "BBS4_2": "VL4_DISCONNECTOR_13_1",
            "BBS4_3": "VL4_DISCONNECTOR_13_2",
            "BBS4_4": "VL4_DISCONNECTOR_13_3",
        },
    )

    assert station.couplers[2].grid_model_id == "VL4_BREAKER#1"
    assert station.couplers[2].coupler_type == "BREAKER"
    assert station.couplers[2].name == "VL4_BREAKER#1"
    assert_coupler_bay(
        station.couplers[2],
        from_busbars=["BBS4_1", "BBS4_2", "BBS4_3"],
        to_busbars=["BBS4_2", "BBS4_3", "BBS4_4"],
        from_switches={
            "BBS4_1": "VL4_DISCONNECTOR_14_0",
            "BBS4_2": "VL4_DISCONNECTOR_14_1",
            "BBS4_3": "VL4_DISCONNECTOR_14_2",
        },
        to_switches={
            "BBS4_2": "VL4_DISCONNECTOR_15_1",
            "BBS4_3": "VL4_DISCONNECTOR_15_2",
            "BBS4_4": "VL4_DISCONNECTOR_15_3",
        },
    )


@pytest.mark.xfail(reason="Failing edge case, there are no lines connected to the busbars in the middle")
def test_switching_tables_failing_edgecase(basic_node_breaker_network_powsybl_grid_v2):
    net = basic_node_breaker_network_powsybl_grid_v2
    # net.get_single_line_diagram('VL5')

    station_info = SubstationInformation(voltage_level_id="VL6", region="BE", nominal_v=380, name="Station6")
    station, _branch_switching_table, _injection_switching_table = build_station_from_bus_id(net, "VL6_0", station_info)
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
            asset_bay_id=build_asset_bay_id(substation_id, "L3"),
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id="L32_BREAKER",
            busbar_disconnector_grid_model_id={"BBS3_1": "L32_DISCONNECTOR_5_0", "BBS3_2": "L32_DISCONNECTOR_5_1"},
        ),
        "L6": AssetBay(
            asset_bay_id=build_asset_bay_id(substation_id, "L6"),
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id="L62_BREAKER",
            busbar_disconnector_grid_model_id={"BBS3_1": "L62_DISCONNECTOR_7_0", "BBS3_2": "L62_DISCONNECTOR_7_1"},
        ),
        "L7": AssetBay(
            asset_bay_id=build_asset_bay_id(substation_id, "L7"),
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id="L72_BREAKER",
            busbar_disconnector_grid_model_id={"BBS3_1": "L72_DISCONNECTOR_9_0", "BBS3_2": "L72_DISCONNECTOR_9_1"},
        ),
        "L9": AssetBay(
            asset_bay_id=build_asset_bay_id(substation_id, "L9"),
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id="L91_BREAKER",
            busbar_disconnector_grid_model_id={"BBS3_1": "L91_DISCONNECTOR_11_0", "BBS3_2": "L91_DISCONNECTOR_11_1"},
        ),
        "load2": AssetBay(
            asset_bay_id=build_asset_bay_id(substation_id, "load2"),
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id="load2_BREAKER",
            busbar_disconnector_grid_model_id={"BBS3_1": "load2_DISCONNECTOR_19_0", "BBS3_2": "load2_DISCONNECTOR_19_1"},
        ),
    }
    asset_bay_dict = {}
    station_logs = []
    for asset_grid_model_id in switchable_assets_df["grid_model_id"].to_list():
        asset_bay, logs = get_asset_bay(
            network_graph_data.switches,
            station_grid_model_id=substation_id,
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
            station_grid_model_id=substation_id,
            asset_grid_model_id="L3",
            busbar_df=busbar_df,
            edge_connection_info=edge_connection_info,
        )
    switches_df.loc[0, "asset_type"] = "BREAKER"
    switches_df.loc[1, "asset_type"] = "BREAKER"
    switches_df.loc[2, "asset_type"] = "BREAKER"

    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        station_grid_model_id=substation_id,
        asset_grid_model_id="L3",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    expected = AssetBay(
        asset_bay_id=build_asset_bay_id(substation_id, "L3"),
        asset_disconnector_grid_model_id=None,
        dv_switch_grid_model_id="L32_BREAKER",
        busbar_disconnector_grid_model_id={"BBS3_1": "L32_DISCONNECTOR_5_0", "BBS3_2": "L32_DISCONNECTOR_5_1"},
    )
    assert asset_grid_model_id == expected
    assert logs == [
        "Warning: There is a BREAKER directly connected to a busbar ['L32_DISCONNECTOR_5_0', 'L32_DISCONNECTOR_5_1'] Will be modelled as busbar disconnector. grid_model_id: L32_DISCONNECTOR_5_0"
    ]

    switches_df.drop(1, inplace=True)
    switches_df.drop(2, inplace=True)
    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        station_grid_model_id=substation_id,
        asset_grid_model_id="L3",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    assert asset_grid_model_id is None
    assert logs == [
        "Warning: There should be at least one busbar disconnector but got 0, AssetBay ignored for grid_model_id: L3"
    ]

    edge_connection_info["L62_DISCONNECTOR_7_0"].direct_busbar_grid_model_id = ""
    asset_grid_model_id, logs = get_asset_bay(
        switches_df=switches_df,
        station_grid_model_id=substation_id,
        asset_grid_model_id="L6",
        busbar_df=busbar_df,
        edge_connection_info=edge_connection_info,
    )
    expected = AssetBay(
        asset_bay_id=build_asset_bay_id(substation_id, "L6"),
        asset_disconnector_grid_model_id="L62_DISCONNECTOR_7_0",
        dv_switch_grid_model_id="L62_BREAKER",
        busbar_disconnector_grid_model_id={"BBS3_2": "L62_DISCONNECTOR_7_1"},
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


def test_get_asset_disconnector():
    # Test case 1: No asset disconnector found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["busbar1", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_asset_disconnector = get_asset_disconnector(asset_bays_df)
    assert result is None, f"Expected None, but got {result}"
    assert n_asset_disconnector == 0
    assert logs == []

    # Test case 2: One asset disconnector found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", "busbar2"],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, False],
        }
    )
    result, logs, n_asset_disconnector = get_asset_disconnector(asset_bays_df)
    assert result == "id1", f"Expected 'switch1', but got {result}"
    assert n_asset_disconnector == 1
    assert logs == []

    # Test case 3: Multiple asset disconnectors found
    asset_bays_df = pd.DataFrame(
        {
            "asset_type": ["DISCONNECTOR", "DISCONNECTOR"],
            "direct_busbar_grid_model_id": ["", ""],
            "foreign_id": ["switch1", "switch2"],
            "grid_model_id": ["id1", "id2"],
            "open": [False, True],
        }
    )
    result, logs, n_asset_disconnector = get_asset_disconnector(asset_bays_df)
    assert result == "id2"  # Expectes the open switch
    assert n_asset_disconnector == 2
    assert "There should be maximum one asset_disconnector but got 2" in logs[0]

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
    result, logs, n_asset_disconnector = get_asset_disconnector(asset_bays_df)
    assert result is None, f"Expected None, but got {result}"
    assert n_asset_disconnector == 0
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
                "from_busbar_ids": ["BBS2_1", "BBS2_2"],
                "from_coupler_ids": ["VL2_DISCONNECTOR_13_0", "VL2_DISCONNECTOR_13_1"],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_13_0",
                "direct_busbar_grid_model_id": "BBS2_1",
                "open": False,
                "from_busbar_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_13_1",
                "direct_busbar_grid_model_id": "BBS2_2",
                "open": True,
                "from_busbar_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_14_1",
                "direct_busbar_grid_model_id": "BBS2_2",
                "open": False,
                "from_busbar_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_DISCONNECTOR_14_2",
                "direct_busbar_grid_model_id": "BBS2_3",
                "open": True,
                "from_busbar_ids": [],
                "from_coupler_ids": [],
            },
            {
                "grid_model_id": "VL2_BREAKER",
                "direct_busbar_grid_model_id": "",
                "open": False,
                "from_busbar_ids": ["BBS2_1", "BBS2_2", "BBS2_3"],
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

    with pytest.raises(ValueError, match="Coupler has no busbar id"):
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

    # Case 9: one side has an asset disconnector
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
