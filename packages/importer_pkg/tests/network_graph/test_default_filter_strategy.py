# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Test default filter strategy functions.

The testdata is on purpose kept outside of the conftest.py to make it easier to understand the test.
Each tests creates it's own test graph with the minimum data needed to test the function.

"""

import copy

import networkx as nx
from toop_engine_importer.network_graph.data_classes import (
    BusbarConnectionInfo,
    EdgeConnectionInfo,
    NetworkGraphData,
    WeightValues,
)
from toop_engine_importer.network_graph.default_filter_strategy import (
    calculate_connectable_busbars,
    calculate_zero_impedance_connected,
    get_asset_bay_update_dict,
    get_connectable_assets_update_dict,
    get_connectable_busbars_update_dict,
    set_asset_bay_edge_attr,
    set_bay_weights,
    set_connectable_busbars,
    set_switch_busbar_connection_info,
    set_zero_impedance_connected,
)
from toop_engine_importer.network_graph.network_graph import generate_graph
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data


def test_set_switch_busbar_connection_info(network_graph_data_test2_helper_branches_removed: NetworkGraphData):
    network_graph_data = network_graph_data_test2_helper_branches_removed

    add_graph_specific_data(network_graph_data)
    graph = generate_graph(network_graph_data)

    # assert test initial state
    for node in graph.nodes:
        if node not in network_graph_data.node_assets["node"].values:
            assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()

    network_graph_data_copy = copy.copy(network_graph_data)
    set_switch_busbar_connection_info(graph=graph)

    assert network_graph_data_copy.switches.equals(network_graph_data.switches)
    assert network_graph_data_copy.helper_branches.equals(network_graph_data.helper_branches)
    assert network_graph_data_copy.branches.equals(network_graph_data.branches)
    assert network_graph_data_copy.nodes.equals(network_graph_data.nodes)
    assert network_graph_data_copy.node_assets.equals(network_graph_data.node_assets)

    # should not change nodes
    for node in graph.nodes:
        if node not in network_graph_data.node_assets["node"].values:
            assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    busbar_cond = network_graph_data.nodes["node_type"] == "busbar"
    busbars = network_graph_data.nodes.loc[busbar_cond].index
    cond = network_graph_data.switches["from_node"].isin(busbars) | network_graph_data.switches["to_node"].isin(busbars)

    edges_changed_list = list(network_graph_data.switches.loc[cond, "node_tuple"].values)
    for edge in graph.edges:
        edge_connection_info = graph.edges[edge]["edge_connection_info"]
        if edge in edges_changed_list:
            for attr in edge_connection_info.__dict__.keys():
                if "direct_busbar_grid_model_id" != attr:
                    assert getattr(edge_connection_info, attr) == getattr(EdgeConnectionInfo(), attr)
                else:
                    assert getattr(edge_connection_info, attr) != getattr(EdgeConnectionInfo(), attr)

                assert graph.edges[edge]["busbar_weight"] != WeightValues.low.value
        else:
            assert edge_connection_info == EdgeConnectionInfo()
            assert graph.edges[edge]["busbar_weight"] == WeightValues.low.value


def test_set_bay_weights_for_asset_nodes():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_node(
        6,
        node_type="node",
        grid_model_id="bb1",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_6_gen"], node_assets_ids=[123]),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})
        if "busbar_connection_info" not in graph.nodes[node]:
            nx.set_node_attributes(graph, {node: {"busbar_connection_info": BusbarConnectionInfo()}})
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"grid_model_id": f"edge_{edge}"}})
        nx.set_edge_attributes(graph, {edge: {"coupler_weight": WeightValues.low.value}})

    set_bay_weights(graph=graph)
    busbar_weight_list = [
        (1, 2, 10.0),
        (1, 7, 10.0),
        (2, 3, 0.0),
        (3, 4, 0.0),
        (4, 5, 0.0),
        (5, 6, 0.0),
        (7, 8, 0.0),
        (8, 9, 10.0),
    ]
    assert busbar_weight_list == list(graph.edges(data="busbar_weight"))

    # assert bay
    bay_ids = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    none_bay_ids = [(1, 7), (7, 8), (8, 9)]
    assert len(bay_ids) + len(none_bay_ids) == len(graph.edges)
    for edge in bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value

    for edge in none_bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.low.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.low.value

    # add by connection to other busbar
    graph.add_edge(
        2,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # reset bay weights
    for edge in graph.edges:
        graph.edges[edge]["bay_weight"] = WeightValues.low.value
        graph.edges[edge]["coupler_weight"] = WeightValues.low.value
    set_bay_weights(graph=graph)
    # assert bay
    bay_ids = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 9)]
    none_bay_ids = [(1, 7), (7, 8), (8, 9)]
    assert len(bay_ids) + len(none_bay_ids) == len(graph.edges)
    for edge in bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value

    for edge in none_bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.low.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.low.value

    # add random node as dead end in asset bay
    graph.add_edge(
        2,
        100,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    nx.set_node_attributes(graph, {100: {"busbar_connection_info": BusbarConnectionInfo()}})
    # reset bay weights
    for edge in graph.edges:
        graph.edges[edge]["bay_weight"] = WeightValues.low.value
        graph.edges[edge]["coupler_weight"] = WeightValues.low.value
    set_bay_weights(graph=graph)
    # assert bay
    bay_ids = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 9)]
    none_bay_ids = [(1, 7), (7, 8), (8, 9), (2, 100)]
    assert len(bay_ids) + len(none_bay_ids) == len(graph.edges)
    for edge in bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value

    for edge in none_bay_ids:
        assert graph.edges[edge]["bay_weight"] == WeightValues.low.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.low.value


def test_set_asset_bay_edge_attr(network_graph_data_test2_helper_branches_removed: NetworkGraphData):
    network_graph_data = network_graph_data_test2_helper_branches_removed

    add_graph_specific_data(network_graph_data)
    graph = generate_graph(network_graph_data)

    # assert test initial state
    for node in graph.nodes:
        if node not in network_graph_data.node_assets["node"].values:
            assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()

    bay1_id = "test_bay_id"
    bay_update_dict = {bay1_id: {}}
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=bay_update_dict)
    # nothing should happen
    for node in graph.nodes:
        if node not in network_graph_data.node_assets["node"].values:
            assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()

    # note: this is just a random dict, the values are not reflecting an actual bay path
    shortest_path_to_busbar_dict = {
        45: [28, 45],
        33: [28, 33],
        37: [28, 37],
        43: [28, 43],
        41: [28, 41],
        47: [28, 47],
        35: [28, 35],
        31: [28, 31],
        39: [28, 39],
    }
    bay1_id = "test_bay_id"
    bay_update_dict = {bay1_id: shortest_path_to_busbar_dict}
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=bay_update_dict)
    # should not change nodes
    for node in graph.nodes:
        if node not in network_graph_data.node_assets["node"].values:
            assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    edges_changed_list = [set(values) for values in shortest_path_to_busbar_dict.values()]
    for edge in graph.edges:
        edge_connection_info = graph.edges[edge]["edge_connection_info"]
        if set(edge) in edges_changed_list:
            for attr in edge_connection_info.__dict__.keys():
                if "bay_id" != attr:
                    assert getattr(edge_connection_info, attr) == getattr(EdgeConnectionInfo(), attr)
                else:
                    assert getattr(edge_connection_info, attr) == bay1_id

                assert graph.edges[edge]["bay_weight"] != WeightValues.low.value
                assert graph.edges[edge]["coupler_weight"] != WeightValues.step.value
        else:
            assert edge_connection_info == EdgeConnectionInfo()
            assert graph.edges[edge]["bay_weight"] == WeightValues.low.value
            assert graph.edges[edge]["coupler_weight"] == WeightValues.step.value


def test_get_node_update_dict():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(12, node_type="busbar", grid_model_id="bb3", busbar_connection_info=BusbarConnectionInfo())

    # following edges are not needed for the test, bus are given as an example to show the connection
    # # asset bay
    # graph.add_edge(1, 2, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.max_step.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(2, 3, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(3, 4, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(4, 5, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # # node 6 = asset
    # graph.add_edge(5, 6, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(7, 2, bay_weight=WeightValues.max_step.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # # busbar coupler to busbar2
    # graph.add_edge(1, 7, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(7, 8, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # # node 9 = busbar2
    # graph.add_edge(8, 9, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value, edge_connection_info=EdgeConnectionInfo())
    # # busbar coupler to busbar3
    # graph.add_edge(9, 10, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(10, 11, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value, edge_connection_info=EdgeConnectionInfo())
    # graph.add_edge(11, 12, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value, edge_connection_info=EdgeConnectionInfo())

    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})

    busbar_interconnectable = {1: [9], 9: [1, 12], 12: [9]}
    node_update_dict = get_connectable_busbars_update_dict(graph=graph, shortest_path=busbar_interconnectable)
    node_update_dict_expected = {
        1: {"connectable_busbars": ["node_9"], "connectable_busbars_node_ids": [9]},
        9: {"connectable_busbars": ["node_1", "node_12"], "connectable_busbars_node_ids": [1, 12]},
        12: {"connectable_busbars": ["node_9"], "connectable_busbars_node_ids": [9]},
    }
    assert node_update_dict == node_update_dict_expected


def test_get_connectable_assets_update_dict():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    # node 6 = asset
    graph.add_node(
        6,
        node_type="node",
        grid_model_id="n6",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_6_gen"], node_assets_ids=[123]),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    connectable_busbars_node_assets = {6: [1]}
    update_dict_res = get_connectable_assets_update_dict(
        graph=graph, connectable_node_assets_to_busbars=connectable_busbars_node_assets
    )
    node_update_dict_expected = {1: {"connectable_assets": ["node_6_gen"], "connectable_assets_node_ids": [6]}}
    assert update_dict_res == node_update_dict_expected

    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(
        7,
        node_type="node",
        grid_model_id="n7",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_7_gen"], node_assets_ids=[124]),
    )

    connectable_busbars_node_assets = {6: [1, 9], 7: [9]}
    update_dict_res = get_connectable_assets_update_dict(
        graph=graph, connectable_node_assets_to_busbars=connectable_busbars_node_assets
    )
    node_update_dict_expected = {
        1: {"connectable_assets": ["node_6_gen"], "connectable_assets_node_ids": [6]},
        9: {"connectable_assets": ["node_6_gen", "node_7_gen"], "connectable_assets_node_ids": [6, 7]},
    }
    assert update_dict_res == node_update_dict_expected


def test_set_connectable_busbars():
    graph = nx.Graph()
    # busbars
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(12, node_type="busbar", grid_model_id="bb3", busbar_connection_info=BusbarConnectionInfo())

    # asset bay
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # busbar coupler to busbar3
    graph.add_edge(
        9,
        10,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        11,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        11,
        12,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"grid_model_id": f"edge_{edge}"}})
        nx.set_edge_attributes(graph, {edge: {"coupler_weight": WeightValues.step.value}})

    # node_update_dict_expected comes from get_node_update_dict(graph=graph, shortest_path=busbar_interconnectable)
    # node_update_dict = get_node_update_dict(graph=graph, shortest_path=busbar_interconnectable)
    node_update_dict_expected = {
        1: {"connectable_busbars": ["node_9"], "connectable_busbars_node_ids": [9]},
        9: {"connectable_busbars": ["node_1", "node_12"], "connectable_busbars_node_ids": [1, 12]},
        12: {"connectable_busbars": ["node_9"], "connectable_busbars_node_ids": [9]},
    }
    busbar_interconnectable = {1: [9], 9: [1, 12], 12: [9]}
    set_connectable_busbars(graph=graph)
    # no change expected
    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()
    for node in node_update_dict_expected:
        connection_info = graph.nodes[node]["busbar_connection_info"]
        assert connection_info == BusbarConnectionInfo(**(node_update_dict_expected[node]))


def test_get_connectable_busbars():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    graph.add_edge(
        7,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar3
    graph.add_node(12, node_type="busbar", grid_model_id="bb3", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        9,
        10,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        11,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        11,
        12,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"grid_model_id": f"edge_{edge}"}})
        nx.set_edge_attributes(graph, {edge: {"coupler_weight": WeightValues.step.value}})

    busbar_interconnectable, busbar_shortest_path = calculate_connectable_busbars(graph=graph)
    # 1 and 9 are connectable
    # 9 is in the middle and connects 1 and 12
    # 1 and 12 are not directly connected
    assert busbar_interconnectable == {1: [9], 9: [1, 12], 12: [9]}
    assert busbar_shortest_path == {
        1: {9: [1, 7, 8, 9]},
        9: {1: [9, 8, 7, 1], 12: [9, 10, 11, 12]},
        12: {9: [12, 11, 10, 9]},
    }


def test_set_zero_impedance_connected():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(
        1,
        node_type="busbar",
        grid_model_id="bb1",
        busbar_connection_info=BusbarConnectionInfo(connectable_assets_node_ids=[6], connectable_busbars_node_ids=[9]),
    )
    graph.add_node(
        9,
        node_type="busbar",
        grid_model_id="bb2",
        busbar_connection_info=BusbarConnectionInfo(connectable_assets_node_ids=[6], connectable_busbars_node_ids=[1]),
    )
    graph.add_node(
        6,
        node_type="node",
        grid_model_id="n6",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_6_gen"], node_assets_ids=[123]),
    )

    # asset bay
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"switch_open_weight": WeightValues.low.value}})
    nx.set_edge_attributes(graph, {(9, 2): {"switch_open_weight": WeightValues.high.value}})
    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})

    set_zero_impedance_connected(graph=graph)

    assert graph.nodes[1]["busbar_connection_info"] == BusbarConnectionInfo(
        connectable_assets_node_ids=[6],
        connectable_busbars_node_ids=[9],
        zero_impedance_connected_assets=["node_6_gen"],
        zero_impedance_connected_assets_node_ids=[6],
        zero_impedance_connected_busbars=["node_9"],
        zero_impedance_connected_busbars_node_ids=[9],
    )

    assert graph.nodes[9]["busbar_connection_info"] == BusbarConnectionInfo(
        connectable_assets_node_ids=[6],
        connectable_busbars_node_ids=[1],
        zero_impedance_connected_busbars=["node_1"],
        zero_impedance_connected_busbars_node_ids=[1],
    )


def test_calculate_zero_impedance_connected():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(
        1,
        node_type="busbar",
        grid_model_id="bb1",
        busbar_connection_info=BusbarConnectionInfo(connectable_assets_node_ids=[6], connectable_busbars_node_ids=[9]),
    )
    graph.add_node(
        9,
        node_type="busbar",
        grid_model_id="bb2",
        busbar_connection_info=BusbarConnectionInfo(connectable_assets_node_ids=[6], connectable_busbars_node_ids=[1]),
    )
    graph.add_node(
        6,
        node_type="node",
        grid_model_id="n6",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_6_gen"], node_assets_ids=[123]),
    )

    # asset bay
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"switch_open_weight": WeightValues.low.value}})
    nx.set_edge_attributes(graph, {(9, 2): {"switch_open_weight": WeightValues.high.value}})
    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})

    connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=1)
    assert connected_assets_dict == {6: ["node_6_gen"], 9: []}

    connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=9)
    assert connected_assets_dict == {1: []}

    # Test with no open switches
    nx.set_edge_attributes(graph, {(9, 2): {"switch_open_weight": WeightValues.low.value}})
    connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=1)
    assert connected_assets_dict == {6: ["node_6_gen"], 9: []}

    connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=9)
    assert connected_assets_dict == {6: ["node_6_gen"], 1: []}

    # Test with no connectable assets
    graph.nodes[1]["busbar_connection_info"].connectable_assets_node_ids = []
    connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=1)
    assert connected_assets_dict == {9: []}


def test_get_asset_bay_update_dict():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1_station1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(9, node_type="busbar", grid_model_id="bb2station1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(6, node_type="node", grid_model_id="n6_station1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(
        7,
        node_type="node",
        grid_model_id="n7_station1",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_7_gen"], node_assets_ids=[124]),
    )

    # asset bay 1
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        2,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        40,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        40,
        41,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        41,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # asset bay 2
    graph.add_edge(
        1,
        10,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        11,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        11,
        12,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        12,
        13,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        13,
        14,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        14,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        10,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # node 1 = busbar1
    graph.add_node(15, node_type="busbar", grid_model_id="bb3_station2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(19, node_type="busbar", grid_model_id="bb4_station2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(16, node_type="node", grid_model_id="n6_station2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(
        17,
        node_type="node",
        grid_model_id="n7_station2",
        busbar_connection_info=BusbarConnectionInfo(node_assets=["node_17_gen"], node_assets_ids=[125]),
    )

    # asset bay 1
    graph.add_edge(
        15,
        18,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        18,
        31,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        31,
        20,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        20,
        21,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        21,
        16,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        18,
        19,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        15,
        22,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        22,
        23,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        23,
        19,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # asset bay 2
    graph.add_edge(
        15,
        24,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        24,
        25,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        25,
        26,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        26,
        27,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        27,
        28,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        28,
        17,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        19,
        24,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # add a branch
    graph.add_edge(
        6,
        16,
        bay_weight=WeightValues.high.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # Note: node_assets_ids are the id's of the assets
    graph.nodes[6]["busbar_connection_info"] = BusbarConnectionInfo(node_assets=["edge(6, 16)"], node_assets_ids=[(6, 16)])
    graph.nodes[16]["busbar_connection_info"] = BusbarConnectionInfo(node_assets=["edge(6, 16)"], node_assets_ids=[(6, 16)])

    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"switch_open_weight": WeightValues.low.value}})
        nx.set_edge_attributes(graph, {edge: {"asset_type": "switch"}})
        nx.set_edge_attributes(graph, {edge: {"node_tuple": edge}})
        nx.set_edge_attributes(graph, {edge: {"grid_model_id": f"edge_{edge}"}})
    nx.set_edge_attributes(graph, {(6, 16): {"asset_type": "LINE"}})
    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})
        if "busbar_connection_info" not in graph.nodes[node]:
            nx.set_node_attributes(graph, {node: {"busbar_connection_info": BusbarConnectionInfo()}})

    update_node_dict, bay_update_dict = get_asset_bay_update_dict(graph=graph)
    expected_update_node_dict = {
        1: {"connectable_assets": ["edge(6, 16)", "node_7_gen"], "connectable_assets_node_ids": [6, 7]},
        9: {"connectable_assets": ["edge(6, 16)", "node_7_gen"], "connectable_assets_node_ids": [6, 7]},
        15: {"connectable_assets": ["edge(6, 16)", "node_17_gen"], "connectable_assets_node_ids": [16, 17]},
        19: {"connectable_assets": ["edge(6, 16)", "node_17_gen"], "connectable_assets_node_ids": [16, 17]},
        6: {"connectable_busbars": ["node_1", "node_9"], "connectable_busbars_node_ids": [1, 9]},
        7: {"connectable_busbars": ["node_1", "node_9"], "connectable_busbars_node_ids": [1, 9]},
        17: {"connectable_busbars": ["node_15", "node_19"], "connectable_busbars_node_ids": [15, 19]},
        16: {"connectable_busbars": ["node_15", "node_19"], "connectable_busbars_node_ids": [15, 19]},
    }
    assert update_node_dict == expected_update_node_dict

    expected_bay_update_dict = {
        "edge(6, 16)": {15: [16, 21, 20, 31, 18, 15], 19: [16, 21, 20, 31, 18, 19]},
        "node_7_gen": {1: [7, 14, 13, 12, 11, 10, 1], 9: [7, 14, 13, 12, 11, 10, 9]},
        "node_17_gen": {15: [17, 28, 27, 26, 25, 24, 15], 19: [17, 28, 27, 26, 25, 24, 19]},
    }
    assert bay_update_dict == expected_bay_update_dict
