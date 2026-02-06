# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import pytest
from toop_engine_importer.network_graph.data_classes import BusbarConnectionInfo, EdgeConnectionInfo, WeightValues
from toop_engine_importer.network_graph.filter_strategy.helper_functions import set_asset_bay_edge_attr
from toop_engine_importer.network_graph.filter_strategy.switches import (
    busbar_coupler_condition,
    get_asset_bay_id_grid_model_update_dict,
    get_busbar_sides_of_coupler,
    get_coupler_bay_edge_ids,
    get_coupler_type,
    get_switch_bay_dict,
    get_switches_with_no_bay_id,
    set_all_busbar_coupling_switches,
    set_coupler_type,
    set_coupler_type_graph,
    set_switch_bay_from_edge_ids,
)


@pytest.fixture()
def graph_with_all_coupling_setups():
    graph = nx.Graph()

    # Add busbar nodes
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(2, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(3, node_type="busbar", grid_model_id="bb3", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(4, node_type="busbar", grid_model_id="bb4", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(100, node_type="busbar", grid_model_id="bb100", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(200, node_type="busbar", grid_model_id="bb200", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(300, node_type="busbar", grid_model_id="bb300", busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(400, node_type="busbar", grid_model_id="bb400", busbar_connection_info=BusbarConnectionInfo())

    # Single paths between busbars
    # B1 -> DS -> B2
    graph.add_edge(
        1,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B2 -> CB -> B3
    graph.add_edge(
        2,
        3,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> DS -> CB -> B2
    graph.add_edge(
        1,
        7,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        2,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> CB -> DS -> B2
    graph.add_edge(
        1,
        9,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> DS -> CB -> DS -> B2
    graph.add_edge(
        1,
        11,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        11,
        12,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        12,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> DS -> CB -> CB -> DS -> B2
    graph.add_edge(
        1,
        30,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        30,
        31,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        31,
        32,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        32,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> DS -> CB -> DS -> CB -> DS -> B2
    graph.add_edge(
        1,
        33,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        33,
        34,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        34,
        35,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        35,
        36,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        36,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # More complex paths with multiple busbars
    # Two busbars on either side
    # B1 -> DS1 -> CB -> DS3 -> B3
    #           ^
    # B2 -> DS2 |
    graph.add_edge(
        1,
        14,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        14,
        15,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        15,
        3,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    #   B2 -> DS2 |
    graph.add_edge(
        2,
        14,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # B1 -> DS1 -> CB -> DS3 -> B3
    #           ^     ^
    # B2 -> DS2 |     | DS4 - > B4
    graph.add_edge(
        1,
        19,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        19,
        20,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        20,
        3,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # B2 -> DS2 |
    graph.add_edge(
        2,
        19,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # | DS4 - > B4
    graph.add_edge(
        20,
        4,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # Three busbars on either side
    # B1 -> DS1 -> CB -> DS3 -> B4
    #           ^
    # B2 -> DS2 |
    #           |
    # B3 -> DS4 |
    graph.add_edge(
        1,
        17,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        17,
        18,
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        18,
        4,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # B2 -> DS2 |
    graph.add_edge(
        2,
        17,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # B3 -> DS4 |
    graph.add_edge(
        3,
        17,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # a series of DS
    # B100 -> DS1 -> B200 -> DS2 -> B300 -> DS3 -> B400
    graph.add_edge(
        100,
        200,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        200,
        201,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(  # helper node
        201,
        300,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        asset_type="",
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(  # helper node
        300,
        301,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        asset_type="",
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        301,
        302,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(  # helper node
        302,
        400,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        asset_type="",
        edge_connection_info=EdgeConnectionInfo(),
    )

    # Add attributes to all edges
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"switch_open_weight": WeightValues.low.value}})
        nx.set_edge_attributes(graph, {edge: {"node_tuple": edge}})
        nx.set_edge_attributes(graph, {edge: {"grid_model_id": f"edge_{edge}"}})

    # Add attributes to all nodes
    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"grid_model_id": f"node_{node}"}})
        if "busbar_connection_info" not in graph.nodes[node]:
            nx.set_node_attributes(graph, {node: {"busbar_connection_info": BusbarConnectionInfo()}})

    return graph


def test_get_switches_with_no_bay_id():
    graph = nx.Graph()
    # edges that should not be found for BREAKER:
    graph.add_edge(
        1,
        2,
        node_tuple=(1, 2),
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(bay_id="bb1"),
    )
    graph.add_edge(
        1,
        3,
        node_tuple=(1, 3),
        asset_type="BREAKER",
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(bay_id="bb1"),
    )
    graph.add_edge(
        1,
        4,
        node_tuple=(1, 4),
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(bay_id=""),
    )

    # edges that should be found for BREAKER:
    graph.add_edge(
        2,
        3,
        node_tuple=(2, 3),
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(
            direct_busbar_grid_model_id="bb2",
            bay_id="",
            coupler_type="",
        ),
    )
    graph.add_edge(
        2,
        4,
        node_tuple=(2, 4),
        asset_type="BREAKER",
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(
            bay_id="",
        ),
    )
    # test BREAKER
    res = get_switches_with_no_bay_id(graph=graph, asset_type="BREAKER")
    assert res == [(2, 3), (2, 4)]
    # test DISCONNECTOR
    res = get_switches_with_no_bay_id(graph=graph, asset_type="DISCONNECTOR")
    assert res == [(1, 4)]


def test_busbar_coupler_condition():
    connectable_assets = {
        1: [10, 11, 12, 13],
        2: [10, 11, 12],
        3: [14, 15],
    }

    # Test case where busbar is a busbar coupler
    assert busbar_coupler_condition(busbar1=1, busbar2=2, connectable_assets=connectable_assets, threshold=0.5)
    assert busbar_coupler_condition(busbar1=2, busbar2=1, connectable_assets=connectable_assets, threshold=0.5)

    # Test case where busbar is not a busbar coupler
    assert not busbar_coupler_condition(busbar1=1, busbar2=3, connectable_assets=connectable_assets, threshold=0.5)
    assert not busbar_coupler_condition(busbar1=3, busbar2=1, connectable_assets=connectable_assets, threshold=0.5)
    assert not busbar_coupler_condition(busbar1=1, busbar2=3, connectable_assets=connectable_assets, threshold=1)

    # Test case where busbar is a busbar coupler with a higher threshold
    assert busbar_coupler_condition(busbar1=1, busbar2=2, connectable_assets=connectable_assets, threshold=0.75)
    assert busbar_coupler_condition(busbar1=2, busbar2=1, connectable_assets=connectable_assets, threshold=0.75)

    # Test case where busbar is not a busbar coupler with a higher threshold
    assert not busbar_coupler_condition(busbar1=1, busbar2=2, connectable_assets=connectable_assets, threshold=0.9)
    assert not busbar_coupler_condition(busbar1=2, busbar2=1, connectable_assets=connectable_assets, threshold=0.9)

    # Test case where busbar has no connectable assets
    connectable_assets_empty = {
        1: [],
        2: [10, 11, 12],
    }
    assert not busbar_coupler_condition(busbar1=1, busbar2=2, connectable_assets=connectable_assets_empty, threshold=0.5)
    assert not busbar_coupler_condition(busbar1=2, busbar2=1, connectable_assets=connectable_assets_empty, threshold=0.5)


def test_get_coupler_type():
    coupler_sides = {(1, 3): ([1], [3]), (2, 4): ([2], [4]), (1, 4): ([1], [4]), (2, 3): ([2], [3])}
    connectable_assets = {
        1: [10, 12, 13, 14, 16, 20, 21],
        3: [10, 12, 13, 14, 16, 20, 21],
        4: [11, 15, 16, 18, 19, 22],
        2: [11, 15, 16, 18, 19, 22],
    }
    result = get_coupler_type(connectable_assets=connectable_assets, coupler_sides=coupler_sides)
    expected_result = {"busbar_coupler": [(1, 3), (2, 4)], "cross_coupler": [(1, 4), (2, 3)]}
    assert result == expected_result

    coupler_sides = {(1, 3): ([1], [3]), (2, 4): ([2], [4]), (1, 4): ([1], [4]), (2, 3): ([2], [3])}
    connectable_assets = {
        1: [10, 12, 13, 14, 16, 20, 50],
        3: [10, 12, 13, 14, 16, 20, 21],
        4: [11, 15, 16, 18, 19, 22],
        2: [11, 15, 16, 18, 19, 22],
    }
    result = get_coupler_type(connectable_assets=connectable_assets, coupler_sides=coupler_sides)
    expected_result = {"busbar_coupler": [(1, 3), (2, 4)], "cross_coupler": [(1, 4), (2, 3)]}
    assert result == expected_result


def test_set_coupler_type():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(
        1,
        node_type="busbar",
        grid_model_id="bb1",
        busbar_connection_info=BusbarConnectionInfo(connectable_busbars=["bb1"], connectable_assets=["edge_3"]),
    )
    graph.add_node(
        4,
        node_type="busbar",
        grid_model_id="bb2",
        busbar_connection_info=BusbarConnectionInfo(connectable_busbars=["bb2"], connectable_assets=["edge_3"]),
    )
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 3 = asset
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        5,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        6,
        4,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for node in graph.nodes:
        if "busbar_connection_info" not in graph.nodes[node]:
            nx.set_node_attributes(graph, {node: {"busbar_connection_info": BusbarConnectionInfo()}})

    coupler_sides = {(5, 6): ([1], [4])}
    set_coupler_type(graph=graph, coupler_sides=coupler_sides)
    assert graph.edges[(5, 6)]["edge_connection_info"].coupler_type == "busbar_coupler"

    assert graph.edges[(1, 2)]["edge_connection_info"] == EdgeConnectionInfo()
    assert graph.edges[(4, 2)]["edge_connection_info"] == EdgeConnectionInfo()
    assert graph.edges[(2, 3)]["edge_connection_info"] == EdgeConnectionInfo()


def test_set_coupler_type_graph():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(
        1,
        node_type="busbar",
        grid_model_id="bb1",
        busbar_connection_info=BusbarConnectionInfo(connectable_busbars=["bb1"], connectable_assets=["edge_3"]),
    )
    graph.add_node(
        4,
        node_type="busbar",
        grid_model_id="bb2",
        busbar_connection_info=BusbarConnectionInfo(connectable_busbars=["bb2"], connectable_assets=["edge_3"]),
    )
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 3 = asset
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        5,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        6,
        4,
        bay_weight=WeightValues.low.value,
        coupler_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    for node in graph.nodes:
        if "busbar_connection_info" not in graph.nodes[node]:
            nx.set_node_attributes(graph, {node: {"busbar_connection_info": BusbarConnectionInfo()}})
    coupler_categories = {"busbar_coupler": [], "cross_coupler": [(5, 6)]}

    set_coupler_type_graph(graph=graph, coupler_categories=coupler_categories)
    assert graph.edges[(5, 6)]["edge_connection_info"].coupler_type == "cross_coupler"

    assert graph.edges[(1, 2)]["edge_connection_info"] == EdgeConnectionInfo()
    assert graph.edges[(4, 2)]["edge_connection_info"] == EdgeConnectionInfo()
    assert graph.edges[(2, 3)]["edge_connection_info"] == EdgeConnectionInfo()


def test_get_switches_with_no_bay_id(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups
    no_bay_breaker_edges = get_switches_with_no_bay_id(graph=graph, asset_type="BREAKER")
    assert set(no_bay_breaker_edges) == set(
        [(2, 7), (35, 36), (19, 20), (31, 32), (2, 3), (11, 12), (30, 31), (33, 34), (17, 18), (14, 15), (1, 9)]
    )


def test_get_switch_bay_dict(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups

    no_bay_breaker_edges = [
        (2, 7),
        (35, 36),
        (19, 20),
        (31, 32),
        (2, 3),
        (11, 12),
        (30, 31),
        (33, 34),
        (17, 18),
        (14, 15),
        (1, 9),
    ]
    # 2. Loop over BREAKER one by one
    # 2.1 Loop: Get all shortest paths from BREAKER to all BUSBARs (respect weights)
    asset_bay_edge_id_update_dict = get_switch_bay_dict(graph=graph, switch_edge_list=no_bay_breaker_edges)
    key_set = set(asset_bay_edge_id_update_dict.keys())
    seen_list = []
    # there is one BREAKER in the following paths -> only one solution
    expected_single_result = [(2, 3), (2, 7), (1, 9), (11, 12), (14, 15), (19, 20), (17, 18)]
    for res in expected_single_result:
        if res in key_set:
            seen_list.append(res)
    # there are two BREAKER in the following paths -> either one of them is a solution, but not both
    expected_multi_result = [((30, 31), (31, 32)), ((33, 34), (35, 36))]
    for res in expected_multi_result:
        if res[0] in key_set and res[1] not in key_set:
            seen_list.append(res[0])
        elif res[1] in key_set and res[0] not in key_set:
            seen_list.append(res[1])
    # TODO: fix ((33, 34),(35, 36)) being part of key_set -> bughunt
    assert len(seen_list) == len(expected_single_result) + len(expected_multi_result)
    assert len(seen_list) == len(key_set)

    # Note: the order of the nodes in the list may vary if the order of the nodes in the edge tuple varies
    asset_bay_edge_dict_expected = {
        (2, 7): ({2: [2]}, {1: [7, 1]}),
        (35, 36): ({1: [35, 34, 33, 1]}, {2: [36, 2]}),
        (19, 20): ({1: [19, 1], 2: [19, 2]}, {3: [20, 3], 4: [20, 4]}),
        (31, 32): ({1: [31, 30, 1]}, {2: [32, 2]}),
        (2, 3): ({2: [2]}, {3: [3]}),
        (11, 12): ({1: [11, 1]}, {2: [12, 2]}),
        (17, 18): ({1: [17, 1], 2: [17, 2], 3: [17, 3]}, {4: [18, 4]}),
        (14, 15): ({1: [14, 1], 2: [14, 2]}, {3: [15, 3]}),
        (1, 9): ({1: [1]}, {2: [9, 2]}),
    }
    for key, value in asset_bay_edge_dict_expected.items():
        assert key in asset_bay_edge_id_update_dict, f"Key {key} not found in asset_bay_edge_id_update_dict"
        assert asset_bay_edge_id_update_dict[key] == value, (
            f"Value for key {key} does not match expected value. Expected: {value}, Found: {asset_bay_edge_id_update_dict[key]}"
        )

    # Test only DISCONNECTOR
    no_bay_breaker_edges_ds = [(1, 2), (100, 200), (200, 201), (301, 302)]
    asset_bay_edge_id_update_dict_ds = get_switch_bay_dict(graph=graph, switch_edge_list=no_bay_breaker_edges_ds)
    asset_bay_edge_dict_expected_ds = {
        (1, 2): ({1: [1]}, {2: [2]}),
        (100, 200): ({100: [100]}, {200: [200]}),
        (200, 201): ({200: [200]}, {300: [201, 300]}),
        (301, 302): ({300: [301, 300]}, {400: [302, 400]}),
    }
    assert asset_bay_edge_id_update_dict_ds == asset_bay_edge_dict_expected_ds


def test_set_asset_bay_edge_attr(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups
    # test input data
    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()
        assert graph.edges[edge]["bay_weight"] == WeightValues.low.value
        assert graph.edges[edge]["coupler_weight"] == WeightValues.low.value

    asset_bay_edge_dict_input = {
        (2, 7): {1: [7, 1], 2: [7, 2]},
        (19, 20): {1: [19, 1], 2: [19, 2], 3: [19, 20, 3], 4: [19, 20, 4]},
    }
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=asset_bay_edge_dict_input)
    bay1_id = (2, 7)
    bay2_id = (19, 20)
    bay1_edge_ids = [(1, 7), (2, 7)]
    bay2_edge_ids = [(19, 1), (19, 2), (19, 20), (20, 4), (20, 3)]

    for edge in bay1_edge_ids:
        assert graph.edges[edge]["edge_connection_info"].bay_id == bay1_id
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value

    for edge in bay2_edge_ids:
        assert graph.edges[edge]["edge_connection_info"].bay_id == bay2_id
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value


def test_get_sides_of_coupler(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups

    asset_bay_edge_dict_input = {
        (2, 7): ({2: [2]}, {1: [7, 1]}),
        (35, 36): ({1: [35, 34, 33, 1]}, {2: [36, 2]}),
        (19, 20): ({1: [19, 1], 2: [19, 2]}, {3: [20, 3], 4: [20, 4]}),
        (31, 32): ({1: [31, 30, 1]}, {2: [32, 2]}),
        (2, 3): ({2: [2]}, {3: [3]}),
        (11, 12): ({1: [11, 1]}, {2: [12, 2]}),
        (17, 18): ({1: [17, 1], 2: [17, 2], 3: [17, 3]}, {4: [18, 4]}),
        (14, 15): ({1: [14, 1], 2: [14, 2]}, {3: [15, 3]}),
        (1, 9): ({1: [1]}, {2: [9, 2]}),
    }
    coupler_sides = get_busbar_sides_of_coupler(
        graph=graph,
        asset_bay_edge_id_update_dict=asset_bay_edge_dict_input,
    )
    # note: the order of the nodes in the list may vary if the order of the nodes in the edge tuple varies
    # Note: which side is one or two is not defined
    expected_coupler_sides = {
        (2, 7): ([2], [1]),
        (35, 36): ([1], [2]),
        (19, 20): ([1, 2], [3, 4]),
        (31, 32): ([1], [2]),
        (2, 3): ([2], [3]),
        (11, 12): ([1], [2]),
        (17, 18): ([1, 2, 3], [4]),
        (14, 15): ([1, 2], [3]),
        (1, 9): ([1], [2]),
    }
    for key, value in expected_coupler_sides.items():
        assert key in coupler_sides, f"Key {key} not found in coupler_sides"
        assert (coupler_sides[key]) == value, (
            f"Value for key {key} does not match expected value. Expected: {value}, Found: {coupler_sides[key]}"
        )
    assert coupler_sides == expected_coupler_sides


def test_set_switch_bay_from_edge_ids(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups
    bay1_id = (2, 7)
    bay2_id = (19, 20)
    bay3_id = (2, 3)  # single BREAKER
    no_bay_breaker_edges = [bay1_id, bay2_id, bay3_id]
    set_switch_bay_from_edge_ids(graph=graph, edge_ids=no_bay_breaker_edges)

    bay1_edge_ids = [(1, 7), (2, 7)]
    bay2_edge_ids = [(19, 1), (19, 2), (19, 20), (20, 4), (20, 3)]
    for edge in bay1_edge_ids:
        assert graph.edges[edge]["edge_connection_info"].bay_id == graph.edges[bay1_id]["grid_model_id"]
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value
    for edge in bay2_edge_ids:
        assert graph.edges[edge]["edge_connection_info"].bay_id == graph.edges[bay2_id]["grid_model_id"]
        assert graph.edges[edge]["coupler_weight"] == WeightValues.over_step.value
        assert graph.edges[edge]["bay_weight"] == WeightValues.over_step.value
    # Note: the from and to node could be reversed as the side is not explicitly defined
    assert graph.edges[bay1_id]["edge_connection_info"].from_busbar_grid_model_ids == ["node_2"]
    assert graph.edges[bay1_id]["edge_connection_info"].to_busbar_grid_model_ids == ["node_1"]
    assert graph.edges[(1, 7)]["edge_connection_info"].to_busbar_grid_model_ids == []
    assert graph.edges[(1, 7)]["edge_connection_info"].from_busbar_grid_model_ids == []
    assert graph.edges[bay1_id]["edge_connection_info"].from_coupler_ids == []
    assert graph.edges[bay1_id]["edge_connection_info"].to_coupler_ids == ["edge_(1, 7)"]
    # Note the order of the list is not defined
    assert graph.edges[bay2_id]["edge_connection_info"].from_busbar_grid_model_ids == ["node_1", "node_2"]
    assert graph.edges[bay2_id]["edge_connection_info"].to_busbar_grid_model_ids == ["node_3", "node_4"]
    assert graph.edges[bay2_id]["edge_connection_info"].from_coupler_ids == ["edge_(1, 19)", "edge_(2, 19)"]
    assert graph.edges[bay2_id]["edge_connection_info"].to_coupler_ids == ["edge_(3, 20)", "edge_(4, 20)"]
    for edge in bay2_edge_ids:
        if edge != bay2_id:
            assert graph.edges[edge]["edge_connection_info"].from_busbar_grid_model_ids == []
            assert graph.edges[edge]["edge_connection_info"].to_busbar_grid_model_ids == []
    # edge case with only one switch
    assert graph.edges[bay3_id]["edge_connection_info"].from_busbar_grid_model_ids == ["node_2"]
    assert graph.edges[bay3_id]["edge_connection_info"].to_busbar_grid_model_ids == ["node_3"]
    assert graph.edges[bay3_id]["edge_connection_info"].bay_id == graph.edges[bay3_id]["grid_model_id"]
    assert graph.edges[bay3_id]["coupler_weight"] == WeightValues.over_step.value
    assert graph.edges[bay3_id]["bay_weight"] == WeightValues.over_step.value
    assert graph.edges[bay3_id]["edge_connection_info"].from_coupler_ids == []
    assert graph.edges[bay3_id]["edge_connection_info"].to_coupler_ids == []


def test_set_all_busbar_coupling_switches(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups
    set_all_busbar_coupling_switches(graph=graph)
    bay1_id = (1, 2)  # single DISCONNECTOR
    bay2_id = (2, 3)  # single BREAKER
    assert graph.edges[bay1_id]["edge_connection_info"].bay_id == graph.edges[bay1_id]["grid_model_id"]
    assert graph.edges[bay1_id]["coupler_weight"] == WeightValues.over_step.value
    assert graph.edges[bay1_id]["bay_weight"] == WeightValues.over_step.value
    assert graph.edges[bay2_id]["edge_connection_info"].bay_id == graph.edges[bay2_id]["grid_model_id"]
    assert graph.edges[bay2_id]["coupler_weight"] == WeightValues.over_step.value
    assert graph.edges[bay2_id]["bay_weight"] == WeightValues.over_step.value


def test_get_asset_bay_id_grid_model_update_dict(graph_with_all_coupling_setups):
    graph = graph_with_all_coupling_setups
    asset_bay_edge_dict_input = {
        (2, 7): ({2: [2]}, {1: [7, 1]}),
        (35, 36): ({1: [35, 34, 33, 1]}, {2: [36, 2]}),
        (2, 3): ({2: [2]}, {3: [3]}),
    }

    coupler_update_expected = {(2, 7): {2: [2, 7]}, (35, 36): {35: [35, 36]}, (2, 3): {2: [2, 3]}}
    side1_update_expected = {(2, 7): {2: [2]}, (35, 36): {1: [35, 34, 33, 1]}, (2, 3): {2: [2]}}
    side2_update_expected = {(2, 7): {1: [7, 1]}, (35, 36): {2: [36, 2]}, (2, 3): {3: [3]}}
    coupler_update, side1_update, side2_update = get_asset_bay_id_grid_model_update_dict(
        asset_bay_edge_id_update_dict=asset_bay_edge_dict_input,
    )
    assert coupler_update == coupler_update_expected
    assert side1_update == side1_update_expected
    assert side2_update == side2_update_expected


def testget_coupler_bay_edge_ids():
    data = {(2, 7): {1: [7, 1]}, (19, 20): {3: [20, 3], 4: [20, 4]}, (2, 3): {3: [3]}}
    res = get_coupler_bay_edge_ids(asset_bay_edge_id_update_dict=data)
    expected = {(2, 7): [(7, 1)], (19, 20): [(20, 3), (20, 4)], (2, 3): []}
    assert res == expected
