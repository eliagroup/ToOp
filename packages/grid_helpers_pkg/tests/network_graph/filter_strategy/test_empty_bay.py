# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
from toop_engine_grid_helpers.network_graph.data_classes import BusbarConnectionInfo, EdgeConnectionInfo, WeightValues
from toop_engine_grid_helpers.network_graph.filter_strategy.empty_bay import set_empty_bay_weights


def test_empty_bay():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1")
    graph.add_node(10, node_type="busbar", grid_model_id="bb2")

    # bay 1 has been set
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
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

    # bay 2 has been set
    graph.add_edge(
        1,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
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
        6,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        8,
        9,
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

    set_empty_bay_weights(graph=graph)
    # empty bay should be set
    assert graph.edges[(1, 5)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(10, 5)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(5, 6)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(6, 7)]["bay_weight"] == WeightValues.max_step.value

    # coupler should not be set
    assert graph.edges[(1, 8)]["bay_weight"] == WeightValues.low.value
    assert graph.edges[(8, 9)]["bay_weight"] == WeightValues.low.value
    assert graph.edges[(9, 10)]["bay_weight"] == WeightValues.low.value


def test_empty_bay_marks_two_disconnector_empty_bay_between_busbars() -> None:
    graph = nx.Graph()
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", helper_node=False)
    graph.add_node(2, node_type="busbar", grid_model_id="bb2", helper_node=False)
    graph.add_node(3, node_type="node", grid_model_id="mid", helper_node=False)

    for node_id in [1, 2, 3]:
        graph.nodes[node_id]["busbar_connection_info"] = BusbarConnectionInfo()

    graph.add_edge(
        1,
        3,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        2,
        asset_type="DISCONNECTOR",
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    set_empty_bay_weights(graph=graph)

    assert graph.edges[(1, 3)]["empty_bay"] is True
    assert graph.edges[(3, 2)]["empty_bay"] is True
    assert graph.edges[(1, 3)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(3, 2)]["bay_weight"] == WeightValues.max_step.value
