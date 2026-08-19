# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Empty Bay Filter Strategy.

This module contains functions to identify and handle empty bays in a network graph.
An empty bay is defined as a node in the graph that has no connected assets, but has still a bay.

These empty bays need to be identified, so couplers can be found and categorized correctly.
The main function is `set_empty_bay_weights`, which sets the bay weight for empty bays.
"""

import networkx as nx
from toop_engine_grid_helpers.network_graph.data_classes import WeightValues
from toop_engine_grid_helpers.network_graph.network_graph import (
    flatten_list_of_mixed_entries,
    get_busbar_true_nodes,
    multi_weight_function,
)
from toop_engine_grid_helpers.network_graph.network_graph_helper_functions import (
    find_longest_path_ids,
    get_pair_tuples_from_list,
)


def set_empty_bay_weights(graph: nx.Graph) -> None:
    """Set a bay weight for empty bays.

    Due to data quality issues, some bays may be empty.
    Finding and categorizing couplers depends on all bays weights being set.
    This function sets a bay weight for empty bays.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    """
    empty_asset_bay_lists = get_empty_bay_list(graph=graph)
    update_dict = get_empty_bay_update_dict(empty_bay_lists=empty_asset_bay_lists)
    nx.set_edge_attributes(graph, update_dict)


def get_empty_bay_list(graph: nx.Graph) -> list[list[int]]:
    """Get a list of empty bays.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.

    Returns
    -------
    empty_asset_bay_lists : list[list[int]]
        A list of empty bays.
        contains the node_ids of the empty bay path.
    """
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    weight_list = ["busbar_weight", "bay_weight"]
    cutoff = WeightValues.max_step.value
    empty_asset_bay_lists = []
    for busbar_id in busbars_helper_nodes:
        station_node_paths = nx.single_source_dijkstra_path(
            graph, source=busbar_id, weight=multi_weight_function(weight_list), cutoff=cutoff
        )
        longest_path_ids = find_longest_path_ids(station_node_paths)
        # has only one neighbor -> dead end -> empty bay
        # dead end is not a busbar in the helper nodes system
        asset_bay_nodes = [node_id for node_id in longest_path_ids if len(graph[node_id]) == 1 and node_id not in busbars]
        empty_asset_bay_lists += [station_node_paths[asset_bay_node] for asset_bay_node in asset_bay_nodes]
    empty_asset_bay_lists += get_empty_busbar_to_busbar_disconnector_paths(graph=graph)
    return empty_asset_bay_lists


def get_empty_busbar_to_busbar_disconnector_paths(graph: nx.Graph) -> list[list[int]]:
    """Identify empty-bay remnants made of two disconnectors between two busbars.

    Imported node-breaker models can contain a deleted asset position where only the two
    selector disconnectors to the busbars remain. Those structures are not real busbar
    couplers and must be marked as empty bays before coupler extraction runs.
    """
    busbars, _busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    busbar_node_ids = set(busbars)
    empty_paths: list[list[int]] = []

    for node_id, node_data in graph.nodes(data=True):
        if node_id in busbar_node_ids or node_data.get("helper_node", False):
            continue
        if len(graph[node_id]) != 2:
            continue
        busbar_connection_info = node_data.get("busbar_connection_info")
        if busbar_connection_info is not None and busbar_connection_info.node_assets:
            continue

        neighbor_ids = list(graph.neighbors(node_id))
        if any(neighbor_id not in busbar_node_ids for neighbor_id in neighbor_ids):
            continue

        edge_data = [graph.get_edge_data(node_id, neighbor_id) for neighbor_id in neighbor_ids]
        if not all(str(edge["asset_type"]) == "DISCONNECTOR" for edge in edge_data):
            continue

        empty_paths.append([neighbor_ids[0], node_id, neighbor_ids[1]])

    return empty_paths


def get_empty_bay_update_dict(empty_bay_lists: list[list[int]]) -> dict[tuple[int, int], dict[str, float]]:
    """Get the empty bay update dictionary for the nx.Graph.

    The empty bay update dictionary is used set the bay weight.

    Parameters
    ----------
    empty_bay_lists : list[list[int]]
        A list of empty bays.
        contains the node_ids of the empty bay path.

    Returns
    -------
    update_edge_dict : dict[tuple[int,int], dict[str, float]]
        A dictionary containing the update information for each edge.
        keys: edge_id (a tuple of node_ids)
        values: {"bay_weight": WeightValues.max_step.value}

    """
    content = {"bay_weight": WeightValues.max_step.value, "empty_bay": True}
    empty_bay_paired_tuples = [get_pair_tuples_from_list(empty_bay_list) for empty_bay_list in empty_bay_lists]
    empty_bay_paired_tuples = flatten_list_of_mixed_entries(empty_bay_paired_tuples)
    # remove duplicates
    empty_bay_paired_tuples = list(set(empty_bay_paired_tuples))
    update_edge_dict = {edge_id: content for edge_id in empty_bay_paired_tuples}
    return update_edge_dict
