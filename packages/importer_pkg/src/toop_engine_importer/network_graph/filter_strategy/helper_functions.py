# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for the filter strategy."""

from itertools import pairwise
from typing import Any, TypeVar

import networkx as nx
from toop_engine_importer.network_graph.data_classes import WeightValues
from toop_engine_importer.network_graph.network_graph import (
    multi_weight_function,
    shortest_paths_to_target_ids,
    update_edge_connection_info,
)

T = TypeVar("T")


def set_asset_bay_edge_attr(
    graph: nx.Graph,
    asset_bay_update_dict: dict[str, dict[int, list[int]]],
) -> None:
    """Set the bay information in the nx.Graph.

    This function updates the initial edge weights and id related to the asset bay.
    Sets the bay_id, bay_weight, and coupler_weight for each edge in the shortest path to a busbar.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    asset_bay_update_dict : dict[str, dict[int, list[int]]]
        A dictionary containing the shortest path to a busbar for each busbar.
        key: bay_id (a str grid_model_id)
        value: dictionary from shortest_paths_to_target_ids
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    edge_update_dict = {}
    content_dict = {
        "bay_weight": WeightValues.over_step.value,
        "coupler_weight": WeightValues.over_step.value,
    }
    for grid_model_id, shortest_path_to_busbar_dict in asset_bay_update_dict.items():
        for path in shortest_path_to_busbar_dict.values():
            # TODO: edge_update_dict.update does not work if one bay has multiple assets
            # refactor bay_id to list[str]
            # current behavior: if multiple assets are connected to the same bay,
            # the bay_id is set to the last asset in the path
            # set bay_id, bay_weight, coupler_weight
            edge_update_dict.update({(from_id, to_id): {"bay_id": grid_model_id} for from_id, to_id in pairwise(path)})
            update_dict = {(s_id, t_id): content_dict for s_id, t_id in pairwise(path)}
            nx.set_edge_attributes(graph, update_dict)
    update_edge_connection_info(graph=graph, update_edge_dict=edge_update_dict)


def calculate_asset_bay_for_node_assets(
    graph: nx.Graph,
    asset_node: int,
    busbars_helper_nodes: list[int],
) -> dict[int, list[int]]:
    """Calculate the bay for an asset starting from a node.

    The "busbar_weight" and "bay_weight" is used to find the bay of an asset,
    a connection path from the asset to a busbar.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    asset_node : int
        The node id of the asset.
    busbars_helper_nodes : list[int]
        A list of busbar helper, if the true connection of the busbar is at a different node.
        See network_graph.get_helper_node_ids() for more information.

    Returns
    -------
    shortest_path_to_busbar_dict : dict[int, list[int]]
        key: busbars_helper_nodes
        value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    weights_list = ["busbar_weight", "bay_weight"]
    shortest_path_to_busbar_dict = shortest_paths_to_target_ids(
        graph,
        target_node_ids=busbars_helper_nodes,
        start_node_id=asset_node,
        weight=multi_weight_function(weights_list),
        cutoff=WeightValues.max_step.value,
    )
    return shortest_path_to_busbar_dict


def set_single_bay_weight(
    graph: nx.Graph,
    edge_id: tuple[int, int],
    bay_weight: WeightValues,
) -> None:
    """Set the bay weight for a single edge in the nx.Graph.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    edge_id : tuple[int, int]
        The edge id (a tuple of node_ids).
    bay_weight : float
        The bay weight to set.
    """
    content_dict = {
        "bay_weight": bay_weight,
    }
    update_dict = {edge_id: content_dict}
    nx.set_edge_attributes(graph, update_dict)


def get_edge_attr_for_dict_list(
    graph: nx.Graph,
    input_dict: dict[T, list[tuple[int, int]]],
    attribute: str = "grid_model_id",
) -> dict[T, list[Any]]:
    """Get the grid model id from a dictionary of edges and their connections.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    input_dict : dict[T, list[tuple[int, int]]]
        A dictionary containing lists of edge_ids
        key: T
        value: list of tuples of node_ids (edge_id)
    attribute : Any
        The attribute to get from the edge.
        Default is "grid_model_id".


    Returns
    -------
    dict[T, list[Any]]
        A dictionary with edges as keys and a list it's attribute as values.
        Return type is "Any" as the the type depends on the attribute in the graph.
    """
    output_dict = {}
    for key, edge_list in input_dict.items():
        output_dict[key] = [graph.edges[edge_id][attribute] for edge_id in edge_list]
    return output_dict


def get_edge_attr_for_dict_key(
    graph: nx.Graph,
    input_dict: dict[tuple[int, int], T],
    attribute: str = "grid_model_id",
) -> dict[Any, T]:
    """Get the grid model id from a dictionary of edges and their connections.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    input_dict : dict[tuple[int,int], T]
        A dictionary containing lists of edge_ids
        key: tuple of node_ids (edge_id)
        value: T
    attribute : str
        The attribute to get from the edge.
        Default is "grid_model_id".

    Returns
    -------
    dict[Any, T]
        A dictionary with the attribute as keys and a list of edge_ids as values.
        Return type is "Any" as the the type depends on the attribute in the graph.
    """
    output_dict = {}
    for edge_id, value in input_dict.items():
        attribute_value = graph.edges[edge_id][attribute]
        output_dict[attribute_value] = value
    return output_dict
