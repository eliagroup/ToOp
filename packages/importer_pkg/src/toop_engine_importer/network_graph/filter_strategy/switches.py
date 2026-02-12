# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Filter strategy for switches.

The Issue:
There are many paths between two busbars, list of all known connection paths.
DISCONNECTOR -> DS
CIRCUIT BREAKER -> CB
BUSBAR -> B

Busbar to Busbar (single path with no additional parallel busbars):
- B -> DS -> B
- B -> CB -> B
- B -> DS -> CB -> B
- B -> CB -> DS -> B
- B -> DS -> CB -> DS -> B
- B -> DS -> CB -> CB -> DS -> B
- B -> DS -> CB -> DS -> CB -> DS -> B

More complex paths, e.g. with multiple busbars, all paths above can be extended with additional busbars:
only few examples.
Two busbars on either/both sides of a busbar:
- B1 -> DS1 -> CB -> DS3 -> B3
            ^
- B2 -> DS2 |

Three busbars on either/both sides of a busbar:
- B1 -> DS1 -> CB -> DS3 -> B3
            ^
- B2 -> DS2 |
            |
- B4 -> DS4 |

Solution:
1. Get all BREAKER, which have no bay_id (e.g. excludes Line/Load breaker)
2. Loop over BREAKER one by one
    2.1 Loop: Get all shortest paths from BREAKER to all BUSBARs (respect weights)
    2.2 Set coupler sides
    2.3 Set bay_id
    2.4 Set coupler type
3. Get all DISCONNECTOR (to find left over connections like B -> DS -> B)
4. Repeat step 2. for DS
5. assert all bay_id are set for all DS and CB in the graph

"""

from itertools import pairwise

import logbook
import networkx as nx
from beartype.typing import Literal, Union
from toop_engine_importer.network_graph.data_classes import EDGE_ID, WeightValues
from toop_engine_importer.network_graph.filter_strategy.helper_functions import (
    calculate_asset_bay_for_node_assets,
    get_edge_attr_for_dict_key,
    get_edge_attr_for_dict_list,
    set_asset_bay_edge_attr,
    set_single_bay_weight,
)
from toop_engine_importer.network_graph.network_graph import (
    get_busbar_connection_info_attribute,
    get_busbar_true_nodes,
    get_edge_list_by_attribute,
    update_edge_connection_info,
)
from toop_engine_importer.network_graph.network_graph_helper_functions import (
    find_matching_node_in_list,
    remove_path_multiple_busbars,
    reverse_dict_list,
)

logger = logbook.Logger(__name__)


def set_all_busbar_coupling_switches(
    graph: nx.Graph,
) -> None:
    """Set all connection paths between busbars, be it BREAKER or DISCONNECTOR.

    1. Get all BREAKER, which have no bay_id (e.g. excludes Line/Load breaker)
    2. Loop over BREAKER one by one
        2.1 Loop: Get all shortest paths from BREAKER to all BUSBARs (respect weights)
        2.2 Set coupler sides
        2.3 After loop: Set bay_id
        2.4 Set coupler type
    3. Get all DISCONNECTOR (to find left over connections like B -> DS -> B)
    4. Repeat step 2. for DS

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    """
    # 1. Get all BREAKER, which have no bay_id (e.g. excludes Line/Load breaker)
    no_bay_breaker_edges = get_switches_with_no_bay_id(graph=graph, asset_type="BREAKER")
    # 2. Loop over BREAKER one by one
    # 2.1 Loop: Get all shortest paths from BREAKER to all BUSBARs (respect weights)
    set_switch_bay_from_edge_ids(graph=graph, edge_ids=no_bay_breaker_edges)
    # 3. Get all DISCONNECTOR (to find left over connections like B -> DS -> B)
    no_bay_breaker_edges = get_switches_with_no_bay_id(graph=graph, asset_type="DISCONNECTOR")
    set_switch_bay_from_edge_ids(graph=graph, edge_ids=no_bay_breaker_edges)


def set_switch_bay_from_edge_ids(
    graph: nx.Graph,
    edge_ids: list[EDGE_ID],
) -> None:
    """Set the bay for a switch.

    Loops over the edge_ids and sets the bay for each edge_id.
    If there are multiple edges with the same bay_id, the first one is used.
    Note: if you provide all switches of one bay, the function will
    set the first edge as the bay_id, no matter the type of the edge.
    You may provide e.g. all BREAKER first, then all DISCONNECTOR with no bay_id.
    This way the bay id will be set for a BREAKER if there is one in the path.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    edge_ids : list[EDGE_ID]
        A list of edges to find the bay for.
        Note: only the first edge (regardless of type) in a path is used to set the bay_id.
    """
    # 2. Loop over BREAKER or DISCONNECTOR one by one
    asset_bay_edge_id_update_dict = get_switch_bay_dict(graph=graph, switch_edge_list=edge_ids)
    # 2.2 Set coupler sides
    coupler_busbar_sides = get_busbar_sides_of_coupler(
        graph=graph,
        asset_bay_edge_id_update_dict=asset_bay_edge_id_update_dict,
    )
    set_coupler_busbar_sides(graph=graph, busbar_sides_of_coupler=coupler_busbar_sides)
    # 2.3 After loop: Set bay_id for whole path
    coupler_update, side1_update, side2_update = get_asset_bay_id_grid_model_update_dict(
        asset_bay_edge_id_update_dict=asset_bay_edge_id_update_dict
    )
    set_coupler_bay_ids(
        graph=graph,
        side1_update=side1_update,
        side2_update=side2_update,
    )
    set_bay_attr_for_coupler_paths(
        graph=graph,
        coupler_update=coupler_update,
        side1_update=side1_update,
        side2_update=side2_update,
    )
    # 2.4 Set coupler type
    set_coupler_type(graph=graph, coupler_sides=coupler_busbar_sides)


def set_bay_attr_for_coupler_paths(
    graph: nx.Graph,
    coupler_update: dict[EDGE_ID, dict[int, list[int]]],
    side1_update: dict[EDGE_ID, dict[int, list[int]]],
    side2_update: dict[EDGE_ID, dict[int, list[int]]],
) -> None:
    """Set the bay attributes for the coupler paths.

    This function sets the bay attributes for the coupler paths.
    The coupler paths are the paths between the busbars and the coupler.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    coupler_update : dict[EDGE_ID, dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    side1_update : dict[EDGE_ID, dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    side2_update : dict[EDGE_ID, dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    # get the bay_id for the coupler paths
    coupler_grid_model_id_update = get_edge_attr_for_dict_key(
        graph=graph,
        input_dict=coupler_update,
        attribute="grid_model_id",
    )
    side1_grid_model_id_update = get_edge_attr_for_dict_key(
        graph=graph,
        input_dict=side1_update,
        attribute="grid_model_id",
    )
    side2_grid_model_id_update = get_edge_attr_for_dict_key(
        graph=graph,
        input_dict=side2_update,
        attribute="grid_model_id",
    )
    # set the bay_id for the coupler paths
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=coupler_grid_model_id_update)
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=side1_grid_model_id_update)
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=side2_grid_model_id_update)


def set_coupler_bay_ids(
    graph: nx.Graph,
    side1_update: dict[EDGE_ID, dict[int, list[int]]],
    side2_update: dict[EDGE_ID, dict[int, list[int]]],
) -> None:
    """Set the from_coupler_ids and to_coupler_ids in the EdgeConnectionInfo of the coupler.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    side1_update : dict[EDGE_ID, dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    side2_update : dict[EDGE_ID, dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    side1_bay_dict = get_coupler_bay_edge_ids(asset_bay_edge_id_update_dict=side1_update)
    side2_bay_dict = get_coupler_bay_edge_ids(asset_bay_edge_id_update_dict=side2_update)
    side_1_grid_model_ids = get_edge_attr_for_dict_list(
        graph=graph,
        input_dict=side1_bay_dict,
        attribute="grid_model_id",
    )
    side_2_grid_model_ids = get_edge_attr_for_dict_list(
        graph=graph,
        input_dict=side2_bay_dict,
        attribute="grid_model_id",
    )
    side1_update_dict = {}
    side2_update_dict = {}
    for edge_id, grid_model_ids in side_1_grid_model_ids.items():
        side1_update_dict[edge_id] = {"from_coupler_ids": grid_model_ids}
    for edge_id, grid_model_ids in side_2_grid_model_ids.items():
        side2_update_dict[edge_id] = {"to_coupler_ids": grid_model_ids}
    update_edge_connection_info(graph, side1_update_dict)
    update_edge_connection_info(graph, side2_update_dict)


def get_coupler_bay_edge_ids(
    asset_bay_edge_id_update_dict: dict[EDGE_ID, dict[int, list[int]]],
) -> dict[EDGE_ID, list[EDGE_ID]]:
    """Get the coupler bay ids.

    Parameters
    ----------
    asset_bay_edge_id_update_dict : dict[tuple[int,int], dict[int, list[int]]]
        A dictionary containing the found busbars.
        key: coupler_edge_id
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)

    Returns
    -------
    coupler_bay_ids : dict[tuple[int,int], list[tuple[int,int]]]
        A dictionary containing the coupler bay ids.
        key: coupler_edge_id
        value: list of edge_ids (a tuple of node_ids) that are part of the coupler
               leading to the busbar
    """
    bay_dict = {}
    for edge_id, shortest_path_to_busbar_dict in asset_bay_edge_id_update_dict.items():
        bay_list = []
        for path in shortest_path_to_busbar_dict.values():
            bay_list += [(from_id, to_id) for from_id, to_id in pairwise(path)]
        bay_dict[edge_id] = bay_list

    return bay_dict


def set_coupler_type(
    graph: nx.Graph,
    coupler_sides: dict[EDGE_ID, tuple[list[int], list[int]]],
) -> None:
    """Set the coupler type in the nx.Graph (based on NetworkGraphData model).

    Warning: this assumes that all assets of horizontal connections are connected have an bay to all busbars.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    coupler_sides : dict[EDGE_ID, tuple[list[int], list[int]]]
        A dictionary containing the sides of the coupler.
        key: edge_id (a tuple of node_ids)
        value: tuple of two lists of busbar ids
        which side is from and which side is to is not defined.
        The order of the busbar ids is not important.
    """
    connectable_assets_dict = get_busbar_connection_info_attribute(graph, "connectable_assets", node_type="busbar")
    coupler_categories = get_coupler_type(connectable_assets=connectable_assets_dict, coupler_sides=coupler_sides)
    set_coupler_type_graph(graph=graph, coupler_categories=coupler_categories)


def get_coupler_type(
    connectable_assets: dict[int, list[Union[str, int]]],
    coupler_sides: dict[EDGE_ID, tuple[list[int], list[int]]],
) -> dict[str, list[EDGE_ID]]:
    """Categorize the coupler in the NetworkGraphData model.

    A logic to categorize the coupler into busbar coupler and cross coupler.
    Uses the BusbarConnectionInfo of each node to find the coupler.

    Note: this function is independent of any coupler information.
    Match the results with the couplers in the graph.

    Parameters
    ----------
    connectable_assets : dict[int, list[Union[str, int]]]
        A dictionary containing the connectable assets.
        Key: busbar_id
        Value: list of connectable asset ids
    coupler_sides : dict[EDGE_ID, tuple[list[int], list[int]]]
        A dictionary containing the sides of the coupler.
        key: edge_id (a tuple of node_ids)
        value: tuple of two lists of busbar ids
        which side is from and which side is to is not defined.
        The order of the busbar ids is not important.

    Returns
    -------
    coupler_categories: dict[str, list[EDGE_ID]]
        A dictionary containing the coupler_categories.
        Key: busbar_coupler or cross_coupler
        Value: list of busbar id tuples
    """
    coupler_categories = {"busbar_coupler": [], "cross_coupler": []}
    # get A set of busbar ids for each coupler
    busbar_coupler_tuple = {
        coupler: (busbar_side1[0], busbar_side2[0]) for coupler, (busbar_side1, busbar_side2) in coupler_sides.items()
    }

    for coupler, (busbar1, busbar2) in busbar_coupler_tuple.items():
        if busbar_coupler_condition(busbar1=busbar1, busbar2=busbar2, connectable_assets=connectable_assets):
            coupler_categories["busbar_coupler"].append(coupler)
        else:
            coupler_categories["cross_coupler"].append(coupler)
    return coupler_categories


def busbar_coupler_condition(
    busbar1: int, busbar2: int, connectable_assets: dict[int, list[str]], threshold: float = 0.5
) -> bool:
    """Check if a busbar is a busbar coupler.

    A busbar is a busbar coupler if a percentage of connected assets are connected to all other busbars.

    Parameters
    ----------
    busbar1 : int
        The busbar1 id.
    busbar2 : int
        The busbar2 id.
    connectable_assets : dict[int, list[str]]
        A dictionary containing the connectable assets.
        Key: busbar_id
        Value: list of connectable asset ids
    threshold : float
        The threshold of connected assets to all other busbars.
        The percentage above the threshold is considered a busbar coupler.
        example:
            connectable_assets[busbar] = [1,2,3,4]
            connectable_assets[connected_bus] = [1,2,3]
            threshold = 0.5
            -> busbar is still a busbar coupler,
            even if only 3 out of 4 assets are connected to both busbars.

    Returns
    -------
    bool
        True if the busbar is a busbar coupler.
    """
    n_connected_elements = sum(elem in connectable_assets[busbar2] for elem in connectable_assets[busbar1])
    if len(connectable_assets[busbar1]) >= len(connectable_assets[busbar2]):
        n_total_elements = len(connectable_assets[busbar1])
    else:
        n_total_elements = len(connectable_assets[busbar2])
    if n_total_elements > 0 and n_connected_elements / n_total_elements >= threshold:
        return True
    return False


def set_coupler_type_graph(
    graph: nx.Graph,
    coupler_categories: dict[str, list[EDGE_ID]],
) -> None:
    """Set the "coupler_type" in the NetworkX Graph EdgeConnectionInfo.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    coupler_categories : dict[str, list[EDGE_ID]]
        A dictionary containing the coupler_categories.
        Key: busbar_coupler or cross_coupler
        Value: list of busbar id tuples
    """
    update_edge = reverse_dict_list(dict_of_lists=coupler_categories)
    update_edge_dict = {}
    for edge, type_content in update_edge.items():
        # check if the edge is already in the update_edge_dict
        update_edge_dict[edge] = {"coupler_type": type_content[0]}
    update_edge_connection_info(graph=graph, update_edge_dict=update_edge_dict)


def get_switches_with_no_bay_id(graph: nx.Graph, asset_type: Literal["BREAKER", "DISCONNECTOR"]) -> list[EDGE_ID]:
    """Get all switches with no bay_id.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    asset_type : Literal["BREAKER", "DISCONNECTOR"]
        The asset type to find.
        Can be either "BREAKER" or "DISCONNECTOR".

    Returns
    -------
    switches : list[EDGE_ID]
        A list of all switches with no bay_id.
    """
    breaker_switches_tuple = get_edge_list_by_attribute(graph, attribute="asset_type", value=[asset_type])
    no_bay_edges = get_edge_list_by_attribute(graph, attribute="bay_weight", value=[0.0])
    no_bay_breaker_edges = list(set(breaker_switches_tuple) & set(no_bay_edges))
    return no_bay_breaker_edges


def get_switch_bay_dict(
    graph: nx.Graph,
    switch_edge_list: list[EDGE_ID],
) -> dict[EDGE_ID, tuple[dict[int, list[int]], dict[int, list[int]]]]:
    """Get the bay for a switch.

    Note: if there are two BREAKER in one connection path, this function will return
    only the first one.

    Examples
    --------
    - B -> DS -> CB1 -> CB2 -> DS -> B
    - B -> DS -> CB1 -> DS -> CB2 -> DS -> B
    Assuming CB1 is the first in the list, the function will return only CB1.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    switch_edge_list : list[EDGE_ID]
        A list of edges to find the bay for.
        This can be a list of edges with a BREAKER or DISCONNECTOR.
        The edges are used to find the bay for the asset.

    Returns
    -------
    asset_bay_update_dict : dict[tuple[int,int], tuple[dict[int, list[int]], dict[int, list[int]]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: tuple of dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    _busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    asset_bay_edge_id_update_dict = {}

    for edge_id in switch_edge_list:
        # set the bay weight for the edge_id, ensures that only one side is found
        set_single_bay_weight(graph=graph, edge_id=edge_id, bay_weight=WeightValues.over_step.value)

        if edge_id[0] in busbars_helper_nodes:
            # the edge is directly connected to the busbar
            shortest_path_to_busbar_side1_dict = {edge_id[0]: [edge_id[0]]}
        else:
            shortest_path_to_busbar_side1_dict = calculate_asset_bay_for_node_assets(
                graph=graph, busbars_helper_nodes=busbars_helper_nodes, asset_node=edge_id[0]
            )
        if edge_id[1] in busbars_helper_nodes:
            # the edge is directly connected to the busbar
            shortest_path_to_busbar_side2_dict = {edge_id[1]: [edge_id[1]]}
        else:
            shortest_path_to_busbar_side2_dict = calculate_asset_bay_for_node_assets(
                graph=graph, busbars_helper_nodes=busbars_helper_nodes, asset_node=edge_id[1]
            )
        # sort out paths with multiple busbars
        shortest_path_to_busbar_side1_dict = remove_path_multiple_busbars(
            path_dict=shortest_path_to_busbar_side1_dict, busbars=busbars_helper_nodes
        )
        shortest_path_to_busbar_side2_dict = remove_path_multiple_busbars(
            path_dict=shortest_path_to_busbar_side2_dict, busbars=busbars_helper_nodes
        )

        if len(shortest_path_to_busbar_side1_dict) == 0 or len(shortest_path_to_busbar_side2_dict) == 0:
            # switch is part of a previous found path
            # -> skip
            set_single_bay_weight(graph=graph, edge_id=edge_id, bay_weight=WeightValues.low.value)
        else:
            # current path is not a sub path of a previous found path
            # -> save path
            asset_bay_edge_id_update_dict[edge_id] = (shortest_path_to_busbar_side1_dict, shortest_path_to_busbar_side2_dict)
    return asset_bay_edge_id_update_dict


def get_busbar_sides_of_coupler(
    graph: nx.Graph,
    asset_bay_edge_id_update_dict: dict[EDGE_ID, tuple[dict[int, list[int]], dict[int, list[int]]]],
) -> dict[EDGE_ID, tuple[list[int], list[int]]]:
    """Get the sides of the coupler.

    A coupler has two sides, the from and to side. Both sides can connect to
    multiple busbars.
    This function gets the busbar ids for the busbar_helper_node ids.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
    asset_bay_edge_id_update_dict : dict[EDGE_ID, tuple[dict[int, list[int]], dict[int, list[int]]]]
        A dictionary containing the found busbars.
        key: EDGE_ID (a tuple of node_ids)
        value: list of busbar_helper_node ids.

    Returns
    -------
    sides_of_coupler : dict[EDGE_ID, tuple[list[int], list[int]]]
        A dictionary containing the sides of the coupler.
        key: edge_id (a tuple of node_ids)
        value: tuple of two lists of busbar ids
        Which side is from and which side is to is not defined.
        The order of the tuple is defined by the edge_id order.
        Note: the busbar_helper_node are transformed to busbar ids.
    """
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    sides_of_coupler = {}

    for edge_id, side_paths in asset_bay_edge_id_update_dict.items():
        side1_path = side_paths[0]
        side2_path = side_paths[1]
        side1 = [find_matching_node_in_list(node_id, busbars_helper_nodes, busbars) for node_id in side1_path.keys()]
        side2 = [find_matching_node_in_list(node_id, busbars_helper_nodes, busbars) for node_id in side2_path.keys()]
        sides_of_coupler[edge_id] = (side1, side2)

    return sides_of_coupler


def set_coupler_busbar_sides(
    graph: nx.Graph,
    busbar_sides_of_coupler: dict[EDGE_ID, tuple[list[int], list[int]]],
) -> None:
    """Set the coupler busbar side in the nx.Graph (based on NetworkGraphData model).

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    busbar_sides_of_coupler : dict[EDGE_ID, tuple[list[int], list[int]]]
        A dictionary containing the sides of the coupler.
        key: EDGE_ID (a tuple of node_ids)
        value: tuple of two lists of busbar ids
    """
    update_edge_dict = {}
    for edge_id, (from_busbars, to_busbars) in busbar_sides_of_coupler.items():
        from_busbar_grid_model_ids = [graph.nodes[busbar_id]["grid_model_id"] for busbar_id in from_busbars]
        to_busbar_grid_model_ids = [graph.nodes[busbar_id]["grid_model_id"] for busbar_id in to_busbars]
        # set the coupler side
        update_edge_dict[edge_id] = {
            "from_busbar_grid_model_ids": from_busbar_grid_model_ids,
            "to_busbar_grid_model_ids": to_busbar_grid_model_ids,
        }

    update_edge_connection_info(graph=graph, update_edge_dict=update_edge_dict)


def get_asset_bay_id_grid_model_update_dict(
    asset_bay_edge_id_update_dict: dict[EDGE_ID, tuple[dict[int, list[int]], dict[int, list[int]]]],
) -> tuple[
    dict[EDGE_ID, dict[int, list[int]]],
    dict[EDGE_ID, dict[int, list[int]]],
    dict[EDGE_ID, dict[int, list[int]]],
]:
    """Get the asset bay id grid model update dict for set_asset_bay_edge_attr.

    Replaces the edge ids with the grid model ids.
    Adds the coupler itself to the dict if missing.

    Parameters
    ----------
    asset_bay_edge_id_update_dict : dict[tuple[int,int], tuple[dict[int, list[int]], dict[int, list[int]]]]
        A dictionary containing the found busbars.
        key: bay_edge_id
        value: tuple of dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)

    Returns
    -------
    asset_bay_id_grid_model_update_dict : tuple[
                                                dict[EDGE_ID, dict[int, list[int]]],
                                                dict[EDGE_ID, dict[int, list[int]]],
                                                dict[EDGE_ID, dict[int, list[int]]]
                                                ]
        A tuple (coupler, side1, side2) of dictionary containing the found busbars.
        key: bay_id (an EDGE_ID)
        value: dictionary with
            key: busbar_id and
            value: list of node_ids from the asset_node the key (a busbars_helper_node)
    """
    asset_bay_id_side1_update_dict = {}
    asset_bay_id_side2_update_dict = {}
    asset_bay_id_coupler_update_dict = {}
    for edge_id, shortest_path_to_busbar_dict in asset_bay_edge_id_update_dict.items():
        # get coupler update dict, enforces that the coupler itself gets updated
        asset_bay_id_coupler_update_dict[edge_id] = {edge_id[0]: [edge_id[0], edge_id[1]]}
        asset_bay_id_side1_update_dict[edge_id] = shortest_path_to_busbar_dict[0]
        asset_bay_id_side2_update_dict[edge_id] = shortest_path_to_busbar_dict[1]

    return asset_bay_id_coupler_update_dict, asset_bay_id_side1_update_dict, asset_bay_id_side2_update_dict
