"""The default filter strategy for the NetworkGraphData model.

The default strategy fills:
    - the BusbarConnectionInfo of all relevant nodes in the network graph.
    - the bay weights for asset nodes.
    - EdgeConnectionInfo
"""

from typing import Union

import logbook
import networkx as nx
from toop_engine_importer.network_graph.data_classes import WeightValues
from toop_engine_importer.network_graph.filter_strategy.empty_bay import set_empty_bay_weights
from toop_engine_importer.network_graph.filter_strategy.helper_functions import (
    calculate_asset_bay_for_node_assets,
    set_asset_bay_edge_attr,
)
from toop_engine_importer.network_graph.filter_strategy.switches import set_all_busbar_coupling_switches
from toop_engine_importer.network_graph.network_graph import (
    flatten_list_of_mixed_entries,
    get_busbar_true_nodes,
    get_helper_node_ids,
    get_node_list_by_attribute,
    get_nodes_ids_with_a_connected_asset,
    multi_weight_function,
    shortest_paths_to_target_ids,
    update_busbar_connection_info,
    update_edge_connection_info,
)
from toop_engine_importer.network_graph.network_graph_helper_functions import (
    find_matching_node_in_list,
    find_shortest_path_ids,
    reverse_dict_list,
)

logger = logbook.Logger(__name__)


def run_default_filter_strategy(graph: nx.Graph) -> None:
    """Run the default strategy on the nx.Graph created by the NetworkGraphData model.

    This is the main function for the default strategy.
    The default strategy fills
    - BusbarConnectionInfo of all busbars in the network graph.
    - EdgeConnectionInfo of all edges in the network graph.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    """
    set_switch_busbar_connection_info(graph=graph)
    set_bay_weights(graph=graph)
    set_empty_bay_weights(graph=graph)
    set_connectable_busbars(graph=graph)
    set_all_busbar_coupling_switches(graph=graph)
    set_zero_impedance_connected(graph=graph)


def set_switch_busbar_connection_info(graph: nx.Graph) -> None:
    """Set the switch busbar connection information in the network graph model.

    why:
        - cut off the shortest path at a busbar
        - direct_busbar_grid_model_id in combination with bay_id for switching table
    uses:
        - graph.nodes[node_id][`"node_type"`] == "busbar"
    sets:
        - set the direct_busbar_grid_model_id in the switches
        - set the busbar_weight to step for all edges around a busbar.
          This enables to cut off a shortest path at a busbar.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    """
    busbars = get_node_list_by_attribute(graph=graph, attribute="node_type", value=["busbar"])
    min_len_path = 2  # since we are searching for an edge, we need two nodes
    update_edge_dict = {}
    for busbar_id in busbars:
        # helper_nodes = get_helper_node_ids(network_graph_data)
        helper_nodes = get_helper_node_ids(graph=graph)
        length, path = nx.single_source_dijkstra(
            graph, source=busbar_id, cutoff=WeightValues.step.value, weight="station_weight"
        )
        # sort out connections that span over multiple busbars
        # take only none helper nodes -> jump over to the next true node
        # save k: busbar v: a non helper node and none busbar
        station_switch_nodes = {
            k: v
            for k, v in path.items()
            if (k not in helper_nodes)
            and (len(v) >= min_len_path)
            and (v[-2] not in helper_nodes)
            and (length[k] == WeightValues.step.value)
        }
        # get only the shortest path -> if there is a helper branch at the end, it is filtered out here
        station_switch_nodes = {key: station_switch_nodes[key] for key in find_shortest_path_ids(station_switch_nodes)}
        busbar_grid_model_id = graph.nodes[busbar_id]["grid_model_id"]
        update_edge_dict.update(
            {(v[-2], v[-1]): {"direct_busbar_grid_model_id": busbar_grid_model_id} for v in station_switch_nodes.values()}
        )
        # set busbar_weight to graph
        update_dict = {(v[-2], v[-1]): {"busbar_weight": WeightValues.max_step.value} for v in station_switch_nodes.values()}
        nx.set_edge_attributes(graph, update_dict)
    update_edge_connection_info(graph=graph, update_edge_dict=update_edge_dict)


def set_bay_weights(graph: nx.Graph) -> dict[int, list[int]]:
    """Set the bay weights in the NetworkGraphData model for asset_nodes and branches.

    The bay weight is used to categorize paths in the network and assign a bay to an asset.
    Enables DGS export and switching into topology using a branch_id, by linking the branch_id to a switch.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    """
    update_node_dict, bay_update_dict = get_asset_bay_update_dict(graph=graph)
    set_asset_bay_edge_attr(graph=graph, asset_bay_update_dict=bay_update_dict)
    update_busbar_connection_info(graph=graph, update_node_dict=update_node_dict)


def get_asset_bay_update_dict(
    graph: nx.Graph,
) -> tuple[dict[int, dict[str, list[Union[str, int]]]], dict[str, dict[int, list[int]]]]:
    """Get the asset bay update dictionary for the nx.Graph (bases on NetworkGraphData model).

    The asset bay update dictionary is used to categorize the asset nodes in the network graph.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.

    Returns
    -------
    update_node_dict : dict
        A dictionary containing the update information for each node.
        keys: BusbarConnectionInfo attributes
        values: list of values to update
    update_bay_dict : dict
        A dictionary containing the update information for each bay.
        keys: bay_id
        values: dict
            key: busbar_id
            value: list of node_ids from the asset_node
    """
    # get update dict for node asset
    node_ids_with_node_assets = get_nodes_ids_with_a_connected_asset(graph=graph)
    connectable_node_assets_to_busbar, update_bay_dict = get_asset_bay_node_asset_dict(
        graph=graph, node_ids_with_node_assets=node_ids_with_node_assets
    )
    connectable_assets_dict = get_connectable_assets_update_dict(connectable_node_assets_to_busbar, graph=graph)
    connectable_busbars_dict = get_connectable_busbars_update_dict(
        shortest_path=connectable_node_assets_to_busbar, graph=graph
    )
    update_node_dict = {**connectable_assets_dict, **connectable_busbars_dict}
    return update_node_dict, update_bay_dict


def get_asset_bay_node_asset_dict(
    graph: nx.Graph,
    node_ids_with_node_assets: list[int],
) -> tuple[dict[int, list[str]], dict[str, dict[int, list[int]]]]:
    """Get the asset bay node asset dictionary for the nx.Graph (based on NetworkGraphData model).

    The asset bay node asset dictionary is used to categorize the asset nodes in the network graph.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    node_ids_with_node_assets : list[int]
        A list of node ids with a connected asset.
        This can be a node at the outer end of a Graph, e.g. a node where a line is incoming.
        Can also be a in the middle of a Graph, e.g. a BREAKER to be mapped.

    Returns
    -------
    connectable_asset_node_to_busbar : dict[int, list[int]]
        A dictionary containing the connectable busbars for each node.
        Key: node_id
        Value: list of connectable busbar ids
    asset_bay_update_dict : dict[str, dict[int, list[int]]]
        A dictionary containing the update information for each bay.
        keys: bay_id
        values: dict
            key: busbar_id
            value: list of node_ids from the asset_node
    """
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    connectable_node_assets_to_busbar = {}
    asset_bay_update_dict = {}
    for node_assets_id in node_ids_with_node_assets:
        grid_model_ids = graph.nodes[node_assets_id]["busbar_connection_info"].node_assets
        bay_id = " + ".join(grid_model_ids)
        shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
            graph=graph, asset_node=node_assets_id, busbars_helper_nodes=busbars_helper_nodes
        )
        asset_bay_update_dict[bay_id] = shortest_path_to_busbar_dict
        connectable_node_assets_to_busbar[node_assets_id] = [
            find_matching_node_in_list(busbar_node_id, busbars_helper_nodes, busbars)
            for busbar_node_id in shortest_path_to_busbar_dict.keys()
        ]
    return connectable_node_assets_to_busbar, asset_bay_update_dict


def get_connectable_busbars_update_dict(
    graph: nx.Graph, shortest_path: dict[int, list[int]]
) -> dict[int, dict[str, list[Union[str, int]]]]:
    """Get the node update dictionary for the BusbarConnectionInfo.

    Parameters
    ----------
    graph : nx.Graph
        The network graph from the NetworkGraphData.
    shortest_path : dict
        A dictionary containing the shortest path to a busbar for each busbar.
        key: busbars_helper_nodes
        value: list of node_ids from the asset_node the key (a busbars_helper_node)

    Returns
    -------
    update_nodes : dict
        A dictionary containing the update information for each node.
        keys: BusbarConnectionInfo attributes
        values: list of values to update
    """
    update_nodes = {}
    for node_id, connectable_list in shortest_path.items():
        grid_model_id_list = [graph.nodes[node]["grid_model_id"] for node in connectable_list]
        update_nodes[node_id] = {
            "connectable_busbars": grid_model_id_list,
            "connectable_busbars_node_ids": connectable_list,
        }
    return update_nodes


def get_connectable_assets_update_dict(
    connectable_node_assets_to_busbars: dict[int, list[int]], graph: nx.Graph
) -> dict[int, dict[str, list[Union[str, int]]] :]:
    """Get the connectable assets update dictionary for the BusbarConnectionInfo.

    Uses the reversed connectable_node_assets_to_busbars to update the connectable assets in the BusbarConnectionInfo.

    Parameters
    ----------
    connectable_node_assets_to_busbars : dict
        A dictionary containing the connectable busbars for each asset node.
        key: node_id
        value: list of connectable busbar ids
    graph : nx.Graph
        The network graph from the NetworkGraphData.

    Returns
    -------
    update_nodes : dict
        A dictionary containing the update information for each node.
        keys: BusbarConnectionInfo attributes
        values: list of values to update
    """
    connectable_node_assets = reverse_dict_list(connectable_node_assets_to_busbars)
    update_nodes = {}
    for node_id, connectable_list in connectable_node_assets.items():
        grid_model_id_lists = [graph.nodes[node]["busbar_connection_info"].node_assets for node in connectable_list]
        grid_model_id_list = list(flatten_list_of_mixed_entries(grid_model_id_lists))

        update_nodes[node_id] = {"connectable_assets": grid_model_id_list, "connectable_assets_node_ids": connectable_list}
    return update_nodes


def set_connectable_busbars(graph: nx.Graph) -> None:
    """Set the connectable busbars in BusbarConnectionInfo for each node in connectable_busbars.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    """
    connectable_busbars, _busbar_shortest_path = calculate_connectable_busbars(graph=graph)
    update_node_dict = get_connectable_busbars_update_dict(shortest_path=connectable_busbars, graph=graph)
    update_busbar_connection_info(graph=graph, update_node_dict=update_node_dict)


def calculate_connectable_busbars(graph: nx.Graph) -> tuple[dict[int, list[int]], dict[int, dict[int, list[int]]]]:
    """Calculate the connectable busbars in the NetworkGraphData model.

    The connectable busbars are the busbars that are reachable from a busbar.

    Parameters
    ----------
    graph : nx.Graph
        The network graph from the NetworkGraphData.

    Returns
    -------
    (busbar_interconnectable, busbar_shortest_path) : tuple[dict[int, list[int]], dict[int, dict[int, list[int]]]]
        the first dictionary contains the connectable busbars for each node.
        the second dictionary contains the shortest path to a busbar for each busbar.
    """
    busbars, _busbars_helper_nodes = get_busbar_true_nodes(graph=graph)
    busbar_interconnectable = {busbar_id: [] for busbar_id in busbars}
    busbar_shortest_path = {busbar_id: {} for busbar_id in busbars}
    weights_list = ["coupler_weight", "bay_weight"]
    for busbar in busbars:
        shortest_path_dict = shortest_paths_to_target_ids(
            graph=graph,
            target_node_ids=busbars,
            start_node_id=busbar,
            weight=multi_weight_function(weights_list),
            cutoff=WeightValues.max_coupler.value,
        )
        # sort out connections that span over multiple busbars
        # shortest path always contains the busbar itself + the target busbar -> 3 is not allowed
        shortest_path_dict = {k: v for k, v in shortest_path_dict.items() if sum(1 for x in v if x in busbars) == 2}
        busbar_interconnectable[busbar] = [busbar_id for busbar_id in shortest_path_dict.keys()]
        busbar_shortest_path[busbar] = {busbar_id: path for busbar_id, path in shortest_path_dict.items()}

    return busbar_interconnectable, busbar_shortest_path


def calculate_zero_impedance_connected(graph: nx.Graph, busbar_id: int) -> dict[int, list[str]]:
    """Calculate the zero impedance connected assets and busbars for a busbar_id.

    The zero impedance connections, are e.g. connections where open switches are filtered out.

    Parameters
    ----------
    graph : nx.Graph
        The network graph from the NetworkGraphData.
    busbar_id : int
        The node id of the busbar.

    Returns
    -------
    connected_assets_dict : dict[int, list[str]]
        A dictionary containing the connected assets for each node.
        key: node_id (a node asset id or a different busbar id than the input busbar_id)
        value: BusbarConnectionInfo.node_assets
    """
    # We now filter for connections that have a zero impedance connection
    # recap of the filter weights:
    # switch_open_weight: is set to "low" for all switches, except for open switches
    # -> any path that contains an open switch is cut off
    # busbar_weight: is set to "max_step" for all edges around a busbar
    # -> to reach an asset we need at least the "max_step" to not cut off
    # -> to reach another busbar we need at least 2*max_step to not cut off
    # station_weight: is set to "step" for all edges around a substation
    # -> to reach an asset we need up to 5*"step to not cut off
    # -> to reach another busbar we need at least 2*max_step to not cut off + the steps for each edge
    # The maximum step we expect is 2.4*max_step = 2*max_step (busbar) + 4*step (coupler)
    # The value 2.9 is choses to indicate that we search everything below 3 max_step,
    # which refers to the hops we need due to the busbar_weight
    # To conclude: with 2.9 all connections are found to the connected busbars and assets
    # respecting the switch open state with no busbar in between.
    step_multiplier = 2.9

    weights_list = ["switch_open_weight", "busbar_weight", "station_weight"]
    connectable_assets_node_ids = list(
        flatten_list_of_mixed_entries(graph.nodes[busbar_id]["busbar_connection_info"].connectable_assets_node_ids)
    )
    target_node_ids = (
        connectable_assets_node_ids + graph.nodes[busbar_id]["busbar_connection_info"].connectable_busbars_node_ids
    )
    path = shortest_paths_to_target_ids(
        graph=graph,
        target_node_ids=target_node_ids,
        start_node_id=busbar_id,
        weight=multi_weight_function(weights_list),
        cutoff=step_multiplier * WeightValues.max_step.value,
    )
    path_ids = list(path.keys())
    connected_assets_dict = {node_id: graph.nodes[node_id]["busbar_connection_info"].node_assets for node_id in path_ids}
    return connected_assets_dict


def set_zero_impedance_connected(graph: nx.Graph) -> None:
    """Set zero_impedance_connected in the graph model.

    A shortest path to the connected assets respecting the switch open state is calculated.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified
    """
    busbar_ids = get_node_list_by_attribute(graph=graph, attribute="node_type", value=["busbar"])
    update_node_dict = {}
    for busbar_id in busbar_ids:
        connected_assets_dict = calculate_zero_impedance_connected(graph=graph, busbar_id=busbar_id)
        connected_assets = []
        connected_asset_node_ids = []
        for node_id, connected_asset_list in connected_assets_dict.items():
            if connected_asset_list != []:
                connected_assets += connected_asset_list
                connected_asset_node_ids.append(node_id)

        connected_busbar_node_ids = [node_id for node_id in connected_assets_dict.keys() if node_id in busbar_ids]
        connected_busbars = [graph.nodes[node_id]["grid_model_id"] for node_id in connected_busbar_node_ids]
        update_node_dict[busbar_id] = {
            "zero_impedance_connected_assets": connected_assets,
            "zero_impedance_connected_assets_node_ids": connected_asset_node_ids,
            "zero_impedance_connected_busbars": connected_busbars,
            "zero_impedance_connected_busbars_node_ids": connected_busbar_node_ids,
        }
    update_busbar_connection_info(graph=graph, update_node_dict=update_node_dict)
