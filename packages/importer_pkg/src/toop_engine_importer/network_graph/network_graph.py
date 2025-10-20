"""Helper functions for the NetworkGraphData model."""

from typing import Any, Iterable, Iterator, Literal, Optional, Union

import networkx as nx
from toop_engine_importer.network_graph.data_classes import (
    BRANCH_TYPES,
    BranchSchema,
    BusbarConnectionInfo,
    EdgeConnectionInfo,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SwitchSchema,
    WeightValues,
)
from toop_engine_importer.network_graph.network_graph_helper_functions import add_dict_list


def generate_graph(network_graph_data: NetworkGraphData) -> nx.Graph:
    """Generate a NetworkX graph from the NetworkGraphData model.

    Parameters
    ----------
    network_graph_data : NetworkGraphData
        The NetworkGraphData model.
        Needs to be filled e.g. using pandapower_network_to_graph or powsybl_station_to_graph

    Returns
    -------
    nx.Graph
        The NetworkX graph from the NetworkGraphData model.
    """
    graph = nx.Graph()

    # create nodes
    graph_creation_nodes_helper(nodes_df=network_graph_data.nodes, graph=graph)
    # add nodes_assets to the nodes
    if not network_graph_data.node_assets.empty:
        graph_creation_node_assets_helper(node_assets_df=network_graph_data.node_assets, graph=graph)
    # create edges
    if not network_graph_data.switches.empty:
        graph_creation_edge_helper(edge_df=network_graph_data.switches, graph=graph)
    if not network_graph_data.branches.empty:
        graph_creation_edge_helper(edge_df=network_graph_data.branches, graph=graph)
        update_node_dict = get_branch_node_asset_update_dict(branch_df=network_graph_data.branches)
        update_busbar_connection_info(graph=graph, update_node_dict=update_node_dict, method="append")
    if not network_graph_data.helper_branches.empty:
        # IMPORTANT: the helper_branches need to be added last
        # -> existing branches and nodes are being ignored
        graph_creation_edge_helper(edge_df=network_graph_data.helper_branches, graph=graph)
    return graph


def graph_creation_nodes_helper(nodes_df: NodeSchema, graph: nx.Graph) -> None:
    """Create nodes in the NetworkX graph from the node DataFrame.

    Adds BusbarConnectionInfo to the nodes

    Parameters
    ----------
    nodes_df : NodeSchema
        The DataFrame containing the nodes.
    graph : nx.Graph
        The NetworkX graph to which the nodes will be added.
        Note: The graph is modified in place.
    """
    for _index, row in nodes_df.iterrows():
        row_dict = row.to_dict()
        row_dict["busbar_connection_info"] = BusbarConnectionInfo()
        graph.add_node(_index, **row_dict)


def graph_creation_edge_helper(edge_df: Union[BranchSchema, SwitchSchema], graph: nx.Graph) -> None:
    """Create edges in the NetworkX graph from the edge DataFrame.

    Parameters
    ----------
    edge_df : Union[BranchSchema, SwitchSchema]
        The DataFrame containing the edges.
    graph : nx.Graph
        The NetworkX graph to which the edges will be added.
        Note: The graph is modified in place.

    Returns
    -------
    nx.Graph
        The NetworkX graph with the added edges.
    """
    for _index, row in edge_df.iterrows():
        row_dict = row.to_dict()
        row_dict.pop("from_node")
        row_dict.pop("to_node")
        row_dict["edge_connection_info"] = EdgeConnectionInfo()
        graph.add_edge(row["from_node"], row["to_node"], **row_dict)


def graph_creation_node_assets_helper(node_assets_df: NodeAssetSchema, graph: nx.Graph) -> None:
    """Create node assets_list from node_assets_df in the nodes_df.

    Parameters
    ----------
    node_assets_df : NodeAssetSchema
        The DataFrame containing the node_assets.
    graph : nx.Graph
        The NetworkX graph to which the nodes will be added.
        Note: The graph is modified in place
    """
    for node in node_assets_df["node"].unique():
        node_content = node_assets_df[node_assets_df["node"] == node]
        graph.nodes[node]["busbar_connection_info"].node_assets = node_content["grid_model_id"].to_list()
        graph.nodes[node]["busbar_connection_info"].node_assets_ids = node_content.index.to_list()


def get_branch_node_asset_update_dict(branch_df: BranchSchema) -> dict[int, dict[str, Any]]:
    """Get the update dict for the node assets from the branches.

    Gets an update for BusbarConnectionInfo.node_assets and
    BusbarConnectionInfo.node_assets_ids from branches nodes ("from" and "to").

    Parameters
    ----------
    branch_df : BranchSchema
        The DataFrame containing the branches.

    Returns
    -------
    update_node_dict : dict[int, dict[str, Any]]
        A dictionary containing the update information for each node.
        key: node_id
        value: dictionary with the update, with keys BusbarConnectionInfo
    """
    update_node_dict = {}
    for _index, row in branch_df.iterrows():
        grid_model_id = row["grid_model_id"]
        from_node = row["from_node"]
        to_node = row["to_node"]
        update = {"node_assets": [grid_model_id], "node_assets_ids": [row["node_tuple"]]}
        update_node_dict[from_node] = add_dict_list(
            dict_list1=update, dict_list2=update_node_dict.get(from_node, {}), mode="append"
        )
        update_node_dict[to_node] = add_dict_list(
            dict_list1=update, dict_list2=update_node_dict.get(to_node, {}), mode="append"
        )

    return update_node_dict


def shortest_paths_to_target_ids(
    graph: nx.Graph,
    target_node_ids: list[int],
    start_node_id: int,
    weight: Union[str, callable] = "station_weight",
    cutoff: int = WeightValues.high.value,
) -> dict[int, list[int]]:
    """Find the shortest paths from one busbar to a list of busbars in the NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    target_node_ids : list[int]
        The list of busbar node ids, to which the shortest path is calculated.
    start_node_id : int
        The node id from which the shortest path is calculated.
    weight : Union[str, callable], default: "station_weight"
        string or callable
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.
    cutoff : int, optional
        The cutoff value for the shortest path.

    Returns
    -------
    shortest_path_dict dict[int, list[int]]
        The shortest path to the busbar.
        key: target_node_ids
        value: list of node_ids from the start_node_id to the key (the target_node_id)
        Note: not all target_node_ids are in the dict, only the ones that are reachable from the start_node_id.
              Reachable means that the path is shorter than the cutoff value.
    """
    shortest_path_dict = nx.single_source_dijkstra_path(graph, source=start_node_id, weight=weight, cutoff=cutoff)
    shortest_path_dict = {k: v for k, v in shortest_path_dict.items() if k in target_node_ids}
    return shortest_path_dict


def flatten_list_of_mixed_entries(
    stacked_list: list[Iterable[str | int] | str | int],
) -> Iterator[str | int]:
    """Generate flattened entries from a list of iterables and non-iterables.

    Parameters
    ----------
    stacked_list: list[Iterable[str | int] | str | int]
        A list containing int, str, tuple, list, or set.

    Yields
    ------
    A Generator with the flattened entries
    """
    for item in stacked_list:
        if isinstance(item, (int, str)):
            yield item
        else:
            for subitem in item:
                yield subitem


# TODO: add test
def set_substation_id(
    graph: nx.Graph,
    network_graph_data: NetworkGraphData,
) -> None:
    """Set the substation id in the NetworkGraphData model.

    The substation id is set in the nodes DataFrame if the substaion_id is not already set.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
        Note: The graph is modified in place.
    network_graph_data : NetworkGraphData
        The NetworkGraphData model.
    """
    if all(network_graph_data.nodes["substation_id"].unique() == ""):
        found_list = []
        for node in network_graph_data.nodes.index:
            if node in found_list:
                continue
            substation = list((get_all_node_paths_of_a_station_from_a_node(graph=graph, node_id=node)).keys())

            found_list += substation
            network_graph_data.nodes.loc[substation, "substation_id"] = network_graph_data.nodes.loc[node, "grid_model_id"]

        substation_id_dict = network_graph_data.nodes[["substation_id"]].to_dict(orient="index")
        nx.set_node_attributes(graph, substation_id_dict)


def multi_weight_function(weight_list: list[str], weight_multiplier: Optional[dict[str, float]] = None) -> callable:
    """Create a multi weight function for the NetworkGraphData model.

    Parameters
    ----------
    weight_list : list[str]
        A list of weights used to find the shortest path.
    weight_multiplier : Optional[dict[str, float]], optional
        A dictionary containing the weights to be modified.
        The keys are the weights and the values are the multipliers.
        The multipliers are used to modify the retrieved weights in the weight_list.

    Returns
    -------
    multi_weight_function : function
        A function that returns the sum of the weights in the weight_list.
    """
    if weight_multiplier is None:
        weight_multiplier = {}

    for weight in weight_list:
        if weight not in weight_multiplier:
            weight_multiplier[weight] = 1.0

    # ruff: noqa: ARG001
    def multi_weight_function(from_id: int, to_id: int, data: dict[str, Any]) -> int:
        """Return the sum of the weights in the weight_list."""
        return sum(data.get(weight, 0) * weight_multiplier[weight] for weight in weight_list)

    return multi_weight_function


def get_busbar_true_nodes(graph: nx.Graph) -> tuple[list[int], list[int]]:
    """Get the busbar nodes and helper nodes in the NetworkGraphData model.

    The busbars can be modeled as nodes or helper nodes in the NetworkGraphData model.
    This function returns the busbars as nodes and the busbars as helper nodes.
    If there is a helper node for a busbar, the busbar node has only one neighbor
    and the neighbor has all edges connected to it.
    A helper node does not store any information and is only used to connect the busbar to the network.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph based on NetworkGraphData model.

    Returns
    -------
    (busbars, busbars_helper_nodes) : tuple[list[int], list[int]]
        the first list contains busbars, where they are modeled as nodes
        the second list contains busbars, where they are modeled as helper nodes
        The returned lists are of the same length
    """
    busbars = get_node_list_by_attribute(graph=graph, attribute="node_type", value=["busbar"])
    busbars_helper_nodes = []
    for node_id in busbars:
        neighbors = graph[node_id]
        if len(neighbors) == 1:
            for neighbor in neighbors.keys():
                if graph.nodes[neighbor]["grid_model_id"] == "":
                    busbars_helper_nodes.append(neighbor)
    if len(busbars_helper_nodes) == len(busbars):
        return (busbars, busbars_helper_nodes)
    if len(busbars_helper_nodes) == 0:
        return (busbars, busbars)
    raise ValueError("The busbar helper nodes are not correctly set.")


def get_busbar_connection_info(
    graph: nx.Graph, busbar_grid_model_id: Optional[list[str]] = None
) -> dict[str, BusbarConnectionInfo]:
    """Return the BusbarConnectionInfo of the busbars in the NetworkGraphData model.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    busbar_grid_model_id : Optional[list[str]]
        The grid_model_ids of the busbars that should be returned.
        if None, all BusbarConnectionInfo is returned.

    Returns
    -------
    busbar_connection_info : dict
        A dictionary containing the BusbarConnectionInfo for each busbar.
            Key: grid_model_id
            Value: BusbarConnectionInfo
        Note: always returns a dict. Dict is empty if no busbar is found.

    """
    if busbar_grid_model_id is not None:
        busbars_node_id = get_node_list_by_attribute(graph=graph, attribute="grid_model_id", value=[busbar_grid_model_id])
    else:
        busbars_node_id = get_node_list_by_attribute(graph=graph, attribute="node_type", value=["busbar"])

    busbar_connection_info = {
        graph.nodes[node_id]["grid_model_id"]: graph.nodes[node_id]["busbar_connection_info"] for node_id in busbars_node_id
    }
    return busbar_connection_info


def get_edge_connection_info(
    graph: nx.Graph, edge_grid_model_ids: Optional[list[str]] = None
) -> dict[str, BusbarConnectionInfo]:
    """Return the EdgeConnectionInfo of the edges in the NetworkGraphData model.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    edge_grid_model_ids : Optional[list[str]]
        The grid_model_ids of the edges that should be returned, e.g. switch or branch.
        if None, all EdgeConnectionInfo is returned.

    Returns
    -------
    edge_connection_info : dict
        A dictionary containing the EdgeConnectionInfo for each busbar.
            Key: grid_model_id
            Value: EdgeConnectionInfo
        Note: always returns a dict. Dict is empty if no edge is found.

    """
    if edge_grid_model_ids is not None:
        edge_ids_to_find = [
            (node_id1, node_id2)
            for node_id1, node_id2, grid_model_id in graph.edges(data="grid_model_id")
            if grid_model_id in edge_grid_model_ids
        ]

    else:
        edge_ids_to_find = graph.edges

    edge_connection_info = {
        graph.edges[edge_id]["grid_model_id"]: graph.edges[edge_id]["edge_connection_info"] for edge_id in edge_ids_to_find
    }
    return edge_connection_info


def get_node_list_by_attribute(graph: nx.Graph, attribute: str, value: list[Any]) -> list[int]:
    """Get a list of nodes by an attribute value.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    attribute : str
        The attribute name.
    value : list[Any]
        The attribute value list to find.

    Returns
    -------
    list[int]
        A list of node ids.
    """
    return [node_id for node_id, node_attr in graph.nodes(data=True) if node_attr.get(attribute, []) in value]


def get_helper_node_ids(graph: nx.Graph) -> list[int]:
    """Get the helper node ids in the nx.Graph, based on the NetworkGraphData model.

    The helper nodes are used to connect the busbars to the network.
    They are not part of the network and do not contain any information.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.

    Returns
    -------
    helper_node_ids : list[int]
        A list of helper node ids.
    """
    helper_node_ids = get_node_list_by_attribute(graph=graph, attribute="helper_node", value=[True])
    return helper_node_ids


def get_nodes_ids_with_a_connected_asset(graph: nx.Graph) -> list[int]:
    """Get the asset nodes is in the nx.Graph, where there is an asset.

    The asset nodes are nodes that contain assets node_assets like a generator or a load.
    They also contain the information about branches. A branch has two asset nodes.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.

    Returns
    -------
    asset_node_ids : list[int]
        A list of asset node ids.
    """
    asset_node_ids = [
        node_id
        for node_id, node_connection in graph.nodes(data="busbar_connection_info")
        if node_connection.node_assets_ids != []
    ]
    return asset_node_ids


def get_edge_list_by_attribute(graph: nx.Graph, attribute: str, value: list[Any]) -> list[tuple[int, int]]:
    """Get a list of edge ids by an attribute value.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    attribute : str
        The attribute name.
    value : list[Any]
        The attribute value list to find.

    Returns
    -------
    list[tuple[int, int]]
        A list of edge id tuple.
        An edge id tuple contains two node ids. The Direction does not matter.
    """
    return [
        edge_attr["node_tuple"] for _id1, id2, edge_attr in graph.edges(data=True) if edge_attr.get(attribute, []) in value
    ]


def get_busbar_connection_info_attribute(
    graph: nx.Graph, attribute: str, node_type: Optional[Literal["busbar", "node"]] = None
) -> dict[int, list[int]]:
    """Get an attribute from the BusbarConnectionInfo from graph.nodes.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    attribute : str
        The attribute name from the BusbarConnectionInfo, e.g. "connectable_busbars"
    node_type : Optional[Literal["busbar", "node"]], optional
        The node_type of the busbar, e.g. "busbar" or "node".
        If None, all graph.nodes are returned.

    Returns
    -------
    connectable_busbars : dict
        A dictionary containing the connectable busbars.
        Key: node_id
        Value: list of node_ids of connectable busbars.
    """
    if node_type is not None:
        node_list = get_node_list_by_attribute(graph, attribute="node_type", value=[node_type])
    else:
        node_list = list(graph.nodes)

    connectable_busbars_dict = {
        node_id: value.__dict__[attribute]
        for node_id, value in graph.nodes(data="busbar_connection_info")
        if node_id in node_list
    }
    return connectable_busbars_dict


def get_branch_ids_by_type_list(graph: nx.Graph, branch_types: Optional[list[str]] = None) -> list[int]:
    """Get the branch ids by the branch types.

    Returns the branch ids of the branch types in the NetworkX graph.
    Only returns BRANCH_TYPES.
    Ignores other Types like SWITCH_TYPES.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    branch_types : Optional[list[str]], optional
        The list of branch types to find.
        If None, all BRANCH_TYPES are returned. Note: this excludes SWITCH_TYPES.

    Returns
    -------
    branch_ids : list[int]
        A list of branch ids.
    """
    if branch_types is None:
        branch_types = [
            branch_type for branch_type_model in BRANCH_TYPES.__args__ for branch_type in branch_type_model.__args__
        ]
    branch_ids = get_edge_list_by_attribute(graph=graph, attribute="asset_type", value=branch_types)
    return branch_ids


def update_busbar_connection_info(
    graph: nx.Graph,
    update_node_dict: dict[int, dict[str, Any]],
    method: Literal["set", "append"] = "set",
) -> None:
    """Update the BusbarConnectionInfo in the graph model.

    The BusbarConnectionInfo is initialized empty for each node in the network graph.
    This function updates the BusbarConnectionInfo with the given update_node_dict.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    update_node_dict : dict[int, dict[str, Any]]
        A dictionary containing the update information for each node.
        key: node_id
        value: dictionary with the update, with keys BusbarConnectionInfo attributes
    method : Literal["set", "append"], optional
        The mode to update the lists.
        "set": The keys and values will replace the existing values.
        "append": If the value exists and is a list, the values will be appended to the existing values.
        Raises an error if the value is not a list.

    Raises
    ------
    ValueError
        If the method is not set or append.
    """
    for asset_node, update_dict in update_node_dict.items():
        busbar_connection_info = graph.nodes[asset_node]["busbar_connection_info"]
        validate_update_dict_for_connection_info(busbar_connection_info, update_dict)
        if method == "set":
            graph.nodes[asset_node]["busbar_connection_info"] = graph.nodes[asset_node]["busbar_connection_info"].model_copy(
                update=update_dict
            )
        elif method == "append":
            busbar_connection_info = append_connection_info(connection_info=busbar_connection_info, update_dict=update_dict)
            graph.nodes[asset_node]["busbar_connection_info"] = busbar_connection_info
        else:
            raise ValueError("The method is not set or append.")
        BusbarConnectionInfo().model_validate(graph.nodes[asset_node]["busbar_connection_info"])


def validate_update_dict_for_connection_info(
    connection_info: Union[BusbarConnectionInfo, EdgeConnectionInfo], update_dict: dict[int, dict[str, Any]]
) -> bool:
    """Test if the keys in the update_dict are in the connection_info.

    update_dict: dict[int, dict[str, Any]]
    A dictionary containing the update information.
    key: node_id
    value: dictionary with the update, with keys BusbarConnectionInfo or EdgeConnectionInfo attributes

    Raises
    ------
    ValueError
        If the update_dict contains unknown keys for the connection

    """
    if not all([arg in connection_info.__annotations__.keys() for arg in update_dict.keys()]):
        raise ValueError(
            f"Update dictionary contains unknown keys for {type(connection_info)}."
            f" Update keys: {update_dict.keys()}. Allowed keys: {connection_info.__annotations__.keys()}"
        )
    return True


def append_connection_info(
    connection_info: Union[BusbarConnectionInfo, EdgeConnectionInfo], update_dict: dict[int, dict[str, Any]]
) -> Union[BusbarConnectionInfo, EdgeConnectionInfo]:
    """Append the ConnectionInfo in the graph model.

    Additional information is added to te ConnectionInfo.
    Note: this only works for lists.

    Parameters
    ----------
    connection_info : Union[BusbarConnectionInfo, EdgeConnectionInfo]
        The ConnectionInfo model.
    update_dict : dict[int, dict[str, Any]]
        A dictionary containing the update information.
        key: node_id
        value: dictionary with the update, with keys ConnectionInfo attributes
        Note: The values need to be lists.

    Returns
    -------
    connection_info : Union[BusbarConnectionInfo, EdgeConnectionInfo]
        The updated ConnectionInfo model.
        The values are appended to the existing values.

    Raises
    ------
    ValueError
        If the dict or connection info is not is not a list for a given dict key.
    """
    for key, value in update_dict.items():
        if isinstance(value, list) and isinstance(connection_info.__dict__[key], list):
            connection_info.__dict__[key] += value
        else:
            raise ValueError(f"The value for key: {key} is not a list.")
    return connection_info


def update_edge_connection_info(
    graph: nx.Graph,
    update_edge_dict: dict,
    method: Literal["set", "append"] = "set",
) -> None:
    """Update the ConnectionInfo in the graph model.

    The ConnectionInfo is initialized empty for each edge in the network graph.
    This function updates the ConnectionInfo with the given update_edge_dict.

    Parameters
    ----------
    graph : nx.Graph
        The network graph.
        Note: The graph is modified in place.
    update_edge_dict : dict
        A dictionary containing the update information for each edge.
        key: tuple[node1,node2]
        value: dictionary with the update, with keys ConnectionInfo attributes
    method : Literal["set", "append"], optional
        The mode to update the lists.
        "set": The keys and values will replace the existing values.
        "append": If the value exists and is a list, the values will be appended to the existing values.
        Raises an error if the value is not a list.
    """
    for edge, update_dict in update_edge_dict.items():
        edge_connection_info = graph.edges[edge]["edge_connection_info"]
        validate_update_dict_for_connection_info(edge_connection_info, update_dict)
        if method == "set":
            graph.edges[edge]["edge_connection_info"] = graph.edges[edge]["edge_connection_info"].model_copy(
                update=update_dict
            )
        elif method == "append":
            edge_connection_info = append_connection_info(connection_info=edge_connection_info, update_dict=update_dict)
            graph.edges[edge]["edge_connection_info"] = edge_connection_info
        else:
            raise ValueError("The method is not set or append.")
        EdgeConnectionInfo().model_validate(graph.edges[edge]["edge_connection_info"])


def get_all_node_paths_of_a_station_from_a_node(
    graph: nx.Graph, node_id: int, weights_list: Optional[list[str]] = None
) -> dict[int, list[int]]:
    """Get all nodes of a station starting from a single node in the NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph.
    node_id : int
        The node id from which the station is searched.
    weights_list : Optional[list[str]]
        The weights used to find the shortest path.
        If None, the default is ["station_weight"].

    Returns
    -------
    station_nodes_list : dict[int, list[int]]
        A dictionary containing the station nodes.
        key: node_id
        value: the shortest path list of node_ids
    """
    if weights_list is None:
        weights_list = ["station_weight"]

    cutoff = WeightValues.high.value
    # the cutoff behaves like <=
    # the outer edges of a substation are set to WeightValues.high.value
    # if a node is selected that is directly connected to the outer edges, another substation would be included
    # the cutoff is set to WeightValues.high.value - 1 to avoid this issue
    cutoff -= 1
    station_nodes = nx.single_source_dijkstra_path(
        graph, source=node_id, weight=multi_weight_function(weights_list), cutoff=cutoff
    )
    return station_nodes
