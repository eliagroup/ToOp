# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for the NetworkGraphData model or network_graph (nx.Graph)."""

from typing import Literal, TypeVar

import pandas as pd
from beartype.typing import get_args
from toop_engine_importer.network_graph.data_classes import DUPLICATED_EDGE_SUFFIX
from toop_engine_interfaces.asset_topology import SwitchableAsset

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def find_matching_node_in_list(node_id: int, search_list: list[int], return_list: list[int]) -> int:
    """Find the index of a node in one list and return the matching entry in a second list of the same length.

    Enables the changing of busbar and busbar_helper nodes in the NetworkGraphData model.
    For instance if information is stored in the busbar nodes, but the helper nodes are used to connect the busbar.

    Parameters
    ----------
    node_id: int
        The node id that should be searched
    search_list: list[int]
        The list of busbar node ids. where the available node_id can be found
    return_list: list[int]
        The list of busbar node ids where the corresponding node id is returned from

    Returns
    -------
    int
        The busbar node id.
    """
    assert len(search_list) == len(return_list), "The search and return list do not have the same length!"
    found_index = search_list.index(node_id)
    return return_list[found_index]


def reverse_dict_list(dict_of_lists: dict[K, list[V]]) -> dict[K, list[V]]:
    """Reverse a dictionary of lists.

    Parameters
    ----------
    dict_of_lists : dict
        A dictionary of lists.
        Key: int
        Value: list of int

    Returns
    -------
    reversed_dict: dict[K, list[V]]
        A reversed dictionary of lists.
        Key: every value from all lists from dict_of_lists
        Value: list, containing the keys of dict_of_lists

    Examples
    --------
    dict_of_lists = {1: [2, 3], 2: [3, 4]}
    reversed_dict = {2: [1], 3: [1, 2], 4: [2]}
    """
    reversed_dict = {}
    for key, value in dict_of_lists.items():
        for v in value:
            if v not in reversed_dict:
                reversed_dict[v] = []
            reversed_dict[v].append(key)
    return reversed_dict


def add_dict_list(
    dict_list1: dict[K, list[V]], dict_list2: dict[K, list[V]], mode: Literal["set", "append"] = "set"
) -> dict[K, list[V]]:
    """Add two dictionaries of lists.

    Parameters
    ----------
    dict_list1 : dict[K, list[V]]
        A dictionary of lists.
        Key: K
        Value: list of V
    dict_list2 : dict[K, list[V]]
        A dictionary of lists.
        Key: K
        Value: list of V
    mode : Literal["set", "append"], optional
        The mode to add the lists.
        "set": The values of the lists are unique.
        Note: The values of the lists are unique and do not keep order.
        "append": The values of the lists are not unique.
        Note: There might be duplicates in the list.

    Returns
    -------
    added_dict: dict[K, list[V]]
        A dictionary of lists.
        Key: K
        Value: list of V, containing the values of dict_list1 and dict_list2

    Examples
    --------
    dict1 ={0:[1,2,3], 1:[4,5,6], 2:[7,8,9]}
    dict2 = {0:[1,33,34], 4:[22,23,24], 5:[12,11,13]}
    added_dict = {0: [1, 2, 3, 34, 33], 1: [4, 5, 6], 2: [8, 9, 7], 4: [24, 22, 23], 5: [11, 12, 13]}
    """
    added_dict = {}
    for key in set(list(dict_list1.keys()) + list(dict_list2.keys())):
        res_list = list(dict_list1.get(key, []) + dict_list2.get(key, []))
        if mode == "set":
            added_dict[key] = list(set(res_list))
        elif mode == "append":
            added_dict[key] = res_list
        else:
            raise ValueError("The mode is not set or append.")
    return added_dict


def find_longest_path_ids(path_dict: dict[int, list[int]]) -> list[int]:
    """Find the longest path node_id in a list of paths.

    The path dict contains the path of nodes from a source node to a target node.
    This functions returns key (node_id) all longest paths.

    Parameters
    ----------
    path_dict : dict[int, list[int]]
        Dictionary of shortest path lengths keyed by target.
        key: int, target node_id
        value: list of paths from source node to target node.
        e.g. from nx.single_source_dijkstra_path

    Returns
    -------
    longest_path : list[int]
        List of node_id at the end of the longest path.

    Examples
    --------
    # from nx.single_source_dijkstra_path
    path_dict = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]}
    find_longest_path_id(path_list) = [3]

    path_list = {3: [1, 2, 3], 2: [1, 2], 4: [1, 2, 3, 4], 5: [1, 5], 6: [1, 2, 6]}
    find_longest_path_id(path_list) = [4, 5, 6]


    """
    result = []
    path_list = path_dict.values()
    for path_key, path in path_dict.items():
        # Check if this path is NOT a subset of any other path
        if not any(set(path).issubset(set(other_path)) and path != other_path for other_path in path_list):
            result.append(path_key)  # Append the last element of the longest paths

    return result


def find_shortest_path_ids(path_dict: dict[int, list[int]]) -> list[int]:
    """Find the shortest path node_id in a list of paths.

    The path dict contains the path of nodes from a source node to a target node.
    This functions returns key (node_id) all shortest paths.

    Parameters
    ----------
    path_dict : dict[int, list[list[int]]]
        Dictionary of shortest path lengths keyed by target.
        key: int, target node_id
        value: list of paths from source node to target node.
        e.g. from nx.single_source_dijkstra

    Returns
    -------
    shortest_path : list[int]
        List of node_id at the end of the shortest path.

    Examples
    --------
    # from nx.single_source_dijkstra_path
    path_dict = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]}
    find_shortest_path_id(path_list) = [0]

    path_list = {3: [1, 2, 3], 2: [1, 2], 4: [1, 2, 3, 4], 5: [1, 5], 6: [1, 2, 6]}
    find_shortest_path_id(path_list) = [2, 5]
    """
    result = []
    for path_key, path in path_dict.items():
        # Check if this path is a subset of any other path
        if all([path[: len(other_path)] != other_path for other_path in path_dict.values() if other_path != path]):
            result.append(path_key)

    return result


def get_pair_tuples_from_list(input_list: list[T]) -> list[tuple[T, T]]:
    """Convert a list of elements to a list of paired tuples.

    Parameters
    ----------
    input_list : list[T]
        A list of elements.

    Returns
    -------
    list[tuple[T, T]]
        A list of paired tuples.

    Examples
    --------
    input_list = [1, 2, 3, 4]
    get_pair_tuples_from_list(input_list) = [(1, 2), (2, 3), (3, 4)]
    """
    return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]


def remove_path_multiple_busbars(path_dict: dict[int, list[int]], busbars: list[int]) -> dict[int, list[int]]:
    """Remove a path from a dictionary of paths if it contains multiple busbars.

    Parameters
    ----------
    path_dict : dict[int, list[int]]
        Dictionary of paths keyed by target.
        key: int, target node_id
        value: list of node_ids representing the path from source to target.
    busbars : list[int]
        List of busbar node ids to check against the paths.

    Returns
    -------
    dict[int, list[int]]
        The updated dictionary of paths with the specified path removed.
    """
    # If one busbar is in the path, it is not removed
    # If multiple busbars are in the path, it is removed
    updated_path_dict = {}
    for target, path in path_dict.items():
        busbar_count = sum(1 for node in path if node in busbars)
        if busbar_count <= 1:  # Keep paths with 0 or 1 busbar
            updated_path_dict[target] = path
    return updated_path_dict


def add_suffix_to_duplicated_grid_model_id(df: pd.DataFrame, column: str = "grid_model_id") -> None:
    """Add a suffix to duplicated grid_model_id.

    There might be an asset like a PST or line connected to two busbars within the same substation.
    This functions adds a suffix to the grid_model_id to make it unique.

    Parameters
    ----------
    df : pd.DataFrame
        The node assets DataFrame.
        Note: df is modified in place.
    column : str
        The column to modify. Default is "grid_model_id".
    """
    # Get suffixes from the Literal type
    suffixes = get_args(DUPLICATED_EDGE_SUFFIX)
    cond = df.duplicated(subset=[column], keep=False)
    duplicates = df[cond][column].unique()
    for grid_model_id in duplicates:
        to_be_modified = df[df[column] == grid_model_id]
        assert len(suffixes) == len(to_be_modified)
        for i, suffix in enumerate(suffixes):
            df.loc[to_be_modified.index[i], column] = f"{grid_model_id}{suffix}"


def remove_suffix_from_switchable_assets(switchable_asset: list[SwitchableAsset]) -> list[SwitchableAsset]:
    """Remove the grid model id suffix from a switchable asset.

    Parameters
    ----------
    switchable_asset : list[SwitchableAsset]
        The switchable asset with a suffix.
        Note: The switchable asset is modified in place.

    Returns
    -------
    str
        The switchable asset without the suffix.
    """
    for asset in switchable_asset:
        for suffix in get_args(DUPLICATED_EDGE_SUFFIX):
            if asset.grid_model_id.endswith(suffix):
                asset.grid_model_id = asset.grid_model_id.split(suffix)[0]
