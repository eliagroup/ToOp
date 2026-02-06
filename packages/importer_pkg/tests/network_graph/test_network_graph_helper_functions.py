# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_importer.network_graph.data_classes import DUPLICATED_EDGE_SUFFIX
from toop_engine_importer.network_graph.network_graph_helper_functions import (
    add_dict_list,
    find_longest_path_ids,
    find_matching_node_in_list,
    find_shortest_path_ids,
    get_pair_tuples_from_list,
    remove_path_multiple_busbars,
    remove_suffix_from_switchable_assets,
    reverse_dict_list,
)


def test_find_busbar_in_list():
    busbar_list1 = [1, 2, 3]
    busbar_list2 = [4, 5, 6]

    assert find_matching_node_in_list(1, busbar_list1, busbar_list2) == 4
    assert find_matching_node_in_list(2, busbar_list1, busbar_list2) == 5
    assert find_matching_node_in_list(3, busbar_list1, busbar_list2) == 6

    with pytest.raises(ValueError):
        find_matching_node_in_list(7, busbar_list1, busbar_list2)


def test_reverse_dict_list():
    dict_of_lists = {1: [2, 3], 2: [3, 4]}
    expected_reversed_dict = {2: [1], 3: [1, 2], 4: [2]}
    assert reverse_dict_list(dict_of_lists) == expected_reversed_dict

    dict_of_lists = {1: [2, 3, 4], 2: [4, 5], 3: [5, 6]}
    expected_reversed_dict = {2: [1], 3: [1], 4: [1, 2], 5: [2, 3], 6: [3]}
    assert reverse_dict_list(dict_of_lists) == expected_reversed_dict

    dict_of_lists = {}
    expected_reversed_dict = {}
    assert reverse_dict_list(dict_of_lists) == expected_reversed_dict

    dict_of_lists = {1: [], 2: [3], 3: [1, 2]}
    expected_reversed_dict = {3: [2], 1: [3], 2: [3]}
    assert reverse_dict_list(dict_of_lists) == expected_reversed_dict

    dict_of_lists = {1: [1, 2], 2: [2, 3], 3: [3, 1]}
    expected_reversed_dict = {1: [1, 3], 2: [1, 2], 3: [2, 3]}
    assert reverse_dict_list(dict_of_lists) == expected_reversed_dict


def test_add_dict_list():
    dict_list1 = {1: [2, 3], 2: [3, 4]}
    dict_list2 = {1: [3, 5], 3: [6, 7]}
    expected_result_set = {1: [2, 3, 5], 2: [3, 4], 3: [6, 7]}
    expected_result_append = {1: [2, 3, 3, 5], 2: [3, 4], 3: [6, 7]}
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    dict_list1 = {1: [2, 3, 4], 2: [4, 5], 3: [5, 6]}
    dict_list2 = {1: [4, 5], 2: [4, 5], 3: [6, 7]}
    expected_result_set = {1: [2, 3, 4, 5], 2: [4, 5], 3: [5, 6, 7]}
    expected_result_append = {1: [2, 3, 4, 4, 5], 2: [4, 5, 4, 5], 3: [5, 6, 6, 7]}
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    dict_list1 = {}
    dict_list2 = {1: [2, 3], 2: [3, 4]}
    expected_result_set = dict_list2
    expected_result_append = dict_list2
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    dict_list1 = {1: [2, 3], 2: [3, 4]}
    dict_list2 = {}
    expected_result_set = dict_list1
    expected_result_append = dict_list1
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    dict_list1 = {1: [1, 2], 2: [2, 3], 3: [3, 1]}
    dict_list2 = {1: [3, 4], 2: [4, 5], 3: [5, 6]}
    expected_result_set = {1: [1, 2, 3, 4], 2: [2, 3, 4, 5], 3: [1, 3, 5, 6]}
    expected_result_append = {1: [1, 2, 3, 4], 2: [2, 3, 4, 5], 3: [3, 1, 5, 6]}
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    dict_list1 = {1: [1, 2], 2: [2, 3], 3: [3, 1]}
    dict_list2 = {4: [3, 4], 5: [4, 5], 6: [5, 6]}
    expected_result_set = {1: [1, 2], 2: [2, 3], 3: [1, 3], 4: [3, 4], 5: [4, 5], 6: [5, 6]}
    expected_result_append = {1: [1, 2], 2: [2, 3], 3: [3, 1], 4: [3, 4], 5: [4, 5], 6: [5, 6]}
    assert add_dict_list(dict_list1, dict_list2, mode="set") == expected_result_set
    assert add_dict_list(dict_list1, dict_list2, mode="append") == expected_result_append

    with pytest.raises(ValueError):
        add_dict_list(dict_list1, dict_list2, mode="invalid_mode")


def test_find_longest_path():
    path_dict = {3: [1, 2, 3], 2: [1, 2], 4: [1, 2, 3, 4], 5: [1, 5], 6: [1, 2, 6]}
    expected_result = [4, 5, 6]
    assert find_longest_path_ids(path_dict) == expected_result

    path_dict = {3: [1, 2, 3], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4, 5], 6: [1, 2, 3, 4, 5, 6]}
    expected_result = [6]
    assert find_longest_path_ids(path_dict) == expected_result

    path_dict = {2: [1, 2], 3: [1, 3], 4: [1, 4], 5: [1, 5]}
    expected_result = [2, 3, 4, 5]
    assert find_longest_path_ids(path_dict) == expected_result

    path_dict = {}
    expected_result = []
    assert find_longest_path_ids(path_dict) == expected_result


def test_get_pair_tuples_from_list():
    input_list = [1, 2, 3, 4]
    expected_result = [(1, 2), (2, 3), (3, 4)]
    assert get_pair_tuples_from_list(input_list) == expected_result

    input_list = ["a", "b", "c"]
    expected_result = [("a", "b"), ("b", "c")]
    assert get_pair_tuples_from_list(input_list) == expected_result

    input_list = [1]
    expected_result = []
    assert get_pair_tuples_from_list(input_list) == expected_result

    input_list = []
    expected_result = []
    assert get_pair_tuples_from_list(input_list) == expected_result

    input_list = [1, 2, 2, 3]
    expected_result = [(1, 2), (2, 2), (2, 3)]
    assert get_pair_tuples_from_list(input_list) == expected_result


def test_remove_path_multiple_busbars_basic():
    path_dict = {
        1: [1, 2, 3],
        2: [2, 5, 6],
        3: [3, 7, 8],
        4: [4, 5, 6, 2],
        5: [5, 5, 5, 5, 5],
        6: [5, 2],
        7: [7, 8, 9, 5, 20, 55, 22],
    }
    busbars = [2, 5]
    # Only keep paths with at most one busbar in the path
    expected = {1: [1, 2, 3], 3: [3, 7, 8], 7: [7, 8, 9, 5, 20, 55, 22]}
    assert remove_path_multiple_busbars(path_dict, busbars) == expected

    busbars = [2, 5, 1, 3, 4, 8]
    # All paths have more than one busbar
    expected = {}
    assert remove_path_multiple_busbars(path_dict, busbars) == expected

    busbars = []
    # No busbars, all paths should remain
    assert remove_path_multiple_busbars(path_dict, busbars) == path_dict

    path_dict = {}
    busbars = [1, 2]
    expected = {}
    assert remove_path_multiple_busbars(path_dict, busbars) == expected


class DummySwitchableAsset:
    def __init__(self, grid_model_id):
        self.grid_model_id = grid_model_id


def test_remove_suffix_from_switchable_assets_removes_suffix():
    suffixes = list(DUPLICATED_EDGE_SUFFIX.__args__)
    assets = [
        DummySwitchableAsset(f"asset1{suffixes[0]}"),
        DummySwitchableAsset(f"asset2{suffixes[1]}"),
        DummySwitchableAsset("asset3"),
    ]
    remove_suffix_from_switchable_assets(assets)
    assert assets[0].grid_model_id == "asset1"
    assert assets[1].grid_model_id == "asset2"
    assert assets[2].grid_model_id == "asset3"

    # test empty list
    assets = []
    remove_suffix_from_switchable_assets(assets)  # Should not raise
    assert assets == []


def test_find_shortest_path_ids_basic():
    # Example from docstring: only [0] is not a prefix of any other
    path_dict = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]}
    assert find_shortest_path_ids(path_dict) == [0]

    # Example from docstring: [2] and [5] are not prefixes of any other
    path_dict = {3: [1, 2, 3], 2: [1, 2], 4: [1, 2, 3, 4], 5: [1, 5], 6: [1, 2, 6]}
    result = find_shortest_path_ids(path_dict)
    assert set(result) == {2, 5}

    # All paths are unique and not prefixes of each other
    path_dict = {1: [1], 2: [2], 3: [3]}
    assert set(find_shortest_path_ids(path_dict)) == {1, 2, 3}

    # Only one path
    path_dict = {10: [1, 2, 3]}
    assert find_shortest_path_ids(path_dict) == [10]

    # Empty dict
    assert find_shortest_path_ids({}) == []

    # Multiple shortest paths
    path_dict = {1: [1], 2: [2], 3: [1, 2], 4: [2, 3]}
    assert set(find_shortest_path_ids(path_dict)) == {1, 2}

    # Paths with shared prefixes
    path_dict = {1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 4]}
    assert set(find_shortest_path_ids(path_dict)) == {1, 4}

    path_dict = {
        58: [31, 12, 59, 58],
        68: [31, 12, 69, 68],
        74: [31, 12, 75, 74],
        92: [31, 12, 93, 92],
        98: [31, 12, 99, 98],
        132: [31, 12, 133, 132],
        24: [31, 12, 93, 92, 24],
        67: [31, 12, 93, 92, 24, 67],
    }
    expected = [58, 68, 74, 92, 98, 132]
    assert set(find_shortest_path_ids(path_dict)) == set(expected)
