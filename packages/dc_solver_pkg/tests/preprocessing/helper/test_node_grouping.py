# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.node_grouping import (
    convert_boolean_mask_to_index_array,
    get_num_elements_per_node,
    group_by_node,
)


def test_group_branch_by_bus():
    branch_node_vector = np.array([0, 0, 1, 2, 3, 3], dtype=int)
    relevant_nodes = np.array([0, 2, 3], dtype=int)
    expected_grouping = [
        np.array([0, 1], dtype=int),
        np.array([3], dtype=int),
        np.array([4, 5], dtype=int),
    ]
    grouping = group_by_node(branch_node_vector, relevant_nodes)
    for result, expectation in zip(grouping, expected_grouping):
        assert np.array_equal(result, expectation)


def test_get_num_elements_per_node():
    elements_at_nodes = [np.array([0, 1], dtype=int), np.array([3], dtype=int)]
    expected_count = np.array([2, 1], dtype=int)
    count = get_num_elements_per_node(elements_at_nodes)
    assert np.array_equal(count, expected_count)


def test_convert_boolean_mask_to_index_array():
    mask = np.array([[0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=bool)
    expected = np.array([[1, 4, 5], [2, 3, -1], [4, 5, -1]], dtype=int)

    result = convert_boolean_mask_to_index_array(mask)
    assert np.array_equal(result, expected)

    mask = np.array([[0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=bool)
    expected = np.array([[1, 5], [2, 3], [4, 5]], dtype=int)

    result = convert_boolean_mask_to_index_array(mask)
    assert np.array_equal(result, expected)
