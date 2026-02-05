# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.branch_topology import (
    get_branch_direction,
    zip_branch_lists,
)


def test_zip_branch_lists():
    from_branches_at_nodes = [np.array([0, 1], dtype=int), np.array([3], dtype=int)]
    to_branches_at_nodes = [np.array([2], dtype=int), np.array([4, 5], dtype=int)]
    expected_branches_at_nodes = [
        np.array([0, 1, 2], dtype=int),
        np.array([3, 4, 5], dtype=int),
    ]
    branches_at_nodes = zip_branch_lists(from_branches_at_nodes, to_branches_at_nodes)
    for result, expectation in zip(branches_at_nodes, expected_branches_at_nodes):
        assert np.array_equal(result, expectation)


def test_get_branch_direction():
    from_branches_at_nodes = [np.array([0, 1], dtype=int), np.array([3], dtype=int)]
    branches_at_nodes = branches_at_nodes = [
        np.array([0, 1, 2], dtype=int),
        np.array([3, 4, 5], dtype=int),
    ]
    expected_branch_direction = [
        np.array([True, True, False], dtype=bool),
        np.array([True, False, False], dtype=bool),
    ]

    branch_direction = get_branch_direction(branches_at_nodes, from_branches_at_nodes)
    for result, expectation in zip(branch_direction, expected_branch_direction):
        assert np.array_equal(result, expectation)
