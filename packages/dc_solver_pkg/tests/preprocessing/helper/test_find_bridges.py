# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_bridges,
    find_n_minus_2_safe_branches,
)


def test_find_bridges_2_nodes() -> None:
    # Make sure the whole grid is connected
    from_node = np.array([0, 0], dtype=int)
    to_node = np.array([1, 1], dtype=int)
    bridge_mask = find_bridges(from_node, to_node, 2, 2)
    expected_bridge_mask = np.array(
        [
            False,
            False,  # Since there are two branches connecting 0 and 1
        ],
        dtype=bool,
    )

    assert np.array_equal(bridge_mask, expected_bridge_mask)


def test_find_bridges_2_nodes_directed() -> None:
    # Make sure the whole grid is connected
    from_node = np.array([0, 1], dtype=int)
    to_node = np.array([1, 0], dtype=int)
    bridge_mask = find_bridges(from_node, to_node, 2, 2)
    expected_bridge_mask = np.array(
        [
            False,
            False,  # Since there are two branches connecting 0 and 1
        ],
        dtype=bool,
    )

    assert np.array_equal(bridge_mask, expected_bridge_mask)


def test_find_bridges_6_branches() -> None:
    # Make sure the whole grid is connected
    from_node = np.array([0, 1, 1, 2, 4, 1], dtype=int)
    to_node = np.array([1, 0, 4, 3, 1, 2], dtype=int)
    bridge_mask = find_bridges(from_node, to_node, 6, 5)
    expected_bridge_mask = np.array(
        [
            False,
            False,  # Since there are two branches connecting 0 and 1
            False,  # Since node 1 and 4 are connected twice
            True,  # Since node 2 and 3 are only connected here and node 3 is not connected in any other way
            False,  # Since node 1 and 4 are connected twice
            True,  # Since this is the only connection to the nodes 1 and 2
        ],
        dtype=bool,
    )

    assert np.array_equal(bridge_mask, expected_bridge_mask)


def test_find_n_minus_2_safe_branches() -> None:
    # Create a graph with 4 nodes in a square and one branch in the middle
    # The middle branch should be the only one that is n-2 safe

    from_node = np.array([0, 1, 2, 3, 1], dtype=int)
    to_node = np.array([1, 2, 3, 0, 3], dtype=int)
    n_minus_2_safe = find_n_minus_2_safe_branches(from_node, to_node, 5, 4)
    expected_n_minus_2_safe = np.array(
        [
            False,
            False,
            False,
            False,
            True,
        ],
        dtype=bool,
    )

    assert np.array_equal(n_minus_2_safe, expected_n_minus_2_safe)
