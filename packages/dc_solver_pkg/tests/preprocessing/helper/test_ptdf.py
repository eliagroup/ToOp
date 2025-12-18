# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.ptdf import (
    compute_ptdf,
    get_connectivity_matrix,
    get_extended_nodal_injections,
    get_extended_ptdf,
    get_susceptance_matrices,
)


def test_get_susceptance_matrices() -> None:
    from_node = np.array([0, 0, 1], dtype=int)
    to_node = np.array([2, 1, 0], dtype=int)
    susceptances = np.array([1, 2, 3], dtype=float)
    node_node_susceptance, branch_node_susceptance = get_susceptance_matrices(from_node, to_node, susceptances, 3, 3)

    expected_branch_node_susceptance = np.array([[1, 0, -1], [2, -2, 0], [-3, 3, 0]])
    assert np.allclose(branch_node_susceptance.toarray(), expected_branch_node_susceptance)
    # manual connectivity_matrix.T * branch_node_susceptance
    expected_node_node_susceptance = np.array(
        [
            [1 * 1 + 2 * 1 + -3 * -1, 0 * 1 + -2 * 1 + 3 * -1, -1 * 1 + 0 * 1 + 0 * -1],
            [1 * 0 + 2 * -1 + -3 * 1, 0 * 0 + -2 * -1 + 3 * 1, -1 * 0 + 0 * -1 + 0 * 1],
            [1 * -1 + 2 * 0 + -3 * 0, 0 * -1 + -2 * 0 + 3 * 0, -1 * -1 + 0 * 0 + 0 * 0],
        ]
    )
    assert np.allclose(node_node_susceptance.toarray(), expected_node_node_susceptance)


def test_get_connectivity_matrix() -> None:
    from_node = np.array([0, 0, 3, 2, 2, 3, 4], dtype=int)
    to_node = np.array([1, 2, 1, 3, 4, 5, 5], dtype=int)
    expected_connectivity = np.array(
        [
            [1, -1, 0, 0, 0, 0],
            [1, 0, -1, 0, 0, 0],
            [0, -1, 0, 1, 0, 0],
            [0, 0, 1, -1, 0, 0],
            [0, 0, 1, 0, -1, 0],
            [0, 0, 0, 1, 0, -1],
            [0, 0, 0, 0, 1, -1],
        ]
    )

    connectivity_matrix = get_connectivity_matrix(from_node, to_node, 7, 6)
    assert np.allclose(connectivity_matrix.toarray(), expected_connectivity)


def test_compute_ptdf() -> None:
    # Example values from bsdf paper https://www.techrxiv.org/users/689474/articles/681212-bus-split-distribution-factors
    from_node = np.array([0, 0, 3, 2, 2, 3, 4], dtype=int)
    to_node = np.array([1, 2, 1, 3, 4, 5, 5], dtype=int)
    susceptances = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float)
    slack_bus = 2
    ptdf_expected = (
        np.array(
            [
                [8, -14, 0, -6, -2, -4],
                [22, 14, 0, 6, 2, 4],
                [-8, -16, 0, 6, 2, 4],
                [-6, -12, 0, -18, -6, -12],
                [-2, -4, 0, -6, -22, -14],
                [2, 4, 0, 6, -8, -16],
                [-2, -4, 0, -6, 8, -14],
            ],
            dtype=float,
        )
        * 1
        / 30
    )
    ptdf = compute_ptdf(from_node, to_node, susceptances, slack_bus)
    assert ptdf_expected.shape == ptdf.shape
    assert np.allclose(ptdf_expected, ptdf)


def test_get_extended_ptdf() -> None:
    ptdf = np.array([[1, 2], [2, 3]], dtype=float)
    relevant_node_mask = np.ones(2, dtype=bool)
    extended_ptdf = get_extended_ptdf(ptdf, relevant_node_mask)

    expected_result = np.array([[1, 2, 1, 2], [2, 3, 2, 3]], dtype=float)
    assert np.array_equal(extended_ptdf, expected_result)


def test_get_extended_nodal_injections() -> None:
    nodal_injection = np.array([[1, 2], [2, 3]], dtype=float)
    relevant_node_mask = np.ones(2, dtype=bool)
    extended_nodal_injections = get_extended_nodal_injections(nodal_injection, relevant_node_mask)

    expected_result = np.array([[1, 2, 0, 0], [2, 3, 0, 0]], dtype=float)
    assert np.array_equal(extended_nodal_injections, expected_result)


def test_get_extended_nodal_injections_with_none() -> None:
    nodal_injection = None
    relevant_node_mask = np.ones(2, dtype=bool)
    extended_nodal_injections = get_extended_nodal_injections(nodal_injection, relevant_node_mask)

    assert extended_nodal_injections is None
