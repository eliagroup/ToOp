# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.helpers.injection_topology import (
    compute_nodal_injection,
    get_mw_injections_at_nodes,
    identify_inactive_injections,
)
from toop_engine_dc_solver.preprocess.helpers.node_grouping import group_by_node


def test_compute_nodal_injection() -> None:
    gen_loads_mw = np.array([[-1, -2, -3, 0, 1, 2, 3, 4]], dtype=float)
    gen_loads_bus = np.array([0, 1, 3, 4, 2, 3, 4, 5], dtype=int)
    num_bus = 10

    expected_nodal_injections = np.array([-1, -2, 1, -3 + 2, 0 + 3, 4, 0, 0, 0, 0], dtype=float)

    nodal_injections = compute_nodal_injection(gen_loads_mw, gen_loads_bus, num_bus)
    assert np.allclose(expected_nodal_injections, nodal_injections)


def test_compute_nodal_injection_timesteps() -> None:
    gen_loads_mw = np.array([[1, 2, 3], [0, 3, 3]], dtype=float)
    gen_loads_bus = np.array([0, 1, 1], dtype=int)
    num_bus = 2
    expected_nodal_injections = np.array([[1, 5], [0, 6]], dtype=float)
    nodal_injections = compute_nodal_injection(gen_loads_mw, gen_loads_bus, num_bus)
    assert np.allclose(expected_nodal_injections, nodal_injections)


def test_get_mw_injections_at_nodes() -> None:
    injection_node = np.array([0, 0, 1, 1, 2, 3], dtype=int)
    mw_injection = np.array([[1, 2, 3, 4, 5, 6], [0, 1, 1, 0, 0, 0]], dtype=float)
    relevant_nodes = np.array([0, 1, 2], dtype=int)
    injection_idx_at_nodes = group_by_node(injection_node, relevant_nodes)
    mw_injections_at_node = get_mw_injections_at_nodes(injection_idx_at_nodes, mw_injection)
    expected_mw_injections = [
        np.array([[1, 2], [0, 1]], dtype=float),
        np.array([[3, 4], [1, 0]], dtype=float),
        np.array([[5], [0]], dtype=float),
    ]
    for result, expectation in zip(mw_injections_at_node, expected_mw_injections):
        assert np.array_equal(result, expectation)


def test_identify_inactive_injections() -> None:
    mw_injections = [np.array([[0, 1, 0, 2], [0, 0, 0, 3]], dtype=float)]
    active_injections = identify_inactive_injections(mw_injections)
    expected_result = [
        np.array(
            [
                False,  # All values in the first column are 0
                True,  # Some values in the second column are not zero
                False,  # All values in third are zero
                True,  # All values are non-zero
            ]
        )
    ]
    assert np.array_equal(active_injections[0], expected_result[0])
