# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from toop_engine_dc_solver.preprocess.helpers.reduce_node_dimension import (
    get_significant_nodes,
    get_updated_indices_due_to_filtering,
    reduce_ptdf_and_nodal_injections,
    update_ids_linking_to_nodes,
)


def test_get_significant_nodes():
    # Grid:  0 -- 1 -- 2 -- 3 -- 4 -- 5 -- 6 -- 7 -- 8 == 9

    # The first entry is relevant, so the bus 0 should be marked as True
    relevant_node_mask = np.array([True, False, False, False, False, False, False, False, False, False])
    # The 6th bus is outaged, so bus 5 should be marked as True
    multi_outage_node_mask = np.array(
        [
            [False, False, False, False, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
        ]
    )
    from_nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    to_nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 8])
    relevant_branches = np.array([0])
    # The first branch is relevant, as well as the same bus
    # We want to include 2 buses away from all relevant busses, so bus 1 and 2 should be included

    # the slack is also significant, so result 9 -> True
    slack = 9
    significant_nodes = get_significant_nodes(
        relevant_node_mask, multi_outage_node_mask, relevant_branches, from_nodes, to_nodes, slack
    )
    expected_significant_node_mask = np.array([True, True, True, False, False, True, False, False, False, True])
    assert np.all(significant_nodes == expected_significant_node_mask)


def test_get_updated_indices_due_to_filtering():
    ids_to_keep = np.array([0, 2, 4, 6, 8])
    ids_to_filter = np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 0])
    fill_value = 10
    # The indices 0, 2, 4, 6, 8 should be kept, so the new indices should be 0, 1, 2, 3, 4
    expected_filtered_ids = np.array([0, 0, 1, fill_value, 2, fill_value, 3, fill_value, 4, 0])
    filtered_ids = get_updated_indices_due_to_filtering(ids_to_keep, ids_to_filter, fill_value)
    assert np.all(filtered_ids == expected_filtered_ids)


def test_reduce_ptdf_and_nodal_injections():
    ptdf = np.array([[10, 10, 10, 10], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=float)
    nodal_injection = np.array([[2, 2, 2, 2]], dtype=float)
    expected_loadflow = ptdf @ nodal_injection[0]
    significant_nodes = np.array([True, False, False, False])
    reduced_ptdf, reduced_injections = reduce_ptdf_and_nodal_injections(ptdf, nodal_injection, significant_nodes)

    reduced_loadflow = reduced_ptdf @ reduced_injections[0]
    assert np.all(reduced_loadflow == expected_loadflow)

    assert reduced_ptdf.shape[1] == 2, (
        "Since only the first node is significant and the other columns are getting grouped, the reduced ptdf should have 2 columns"
    )
    assert reduced_injections.shape[1] == 2, (
        "Since only the first node is significant and the other nodes are getting grouped, the reduced nodal injections should have 2 columns"
    )

    expected_ptdf = np.array([[10, 60], [1, 6], [1, 0]])
    expected_nodal_injection = np.array([[2, 1]])
    assert np.all(reduced_ptdf == expected_ptdf)
    assert np.all(reduced_injections == expected_nodal_injection)


def test_reduce_ptdf_and_nodal_injections_multi_timestep():
    ptdf = np.array([[10, 10, 10, 10], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=float)
    nodal_injection = np.array([[2, 2, 2, 2], [1, 1, 1, 1]], dtype=float)
    expected_loadflow = ptdf @ nodal_injection.T
    significant_nodes = np.array([True, False, False, False])
    reduced_ptdf, reduced_injections = reduce_ptdf_and_nodal_injections(ptdf, nodal_injection, significant_nodes)

    reduced_loadflow = reduced_ptdf @ reduced_injections.T
    assert np.all(reduced_loadflow == expected_loadflow)

    assert reduced_ptdf.shape[1] == 3, (
        "Since only the first node is significant and there are two timesteps, the reduced ptdf should have 3 columns"
    )
    assert reduced_injections.shape[1] == 3, (
        "Since only the first node is significant and there are 2 timesteps, the reduced nodal injections should have 3 columns"
    )

    expected_ptdf = np.array([[10, 60, 30], [1, 6, 3], [1, 0, 0]])
    expected_nodal_injection = np.array([[2, 1, 0], [1, 0, 1]])
    assert np.all(reduced_ptdf == expected_ptdf)
    assert np.all(reduced_injections == expected_nodal_injection)


def test_update_ids_linking_to_nodes():
    significant_node_mask = np.array([0, 8])
    # Node 0 and 8 are significant
    from_nodes = np.array([0, 0, 2, 2])
    to_nodes = np.array([1, 2, 3, 8])
    injection_nodes = np.array([4, 4, 5, 6, 7])
    # Fill value = 2, because there are 2 significant nodes. So at index=2 is the grouped node
    fill_value = 2
    slack = 8  # Significant
    updated_from_nodes, updated_to_nodes, updated_inj_nodes, updated_slack = update_ids_linking_to_nodes(
        from_nodes, to_nodes, injection_nodes, slack, significant_node_mask, fill_value
    )
    assert np.all(updated_from_nodes == np.array([0, 0, fill_value, fill_value]))
    assert np.all(updated_to_nodes == np.array([fill_value, fill_value, fill_value, 1]))
    assert np.all(updated_inj_nodes == np.array([fill_value, fill_value, fill_value, fill_value, fill_value]))
    assert updated_slack == 1
    slack = 4  # Insignificant
    with pytest.raises(ValueError):
        update_ids_linking_to_nodes(from_nodes, to_nodes, injection_nodes, slack, significant_node_mask, fill_value)
