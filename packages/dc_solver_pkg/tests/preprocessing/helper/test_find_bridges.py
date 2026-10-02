# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import numpy as np
from tests.network_data_pickle import load_network_data
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_branches_to_spare_from_groups,
    find_bridges,
    find_islanding_branch_groups,
    find_n_minus_2_safe_branches,
    get_number_of_bridges_after_outage,
)
from toop_engine_dc_solver.preprocess.helpers.ptdf import get_connectivity_matrix


def _get_graph_old_path(
    from_node: np.ndarray,
    to_node: np.ndarray,
    number_of_branches: int,
    number_of_nodes: int,
) -> nx.MultiGraph:
    connectivity_matrix = get_connectivity_matrix(
        from_node,
        to_node,
        number_of_branches,
        number_of_nodes,
        directed=False,
    )
    graph = connectivity_matrix.T @ connectivity_matrix
    return nx.from_scipy_sparse_array(graph, parallel_edges=True, create_using=nx.MultiGraph)


def _get_number_of_bridges_after_outage_old_path(
    cases_to_check: np.ndarray,
    from_node: np.ndarray,
    to_node: np.ndarray,
    number_of_branches: int,
    number_of_nodes: int,
) -> np.ndarray:
    n_bridges = np.zeros(len(cases_to_check), dtype=int)
    for index, branch in enumerate(cases_to_check):
        from_node_temp = np.delete(from_node, branch)
        to_node_temp = np.delete(to_node, branch)
        temp_graph = _get_graph_old_path(from_node_temp, to_node_temp, number_of_branches - 1, number_of_nodes)
        n_bridges[index] = len(set(nx.bridges(temp_graph)))
    return n_bridges


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

    # Make sure only the cases we want to check are checked.
    cases_to_check = np.array([1, 4])
    n_minus_2_safe = find_n_minus_2_safe_branches(from_node, to_node, 5, 4, cases_to_check)
    expected_n_minus_2_safe = np.array(
        [
            False,
            True,
        ],
        dtype=bool,
    )
    assert np.array_equal(n_minus_2_safe, expected_n_minus_2_safe)

    # If we only want to outage the 0-3 branch, then the 1-2 branch is also n-2 safe
    cases_to_check = np.array([1, 4])
    cases_to_outage = np.array([3])
    n_minus_2_safe = find_n_minus_2_safe_branches(from_node, to_node, 5, 4, cases_to_check, cases_to_outage)
    expected_n_minus_2_safe = np.array(
        [
            True,
            True,
        ],
        dtype=bool,
    )
    assert np.array_equal(n_minus_2_safe, expected_n_minus_2_safe)

    # If we do not check any outages, then all branches are n-2 safe, since we do not check any outages
    cases_to_check = np.array([1, 4])
    cases_to_outage = np.array([], dtype=int)
    n_minus_2_safe = find_n_minus_2_safe_branches(from_node, to_node, 5, 4, cases_to_check, cases_to_outage)
    expected_n_minus_2_safe = np.array(
        [
            True,
            True,
        ],
        dtype=bool,
    )
    assert np.array_equal(n_minus_2_safe, expected_n_minus_2_safe)


def test_get_number_of_bridges_after_outage_matches_old_path_for_complex_grid(
    create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path,
) -> None:
    network_data = load_network_data(create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path / "network_data.pkl")
    from_node = network_data.from_nodes
    to_node = network_data.to_nodes
    number_of_branches = len(network_data.branch_ids)
    number_of_nodes = len(network_data.node_ids)
    cases_to_check = np.arange(number_of_branches)
    outage_edges = {(int(from_bus), int(to_bus)) for from_bus, to_bus in zip(from_node, to_node, strict=False)}
    outage_edges |= {(int(to_bus), int(from_bus)) for from_bus, to_bus in zip(from_node, to_node, strict=False)}

    old_counts = _get_number_of_bridges_after_outage_old_path(
        cases_to_check=cases_to_check,
        from_node=from_node,
        to_node=to_node,
        number_of_branches=number_of_branches,
        number_of_nodes=number_of_nodes,
    )
    new_counts = get_number_of_bridges_after_outage(
        cases_to_check=cases_to_check,
        outage_edges=outage_edges,
        from_node=from_node,
        to_node=to_node,
        number_of_nodes=number_of_nodes,
    )

    assert np.array_equal(new_counts, old_counts)


def _three_winding_transformer_grid() -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """A meshed triangle plus a three-winding transformer hanging off it.

    Nodes 0, 1 and 2 are the HV/MV/LV sides, node 3 is the star node that only exists inside the
    transformer. Branches 3, 4 and 5 are its legs, so they are the only branches touching node 3.
    """
    from_node = np.array([0, 1, 2, 3, 3, 3], dtype=int)
    to_node = np.array([1, 2, 0, 0, 1, 2], dtype=int)
    legs = np.zeros((1, 6), dtype=bool)
    legs[0, [3, 4, 5]] = True
    return from_node, to_node, 4, legs


def test_three_winding_transformer_group_islands_its_own_star_node() -> None:
    from_node, to_node, number_of_nodes, legs = _three_winding_transformer_grid()

    # No single leg is a bridge - the star node keeps two connections - yet the three of them
    # together are a cut set, which is exactly what find_bridges alone cannot see.
    assert not find_bridges(from_node, to_node, 6, number_of_nodes)[[3, 4, 5]].any()
    assert find_islanding_branch_groups(from_node, to_node, number_of_nodes, legs)[0]


def test_three_winding_transformer_group_spares_exactly_one_leg() -> None:
    from_node, to_node, number_of_nodes, legs = _three_winding_transformer_grid()

    spared = find_branches_to_spare_from_groups(from_node, to_node, number_of_nodes, legs)

    # One leg stays in service, so the star node keeps a path to the rest of the grid, and it is
    # the lowest-index leg because sparing it on its own already resolves the islanding.
    assert np.array_equal(np.flatnonzero(spared[0]), np.array([3]))
    computed = legs & ~spared
    assert not find_islanding_branch_groups(from_node, to_node, number_of_nodes, computed)[0]
    # The other two legs are still computed: the repair is minimal, not a fallback to N-1.
    assert np.array_equal(np.flatnonzero(computed[0]), np.array([4, 5]))


def test_two_legs_of_a_three_winding_transformer_need_no_spare() -> None:
    """Sparing is keyed off islanding, not off the element type."""
    from_node, to_node, number_of_nodes, _ = _three_winding_transformer_grid()
    two_legs = np.zeros((1, 6), dtype=bool)
    two_legs[0, [3, 4]] = True

    assert not find_islanding_branch_groups(from_node, to_node, number_of_nodes, two_legs)[0]
    assert not find_branches_to_spare_from_groups(from_node, to_node, number_of_nodes, two_legs).any()
