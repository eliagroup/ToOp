# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions to reduce the amount of nodes to the significant ones.

This enables a reduction of the PTDF-Matrix, thus accelerating the preprocessing.
The functions are used to identify and update the significant nodes, removing the insignificant,
and updating all node-references accordingly.

Author: Leonard Hilfrich
Date: 2025-04-17

"""

import numpy as np
from jaxtyping import Bool, Float, Int


def get_significant_nodes(
    relevant_node_mask: Bool[np.ndarray, " n_nodes"],
    multi_outage_node_mask: Bool[np.ndarray, " n_multioutages n_nodes"],
    relevant_branches: Int[np.ndarray, " n_relevant_branches"],
    from_nodes: Int[np.ndarray, " n_nodes"],
    to_nodes: Int[np.ndarray, " n_nodes"],
    slack: int,
) -> Bool[np.ndarray, " n_nodes"]:
    """Get all nodes that are in some way significant for the loadflow computations and actions.

    This includes all nodes that are already part of relevant subs, connected to a relevant branch or
    part of a multi outage.
    To be sure we don't break stuff after switching, we also add nodes that are 2 branches away.

    Parameters
    ----------
    relevant_node_mask : Bool[np.ndarray, " n_nodes"]
        A mask indicating which nodes are significant.
    multi_outage_node_mask : Bool[np.ndarray, " n_nodes"]
        A mask indicating which nodes are part of a multi outage.
    relevant_branches : Int[np.ndarray, " n_relevant_branches"]
        The indices of the relevant branches.
    from_nodes : Int[np.ndarray, " n_branches"]
        The from nodes of each branch.
    to_nodes : Int[np.ndarray, " n_branches"]
        The to nodes of each branch.
    slack : int
        The slack node.

    Returns
    -------
    Bool[np.ndarray, " n_nodes"]
        A mask indicating which nodes are significant.
    """
    relevant_branch_mask = np.zeros_like(from_nodes, dtype=bool)
    relevant_branch_mask[relevant_branches] = True
    significant_nodes = relevant_node_mask.copy()
    nodes_within_range_of = 2
    for _ in range(nodes_within_range_of):
        from_nodes_significant = significant_nodes[from_nodes]
        to_nodes_significant = significant_nodes[to_nodes]
        relevant_branch_mask |= from_nodes_significant | to_nodes_significant
        # Add all nodes that are connected to relevant branches
        significant_nodes[from_nodes[relevant_branch_mask]] = True
        significant_nodes[to_nodes[relevant_branch_mask]] = True
    significant_nodes |= multi_outage_node_mask.any(axis=0)
    significant_nodes[slack] = True
    return significant_nodes


def get_updated_indices_due_to_filtering(
    ids_to_keep: Int[np.ndarray, " n_reduced_nodes"],
    ids_to_filter: Int[np.ndarray, " n_branches_or_injections"],
    fill_value: int,
) -> Int[np.ndarray, " n_branches_or_injections"]:
    """Update the indices of an array after removing some of the elements.

    Parameters
    ----------
    ids_to_keep : Int[np.ndarray, " n_reduced_nodes"]
        The indices that should be kept. Unique and sorted.
    ids_to_filter : Int[np.ndarray, " n_branches_or_injections"]
        The indices that should be updated. Not necessarily unique or sorted.
    fill_value : int
        The value to fill the indices that are not in ids_to_keep.

    Returns
    -------
    filtered_ids: Int[np.ndarray, " n_branches_or_injections"]
        The updated indices.
    """
    filtered_ids = np.full(ids_to_filter.shape[0], fill_value)
    idx_kept, new_ids = np.nonzero(ids_to_filter[:, None] == ids_to_keep)
    filtered_ids[idx_kept] = new_ids
    return filtered_ids


def reduce_ptdf_and_nodal_injections(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    nodal_injection: Float[np.ndarray, " n_timesteps n_nodes"],
    significant_node_mask: Bool[np.ndarray, " n_nodes"],
) -> tuple[
    Float[np.ndarray, " n_branches n_reduced_nodes_pluts_1"], Float[np.ndarray, " n_timesteps n_reduced_nodes_pluts_1"]
]:
    """Reduce the PTDF matrix and nodal injections to only include significant nodes.

    For each nodal injection timestep, an additional column will be added at the end of the nodal arrays.

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        The PTDF matrix.
    nodal_injection : Float[np.ndarray, " n_timesteps n_nodes"]
        The nodal injections.
    significant_node_mask : Bool[np.ndarray, " n_nodes"]
        A mask indicating which nodes are significant.

    Returns
    -------
    reduced_ptdf : Float[np.ndarray, " n_branches n_reduced_nodes_pluts_1"]
        The reduced PTDF matrix. Includes all columns for the significant nodes and one additional column
        for all of the insignificant nodes.
        This additional column contains the loadflow resulting from all insignificant nodes.
    reduced_nodal_injection : Float[np.ndarray, " n_timesteps n_reduced_nodes_pluts_1"]
        The reduced nodal injections. Includes all columns for the significant nodes and one additional column
        for all of the insignificant nodes.
        Since the loadflow is set in the last column of the PTDF the nodal injection value is 1
    """
    lf_from_insignificant = ptdf[:, ~significant_node_mask] @ nodal_injection[:, ~significant_node_mask].T
    reduced_ptdf = np.c_[ptdf[:, significant_node_mask], lf_from_insignificant]
    reduced_nodal_injection = np.c_[nodal_injection[:, significant_node_mask], np.eye(nodal_injection.shape[0])]
    return reduced_ptdf, reduced_nodal_injection


def update_ids_linking_to_nodes(
    from_nodes: Int[np.ndarray, " n_branches"],
    to_nodes: Int[np.ndarray, " n_branches"],
    injection_nodes: Int[np.ndarray, " n_injections"],
    slack: int,
    significant_node_ids: Int[np.ndarray, " n_nodes"],
    index_of_last_column: int,
) -> tuple[Int[np.ndarray, " n_branches"], Int[np.ndarray, " n_branches"], Int[np.ndarray, " n_injections"], int]:
    """Update ids to nodes to refer to the correct nodes after reducing the network to only significant nodes.

    All nodes that are irrelevant will refer to an additional artificial node at the end of the node arrays.

    Parameters
    ----------
    from_nodes : Int[np.ndarray, " n_branches"]
        The from nodes of each branch.
    to_nodes : Int[np.ndarray, " n_branches"]
        The to nodes of each branch.
    injection_nodes : Int[np.ndarray, " n_injections"]
        The node of each injection
    slack : int
        The slack node.
    significant_node_ids : Bool[np.ndarray, " n_nodes"]
        The ids of the significant nodes.
    index_of_last_column: int
        The index of the last column added to the reduced ptdf

    Returns
    -------
    from_nodes : Int[np.ndarray, " n_branches"]
        The from nodes of each branch after reduction.
    to_nodes : Int[np.ndarray, " n_branches"]
        The to nodes of each branch after reduction.
    injection_nodes : Int[np.ndarray, " n_injections"]
        The node of each injection after reduction.
    slack : int
        The updated slack node.

    """
    from_nodes = get_updated_indices_due_to_filtering(significant_node_ids, from_nodes, index_of_last_column)
    to_nodes = get_updated_indices_due_to_filtering(significant_node_ids, to_nodes, index_of_last_column)
    injection_nodes = get_updated_indices_due_to_filtering(significant_node_ids, injection_nodes, index_of_last_column)
    if slack in significant_node_ids:
        slack = int(np.flatnonzero(slack == significant_node_ids)[0])
    else:
        raise ValueError(
            "The slack node should always be significant, so it doesnt break things reliant on being on the slack"
        )
    return from_nodes, to_nodes, injection_nodes, slack
