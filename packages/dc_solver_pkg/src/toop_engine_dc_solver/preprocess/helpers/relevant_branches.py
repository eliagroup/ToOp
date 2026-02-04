# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Finds relevant branches in network data"""

import numpy as np
from jaxtyping import Bool, Int


def get_relevant_branches(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    relevant_node_mask: Bool[np.ndarray, " n_node"],
    monitored_branch_mask: Bool[np.ndarray, " n_branch"],
    outaged_branch_mask: Bool[np.ndarray, " n_branch"],
    multi_outage_mask: Bool[np.ndarray, " n_multi_outages n_branch"],
    busbar_outage_branch_mask: Bool[np.ndarray, " n_branch"],
) -> Int[np.ndarray, " n_branch_reduced"]:
    """Get all relevant branches.

    Filters out all branches that are not monitored, part of the N-1 or connected to a relevant
    substation.

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector.
    relevant_node_mask : Bool[np.ndarray, " n_node"]
        A mask indicating which nodes are relevant.
    monitored_branch_mask : Bool[np.ndarray, " n_branch"]
        A mask indicating which branches are monitored.
    outaged_branch_mask : Bool[np.ndarray, " n_branch"]
        A mask indicating which branches are outaged.
    multi_outage_mask : Bool[np.ndarray, " n_multi_outages n_branch"]
        A mask indicating which branches are outaged in the multi-outage case.
    busbar_outage_branch_mask : Bool[np.ndarray, " n_branch"]
        A mask indicating which branches are outaged due to busbar outages.

    Returns
    -------
    Int[np.ndarray, " n_branch_reduced"]
        The indices of the relevant branches
    """
    from_nodes_relevant = relevant_node_mask[from_node]
    to_nodes_relevant = relevant_node_mask[to_node]

    branches_relevant = (
        from_nodes_relevant
        | to_nodes_relevant
        | monitored_branch_mask
        | outaged_branch_mask
        | multi_outage_mask.any(axis=0)
        | busbar_outage_branch_mask
    )

    return np.flatnonzero(branches_relevant)
