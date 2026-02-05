# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A numpy reference implementing the basic LODF and BSDF formulas. This is mostly used for testing,
though the bsdf_check_split function is also used to pre-filter the action set.

Bus Split Distribution Factors
DOI:10.36227/techrxiv.22298950.v1

Unified algebraic deviation of distribution factors in linear power flow
https://doi.org/10.48550/arXiv.2412.16164
"""

import numpy as np
from jaxtyping import Bool, Float, Int


def _validate_assignment(assignment: Bool[np.ndarray, " n_branches_at_node"]) -> None:
    """
    Validate that both busbars receive at least one branch.

    Parameters
    ----------
    assignment : Bool[np.ndarray, " n_branches_at_node"]
        Boolean mask (False -> bus A, True -> bus B).

    Raises
    ------
    AssertionError
        If one of the busbars receives zero branches.
    """
    if np.sum(assignment) < 1 or np.sum(~assignment) < 1:
        raise AssertionError("Each busbar must have at least one branch assigned.")


def _get_local_branch_data(
    switched_node: int,
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    branch_direction: list[Bool[np.ndarray, " n_branches_at_node"]],
) -> tuple[
    Int[np.ndarray, " n_branches_at_node"],
    Bool[np.ndarray, " n_branches_at_node"],
]:
    """
    Retrieve branch indices and direction arrays for the switched node.

    Parameters
    ----------
    switched_node : int
        Index of the node being split.
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        List of branch index arrays per node.
    branch_direction : list[Bool[np.ndarray, " n_branches_at_node"]]
        List of direction masks per node (True = leaving, False = entering).

    Returns
    -------
    branches_local : Int[np.ndarray, " n_branches_at_node"]
        Branch indices local to the switched node.
    direction_local : Bool[np.ndarray, " n_branches_at_node"]
        Direction mask local to the switched node.
    """
    return branches_at_nodes[switched_node], branch_direction[switched_node]


def _compute_ptdf_difference(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    branch_from_a: Int[np.ndarray, " branches_from_a"],
    branch_to_a: Int[np.ndarray, " branches_to_a"],
) -> Float[np.ndarray, " n_nodes"]:
    """
    Compute PTDF difference vector for the switched node.

    Defined as sum(PTDF of branches to bus A) - sum(PTDF of branches from bus A).

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        PTDF matrix.
    branch_from_a : Int[np.ndarray, " branches_from_a"]
        Indices of branches with from-end on bus A.
    branch_to_a : Int[np.ndarray, " branches_to_a"]
        Indices of branches with to-end on bus A.

    Returns
    -------
    ptdf_diff : Float[np.ndarray, " n_nodes"]
        PTDF difference vector.
    """
    return np.sum(ptdf[branch_to_a, :], axis=0) - np.sum(ptdf[branch_from_a, :], axis=0)


def _adjust_ptdf_for_slack(
    ptdf_diff: Float[np.ndarray, " n_nodes"],
    is_slack: bool,
    bus_a_ptdf: int,
    bus_b_ptdf: int,
) -> tuple[
    Float[np.ndarray, " n_nodes"],
    Float[np.ndarray, " n_nodes"],
]:
    """
    Apply slack node adjustments and derive relative PTDF difference.

    Parameters
    ----------
    ptdf_diff : Float[np.ndarray, " n_nodes"]
        Raw PTDF difference vector.
    is_slack : bool
        Whether the switched node is the system slack.
    bus_a_ptdf : int
        Column index for bus A.
    bus_b_ptdf : int
        Column index for bus B.

    Returns
    -------
    ptdf_diff_adj : Float[np.ndarray, " n_nodes"]
        Adjusted PTDF difference.
    ptdf_b : Float[np.ndarray, " n_nodes"]
        PTDF difference shifted by bus B reference.
    """
    ptdf_diff_adj = ptdf_diff.copy()
    if is_slack:
        ptdf_diff_adj -= 1
        ptdf_diff_adj[bus_a_ptdf] = 0
    else:
        ptdf_diff_adj[bus_a_ptdf] += 1
    ptdf_b = ptdf_diff_adj - ptdf_diff_adj[bus_b_ptdf]
    return ptdf_diff_adj, ptdf_b


def _compute_bsdf_denominator(
    suscept_from_a: Float[np.ndarray, " branches_from_a"],
    suscept_to_a: Float[np.ndarray, " branches_to_a"],
    ptdf_b: Float[np.ndarray, " n_nodes"],
    node_from_a: Int[np.ndarray, " branches_from_a"],
    node_to_a: Int[np.ndarray, " branches_to_a"],
    tol: float = 1e-5,
) -> float:
    """
    Compute BSDF denominator and validate numerical stability.

    Parameters
    ----------
    suscept_from_a : Float[np.ndarray, " branches_from_a"]
        Susceptances of branches with from-end on bus A.
    suscept_to_a : Float[np.ndarray, " branches_to_a"]
        Susceptances of branches with to-end on bus A.
    ptdf_b : Float[np.ndarray, " n_nodes"]
        Adjusted PTDF difference vector.
    node_from_a : Int[np.ndarray, " branches_from_a"]
        Node indices connected via from-end on bus A.
    node_to_a : Int[np.ndarray, " branches_to_a"]
        Node indices connected via to-end on bus A.
    tol : float, default 1e-5
        Minimum absolute denominator tolerance.

    Returns
    -------
    denom : float
        BSDF denominator.

    Raises
    ------
    ValueError
        If |denom| < tol indicating potential grid split.
    """
    g_sw = np.dot(suscept_from_a, ptdf_b[node_from_a]) + np.dot(suscept_to_a, ptdf_b[node_to_a])
    denom = suscept_from_a.sum() + suscept_to_a.sum() - g_sw
    if np.abs(denom) < tol:
        raise ValueError("BSDF denominator too small; grid would split.")
    return denom


def _compute_pedf_terms(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    bus_a_ptdf: int,
    node_from_a: Int[np.ndarray, " branches_from_a"],
    node_to_a: Int[np.ndarray, " branches_to_a"],
) -> tuple[
    Float[np.ndarray, " n_branches branches_from_a"],
    Float[np.ndarray, " n_branches branches_to_a"],
    Float[np.ndarray, " n_branches 1"],
]:
    """
    Compute Partial PTDF (PEDF) terms relative to bus A.

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        PTDF matrix.
    bus_a_ptdf : int
        Column index for bus A.
    node_from_a : Int[np.ndarray, " branches_from_a"]
        Node indices for from-end connections to bus A.
    node_to_a : Int[np.ndarray, " branches_to_a"]
        Node indices for to-end connections to bus A.

    Returns
    -------
    pedf_from : Float[np.ndarray, " n_branches branches_from_a"]
        PEDF terms for from-end nodes.
    pedf_to : Float[np.ndarray, " n_branches branches_to_a"]
        PEDF terms for to-end nodes.
    ptdf_bus_a : Float[np.ndarray, " n_branches 1"]
        Column vector of PTDF for bus A.
    """
    ptdf_bus_a = ptdf[:, bus_a_ptdf][:, None]
    pedf_from = ptdf[:, node_from_a] - ptdf_bus_a
    pedf_to = ptdf[:, node_to_a] - ptdf_bus_a
    return pedf_from, pedf_to, ptdf_bus_a


def _compute_bsdf_vector(
    suscept_from_a: Float[np.ndarray, " branches_from_a"],
    suscept_to_a: Float[np.ndarray, " branches_to_a"],
    pedf_from: Float[np.ndarray, " n_branches branches_from_a"],
    pedf_to: Float[np.ndarray, " n_branches branches_to_a"],
    branch_from_a: Int[np.ndarray, " branches_from_a"],
    branch_to_a: Int[np.ndarray, " branches_to_a"],
    denom: float,
) -> Float[np.ndarray, " n_branches"]:
    """
    Compute the BSDF vector.

    Parameters
    ----------
    suscept_from_a : Float[np.ndarray, " branches_from_a"]
        Susceptances for from-end branches on bus A.
    suscept_to_a : Float[np.ndarray, " branches_to_a"]
        Susceptances for to-end branches on bus A.
    pedf_from : Float[np.ndarray, " n_branches branches_from_a"]
        PEDF matrix for from-end nodes.
    pedf_to : Float[np.ndarray, " n_branches branches_to_a"]
        PEDF matrix for to-end nodes.
    branch_from_a : Int[np.ndarray, " branches_from_a"]
        Indices of branches with from-end on bus A.
    branch_to_a : Int[np.ndarray, " branches_to_a"]
        Indices of branches with to-end on bus A.
    denom : float
        BSDF denominator.

    Returns
    -------
    bsdf : Float[np.ndarray, " n_branches"]
        BSDF vector.
    """
    nom = (suscept_from_a * pedf_from).sum(axis=1) + (suscept_to_a * pedf_to).sum(axis=1)
    nom[branch_from_a] += suscept_from_a
    nom[branch_to_a] -= suscept_to_a
    return nom / denom


def _update_network_after_split(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    bsdf: Float[np.ndarray, " n_branches"],
    ptdf_diff: Float[np.ndarray, " n_nodes"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branch_from_b: Int[np.ndarray, " branches_from_b"],
    branch_to_b: Int[np.ndarray, " branches_to_b"],
    bus_b_ptdf: int,
) -> tuple[
    Float[np.ndarray, " n_branches n_nodes"],
    Int[np.ndarray, " n_branches"],
    Int[np.ndarray, " n_branches"],
]:
    """
    Update PTDF and node incidence arrays after applying a split.

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        Original PTDF matrix.
    bsdf : Float[np.ndarray, " n_branches"]
        BSDF vector.
    ptdf_diff : Float[np.ndarray, " n_nodes"]
        PTDF difference vector.
    from_node : Int[np.ndarray, " n_branches"]
        Original from-node indices.
    to_node : Int[np.ndarray, " n_branches"]
        Original to-node indices.
    branch_from_b : Int[np.ndarray, " branches_from_b"]
        Branch indices with from-end reassigned to bus B.
    branch_to_b : Int[np.ndarray, " branches_to_b"]
        Branch indices with to-end reassigned to bus B.
    bus_b_ptdf : int
        PTDF column index for bus B.

    Returns
    -------
    ptdf_new : Float[np.ndarray, " n_branches n_nodes"]
        Updated PTDF matrix.
    from_node_new : Int[np.ndarray, " n_branches"]
        Updated from-node indices.
    to_node_new : Int[np.ndarray, " n_branches"]
        Updated to-node indices.
    """
    ptdf_new = ptdf + np.outer(bsdf, ptdf_diff)
    from_node_new = from_node.copy()
    to_node_new = to_node.copy()
    from_node_new[branch_from_b] = bus_b_ptdf
    to_node_new[branch_to_b] = bus_b_ptdf
    return ptdf_new, from_node_new, to_node_new


def get_bsdf_branch_indices(
    assignment: Bool[np.ndarray, " n_branches_at_node"],
    branches_locally: Int[np.ndarray, " n_branches_at_node"],
    branch_direction_locally: Bool[np.ndarray, " n_branches_at_node"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
) -> tuple[
    Int[np.ndarray, " branches_from_a"],
    Int[np.ndarray, "branches_to_a"],
    Int[np.ndarray, "branches_from_b"],
    Int[np.ndarray, "branches_to_b"],
    Int[np.ndarray, "node_from_a"],
    Int[np.ndarray, "node_to_a"],
]:
    """Extract the branch indices for the BSDF computation

    Parameters
    ----------
    assignment : Bool[np.ndarray, " n_branches_at_node"]
        A boolean array indicating which branches are assigned to busbar A (false) and which to
        busbar B (true) in the station
    branches_locally : Int[np.ndarray, " n_branches_at_node"]
        The branches entering/leaving the switched node - from tot_stat
    branch_direction_locally : Bool[np.ndarray, " n_branches_at_node"]
        A boolean array indicating wether the according branch in the branches_at_node
        list is leaving (True) or entering (False) the given node, from from_stat_bool
    from_node : Int[np.ndarray, " n_branches"]
        The from nodes of the branches
    to_node : Int[np.ndarray, " n_branches"]
        The to nodes of the branches

    Returns
    -------
    Int[np.ndarray, " branches_from_a"]
        The branch indices that have a from-end on bus A
    Int[np.ndarray, "branches_to_a"]
        The branch indices that have a to-end on bus A
    Int[np.ndarray, "branches_from_b"]
        The branch indices that have a from-end on bus B
    Int[np.ndarray, "branches_to_b"]
        The branch indices that have a to-end on bus B
    Int[np.ndarray, "node_from_a"]
        The nodes that are connected through a branch with a from-end on bus A
    Int[np.ndarray, "node_to_a"]
        The nodes that are connected through a branch with a to-end on bus A
    """
    assert len(assignment) == len(branches_locally) == len(branch_direction_locally), (
        "Assignment is for a different number of branches than available"
    )

    brh_from_a = branches_locally[~assignment & branch_direction_locally]
    brh_to_a = branches_locally[~assignment & ~branch_direction_locally]
    brh_from_b = branches_locally[assignment & branch_direction_locally]
    brh_to_b = branches_locally[assignment & ~branch_direction_locally]

    node_from_a = to_node[brh_from_a]
    node_to_a = from_node[brh_to_a]

    return brh_from_a, brh_to_a, brh_from_b, brh_to_b, node_from_a, node_to_a


def calc_bsdf(
    switched_node: int,
    assignment: Bool[np.ndarray, " n_branches_at_node"],
    is_slack: bool,
    bus_a_ptdf: int,
    bus_b_ptdf: int,
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    susceptance: Float[np.ndarray, " n_branches"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    branch_direction: list[Bool[np.ndarray, " n_branches_at_node"]],
) -> tuple[
    Float[np.ndarray, " n_branches"],
    Float[np.ndarray, " n_branches n_nodes"],
    Int[np.ndarray, " n_branches"],
    Int[np.ndarray, " n_branches"],
]:
    """
    Compute the Bus Split Distribution Factors (BSDF) for a node split.

    Parameters
    ----------
    switched_node : int
        Index of the node being split.
    assignment : Bool[np.ndarray, " n_branches_at_node"]
        Boolean mask for branch assignment to busbars.
    is_slack : bool
        True if the node is the slack node.
    bus_a_ptdf : int
        PTDF column index for busbar A.
    bus_b_ptdf : int
        PTDF column index for busbar B.
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        PTDF matrix.
    susceptance : Float[np.ndarray, " n_branches"]
        Branch susceptances.
    from_node : Int[np.ndarray, " n_branches"]
        Branch from-nodes.
    to_node : Int[np.ndarray, " n_branches"]
        Branch to-nodes.
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        Branch indices at each node.
    branch_direction : list[Bool[np.ndarray, " n_branches_at_node"]]
        Direction mask for branches at each node.

    Returns
    -------
    bsdf : Float[np.ndarray, " n_branches"]
        BSDF vector.
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        Updated PTDF matrix.
    from_node : Int[np.ndarray, " n_branches"]
        Updated from-nodes.
    to_node : Int[np.ndarray, " n_branches"]
        Updated to-nodes.

    Raises
    ------
    ValueError
        If the denominator is too small, indicating a grid split.
    """
    # 1. Validate assignment
    _validate_assignment(assignment)

    # 2. Local branch data
    branches_local, direction_local = _get_local_branch_data(
        switched_node=switched_node,
        branches_at_nodes=branches_at_nodes,
        branch_direction=branch_direction,
    )

    # 3. Partition branches by busbar & direction
    branch_from_a, branch_to_a, branch_from_b, branch_to_b, node_from_a, node_to_a = get_bsdf_branch_indices(
        assignment=assignment,
        branches_locally=branches_local,
        branch_direction_locally=direction_local,
        from_node=from_node,
        to_node=to_node,
    )

    # 4. PTDF difference & slack adjustment
    ptdf_diff_raw = _compute_ptdf_difference(
        ptdf=ptdf,
        branch_from_a=branch_from_a,
        branch_to_a=branch_to_a,
    )
    ptdf_diff, ptdf_b = _adjust_ptdf_for_slack(
        ptdf_diff=ptdf_diff_raw,
        is_slack=is_slack,
        bus_a_ptdf=bus_a_ptdf,
        bus_b_ptdf=bus_b_ptdf,
    )

    # 5. Susceptance slices
    suscept_from_a = susceptance[branch_from_a]
    suscept_to_a = susceptance[branch_to_a]

    # 6. Denominator
    denom = _compute_bsdf_denominator(
        suscept_from_a=suscept_from_a,
        suscept_to_a=suscept_to_a,
        ptdf_b=ptdf_b,
        node_from_a=node_from_a,
        node_to_a=node_to_a,
    )

    # 7. PEDF terms
    pedf_from, pedf_to, _ptdf_bus_a = _compute_pedf_terms(
        ptdf=ptdf,
        bus_a_ptdf=bus_a_ptdf,
        node_from_a=node_from_a,
        node_to_a=node_to_a,
    )

    # 8. BSDF vector
    bsdf = _compute_bsdf_vector(
        suscept_from_a=suscept_from_a,
        suscept_to_a=suscept_to_a,
        pedf_from=pedf_from,
        pedf_to=pedf_to,
        branch_from_a=branch_from_a,
        branch_to_a=branch_to_a,
        denom=denom,
    )

    # 9. Apply update to PTDF and node mappings
    ptdf_new, from_node_new, to_node_new = _update_network_after_split(
        ptdf=ptdf,
        bsdf=bsdf,
        ptdf_diff=ptdf_diff,
        from_node=from_node,
        to_node=to_node,
        branch_from_b=branch_from_b,
        branch_to_b=branch_to_b,
        bus_b_ptdf=bus_b_ptdf,
    )

    return bsdf, ptdf_new, from_node_new, to_node_new


def extract_bsdf_data(
    topo_vect: Bool[np.ndarray, " n_branches_at_subs"],
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    relevant_nodes: Int[np.ndarray, " n_relevant_subs"],
    slack: int,
    n_stat: int,
) -> list[dict]:
    """Takes a topo vect in dense numpy format and returns a list of parameters to calc_bsdf

    Parameters
    ----------
    topo_vect : Bool[np.ndarray, " n_branches_at_subs"]
        The topology vector in dense numpy format, i.e. a boolean for every branch end in the grid
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        List of arrays containing all branch indices leaving the relevant nodes.
        The branch index points into the list of all branches, length N_relevant_nodes
    relevant_nodes : Int[np.ndarray, " n_relevant_subs"]
        The relevant nodes in the grid
    slack : int
        The index of the slack node
    n_stat : int
        The number of static nodes in the grid

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing the parameters for calc_bsdf for one switched node
        The keys are:
            - switched_node: The index of the switched node
            - assignment: The assignment of branches to busbar A (False) and busbar B (True)
            - is_slack: True if the switched node is the slack node
            - bus_a_ptdf: The column of the busbar A in the ptdf matrix
            - bus_b_ptdf: The column of the busbar B in the ptdf matrix
        The list will have as many entries as there are split substations in the topo_vect
    """

    res = []
    topo_vect_index = 0
    for sub_id, branches_at_node in enumerate(branches_at_nodes):
        local_topo_vect = topo_vect[topo_vect_index : topo_vect_index + len(branches_at_node)]
        topo_vect_index += len(branches_at_node)
        if not np.any(local_topo_vect):
            continue

        res.append(
            {
                "switched_node": sub_id,
                "assignment": local_topo_vect,
                "is_slack": relevant_nodes[sub_id] == slack,
                "bus_a_ptdf": relevant_nodes[sub_id],
                "bus_b_ptdf": n_stat + sub_id,
            }
        )
    assert topo_vect_index == len(topo_vect)
    return res


def compute_bus_splits(
    topo_vect: Bool[np.ndarray, " n_branches_at_subs"],
    relevant_nodes: Int[np.ndarray, " n_relevant_subs"],
    slack: int,
    n_stat: int,
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    susceptance: Float[np.ndarray, " n_branches"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    branch_direction: list[Bool[np.ndarray, " n_branches_at_node"]],
) -> tuple[
    Float[np.ndarray, " n_branches n_nodes"],
    Int[np.ndarray, " n_branches"],
    Int[np.ndarray, " n_branches"],
]:
    """Compute multiple bus splits and return the updated PTDF, from and to-nodes.

    Parameters
    ----------
    topo_vect : Bool[np.ndarray, " n_branches_at_subs"]
        The topology vector in dense numpy format, i.e. a boolean for every branch end in the grid
    relevant_nodes : Int[np.ndarray, " n_relevant_subs"]
        The relevant nodes in the grid
    slack : int
        The index of the slack node
    n_stat : int
        The number of static nodes in the grid
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        The ptdf matrix before the split, but already including the busbar b (can be just the copy
        of busbar a)
    susceptance : Float[np.ndarray, " n_branches"]
        The susceptances of the branches
    from_node : Int[np.ndarray, " n_branches"]
        The from nodes of the branches
    to_node : Int[np.ndarray, " n_branches"]
        The to nodes of the branches
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        List of arrays containing all branch indices leaving or entering the relevant nodes.
    branch_direction : list[Bool[np.ndarray, " n_branches_at_node"]]
        A list of boolean Arrays indicating whether the according branch in the branches_at_node
        enters (True) or leaves (False) the given node, length N_relevant_nodes

    Returns
    -------
    Float[np.ndarray, " n_branches n_nodes"]
        The PTDF matrix after the split
    Int[np.ndarray, " n_branches"]
        The from nodes of the branches after the split
    Int[np.ndarray, " n_branches"]
        The to nodes of the branches after the split

    Raises
    ------
    ValueError
        If a BSDF computation split the grid
    """
    for data in extract_bsdf_data(
        topo_vect=topo_vect,
        branches_at_nodes=branches_at_nodes,
        relevant_nodes=relevant_nodes,
        slack=slack,
        n_stat=n_stat,
    ):
        _bsdf, ptdf, from_node, to_node = calc_bsdf(
            ptdf=ptdf,
            susceptance=susceptance,
            from_node=from_node,
            to_node=to_node,
            branches_at_nodes=branches_at_nodes,
            branch_direction=branch_direction,
            **data,
        )

    return ptdf, from_node, to_node


def calc_lodf(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branch_to_outage: int,
) -> Float[np.ndarray, " n_branches"]:
    """Calculate the LODF vector for a branch outage

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        The PTDF matrix
    from_node : Int[np.ndarray, " n_branches"]
        The from nodes of the branches
    to_node : Int[np.ndarray, " n_branches"]
        The to nodes of the branches
    branch_to_outage : int
        The index of the branch to outage

    Returns
    -------
    Float[np.ndarray, " n_branches"]
        The LODF vector for the branch outage

    Raises
    ------
    ValueError
        If the denominator of the LODF is too small, indicating that the outage will split the grid.
    """
    i = from_node[branch_to_outage]  # shape (M,)
    j = to_node[branch_to_outage]  # shape (M,)

    # Numerators: PTDF[:, i] - PTDF[:, j] for each outage (broadcast to (E, M))
    numerators = ptdf[:, i] - ptdf[:, j]  # shape (E, M)
    # Denominators: 1 - (PTDF[e, i] - PTDF[e, j]) per outage e
    denominators = 1.0 - (ptdf[branch_to_outage, i] - ptdf[branch_to_outage, j])  # shape (M,)
    numerators[branch_to_outage] = -denominators

    if np.abs(denominators) < 1e-11:
        raise ValueError("Denominator of LODF is too small - this outage will split the grid")

    return numerators / denominators


def contingency_analysis(
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branches_to_outage: Int[np.ndarray, " n_failures"],
    nodal_injections: Float[np.ndarray, " n_timesteps n_nodes"],
) -> tuple[Float[np.ndarray, " n_failures n_branches"], Bool[np.ndarray, " n_failures"]]:
    """Perform a branch contingency analysis

    This will compute N-0 flows and all N-1 flows after a branch outage.

    Note that this does not support multi-outages, injection outages or busbar outages.

    Parameters
    ----------
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        The PTDF matrix
    from_node : Int[np.ndarray, " n_branches"]
        The from nodes of the branches
    to_node : Int[np.ndarray, " n_branches"]
        The to nodes of the branches
    branches_to_outage : Int[np.ndarray, " n_failures"]
        The indices of the branches to outage
    nodal_injections : Float[np.ndarray, " n_timesteps n_nodes"]
        The nodal injections

    Returns
    -------
    Float[np.ndarray, " n_failures n_branches"]
        The N-1 flows for each outage. Unsuccessful N-1 cases will contain all zero.
    Bool[np.ndarray, " n_failures"]
        Whether the LODF calculation was successful (true) or if it split the grid (false).
    """

    n_0_flows = ptdf @ nodal_injections
    n_1_flows = []
    success = []
    for branch_to_outage in branches_to_outage:
        try:
            lodf = calc_lodf(ptdf, from_node, to_node, branch_to_outage)
            n_1_flows.append(n_0_flows + lodf * n_0_flows[branch_to_outage])
            success.append(True)
        except ValueError:
            n_1_flows.append(np.zeros_like(ptdf[:, 0]))
            success.append(False)

    return np.array(n_1_flows), np.array(success)


def run_solver(
    branch_topo_vect: Bool[np.ndarray, " n_branches_at_subs"],
    relevant_nodes: Int[np.ndarray, " n_relevant_subs"],
    slack: int,
    n_stat: int,
    ptdf: Float[np.ndarray, " n_branches n_nodes"],
    susceptance: Float[np.ndarray, " n_branches"],
    from_node: Int[np.ndarray, " n_branches"],
    to_node: Int[np.ndarray, " n_branches"],
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    branch_direction: list[Bool[np.ndarray, " n_branches_at_node"]],
    branches_to_outage: Int[np.ndarray, " n_failures"],
    nodal_injections: Float[np.ndarray, " n_timesteps n_nodes"],
) -> tuple[Float[np.ndarray, " n_failures n_branches"], Bool[np.ndarray, " n_failures"]]:
    """Compute the N-1 flows for a given topology

    This will apply the topology using the BSDF formula and then run a branch-only N-1 computation

    Parameters
    ----------
    branch_topo_vect : Bool[np.ndarray, " n_branches_at_subs"]
        The topology vector in dense numpy format, i.e. a boolean for every branch end in the grid with
        True if on bus B and False if on bus A
    relevant_nodes : Int[np.ndarray, " n_relevant_subs"]
        The relevant nodes in the grid
    slack : int
        The index of the slack node
    n_stat : int
        The number of static nodes in the grid
    ptdf : Float[np.ndarray, " n_branches n_nodes"]
        The ptdf matrix before the split, but already including the busbar b (can be just the copy
        of busbar a)
    susceptance : Float[np.ndarray, " n_branches"]
        The susceptances of the branches
    from_node : Int[np.ndarray, " n_branches"]
        The from nodes of the branches
    to_node : Int[np.ndarray, " n_branches"]
        The to nodes of the branches
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        List of arrays containing all branch indices leaving or entering the relevant nodes.
    branch_direction : list[Bool[np.ndarray, " n_branches_at_node"]]
        A list of boolean Arrays indicating whether the according branch in the branches_at_node
        enters (True) or leaves (False) the given node, length N_relevant_nodes
    branches_to_outage : Int[np.ndarray, " n_failures"]
        The indices of the branches to outage
    nodal_injections : Float[np.ndarray, " n_timesteps n_nodes"]
        The nodal injections

    Returns
    -------
    Float[np.ndarray, " n_failures n_branches"]
        The N-1 flows for each outage. Unsuccessful N-1 cases will contain all zero.
    Bool[np.ndarray, " n_failures"]
        Whether the LODF calculation was successful (true) or if it split the grid (false).

    Raises
    ------
    ValueError
        If a BSDF computation split the grid
    """
    ptdf, from_node, to_node = compute_bus_splits(
        topo_vect=branch_topo_vect,
        relevant_nodes=relevant_nodes,
        slack=slack,
        n_stat=n_stat,
        ptdf=ptdf,
        susceptance=susceptance,
        from_node=from_node,
        to_node=to_node,
        branches_at_nodes=branches_at_nodes,
        branch_direction=branch_direction,
    )

    return contingency_analysis(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        branches_to_outage=branches_to_outage,
        nodal_injections=nodal_injections,
    )
