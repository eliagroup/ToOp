"""Busbar Outage Simulation Module

This module provides functions to simulate and handle busbar outages in a power grid network.
It includes methods to calculate the resulting load flows, update network parameters, and
handle critical busbars during outages. The calculations leverage JAX for efficient numerical
computations and support batch processing for multiple busbar outages.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.aggregate_results import get_overload_energy_n_1_matrix
from toop_engine_dc_solver.jax.multi_outages import compute_multi_outage
from toop_engine_dc_solver.jax.topology_computations import pad_action_with_unsplit_action_indices
from toop_engine_dc_solver.jax.types import (
    ActionSet,
    BBOutageBaselineAnalysis,
    NonRelBBOutageData,
    RelBBOutageData,
    int_max,
)


def perform_outage_single_busbar(
    connected_branches_to_outage: Int[Array, " max_n_branches_failed"],
    injection_deltap_to_outage: Float[Array, " n_timesteps"],
    node_index_busbar: Int[Array, " "],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> tuple[Float[Array, " n_timesteps n_branches_monitored"], Bool[Array, " "]]:
    """Perform an outage on a single busbar and calculate the resulting load flows.

    Loadflows will be full of zeros, if a branch gets deleted in the busbar outage
    which is a bridge branch. Generally the bridge branches are filtered in the
    preprocessing step, howevere there are certain cases where the non-bridge branches
    become a bridge mask after the station split.

    Loadflows will also be full of 0 if nodal_index is invalid which means that this busbar
    should not be outaged.

    This functions tries twice to perform outages using the compute_multi_outage function.
    The first attempt uses the original connected_branches_to_outage, while the second
    trial attempts to outage all but one branches in connected_branches_to_outage. The
    idea is to handle cases where the first attempt fails due to grid splitting. This way,
    we can skip leaving a skeleton branch connected to the busbar in the preprocessing step.

    The final load flows are determined based on the success of the first attempt, and if
    it fails, the second attempt is used. If both attempts fail,
    the load flows are set to zero.

    Parameters
    ----------
    connected_branches_to_outage : Int[Array, " max_n_branches_failed"]
        Array of branch indices that are connected to the busbar and will be disconnected.
    injection_deltap_to_outage : Float[Array, " n_timesteps"]
        Array of injection changes (delta P) at the busbar for each timestep.
    node_index_busbar : Int[Array, " "]
        Index of the busbar node in the network.
    ptdf : Float[Array, " n_branches n_nodes"]
        Power Transfer Distribution Factors (PTDF) matrix.
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        Nodal injections for each timestep.
    from_node : Int[Array, " n_branches"]
        Array of from nodes for each branch.
    to_node : Int[Array, " n_branches"]
        Array of to nodes for each branch.
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        Initial load flows for each branch and timestep.
    branches_monitored : Int[Array, " n_branches_monitored"]
        Array of branch indices being monitored.

    Returns
    -------
    Float[Array, " n_timesteps n_branches_monitored"]
        Array of load flows for each branch and each timestep after the outage.
    Bool[Array, " "]
        Success flag indicating whether the outage was successful.
    """
    # Step 1: Otutage the injections
    delta_p = -1 * injection_deltap_to_outage
    del_lfs = jnp.einsum("b, t -> tb", ptdf.at[:, node_index_busbar].get(mode="fill", fill_value=int_max()), delta_p)
    n_0_flows_inj_outaged = n_0_flows + del_lfs
    # Step 2: Update the branches
    # Here for the loadflows, updated_nodal_injection are not considered
    lfs, success = compute_multi_outage(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        n_0_flow=n_0_flows_inj_outaged,
        multi_outages=connected_branches_to_outage,
        branches_monitored=branches_monitored,
    )

    # Here the success can be false if the outage of a branch leads to grid splitting.
    # This can happen if a skeleton branch is outaged

    lfs_retry, success_retry = compute_multi_outage(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        n_0_flow=n_0_flows_inj_outaged,
        multi_outages=connected_branches_to_outage.at[0].set(int_max()),
        branches_monitored=branches_monitored,
    )

    lfs = jnp.where(success, lfs, lfs_retry)
    success = jnp.logical_or(success, success_retry)

    lfs = jnp.where(success, lfs, jnp.nan)
    lfs = jnp.where(node_index_busbar <= nodal_injections.shape[1], lfs, 0.0)

    return lfs, success


def perform_outage_multi_busbars(
    connected_branches_to_outage: Int[Array, " n_bb_outages max_n_branches_failed"],
    injection_deltap_to_outage: Float[Array, " n_bb_outages n_timesteps"],
    node_index_busbar: Int[Array, " n_bb_outages"],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    branches_monitored: Int[Array, " n_branches_monitored"],
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    disconnections: Int[Array, " n_disconnections"] = None,
) -> tuple[Float[Array, " n_bb_outages n_timesteps n_branches_monitored"], Bool[Array, " n_bb_outages"]]:
    """Simulate outages for multiple busbars and computes the resulting load flow solutions.

    Parameters
    ----------
    connected_branches_to_outage : Int[Array, "n_bb_outages n_branches_failed"]
        Array indicating the branches connected to each busbar outage.
    injection_deltap_to_outage : Float[Array, "n_bb_outages n_timesteps"]
        Array of injection changes (delta P) for each busbar outage over timesteps.
    node_index_busbar : Int[Array, "n_bb_outages"]
        Array of busbar node indices for each outage.
    ptdf : Float[Array, "n_branches n_nodes"]
        Power Transfer Distribution Factor (PTDF) matrix.
    nodal_injections : Float[Array, "n_timesteps n_nodes"]
        Array of nodal injections over timesteps.
    from_nodes : Int[Array, "n_branches"]
        Array of "from" node indices for each branch.
    to_nodes : Int[Array, "n_branches"]
        Array of "to" node indices for each branch.
    branches_monitored : Int[Array, "n_branches_monitored"]
        Array of branch indices that are monitored.
    n_0_flows : Float[Array, "n_timesteps n_branches"]
        Initial load flows for each branch over timesteps.
    disconnections : Int[Array, "n_disconnections"], optional
        Array of disconnection actions which were performed before the busbar outage as part of
        topological actions. If provided, branches that are already outaged by these disconnections
        will not be outaged again in the busbar outage.

    Returns
    -------
    Float[Array, " n_bb_outages n_timesteps n_branches_monitored"]: Load flow solutions for each busbar outage over time.

    Bool[Array, " n_bb_outages"]: Success flags indicating whether the load flow solution was successfully computed
    for each outage.

    Raises
    ------
    AssertionError
        If the input arrays have inconsistent shapes or if the node index is invalid.
    """
    # Handle disconnection actions: Prevent double outage of branches
    # that are already outaged by the disconnection action.
    if disconnections is not None:
        connected_branches_to_outage = jax.vmap(filter_already_outaged_branches_single_outage, in_axes=(0, None))(
            connected_branches_to_outage, disconnections
        )

    lfs_list, sucess_list = jax.vmap(perform_outage_single_busbar, in_axes=(0, 0, 0, None, None, None, None, None, None))(
        connected_branches_to_outage,
        injection_deltap_to_outage,
        node_index_busbar,
        ptdf,
        nodal_injections,
        from_nodes,
        to_nodes,
        n_0_flows,
        branches_monitored,
    )
    return lfs_list, sucess_list


def perform_non_rel_bb_outages(
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    non_rel_bb_outage_data: NonRelBBOutageData,
    branches_monitored: Int[Array, " n_branches_monitored"],
    disconnections: Int[Array, " n_disconnections"] = None,
) -> tuple[Float[Array, " n_bb_outages n_timesteps n_branches_monitored"], Bool[Array, " n_bb_outages"]]:
    """
    Perform non-rel busbar outages and compute the resulting line flows.

    Parameters
    ----------
    n_0_flows : Float[Array, "n_timesteps n_branches"]
        Initial load flows for each branch over timesteps.
    ptdf : Float[Array, "n_branches n_nodes"]
        Power Transfer Distribution Factors (PTDF) matrix.
    nodal_injections : Float[Array, "n_timesteps n_nodes"]
        Nodal injection values for each timestep.
    from_node : Int[Array, "n_branches"]
        Array indicating the starting node of each branch.
    to_node : Int[Array, "n_branches"]
        Array indicating the ending node of each branch.
    non_rel_bb_outage_data : NonRelBBOutageData
        Data structure containing information about non-reliable busbar outages,
        including branch outages, delta injections, and nodal indices.
    branches_monitored : Int[Array, "n_branches_monitored"]
        Array of branch indices that are monitored during the outage.
    disconnections : Int[Array, "n_disconnections"], optional
        Array of disconnection actions which were performed before the busbar outage as part of
        topological actions. If provided, branches that are already outaged by these disconnections
        will not be outaged again in the busbar outage.

    Returns
    -------
    Float[Array, "n_bb_outages n_timesteps n_branches_monitored"]
        Line flows for each busbar outage, timestep, and branch. The length of the list
        corresponds to the number of busbar outages.
    success : Bool[Array, " n_bb_outages"]
        Boolean indicating whether the outage calculations were successful.

    Notes
    -----
    This function calculates the impact of non-rel busbar outages on the
    power grid by modifying nodal injections and branch flows based on the
    provided outage data. It uses the `perform_outage_multi_busbars` function
    to compute the line flows and success status.
    """
    connected_branches_to_outage = non_rel_bb_outage_data.branch_outages
    injection_deltap_to_outage = non_rel_bb_outage_data.deltap
    node_index_busbar = non_rel_bb_outage_data.nodal_indices

    lfs, success = perform_outage_multi_busbars(
        connected_branches_to_outage,
        injection_deltap_to_outage,
        node_index_busbar,
        ptdf,
        nodal_injections,
        from_node,
        to_node,
        branches_monitored,
        n_0_flows=n_0_flows,
        disconnections=disconnections,
    )
    return lfs, success


def remove_articulation_nodes_from_bb_outage(
    rel_bb_outage_data: RelBBOutageData, branch_action_indices: Int[Array, " n_rel_subs"]
) -> tuple[
    Int[Array, " n_rel_subs max_n_physical_bb_per_sub max_branches_per_sub"],
    Int[Array, " n_rel_subs max_n_physical_bb_per_sub"],
    Float[Array, " n_rel_subs max_n_physical_bb_per_sub n_timesteps"],
]:
    """
    Remove critical busbars from outage data.

    Parameters
    ----------
    rel_bb_outage_data : RelBBOutageData
        The outage data containing information about busbars, branches, and nodal indices.
    branch_action_indices : Int[Array, "n_rel_subs"]
        Indices of the branche_actions.

    Returns
    -------
    branch_outages : Int[Array, "n_rel_subs max_n_physical_bb_per_sub max_branches_per_sub"]
        Updated branch outages with critical busbars removed.
    nodal_indices : Int[Array, "n_rel_subs max_n_physical_bb_per_sub"]
        Updated nodal indices with critical busbars removed.
    deltap_outages : Float[Array, "n_rel_subs max_n_physical_bb_per_sub n_timesteps"]
        Updated delta power outages with critical busbars removed.
    """
    articulation_node_mask: Bool[Array, " n_rel_subs n_max_bb_to_outage_per_sub"] = (
        rel_bb_outage_data.articulation_node_mask.at[branch_action_indices].get()
    )
    max_n_branches_per_sub = rel_bb_outage_data.branch_outage_set.shape[2]

    branch_outages = rel_bb_outage_data.branch_outage_set.at[branch_action_indices].get()
    branch_outages = jnp.where(
        jnp.repeat(articulation_node_mask[..., None], max_n_branches_per_sub, 2), int_max(), branch_outages
    )

    nodal_indices = rel_bb_outage_data.nodal_indices.at[branch_action_indices].get()
    nodal_indices = jnp.where(articulation_node_mask, -1, nodal_indices)

    deltap_outages = rel_bb_outage_data.deltap_set.at[branch_action_indices].get()
    deltap_outages = jnp.where(articulation_node_mask[..., None], 0.0, deltap_outages)

    return branch_outages, nodal_indices, deltap_outages


def filter_already_outaged_branches_single_outage(
    branch_outages: Int[Array, " max_branches_per_sub"], disconnections: Int[Array, " n_disconnections"]
) -> Int[Array, " max_branches_per_sub"]:
    """Filter out branches that are already outaged by disconnection actions.

    If a branch is already outaged by the disconnection action, then we need to
    set the branch to int_max() in the branch_outages array.
    This will make sure that the branch is not outaged again and the loadflows
    are computed correctly.

    Parameters
    ----------
    branch_outages : Int[Array, " max_branches_per_sub"]
        Array of branch indices to be outaged.
    disconnections : Int[Array, "n_disconnections"]
        Array of disconnection actions which were performed before the busbar outage as part of
        topological actions.

    Returns
    -------
    Int[Array, " max_branches_per_sub"]
        Filtered array of branch indices to be outaged.
    """
    comparison_matrix = branch_outages[:, None] == disconnections[None, :]
    filtered_branches = jnp.where(jnp.any(comparison_matrix, axis=1), int_max(), branch_outages)

    # We need to sort to ensure that all the int_max is pushed to the end of the array
    filtered_branches = jnp.sort(filtered_branches)
    return filtered_branches


def perform_rel_bb_outage_single_topo(
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    action_indices: Int[Array, " n_rel_subs"],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
    disconnections: Int[Array, " n_disconnections"] = None,
) -> tuple[Float[Array, " n_bb_outages n_timesteps n_branches_monitored"], Bool[Array, " n_bb_outages"]]:
    """Perform a relevant busbar outage for a single topology.

    This function calculates the impact of a relevant busbar outage on the power grid
    by updating nodal injections and loadlfows.

    Parameters
    ----------
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        The initial loadflows over all timesteps.
    action_indices : Int[Array, " n_rel_subs"]
        Indices of the actions to be performed for the relative busbar outage.
    ptdf : Float[Array, " n_branches n_nodes"]
        Power Transfer Distribution Factor (PTDF) matrix.
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        Nodal injection values for each timestep and node.
    from_nodes : Int[Array, " n_branches"]
        Array of "from" nodes for each branch.
    to_nodes : Int[Array, " n_branches"]
        Array of "to" nodes for each branch.
    action_set : ActionSet
        ActionSet object containing information about branch actions and relative busbar outage data.
    branches_monitored : Int[Array, " n_branches_monitored"]
        Indices of branches to be monitored during the outage.
    disconnections : Int[Array, "n_disconnections"], optional
        Array of disconnection actions which were performed before the busbar outage as part of
        topological actions.

    Returns
    -------
    lfs_list : Float[Array, " n_bb_outages n_timesteps n_branches_monitored"]
        Array of load flow solutions for each busbar outages, timestep and branch.
    success : list[Bool[Array, " "]]
        Array indicating the success or failure of the outage calculations for each busbar outage.

    Raises
    ------
    AssertionError
        If the branch outage set is None or if there is a mismatch between the branch action set
        and the branch outage set.
    """
    branch_action_set = action_set.branch_actions

    assert action_set.rel_bb_outage_data.branch_outage_set is not None, (
        "Branch outage set is None in dynamic information. Perform the outage calculation first."
    )
    assert branch_action_set.shape[0] == action_set.rel_bb_outage_data.branch_outage_set.shape[0], (
        "Mismatch in branch action set and branch outage set."
    )

    branch_outages, nodal_indices_outages, deltap_outages = remove_articulation_nodes_from_bb_outage(
        action_set.rel_bb_outage_data, action_indices
    )
    # Note: branch_indices with value -1 or int_max are automatically ignored in the  build_modf_matrix  function
    branch_outages: Int[Array, " n_rel_subs*max_n_physical_bb_per_sub max_branches_per_sub"] = jnp.concatenate(
        branch_outages, axis=0
    )
    deltap_outages: Float[Array, " n_rel_subs*max_n_physical_bb_per_sub n_timesteps"] = jnp.concatenate(
        deltap_outages, axis=0
    )
    nodal_indices_outages: Int[Array, " n_rel_subs*max_n_physical_bb_per_sub "] = jnp.concatenate(
        nodal_indices_outages, axis=0
    )

    lfs_list, success = perform_outage_multi_busbars(
        branch_outages,
        deltap_outages,
        nodal_indices_outages,
        ptdf,
        nodal_injections,
        from_nodes,
        to_nodes,
        branches_monitored=branches_monitored,
        n_0_flows=n_0_flows,
        disconnections=disconnections,
    )
    return lfs_list, success


def perform_rel_bb_outage_for_unsplit_grid(
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> tuple[Float[Array, " n_bb_outages n_timesteps n_branches_monitored"], Bool[Array, " n_bb_outages"]]:
    """
    Perform relative busbar outages for an unsplit grid.

    This function calculates the loadflows (LFS) and success flags
    for busbar outages in an unsplit grid configuration.

    Parameters
    ----------
    n_0_flows : Float[Array, "n_timesteps n_branches"]
        The initial loadflows over all timesteps.
    ptdf : Float[Array, "n_branches n_nodes"]
        Power Transfer Distribution Factor (PTDF) matrix representing the sensitivity
        of branch flows to nodal power injections.
    nodal_injections : Float[Array, "n_timesteps n_nodes"]
        Array of nodal power injections for each timestep.
    from_nodes : Int[Array, "n_branches"]
        Array indicating the "from" nodes for each branch.
    to_nodes : Int[Array, "n_branches"]
        Array indicating the "to" nodes for each branch.
    action_set : ActionSet
        Set of actions representing possible busbar outages.
    branches_monitored : Int[Array, "n_branches_monitored"]
        Array of branch indices that are monitored for outages.

    Returns
    -------
    lfs_list : Float[Array, "n_bb_outages n_timesteps n_branches_monitored"]
        Line flow sensitivities for each busbar outage, timestep, and monitored branch.
    success : Bool[Array, "n_bb_outages"]
        Boolean array indicating whether each busbar outage was successfully simulated.
    """
    # Get a list of action_indices corresponding to unplit action
    # for each relevant substation.
    action_indices = pad_action_with_unsplit_action_indices(
        action_set,
        jnp.full((1,), int_max(), dtype=int),
    )

    lfs_list, success = perform_rel_bb_outage_single_topo(
        n_0_flows, action_indices, ptdf, nodal_injections, from_nodes, to_nodes, action_set, branches_monitored
    )

    return lfs_list, success


def perform_rel_bb_outage_batched(
    n_0_flows: Float[Array, " batch_size n_timesteps n_branches"],
    action_indices: Int[Array, " batch_size n_rel_subs"],
    ptdf: Float[Array, " batch_size n_branches n_nodes"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_nodes"],
    from_nodes: Int[Array, " batch_size n_branches"],
    to_nodes: Int[Array, " batch_size n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
    disconnections: Int[Array, " batch_size n_disconnections"] = None,
) -> tuple[Float[Array, " batch_size n_bb_outages n_timesteps n_branches"], Bool[Array, " batch_size n_bb_outages"]]:
    """
    Perform relative busbar outages for a batch of topologies.

    Parameters
    ----------
    n_0_flows : Float[Array, " batch_size n_timesteps n_branches"]
        The initial load flows for each batch and timestep.
    action_indices : Bool[Array, "batch_size n_rel_subs"]
        A boolean array indicating the action indices for relative busbar
        outages for each batch.
    ptdf : Float[Array, "batch_size n_branches n_nodes"]
        The Power Transfer Distribution Factors (PTDF) for each batch.
    nodal_injections : Float[Array, "batch_size n_timesteps n_nodes"]
        The nodal injections for each batch and timestep.
    from_nodes : Int[Array, "batch_size n_branches"]
        The indices of the "from" nodes for each branch in the network.
    to_nodes : Int[Array, "batch_size n_branches"]
        The indices of the "to" nodes for each branch in the network.
    action_set : ActionSet
        The set of actions defining the busbar outages.
    branches_monitored : Int[Array, "n_branches_monitored"]
        The indices of the branches being monitored.
    disconnections : Int[Array, "batch_size n_disconnections"], optional
        An array of disconnection actions which were performed before the busbar outage as part of
        topological actions. If provided, branches that are already outaged by these disconnections
        will not be outaged again in the busbar outage.

    Returns
    -------
    Float[Array, " batch_size n_bb_outages n_timesteps n_branches"]
        The load flows for each topology, busbar outage, timestep, and monitored branch.
    Bool[Array, " batch_size n_bb_outages"]
        A boolean matrix indicating whether each busbar outage was successfully simulated.
    """
    batched_lfs, batch_success = jax.vmap(perform_rel_bb_outage_single_topo, in_axes=(0, 0, 0, 0, 0, 0, None, None, 0))(
        n_0_flows,
        action_indices,
        ptdf,
        nodal_injections,
        from_nodes,
        to_nodes,
        action_set,
        branches_monitored,
        disconnections,
    )
    return batched_lfs, batch_success


def get_busbar_outage_penalty_batched(
    n_0_flows: Float[Array, " batch_size n_timesteps n_branches"],
    action_indices: Int[Array, " batch_size n_rel_subs"],
    ptdf: Float[Array, " batch_size n_branches n_nodes"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_nodes"],
    from_nodes: Int[Array, " batch_size n_branches"],
    to_nodes: Int[Array, " batch_size n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
    unsplit_bb_outage_analysis: BBOutageBaselineAnalysis,
    lower_bound: Optional[Float[Array, ""]] = None,
) -> tuple[Float[Array, " batch_size"], Float[Array, " batch_size"], Int[Array, " batch_size"]]:
    """
    Compute the batched busbar outage penalty for a given set of actions and system parameters.

    This function evaluates the penalty associated with busbar outages by performing
    a relative busbar outage analysis in a batched manner.

    Parameters
    ----------
    n_0_flows : Float[Array, " batch_size n_timesteps n_branches"]
        The initial load flows for each timestep.
    action_indices : Int[Array, "batch_size n_rel_subs"]
        Indices of the actions to be applied for each batch.
    ptdf : Float[Array, "batch_size n_branches n_nodes"]
        Power Transfer Distribution Factors (PTDF) for each batch.
    nodal_injections : Float[Array, "batch_size n_timesteps n_nodes"]
        Nodal power injections for each batch and timestep.
    from_nodes : Int[Array, "batch_size n_branches"]
        Indices of the "from" nodes for each branch in each batch.
    to_nodes : Int[Array, "batch_size n_branches"]
        Indices of the "to" nodes for each branch in each batch.
    action_set : ActionSet
        The set of actions that can be applied to the system.
    branches_monitored : Int[Array, "n_branches_monitored"]
        Indices of the branches that are monitored for outages.
    unsplit_bb_outage_analysis : BBOutageBaselineAnalysis
        Precomputed baseline analysis for busbar outages.
    lower_bound : Optional[Float[Array, ""]], defaults to None
        A scalar value that sets the lower bound for the busbar outage penalty.

    Returns
    -------
    Float[Array, "batch_size"]
        The computed busbar outage penalties for each batch.
    Float[Array, "batch_size"]
        The overload energy due to busbar outages for each batch.
    Int[Array, "batch_size"]
        The total number of grid splits caused by busbar outages for each batch.

    See Also
    --------
    perform_rel_bb_outage_batched : Performs the relevant busbar outage for a batch of topologies.
    get_busbar_outage_penalty : Computes the busbar outage penalty for a single topology.
    """
    lfs, success = perform_rel_bb_outage_batched(
        n_0_flows,
        action_indices,
        ptdf,
        nodal_injections,
        from_nodes,
        to_nodes,
        action_set,
        branches_monitored,
    )

    batch_penalty, batch_overload, batch_n_grid_splits = jax.vmap(get_busbar_outage_penalty, in_axes=(None, 0, 0, None))(
        unsplit_bb_outage_analysis, lfs, success, lower_bound
    )

    return batch_penalty, batch_overload, batch_n_grid_splits


def get_busbar_outage_penalty(
    baseline: BBOutageBaselineAnalysis,
    lfs: Float[Array, " n_bb_outages n_timesteps n_branches_monitored"],
    success: Bool[Array, " n_bb_outages"],
    lower_bound: Optional[Float[Array, ""]] = None,
) -> tuple[Float[Array, ""], Float[Array, ""], Int[Array, ""]]:
    """
    Compute the penalty for busbar outages for a single topology.

    This method is called for each topology in the batch.

    Parameters
    ----------
    baseline : BBOutageBaselineAnalysis
        The baseline analysis object containing reference values for success count,
        overload energy, maximum MW flow, and overload weight.
    lfs : Float[Array, " n_bb_outages n_timesteps n_branches_monitored"]
        A 3D array representing the load flow solution (LFS) for each busbar outage,
        across multiple timesteps and monitored branches.
    success : Bool[Array, " n_bb_outages"]
        A 1D boolean array indicating whether each busbar outage scenario was successful.
    lower_bound : Optional[Float[Array, ""]], defaults to None
        A scalar value that sets the lower bound for the busbar outage penalty.
        If not provided, it defaults to None and the penalty will not be clipped to
        a particular lower bound.

    Returns
    -------
    Float[Array, ""]
        The penalty value is non-negative and is computed as the sum of the overload
        difference and the success difference, scaled by the more_splits_penalty.
    Float[Array, ""]
        The overload energy due to busbar outage
    Int[Array, ""]
        The total number of grid splits caused due to busbar outages.
    """
    overload = get_overload_energy_n_1_matrix(
        n_1_matrix=jnp.transpose(lfs, (1, 0, 2)),
        max_mw_flow=baseline.max_mw_flow,
        overload_weight=baseline.overload_weight,
        aggregate_strategy="nanmax",
    )
    success_diff = jnp.clip(baseline.success_count - jnp.sum(success), lower_bound, None)
    overload_diff = jnp.clip(overload - baseline.overload, lower_bound, None)
    penalty = overload_diff + baseline.more_splits_penalty * success_diff
    return penalty, overload, jnp.sum(~success)
