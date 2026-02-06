# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The contingency analysis module, holding the methods n_0_analysis and n_1_analysis and helpers"""

from __future__ import annotations

import jax
from beartype.typing import Optional
from jax import numpy as jnp
from jax_dataclasses import Static, pytree_dataclass
from jaxtyping import Array, Float, Int
from toop_engine_dc_solver.jax.busbar_outage import perform_non_rel_bb_outages, perform_rel_bb_outage_single_topo
from toop_engine_dc_solver.jax.multi_outages import MODFMatrix, apply_modf_matrices
from toop_engine_dc_solver.jax.topology_computations import pad_action_with_unsplit_action_indices
from toop_engine_dc_solver.jax.types import ActionSet, NonRelBBOutageData


@pytree_dataclass
class UnBatchedContingencyAnalysisParams:
    """Parameters for the contingency analysis which do not have a batch axis."""

    branches_to_fail: Int[Array, " n_branch_failures"]
    """
    The branches to fail in the contingency analysis."""

    injection_outage_deltap: Float[Array, " n_timesteps n_inj_failures"]
    """
    The effect of removing the injections in MW real power for every injection outage.
    """
    branches_monitored: Int[Array, " n_branches_monitored"]
    """
    The branches that are monitored in the contingency analysis.
    """
    enable_bb_outages: Static[bool]
    """
    Whether to enable busbar outages in the contingency analysis. If True, the
    contingency analysis will include the effects of busbar outages.
    """
    action_set: ActionSet = None
    """
    The action set of the topology."""

    non_rel_bb_outage_data: NonRelBBOutageData = None
    """
    The non-rel busbar outage data for the contingency analysis. The RelBBOutageData
    can be found in the action_set.
    """


@pytree_dataclass
class BatchedContingencyAnalysisParams:
    """Batched parameters for the contingency analysis, containing matrices and other data.

    Note that the batched parameters is per-topology while the unbatched parameters is per-grid
    and hence the unbatched parameters can be broadcasted.
    """

    lodf: Float[Array, " ... n_failures n_branches_monitored"]
    """
    The Line Outage Distribution Factors (LODF) matrix, representing the impact of line outages on monitored branches.
    """
    ptdf: Float[Array, " ... n_branches n_bus"]
    """
    The Power Transfer Distribution Factors (PTDF) matrix, representing the sensitivity of branch flows to nodal injections.
    """
    modf: list[MODFMatrix]
    """
    Multi-Outage Distribution Factor (MODF) matrices for handling multiple simultaneous outages.
    """
    nodal_injections: Float[Array, " ... n_timesteps n_bus"]
    """
    Nodal power injections for each timestep, representing the power injected at each node.
    """
    n_0_flow: Float[Array, " ... n_timesteps n_branches"]
    """
    Base case branch flows for each timestep, representing the initial power flow before any outages.
    """
    injection_outage_node: Int[Array, " ... n_inj_failures"]
    """
    Indices of nodes where injection outages occur.
    """
    action_indices: Optional[Int[Array, " ... n_split_subs"]] = None
    """
    Indices of the topolgical actions
    """
    from_nodes: Optional[Int[Array, " ... n_branches"]] = None
    """
    Indices of "from" nodes for each branch, used to identify the starting point of each branch.
    """
    to_nodes: Optional[Int[Array, " ... n_branches"]] = None
    """
    Indices of "to" nodes for each branch, used to identify the endpoint of each branch.
    """
    disconnections: Optional[Int[Array, " ... n_disconnections"]] = None
    """
    Indices of branches that are disconnected as part of topological actions, used to apply specific disconnection actions.
    """


def contingency_analysis_matrix(
    unbatched_params: UnBatchedContingencyAnalysisParams,
    batched_params: BatchedContingencyAnalysisParams,
) -> Float[Array, " n_timesteps n_branch_failures+n_multi_failures+n_inj_failures+n_bb_outages n_branches_monitored"]:
    """
    Perform a n-0 and n-1 analysis and returns the full n-0 loads and n-1 matrix.

    Whether to include busbar outages is determined by the `params.enable_bb_outages` parameter.

    Parameters
    ----------
    unbatched_params : UnBatchedContingencyAnalysisParams
        Parameters for contingency analysis, including monitored branches and failure configurations.
    batched_params : BatchedContingencyAnalysisParams
        Batched parameters for contingency analysis, including LODF, PTDF, MODF matrices, nodal injections,
        and base case flows.

    Returns
    -------
    Float[Array, "n_timesteps n_branch_failures+n_multi_failures+n_inj_failures+n_bb_outages n_branches_monitored"]
        Contingency analysis matrix containing the impact of branch failures, multi-outages, injection outages,
        and optionally busbar outages on monitored branches.

    Notes
    -----
    - The function computes the N-1 contingency matrix for branch failures, multi-outages and injection outages.
    - If busbar outages are enabled (`params.enable_bb_outages`), their impact is also included.
    - The results are concatenated along the failure dimension to form the final contingency analysis matrix.
    """
    n_0_flow_monitors = batched_params.n_0_flow.at[:, unbatched_params.branches_monitored].get(
        mode="fill", fill_value=jnp.nan
    )

    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches_monitored"] = calc_n_1_matrix(
        lodf=batched_params.lodf,
        branches_to_outage=unbatched_params.branches_to_fail,
        n_0_flow=batched_params.n_0_flow,
        n_0_flow_monitors=n_0_flow_monitors,
    )

    multi_n_1_matrix = apply_modf_matrices(
        modf_matrices=batched_params.modf,
        n_0_flow=batched_params.n_0_flow,
        branches_monitored=unbatched_params.branches_monitored,
    )

    inj_n_1_matrix = calc_injection_outages(
        ptdf=batched_params.ptdf,
        n_0_flow=batched_params.n_0_flow,
        injection_outage_deltap=unbatched_params.injection_outage_deltap,
        injection_outage_node=batched_params.injection_outage_node,
        branches_monitored=unbatched_params.branches_monitored,
    )

    if unbatched_params.enable_bb_outages:
        bb_outage_n_1_matrix = calc_bb_outage_contingency(
            n_0_flows=batched_params.n_0_flow,
            ptdf=batched_params.ptdf,
            nodal_injections=batched_params.nodal_injections,
            action_indices=batched_params.action_indices,
            from_nodes=batched_params.from_nodes,
            to_nodes=batched_params.to_nodes,
            action_set=unbatched_params.action_set,
            branches_monitored=unbatched_params.branches_monitored,
            non_rel_bb_outage_data=unbatched_params.non_rel_bb_outage_data,
            disconnections=batched_params.disconnections,
        )
    else:
        bb_outage_n_1_matrix = jnp.zeros((n_1_matrix.shape[0], 0, n_1_matrix.shape[2]), dtype=n_1_matrix.dtype)

    n_1_matrix = jnp.concatenate([n_1_matrix, multi_n_1_matrix, inj_n_1_matrix, bb_outage_n_1_matrix], axis=1)

    return n_1_matrix


def calc_bb_outage_contingency(
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    ptdf: Float[Array, " n_branches n_bus"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    action_indices: Int[Array, " n_split_subs"],
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
    non_rel_bb_outage_data: NonRelBBOutageData,
    disconnections: Optional[Int[Array, " n_disconnections"]] = None,
) -> Float[
    Array,
    " n_timesteps n_bb_outages n_branches_monitored",
]:
    """
    Calculate the busbar outage contingency matrix for both relevant and non-relevant bb outages.

    Parameters
    ----------
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        The initial load flows for each timestep.
    ptdf : Float[Array, "n_branches n_bus"]
        Power Transfer Distribution Factors (PTDF) matrix.
    nodal_injections : Float[Array, "n_timesteps n_bus"]
        Nodal injection values for each timestep.
    action_indices : Int[Array, "n_split_subs"]
        Indices of the actions to be applied .
    from_nodes : Int[Array, "n_branches"]
        Array of "from" nodes for each branch.
    to_nodes : Int[Array, "n_branches"]
        Array of "to" nodes for each branch.
    action_set : ActionSet
        Set of actions defining the topology changes.
    branches_monitored : Int[Array, "n_branches_monitored"]
        Indices of branches to be monitored for outages.
    non_rel_bb_outage_data : NonRelBBOutageData
        Data related to non-relevant branch outages.
    disconnections : Optional[Int[Array, " n_disconnections"]], optional
        Disconnection action to be considered, by default None.

    Returns
    -------
    Float[Array, "n_timesteps n_bb_outages n_branches_monitored"]
        The branch outage flows for all timesteps, branch outages, and monitored branches.
    """
    padded_action_indices: Int[Array, " n_rel_subs"] = pad_action_with_unsplit_action_indices(action_set, action_indices)
    bb_outage_flows, _success_rel_bbs = perform_rel_bb_outage_single_topo(
        n_0_flows=n_0_flows,
        action_indices=padded_action_indices,
        ptdf=ptdf,
        nodal_injections=nodal_injections,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        action_set=action_set,
        branches_monitored=branches_monitored,
        disconnections=disconnections,
    )
    bb_outage_flows = jnp.transpose(bb_outage_flows, (1, 0, 2))

    if non_rel_bb_outage_data.branch_outages.shape[0] > 0:
        bb_outage_flows_non_rel_bbs, _success_non_rel_bbs = perform_non_rel_bb_outages(
            n_0_flows=n_0_flows,
            ptdf=ptdf,
            nodal_injections=nodal_injections,
            from_node=from_nodes,
            to_node=to_nodes,
            branches_monitored=branches_monitored,
            non_rel_bb_outage_data=non_rel_bb_outage_data,
            disconnections=disconnections,
        )
        bb_outage_flows_non_rel_bbs = jnp.transpose(bb_outage_flows_non_rel_bbs, (1, 0, 2))
        bb_outage_flows = jnp.concatenate([bb_outage_flows, bb_outage_flows_non_rel_bbs], axis=1)

    return bb_outage_flows


def calc_n_1_matrix(
    lodf: Float[Array, " n_failures n_branches_monitored"],
    branches_to_outage: Int[Array, " n_failures"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    n_0_flow_monitors: Float[Array, " n_timesteps n_branches_monitored"],
) -> Float[Array, " n_timesteps n_failures n_branches_monitored"]:
    """Compute the loading after all n-1 cases

    Parameters
    ----------
    lodf : Float[Array, " n_failures n_branches_monitored"]
        The LODF matrix as obtained by calc_lodf_matrix
    branches_to_outage : Int[Array, " n_failures"]
        The list of N-1 failure cases
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The n-0 flows as obtained by n_0_analysis
    n_0_flow_monitors : Float[Array, " n_timesteps n_branches_monitored"]
        The n-0 flows of monitored branches as obtained by n_0_analysis


    Returns
    -------
    Float[Array, " n_timesteps n_failures n_branches_monitored"]
        The loading after all n-1 cases
    """
    delta_flow: Float[Array, " n_timesteps n_failures n_branches_monitored"] = jnp.einsum(
        "ij,ti -> tij", lodf, n_0_flow[:, branches_to_outage]
    )
    flow_n_1 = n_0_flow_monitors[:, None, :] + delta_flow

    return flow_n_1


def calc_injection_outages(
    ptdf: Float[Array, " n_branches n_bus"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    injection_outage_deltap: Float[Array, " n_timesteps n_inj_failures"],
    injection_outage_node: Int[Array, " n_inj_failures"],
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> Float[Array, " n_timesteps n_inj_failures n_branches"]:
    """Compute the post-outage flow after taking out a multiple injections.

    Just vmaps over calc_injection_outage

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The n-0 flows
    injection_outage_deltap : Float[Array, " n_timesteps n_inj_failures"]
        The effect of removing the injections in MW real power.
    injection_outage_node : Int[Array, " n_inj_failures"]
        The nodes where the delta p is to be applied.
    branches_monitored : Int[Array, " n_branches_monitored"]
        Which branches are monitored (static argument)

    Returns
    -------
    Float[Array, " n_timesteps n_inj_failures n_branches"]
        The post-outage flows
    """
    return jax.vmap(
        lambda delta_p, node: calc_injection_outage(ptdf, n_0_flow, delta_p, node, branches_monitored),
        in_axes=(1, 0),
        out_axes=1,
    )(injection_outage_deltap, injection_outage_node)


def calc_injection_outage(
    ptdf: Float[Array, " n_branches n_bus"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    delta_p: Float[Array, " n_timesteps"],
    outage_node: Int[Array, " "],
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> Float[Array, " n_timesteps n_branches_monitored"]:
    """Compute the post-outage flow after taking out a single injection.

    The effect of removing that injection should be represented through a delta_p value.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The n-0 flows
    delta_p : Float[Array, " n_timesteps"]
        The effect of removing the injection in MW real power.
    outage_node : Int[Array, " "]
        The node where the delta p is to be applied.
    branches_monitored : Int[Array, " n_branches_monitored"]
        Which branches are monitored (static argument)

    Returns
    -------
    Float[Array, " n_timesteps n_branches_monitored"]
        The post-outage flows
    """
    # The n_0_flow for a branch is ptdf @ nodal_injections. One of the nodal injections is changed
    # now, hence we only need to compute the PTDF for the branch in question.
    delta_flow = jnp.einsum(
        "i,t->ti",
        ptdf.at[branches_monitored, outage_node].get(mode="fill", fill_value=0),
        delta_p,
    )
    return n_0_flow.at[:, branches_monitored].get(mode="fill", fill_value=jnp.nan) + delta_flow
