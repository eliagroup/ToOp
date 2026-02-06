# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Perform N-2 analysis.

1. Calculate the N-2 sequentially for each topology in the pre-split topology
2. Calculate the N-2 sequentially for each topology in the post-split topology
3. Return the difference between the overload energies of the
pre-split and post-split topologies
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Optional, Protocol
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped
from toop_engine_dc_solver.jax.aggregate_results import get_overload_energy_n_1_matrix
from toop_engine_dc_solver.jax.contingency_analysis import calc_n_1_matrix
from toop_engine_dc_solver.jax.disconnections import apply_single_disconnection_lodf
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix, get_failure_cases_to_zero
from toop_engine_dc_solver.jax.types import DynamicInformation, N2BaselineAnalysis


class N2AggregateProtocol(Protocol):
    """Protocol for the N-2 aggregation function.

    A protocol for an aggregation routine that takes in the L1 case and the N-1 and N-2 loadflows
    and returns some metric. This should work for a single topology and be vmappable.
    """

    def __call__(
        self,
        l1_case: Int[Array, ""],
        n_2: Float[Array, " n_timesteps n_l2_outages n_branches_monitored"],
        l1_success: Bool[Array, ""],
        l2_success: Bool[Array, " n_l2_outages"],
    ) -> PyTree:
        """Run Protocol method.

        This will be vmapped over all topologies.

        Parameters
        ----------
        l1_case : Int[Array, ""]
            The L1 branch that was disconnected, indexes into all branches
        n_2 : Float[Array, " n_timesteps n_l2_outages n_branches_monitored"]
            The N-2 loadflows, which are the main output of the N-2 analysis. This is a N-1 analysis
            after disconnecting the L1 branch. Cases with splits are zeroed out.
        l1_success : Bool[Array, ""]
            Whether the L1 case was successful. If False, the L1 case caused a split in the grid.
            In that case, the N-2 results are completely unuseable.
        l2_success : Bool[Array, " n_l2_outages"]
            Whether the N-2 case was successful. If False, the N-2 case caused a split in the grid.
            The corresponding row in the N-2 matrix will be zeroed out.

        Returns
        -------
        PyTree
            The aggregated data, which can be anything jax-compatible.
        """


def run_single_l1_case(
    l1_branch: Int[Array, ""],
    topological_disconnections: Int[Array, " n_disconnections"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    ptdf: Float[Array, " n_branches n_nodes"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    l2_outages: Int[Array, " n_l2_outages"],
    branches_monitored: Int[Array, " n_branches_monitored"],
    aggregator: N2AggregateProtocol,
) -> PyTree:
    """Compute the aggregated results for a single L1 case with the corresponding L2 analysis

    Parameters
    ----------
    l1_branch : Int[Array, ""]
        The L1 branch that was disconnected, indexes into all branches
    topological_disconnections : Int[Array, " n_disconnections"]
        The branches that were disconnected as part of topological actions, used to zero out the
        N-2 results. If no disconnections were applied, pass jnp.array([], dtype=int)
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        The updated nodal injection vector with the reassigned injections already included.
    ptdf : Float[Array, " n_branches n_nodes"]
        The updated PTDF matrix with splits and disconnections applied.
    from_node : Int[Array, " n_branches"]
        The from node of each branch.
    to_node : Int[Array, " n_branches"]
        The to node of each branch.
    l2_outages : Int[Array, " n_l2_outages"]
        The branches to disconnect in the N-2 analysis.
    branches_monitored : Int[Array, " n_branches_monitored"]
        The branches to monitor
    aggregator : N2AggregateProtocol
        The aggregation function to use

    Returns
    -------
    PyTree
        The aggregated data for the L1 case, will be the output of the aggregator function
    """
    ptdf_local, l1_success = apply_single_disconnection_lodf(
        disconnection=l1_branch,
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
    )

    # Use the l1 outage ptdf to calculate the LODF matrix for the n-1 cases
    # We do not need to update from/to nodes as we will zero out the failure case with that
    # branch anyway
    lodf, lodf_success = calc_lodf_matrix(
        branches_to_outage=l2_outages,
        ptdf=ptdf_local,
        from_node=from_node,
        to_node=to_node,
        branches_monitored=branches_monitored,
    )

    # Calculate the padding for the branches which are already disconnected
    # as part of topological actions
    failure_cases_to_zero = None
    failure_cases_to_zero = get_failure_cases_to_zero(
        jnp.concatenate([topological_disconnections, jnp.array([l1_branch])]),
        l2_outages,
    )

    # We don't split the N-0 computation into static/dynamic parts because we have no
    # injection bruteforcing here.
    n_1_all = jnp.einsum("bn, tn -> tb", ptdf_local, nodal_injections)
    n_1 = n_1_all[:, branches_monitored]

    n_2 = calc_n_1_matrix(
        lodf=lodf,
        branches_to_outage=l2_outages,
        n_0_flow=n_1_all,
        n_0_flow_monitors=n_1,
    )

    n_2 = jnp.where(failure_cases_to_zero[None, :, None] | ~lodf_success[None, :, None], 0, n_2)
    lodf_success = jnp.where(failure_cases_to_zero, True, lodf_success)

    return aggregator(l1_branch, n_2, l1_success, lodf_success)


def n_2_analysis(
    l1_outages: Int[Array, " n_l1_outages"],
    topological_disconnections: Optional[Int[Array, " n_disconnections"]],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    ptdf: Float[Array, " n_branches n_nodes"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    l2_outages: Int[Array, " n_l2_outages"],
    branches_monitored: Int[Array, " n_branches_monitored"],
    aggregator: N2AggregateProtocol,
) -> tuple[Shaped[PyTree, " n_l1_outages ..."], Bool[Array, " n_l1_outages"]]:
    """
    Calculate the N-2 outages sequentially for each L1 case

    Parameters
    ----------
    l1_outages : Int[Array, " n_l1_outages"]
        Array of level 1 disconnections (N-1 outages). The level 1 disconnections
        are the branches that should be discon nected first. These branches are the ones
        connected to the split substation. If an out-of-bounds entry is given or the case was already
        disconnected as part of topological actions, the case will be ignored. Ignores are returned.
    topological_disconnections : Int[Array, " n_disconnections"]
        Array of topological disconnections. These are the branches that are disconnected
        as part of topological actions
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        The updated nodal injection vector with the reassigned injections already included.
    ptdf : Float[Array, " n_branches n_nodes"]
        The updated PTDF matrix with splits and disconnections applied.
    from_node : Int[Array, " n_branches"]
        The from node of each branch.
    to_node : Int[Array, " n_branches"]
        The to node of each branch.
    l2_outages : Int[Array, " n_l2_outages"]
        The branches to disconnect in the N-2 analysis.
    branches_monitored : Int[Array, " n_branches_monitored"]
        The branches to monitor
    aggregator : N2AggregateProtocol
        The aggregation function to use

    Returns
    -------
    Shaped[PyTree, " n_l1_outages ..."]
        The aggregated data for each N-2 case, will be the output of the aggregator function with one
        leading dimension for each L1 disconnection. For ignored cases, the output will be zero.
    Bool[Array, " n_l1_outages"]
        Array of ignore flags for each cases. If true, the case was ignored either because it was
        out of bounds or because it was already present in the disconnections.
    """
    n_timesteps = nodal_injections.shape[0]
    n_branches = ptdf.shape[0]
    n_branches_monitored = branches_monitored.shape[0]
    n_l1_outages = l1_outages.shape[0]
    n_l2_outages = l2_outages.shape[0]

    if topological_disconnections is None:
        topological_disconnections = jnp.array([], dtype=int)
        ignores = jnp.zeros(n_l1_outages, dtype=bool)
    else:
        # Ignore L1 disconnections that were disconnected as part of topological actions
        ignores = jnp.isin(l1_outages, topological_disconnections)

    # Furthermore ignore l1 disconnections which are already out of bounds
    ignores = ignores | (l1_outages < 0) | (l1_outages >= n_branches)
    l1_outages = jnp.where(ignores, jnp.iinfo(jnp.int32).max, l1_outages)

    # Sort the L1 disconnections so we know that once a l1 disconnection is out of bounds (e.g. int
    # max) then we can ignore the rest of the disconnections
    sorting_indices = jnp.argsort(l1_outages)
    l1_outages = l1_outages[sorting_indices]

    # Prepare some storage for the results
    results_dtype = jax.eval_shape(
        aggregator,
        l1_outages[0],
        jnp.zeros((n_timesteps, n_l2_outages, n_branches_monitored), dtype=float),
        jnp.array(False, dtype=bool),
        jnp.zeros(n_l2_outages, dtype=bool),
    )
    buffer = jax.tree_util.tree_map(lambda x: jnp.zeros((n_l1_outages, *x.shape), dtype=x.dtype), results_dtype)

    def body_fun(
        val_tuple: tuple[Int[Array, ""], Shaped[PyTree, " n_l1_outages ..."]],
    ) -> tuple[Int[Array, ""], Shaped[PyTree, " n_l1_outages ..."]]:
        i, buffer = val_tuple
        l1_branch = l1_outages[i]
        storage_index = sorting_indices[i]

        aggregate = run_single_l1_case(
            l1_branch=l1_branch,
            topological_disconnections=topological_disconnections,
            nodal_injections=nodal_injections,
            ptdf=ptdf,
            from_node=from_node,
            to_node=to_node,
            l2_outages=l2_outages,
            branches_monitored=branches_monitored,
            aggregator=aggregator,
        )

        buffer = jax.tree_util.tree_map(lambda buf, new: buf.at[storage_index].set(new), buffer, aggregate)

        return i + 1, buffer

    # We use a while loop instead of a for loop to be able to ignore l1 outages that have been
    # masked away
    _last_iter, buffer = jax.lax.while_loop(
        cond_fun=lambda val_tuple: (l1_outages[val_tuple[0]] < n_branches) & (val_tuple[0] < n_l1_outages),
        body_fun=body_fun,
        init_val=(jnp.array(0, dtype=int), buffer),
    )
    return buffer, ignores


def unsplit_n_2_analysis(
    dynamic_information: DynamicInformation,
    more_splits_penalty: float,
) -> N2BaselineAnalysis:
    """Perform the unsplit N-2 analysis.

    Performs the unsplit N-2 analysis where all branches on all relevant substations are outaged
    to obtain a baseline N-2 overload for each branch.

    This method is not jit compatible, as it is designed to be called once during preprocessing.

    Parameters
    ----------
    dynamic_information : DynamicInformation
        The dynamic information of the grid
    more_splits_penalty : float
        The penalty for additional N-2 cases that could not be computed due to splits in the grid


    Returns
    -------
    N2BaselineAnalysis
        The N-2 baseline analysis results
    """
    int_max = jnp.iinfo(dynamic_information.tot_stat).max
    l1_outages = dynamic_information.tot_stat[dynamic_information.tot_stat < dynamic_information.n_branches]
    l1_outages = jnp.unique(l1_outages.flatten())

    (l1_branches, n_2_overloads, n_2_success_count), ignores = n_2_analysis(
        l1_outages=l1_outages,
        topological_disconnections=None,
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        l2_outages=dynamic_information.branches_to_fail,
        branches_monitored=dynamic_information.branches_monitored,
        aggregator=lambda l1_branch, n_2, l1_success, lodf_success: (
            jnp.where(l1_success, l1_branch, int_max),
            get_overload_energy_n_1_matrix(
                n_1_matrix=n_2,
                max_mw_flow=dynamic_information.branch_limits.max_mw_flow,
                overload_weight=dynamic_information.branch_limits.overload_weight,
            ),
            jnp.sum(lodf_success),
        ),
    )
    assert not jnp.any(ignores), "Some L1 outages were ignored"

    # Some branches might have been blacklisted, filter the results accordingly
    # Also mask tot_stat to not include the blacklisted branches
    # TODO potentially the tot_stat array could be optimized if the max_branches_per_sub changed.
    l1_success = l1_branches != int_max
    blacklist = l1_outages[~l1_success]
    tot_stat_mask = jnp.isin(dynamic_information.tot_stat, blacklist)

    return N2BaselineAnalysis(
        l1_branches=l1_branches[l1_success],
        n_2_overloads=n_2_overloads[l1_success],
        n_2_success_count=n_2_success_count[l1_success],
        tot_stat_blacklisted=jnp.where(tot_stat_mask, int_max, dynamic_information.tot_stat),
        max_mw_flow=dynamic_information.branch_limits.max_mw_flow,
        overload_weight=dynamic_information.branch_limits.overload_weight,
        more_splits_penalty=jnp.array(more_splits_penalty, dtype=float),
    )


def gather_l1_cases(
    has_splits: Bool[Array, " n_subs_split"],
    sub_ids: Int[Array, " n_subs_split"],
    tot_stat: Int[Array, " n_subs_relevant max_branch_per_sub"],
    topological_disconnections: Optional[Int[Array, " n_disconnections"]],
) -> Int[Array, " n_subs_split*max_branch_per_sub"]:
    """Gathers the L1 cases for all relevant substations

    To make this jit compatible, this returns a flattened array with the worst-case shape, but it
    will likely be padded with int_max values.

    Parameters
    ----------
    has_splits : Bool[Array, " n_subs_split"]
        Whether the substation has a split. If False, the L1 cases from that sub will be ignored.
    sub_ids : Int[Array, " n_subs_split"]
        The substation ids. Out of bounds sub ids will be ignored.
    tot_stat : Int[Array, " n_subs_relevant max_branch_per_sub"]
        The L1 cases for each substation
    topological_disconnections : Optional[Int[Array, " n_disconnections"]
        The topological disconnections that were applied, used to ignore the L1 cases that were
        already disconnected

    Returns
    -------
    Int[Array, " n_subs_split*max_branch_per_sub"]
        The L1 cases that were gathered. Due to jit limitations this array is quite large, but it
        will have int_max padding for cases that are not relevant. An order of this can not be
        assumed.
    """
    int_max = jnp.iinfo(tot_stat).max
    tot_stat_local = tot_stat.at[sub_ids].get(mode="fill", fill_value=int_max)

    l1_cases = jnp.where(
        has_splits[:, None],
        tot_stat_local,
        int_max,
    ).flatten()

    if topological_disconnections is not None:
        mask = jnp.isin(l1_cases, topological_disconnections)
        l1_cases = jnp.where(mask, int_max, l1_cases)

    # This does not reduce array size, but potentially increases the amount of padded values
    # if a branch is duplicated along multiple substations at the cost of a unique operation
    # TODO determine if this makes sense to leave in
    l1_cases = jnp.unique_values(l1_cases, size=l1_cases.size, fill_value=int_max)

    return l1_cases


def retrieve_baseline_case(
    l1_branch: Int[Array, ""],
    baseline: N2BaselineAnalysis,
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Retrieve the number of splits and overload for a baseline N-2 case.

    Parameters
    ----------
    l1_branch : Int[Array, ""]
        The L1 branch that was disconnected and for which the baseline results are to be
        retrieved, indexes into all branches
    baseline : N2BaselineAnalysis
        The baseline N-2 analysis results

    Returns
    -------
    Int[Array, ""]
        The number of splits for the L1 case, or int_max if the case was not found
    Float[Array, ""]
        The overload energy for the L1 case, or nan if the case was not found
    """
    int_max = jnp.iinfo(l1_branch.dtype).max
    index = jnp.flatnonzero(baseline.l1_branches == l1_branch, size=1, fill_value=int_max)

    success_count = baseline.n_2_success_count.at[index].get(mode="fill", fill_value=int_max)
    overload = baseline.n_2_overloads.at[index].get(mode="fill", fill_value=jnp.nan)

    return success_count, overload


class SplitAggregator(N2AggregateProtocol):
    """Implementation of the N2AggregateProtocol for an aggregator.

    Implements the N2AggregateProtocol for an aggregator that runs during the split analysis,
    aggregating down the N-2 results to a single scalar penalty. It needs the baseline analysis
    as a static input to compare arbitrary topologies to.
    """

    def __init__(self, baseline: N2BaselineAnalysis) -> None:
        self.baseline = baseline

    def __call__(
        self,
        l1_branch: Int[Array, ""],
        n_2: Float[Array, " n_timesteps n_l2_outages n_branches_monitored"],
        l1_success: Bool[Array, ""],
        lodf_success: Bool[Array, " n_l2_outages"],
    ) -> Float[Array, ""]:
        """Call the aggregate function for a single topology, can be vmapped over all topologies"""
        success_count_ref, overload_ref = retrieve_baseline_case(l1_branch, self.baseline)

        success_count = jnp.sum(lodf_success) * l1_success
        overload = get_overload_energy_n_1_matrix(
            n_1_matrix=n_2,
            max_mw_flow=self.baseline.max_mw_flow,
            overload_weight=self.baseline.overload_weight,
        )

        success_diff = jnp.clip(success_count_ref - success_count, 0, None)
        overload_diff = jnp.clip(overload - overload_ref, 0, None)

        return overload_diff * l1_success + success_diff * self.baseline.more_splits_penalty


def split_n_2_analysis_batched(
    has_splits: Bool[Array, " batch n_subs_relevant"],
    sub_ids: Int[Array, " batch n_subs_relevant"],
    disconnections: Optional[Int[Array, " batch n_disconnections"]],
    nodal_injections: Float[Array, " batch n_timesteps n_nodes"],
    ptdf: Float[Array, " batch n_branches n_nodes"],
    from_node: Int[Array, " batch n_branches"],
    to_node: Int[Array, " batch n_branches"],
    l2_outages: Int[Array, " n_l2_outages"],
    baseline: N2BaselineAnalysis,
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> Float[Array, " batch"]:
    """Wrap around split_n_2_analysis with vmapping for batched inputs

    Parameters
    ----------
    has_splits : Bool[Array, " batch n_subs_relevant"]
        Whether the substation has a split
    sub_ids : Int[Array, " batch n_subs_relevant"]
        The substation ids
    disconnections : Optional[Int[Array, " batch n_disconnections"]]
        The topological disconnections that were applied, used to zero out some results
    nodal_injections : Float[Array, " batch n_timesteps n_nodes"]
        The nodal injection vector with splits already applied
    ptdf : Float[Array, " batch n_branches n_nodes"]
        The PTDF matrix with splits already applied
    from_node : Int[Array, " batch n_branches"]
        The from node of each branch. Updated for changes in the BSDF but not necessarily including
        disconnections
    to_node : Int[Array, " batch n_branches"]
        The to node of each branch. Updated for changes in the BSDF but not necessarily including
        disconnections
    l2_outages : Int[Array, " n_l2_outages"]
        The branches to disconnect in the N-2 analysis (the same for all batch items)
    baseline : N2BaselineAnalysis
        The baseline N-2 analysis results, obtained from the unsplit analysis (the same for all
        batch items, i.e. without leading batch dimension)
    branches_monitored : Int[Array, " n_branches_monitored"]
        The branches to monitor (the same for all batch items)

    Returns
    -------
    Float[Array, " batch"]
        The penalty for the split N-2 analysis for each batch item
    """
    return jax.vmap(
        partial(
            split_n_2_analysis,
            l2_outages=l2_outages,
            baseline=baseline,
            branches_monitored=branches_monitored,
        )
    )(
        has_splits,
        sub_ids,
        disconnections,
        nodal_injections,
        ptdf,
        from_node,
        to_node,
    )


def split_n_2_analysis(
    has_splits: Bool[Array, " n_subs_relevant"],
    sub_ids: Int[Array, " n_subs_relevant"],
    disconnections: Optional[Int[Array, " n_disconnections"]],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    ptdf: Float[Array, " n_branches n_nodes"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    l2_outages: Int[Array, " n_l2_outages"],
    baseline: N2BaselineAnalysis,
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> Float[Array, ""]:
    """Perform the split N-2 analysis.

    Perform the split N-2 analysis, where all branches on all relevant substations are outaged
    and the N-2 overloads are compared to the baseline N-2 overloads.

    from_node and to_node do not need to include the disconnections, as these outage cases will be
    zeroed out anyway so the lodf computation doesn't matter.

    Parameters
    ----------
    has_splits : Bool[Array, " n_subs_relevant"]
        Whether the substation has a split
    sub_ids : Int[Array, " n_subs_relevant"]
        The substation ids
    disconnections : Optional[Int[Array, " n_disconnections"]]
        The topological disconnections that were applied, used to zero out some results
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        The nodal injection vector with splits already applied
    ptdf : Float[Array, " n_branches n_nodes"]
        The PTDF matrix with splits already applied
    from_node : Int[Array, " n_branches"]
        The from node of each branch. Updated for changes in the BSDF but not necessarily including
        disconnections
    to_node : Int[Array, " n_branches"]
        The to node of each branch. Updated for changes in the BSDF but not necessarily including
        disconnections
    l2_outages : Int[Array, " n_l2_outages"]
        The branches to disconnect in the N-2 analysis
    baseline : N2BaselineAnalysis
        The baseline N-2 analysis results, obtained from the unsplit analysis
    branches_monitored : Int[Array, " n_branches_monitored"]
        The branches to monitor

    Returns
    -------
    Float[Array, ""]
        The penalty for the split N-2 analysis
    """
    assert len(has_splits.shape) == 1

    l1_outages = gather_l1_cases(has_splits, sub_ids, baseline.tot_stat_blacklisted, disconnections)

    aggregator = SplitAggregator(baseline)

    penalties, ignores = n_2_analysis(
        l1_outages=l1_outages,
        topological_disconnections=disconnections,
        nodal_injections=nodal_injections,
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        l2_outages=l2_outages,
        branches_monitored=branches_monitored,
        aggregator=aggregator,
    )

    # Not sure why this is necessary
    penalties = jnp.squeeze(penalties)

    # Make sure ignored cases have 0 penalty
    penalty = jnp.sum(penalties * ~ignores)
    return penalty
