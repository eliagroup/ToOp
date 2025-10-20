"""Aggregate routines around result storage.

During the loop, the results need to be stored somehow and especially in injection bruteforce mode,
these stored results need to be updated. Also provides some helper functions around results in general.
"""

from functools import partial

import jax
from beartype.typing import Optional
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped
from toop_engine_dc_solver.jax.types import (
    AggregateOutputProtocol,
    SolverLoadflowResults,
    SparseNMinus0,
    SparseNMinus1,
)
from toop_engine_dc_solver.jax.utils import argmax_top_k


# TODO: PLDR0913 & ARG001 warning -> remove n_sub_relevant
# Too many arguments in function definition (11 > 10)
def prepare_result_storage(  # noqa: PLR0913
    aggregate_output_fn: AggregateOutputProtocol,
    n_timesteps: int,
    n_branches_monitored: int,
    n_failures: int,
    n_splits: int,
    n_disconnections: Optional[int],
    max_branch_per_sub: int,
    max_inj_per_sub: int,
    nminus2: bool,
    bb_outage: bool,
    size: int,
) -> PyTree[Shaped, " size ..."]:
    """Prepare the result storage for the solver

    This invokes aggregate_output_fn with zeros to determine the shape of the output
    and then allocates the resulting pytree but with an additional size dimension.

    Parameters
    ----------
    aggregate_output_fn : AggregateOutputProtocol
        The function that aggregates the results
    n_timesteps:
        The number of timesteps (static_information.n_timesteps)
    n_branches_monitored:
        The number of monitored branches (static_information.n_branches_monitored)
    n_failures:
        The number of failures (static_information.n_outages + static_information.n_multi_outages)
    n_splits:
        The maximum number of splits in the topology
    n_disconnections:
        The maximum number of disconnections in the topology or None if no disconnections are passed
    max_branch_per_sub:
        The maximum number of branches per substation
    max_inj_per_sub:
        The maximum number of injections per substation
    nminus2: bool
        Whether the N-2 feature is enabled
    bb_outage: bool
        Whether the BB outage feature is enabled
    size : int
        The size of the storage

    Returns
    -------
    PyTree[Shaped, " size ..."]
        The allocated storage, where each leaf has size as the first dimension (the other dimensions
        are inferred from the output of aggregate_output_fn). It is filled with zeros by default
    """
    n_0_shape = (
        n_timesteps,
        n_branches_monitored,
    )
    n_1_shape = (
        n_timesteps,
        n_failures,
        n_branches_monitored,
    )
    cross_coupler_flow_shape = (n_splits, n_timesteps)
    topologies_shape = (n_splits, max_branch_per_sub)
    sub_ids_shape = (n_splits,)
    injections_shape = (n_splits, max_inj_per_sub)

    fake_lf_result = SolverLoadflowResults(
        n_0_matrix=jnp.zeros(n_0_shape, dtype=float),
        n_1_matrix=jnp.zeros(n_1_shape, dtype=float),
        cross_coupler_flows=jnp.zeros(cross_coupler_flow_shape, dtype=float),
        branch_action_index=jnp.zeros(n_splits, dtype=int),
        branch_topology=jnp.zeros(topologies_shape, dtype=bool),
        sub_ids=jnp.zeros(sub_ids_shape, dtype=int),
        injection_topology=jnp.zeros(injections_shape, dtype=bool),
        n_2_penalty=jnp.array(0.0, dtype=float) if nminus2 else None,
        disconnections=jnp.zeros(n_disconnections, dtype=bool) if n_disconnections else None,
        bb_outage_penalty=jnp.array(0.0, dtype=float) if bb_outage else None,
    )

    aggregate_pytree = jax.eval_shape(aggregate_output_fn, fake_lf_result)
    result_storage = jax.tree_util.tree_map(lambda x: jnp.zeros((size, *x.shape), dtype=x.dtype), aggregate_pytree)
    return result_storage


def get_best_for_topology(
    topology: Int[Array, " "],
    corresponding_topologies: Int[Array, " n_injections"],
    metric: Float[Array, " n_injections"],
) -> Int[Array, " "]:
    """Get the best injection corresponding to a given topology based on the minimum of the metric

    Returns int-max if topology does not occurs in corresponding_topologies

    Parameters
    ----------
    topology : Int[Array, " "]
        The topology index to look for
    corresponding_topologies : Int[Array, " n_injections"]
        The corresponding topology index for each injection
    metric : Float[Array, " n_injections"]
        The metric to minimize for each injection

    Returns
    -------
    Int[Array, " "]
        The best injection index for the given topology or int-max
        if the topology does not occur in corresponding_topologies
    """
    # Mask out the worst failures with infinity so they won't be chosen by argmin
    metric_masked = jnp.where(corresponding_topologies == topology, metric, jnp.inf)
    best_injection = jnp.argmin(metric_masked)

    # If the result is equal to infinity, return int max
    best_injection = jnp.where(
        metric_masked[best_injection] == jnp.inf,
        jnp.iinfo(best_injection.dtype).max,
        best_injection,
    )

    return best_injection


def get_best_for_topologies(
    corresponding_topologies: Int[Array, " n_injections"],
    metric: Float[Array, " n_injections"],
    batch_size_bsdf: int,
) -> Int[Array, " batch_size_bsdf"]:
    """Get the best injection combination for all topologies in range 0, batch_size_bsdf

    Returns int-max for all topology indices that never occur in corresponding_topologies

    Parameters
    ----------
    corresponding_topologies : Int[Array, " n_injections"]
        The corresponding topology index for each injection
    metric : Float[Array, " n_injections"]
        The metric to minimize for each injection
    batch_size_bsdf : int
        The number of topologies to aggregate down to

    Returns
    -------
    Int[Array, " batch_size_bsdf"]
        The best injection index for each topology
    """
    return jax.vmap(get_best_for_topology, in_axes=(0, None, None))(
        jnp.arange(batch_size_bsdf), corresponding_topologies, metric
    )


def update_aggregate_results(
    injections: Bool[Array, " batch_size_injection n_split max_inj_per_sub"],
    corresponding_topology: Int[Array, " batch_size_injection"],
    results_cur: PyTree[Shaped, " batch_size_injection ..."],
    metrics_cur: Float[Array, " batch_size_injection"],
    pad_mask: Bool[Array, " batch_size_injection"],
    results_acc: PyTree[Shaped, " batch_size_bsdf ..."],
    best_inj_acc: Bool[Array, " batch_size_bsdf n_split max_inj_per_sub"],
    metrics_acc: Float[Array, " batch_size_bsdf"],
) -> tuple[
    PyTree[Shaped, " batch_size_bsdf ..."],
    Bool[Array, " batch_size_bsdf n_split max_inj_per_sub"],
    Float[Array, " batch_size_bsdf"],
]:
    """Aggregate the results for a single injection batch, keeping a running max

    Parameters
    ----------
    injections: Bool[Array, " batch_size_injection n_split max_inj_per_sub"],
        The injection vector tried for every combination in the injection batch
    corresponding_topology: Int[Array, " batch_size_injection"]
        The corresponding topology, should index into 0 - batch_size_bsdf. Entries outside of that
        range will be ignored
    results_cur: PyTree[Shaped, " batch_size_injection ..."]
        The current results for the injection batch
    metrics_cur: Float[Array, " batch_size_injection"]
        The current metrics for the injection batch
    pad_mask: Bool[Array, " batch_size_injection"]
        Which entries in the injection batch to regard (true) and which to ignore (false)
    results_acc: PyTree[Shaped, " batch_size_bsdf ..."]
        The current best results for each topology
    best_inj_acc: Bool[Array, " batch_size_bsdf n_split max_inj_per_sub"]
        The current best injection combination
    metrics_acc: Float[Array, " batch_size_bsdf"]
        The current best metrics for each topology


    Returns
    -------
    PyTree[Shaped, " batch_size_bsdf ..."]
        The updated results_acc, where better results have been overwritten
    Bool[Array, " batch_size_bsdf n_split max_inj_per_sub"]
        The updated best_inj_acc, where better results have been overwritten
    Float[Array, " batch_size_bsdf"]
        The updated metrics_acc, where better results have been overwritten
    """
    batch_size_bsdf = best_inj_acc.shape[0]

    metric = jnp.where(pad_mask, metrics_cur, jnp.inf)

    # Get the best from the current batch
    # If a topology wasn't represented in this batch, returns int_max
    injection_index: Int[Array, " batch_size_bsdf"] = get_best_for_topologies(
        corresponding_topologies=corresponding_topology,
        metric=metric,
        batch_size_bsdf=batch_size_bsdf,
    )

    # We decide for each injection whether we overwrite it if it is better than the current best
    replace_res: Bool[Array, " batch_size_bsdf"] = metrics_acc > metric.at[injection_index].get(
        mode="fill", fill_value=jnp.inf
    )

    # Update the results
    results_acc = jax.tree_util.tree_map(
        lambda acc, cur: jnp.where(
            jnp.expand_dims(replace_res, tuple(-i for i in range(1, len(acc.shape)))),
            cur.at[injection_index].get(mode="fill", fill_value=0),
            acc,
        ),
        results_acc,
        results_cur,
    )

    best_inj_acc = jnp.where(
        jnp.expand_dims(jnp.expand_dims(replace_res, axis=-1), axis=-1),
        injections.at[injection_index].get(mode="fill", fill_value=0),
        best_inj_acc,
    )

    metrics_acc = jnp.where(
        replace_res,
        metric.at[injection_index].get(mode="fill", fill_value=jnp.inf),
        metrics_acc,
    )

    return results_acc, best_inj_acc, metrics_acc


def update_aggregate_metrics(
    injections: Bool[Array, " batch_size_injection n_splits max_inj_per_sub"],
    corresponding_topology: Int[Array, " batch_size_injections"],
    metric: Float[Array, " batch_size_injections"],
    pad_mask: Bool[Array, " batch_size_injections"],
    metrics_acc: Float[Array, " batch_size_bsdf"],
    best_inj_acc: Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"],
) -> tuple[Float[Array, " batch_size_bsdf"], Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]]:
    """Aggregate the results for a single injection batch, keeping a running min

    Parameters
    ----------
    injections: Bool[Array, " batch_size_injection n_splits max_inj_per_sub"],
        The injection vector tried for every combination in the injection batch
    corresponding_topology: Int[Array, " batch_size_injections"]
        The corresponding topology, should index into 0 - batch_size_bsdf. Entries outside of that
        range will be ignored
    metric: Float[Array, " batch_size_injections"]
        The metric to minimize for each injection
    pad_mask: Bool[Array, " batch_size_injections"]
        Which entries in the injection batch to regard (true) and which to ignore (false)
    metrics_acc: Float[Array, " batch_size_bsdf"]
        The current best metric for each topology
    best_inj_acc: Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]
        The current best injection combination for each topology

    Returns
    -------
    Float[Array, " batch_size_bsdf"]
        The updated metrics_acc, where better results have been overwritten
    Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]
        The updated best_inj_acc, where better results have been overwritten
    """
    metric = jnp.where(pad_mask, metric, jnp.inf)
    injection_index: Int[Array, " batch_size_bsdf"] = get_best_for_topologies(
        corresponding_topologies=corresponding_topology,
        metric=metric,
        batch_size_bsdf=metrics_acc.shape[0],
    )

    best_metrics: Float[Array, " batch_size_bsdf"] = metric.at[injection_index].get(mode="fill", fill_value=jnp.inf)

    replace_res: Bool[Array, " batch_size_bsdf"] = metrics_acc > best_metrics

    metrics_acc = jnp.where(
        replace_res,
        best_metrics,
        metrics_acc,
    )

    best_inj_acc = jnp.where(
        replace_res[:, None, None],
        injections.at[injection_index].get(mode="fill", fill_value=False),
        best_inj_acc,
    )

    return metrics_acc, best_inj_acc


def get_worst_failures_single_timestep(
    n_1_matrix: Float[Array, " n_failures n_branches"],
    branches_to_fail: Int[Array, " n_failures"],
    number_most_affected: int,
    number_max_out_in_most_affected: Optional[int] = None,
) -> SparseNMinus1:
    """Get the worst failures directly from the n-1 matrix

    Computes only a single timestep

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_failures n_branches"]
        N-1 matrix as obtained by n_1_analysis
    branches_to_fail : Int[Array, " n_failures"]
        The list of N-1 failure cases. This is needed because hist_out indexes into this vector,
        not into the vector of all branches
    number_most_affected : int
        Number of worst results to track
    number_max_out_in_most_affected : Optional[int], defaults to None
        Limit the number of results per outage that are included to number_max_out_in_most_affected
        If not given, then it is possible that all number_most_affected results are from the same
        outage.

    Returns
    -------
    SparseNMinus1
        Sparse NMinus1 results of shape ( number_most_affected)
    """
    n_failures, n_branches = n_1_matrix.shape
    if number_max_out_in_most_affected is not None and n_failures * number_max_out_in_most_affected < number_most_affected:
        raise ValueError("The max_out_in number is too small to get the required number of results")

    if number_max_out_in_most_affected is None:
        # If we don't have a max_out_in limit, we just take straight from the flattened N-1 matrix
        top_k_val, top_k_idx = jax.lax.top_k(n_1_matrix.flatten(), k=number_most_affected)
        top_k_idx = top_k_idx.astype(int)
        idx_branch = top_k_idx % n_branches
        idx_failure = top_k_idx // n_branches

    else:
        # Limit the number of failures per outage that are included to number_max_out_in_most_affected
        # We get shape (n_failures, number_max_out_in_most_affected) back
        # top_k_per_failure_val, top_k_per_failure_idx = jax.lax.top_k(
        #     n_1_matrix, k=number_max_out_in_most_affected
        # )
        # Argmax-based top-k is currently faster on GPU
        top_k_per_failure_val, top_k_per_failure_idx = argmax_top_k(n_1_matrix, k=number_max_out_in_most_affected)

        # From the flattened values, find the worst number_most_affected
        top_k_val, top_k_global_idx = jax.lax.top_k(top_k_per_failure_val.flatten(), k=number_most_affected)
        top_k_global_idx = top_k_global_idx.astype(int)

        # Recover the indices
        # First, go back to the unflattened top_k_per_failure_idx dimensions
        idx_failure = top_k_global_idx // number_max_out_in_most_affected
        idx_branch_in_selection = top_k_global_idx % number_max_out_in_most_affected

        # The failure index is correct already, as the failure dimension wasn't touched
        # Correct the branch index, as it is currently pointing into the top_k_per_failure_val
        # matrix which only holds a selection of branches
        idx_branch = top_k_per_failure_idx[idx_failure, idx_branch_in_selection]

    # Hist_out indexes into the branches_to_fail vector, not into the vector of all branches
    hist_out = branches_to_fail[idx_failure]

    return SparseNMinus1(
        pf_n_1_max=top_k_val,
        hist_mon=idx_branch,
        hist_out=hist_out,
    )


def get_worst_failures(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    branches_to_fail: Int[Array, " n_failures"],
    number_most_affected: int,
    number_max_out_in_most_affected: Optional[int] = None,
) -> SparseNMinus1:
    """Get the worst failures directly from the n-1 matrix

    Computes across all timesteps

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        N-1 matrix as obtained by n_1_analysis with all timesteps
    branches_to_fail : Int[Array, " n_failures"]
        The list of N-1 failure cases. This is needed because hist_out indexes into this vector,
        not into the vector of all branches
    number_most_affected : int
        Number of worst results to track
    number_max_out_in_most_affected : Optional[int], defaults to None
        Limit the number of results per outage that are included to number_max_out_in_most_affected
        If not given, then it is possible that all number_most_affected results are from the same
        outage.

    Returns
    -------
    SparseNMinus1
        Sparse NMinus1 results of shape ( n_timesteps, number_most_affected)
    """
    assert branches_to_fail.shape[0] == n_1_matrix.shape[1]

    return jax.vmap(
        partial(
            get_worst_failures_single_timestep,
            branches_to_fail=branches_to_fail,
            number_most_affected=number_most_affected,
            number_max_out_in_most_affected=number_max_out_in_most_affected,
        ),
    )(n_1_matrix)


def get_worst_n_0(
    loading_n_0: Float[Array, " ... n_timesteps n_branches"],
    number_most_affected: int,
) -> SparseNMinus0:
    """Get the worst N-0 results directly from the n-0 matrix

    Can deal with arbitrary batch dimensions

    Parameters
    ----------
    loading_n_0 : Float[Array, " ... n_timesteps n_branches"]
        N-0 matrix as obtained by n_0_analysis with all timesteps
    number_most_affected : int
        Number of worst results to track

    Returns
    -------
    SparseNMinus0
        Sparse NMinus0 results of shape ( ... n_timesteps, number_most_affected )
    """
    n_0_flow_max, idx_most_affected_n_0 = jax.lax.top_k(loading_n_0, k=number_most_affected)

    return SparseNMinus0(
        pf_n_0_max=n_0_flow_max,
        hist_mon=idx_most_affected_n_0.astype(int),
    )


def sparsify_results(
    n_0_matrix: Float[Array, " n_timesteps n_branches"],
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    branches_to_fail: Int[Array, " n_failures"],
    max_mw_flow: Float[Array, " n_branches"],
    number_most_affected: int,
    number_max_out_in_most_affected: Optional[int],
    number_most_affected_n_0: int,
) -> tuple[SparseNMinus0, SparseNMinus1]:
    """Sparsify the N-0 and N-1 results

    Uses get_worst_failures and get_worst_n_0

    Parameters
    ----------
    n_0_matrix : Float[" n_timesteps n_branches"]
        The N-0 matrix
    n_1_matrix : Float[" n_timesteps n_failures n_branches"]
        The N-1 matrix
    branches_to_fail : Int[" n_failures"]
        The branch indices that fail in each failure case
    max_mw_flow : Float[" n_branches"]
        The maximum flow for each branch
    number_most_affected : int
        Number of worst results to track for N-1
    number_max_out_in_most_affected : Optional[int]
        Limit the number of results per outage that are included to number_max_out_in_most_affected
    number_most_affected_n_0 : int
        Number of worst results to track for N-0

    Returns
    -------
    SparseNMinus0
        The aggregated N-0 results
    SparseNMinus1
        The aggregated N-1 results
    """
    n_0_matrix = jnp.abs(n_0_matrix / max_mw_flow)
    n_1_matrix = jnp.abs(n_1_matrix / max_mw_flow)

    return (
        get_worst_n_0(n_0_matrix, number_most_affected_n_0),
        get_worst_failures(
            n_1_matrix,
            branches_to_fail,
            number_most_affected,
            number_max_out_in_most_affected,
        ),
    )
