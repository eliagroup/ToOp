# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines for aggregating results across injection batches.

The default mode of operation is to store only one injection candidate based on a selection criterium, and ignore
all the others. The default selection criterium from the numpy code is the argmin selection, taking
the injection combination that yields the minimal worst N-1 case. However, we also include a
selection criterium based on overload energy
"""

import jax
from beartype.typing import Literal, Optional, TypeAlias
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree
from toop_engine_dc_solver.jax.types import (
    BranchLimits,
    SolverLoadflowResults,
)
from toop_engine_interfaces.types import (
    MatrixMetric,
    MetricType,
)

AggregateStrategy: TypeAlias = Literal["max", "nanmax"]


def get_max_flow_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Compute the maximum flow for an N-1 matrix

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " "]
        The maximum flow for the given flow
    """
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    return max_fn(jnp.abs(n_1_matrix / max_mw_flow))


def get_overload_energy_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    overload_weight: Optional[Float[Array, " n_branches"]] = None,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Compute the overload energy for an N-1 matrix

    The overload energy is the amount of energy on branches that exceeds their rating

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    overload_weight : Optional[Float[Array, " n_branches"]], defaults to None
        The overload weight for each branch. If not given, all branches are weighted equally
    aggregate_strategy : Optional[AggregateStartegy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.
        Can be "max" or "nanmax". The "nanmax" will ignore NaN values, while "max" will not.
        This is useful if you want to ignore failures that are not relevant for the metric,
        e.g. if you want to ignore busbar outage failures in the overload energy calculation.

    Returns
    -------
    Float[Array, " "]
        The overload energy for the given flow, i.e. the sum of loads that are greater than the
        maximum flow
    """
    # Only the portion above the crit threshold (1) is overload energy, multiply this with the
    # branch limit to turn the relative overload into absolute overload energy
    overload_matrix = jnp.clip(jnp.abs(n_1_matrix) - max_mw_flow, min=0, max=None)

    # Apply overload weights if given
    if overload_weight is not None:
        overload_matrix = overload_matrix * overload_weight

    # We want to sum the overload energy over all timesteps and max over all failures
    # We need to max first as we want to ignore the non-worst failures
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    return jnp.sum(max_fn(overload_matrix, axis=1))


def get_exponential_overload_energy_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    overload_weight: Optional[Float[Array, " n_branches"]] = None,
    alpha: float = 1.5,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Get an exponentially weighted overload energy with a given factor

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    overload_weight : Optional[Float[Array, " n_branches"]], defaults to None
        The overload weight for each branch. If not given, all branches are weighted equally
    alpha : float, defaults to 1.0
        The exponential factor to use
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " "]
        The exponentially weighted overload energy for the given flow
    """
    # We need to compute the overload energy in relative terms, i.e. how many percent overload
    # do we have on each branch
    # Everything under 100% is clipped, as that will incur a penalty of 0
    overload_matrix = jnp.clip(jnp.abs(n_1_matrix) / max_mw_flow, min=1, max=None)

    # Exponentiate the overloads with the given factor
    overload_matrix = overload_matrix**alpha

    # Transform back into absolute overload energy
    overload_matrix = (overload_matrix - 1) * max_mw_flow

    # Apply overload weights if given
    if overload_weight is not None:
        overload_matrix = overload_matrix * overload_weight

    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax

    # Sum over all timesteps and max over all failures
    return jnp.sum(max_fn(overload_matrix, axis=1))


def get_critical_branch_count_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Int[Array, " "]:
    """Compute the number of critical branches for an N-1 matrix

    A critical branch is a branch that is overloaded in at least one failure. It only returns
    the worst timestep, i.e. if one timestep has 20 problems and the next 25, the result will be 25.

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Int[Array, " "]
        The number of critical branches for the given flow
    """
    # A branch is critical if it is overloaded in at least one failure
    is_critical = jnp.any(jnp.abs(n_1_matrix) > max_mw_flow, axis=1)
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    return max_fn(jnp.sum(is_critical, axis=1))


def get_transport_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Compute the transport for an N-1 matrix

    The transport is the total amount of energy that is transported over all branches, normalized by
    the total capacity

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.


    Returns
    -------
    Float[Array, " "]
        The transport for the given flow
    """
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    total_mw_flow = jnp.sum(max_fn(jnp.abs(n_1_matrix), axis=1))
    capacity = jnp.sum(max_mw_flow) * n_1_matrix.shape[0]

    return total_mw_flow / capacity


def get_number_of_splits(
    branch_topology: Bool[Array, " n_splits max_branch_per_sub"],
    sub_ids: Int[Array, " n_splits"],
    n_relevant_subs: int,
) -> Int[Array, " "]:
    """Count the number of split substations in a topology

    Parameters
    ----------
    branch_topology : Bool[Array, " n_splits max_branch_per_sub"]
        The branch topology for each split
    sub_ids : Int[Array, " n_splits"]
        The substation ids for each split, padded with int-max for unsplit substations
    n_relevant_subs : int
        The number of relevant substations in the grid, used for split_subs metric

    Returns
    -------
    Int[Array, " "]
        The number of split substations in the topology
    """
    # Only count the substations that are valid
    has_splits = jnp.any(branch_topology, axis=-1)
    subid_valid = (sub_ids >= 0) & (sub_ids < n_relevant_subs)
    return jnp.sum(has_splits & subid_valid)


def get_number_of_disconnections(
    disconnections: Optional[Int[Array, " max_n_disconnections"]], n_branches: int
) -> Int[Array, " "]:
    """Count the number of actual disconnections in the disconnections vector

    A disconnection is assumed valid if it refers to an existing branch - this does
    not check if the branch is in the disconnectable_branches array as per default
    unused disconnections slots are padded with int_max and not a valid branch outside
    of the disconnectable branches array.

    Parameters
    ----------
    disconnections : Optional[Int[Array, " max_n_disconnections"]]
        The disconnections vector, padded with int_max for unused slots
    n_branches : int
        The number of branches in the grid, used to check if the disconnection is valid

    Returns
    -------
    Int[Array, " "]
        The number of disconnections in the disconnections vector
    """
    if disconnections is None:
        return 0
    # Only count the disconnections that are valid
    return jnp.sum((disconnections >= 0) & (disconnections < n_branches))


def get_cross_coupler_flow_penalty(
    cross_coupler_flows: Float[Array, " n_splits n_timesteps"],
    sub_ids: Int[Array, " n_splits"],
    coupler_limits: Optional[Float[Array, " n_subs_rel"]],
) -> Float[Array, " "]:
    """Compute the penalty for violating the maximum cross coupler flow

    Parameters
    ----------
    cross_coupler_flows : Float[Array, " n_splits n_timesteps"]
        The cross coupler flow for each split (following the order of splits in bsdf).
        Basically the output of compute_cross_coupler_flows().
    sub_ids : Int[Array, " n_splits"]
        The substation ids for each split, padded with int-max for unsplit substations
    coupler_limits : Optional[Float[Array, " n_subs_rel"]]
        The maximum cross flow allowed for the coupler in every relevant substation. This
        is optional because it is optional in the branch limits dataclass, however it is
        required for this metric computation. Will raise a ValueError upon None.

    Returns
    -------
    Float[Array, " "]
        The penalty for violating the maximum cross coupler flow, which is equal to
        the MW exceeding the maximum cross coupler flow summed over all splits and timesteps

    Raises
    ------
    ValueError
        If coupler_limits is None
    """
    if coupler_limits is None:
        raise ValueError("No coupler limits given for cross_coupler_flow metric")

    limits = coupler_limits.at[sub_ids].get(mode="fill", fill_value=jnp.inf)

    penalty = jnp.clip(jnp.abs(cross_coupler_flows) - limits[:, None], min=0, max=None)
    return jnp.sum(penalty)


def get_median_flow_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Compute the median flow for an N-1 matrix

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " "]
        The median flow for the given flow
    """
    n_1_matrix = jnp.abs(n_1_matrix / max_mw_flow)
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    return jnp.median(max_fn(n_1_matrix, axis=1))


def get_underload_energy_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
) -> Float[Array, " "]:
    """Compute the underload energy for an N-1 matrix

    The underload energy is the amount of energy on branches that is below their rating

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch

    Returns
    -------
    Float[Array, " "]
        The underload energy for the given flow, i.e. the sum of loads that are smaller than the
        maximum flow
    """
    # Only the portion below the crit threshold (1) is underload energy, multiply this with the
    # branch limit to turn the relative underload into absolute underload energy
    underload_matrix = jnp.clip(max_mw_flow - jnp.abs(n_1_matrix), min=0, max=None)

    # We want to sum the underload energy over all timesteps and max over all failures
    # We need to min first as we want to ignore the non-worst failures
    return jnp.sum(jnp.min(underload_matrix, axis=1))


def get_n0_n1_delta(
    n_0: Float[Array, " n_timesteps n_branches"],
    n_1: Float[Array, " n_timesteps n_failures n_branches"],
    only_positive: bool = True,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " n_timesteps n_branches"]:
    """Compute the maximum delta between the N-0 base case and any N-1 case

    Parameters
    ----------
    n_0 : Float[Array, " n_timesteps n_branches"]
        The N-0 matrix, in MW
    n_1 : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, in MW
    only_positive : bool, defaults to True
        If True, only positive deltas are returned, i.e. one where abs(N-1) is larger than abs(N-0)
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " n_timesteps n_branches"]
        The maximum delta between N-0 and N-1 in MW
    """
    delta = jnp.abs(n_1) - jnp.abs(n_0)[:, None, :]
    if only_positive:
        delta = jnp.clip(delta, min=0, max=None)
        if aggregate_strategy == "max":
            max_fn = jnp.max
        elif aggregate_strategy == "nanmax":
            max_fn = jnp.nanmax
        return max_fn(delta, axis=1)

    highest_delta = jnp.argmax(jnp.abs(delta), axis=1)
    return delta[jnp.arange(delta.shape[0]), highest_delta]


def compute_n0_n1_max_diff(
    n_0: Float[Array, " n_timesteps n_branches"],
    n_1: Float[Array, " n_timesteps n_failures n_branches"],
    factors: Float[Array, " n_branches"],
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " n_branches"]:
    """Compute the n0_n1_max_diff based off base case loadflow results and a factor for each branch

    If the factor is negative, the branch will be ignored in the penalty calculation
    If the factor is positive, then the base case difference will be multiplied by the factor.

    Parameters
    ----------
    n_0 : Float[Array, " n_timesteps n_branches"]
        The N-0 base case loadflow results
    n_1 : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 base case loadflow results
    factors : Float[Array, " n_branches"]
        The factors to multiply the base case difference with
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " n_branches"]
        The maximum allowed relative difference between N-0 and N-1, in MW. Can be stored as
        static_information.dynamic_information.branch_limits.n0_n1_max_diff
    """
    delta = get_n0_n1_delta(n_0, n_1, only_positive=True)
    # Only regard the worst timestep
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    delta = max_fn(delta, axis=0)
    return jnp.where(factors < 0, -1, delta * factors)


def get_n0_n1_delta_penalty(
    n_0: Float[Array, " n_timesteps n_branches"],
    n_1: Float[Array, " n_timesteps n_failures n_branches"],
    n0_n1_max_diff: Optional[Float[Array, " n_branches"]],
) -> Float[Array, " "]:
    """Compute the penalty for violating maximum N-0 to N-1 delta

    Parameters
    ----------
    n_0 : Float[Array, " n_timesteps n_branches"]
        The N-0 matrix, in MW
    n_1 : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, in MW
    n0_n1_max_diff : Optional[Float[Array, " n_branches"]]
        The maximum allowed relative difference between N-0 and N-1, in MW. If this value is
        negative, the penalty for this branch will always be zero. This is optional because it
        is optional in the branch limits dataclass, but actually required for this metric.
        If None, a ValueError will be raised.

    Returns
    -------
    Float[Array, " "]
        The penalty for violating the maximum N-0 to N-1 delta, which is equal to the MW that
        exceeds the maximum allowed difference summed over all branches and timesteps

    Raises
    ------
    ValueError
        If n0_n1_max_diff is None, as this is required for the penalty calculation
    """
    if n0_n1_max_diff is None:
        raise ValueError("No n0_n1_max_diff given for n0_n1_delta metric")
    delta = get_n0_n1_delta(n_0, n_1, only_positive=True)
    penalty = jnp.clip(delta - n0_n1_max_diff, min=0, max=None)
    penalty = jnp.where(n0_n1_max_diff < 0, 0, penalty)
    return jnp.sum(penalty)


def get_switching_distance(
    branch_action_index: Int[Array, " n_splits"],
    reassignment_distance: Int[Array, " n_branch_actions"],
) -> Int[Array, " "]:
    """Look up the switching distance for a branch action.

    Parameters
    ----------
    branch_action_index : Int[Array, " n_splits"]
        The branch action for each split, as an index into the branch action set. Invalid indices will be ignored
    reassignment_distance : Int[Array, " n_branch_actions"]
        The reassignment distance for each branch action, computed during preprocessing

    Returns
    -------
    Int[Array, " "]
        The switching distance for each split
    """
    return reassignment_distance.at[branch_action_index].get(mode="fill", fill_value=0).sum()


def get_bb_outage_penalty(bb_outage_penalty: Optional[Float[Array, " "]]) -> Float[Array, " "]:
    """Pass the BB outage penalty from the solver or raise if not provided."""
    if bb_outage_penalty is None:
        raise ValueError("No BB outage penalty returned from solver")
    return bb_outage_penalty


def get_bb_outage_overload(bb_outage_overload: Optional[Float[Array, " "]]) -> Float[Array, " "]:
    """Pass the BB outage overload from the solver or raise if not provided."""
    if bb_outage_overload is None:
        raise ValueError("No BB outage overload returned from solver")
    return bb_outage_overload


def get_bb_outage_grid_splits(bb_outage_grid_splits: Optional[Int[Array, " "]]) -> Int[Array, " "]:
    """Pass the BB outage grid splits from the solver or raise if not provided."""
    if bb_outage_grid_splits is None:
        raise ValueError("No BB outage grid splits returned from solver")
    return bb_outage_grid_splits


def get_n_2_penalty(
    n_2_penalty: Optional[Float[Array, " "]],
) -> Float[Array, " "]:
    """Pass the N-2 penalty from the solver or raise if not provided."""
    if n_2_penalty is None:
        raise ValueError("No N-2 results returned from solver, can not pass to metric")
    return n_2_penalty


def choose_max_mw_flow(
    branch_limits: BranchLimits,
    metric: MetricType,
) -> Float[Array, " n_branches"]:
    """Choose the correct max_mw_flow based on the metric

    Parameters
    ----------
    branch_limits : BranchLimits
        The branch limits dataclass, holding max_mw_flow and max_mw_flow_n_1
    metric : MetricType
        The metric to use for aggregation. If it contains _n_1, then max_mw_flow_n_1 is used,
        otherwise max_mw_flow is used

    Returns
    -------
    Float[Array, " n_branches"]
        The maximum flow for each branch
    """
    if metric in [
        "overload_energy_limited_n_0",
        "exponential_overload_energy_limited_n_0",
        "critical_branch_count_limited_n_1",
    ]:
        if branch_limits.max_mw_flow_limited is None:
            raise ValueError(f"No max_mw_flow_limited given for limited N-0 metric computation {metric}")
        return branch_limits.max_mw_flow_limited
    if metric in [
        "overload_energy_limited_n_1",
        "exponential_overload_energy_limited_n_1",
        "critical_branch_count_limited_n_1",
    ]:
        limit = (
            branch_limits.max_mw_flow_n_1_limited
            if branch_limits.max_mw_flow_n_1_limited is not None
            else branch_limits.max_mw_flow_limited
        )
        if limit is None:
            raise ValueError(f"No max_mw_flow_n_1_limited given for limited N-1 metric computation {metric}")
        return limit
    if metric.endswith("_n_1"):
        return branch_limits.max_mw_flow_n_1 if branch_limits.max_mw_flow_n_1 is not None else branch_limits.max_mw_flow
    return branch_limits.max_mw_flow


def aggregate_to_metric_batched(
    lf_res_batch: SolverLoadflowResults,
    branch_limits: BranchLimits,
    reassignment_distance: Int[Array, " n_branch_actions"],
    n_relevant_subs: int,
    metric: MetricType = "max_flow_n_1",
) -> Float[Array, " batch_size"]:
    """Aggregate the N-0 and N-1 results down to a single metric

    This is a batched version of aggregate_to_metric and can be used to aggregate multiple results,
    however if you want to pass an aggregate_fn to run_solver, you should use aggregate_to_metric.

    Parameters
    ----------
    lf_res_batch : LoadflowMatrices
        The loadflow results for a batch of topologies. I.e. every field has a leading batch dimension
    branch_limits : BranchLimits
        The branch limits dataclass. Usually from static_information.dynamic_information.branch_limits
    reassignment_distance: Int[Array, " n_branch_actions"]
        The switching distance information, if available, to compute the switching distance metric
    n_relevant_subs : int
        The number of relevant substations in the grid, used for split_subs metric
    metric : MetricType = "max_flow_n_1"
        The metric to use for aggregation.

    Returns
    -------
    Float[Array, " batch_size"]
        The aggregated metric
    """
    return jax.vmap(aggregate_to_metric, in_axes=(0, None, None, None, None))(
        lf_res_batch,
        branch_limits,
        reassignment_distance,
        n_relevant_subs,
        metric,
    )


def aggregate_matrix_to_metric(
    lf_res: SolverLoadflowResults,
    branch_limits: BranchLimits,
    metric: MatrixMetric,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Aggregate a metric that needs the N-1/N-0 matrices

    This chooses the correct max_mw_flow and calls aggregate_n_1_matrix

    Parameters
    ----------
    lf_res : LoadflowMatrices
        The loadflow results for a single topology.
    branch_limits : BranchLimits
        The branch limits dataclass. Usually from static_information.dynamic_information.branch_limits
    metric : MatrixMetric
        The metric to use for aggregation. If it contains _n_1, then max_mw_flow_n_1 is used,
        otherwise max_mw_flow is used
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.
        Can be "max" or "nanmax". The "nanmax" will ignore NaN values, while "max" will not.
        This is useful if you want to ignore failures that are not relevant for the metric,
        e.g. if you want to ignore busbar outage failures in the overload energy calculation.

    Returns
    -------
    Float[Array, " "]
        The aggregated metric
    """
    max_mw_flow_used = choose_max_mw_flow(branch_limits, metric)
    if metric.endswith("_n_0"):
        matrix_used = jnp.expand_dims(lf_res.n_0_matrix, axis=1)
        metric_used = metric.replace("_n_0", "")
    elif metric.endswith("_n_1"):
        matrix_used = lf_res.n_1_matrix
        metric_used = metric.replace("_n_1", "")
    else:
        raise ValueError(f"Unknown metric {metric}")

    if "_limited" in metric:
        metric_used = metric_used.replace("_limited", "")

    return aggregate_n_1_matrix(
        matrix_used,
        max_mw_flow_used,
        metric=metric_used,
        overload_weight=branch_limits.overload_weight,
        aggregate_strategy=aggregate_strategy,
    )


def aggregate_to_metric(
    lf_res: SolverLoadflowResults,
    branch_limits: BranchLimits,
    reassignment_distance: Int[Array, " n_branch_actions"],
    n_relevant_subs: int,
    metric: MetricType = "max_flow_n_1",
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Aggregate the N-0 and N-1 results down to a single metric

    This gives several options for commonly used aggregation metrics
    To be used as the aggregate_output_fn in run_solver

    Parameters
    ----------
    lf_res : LoadflowMatrices
        The loadflow results for a single topology. If you want to compute for multiple (with a
        leading batch dimension), use aggregate_to_metric_batched
    branch_limits : BranchLimits
        The branch limits dataclass. Usually from static_information.dynamic_information.branch_limits
    reassignment_distance: Int[Array, " n_branch_actions"]
        The switching distance information, if available, to compute the switching distance metric
    n_relevant_subs : int
        The number of relevant substations in the grid, used for split_subs metric
    metric : MetricType
        The metric to use for aggregation.
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.
        Can be "max" or "nanmax". The "nanmax" will ignore NaN values, while "max" will not.
        This is useful if you want to ignore failures that are not relevant for the metric,
        e.g. if you want to ignore busbar outage failures in the overload energy calculation.

    Returns
    -------
    Float[Array, " "]
        The aggregated metric
    """
    if metric == "n0_n1_delta":
        retval = get_n0_n1_delta_penalty(lf_res.n_0_matrix, lf_res.n_1_matrix, branch_limits.n0_n1_max_diff)
    elif metric == "cross_coupler_flow":
        retval = get_cross_coupler_flow_penalty(lf_res.cross_coupler_flows, lf_res.sub_ids, branch_limits.coupler_limits)
    elif metric == "switching_distance":
        retval = get_switching_distance(
            branch_action_index=lf_res.branch_action_index,
            reassignment_distance=reassignment_distance,
        )
    elif metric == "split_subs":
        retval = get_number_of_splits(lf_res.branch_topology, lf_res.sub_ids, n_relevant_subs)
    elif metric == "disconnected_branches":
        retval = get_number_of_disconnections(lf_res.disconnections, branch_limits.max_mw_flow.shape[0])
    elif metric == "n_2_penalty":
        retval = get_n_2_penalty(lf_res.n_2_penalty)
    elif metric == "bb_outage_penalty":
        retval = get_bb_outage_penalty(lf_res.bb_outage_penalty)
    elif metric == "bb_outage_overload":
        retval = get_bb_outage_overload(lf_res.bb_outage_overload)
    elif metric == "bb_outage_grid_splits":
        retval = get_bb_outage_grid_splits(lf_res.bb_outage_splits)
    else:
        retval = aggregate_matrix_to_metric(
            lf_res=lf_res, branch_limits=branch_limits, metric=metric, aggregate_strategy=aggregate_strategy
        )
    return retval


def aggregate_n_1_matrix(
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    metric: Literal[
        "max_flow",
        "median_flow",
        "overload_energy",
        "underload_energy",
        "transport",
        "exponential_overload_energy",
    ] = "max_flow",
    overload_weight: Optional[Float[Array, " n_branches"]] = None,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " "]:
    """Aggregate the N-1 matrix down to a single metric

    Parameters
    ----------
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, i.e. the relative flows for each timestep and each failure
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    metric : Literal["max_flow", "median_flow", "overload_energy", "underload_energy", "transport"]
        The metric to use for aggregation. Possible values are:
        "max_flow", "median_flow", "transport", "overload_energy"
        Max_flow will compute the maximum relative flow over all N-1 results.
        Median_flow will compute the median relative flow over all N-1 results.
        Overload_energy will compute the amount of energy that exceeds the maximum flow
        Underload_energy will compute the amount of energy that is below the maximum flow
        Transport will use the get_transport_n_1_matrix function to compute the metric
    overload_weight : Optional[Float[Array, " n_branches"]], defaults to None
        The overload weight for each branch. If not given, all branches are weighted equally
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.
        Can be "max" or "nanmax". The "nanmax" will ignore NaN values, while "max" will not.
        This is useful if you want to ignore failures that are not relevant for the metric,
        e.g. if you want to ignore busbar outage failures in the overload energy calculation.

    Returns
    -------
    Float[Array, " "]
        The aggregated metric
    """
    assert len(n_1_matrix.shape) == 3, "This method does not support a batch dimension, use aggregate_n_1_matrices instead"

    if metric == "max_flow":
        metric = get_max_flow_n_1_matrix(n_1_matrix, max_mw_flow, aggregate_strategy)
    elif metric == "overload_energy":
        metric = get_overload_energy_n_1_matrix(n_1_matrix, max_mw_flow, overload_weight, aggregate_strategy)
    elif metric == "underload_energy":
        metric = get_underload_energy_n_1_matrix(n_1_matrix, max_mw_flow)
    elif metric == "transport":
        metric = get_transport_n_1_matrix(n_1_matrix, max_mw_flow, aggregate_strategy)
    elif metric == "median_flow":
        metric = get_median_flow_n_1_matrix(n_1_matrix, max_mw_flow, aggregate_strategy)
    elif metric == "exponential_overload_energy":
        metric = get_exponential_overload_energy_n_1_matrix(
            n_1_matrix, max_mw_flow, overload_weight, aggregate_strategy=aggregate_strategy
        )
    elif metric == "critical_branch_count":
        metric = get_critical_branch_count_n_1_matrix(n_1_matrix, max_mw_flow, aggregate_strategy=aggregate_strategy)
    else:
        raise ValueError(f"Unknown metric {metric}")

    return metric


def default_metric(
    _n_0: Float[Array, " n_timesteps n_branches"],
    n_1: Float[Array, " n_timesteps n_failures n_branches"],
    _output: Optional[PyTree],
) -> Float[Array, " "]:
    """Get the default metric for the bruteforce optimization.

    The default metric for the bruteforce optimization, in compatible function signature to
    AggregateMetricFn

    Just returns the max flow over all N-1 results

    Parameters
    ----------
    _n_0 : Float[Array, " n_timesteps n_branches"]
        The N-0 matrix, in MW
    n_1 : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 matrix, in MW
    _output : Optional[PyTree]
        The output of the solver, not used in this case

    Returns
    -------
    Float[Array, " "]
        The maximum flow for the given flow
    """
    return jnp.max(n_1)


def compute_double_limits(
    n_1: Float[Array, " n_timesteps n_failures n_branches"],
    max_mw_flow: Float[Array, " n_branches"],
    lower_limit: float = 0.9,
    upper_limit: float = 1.0,
    aggregate_strategy: Optional[AggregateStrategy] = "max",
) -> Float[Array, " n_branches"]:
    """Update the maximum flow limits with a lower limit

    The idea behind the lower limit is that branches which are loaded below lower_limit shall get
    their maximum flow reduced to lower_limit * max_mw_flow, to prevent bringing them too close to
    criticality. Branches that are between lower_limit and upper_limit shall get their maximum flow
    set to their current flow to prevent them from being loaded further, but also to prevent them
    from being marked as already overloaded. Lastly, branches above upper_limit (usually 100%) shall
    get their maximum flow set to upper_limit * max_mw_flow as they should be healed.

    Parameters
    ----------
    n_1 : Float[Array, " n_timesteps n_failures n_branches"]
        The N-1 base flows (without any topologies applied). If you want to update N-0, you can add
        a dimension with jnp.expand_dims(n_0, axis=1)
    max_mw_flow : Float[Array, " n_branches"]
        The maximum flow for each branch
    lower_limit : float, defaults to 0.9
        The lower limit for the maximum flow
    upper_limit : float, defaults to 1.0
        The upper limit for the maximum flow
    aggregate_strategy : Optional[AggregateStrategy], defaults to "max"
        The literal use to select the function to use for aggregation over the failures.

    Returns
    -------
    Float[Array, " n_branches"]
        The updated maximum flow
    """
    if aggregate_strategy == "max":
        max_fn = jnp.max
    elif aggregate_strategy == "nanmax":
        max_fn = jnp.nanmax
    flows = max_fn(jnp.abs(n_1), axis=(0, 1))

    lower_mask = flows < lower_limit * max_mw_flow
    upper_mask = flows > upper_limit * max_mw_flow

    # Lower limit
    max_mw_flow = jnp.where(
        lower_mask,
        lower_limit * max_mw_flow,
        max_mw_flow,
    )

    # Deadzone in between
    max_mw_flow = jnp.where(
        ~lower_mask & ~upper_mask,
        flows,
        max_mw_flow,
    )

    # Upper limit
    max_mw_flow = jnp.where(
        upper_mask,
        upper_limit * max_mw_flow,
        max_mw_flow,
    )

    return max_mw_flow


def get_worst_k_contingencies(
    k: Int[Array, " "],
    n_1_matrix: Float[Array, " n_timesteps n_failures n_branches_monitored"],
    max_mw_flow: Float[Array, " n_branches"],
) -> tuple[Float[Array, " n_timesteps"], Int[Array, " n_timesteps k"]]:
    """Get the worst k contingencies from the n-1 matrix.

    Parameters
    ----------
    k : Int[Array, " "]
        The number of worst contingencies to select.
    n_1_matrix : Float[Array, " n_timesteps n_failures n_branches_monitored"]
        The n-1 contingency matrix.
    max_mw_flow : Float[Array, " n_branches"]
        The maximum allowed flow for each branch.

    Returns
    -------
    Float[Array, " n_timesteps"]
        The total overload corresponding to the worst k contingencies for each timestep.
    Int[Array, " n_timesteps k"]
        The indices of the worst k contingencies for each timestep.
    """
    overload_matrix = jnp.clip(jnp.abs(n_1_matrix) - max_mw_flow, min=0, max=None)

    # get worst k contingencies after removing nan values
    overload_matrix = jnp.nan_to_num(overload_matrix, nan=0.0)
    top_k_overloads, case_indices = jax.lax.top_k(overload_matrix.sum(axis=2), k)
    top_k_overloads = jnp.sum(top_k_overloads, axis=1)

    return top_k_overloads, case_indices
