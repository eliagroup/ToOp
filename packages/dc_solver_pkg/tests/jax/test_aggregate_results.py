# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pypowsybl
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics
from toop_engine_contingency_analysis.pypowsybl import run_contingency_analysis_powsybl
from toop_engine_dc_solver.jax.aggregate_results import (
    aggregate_n_1_matrix,
    aggregate_to_metric,
    aggregate_to_metric_batched,
    choose_max_mw_flow,
    compute_double_limits,
    compute_n0_n1_max_diff,
    get_critical_branch_count_n_1_matrix,
    get_cross_coupler_flow_penalty,
    get_cumulative_overload_n_1_matrix,
    get_exponential_overload_energy_n_1_matrix,
    get_max_flow_n_1_matrix,
    get_median_flow_n_1_matrix,
    get_n0_n1_delta,
    get_n0_n1_delta_penalty,
    get_number_of_disconnections,
    get_number_of_splits,
    get_overload_energy_n_1_matrix,
    get_pst_setpoint_deviation,
    get_switching_distance,
    get_transport_n_1_matrix,
    get_underload_energy_n_1_matrix,
    get_worst_k_contingencies,
)
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import BranchLimits, SolverLoadflowResults
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition


def test_get_overload_energy_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))
    overload_weights = jax.random.exponential(keys[2], (n_branch,))

    overload = jax.jit(get_overload_energy_n_1_matrix)(flow, jnp.ones_like(max_mw_flow))
    overload_ref = jnp.sum(jnp.clip(jnp.max(flow, axis=1) - 1, min=0, max=None))
    assert jnp.allclose(overload, overload_ref)

    overload = get_overload_energy_n_1_matrix(flow, jnp.ones_like(max_mw_flow), overload_weights)
    overload_ref = jnp.sum(jnp.clip(jnp.max(flow, axis=1) - 1, min=0, max=None) * overload_weights)
    assert jnp.allclose(overload, overload_ref)

    overload = get_overload_energy_n_1_matrix(flow, max_mw_flow)
    overload_ref = jnp.sum(jnp.clip(jnp.max(flow, axis=1) - max_mw_flow, min=0, max=None))


def test_get_critical_branch_count_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))

    flow_max = jnp.max(flow, axis=1)
    num_violations = jnp.sum(flow_max > max_mw_flow, axis=1)
    max_violations = jnp.max(num_violations)

    critical_branch_count = get_critical_branch_count_n_1_matrix(flow, max_mw_flow)
    assert jnp.allclose(critical_branch_count, max_violations)


def test_get_exponential_overload_energy_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))
    overload_weights = jax.random.exponential(keys[2], (n_branch,))

    overload = get_overload_energy_n_1_matrix(flow, max_mw_flow)

    exponential_overload = get_exponential_overload_energy_n_1_matrix(flow, max_mw_flow, alpha=1)
    assert exponential_overload.shape == overload.shape
    assert jnp.allclose(exponential_overload, overload)

    exponential_overload = get_exponential_overload_energy_n_1_matrix(flow, max_mw_flow, alpha=5)
    assert exponential_overload >= overload

    overload = get_overload_energy_n_1_matrix(flow, max_mw_flow, overload_weight=overload_weights)

    exponential_overload = get_exponential_overload_energy_n_1_matrix(
        flow, max_mw_flow, alpha=1, overload_weight=overload_weights
    )
    assert jnp.allclose(exponential_overload, overload)

    exponential_overload = get_exponential_overload_energy_n_1_matrix(
        flow, max_mw_flow, alpha=5, overload_weight=overload_weights
    )
    assert exponential_overload >= overload


def test_get_underload_energy_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))

    underload = get_underload_energy_n_1_matrix(flow, jnp.ones_like(max_mw_flow))
    flow_max = jnp.max(flow, axis=1)
    flow_max = jnp.clip(1 - flow_max, min=0, max=None)
    assert jnp.allclose(underload, jnp.sum(flow_max))


def test_get_cumulative_overload_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))

    cumulative = get_cumulative_overload_n_1_matrix(flow, jnp.ones_like(max_mw_flow))
    overload_matrix = jnp.clip(jnp.abs(flow) - 1, min=0, max=None)
    relative_overload = overload_matrix / 1
    cumulative_ref = jnp.sum(jnp.max(relative_overload, axis=1))
    assert jnp.allclose(cumulative, cumulative_ref)

    cumulative = get_cumulative_overload_n_1_matrix(flow, max_mw_flow)
    overload_matrix = jnp.clip(jnp.abs(flow) - max_mw_flow, min=0, max=None)
    relative_overload = overload_matrix / max_mw_flow
    cumulative_ref = jnp.sum(jnp.max(relative_overload, axis=1))
    assert jnp.allclose(cumulative, cumulative_ref)


def test_get_transport_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))

    transport = get_transport_n_1_matrix(flow, jnp.ones_like(max_mw_flow))

    assert jnp.allclose(transport, jnp.mean(jnp.max(flow, axis=1)))

    transport_2 = get_transport_n_1_matrix(flow, max_mw_flow)
    assert not jnp.allclose(transport, transport_2)


def test_get_median_flow_n_1_matrix() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    flow = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))

    median_flow = get_median_flow_n_1_matrix(flow, jnp.ones(n_branch))

    assert jnp.allclose(median_flow, jnp.median(jnp.max(flow, axis=1)))


def test_aggregate_to_metric_batched() -> None:
    n_batch = 8
    n_timesteps = 5
    n_failures = 30
    n_branch = 50
    n_splits = 10
    n_subs_rel = 60
    max_branch_per_sub = 6
    max_inj_per_sub = 4

    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    flow = jax.random.exponential(keys[0], (n_batch, n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))
    branch_topologies = jax.random.randint(keys[2], (n_batch, n_splits, max_branch_per_sub), 0, 2).astype(bool)
    sub_ids = jax.random.randint(keys[2], (n_batch, n_splits), 0, n_subs_rel)
    injections = jax.random.randint(keys[3], (n_batch, n_splits, max_inj_per_sub), 0, 2).astype(bool)
    cross_coupler_flow = jax.random.exponential(keys[3], (n_batch, n_splits, n_timesteps))
    max_flow_coupler = jax.random.exponential(keys[4], (n_subs_rel,))

    branch_limits = BranchLimits(
        max_mw_flow=max_mw_flow,
        max_mw_flow_limited=max_mw_flow * 0.9,
        coupler_limits=max_flow_coupler,
    )

    lf_res = SolverLoadflowResults(
        n_0_matrix=flow[:, :, 0, :],
        n_1_matrix=flow,
        cross_coupler_flows=cross_coupler_flow,
        branch_action_index=None,
        branch_topology=branch_topologies,
        sub_ids=sub_ids,
        injection_topology=injections,
        n_2_penalty=None,
        disconnections=None,
    )

    transport = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="transport_n_1",
    )
    transport_ref = get_transport_n_1_matrix(flow[0], max_mw_flow)

    assert transport.shape == (n_batch,)
    assert jnp.allclose(transport[0], transport_ref)

    overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="overload_energy_n_1",
    )
    overload_ref = get_overload_energy_n_1_matrix(flow[0], max_mw_flow)

    assert overload.shape == (n_batch,)
    assert jnp.allclose(overload[0], overload_ref)

    overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="overload_energy_limited_n_1",
    )
    overload_ref = get_overload_energy_n_1_matrix(flow[0], branch_limits.max_mw_flow_limited)

    assert overload.shape == (n_batch,)
    assert jnp.allclose(overload[0], overload_ref)

    overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="exponential_overload_energy_n_1",
    )
    overload_ref = get_exponential_overload_energy_n_1_matrix(flow[0], branch_limits.max_mw_flow)

    assert overload.shape == (n_batch,)
    assert jnp.allclose(overload[0], overload_ref)

    overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="exponential_overload_energy_limited_n_1",
    )
    overload_ref = get_exponential_overload_energy_n_1_matrix(flow[0], branch_limits.max_mw_flow_limited)

    assert overload.shape == (n_batch,)
    assert jnp.allclose(overload[0], overload_ref)

    assert overload.shape == (n_batch,)
    assert jnp.allclose(overload[0], overload_ref)

    underload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="underload_energy_n_1",
    )
    underload_ref = get_underload_energy_n_1_matrix(flow[0], max_mw_flow)

    assert underload.shape == (n_batch,)
    assert jnp.allclose(underload[0], underload_ref)

    max_flow = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="max_flow_n_1",
    )
    max_flow_ref = jnp.max(jnp.abs(flow / max_mw_flow), axis=(1, 2, 3))

    assert max_flow.shape == (n_batch,)
    assert jnp.allclose(max_flow, max_flow_ref)

    median_flow = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="median_flow_n_1",
    )
    median_flow_ref = get_median_flow_n_1_matrix(flow[0], max_mw_flow)

    assert median_flow.shape == (n_batch,)
    assert jnp.allclose(median_flow[0], median_flow_ref)

    critical_branch_count = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="critical_branch_count_n_1",
    )
    critical_branch_count_ref = get_critical_branch_count_n_1_matrix(flow[0], max_mw_flow)

    assert critical_branch_count.shape == (n_batch,)
    assert jnp.allclose(critical_branch_count[0], critical_branch_count_ref)

    critical_branch_count = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="critical_branch_count_limited_n_1",
    )
    critical_branch_count_ref = get_critical_branch_count_n_1_matrix(flow[0], branch_limits.max_mw_flow_limited)

    assert critical_branch_count.shape == (n_batch,)
    assert jnp.allclose(critical_branch_count[0], critical_branch_count_ref)

    cumulative_overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="cumulative_overload_n_1",
    )
    cumulative_overload_ref = get_cumulative_overload_n_1_matrix(flow[0], max_mw_flow)

    assert cumulative_overload.shape == (n_batch,)
    assert jnp.allclose(cumulative_overload[0], cumulative_overload_ref)

    cumulative_overload = aggregate_to_metric_batched(
        lf_res,
        branch_limits,
        None,
        n_subs_rel,
        metric="cumulative_overload_n_0",
    )
    cumulative_overload_ref = get_cumulative_overload_n_1_matrix(jnp.expand_dims(flow[0, :, 0, :], axis=1), max_mw_flow)

    assert cumulative_overload.shape == (n_batch,)
    assert jnp.allclose(cumulative_overload[0], cumulative_overload_ref)

    with pytest.raises(ValueError):
        aggregate_to_metric_batched(
            lf_res,
            branch_limits,
            None,
            n_subs_rel,
            metric="unknown_metric",
        )


def test_aggregate_to_metric() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50
    n_splits = 10
    n_subs_rel = 60
    max_branch_per_sub = 6
    max_inj_per_sub = 4

    keys = jax.random.split(jax.random.PRNGKey(0), 6)

    n_0 = jax.random.exponential(keys[0], (n_timesteps, n_branch))
    n_1 = jax.random.exponential(keys[1], (n_timesteps, n_failures, n_branch))
    cross_coupler = jax.random.exponential(keys[2], (n_splits, n_timesteps))
    max_mw_flow = jax.random.exponential(keys[2], (n_branch,))
    max_mw_flow_n_1 = jax.random.exponential(keys[3], (n_branch,))
    max_flow_coupler = jax.random.exponential(keys[4], (n_subs_rel,))
    branch_action_index = jax.random.randint(keys[3], (n_splits,), 0, 5)
    reassignment_distance = jax.random.randint(keys[4], (5,), 0, 100)
    branch_topologies = jax.random.randint(keys[5], (n_splits, max_branch_per_sub), 0, 2).astype(bool)
    sub_ids = jax.random.randint(keys[5], (n_splits,), 0, n_subs_rel)
    injections = jax.random.randint(keys[5], (n_splits, max_inj_per_sub), 0, 2).astype(bool)
    overload_weight = jax.random.exponential(keys[4], (n_branch,))
    n0_n1_max_diff = jax.random.exponential(keys[5], (n_branch,))
    n_2_penalty = jnp.array(12.3456)

    branch_limits = BranchLimits(
        max_mw_flow=max_mw_flow,
        max_mw_flow_n_1=max_mw_flow_n_1,
        overload_weight=overload_weight,
        n0_n1_max_diff=n0_n1_max_diff,
        coupler_limits=max_flow_coupler,
    )

    lf_res = SolverLoadflowResults(
        n_0_matrix=n_0,
        n_1_matrix=n_1,
        cross_coupler_flows=cross_coupler,
        branch_action_index=branch_action_index,
        branch_topology=branch_topologies,
        sub_ids=sub_ids,
        injection_topology=injections,
        n_2_penalty=n_2_penalty,
        disconnections=None,
    )

    for metric in [
        "max_flow_n_0",
        "median_flow_n_0",
        "overload_energy_n_0",
        "underload_energy_n_0",
        "transport_n_0",
        "exponential_overload_energy_n_0",
        "cumulative_overload_n_0",
    ]:
        res = aggregate_to_metric(
            lf_res,
            branch_limits,
            None,
            n_subs_rel,
            metric,
        )
        ref = aggregate_n_1_matrix(
            jnp.expand_dims(n_0, axis=1),
            max_mw_flow,
            metric.replace("_n_0", ""),
            overload_weight,
        )

        assert jnp.allclose(res, ref)

    for metric in [
        "max_flow_n_1",
        "median_flow_n_1",
        "overload_energy_n_1",
        "underload_energy_n_1",
        "transport_n_1",
        "exponential_overload_energy_n_1",
        "cumulative_overload_n_1",
    ]:
        res = aggregate_to_metric(
            lf_res,
            branch_limits,
            reassignment_distance,
            n_subs_rel,
            metric,
        )
        ref = aggregate_n_1_matrix(n_1, max_mw_flow_n_1, metric.replace("_n_1", ""), overload_weight)

        assert jnp.allclose(res, ref)

    res = aggregate_to_metric(
        lf_res,
        branch_limits,
        reassignment_distance,
        n_subs_rel,
        "n0_n1_delta",
    )
    ref = get_n0_n1_delta_penalty(n_0, n_1, n0_n1_max_diff)
    assert jnp.allclose(res, ref)

    res = aggregate_to_metric(
        lf_res,
        branch_limits,
        reassignment_distance,
        n_subs_rel,
        "cross_coupler_flow",
    )
    ref = get_cross_coupler_flow_penalty(cross_coupler, sub_ids, max_flow_coupler)
    assert jnp.allclose(res, ref)

    res = aggregate_to_metric(
        lf_res,
        branch_limits,
        reassignment_distance,
        n_subs_rel,
        "split_subs",
    )
    ref = get_number_of_splits(branch_topology=branch_topologies, sub_ids=sub_ids, n_relevant_subs=n_subs_rel)
    assert jnp.allclose(res, ref)

    res = aggregate_to_metric(lf_res, branch_limits, reassignment_distance, n_subs_rel, "n_2_penalty")
    assert res == n_2_penalty

    res = aggregate_to_metric(lf_res, branch_limits, reassignment_distance, n_subs_rel, "switching_distance")
    ref = get_switching_distance(branch_action_index, reassignment_distance)
    assert jnp.array_equal(res, ref)

    res = aggregate_to_metric(lf_res, branch_limits, reassignment_distance, n_subs_rel, "disconnected_branches")
    assert res == 0

    with pytest.raises(ValueError):
        aggregate_to_metric(
            lf_res=lf_res,
            branch_limits=branch_limits,
            reassignment_distance=reassignment_distance,
            n_relevant_subs=n_subs_rel,
            metric="unknown_metric",
        )

    with pytest.raises(ValueError):
        aggregate_to_metric(
            lf_res=replace(lf_res, n_2_penalty=None),
            branch_limits=branch_limits,
            reassignment_distance=reassignment_distance,
            n_relevant_subs=n_subs_rel,
            metric="n_2_penalty",
        )

    with pytest.raises(ValueError):
        aggregate_to_metric(
            lf_res,
            branch_limits=replace(branch_limits, n0_n1_max_diff=None),
            reassignment_distance=reassignment_distance,
            n_relevant_subs=n_subs_rel,
            metric="n0_n1_delta",
        )

    with pytest.raises(ValueError):
        aggregate_to_metric(
            lf_res,
            branch_limits=replace(branch_limits, coupler_limits=None),
            reassignment_distance=reassignment_distance,
            n_relevant_subs=n_subs_rel,
            metric="cross_coupler_flow",
        )


def test_compute_double_limits() -> None:
    n_timesteps = 5
    n_failures = 30
    n_branch = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    n_1 = jax.random.exponential(keys[0], (n_timesteps, n_failures, n_branch))
    max_mw_flow = jax.random.exponential(keys[1], (n_branch,))

    max_flows_new = compute_double_limits(n_1, max_mw_flow, lower_limit=0.9)

    max_flows_unchanged = compute_double_limits(n_1, max_mw_flow, lower_limit=1.0)
    assert jnp.allclose(max_flows_unchanged, max_mw_flow)

    assert max_flows_new.shape == max_mw_flow.shape
    assert jnp.all(max_flows_new >= 0.9 * max_mw_flow)
    assert jnp.all(max_flows_new <= max_mw_flow)

    overload_before = get_overload_energy_n_1_matrix(n_1, max_mw_flow)
    overload_after = get_overload_energy_n_1_matrix(n_1, max_flows_new)

    assert jnp.allclose(overload_after, overload_before)


def test_n0_n1_delta() -> None:
    assert (
        get_n0_n1_delta(
            jnp.array([[20.0]]),
            jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
            only_positive=False,
        ).item()
        == -10.0
    )
    assert (
        get_n0_n1_delta(
            jnp.array([[20.0]]),
            jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
            only_positive=True,
        ).item()
        == 3.0
    )

    assert (
        get_n0_n1_delta(
            -jnp.array([[20.0]]),
            -jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
            only_positive=False,
        ).item()
        == -10.0
    )
    assert (
        get_n0_n1_delta(
            -jnp.array([[20.0]]),
            -jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
            only_positive=True,
        ).item()
        == 3.0
    )

    assert get_n0_n1_delta_penalty(
        jnp.array([[20.0]]),
        jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
        jnp.array([1.9]),
    ).item() == (3.0 - 1.9)
    assert (
        get_n0_n1_delta_penalty(
            jnp.array([[20.0]]),
            jnp.array([[[10.0], [15.0], [20.0], [23.0]]]),
            jnp.array([-2]),
        ).item()
        == 0.0
    )


def test_cross_coupler_flow_penalty() -> None:
    n_timesteps = 5
    n_splits = 10
    n_rel_subs = 60

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    cross_coupler_flows = jax.random.normal(keys[0], (n_splits, n_timesteps))
    sub_ids = jax.random.randint(keys[1], (n_splits,), 0, n_rel_subs)
    coupler_limits = jnp.abs(jax.random.normal(keys[2], (n_rel_subs,)))

    penalty = get_cross_coupler_flow_penalty(cross_coupler_flows, sub_ids, jnp.zeros_like(coupler_limits))

    assert jnp.isclose(penalty, jnp.sum(jnp.abs(cross_coupler_flows)))

    penalty2 = get_cross_coupler_flow_penalty(cross_coupler_flows, sub_ids, coupler_limits)

    assert penalty2 <= penalty


def test_choose_max_mw_flow() -> None:
    n_branches = 30

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    max_mw_flow = jax.random.exponential(keys[0], (n_branches,))
    max_mw_flow_n_1 = jax.random.exponential(keys[1], (n_branches,))
    max_mw_flow_limited = jax.random.exponential(keys[2], (n_branches,))
    max_mw_flow_n_1_limited = jax.random.exponential(keys[3], (n_branches,))

    assert jnp.array_equal(
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=max_mw_flow_limited,
                max_mw_flow_n_1_limited=max_mw_flow_n_1_limited,
            ),
            "overload_energy_n_0",
        ),
        max_mw_flow,
    )

    assert jnp.array_equal(
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=max_mw_flow_limited,
                max_mw_flow_n_1_limited=max_mw_flow_n_1_limited,
            ),
            "overload_energy_n_1",
        ),
        max_mw_flow_n_1,
    )

    assert jnp.array_equal(
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=max_mw_flow_limited,
                max_mw_flow_n_1_limited=max_mw_flow_n_1_limited,
            ),
            "overload_energy_limited_n_0",
        ),
        max_mw_flow_limited,
    )

    assert jnp.array_equal(
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=max_mw_flow_limited,
                max_mw_flow_n_1_limited=max_mw_flow_n_1_limited,
            ),
            "overload_energy_limited_n_1",
        ),
        max_mw_flow_n_1_limited,
    )

    with pytest.raises(ValueError):
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=None,
                max_mw_flow_n_1_limited=max_mw_flow_n_1_limited,
            ),
            "overload_energy_limited_n_0",
        )

    assert jnp.array_equal(
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=max_mw_flow_limited,
                max_mw_flow_n_1_limited=None,
            ),
            "overload_energy_limited_n_1",
        ),
        max_mw_flow_limited,
    )

    with pytest.raises(ValueError):
        choose_max_mw_flow(
            BranchLimits(
                max_mw_flow=max_mw_flow,
                max_mw_flow_n_1=max_mw_flow_n_1,
                max_mw_flow_limited=None,
                max_mw_flow_n_1_limited=None,
            ),
            "overload_energy_limited_n_1",
        )


def test_compute_n0_n1_max_diff() -> None:
    n_timesteps = 2
    n_branches = 100
    n_outages = 50

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    n_0 = jax.random.exponential(keys[0], (n_timesteps, n_branches))
    n_1 = jax.random.exponential(keys[1], (n_timesteps, n_outages, n_branches))
    factors = jax.random.exponential(keys[2], (n_branches,))

    n0_n1_max_diff = compute_n0_n1_max_diff(n_0, n_1, factors)

    assert n0_n1_max_diff.shape == (n_branches,)


def test_get_number_of_disconnections() -> None:
    n_branches = 50
    max_n_disconnections = 10

    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    # Case 1: No disconnections provided
    disconnections = None
    assert get_number_of_disconnections(disconnections, n_branches) == 0

    # Case 2: Valid disconnections within range
    disconnections = jax.random.randint(keys[0], (max_n_disconnections,), 0, n_branches)
    assert get_number_of_disconnections(disconnections, n_branches) == max_n_disconnections

    # Case 3: Some disconnections out of range
    disconnections = jnp.array([0, 10, 20, 50, 60, -1, 30])
    expected_count = jnp.sum((disconnections >= 0) & (disconnections < n_branches))
    assert get_number_of_disconnections(disconnections, n_branches) == expected_count

    # Case 4: All disconnections out of range
    disconnections = jnp.array([n_branches, n_branches + 1, -5, -10])
    assert get_number_of_disconnections(disconnections, n_branches) == 0

    # Case 5: Empty disconnections array
    disconnections = jnp.array([])
    assert get_number_of_disconnections(disconnections, n_branches) == 0


def test_compute_metric_matches_jax(
    preprocessed_powsybl_data_folder: Path,
) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_definition = load_nminus1_definition(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    base_case = next((cont for cont in nminus1_definition.contingencies if cont.is_basecase()), None)
    assert base_case is not None, "Base case (N-0) not found in n-1 definition."

    lf_res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_definition, job_id="test_job", timestep=0, method="dc", polars=True
    )
    metrics = compute_metrics(lf_res, base_case_id=base_case.id)

    n_0, n_1, success = extract_solver_matrices_polars(lf_res, nminus1_definition=nminus1_definition, timestep=0)
    assert np.all(success)

    solver_max_flow = get_max_flow_n_1_matrix(n_1[None], static_information.dynamic_information.branch_limits.max_mw_flow)
    assert np.isclose(solver_max_flow, metrics["max_flow_n_1"])

    solver_overload = get_overload_energy_n_1_matrix(
        n_1[None], static_information.dynamic_information.branch_limits.max_mw_flow
    )
    assert np.isclose(solver_overload, metrics["overload_energy_n_1"])

    solver_max_flow_n_0 = get_max_flow_n_1_matrix(
        n_0[None, None], static_information.dynamic_information.branch_limits.max_mw_flow
    )
    assert np.isclose(solver_max_flow_n_0, metrics["max_flow_n_0"])

    solver_overload_n_0 = get_overload_energy_n_1_matrix(
        n_0[None, None], static_information.dynamic_information.branch_limits.max_mw_flow
    )
    assert np.isclose(solver_overload_n_0, metrics["overload_energy_n_0"])

    solver_critical_branches = get_critical_branch_count_n_1_matrix(
        n_1[None], static_information.dynamic_information.branch_limits.max_mw_flow
    )
    assert solver_critical_branches == metrics["critical_branch_count_n_1"]


def test_get_worst_n_k_contingency_basic() -> None:
    n_timesteps = 4
    n_failures = 10
    n_branches = 5
    k = 3

    key = jax.random.PRNGKey(42)
    n_1_matrix = jax.random.uniform(key, (n_timesteps, n_failures, n_branches))
    max_mw_flow = jnp.ones((n_branches,))

    worst_k_overload = get_worst_k_contingencies(k, n_1_matrix, max_mw_flow)
    assert worst_k_overload.top_k_overloads.shape == (n_timesteps,)
    assert worst_k_overload.case_indices.shape == (n_timesteps, k)
    assert jnp.all(worst_k_overload.top_k_overloads >= 0)


def test_get_pst_setpoint_deviation() -> None:
    """Test the PST setpoint deviation metric."""
    from toop_engine_dc_solver.jax.types import NodalInjOptimResults

    n_controllable_pst = 5
    n_timesteps = 3

    # Case 1: No PST optimization enabled (both None)
    deviation = get_pst_setpoint_deviation(optimized_taps=None, initial_tap_idx=None)
    assert deviation == 0.0, "Deviation should be 0 when PST optimization is disabled"

    # Case 2: PST optimization enabled but no initial taps provided
    optimized_taps = NodalInjOptimResults(
        pst_tap_idx=jnp.array([[0, 1, 2, 3, 4]], dtype=float)  # shape: (n_timesteps=1, n_controllable_pst)
    )
    deviation = get_pst_setpoint_deviation(optimized_taps=optimized_taps, initial_tap_idx=None)
    assert deviation == 0.0, "Deviation should be 0 when initial tap indices are not provided"

    # Case 3: No deviation - optimized taps match initial taps
    initial_tap_idx = jnp.array([2, 3, 4, 5, 6], dtype=int)
    optimized_taps = NodalInjOptimResults(
        pst_tap_idx=jnp.array([[2, 3, 4, 5, 6]], dtype=float)  # shape: (n_timesteps=1, n_controllable_pst)
    )
    deviation = get_pst_setpoint_deviation(optimized_taps=optimized_taps, initial_tap_idx=initial_tap_idx)
    assert deviation == 0.0, "Deviation should be 0 when taps haven't changed"

    # Case 4: Simple deviation case - single timestep
    initial_tap_idx = jnp.array([2, 3, 4, 5, 6], dtype=int)
    optimized_taps = NodalInjOptimResults(
        pst_tap_idx=jnp.array([[3, 4, 5, 6, 7]], dtype=float)  # All shifted by +1
    )
    deviation = get_pst_setpoint_deviation(optimized_taps=optimized_taps, initial_tap_idx=initial_tap_idx)
    expected_deviation = 5.0  # Sum of |3-2| + |4-3| + |5-4| + |6-5| + |7-6| = 5
    assert deviation == expected_deviation, f"Expected deviation {expected_deviation}, got {deviation}"

    # Case 5: Mixed positive and negative deviations
    initial_tap_idx = jnp.array([5, 5, 5, 5, 5], dtype=int)
    optimized_taps = NodalInjOptimResults(
        pst_tap_idx=jnp.array([[3, 7, 5, 4, 8]], dtype=float)  # Deviations: -2, +2, 0, -1, +3
    )
    deviation = get_pst_setpoint_deviation(optimized_taps=optimized_taps, initial_tap_idx=initial_tap_idx)
    expected_deviation = 2.0 + 2.0 + 0.0 + 1.0 + 3.0  # L1 distance = 8.0
    assert deviation == expected_deviation, f"Expected deviation {expected_deviation}, got {deviation}"

    # Case 6: Multiple timesteps - should use first timestep only
    initial_tap_idx = jnp.array([2, 3, 4, 5, 6], dtype=int)
    optimized_taps = NodalInjOptimResults(
        pst_tap_idx=jnp.array(
            [
                [3, 4, 5, 6, 7],  # First timestep: deviation = 5
                [10, 10, 10, 10, 10],  # Second timestep: would be higher deviation
                [0, 0, 0, 0, 0],  # Third timestep: would be different deviation
            ],
            dtype=float,
        )  # shape: (n_timesteps=3, n_controllable_pst=5)
    )
    deviation = get_pst_setpoint_deviation(optimized_taps=optimized_taps, initial_tap_idx=initial_tap_idx)
    expected_deviation = 5.0  # Only first timestep should be used
    assert deviation == expected_deviation, f"Expected deviation {expected_deviation}, got {deviation}"

    # Case 7: Verify JAX compatibility (can be JIT compiled)
    @jax.jit
    def jitted_deviation(optimized_taps, initial_tap_idx):
        return get_pst_setpoint_deviation(optimized_taps, initial_tap_idx)

    initial_tap_idx = jnp.array([1, 2, 3], dtype=int)
    optimized_taps = NodalInjOptimResults(pst_tap_idx=jnp.array([[2, 3, 4]], dtype=float))
    
    deviation_jitted = jitted_deviation(optimized_taps, initial_tap_idx)
    deviation_normal = get_pst_setpoint_deviation(optimized_taps, initial_tap_idx)
    
    assert jnp.allclose(deviation_jitted, deviation_normal), "JIT and non-JIT versions should produce same result"
    assert deviation_jitted == 3.0, f"Expected deviation 3.0, got {deviation_jitted}"
