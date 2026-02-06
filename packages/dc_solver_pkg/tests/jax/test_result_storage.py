# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import pytest
from toop_engine_dc_solver.jax.result_storage import (
    get_best_for_topologies,
    get_worst_failures,
    prepare_result_storage,
    update_aggregate_metrics,
    update_aggregate_results,
)
from toop_engine_dc_solver.jax.topology_looper import (
    DefaultAggregateMetricsFn,
    DefaultAggregateOutputFn,
)
from toop_engine_dc_solver.jax.types import (
    BranchLimits,
    SolverLoadflowResults,
)


def test_prepare_result_storage() -> None:
    def aggregate_output_fn(lf_res: SolverLoadflowResults):
        return {
            "n_0": lf_res.n_0_matrix,
            "n_1": lf_res.n_1_matrix,
            "worst": jnp.max(lf_res.n_1_matrix),
            "cross_coupler": lf_res.cross_coupler_flows,
            "topos": lf_res.branch_topology,
            "sub_ids": lf_res.sub_ids,
            "inj": lf_res.injection_topology,
        }

    n_timesteps = 4
    n_failures = 30
    n_branches_monitored = 20
    batch = 23
    n_sub_relevant = 5
    max_branch_per_sub = 6
    max_inj_per_sub = 3
    n_splits = 3
    nminus2 = False
    bb_outage = False
    storage = prepare_result_storage(
        aggregate_output_fn=aggregate_output_fn,
        n_timesteps=n_timesteps,
        n_branches_monitored=n_branches_monitored,
        n_failures=n_failures,
        n_splits=n_splits,
        n_disconnections=None,
        max_branch_per_sub=max_branch_per_sub,
        max_inj_per_sub=max_inj_per_sub,
        nminus2=nminus2,
        size=batch,
        bb_outage=bb_outage,
    )

    assert storage["n_0"].shape == (batch, n_timesteps, n_branches_monitored)
    assert storage["n_1"].shape == (
        batch,
        n_timesteps,
        n_failures,
        n_branches_monitored,
    )
    assert storage["worst"].shape == (batch,)
    assert storage["cross_coupler"].shape == (batch, n_splits, n_timesteps)
    assert storage["sub_ids"].shape == (batch, n_splits)
    assert storage["topos"].shape == (batch, n_splits, max_branch_per_sub)
    assert storage["inj"].shape == (batch, n_splits, max_inj_per_sub)


def test_get_best_for_topologies() -> None:
    key = jax.random.PRNGKey(0)
    n_topologies = 16
    n_injections = 1000

    corresponding_topologies = jax.random.randint(key, (n_injections,), 0, n_topologies)
    worst_failures = jax.random.normal(key, (n_injections,))

    res = get_best_for_topologies(corresponding_topologies, worst_failures, n_topologies)
    assert res.shape == (n_topologies,)
    assert jnp.all(res >= 0)
    assert jnp.all(res < n_injections)
    assert jnp.unique(res).shape[0] == n_topologies


def test_update_aggregate_results() -> None:
    n_timesteps = 2
    n_failures = 30
    n_branches_monitored = 20
    n_splits = 3
    number_most_affected = 20
    number_max_out_in_most_affected = 5
    number_most_affected_n_0 = 10
    n_subs_rel = 5
    data_size = 512
    batch_size = 8
    aggregate_size = 16
    max_branch_per_sub = 6
    max_inj_per_sub = 3

    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    n_0_matrix = jax.random.exponential(keys[0], (data_size, n_timesteps, n_branches_monitored))
    n_1_matrix = jax.random.exponential(keys[1], (data_size, n_timesteps, n_failures, n_branches_monitored))
    topologies = jax.random.randint(keys[2], shape=(data_size, n_splits, max_branch_per_sub), minval=0, maxval=2).astype(
        bool
    )
    sub_ids = jax.random.randint(keys[2], shape=(data_size, n_splits), minval=0, maxval=10)
    injections = jax.random.randint(keys[3], shape=(data_size, n_splits, max_inj_per_sub), minval=0, maxval=2).astype(bool)
    corresponding_topology = jax.random.randint(keys[4], shape=(data_size,), minval=0, maxval=aggregate_size)

    output_fn = DefaultAggregateOutputFn(
        branches_to_fail=jnp.arange(n_failures),
        multi_outage_indices=jnp.array([], dtype=int),
        injection_outage_indices=jnp.array([], dtype=int),
        max_mw_flow=jnp.ones(n_branches_monitored),
        number_most_affected=number_most_affected,
        number_max_out_in_most_affected=number_max_out_in_most_affected,
        number_most_affected_n_0=number_most_affected_n_0,
        fixed_hash=0,
    )
    metric_fn = DefaultAggregateMetricsFn(
        branch_limits=BranchLimits(max_mw_flow=jnp.ones(n_branches_monitored)),
        reassignment_distance=None,
        metric="max_flow_n_1",
        n_relevant_subs=n_subs_rel,
        fixed_hash=0,
    )

    results_acc = prepare_result_storage(
        aggregate_output_fn=output_fn,
        n_timesteps=n_timesteps,
        n_branches_monitored=n_branches_monitored,
        n_failures=n_failures,
        n_splits=n_splits,
        n_disconnections=None,
        max_branch_per_sub=max_branch_per_sub,
        max_inj_per_sub=max_inj_per_sub,
        nminus2=False,
        bb_outage=False,
        size=aggregate_size,
    )
    best_inj_acc = jnp.zeros((aggregate_size, n_splits, max_inj_per_sub))
    metrics_acc = jnp.full((aggregate_size,), jnp.inf)

    for batch_id in range(data_size // batch_size):
        slice_start = batch_id * batch_size
        slice_end = (batch_id + 1) * batch_size
        lf_res = SolverLoadflowResults(
            n_0_matrix=n_0_matrix[slice_start:slice_end],
            n_1_matrix=n_1_matrix[slice_start:slice_end],
            cross_coupler_flows=None,
            branch_action_index=None,
            branch_topology=topologies[slice_start:slice_end],
            sub_ids=sub_ids[slice_start:slice_end],
            injection_topology=injections[slice_start:slice_end],
            n_2_penalty=None,
            disconnections=None,
        )
        results_cur = jax.vmap(output_fn)(lf_res)
        metrics_cur = jax.vmap(metric_fn)(
            lf_res,
            None,
        )

        results_acc, best_inj_acc, metrics_acc = update_aggregate_results(
            injections=injections[slice_start:slice_end],
            corresponding_topology=corresponding_topology[slice_start:slice_end],
            results_cur=results_cur,
            metrics_cur=metrics_cur,
            pad_mask=jnp.ones(batch_size, dtype=bool),
            results_acc=results_acc,
            best_inj_acc=best_inj_acc,
            metrics_acc=metrics_acc,
        )

    lf_res = SolverLoadflowResults(
        n_0_matrix=n_0_matrix,
        n_1_matrix=n_1_matrix,
        cross_coupler_flows=None,
        branch_action_index=None,
        branch_topology=topologies,
        sub_ids=sub_ids,
        injection_topology=injections,
        n_2_penalty=None,
        disconnections=None,
    )
    metrics_ref = jax.vmap(metric_fn)(lf_res, None)
    sparse_n_0_ref, sparse_n_1_ref = jax.vmap(output_fn)(lf_res)
    sparse_n_0, sparse_n_1 = results_acc
    for topology in range(aggregate_size):
        mask = corresponding_topology == topology
        best_metric_idx = jnp.argmin(jnp.where(mask, metrics_ref, jnp.inf))
        assert jnp.isclose(metrics_acc[topology], metrics_ref[best_metric_idx]).item()
        assert jnp.allclose(sparse_n_0.pf_n_0_max[topology], sparse_n_0_ref.pf_n_0_max[best_metric_idx])
        assert jnp.array_equal(sparse_n_0.hist_mon[topology], sparse_n_0_ref.hist_mon[best_metric_idx])
        assert jnp.allclose(sparse_n_1.pf_n_1_max[topology], sparse_n_1_ref.pf_n_1_max[best_metric_idx])


def test_update_aggregate_metrics() -> None:
    batch_size_bsdf = 16
    batch_size_injection = 64
    buffer_size_injection = 32
    n_splits = 5
    max_inj_per_sub = 3

    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    metrics = jax.random.normal(keys[0], (buffer_size_injection, batch_size_injection))
    corresponding_topology = jax.random.choice(keys[1], batch_size_bsdf, shape=(buffer_size_injection, batch_size_injection))

    metrics_acc = jnp.full((batch_size_bsdf,), jnp.inf)
    for buffer_id in range(buffer_size_injection):
        metrics_acc, best_inj = update_aggregate_metrics(
            injections=jnp.zeros((batch_size_injection, n_splits, max_inj_per_sub)),
            corresponding_topology=corresponding_topology[buffer_id],
            metric=metrics[buffer_id],
            pad_mask=jnp.ones((batch_size_injection,), dtype=bool),
            metrics_acc=metrics_acc,
            best_inj_acc=jnp.zeros((batch_size_bsdf, n_splits, max_inj_per_sub)),
        )

        assert best_inj.shape == (batch_size_bsdf, n_splits, max_inj_per_sub)

    # We should get the minimum metric for each topology
    for topo in range(batch_size_bsdf):
        topo_mask = corresponding_topology == topo
        if not jnp.any(topo_mask).item():
            continue
        assert jnp.all(metrics_acc[topo] == metrics[topo_mask].min())


def test_get_worst_failures() -> None:
    n_timesteps = 1
    n_branch = 10
    n_failures = 5

    n_1_matrix = jax.random.normal(jax.random.PRNGKey(0), (n_timesteps, n_failures, n_branch))

    worst_failures = get_worst_failures(
        n_1_matrix=n_1_matrix,
        branches_to_fail=jax.random.choice(jax.random.PRNGKey(0), n_branch, shape=(n_failures,), replace=False),
        number_most_affected=3,
        number_max_out_in_most_affected=2,
    )

    assert worst_failures.pf_n_1_max.shape == (
        1,
        3,
    )
    assert jnp.max(n_1_matrix) == jnp.max(worst_failures.pf_n_1_max)

    with pytest.raises(ValueError):
        get_worst_failures(
            n_1_matrix=n_1_matrix,
            branches_to_fail=jax.random.choice(jax.random.PRNGKey(0), n_branch, shape=(n_failures,), replace=False),
            number_most_affected=100,
            number_max_out_in_most_affected=1,
        )
