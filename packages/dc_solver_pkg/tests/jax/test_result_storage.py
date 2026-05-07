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
)
from toop_engine_dc_solver.jax.types import (
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
