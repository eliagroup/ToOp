# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.benchmarks.benchmarks import (
    bench_inj_ratio,
    bench_symmetric,
    load_static_information_from_dict,
    run_benchmark,
)
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)


def test_bench_symmetric(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    batch_size = 16
    n_topologies_per_run = 64
    n_disconnections_per_topology = 2
    n_runs = 3
    limit_n_subs = 3

    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            batch_size_bsdf=batch_size,
            batch_size_injection=batch_size,
            limit_n_subs=limit_n_subs,
        ),
        dynamic_information=replace(
            static_information.dynamic_information,
            disconnectable_branches=jnp.array([1, 4, 8, 12]),
        ),
    )

    rng_key = jax.random.PRNGKey(0)
    res = bench_symmetric(
        rng_key,
        static_information,
        n_topologies_per_run,
        n_disconnections_per_topology,
        n_runs,
    )
    assert "times" in res
    assert len(res["times"]) == n_runs


def test_bench_inj_ratio(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    batch_size = 16
    n_topologies_per_run = 64
    n_disconnections_per_topology = 2
    n_injections_per_topology = 100
    n_runs = 3
    limit_n_subs = 3

    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            batch_size_bsdf=batch_size,
            batch_size_injection=batch_size,
            buffer_size_injection=None,  # Upper bound
            limit_n_subs=limit_n_subs,
        ),
        dynamic_information=replace(
            static_information.dynamic_information,
            disconnectable_branches=jnp.array([1, 4, 8, 12]),
        ),
    )

    rng_key = jax.random.PRNGKey(0)
    res = bench_inj_ratio(
        rng_key,
        static_information,
        n_topologies_per_run,
        n_disconnections_per_topology,
        n_runs,
        n_injections_per_topology,
    )
    assert "times" in res
    assert len(res["times"]) == n_runs


def test_load_static_information_from_dict(benchmark_config: dict) -> None:
    static_information = load_static_information_from_dict(benchmark_config["benchmarks"][0])
    assert isinstance(static_information, StaticInformation)
    assert static_information.dynamic_information.action_set is not None


def test_run_benchmark(benchmark_config: dict) -> None:
    benchmark = benchmark_config["benchmarks"][0]
    res = run_benchmark(benchmark)
    assert "times" in res
    assert len(res["times"]) == benchmark["bench_runner_config"]["n_runs"]
