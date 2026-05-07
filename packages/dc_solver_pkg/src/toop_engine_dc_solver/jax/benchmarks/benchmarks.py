# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Runs symmetric batches for benchmarking."""

import time

import jax
import jax.numpy as jnp
import structlog
from jax_dataclasses import replace
from jaxtyping import PRNGKeyArray
from toop_engine_dc_solver.jax.disconnections import random_disconnections
from toop_engine_dc_solver.jax.injections import (
    random_injection,
)
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import convert_action_set_index_to_topo, random_topology
from toop_engine_dc_solver.jax.topology_looper import (
    DefaultAggregateOutputFn,
    run_solver_symmetric,
)
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import extract_static_information_stats

logger = structlog.get_logger(__name__)


def load_static_information_from_dict(config: dict) -> StaticInformation:
    """Load the static information from a dictionary in the format of the yaml config.

    Parameters
    ----------
    config : dict
        The dictionary get static information path and hyperparameters from

    Returns
    -------
    StaticInformation
        The static information with hyperparameters set and branch actions computed
    """
    static_information = load_static_information(config["static_information_path"])
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, **config["hyperparameters"]),
    )
    assert static_information.dynamic_information.action_set is not None

    return static_information


def run_benchmark(config: dict) -> dict:
    """Run the benchmark with the given configuration.

    Parameters
    ----------
    config : dict
        The configuration for the benchmark

    Returns
    -------
    dict
        The results of the benchmark
    """
    logger.info(f"Initializing benchmark {config['name']}")
    name_mapper = {
        "symmetric": bench_symmetric,
    }

    static_information = load_static_information_from_dict(config)
    bench_name = config["method"]
    results = name_mapper[bench_name](
        rng_key=jax.random.PRNGKey(config["random_seed"]),
        static_information=static_information,
        **config["bench_runner_config"],
    )

    results.update(extract_static_information_stats(static_information).model_dump())
    results["config"] = config
    return results


def bench_symmetric(
    rng_key: PRNGKeyArray,
    static_information: StaticInformation,
    n_topologies_per_run: int,
    n_disconnections_per_topology: int,
    n_runs: int,
) -> dict:
    """Benchmarks run_solver with symmetric batches.

    Parameters
    ----------
    rng_key : PRNGKeyArray
        The random key to use for generating the topologies
    static_information : StaticInformation
        The static information to run the benchmark with. The batch_size_bsdf will be used, which
        should divide n_topologies_per_run.
    n_topologies_per_run : int
        The number of topologies to generate per run
    n_disconnections_per_topology: int
        The number of disconnections per topology
    n_runs : int
        The number of runs to perform

    Returns
    -------
    dict
        The results of the benchmark
    """
    logger.info(f"Running symmetric benchmark for {n_runs} runs with {n_topologies_per_run} topologies per run")
    limit_n_subs = static_information.solver_config.limit_n_subs

    # Run the benchmark
    times = []
    for _ in range(n_runs):
        key1, key2, key3, rng_key = jax.random.split(rng_key, 4)
        topologies = random_topology(
            rng_key=key1,
            branch_action_set=static_information.dynamic_information.action_set,
            limit_n_subs=limit_n_subs,
            batch_size=n_topologies_per_run,
            topo_vect_format=False,
        )
        disconnections = (
            random_disconnections(
                key2,
                batch_size=n_topologies_per_run,
                n_disconnections=n_disconnections_per_topology,
                disconnectable_branches=static_information.dynamic_information.disconnectable_branches,
            )
            if n_disconnections_per_topology > 0
            else None
        )
        injections = random_injection(
            key3,
            n_generators_per_sub=static_information.dynamic_information.generators_per_sub,
            n_inj_per_topology=1,
            for_topology=convert_action_set_index_to_topo(
                topologies=topologies,
                action_set=static_information.dynamic_information.action_set,
            ),
        )

        injections.injection_topology.block_until_ready()
        start = time.time()
        _res, success = run_solver_symmetric(
            topologies=topologies,
            disconnections=disconnections,
            injections=injections.injection_topology,
            dynamic_information=static_information.dynamic_information,
            solver_config=static_information.solver_config,
            aggregate_output_fn=DefaultAggregateOutputFn(
                branches_to_fail=static_information.dynamic_information.branches_to_fail,
                multi_outage_indices=jnp.arange(static_information.dynamic_information.n_multi_outages)
                + jnp.max(static_information.dynamic_information.branches_to_fail),
                injection_outage_indices=jnp.arange(static_information.dynamic_information.n_inj_failures)
                + jnp.max(static_information.dynamic_information.branches_to_fail)
                + static_information.dynamic_information.n_multi_outages,
                max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow,
                number_most_affected=static_information.solver_config.number_most_affected,
                number_max_out_in_most_affected=static_information.solver_config.number_max_out_in_most_affected,
                number_most_affected_n_0=static_information.solver_config.number_most_affected_n_0,
                fixed_hash=hash(static_information),
            ),
        )
        success.block_until_ready()
        times.append(time.time() - start)

    return {"times": times}
