import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax_dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
for pkg in [
    "packages/dc_solver_pkg/src",
    "packages/topology_optimizer_pkg/src",
    "packages/interfaces_pkg/src",
    "packages/grid_helpers_pkg/src",
    "packages/importer_pkg/src",
    "packages/contingency_analysis_pkg/src",
]:
    sys.path.insert(0, str(ROOT / pkg))

from fsspec.implementations.dirfs import DirFileSystem  # noqa: E402
from toop_engine_dc_solver.example_grids import oberrhein_data  # noqa: E402
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric_batched, get_worst_k_contingencies  # noqa: E402
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits  # noqa: E402
from toop_engine_dc_solver.jax.compute_batch import (  # noqa: E402
    compute_bsdf_lodf_static_flows,
    compute_injections,
    compute_symmetric_batch,
)
from toop_engine_dc_solver.jax.contingency_analysis import (  # noqa: E402
    BatchedContingencyAnalysisParams,
    UnBatchedContingencyAnalysisParams,
    calc_injection_outages,
    calc_n_1_matrix,
    contingency_analysis_matrix,
)
from toop_engine_dc_solver.jax.cross_coupler_flow import compute_cross_coupler_flows  # noqa: E402
from toop_engine_dc_solver.jax.disconnections import apply_disconnections, update_n0_flows_after_disconnections  # noqa: E402
from toop_engine_dc_solver.jax.injections import (  # noqa: E402
    get_all_injection_outage_deltap,
    get_all_outaged_injection_nodes_after_reassignment,
)
from toop_engine_dc_solver.jax.inputs import load_static_information  # noqa: E402
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix  # noqa: E402
from toop_engine_dc_solver.jax.multi_outages import build_modf_matrices  # noqa: E402
from toop_engine_dc_solver.jax.nodal_inj_optim import apply_pst_taps, nodal_inj_optimization  # noqa: E402
from toop_engine_dc_solver.jax.topology_computations import convert_action_set_index_to_topo  # noqa: E402
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax  # noqa: E402
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend  # noqa: E402
from toop_engine_dc_solver.preprocess.preprocess import preprocess  # noqa: E402
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (  # noqa: E402
    initialize_genetic_algorithm,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (  # noqa: E402
    DisconnectionMutationConfig,
    MutationConfig,
    NodalInjectionMutationConfig,
    SubstationMutationConfig,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import (  # noqa: E402
    create_random_topology,
    mutate,
    mutate_topology,
    repeat_topologies,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_disconnections import (  # noqa: E402
    change_disconnected_branch,
    disconnect_additional_branch,
    mutate_disconnections,
    reconnect_disconnected_branch,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_nodal_inj import (  # noqa: E402
    mutate_nodal_injections,
    mutate_psts,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_substations import (  # noqa: E402
    change_split_substation,
    mutate_sub_splits,
    split_additional_sub,
    unsplit_substation,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (  # noqa: E402
    compute_overloads,
    translate_topology,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (  # noqa: E402
    add_to_repertoire,
)
from toop_engine_topology_optimizer.dc.worker.optimizer import (  # noqa: E402
    run_single_device_epoch,
    run_single_iteration,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef  # noqa: E402
# flake8-in-file-ignores: noqa: PLW0108,T201 (inline function call and no prints)


def _block(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _bench_jitted(name, fn, *args, runs=5):
    t0 = time.perf_counter()
    out = fn(*args)
    _block(out)
    compile_plus_first_ms = (time.perf_counter() - t0) * 1000.0

    exec_times = []
    for _ in range(runs):
        t1 = time.perf_counter()
        out = fn(*args)
        _block(out)
        exec_times.append((time.perf_counter() - t1) * 1000.0)

    return {
        "name": name,
        "compile_plus_first_ms": round(compile_plus_first_ms, 3),
        "median_exec_ms": round(statistics.median(exec_times), 3),
        "all_exec_ms": [round(x, 3) for x in exec_times],
    }


BASELINE_OBSERVED_METRICS = ("overload_energy_n_1", "split_subs")


def _get_supported_observed_metrics(dynamic_information, solver_config):
    branch_limits = dynamic_information.branch_limits

    metrics = [
        "max_flow_n_0",
        "median_flow_n_0",
        "overload_energy_n_0",
        "underload_energy_n_0",
        "exponential_overload_energy_n_0",
        "critical_branch_count_n_0",
        "cumulative_overload_n_0",
        "transport_n_0",
        "max_flow_n_1",
        "median_flow_n_1",
        "overload_energy_n_1",
        "underload_energy_n_1",
        "exponential_overload_energy_n_1",
        "critical_branch_count_n_1",
        "cumulative_overload_n_1",
        "transport_n_1",
    ]

    if branch_limits.max_mw_flow_limited is not None:
        metrics.extend(
            [
                "overload_energy_limited_n_0",
                "exponential_overload_energy_limited_n_0",
            ]
        )

    if branch_limits.max_mw_flow_n_1_limited is not None or branch_limits.max_mw_flow_limited is not None:
        metrics.extend(
            [
                "overload_energy_limited_n_1",
                "exponential_overload_energy_limited_n_1",
                "critical_branch_count_limited_n_1",
            ]
        )

    metrics.extend(
        [
            "switching_distance",
            "split_subs",
            "disconnected_branches",
        ]
    )

    if dynamic_information.branch_limits.coupler_limits is not None:
        metrics.append("cross_coupler_flow")

    if branch_limits.n0_n1_max_diff is not None:
        metrics.append("n0_n1_delta")

    if dynamic_information.nodal_injection_information is not None:
        metrics.append("pst_switching_distance")

    if dynamic_information.n2_baseline_analysis is not None:
        metrics.append("n_2_penalty")

    if solver_config.enable_bb_outages and not solver_config.bb_outage_as_nminus1:
        metrics.extend(["bb_outage_penalty", "bb_outage_overload", "bb_outage_grid_splits"])

    deduped_metrics = []
    for metric in metrics:
        if metric not in deduped_metrics:
            deduped_metrics.append(metric)
    return tuple(deduped_metrics)


def _initialize_small_optimizer(static_information_file: Path, batch_size: int):
    static_information = _load_or_generate_static_information(static_information_file)
    dynamic_information = static_information.dynamic_information

    mutation_config = MutationConfig(
        mutation_repetition=1,
        random_topo_prob=0.0,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.3,
            change_split_prob=0.4,
            remove_split_prob=0.3,
            n_rel_subs=int(dynamic_information.n_sub_relevant),
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.2,
            change_disconnection_prob=0.6,
            remove_disconnection_prob=0.2,
            n_disconnectable_branches=int(dynamic_information.disconnectable_branches.shape[0]),
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=0.0,
            pst_n_taps=dynamic_information.nodal_injection_information.pst_n_taps,
        )
        if dynamic_information.nodal_injection_information is not None
        else None,
    )

    algo, jax_data = initialize_genetic_algorithm(
        batch_size=batch_size,
        max_num_splits=3,
        max_num_disconnections=2,
        static_informations=(static_information,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        action_set=dynamic_information.action_set,
        proportion_crossover=0.5,
        crossover_mutation_ratio=0.5,
        random_seed=42,
        observed_metrics=("overload_energy_n_1", "split_subs", "switching_distance"),
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=8),
            DescriptorDef(metric="switching_distance", num_cells=40),
        ),
        distributed=False,
        mutation_config=mutation_config,
    )
    return algo, jax_data


def _load_or_generate_static_information(static_information_file: Path):
    try:
        return load_static_information(str(static_information_file))
    except Exception:
        tmp_dir = Path(tempfile.mkdtemp(prefix="toop_benchmark_oberrhein_"))
        oberrhein_data(tmp_dir)
        backend = PandaPowerBackend(DirFileSystem(str(tmp_dir)))
        network_data = preprocess(backend)
        return convert_to_jax(network_data, enable_bb_outage=False)


def _default_static_information_file() -> Path:
    return ROOT / "data" / "test_grid_node_breaker" / "static_information.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark run_single_device_epoch and nested JAX functions.")
    parser.add_argument(
        "--static-information-file",
        type=Path,
        default=_default_static_information_file(),
        help="Path to static_information.hdf5 (default: data/test_grid_node_breaker/static_information.hdf5)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations-per-epoch", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if not args.static_information_file.exists():
        raise FileNotFoundError(f"Static information file not found: {args.static_information_file}")

    algo, jax_data = _initialize_small_optimizer(args.static_information_file, batch_size=args.batch_size)
    static_information = _load_or_generate_static_information(args.static_information_file)
    dynamic_information = jax_data.dynamic_informations[0]
    solver_config = replace(static_information.solver_config, batch_size_bsdf=args.batch_size)
    supported_observed_metrics = _get_supported_observed_metrics(dynamic_information, solver_config)
    additional_observed_metrics = tuple(
        metric for metric in supported_observed_metrics if metric not in BASELINE_OBSERVED_METRICS
    )
    mutation_config = MutationConfig(
        mutation_repetition=1,
        random_topo_prob=0.0,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.3,
            change_split_prob=0.4,
            remove_split_prob=0.3,
            n_rel_subs=int(dynamic_information.n_sub_relevant),
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.2,
            change_disconnection_prob=0.6,
            remove_disconnection_prob=0.2,
            n_disconnectable_branches=int(dynamic_information.disconnectable_branches.shape[0]),
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=0.0,
            pst_n_taps=dynamic_information.nodal_injection_information.pst_n_taps,
        )
        if dynamic_information.nodal_injection_information is not None
        else None,
    )
    action_set = dynamic_information.action_set

    # Prepare staged data for nested function benchmarks.
    genotypes, _extra, random_key_next = algo._emitter.emit(jax_data.repertoire, jax_data.emitter_state, jax_data.random_key)
    fitnesses, descriptors, extra_scores, emitter_info, _rk2, genotypes_scored = algo._scoring_function(
        genotypes, random_key_next, jax_data.dynamic_informations
    )
    topo_comp, disconnections, nodal_inj_start = translate_topology(genotypes_scored)
    bitvector_topology = convert_action_set_index_to_topo(topo_comp, dynamic_information.action_set)
    translated_disconnections = dynamic_information.disconnectable_branches.at[disconnections].get(
        mode="fill", fill_value=jnp.iinfo(disconnections.dtype).max
    )
    bsdf_only_res = jax.vmap(
        partial(
            compute_bus_splits,
            ptdf=dynamic_information.ptdf,
            from_node=dynamic_information.from_node,
            to_node=dynamic_information.to_node,
            tot_stat=dynamic_information.tot_stat,
            from_stat_bool=dynamic_information.from_stat_bool,
            susceptance=dynamic_information.susceptance,
            rel_stat_map=solver_config.rel_stat_map,
            slack=solver_config.slack,
            n_stat=solver_config.n_stat,
        )
    )(bitvector_topology.topologies, bitvector_topology.sub_ids)
    disconnection_res = jax.vmap(
        partial(
            apply_disconnections,
            guarantee_unique=True,
        ),
        in_axes=(0, 0, 0, 0),
    )(
        bsdf_only_res.ptdf,
        bsdf_only_res.from_node,
        bsdf_only_res.to_node,
        translated_disconnections,
    )
    sub_ids = jnp.where(
        bitvector_topology.topologies.any(axis=-1),
        bitvector_topology.sub_ids,
        jnp.iinfo(bitvector_topology.sub_ids.dtype).max,
    )
    injections = dynamic_information.action_set.inj_actions.at[topo_comp.action].get(mode="fill", fill_value=False)
    topo_res = compute_bsdf_lodf_static_flows(
        bitvector_topology, translated_disconnections, dynamic_information, solver_config
    )
    nodal_injections = compute_injections(injections, sub_ids, dynamic_information, solver_config)
    n_0_raw, cross_coupler_flows = jax.vmap(
        compute_cross_coupler_flows,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(
        topo_res.bsdf,
        bitvector_topology.topologies,
        sub_ids,
        injections,
        dynamic_information.relevant_injections,
        dynamic_information.unsplit_flow,
        dynamic_information.tot_stat,
        dynamic_information.from_stat_bool,
    )
    n_0 = jax.vmap(update_n0_flows_after_disconnections)(n_0_raw, topo_res.disconnection_modf)
    n_0_flow_monitors = n_0.at[:, :, dynamic_information.branches_monitored].get(mode="fill", fill_value=jnp.nan)
    unbatched_params = UnBatchedContingencyAnalysisParams(
        branches_to_fail=dynamic_information.branches_to_fail,
        injection_outage_deltap=get_all_injection_outage_deltap(
            injection_outage_deltap=dynamic_information.nonrel_injection_outage_deltap,
            relevant_injections=dynamic_information.relevant_injections,
            relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
            relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        ),
        branches_monitored=dynamic_information.branches_monitored,
        action_set=dynamic_information.action_set,
        non_rel_bb_outage_data=dynamic_information.non_rel_bb_outage_data,
        enable_bb_outages=solver_config.enable_bb_outages and solver_config.bb_outage_as_nminus1,
    )
    batched_params = BatchedContingencyAnalysisParams(
        lodf=topo_res.lodf,
        ptdf=topo_res.ptdf,
        modf=topo_res.outage_modf,
        nodal_injections=nodal_injections,
        n_0_flow=n_0,
        injection_outage_node=get_all_outaged_injection_nodes_after_reassignment(
            injection_assignment=injections,
            sub_ids=sub_ids,
            relevant_injections=dynamic_information.relevant_injections,
            relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
            relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
            nonrel_injection_outage_node=dynamic_information.nonrel_injection_outage_node,
            rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
            n_stat=jnp.array(solver_config.n_stat),
        ),
    )
    lf_res_symmetric, _success = compute_symmetric_batch(
        topology_batch=topo_comp,
        disconnection_batch=disconnections,
        injections=None,
        nodal_inj_start_options=nodal_inj_start,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )

    # Stage inputs for nested mutation benchmarks.
    n_mutation_batch = int(sub_ids.shape[0])
    mutation_keys_batch = jax.random.split(jax.random.PRNGKey(123), n_mutation_batch)
    sub_ids_single = sub_ids[0]
    action_single = topo_comp.action[0]
    disconnections_single = disconnections[0]

    # JIT wrappers for each measured function.
    run_single_device_epoch_jit = jax.jit(
        run_single_device_epoch,
        static_argnames=("iterations_per_epoch", "update_fn"),
    )
    run_single_iteration_jit = jax.jit(lambda data: run_single_iteration(0, data, algo.update))
    algo_update_jit = jax.jit(lambda rep, es, rk, di: algo.update(rep, es, rk, di))
    emitter_emit_jit = jax.jit(lambda rep, es, rk: algo._emitter.emit(rep, es, rk))
    scoring_fn_jit = jax.jit(lambda g, rk, di: algo._scoring_function(g, rk, di))
    emitter_state_update_jit = jax.jit(lambda es, rep, g, f, d, ei: algo._emitter.state_update(es, rep, g, f, d, ei))
    metrics_fn_jit = jax.jit(lambda rep: algo._metrics_function(rep))

    def make_compute_overloads_jit(observed_metrics):
        return jax.jit(
            lambda g, di: compute_overloads(
                topologies=g,
                dynamic_information=di,
                solver_config=solver_config,
                observed_metrics=observed_metrics,
            )
        )

    def make_aggregate_metric_jit(metric_name):
        return jax.jit(
            lambda lf_res: aggregate_to_metric_batched(
                lf_res_batch=lf_res,
                branch_limits=dynamic_information.branch_limits,
                reassignment_distance=dynamic_information.action_set.reassignment_distance,
                n_relevant_subs=dynamic_information.n_sub_relevant,
                metric=metric_name,
                initial_pst_tap_idx=(
                    dynamic_information.nodal_injection_information.starting_tap_idx
                    if dynamic_information.nodal_injection_information is not None
                    else None
                ),
            )
        )

    translate_topology_jit = jax.jit(translate_topology)
    compute_overloads_jit = make_compute_overloads_jit(BASELINE_OBSERVED_METRICS)
    compute_overloads_all_metrics_jit = make_compute_overloads_jit(supported_observed_metrics)
    compute_symmetric_batch_jit = jax.jit(
        lambda tc, dc, inj, nis, di: compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
    )
    compute_symmetric_batch_no_nodal_start_jit = jax.jit(
        lambda tc, dc, inj, di: compute_symmetric_batch(tc, dc, inj, None, di, solver_config)
    )
    compute_symmetric_batch_n1_success_jit = jax.jit(
        lambda tc, dc, inj, nis, di: (lambda lf_res, success: (lf_res.n_1_matrix, success))(
            *compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
        )
    )
    compute_symmetric_batch_n0_success_jit = jax.jit(
        lambda tc, dc, inj, nis, di: (lambda lf_res, success: (lf_res.n_0_matrix, success))(
            *compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
        )
    )
    compute_symmetric_batch_cross_coupler_success_jit = jax.jit(
        lambda tc, dc, inj, nis, di: (lambda lf_res, success: (lf_res.cross_coupler_flows, success))(
            *compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
        )
    )
    compute_symmetric_batch_branch_topology_success_jit = jax.jit(
        lambda tc, dc, inj, nis, di: (lambda lf_res, success: (lf_res.branch_topology, success))(
            *compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
        )
    )
    compute_symmetric_batch_injection_topology_success_jit = jax.jit(
        lambda tc, dc, inj, nis, di: (lambda lf_res, success: (lf_res.injection_topology, success))(
            *compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)
        )
    )
    compute_symmetric_batch_success_only_jit = jax.jit(
        lambda tc, dc, inj, nis, di: compute_symmetric_batch(tc, dc, inj, nis, di, solver_config)[1]
    )
    compute_bsdf_lodf_static_flows_jit = jax.jit(
        lambda tv, dc, di: compute_bsdf_lodf_static_flows(tv, dc, di, solver_config)
    )
    compute_bus_splits_jit = jax.jit(
        lambda topologies, split_sub_ids: jax.vmap(
            partial(
                compute_bus_splits,
                ptdf=dynamic_information.ptdf,
                from_node=dynamic_information.from_node,
                to_node=dynamic_information.to_node,
                tot_stat=dynamic_information.tot_stat,
                from_stat_bool=dynamic_information.from_stat_bool,
                susceptance=dynamic_information.susceptance,
                rel_stat_map=solver_config.rel_stat_map,
                slack=solver_config.slack,
                n_stat=solver_config.n_stat,
            )
        )(topologies, split_sub_ids)
    )
    apply_disconnections_jit = jax.jit(
        lambda ptdf, from_node, to_node, disconnects: jax.vmap(
            partial(apply_disconnections, guarantee_unique=True),
            in_axes=(0, 0, 0, 0),
        )(ptdf, from_node, to_node, disconnects)
    )
    calc_lodf_matrix_jit = jax.jit(
        lambda ptdf, from_node, to_node: jax.vmap(calc_lodf_matrix, in_axes=(None, 0, 0, 0, None))(
            dynamic_information.branches_to_fail,
            ptdf,
            from_node,
            to_node,
            dynamic_information.branches_monitored,
        )
    )
    build_modf_matrices_jit = jax.jit(
        lambda ptdf, from_node, to_node: jax.vmap(
            build_modf_matrices,
            in_axes=(0, 0, 0, None),
        )(
            ptdf,
            from_node,
            to_node,
            dynamic_information.multi_outage_branches,
        )
    )
    mutate_jit = jax.jit(lambda g, rk: mutate(g, rk, mutation_config, action_set))
    repeat_topologies_jit = jax.jit(
        lambda g: repeat_topologies(
            g,
            batch_size=g.action_index.shape[0],
            mutation_repetition=mutation_config.mutation_repetition,
        )
    )
    mutate_topology_jit = jax.jit(
        lambda sub_id, disconnection_single, action, rk: mutate_topology(
            random_key=rk,
            sub_ids=sub_id,
            disconnections_topo=disconnection_single,
            action=action,
            mutate_config=mutation_config,
            action_set=action_set,
        )
    )
    mutate_topology_vmap_jit = jax.jit(
        lambda sub_ids_batch, disconnections_batch, action_batch, keys: jax.vmap(
            lambda sub_id, disconnection_single, action, rk: mutate_topology(
                random_key=rk,
                sub_ids=sub_id,
                disconnections_topo=disconnection_single,
                action=action,
                mutate_config=mutation_config,
                action_set=action_set,
            )
        )(sub_ids_batch, disconnections_batch, action_batch, keys)
    )
    create_random_topology_jit = jax.jit(
        lambda sub_id, disconnection_single, rk: create_random_topology(
            random_key=rk,
            sub_ids=sub_id,
            disconnections=disconnection_single,
            action_set=action_set,
            n_rel_subs=mutation_config.substation_mutation_config.n_rel_subs,
            n_disconnectable_branches=mutation_config.disconnection_mutation_config.n_disconnectable_branches,
        )
    )
    create_random_topology_vmap_jit = jax.jit(
        lambda sub_ids_batch, disconnections_batch, keys: jax.vmap(
            lambda sub_id, disconnection_single, rk: create_random_topology(
                random_key=rk,
                sub_ids=sub_id,
                disconnections=disconnection_single,
                action_set=action_set,
                n_rel_subs=mutation_config.substation_mutation_config.n_rel_subs,
                n_disconnectable_branches=mutation_config.disconnection_mutation_config.n_disconnectable_branches,
            )
        )(sub_ids_batch, disconnections_batch, keys)
    )
    mutate_sub_splits_jit = jax.jit(
        lambda sub_id, action, rk: mutate_sub_splits(
            sub_ids=sub_id,
            action=action,
            random_key=rk,
            sub_mutate_config=mutation_config.substation_mutation_config,
            action_set=action_set,
        )
    )
    split_additional_sub_jit = jax.jit(
        lambda sub_id, rk: split_additional_sub(
            random_key=rk,
            sub_ids=sub_id,
            n_rel_subs=mutation_config.substation_mutation_config.n_rel_subs,
            int_max_value=jnp.iinfo(sub_id.dtype).max,
        )
    )
    unsplit_substation_jit = jax.jit(
        lambda sub_id, rk: unsplit_substation(
            random_key=rk,
            sub_ids=sub_id,
            int_max_value=jnp.iinfo(sub_id.dtype).max,
        )
    )
    change_split_substation_jit = jax.jit(
        lambda sub_id, rk: change_split_substation(
            random_key=rk,
            sub_ids=sub_id,
            n_rel_subs=mutation_config.substation_mutation_config.n_rel_subs,
            int_max_value=jnp.iinfo(sub_id.dtype).max,
        )
    )
    mutate_disconnections_jit = jax.jit(
        lambda sub_id, disconnection_single, rk: mutate_disconnections(
            random_key=rk,
            sub_ids=sub_id,
            disconnections=disconnection_single,
            disconnection_mutation_config=mutation_config.disconnection_mutation_config,
        )
    )
    disconnect_additional_branch_jit = jax.jit(
        lambda disconnection_single, rk: disconnect_additional_branch(
            random_key=rk,
            disconnections=disconnection_single,
            n_disconnectable_branches=mutation_config.disconnection_mutation_config.n_disconnectable_branches,
        )
    )
    reconnect_disconnected_branch_jit = jax.jit(
        lambda disconnection_single, rk: reconnect_disconnected_branch(
            random_key=rk,
            disconnections=disconnection_single,
        )
    )
    change_disconnected_branch_jit = jax.jit(
        lambda disconnection_single, rk: change_disconnected_branch(
            random_key=rk,
            disconnections=disconnection_single,
            n_disconnectable_branches=mutation_config.disconnection_mutation_config.n_disconnectable_branches,
        )
    )
    mutate_psts_jit = jax.jit(
        lambda pst_taps, rk: mutate_psts(
            random_key=rk,
            pst_taps=pst_taps,
            pst_n_taps=mutation_config.nodal_injection_mutation_config.pst_n_taps,
            pst_mutation_sigma=mutation_config.nodal_injection_mutation_config.pst_mutation_sigma,
        )
    )
    mutate_nodal_injections_jit = jax.jit(
        lambda nodal_inj_info, rk: mutate_nodal_injections(
            random_key=rk,
            nodal_inj_info=nodal_inj_info,
            nodal_mutation_config=mutation_config.nodal_injection_mutation_config,
        )
    )
    compute_injections_jit = jax.jit(lambda inj, sid, di: compute_injections(inj, sid, di, solver_config))
    apply_pst_taps_jit = jax.jit(
        lambda n_0_batch, nodal_inj_batch, pst_tap_idx, topo_results: apply_pst_taps(
            n_0=n_0_batch,
            nodal_injections=nodal_inj_batch,
            pst_tap_indices=pst_tap_idx,
            topo_res=topo_results,
            nodal_inj_info=dynamic_information.nodal_injection_information,
        )
    )
    nodal_inj_optimization_jit = jax.jit(
        lambda n_0_batch, nodal_inj_batch, topo_results, start_options, di: nodal_inj_optimization(
            n_0=n_0_batch,
            nodal_injections=nodal_inj_batch,
            topo_res=topo_results,
            start_options=start_options,
            dynamic_information=di,
            solver_config=solver_config,
        )
    )
    cross_coupler_flows_jit = jax.jit(
        lambda bsdf, topo, sid, inj: jax.vmap(
            compute_cross_coupler_flows,
            in_axes=(0, 0, 0, 0, None, None, None, None),
        )(
            bsdf,
            topo,
            sid,
            inj,
            dynamic_information.relevant_injections,
            dynamic_information.unsplit_flow,
            dynamic_information.tot_stat,
            dynamic_information.from_stat_bool,
        )
    )
    contingency_analysis_matrix_jit = jax.jit(
        lambda params: jax.vmap(partial(contingency_analysis_matrix, unbatched_params=unbatched_params))(
            batched_params=params
        )
    )
    calc_n_1_matrix_jit = jax.jit(
        lambda lodf, n_0_flow_batch, n_0_flow_monitors_batch: jax.vmap(
            calc_n_1_matrix,
            in_axes=(0, None, 0, 0),
        )(
            lodf,
            dynamic_information.branches_to_fail,
            n_0_flow_batch,
            n_0_flow_monitors_batch,
        )
    )
    calc_injection_outages_jit = jax.jit(
        lambda ptdf, n_0_flow_batch, injection_outage_node: jax.vmap(
            calc_injection_outages,
            in_axes=(0, 0, None, 0, None),
        )(
            ptdf,
            n_0_flow_batch,
            unbatched_params.injection_outage_deltap,
            injection_outage_node,
            dynamic_information.branches_monitored,
        )
    )
    aggregate_metric_jit = make_aggregate_metric_jit("overload_energy_n_1")
    aggregate_all_metrics_jit = jax.jit(
        lambda lf_res: {
            metric_name: aggregate_to_metric_batched(
                lf_res_batch=lf_res,
                branch_limits=dynamic_information.branch_limits,
                reassignment_distance=dynamic_information.action_set.reassignment_distance,
                n_relevant_subs=dynamic_information.n_sub_relevant,
                metric=metric_name,
                initial_pst_tap_idx=(
                    dynamic_information.nodal_injection_information.starting_tap_idx
                    if dynamic_information.nodal_injection_information is not None
                    else None
                ),
            )
            for metric_name in supported_observed_metrics
        }
    )
    get_worst_k_contingencies_jit = jax.jit(
        lambda n_1_matrix: jax.vmap(get_worst_k_contingencies, in_axes=(None, 0, None))(
            10, n_1_matrix, dynamic_information.branch_limits.max_mw_flow
        )
    )
    add_to_repertoire_jit = jax.jit(
        lambda rep, g, d, f, e: add_to_repertoire(
            repertoire=rep,
            batch_of_genotypes=g,
            batch_of_descriptors=d,
            batch_of_fitnesses=f,
            batch_of_extra_scores=e,
        )
    )

    timings = [
        _bench_jitted(
            "run_single_device_epoch",
            run_single_device_epoch_jit,
            jax_data,
            args.iterations_per_epoch,
            algo.update,
            runs=args.runs,
        ),
        _bench_jitted("run_single_iteration", run_single_iteration_jit, jax_data, runs=args.runs),
        _bench_jitted(
            "DiscreteMapElites.update",
            algo_update_jit,
            jax_data.repertoire,
            jax_data.emitter_state,
            jax_data.random_key,
            jax_data.dynamic_informations,
            runs=args.runs,
        ),
        _bench_jitted(
            "TrackingMixingEmitter.emit",
            emitter_emit_jit,
            jax_data.repertoire,
            jax_data.emitter_state,
            jax_data.random_key,
            runs=args.runs,
        ),
        _bench_jitted(
            "scoring_function",
            scoring_fn_jit,
            genotypes,
            random_key_next,
            jax_data.dynamic_informations,
            runs=args.runs,
        ),
        _bench_jitted("mutate", mutate_jit, genotypes, random_key_next, runs=args.runs),
        _bench_jitted("repeat_topologies", repeat_topologies_jit, genotypes, runs=args.runs),
        _bench_jitted(
            "mutate_topology_single",
            mutate_topology_jit,
            sub_ids_single,
            disconnections_single,
            action_single,
            jax.random.PRNGKey(124),
            runs=args.runs,
        ),
        _bench_jitted(
            "mutate_topology_vmap",
            mutate_topology_vmap_jit,
            sub_ids,
            disconnections,
            topo_comp.action,
            mutation_keys_batch,
            runs=args.runs,
        ),
        _bench_jitted(
            "create_random_topology_single",
            create_random_topology_jit,
            sub_ids_single,
            disconnections_single,
            jax.random.PRNGKey(125),
            runs=args.runs,
        ),
        _bench_jitted(
            "create_random_topology_vmap",
            create_random_topology_vmap_jit,
            sub_ids,
            disconnections,
            mutation_keys_batch,
            runs=args.runs,
        ),
        _bench_jitted(
            "mutate_sub_splits",
            mutate_sub_splits_jit,
            sub_ids_single,
            action_single,
            jax.random.PRNGKey(126),
            runs=args.runs,
        ),
        _bench_jitted(
            "split_additional_sub",
            split_additional_sub_jit,
            sub_ids_single,
            jax.random.PRNGKey(127),
            runs=args.runs,
        ),
        _bench_jitted(
            "unsplit_substation",
            unsplit_substation_jit,
            sub_ids_single,
            jax.random.PRNGKey(128),
            runs=args.runs,
        ),
        _bench_jitted(
            "change_split_substation",
            change_split_substation_jit,
            sub_ids_single,
            jax.random.PRNGKey(129),
            runs=args.runs,
        ),
        _bench_jitted(
            "mutate_disconnections",
            mutate_disconnections_jit,
            sub_ids_single,
            disconnections_single,
            jax.random.PRNGKey(130),
            runs=args.runs,
        ),
        _bench_jitted(
            "disconnect_additional_branch",
            disconnect_additional_branch_jit,
            disconnections_single,
            jax.random.PRNGKey(131),
            runs=args.runs,
        ),
        _bench_jitted(
            "reconnect_disconnected_branch",
            reconnect_disconnected_branch_jit,
            disconnections_single,
            jax.random.PRNGKey(132),
            runs=args.runs,
        ),
        _bench_jitted(
            "change_disconnected_branch",
            change_disconnected_branch_jit,
            disconnections_single,
            jax.random.PRNGKey(133),
            runs=args.runs,
        ),
        *(
            [
                _bench_jitted(
                    "mutate_nodal_injections",
                    mutate_nodal_injections_jit,
                    genotypes.nodal_injections_optimized,
                    jax.random.PRNGKey(134),
                    runs=args.runs,
                ),
                _bench_jitted(
                    "mutate_psts",
                    mutate_psts_jit,
                    genotypes.nodal_injections_optimized.pst_tap_idx[0, 0],
                    jax.random.PRNGKey(135),
                    runs=args.runs,
                ),
            ]
            if (
                mutation_config.nodal_injection_mutation_config is not None
                and genotypes.nodal_injections_optimized is not None
            )
            else []
        ),
        _bench_jitted(
            "compute_overloads",
            compute_overloads_jit,
            genotypes_scored,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_overloads_all_supported_metrics",
            compute_overloads_all_metrics_jit,
            genotypes_scored,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted("translate_topology", translate_topology_jit, genotypes_scored, runs=args.runs),
        _bench_jitted(
            "compute_symmetric_batch",
            compute_symmetric_batch_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_no_nodal_start",
            compute_symmetric_batch_no_nodal_start_jit,
            topo_comp,
            disconnections,
            None,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_n1_success",
            compute_symmetric_batch_n1_success_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_n0_success",
            compute_symmetric_batch_n0_success_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_cross_coupler_success",
            compute_symmetric_batch_cross_coupler_success_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_branch_topology_success",
            compute_symmetric_batch_branch_topology_success_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_injection_topology_success",
            compute_symmetric_batch_injection_topology_success_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_symmetric_batch_success_only",
            compute_symmetric_batch_success_only_jit,
            topo_comp,
            disconnections,
            None,
            nodal_inj_start,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_injections",
            compute_injections_jit,
            injections,
            sub_ids,
            dynamic_information,
            runs=args.runs,
        ),
        *(
            [
                _bench_jitted(
                    "apply_pst_taps",
                    apply_pst_taps_jit,
                    n_0,
                    nodal_injections,
                    nodal_inj_start.previous_results.pst_tap_idx,
                    topo_res,
                    runs=args.runs,
                ),
                _bench_jitted(
                    "nodal_inj_optimization",
                    nodal_inj_optimization_jit,
                    n_0,
                    nodal_injections,
                    topo_res,
                    nodal_inj_start,
                    dynamic_information,
                    runs=args.runs,
                ),
            ]
            if nodal_inj_start is not None and dynamic_information.nodal_injection_information is not None
            else []
        ),
        _bench_jitted(
            "compute_cross_coupler_flows_vmap",
            cross_coupler_flows_jit,
            topo_res.bsdf,
            bitvector_topology.topologies,
            sub_ids,
            injections,
            runs=args.runs,
        ),
        _bench_jitted(
            "contingency_analysis_matrix_vmap",
            contingency_analysis_matrix_jit,
            batched_params,
            runs=args.runs,
        ),
        _bench_jitted(
            "calc_n_1_matrix_vmap",
            calc_n_1_matrix_jit,
            topo_res.lodf,
            n_0,
            n_0_flow_monitors,
            runs=args.runs,
        ),
        _bench_jitted(
            "calc_injection_outages_vmap",
            calc_injection_outages_jit,
            topo_res.ptdf,
            n_0,
            batched_params.injection_outage_node,
            runs=args.runs,
        ),
        _bench_jitted(
            "aggregate_to_metric_batched",
            aggregate_metric_jit,
            lf_res_symmetric,
            runs=args.runs,
        ),
        _bench_jitted(
            "aggregate_all_supported_metrics",
            aggregate_all_metrics_jit,
            lf_res_symmetric,
            runs=args.runs,
        ),
        _bench_jitted(
            "get_worst_k_contingencies_vmap",
            get_worst_k_contingencies_jit,
            lf_res_symmetric.n_1_matrix,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_bsdf_lodf_static_flows",
            compute_bsdf_lodf_static_flows_jit,
            bitvector_topology,
            translated_disconnections,
            dynamic_information,
            runs=args.runs,
        ),
        _bench_jitted(
            "compute_bus_splits_vmap",
            compute_bus_splits_jit,
            bitvector_topology.topologies,
            bitvector_topology.sub_ids,
            runs=args.runs,
        ),
        _bench_jitted(
            "apply_disconnections_vmap",
            apply_disconnections_jit,
            bsdf_only_res.ptdf,
            bsdf_only_res.from_node,
            bsdf_only_res.to_node,
            translated_disconnections,
            runs=args.runs,
        ),
        _bench_jitted(
            "calc_lodf_matrix_vmap",
            calc_lodf_matrix_jit,
            disconnection_res.ptdf,
            disconnection_res.from_node,
            disconnection_res.to_node,
            runs=args.runs,
        ),
        _bench_jitted(
            "build_modf_matrices_vmap",
            build_modf_matrices_jit,
            disconnection_res.ptdf,
            disconnection_res.from_node,
            disconnection_res.to_node,
            runs=args.runs,
        ),
        _bench_jitted(
            "add_to_repertoire",
            add_to_repertoire_jit,
            jax_data.repertoire,
            genotypes_scored,
            descriptors,
            fitnesses,
            extra_scores,
            runs=args.runs,
        ),
        _bench_jitted(
            "TrackingMixingEmitter.state_update",
            emitter_state_update_jit,
            jax_data.emitter_state,
            jax_data.repertoire,
            genotypes_scored,
            fitnesses,
            descriptors,
            emitter_info,
            runs=args.runs,
        ),
        _bench_jitted("metrics_function", metrics_fn_jit, jax_data.repertoire, runs=args.runs),
    ]

    aggregate_metric_timings = [
        _bench_jitted(
            f"aggregate_metric::{metric_name}",
            make_aggregate_metric_jit(metric_name),
            lf_res_symmetric,
            runs=args.runs,
        )
        for metric_name in supported_observed_metrics
    ]

    compute_overloads_metric_timings = [
        _bench_jitted(
            f"compute_overloads_with_metric::{metric_name}",
            make_compute_overloads_jit(BASELINE_OBSERVED_METRICS + (metric_name,)),
            genotypes_scored,
            dynamic_information,
            runs=args.runs,
        )
        for metric_name in additional_observed_metrics
    ]

    top_exec = sorted(timings, key=lambda x: x["median_exec_ms"], reverse=True)[:5]
    top_compile = sorted(timings, key=lambda x: x["compile_plus_first_ms"], reverse=True)[:5]
    baseline_compute_overloads = next(item for item in timings if item["name"] == "compute_overloads")
    all_metrics_compute_overloads = next(
        item for item in timings if item["name"] == "compute_overloads_all_supported_metrics"
    )
    aggregate_metric_timings_sorted = sorted(aggregate_metric_timings, key=lambda x: x["median_exec_ms"], reverse=True)
    compute_overloads_metric_effects = sorted(
        [
            {
                **item,
                "metric": item["name"].split("::", 1)[1],
                "delta_vs_baseline_ms": round(item["median_exec_ms"] - baseline_compute_overloads["median_exec_ms"], 3),
            }
            for item in compute_overloads_metric_timings
        ],
        key=lambda x: x["delta_vs_baseline_ms"],
        reverse=True,
    )

    results = {
        "device": str(jax.devices()[0]),
        "jax_platform": os.environ.get("JAX_PLATFORMS", "default"),
        "static_information_file": str(args.static_information_file),
        "batch_size": args.batch_size,
        "iterations_per_epoch": args.iterations_per_epoch,
        "runs": args.runs,
        "timings": timings,
        "potential_bottlenecks": {
            "highest_median_exec_ms": top_exec,
            "highest_compile_plus_first_ms": top_compile,
        },
        "metric_analysis": {
            "baseline_observed_metrics": BASELINE_OBSERVED_METRICS,
            "all_supported_observed_metrics": supported_observed_metrics,
            "additional_observed_metrics": additional_observed_metrics,
            "compute_overloads_baseline_median_ms": baseline_compute_overloads["median_exec_ms"],
            "compute_overloads_all_supported_median_ms": all_metrics_compute_overloads["median_exec_ms"],
            "compute_overloads_all_supported_delta_ms": round(
                all_metrics_compute_overloads["median_exec_ms"] - baseline_compute_overloads["median_exec_ms"], 3
            ),
            "aggregate_metric_timings": aggregate_metric_timings_sorted,
            "compute_overloads_metric_effects": compute_overloads_metric_effects,
            "top_aggregate_metric_costs": aggregate_metric_timings_sorted[:10],
            "top_compute_overloads_metric_effects": compute_overloads_metric_effects[:10],
        },
        "note": "This benchmarks the primary nested call chain reachable from "
        "run_single_device_epoch with direct JIT wrappers.",
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
