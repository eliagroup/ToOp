# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax_dataclasses import replace
from tests.numpy_reference import run_solver as run_solver_ref
from toop_engine_dc_solver.jax.aggregate_results import (
    aggregate_n_1_matrix,
    aggregate_to_metric_batched,
)
from toop_engine_dc_solver.jax.batching import slice_topologies
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.injections import default_injection
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    convert_single_branch_topo_vect,
    convert_topo_to_action_set_index,
    random_topology,
)
from toop_engine_dc_solver.jax.topology_looper import (
    DefaultAggregateMetricsFn,
    DefaultAggregateOutputFn,
    run_solver,
    run_solver_inj_bruteforce,
    run_solver_symmetric,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    SolverLoadflowResults,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def test_run_solver(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config
    topologies = jax_inputs[0]
    nd = case14_network_data

    action_index_topo, _ = convert_topo_to_action_set_index(topologies, dynamic_information.action_set)

    injections = default_injection(
        n_splits=action_index_topo.action.shape[1],
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        batch_size=action_index_topo.action.shape[0],
    )

    n_1_flows, success = run_solver_symmetric(
        topologies=action_index_topo,
        disconnections=None,
        injections=injections.injection_topology,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=lambda x: x.n_1_matrix,
    )
    assert jnp.all(success)

    for topo_id in range(topologies.topologies.shape[0]):
        topo_vect = convert_single_branch_topo_vect(
            topologies.topologies[topo_id],
            topologies.sub_ids[topo_id],
            branches_per_sub=solver_config.branches_per_sub,
        )
        n_1_ref, success = run_solver_ref(
            branch_topo_vect=topo_vect,
            relevant_nodes=nd.relevant_nodes,
            slack=nd.slack,
            n_stat=nd.n_original_nodes,
            ptdf=nd.ptdf,
            susceptance=nd.susceptances,
            from_node=nd.from_nodes,
            to_node=nd.to_nodes,
            branches_at_nodes=nd.branches_at_nodes,
            branch_direction=nd.branch_direction,
            branches_to_outage=np.flatnonzero(nd.outaged_branch_mask),
            nodal_injections=nd.nodal_injection[0],
        )
        assert np.all(success)
        # The reference solver can only do line outages
        n_1_solver = n_1_flows[topo_id, 0, : sum(nd.outaged_branch_mask)]
        assert n_1_solver.shape == n_1_ref.shape
        assert jnp.allclose(n_1_solver, n_1_ref)

    # Test "bruteforce" mode but with only default injections
    n_1_flow_bruteforce, _, success = run_solver_inj_bruteforce(
        topologies=action_index_topo,
        disconnections=None,
        injections=injections,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_metric_fn=lambda _1, _2: 0,
        aggregate_output_fn=lambda x: x.n_1_matrix,
    )

    assert jnp.all(success)
    assert jnp.allclose(n_1_flows, n_1_flow_bruteforce)


def test_run_solver_random_topo(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config
    nd = case14_network_data

    n_topologies = 10
    n_splits = 3

    act_index_topo = random_topology(
        rng_key=jax.random.PRNGKey(0),
        branch_action_set=dynamic_information.action_set,
        limit_n_subs=n_splits,
        batch_size=n_topologies,
    )
    topologies = convert_action_set_index_to_topo(act_index_topo, dynamic_information.action_set)
    inj = default_injection(
        n_splits=n_splits,
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        batch_size=n_topologies,
    )

    n_1_flows, success = run_solver_symmetric(
        topologies=act_index_topo,
        disconnections=None,
        injections=inj.injection_topology,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=lambda x: x.n_1_matrix,
    )
    # Some might fail
    assert jnp.sum(success) > 0

    outage_branches = np.flatnonzero(nd.outaged_branch_mask)
    for topo_id in range(topologies.topologies.shape[0]):
        topo_vect = convert_single_branch_topo_vect(
            topologies.topologies[topo_id],
            topologies.sub_ids[topo_id],
            branches_per_sub=solver_config.branches_per_sub,
        )
        if not success[topo_id]:
            with pytest.raises(ValueError):
                _, success = run_solver_ref(
                    branch_topo_vect=topo_vect,
                    relevant_nodes=nd.relevant_nodes,
                    slack=nd.slack,
                    n_stat=nd.n_original_nodes,
                    ptdf=nd.ptdf,
                    susceptance=nd.susceptances,
                    from_node=nd.from_nodes,
                    to_node=nd.to_nodes,
                    branches_at_nodes=nd.branches_at_nodes,
                    branch_direction=nd.branch_direction,
                    branches_to_outage=outage_branches,
                    nodal_injections=nd.nodal_injection[0],
                )
                if not np.all(success):
                    raise ValueError()
        else:
            n_1_ref, success_ref = run_solver_ref(
                branch_topo_vect=topo_vect,
                relevant_nodes=nd.relevant_nodes,
                slack=nd.slack,
                n_stat=nd.n_original_nodes,
                ptdf=nd.ptdf,
                susceptance=nd.susceptances,
                from_node=nd.from_nodes,
                to_node=nd.to_nodes,
                branches_at_nodes=nd.branches_at_nodes,
                branch_direction=nd.branch_direction,
                branches_to_outage=outage_branches,
                nodal_injections=nd.nodal_injection[0],
            )
            assert np.all(success_ref)
            n_1_solver = n_1_flows[topo_id, 0, : len(outage_branches)]
            assert n_1_solver.shape == n_1_ref.shape
            assert jnp.allclose(n_1_solver, n_1_ref)


def test_run_solver_multiple_timesteps(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    action_index_topo, _ = convert_topo_to_action_set_index(topologies, static_information.dynamic_information.action_set)

    reference = run_solver(
        topologies=action_index_topo,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )

    n_timesteps = 4

    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            nodal_injections=jnp.repeat(
                static_information.dynamic_information.nodal_injections,
                n_timesteps,
                axis=0,
            ),
            relevant_injections=jnp.repeat(
                static_information.dynamic_information.relevant_injections,
                n_timesteps,
                axis=0,
            ),
            nonrel_injection_outage_deltap=jnp.repeat(
                static_information.dynamic_information.nonrel_injection_outage_deltap, n_timesteps, axis=0
            ),
            unsplit_flow=jnp.repeat(static_information.dynamic_information.unsplit_flow, n_timesteps, axis=0),
        ),
    )

    multi_timestep = run_solver(
        topologies=action_index_topo,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )

    assert multi_timestep.n_0_results.hist_mon.shape == (
        topologies.topologies.shape[0],
        n_timesteps,
        static_information.solver_config.number_most_affected,
    )
    assert multi_timestep.n_0_results.pf_n_0_max.shape == (
        topologies.topologies.shape[0],
        n_timesteps,
        static_information.solver_config.number_most_affected,
    )
    assert multi_timestep.n_1_results.hist_mon.shape == (
        topologies.topologies.shape[0],
        n_timesteps,
        static_information.solver_config.number_most_affected,
    )
    assert multi_timestep.n_1_results.hist_out.shape == (
        topologies.topologies.shape[0],
        n_timesteps,
        static_information.solver_config.number_most_affected,
    )
    assert multi_timestep.n_1_results.pf_n_1_max.shape == (
        topologies.topologies.shape[0],
        n_timesteps,
        static_information.solver_config.number_most_affected,
    )

    assert jnp.allclose(
        jnp.max(multi_timestep.n_1_results.pf_n_1_max, axis=(1, 2)),
        jnp.max(reference.n_1_results.pf_n_1_max, axis=(1, 2)),
    )


def test_run_solver_with_disconnections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    action_index_topo, _ = convert_topo_to_action_set_index(topologies, static_information.dynamic_information.action_set)

    disconnections = jnp.repeat(jnp.array([[8, 999]]), topologies.topologies.shape[0], axis=0)

    res = run_solver(
        topologies=action_index_topo,
        disconnections=disconnections,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert res is not None


def test_run_solver_inj_candidates(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, candidates, static_information = jax_inputs

    action_index_topos, _ = convert_topo_to_action_set_index(topologies, static_information.dynamic_information.action_set)

    aggregate_metrics_fn = DefaultAggregateMetricsFn(
        branch_limits=static_information.dynamic_information.branch_limits,
        reassignment_distance=static_information.dynamic_information.action_set.reassignment_distance,
        n_relevant_subs=static_information.n_sub_relevant,
        metric=static_information.solver_config.aggregation_metric,
        fixed_hash=hash(static_information),
    )
    aggregate_output_fn = DefaultAggregateOutputFn(
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
    )
    # Auto-determine the buffer size
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, buffer_size_injection=9999),
    )

    res, best_inj, succ = run_solver_inj_bruteforce(
        topologies=action_index_topos,
        disconnections=None,
        injections=candidates,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_metric_fn=aggregate_metrics_fn,
        aggregate_output_fn=aggregate_output_fn,
    )
    assert jnp.all(succ)

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, distributed=True),
    )

    res2, best_inj_2, succ = run_solver_inj_bruteforce(
        topologies=action_index_topos,
        disconnections=None,
        injections=candidates,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_metric_fn=aggregate_metrics_fn,
        aggregate_output_fn=aggregate_output_fn,
    )
    assert jnp.all(succ)
    assert jnp.array_equal(best_inj, best_inj_2)

    assert jnp.allclose(
        jnp.max(res[0].pf_n_0_max, axis=-1),
        jnp.max(res2[0].pf_n_0_max, axis=-1),
    )
    assert jnp.allclose(
        jnp.max(res[1].pf_n_1_max, axis=-1),
        jnp.max(res2[1].pf_n_1_max, axis=-1),
    )


def test_run_solver_symmetric(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    action_index_topo, _ = convert_topo_to_action_set_index(topologies, static_information.dynamic_information.action_set)
    injections = default_injection(
        n_splits=action_index_topo.action.shape[1],
        max_inj_per_sub=static_information.dynamic_information.max_inj_per_sub,
        batch_size=action_index_topo.action.shape[0],
    )

    def aggregate_output_fn(lf_res: SolverLoadflowResults):
        return aggregate_n_1_matrix(
            lf_res.n_1_matrix,
            max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow,
            metric="overload_energy",
        )

    res, success = run_solver_symmetric(
        topologies=action_index_topo,
        disconnections=None,
        injections=injections.injection_topology,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=aggregate_output_fn,
    )

    assert len(res) == topologies.topologies.shape[0]
    assert len(success) == topologies.topologies.shape[0]
    assert jnp.all(success)

    batch_size = static_information.solver_config.batch_size_bsdf
    topology_batch = slice_topologies(topologies, 0, batch_size)
    topology_batch, _ = convert_topo_to_action_set_index(
        topology_batch, static_information.dynamic_information.action_set, extend_action_set=False
    )
    injection_batch = default_injection(
        n_splits=topology_batch.action.shape[1],
        max_inj_per_sub=static_information.dynamic_information.max_inj_per_sub,
        batch_size=batch_size,
    )
    lf_res, _ = compute_symmetric_batch(
        topology_batch=topology_batch,
        disconnection_batch=None,
        injections=injection_batch.injection_topology,
        nodal_inj_start_options=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    ref = aggregate_to_metric_batched(
        lf_res,
        branch_limits=static_information.dynamic_information.branch_limits,
        reassignment_distance=static_information.dynamic_information.action_set.reassignment_distance,
        n_relevant_subs=static_information.n_sub_relevant,
        metric="overload_energy_n_1",
    )
    assert jnp.allclose(res[:batch_size], ref)

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, distributed=True),
    )

    res2, success = run_solver_symmetric(
        topologies=action_index_topo,
        disconnections=None,
        injections=injections.injection_topology,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=aggregate_output_fn,
    )

    assert len(res2) == topologies.topologies.shape[0]
    assert jnp.allclose(res, res2)
    assert jnp.all(success)


def test_run_solver_symmetric_with_bb_outage(
    network_data_preprocessed: NetworkData,
) -> None:
    static_information = convert_to_jax(network_data_preprocessed, enable_bb_outage=True, bb_outage_as_nminus1=False)
    action_index_topo: ActionIndexComputations = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=static_information.solver_config.batch_size_bsdf,
        topo_vect_format=False,
    )
    solver_config = static_information.solver_config

    def aggregate_output_fn(lf_res: SolverLoadflowResults):
        return aggregate_n_1_matrix(
            lf_res.n_1_matrix,
            max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow,
            metric="overload_energy",
        )

    res, success = run_solver_symmetric(
        topologies=action_index_topo,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=aggregate_output_fn,
    )

    assert len(res) == action_index_topo.action.shape[0]
    assert len(success) == action_index_topo.action.shape[0]
    assert jnp.all(success)
