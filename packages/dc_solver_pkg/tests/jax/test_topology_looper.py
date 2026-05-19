# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jaxtyping
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
    run_solver,
    run_solver_symmetric,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    SolverLoadflowResults,
    SparseSolverOutput,
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
    # Some random topologies can be numerically unstable. Compare only successful solver runs.
    topology_success = jnp.all(success, axis=1)
    assert jnp.sum(topology_success) > 0

    outage_branches = np.flatnonzero(nd.outaged_branch_mask)
    for topo_id in range(topologies.topologies.shape[0]):
        topo_vect = convert_single_branch_topo_vect(
            topologies.topologies[topo_id],
            topologies.sub_ids[topo_id],
            branches_per_sub=solver_config.branches_per_sub,
        )

        if not bool(topology_success[topo_id]):
            ref_deemed_unsuccessful = False
            try:
                _, success_ref = run_solver_ref(
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
                ref_deemed_unsuccessful = not np.all(success_ref)
            except ValueError:
                ref_deemed_unsuccessful = True

            assert ref_deemed_unsuccessful
            continue

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


def test_typecheck():
    with pytest.raises(jaxtyping.TypeCheckError):
        SparseSolverOutput(n_0_results=123, n_1_results=123, best_inj_combi=True, success="True")


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
    static_information = convert_to_jax(network_data_preprocessed, preprocess_bb_outages=True)
    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            enable_bb_outages=True,
            bb_outage_as_nminus1=True,
        ),
        dynamic_information=replace(
            static_information.dynamic_information,
            bb_outage_baseline_analysis=None,
        ),
    )
    action_index_topo: ActionIndexComputations = random_topology(
        jax.random.PRNGKey(41),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=static_information.solver_config.batch_size_bsdf,
        topo_vect_format=False,
    )
    action_index_topo2: ActionIndexComputations = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=static_information.solver_config.batch_size_bsdf,
        topo_vect_format=False,
    )
    solver_config = static_information.solver_config

    n_1_matrix, success = run_solver_symmetric(
        topologies=action_index_topo,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=lambda lf_res: lf_res.n_1_matrix,
    )

    assert n_1_matrix.shape == (
        action_index_topo.action.shape[0],
        static_information.dynamic_information.n_timesteps,
        static_information.dynamic_information.n_nminus1_cases,
        static_information.dynamic_information.n_branches_monitored,
    )
    assert len(success) == action_index_topo.action.shape[0]
    assert jnp.all(success)
