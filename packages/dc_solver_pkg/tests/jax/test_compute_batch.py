# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import numpy as np
from jax import numpy as jnp
from jax_dataclasses import replace
from tests.jax.test_busbar_outage import compute_splits_and_injections
from toop_engine_dc_solver.jax.batching import batch_injections, batch_topologies
from toop_engine_dc_solver.jax.busbar_outage import perform_rel_bb_outage_single_topo
from toop_engine_dc_solver.jax.compute_batch import (
    compute_batch,
    compute_bsdf_lodf_static_flows,
    compute_symmetric_batch,
)
from toop_engine_dc_solver.jax.disconnections import random_disconnection_indices
from toop_engine_dc_solver.jax.injections import (
    default_injection,
    random_injection,
)
from toop_engine_dc_solver.jax.nminus2_outage import unsplit_n_2_analysis
from toop_engine_dc_solver.jax.topology_computations import (
    convert_topo_to_action_set_index_jittable,
    pad_action_with_unsplit_action_indices,
)
from toop_engine_dc_solver.jax.topology_looper import (
    DefaultAggregateMetricsFn,
    DefaultAggregateOutputFn,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import get_bb_outage_baseline_analysis
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_n_minus_2_safe_branches,
)


def test_compute_bsdf_lodf_static_flows(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    n_failures = len(static_information.dynamic_information.branches_to_fail)
    batch_size_bsdf = static_information.solver_config.batch_size_bsdf
    n_branches = static_information.dynamic_information.ptdf.shape[0]
    n_bus = static_information.dynamic_information.ptdf.shape[1]
    n_timesteps = static_information.dynamic_information.nodal_injections.shape[0]

    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            branch_limits=replace(
                static_information.dynamic_information.branch_limits,
                max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow * 0.01,  # The data is in 10kW
            ),
        ),
        solver_config=replace(
            static_information.solver_config,
            aggregation_metric="overload_energy",
            number_most_affected=n_branches * n_failures,
            number_most_affected_n_0=n_branches,
            number_max_out_in_most_affected=n_branches,
        ),
    )

    batched_topologies = batch_topologies(topologies, batch_size_bsdf)

    batch_index = jnp.array(0, dtype=int)

    topology_batch = batched_topologies[batch_index]

    topo_res = compute_bsdf_lodf_static_flows(
        topology_batch,
        None,
        static_information.dynamic_information,
        static_information.solver_config,
    )

    assert topo_res.lodf.shape == (batch_size_bsdf, n_failures, n_branches)
    assert topo_res.ptdf.shape == (batch_size_bsdf, n_branches, n_bus)
    assert topo_res.success.shape == (batch_size_bsdf,)
    assert jnp.all(topo_res.success)


def test_compute_bsdf_lodf_static_flows_with_outages(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    batch_size_bsdf = static_information.solver_config.batch_size_bsdf

    batched_topologies = batch_topologies(topologies, batch_size_bsdf)

    batch_index = jnp.array(0, dtype=int)

    n_minus_2_safe_mask = find_n_minus_2_safe_branches(
        from_node=np.array(static_information.dynamic_information.from_node),
        to_node=np.array(static_information.dynamic_information.to_node),
        number_of_branches=static_information.dynamic_information.ptdf.shape[0],
        number_of_nodes=static_information.dynamic_information.ptdf.shape[1],
    )

    disconnections = jnp.argwhere(n_minus_2_safe_mask == 1).flatten()[0:1][None, :]

    topo_res = compute_bsdf_lodf_static_flows(
        topology_batch=batched_topologies[batch_index],
        disconnection_batch=disconnections,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )

    assert jnp.all(topo_res.success)

    disconnections = jnp.argwhere(n_minus_2_safe_mask == 0).flatten()[0:1][None, :]

    topo_res = compute_bsdf_lodf_static_flows(
        topology_batch=batched_topologies[batch_index],
        disconnection_batch=disconnections,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )

    assert not jnp.all(topo_res.success)


def test_compute_batch(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, candidates, static_information = jax_inputs

    batched_topologies = batch_topologies(topologies, static_information.solver_config.batch_size_bsdf)

    batched_injections = batch_injections(
        all_injections=candidates,
        batched_topologies=batched_topologies,
        batch_size_injection=static_information.solver_config.batch_size_injection,
        buffer_size_injection=static_information.solver_config.buffer_size_injection,
    )

    batch_index = jnp.array(0, dtype=int)
    # We don't have to adjust corresponding topologies for batch 0

    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=batched_topologies[batch_index],
        branch_actions=static_information.dynamic_information.action_set,
    )

    _, best_inj, success = compute_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injection_batch=batched_injections[batch_index],
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_metric_fn=DefaultAggregateMetricsFn(
            branch_limits=static_information.dynamic_information.branch_limits,
            reassignment_distance=static_information.dynamic_information.action_set.reassignment_distance,
            metric=static_information.solver_config.aggregation_metric,
            n_relevant_subs=static_information.n_sub_relevant,
            fixed_hash=hash(static_information),
        ),
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

    assert best_inj.shape == (
        static_information.solver_config.batch_size_bsdf,
        static_information.dynamic_information.n_sub_relevant,
        static_information.dynamic_information.max_inj_per_sub,
    )
    assert success.shape == (static_information.solver_config.batch_size_bsdf,)


def test_compute_batch_nminus2(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    solver_config = static_information.solver_config
    dynamic_information = static_information.dynamic_information

    baseline = unsplit_n_2_analysis(dynamic_information=dynamic_information, more_splits_penalty=100.0)

    dynamic_information = replace(dynamic_information, n2_baseline_analysis=baseline)

    injections = random_injection(
        jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=8,
        for_topology=topologies,
    )

    topologies = batch_topologies(topologies, static_information.solver_config.batch_size_bsdf)
    injections = batch_injections(
        all_injections=injections,
        batched_topologies=topologies,
        batch_size_injection=static_information.solver_config.batch_size_injection,
        buffer_size_injection=static_information.solver_config.buffer_size_injection,
    )

    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=topologies,
        branch_actions=dynamic_information.action_set,
    )
    n_2_penalty, _, success = compute_batch(
        topology_batch=action_index_topo[0],
        disconnection_batch=None,
        injection_batch=injections[0],
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_metric_fn=DefaultAggregateMetricsFn(
            branch_limits=dynamic_information.branch_limits,
            reassignment_distance=dynamic_information.action_set.reassignment_distance,
            metric=solver_config.aggregation_metric,
            n_relevant_subs=static_information.n_sub_relevant,
            fixed_hash=hash(static_information),
        ),
        aggregate_output_fn=lambda lf_res: lf_res.n_2_penalty,
    )
    assert jnp.all(success)
    assert n_2_penalty is not None
    assert n_2_penalty.shape == success.shape
    assert jnp.all(n_2_penalty >= 0.0)
    # Unsplit grid is first topology
    assert n_2_penalty[0] == 0


def test_compute_symmetric_batch(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    n_sub_relevant = static_information.solver_config.branches_per_sub.shape[0]
    batch_size_bsdf = static_information.solver_config.batch_size_bsdf
    batch_size_injection = static_information.solver_config.batch_size_injection
    buffer_size_injection = static_information.solver_config.buffer_size_injection
    max_mw_flow = static_information.dynamic_information.branch_limits.max_mw_flow
    max_inj_per_sub = static_information.dynamic_information.max_inj_per_sub

    # this test relies on this
    assert batch_size_bsdf <= batch_size_injection

    batched_topologies = batch_topologies(topologies, batch_size_bsdf)
    batch_index = jnp.array(0, dtype=int)

    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=batched_topologies,
        branch_actions=static_information.dynamic_information.action_set,
    )[batch_index]
    n_splits = action_index_topo.action.shape[1]

    injections = jnp.zeros((batch_size_bsdf, n_splits, max_inj_per_sub), dtype=bool)
    corresponding_topology = jnp.arange(batch_size_bsdf, dtype=int)

    inj_computations = InjectionComputations(
        injection_topology=jnp.zeros((buffer_size_injection, batch_size_injection, n_splits, max_inj_per_sub), dtype=bool)
        .at[0, 0:batch_size_bsdf]
        .set(injections),
        corresponding_topology=jnp.zeros((buffer_size_injection, batch_size_injection), dtype=int)
        .at[0, 0:batch_size_bsdf]
        .set(corresponding_topology),
        pad_mask=jnp.zeros((buffer_size_injection, batch_size_injection), dtype=int).at[0, 0:batch_size_bsdf].set(1),
    )

    res_ref, best_inj_combo_ref, success_ref = compute_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injection_batch=inj_computations,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_metric_fn=DefaultAggregateMetricsFn(
            branch_limits=static_information.dynamic_information.branch_limits,
            reassignment_distance=static_information.dynamic_information.action_set.reassignment_distance,
            metric=static_information.solver_config.aggregation_metric,
            n_relevant_subs=static_information.n_sub_relevant,
            fixed_hash=hash(static_information),
        ),
        aggregate_output_fn=DefaultAggregateOutputFn(
            branches_to_fail=static_information.dynamic_information.branches_to_fail,
            multi_outage_indices=jnp.arange(static_information.dynamic_information.n_multi_outages)
            + jnp.max(static_information.dynamic_information.branches_to_fail),
            injection_outage_indices=jnp.arange(static_information.dynamic_information.n_inj_failures)
            + jnp.max(static_information.dynamic_information.branches_to_fail)
            + static_information.dynamic_information.n_multi_outages,
            max_mw_flow=max_mw_flow,
            number_most_affected=static_information.solver_config.number_most_affected,
            number_max_out_in_most_affected=static_information.solver_config.number_max_out_in_most_affected,
            number_most_affected_n_0=static_information.solver_config.number_most_affected_n_0,
            fixed_hash=hash(static_information),
        ),
    )
    n_0_ref, n_1_ref = res_ref

    assert jnp.array_equal(best_inj_combo_ref, injections)

    lf_res, success = compute_symmetric_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injections=injections,
        nodal_inj_start_options=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert jnp.array_equal(success_ref, success)
    full_n_0 = lf_res.n_0_matrix
    full_n_1 = lf_res.n_1_matrix

    full_n_0 = jnp.abs(full_n_0 / max_mw_flow)
    full_n_1 = jnp.abs(full_n_1 / max_mw_flow)

    worst_extracted = jax.lax.top_k(full_n_1, k=static_information.solver_config.number_max_out_in_most_affected)[0]
    worst_extracted = worst_extracted.reshape(batch_size_bsdf, 1, -1)
    worst_extracted = jax.lax.top_k(worst_extracted, k=static_information.solver_config.number_most_affected)[0]

    assert jnp.allclose(n_1_ref.pf_n_1_max, worst_extracted)

    worst_n_0_extracted = jax.lax.top_k(full_n_0, k=static_information.solver_config.number_most_affected)[0]

    assert jnp.allclose(n_0_ref.pf_n_0_max, worst_n_0_extracted)


def test_compute_batch_symmetric_with_bb_outage(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation],
) -> None:
    topo_indices, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information

    assert static_information.solver_config.enable_bb_outages, "This test is only for the case with busbar outages"

    lf_res, success = compute_symmetric_batch(
        topology_batch=topo_indices,
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert lf_res.bb_outage_penalty is None
    full_n_1 = lf_res.n_1_matrix

    n_timesteps = static_information.dynamic_information.n_timesteps
    n_n_1_cases = di.n_nminus1_cases
    assert full_n_1.shape == (
        topo_indices.action.shape[0],
        n_timesteps,
        n_n_1_cases,
        static_information.dynamic_information.n_branches_monitored,
    )

    # Calculate loadflows due to busbar outage for reference.
    for batch_id in range(topo_indices.action.shape[0]):
        topo_index = topo_indices[batch_id]
        updated_topo_indices = jax.jit(pad_action_with_unsplit_action_indices)(di.action_set, topo_index.action)
        branch_actions = np.array(di.action_set.branch_actions[updated_topo_indices])
        affected_sub_ids = di.action_set.substation_correspondence[updated_topo_indices]
        splitted_ptdf, from_node, to_node, input_nodal_injections, _ = compute_splits_and_injections(
            static_information, branch_actions, updated_topo_indices, affected_sub_ids
        )
        n_0_flows = jnp.einsum("ij,tj -> ti", splitted_ptdf, input_nodal_injections)
        # Perform busbar outage for the current topology
        lfs_bb_outage_ref, success = perform_rel_bb_outage_single_topo(
            n_0_flows=n_0_flows,
            action_set=di.action_set,
            action_indices=updated_topo_indices,
            ptdf=splitted_ptdf,
            nodal_injections=input_nodal_injections,
            from_nodes=from_node,
            to_nodes=to_node,
            branches_monitored=di.branches_monitored,
        )
        bb_outage_index = n_n_1_cases - di.n_bb_outages
        # Note: In oberrhein data, there are total 6 relevant busbars. We compare the load flows corresponding
        # to these busbars in the n-1 matrix with the load flows calculated using the busbar outage function.
        n_minus_1_bb_outage = jnp.transpose(full_n_1[batch_id, :, bb_outage_index : bb_outage_index + 6, :], (1, 0, 2))

        assert jnp.allclose(n_minus_1_bb_outage, lfs_bb_outage_ref, equal_nan=True)

    # Case 2: bb_outage computed as penalty.
    solver_config = static_information.solver_config
    solver_config = replace(
        solver_config,
        bb_outage_as_nminus1=False,
    )
    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            bb_outage_baseline_analysis=get_bb_outage_baseline_analysis(
                di=static_information.dynamic_information,
                more_splits_penalty=1000.0,
            ),
        ),
    )

    lf_res, success = compute_symmetric_batch(
        topology_batch=topo_indices,
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=solver_config,
    )
    assert lf_res.bb_outage_penalty is not None
    assert lf_res.bb_outage_penalty.shape == (solver_config.batch_size_bsdf,)


def test_compute_symmetric_batch_with_disconnection(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=64),
        dynamic_information=replace(
            static_information.dynamic_information, branches_monitored=jnp.arange(static_information.n_branches)
        ),
    )

    topologies = batch_topologies(topologies, static_information.solver_config.batch_size_bsdf)[0]
    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=topologies,
        branch_actions=static_information.dynamic_information.action_set,
    )

    for n_disconnections in range(1, 4):
        disconnection_indices_batch = random_disconnection_indices(
            rng_key=jax.random.PRNGKey(876376834678),
            n_disconnections=n_disconnections,
            batch_size=static_information.solver_config.batch_size_bsdf,
            disconnectable_branches=static_information.dynamic_information.disconnectable_branches,
            chance_for_empty_disconnection=0.1,
        )

        injections = random_injection(
            jax.random.PRNGKey(0),
            for_topology=topologies,
            n_generators_per_sub=static_information.dynamic_information.generators_per_sub,
            n_inj_per_topology=1,
        ).injection_topology

        lf_res, success = compute_symmetric_batch(
            topology_batch=action_index_topo,
            disconnection_batch=disconnection_indices_batch,
            injections=injections,
            nodal_inj_start_options=None,
            dynamic_information=static_information.dynamic_information,
            solver_config=static_information.solver_config,
        )
        # 1 Disconnection should have some successful cases
        if n_disconnections == 1:
            assert jnp.any(success)
        if jnp.any(success):
            n_0 = lf_res.n_0_matrix[success]
            n_1 = lf_res.n_1_matrix[success]
            disc = disconnection_indices_batch[success]
            for n_0_i, n_1_1, disc_i in zip(n_0, n_1, disc):
                # Filter out disconnections that are padded
                disc_i = disc_i[disc_i != int_max()]
                disc_i = static_information.dynamic_information.disconnectable_branches[disc_i]
                assert jnp.allclose(n_0_i[:, disc_i], 0)
                assert jnp.allclose(n_1_1[:, :, disc_i], 0)


def test_compute_symmetric_batch_nminus2(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    solver_config = static_information.solver_config
    dynamic_information = static_information.dynamic_information

    baseline = unsplit_n_2_analysis(dynamic_information=dynamic_information, more_splits_penalty=100.0)

    dynamic_information = replace(dynamic_information, n2_baseline_analysis=baseline)

    topologies = batch_topologies(topologies, static_information.solver_config.batch_size_bsdf)[0]
    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=topologies,
        branch_actions=dynamic_information.action_set,
    )
    injections = default_injection(
        n_splits=action_index_topo.action.shape[1],
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        batch_size=action_index_topo.action.shape[0],
    )

    lf_res, success = compute_symmetric_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injections=injections.injection_topology,
        nodal_inj_start_options=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    assert jnp.all(success)
    assert lf_res.n_2_penalty is not None
    assert lf_res.n_2_penalty.shape == success.shape
    assert jnp.all(lf_res.n_2_penalty >= 0.0)
    # Unsplit grid is first topology
    assert lf_res.n_2_penalty[0] == 0


def test_compute_symmetric_batch_multiple_timesteps(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    n_timesteps = 3
    topologies, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    action_index_topo = convert_topo_to_action_set_index_jittable(
        topologies=topologies,
        branch_actions=dynamic_information.action_set,
    )[0 : solver_config.batch_size_bsdf]

    single_timestep_ref, success = compute_symmetric_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    assert jnp.all(success)

    dynamic_information = replace(
        dynamic_information,
        unsplit_flow=jnp.repeat(
            dynamic_information.unsplit_flow,
            n_timesteps,
            axis=0,
        ),
        nodal_injections=jnp.repeat(
            dynamic_information.nodal_injections,
            n_timesteps,
            axis=0,
        ),
        relevant_injections=jnp.repeat(
            dynamic_information.relevant_injections,
            n_timesteps,
            axis=0,
        ),
        nonrel_injection_outage_deltap=jnp.repeat(dynamic_information.nonrel_injection_outage_deltap, n_timesteps, axis=0),
    )

    multi_timestep, success = compute_symmetric_batch(
        topology_batch=action_index_topo,
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    assert jnp.all(success)

    assert multi_timestep.n_1_matrix.shape == (
        solver_config.batch_size_bsdf,
        n_timesteps,
        dynamic_information.n_nminus1_cases,
        dynamic_information.n_branches_monitored,
    )
    assert jnp.allclose(multi_timestep.n_1_matrix[:, 0], single_timestep_ref.n_1_matrix[:, 0])
