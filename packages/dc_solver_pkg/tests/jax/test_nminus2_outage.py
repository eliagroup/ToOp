# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits
from toop_engine_dc_solver.jax.compute_batch import compute_bsdf_lodf_static_flows
from toop_engine_dc_solver.jax.contingency_analysis import calc_n_1_matrix
from toop_engine_dc_solver.jax.disconnections import apply_disconnections
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix, get_failure_cases_to_zero
from toop_engine_dc_solver.jax.nminus2_outage import (
    SplitAggregator,
    gather_l1_cases,
    n_2_analysis,
    run_single_l1_case,
    split_n_2_analysis,
    unsplit_n_2_analysis,
)
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import (
    DynamicInformation,
    InjectionComputations,
    SolverConfig,
    StaticInformation,
    TopoVectBranchComputations,
)


def solver_reference(
    topology: TopoVectBranchComputations,
    disconnections: Optional[Int[Array, " n_disconnections"]],
    l1_outages: Int[Array, " n_l1_outages"],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> tuple[
    Float[Array, " n_l1_outages n_timesteps n_branches_monitored"],
    Float[Array, " n_l1_outages n_timesteps n_l2_outages n_branches_monitored"],
    Bool[Array, " n_l1_outages"],
    Bool[Array, " n_l1_outages n_l2_outages"],
]:
    """A solver reference to compare to, using the disconnections mechanic. Can only do a single
    topology at a time and has no injection mechanic"""

    if disconnections is None:
        disconnections = jnp.array([], dtype=int)

    # Repeat the disconnections and concatenate the l1_outages
    assert len(disconnections.shape) == 1
    assert len(l1_outages.shape) == 1
    assert len(l1_outages)
    disconnections = jnp.repeat(disconnections[None], l1_outages.shape[0], axis=0)
    disconnections = jnp.concatenate([disconnections, l1_outages[:, None]], axis=1)

    # Repeat the topology and injection
    assert len(topology.topologies.shape) == 2
    topology_batch = TopoVectBranchComputations(
        topologies=jnp.repeat(topology.topologies[None], l1_outages.shape[0], axis=0),
        sub_ids=jnp.repeat(topology.sub_ids[None], l1_outages.shape[0], axis=0),
        pad_mask=jnp.repeat(topology.pad_mask[None], l1_outages.shape[0], axis=0),
    )

    # Run the "solver", code copied from compute_batch
    topo_res = jax.vmap(
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
    )(topology_batch.topologies, topology_batch.sub_ids)

    assert jnp.all(topo_res.success)

    disc_res = jax.vmap(
        partial(
            apply_disconnections,
            guarantee_unique=True,
        ),
        in_axes=(0, 0, 0, 0),
    )(topo_res.ptdf, topo_res.from_node, topo_res.to_node, disconnections)
    new_ptdf, l1_success = disc_res.ptdf, disc_res.success

    lodf, l2_success = jax.vmap(calc_lodf_matrix, in_axes=(None, 0, 0, 0, None))(
        dynamic_information.branches_to_fail,
        new_ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.branches_monitored,
    )

    failure_cases_to_zero = jax.vmap(get_failure_cases_to_zero, in_axes=(0, None))(
        disconnections, dynamic_information.branches_to_fail
    )

    # x = batch, b = branch, n = node, t = timestep
    n_0_all = jnp.einsum("xbn,tn->xtb", new_ptdf, dynamic_information.nodal_injections)
    n_0 = n_0_all[:, :, dynamic_information.branches_monitored]
    n_1 = jax.vmap(calc_n_1_matrix, in_axes=(0, None, 0, 0))(
        lodf,
        dynamic_information.branches_to_fail,
        n_0_all,
        n_0,
    )
    n_1 = jnp.where(failure_cases_to_zero[:, None, :, None], 0.0, n_1)
    n_1 = jnp.where(l2_success[:, None, :, None], n_1, 0.0)

    return n_0, n_1, l1_success, l2_success


def test_run_single_l1_case(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    # The result of the solver looping through disconnections should be the same as running the N-2
    # analysis function
    # However we have to remove all injection and multi outages before
    dynamic_information = replace(
        dynamic_information,
        multi_outage_branches=[],
        multi_outage_nodes=[],
        nonrel_injection_outage_deltap=jnp.zeros((0, 1), dtype=float),
        nonrel_injection_outage_node=jnp.zeros((0,), dtype=int),
    )

    l1_outage = jnp.array(3, dtype=int)

    n_0_ref, n_1_ref, l1_success, l2_success = solver_reference(
        topology=default_topology(solver_config, batch_size=1, topo_vect_format=True)[0],
        disconnections=None,
        l1_outages=l1_outage[None],
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    assert jnp.all(l1_success)
    assert jnp.all(l2_success)

    (l1_branch, n_2, l1_success, l2_success) = run_single_l1_case(
        l1_branch=l1_outage,
        topological_disconnections=jnp.array([], dtype=int),
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        l2_outages=dynamic_information.branches_to_fail,
        branches_monitored=dynamic_information.branches_monitored,
        aggregator=lambda l1_branch, n_2, l1_success, l2_success: (
            l1_branch,
            n_2,
            l1_success,
            l2_success,
        ),
    )

    assert l1_success.item()
    assert jnp.all(l2_success)
    assert l1_outage.item() == l1_branch.item()
    assert n_1_ref[0].shape == n_2.shape
    assert jnp.allclose(n_1_ref[0], n_2)


def test_n_2_analysis_against_solver(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    # The result of the solver looping through disconnections should be the same as running the N-2
    # analysis function
    # However we have to remove all injection and multi outages before
    dynamic_information = replace(
        dynamic_information,
        multi_outage_branches=[],
        multi_outage_nodes=[],
        nonrel_injection_outage_deltap=jnp.zeros((0, 1), dtype=float),
        nonrel_injection_outage_node=jnp.zeros((0,), dtype=int),
    )

    l1_outages = jnp.arange(static_information.n_branches, dtype=int)

    n_1_ref, n_2_ref, l1_success, l2_success_ref = solver_reference(
        topology=default_topology(solver_config, batch_size=1, topo_vect_format=True)[0],
        disconnections=None,
        l1_outages=l1_outages,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    # Only use the L1 successful cases
    n_1_ref = n_1_ref[l1_success]
    n_2_ref = n_2_ref[l1_success]
    l1_outages = l1_outages[l1_success]
    l2_success_ref = l2_success_ref[l1_success]
    assert n_1_ref.shape[0] > 0, "No successful cases found"

    (l1_branch, n_2, l1_success, l2_success), ignore = n_2_analysis(
        l1_outages=l1_outages,
        topological_disconnections=None,
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        l2_outages=dynamic_information.branches_to_fail,
        branches_monitored=dynamic_information.branches_monitored,
        aggregator=lambda l1_branch, n_2, l1_success, l2_success: (
            l1_branch,
            n_2,
            l1_success,
            l2_success,
        ),
    )
    assert not jnp.any(ignore)
    assert jnp.all(l1_success)
    assert jnp.array_equal(l2_success, l2_success_ref)
    assert jnp.array_equal(l1_outages, l1_branch)

    assert n_2.shape == n_2_ref.shape
    assert jnp.allclose(n_2, n_2_ref)


def test_n_2_padded_elements(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    # The result of the solver looping through disconnections should be the same as running the N-2
    # analysis function
    # However we have to remove all injection and multi outages before
    dynamic_information = replace(
        dynamic_information,
        multi_outage_branches=[],
        multi_outage_nodes=[],
        nonrel_injection_outage_deltap=jnp.zeros((0, 1), dtype=float),
        nonrel_injection_outage_node=jnp.zeros((0,), dtype=int),
    )

    l1_outages = jnp.arange(static_information.n_branches, dtype=int)

    n_0_ref, n_1_ref, l1_success, l2_success = solver_reference(
        topology=default_topology(solver_config, batch_size=1, topo_vect_format=True)[0],
        disconnections=None,
        l1_outages=l1_outages,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )
    # Only use fully successful cases
    full_success = l1_success & jnp.all(l2_success, axis=-1)
    n_0_ref = n_0_ref[full_success]
    n_1_ref = n_1_ref[full_success]
    l1_outages = l1_outages[full_success]
    assert n_0_ref.shape[0] > 0, "No successful cases found"

    # Add padded elements
    l1_outages = jnp.insert(l1_outages, jnp.array([1, 3, 4]), 99999)

    (l1_branch, n_2, l1_success, lodf_success), ignore = n_2_analysis(
        l1_outages=l1_outages,
        topological_disconnections=None,
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        l2_outages=dynamic_information.branches_to_fail,
        branches_monitored=dynamic_information.branches_monitored,
        aggregator=lambda l1_branch, n_2, l1_success, lodf_success: (
            l1_branch,
            n_2,
            l1_success,
            lodf_success,
        ),
    )
    assert jnp.array_equal(ignore, l1_outages == 99999)
    assert jnp.all(l1_branch[ignore] == 0)
    assert jnp.all(n_2[ignore] == 0)
    assert jnp.all(~l1_success[ignore])
    assert jnp.all(~lodf_success[ignore])

    assert jnp.all(l1_success[~ignore])
    assert jnp.all(lodf_success[~ignore])
    assert jnp.array_equal(l1_outages[~ignore], l1_branch[~ignore])
    assert n_2[~ignore].shape == n_1_ref.shape
    assert jnp.allclose(n_2[~ignore], n_1_ref)


def test_unsplit_n_2_analysis(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information

    baseline = unsplit_n_2_analysis(dynamic_information, 1000.0)

    # -1 because of int_max padding
    assert baseline.l1_branches.shape[0] == jnp.unique(baseline.tot_stat_blacklisted.flatten()).shape[0] - 1
    assert baseline.l1_branches.size
    assert baseline.n_2_overloads.shape == baseline.l1_branches.shape
    assert baseline.n_2_success_count.shape == baseline.l1_branches.shape
    assert jnp.all(baseline.n_2_success_count > 0)
    assert jnp.all(jnp.isfinite(baseline.n_2_overloads))
    assert jnp.all(baseline.n_2_overloads >= 0)
    assert jnp.unique(baseline.l1_branches).shape[0] == baseline.l1_branches.shape[0]
    assert baseline.more_splits_penalty == 1000.0
    assert jnp.array_equal(baseline.max_mw_flow, dynamic_information.branch_limits.max_mw_flow)
    if dynamic_information.branch_limits.overload_weight is not None:
        assert jnp.array_equal(baseline.overload_weight, dynamic_information.branch_limits.overload_weight)
    else:
        assert baseline.overload_weight is None


def test_gather_l1_cases_no_splits() -> None:
    has_splits = jnp.array([False, False, False], dtype=bool)
    sub_ids = jnp.array([0, 1, 2], dtype=int)
    tot_stat = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    disconnections = None

    result = gather_l1_cases(has_splits, sub_ids, tot_stat, disconnections)
    expected = jnp.array([jnp.iinfo(tot_stat.dtype).max] * 6, dtype=int)

    assert jnp.array_equal(result, expected)


def test_gather_l1_cases_with_splits() -> None:
    has_splits = jnp.array([True, False, True], dtype=bool)
    sub_ids = jnp.array([0, 1, 2], dtype=int)
    tot_stat = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    disconnections = None

    result = jnp.sort(gather_l1_cases(has_splits, sub_ids, tot_stat, disconnections))
    expected = jnp.array([1, 2, 5, 6, jnp.iinfo(tot_stat.dtype).max, jnp.iinfo(tot_stat.dtype).max])

    assert jnp.array_equal(result, expected)


def test_gather_l1_cases_out_of_bounds_sub_ids() -> None:
    has_splits = jnp.array([True, True], dtype=bool)
    sub_ids = jnp.array([0, 3], dtype=int)  # 3 is out of bounds
    tot_stat = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    disconnections = None

    result = gather_l1_cases(has_splits, sub_ids, tot_stat, disconnections)
    expected = jnp.array([1, 2, jnp.iinfo(tot_stat.dtype).max, jnp.iinfo(tot_stat.dtype).max], dtype=int)

    assert jnp.array_equal(result, expected)


def test_gather_l1_cases_duplicate_branches() -> None:
    has_splits = jnp.array([True, True], dtype=bool)
    sub_ids = jnp.array([0, 1], dtype=int)
    tot_stat = jnp.array([[1, 2], [1, 3]], dtype=int)  # Duplicate branch 1
    disconnections = None

    result = jnp.sort(gather_l1_cases(has_splits, sub_ids, tot_stat, disconnections))
    expected = jnp.array([1, 2, 3, jnp.iinfo(tot_stat.dtype).max])

    assert jnp.array_equal(result, expected)


def test_gather_l1_cases_with_disconnections() -> None:
    has_splits = jnp.array([True, True], dtype=bool)
    sub_ids = jnp.array([0, 1], dtype=int)
    tot_stat = jnp.array([[1, 2], [3, 4]], dtype=int)
    disconnections = jnp.array([1, 3], dtype=int)

    result = jnp.sort(gather_l1_cases(has_splits, sub_ids, tot_stat, disconnections))
    expected = jnp.array([2, 4, jnp.iinfo(tot_stat.dtype).max, jnp.iinfo(tot_stat.dtype).max])

    assert jnp.array_equal(result, expected)


def test_split_analysis(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    baseline = unsplit_n_2_analysis(dynamic_information, 1000.0)

    topology = default_topology(solver_config, batch_size=1, topo_vect_format=True)

    # The penalty should be 0 if no splits were actually performed
    penalty = split_n_2_analysis(
        has_splits=jnp.any(topology.topologies, axis=-1)[0],
        sub_ids=topology.sub_ids[0],
        disconnections=None,
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        l2_outages=dynamic_information.branches_to_fail,
        baseline=baseline,
        branches_monitored=dynamic_information.branches_monitored,
    )

    assert jnp.isclose(penalty, 0)

    # With a split, it should be positive
    topology = topologies[1:2]

    topo_res = compute_bsdf_lodf_static_flows(
        topology_batch=topology,
        disconnection_batch=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )

    penalty = split_n_2_analysis(
        has_splits=jnp.any(topology.topologies, axis=-1)[0],
        sub_ids=topology.sub_ids[0],
        disconnections=None,
        nodal_injections=dynamic_information.nodal_injections,
        ptdf=topo_res.ptdf[0],
        from_node=topo_res.from_node[0],
        to_node=topo_res.to_node[0],
        l2_outages=dynamic_information.branches_to_fail,
        baseline=baseline,
        branches_monitored=dynamic_information.branches_monitored,
    )

    assert not jnp.isclose(penalty, 0)
    assert jnp.isfinite(penalty)
    assert penalty > 0

    # TODO compute within compute_batch and check if the same penalty came out


def test_split_analysis_against_solver(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    baseline = unsplit_n_2_analysis(dynamic_information, 0.0)
    # Change the overloads to all zeros so that the penalty is just the sum of N-2 overloads
    baseline = replace(
        baseline,
        n_2_overloads=jnp.zeros_like(baseline.n_2_overloads),
        n_2_success_count=jnp.zeros_like(baseline.n_2_success_count),
    )

    aggregator_ref = SplitAggregator(baseline)

    # Randomly choose a topology to try
    np.random.seed(0)
    topologies = topologies[1:]
    topos_to_try = np.random.choice(topologies.topologies.shape[0], 2, replace=False)

    for topo_idx in topos_to_try:
        topology = topologies[topo_idx : topo_idx + 1]

        topo_res = compute_bsdf_lodf_static_flows(
            topology_batch=topology,
            disconnection_batch=None,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )

        topology = topology[0]

        penalty = split_n_2_analysis(
            has_splits=jnp.any(topology.topologies, axis=-1),
            sub_ids=topology.sub_ids,
            disconnections=None,
            nodal_injections=dynamic_information.nodal_injections,
            ptdf=topo_res.ptdf[0],
            from_node=topo_res.from_node[0],
            to_node=topo_res.to_node[0],
            l2_outages=dynamic_information.branches_to_fail,
            baseline=baseline,
            branches_monitored=dynamic_information.branches_monitored,
        )

        l1_outages = gather_l1_cases(
            has_splits=jnp.any(topology.topologies, axis=-1),
            sub_ids=topology.sub_ids,
            tot_stat=baseline.tot_stat_blacklisted,
            topological_disconnections=None,
        )

        l1_outages = l1_outages[l1_outages != jnp.iinfo(l1_outages.dtype).max]

        penalty_direct, ignores = n_2_analysis(
            l1_outages=l1_outages,
            topological_disconnections=None,
            nodal_injections=dynamic_information.nodal_injections,
            ptdf=topo_res.ptdf[0],
            from_node=topo_res.from_node[0],
            to_node=topo_res.to_node[0],
            l2_outages=dynamic_information.branches_to_fail,
            branches_monitored=dynamic_information.branches_monitored,
            aggregator=aggregator_ref,
        )
        assert not jnp.any(ignores)

        n_1_ref, n_2_ref, l1_success_ref, l2_success_ref = solver_reference(
            topology=topology,
            disconnections=None,
            l1_outages=l1_outages,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )

        penalty_ref = jax.vmap(aggregator_ref)(
            l1_outages,
            n_2_ref,
            l1_success_ref,
            l2_success_ref,
        )

        assert jnp.allclose(penalty_ref, penalty_direct)
        assert jnp.isclose(penalty, penalty_ref.sum())

        # Try again with a disconnection
        topology = topologies[topo_idx : topo_idx + 1]
        disconnection = jnp.array([[5]], dtype=int)

        topo_res = compute_bsdf_lodf_static_flows(
            topology_batch=topology,
            disconnection_batch=disconnection,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )

        topology = topology[0]
        disconnection = disconnection[0]

        penalty = split_n_2_analysis(
            has_splits=jnp.any(topology.topologies, axis=-1),
            sub_ids=topology.sub_ids,
            disconnections=disconnection,
            nodal_injections=dynamic_information.nodal_injections,
            ptdf=topo_res.ptdf[0],
            from_node=topo_res.from_node[0],
            to_node=topo_res.to_node[0],
            l2_outages=dynamic_information.branches_to_fail,
            baseline=baseline,
            branches_monitored=dynamic_information.branches_monitored,
        )

        l1_outages = gather_l1_cases(
            has_splits=jnp.any(topology.topologies, axis=-1),
            sub_ids=topology.sub_ids,
            tot_stat=baseline.tot_stat_blacklisted,
            topological_disconnections=disconnection,
        )

        l1_outages = l1_outages[l1_outages != jnp.iinfo(l1_outages.dtype).max]

        penalty_direct, ignores = n_2_analysis(
            l1_outages=l1_outages,
            topological_disconnections=disconnection,
            nodal_injections=dynamic_information.nodal_injections,
            ptdf=topo_res.ptdf[0],
            from_node=topo_res.from_node[0],
            to_node=topo_res.to_node[0],
            l2_outages=dynamic_information.branches_to_fail,
            branches_monitored=dynamic_information.branches_monitored,
            aggregator=aggregator_ref,
        )
        assert not jnp.any(ignores)

        n_1_ref, n_2_ref, l1_success_ref, l2_success_ref = solver_reference(
            topology=topology,
            disconnections=disconnection,
            l1_outages=l1_outages,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )

        penalty_ref = jax.vmap(aggregator_ref)(
            l1_outages,
            n_2_ref,
            l1_success_ref,
            l2_success_ref,
        )

        assert jnp.allclose(penalty_ref, penalty_direct)
        assert jnp.isclose(penalty, penalty_ref.sum())
