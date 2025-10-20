from functools import partial

import jax
import jax.numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits
from toop_engine_dc_solver.jax.compute_batch import compute_injections
from toop_engine_dc_solver.jax.cross_coupler_flow import (
    compute_cross_coupler_flows,
)
from toop_engine_dc_solver.jax.injections import default_injection, random_injection
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    convert_topo_to_action_set_index,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
)


def test_compute_cross_coupler_flow_against_ptdf_no_injection(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    act_index_topos, action_set = convert_topo_to_action_set_index(
        topologies, static_information.dynamic_information.action_set, extend_action_set=True
    )
    if action_set is not None:
        dynamic_information = replace(
            dynamic_information,
            action_set=action_set,
        )

    injections = default_injection(
        n_splits=act_index_topos.action.shape[1],
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        batch_size=act_index_topos.action.shape[0],
    )

    dynamic_information = replace(
        dynamic_information,
        unsplit_flow=jnp.einsum(
            "bn,tn->tb",
            dynamic_information.ptdf,
            dynamic_information.nodal_injections,
        ),
    )

    # Compute the cross-coupler flow and the post-split N-0 flows using the cross-coupler methods
    bsdf_res = jax.vmap(
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
    )(topologies.topologies, topologies.sub_ids)

    n_0_ref = jnp.einsum("xbn, tn ->xtb", bsdf_res.ptdf, dynamic_information.nodal_injections)

    n_0, cross_coupler = jax.vmap(compute_cross_coupler_flows, in_axes=(0, 0, 0, 0, None, None, None, None))(
        bsdf_res.bsdf,
        topologies.topologies,
        topologies.sub_ids,
        injections.injection_topology,
        dynamic_information.relevant_injections,
        dynamic_information.unsplit_flow,
        dynamic_information.tot_stat,
        dynamic_information.from_stat_bool,
    )

    assert n_0.shape == n_0_ref.shape
    assert jnp.allclose(n_0, n_0_ref)

    has_splits = jnp.any(topologies.topologies, axis=-1)
    assert has_splits.shape + (1,) == cross_coupler.shape
    assert jnp.all(cross_coupler[~has_splits] == 0)


def test_compute_cross_coupler_flow_against_ptdf(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    act_index_topos, action_set = convert_topo_to_action_set_index(
        topologies, dynamic_information.action_set, extend_action_set=True
    )
    if action_set is not None:
        dynamic_information = replace(
            dynamic_information,
            action_set=action_set,
        )

    injections = random_injection(
        rng_key=jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=1,
        for_topology=topologies,
    )

    dynamic_information = replace(
        dynamic_information,
        unsplit_flow=jnp.einsum(
            "bn,tn->tb",
            dynamic_information.ptdf,
            dynamic_information.nodal_injections,
        ),
    )

    # Compute the cross-coupler flow and the post-split N-0 flows using the cross-coupler methods
    bsdf_res = jax.vmap(
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
    )(topologies.topologies, topologies.sub_ids)

    nodal_injections = compute_injections(
        injections=injections.injection_topology,
        sub_ids=topologies.sub_ids,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )

    n_0_ref = jnp.einsum("xbn, xtn ->xtb", bsdf_res.ptdf, nodal_injections)

    n_0, cross_coupler = jax.vmap(compute_cross_coupler_flows, in_axes=(0, 0, 0, 0, None, None, None, None))(
        bsdf_res.bsdf,
        topologies.topologies,
        topologies.sub_ids,
        injections.injection_topology,
        dynamic_information.relevant_injections,
        dynamic_information.unsplit_flow,
        dynamic_information.tot_stat,
        dynamic_information.from_stat_bool,
    )

    assert n_0.shape == n_0_ref.shape
    assert jnp.allclose(n_0, n_0_ref)

    has_splits = jnp.any(topologies.topologies, axis=-1)
    assert has_splits.shape + (1,) == cross_coupler.shape
    assert jnp.all(cross_coupler[~has_splits] == 0)


def test_compute_cross_coupler_flow_invalid_subid(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    act_index_topo = ActionIndexComputations(
        action=jnp.array([[int_max()]], dtype=int),
        pad_mask=jnp.array([True], dtype=bool),
    )
    topo = convert_action_set_index_to_topo(act_index_topo, dynamic_information.action_set)
    assert jnp.array_equal(topo.sub_ids, jnp.array([[int_max()]], dtype=int))
    assert jnp.array_equal(topo.topologies, jnp.zeros_like(topo.topologies))
    injections = InjectionComputations(
        injection_topology=jnp.zeros((1, 1, dynamic_information.max_inj_per_sub), dtype=bool),
        corresponding_topology=jnp.array([0], dtype=int),
        pad_mask=jnp.array([True], dtype=bool),
    )

    unsplit_flow = jnp.einsum(
        "bn,tn->tb",
        dynamic_information.ptdf,
        dynamic_information.nodal_injections,
    )

    bsdf_res = compute_bus_splits(
        topologies=topo.topologies[0],
        sub_ids=topo.sub_ids[0],
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

    n_0, cross_coupler = compute_cross_coupler_flows(
        bsdf=bsdf_res.bsdf,
        topologies=topo.topologies[0],
        substation_ids=topo.sub_ids[0],
        injections=injections.injection_topology[0],
        relevant_injections=dynamic_information.relevant_injections,
        n_0_flows=unsplit_flow,
        substation_branch_status=dynamic_information.tot_stat,
        from_stat_bool=dynamic_information.from_stat_bool,
    )

    assert jnp.allclose(cross_coupler, 0)
    assert jnp.allclose(n_0, unsplit_flow)
