import jax
import numpy as np
import pytest
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits
from toop_engine_dc_solver.jax.injections import (
    apply_reassignments_deltap,
    convert_action_index_to_numpy,
    convert_inj_candidates,
    convert_inj_topo_vect,
    convert_relevant_sub_injection_outages,
    default_injection,
    get_all_injection_outage_deltap,
    get_all_outaged_injection_nodes_after_reassignment,
    get_injection_per_bus,
    get_injection_vector,
    get_reassignment_deltap,
    get_relevant_injection_outage_deltap,
    get_single_injection_vector,
    random_injection,
)
from toop_engine_dc_solver.jax.topology_computations import (
    default_topology,
)
from toop_engine_dc_solver.jax.types import ActionSet, InjectionComputations, StaticInformation, TopoVectBranchComputations


def test_default_injection() -> None:
    inj = default_injection(n_splits=4, max_inj_per_sub=23, batch_size=99)
    assert jnp.array_equal(inj.injection_topology, jnp.zeros((99, 4, 23), dtype=bool))
    assert jnp.all(inj.pad_mask)
    assert jnp.array_equal(inj.corresponding_topology, jnp.arange(99))

    inj = default_injection(n_splits=4, max_inj_per_sub=23, batch_size=9, buffer_size=3)
    assert jnp.array_equal(inj.injection_topology, jnp.zeros((3, 9, 4, 23), dtype=bool))
    assert jnp.all(inj.pad_mask)
    assert inj.corresponding_topology.shape == (3, 9)
    assert jnp.all(inj.corresponding_topology < 9)
    assert jnp.all(inj.corresponding_topology >= 0)


def test_get_all_injection_outage_deltap(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information

    deltap = get_all_injection_outage_deltap(
        injection_outage_deltap=dynamic_information.nonrel_injection_outage_deltap,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
    )

    assert deltap.shape == (dynamic_information.n_timesteps, dynamic_information.n_inj_failures)
    assert jnp.array_equal(
        deltap[:, : dynamic_information.n_nonrel_inj_failures], dynamic_information.nonrel_injection_outage_deltap
    )


def test_random_injection(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topos, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information

    topo_batch = topos[0:2]
    inj = random_injection(
        jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=1,
        for_topology=topo_batch,
    )
    assert inj.injection_topology.shape == (2, dynamic_information.n_sub_relevant, dynamic_information.max_inj_per_sub)
    for sub, degree in enumerate(dynamic_information.generators_per_sub):
        assert not jnp.any(inj.injection_topology[:, sub, degree:])

    topo_batch = topos[0:16]
    inj = random_injection(
        jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=5,
        for_topology=topo_batch,
    )

    n_splits = topo_batch.topologies.shape[1]

    assert inj.injection_topology.shape == (16 * 5, n_splits, dynamic_information.max_inj_per_sub)
    assert inj.pad_mask.shape == (16 * 5,)
    assert inj.corresponding_topology.shape == (16 * 5,)
    assert jnp.all(inj.corresponding_topology >= 0)
    assert jnp.all(inj.corresponding_topology < 16)

    inj = random_injection(
        rng_key=jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=1,
        for_topology=default_topology(static_information.solver_config, batch_size=16, topo_vect_format=True),
    )
    assert jnp.array_equal(inj.corresponding_topology, jnp.arange(16))
    assert not jnp.any(inj.injection_topology)
    assert jnp.all(inj.pad_mask)

    inj = random_injection(
        rng_key=jax.random.PRNGKey(0),
        n_generators_per_sub=dynamic_information.generators_per_sub,
        n_inj_per_topology=1,
        for_topology=topos,
    )
    assert jnp.array_equal(inj.corresponding_topology, jnp.arange(len(topos)))
    for sub, degree in enumerate(dynamic_information.generators_per_sub):
        assert not jnp.any(inj.injection_topology[:, sub, degree:])


def test_get_injection_per_bus() -> None:
    relevant_injections = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ],
        dtype=float,
    )

    bus_a_inj, bus_b_inj = get_injection_per_bus(
        jnp.array([False, True, False], dtype=bool), jnp.array(1, dtype=int), relevant_injections
    )

    expected_bus_a_inj = jnp.array([10, 28], dtype=float)
    expected_bus_b_inj = jnp.array([5, 14], dtype=float)
    assert jnp.array_equal(bus_a_inj, expected_bus_a_inj)
    assert jnp.array_equal(bus_b_inj, expected_bus_b_inj)

    bus_a_inj, bus_b_inj = get_injection_per_bus(
        jnp.array([True, True, False], dtype=bool), jnp.array(0, dtype=int), relevant_injections
    )

    expected_bus_a_inj = jnp.array([3, 12], dtype=float)
    expected_bus_b_inj = jnp.array([3, 21], dtype=float)
    assert jnp.array_equal(bus_a_inj, expected_bus_a_inj)
    assert jnp.array_equal(bus_b_inj, expected_bus_b_inj)


def test_get_single_injection_vector() -> None:
    relevant_injections = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ],
        dtype=float,
    )
    nodal_injections = jax.random.normal(jax.random.PRNGKey(0), (2, 9), dtype=float)
    rel_stat_map = jnp.array([0, 2, 4], dtype=int)
    n_stat = 6

    injection_assignment = jnp.array([False, True, False], dtype=bool)
    sub_id = jnp.array(1, dtype=int)

    updated_nodal_injections = get_single_injection_vector(
        injection_assignment,
        sub_id,
        relevant_injections,
        nodal_injections,
        n_stat,
        rel_stat_map,
    )

    assert jnp.array_equal(updated_nodal_injections[:, rel_stat_map[sub_id]], jnp.array([10.0, 28.0], dtype=float))
    assert jnp.array_equal(updated_nodal_injections[:, n_stat + sub_id], jnp.array([5.0, 14.0], dtype=float))
    assert jnp.array_equal(updated_nodal_injections[:, : rel_stat_map[sub_id]], nodal_injections[:, : rel_stat_map[sub_id]])
    assert jnp.array_equal(
        updated_nodal_injections[:, rel_stat_map[sub_id] + 1 : n_stat + sub_id],
        nodal_injections[:, rel_stat_map[sub_id] + 1 : n_stat + sub_id],
    )
    assert jnp.array_equal(updated_nodal_injections[:, n_stat + sub_id + 1 :], nodal_injections[:, n_stat + sub_id + 1 :])

    # Test with invalid sub_id
    sub_id = jnp.array(3, dtype=int)

    updated_nodal_injections = get_single_injection_vector(
        injection_assignment,
        sub_id,
        relevant_injections,
        nodal_injections,
        n_stat,
        rel_stat_map,
    )

    assert jnp.array_equal(updated_nodal_injections, nodal_injections)


def test_convert_relevant_sub_injection_outages(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config

    # Everything on bus A
    injection_assignment = jnp.zeros((3, dynamic_information.max_inj_per_sub), dtype=bool)
    sub_ids = jnp.array([1, 2, 3], dtype=int)

    outage_node = convert_relevant_sub_injection_outages(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
        n_stat=solver_config.n_stat,
    )
    outage_deltap = get_relevant_injection_outage_deltap(
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
    )

    assert len(outage_node) == len(dynamic_information.relevant_injection_outage_idx)

    for o_node, o_deltap, o_idx, o_sub in zip(
        outage_node,
        outage_deltap,
        dynamic_information.relevant_injection_outage_idx,
        dynamic_information.relevant_injection_outage_sub,
    ):
        p = dynamic_information.relevant_injections[:, o_sub, o_idx]
        assert jnp.array_equal(p, -o_deltap)
        assert jnp.array_equal(o_node, solver_config.rel_stat_map.val[o_sub])

    # Invalid sub ids
    # Should be the same as everything on bus A
    injection_assignment = jnp.ones((3, dynamic_information.max_inj_per_sub), dtype=bool)
    sub_ids = jnp.array([9999, 9999, 9999], dtype=int)

    outage_node_inv = convert_relevant_sub_injection_outages(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
        n_stat=solver_config.n_stat,
    )

    assert jnp.array_equal(outage_node_inv, outage_node)

    # Everything on bus B
    injection_assignment = jnp.ones((3, dynamic_information.max_inj_per_sub), dtype=bool)
    sub_ids = jnp.array([1, 2, 3], dtype=int)

    outage_node_b = convert_relevant_sub_injection_outages(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
        n_stat=solver_config.n_stat,
    )

    assert len(outage_node_b) == len(dynamic_information.relevant_injection_outage_idx)

    for o_node, o_idx, o_sub in zip(
        outage_node_b,
        dynamic_information.relevant_injection_outage_idx,
        dynamic_information.relevant_injection_outage_sub,
    ):
        if o_sub in sub_ids.tolist():
            assert jnp.array_equal(o_node, solver_config.n_stat + o_sub)
        else:
            assert jnp.array_equal(o_node, solver_config.rel_stat_map.val[o_sub])

    # Everything in batch gives the same results
    injection_assignment = jnp.stack(
        [
            jnp.zeros((3, dynamic_information.max_inj_per_sub), dtype=bool),
            jnp.ones((3, dynamic_information.max_inj_per_sub), dtype=bool),
            jnp.ones((3, dynamic_information.max_inj_per_sub), dtype=bool),
        ]
    )
    sub_ids = jnp.stack([jnp.array([1, 2, 3], dtype=int)] * 3)

    outage_node_batch = get_all_outaged_injection_nodes_after_reassignment(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        nonrel_injection_outage_node=dynamic_information.nonrel_injection_outage_node,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
        n_stat=jnp.array(solver_config.n_stat),
    )
    assert outage_node_batch.shape == (3, dynamic_information.n_inj_failures)

    outage_node_batch_nonrel = outage_node_batch[:, : dynamic_information.n_nonrel_inj_failures]
    assert jnp.all(outage_node_batch_nonrel == dynamic_information.nonrel_injection_outage_node)

    outage_node_batch_rel = outage_node_batch[:, dynamic_information.n_nonrel_inj_failures :]
    assert jnp.array_equal(outage_node_batch_rel, jnp.stack([outage_node, outage_node_inv, outage_node_b]))


def test_convert_inj_candidates():
    n_gens_per_sub = np.array([2, 3, 4], dtype=int)

    inj_topo = np.array(
        [
            [[True, True, False, False], [False, False, True, True]],
            [[True, False, True, False], [False, True, False, False]],
        ],
        dtype=bool,
    )
    sub_ids = np.array([[0, 2], [1, 0]], dtype=int)

    inj_candidates = convert_inj_candidates(inj_topo, sub_ids, n_gens_per_sub)

    expected = np.array(
        [
            [
                True,
                True,  # sub 0
                False,
                False,
                False,  # sub 1
                False,
                False,
                True,
                True,  # sub 2
            ],
            [
                False,
                True,  # sub 0
                True,
                False,
                True,  # sub 1
                False,
                False,
                False,
                False,  # sub 2
            ],
        ]
    )

    assert np.array_equal(inj_candidates, expected)

    # Try the reverse direction
    reversed = convert_inj_topo_vect(
        numpy_topo_vect=inj_candidates,
        sub_ids=sub_ids,
        generators_per_sub=n_gens_per_sub,
        missing_split_behavior="raise",
    )

    assert np.array_equal(reversed, inj_topo)

    # Try with a missing subid
    sub_ids = np.array([[0, 2], [999, 0]])
    with pytest.raises(ValueError):
        convert_inj_topo_vect(inj_candidates, sub_ids, n_gens_per_sub, "raise")


def test_convert_action_index_to_numpy():
    action_index = jnp.array([[0, 1], [1, 3], [3, 0]], dtype=int)
    n_generators_per_sub = jnp.array([2, 3, 4], dtype=int)

    action_set = ActionSet(
        substation_correspondence=jnp.array([0, 1, 1, 2], dtype=int),
        inj_actions=jnp.array(
            [
                [True, True, False, False],
                [False, False, True, False],
                [True, False, True, False],
                [False, True, False, True],
            ],
            dtype=bool,
        ),
        branch_actions=jnp.zeros((4, 12), dtype=bool),
        n_actions_per_sub=jnp.array([1, 2, 1], dtype=int),
        unsplit_action_mask=jnp.zeros(4, dtype=bool),
        reassignment_distance=jnp.zeros(4, dtype=int),
    )

    inj_candidates = convert_action_index_to_numpy(action_index, action_set, n_generators_per_sub)

    expected = np.array(
        [
            [
                True,
                True,  # sub 0
                False,
                False,
                True,  # sub 1
                False,
                False,
                False,
                False,  # sub 2
            ],
            [
                False,
                False,  # sub 0
                False,
                False,
                True,  # sub 1
                False,
                True,
                False,
                True,  # sub 2
            ],
            [
                True,
                True,  # sub 0
                False,
                False,
                False,  # sub 1
                False,
                True,
                False,
                True,  # sub 2
            ],
        ]
    )

    assert np.array_equal(inj_candidates, expected)

    # Test with invalid action index
    action_index = jnp.array([[999, 999], [999, 999]], dtype=int)
    inj_candidates = convert_action_index_to_numpy(action_index, action_set, n_generators_per_sub)

    expected_invalid = np.zeros((2, 9), dtype=bool)
    assert np.array_equal(inj_candidates, expected_invalid)


def test_get_reassignment_deltap(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    sub_id = 0
    assert dynamic_information.generators_per_sub[sub_id] == 2
    reassignment = jnp.array([0, 1], dtype=bool)

    busa_idx, busb_idx, busa_deltap, busb_deltap = get_reassignment_deltap(
        injection_assignment=reassignment,
        sub_id=sub_id,
        relevant_injections=dynamic_information.relevant_injections,
        n_stat=solver_config.n_stat,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
    )
    assert busa_idx == solver_config.rel_stat_map.val[sub_id]
    assert busb_idx == solver_config.n_stat + sub_id
    assert busa_deltap == -dynamic_information.relevant_injections[:, sub_id, 1]
    assert busb_deltap == dynamic_information.relevant_injections[:, sub_id, 1]


def test_apply_reassignment_deltap(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    # Test with a single split station
    sub_id = jnp.array([0], dtype=int)
    assert dynamic_information.generators_per_sub[sub_id] == 2
    inj_reassignment = jnp.array([[0, 1]], dtype=bool)
    branch_reassignment = jnp.array([[1, 1, 0, 0, 0]], dtype=bool)

    bsdf_res = compute_bus_splits(
        topologies=branch_reassignment,
        sub_ids=sub_id,
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

    br_split_n0_flow = jnp.einsum("bn, tn -> tb", bsdf_res.ptdf, dynamic_information.nodal_injections)

    full_split_n0_flow = apply_reassignments_deltap(
        injection_assignment=inj_reassignment,
        sub_ids=sub_id,
        split_n0_flow=br_split_n0_flow,
        ptdf=bsdf_res.ptdf,
        relevant_injections=dynamic_information.relevant_injections,
        n_stat=solver_config.n_stat,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
    )

    nodal_inj_changed = get_injection_vector(
        injection_assignment=inj_reassignment,
        sub_ids=sub_id,
        relevant_injections=dynamic_information.relevant_injections,
        nodal_injections=dynamic_information.nodal_injections,
        n_stat=solver_config.n_stat,
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
    )
    full_split_n0_flow_ref = jnp.einsum("bn, tn -> tb", bsdf_res.ptdf, nodal_inj_changed)
    assert jnp.allclose(full_split_n0_flow, full_split_n0_flow_ref)
