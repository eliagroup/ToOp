import jax
from jax import numpy as jnp
from toop_engine_dc_solver.jax.disconnections import (
    apply_disconnections,
    apply_single_disconnection_lodf,
    enumerate_disconnectable_branches,
    random_disconnections,
    update_from_to_nodes_after_disconnections,
)
from toop_engine_dc_solver.jax.lodf import calc_lodf, calc_lodf_matrix
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
)


def test_apply_disconnections_empty_disconnections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node

    disconnections = jnp.array([int_max(), int_max(), int_max()])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success

    assert success.shape == ()
    assert jnp.all(success)
    assert ptdf_new.shape == ptdf.shape
    assert jnp.allclose(ptdf_new, ptdf)

    disconnections = jnp.array([])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success

    assert success.shape == ()
    assert jnp.all(success)
    assert ptdf_new.shape == ptdf.shape
    assert jnp.allclose(ptdf_new, ptdf)

    disconnections = jnp.array([int_max()])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success

    assert success.shape == ()
    assert jnp.all(success)
    assert ptdf_new.shape == ptdf.shape
    assert jnp.allclose(ptdf_new, ptdf)

    disconnections = jnp.array([int_max(), int_max()])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success

    assert success.shape == ()
    assert jnp.all(success)
    assert ptdf_new.shape == ptdf.shape
    assert jnp.allclose(ptdf_new, ptdf)


def test_apply_disconnections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node

    disconnections = jnp.array([0, 8, 12])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success

    assert success.shape == ()
    assert jnp.all(success)
    assert ptdf_new.shape == ptdf.shape

    assert not jnp.allclose(ptdf_new, ptdf)
    assert jnp.allclose(ptdf_new[disconnections], 0)

    flows = jnp.dot(ptdf_new, static_information.dynamic_information.nodal_injections[0])
    assert jnp.allclose(flows[disconnections], 0)

    # Validate against repeated single variant
    ptdf_single = ptdf.copy()
    for disconnection in disconnections:
        ptdf_single, success = apply_single_disconnection_lodf(
            disconnection=disconnection,
            ptdf=ptdf_single,
            from_node=from_node,
            to_node=to_node,
        )
        assert success.item()
    assert ptdf_single.shape == ptdf.shape
    assert jnp.allclose(ptdf_new, ptdf_single)

    # It's possible to mask disconnections by giving an invalid outage index
    disconnections = jnp.array([0, 999999, 8, 12, -1, 1000])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new_2, success = disc_res.ptdf, disc_res.success

    assert jnp.all(success)
    assert ptdf_new_2.shape == ptdf.shape
    assert jnp.allclose(ptdf_new_2, ptdf_new)

    # The order of disconnections does not matter
    disconnections = jnp.array([12, 8, 0])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new_3, success = disc_res.ptdf, disc_res.success

    assert jnp.all(success)
    assert ptdf_new_3.shape == ptdf.shape
    assert jnp.allclose(ptdf_new_3, ptdf_new)

    # Too many disconnections lead to a split
    disconnections = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    assert not jnp.all(disc_res.success)

    # disconnections can be repeated
    disconnections = jnp.array([0, 0, 0, 0, 8, 12, 12])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new_4, success = disc_res.ptdf, disc_res.success

    assert jnp.all(success)
    assert ptdf_new_4.shape == ptdf.shape
    assert jnp.allclose(ptdf_new_4, ptdf_new)

    # disconnections can be empty
    disconnections = jnp.array([], dtype=int)

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new_5, success = disc_res.ptdf, disc_res.success

    assert jnp.all(success)
    assert jnp.array_equal(ptdf_new_5, ptdf)


def test_contingency_compatible(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information
    static_information.solver_config

    ptdf = dynamic_information.ptdf
    from_node = dynamic_information.from_node
    to_node = dynamic_information.to_node

    disconnections = dynamic_information.disconnectable_branches[0:1]

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf, success = disc_res.ptdf, disc_res.success

    lodf, lodf_success = calc_lodf_matrix(
        dynamic_information.branches_to_fail,
        ptdf,
        from_node,
        to_node,
        dynamic_information.branches_monitored,
    )

    brh = disconnections.item()
    assert jnp.all(lodf[:brh, brh] == 0)
    assert lodf[brh, brh].item() == -1
    assert jnp.all(lodf[brh + 1 :, brh] == 0)

    assert jnp.all(lodf_success)

    # Too many disconnections lead to a split in the contingency analysis
    disconnections = jnp.array([0, 8, 12])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf, success = disc_res.ptdf, disc_res.success

    lodf, lodf_success = calc_lodf_matrix(
        dynamic_information.branches_to_fail,
        ptdf,
        from_node,
        to_node,
        dynamic_information.branches_monitored,
    )

    assert jnp.all(success)
    assert not jnp.all(lodf_success)


def test_equivalent_to_lodf(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node

    disconnections = jnp.array([0])

    unsplit_flows = jnp.einsum("ij,j->i", ptdf, static_information.dynamic_information.nodal_injections[0])

    disc_res = apply_disconnections(ptdf, from_node, to_node, disconnections)
    ptdf_new, success = disc_res.ptdf, disc_res.success
    assert jnp.all(success)

    split_flows = jnp.einsum("ij,j->i", ptdf_new, static_information.dynamic_information.nodal_injections[0])

    lodf, succ = calc_lodf(
        branch_to_outage=disconnections[0],
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        branches_monitored=static_information.dynamic_information.branches_monitored,
    )
    assert succ

    delta_flow = lodf * unsplit_flows[disconnections[0]]
    flows = unsplit_flows + delta_flow

    assert jnp.allclose(flows, split_flows)


def test_random_disconnection() -> None:
    disconnectable_branches = jnp.arange(90) + 10

    x = random_disconnections(jax.random.PRNGKey(0), 10, 5, disconnectable_branches, 0.0)

    assert x.shape == (10, 5)
    assert jnp.all(x >= 10)
    assert jnp.all(x < 100)

    x = random_disconnections(jax.random.PRNGKey(0), 10, 5, disconnectable_branches, 1.0)

    assert x.shape == (10, 5)
    assert jnp.all(x == int_max())


def test_enumerate_disconnectable_branches(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    disconnectable_branches = enumerate_disconnectable_branches(
        static_information.dynamic_information.from_node,
        static_information.dynamic_information.to_node,
    )

    assert len(disconnectable_branches)
    assert jnp.all(disconnectable_branches >= 0)
    assert jnp.all(disconnectable_branches < static_information.dynamic_information.ptdf.shape[0])


def test_update_from_to_nodes_after_disconnections() -> None:
    from_nodes = jnp.array([0, 1, 2, 3, 4, 5], dtype=int)
    to_nodes = jnp.array([5, 4, 3, 2, 1, 0], dtype=int)
    disconnections = jnp.array([1, 3], dtype=int)

    updated_from_nodes, updated_to_nodes = update_from_to_nodes_after_disconnections(from_nodes, to_nodes, disconnections)

    expected_from_nodes = jnp.array([0, jnp.iinfo(jnp.int32).max, 2, jnp.iinfo(jnp.int32).max, 4, 5], dtype=int)
    expected_to_nodes = jnp.array([5, jnp.iinfo(jnp.int32).max, 3, jnp.iinfo(jnp.int32).max, 1, 0], dtype=int)

    assert jnp.array_equal(updated_from_nodes, expected_from_nodes)
    assert jnp.array_equal(updated_to_nodes, expected_to_nodes)

    # Test with no disconnections
    disconnections = jnp.array([], dtype=int)

    updated_from_nodes, updated_to_nodes = update_from_to_nodes_after_disconnections(from_nodes, to_nodes, disconnections)

    assert jnp.array_equal(updated_from_nodes, from_nodes)
    assert jnp.array_equal(updated_to_nodes, to_nodes)

    # Test with all nodes disconnected
    disconnections = jnp.array([0, 1, 2, 3, 4, 5], dtype=int)

    updated_from_nodes, updated_to_nodes = update_from_to_nodes_after_disconnections(from_nodes, to_nodes, disconnections)

    expected_from_nodes = jnp.full_like(from_nodes, jnp.iinfo(jnp.int32).max)
    expected_to_nodes = jnp.full_like(to_nodes, jnp.iinfo(jnp.int32).max)

    assert jnp.array_equal(updated_from_nodes, expected_from_nodes)
    assert jnp.array_equal(updated_to_nodes, expected_to_nodes)
