import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.disconnections import apply_single_disconnection_lodf
from toop_engine_dc_solver.jax.multi_outages import (
    apply_modf_matrix,
    build_modf_matrix,
    compute_multi_outage,
    update_ptdf_with_modf,
)
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)


# Have two reference functions for the multi-outage computation,
# one using the ptdf and one using the MODF formulation
def compute_multi_outage_ptdf(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    multi_outages: Int[Array, " max_n_outaged_branches"],
) -> tuple[Float[Array, "n_branches n_bus"], Float[Array, " n_timesteps n_branches"], Bool[Array, " "]]:
    """Compute the flow after a single multi-outage using ptdf updates

    A multi-outage outages a set of branches and zeros injections to a set of nodes.
    They can be used to represent trafo3w or busbar outages in the N-1 contingency analysis.

    This function computes the entire flow after the multi-outages in one go instead of the
    build/apply approach. It is not as efficient as the build/apply approach and not as efficient
    as the MODF formulation, but it can be used as a reference implementation.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after all busbar splits/branch outages that are part of actions.
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    nodal_injections : Float[Array, " n_timesteps n_bus"]
        The nodal injection vector
    multi_outages : Int[Array, " max_n_outaged_branches"]
        The branches to be outaged. Can be padded with invalid branch indices.

    Returns
    -------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after the multi-outages
    Float[Array, " n_timesteps n_branches"]
        The flow after the multi-outages
    Bool[Array, " "]
        A boolean vector indicating whether the multi-outage was valid or not.
    """
    # Run using the LODF method
    ptdf, success = jax.lax.scan(
        lambda new_ptdf, outage: apply_single_disconnection_lodf(
            disconnection=outage, ptdf=new_ptdf, from_node=from_node, to_node=to_node
        ),
        ptdf,
        multi_outages,
    )
    assert isinstance(ptdf, Float[Array, " n_branches n_bus"])

    res_flows = jnp.einsum("ij,tj->ti", ptdf, nodal_injections)
    assert isinstance(res_flows, Float[Array, " n_timesteps n_branches"])

    return ptdf, res_flows, success


def test_build_modf_matrix(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node
    nodal_injections = static_information.dynamic_information.nodal_injections

    n_0_flows = jnp.einsum("ij,tj -> ti", ptdf, nodal_injections)

    outage_scenarios = [[0], [0, 8], [0, 8, 12], [12, 8], [12, 8, 0], [0, 4, 8]]

    for outage in outage_scenarios:
        outage = jnp.array(outage, dtype=int)

        _, flows_ref, success = compute_multi_outage_ptdf(
            ptdf,
            from_node,
            to_node,
            nodal_injections,
            outage,
        )
        assert jnp.allclose(flows_ref[:, outage], 0)
        assert jnp.all(success)

        modf_matrix, success = build_modf_matrix(ptdf, from_node, to_node, outage)
        assert jnp.all(success)

        flows = apply_modf_matrix(
            modf_matrix=modf_matrix,
            n_0_flow=n_0_flows,
            branches_monitored=None,
        )
        assert jnp.allclose(flows[:, outage], 0)
        assert flows.shape == flows_ref.shape
        assert jnp.allclose(flows, flows_ref)

        flows_end_to_end, success = compute_multi_outage(
            ptdf=ptdf,
            from_node=from_node,
            to_node=to_node,
            n_0_flow=n_0_flows,
            multi_outages=outage,
            branches_monitored=None,
        )
        assert jnp.all(success)
        assert flows_end_to_end.shape == flows_ref.shape
        assert jnp.allclose(flows_end_to_end, flows_ref)


def test_padded_outages(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node
    nodal_injections = static_information.dynamic_information.nodal_injections

    n_0_flows = jnp.einsum("ij,tj -> ti", ptdf, nodal_injections)

    outage_scenarios = [[0, 8, -1], [-1, 0, 8], [0, -1, 8], [0, 8]]

    for outage in outage_scenarios:
        outage = jnp.array(outage, dtype=int)
        real_outages = outage[outage >= 0]

        _, flows_ref, success = compute_multi_outage_ptdf(
            ptdf,
            from_node,
            to_node,
            nodal_injections,
            outage,
        )
        assert jnp.all(success)
        assert jnp.allclose(flows_ref[:, real_outages], 0)

        modf_matrix, success = build_modf_matrix(ptdf, from_node, to_node, outage)
        assert jnp.all(success)

        flows = apply_modf_matrix(
            modf_matrix,
            n_0_flows,
            static_information.dynamic_information.branches_monitored,
        )
        assert flows.shape == flows_ref.shape
        assert jnp.allclose(flows[:, real_outages], 0)
        assert jnp.allclose(flows, flows_ref)


# @pytest.mark.skip(
#     reason="Fails due to MODF not detecting splits - find a way how to detect splits"
# )
def test_detects_splits(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node
    nodal_injections = static_information.dynamic_information.nodal_injections

    outage_scenarios = [[0, 1, 2, 3, 4, 5], [13, 14, 18]]
    for outage in outage_scenarios:
        outage = jnp.array(outage, dtype=int)

        _, success = build_modf_matrix(ptdf, from_node, to_node, outage)
        assert not jnp.all(success)

        _, _, success = compute_multi_outage_ptdf(
            ptdf,
            from_node,
            to_node,
            nodal_injections,
            outage,
        )
        assert not jnp.all(success)


def test_build_modf_matrices_branches_monitored(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs

    ptdf = static_information.dynamic_information.ptdf
    from_node = static_information.dynamic_information.from_node
    to_node = static_information.dynamic_information.to_node
    nodal_injections = static_information.dynamic_information.nodal_injections
    branches_monitored = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12])

    n_0_flows = jnp.einsum("ij,tj -> ti", ptdf, nodal_injections)

    outage_scenarios = [[0], [0, 8], [0, 8, 12], [12, 8], [12, 8, 0], [0, 4, 8]]

    for outage in outage_scenarios:
        outage = jnp.array(outage, dtype=int)

        modf_matrix, success = build_modf_matrix(ptdf, from_node, to_node, outage)
        assert jnp.all(success)
        flows = apply_modf_matrix(
            modf_matrix,
            n_0_flows,
            branches_monitored,
        )

        _, flows_ref, success = compute_multi_outage_ptdf(
            ptdf,
            from_node,
            to_node,
            nodal_injections,
            outage,
        )
        assert jnp.all(success)

        assert flows.shape == (nodal_injections.shape[0], len(branches_monitored))
        # assert flows.shape == flows_2.shape
        # assert jnp.allclose(flows, flows_2)
        assert jnp.allclose(flows, flows_ref[:, branches_monitored])


def test_update_ptdf_with_modf(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    dynamic_information = static_information.dynamic_information

    outage_scenarios = [
        [0],
        [0, 8],
        [0, 8, 12],
        [12, 8],
        [12, 8, 0],
        [0, 4, 8],
        [-1, 0, 4, 8],
        [0, 4, 8, 99999],
    ]

    for outage in outage_scenarios:
        modf, success = build_modf_matrix(
            static_information.dynamic_information.ptdf,
            static_information.dynamic_information.from_node,
            static_information.dynamic_information.to_node,
            jnp.array(outage),
        )
        assert jnp.all(success)

        ptdf_new = update_ptdf_with_modf(
            modf,
            static_information.dynamic_information.ptdf,
        )

        ptdf_ref, _, success = compute_multi_outage_ptdf(
            ptdf=dynamic_information.ptdf,
            from_node=dynamic_information.from_node,
            to_node=dynamic_information.to_node,
            nodal_injections=dynamic_information.nodal_injections,
            multi_outages=jnp.array(outage, dtype=int),
        )
        assert jnp.all(success)

        assert ptdf_new.shape == ptdf_ref.shape
        assert jnp.allclose(ptdf_new, ptdf_ref)
