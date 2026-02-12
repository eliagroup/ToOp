# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to compute the flow after multi-outages as part of the N-1 contingency analysis.

Some elements like trafo3ws and busbars have to be represented as multi-outages, as they correspond to
multiple outages at once.

There are fundamentally two ways to compute the flow after multi-outages:
- Update the PTDF for all except the last outage, then use the LODF
- Use the MODF to compute all the flows in one go

There are situations where the first approach is more efficient, and situations where the second
approach is more efficient. This module provides both approaches, and the user can choose which one
to use based on the specific use case.
"""

from __future__ import annotations

import jax
from beartype.typing import Optional
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.types import MODFMatrix, int_max
from toop_engine_dc_solver.jax.unrolled_linalg import solve_and_check_det

# import lineax as lx


def build_modf_matrices(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    multi_outage_branches: list[Int[Array, " n_multi_outages max_n_outaged_branches"]],
) -> tuple[list[MODFMatrix], Bool[Array, " all_multi_outages"]]:
    """Build the MODF matrix for all multi-outages

    This function computes the MODF matrix for all multi-outages. The MODF matrix gives the impact
    of a multi-outage on the flow on all other branches.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after all busbar splits/branch outages that are part of actions.
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    multi_outage_branches : list[Int[Array, " n_multi_outages max_n_outaged_branches"]]
        The list of multi-outage cases as stored in static_information.dynamic_information.multi_outage_branches

    Returns
    -------
    list[MODFMatrix]
        The MODF matrix for all multi-outages, still separated into batches
    Bool[Array, " all_multi_outages"]
        Whether the computation was successful for each outage - false if the network split
    """
    if len(multi_outage_branches) == 0:
        return [], jnp.array([], dtype=bool)

    batches = []
    success = []
    for branch in multi_outage_branches:
        modf, succ = jax.vmap(build_modf_matrix, in_axes=(None, None, None, 0))(
            ptdf,
            from_node,
            to_node,
            branch,
        )

        batches.append(modf)
        success.append(succ)
    success = jnp.concatenate(success)
    return batches, success


def build_modf_matrix(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    multi_outages: Int[Array, " n_outaged_branches"],
) -> tuple[MODFMatrix, Bool[Array, " "]]:
    """Compute the flow after a single multi-outage using the MODF formulation

    You can look up the theory in the paper:
    https://doi.org/10.1109/TPWRS.2009.2023273

    This formulation can not work with padded multi-outages and it's using a solve operation, which
    can be cheaper than compute_multi_outage.

    Obtain the actual flows using apply_modf_matrix_modf

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after all busbar splits/branch outages that are part of actions.
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branchesule
    multi_outages : Int[Array, " n_outaged_branches"]
        The branches to be outaged. This can contain padded elements, i.e. invalid branch indices.
        If such padded elements are present, the corresponding row in the MODF matrix will be zero.

    Returns
    -------
    MODFMatrix
        The MODF matrix for all multi-outages
    Bool[Array, " "]
        Whether the computation was successful (i.e. the network did not split in any of the outages)
    """
    assert len(multi_outages.shape) == 1

    # Roughly according to the paper
    # https://doi.org/10.1109/TPWRS.2009.2023273
    from_node_outage = from_node.at[multi_outages].get(mode="fill", fill_value=int_max())
    to_node_outage = to_node.at[multi_outages].get(mode="fill", fill_value=int_max())

    denom = (
        jnp.identity(len(multi_outages))
        - ptdf.at[jnp.ix_(multi_outages, from_node_outage)].get(mode="fill", fill_value=0.0)
        + ptdf.at[jnp.ix_(multi_outages, to_node_outage)].get(mode="fill", fill_value=0.0)
    )
    assert isinstance(denom, Float[Array, " n_outaged_branches n_outaged_branches"])

    # Denom is equivalent to PTDF^0_{O, O} in the paper
    # Nom is equivalent to PTDF^0_{M, O} in the paper
    nom = ptdf.at[:, from_node_outage].get(mode="fill", fill_value=0) - ptdf.at[:, to_node_outage].get(
        mode="fill", fill_value=0
    )
    nom = nom.at[multi_outages].set(-denom, mode="drop")
    assert isinstance(nom, Float[Array, " n_branches n_outaged_branches"])

    # Make sure invalid outage indices are padded properly
    invalid_outages = (multi_outages < 0) | (multi_outages >= ptdf.shape[0])
    nom = jnp.where(invalid_outages[None, :], 0.0, nom)
    denom = jnp.where(invalid_outages[:, None], jnp.identity(len(multi_outages)), denom)
    denom = jnp.where(invalid_outages[None, :], jnp.identity(len(multi_outages)), denom)
    multi_outages = jnp.where(invalid_outages, int_max(), multi_outages)

    # Use an optimized version for 2x2 systems (trafo3w) and 3x3 systems
    modf, success = solve_and_check_det(denom.T, nom.T)
    modf = modf.T

    assert isinstance(modf, Float[Array, " n_branches n_outaged_branches"])

    return (
        MODFMatrix(
            modf=modf,
            branch_indices=multi_outages,
        ),
        success,
    )


def apply_modf_matrices(
    modf_matrices: list[MODFMatrix],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    branches_monitored: Int[Array, " n_branches_monitored"],
) -> Float[Array, "  n_timesteps n_all_multi_outages n_branches_monitored"]:
    """Apply all MODF matrices to perform a contingency analysis for multi outages

    Parameters
    ----------
    modf_matrices : MODFMatrices
        The MODF matrices for all multi-outages
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The N-0 flows as computed in the contingency module
    branches_monitored : Int[Array, " n_branches_monitored"]
        The branches that are monitored (static argument)

    Returns
    -------
    Float[Array, " n_timesteps n_all_multi_outages n_branches_monitored"]
        The loading on the branches after the multi-outages, can be concatenated to the N-1 matrix
    """
    if len(modf_matrices) == 0:
        return jnp.zeros((n_0_flow.shape[0], 0, branches_monitored.shape[0]), dtype=float)

    flows = []
    for modf_matrix in modf_matrices:
        flows.append(jax.vmap(apply_modf_matrix, in_axes=(0, None, None))(modf_matrix, n_0_flow, branches_monitored))

    flows = jnp.concatenate(flows, axis=0)
    flows = jnp.transpose(flows, (1, 0, 2))
    return flows


def apply_modf_matrix(
    modf_matrix: MODFMatrix,
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    branches_monitored: Optional[Int[Array, " n_branches_monitored"]],
) -> Float[Array, " n_timesteps n_branches_monitored"]:
    """Apply the MODF matrix to compute the flow after multi-outages

    Parameters
    ----------
    modf_matrix : MODFMatrix
        The MODF matrix for this multi-outage
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The N-0 flows as computed in the contingency module
    branches_monitored : Optional[Int[Array, " n_branches_monitored"]]
        The branches that are monitored. If passed, the return value will have size
        n_branches_monitored, otherwise n_branches. (static argument)

    Returns
    -------
    Float[Array, " n_timesteps n_branches_monitored"]
        The loading on the branches after the multi-outages
    """
    assert len(modf_matrix.modf.shape) == 2
    assert modf_matrix.modf.shape == (
        n_0_flow.shape[1],
        modf_matrix.branch_indices.shape[0],
    )
    assert len(modf_matrix.branch_indices.shape) == 1

    res = n_0_flow + jnp.einsum(
        "io,to->ti",
        modf_matrix.modf,
        n_0_flow[:, modf_matrix.branch_indices],
    )
    res = res.at[:, modf_matrix.branch_indices].set(0.0, mode="drop")
    if branches_monitored is not None:
        res = res[:, branches_monitored]

    return res


def compute_multi_outage(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    n_0_flow: Float[Array, " n_timesteps n_branches"],
    multi_outages: Int[Array, " n_multi_outages"],
    branches_monitored: Optional[Int[Array, " n_branches_monitored"]],
) -> tuple[Float[Array, " n_timesteps n_branches"], Bool[Array, " "]]:
    """Compute the flow after a single multi-outage using the MODF formulation

    You can look up the theory in the paper:
    https://doi.org/10.1109/TPWRS.2009.2023273

    This formulation can not work with padded multi-outages and it's using a solve operation, which
    can be cheaper than compute_multi_outage.

    This computes an end-to-end multi outage instead of the build/apply approach and is intended
    rather as a reference than a performance optimized implementation.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after all busbar splits/branch outages that are part of actions.
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    n_0_flow : Float[Array, " n_timesteps n_branches"]
        The N-0 flows as computed in the contingency module
    multi_outages : Int[Array, " n_multi_outages"]
        The branches to be outaged
    branches_monitored : Optional[Int[Array, " n_branches_monitored"]]
        The branches that are monitored (static argument)

    Returns
    -------
    Float[Array, " n_timesteps n_branches"]
        The flow after the multi-outages
    Bool[Array, " "]
        Whether the computation was successful (i.e. the network did not split in any of the outages)
    """
    modf_matrix, success = build_modf_matrix(
        ptdf,
        from_node,
        to_node,
        multi_outages,
    )

    n_0_flow = apply_modf_matrix(modf_matrix, n_0_flow, branches_monitored)

    return n_0_flow, success


def update_ptdf_with_modf(
    modf: MODFMatrix,
    ptdf: Float[Array, " n_branches n_bus"],
) -> Float[Array, " n_branches n_bus"]:
    """Update the PTDF matrix with the MODF matrix

    Uses equation 7 from https://doi.org/10.1109/TPWRS.2009.2023273

    Parameters
    ----------
    modf : MODFMatrix
        The MODF matrix for this multi-outage
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix, usually after applying the BSDF formula as outages are applied after splits
        by convention.

    Returns
    -------
    Float[Array, " n_branches n_bus"]
        The updated PTDF matrix
    """
    delta = jnp.einsum("ij,jk->ik", modf.modf, ptdf[modf.branch_indices, :])
    assert delta.shape == ptdf.shape
    return ptdf + delta
