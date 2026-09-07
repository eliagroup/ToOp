# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
"""PTDF updates for simultaneous branch-parameter changes.

This module handles low-rank PTDF updates when one or more branch parameters change at once,
for example when multiple phase-shifting transformers move to new taps in the same batch.

The core idea is the same as in :mod:`toop_engine_dc_solver.jax.multi_outages`: isolate the
small coupled subproblem for the changed elements, solve that reduced system, and then apply
the correction to the full PTDF matrix.
"""

from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.unrolled_linalg import solve_and_check_det


def update_ptdf_with_branch_parameter_change(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    base_susceptance: Float[Array, " n_branches"],
    updated_susceptance: Float[Array, " n_changed_branches"],
    changed_branch_indices: Int[Array, " n_changed_branches"],
) -> tuple[Float[Array, " n_branches n_bus"], Bool[Array, " "]]:
    """Update a PTDF after simultaneous branch-parameter changes.

    This uses the same low-rank solve pattern as the multi-outage PTDF update in
    :mod:`toop_engine_dc_solver.jax.multi_outages`. The difference is that branch-parameter
    changes modify the branch reactance/susceptance rather than removing a branch entirely.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix before the parameter changes.
    from_node : Int[Array, " n_branches"]
        The from-node index for each branch.
    to_node : Int[Array, " n_branches"]
        The to-node index for each branch.
    base_susceptance : Float[Array, " n_branches"]
        The branch susceptance values before the change.
    updated_susceptance : Float[Array, " n_changed_branches"]
        The new susceptance values for the changed branches.
    changed_branch_indices : Int[Array, " n_changed_branches"]
        The indices of the branches whose susceptance changes.

    Returns
    -------
    tuple[Float[Array, " n_branches n_bus"], Bool[Array, " "]]
        The updated PTDF and a success flag.
    """
    n_changed = changed_branch_indices.shape[0]
    if n_changed == 0:
        return ptdf, jnp.array(True)

    base_changed_susceptance = base_susceptance[changed_branch_indices]
    delta_susceptance = updated_susceptance - base_changed_susceptance
    alpha = delta_susceptance / base_changed_susceptance

    changed_from = from_node[changed_branch_indices]
    changed_to = to_node[changed_branch_indices]
    h_columns = ptdf[:, changed_from] - ptdf[:, changed_to]
    h_oo = h_columns[changed_branch_indices, :]

    d_alpha = jnp.diag(alpha)
    coupling_matrix = jnp.eye(n_changed, dtype=ptdf.dtype) + d_alpha @ h_oo
    rhs = d_alpha @ ptdf[changed_branch_indices, :]
    correction, success = solve_and_check_det(coupling_matrix, rhs)

    branch_parameter_influence = -h_columns
    branch_parameter_influence = branch_parameter_influence.at[changed_branch_indices, jnp.arange(n_changed)].add(1.0)
    updated_ptdf = ptdf + branch_parameter_influence @ correction
    return updated_ptdf, success
