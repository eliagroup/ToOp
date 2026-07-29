# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
import numpy as np
from toop_engine_dc_solver.jax.branch_parameter_changes import update_ptdf_with_branch_parameter_change
from toop_engine_dc_solver.jax.lodf import calc_lodf


def test_update_ptdf_with_branch_parameter_change_no_changed_branches() -> None:
    """No-op branch parameter updates should return the original PTDF and succeed."""
    ptdf = jnp.array([[0.2, -0.1], [0.3, 0.4]], dtype=float)
    from_node = jnp.array([0, 1], dtype=int)
    to_node = jnp.array([1, 0], dtype=int)
    base_susceptance = jnp.array([0.5, 0.25], dtype=float)
    updated_susceptance = jnp.array([], dtype=float)
    changed_branch_indices = jnp.array([], dtype=int)

    updated_ptdf, success = update_ptdf_with_branch_parameter_change(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        base_susceptance=base_susceptance,
        updated_susceptance=updated_susceptance,
        changed_branch_indices=changed_branch_indices,
    )

    assert jnp.array_equal(updated_ptdf, ptdf)
    assert bool(success)


def test_update_ptdf_with_branch_parameter_change_matches_line_outage_when_susceptance_goes_to_zero() -> None:
    """Zeroing a branch susceptance should match the equivalent single-outage PTDF update."""
    ptdf = jnp.array(
        [
            [0.5, -0.2],
            [0.2, 0.1],
            [-0.1, 0.3],
        ],
        dtype=float,
    )
    from_node = jnp.array([0, 0, 1], dtype=int)
    to_node = jnp.array([1, 1, 0], dtype=int)
    base_susceptance = jnp.array([1.0, 1.0, 1.0], dtype=float)
    changed_branch_index = jnp.array([0], dtype=int)

    updated_ptdf, success = update_ptdf_with_branch_parameter_change(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        base_susceptance=base_susceptance,
        updated_susceptance=jnp.array([0.0], dtype=float),
        changed_branch_indices=changed_branch_index,
    )

    lodf, lodf_success = calc_lodf(
        branch_to_outage=changed_branch_index[0],
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        branches_monitored=None,
    )
    expected_ptdf = ptdf + lodf[:, None] * ptdf[changed_branch_index[0]][None, :]

    assert bool(success)
    assert bool(lodf_success)
    assert np.allclose(updated_ptdf, expected_ptdf)
    assert np.allclose(updated_ptdf[changed_branch_index[0]], 0.0)
