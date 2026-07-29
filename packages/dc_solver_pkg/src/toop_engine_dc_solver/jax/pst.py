# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for applying PST state in the JAX solver.

This module keeps PST-specific logic out of the nodal injection optimization layer.
The topology-sensitive part stays in ``compute_bsdf_lodf_static_flows`` via branch-parameter
updates, while the helpers here prepare requested tap states and write the corresponding PST
angles into the nodal injection tensor.
"""

import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Float, Int
from toop_engine_dc_solver.jax.types import (
    NodalInjectionInformation,
    NodalInjOptimResults,
    NodalInjStartOptions,
    TopologyResults,
)


def prepare_pst_tap_state(
    start_options: Optional[NodalInjStartOptions],
    nodal_inj_info: Optional[NodalInjectionInformation],
) -> tuple[
    Optional[Int[Array, " batch_size n_timesteps n_controllable_pst"]],
    Optional[Float[Array, " batch_size n_controllable_pst"]],
    Optional[NodalInjOptimResults],
]:
    """Normalize requested PST taps and derive topology-level susceptance updates.

    The PTDF update path is topology-scoped, so it needs one tap state per topology batch item.
    The nodal injection path remains timestep-scoped and therefore keeps the full tap tensor.
    """
    if start_options is None or nodal_inj_info is None:
        return None, None, None

    pst_tap_indices = start_options.previous_results.pst_tap_idx
    if pst_tap_indices.ndim == 2:
        pst_tap_indices = pst_tap_indices[None, :, :]
    if pst_tap_indices.shape[1] == 0:
        raise ValueError("PST tap index batches must include at least one timestep.")

    starting_tap_idx = nodal_inj_info.starting_tap_idx
    requested_diff_mask = jnp.any(pst_tap_indices != starting_tap_idx[None, None, :], axis=-1)
    topology_level_step = jnp.argmax(requested_diff_mask, axis=1)
    topology_level_tap_idx = pst_tap_indices[jnp.arange(pst_tap_indices.shape[0]), topology_level_step]
    pst_tap_susceptance_values = nodal_inj_info.pst_tap_susceptance_values[
        jnp.arange(topology_level_tap_idx.shape[1]), topology_level_tap_idx
    ]

    return (
        pst_tap_indices,
        pst_tap_susceptance_values,
        NodalInjOptimResults(pst_tap_idx=pst_tap_indices),
    )


def _gather_pst_table(
    pst_table: Float[Array, " n_controllable_pst max_n_tap_positions"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
) -> Float[Array, " batch_size n_timesteps n_controllable_pst"]:
    """Gather a per-tap PST table for a batch of tap indices."""
    n_controllable_pst = pst_table.shape[0]
    pst_idx = jnp.arange(n_controllable_pst)[None, None, :]
    return pst_table[pst_idx, pst_tap_indices]


def write_pst_taps_to_nodal_injections(
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
    nodal_inj_info: NodalInjectionInformation,
) -> Float[Array, " batch_size n_timesteps n_buses"]:
    """Write requested PST tap angles into the nodal injection tensor."""
    new_shift_angles = _gather_pst_table(nodal_inj_info.pst_tap_values, pst_tap_indices)
    return nodal_injections.at[:, :, nodal_inj_info.controllable_pst_indices].set(new_shift_angles)


def update_n0_for_pst_taps(
    n_0: Float[Array, " batch_size n_timesteps n_branches"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    updated_nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
    topo_res: TopologyResults,
    nodal_inj_info: NodalInjectionInformation,
) -> Float[Array, " batch_size n_timesteps n_branches"]:
    """Update N-0 flows after PST taps have been applied to nodal injections."""
    del n_0, nodal_injections, pst_tap_indices, nodal_inj_info
    return jnp.einsum("bij,btj->bti", topo_res.ptdf, updated_nodal_injections)
