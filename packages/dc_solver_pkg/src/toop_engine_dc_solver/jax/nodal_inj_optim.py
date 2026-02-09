# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains nodal injection optimization routines.

Nodal injection optimization includes PST Optimization routines.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from toop_engine_dc_solver.jax.types import (
    DynamicInformation,
    NodalInjectionInformation,
    NodalInjOptimResults,
    NodalInjStartOptions,
    SolverConfig,
    TopologyResults,
)


def make_start_options(
    old_res: NodalInjOptimResults | None,
) -> NodalInjStartOptions | None:
    """Create start options for nodal injection optimization from previous results.

    Returns None if old_res is None, indicating no previous results to use as starting point.
    """
    if old_res is None:
        return None
    return NodalInjStartOptions(
        previous_results=old_res,
        precision_percent=jnp.array(1.0),  # TODO
    )


def apply_pst_taps(
    n_0: Float[Array, " batch_size n_timesteps n_branches"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
    topo_res: TopologyResults,
    nodal_inj_info: NodalInjectionInformation,
) -> Float[Array, " batch_size n_timesteps n_branches"]:
    """Apply PST taps from start options and update N-0 flows incrementally.

    This implementation takes the PST tap settings from the start options and applies
    only the delta to the N-0 flows using the PSDF columns in the PTDF matrix, rather
    than recomputing the entire flow from scratch.

    Parameters
    ----------
    n_0 : Float[Array, " batch_size n_timesteps n_branches"]
        The base case N-0 flows for each topology (with current PST settings)
    nodal_injections : Float[Array, " batch_size n_timesteps n_buses"]
        The nodal injections including current PST angles at the beginning
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"]
        The tap indices for the controllable PSTs from the start options, which indicate the new tap settings to apply.
    topo_res : TopologyResults
        The topology results containing PTDF matrix (with PSDF prepended)
    nodal_inj_info : NodalInjectionInformation
        Contains PST information

    Returns
    -------
    n_0_updated : Float[Array, " batch_size n_timesteps n_branches
        The updated N-0 flows after applying PST taps
    """
    # Convert tap indices to shift angles in degrees using pst_tap_values
    # pst_tap_values shape: (n_controllable_pst, max_n_tap_positions)
    n_controllable_pst = nodal_inj_info.controllable_pst_indices.shape[0]
    pst_indices = jnp.arange(n_controllable_pst)

    # Vectorized gather: for each batch and timestep, get tap values
    def get_tap_values_single(tap_idx_row: Int[Array, " n_controllable_pst"]) -> Float[Array, " n_controllable_pst"]:
        """Get tap values for a single batch/timestep combination."""
        return nodal_inj_info.pst_tap_values[pst_indices, tap_idx_row]

    # Apply vmap over batch dimension, then timestep dimension
    new_shift_angles = jax.vmap(jax.vmap(get_tap_values_single))(pst_tap_indices)
    # Shape: (batch_size, n_timesteps, n_controllable_pst)

    # Get current PST angles from nodal_injections using controllable_pst_indices
    # controllable_pst_indices maps PST positions to node array indices
    current_shift_angles = nodal_injections[:, :, nodal_inj_info.controllable_pst_indices]

    # Compute the delta in shift angles
    delta_shift_angles = new_shift_angles - current_shift_angles
    # Shape: (batch_size, n_timesteps, n_controllable_pst)

    # Extract PSDF columns from PTDF using controllable_pst_indices
    # PTDF shape: (batch_size, n_branches, n_buses)
    # PSDF columns are at node indices specified by controllable_pst_indices
    psdf_columns = topo_res.ptdf[:, :, nodal_inj_info.controllable_pst_indices]
    # Shape: (batch_size, n_branches, n_controllable_pst)

    # Compute the flow delta using PSDF: delta_flows = PSDF @ delta_angles
    # Use einsum for batched matrix multiplication:
    # (batch, branches, pst) @ (batch, timesteps, pst) -> (batch, timesteps, branches)
    delta_flows = jnp.einsum("bij,btj->bti", psdf_columns, delta_shift_angles)

    # Apply the delta to the existing N-0 flows
    # Note: PSDF is computed with a negative sign in preprocessing (psdf.py:56),
    # so we subtract the delta to match PowerSybl convention
    n_0_updated = n_0 - delta_flows

    return n_0_updated


def nodal_inj_optimization(
    n_0: Float[Array, " batch_size n_timesteps n_branches"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    topo_res: TopologyResults,
    start_options: NodalInjStartOptions,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,  # noqa: ARG001
) -> tuple[
    Float[Array, " batch_size n_timesteps n_branches"],
    Float[Array, " batch_size n_timesteps n_outages n_branches_monitored"],
    NodalInjOptimResults,
]:
    """Optimize nodal injections in the loop

    Currently only applies PST taps, in the future this will also perform an HVDC optimization and
    potentially an easy redispatch.

    Parameters
    ----------
    n_0 : Float[Array, " batch_size n_timesteps n_branches"]
        The base case N-0 flows for each topology (with current PST settings)
    nodal_injections : Float[Array, " batch_size n_timesteps n_buses"]
        The nodal injections including current PST angles at the beginning
    topo_res : TopologyResults
        The topology results containing PTDF matrix (with PSDF prepended)
    start_options : NodalInjStartOptions
        Contains previous PST tap results to apply
    dynamic_information : DynamicInformation
        Contains PST information and grid data
    solver_config : SolverConfig
        Solver configuration

    Returns
    -------
    n_0_updated : Float[Array, " batch_size n_timesteps n_branches"]
        The updated N-0 flows after applying PST taps
    n_1_dummy : Float[Array, " batch_size n_timesteps n_outages n_branches_monitored"]
        Dummy N-1 matrix (to be properly implemented later)
    results : NodalInjOptimResults
        The PST taps that were applied
    """
    nodal_inj_info = dynamic_information.nodal_injection_information
    assert nodal_inj_info is not None, "Nodal injection information must be provided for PST optimization"

    # Get PST tap indices from start options (shape: batch_size x n_timesteps x n_controllable_pst)
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"] = (
        start_options.previous_results.pst_taps.astype(jnp.int32)
    )

    n_0_updated = apply_pst_taps(
        n_0=n_0,
        nodal_injections=nodal_injections,
        pst_tap_indices=pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=nodal_inj_info,
    )

    # Create dummy N-1 matrix (will be properly implemented later)
    # Shape should be (batch_size, n_timesteps, n_outages, n_branches_monitored)
    n_branches_monitored = dynamic_information.branches_monitored.shape[0]
    n_outages = dynamic_information.n_nminus1_cases
    batch_size = n_0.shape[0]
    n_timesteps = n_0.shape[1]
    n_1_dummy = jnp.zeros((batch_size, n_timesteps, n_outages, n_branches_monitored))

    # Return the applied taps
    results = NodalInjOptimResults(pst_taps=pst_tap_indices)

    return n_0_updated, n_1_dummy, results
