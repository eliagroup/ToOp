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
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix
from toop_engine_dc_solver.jax.multi_outages import build_modf_matrices
from toop_engine_dc_solver.jax.types import (
    DynamicInformation,
    NodalInjectionInformation,
    NodalInjOptimResults,
    NodalInjStartOptions,
    SolverConfig,
    TopologyResults,
)
from toop_engine_dc_solver.jax.unrolled_linalg import solve_and_check_det


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
        The tap indices for the controllable PSTs from the start options,
        which indicate the new tap settings to apply.
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
    # pst_tap_indices shape: (batch_size, n_timesteps, n_controllable_pst)
    n_controllable_pst = nodal_inj_info.controllable_pst_indices.shape[0]

    # Use advanced indexing to gather tap values
    # Create index array for first dimension (PST index)
    pst_idx = jnp.arange(n_controllable_pst)[None, None, :]  # Shape: (1, 1, n_controllable_pst)
    # Broadcasting allows: result[b, t, i] = pst_tap_values[i, pst_tap_indices[b, t, i]]
    new_shift_angles = nodal_inj_info.pst_tap_values[pst_idx, pst_tap_indices]
    # Shape: (batch_size, n_timesteps, n_controllable_pst)

    # Get current PST angles from nodal_injections using controllable_pst_indices
    # controllable_pst_indices maps PST positions to node array indices
    current_shift_angles = nodal_injections[:, :, nodal_inj_info.controllable_pst_indices]

    # Compute the delta in shift angles # TODO: Check sign
    delta_shift_angles = -new_shift_angles + current_shift_angles
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


def _gather_pst_table(
    pst_table: Float[Array, " n_controllable_pst max_n_tap_positions"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
) -> Float[Array, " batch_size n_timesteps n_controllable_pst"]:
    """Gather a per-tap PST table for a batch of tap indices."""
    n_controllable_pst = pst_table.shape[0]
    pst_idx = jnp.arange(n_controllable_pst)[None, None, :]
    return pst_table[pst_idx, pst_tap_indices]


def _update_nodal_injections_with_pst_taps(
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    pst_tap_indices: Int[Array, " batch_size n_timesteps n_controllable_pst"],
    nodal_inj_info: NodalInjectionInformation,
) -> tuple[
    Float[Array, " batch_size n_timesteps n_buses"],
    Float[Array, " batch_size n_timesteps n_controllable_pst"],
]:
    """Write requested PST angles into the nodal injection vector."""
    new_shift_angles = _gather_pst_table(nodal_inj_info.pst_tap_values, pst_tap_indices)
    updated_nodal_injections = nodal_injections.at[:, :, nodal_inj_info.controllable_pst_indices].set(new_shift_angles)
    return updated_nodal_injections, new_shift_angles


def _apply_branch_parameter_update_in_ptdf_world_single(
    full_ptdf: Float[Array, " n_branches n_buses"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    base_susceptance: Float[Array, " n_branches"],
    updated_controllable_pst_susceptance: Float[Array, " n_controllable_pst"],
    nodal_inj_info: NodalInjectionInformation,
) -> tuple[Float[Array, " n_branches n_buses"], Bool[Array, " "]]:
    """Apply a generalized branch-parameter update in PTDF space.

    This treats parameter updates like a MODF-style correction with a small coupled solve over only
    the changed branches. Branch outages are the special case with alpha=-1.
    """
    changed_branch_indices = nodal_inj_info.controllable_pst_branch_indices
    n_changed = changed_branch_indices.shape[0]
    if n_changed == 0:
        return full_ptdf, jnp.array(True)

    base_changed_susceptance = base_susceptance[changed_branch_indices]
    delta_susceptance = updated_controllable_pst_susceptance - base_changed_susceptance
    alpha = delta_susceptance / base_changed_susceptance

    changed_from = from_node[changed_branch_indices]
    changed_to = to_node[changed_branch_indices]
    h_columns = full_ptdf[:, changed_from] - full_ptdf[:, changed_to]
    h_oo = h_columns[changed_branch_indices, :]

    d_alpha = jnp.diag(alpha)
    coupling_matrix = jnp.eye(n_changed, dtype=full_ptdf.dtype) + d_alpha @ h_oo
    rhs = d_alpha @ full_ptdf[changed_branch_indices, :]
    correction, success = solve_and_check_det(coupling_matrix, rhs)

    branch_parameter_influence = -h_columns
    branch_parameter_influence = branch_parameter_influence.at[changed_branch_indices, jnp.arange(n_changed)].add(1.0)
    updated_ptdf = full_ptdf + branch_parameter_influence @ correction
    return updated_ptdf, success


def _recompute_topology_results_for_nonlinear_psts(
    topo_res: TopologyResults,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    pst_tap_susceptance_values: Float[Array, " batch_size n_timesteps n_controllable_pst"],
    nodal_inj_info: NodalInjectionInformation,
) -> TopologyResults:
    """Recompute PTDF and contingency sensitivities after nonlinear PST updates."""
    updated_ptdf, ptdf_update_success = jax.vmap(
        _apply_branch_parameter_update_in_ptdf_world_single,
        in_axes=(0, 0, 0, None, 0, None),
    )(
        topo_res.ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.susceptance,
        pst_tap_susceptance_values[:, 0, :],
        nodal_inj_info,
    )

    lodf, lodf_success = jax.vmap(calc_lodf_matrix, in_axes=(None, 0, 0, 0, None))(
        dynamic_information.branches_to_fail,
        updated_ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.branches_monitored,
    )
    if topo_res.failure_cases_to_zero is not None:
        single_outage_cases_to_zero = topo_res.failure_cases_to_zero[:, : lodf_success.shape[1]]
        lodf_success = jnp.where(single_outage_cases_to_zero, True, lodf_success)
    lodf_success = lodf_success & ptdf_update_success[:, None]

    outage_modf, outage_modf_success = jax.vmap(
        build_modf_matrices,
        in_axes=(0, 0, 0, None),
    )(
        updated_ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.multi_outage_branches,
    )
    outage_modf_success = outage_modf_success & ptdf_update_success[:, None]

    base_success = topo_res.success & ptdf_update_success
    injection_outage_success = jnp.broadcast_to(
        base_success[:, None],
        (topo_res.success.shape[0], dynamic_information.n_inj_failures),
    )
    bb_outage_success = jnp.broadcast_to(
        base_success[:, None],
        (
            topo_res.success.shape[0],
            dynamic_information.n_bb_outages
            if solver_config.enable_bb_outages and solver_config.bb_outage_as_nminus1
            else 0,
        ),
    )
    contingency_success = jnp.concatenate(
        [
            lodf_success,
            outage_modf_success,
            injection_outage_success,
            bb_outage_success,
        ],
        axis=1,
    )

    return replace(
        topo_res,
        ptdf=updated_ptdf,
        lodf=lodf,
        outage_modf=outage_modf,
        contingency_success=contingency_success,
        success=base_success & jnp.all(contingency_success, axis=1),
    )


def nodal_inj_optimization(
    n_0: Float[Array, " batch_size n_timesteps n_branches"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    topo_res: TopologyResults,
    start_options: NodalInjStartOptions,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> tuple[
    Float[Array, " batch_size n_timesteps n_branches"],
    Float[Array, " batch_size n_timesteps n_outages n_branches_monitored"],
    NodalInjOptimResults,
    TopologyResults,
    Float[Array, " batch_size n_timesteps n_buses"],
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
        Contains the slack bus and contingency toggles required to rebuild sensitivity matrices.

    Returns
    -------
    n_0_updated : Float[Array, " batch_size n_timesteps n_branches"]
        The updated N-0 flows after applying PST taps
    n_1_dummy : Float[Array, " batch_size n_timesteps n_outages n_branches_monitored"]
        Dummy N-1 matrix (to be properly implemented later)
    results : NodalInjOptimResults
        The PST taps that were applied
    updated_topology_results : TopologyResults
        The topology result updated for nonlinear PST susceptance changes.
    updated_nodal_injections : Float[Array, " batch_size n_timesteps n_buses"]
        Nodal injections with the requested PST angles written into the PST pseudo-nodes.
    """
    nodal_inj_info = dynamic_information.nodal_injection_information
    assert nodal_inj_info is not None, "Nodal injection information must be provided for PST optimization"

    # Get PST tap indices from start options (shape: batch_size x n_timesteps x n_controllable_pst)
    pst_tap_indices = start_options.previous_results.pst_tap_idx
    if pst_tap_indices.ndim == 2:
        pst_tap_indices = pst_tap_indices[None, :, :]

    updated_nodal_injections, _new_shift_angles = _update_nodal_injections_with_pst_taps(
        nodal_injections=nodal_injections,
        pst_tap_indices=pst_tap_indices,
        nodal_inj_info=nodal_inj_info,
    )

    can_recompute_contingencies = (
        topo_res.contingency_success is not None and topo_res.lodf is not None and topo_res.outage_modf is not None
    )

    if not can_recompute_contingencies:
        if bool(jnp.any(~nodal_inj_info.phase_shift_linearity)):
            pst_tap_susceptance_values = _gather_pst_table(
                nodal_inj_info.pst_tap_susceptance_values,
                pst_tap_indices,
            )
            updated_ptdf, ptdf_update_success = jax.vmap(
                _apply_branch_parameter_update_in_ptdf_world_single,
                in_axes=(0, 0, 0, None, 0, None),
            )(
                topo_res.ptdf,
                topo_res.from_node,
                topo_res.to_node,
                dynamic_information.susceptance,
                pst_tap_susceptance_values[:, 0, :],
                nodal_inj_info,
            )
            updated_topo_res = replace(topo_res, ptdf=updated_ptdf, success=topo_res.success & ptdf_update_success)
            n_0_updated = jnp.einsum("bij,btj->bti", updated_ptdf, updated_nodal_injections)
        else:
            updated_topo_res = topo_res
            n_0_updated = apply_pst_taps(
                n_0=n_0,
                nodal_injections=nodal_injections,
                pst_tap_indices=pst_tap_indices,
                topo_res=topo_res,
                nodal_inj_info=nodal_inj_info,
            )
    else:

        def _nonlinear_branch(
            _args: None,
        ) -> tuple[
            Float[Array, " batch_size n_timesteps n_branches"],
            TopologyResults,
        ]:
            pst_tap_susceptance_values = _gather_pst_table(
                nodal_inj_info.pst_tap_susceptance_values,
                pst_tap_indices,
            )
            updated_topo_res = _recompute_topology_results_for_nonlinear_psts(
                topo_res=topo_res,
                dynamic_information=dynamic_information,
                solver_config=solver_config,
                pst_tap_susceptance_values=pst_tap_susceptance_values,
                nodal_inj_info=nodal_inj_info,
            )
            n_0_updated = jnp.einsum("bij,btj->bti", updated_topo_res.ptdf, updated_nodal_injections)
            return n_0_updated, updated_topo_res

        def _linear_branch(
            _args: None,
        ) -> tuple[
            Float[Array, " batch_size n_timesteps n_branches"],
            TopologyResults,
        ]:
            n_0_updated = apply_pst_taps(
                n_0=n_0,
                nodal_injections=nodal_injections,
                pst_tap_indices=pst_tap_indices,
                topo_res=topo_res,
                nodal_inj_info=nodal_inj_info,
            )
            return n_0_updated, topo_res

        n_0_updated, updated_topo_res = jax.lax.cond(
            jnp.any(~nodal_inj_info.phase_shift_linearity),
            _nonlinear_branch,
            _linear_branch,
            operand=None,
        )

    # Create dummy N-1 matrix (will be properly implemented later)
    # Shape should be (batch_size, n_timesteps, n_outages, n_branches_monitored)
    n_branches_monitored = dynamic_information.branches_monitored.shape[0]
    n_outages = dynamic_information.n_nminus1_cases
    batch_size = n_0.shape[0]
    n_timesteps = n_0.shape[1]
    n_1_dummy = jnp.zeros((batch_size, n_timesteps, n_outages, n_branches_monitored))

    # Return the applied taps
    results = NodalInjOptimResults(pst_tap_idx=pst_tap_indices)

    return n_0_updated, n_1_dummy, results, updated_topo_res, updated_nodal_injections
