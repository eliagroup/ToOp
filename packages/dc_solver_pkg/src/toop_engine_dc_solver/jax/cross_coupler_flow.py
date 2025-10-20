"""Compute cross-coupler related flows using BSDF formulations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.bsdf import get_bus_data


def _gather_bus_a_injection(
    substation_id: Int[Array, " "],
    injection_config_row: Bool[Array, " max_inj_per_sub"],
    relevant_injections: Float[Array, " n_timesteps n_sub_relevant max_inj_per_sub"],
) -> Float[Array, " n_timesteps"]:
    """Build the per-timestep injection at busbar A for a single split.

    Parameters
    ----------
    substation_id : Int[Array, " "]
        The substation index for which to gather the injection. Returns zero if out of range.
    injection_config_row : Bool[Array, " max_inj_per_sub"]
        Injection configuration row for the split.
        "True" indicates Bus A is connected to that injection point.
        "False" indicates Bus B is connected to that injection point.
    relevant_injections : Float[Array, " n_timesteps n_sub_relevant max_inj_per_sub"]
        The relevant power injections for all substations and all timesteps.

    Returns
    -------
    bus_a_injection: Float[Array, " n_timesteps"]
        The injection at busbar A for all timesteps.
    """
    injection_block = relevant_injections.at[:, substation_id].get(mode="fill", fill_value=0.0)  # (T, max_inj)
    mask = (~injection_config_row).astype(injection_block.dtype)
    # Weighted sum across injection dimension
    bus_a_injection = jnp.einsum("ti,i->t", injection_block, mask)
    return bus_a_injection


def _compute_bus_a_imbalance(
    flows: Float[Array, " n_timesteps n_branches"],
    branch_to_a: Int[Array, " n_to"],
    branch_from_a: Int[Array, " n_from"],
    bus_a_injection: Float[Array, " n_timesteps"],
    substation_id: Int[Array, " "],
    substation_branch_status: Int[Array, " n_sub_rel max_branch_per_sub"],
    substation_configuration: Bool[Array, " max_branch_per_sub"],
) -> Float[Array, " n_timesteps"]:
    """Compute the net imbalance at busbar A (cross-coupler flow).

    The imbalance equals:
        sum(flows into A) - sum(flows out of A) + injection at A

    Parameters
    ----------
    flows : Float[Array, " n_timesteps n_branches"]
        Current branch flows (potentially already updated by previous splits).
    branch_to_a : Int[Array, " n_to"]
        Indices of branches whose flow direction is towards bus A.
    branch_from_a : Int[Array, " n_from"]
        Indices of branches whose flow direction is away from bus A.
    bus_a_injection : Float[Array, " n_timesteps"]
        Net power injection assigned to bus A.
    substation_id : Int[Array, " "]
        Substation index (used for bounds checking).
    substation_branch_status : Int[Array, " n_sub_rel max_branch_per_sub"]
        Branch status table for all relevant substations.
    substation_configuration : Bool[Array, " max_branch_per_sub"]
        Topology mask (True = connected to bus A, False = bus B).

    Returns
    -------
    Float[Array, " n_timesteps"]
        Cross-coupler flow (imbalance at bus A) for each timestep.
    """
    # Gather flows (fill zeros if padding indices)
    to_bus_a = flows.at[:, branch_to_a].get(mode="fill", fill_value=0.0)
    from_bus_a = flows.at[:, branch_from_a].get(mode="fill", fill_value=0.0)

    imbalance_a = jnp.sum(to_bus_a, axis=1) - jnp.sum(from_bus_a, axis=1) + bus_a_injection

    # Zero-out invalid / disabled situations
    invalid = (substation_id < 0) | (substation_id >= substation_branch_status.shape[0]) | jnp.all(~substation_configuration)
    return jnp.where(invalid, jnp.zeros_like(imbalance_a), imbalance_a)


def compute_cross_coupler_flow_single(
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    substation_configuration: Bool[Array, " max_branch_per_sub"],
    substation_id: Int[Array, " "],
    bus_a_injection: Float[Array, " n_timesteps"],
    substation_branch_status: Int[Array, " n_sub_rel max_branch_per_sub"],
    from_status: Bool[Array, " n_sub_rel max_branch_per_sub"],
) -> Float[Array, " n_timesteps"]:
    """Compute cross-coupler flow between two busbars A and B.

    Using Kirchhoff's Current Law, the cross-coupler flow equals the imbalance at bus A.

    Parameters
    ----------
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        The N-0 flows in all branches for all timesteps.
    substation_configuration : Bool[Array, " max_branch_per_sub"]
        The topology vector for the substation that we want to split.
        "True" indicates that the branch is connected to busbar A.
        "False" indicates that the branch is connected to busbar B.
    substation_id : Int[Array, " "]
        The substation index for which to compute the cross-coupler flow. Returns zero if out of range.
    bus_a_injection : Float[Array, " n_timesteps"]
        The injection at busbar A for all timesteps.
    substation_branch_status : Int[Array, " n_sub_rel max_branch_per_sub"]
        The status of branches for all relevant substations. Contains all from and to branch indices.
        Padded to the maximum number of branches allowed, `max_branch_per_sub`.
    from_status : Bool[Array, " n_sub_rel max_branch_per_sub"]
        The "from" status of branches for all relevant substations.
        "True" indicates that the branch is a "from" branch.
        "False" indicates that the branch is a "to" branch.

    Returns
    -------
    cross_coupler_flow : Float[Array, " n_timesteps"]
        The cross-coupler flow for all timesteps.
    """
    status_substation = substation_branch_status.at[substation_id].get(mode="fill", fill_value=0)
    from_status_bool_substation = from_status.at[substation_id].get(mode="fill", fill_value=False)

    # get BSDF data
    branch_to_a, branch_from_a = get_bus_data(
        substation_topology=substation_configuration,
        tot_stat=status_substation,
        from_stat_bool=from_status_bool_substation,
        return_bus_b=False,
        n_branches=n_0_flows.shape[1],
    )

    imbalance_a = _compute_bus_a_imbalance(
        flows=n_0_flows,
        branch_to_a=branch_to_a,
        branch_from_a=branch_from_a,
        bus_a_injection=bus_a_injection,
        substation_id=substation_id,
        substation_branch_status=substation_branch_status,
        substation_configuration=substation_configuration,
    )
    # This equals the cross-coupler flow by Kirchhoff
    return imbalance_a


def _apply_bsdf_update(
    flows: Float[Array, " n_timesteps n_branches"],
    bsdf_row: Float[Array, " n_branches"],
    cross_flow: Float[Array, " n_timesteps"],
) -> Float[Array, " n_timesteps n_branches"]:
    """Apply the BSDF update to the flows.

    Parameters
    ----------
    flows : Float[Array, " n_timesteps n_branches"]
        The current flows in all branches for all timesteps.
    bsdf_row : Float[Array, " n_branches"]
        The BSDF row for the current split.
    cross_flow : Float[Array, " n_timesteps"]
        The cross-coupler flow for all timesteps.

    Returns
    -------
    Float[Array, " n_timesteps n_branches"]
        The updated flows after applying the BSDF update.
    """
    return flows + cross_flow[:, None] * bsdf_row[None, :]


def get_unsplit_flows(
    ptdf: Float[Array, "n_branches n_bus"],
    nodal_injections: Float[Array, "n_timesteps n_bus"],
    ac_dc_mismatch: Float[Array, "n_timesteps n_branch"],
    ac_dc_interpolation: Float[Array, ""],
) -> Float[Array, "n_timesteps n_branches"]:
    """Compute the N-0 flows for the unsplit case.

    Parameters
    ----------
    ptdf : Float[Array, "n_branches n_bus"]
        The unsplit PTDF matrix
    nodal_injections : Float[Array, "n_timesteps n_bus"]
        The nodal injections
    ac_dc_mismatch : Float[Array, "n_timesteps n_branch"]
        The AC-DC mismatch, to correct the DC results towards AC results
    ac_dc_interpolation : Float[Array, ""]
        The AC-DC interpolation factor, how much to correct the DC results towards AC results

    Returns
    -------
    Float[Array, "n_timesteps n_branches"]
        The N-0 flows for the unsplit case
    """
    return jnp.einsum("ij,tj -> ti", ptdf, nodal_injections) + ac_dc_mismatch * ac_dc_interpolation


def compute_cross_coupler_flows(
    bsdf: Float[Array, " n_splits n_branches"],
    topologies: Bool[Array, " n_splits max_branch_per_sub"],
    substation_ids: Int[Array, " n_splits"],
    injections: Bool[Array, " n_splits max_inj_per_sub"],
    relevant_injections: Float[Array, " n_timesteps n_sub_relevant max_inj_per_sub"],
    n_0_flows: Float[Array, " n_timesteps n_branches"],
    substation_branch_status: Int[Array, " n_sub_rel max_branch_per_sub"],
    from_stat_bool: Bool[Array, " n_sub_rel max_branch_per_sub"],
) -> tuple[
    Float[Array, " n_timesteps n_branches"],
    Float[Array, " n_splits n_timesteps"],
]:
    """Compute for n splits the cross-coupler flows.

    Sequentially apply each split (defined by its BSDF vector and topology) and record
    the pre-update cross-coupler flow.

    Parameters
    ----------
    bsdf : Float[Array, " n_splits n_branches"]
        The BSDF matrix for all splits.
    topologies : Bool[Array, " n_splits max_branch_per_sub"]
        The topology matrix for all splits.
        "True" indicates that the branch is connected to busbar A.
        "False" indicates that the branch is connected to busbar B.
    substation_ids : Int[Array, " n_splits"]
        The substation indices for which to compute the cross-coupler flows.
        Returns zero if out of range.
    injections : Bool[Array, " n_splits max_inj_per_sub"]
        The injection configuration for each split.
        "True" indicates Bus A is connected to that injection point.
        "False" indicates Bus B is connected to that injection point.
    relevant_injections : Float[Array, " n_timesteps n_sub_relevant max_inj_per_sub"]
        The relevant power injections for all substations and all timesteps.
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        The N-0 flows in all branches for all timesteps (before any split is applied
        this is the unsplit case).
    substation_branch_status : Int[Array, " n_sub_rel max_branch_per_sub"]
        The status of branches for all relevant substations. Contains all from and to branch indices.
        Padded to the maximum number of branches allowed, `max_branch_per_sub`.
    from_stat_bool : Bool[Array, " n_sub_rel max_branch_per_sub"]
        The "from" status of branches for all relevant substations.
        "True" indicates that the branch is a "from" branch.
        "False" indicates that the branch is a "to" branch.
        Padded to the maximum number of branches allowed, `max_branch_per_sub`.

    Returns
    -------
    updated_flows : Float[Array, " n_timesteps n_branches"]
        Flows after the final split has been applied.
    cross_coupler_flows : Float[Array, " n_splits n_timesteps"]
        Cross-coupler flow for each split (before its flow update).
    """

    def body_fun(
        n_0_flows: Float[Array, " n_timesteps n_branches"],
        scan_inputs: tuple[
            Float[Array, " n_branches"], Bool[Array, " max_branch_per_sub"], Int[Array, " "], Bool[Array, " max_inj_per_sub"]
        ],
    ) -> tuple[
        Float[Array, " n_timesteps n_branches"],
        Float[Array, " n_timesteps"],
    ]:
        """Body function for jax.lax.scan over splits.

        Parameters
        ----------
        n_0_flows : Float[Array, " n_timesteps n_branches"]
            The current N-0 flows.
        scan_inputs : tuple
            The inputs for the current split, containing:
            - bsdf_row : Float[Array, " n_branches"]
                The BSDF row for the current split.
            - topo_row : Bool[Array, " n_branches"]
                The topology row for the current split.
            - sub_id : Int[Array, " 1"]
                The substation ID for the current split.
            - inj_cfg : Bool[Array, " n_branches"]
                The injection configuration for the current split.

        Returns
        -------
        new_flows : Float[Array, " n_timesteps n_branches"]
            The updated flows after applying the BSDF update for the current split.
        cross_coupler_flow : Float[Array, " n_timesteps"]
            The cross-coupler flow for the current split (before applying the update).
        """
        bsdf_row, topo_row, sub_id, inj_cfg = scan_inputs

        # Compute bus A injection
        bus_a_injection = _gather_bus_a_injection(
            substation_id=sub_id,
            injection_config_row=inj_cfg,
            relevant_injections=relevant_injections,
        )

        # Cross-coupler flow for this split (before updating flows)
        cross_coupler_flow = compute_cross_coupler_flow_single(
            n_0_flows=n_0_flows,
            substation_configuration=topo_row,
            substation_id=sub_id,
            bus_a_injection=bus_a_injection,
            substation_branch_status=substation_branch_status,
            from_status=from_stat_bool,
        )

        # Update flows by BSDF rank-1 adjustment
        new_flows = _apply_bsdf_update(n_0_flows, bsdf_row, cross_coupler_flow)

        return new_flows, cross_coupler_flow  # carry, output

    scan_pack = (bsdf, topologies, substation_ids, injections)

    # Run scan
    n_0_flows, cross_coupler_flows = jax.lax.scan(
        body_fun,
        n_0_flows,
        scan_pack,
    )

    return n_0_flows, cross_coupler_flows
