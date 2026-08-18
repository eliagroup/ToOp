# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Does the same thing as preprocess_station_realizations.py but using SIMD paralelization for speed."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import structlog
from beartype.typing import Literal, Optional
from jaxtyping import Array, Bool, Int
from toop_engine_dc_solver.preprocess.helpers.switching_distance import per_station_switching_distance
from toop_engine_dc_solver.preprocess.preprocess_switching import OptimalSeparationSetInfo
from toop_engine_interfaces.asset_topology.assets_runtime import RuntimeBusbarCoupler
from toop_engine_interfaces.asset_topology.runtime_topology import (
    _validate_busgroup_physical_assignments,
    _validate_busgroup_switching_tables,
)
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import SimplifiedBusGroup
from toop_engine_interfaces.messages.preprocess.preprocess_commands import ReassignmentLimits

logger = structlog.get_logger(__name__)


def _construct_runtime_couplers(
    couplers: list[RuntimeBusbarCoupler],
    coupler_state_key: tuple[bool, ...],
) -> list[RuntimeBusbarCoupler]:
    """Construct runtime couplers for one open-state variant without revalidation."""
    return [
        RuntimeBusbarCoupler.model_construct(
            grid_model_id=coupler.grid_model_id,
            coupler_type=coupler.coupler_type,
            name=coupler.name,
            busbar_from_id=coupler.busbar_from_id,
            busbar_to_id=coupler.busbar_to_id,
            asset_bay=coupler.asset_bay,
            coupler_bay=coupler.coupler_bay,
            open=open_state,
            in_service=coupler.in_service,
        )
        for coupler, open_state in zip(couplers, coupler_state_key, strict=True)
    ]


def _construct_realized_station(
    station: SimplifiedBusGroup,
    couplers: list[RuntimeBusbarCoupler],
    branch_switching_table: np.ndarray,
) -> SimplifiedBusGroup:
    """Construct one realized station without revalidation in the hot loop."""
    return SimplifiedBusGroup.model_construct(
        bus_group_id=station.bus_group_id,
        voltage_level_id=station.voltage_level_id,
        name=station.name,
        station_type=station.station_type,
        region=station.region,
        voltage_level=station.voltage_level,
        busbars=station.busbars,
        bus_branch_bus_ids=station.bus_branch_bus_ids,
        couplers=couplers,
        branch_switching_table=branch_switching_table,
        injection_switching_table=station.injection_switching_table,
        branch_connectivity=station.branch_connectivity,
        injection_connectivity=station.injection_connectivity,
        model_log=station.model_log,
        branch_connections=station.branch_connections,
        injection_connections=station.injection_connections,
    )


def _validate_realized_station_update(
    station: SimplifiedBusGroup,
    action_switching: Bool[np.ndarray, " n_busbars n_branches"],
    action_coupler_states: Bool[np.ndarray, " n_couplers"],
) -> None:
    """Validate the mutated fields of a realized station.

    The realization step only changes the branch switching table and coupler open states.
    All other station-local payloads stay identical to the validated input station, so a full
    pydantic round-trip per action is unnecessary in this hot path.
    """
    if len(action_coupler_states) != len(station.couplers):
        raise ValueError(
            f"Coupler state count {len(action_coupler_states)} does not match station coupler count "
            f"{len(station.couplers)} for station {station.bus_group_id}."
        )

    _validate_busgroup_switching_tables(
        station_grid_model_id=station.bus_group_id,
        station_name=station.name,
        busbar_count=len(station.busbars),
        asset_count=len(station.branch_connections),
        asset_switching_table=np.asarray(action_switching, dtype=bool),
        asset_connectivity=station.branch_connectivity,
        asset_kind="branch",
    )
    _validate_busgroup_physical_assignments(
        station_grid_model_id=station.bus_group_id,
        station_name=station.name,
        asset_switching_table=np.asarray(action_switching, dtype=bool),
        asset_connectivity=station.branch_connectivity,
        asset_kind="branch",
    )


def _get_padded_action_batch_size(n_actions: int, max_batch_size: int) -> int:
    """Round batch sizes to powers of two to reduce JAX recompilations.

    For the hot station-realization path, the dominant source of wall time is repeated JAX
    compilation for slightly different action counts. Rounding the action-batch length to a small
    family of powers of two keeps execution batches bounded while collapsing many compile shapes.
    """
    if n_actions <= 1:
        return 1
    return min(max_batch_size, 1 << (n_actions - 1).bit_length())


@partial(jax.jit, static_argnames=("choice_heuristic",))
def _compute_switching_tables_for_action_batch(
    local_branch_action_batch: Bool[Array, " n_batch_actions n_assets"],
    current_coupler_state: Bool[Array, " n_couplers"],
    separation_set: Bool[Array, " n_separations 2 n_assets"],
    coupler_states: Bool[Array, " n_separations n_couplers"],
    busbar_mapping: Bool[Array, " n_separations n_busbars"],
    current_switching_table: Bool[Array, " n_busbars n_assets"],
    asset_connectivity: Bool[Array, " n_busbars n_assets"],
    choice_heuristic: Literal["first", "least_connected_busbar", "most_connected_busbar"],
) -> tuple[
    Bool[Array, " n_batch_actions n_busbars n_assets"],
    Bool[Array, " n_batch_actions n_couplers"],
    Bool[Array, " n_batch_actions n_busbars"],
    Int[Array, " n_batch_actions"],
    Int[Array, " n_batch_actions"],
    Bool[Array, " n_batch_actions"],
]:
    """Compute switching tables for one padded action batch.

    The batch length is rounded outside this function to a small family of powers of two.
    This keeps the compiled shapes stable across stations that differ mainly in the number of
    candidate branch actions.
    """
    return jax.lax.map(
        lambda local_action: compute_switching_table(
            local_action=local_action,
            current_coupler_state=current_coupler_state,
            separation_set=separation_set,
            coupler_states=coupler_states,
            busbar_mapping=busbar_mapping,
            current_switching_table=current_switching_table,
            asset_connectivity=asset_connectivity,
            choice_heuristic=choice_heuristic,
        ),
        local_branch_action_batch,
    )


def _compute_switching_tables_for_actions(
    local_branch_action_set: Bool[Array, " n_combinations n_assets"],
    current_coupler_state: Bool[Array, " n_couplers"],
    separation_set: Bool[Array, " n_separations 2 n_assets"],
    coupler_states: Bool[Array, " n_separations n_couplers"],
    busbar_mapping: Bool[Array, " n_separations n_busbars"],
    current_switching_table: Bool[Array, " n_busbars n_assets"],
    asset_connectivity: Bool[Array, " n_busbars n_assets"],
    batch_size: int,
    choice_heuristic: Literal["first", "least_connected_busbar", "most_connected_busbar"],
) -> tuple[
    Bool[np.ndarray, " n_combinations n_busbars n_assets"],
    Bool[np.ndarray, " n_combinations n_couplers"],
    Bool[np.ndarray, " n_combinations n_busbars"],
    Int[np.ndarray, " n_combinations"],
    Int[np.ndarray, " n_combinations"],
    Bool[np.ndarray, " n_combinations"],
]:
    """Compute switching tables while collapsing action-count-specific JAX shapes."""
    n_combinations = local_branch_action_set.shape[0]
    if n_combinations == 0:
        n_busbars, n_assets = current_switching_table.shape
        n_couplers = current_coupler_state.shape[0]
        return (
            np.zeros((0, n_busbars, n_assets), dtype=bool),
            np.zeros((0, n_couplers), dtype=bool),
            np.zeros((0, n_busbars), dtype=bool),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=bool),
        )

    switching_tables: list[np.ndarray] = []
    chosen_coupler_states: list[np.ndarray] = []
    chosen_busbar_mappings: list[np.ndarray] = []
    electrical_reassignment_distances: list[np.ndarray] = []
    physical_reassignment_distances: list[np.ndarray] = []
    failed_assignment_masks: list[np.ndarray] = []

    batch_start = 0
    while batch_start < n_combinations:
        remaining_actions = n_combinations - batch_start
        padded_batch_size = _get_padded_action_batch_size(remaining_actions, batch_size)
        batch_stop = min(batch_start + padded_batch_size, n_combinations)
        local_action_batch = np.asarray(local_branch_action_set[batch_start:batch_stop], dtype=bool)
        actual_batch_size = local_action_batch.shape[0]
        if actual_batch_size < padded_batch_size:
            local_action_batch = np.pad(
                local_action_batch,
                pad_width=((0, padded_batch_size - actual_batch_size), (0, 0)),
                mode="constant",
                constant_values=False,
            )

        (
            local_switching_table,
            local_chosen_coupler_state,
            local_chosen_busbar_mapping,
            local_el_reassignment_distance,
            local_phy_reassignment_distance,
            local_failed_assignments,
        ) = _compute_switching_tables_for_action_batch(
            local_branch_action_batch=jnp.asarray(local_action_batch, dtype=bool),
            current_coupler_state=current_coupler_state,
            separation_set=separation_set,
            coupler_states=coupler_states,
            busbar_mapping=busbar_mapping,
            current_switching_table=current_switching_table,
            asset_connectivity=asset_connectivity,
            choice_heuristic=choice_heuristic,
        )

        switching_tables.append(np.asarray(local_switching_table[:actual_batch_size]))
        chosen_coupler_states.append(np.asarray(local_chosen_coupler_state[:actual_batch_size]))
        chosen_busbar_mappings.append(np.asarray(local_chosen_busbar_mapping[:actual_batch_size]))
        electrical_reassignment_distances.append(np.asarray(local_el_reassignment_distance[:actual_batch_size]))
        physical_reassignment_distances.append(np.asarray(local_phy_reassignment_distance[:actual_batch_size]))
        failed_assignment_masks.append(np.asarray(local_failed_assignments[:actual_batch_size]))
        batch_start = batch_stop

    return (
        np.concatenate(switching_tables, axis=0),
        np.concatenate(chosen_coupler_states, axis=0),
        np.concatenate(chosen_busbar_mappings, axis=0),
        np.concatenate(electrical_reassignment_distances, axis=0),
        np.concatenate(physical_reassignment_distances, axis=0),
        np.concatenate(failed_assignment_masks, axis=0),
    )


def heuristic_first(
    requires_addition: Bool[Array, " n_assets"],
    current_switching_table: Bool[Array, " n_busbars n_assets"],
    possible_switching_table: Bool[Array, " n_busbars n_assets"],
) -> Bool[Array, " n_busbars n_assets"]:
    """Heuristic to choose the first True entry in the possible switching table.

    Parameters
    ----------
    requires_addition : Bool[Array, " n_assets"]
        A boolean array indicating whether an addition is required for the asset. This means at least one true entry in the
        possible switching table for the asset. Where this is false, the returned switching table will be equal to the
        current switching table.
    current_switching_table : Bool[Array, " n_busbars n_assets"]
        The current switching table from the station. It is expected that removals already have been applied.
    possible_switching_table : Bool[Array, " n_busbars n_assets"]
        The possible switching table for the asset. This is True if the asset could be on the phy busbar after the action.

    Returns
    -------
    Bool[Array, " n_busbars n_assets"]
        The updated switching table after applying the heuristic. This will be equal to the current switching table if no
        addition is required for the asset and will have exactly one True entry for each asset if an addition is required.
    """
    first_true: Int[Array, " n_assets"] = jnp.argmax(possible_switching_table, axis=0)

    return jnp.where(
        requires_addition[None, :],
        current_switching_table.at[first_true, range(requires_addition.shape[0])].set(True),
        current_switching_table,
    )


def heuristic_least_connected_busbar(
    requires_addition: Bool[Array, " n_assets"],
    current_switching_table: Bool[Array, " n_busbars n_assets"],
    possible_switching_table: Bool[Array, " n_busbars n_assets"],
    least_connected: bool = True,
) -> Bool[Array, " n_busbars n_assets"]:
    """Heuristic to choose the busbar with the least number of connected assets

    Parameters
    ----------
    requires_addition : Bool[Array, " n_assets"]
        A boolean array indicating whether an addition is required for the asset. This means at least one true entry in the
        possible switching table for the asset. In other words, for asset index i, if requires_addition[i] is True, it must
        hold that any(possible_switching_table[:, i]. For the assets for which this is false, the returned switching table
        will be equal to the current switching table.
    current_switching_table : Bool[Array, " n_busbars n_assets"]
        The current switching table from the station. It is expected that removals already have been applied.
    possible_switching_table : Bool[Array, " n_busbars n_assets"]
        The possible switching table for the asset. This is True if the asset could be on the phy busbar after the action.
    least_connected : bool, optional
        If True, the heuristic will select the busbar with the least number of connected assets. If False, it will select
        the busbar with the most connected assets. Least connected assets is better under busbar outages.

    Returns
    -------
    Bool[Array, " n_busbars n_assets"]
        The updated switching table after applying the heuristic. This will be equal to the current switching table if no
        addition is required for the asset and will have exactly one True entry for each asset if an addition is required.
    """
    # There could be a situation where the assets on the left half of the station are flexible in where they can be connected
    # but the right half can only be on one busbar. In such a scenario, naively running from left to right through the assets
    # would yield a highly unbalanced asset distribution over the busbars. Hence, we first apply the assets that have exactly
    # one possible busbar to connect to and then apply the rest.
    single_busbar_assets: Bool[Array, " n_assets"] = jnp.sum(possible_switching_table, axis=0) == 1
    first_hit: Int[Array, " n_assets"] = jnp.argmax(possible_switching_table, axis=0)

    current_switching_table = jnp.where(
        single_busbar_assets[None, :],
        current_switching_table.at[first_hit, range(requires_addition.shape[0])].set(True),
        current_switching_table,
    )
    requires_addition = requires_addition & ~single_busbar_assets

    # Count the number of assets connected to each busbar
    busbar_connection_count: Int[Array, " n_busbars"] = jnp.sum(current_switching_table, axis=1)

    def _select_busbar(
        connection_count: Int[Array, " n_busbars"], xs: tuple[Bool[Array, " n_busbars"], Bool[Array, " "]]
    ) -> tuple[Int[Array, " n_busbars"], Int[Array, " "]]:
        """Select the busbar with the least number of connected assets.

        Parameters
        ----------
        connection_count : Int[Array, " n_busbars"]
            The number of assets connected to each busbar until now
        xs : tuple[Bool[Array, " n_busbars"], Bool[Array, " "]]
            The entries for the asset in the possible switching table and whether an addition is required.

        Returns
        -------
        Int[Array, " n_busbars"]
            The updated connection count after the selection.
        Int[Array, " "]
            The index of the busbar that was selected for switching.
        """
        asset_possible_switching, asset_requires_addition = xs

        if least_connected:
            masked_connection_count = jnp.where(asset_possible_switching, connection_count, jnp.inf)
            selected_busbar_index = jnp.argmin(masked_connection_count)
        else:
            masked_connection_count = jnp.where(asset_possible_switching, connection_count, -jnp.inf)
            selected_busbar_index = jnp.argmax(masked_connection_count)
        connection_count = jnp.where(
            asset_requires_addition, connection_count.at[selected_busbar_index].add(1), connection_count
        )
        return connection_count, selected_busbar_index

    _, selected_busbar_indices = jax.lax.scan(
        f=_select_busbar, init=busbar_connection_count, xs=(possible_switching_table.T, requires_addition.T), unroll=True
    )

    # Now we can update the current switching table with the selected busbar indices
    current_switching_table = jnp.where(
        requires_addition[None, :],
        current_switching_table.at[selected_busbar_indices, range(requires_addition.shape[0])].set(True),
        current_switching_table,
    )
    return current_switching_table


def compute_switching_table(
    local_action: Bool[Array, " n_assets"],
    current_coupler_state: Bool[Array, " n_couplers"],
    separation_set: Bool[Array, " n_separations 2 n_assets"],
    coupler_states: Bool[Array, " n_separations n_couplers"],
    busbar_mapping: Bool[Array, " n_separations n_busbars"],
    current_switching_table: Bool[Array, " n_busbars n_assets"],
    asset_connectivity: Bool[Array, " n_busbars n_assets"],
    choice_heuristic: Literal["first", "least_connected_busbar", "most_connected_busbar"],
) -> tuple[
    Bool[Array, " n_busbars n_assets"],
    Bool[Array, " n_couplers"],
    Bool[Array, " n_busbars"],
    Int[Array, " "],
    Int[Array, " "],
    Bool[Array, " "],
]:
    """Compute the switching table for a given local action.

    This translates the electrical action to a physical asset switching table minimizing switching distance and applying
    a heuristic to choose a physical busbar if multiple are possible.

    Parameters
    ----------
    local_action : Bool[Array, " n_assets"]
        A boolean array indicating the local branch action to be realised. It is an array of length n_assets with a True
        if that asset is to be on el. busbar B
    current_coupler_state : Bool[Array, " n_couplers"]
        A boolean array indicating the current state of the couplers. It is an array of length n_couplers with a True
        if that coupler is open.
    separation_set : Bool[Array, " n_separations 2 n_assets"]
        A boolean array indicating the separation set. It is an array of shape (n_separations, 2, n_assets) where the
        first dimension is the number of separations in the separation set, the second dimension is 2 (indicating busbar A
        and B), and the third dimension is the number of assets. Each entry represents whether the asset is connected to the
        corresponding el. busbar (True) or not (False).
    coupler_states : Bool[Array, " n_separations n_couplers"]
        A boolean array indicating the coupler states for each separation in the separation set. It is an array of shape
        (n_separations, n_couplers) where each entry represents whether the coupler is open (True) or closed (False)
        for that separation.
    busbar_mapping : Bool[Array, " n_separations n_busbars"]
        A boolean array indicating which phy. busbars are el. busbar B for each separation in the separation set. It is an
        array of shape (n_separations, n_busbars) where each entry is True if the phy. busbar is el. busbar B for that
        separation and False if it is el. busbar A.
    current_switching_table : Bool[Array, " n_busbars n_assets"]
        The current switching table from the station. This is an array of shape (n_assets, n_busbars) where each
        entry is True if the asset is connected to the corresponding phy. busbar and False if it is not.
    asset_connectivity : Bool[Array, " n_busbars n_assets"]
        A boolean array indicating the connectivity of assets to phy busbars. It is an array of shape (n_assets, n_busbars)
        where each entry is True if the asset can be connected to the corresponding phy busbar and False if it cannot.
    choice_heuristic : Literal["first", "least_connected_busbar", "most_connected_busbar"], optional
        The heuristic to use when multiple busbars are possible for an asset.

    Returns
    -------
    current_switching_table : Bool[Array, " n_busbars n_assets"]
        The updated switching table after applying the local action.
    chosen_coupler_state : Bool[Array, " n_couplers"]
        The coupler state that results in the chosen separation, from the coupler_states array.
    chosen_busbar_mapping : Bool[Array, " n_busbars"]
        The busbar mapping from the separation, True if the phy. busbar is el. busbar B, False if it is el. busbar A.
    el_reassignment_distance : Int[Array, " "]
        The electrical reassignment distance, i.e. the number of assets that had to be reassigned to a different el. busbar
    phy_reassignment_distance : Int[Array, " "]
        The physical reassignment distance, i.e. the number of assets that had to be reassigned to a different phy. busbar.
        This is always equal or larger than the electrical reassignment distance.
    failed_assignments : Bool[Array, " "]
        If there were any failed assignments, i.e. a local action that could not be realized due to
        asset connectivity constraints. If True, it means that the function returned
        undefined results for all other outputs and the local action should be removed from the action set.
    """
    original_switching_table = current_switching_table.copy()

    (
        best_configuration,
        invert,
        el_reassignment_distance,
        _coupler_distance,
    ) = per_station_switching_distance(
        target_configuration=local_action,
        current_coupler_state=current_coupler_state,
        separation_set=separation_set,
        coupler_states=coupler_states,
    )

    # Select the best configuration from the separation set
    chosen_coupler_state = coupler_states[best_configuration]
    chosen_busbar_mapping = busbar_mapping[best_configuration]

    # We need to invert the busbar b mappings if the best configuration is inverted
    # We also keep the inverse busbar mapping if we want to invert again due to failed assignments
    chosen_busbar_mapping: Bool[Array, " n_busbars"] = jnp.where(invert, ~chosen_busbar_mapping, chosen_busbar_mapping)
    chosen_busbar_mapping_inv = ~chosen_busbar_mapping

    # The possible switching table is true if an asset could be on the phy busbar after the action
    # This is the case if either the asset should be on el busbar B and the phy busbar is el busbar B or if the asset
    # should be on el busbar A and the phy busbar is el busbar A.
    # We also keep the inverse possible switching table in case we need to invert again due to failed assignments.
    possible_switching_table: Bool[Array, " n_busbars n_assets"] = (
        local_action[None, :] & chosen_busbar_mapping[:, None]
    ) | (~local_action[None, :] & ~chosen_busbar_mapping[:, None])
    possible_switching_table_inv = (local_action[None, :] & chosen_busbar_mapping_inv[:, None]) | (
        ~local_action[None, :] & ~chosen_busbar_mapping_inv[:, None]
    )

    # Erase entries from the possible switching table which are not allowed per asset connectivity
    possible_switching_table = possible_switching_table & asset_connectivity
    possible_switching_table_inv = possible_switching_table_inv & asset_connectivity

    # It is possible that the action is not feasible, i.e. that the asset cannot be on the busbar
    # In that case, the action needs to be removed from the local action set (outside of this function)
    # This function will return undefined results if the action is not feasible.
    failed_assignments: Bool[Array, " "] = jnp.any(jnp.all(~possible_switching_table, axis=0))
    failed_assignments_inv = jnp.any(jnp.all(~possible_switching_table_inv, axis=0))

    # When we have a failed attempt, invert again to see if this fixes the issue.
    # If both attempts fail, we likely have an unresolvable action.
    chosen_busbar_mapping = jnp.where(failed_assignments, chosen_busbar_mapping_inv, chosen_busbar_mapping)
    possible_switching_table = jnp.where(failed_assignments, possible_switching_table_inv, possible_switching_table)
    failed_assignments = failed_assignments & failed_assignments_inv

    # Gather and apply all the asset removals
    asset_removals: Bool[Array, " n_busbars n_assets"] = current_switching_table & ~possible_switching_table
    current_switching_table = jnp.where(asset_removals, False, current_switching_table)

    # Gather all the necessary additions
    # We only have to act when the current switching table is False for all True entries in the possible switching table
    # because it is enough if the asset is connected to any of the busbars.
    requires_addition: Bool[Array, " n_assets"] = ~jnp.any(current_switching_table & possible_switching_table, axis=0)

    # For additions we need to choose a busbar out of the possible switching table.
    # As multiple might be available, we apply a heuristic to choose one.
    heuristic_map = {
        "first": heuristic_first,
        "least_connected_busbar": partial(heuristic_least_connected_busbar, least_connected=True),
        "most_connected_busbar": partial(heuristic_least_connected_busbar, least_connected=False),
    }
    current_switching_table = heuristic_map[choice_heuristic](
        requires_addition=requires_addition,
        current_switching_table=current_switching_table,
        possible_switching_table=possible_switching_table,
    )

    # Also compute the physical reassignment distance, which can be larger than the electrical
    # distance if an asset was connected to multiple busbars before the action.
    phy_reassignment_distance = jnp.sum(jnp.logical_xor(original_switching_table, current_switching_table))

    return (
        current_switching_table,
        chosen_coupler_state,
        chosen_busbar_mapping,
        el_reassignment_distance,
        phy_reassignment_distance,
        failed_assignments,
    )


def realise_ba_to_physical_topo_per_station_jax(
    local_branch_action_set: Bool[np.ndarray, " n_combinations n_branches"],
    bus_group: SimplifiedBusGroup,
    separation_set_info: OptimalSeparationSetInfo,
    batch_size: int = 1024,
    choice_heuristic: Literal["first", "least_connected_busbar", "most_connected_busbar"] = "least_connected_busbar",
    validate: bool = True,
    reassignment_limits: Optional[ReassignmentLimits] = None,
) -> tuple[list[SimplifiedBusGroup], Bool[np.ndarray, "n_combinations n_branches"], list[list[int]], list[int]]:
    """Realize the branch actions to physical topology per station.

    This iterates over all actions in the local branch action set and tries to find a realization for them.

    Parameters
    ----------
    local_branch_action_set : Bool[np.ndarray, "n_combinations n_branches"]
        A boolean array indicating the set of local branch actions to be realised.
    bus_group : SimplifiedBusGroup
        The bus group object representing the electrical station where the actions are to be realised. This assumes a
        simplified station
    separation_set_info : OptimalSeparationSetInfo
        The optimal separation set info for the station as computed by make_optimal_separation_set.
    batch_size : int, optional
        The batch size for SIMD parallelization during the computation of the switching tables, by default 1024.
        Reduce if the memory usage is too high.
    choice_heuristic : Literal["first", "least_connected_busbar", "most_connected_busbar"], optional
        The heuristic to use when multiple busbars are possible for an asset. This is used to choose which busbar to
        connect the asset to if multiple are possible. The options are:
        - "first": Choose the first True entry in the possible switching table.
        - "least_connected_busbar": Choose the busbar with the least number of connected assets.
        - "most_connected_busbar": Choose the busbar with the most connected assets.
        By default, it is set to "least_connected_busbar".
    validate : bool, optional
        Whether to validate the station before processing. This will invoke the station validators for all stations.
    reassignment_limits : Optional[ReassignmentLimits], optional
        If given, settings to limit the amount of reassignment during the physical reconfiguration.

    Returns
    -------
    realised_stations : list
        A list of bus groups with the realised branch actions. This will not include the full station as it was in the
        grid model, but a simplified version of it. For the list of simplifications, consult prepare_for_separation_set.
    local_branch_action_set : Bool[np.ndarray, "n_combinations n_branches"]
        The updated local branch action set with infeasible actions removed.
    busbar_a_mappings : list[list[int]]
        A list of busbar A mappings for each branch action. The outer list is of
        length equal to the number of branch_actions for the station. The inner list
        contains the busbar A mappings for the corresponding branch action.
    reassignment_distances : list[int]
        A list of reassignment distances for each branch action. The length of the list
        is equal to the number of branch actions for the station. Each element
        represents the number of reassignments needed to reach the target configuration
    """
    separation_set, coupler_states, _coupler_distances, busbar_a_separation = (
        separation_set_info.separation_set,
        separation_set_info.coupler_states,
        separation_set_info.coupler_distance,
        separation_set_info.busbar_a,
    )
    current_coupler_state = [c.open for c in bus_group.couplers]

    if local_branch_action_set.shape[0] == 0:
        return [], local_branch_action_set, [], []

    if separation_set.size == 0 or not np.any(local_branch_action_set[1:]):
        logger.warning(
            f"No separation set is possible for the station {bus_group.bus_group_id}.",
        )
        return [bus_group], local_branch_action_set[:1].copy(), [list(range(len(bus_group.busbars)))], [0]

    # Make an array out of busbar_a_separation
    # This is an array which is True if a phy busbar is el busbar B for that configuration
    n_buses = len(bus_group.busbars)
    busbar_b_array: Bool[np.ndarray, " n_configurations n_busbars"] = np.stack(
        [
            np.array([busbar_index not in separation for busbar_index in range(n_buses)], dtype=bool)
            for separation in busbar_a_separation
        ]
    )

    # We expect the first action to be the unsplit action - for that we don't have to do anything
    assert not np.any(local_branch_action_set[0])
    local_branch_action_set = local_branch_action_set[1:]

    n_branches = local_branch_action_set.shape[1]
    separation_set = separation_set[:, :, :n_branches]

    branch_switching_table = np.asarray(bus_group.branch_switching_table, dtype=bool)
    branch_connectivity = (
        bus_group.branch_connectivity
        if bus_group.branch_connectivity is not None
        else np.ones_like(bus_group.branch_switching_table, dtype=bool)
    )
    assert branch_switching_table.shape[1] == local_branch_action_set.shape[1], (
        "The number of branches in the station must match the number of branches in the local action set."
    )
    assert branch_connectivity.shape[1] == local_branch_action_set.shape[1], (
        "The number of branches in the station must match the number of branches in the local action set."
    )
    # Map over the local action set and compute a switching table for each action
    with jax.default_device(jax.devices("cpu")[0]):
        (
            switching_table,
            chosen_coupler_state,
            chosen_busbar_mapping,
            _el_reassignment_distance,
            phy_reassignment_distance,
            failed_assignments,
        ) = _compute_switching_tables_for_actions(
            local_branch_action_set=jnp.array(local_branch_action_set, dtype=bool),
            current_coupler_state=jnp.array(current_coupler_state, dtype=bool),
            separation_set=jnp.array(separation_set, dtype=bool),
            coupler_states=jnp.array(coupler_states, dtype=bool),
            busbar_mapping=jnp.array(busbar_b_array, dtype=bool),
            current_switching_table=jnp.array(branch_switching_table, dtype=bool),
            asset_connectivity=jnp.array(branch_connectivity, dtype=bool),
            batch_size=batch_size,
            choice_heuristic=choice_heuristic,
        )
    # Remove failed assignments from the local branch action set
    success = ~failed_assignments
    switching_table = np.array(switching_table[success])
    local_branch_action_set = np.array(local_branch_action_set[success])
    chosen_coupler_state = np.array(chosen_coupler_state[success])
    chosen_busbar_mapping = np.array(chosen_busbar_mapping[success])
    phy_reassignment_distance = np.array(phy_reassignment_distance[success])

    # Only keep those within the reassignment limits
    if reassignment_limits is not None:
        max_reassignments = reassignment_limits.station_specific_limits.get(
            bus_group.bus_group_id, reassignment_limits.max_reassignments_per_sub
        )
        within_limit = phy_reassignment_distance <= max_reassignments
        switching_table = switching_table[within_limit]
        local_branch_action_set = local_branch_action_set[within_limit]
        chosen_coupler_state = chosen_coupler_state[within_limit]
        chosen_busbar_mapping = chosen_busbar_mapping[within_limit]
        phy_reassignment_distance = phy_reassignment_distance[within_limit]

    # Create the realised stations. This dominates runtime for large action sets, so reuse
    # the already validated station payload and cache the few repeated coupler-state variants.
    coupler_state_cache: dict[tuple[bool, ...], list[RuntimeBusbarCoupler]] = {}
    realised_stations: list[SimplifiedBusGroup] = []
    for action_switching, action_coupler_states in zip(switching_table, chosen_coupler_state, strict=True):
        if validate:
            _validate_realized_station_update(
                station=bus_group,
                action_switching=action_switching,
                action_coupler_states=action_coupler_states,
            )

        coupler_state_key = tuple(bool(open_state) for open_state in action_coupler_states)
        cached_couplers = coupler_state_cache.get(coupler_state_key)
        if cached_couplers is None:
            cached_couplers = _construct_runtime_couplers(bus_group.couplers, coupler_state_key)
            coupler_state_cache[coupler_state_key] = cached_couplers

        realised_stations.append(_construct_realized_station(bus_group, cached_couplers, action_switching))

    # Convert the busbar mapping to a list of busbar A mappings
    busbar_mappings_converted = [np.flatnonzero(~mapping).tolist() for mapping in chosen_busbar_mapping]
    # Add the unsplit action for every returned element
    return (
        [bus_group, *realised_stations],
        np.concatenate([np.zeros((1, n_branches), dtype=bool), local_branch_action_set], axis=0),
        [list(range(n_buses)), *busbar_mappings_converted],
        [0, *phy_reassignment_distance.tolist()],
    )
