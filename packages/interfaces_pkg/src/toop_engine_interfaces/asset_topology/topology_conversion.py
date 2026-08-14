# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Conversions between materialized bus groups and canonical master data."""

from dataclasses import dataclass

import numpy as np
import structlog
from beartype.typing import Optional
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    BusbarCoupler,
    InjectionAsset,
)
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeBusGroup,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RuntimeSwitchingState:
    """Compact runtime overlay describing the current live switching state of a bus group."""

    busbar_bus_branch_bus_ids: Optional[dict[str, str]] = None
    """Mapping from canonical busbar ids to current runtime bus-branch bus ids."""

    branch_current_bus_ids: Optional[list[str | None]] = None
    """Current runtime bus ids for branch connections, aligned with the bus group's
    canonical branch connections."""

    injection_current_bus_ids: Optional[list[str | None]] = None
    """Current runtime bus ids for injection connections, aligned with the bus group's
    canonical injection connections."""
    busbar_out_of_service_ids: set[str] = frozenset()
    """Canonical busbar ids that are currently out of service."""
    open_coupler_ids: set[str] = frozenset()
    """Canonical coupler ids that are currently open."""
    out_of_service_coupler_ids: set[str] = frozenset()
    """Canonical coupler ids that are currently out of service."""
    open_switch_ids: set[str] = frozenset()
    """Asset-bay switch ids that are currently open."""


def _copy_runtime_or_raise(runtime_map: dict[str, object], grid_model_id: str, runtime_kind: str) -> object:
    """Return a runtime payload by id or raise a targeted validation error.

    Parameters
    ----------
    runtime_map : dict[str, object]
        Mapping from grid-model ids to runtime payload objects.
    grid_model_id : str
        Requested runtime payload id.
    runtime_kind : str
        Human-readable payload kind used in the error message.

    Returns
    -------
    object
        Runtime payload stored under ``grid_model_id``.

    Raises
    ------
    ValueError
        If the requested runtime payload is missing.
    """
    try:
        return runtime_map[grid_model_id]
    except KeyError as error:
        raise ValueError(f"Missing runtime {runtime_kind} {grid_model_id} during topology realization") from error


def validate_complete_master_asset_topology(master_data: MasterAssetTopology) -> None:
    """Validate that canonical master data already contains productive asset metadata.

    Parameters
    ----------
    master_data : MasterAssetTopology
        Canonical master data to validate.

    Raises
    ------
    ValueError
        If canonical assets or station-local branch references are missing metadata that productive
        preprocessing requires.
    """
    for asset in master_data.branch_assets:
        if asset.name is None:
            raise ValueError(f"Branch asset {asset.grid_model_id} is missing canonical name in master data")

    for station in master_data.bus_groups:
        for asset_connection in station.branch_connections:
            if asset_connection.branch_end is None:
                raise ValueError(
                    f"Branch asset {asset_connection.asset_id} in station {station.bus_group_id} "
                    "is missing canonical branch_end in master data"
                )


def _build_switching_table_from_compact_runtime(
    master_bus_group: MasterBusGroup,
    asset_connections: list[BusGroupAssetConnection],
    asset_connectivity: Optional[np.ndarray],
    asset_bay_map: dict[str, AssetBay],
    runtime_switching_state: RuntimeSwitchingState,
    asset_kind: str,
) -> np.ndarray:
    """Rebuild one station switching table directly from compact runtime state.

    Parameters
    ----------
    master_bus_group : MasterBusGroup
        Canonical bus group whose runtime switching table should be reconstructed.
    asset_connections : list[StationAssetConnection]
        Canonical station-local asset references aligned with the target table.
    asset_connectivity : Optional[np.ndarray]
        Canonical connectivity table aligned with ``asset_connections``.
    asset_bay_map : dict[str, AssetBay]
        Canonical asset bays keyed by asset-bay id.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay describing the current network state.
    asset_kind : str
        Human-readable asset kind used in validation errors.

    Returns
    -------
    np.ndarray
        Boolean switching table with shape ``(n_busbar, n_asset)``.

    Raises
    ------
    ValueError
        If the compact runtime state cannot be mapped consistently back to the
        canonical station structure.
    """
    switching_table = np.zeros((len(master_bus_group.busbars), len(asset_connections)), dtype=bool)
    busbar_index_by_id = {busbar.grid_model_id: index for index, busbar in enumerate(master_bus_group.busbars)}
    asset_current_bus_ids = (
        runtime_switching_state.branch_current_bus_ids
        if asset_kind == "branch"
        else runtime_switching_state.injection_current_bus_ids
    )

    for asset_index, asset_connection in enumerate(asset_connections):
        if asset_connection.asset_bay_id is None:
            _assign_switching_from_connectivity(
                switching_table=switching_table,
                master_bus_group=master_bus_group,
                asset_connection=asset_connection,
                asset_connectivity=asset_connectivity,
                runtime_switching_state=runtime_switching_state,
                asset_current_bus_ids=asset_current_bus_ids,
                asset_kind=asset_kind,
                asset_index=asset_index,
            )
            continue

        _assign_switching_from_asset_bay(
            switching_table=switching_table,
            master_bus_group=master_bus_group,
            asset_connection=asset_connection,
            asset_bay_map=asset_bay_map,
            runtime_switching_state=runtime_switching_state,
            busbar_index_by_id=busbar_index_by_id,
            asset_index=asset_index,
        )

        if asset_connectivity is not None:
            invalid_assignments = np.logical_and(
                switching_table[:, asset_index], np.logical_not(asset_connectivity[:, asset_index])
            )
            if invalid_assignments.any():
                raise ValueError(
                    f"Compact runtime reconstruction violates {asset_kind} connectivity for asset "
                    f"{asset_connection.asset_id} in station {master_bus_group.bus_group_id}"
                )

    return switching_table


def _assign_switching_from_connectivity(
    switching_table: np.ndarray,
    master_bus_group: MasterBusGroup,
    asset_connection: BusGroupAssetConnection,
    asset_connectivity: Optional[np.ndarray],
    runtime_switching_state: RuntimeSwitchingState,
    asset_current_bus_ids: Optional[list[str | None]],
    asset_kind: str,
    asset_index: int,
) -> None:
    """Assign one switching-table column from compact connectivity information.

    Parameters
    ----------
    switching_table : np.ndarray
        Mutable switching table under construction.
    master_bus_group : MasterBusGroup
        Canonical bus group owning the switching table.
    asset_connection : StationAssetConnection
        Canonical asset reference for the column being assigned.
    asset_connectivity : Optional[np.ndarray]
        Canonical connectivity constraints for the station.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay for the station.
    asset_current_bus_ids : Optional[list[str | None]]
        Runtime bus ids aligned with the asset columns.
    asset_kind : str
        Human-readable asset kind used in validation errors.
    asset_index : int
        Column index within the switching table.

    Raises
    ------
    ValueError
        If the runtime state cannot be resolved unambiguously for the asset.
    """
    if asset_connectivity is None:
        raise ValueError(
            f"Missing asset bay and connectivity for {asset_kind} asset {asset_connection.asset_id} "
            f"in bus group {master_bus_group.bus_group_id}"
        )

    current_bus_id = asset_current_bus_ids[asset_index] if asset_current_bus_ids is not None else None
    matching_busbar_indices = _get_matching_busbar_indices(
        master_bus_group=master_bus_group,
        asset_connectivity=asset_connectivity,
        runtime_switching_state=runtime_switching_state,
        current_bus_id=current_bus_id,
        asset_index=asset_index,
    )
    if len(matching_busbar_indices) == 1:
        switching_table[matching_busbar_indices[0], asset_index] = True
        return

    candidate_busbars = np.flatnonzero(asset_connectivity[:, asset_index])
    if len(candidate_busbars) == 1:
        switching_table[int(candidate_busbars[0]), asset_index] = True
        return
    if current_bus_id is None:
        return

    if len(candidate_busbars) > 1:
        # Legacy pandapower master data can lack asset-bay mappings while collapsing multiple
        # physical busbars onto one runtime electrical bus. In that case there is no unique
        # reconstruction signal left, so keep the realization deterministic by choosing the
        # first physically admissible busbar slot.
        switching_table[int(candidate_busbars[0]), asset_index] = True
        return

    raise ValueError(
        f"Cannot compactly realize {asset_kind} asset {asset_connection.asset_id} in bus group "
        f"{master_bus_group.bus_group_id} without asset bay mapping"
    )


def _get_matching_busbar_indices(
    master_bus_group: MasterBusGroup,
    asset_connectivity: np.ndarray,
    runtime_switching_state: RuntimeSwitchingState,
    current_bus_id: str | None,
    asset_index: int,
) -> list[int]:
    """Return busbar indices matching the current runtime bus id for one asset.

    Parameters
    ----------
    master_bus_group : MasterBusGroup
        Canonical bus group owning the busbars.
    asset_connectivity : np.ndarray
        Canonical connectivity table for the relevant asset class.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay for the station.
    current_bus_id : str | None
        Current runtime bus id for the asset column.
    asset_index : int
        Column index within the switching table.

    Returns
    -------
    list[int]
        Busbar row indices that match the runtime bus id and remain physically allowed.
    """
    if current_bus_id is None or runtime_switching_state.busbar_bus_branch_bus_ids is None:
        return []

    matching_busbar_indices: list[int] = []
    for busbar_index, busbar in enumerate(master_bus_group.busbars):
        mapped_bus_id = runtime_switching_state.busbar_bus_branch_bus_ids.get(busbar.grid_model_id)
        matches_current_bus_id = current_bus_id in {mapped_bus_id, busbar.grid_model_id}
        if matches_current_bus_id and asset_connectivity[busbar_index, asset_index]:
            matching_busbar_indices.append(busbar_index)
    return matching_busbar_indices


def _assign_switching_from_asset_bay(
    switching_table: np.ndarray,
    master_bus_group: MasterBusGroup,
    asset_connection: BusGroupAssetConnection,
    asset_bay_map: dict[str, AssetBay],
    runtime_switching_state: RuntimeSwitchingState,
    busbar_index_by_id: dict[str, int],
    asset_index: int,
) -> None:
    """Assign one switching-table column from runtime asset-bay switch state.

    Parameters
    ----------
    switching_table : np.ndarray
        Mutable switching table under construction.
    master_bus_group : MasterBusGroup
        Canonical bus group owning the switching table.
    asset_connection : StationAssetConnection
        Canonical asset reference for the column being assigned.
    asset_bay_map : dict[str, AssetBay]
        Canonical asset bays keyed by asset-bay id.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay for the station.
    busbar_index_by_id : dict[str, int]
        Mapping from canonical busbar id to row index in the switching table.
    asset_index : int
        Column index within the switching table.

    Raises
    ------
    ValueError
        If an asset bay references a busbar that is not part of the station.
    """
    asset_bay = _copy_runtime_or_raise(asset_bay_map, asset_connection.asset_bay_id, "asset bay")
    assert isinstance(asset_bay, AssetBay)
    if asset_bay.breaker_grid_model_id in runtime_switching_state.open_switch_ids:
        return

    for busbar_id, switch_id in asset_bay.busbar_disconnector_grid_model_id.items():
        if switch_id in runtime_switching_state.open_switch_ids:
            continue
        try:
            busbar_index = busbar_index_by_id[busbar_id]
        except KeyError as error:
            raise ValueError(
                f"Asset bay {asset_bay.asset_bay_id} references unknown busbar {busbar_id} in bus group "
                f"{master_bus_group.bus_group_id}"
            ) from error
        switching_table[busbar_index, asset_index] = True


def _resolve_runtime_coupler_busbar_id(
    selector_switch_ids: dict[str, str],
    runtime_switching_state: RuntimeSwitchingState,
    direct_busbar_grid_model_ids: list[str],
) -> tuple[str | None, bool]:
    """Resolve one coupler side from the live switch state.

    If selector switches exist, exactly one closed selector determines the endpoint.
    If no selector switches exist, a fixed one-busbar side is resolved from the physical
    coupler-bay metadata. Ambiguous sides keep a best-effort endpoint id from the
    visible switch metadata, but are reported as unresolved.
    """
    if not selector_switch_ids:
        if len(direct_busbar_grid_model_ids) != 1:
            runtime_bus_ids_by_direct_busbar = {
                busbar_grid_model_id: runtime_switching_state.busbar_bus_branch_bus_ids.get(busbar_grid_model_id)
                for busbar_grid_model_id in direct_busbar_grid_model_ids
            }
            connected_direct_busbars = [
                busbar_grid_model_id
                for busbar_grid_model_id, runtime_bus_id in runtime_bus_ids_by_direct_busbar.items()
                if runtime_bus_id not in {None, ""}
            ]
            fallback_busbar_grid_model_id = connected_direct_busbars[0] if connected_direct_busbars else None
            if fallback_busbar_grid_model_id is None and direct_busbar_grid_model_ids:
                fallback_busbar_grid_model_id = direct_busbar_grid_model_ids[0]
            return fallback_busbar_grid_model_id, False
        resolved_busbar_grid_model_id = direct_busbar_grid_model_ids[0]
        if resolved_busbar_grid_model_id in runtime_switching_state.busbar_out_of_service_ids:
            return resolved_busbar_grid_model_id, False
        return resolved_busbar_grid_model_id, True

    connected_busbar_ids = [
        busbar_id
        for busbar_id, switch_id in selector_switch_ids.items()
        if switch_id not in runtime_switching_state.open_switch_ids
    ]
    if len(connected_busbar_ids) != 1:
        fallback_busbar_id = connected_busbar_ids[0] if connected_busbar_ids else next(iter(selector_switch_ids), None)
        return fallback_busbar_id, False
    resolved_busbar_grid_model_id = connected_busbar_ids[0]
    if resolved_busbar_grid_model_id in runtime_switching_state.busbar_out_of_service_ids:
        return resolved_busbar_grid_model_id, False
    return resolved_busbar_grid_model_id, True


def _get_unresolved_coupler_side_candidates(
    selector_switch_ids: dict[str, str],
    direct_busbar_grid_model_ids: list[str],
) -> list[str]:
    """Return ordered candidate busbars for one unresolved coupler side."""
    if selector_switch_ids:
        return list(selector_switch_ids)
    return direct_busbar_grid_model_ids


def _pick_distinct_coupler_busbar_id(candidate_busbar_ids: list[str], other_busbar_id: str | None) -> str | None:
    """Prefer a candidate busbar different from the opposite coupler side."""
    for candidate_busbar_id in candidate_busbar_ids:
        if candidate_busbar_id != other_busbar_id:
            return candidate_busbar_id
    return candidate_busbar_ids[0] if candidate_busbar_ids else None


def _materialize_runtime_coupler(
    coupler: BusbarCoupler,
    busbar_int_id_by_grid_model_id: dict[str, int],
    runtime_switching_state: RuntimeSwitchingState,
) -> RuntimeBusbarCoupler:
    """Build one runtime coupler payload from canonical structure and live switch state."""
    coupler_data = coupler.model_dump()
    open_state = coupler.grid_model_id in runtime_switching_state.open_coupler_ids
    in_service = coupler.grid_model_id not in runtime_switching_state.out_of_service_coupler_ids
    update: dict[str, object] = {"open": open_state, "in_service": in_service}

    if coupler.coupler_bay is None:
        raise ValueError(f"Coupler {coupler.grid_model_id} cannot be materialized without coupler_bay")

    if any(
        switch_id in runtime_switching_state.open_switch_ids
        for switch_id in [
            *coupler.coupler_bay.coupler_breaker_ids,
            *coupler.coupler_bay.coupler_disconnector_ids,
        ]
    ):
        open_state = True

    from_busbar_grid_model_id, from_side_resolved = _resolve_runtime_coupler_busbar_id(
        selector_switch_ids=coupler.coupler_bay.from_busbar_disconnector_ids,
        runtime_switching_state=runtime_switching_state,
        direct_busbar_grid_model_ids=coupler.coupler_bay.from_busbar_ids,
    )

    to_busbar_grid_model_id, to_side_resolved = _resolve_runtime_coupler_busbar_id(
        selector_switch_ids=coupler.coupler_bay.to_busbar_disconnector_ids,
        runtime_switching_state=runtime_switching_state,
        direct_busbar_grid_model_ids=coupler.coupler_bay.to_busbar_ids,
    )

    if from_busbar_grid_model_id == to_busbar_grid_model_id and not from_side_resolved:
        from_busbar_grid_model_id = _pick_distinct_coupler_busbar_id(
            candidate_busbar_ids=_get_unresolved_coupler_side_candidates(
                selector_switch_ids=coupler.coupler_bay.from_busbar_disconnector_ids,
                direct_busbar_grid_model_ids=coupler.coupler_bay.from_busbar_ids,
            ),
            other_busbar_id=to_busbar_grid_model_id,
        )
    if from_busbar_grid_model_id == to_busbar_grid_model_id and not to_side_resolved:
        to_busbar_grid_model_id = _pick_distinct_coupler_busbar_id(
            candidate_busbar_ids=_get_unresolved_coupler_side_candidates(
                selector_switch_ids=coupler.coupler_bay.to_busbar_disconnector_ids,
                direct_busbar_grid_model_ids=coupler.coupler_bay.to_busbar_ids,
            ),
            other_busbar_id=from_busbar_grid_model_id,
        )

    if from_busbar_grid_model_id is not None:
        try:
            update["busbar_from_id"] = busbar_int_id_by_grid_model_id[from_busbar_grid_model_id]
        except KeyError as error:
            raise ValueError(
                f"Coupler {coupler.grid_model_id} references unknown from-side busbar {from_busbar_grid_model_id}"
            ) from error

    if to_busbar_grid_model_id is not None:
        try:
            update["busbar_to_id"] = busbar_int_id_by_grid_model_id[to_busbar_grid_model_id]
        except KeyError as error:
            raise ValueError(
                f"Coupler {coupler.grid_model_id} references unknown to-side busbar {to_busbar_grid_model_id}"
            ) from error

    if (
        from_busbar_grid_model_id is None
        or to_busbar_grid_model_id is None
        or not from_side_resolved
        or not to_side_resolved
        or from_busbar_grid_model_id == to_busbar_grid_model_id
    ):
        open_state = True

    update["open"] = open_state

    return RuntimeBusbarCoupler(
        **{k: v for k, v in coupler_data.items() if k not in update},
        **update,
    )


def materialize_runtime_bus_group_from_runtime_state(
    canonical_bus_group: MasterBusGroup,
    branch_asset_map: dict[str, BranchAsset],
    injection_asset_map: dict[str, InjectionAsset],
    asset_bay_map: dict[str, AssetBay],
    runtime_switching_state: RuntimeSwitchingState,
    *,
    model_log: Optional[list[str]] = None,
) -> RuntimeBusGroup:
    """Materialize one master bus group from compact runtime switching state.

    Parameters
    ----------
    canonical_bus_group : MasterBusGroup
        Canonical bus-group definition.
    branch_asset_map : dict[str, BranchAsset]
        Canonical branch assets keyed by grid-model id.
    injection_asset_map : dict[str, InjectionAsset]
        Canonical injection assets keyed by grid-model id.
    asset_bay_map : dict[str, AssetBay]
        Canonical asset bays keyed by asset-bay id.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay describing the current live state.
    model_log : Optional[list[str]], optional
        Optional log messages to attach to the materialized bus group.

    Returns
    -------
    RuntimeBusGroup
        Runtime bus-group snapshot combining canonical structure and live switching state.
    """
    materialized_busbars = [
        RuntimeBusbar(
            **busbar.model_dump(),
            in_service=busbar.grid_model_id not in runtime_switching_state.busbar_out_of_service_ids,
            bus_branch_bus_id=(
                runtime_switching_state.busbar_bus_branch_bus_ids.get(busbar.grid_model_id)
                if runtime_switching_state.busbar_bus_branch_bus_ids is not None
                else None
            ),
        )
        for busbar in canonical_bus_group.busbars
    ]
    busbar_int_id_by_grid_model_id = {busbar.grid_model_id: busbar.int_id for busbar in canonical_bus_group.busbars}

    return RuntimeBusGroup(
        bus_group_id=canonical_bus_group.bus_group_id,
        voltage_level_id=canonical_bus_group.voltage_level_id,
        name=canonical_bus_group.name,
        station_type=canonical_bus_group.station_type,
        region=canonical_bus_group.region,
        voltage_level=canonical_bus_group.voltage_level,
        busbars=materialized_busbars,
        couplers=[
            _materialize_runtime_coupler(
                coupler=coupler,
                busbar_int_id_by_grid_model_id=busbar_int_id_by_grid_model_id,
                runtime_switching_state=runtime_switching_state,
            )
            for coupler in canonical_bus_group.couplers
        ],
        branch_connections=[
            RuntimeAssetConnection(
                asset=RuntimeBranchAsset.model_validate(
                    _copy_runtime_or_raise(branch_asset_map, asset_connection.asset_id, "branch asset").model_dump()
                ),
                branch_end=asset_connection.branch_end,
                asset_bay=(
                    _copy_runtime_or_raise(asset_bay_map, asset_connection.asset_bay_id, "asset bay").model_copy(deep=True)
                    if asset_connection.asset_bay_id is not None
                    else None
                ),
            )
            for asset_connection in canonical_bus_group.branch_connections
        ],
        injection_connections=[
            RuntimeAssetConnection(
                asset=RuntimeInjectionAsset.model_validate(
                    _copy_runtime_or_raise(
                        injection_asset_map,
                        asset_connection.asset_id,
                        "injection asset",
                    ).model_dump()
                ),
                branch_end=asset_connection.branch_end,
                asset_bay=(
                    _copy_runtime_or_raise(asset_bay_map, asset_connection.asset_bay_id, "asset bay").model_copy(deep=True)
                    if asset_connection.asset_bay_id is not None
                    else None
                ),
            )
            for asset_connection in canonical_bus_group.injection_connections
        ],
        branch_switching_table=_build_switching_table_from_compact_runtime(
            master_bus_group=canonical_bus_group,
            asset_connections=canonical_bus_group.branch_connections,
            asset_connectivity=(
                np.asarray(canonical_bus_group.branch_connectivity, dtype=bool)
                if canonical_bus_group.branch_connectivity is not None
                else None
            ),
            asset_bay_map=asset_bay_map,
            runtime_switching_state=runtime_switching_state,
            asset_kind="branch",
        ),
        injection_switching_table=_build_switching_table_from_compact_runtime(
            master_bus_group=canonical_bus_group,
            asset_connections=canonical_bus_group.injection_connections,
            asset_connectivity=(
                np.asarray(canonical_bus_group.injection_connectivity, dtype=bool)
                if canonical_bus_group.injection_connectivity is not None
                else None
            ),
            asset_bay_map=asset_bay_map,
            runtime_switching_state=runtime_switching_state,
            asset_kind="injection",
        ),
        branch_connectivity=(
            np.asarray(canonical_bus_group.branch_connectivity, dtype=bool)
            if canonical_bus_group.branch_connectivity is not None
            else None
        ),
        injection_connectivity=(
            np.asarray(canonical_bus_group.injection_connectivity, dtype=bool)
            if canonical_bus_group.injection_connectivity is not None
            else None
        ),
        model_log=list(model_log) if model_log is not None else None,
    )
