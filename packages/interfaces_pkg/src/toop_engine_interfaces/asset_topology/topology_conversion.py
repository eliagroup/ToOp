# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Conversions between materialized stations and canonical master data."""

from dataclasses import dataclass

import numpy as np
import structlog
from beartype.typing import Optional
from toop_engine_interfaces.asset_topology.asset_topology import MasterStation, TopologyMasterData
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    BusbarCoupler,
    InjectionAsset,
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.materialized_topology import (
    MaterializedAssetConnection,
    MaterializedStation,
    StationAssetConnection,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RuntimeSwitchingState:
    """Compact runtime overlay describing the current live switching state of a station."""

    busbar_bus_branch_bus_ids: Optional[dict[str, str]] = None
    """Mapping from canonical busbar ids to current runtime bus-branch bus ids."""

    branch_current_bus_ids: Optional[list[str | None]] = None
    """Current runtime bus ids for branch connections, aligned with the station's
    canonical branch connections."""

    injection_current_bus_ids: Optional[list[str | None]] = None
    """Current runtime bus ids for injection connections, aligned with the station's
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


def validate_complete_master_data(master_data: TopologyMasterData) -> None:
    """Validate that canonical master data already contains productive asset metadata.

    Parameters
    ----------
    master_data : TopologyMasterData
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

    for station in master_data.stations:
        for asset_connection in station.branch_connections:
            if asset_connection.branch_end is None:
                raise ValueError(
                    f"Branch asset {asset_connection.asset_id} in station {station.bus_group_id} "
                    "is missing canonical branch_end in master data"
                )


def _build_switching_table_from_compact_runtime(
    station: MasterStation,
    asset_connections: list[StationAssetConnection],
    asset_connectivity: Optional[np.ndarray],
    asset_bay_map: dict[str, AssetBay],
    runtime_switching_state: RuntimeSwitchingState,
    asset_kind: str,
) -> np.ndarray:
    """Rebuild one station switching table directly from compact runtime state.

    Parameters
    ----------
    station : MasterStation
        Canonical station whose runtime switching table should be reconstructed.
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
    switching_table = np.zeros((len(station.busbars), len(asset_connections)), dtype=bool)
    busbar_index_by_id = {busbar.grid_model_id: index for index, busbar in enumerate(station.busbars)}
    asset_current_bus_ids = (
        runtime_switching_state.branch_current_bus_ids
        if asset_kind == "branch"
        else runtime_switching_state.injection_current_bus_ids
    )

    for asset_index, asset_connection in enumerate(asset_connections):
        if asset_connection.asset_bay_id is None:
            _assign_switching_from_connectivity(
                switching_table=switching_table,
                station=station,
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
            station=station,
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
                    f"{asset_connection.asset_id} in station {station.bus_group_id}"
                )

    return switching_table


def _assign_switching_from_connectivity(
    switching_table: np.ndarray,
    station: MasterStation,
    asset_connection: StationAssetConnection,
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
    station : MasterStation
        Canonical station owning the switching table.
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
            f"in station {station.bus_group_id}"
        )

    current_bus_id = asset_current_bus_ids[asset_index] if asset_current_bus_ids is not None else None
    matching_busbar_indices = _get_matching_busbar_indices(
        station=station,
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
        f"Cannot compactly realize {asset_kind} asset {asset_connection.asset_id} in station "
        f"{station.bus_group_id} without asset bay mapping"
    )


def _get_matching_busbar_indices(
    station: MasterStation,
    asset_connectivity: np.ndarray,
    runtime_switching_state: RuntimeSwitchingState,
    current_bus_id: str | None,
    asset_index: int,
) -> list[int]:
    """Return busbar indices matching the current runtime bus id for one asset.

    Parameters
    ----------
    station : MasterStation
        Canonical station owning the busbars.
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
    for busbar_index, busbar in enumerate(station.busbars):
        mapped_bus_id = runtime_switching_state.busbar_bus_branch_bus_ids.get(busbar.grid_model_id)
        matches_current_bus_id = current_bus_id in {mapped_bus_id, busbar.grid_model_id}
        if matches_current_bus_id and asset_connectivity[busbar_index, asset_index]:
            matching_busbar_indices.append(busbar_index)
    return matching_busbar_indices


def _assign_switching_from_asset_bay(
    switching_table: np.ndarray,
    station: MasterStation,
    asset_connection: StationAssetConnection,
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
    station : MasterStation
        Canonical station owning the switching table.
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
    if asset_bay.dv_switch_grid_model_id in runtime_switching_state.open_switch_ids:
        return

    for busbar_id, switch_id in asset_bay.sr_switch_grid_model_id.items():
        if switch_id in runtime_switching_state.open_switch_ids:
            continue
        try:
            busbar_index = busbar_index_by_id[busbar_id]
        except KeyError as error:
            raise ValueError(
                f"Asset bay {asset_bay.asset_bay_id} references unknown busbar {busbar_id} in station {station.bus_group_id}"
            ) from error
        switching_table[busbar_index, asset_index] = True


def _resolve_runtime_coupler_busbar_id(
    selector_switch_ids: dict[str, str],
    runtime_switching_state: RuntimeSwitchingState,
) -> str | None:
    """Return the uniquely connected canonical busbar id for one coupler side."""
    connected_busbar_ids = [
        busbar_id
        for busbar_id, switch_id in selector_switch_ids.items()
        if switch_id not in runtime_switching_state.open_switch_ids
    ]
    if len(connected_busbar_ids) == 1:
        return connected_busbar_ids[0]
    return None


def _get_canonical_coupler_busbar_grid_model_id(
    coupler_busbar_int_id: int,
    busbar_grid_model_id_by_int_id: dict[int, str],
    side: str,
    coupler_id: str,
) -> str:
    """Return the canonical busbar grid-model id for a fixed coupler side."""
    try:
        return busbar_grid_model_id_by_int_id[coupler_busbar_int_id]
    except KeyError as error:
        raise ValueError(
            f"Coupler {coupler_id} references unknown canonical {side}-side busbar int id {coupler_busbar_int_id}"
        ) from error


def _resolve_runtime_or_canonical_coupler_busbar_id(
    selector_switch_ids: dict[str, str],
    runtime_switching_state: RuntimeSwitchingState,
    coupler_busbar_int_id: int,
    busbar_grid_model_id_by_int_id: dict[int, str],
    side: str,
    coupler_id: str,
) -> str | None:
    """Resolve one coupler side from runtime selectors and fall back to canonical endpoint if still connected."""
    if not selector_switch_ids:
        return _get_canonical_coupler_busbar_grid_model_id(
            coupler_busbar_int_id=coupler_busbar_int_id,
            busbar_grid_model_id_by_int_id=busbar_grid_model_id_by_int_id,
            side=side,
            coupler_id=coupler_id,
        )

    resolved_busbar_grid_model_id = _resolve_runtime_coupler_busbar_id(
        selector_switch_ids=selector_switch_ids,
        runtime_switching_state=runtime_switching_state,
    )
    if resolved_busbar_grid_model_id is not None:
        if resolved_busbar_grid_model_id in runtime_switching_state.busbar_out_of_service_ids:
            return None
        return resolved_busbar_grid_model_id

    canonical_busbar_grid_model_id = _get_canonical_coupler_busbar_grid_model_id(
        coupler_busbar_int_id=coupler_busbar_int_id,
        busbar_grid_model_id_by_int_id=busbar_grid_model_id_by_int_id,
        side=side,
        coupler_id=coupler_id,
    )

    if any(switch_id not in runtime_switching_state.open_switch_ids for switch_id in selector_switch_ids.values()):
        if canonical_busbar_grid_model_id in runtime_switching_state.busbar_out_of_service_ids:
            return None
        return canonical_busbar_grid_model_id
    return None


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
    busbar_grid_model_id_by_int_id = {
        int_id: grid_model_id for grid_model_id, int_id in busbar_int_id_by_grid_model_id.items()
    }

    if coupler.coupler_bay is not None:
        if coupler.coupler_bay.connection_kind == "disconnector":
            if coupler.coupler_bay.dv_switch_grid_model_id in runtime_switching_state.open_switch_ids:
                open_state = True
            update["open"] = open_state
            return RuntimeBusbarCoupler(
                **{k: v for k, v in coupler_data.items() if k not in update},
                **update,
            )

        if coupler.coupler_bay.dv_switch_grid_model_id in runtime_switching_state.open_switch_ids:
            open_state = True

        from_busbar_grid_model_id = _resolve_runtime_or_canonical_coupler_busbar_id(
            selector_switch_ids=coupler.coupler_bay.from_sr_switch_grid_model_id,
            runtime_switching_state=runtime_switching_state,
            coupler_busbar_int_id=coupler.busbar_from_id,
            busbar_grid_model_id_by_int_id=busbar_grid_model_id_by_int_id,
            side="from",
            coupler_id=coupler.grid_model_id,
        )

        to_busbar_grid_model_id = _resolve_runtime_or_canonical_coupler_busbar_id(
            selector_switch_ids=coupler.coupler_bay.to_sr_switch_grid_model_id,
            runtime_switching_state=runtime_switching_state,
            coupler_busbar_int_id=coupler.busbar_to_id,
            busbar_grid_model_id_by_int_id=busbar_grid_model_id_by_int_id,
            side="to",
            coupler_id=coupler.grid_model_id,
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
            or from_busbar_grid_model_id == to_busbar_grid_model_id
        ):
            open_state = True

        update["open"] = open_state

    return RuntimeBusbarCoupler(
        **{k: v for k, v in coupler_data.items() if k not in update},
        **update,
    )


def materialize_station_from_runtime_state(
    station: MasterStation,
    branch_asset_map: dict[str, BranchAsset],
    injection_asset_map: dict[str, InjectionAsset],
    asset_bay_map: dict[str, AssetBay],
    runtime_switching_state: RuntimeSwitchingState,
    *,
    model_log: Optional[list[str]] = None,
) -> MaterializedStation:
    """Materialize one master station from compact runtime switching state.

    Parameters
    ----------
    station : MasterStation
        Canonical station definition.
    branch_asset_map : dict[str, BranchAsset]
        Canonical branch assets keyed by grid-model id.
    injection_asset_map : dict[str, InjectionAsset]
        Canonical injection assets keyed by grid-model id.
    asset_bay_map : dict[str, AssetBay]
        Canonical asset bays keyed by asset-bay id.
    runtime_switching_state : RuntimeSwitchingState
        Compact runtime overlay describing the current live state.
    model_log : Optional[list[str]], optional
        Optional log messages to attach to the materialized station.

    Returns
    -------
    MaterializedStation
        Runtime station snapshot combining canonical structure and live switching state.
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
        for busbar in station.busbars
    ]
    busbar_int_id_by_grid_model_id = {busbar.grid_model_id: busbar.int_id for busbar in station.busbars}

    return MaterializedStation(
        bus_group_id=station.bus_group_id,
        voltage_level_id=station.voltage_level_id,
        name=station.name,
        station_type=station.station_type,
        region=station.region,
        voltage_level=station.voltage_level,
        busbars=materialized_busbars,
        couplers=[
            _materialize_runtime_coupler(
                coupler=coupler,
                busbar_int_id_by_grid_model_id=busbar_int_id_by_grid_model_id,
                runtime_switching_state=runtime_switching_state,
            )
            for coupler in station.couplers
        ],
        branch_connections=[
            MaterializedAssetConnection(
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
            for asset_connection in station.branch_connections
        ],
        injection_connections=[
            MaterializedAssetConnection(
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
            for asset_connection in station.injection_connections
        ],
        branch_switching_table=_build_switching_table_from_compact_runtime(
            station=station,
            asset_connections=station.branch_connections,
            asset_connectivity=(
                np.asarray(station.branch_connectivity, dtype=bool) if station.branch_connectivity is not None else None
            ),
            asset_bay_map=asset_bay_map,
            runtime_switching_state=runtime_switching_state,
            asset_kind="branch",
        ),
        injection_switching_table=_build_switching_table_from_compact_runtime(
            station=station,
            asset_connections=station.injection_connections,
            asset_connectivity=(
                np.asarray(station.injection_connectivity, dtype=bool)
                if station.injection_connectivity is not None
                else None
            ),
            asset_bay_map=asset_bay_map,
            runtime_switching_state=runtime_switching_state,
            asset_kind="injection",
        ),
        branch_connectivity=(
            np.asarray(station.branch_connectivity, dtype=bool) if station.branch_connectivity is not None else None
        ),
        injection_connectivity=(
            np.asarray(station.injection_connectivity, dtype=bool) if station.injection_connectivity is not None else None
        ),
        model_log=list(model_log) if model_log is not None else None,
    )
