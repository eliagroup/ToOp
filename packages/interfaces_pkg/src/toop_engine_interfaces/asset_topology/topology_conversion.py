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
    InjectionAsset,
)
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import StationAssetConnection

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RuntimeSwitchingState:
    """Compact runtime overlay inputs for materializing one station.

    Attributes
    ----------
    busbar_bus_branch_bus_ids : Optional[dict[str, str]]
        Mapping from canonical busbar ids to current runtime bus-branch bus ids.
    branch_current_bus_ids : Optional[list[str | None]]
        Current runtime bus ids for branch connections, aligned with the station's
        canonical branch connections.
    injection_current_bus_ids : Optional[list[str | None]]
        Current runtime bus ids for injection connections, aligned with the station's
        canonical injection connections.
    busbar_out_of_service_ids : set[str]
        Canonical busbar ids that are currently out of service.
    open_coupler_ids : set[str]
        Canonical coupler ids that are currently open.
    out_of_service_coupler_ids : set[str]
        Canonical coupler ids that are currently out of service.
    open_switch_ids : set[str]
        Asset-bay switch ids that are currently open.
    """

    busbar_bus_branch_bus_ids: Optional[dict[str, str]] = None
    branch_current_bus_ids: Optional[list[str | None]] = None
    injection_current_bus_ids: Optional[list[str | None]] = None
    busbar_out_of_service_ids: set[str] = frozenset()
    open_coupler_ids: set[str] = frozenset()
    out_of_service_coupler_ids: set[str] = frozenset()
    open_switch_ids: set[str] = frozenset()


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
    return MaterializedStation(
        bus_group_id=station.bus_group_id,
        voltage_level_id=station.voltage_level_id,
        name=station.name,
        station_type=station.station_type,
        region=station.region,
        voltage_level=station.voltage_level,
        busbars=[
            busbar.model_copy(
                update={
                    "in_service": busbar.grid_model_id not in runtime_switching_state.busbar_out_of_service_ids,
                    "bus_branch_bus_id": (
                        runtime_switching_state.busbar_bus_branch_bus_ids.get(busbar.grid_model_id)
                        if runtime_switching_state.busbar_bus_branch_bus_ids is not None
                        else None
                    ),
                },
                deep=True,
            )
            for busbar in station.busbars
        ],
        couplers=[
            coupler.model_copy(
                update={
                    "open": coupler.grid_model_id in runtime_switching_state.open_coupler_ids,
                    "in_service": coupler.grid_model_id not in runtime_switching_state.out_of_service_coupler_ids,
                },
                deep=True,
            )
            for coupler in station.couplers
        ],
        branch_connections=[
            MaterializedAssetConnection(
                asset=_copy_runtime_or_raise(branch_asset_map, asset_connection.asset_id, "branch asset").model_copy(
                    deep=True
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
                asset=_copy_runtime_or_raise(
                    injection_asset_map,
                    asset_connection.asset_id,
                    "injection asset",
                ).model_copy(deep=True),
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
