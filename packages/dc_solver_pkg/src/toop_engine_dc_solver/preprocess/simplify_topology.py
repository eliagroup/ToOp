# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for building simplified runtime topology views during preprocessing."""

from dataclasses import replace

import numpy as np
import structlog
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess_switching import StationProblems, prepare_for_separation_set
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeBusGroup,
)
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import (
    SimplifiedAssetTopology,
    SimplifiedBusGroup,
)

logger = structlog.get_logger(__name__)


def _get_local_busbar_keep_indices(station: RuntimeBusGroup, node_id: str | None) -> list[int]:
    """Return busbar indices for the requested electrical node slice."""
    if node_id is None or node_id not in station.bus_branch_bus_ids or len(station.bus_branch_bus_ids) <= 1:
        return list(range(len(station.busbars)))

    return [index for index, busbar in enumerate(station.busbars) if busbar.bus_branch_bus_id == node_id]


def _select_locally_switched_asset_indices(
    asset_connections: list[RuntimeAssetConnection],
    allowed_asset_ids: set[str],
    switching_rows: np.ndarray,
) -> list[int]:
    """Keep the first locally switched occurrence of each requested asset id."""
    selected_indices: list[int] = []
    seen_asset_ids: set[str] = set()

    for index, connection in enumerate(asset_connections):
        asset_id = connection.asset.grid_model_id
        if asset_id not in allowed_asset_ids or asset_id in seen_asset_ids:
            continue
        if not np.any(switching_rows[:, index]):
            continue
        selected_indices.append(index)
        seen_asset_ids.add(asset_id)

    return selected_indices


def _filter_asset_bay_to_kept_busbars(
    asset_connection: RuntimeAssetConnection,
    kept_busbar_grid_model_ids: set[str],
) -> RuntimeAssetConnection:
    """Drop asset-bay SR references to busbars that were removed from the slice."""
    if asset_connection.asset_bay is None:
        return asset_connection

    return asset_connection.model_copy(
        update={
            "asset_bay": asset_connection.asset_bay.model_copy(
                update={
                    "busbar_disconnector_grid_model_id": {
                        busbar_id: switch_id
                        for busbar_id, switch_id in asset_connection.asset_bay.busbar_disconnector_grid_model_id.items()
                        if busbar_id in kept_busbar_grid_model_ids
                    }
                }
            )
        }
    )


def _slice_optional_matrix(
    matrix: np.ndarray | None,
    row_indices: list[int],
    column_indices: list[int],
) -> np.ndarray | None:
    """Slice an optional switching/connectivity matrix if present."""
    if matrix is None:
        return None
    return matrix[np.ix_(row_indices, column_indices)]


def _project_station_to_local_assets(
    station: RuntimeBusGroup,
    branch_ids: list[str],
    injection_ids: list[str],
    node_id: str | None = None,
) -> RuntimeBusGroup:
    """Restrict a runtime station to the assets represented by one relevant node slice."""
    busbar_keep_indices = _get_local_busbar_keep_indices(station, node_id)
    kept_busbars = [station.busbars[index] for index in busbar_keep_indices]
    kept_busbar_int_ids = {busbar.int_id for busbar in kept_busbars}
    kept_busbar_grid_model_ids = {busbar.grid_model_id for busbar in kept_busbars}

    branch_switching_rows = station.branch_switching_table[busbar_keep_indices, :]
    injection_switching_rows = station.injection_switching_table[busbar_keep_indices, :]
    branch_keep_indices = _select_locally_switched_asset_indices(
        asset_connections=station.branch_connections,
        allowed_asset_ids=set(branch_ids),
        switching_rows=branch_switching_rows,
    )
    injection_keep_indices = _select_locally_switched_asset_indices(
        asset_connections=station.injection_connections,
        allowed_asset_ids=set(injection_ids),
        switching_rows=injection_switching_rows,
    )

    return station.model_copy(
        update={
            "bus_branch_bus_ids": [node_id] if node_id is not None else station.bus_branch_bus_ids,
            "busbars": kept_busbars,
            "couplers": [
                coupler
                for coupler in station.couplers
                if coupler.busbar_from_id in kept_busbar_int_ids and coupler.busbar_to_id in kept_busbar_int_ids
            ],
            "branch_connections": [
                _filter_asset_bay_to_kept_busbars(station.branch_connections[index], kept_busbar_grid_model_ids)
                for index in branch_keep_indices
            ],
            "injection_connections": [
                _filter_asset_bay_to_kept_busbars(station.injection_connections[index], kept_busbar_grid_model_ids)
                for index in injection_keep_indices
            ],
            "branch_switching_table": station.branch_switching_table[np.ix_(busbar_keep_indices, branch_keep_indices)],
            "injection_switching_table": station.injection_switching_table[
                np.ix_(busbar_keep_indices, injection_keep_indices)
            ],
            "branch_connectivity": _slice_optional_matrix(
                station.branch_connectivity,
                busbar_keep_indices,
                branch_keep_indices,
            ),
            "injection_connectivity": _slice_optional_matrix(
                station.injection_connectivity,
                busbar_keep_indices,
                injection_keep_indices,
            ),
        }
    )


def _build_simplified_topology(
    stations: list[SimplifiedBusGroup],
    circuit_groups: list | None,
) -> SimplifiedAssetTopology:
    """Wrap simplified runtime stations in the explicit simplified topology subtype."""
    return SimplifiedAssetTopology(stations=stations, circuit_groups=circuit_groups)


def _simplify_station_slice(
    station: RuntimeBusGroup,
    branch_ids: list[str],
    injection_ids: list[str],
    close_couplers: bool,
    node_id: str | None = None,
) -> tuple[SimplifiedBusGroup, StationProblems]:
    """Simplify one projected runtime station slice."""
    electrical_bus_station = _project_station_to_local_assets(
        station=station,
        branch_ids=branch_ids,
        injection_ids=injection_ids,
        node_id=node_id,
    )
    simplified_station, problems = prepare_for_separation_set(
        station=electrical_bus_station,
        branch_ids=branch_ids,
        injection_ids=injection_ids,
        close_couplers=close_couplers,
    )
    return simplified_station, problems


def simplify_asset_topology_for_bb_outages(network_data: NetworkData, close_couplers: bool = False) -> NetworkData:
    """Simplify all stations needed by busbar-outage preprocessing."""
    assert network_data.asset_topology is not None, "Missing runtime asset-topology stations"

    if network_data.busbar_outage_map is None:
        return network_data

    runtime_stations_by_id = {station.bus_group_id: station for station in network_data.asset_topology.stations}
    simplified_stations_by_id: dict[str, SimplifiedBusGroup] = {
        station.bus_group_id: station
        for station in (network_data.simplified_asset_topology.stations if network_data.simplified_asset_topology else [])
    }
    updated_busbar_outage_map: dict[str, list[str]] = {}

    for station_id, configured_busbar_ids in network_data.busbar_outage_map.items():
        simplified_station = simplified_stations_by_id.get(station_id)
        if simplified_station is None:
            runtime_station = runtime_stations_by_id.get(station_id)
            if runtime_station is None:
                logger.warning(
                    "Skipping busbar-outage simplification for unknown station.",
                    station_id=station_id,
                )
                continue

            branch_ids = [connection.asset.grid_model_id for connection in runtime_station.branch_connections]
            injection_ids = [connection.asset.grid_model_id for connection in runtime_station.injection_connections]
            simplified_station, _problems = _simplify_station_slice(
                station=runtime_station,
                branch_ids=branch_ids,
                injection_ids=injection_ids,
                close_couplers=close_couplers,
            )
            simplified_stations_by_id[station_id] = simplified_station

        simplified_busbar_ids = {busbar.grid_model_id for busbar in simplified_station.busbars}
        updated_busbar_outage_map[station_id] = [
            busbar_id for busbar_id in configured_busbar_ids if busbar_id in simplified_busbar_ids
        ]

    return replace(
        network_data,
        simplified_bb_outage_topology=_build_simplified_topology(
            stations=list(simplified_stations_by_id.values()),
            circuit_groups=network_data.asset_topology.circuit_groups,
        ),
        busbar_outage_map=updated_busbar_outage_map,
    )
