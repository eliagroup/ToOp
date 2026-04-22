# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers to represent explicit branch disconnections as switch updates."""

import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import cast
from toop_engine_interfaces.asset_topology import AssetBay, Station, Topology
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema

logger = structlog.get_logger(__name__)


def get_disconnected_asset_ids(
    stations: list[Station],
    disconnections: list[GridElement],
) -> dict[str, list[AssetBay]]:
    """Collect representable disconnection asset ids from the provided topology.

    Disconnections are represented via switch updates only when at least one switchable station in
    ``starting_topology`` contains the disconnected asset. Assets that are not present in the
    topology are skipped and logged as warnings instead of raising, because non-switchable terminal
    stations are not represented there.

    Parameters
    ----------
    stations : list[Station]
        Stations to be searched for the disconnections.
    disconnections : list[GridElement]
        Explicit branch disconnections requested for the target state.

    Returns
    -------
    dict[str, list[AssetBay]]
        Mapping of grid element IDs to be disconnected to the switchable assets that can perform the disconnection.
        Note that not all requested disconnections might be representable as switch updates, so this mapping might be
        incomplete.
    """
    disconnection_map: dict[str, GridElement] = {disconnection.id: disconnection for disconnection in disconnections}
    disconnection_asset_map: dict[str, list[AssetBay]] = {disconnection.id: [] for disconnection in disconnections}
    for station in stations:
        for asset in station.assets:
            if asset.grid_model_id in disconnection_map and asset.asset_bay is not None:
                corresponding_disconnection = disconnection_map[asset.grid_model_id]
                disconnection_asset_map[corresponding_disconnection.id].append(asset.asset_bay)
    return disconnection_asset_map


@pa.check_types
def get_changing_switches_from_disconnections(
    starting_topology: Topology,
    disconnections: list[GridElement],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates that represent explicit disconnections.

    Parameters
    ----------
    starting_topology : Topology
        Simplified starting topology containing the switchable stations available for export.
    disconnections : list[GridElement]
        Explicit branch disconnections requested for the target state.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing the requested disconnections where possible.
    """
    disconnection_asset_map: dict[str, list[AssetBay]] = get_disconnected_asset_ids(
        stations=starting_topology.stations,
        disconnections=disconnections,
    )

    # Open all dv-switches in all asset bays
    switch_updates = []
    for disconnection_id, assets in disconnection_asset_map.items():
        if not assets:
            disconnection_map: dict[str, GridElement] = {disconnection.id: disconnection for disconnection in disconnections}
            disconnection = disconnection_map[disconnection_id]

            logger.warning(
                "Disconnected asset cannot be represented as switch update due to missing corresponding switchable asset "
                "in the topology.",
                disconnection_id=disconnection.id,
                disconnection_name=disconnection.name,
                disconnection_type=disconnection.type,
                available_station_ids=[station.grid_model_id for station in starting_topology.stations],
            )
        for asset in assets:
            switch_updates.append(
                {
                    "grid_model_id": asset.dv_switch_grid_model_id,
                    "open": True,
                }
            )

    return cast(
        pat.DataFrame[SwitchUpdateSchema], pd.DataFrame.from_records(switch_updates, columns=["grid_model_id", "open"])
    )
