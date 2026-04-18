# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers to represent explicit branch disconnections as switch updates."""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import cast
from jaxtyping import Bool
from toop_engine_dc_solver.export.asset_topology_to_dgs import SwitchUpdateSchema
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import GridElement

logger = structlog.get_logger(__name__)


def get_disconnected_asset_ids(
    disconnections: list[GridElement] | None,
    starting_topology: Topology,
) -> set[str]:
    """Collect representable disconnection asset ids from the provided topology.

    Disconnections are represented via switch updates only when at least one switchable station in
    ``starting_topology`` contains the disconnected asset. Assets that are not present in the
    topology are skipped and logged as warnings instead of raising, because non-switchable terminal
    stations are not represented there.

    Parameters
    ----------
    disconnections : list[GridElement] | None
        Explicit branch disconnections requested for the target state.
    starting_topology : Topology
        Simplified starting topology containing only the switchable stations available for switch
        update generation.

    Returns
    -------
    set[str]
        Asset ids that can be represented as switch updates using the provided topology.
    """
    if disconnections is None:
        return set()

    known_asset_ids = {asset.grid_model_id for station in starting_topology.stations for asset in station.assets}
    disconnected_asset_ids: set[str] = set()
    for disconnection in disconnections:
        if disconnection.id not in known_asset_ids:
            logger.warning(
                "Disconnected asset cannot be represented because it is not present in the provided switchable topology.",
                disconnection_id=disconnection.id,
                disconnection_name=disconnection.name,
                disconnection_type=disconnection.type,
                available_station_ids=[station.grid_model_id for station in starting_topology.stations],
            )
            continue
        disconnected_asset_ids.add(disconnection.id)
    return disconnected_asset_ids


def get_station_ids_affected_by_disconnections(
    starting_topology: Topology,
    disconnected_asset_ids: set[str],
) -> set[str]:
    """Match representable disconnections to stations in the provided topology.

    Parameters
    ----------
    starting_topology : Topology
        Simplified starting topology containing switchable stations.
    disconnected_asset_ids : set[str]
        Asset ids that can be represented through switch updates.

    Returns
    -------
    set[str]
        Station ids whose assets include one of the disconnected asset ids.
    """
    affected_station_ids: set[str] = set()
    for station in starting_topology.stations:
        if any(asset.grid_model_id in disconnected_asset_ids for asset in station.assets):
            affected_station_ids.add(station.grid_model_id)
    return affected_station_ids


def apply_disconnections_to_station_switching_table(
    station: Station,
    switching_table: Bool[np.ndarray, " n_busbar n_asset"],
    disconnected_asset_ids: set[str],
) -> Bool[np.ndarray, " n_busbar n_asset"]:
    """Apply explicit disconnections by zeroing the affected asset columns.

    Parameters
    ----------
    station : Station
        Station whose asset ordering defines the column mapping of ``switching_table``.
    switching_table : Bool[np.ndarray, " n_busbar n_asset"]
        Switching table to modify.
    disconnected_asset_ids : set[str]
        Asset ids that should be treated as disconnected.

    Returns
    -------
    Bool[np.ndarray, " n_busbar n_asset"]
        Copy of the switching table where disconnected asset columns are set to ``False``.
    """
    effective_switching_table = switching_table.copy()
    for column, asset in enumerate(station.assets):
        if asset.grid_model_id in disconnected_asset_ids:
            effective_switching_table[:, column] = False
    return effective_switching_table


def _get_disconnection_switch_diffs(
    station: Station,
    disconnected_asset_ids: set[str],
) -> list[dict[str, str | bool]]:
    """Collect breaker-open updates for disconnected assets in one station.

    Parameters
    ----------
    station : Station
        Station whose assets should be checked for requested disconnections.
    disconnected_asset_ids : set[str]
        Asset ids that should be represented as explicit disconnections.

    Returns
    -------
    list[dict[str, str | bool]]
        Breaker switch updates required to disconnect the requested assets.
    """
    switching_table = np.asarray(station.asset_switching_table, dtype=bool)
    diff_switches: list[dict[str, str | bool]] = []
    for column, asset in enumerate(station.assets):
        if asset.grid_model_id not in disconnected_asset_ids:
            continue
        asset_bay = asset.asset_bay
        if asset_bay is None:
            continue
        switch_states = switching_table[:, column]
        active_busbars = int(switch_states.sum())
        if active_busbars > 1:
            raise ValueError(
                f"Starting station asset {asset.grid_model_id} has more than one active busbar: {switch_states}"
            )
        if active_busbars == 0:
            continue
        diff_switches.append({"grid_model_id": asset_bay.dv_switch_grid_model_id, "open": True})
    return diff_switches


@pa.check_types
def get_changing_switches_from_disconnections(
    starting_topology: Topology,
    disconnections: list[GridElement] | None = None,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates that represent explicit disconnections.

    Parameters
    ----------
    starting_topology : Topology
        Simplified starting topology containing the switchable stations available for export.
    disconnections : list[GridElement] | None, optional
        Explicit branch disconnections requested for the target state.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing the requested disconnections where possible.
    """
    disconnected_asset_ids = get_disconnected_asset_ids(
        disconnections=disconnections,
        starting_topology=starting_topology,
    )
    affected_station_ids = get_station_ids_affected_by_disconnections(
        starting_topology=starting_topology,
        disconnected_asset_ids=disconnected_asset_ids,
    )

    diff_switches: list[dict[str, str | bool]] = []
    for station in starting_topology.stations:
        if station.grid_model_id not in affected_station_ids:
            continue
        diff_switches.extend(_get_disconnection_switch_diffs(station=station, disconnected_asset_ids=disconnected_asset_ids))

    diff_switch_df = pd.DataFrame.from_records(diff_switches, columns=["grid_model_id", "open"])
    if diff_switch_df.empty:
        diff_switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    if diff_switch_df.duplicated(subset=["grid_model_id"]).any():
        logger.warning(
            "Duplicate switch ids found in disconnection switch updates.",
            duplicate_switch_ids=diff_switch_df.loc[
                diff_switch_df.duplicated(subset=["grid_model_id"]), "grid_model_id"
            ].to_list(),
        )
        diff_switch_df = diff_switch_df.drop_duplicates(subset=["grid_model_id"])
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], diff_switch_df)
