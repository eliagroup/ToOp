# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""High-level switch update export entrypoints."""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import cast
from pypowsybl.network import Network
from toop_engine_dc_solver.export.disconnection_switch_updates import get_changing_switches_from_disconnections
from toop_engine_dc_solver.export.station_switch_updates import get_changing_switches_from_changed_stations
from toop_engine_interfaces.asset_topology import BusbarCoupler, Station, Topology
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema

logger = structlog.get_logger(__name__)


@pa.check_types
def get_coupler_states_from_busbar_couplers(station_couplers: list[BusbarCoupler]) -> pat.DataFrame[SwitchUpdateSchema]:
    """Translate the coupler states to the switch update schema format."""
    switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    for coupler in station_couplers:
        if not coupler.in_service:
            raise ValueError(f"Coupler {coupler.grid_model_id} is not in service, undefined behavior")
        switch_df.loc[switch_df.shape[0]] = {
            "grid_model_id": coupler.grid_model_id,
            "open": coupler.open,
        }
    switch_df = switch_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], switch_df)


def get_asset_bay_grid_model_id_list(station: Station) -> list[dict[str, str] | None]:
    """Get selector switch ids for each station asset."""
    asset_bays = [asset.asset_bay for asset in station.assets]
    sr_switch_grid_model_id_list: list[dict[str, str] | None] = []
    for asset_bay in asset_bays:
        if asset_bay is None or asset_bay.sr_switch_grid_model_id is None:
            sr_switch_grid_model_id_list.append(None)
        else:
            sr_switch_grid_model_id_list.append(asset_bay.sr_switch_grid_model_id)
    return sr_switch_grid_model_id_list


def get_busbar_lookup(station: Station) -> dict[int, str]:
    """Get the busbar lookup for the given station."""
    return {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}


@pa.check_types
def get_asset_switch_states_from_station(
    station: Station,
) -> tuple[pat.DataFrame[SwitchUpdateSchema], pat.DataFrame[SwitchUpdateSchema]]:
    """Translate asset selector and breaker states of one station to switch updates."""
    switch_reassignment_list: list[dict[str, str | bool]] = []
    switch_disconnection_list: list[dict[str, str | bool]] = []
    busbar_id_dict = get_busbar_lookup(station)
    switching_table = np.asarray(station.asset_switching_table, dtype=bool)
    asset_reassignment_list = get_asset_bay_grid_model_id_list(station)
    assert switching_table.shape[1] == len(asset_reassignment_list), (
        "The asset switching table has a different number of columns than the asset reassignment list. "
        f"Columns: {switching_table.shape[1]}, Reassignment list: {len(asset_reassignment_list)}"
    )

    for column in range(switching_table.shape[1]):
        asset_switch_ids = asset_reassignment_list[column]
        if asset_switch_ids is None:
            continue
        asset_switch_states = switching_table[:, column]
        active_busbars = int(asset_switch_states.sum())
        if active_busbars == 1:
            assigned_busbar = int(np.nonzero(asset_switch_states)[0][0])
            for busbar, switch_id in asset_switch_ids.items():
                switch_reassignment_list.append(
                    {
                        "grid_model_id": switch_id,
                        "open": busbar_id_dict[assigned_busbar] != busbar,
                    }
                )
        elif active_busbars == 0:
            asset_bay = station.assets[column].asset_bay
            assert asset_bay is not None
            switch_disconnection_list.append(
                {
                    "grid_model_id": asset_bay.dv_switch_grid_model_id,
                    "open": True,
                }
            )
        else:
            raise ValueError(f"Switching table column {column} has more than one True value: {switching_table[:, column]}")

    switch_reassignment_df = pd.DataFrame.from_records(switch_reassignment_list, columns=["grid_model_id", "open"])
    switch_disconnection_df = pd.DataFrame.from_records(switch_disconnection_list, columns=["grid_model_id", "open"])
    if switch_reassignment_df.empty:
        switch_reassignment_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    if switch_disconnection_df.empty:
        switch_disconnection_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    switch_reassignment_df = switch_reassignment_df.astype({"grid_model_id": str, "open": bool})
    switch_disconnection_df = switch_disconnection_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], switch_reassignment_df), cast(
        pat.DataFrame[SwitchUpdateSchema], switch_disconnection_df
    )


@pa.check_types
def get_switch_update_schema_from_topology(topology: Topology) -> pat.DataFrame[SwitchUpdateSchema]:
    """Translate a target topology to the full switch update schema without diffing."""
    switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)

    for station in topology.stations:
        coupler_df = get_coupler_states_from_busbar_couplers(station.couplers)
        switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
        switch_df_update = pd.concat([coupler_df, switch_reassignment_df, switch_disconnection_df], ignore_index=True)
        switch_df = pd.concat([switch_df, switch_df_update], ignore_index=True)

    if switch_df.duplicated(subset=["grid_model_id"]).any():
        logger.warning(
            "Duplicate switch ids found in the switch update schema",
            duplicate_switch_ids=switch_df.loc[switch_df.duplicated(subset=["grid_model_id"]), "grid_model_id"].to_list(),
        )
        switch_df = switch_df.drop_duplicates(subset=["grid_model_id"])
    switch_df = switch_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], switch_df)


@pa.check_types
def get_diff_switch_states(
    network: Network,
    switch_df: pat.DataFrame[SwitchUpdateSchema],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Filter switch updates down to the ones that differ from the current network state."""
    diff_switch_df = switch_df.merge(
        network.get_switches(attributes=["open"]),
        left_on="grid_model_id",
        right_index=True,
        how="left",
        suffixes=("", "_network"),
    )
    if diff_switch_df["open_network"].isna().any():
        raise ValueError(
            "Switch id not found in the networkSwitch id: "
            f"{diff_switch_df.loc[diff_switch_df['open_network'].isna(), 'grid_model_id']}"
        )
    diff_switch_df = diff_switch_df[diff_switch_df["open"] != diff_switch_df["open_network"]]
    diff_switch_df = diff_switch_df[["grid_model_id", "open"]]
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], diff_switch_df)


@pa.check_types
def get_changing_switches_from_topology(network: Network, target_topology: Topology) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get the switch updates needed to realize a target topology on a node-breaker network."""
    switch_update_df = get_switch_update_schema_from_topology(topology=target_topology)
    return get_diff_switch_states(network=network, switch_df=switch_update_df)


@pa.check_types
def get_changing_switches_from_actions(
    changed_stations: list[Station],
    starting_topology: Topology,
    disconnections: list[GridElement] | None = None,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates for changed stations and explicit disconnections.

    This is the composed export entrypoint for switch updates derived from the simplified action
    representation. Changed stations contribute action-driven switch updates, while explicit
    disconnections are translated into breaker-open updates when they can be represented through
    switchable stations in the provided topology.

    Parameters
    ----------
    changed_stations : list[Station]
        Stations describing the target state for switchable substations.
    starting_topology : Topology
        Simplified starting topology used as reference for station ordering and switch layout.
    disconnections : list[GridElement] | None, optional
        Explicit branch disconnections requested for the target state.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing both station actions and representable disconnections.
    """
    action_switch_updates = get_changing_switches_from_changed_stations(
        changed_stations=changed_stations,
        starting_topology=starting_topology,
    )
    disconnection_switch_updates = get_changing_switches_from_disconnections(
        starting_topology=starting_topology,
        disconnections=disconnections,
    )

    overlapping_switch_ids = sorted(
        set(action_switch_updates["grid_model_id"]).intersection(disconnection_switch_updates["grid_model_id"])
    )
    if overlapping_switch_ids:
        logger.warning(
            "Action and disconnection switch updates overlap. Disconnection updates take precedence.",
            overlapping_switch_ids=overlapping_switch_ids,
        )
        action_switch_updates = action_switch_updates.loc[
            ~action_switch_updates["grid_model_id"].isin(overlapping_switch_ids)
        ]

    combined_switch_updates = pd.concat(
        [action_switch_updates, disconnection_switch_updates],
        ignore_index=True,
    )
    if combined_switch_updates.empty:
        combined_switch_updates = get_empty_dataframe_from_model(SwitchUpdateSchema)
    combined_switch_updates = combined_switch_updates.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], combined_switch_updates)
