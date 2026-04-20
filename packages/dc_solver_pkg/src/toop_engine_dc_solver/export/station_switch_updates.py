# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers to create switch update data from changed stations.

This module complements the network-based helpers in ``asset_topology_to_dgs`` by deriving the
same switch update schema from changed stations and a starting topology.
"""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import cast
from jaxtyping import Bool
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def _get_busbar_lookup(station: Station) -> dict[int, str]:
    """Map busbar row indices in the switching table to busbar ids."""
    return {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}


def _resolve_changed_stations(
    changed_stations: list[Station],
    starting_topology: Topology,
) -> tuple[dict[str, Station], dict[str, Station], list[str]]:
    """Resolve station lookups and preserve changed-station ordering.

    This helper is intentionally limited to station actions. It validates that all changed stations
    are unique and present in the starting topology, then returns them in the same station order as
    the starting topology.

    Parameters
    ----------
    changed_stations : list[Station]
        Stations that contain topology changes relative to the starting topology.
    starting_topology : Topology
        Reference topology used to validate station identities and derive stable ordering.

    Returns
    -------
    tuple[dict[str, Station], dict[str, Station], list[str]]
        A tuple containing:
        1. a lookup for stations in the starting topology by ``grid_model_id``,
        2. a lookup for changed stations by ``grid_model_id``,
        3. the changed station ids in starting-topology order.

    Raises
    ------
    ValueError
        If ``changed_stations`` contains duplicate station ids or if a changed station is not
        present in the starting topology.
    """
    changed_station_ids = [station.grid_model_id for station in changed_stations]
    if len(changed_station_ids) != len(set(changed_station_ids)):
        raise ValueError("Changed stations must be unique by grid_model_id.")

    starting_station_lookup = {station.grid_model_id: station for station in starting_topology.stations}
    changed_station_lookup = {station.grid_model_id: station for station in changed_stations}
    missing_station_ids = set(changed_station_lookup).difference(starting_station_lookup)
    if missing_station_ids:
        raise ValueError(f"Changed stations not found in starting topology: {sorted(missing_station_ids)}")

    ordered_changed_station_ids = [
        station.grid_model_id for station in starting_topology.stations if station.grid_model_id in changed_station_lookup
    ]
    return starting_station_lookup, changed_station_lookup, ordered_changed_station_ids


def _get_coupler_switch_diffs(
    changed_station: Station,
    starting_station: Station,
) -> list[dict[str, str | bool]]:
    """Collect coupler switch changes between two station states.

    Parameters
    ----------
    changed_station : Station
        Station describing the target coupler states.
    starting_station : Station
        Station describing the reference coupler states.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records for couplers whose ``open`` state changes.

    Raises
    ------
    ValueError
        If the stations do not expose the same couplers.
    """
    if len(changed_station.couplers) != len(starting_station.couplers):
        raise ValueError(
            f"Changed station coupler count does not match starting topology for station {changed_station.grid_model_id}."
        )

    diff_switches: list[dict[str, str | bool]] = []
    starting_couplers = {coupler.grid_model_id: coupler for coupler in starting_station.couplers}
    for changed_coupler in changed_station.couplers:
        if changed_coupler.grid_model_id not in starting_couplers:
            raise ValueError(
                f"Coupler {changed_coupler.grid_model_id} not found in starting topology for station "
                f"{changed_station.grid_model_id}."
            )
        if changed_coupler.open != starting_couplers[changed_coupler.grid_model_id].open:
            diff_switches.append(
                {
                    "grid_model_id": changed_coupler.grid_model_id,
                    "open": changed_coupler.open,
                }
            )
    return diff_switches


def _get_asset_switch_diffs(
    changed_station: Station,
    starting_station: Station,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect selector and breaker switch changes between two station states.

    Parameters
    ----------
    changed_station : Station
        Station describing the target asset-to-busbar assignments.
    starting_station : Station
        Station describing the reference asset-to-busbar assignments. The station assets must be
        in the same order as ``changed_station``. This is the ordering contract provided by
        ``ActionSet.simplified_starting_topology``.
    fail_on_disconnect: bool
        Fundamentally, the stations should never disconnect an element. If this is detected, we can either raise
        or open the breaker. If fail_on_disconnect is true, a ValueError will be raised

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records for selector and breaker switches whose state changes.

    Raises
    ------
    ValueError
        If the station switching tables are structurally incompatible or asset order does not match.
    """
    changed_switching_table = _get_station_switching_table(changed_station)
    starting_switching_table = _get_station_switching_table(starting_station)
    if changed_switching_table.shape != starting_switching_table.shape:
        raise ValueError(
            "Changed station asset switching table shape does not match starting topology for station "
            f"{changed_station.grid_model_id}."
        )

    changed_busbar_lookup = _get_busbar_lookup(changed_station)
    changed_asset_ids = [asset.grid_model_id for asset in changed_station.assets]
    starting_asset_ids = [asset.grid_model_id for asset in starting_station.assets]
    if changed_asset_ids != starting_asset_ids:
        raise ValueError(
            "Changed station assets are not ordered like the starting topology for station "
            f"{changed_station.grid_model_id}. Use ActionSet.simplified_starting_topology as input."
        )

    switching_xor = np.logical_xor(starting_switching_table, changed_switching_table)

    diff_switches: list[dict[str, str | bool]] = []
    for column, changed_asset in enumerate(changed_station.assets):
        asset_bay = changed_asset.asset_bay
        if asset_bay is None:
            continue
        changed_switch_states = changed_switching_table[:, column]
        starting_switch_states = starting_switching_table[:, column]
        changed_rows = np.flatnonzero(switching_xor[:, column])
        changed_active = int(changed_switch_states.sum())
        starting_active = int(starting_switch_states.sum())

        if changed_active == 0:
            breaker_id = asset_bay.dv_switch_grid_model_id
            # The asset was disconnected by disconnecting all entries in the switching table. This can only be represented
            # through a breaker-open if there was at least one active busbar connection in the starting state.
            if starting_active > 0:
                if fail_on_disconnect:
                    raise ValueError(
                        f"Station action in station {changed_station.grid_model_id} would disconnect "
                        f"asset {changed_asset.grid_model_id}."
                    )
                diff_switches.append({"grid_model_id": breaker_id, "open": True})
            continue

        for row in changed_rows:
            busbar_id = changed_busbar_lookup[int(row)]
            switch_id = asset_bay.sr_switch_grid_model_id[busbar_id]
            diff_switches.append({"grid_model_id": switch_id, "open": not bool(changed_switch_states[row])})

    return diff_switches


def _get_switch_updates_from_station_ids(
    changed_station_lookup: dict[str, Station],
    starting_station_lookup: dict[str, Station],
    ordered_station_ids: list[str],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Build switch updates for a specific ordered list of stations.

    Parameters
    ----------
    changed_station_lookup : dict[str, Station]
        Changed stations by station id.
    starting_station_lookup : dict[str, Station]
        Reference stations by station id.
    ordered_station_ids : list[str]
        Station ids to process in output order.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows for the requested stations.
    """
    diff_switches: list[dict[str, str | bool]] = []
    for station_id in ordered_station_ids:
        if station_id not in starting_station_lookup:
            raise ValueError(f"Changed station {station_id} not found in starting topology.")
        starting_station = starting_station_lookup[station_id]
        changed_station = changed_station_lookup.get(station_id, starting_station)
        diff_switches.extend(_get_coupler_switch_diffs(changed_station=changed_station, starting_station=starting_station))
        diff_switches.extend(
            _get_asset_switch_diffs(
                changed_station=changed_station,
                starting_station=starting_station,
            )
        )

    diff_switch_df = pd.DataFrame.from_records(diff_switches, columns=["grid_model_id", "open"])
    if diff_switch_df.empty:
        diff_switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], diff_switch_df)


def _get_station_switching_table(station: Station) -> Bool[np.ndarray, " n_busbar n_asset"]:
    """Return the station switching table as a boolean numpy array.

    Parameters
    ----------
    station : Station
        Station whose switching table should be normalized to a boolean ndarray.

    Returns
    -------
    Bool[np.ndarray, " n_busbar n_asset"]
        Boolean switching table indexed by busbar and asset.
    """
    return np.asarray(station.asset_switching_table, dtype=bool)


@pa.check_types
def get_changing_switches_from_changed_stations(
    changed_stations: list[Station],
    starting_topology: Topology,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get changed switches by comparing changed stations to the starting topology.

    This is intended for changed stations originating from ``ActionSet.local_actions`` where only
    coupler open states and the asset switching table differ from the starting topology.

    Parameters
    ----------
    changed_stations : list[Station]
        Stations describing the target state for the affected substations.
    starting_topology : Topology
        Starting topology containing the reference state for all stations. This is expected to be
        ``ActionSet.simplified_starting_topology`` so that station asset ordering matches the
        ordering used by ``ActionSet.local_actions``.


    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows containing only switches whose state differs from the starting topology.

    Raises
    ------
    ValueError
        If a changed station is duplicated, missing from the starting topology, or is structurally
        incompatible with the reference station.
    """
    starting_station_lookup, changed_station_lookup, ordered_changed_station_ids = _resolve_changed_stations(
        changed_stations=changed_stations,
        starting_topology=starting_topology,
    )
    return _get_switch_updates_from_station_ids(
        changed_station_lookup=changed_station_lookup,
        starting_station_lookup=starting_station_lookup,
        ordered_station_ids=ordered_changed_station_ids,
    )
