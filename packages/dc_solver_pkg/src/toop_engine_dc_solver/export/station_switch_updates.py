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
from toop_engine_interfaces.asset_topology.asset_topology import Topology
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def _get_busbar_lookup(station: MaterializedStation) -> dict[int, str]:
    """Map busbar row indices in the switching table to busbar ids."""
    return {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}


def _get_asset_busbar_lookup(
    station: MaterializedStation,
    asset_connection: MaterializedAssetConnection,
) -> dict[int, str]:
    """Resolve row-to-busbar ids for one asset connection.

    Simplified split-station actions can use per-asset logical rows that no longer match the
    station-level busbar ids. When that happens, the asset bay selector-switch keys preserve the
    physical busbar ordering needed to translate row changes back into switch updates.
    """
    station_busbar_lookup = _get_busbar_lookup(station)
    asset_bay = asset_connection.asset_bay
    if asset_bay is None:
        return station_busbar_lookup

    station_busbar_ids = list(station_busbar_lookup.values())
    asset_busbar_ids = list(asset_bay.sr_switch_grid_model_id.keys())
    if len(asset_busbar_ids) == len(station_busbar_ids) and not set(station_busbar_ids).issubset(
        asset_bay.sr_switch_grid_model_id
    ):
        return {index: busbar_id for index, busbar_id in enumerate(asset_busbar_ids)}

    return station_busbar_lookup


def _resolve_changed_stations(
    changed_stations: list[MaterializedStation],
    starting_topology: Topology,
) -> tuple[dict[str, MaterializedStation], dict[str, MaterializedStation], list[str]]:
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

    starting_stations = starting_topology.materialize_stations()
    starting_station_lookup = {station.grid_model_id: station for station in starting_stations}
    changed_station_lookup = {station.grid_model_id: station for station in changed_stations}
    missing_station_ids = set(changed_station_lookup).difference(starting_station_lookup)
    if missing_station_ids:
        raise ValueError(f"Changed stations not found in starting topology: {sorted(missing_station_ids)}")

    ordered_changed_station_ids = [
        station.grid_model_id for station in starting_stations if station.grid_model_id in changed_station_lookup
    ]
    return starting_station_lookup, changed_station_lookup, ordered_changed_station_ids


def _get_coupler_switch_diffs(
    changed_station: MaterializedStation,
    starting_station: MaterializedStation,
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


def _get_branch_switch_diffs(
    changed_station: MaterializedStation,
    starting_station: MaterializedStation,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect branch selector and breaker switch changes between two station states.

    Parameters
    ----------
    changed_station : MaterializedStation
        Station describing the target branch-to-busbar assignments.
    starting_station : MaterializedStation
        Station describing the reference branch assignments. The branch connection
        array must stay in the same order as ``changed_station``.
    fail_on_disconnect : bool, default=False
        Whether to raise when a changed branch becomes fully disconnected instead
        of emitting a breaker-opening update.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records derived from ``branch_connections`` and
        ``branch_switching_table``. A changed branch column can produce selector
        switch updates for busbar reassignments and, when the branch becomes
        fully disconnected, one breaker update via the asset-bay disconnecting
        switch.

    Raises
    ------
    ValueError
        If the branch switching tables are structurally incompatible, the branch
        order does not match, or a disconnect is detected while
        ``fail_on_disconnect`` is true.
    """
    if changed_station.branch_switching_table.shape != starting_station.branch_switching_table.shape:
        raise ValueError(
            "Changed station asset switching table shape does not match starting topology for station "
            f"{changed_station.grid_model_id}."
        )

    changed_asset_ids = [asset_connection.asset.grid_model_id for asset_connection in changed_station.branch_connections]
    starting_asset_ids = [asset_connection.asset.grid_model_id for asset_connection in starting_station.branch_connections]
    if changed_asset_ids != starting_asset_ids:
        raise ValueError(
            "Changed station assets are not ordered like the starting topology for station "
            f"{changed_station.grid_model_id}. Use ActionSet.simplified_starting_topology as input."
        )

    switching_xor = np.logical_xor(starting_station.branch_switching_table, changed_station.branch_switching_table)
    diff_switches: list[dict[str, str | bool]] = []

    for column, changed_asset_connection in enumerate(changed_station.branch_connections):
        changed_busbar_lookup = _get_asset_busbar_lookup(changed_station, changed_asset_connection)
        asset_bay = changed_asset_connection.asset_bay
        if asset_bay is None:
            continue
        changed_switch_states = changed_station.branch_switching_table[:, column]
        starting_switch_states = starting_station.branch_switching_table[:, column]
        changed_rows = np.flatnonzero(switching_xor[:, column])
        changed_active = int(changed_switch_states.sum())
        starting_active = int(starting_switch_states.sum())

        if changed_active == 0:
            if starting_active > 0:
                if fail_on_disconnect:
                    raise ValueError(
                        f"Station action in station {changed_station.grid_model_id} would disconnect "
                        f"asset {changed_asset_connection.asset.grid_model_id}."
                    )
                diff_switches.append({"grid_model_id": asset_bay.dv_switch_grid_model_id, "open": True})
            continue

        for row in changed_rows:
            busbar_id = changed_busbar_lookup[int(row)]
            switch_id = asset_bay.sr_switch_grid_model_id[busbar_id]
            diff_switches.append({"grid_model_id": switch_id, "open": not bool(changed_switch_states[row])})

    return diff_switches


def _get_injection_switch_diffs(
    changed_station: MaterializedStation,
    starting_station: MaterializedStation,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect injection selector and breaker switch changes between two station states.

    Parameters
    ----------
    changed_station : MaterializedStation
        Station describing the target injection-to-busbar assignments.
    starting_station : MaterializedStation
        Station describing the reference injection assignments. The injection
        connection array must stay in the same order as ``changed_station``.
    fail_on_disconnect : bool, default=False
        Whether to raise when a changed injection becomes fully disconnected
        instead of emitting a breaker-opening update.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records derived from ``injection_connections`` and
        ``injection_switching_table``. This mirrors
        ``_get_branch_switch_diffs`` for the injection-side station tables.

    Raises
    ------
    ValueError
        If the injection switching tables are structurally incompatible, the
        injection order does not match, or a disconnect is detected while
        ``fail_on_disconnect`` is true.
    """
    if changed_station.injection_switching_table.shape != starting_station.injection_switching_table.shape:
        raise ValueError(
            "Changed station asset switching table shape does not match starting topology for station "
            f"{changed_station.grid_model_id}."
        )

    changed_asset_ids = [asset_connection.asset.grid_model_id for asset_connection in changed_station.injection_connections]
    starting_asset_ids = [
        asset_connection.asset.grid_model_id for asset_connection in starting_station.injection_connections
    ]
    if changed_asset_ids != starting_asset_ids:
        raise ValueError(
            "Changed station assets are not ordered like the starting topology for station "
            f"{changed_station.grid_model_id}. Use ActionSet.simplified_starting_topology as input."
        )

    switching_xor = np.logical_xor(
        starting_station.injection_switching_table,
        changed_station.injection_switching_table,
    )
    diff_switches: list[dict[str, str | bool]] = []

    for column, changed_asset_connection in enumerate(changed_station.injection_connections):
        changed_busbar_lookup = _get_asset_busbar_lookup(changed_station, changed_asset_connection)
        asset_bay = changed_asset_connection.asset_bay
        if asset_bay is None:
            continue
        changed_switch_states = changed_station.injection_switching_table[:, column]
        starting_switch_states = starting_station.injection_switching_table[:, column]
        changed_rows = np.flatnonzero(switching_xor[:, column])
        changed_active = int(changed_switch_states.sum())
        starting_active = int(starting_switch_states.sum())

        if changed_active == 0:
            if starting_active > 0:
                if fail_on_disconnect:
                    raise ValueError(
                        f"Station action in station {changed_station.grid_model_id} would disconnect "
                        f"asset {changed_asset_connection.asset.grid_model_id}."
                    )
                diff_switches.append({"grid_model_id": asset_bay.dv_switch_grid_model_id, "open": True})
            continue

        for row in changed_rows:
            busbar_id = changed_busbar_lookup[int(row)]
            switch_id = asset_bay.sr_switch_grid_model_id[busbar_id]
            diff_switches.append({"grid_model_id": switch_id, "open": not bool(changed_switch_states[row])})

    return diff_switches


def _get_asset_switch_diffs(
    changed_station: MaterializedStation,
    starting_station: MaterializedStation,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect selector and breaker switch changes between two station states.

    Parameters
    ----------
    changed_station : MaterializedStation
        Station describing the target branch/injection-to-busbar assignments.
    starting_station : MaterializedStation
        Station describing the reference branch/injection assignments. The branch and injection
        connection arrays must each stay in the same order as ``changed_station``. This is the
        ordering contract provided by ``ActionSet.simplified_starting_topology``.
    fail_on_disconnect : bool, default=False
        Fundamentally, the stations should never disconnect an element. If this is detected, the
        helper can either raise or emit a breaker-opening update. If ``fail_on_disconnect`` is
        true, a ``ValueError`` is raised instead.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records for selector and breaker switches whose state changes across both the
        branch and injection station tables.

    Raises
    ------
    ValueError
        If the station switching tables are structurally incompatible or asset order does not match.
    """
    return [
        *_get_branch_switch_diffs(
            changed_station=changed_station,
            starting_station=starting_station,
            fail_on_disconnect=fail_on_disconnect,
        ),
        *_get_injection_switch_diffs(
            changed_station=changed_station,
            starting_station=starting_station,
            fail_on_disconnect=fail_on_disconnect,
        ),
    ]


def _get_switch_updates_from_station_ids(
    changed_station_lookup: dict[str, MaterializedStation],
    starting_station_lookup: dict[str, MaterializedStation],
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
    return diff_switch_df


@pa.check_types
def get_changing_switches_from_changed_stations(
    changed_stations: list[MaterializedStation],
    starting_topology: Topology,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get changed switches by comparing changed stations to the starting topology.

    This is intended for changed stations originating from ``ActionSet.local_actions`` where only
    coupler open states and the split station switching tables differ from the starting topology.
    In the split topology model that means ``branch_switching_table`` and
    ``injection_switching_table`` are compared independently and then merged into one switch-update
    table.

    Parameters
    ----------
    changed_stations : list[MaterializedStation]
        Stations describing the target state for the affected substations.
    starting_topology : Topology
        Starting topology containing the reference state for all stations. This is expected to be
        ``ActionSet.simplified_starting_topology`` so that both branch and injection connection
        ordering match the ordering used by ``ActionSet.local_actions``.


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
