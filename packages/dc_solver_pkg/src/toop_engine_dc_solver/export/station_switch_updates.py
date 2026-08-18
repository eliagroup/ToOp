# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers to create switch update data from changed bus groups.

This module complements the network-based helpers in ``asset_topology_to_dgs`` by deriving the
same switch update schema from changed bus groups and reference bus-group snapshots.
"""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetConnection, RuntimeBusGroup
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import SimplifiedBusGroup
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def _get_busbar_lookup(bus_group: RuntimeBusGroup) -> dict[int, str]:
    """Map busbar row indices in the switching table to busbar ids."""
    return {index: busbar.grid_model_id for index, busbar in enumerate(bus_group.busbars)}


def _get_asset_busbar_lookup(
    bus_group: RuntimeBusGroup,
    asset_connection: RuntimeAssetConnection,
) -> dict[int, str]:
    """Resolve row-to-busbar ids for one asset connection.

    Simplified split-bus-group actions can use per-asset logical rows that no longer match the
    bus-group-level busbar ids. When that happens, the asset bay selector-switch keys preserve the
    physical busbar ordering needed to translate row changes back into switch updates.
    """
    bus_group_busbar_lookup = _get_busbar_lookup(bus_group)
    asset_bay = asset_connection.asset_bay
    if asset_bay is None:
        return bus_group_busbar_lookup

    bus_group_busbar_ids = list(bus_group_busbar_lookup.values())
    asset_busbar_ids = list(asset_bay.busbar_disconnector_grid_model_id.keys())
    if len(asset_busbar_ids) == len(bus_group_busbar_ids) and not set(bus_group_busbar_ids).issubset(
        asset_bay.busbar_disconnector_grid_model_id
    ):
        return {index: busbar_id for index, busbar_id in enumerate(asset_busbar_ids)}

    return bus_group_busbar_lookup


def _resolve_changed_bus_groups(
    changed_bus_groups: list[SimplifiedBusGroup],
    starting_bus_groups: list[SimplifiedBusGroup],
) -> tuple[dict[str, SimplifiedBusGroup], dict[str, SimplifiedBusGroup], list[str]]:
    """Resolve bus-group lookups and preserve changed-bus-group ordering.

    This helper is intentionally limited to bus-group actions. It validates that all changed bus groups
    are unique and present in the starting bus groups, then returns them in the same order as the
    starting bus groups.

    Parameters
    ----------
    changed_bus_groups : list[SimplifiedBusGroup]
        Bus groups that contain topology changes relative to the starting topology.
    starting_bus_groups : list[SimplifiedBusGroup]
        Reference bus groups used to validate identities and derive stable ordering.

    Returns
    -------
    tuple[dict[str, SimplifiedBusGroup], dict[str, SimplifiedBusGroup], list[str]]
        A tuple containing:
        1. a lookup for starting bus groups by ``bus_group_id``,
        2. a lookup for changed bus groups by ``bus_group_id``,
        3. the changed bus-group ids in starting-bus-group order.

    Raises
    ------
    ValueError
        If ``changed_bus_groups`` contains duplicate ids or if a changed bus group is not
        present in the starting bus groups.
    """
    changed_bus_group_ids = [bus_group.bus_group_id for bus_group in changed_bus_groups]
    if len(changed_bus_group_ids) != len(set(changed_bus_group_ids)):
        raise ValueError("Changed bus groups must be unique by bus_group_id.")

    starting_bus_group_lookup = {bus_group.bus_group_id: bus_group for bus_group in starting_bus_groups}
    changed_bus_group_lookup = {bus_group.bus_group_id: bus_group for bus_group in changed_bus_groups}
    missing_bus_group_ids = set(changed_bus_group_lookup).difference(starting_bus_group_lookup)
    if missing_bus_group_ids:
        raise ValueError(f"Changed bus groups not found in starting bus groups: {sorted(missing_bus_group_ids)}")

    ordered_changed_bus_group_ids = [
        bus_group.bus_group_id for bus_group in starting_bus_groups if bus_group.bus_group_id in changed_bus_group_lookup
    ]
    return starting_bus_group_lookup, changed_bus_group_lookup, ordered_changed_bus_group_ids


def _get_coupler_switch_diffs(
    changed_bus_group: SimplifiedBusGroup,
    starting_bus_group: SimplifiedBusGroup,
) -> list[dict[str, str | bool]]:
    """Collect coupler switch changes between two bus-group states.

    Parameters
    ----------
    changed_bus_group : SimplifiedBusGroup
        Bus group describing the target coupler states.
    starting_bus_group : SimplifiedBusGroup
        Bus group describing the reference coupler states.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records for couplers whose ``open`` state changes.

    Raises
    ------
    ValueError
        If the bus groups do not expose the same couplers.
    """
    if len(changed_bus_group.couplers) != len(starting_bus_group.couplers):
        raise ValueError(
            "Changed bus-group coupler count does not match the starting bus group for "
            f"bus group {changed_bus_group.bus_group_id}."
        )

    diff_switches: list[dict[str, str | bool]] = []
    starting_couplers = {coupler.grid_model_id: coupler for coupler in starting_bus_group.couplers}
    for changed_coupler in changed_bus_group.couplers:
        if changed_coupler.grid_model_id not in starting_couplers:
            raise ValueError(
                f"Coupler {changed_coupler.grid_model_id} not found in starting bus groups for bus group "
                f"{changed_bus_group.bus_group_id}."
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
    changed_bus_group: SimplifiedBusGroup,
    starting_bus_group: SimplifiedBusGroup,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect branch selector and breaker switch changes between two bus-group states.

    Parameters
    ----------
    changed_bus_group : SimplifiedBusGroup
        Bus group describing the target branch-to-busbar assignments.
    starting_bus_group : SimplifiedBusGroup
        Bus group describing the reference branch assignments. The branch connection
        array must stay in the same order as ``changed_bus_group``.
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
    if changed_bus_group.branch_switching_table.shape != starting_bus_group.branch_switching_table.shape:
        raise ValueError(
            "Changed bus-group asset switching table shape does not match the starting bus group for bus group "
            f"{changed_bus_group.bus_group_id}."
        )

    changed_asset_ids = [asset_connection.asset.grid_model_id for asset_connection in changed_bus_group.branch_connections]
    starting_asset_ids = [asset_connection.asset.grid_model_id for asset_connection in starting_bus_group.branch_connections]
    if changed_asset_ids != starting_asset_ids:
        raise ValueError(
            "Changed bus-group assets are not ordered like the starting bus group for bus group "
            f"{changed_bus_group.bus_group_id}. Use ActionSet.get_simplified_starting_bus_groups() as input."
        )

    switching_xor = np.logical_xor(starting_bus_group.branch_switching_table, changed_bus_group.branch_switching_table)
    diff_switches: list[dict[str, str | bool]] = []

    for column, changed_asset_connection in enumerate(changed_bus_group.branch_connections):
        changed_busbar_lookup = _get_asset_busbar_lookup(changed_bus_group, changed_asset_connection)
        asset_bay = changed_asset_connection.asset_bay
        if asset_bay is None:
            continue
        changed_switch_states = changed_bus_group.branch_switching_table[:, column]
        starting_switch_states = starting_bus_group.branch_switching_table[:, column]
        changed_rows = np.flatnonzero(switching_xor[:, column])
        changed_active = int(changed_switch_states.sum())
        starting_active = int(starting_switch_states.sum())

        if changed_active == 0:
            if starting_active > 0:
                if fail_on_disconnect:
                    raise ValueError(
                        f"Bus-group action in bus group {changed_bus_group.bus_group_id} would disconnect "
                        f"asset {changed_asset_connection.asset.grid_model_id}."
                    )
                diff_switches.append({"grid_model_id": asset_bay.breaker_grid_model_id, "open": True})
            continue

        for row in changed_rows:
            busbar_id = changed_busbar_lookup[int(row)]
            switch_id = asset_bay.busbar_disconnector_grid_model_id[busbar_id]
            diff_switches.append({"grid_model_id": switch_id, "open": not bool(changed_switch_states[row])})

    return diff_switches


def _get_injection_switch_diffs(
    changed_bus_group: SimplifiedBusGroup,
    starting_bus_group: SimplifiedBusGroup,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect injection selector and breaker switch changes between two bus-group states.

    Parameters
    ----------
    changed_bus_group : SimplifiedBusGroup
        Bus group describing the target injection-to-busbar assignments.
    starting_bus_group : SimplifiedBusGroup
        Bus group describing the reference injection assignments. The injection
        connection array must stay in the same order as ``changed_bus_group``.
    fail_on_disconnect : bool, default=False
        Whether to raise when a changed injection becomes fully disconnected
        instead of emitting a breaker-opening update.

    Returns
    -------
    list[dict[str, str | bool]]
        Switch update records derived from ``injection_connections`` and
        ``injection_switching_table``. This mirrors
        ``_get_branch_switch_diffs`` for the injection-side bus-group tables.

    Raises
    ------
    ValueError
        If the injection switching tables are structurally incompatible, the
        injection order does not match, or a disconnect is detected while
        ``fail_on_disconnect`` is true.
    """
    if changed_bus_group.injection_switching_table.shape != starting_bus_group.injection_switching_table.shape:
        raise ValueError(
            "Changed station asset switching table shape does not match starting stations for station "
            f"{changed_bus_group.bus_group_id}."
        )

    changed_asset_ids = [
        asset_connection.asset.grid_model_id for asset_connection in changed_bus_group.injection_connections
    ]
    starting_asset_ids = [
        asset_connection.asset.grid_model_id for asset_connection in starting_bus_group.injection_connections
    ]
    if changed_asset_ids != starting_asset_ids:
        raise ValueError(
            "Changed station assets are not ordered like the starting stations for station "
            f"{changed_bus_group.bus_group_id}. Use ActionSet.get_simplified_starting_bus_groups() as input."
        )

    switching_xor = np.logical_xor(
        starting_bus_group.injection_switching_table,
        changed_bus_group.injection_switching_table,
    )
    diff_switches: list[dict[str, str | bool]] = []

    for column, changed_asset_connection in enumerate(changed_bus_group.injection_connections):
        changed_busbar_lookup = _get_asset_busbar_lookup(changed_bus_group, changed_asset_connection)
        asset_bay = changed_asset_connection.asset_bay
        if asset_bay is None:
            continue
        changed_switch_states = changed_bus_group.injection_switching_table[:, column]
        starting_switch_states = starting_bus_group.injection_switching_table[:, column]
        changed_rows = np.flatnonzero(switching_xor[:, column])
        changed_active = int(changed_switch_states.sum())
        starting_active = int(starting_switch_states.sum())

        if changed_active == 0:
            if starting_active > 0:
                if fail_on_disconnect:
                    raise ValueError(
                        f"Station action in station {changed_bus_group.bus_group_id} would disconnect "
                        f"asset {changed_asset_connection.asset.grid_model_id}."
                    )
                diff_switches.append({"grid_model_id": asset_bay.breaker_grid_model_id, "open": True})
            continue

        for row in changed_rows:
            busbar_id = changed_busbar_lookup[int(row)]
            switch_id = asset_bay.busbar_disconnector_grid_model_id[busbar_id]
            diff_switches.append({"grid_model_id": switch_id, "open": not bool(changed_switch_states[row])})

    return diff_switches


def _get_asset_switch_diffs(
    changed_bus_group: SimplifiedBusGroup,
    starting_bus_group: SimplifiedBusGroup,
    fail_on_disconnect: bool = False,
) -> list[dict[str, str | bool]]:
    """Collect selector and breaker switch changes between two station states.

    Parameters
    ----------
    changed_bus_group : SimplifiedBusGroup
        Bus group describing the target branch/injection-to-busbar assignments.
    starting_bus_group : SimplifiedBusGroup
        Bus group describing the reference branch/injection assignments. The branch and injection
        connection arrays must each stay in the same order as ``changed_bus_group``. This is the
        ordering contract provided by ``ActionSet.get_simplified_starting_bus_groups()``.
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
            changed_bus_group=changed_bus_group,
            starting_bus_group=starting_bus_group,
            fail_on_disconnect=fail_on_disconnect,
        ),
        *_get_injection_switch_diffs(
            changed_bus_group=changed_bus_group,
            starting_bus_group=starting_bus_group,
            fail_on_disconnect=fail_on_disconnect,
        ),
    ]


def _get_switch_updates_from_bus_group_ids(
    changed_bus_group_lookup: dict[str, SimplifiedBusGroup],
    starting_bus_group_lookup: dict[str, SimplifiedBusGroup],
    ordered_bus_group_ids: list[str],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Build switch updates for a specific ordered list of stations.

    Parameters
    ----------
    changed_bus_group_lookup : dict[str, SimplifiedBusGroup]
        Changed bus groups by bus-group id.
    starting_bus_group_lookup : dict[str, SimplifiedBusGroup]
        Reference bus groups by bus-group id.
    ordered_bus_group_ids : list[str]
        Bus-group ids to process in output order.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows for the requested stations.
    """
    diff_switches: list[dict[str, str | bool]] = []
    for station_id in ordered_bus_group_ids:
        if station_id not in starting_bus_group_lookup:
            raise ValueError(f"Changed station {station_id} not found in starting stations.")
        starting_station = starting_bus_group_lookup[station_id]
        changed_station = changed_bus_group_lookup.get(station_id, starting_station)
        diff_switches.extend(
            _get_coupler_switch_diffs(
                changed_bus_group=changed_station,
                starting_bus_group=starting_station,
            )
        )
        diff_switches.extend(
            _get_asset_switch_diffs(
                changed_bus_group=changed_station,
                starting_bus_group=starting_station,
            )
        )

    diff_switch_df = pd.DataFrame.from_records(diff_switches, columns=["grid_model_id", "open"])
    if diff_switch_df.empty:
        diff_switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return diff_switch_df


@pa.check_types
def get_changing_switches_from_changed_bus_groups(
    changed_bus_groups: list[SimplifiedBusGroup],
    starting_bus_groups: list[SimplifiedBusGroup],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get changed switches by comparing changed bus groups to reference bus groups.

    This is intended for changed bus groups originating from ``ActionSet.local_actions`` where only
    coupler open states and the split bus-group switching tables differ from the reference bus groups.
    In the split topology model that means ``branch_switching_table`` and
    ``injection_switching_table`` are compared independently and then merged into one switch-update
    table.

    Parameters
    ----------
    changed_bus_groups : list[SimplifiedBusGroup]
        Bus groups describing the target state for the affected substations.
    starting_bus_groups : list[SimplifiedBusGroup]
        Reference bus groups containing the baseline state for all bus groups. This is expected to be
        ``ActionSet.simplified_starting_stations`` so that both branch and injection
        connection ordering match the ordering used by ``ActionSet.local_actions``.


    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows containing only switches whose state differs from the starting bus groups.

    Raises
    ------
    ValueError
        If a changed bus group is duplicated, missing from the starting bus groups, or is structurally
        incompatible with the reference bus group.
    """
    starting_bus_group_lookup, changed_bus_group_lookup, ordered_changed_bus_group_ids = _resolve_changed_bus_groups(
        changed_bus_groups=changed_bus_groups,
        starting_bus_groups=starting_bus_groups,
    )
    return _get_switch_updates_from_bus_group_ids(
        changed_bus_group_lookup=changed_bus_group_lookup,
        starting_bus_group_lookup=starting_bus_group_lookup,
        ordered_bus_group_ids=ordered_changed_bus_group_ids,
    )
