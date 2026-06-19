# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines to apply an asset topology to a powsybl model.

Fundamentally there are two ways to do it: A node/breaker way which just changes switch states and is equivalent to the .dgs
export and a bus/branch way which reassigns the elements to their new locations.

Furthermore, there are two ways to get the switch updates that are needed - either by comparing to the switch state of the
network (this file) or by comparing to the asset topology starting state (export.py). Comparing directly to the grid is safer
in case multiple changes have been made to the grid for some reason.

For the bus/branch way, see the function apply_topology_bus_branch.
the node/breaker way is still TODO.
"""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import Literal, Optional, Union
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import assert_station_in_network
from toop_engine_interfaces.asset_topology import (
    AppliedStation,
    BusbarCoupler,
    MaterializedStation,
    RawStation,
    RealizedTopology,
    Topology,
    copy_topology_with_updates,
)
from toop_engine_interfaces.asset_topology_helpers import accumulate_diffs
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema

logger = structlog.get_logger(__name__)


@pa.check_types
def get_coupler_states_from_busbar_couplers(station_couplers: list[BusbarCoupler]) -> pat.DataFrame[SwitchUpdateSchema]:
    """Translate coupler states to the switch-update schema format.

    Parameters
    ----------
    station_couplers : list[BusbarCoupler]
        Couplers whose target open state should be expressed as switch updates.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch-update rows containing the desired state for each coupler.

    Raises
    ------
    ValueError
        If a coupler is marked out of service.
    """
    switch_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    for coupler in station_couplers:
        if not coupler.in_service:
            raise ValueError(f"Coupler {coupler.grid_model_id} is not in service, undefined behavior")
        switch_df.loc[switch_df.shape[0]] = {
            "grid_model_id": coupler.grid_model_id,
            "open": coupler.open,
        }
    switch_df = switch_df.astype({"grid_model_id": str, "open": bool})
    return switch_df


def get_busbar_lookup(station: MaterializedStation) -> dict[int, str]:
    """Map switching-table row indices to station busbar ids.

    Parameters
    ----------
    station : MaterializedStation
        Station whose busbar ordering defines the switching-table row order.

    Returns
    -------
    dict[int, str]
        Mapping from switching-table row index to busbar ``grid_model_id``.
    """
    return {index: busbar.grid_model_id for index, busbar in enumerate(station.busbars)}


def _get_branch_switch_states_from_station(
    station: MaterializedStation,
    busbar_id_dict: dict[int, str],
) -> tuple[list[dict[str, str | bool]], list[dict[str, str | bool]]]:
    """Translate branch selector and breaker states of one station to switch updates.

    Parameters
    ----------
    station : MaterializedStation
        Station whose branch-side switching state should be exported.
    busbar_id_dict : dict[int, str]
        Mapping from branch switching-table row index to busbar id.

    Returns
    -------
    tuple[list[dict[str, str | bool]], list[dict[str, str | bool]]]
        Two lists containing branch reassignment switch states and branch
        disconnection switch states.

    Raises
    ------
    ValueError
        If a branch switching-table column connects the same branch to multiple
        busbars.
    """
    switch_reassignment_list: list[dict[str, str | bool]] = []
    switch_disconnection_list: list[dict[str, str | bool]] = []

    assert station.branch_switching_table.shape[1] == len(station.branch_connections), (
        "The branch switching table has a different number of columns than the branch reassignment list. "
        f"Columns: {station.branch_switching_table.shape[1]}, Assets: {len(station.branch_connections)}"
    )

    for column, asset_connection in enumerate(station.branch_connections):
        asset_switch_ids = asset_connection.get_sr_switch()
        if asset_switch_ids is None:
            continue
        asset_switch_states = station.branch_switching_table[:, column]
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
            asset_bay = asset_connection.asset_bay
            assert asset_bay is not None
            switch_disconnection_list.append(
                {
                    "grid_model_id": asset_bay.dv_switch_grid_model_id,
                    "open": True,
                }
            )
        else:
            raise ValueError(
                f"Switching table column {column} has more than one True value: {station.branch_switching_table[:, column]}"
            )

    return switch_reassignment_list, switch_disconnection_list


def _get_injection_switch_states_from_station(
    station: MaterializedStation,
    busbar_id_dict: dict[int, str],
) -> tuple[list[dict[str, str | bool]], list[dict[str, str | bool]]]:
    """Translate injection selector and breaker states of one station to switch updates.

    Parameters
    ----------
    station : MaterializedStation
        Station whose injection-side switching state should be exported.
    busbar_id_dict : dict[int, str]
        Mapping from injection switching-table row index to busbar id.

    Returns
    -------
    tuple[list[dict[str, str | bool]], list[dict[str, str | bool]]]
        Two lists containing injection reassignment switch states and injection
        disconnection switch states.

    Raises
    ------
    ValueError
        If an injection switching-table column connects the same injection to
        multiple busbars.
    """
    switch_reassignment_list: list[dict[str, str | bool]] = []
    switch_disconnection_list: list[dict[str, str | bool]] = []

    assert station.injection_switching_table.shape[1] == len(station.injection_connections), (
        "The injection switching table has a different number of columns than the injection reassignment list. "
        f"Columns: {station.injection_switching_table.shape[1]}, Assets: {len(station.injection_connections)}"
    )

    for column, asset_connection in enumerate(station.injection_connections):
        asset_switch_ids = asset_connection.get_sr_switch()
        if asset_switch_ids is None:
            continue
        asset_switch_states = station.injection_switching_table[:, column]
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
            asset_bay = asset_connection.asset_bay
            assert asset_bay is not None
            switch_disconnection_list.append(
                {
                    "grid_model_id": asset_bay.dv_switch_grid_model_id,
                    "open": True,
                }
            )
        else:
            raise ValueError(
                f"Switching table column {column} has more than one True value: "
                f"{station.injection_switching_table[:, column]}"
            )

    return switch_reassignment_list, switch_disconnection_list


@pa.check_types
def get_asset_switch_states_from_station(
    station: MaterializedStation,
) -> tuple[pat.DataFrame[SwitchUpdateSchema], pat.DataFrame[SwitchUpdateSchema]]:
    """Translate station asset switch states to switch-update dataframes.

    Parameters
    ----------
    station : MaterializedStation
        Station whose branch and injection switching tables should be exported.

    Returns
    -------
    tuple[pat.DataFrame[SwitchUpdateSchema], pat.DataFrame[SwitchUpdateSchema]]
        Two dataframes containing reassignment-related switch updates and
        disconnection-related switch updates.
    """
    busbar_id_dict = get_busbar_lookup(station)
    branch_reassignment_list, branch_disconnection_list = _get_branch_switch_states_from_station(
        station=station,
        busbar_id_dict=busbar_id_dict,
    )
    injection_reassignment_list, injection_disconnection_list = _get_injection_switch_states_from_station(
        station=station,
        busbar_id_dict=busbar_id_dict,
    )
    switch_reassignment_list = [*branch_reassignment_list, *injection_reassignment_list]
    switch_disconnection_list = [*branch_disconnection_list, *injection_disconnection_list]

    switch_reassignment_df = pd.DataFrame.from_records(switch_reassignment_list, columns=["grid_model_id", "open"])
    switch_disconnection_df = pd.DataFrame.from_records(switch_disconnection_list, columns=["grid_model_id", "open"])
    if switch_reassignment_df.empty:
        switch_reassignment_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    if switch_disconnection_df.empty:
        switch_disconnection_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    switch_reassignment_df = switch_reassignment_df.astype({"grid_model_id": str, "open": bool})
    switch_disconnection_df = switch_disconnection_df.astype({"grid_model_id": str, "open": bool})
    return switch_reassignment_df, switch_disconnection_df


@pa.check_types
def get_diff_switch_states(
    network: Network,
    switch_df: pat.DataFrame[SwitchUpdateSchema],
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Filter switch updates down to the ones differing from the network state.

    Parameters
    ----------
    network : Network
        Powsybl network providing the current switch states.
    switch_df : pat.DataFrame[SwitchUpdateSchema]
        Candidate switch updates.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Only the switch updates whose target ``open`` value differs from the
        current network state.

    Raises
    ------
    ValueError
        If a switch id from ``switch_df`` is missing in the network.
    """
    diff_switch_df = switch_df.merge(
        network.get_switches(attributes=["open"]),
        left_on="grid_model_id",
        right_index=True,
        how="left",
        suffixes=("", "_network"),
    )
    if diff_switch_df["open_network"].isna().any():
        raise ValueError(
            "Switch id not found in the network - Switch id: "
            f"{diff_switch_df.loc[diff_switch_df['open_network'].isna(), 'grid_model_id']}"
        )
    diff_switch_df = diff_switch_df[diff_switch_df["open"] != diff_switch_df["open_network"]]
    diff_switch_df = diff_switch_df[["grid_model_id", "open"]]
    diff_switch_df = diff_switch_df.astype({"grid_model_id": str, "open": bool})
    return diff_switch_df


@pa.check_types
def get_changing_switches_from_topology(network: Network, target_topology: Topology) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates needed to realize a topology on a node-breaker network.

    Parameters
    ----------
    network : Network
        Powsybl network whose current switch states act as the reference.
    target_topology : Topology
        Topology whose couplers and split branch/injection station tables define
        the target switch states.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch updates required to transform the current network state into the
        target node-breaker topology.
    """
    switch_update_df = get_empty_dataframe_from_model(SwitchUpdateSchema)
    for station in target_topology.materialize_stations():
        coupler_df = get_coupler_states_from_busbar_couplers(station.couplers)
        switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
        station_switch_updates = pd.concat([coupler_df, switch_reassignment_df, switch_disconnection_df], ignore_index=True)
        switch_update_df = pd.concat([switch_update_df, station_switch_updates], ignore_index=True)

    if switch_update_df.duplicated(subset=["grid_model_id"]).any():
        logger.warning(
            "Duplicate switch ids found in the switch update schema",
            duplicate_switch_ids=switch_update_df.loc[
                switch_update_df.duplicated(subset=["grid_model_id"]), "grid_model_id"
            ].to_list(),
        )
        switch_update_df = switch_update_df.drop_duplicates(subset=["grid_model_id"])
    switch_update_df = switch_update_df.astype({"grid_model_id": str, "open": bool})
    return get_diff_switch_states(
        network=network,
        switch_df=switch_update_df,
    )


def find_branch(net: Network, elem_id: str, voltage_level_id: str, bus_id: str) -> tuple[bool, bool, Optional[str]]:
    """Find a branch connection in one station and resolve its local orientation.

    It will try to match the branch with two methods. First, the voltage level id is compared.
    However, there are branches which have the same voltage level id on both sides, in which case
    the bus_id is used as a fallback check. This is the case for example on PSTs.

    Parameters
    ----------
    net: Network
        The powsybl network to check the element in
    elem_id: str
        The id of the element to check
    voltage_level_id: str
        The voltage level id of the branch to find out the direction
    bus_id: str
        The id of the busbar to find out the direction, will only be used if the voltage
        level is ambiguous

    Returns
    -------
    bool
        Whether the element is connected.
    bool
        Whether side 1 is on the busbar that was passed in. This is always false if the element
        is not connected.
    Optional[str]
        The bus_breaker_id on the matched branch side. Might be None if the element is not connected.

    Raises
    ------
    ValueError
        If the branch is not found in the station.
    """
    branches_df = net.get_branches(
        attributes=[
            "voltage_level1_id",
            "voltage_level2_id",
            "connected1",
            "connected2",
            "bus1_id",
            "bus2_id",
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
        ]
    )
    if elem_id not in branches_df.index:
        raise ValueError(f"Branch {elem_id} not found in the station, assumed to be in {voltage_level_id}")

    branch_entry = branches_df.loc[elem_id]

    same_vl = branch_entry["voltage_level1_id"] == branch_entry["voltage_level2_id"]

    if (not same_vl and branch_entry["voltage_level1_id"] == voltage_level_id) or branch_entry["bus1_id"] == bus_id:
        return (
            bool(branch_entry["connected1"] and branch_entry["connected2"]),
            True,
            branch_entry["bus_breaker_bus1_id"],
        )

    if (not same_vl and branch_entry["voltage_level2_id"] == voltage_level_id) or branch_entry["bus2_id"] == bus_id:
        return (
            bool(branch_entry["connected1"] and branch_entry["connected2"]),
            False,
            branch_entry["bus_breaker_bus2_id"],
        )

    raise ValueError(f"Branch {elem_id} not found in the station, assumed to be in {voltage_level_id}")


def find_injection(net: Network, elem_id: str, voltage_level_id: str) -> tuple[bool, Optional[str]]:
    """Find an injection connection in one station.

    Parameters
    ----------
    net : Network
        Powsybl network to query.
    elem_id : str
        Injection id to locate.
    voltage_level_id : str
        Voltage level id expected for the injection.

    Returns
    -------
    tuple[bool, Optional[str]]
        Whether the injection is connected and the bus-breaker bus id on which
        it is currently placed.

    Raises
    ------
    ValueError
        If the injection is not present in the expected station.
    """
    injections_df = net.get_injections(attributes=["connected", "voltage_level_id", "bus_id", "bus_breaker_bus_id"])

    if elem_id not in injections_df.index or injections_df.loc[elem_id]["voltage_level_id"] != voltage_level_id:
        raise ValueError(f"Element {elem_id} not found in the station.")

    return bool(injections_df.loc[elem_id]["connected"]), injections_df.loc[elem_id]["bus_breaker_bus_id"]


def move_branch(net: Network, elem_id: str, bus_breaker_id: str, from_end: bool) -> None:
    """Move a branch to a new busbar in the network.

    Parameters
    ----------
    net : Network
        The powsybl network to update in place.
    elem_id : str
        The id of the branch to move.
    bus_breaker_id : str
        The id of the target bus-breaker bus.
    from_end : bool
        Whether the matched station side corresponds to branch end 1.
    """
    if from_end:
        net.update_branches(id=elem_id, bus_breaker_bus1_id=bus_breaker_id, connected1=True)
    else:
        net.update_branches(id=elem_id, bus_breaker_bus2_id=bus_breaker_id, connected2=True)


def disconnect_branch(
    net: Network,
    elem_id: str,
) -> None:
    """Disconnect a branch in the network.

    Parameters
    ----------
    net : Network
        The powsybl network to update in place.
    elem_id : str
        The id of the branch to disconnect.
    """
    net.update_branches(id=elem_id, connected1=False, connected2=False)


def move_injection(
    net: Network,
    elem_id: str,
    bus_breaker_id: str,
) -> None:
    """Move an injection to a new busbar in the network.

    Parameters
    ----------
    net : Network
        The powsybl network to update in place.
    elem_id : str
        The id of the injection to move.
    bus_breaker_id : str
        The id of the target bus-breaker bus.
    """
    net.update_injections(id=elem_id, bus_breaker_bus_id=bus_breaker_id, connected=True)


def disconnect_injection(
    net: Network,
    elem_id: str,
) -> None:
    """Disconnect an injection in the network.

    Parameters
    ----------
    net : Network
        The powsybl network to update in place.
    elem_id : str
        The id of the injection to disconnect.
    """
    net.update_injections(id=elem_id, connected=False)


def _get_target_bus_index(
    station: MaterializedStation,
    switching_column: np.ndarray,
    bus_breaker_id: Optional[str],
) -> int:
    """Resolve the target busbar index for one switching-table column.

    Parameters
    ----------
    station : MaterializedStation
        Station whose busbar order defines valid target indices.
    switching_column : np.ndarray
        One branch or injection switching-table column.
    bus_breaker_id : Optional[str]
        Current bus-breaker id of the asset, if known.

    Returns
    -------
    int
        Index of the target busbar. If the current bus-breaker id matches one of
        the active busbars, that index is preferred; otherwise the first active
        entry is used.
    """
    target_bus_indices = [
        index
        for index, (connected, busbar) in enumerate(zip(switching_column.tolist(), station.busbars, strict=True))
        if connected and busbar.grid_model_id == bus_breaker_id
    ]
    if len(target_bus_indices) == 1:
        return target_bus_indices[0]
    return int(np.argmax(switching_column))


def _apply_single_branch_bus_branch(
    net: Network,
    station: MaterializedStation,
    asset_index: int,
) -> tuple[Literal["disconnected", "reassigned", "nothing"], list[tuple[int, int, bool]]]:
    """Reassign or disconnect a single branch connection in a bus/branch topology.

    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station is fully present in the
        network.
    station : MaterializedStation
        The asset topology station in which the branch at position asset_index will be applied to the powsybl network.
    asset_index : int
        The index of the branch connection inside station.branch_connections.

    Returns
    -------
    Literal["disconnected", "reassigned", "nothing"]
        A string indicating whether the branch was disconnected, reassigned or left as is.
    list[tuple[int, int, bool]]
        A list of reassignment diffs for this branch.
    """
    vl_id = net.get_buses(attributes=["voltage_level_id"]).loc[station.grid_model_id]["voltage_level_id"]

    asset = station.branch_connections[asset_index].asset
    switching_column = station.branch_switching_table[:, asset_index]

    is_connected, from_side, bus_breaker_id = find_branch(
        net=net, elem_id=asset.grid_model_id, voltage_level_id=vl_id, bus_id=station.grid_model_id
    )

    if not np.any(switching_column):
        if is_connected:
            disconnect_branch(net=net, elem_id=asset.grid_model_id)
            return "disconnected", []
        return "nothing", []

    target_bus_index = _get_target_bus_index(station, switching_column, bus_breaker_id)
    target_bus_breaker_id = station.busbars[target_bus_index].grid_model_id
    if target_bus_breaker_id == bus_breaker_id:
        return "nothing", []

    move_branch(
        net=net,
        elem_id=asset.grid_model_id,
        bus_breaker_id=target_bus_breaker_id,
        from_end=from_side,
    )
    reassignments = [(asset_index, target_bus_index, True)]

    old_indices = [index for index, busbar in enumerate(station.busbars) if busbar.grid_model_id == bus_breaker_id]
    if len(old_indices) == 1:
        reassignments.append((asset_index, old_indices[0], False))

    return "reassigned", reassignments


def apply_single_branch_bus_branch(
    net: Network,
    station: MaterializedStation,
) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """Apply all branch connection updates for one station in bus/branch topology.

    Parameters
    ----------
    net : Network
        Powsybl network to modify in place.
    station : MaterializedStation
        Station whose branch-side topology should be realized.

    Returns
    -------
    tuple[list[int], list[tuple[int, int, bool]]]
        Disconnected branch indices and branch reassignment diffs.
    """
    branch_disconnection_diff: list[int] = []
    branch_reassignment_diff: list[tuple[int, int, bool]] = []

    for asset_index in range(len(station.branch_connections)):
        result, new_reassignments = _apply_single_branch_bus_branch(net, station, asset_index)
        if result == "disconnected":
            branch_disconnection_diff.append(asset_index)
        elif result == "reassigned":
            branch_reassignment_diff.extend(
                [(int(asset_index), int(busbar_index), bool(connected)) for _, busbar_index, connected in new_reassignments]
            )

    return branch_disconnection_diff, branch_reassignment_diff


def _apply_single_injection_bus_branch(
    net: Network,
    station: MaterializedStation,
    asset_index: int,
) -> tuple[Literal["disconnected", "reassigned", "nothing"], list[tuple[int, int, bool]]]:
    """Reassign or disconnect a single injection connection in a bus/branch topology.

    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station is fully present in the
        network.
    station : MaterializedStation
        The asset topology station in which the injection at position asset_index will be applied to the powsybl network.
    asset_index : int
        The index of the injection connection inside station.injection_connections.

    Returns
    -------
    Literal["disconnected", "reassigned", "nothing"]
        A string indicating whether the injection was disconnected, reassigned or left as is.
    list[tuple[int, int, bool]]
        A list of reassignment diffs for this injection.
    """
    vl_id = net.get_buses(attributes=["voltage_level_id"]).loc[station.grid_model_id]["voltage_level_id"]

    asset = station.injection_connections[asset_index].asset
    switching_column = station.injection_switching_table[:, asset_index]

    is_connected, bus_breaker_id = find_injection(net=net, elem_id=asset.grid_model_id, voltage_level_id=vl_id)

    if not np.any(switching_column):
        if is_connected:
            disconnect_injection(net=net, elem_id=asset.grid_model_id)
            return "disconnected", []
        return "nothing", []

    target_bus_index = _get_target_bus_index(station, switching_column, bus_breaker_id)
    target_bus_breaker_id = station.busbars[target_bus_index].grid_model_id
    if target_bus_breaker_id == bus_breaker_id:
        return "nothing", []

    move_injection(
        net=net,
        elem_id=asset.grid_model_id,
        bus_breaker_id=target_bus_breaker_id,
    )
    reassignments = [(asset_index, target_bus_index, True)]

    old_indices = [index for index, busbar in enumerate(station.busbars) if busbar.grid_model_id == bus_breaker_id]
    if len(old_indices) == 1:
        reassignments.append((asset_index, old_indices[0], False))

    return "reassigned", reassignments


def apply_single_injection_bus_branch(
    net: Network,
    station: MaterializedStation,
) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """Apply all injection connection updates for one station in bus/branch topology.

    Parameters
    ----------
    net : Network
        Powsybl network to modify in place.
    station : MaterializedStation
        Station whose injection-side topology should be realized.

    Returns
    -------
    tuple[list[int], list[tuple[int, int, bool]]]
        Disconnected injection indices and injection reassignment diffs.
    """
    injection_disconnection_diff: list[int] = []
    injection_reassignment_diff: list[tuple[int, int, bool]] = []

    for asset_index in range(len(station.injection_connections)):
        result, new_reassignments = _apply_single_injection_bus_branch(net, station, asset_index)
        if result == "disconnected":
            injection_disconnection_diff.append(asset_index)
        elif result == "reassigned":
            injection_reassignment_diff.extend(
                [(int(asset_index), int(busbar_index), bool(connected)) for _, busbar_index, connected in new_reassignments]
            )

    return injection_disconnection_diff, injection_reassignment_diff


def set_coupler(
    net: Network,
    coupler_id: str,
    target_state: bool,
) -> bool:
    """Set the state of a coupler in the network.

    Parameters
    ----------
    net : Network
        The powsybl network to switch the coupler in, modified in place.
    coupler_id : str
        The id of the coupler to switch, expected in ``net.get_switches()``.
    target_state : bool
        Target switch state, ``True`` for open and ``False`` for closed.

    Returns
    -------
    bool
        ``True`` if the coupler was switched, ``False`` if it already had the target state.

    Raises
    ------
    ValueError
        If the coupler is missing in the network.
    """
    switches_df = net.get_switches(attributes=["open"])
    if coupler_id not in switches_df.index:
        raise ValueError(f"Coupler {coupler_id} not found in the network")
    if switches_df.loc[coupler_id]["open"] == target_state:
        return False
    net.update_switches(id=coupler_id, open=target_state)
    return True


def apply_station_bus_branch(net: Network, station: MaterializedStation) -> AppliedStation:
    """Apply a station topology to a powsybl model in bus/branch format.

    This will assume that the substations are in bus/branch format and that the busbars in the station are the same as in
    the asset topology, just that the assets are not yet on their spot and the busbar couplers are not yet correctly switched

    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station name corresponds to a busbar id in the
        network (busbar in terms of net.get_buses()) and that the bus-breaker buses are in the same voltage level as that
        busbar. Furthermore, we assume to find the bus-breaker buses in the voltage level to represent the busbars in the
        asset topology. If there are more buses, these additional buses will be ignored, if there are fewer buses, an
        exception is raised. Will be modified in-place.
    station : MaterializedStation
        The asset topology station. The split branch and injection switching
        tables, plus the coupler states, are applied to the matched station in
        the powsybl grid.

    Returns
    -------
    AppliedStation
        The realized station object which contains the input station plus a diff of switched couplers, reassignments and
        disconnections.

    Raises
    ------
    ValueError
        If the station is not in the network or if some of the assets/busbars are not in the station
    """
    assert_station_in_network(net, station, couplers_strict=False, assets_strict=False, busbars_strict=False)

    coupler_diff = []
    branch_disconnection_diff, branch_reassignment_diff = apply_single_branch_bus_branch(net, station)
    injection_disconnection_diff, injection_reassignment_diff = apply_single_injection_bus_branch(net, station)

    for coupler in station.couplers:
        if set_coupler(net, coupler.grid_model_id, coupler.open):
            coupler_diff.append(coupler)

    return AppliedStation(
        station=station,
        coupler_diff=coupler_diff,
        branch_reassignment_diff=branch_reassignment_diff,
        injection_reassignment_diff=injection_reassignment_diff,
        branch_disconnection_diff=branch_disconnection_diff,
        injection_disconnection_diff=injection_disconnection_diff,
    )


def apply_topology_bus_branch(net: Network, topology: Topology) -> RealizedTopology:
    """Apply an asset topology to a network and return the diff.

    This takes an asset topology and applies it to the network. It will return the diff that it had to do to reach
    the asset topology in the form of a RealizedTopology


    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station names correspond to busbar ids in the
        network (busbar in terms of net.get_buses()) and that the bus-breaker buses are in the same voltage level as that
        busbar. Furthermore, we assume to find the bus-breaker buses in the voltage level to represent the busbars in the
        asset topology. If there are more buses, these additional buses will be ignored, if there are fewer buses, an
        exception is raised. Will be modified in-place.
    topology : Topology
        The asset topology to apply to the network. The stations in the topology will be applied to the network and the
        diff will be returned.

    Returns
    -------
    RealizedTopology
        The realized topology object containing the input topology plus the
        coupler, reassignment, and disconnection diffs.
    """
    realized_stations = [apply_station_bus_branch(net, station) for station in topology.materialize_stations()]

    (
        coupler_diff,
        branch_reassignment_diff,
        injection_reassignment_diff,
        branch_disconnection_diff,
        injection_disconnection_diff,
    ) = accumulate_diffs(realized_stations)

    return RealizedTopology(
        topology=topology,
        coupler_diff=coupler_diff,
        branch_reassignment_diff=branch_reassignment_diff,
        injection_reassignment_diff=injection_reassignment_diff,
        branch_disconnection_diff=branch_disconnection_diff,
        injection_disconnection_diff=injection_disconnection_diff,
    )


@pa.check_types
def apply_node_breaker_topology(net: Network, target_topology: Topology) -> pa.typing.DataFrame[SwitchUpdateSchema]:
    """Apply a node-breaker topology to a powsybl network.

    Parameters
    ----------
    net : Network
        The powsybl network to modify, will be modified in place.
    target_topology : Topology
        The target topology to apply to the network.

    Returns
    -------
    pa.typing.DataFrame[SwitchUpdateSchema]
        The dataframe of switches that were updated.
    """
    switch_update_df = get_changing_switches_from_topology(network=net, target_topology=target_topology)
    switch_update_df.rename(columns={"grid_model_id": "id"}, inplace=True)
    switch_update_df.set_index("id", inplace=True)
    # Update the network with the new switch states
    net.update_switches(switch_update_df)
    # reset id to be compliant with the schema
    switch_update_df.reset_index(inplace=True)
    switch_update_df.rename(columns={"id": "grid_model_id"}, inplace=True)
    return switch_update_df


def is_node_breaker_grid(net: Network, relevant_station: Optional[str] = None) -> bool:
    """Check if the network is in node-breaker format.

    Parameters
    ----------
    net : Network
        The powsybl network to check.
    relevant_station : Optional[str]
        The name of any relevant station to check. It is possible that a grid has a mix of node-breaker and bus/branch but
        the relevant stations should be all uniform, in one of the two formats.

    Returns
    -------
    bool
        True if the network is in node-breaker format, False otherwise.
    """
    bus = net.get_buses(attributes=["voltage_level_id"])
    if relevant_station is not None:
        bus = bus.loc[relevant_station]
    else:
        bus = bus.iloc[0]
    return (
        net.get_voltage_levels(attributes=["topology_kind"]).loc[bus["voltage_level_id"]]["topology_kind"] == "NODE_BREAKER"
    )


def apply_station(
    net: Network, topology: Topology, raw_station: RawStation
) -> Union[pa.typing.DataFrame[SwitchUpdateSchema], AppliedStation]:
    """Apply a station topology to a powsybl model.

    This will apply the station topology to the network. If the network is in bus/branch format, it will return the
    realized station. If the network is in node/breaker format, it will return the switch update dataframe.

    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station name corresponds to a busbar id in the
        network (busbar in terms of net.get_buses()) and that the bus-breaker buses are in the same voltage level as that
        busbar. Furthermore, we assume to find the bus-breaker buses in the voltage level to represent the busbars in the
        asset topology. If there are more buses, these additional buses will be ignored, if there are fewer buses, an
        exception is raised. Will be modified in-place.
    topology : Topology
        The owning topology that provides canonical assets and asset bays for the raw station.
    raw_station : RawStation
        The lean station view to apply. The switching state of the assets and busbar couplers shall be applied to the
        matched station in the powsybl grid.

    Returns
    -------
    Union[pa.typing.DataFrame[SwitchUpdateSchema], AppliedStation]
        The realized station object which contains the input station plus a diff of switched couplers, reassignments and
        disconnections or a dataframe of switches that were updated.

    Raises
    ------
    ValueError
        If the station cannot be matched into the expected powsybl representation.
    """
    station_topology = copy_topology_with_updates(
        topology,
        [raw_station],
        topology.asset_bays,
        branch_assets=topology.branch_assets,
        injection_assets=topology.injection_assets,
    )

    if is_node_breaker_grid(net=net, relevant_station=raw_station.grid_model_id):
        return apply_node_breaker_topology(
            net=net,
            target_topology=station_topology,
        )
    return apply_station_bus_branch(net=net, station=station_topology.materialize_stations()[0])
