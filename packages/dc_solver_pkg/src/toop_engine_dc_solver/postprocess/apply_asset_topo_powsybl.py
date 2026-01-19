# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines to apply an asset topology to a powsybl model.

Fundamentally there are two ways to do it: A node/breaker way which just changes switch states and is equivalent to the .dgs
export and a bus/branch way which reassigns the elements to their new locations.

For the bus/branch way, see the function apply_topology_bus_branch.
the node/breaker way is still TODO.
"""

from datetime import datetime

import numpy as np
import pandera as pa
from beartype.typing import Literal, Optional, Union
from pypowsybl.network import Network
from toop_engine_dc_solver.export.asset_topology_to_dgs import (
    SwitchUpdateSchema,
    get_changing_switches_from_topology,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import assert_station_in_network
from toop_engine_interfaces.asset_topology import (
    RealizedStation,
    RealizedTopology,
    Station,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import accumulate_diffs


def find_asset(net: Network, elem_id: str, voltage_level_id: str, bus_id: str) -> tuple[bool, bool, bool, Optional[str]]:
    """Find out whether a given element is a branch, if its connected and if branch, which direction it has.

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
        Whether the element is a branch
    bool
        Whether the element is connected.
    bool
        Whether side 1 is on the busbar that was passed in. This is always false if the element
        is not a branch or if it is not connected.
    Optional[str]
        The bus_breaker_id that the element is connected to. If the element is a branch it refers
        to the side that was found. Might be None if the element is not connected.

    Raises
    ------
    ValueError
        If the element is a branch and connected, but not found in the station
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
        injections_df = net.get_injections(attributes=["connected", "voltage_level_id", "bus_id", "bus_breaker_bus_id"])

        if elem_id not in injections_df.index or injections_df.loc[elem_id]["voltage_level_id"] != voltage_level_id:
            raise ValueError(f"Element {elem_id} not found in the station.")

        return False, bool(injections_df.loc[elem_id]["connected"]), False, injections_df.loc[elem_id]["bus_breaker_bus_id"]

    branch_entry = branches_df.loc[elem_id]

    same_vl = branch_entry["voltage_level1_id"] == branch_entry["voltage_level2_id"]

    if (not same_vl and branch_entry["voltage_level1_id"] == voltage_level_id) or branch_entry["bus1_id"] == bus_id:
        return (
            True,
            bool(branch_entry["connected1"] and branch_entry["connected2"]),
            True,
            branch_entry["bus_breaker_bus1_id"],
        )

    if (not same_vl and branch_entry["voltage_level2_id"] == voltage_level_id) or branch_entry["bus2_id"] == bus_id:
        return (
            True,
            bool(branch_entry["connected1"] and branch_entry["connected2"]),
            False,
            branch_entry["bus_breaker_bus2_id"],
        )

    raise ValueError(f"Branch {elem_id} not found in the station, assumed to be in {voltage_level_id}")


def move_branch(net: Network, elem_id: str, bus_breaker_id: str, from_end: bool) -> None:
    """Move a branch to a new busbar in the network

    Parameters
    ----------
    net : Network
        The powsybl network to disconnect the injection in
    elem_id : str
        The id of the injection to disconnect
    bus_breaker_id : str
        The id of the busbar to move the injection to
    from_end : bool
        Whether the branch is connected "from" side the busbar or "to" side the busbar.
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
        The powsybl network to disconnect the injection in
    elem_id : str
        The id of the injection to disconnect
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
        The powsybl network to disconnect the injection in
    elem_id : str
        The id of the injection to disconnect
    bus_breaker_id : str
        The id of the busbar to move the injection to
    """
    net.update_injections(id=elem_id, bus_breaker_bus_id=bus_breaker_id, connected=True)


def disconnect_injection(
    net: Network,
    elem_id: str,
) -> None:
    """Disconnects an injection in the network

    Parameters
    ----------
    net : Network
        The powsybl network to disconnect the injection in
    elem_id : str
        The id of the injection to disconnect
    """
    net.update_injections(id=elem_id, connected=False)


def apply_single_asset_bus_branch(
    net: Network,
    station: Station,
    asset_index: int,
) -> tuple[Literal["disconnected", "reassigned", "nothing"], list[tuple[int, int, bool]]]:
    """Reassign or disconnect a single asset in a bus/branch topology

    If the asset has no connections in the switching table, it is set to be disconnected. If it isn't already
    disconnected, it will be disconnected in the net and 'disconnected' will be returned.

    If the asset has one or more switching table entries, but none of them matches the current bus_breaker assignment,
    the asset needs to be reassigned. In this case it will be reassigned to the first busbar that is True in the switching
    table and 'reassigned' will be returned.

    If neither a disconnection or a reassignment is needed, 'nothing' will be returned and the resulting powsybl network
    will be the same as before.

    Parameters
    ----------
    net : Network
        The powsybl network to apply the topology to. It is assumed that the station is fully present in the
        network.
    station : Station
        The asset topology station in which the asset at position asset_index will be applied to the powsybl network.
    asset_index : int
        The index of the asset in the station.assets list.

    Returns
    -------
    Literal["disconnected", "reassigned", "nothing"]
        A string indicating whether the asset was disconnected, reassigned or left as is.
    list[tuple[int, int, bool]]
        A list of reassignments that have been made. Each tuple contains the asset index that was
        affected (not the asset grid_model_id but the index into the asset_switching_table), the busbar
        index (again the index into the switching table) and whether the asset was connected (True) or
        disconnected (False) to that busbar. Note that this list will have either 0, 1 or 2 entries and
        the first entry of the tuple, the asset index, will be the same as the asset_index passed in.
        If no reassignments happened, this will be the empty list.

    Raises
    ------
    ValueError
        If the asset is not in the station or if the asset is not in the network
    """
    vl_id = net.get_buses(attributes=["voltage_level_id"]).loc[station.grid_model_id]["voltage_level_id"]

    is_branch, is_connected, from_side, bus_breaker_id = find_asset(
        net=net, elem_id=station.assets[asset_index].grid_model_id, voltage_level_id=vl_id, bus_id=station.grid_model_id
    )

    switching_column = station.asset_switching_table[:, asset_index]

    target_disconnected = not np.any(switching_column)

    # If the bus_breaker_id of the asset is currently set in the switching table, use
    # that as the target bus_breaker_id. Otherwise, just use the first True entry.
    target_bus_indices = [
        index
        for index, (connected, busbar) in enumerate(zip(switching_column.tolist(), station.busbars, strict=True))
        if connected and busbar.grid_model_id == bus_breaker_id
    ]
    if len(target_bus_indices) == 1:
        target_bus_index = target_bus_indices[0]
    else:
        target_bus_index = int(np.argmax(switching_column))

    if target_disconnected and is_connected:
        if is_branch:
            disconnect_branch(net=net, elem_id=station.assets[asset_index].grid_model_id)
        else:
            disconnect_injection(net=net, elem_id=station.assets[asset_index].grid_model_id)
        return "disconnected", []
    if station.busbars[target_bus_index].grid_model_id != bus_breaker_id:
        if is_branch:
            move_branch(
                net=net,
                elem_id=station.assets[asset_index].grid_model_id,
                bus_breaker_id=station.busbars[target_bus_index].grid_model_id,
                from_end=from_side,
            )
        else:
            move_injection(
                net=net,
                elem_id=station.assets[asset_index].grid_model_id,
                bus_breaker_id=station.busbars[target_bus_index].grid_model_id,
            )
        reassignments = [(asset_index, target_bus_index, True)]

        old_indices = [index for index, busbar in enumerate(station.busbars) if busbar.grid_model_id == bus_breaker_id]
        if len(old_indices) == 1:
            reassignments.append((asset_index, old_indices[0], False))

        return "reassigned", reassignments
    return "nothing", []


def set_coupler(
    net: Network,
    coupler_id: str,
    target_state: bool,
) -> bool:
    """Set the state of a coupler in the network

    Parameters
    ----------
    net : Network
        The powsybl network to switch the coupler in, will be modified in-place
    coupler_id : str
        The id of the coupler to switch, should be in net.get_switches
    target_state : bool
        The target state of the coupler, True for open, False for closed

    Returns
    -------
    bool
        True if the coupler was switched, False if it was already in the target state
    """
    switches_df = net.get_switches(attributes=["open"])
    if coupler_id not in switches_df.index:
        raise ValueError(f"Coupler {coupler_id} not found in the network")
    if switches_df.loc[coupler_id]["open"] == target_state:
        return False
    net.update_switches(id=coupler_id, open=target_state)
    return True


def apply_station_bus_branch(net: Network, station: Station) -> RealizedStation:
    """Apply a station topology to a powsybl model in bus/branch format

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
    station : Station
        The asset topology station. The switching state of the assets and busbar couplers shall be applied to the matched
        station in the powsybl grid

    Returns
    -------
    RealizedStation
        The realized station object which contains the input station plus a diff of switched couplers, reassignments and
        disconnections.

    Raises
    ------
    ValueError
        If the station is not in the network or if some of the assets/busbars are not in the station
    """
    assert_station_in_network(net, station, couplers_strict=False, assets_strict=False, busbars_strict=False)

    disconnection_diff = []
    reassignment_diff = []
    coupler_diff = []

    for asset_index in range(len(station.assets)):
        result, new_reassignments = apply_single_asset_bus_branch(net, station, asset_index)
        if result == "disconnected":
            disconnection_diff.append(asset_index)
        elif result == "reassigned":
            reassignment_diff.extend(new_reassignments)

    for coupler in station.couplers:
        if set_coupler(net, coupler.grid_model_id, coupler.open):
            coupler_diff.append(coupler)

    return RealizedStation(
        station=station,
        disconnection_diff=disconnection_diff,
        reassignment_diff=reassignment_diff,
        coupler_diff=coupler_diff,
    )


def apply_topology_bus_branch(net: Network, topology: Topology) -> RealizedTopology:
    """Apply an asset topology to a network and return the diff

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
        The realized topology object which contains the input topology plus a diff of switched couplers, reassignments and
        disconnections.
    """
    realized_stations = [apply_station_bus_branch(net, station) for station in topology.stations]

    coupler_diff, reassignment_diff, disconnection_diff = accumulate_diffs(realized_stations)

    return RealizedTopology(
        topology=topology,
        coupler_diff=coupler_diff,
        reassignment_diff=reassignment_diff,
        disconnection_diff=disconnection_diff,
    )


@pa.check_types
def apply_node_breaker_topology(net: Network, target_topology: Topology) -> pa.typing.DataFrame[SwitchUpdateSchema]:
    """Apply a node-breaker Topology to a powsybl network

    Parameters
    ----------
    net : Network
        The powsybl network to modify, will be modified in place.
    target_topology : Topology
        The target topology to apply to the network, i.e. how you want the network to look like

    Returns
    -------
    switch_update_df: SwitchUpdateSchema
        The dataframe of the switches that were updated. This is the same as the one returned by get_diff_switch_states
        but with the index set to the switch id.
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


def is_node_breaker_grid(net: Network, relevant_station: str) -> bool:
    """Check if the network is in node-breaker format

    Parameters
    ----------
    net : Network
        The powsybl network to check
    relevant_station : str
        The name of any relevant station to check. It is possible that a grid has a mix of node-breaker and bus/branch but
        the relevant stations should be all uniform, in one of the two formats.

    Returns
    -------
    bool
        True if the network is in node-breaker format, False otherwise.
    """
    bus = net.get_buses(attributes=["voltage_level_id"]).loc[relevant_station]
    return (
        net.get_voltage_levels(attributes=["topology_kind"]).loc[bus["voltage_level_id"]]["topology_kind"] == "NODE_BREAKER"
    )


def apply_station(net: Network, station: Station) -> Union[pa.typing.DataFrame[SwitchUpdateSchema], RealizedStation]:
    """Apply a station topology to a powsybl model

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
    station : Station
        The asset topology station. The switching state of the assets and busbar couplers shall be applied to the matched
        station in the powsybl grid

    Returns
    -------
    Union[pa.typing.DataFrame[SwitchUpdateSchema], RealizedStation]
        The realized station object which contains the input station plus a diff of switched couplers, reassignments and
        disconnections or a dataframe of switches that were updated.
    """
    if is_node_breaker_grid(net=net, relevant_station=station.grid_model_id):
        return apply_node_breaker_topology(
            net=net,
            target_topology=Topology(
                topology_id="this_id_will_be_ignored",
                stations=[station],
                timestamp=datetime.now(),
            ),
        )
    return apply_station_bus_branch(net=net, station=station)
