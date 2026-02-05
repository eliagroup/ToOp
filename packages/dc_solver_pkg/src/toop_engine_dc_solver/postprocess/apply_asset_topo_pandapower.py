# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines to apply an asset topology to a pandapower model.

Fundamentally there are two ways to do it: A node/breaker way which just changes switch states and is equivalent to the .dgs
export and a bus/branch way which reassigns the elements to their new locations.
"""

from dataclasses import dataclass

import logbook
import numpy as np
import pandapower as pp
from beartype.typing import Iterable
from pandapower.toolbox import element_bus_tuples, get_connected_elements_dict
from toop_engine_grid_helpers.pandapower.pandapower_helpers import get_element_table, get_remotely_connected_buses
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import parse_globally_unique_id, table_id
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    RealizedStation,
    RealizedTopology,
    Station,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import accumulate_diffs, find_busbars_for_coupler

logger = logbook.Logger(__name__)


def reassign_asset_to_bus(
    net: pp.pandapowerNet,
    asset_id: int,
    asset_table: str,
    target_bus_id: int,
    station_buses: Iterable[int],
) -> int:
    """Reassign an asset to a new bus in the station.

    Assumes the direction of the asset is not known and will be inferred by finding out which bus in station_buses it is
    currently connected to (raising if it's connected to none or multiple buses).

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    asset_id : int
        The id of the asset to reassign where the id refers to the index of the asset_table in the pandapower network.
    asset_table : str
        The table of the asset to reassign.
    target_bus_id : int
        The id of the bus to reassign the asset to where the id refers to the index of the bus in the pandapower network
        (net.buses table).
    station_buses : list[int]
        The list of bus ids in the station to which the asset might be connected. It is assumed that the asset is only
        connected to one of these buses.

    Returns
    -------
    int
        The id of the bus to which the asset was connected before the reassignment, referring to the pandapower net.buses
        table.

    Raises
    ------
    ValueError
        If the asset is not connected to any bus in the station.
    """
    ebts = [(table, bus_col) for (table, bus_col) in element_bus_tuples() if table == asset_table]

    connected_bus = None
    for table, bus_col in ebts:
        asset_bus = net[table].loc[asset_id, bus_col]
        if asset_bus in station_buses:
            connected_bus = asset_bus
            break

    if connected_bus is None:
        raise ValueError(f"Asset {asset_id} is not connected to any bus in the station.")

    net[table].loc[asset_id, bus_col] = target_bus_id
    return int(connected_bus)


def create_missing_busbars(
    net: pp.pandapowerNet,
    busbars: list[Busbar],
) -> list[Busbar]:
    """Create missing busbars in a station.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    busbars : list[Busbar]
        The busbars from the asset topology station that are expected to be present
        in the pandapower network.

    Returns
    -------
    list[Busbar]
        A list of busbars that were created in the pandapower network.

    Raises
    ------
    ValueError
        If a busbar with the same id already exists in the pandapower network, but in a different station.
    """
    asset_busbar_set = set(table_id(bus.grid_model_id) for bus in busbars)
    assert len(asset_busbar_set) == len(busbars), "Busbar ids are not unique"
    station_buses = get_remotely_connected_buses(net=net, buses=asset_busbar_set, consider=("s",), respect_switches=False)

    station_voltages = net.bus.loc[list(station_buses), "vn_kv"].unique()
    assert len(station_voltages) == 1, "Busbars with different voltages are conneted via a switch"

    # Create missing busbars
    created = []
    for busbar in busbars:
        busbar_id = table_id(busbar.grid_model_id)
        if busbar_id not in station_buses:
            if busbar_id in net.bus.index:
                raise ValueError(f"Busbar {busbar_id} is part of the wrong station")
            # We have to use .loc instead of the creation routine as we need the specific bus id
            # and not the next available one
            net.bus.loc[busbar_id] = {
                "name": busbar.name,
                "vn_kv": station_voltages[0],
                "in_service": busbar.in_service,
                "type": "b",
            }
            created.append(busbar)

    return created


def create_missing_switches(
    net: pp.pandapowerNet, busbars: list[Busbar], couplers: list[BusbarCoupler]
) -> list[BusbarCoupler]:
    """Create missing busbar couplers in a station.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    busbars : list[Busbar]
        The busbars from the asset topology station. This function assumes that they all have a corresponding bus in the
        pandapower network, use create_missing_busbars to make sure this is the case before calling this function.
    couplers : list[BusbarCoupler]
        The busbar couplers from the asset topology station that are expected to be present
        in the pandapower network. It will create missing busbar couplers in the pandapower network.

    Returns
    -------
    list[BusbarCoupler]
        A list of busbar couplers that were created in the pandapower network.

    Raises
    ------
    ValueError
        If a busbar does not exist in the pandapower network.
    ValueError
        If a busbar coupler with the same id already exists in the pandapower network, but is connected to different busbars
        than expected.
    """
    created = []
    for coupler in couplers:
        from_busbar, to_busbar = find_busbars_for_coupler(
            busbars=busbars,
            coupler=coupler,
        )
        from_busbar_id = table_id(from_busbar.grid_model_id)
        to_busbar_id = table_id(to_busbar.grid_model_id)
        if from_busbar_id not in net.bus.index or to_busbar_id not in net.bus.index:
            raise ValueError(f"Busbar {from_busbar_id} or {to_busbar_id} is not in the pandapower network.")

        coupler_id = table_id(coupler.grid_model_id)
        if coupler_id not in net.switch.index:
            net.switch.loc[coupler_id] = {
                "name": coupler.name,
                "et": "b",
                "closed": not coupler.open,
                "bus": from_busbar_id,
                "element": to_busbar_id,
                "type": "LBS",
            }
            created.append(coupler)

        if (net.switch.loc[coupler_id, "bus"] not in [from_busbar_id, to_busbar_id]) or (
            net.switch.loc[coupler_id, "element"] not in [from_busbar_id, to_busbar_id]
        ):
            raise ValueError(f"Busbar coupler {coupler_id} is connected to the wrong busbars.")

    return created


def delete_excess_busbars(
    net: pp.pandapowerNet,
    busbars: list[Busbar],
) -> list[int]:
    """Delete excess busbars in a station.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    busbars : list[Busbar]
        The busbars from the asset topology station that are expected to be present
        in the pandapower network.

    Returns
    -------
    list[int]
        A list of busbars that were deleted from the pandapower network, where the id is the index of the busbar in the
        pandapower net.buses table.
    """
    asset_busbar_set = set(table_id(bus.grid_model_id) for bus in busbars)
    station_buses = get_remotely_connected_buses(net=net, buses=asset_busbar_set, consider=("s",), respect_switches=False)

    # Delete excess busbars, but not connected elements. We expect switches to be still connected
    # and we want to delete them separately
    deleted = []
    for bus_id in station_buses:
        if bus_id not in asset_busbar_set:
            net.bus.drop(index=bus_id, inplace=True)
            deleted.append(bus_id)

    return deleted


def delete_excess_switches(
    net: pp.pandapowerNet,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
) -> list[int]:
    """Delete excess switches from the pandapower network

    Finds all switches connected somewhere to the substation and deletes all that are not in the list of couplers

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    busbars : list[Busbar]
        The busbars from the asset topology station that are expected to be present
        in the pandapower network. Only used to collect switches
    couplers : list[BusbarCoupler]
        The busbar couplers from the asset topology station that are expected to be present
        in the pandapower network. This will be enforced

    Returns
    -------
    list[int]
        A list of switches that were deleted from the pandapower network, where the id is the index of the switch in the
        pandapower net.switch table.
    """
    asset_busbar_set = set(table_id(bus.grid_model_id) for bus in busbars)
    station_buses = get_remotely_connected_buses(net=net, buses=asset_busbar_set, consider=("s",), respect_switches=False)
    connected_switches = net.switch[
        net.switch["bus"].isin(station_buses) | (net.switch["element"].isin(station_buses) & net.switch["et"] == "b")
    ].index

    # Delete excess switches
    expected_switches = [table_id(coupler.grid_model_id) for coupler in couplers]
    deleted = []
    for switch_id in connected_switches:
        if switch_id not in expected_switches:
            # Delete the switch
            net.switch.drop(index=switch_id, inplace=True)
            deleted.append(switch_id)

    return deleted


def apply_station_assets(
    net: pp.pandapowerNet,
    station: Station,
) -> tuple[list[int], list[tuple[int, int, bool]]]:
    """Apply a station topology to a pandapower network.

    This assumes that exactly the assets in the station also exist in the pandapower network.
    If the assets are connected via asset bays and switches before the application of this function,
    these asset bays will be dangling after.

    It returns diffs compatible with the RealizedStation format.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    station : Station
        The station to apply. It is assumed that the station grid_model_id refers to a bus in the pandapower network.

    Returns
    -------
    list[int]
        A list of asset indices that were disconnected from the grid.
    list[tuple[int, int, bool]]
        A list of tuples containing the asset index, the bus id to which it was assigned/unassigned and a boolean indicating
        whether the asset was assigned (true) or unassigned (false).
    """
    assert all(table_id(bus.grid_model_id) in net.bus.index for bus in station.busbars), (
        "All busbars must be present in the pandapower network."
    )

    station_buses = get_remotely_connected_buses(
        net=net, buses=[table_id(bus.grid_model_id) for bus in station.busbars], consider=("s",), respect_switches=False
    )

    disconnection_diff = []
    reassignment_diff = []
    for asset_index, asset in enumerate(station.assets):
        pp_id, asset_type = parse_globally_unique_id(asset.grid_model_id)
        pp_table = get_element_table(asset_type)
        pp_id = int(pp_id)
        # Note that this is an index into the asset switching table, not into the pandapower net.buses table
        target_buses: list[int] = np.flatnonzero(station.asset_switching_table[:, asset_index]).tolist()

        if len(target_buses) > 1:
            raise NotImplementedError("Connecting an asset to multiple buses is not supported.")

        if len(target_buses) == 0:
            # We are supposed to disconnect the asset, we do that by setting the in_service flag to False
            net[pp_table].loc[pp_id, "in_service"] = False
            disconnection_diff.append(asset_index)
        else:
            # We shall reassign the asset to a new bus
            # Note that the target_bus_index indexes into the asset switching table and the target_bus_id indexes into the
            # pandapower net.buses table
            target_bus_index = target_buses[0]
            target_bus_id = table_id(station.busbars[target_bus_index].grid_model_id)
            previous_bus = reassign_asset_to_bus(
                net=net,
                asset_id=pp_id,
                asset_table=pp_table,
                target_bus_id=target_bus_id,
                station_buses=station_buses,
            )
            if previous_bus != target_bus_id:
                reassignment_diff.append((asset_index, target_bus_index, True))
                try:
                    # It could be that the asset is connected to a busbar that is not in the station but in the pandapower
                    # grid. In that case, we don't know where it came from.
                    previous_bus_index = [table_id(bus.grid_model_id) for bus in station.busbars].index(previous_bus)
                except ValueError:
                    previous_bus_index = -1
                    logger.warning(
                        f"Asset {asset.grid_model_id} was reassigned from bus {previous_bus} which is not in the station."
                    )
                reassignment_diff.append((asset_index, previous_bus_index, False))

    return disconnection_diff, reassignment_diff


def apply_station_couplers(
    net: pp.pandapowerNet,
    couplers: list[BusbarCoupler],
) -> list[BusbarCoupler]:
    """Apply coupler changes from an asset topology station to a pandapower network

    This will find all couplers and change them to their desired state. If they have been switched, they will be returned
    as a coupler diff.

    This expects all couplers to be present in the grid and to be connected to the correct busbars.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    couplers : list[BusbarCoupler]
        The desired coupler states.

    Returns
    -------
    list[BusbarCoupler]
        A list of couplers that were switched in the pandapower network.

    Raises
    ------
    ValueError
        If a coupler does not exist in the pandapower network.
    """
    changed_couplers = []
    for coupler in couplers:
        coupler_id = table_id(coupler.grid_model_id)
        if coupler_id not in net.switch.index:
            raise ValueError(f"Coupler {coupler_id} is not in the pandapower network.")
        target_state = not coupler.open
        if net.switch.loc[coupler_id, "closed"] != target_state:
            net.switch.loc[coupler_id, "closed"] = target_state
            changed_couplers.append(coupler)
    return changed_couplers


@dataclass
class ApplyGridDiff:
    """Holds the information about the difference between the asset topology and the pandapower network.

    The difference between the switches and busbars that were expected by the asset topology and the actual switches
    and busbars in the grid. This diff is created when using apply_station on a grid that does not exactly match the asset
    topology.
    """

    busbars_created: list[Busbar]
    """The busbars that were present in the asset topology but not present in the grid. Represented by the busbar object"""

    switches_created: list[BusbarCoupler]
    """The switches that were present in the asset topology but not present in the grid. Represented by the busbar coupler
    object"""

    busbars_deleted: list[int]
    """The busbars that were present in the grid but not present in the asset topology. Represented by the index of the
    busbar in the pandapower network prior to deletion."""

    switches_deleted: list[int]
    """The switches that were present in the grid but not present in the asset topology. Represented by the index of the
    switch in the pandapower network prior to deletion."""


def apply_station(
    net: pp.pandapowerNet,
    station: Station,
) -> tuple[ApplyGridDiff, RealizedStation]:
    """Apply an asset topology station to a pandapower network.

    This will force the station into the format of the asset topology, meaning missing busbars and switches will be created,
    excess busbars and switches will be deleted, the asset_switching_table will be applied and the couplers will be set to
    their desired state.

    The asset bays and coupler bays in the station will not be recreated, i.e. if a node/breaker model is passed into this
    function, it will return as a bus/branch model.

    If there are assets missing, it will raise, if there are additional assets in the station the function has undefined
    behaviour.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    station : Station
        The station to apply. It is assumed that the station grid_model_id refers to a bus in the pandapower network.

    Returns
    -------
    ApplyGridDiff
        The difference between the switches and busbars that were expected by the asset topology and the actual switches
        and busbars in the grid.
    RealizedStation
        The realized station, containing the coupler diff, reassignment diff and disconnection diff.
    """
    busbars_created = create_missing_busbars(
        net=net,
        busbars=station.busbars,
    )
    switches_created = create_missing_switches(
        net=net,
        busbars=station.busbars,
        couplers=station.couplers,
    )

    disconnection_diff, reassignment_diff = apply_station_assets(
        net=net,
        station=station,
    )

    coupler_diff = apply_station_couplers(
        net=net,
        couplers=station.couplers,
    )

    # Delete excess elements
    # This has to happen after apply_station_assets because before there could be assets connected to the deleted
    # busbars that we still want to reassign. Deleting the busbar would bring these assets into a state where the busbar
    # they connect to does not exist anymore.
    busbars_deleted = delete_excess_busbars(
        net=net,
        busbars=station.busbars,
    )
    switches_deleted = delete_excess_switches(
        net=net,
        busbars=station.busbars,
        couplers=station.couplers,
    )

    # After the deletion, we expect to have nothing in the grid connected to the busbars that were deleted
    residual_connected_elements = get_connected_elements_dict(
        net=net,
        buses=busbars_deleted,
    )
    for key, value in residual_connected_elements.items():
        if len(value):
            raise ValueError(
                f"Residual connected {key} after deletion: {value}"
                "This can be the case when an asset is present in the grid but not the asset topology and"
                "hence was not reassigned to a valid bus"
            )

    return (
        ApplyGridDiff(
            busbars_created=busbars_created,
            switches_created=switches_created,
            busbars_deleted=busbars_deleted,
            switches_deleted=switches_deleted,
        ),
        RealizedStation(
            station=station,
            coupler_diff=coupler_diff,
            reassignment_diff=reassignment_diff,
            disconnection_diff=disconnection_diff,
        ),
    )


def apply_topology(net: pp.pandapowerNet, topology: Topology) -> tuple[list[tuple[str, ApplyGridDiff]], RealizedTopology]:
    """Apply an asset topology to a pandapower network.

    This will apply all stations in the topology to the pandapower network. It will create missing busbars and switches,
    delete excess busbars and switches, apply the asset_switching_table and set the couplers to their desired state.

    If switches or busbars had to be adjusted, this is returned separately.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to apply the topology to. Will be modified in place.
    topology : Topology
        The topology to apply.

    Returns
    -------
    list[tuple[str, ApplyGridDiff]]
        A list of tuples containing the station id and the difference between the switches and busbars that were expected by
        the asset topology and the actual switches and busbars in the grid.
    RealizedTopology
        The realized topology, containing the coupler diff, reassignment diff and disconnection diff for each station.
    """
    realizations = [apply_station(net, station) for station in topology.stations]
    apply_diffs = [(rs.station.grid_model_id, apply_diff) for apply_diff, rs in realizations]
    realized_stations = [rs for _, rs in realizations]

    coupler_diff, reassignment_diff, disconnection_diff = accumulate_diffs(realized_stations)

    return apply_diffs, RealizedTopology(
        topology=topology,
        coupler_diff=coupler_diff,
        reassignment_diff=reassignment_diff,
        disconnection_diff=disconnection_diff,
    )
