# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to process pandapower node/breaker models.

File: pandapower_toolset_node_breaker.py
Author:  Benjamin Petrick
Created: 2024-11-21

This diagram shows the structure of a substation with two branches, two busbar couplers and two cross couplers.
It is referred to in some functions in this module.
The variable substation_bus_list is referring to all buses/nodes,
which for instance includes the nodes where the branch is connected to + the two following nodes where the
switches are connected to.
```
DS- Disconnector
BB- Busbar (type b)
BC- Busbar coupler
CC- Cross coupler
PS- Power switch / Branch switch
                        Branch 1                         Branch 2
                            |                                    |
                            / DS                                 / DS
                            |                                    |
                            |                                    |
                            / CB                                 / CB
                            |                                    |
                        ____|                                ____|
                        |   |                               |   |
                        |   / DS                            |   / DS
                        |   |              CC 1/3           |   |
    BB 1----------------|----------- _______/______ --------|-------------- BB 3
            |           |                                   |          |
        DS /       DS /                                 DS /       DS /
            |           |             CC 2/4                |          |
    BB 2----|-----------/--------- _______/________ -------------------|--------BB 4
        |   |                                                          |   |
     DS /   |                                                          |   / DS
        |   |                                                          |   |
        |_/_|                                                          |_/_|
        BC 1/2                                                         BC 3/4
```
"""

from typing import Optional, Union

import logbook
import numpy as np
import pandapower as pp
import pandas as pd
from toop_engine_grid_helpers.pandapower.pandapower_import_helpers import move_elements_based_on_labels

logger = logbook.Logger(__name__)


def get_type_b_nodes(network: pp.pandapowerNet, substation_bus_list: Optional[list[int]] = None) -> pd.DataFrame:
    """Get all nodes of type 'b' (busbar) in a network or substation.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the busbars from.
    substation_bus_list: Optional[list[int]]
        The bus ids of the substation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all busbars of type 'b' in the substation.
    """
    if substation_bus_list is None:
        substation_bus_list = network.bus.index
    substation_buses = network.bus.loc[substation_bus_list]
    bus_type_b = substation_buses[substation_buses.type == "b"]
    return bus_type_b


# TODO: replace by networkX_logic_modules
def get_indirect_connected_switch(
    net: pp.pandapowerNet,
    bus_1: int,
    bus_2: int,
    only_closed_switches: bool = True,
    consider_three_buses: bool = False,
    exclude_buses: Optional[list[int]] = None,
) -> dict[str, list[int]]:
    """Get a switch, that is indirectly connected by two buses and only by two buses.

    This function will only return the indirect connection between two buses.
    e.g. switchB or any switch that is parallel to switchB.
    ```
    busA---switchA---busB---switchB---busC---switchC---busD
    ```
    Note: this function will also return an empty dict for bus1 and bus3.
    Note: this function will return an empty dict if e.g. switch1 & bus2 are missing.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to get the indirect connections from.
    bus_1: int
        The bus to get the indirect connections from.
    bus_2: int
        The bus to get the indirect connections to.
    only_closed_switches: bool
        If True, only closed switches are considered.
    consider_three_buses: bool
        If True, the function will also consider three buses in between.
        Bus1---switch1---bus2---switch2---bus3---switch3---bus4---switch4---bus5
    exclude_buses: Optional[list[int]]
        The buses to exclude from the indirect connection.
        e.g. give all other busbars (type b) in the substation.

    Returns
    -------
    dict[str, list[int]]
        A dictionary with the indirect connections from bus_1 to bus_2

    Raises
    ------
    ValueError
        If the indirect connection contains more than one switch.
        e.g. a parallel line to the switch.
    """
    if exclude_buses is None:
        exclude_buses = [bus_1, bus_2]
    bus_1_connected = list(pp.toolbox.get_connected_buses(net, [bus_1], respect_switches=only_closed_switches, consider="s"))
    bus_1_connected = [el for el in bus_1_connected if el not in exclude_buses]
    bus_2_connected = list(pp.toolbox.get_connected_buses(net, [bus_2], respect_switches=only_closed_switches, consider="s"))
    bus_2_connected = [el for el in bus_2_connected if el not in exclude_buses]

    indirect_connection = pp.toolbox.get_connecting_branches(net, bus_1_connected, bus_2_connected)
    if consider_three_buses:
        indirect_connection_3 = get_indirect_connected_switches_three_buses(
            net,
            bus_1,
            bus_2,
            bus_1_connected,
            bus_2_connected,
            only_closed_switches,
            exclude_buses,
        )
        if "switch" in indirect_connection:
            indirect_connection["switch"] = indirect_connection["switch"] | set(indirect_connection_3["switch"])
        else:
            indirect_connection["switch"] = set(indirect_connection_3["switch"])

    indirect_connection = {
        key: list(indirect_connection[key])
        for key in indirect_connection
        if len(indirect_connection[key]) > 0 or key == "switch"
    }
    # filter only closed switches in the indirect connection
    closed_switches = []
    if "switch" in indirect_connection and only_closed_switches:
        for switch_id in indirect_connection["switch"]:
            if net.switch.loc[switch_id].closed:
                closed_switches.append(switch_id)
        indirect_connection["switch"] = closed_switches
        if len(indirect_connection["switch"]) == 0:
            del indirect_connection["switch"]
    if ("switch" in indirect_connection and len(indirect_connection) != 1) or (
        "switch" not in indirect_connection and len(indirect_connection) > 0
    ):
        error_value = [f"{key!s}:{value!s}" for key, values in indirect_connection.items() for value in values]
        raise ValueError(
            f"Indirect connection between bus {bus_1} and {bus_2} must contain only switches {' '.join(error_value)}"
        )
    return indirect_connection


# TODO: replace by networkX_logic_modules
def get_indirect_connected_switches_three_buses(
    net: pp.pandapowerNet,
    bus_1: int,
    bus_2: int,
    bus_1_connected: list[int],
    bus_2_connected: list[int],
    only_closed_switches: bool = True,
    exclude_buses: Optional[list[int]] = None,
) -> dict[str, list[int]]:
    """Get both switches that indirectly connect two buses with exactly three buses in between.

    This function will only return the indirect connection between two buses.
    e.g. switchB and switchC or parallel switches.
    busA---switchA---busB---switchB---busC---switchC---busD---switchD---busE
    Note: this function will also return an empty dict for bus1 and bus4.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to get the indirect connections from.
    bus_1: int
        The bus to get the indirect connections from.
    bus_2: int
        The bus to get the indirect connections to.
    bus_1_connected: list[int]
        The buses connected to bus_1.
    bus_2_connected: list[int]
        The buses connected to bus_2.
    only_closed_switches: bool
        If True, only closed switches are considered.
    exclude_buses: Optional[list[int]]
        The buses to exclude from the indirect connection.
        e.g. give all other busbars (type b) in the substation.

    Returns
    -------
    dict[str, list[int]]
        A dictionary with the indirect connections from bus_1 to bus_2
    """
    n_max_expected_switches = 2
    if exclude_buses is None:
        exclude_buses = [bus_1, bus_2]
    bus3_candidates_1 = [el for el in bus_1_connected if el not in bus_2_connected and el not in exclude_buses]
    bus3_candidates_2 = [el for el in bus_2_connected if el not in bus_1_connected and el not in exclude_buses]
    bus_3_connected_1 = list(
        pp.toolbox.get_connected_buses(net, bus3_candidates_1, respect_switches=only_closed_switches, consider="s")
    )
    bus_3_connected_2 = list(
        pp.toolbox.get_connected_buses(net, bus3_candidates_2, respect_switches=only_closed_switches, consider="s")
    )
    # Note: bus_3 is a list and can contain multiple buses
    # -> parallel switches will be found or paths with multiple switches
    bus_3 = [el for el in bus_3_connected_1 if el in bus_3_connected_2]
    indirect_connection = pp.toolbox.get_connected_elements_dict(net, bus_3, include_empty_lists=True)
    del indirect_connection["bus"]
    if len(indirect_connection["switch"]) > n_max_expected_switches:
        logger.warning(
            f"Unknown Switch configuration: {bus_1_connected} and {bus_2_connected}"
            + f" bus1: '{net.bus.loc[bus_1, 'name']}' bus2: '{net.bus.loc[bus_2, 'name']}'"
            + " Switches will be ignored"
        )
        indirect_connection["switch"] = []
    return indirect_connection


def get_all_switches_from_bus_ids(
    network: pp.pandapowerNet, bus_ids: list[int], only_closed_switches: bool = True
) -> pd.DataFrame:
    """Get all switches connected to a list of buses.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the switches from.
    bus_ids: list[int]
        The buses to get the switches from.
    only_closed_switches: bool
        If True, only closed switches are considered.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all switches connected to the buses in bus_ids.
    """
    connected = pp.toolbox.get_connected_elements_dict(
        network,
        bus_ids,
        respect_switches=only_closed_switches,
        include_empty_lists=True,
    )
    station_switches = network.switch[network.switch.index.isin(connected["switch"])]
    return station_switches


def get_closed_switch(switches: pd.DataFrame, column: str, column_ids: list[Union[str, int, float]]) -> pd.DataFrame:
    """Get the closed switch based on the column and column_ids.

    Parameters
    ----------
    switches: pd.DataFrame
        The switches df to filter the closed switch from.
    column: str
        The column to filter the column_ids. e.g. foreign_id
    column_ids: list[Union[str, int, float]]
        The column ids to filter the closed switch from.

    Returns
    -------
    pd.DataFrame
        The closed switch filtered by the column_ids.
    """
    closed_switch = switches[(switches[column].isin(column_ids)) & (switches.closed)]
    return closed_switch


def fuse_closed_switches_by_bus_ids(network: pp.pandapowerNet, switch_bus_ids: list[int]) -> np.array:
    """Fuse a series of closed switches in the network by merging busbars (type b).

    ```
    Note: this function expects that there are only switches between the busbars.
    Warning: this function will break the model if gaps are between the buses or other elements in between.
    This function will not work if you try to fuse multiple busbars,
    that are not directly connected by the the switch_bus_ids.
    e.g.
    ----busbar1----switch1----switch2---switch3----busbar2----
    will result in:
    ----busbar1---

    This will not work:
    (no connection between busbar1 and busbar3 / busbar2 and 4)
    ----busbar1----switch1----switch2---switch3----busbar2----
    ----busbar3----switch4----switch5---switch6----busbar4----
    call this function twice to fuse busbar2 into busbar1 and busbar4 into busbar3
    ```

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to fuse closed switches in, will be modified in-place.
    switch_bus_ids: list[int]
        The bus ids of the switches to fuse.
        Note: this must include the bus_id that is expected to be the final busbar.

    Returns
    -------
    bus_labels: np.array
        An with the length of the highest bus id in the network representing the busbar index.
        At the index of the array(old busbar index), the new busbar index is stored.

    """
    # get a label dict for the buses
    # make sure that missing/deleted buses do not break the algorithm
    bus_labels = np.arange(np.max(network.bus.index) + 1)
    # remove duplicate bus ids -> can happen if cross coupler have only one DV switch
    switch_bus_ids_pruned = list(set(switch_bus_ids))
    switch_buses = network.bus.loc[switch_bus_ids_pruned]
    switch_buses_type_b = switch_buses[switch_buses["type"] == "b"]
    if len(switch_buses_type_b) == 0:
        raise ValueError(f"No busbars found in the switch_bus_ids list {switch_bus_ids}")
    # select the first busbar of type 'b' to be the reference busbar
    for bus_id in switch_bus_ids_pruned:
        # set all busbars to the first busbar
        bus_labels[bus_id] = switch_buses_type_b.index[0]
        # Move all elements over to the lowest index busbar

    move_elements_based_on_labels(network, bus_labels)
    # Drop all busbars that were re-labeled because they were connected to a lower-labeled bus
    buses_to_drop = network.bus[~np.isin(network.bus.index, bus_labels)]
    # drop switches that are connected to one bus -> have been fused
    network["switch"] = network["switch"][network["switch"]["bus"] != network["switch"]["element"]]
    pp.drop_buses(network, buses_to_drop.index)

    return bus_labels


# TODO: replace by networkX_logic_modules
def get_vertical_connected_busbars(network: pp.pandapowerNet, substation_bus_list: list[int]) -> dict[int, list[int]]:
    """Get all vertical busbars combinations.

    respect_switches = False, because we want to get the horizontal busbars.
    By ignoring the switches we will find all paths between the busbars with only one node in between.
    See Branch 1 in the diagram (in the module header ), we are looking for the busbars that connect to one branch.
    If we found a busbar that connects to one branch -> a coupler between these two busbars is
    a busbar coupler.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the horizontal busbars from.
    substation_bus_list: list[int]
        The bus ids of the busbars to get the horizontal busbars from.
        This list is expected to be from one substation only.
        The list will be filtered for the type 'b'.
        Include all busbars you want to find the connections for.

    Returns
    -------
    dict[int, list[int]]
        A dictionary with the vertical busbars combinations.
        e.g. for two busbars connected to each other:
        {0: [1], 1: [0]}
        for busbar three busbars connected to each other:
        {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        if not connected, the list will be empty
        {0: [1], 1: [0], 30: []}

    """
    bus_type_b = get_type_b_nodes(network, substation_bus_list)
    vertical_busbars = {}
    for station_id in bus_type_b.index:
        first_layer_connection = pp.toolbox.get_connected_buses(network, [station_id], respect_switches=False, consider="s")
        first_layer_connection = first_layer_connection.difference(set(bus_type_b.index))
        second_lay_connection = pp.toolbox.get_connected_buses(
            network, first_layer_connection, respect_switches=False, consider="s"
        )
        connection = first_layer_connection | second_lay_connection
        vertical_busbars[station_id] = [
            station for station in connection if station in bus_type_b.index and station != station_id
        ]
    return vertical_busbars


# TODO: replace by networkX_logic_modules
def get_connection_between_busbars(
    network: pp.pandapowerNet,
    bus_1: int,
    bus_2: int,
    exlcude_ids: list[int],
    only_closed_switches: bool = True,
) -> tuple[list[int], dict[str, set[int]]]:
    """Get the connection between two busbars.

    This function will return the connection between two busbars.
    The connection can be direct or indirect, with up to four switches.
    The function will return the connection with the least amount of switches.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the connection from.
    bus_1: int
        The bus id of the first busbar.
    bus_2: int
        The bus id of the second busbar.
    exlcude_ids: list[int]
        The bus ids to exclude from the connection.
        e.g. give all other busbars (type b) in the substation.
    only_closed_switches: bool
        If True, only closed switches are considered.

    Returns
    -------
    connection: tuple[list[int], dict[str, set[int]]]
        A dictionary with the connection between the two busbars.
        The dictionary will contain the key "switch" with the switch ids.
        If there is no connection, the dictionary will be empty.
        Note: This function does not handle other parallel connections and will return these parallel connections.
              Look for other keys in the dict to find parallel elements.

    Raises
    ------
    ValueError
        If the connection contains more than four switches.

    """
    # get connection with one switch
    connection = pp.toolbox.get_connecting_branches(network, [bus_1], [bus_2])
    # get connection with three switches
    if "switch" not in connection:
        connection = get_indirect_connected_switch(
            net=network,
            bus_1=bus_1,
            bus_2=bus_2,
            only_closed_switches=only_closed_switches,
            exclude_buses=exlcude_ids,
        )
    if "switch" in connection:
        switches = list(connection["switch"])
    else:
        switches = []
    # # get connection with four switches
    # consider_three_buses = False
    # if "switch" not in connection:
    #     consider_three_buses = True
    #     connection = get_indirect_connected_switch(
    #         net=network,
    #         bus_1=bus_1,
    #         bus_2=bus_2,
    #         only_closed_switches=only_closed_switches,
    #         consider_three_buses=True,
    #         exclude_buses=exlcude_ids,
    #     )
    #     switches = connection["switch"]
    #     if "switch" in connection and len(connection["switch"]) > 0:
    #         # no parallel switches implemented
    #         switches = [switch for switch in connection["switch"]]

    return switches, connection


# TODO: replace by networkX_logic_modules
def get_coupler_types_of_substation(
    network: pp.pandapowerNet,
    substation_bus_list: list[int],
    only_closed_switches: bool = True,
) -> dict[str, list[list[int]]]:
    """Get the cross coupler (German: Querkuppler),  busbar coupler and a cross connector of a substation.

    A busbar coupler is a connection between two busbars, where assets can be connected to both busbars.
    A cross coupler is a connection between two busbars B1 and B2,
    where assets A1 can not be connected to both busbars directly.
    Asset A1 can only be connected directly to B1 and is connected indirectly to B2 by the cross coupler.
    A coupler is always a disconnector (DS), a power switch (CB) and a DS in series. In unique cases, there can be
    two CB switches in series.
    A cross connector is a single disconnector between two busbars.


    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the Cross coupler/quercoupler from.
    substation_bus_list: list[int]
        The bus list of the substation.
        All buses in the list represent a substation.
    only_closed_switches: bool
        If True, only closed switches are considered.

    Returns
    -------
    coupler : dict[str, list[list[int]]]
        a dictionary with 4 keys:
        - 1. key: "busbar_coupler_bus_ids"
        - 2. key: "cross_coupler_bus_ids"
        - 3. key: "busbar_coupler_switch_ids"
        - 4. key: "cross_coupler_switch_ids"
        bus_ids: list of bus ids representing the busbar coupler and cross coupler
        switch_ids: list of switch ids representing the busbar coupler and cross coupler
            switch_ids = [CB, DS1, DS2]
        Note: the switches are not filtered by open/closed.
        Note: if there is only one switch or two switches:
            switch_ids_1sw = [CB, CB, CB]
            switch_ids_2sw = [CB, CB, DS2]
    """
    coupler = {
        "busbar_coupler_bus_ids": [],
        "cross_coupler_bus_ids": [],
        "busbar_coupler_switch_ids": [],
        "cross_coupler_switch_ids": [],
    }  # type: dict[str, list[list[int]]]
    bus_type_b = get_type_b_nodes(network, substation_bus_list)
    if len(bus_type_b) == 0 or len(bus_type_b) == 1:
        # no coupled busbars
        return coupler
    vertical_busbars = get_vertical_connected_busbars(network, substation_bus_list)
    busbar_combinations = [
        (int(bus_1), int(bus_2)) for i, bus_1 in enumerate(bus_type_b.index) for bus_2 in bus_type_b.index[i + 1 :]
    ]
    # sort by busbar coupler and cross coupler
    for bus_1, bus_2 in busbar_combinations:
        # get connection between busbars
        switches, _connection = get_connection_between_busbars(
            network=network,
            bus_1=bus_1,
            bus_2=bus_2,
            exlcude_ids=bus_type_b.index,
            only_closed_switches=only_closed_switches,
        )
        if len(switches) != 0:
            # check for parallel switches
            for cb_switch_id in switches:
                # if not consider_three_buses:
                #     cb_switch_id_list = [cb_switch_id]
                # else:
                #     cb_switch_id_list = cb_switch_id
                cb_switch_id_list = [cb_switch_id]
                power_switch = network.switch.loc[cb_switch_id_list]
                switch_buses = np.append(power_switch.element.values, power_switch.bus.values)
                ds_switch_1 = pp.toolbox.get_connecting_branches(
                    network,
                    [bus_1],
                    switch_buses,
                )
                ds_switch_2 = pp.toolbox.get_connecting_branches(
                    network,
                    [bus_2],
                    switch_buses,
                )
                if bus_1 in vertical_busbars and bus_2 in vertical_busbars[bus_1]:
                    bus_key = "busbar_coupler_bus_ids"
                    switch_key = "busbar_coupler_switch_ids"
                else:
                    bus_key = "cross_coupler_bus_ids"
                    switch_key = "cross_coupler_switch_ids"

                # handle cases with two buses
                bus_res = [
                    bus_1,
                    bus_2,
                    power_switch.element.values[0],
                    power_switch.bus.values[0],
                ]
                switch_res = [
                    cb_switch_id_list[0],
                    int(list(ds_switch_1["switch"])[0]),  # noqa: RUF015
                    int(list(ds_switch_2["switch"])[0]),  # noqa: RUF015
                ]
                # # handle cases with three buses
                # # if consider_three_buses:
                # if len(cb_switch_id_list) > 1:
                #     switch_res.append(cb_switch_id_list[1])
                #     node_list = list(
                #         set(
                #             np.append(
                #                 power_switch.element.values, power_switch.bus.values
                #             )
                #         )
                #     )
                #     bus_res = [bus_1, bus_2] + node_list
                coupler[bus_key].append(bus_res)
                coupler[switch_key].append(switch_res)

    return coupler


# TODO: replace by networkX_logic_modules
def get_substation_buses_from_bus_id(
    network: pp.pandapowerNet, start_bus_id: int, only_closed_switches: bool = False
) -> set[int]:
    """Get all buses of a substation from a start bus id.

    This function will return all buses that are connected to the start bus id via switches.
    Note: The input expects a bus ids only containing the busbars you want to get the connection for.
    See diagram for references. e.g::
    input [BB1, BB2] -> get BC1/2
    input [BB1, BB2, BB3, BB4] -> get BC1/2, BC3/4, CC1/3, CC2/4

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to get the substation buses from.
    start_bus_id: int
        The bus id to start the search from.
    only_closed_switches: bool
        If True, only closed switches are considered.

    Returns
    -------
    set[int]
        A set of bus ids that are connected to the start bus id.

    Raises
    ------
    RuntimeError
        If the function detects an infinite loop.
    """
    station_buses = {start_bus_id}
    len_station = len(station_buses)
    len_update = 0
    break_counter = 0
    max_loop_count = 25
    while len_station != len_update:
        len_station = len(station_buses)
        update_bus = pp.toolbox.get_connected_buses(
            network, station_buses, consider="s", respect_switches=only_closed_switches
        )
        station_buses.update(update_bus)
        len_update = len(station_buses)
        break_counter += 1
        # maximum hops is 7 for a standard substation as drawn the module header if you start at a branch
        if break_counter > max_loop_count:
            raise RuntimeError(
                "Infinite loop detected, please check the network model. "
                + f"Substation: {network.bus.loc[start_bus_id, 'name']}, with bus_id: {start_bus_id}"
            )
    return station_buses


# TODO: replace by networkX_logic_modules
def add_substation_column_to_bus(
    network: pp.pandapowerNet,
    substation_col: Optional[str] = "substat",
    get_name_col: Optional[str] = "name",
    only_closed_switches: bool = False,
) -> None:
    """Add a substation column to the bus DataFrame.

    This function will go through all busbars of type 'b' and add the substation name to all buses connected to the busbar.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network to add the substation column to.
        Note: the network will be modified in-place.
    substation_col: Optional[str]
        The name of the new substation column where the value from the get_name_col is added.
    get_name_col: Optional[str]
        The name of the column to get the substation name from.
    only_closed_switches: bool
        If True, only closed switches are considered.
        The result will lead substation naming after the the electrical voltage level.
    """
    bus_type_b = get_type_b_nodes(network).index
    network.bus[substation_col] = ""
    found_list = []
    name_list = []
    for bus_id in bus_type_b:
        if bus_id in found_list:
            continue
        station_buses = list(get_substation_buses_from_bus_id(network, bus_id, only_closed_switches=only_closed_switches))
        station_name = str(network.bus.loc[bus_id, get_name_col])
        counter = 0
        while station_name in name_list:
            station_name = str(network.bus.loc[bus_id, get_name_col]) + f"_{counter}"
            counter += 1
        network.bus.loc[station_buses, substation_col] = station_name
        found_list.extend(station_buses)
        name_list.append(station_name)


def get_station_id_list(bus_df: pd.DataFrame, substation_col: str = "substat") -> list[int]:
    """Get all station ids from the network.

    This function will return all unique station ids from the network.

    Parameters
    ----------
    bus_df: pd.DataFrame
        The bus DataFrame to get the station ids from.
        e.g. pre filtered bus DataFrame with only busbars of type 'b'.
    substation_col: str
        The column name of the substation

    Returns
    -------
    list[int]
        A list of station ids in the order of the stations in the substation_col.
    """
    substation_names = bus_df[substation_col].unique()
    return [bus_df[bus_df[substation_col] == substation_name].index for substation_name in substation_names]
