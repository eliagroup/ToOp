# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Specific functions to extract masks from pypowsybl network for UCTE data."""

from beartype.typing import Optional
from pypowsybl.network.impl.network import Network
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    RegionType,
)


def get_switchable_buses_ucte(
    net: Network,
    area_codes: list[RegionType],
    cutoff_voltage: int = 220,
    select_by_voltage_level_id_list: Optional[list[int]] = None,
) -> list[str]:
    """Return the buses in the given voltage level and area, if they have more than one busbar and connected switches.

    Checks for number of branches are happening later during network data preprocessing

    Parameters
    ----------
    net: Network
        The network to analyze.
    area_codes: list[RegionType]
        The prefixes of the voltage levels to consider.
    cutoff_voltage: int
        The minimal voltage to be considered for relevant substations. Defaults to 220
    select_by_voltage_level_id_list: Optional[list[int]]
        If given, only voltage levels with these IDs are considered.
        Note: This overrides the area_codes and cutoff_voltage parameters.

    Returns
    -------
    relevant_busbars: list[str]
        The most connected buses in the relevant voltage levels.

    """
    voltage_levels = net.get_voltage_levels()
    if select_by_voltage_level_id_list is None:
        # Gets all voltage levels in the area
        voltage_levels = voltage_levels[
            voltage_levels.index.str.startswith(tuple(area_codes)) & (voltage_levels["nominal_v"] >= cutoff_voltage)
        ]
        voltage_level_list = voltage_levels.index.tolist()
    else:
        voltage_level_list = [vl for vl in select_by_voltage_level_id_list if vl in voltage_levels.index]
    switchable_buses = []
    for vl in voltage_level_list:
        bus_breaker_topology = net.get_bus_breaker_topology(vl)
        if bus_breaker_topology.switches.empty:
            continue
        busbars_per_bus = bus_breaker_topology.buses[bus_breaker_topology.buses["bus_id"] != ""].groupby("bus_id").size()
        if busbars_per_bus.empty:
            continue
        switchable_buses.extend(busbars_per_bus[busbars_per_bus > 1].index)
    return switchable_buses
