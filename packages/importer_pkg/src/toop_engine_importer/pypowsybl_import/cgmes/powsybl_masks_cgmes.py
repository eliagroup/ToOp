# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Specific functions to extract masks from pypowsybl network for CGMES data."""

from beartype.typing import Optional
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import get_voltage_level_with_region
from toop_engine_interfaces.asset_topology import AssetBranchTypePowsybl, AssetInjectionTypePowsybl
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    RegionType,
    RelevantStationRules,
)


def get_switchable_buses_cgmes(
    net: Network,
    area_codes: list[RegionType],
    cutoff_voltage: int = 220,
    select_by_voltage_level_id_list: Optional[list[int]] = None,
    relevant_station_rules: Optional[RelevantStationRules] = None,
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
    relevant_station_rules: Optional[RelevantStationRules]
        The rules to consider a station as relevant. If None, default rules are applied.

    Returns
    -------
    relevant_busbars: list[str]
        The most connected buses in the relevant voltage levels.

    """
    if relevant_station_rules is None:
        relevant_station_rules = RelevantStationRules()
    voltage_levels = get_voltage_level_with_region(network=net)
    if select_by_voltage_level_id_list is None:
        # Gets all voltage levels in the area
        voltage_levels = voltage_levels[
            voltage_levels["region"].str.startswith(tuple(area_codes)) & (voltage_levels["nominal_v"] >= cutoff_voltage)
        ]
        voltage_level_list = voltage_levels.index.tolist()
    else:
        voltage_level_list = [vl for vl in select_by_voltage_level_id_list if vl in voltage_levels.index]

    allowed_injections_types = list(AssetInjectionTypePowsybl.__args__)
    allowed_branch_types = list(AssetBranchTypePowsybl.__args__)
    allowed_elements_types = allowed_branch_types + allowed_injections_types

    switchable_buses = []
    for voltage_level_id in voltage_level_list:
        bus = get_most_connected_bus_at_voltage_level(
            voltage_level_id=voltage_level_id,
            net=net,
            relevant_station_rules=relevant_station_rules,
            allowed_branch_types=allowed_branch_types,
            allowed_elements_types=allowed_elements_types,
        )
        if bus:
            switchable_buses.append(bus)
    return switchable_buses


def get_most_connected_bus_at_voltage_level(
    voltage_level_id: int,
    net: Network,
    relevant_station_rules: RelevantStationRules,
    allowed_branch_types: list[str],
    allowed_elements_types: list[str],
) -> str | None:
    """Get the most connected bus at the given voltage level, if it passes the relevant station rules.

    Parameters
    ----------
    voltage_level_id: int
        The voltage level to analyze.
    net: Network
        The network to analyze.
    relevant_station_rules: RelevantStationRules
        The rules to consider a station as relevant.
    allowed_branch_types: list[str]
        The allowed branch types to consider.
    allowed_elements_types: list[str]
        The allowed element types to consider.

    Returns
    -------
    str | None
        The most connected bus at the given voltage level, or None if no bus passes the rules
    """
    bus_breaker_topology = net.get_bus_breaker_topology(voltage_level_id)
    node_breaker_topology = net.get_node_breaker_topology(voltage_level_id)
    switches = bus_breaker_topology.switches
    elements = bus_breaker_topology.elements
    if switches[switches["kind"] == "BREAKER"].empty:
        return None

    busbars_per_bus = node_breaker_topology.nodes[node_breaker_topology.nodes["connectable_type"] == "BUSBAR_SECTION"]
    busbars_per_bus = busbars_per_bus.merge(
        net.get_busbar_sections()[["bus_id"]], left_on="connectable_id", right_on="id", how="left"
    )
    n_voltage_level_per_station = len(busbars_per_bus["bus_id"].unique())
    n_busbars_per_bus = len(busbars_per_bus)
    if (n_busbars_per_bus < relevant_station_rules.min_busbars) or (n_busbars_per_bus - 1 < n_voltage_level_per_station):
        return None  # number of busbars too low
    if elements["type"].isin(allowed_elements_types).sum() < relevant_station_rules.min_connected_elements:
        return None  # number of connected elements too low
    if elements["type"].isin(allowed_branch_types).sum() < relevant_station_rules.min_connected_branches:
        return None  # number of connected elements too low
    busbars_per_bus_count = busbars_per_bus[busbars_per_bus["bus_id"] != ""].groupby("bus_id").size()
    # relevant bus is only the most connected busbar in the bus
    busbars_per_bus_count = busbars_per_bus_count[busbars_per_bus_count > 1]
    if busbars_per_bus_count.empty:
        return None
    busbars_per_bus_count = busbars_per_bus_count.sort_values(ascending=False)
    most_connected_bus = busbars_per_bus_count.index[0]
    return most_connected_bus


def get_potentially_relevant_voltage_levels(
    net: Network,
    area_codes: list[RegionType],
    cutoff_voltage: int,
    select_by_voltage_level_id_list: Optional[list[int]] = None,
) -> list[int]:
    """Get the voltage levels in the given area and above the cutoff voltage.

    Parameters
    ----------
    net: Network
        The network to analyze.
    area_codes: list[RegionType]
        The prefixes of the voltage levels to consider.
    cutoff_voltage: int
        The minimal voltage to be considered for relevant substations.
    select_by_voltage_level_id_list: Optional[list[int]]
        If given, only voltage levels with these IDs are considered.

    Returns
    -------
    voltage_level_list: list[int]
        The voltage levels in the given area and above the cutoff voltage.
    """
    voltage_levels = get_voltage_level_with_region(network=net)
    if select_by_voltage_level_id_list is None:
        # Gets all voltage levels in the area
        voltage_levels = voltage_levels[
            voltage_levels["region"].str.startswith(tuple(area_codes)) & (voltage_levels["nominal_v"] >= cutoff_voltage)
        ]
        voltage_level_list = voltage_levels.index.tolist()
    else:
        voltage_level_list = [vl for vl in select_by_voltage_level_id_list if vl in voltage_levels.index]
    return voltage_level_list
