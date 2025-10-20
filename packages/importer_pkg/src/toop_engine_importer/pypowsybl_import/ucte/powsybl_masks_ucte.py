"""Specific functions to extract masks from pypowsybl network for UCTE data."""

from pypowsybl.network.impl.network import Network
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    RegionType,
)


def get_switchable_buses_ucte(net: Network, area_codes: list[RegionType], cutoff_voltage: int = 220) -> list[str]:
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

    Returns
    -------
    relevant_busbars: list[str]
        The most connected buses in the relevant voltage levels.

    """
    # Gets all voltage levels in the area
    voltage_levels = net.get_voltage_levels()
    voltage_levels = voltage_levels[
        voltage_levels.index.str.startswith(tuple(area_codes)) & (voltage_levels["nominal_v"] >= cutoff_voltage)
    ]
    switchable_buses = []
    for vl in voltage_levels.index:
        bus_breaker_topology = net.get_bus_breaker_topology(vl)
        if bus_breaker_topology.switches.empty:
            continue
        busbars_per_bus = bus_breaker_topology.buses[bus_breaker_topology.buses["bus_id"] != ""].groupby("bus_id").size()
        if busbars_per_bus.empty:
            continue
        switchable_buses.extend(busbars_per_bus[busbars_per_bus > 1].index)
    return switchable_buses
