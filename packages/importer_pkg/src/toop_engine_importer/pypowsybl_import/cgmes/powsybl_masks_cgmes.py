"""Specific functions to extract masks from pypowsybl network for CGMES data."""

from beartype.typing import Optional
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import get_voltage_level_with_region
from toop_engine_interfaces.asset_topology import AssetBranchTypePowsybl, AssetInjectionTypePowsybl
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    RegionType,
)


def get_switchable_buses_cgmes(
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
    # TODO: set as import parameter
    rules = {"min_busbars": 2, "min_elements": 4, "allow_pst": False}
    voltage_levels = get_voltage_level_with_region(network=net)
    if select_by_voltage_level_id_list is None:
        # Gets all voltage levels in the area
        voltage_levels = voltage_levels[
            voltage_levels["region"].str.startswith(tuple(area_codes)) & (voltage_levels["nominal_v"] >= cutoff_voltage)
        ]
        voltage_level_list = voltage_levels.index.tolist()
    else:
        voltage_level_list = [vl for vl in select_by_voltage_level_id_list if vl in voltage_levels.index]

    allowed_elements = list(AssetBranchTypePowsybl.__args__ + AssetInjectionTypePowsybl.__args__)
    switchable_buses = []
    for vl_index in voltage_level_list:
        bus_breaker_topology = net.get_bus_breaker_topology(vl_index)
        node_breaker_topology = net.get_node_breaker_topology(vl_index)
        switches = bus_breaker_topology.switches
        elements = bus_breaker_topology.elements
        if switches[switches["kind"] == "BREAKER"].empty:
            continue

        busbars_per_bus = node_breaker_topology.nodes[node_breaker_topology.nodes["connectable_type"] == "BUSBAR_SECTION"]
        busbars_per_bus = busbars_per_bus.merge(
            net.get_busbar_sections()[["bus_id"]], left_on="connectable_id", right_on="id", how="left"
        )
        n_voltage_level_per_station = len(busbars_per_bus["bus_id"].unique())
        n_busbars_per_bus = len(busbars_per_bus)
        if n_busbars_per_bus < rules["min_busbars"]:
            continue  # less than 2 busbars -> not switchable
        if n_busbars_per_bus - 1 < n_voltage_level_per_station:
            continue  # at least two busbars need to have the same voltage level
        if elements["type"].isin(allowed_elements).sum() < rules["min_elements"]:
            continue  # at least 4 elements are needed for splits
        # check for PSTs
        if not rules["allow_pst"]:
            trafos = elements.merge(
                net.get_2_windings_transformers(attributes=["voltage_level1_id", "voltage_level2_id"]),
                left_index=True,
                right_index=True,
            )
            if len(trafos[trafos["voltage_level1_id"] == trafos["voltage_level2_id"]]) > 0:
                continue
        busbars_per_bus_count = busbars_per_bus[busbars_per_bus["bus_id"] != ""].groupby("bus_id").size()
        # relevant bus is only the most connected busbar in the bus
        busbars_per_bus_count = busbars_per_bus_count[busbars_per_bus_count > 1]
        if busbars_per_bus_count.empty:
            continue
        busbars_per_bus_count = busbars_per_bus_count.sort_values(ascending=False)
        switchable_buses.append(busbars_per_bus_count.index[0])
    return switchable_buses
