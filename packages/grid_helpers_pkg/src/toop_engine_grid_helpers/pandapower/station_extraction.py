# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Pandapower station extraction helpers used by the DC solver backend.

These helpers mirror the importer-side pandapower station extraction logic but live in the
solver package so the backend can materialize runtime stations without importing the importer
package at module import time.
"""

import numpy as np
import pandapower as pp
import pandas as pd
import structlog
from beartype.typing import Iterable, List, Literal, Optional, Tuple, Union
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import get_asset_switching_table
from toop_engine_interfaces.asset_topology.assets import AssetBay, CouplerBay, build_asset_bay_id

logger = structlog.get_logger(__name__)


def get_type_b_nodes(
    network: pp.pandapowerNet,
    substation_bus_list: Optional[list[int] | pd.Index] = None,
    substation_column: str = "substat",
) -> pd.DataFrame:
    """Get all nodes of type ``b`` in a network or substation."""
    if substation_bus_list is None:
        substation_bus_list = network.bus.index
    substation_buses = network.bus.loc[substation_bus_list]
    bus_type_b = substation_buses[substation_buses.type == "b"]
    if substation_column not in bus_type_b.columns:
        bus_type_b[substation_column] = np.nan
    no_substations_name = bus_type_b[substation_column].isna() | (bus_type_b[substation_column] == "")
    bus_type_b.loc[no_substations_name, substation_column] = bus_type_b.loc[no_substations_name].index.astype(str)
    return bus_type_b


def get_indirect_connected_switch(
    net: pp.pandapowerNet,
    bus_1: int,
    bus_2: int,
    only_closed_switches: bool = True,
    consider_three_buses: bool = False,
    exclude_buses: Optional[list[int] | pd.Index] = None,
) -> dict[str, list[int]]:
    """Get switch-only indirect connections between two buses."""
    if exclude_buses is None:
        exclude_buses = [bus_1, bus_2]
    bus_1_connected = list(pp.toolbox.get_connected_buses(net, [bus_1], respect_switches=only_closed_switches, consider="s"))
    bus_1_connected = [bus_id for bus_id in bus_1_connected if bus_id not in exclude_buses]
    bus_2_connected = list(pp.toolbox.get_connected_buses(net, [bus_2], respect_switches=only_closed_switches, consider="s"))
    bus_2_connected = [bus_id for bus_id in bus_2_connected if bus_id not in exclude_buses]

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
    if "switch" in indirect_connection and only_closed_switches:
        indirect_connection["switch"] = [
            switch_id for switch_id in indirect_connection["switch"] if net.switch.loc[switch_id].closed
        ]
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


def get_indirect_connected_switches_three_buses(
    net: pp.pandapowerNet,
    bus_1: int,
    bus_2: int,
    bus_1_connected: list[int],
    bus_2_connected: list[int],
    only_closed_switches: bool = True,
    exclude_buses: Optional[list[int] | pd.Index] = None,
) -> dict[str, list[int]]:
    """Get both switches that indirectly connect two buses with exactly three buses in between."""
    if exclude_buses is None:
        exclude_buses = [bus_1, bus_2]
    indirect_connection = {"switch": []}
    for bus_1_con in bus_1_connected:
        for bus_2_con in bus_2_connected:
            bus_1_connected_2 = list(
                pp.toolbox.get_connected_buses(net, [bus_1_con], respect_switches=only_closed_switches, consider="s")
            )
            bus_1_connected_2 = [bus_id for bus_id in bus_1_connected_2 if bus_id not in exclude_buses]
            bus_2_connected_2 = list(
                pp.toolbox.get_connected_buses(net, [bus_2_con], respect_switches=only_closed_switches, consider="s")
            )
            bus_2_connected_2 = [bus_id for bus_id in bus_2_connected_2 if bus_id not in exclude_buses]
            connection = pp.toolbox.get_connecting_branches(net, bus_1_connected_2, bus_2_connected_2)
            if list(connection.keys()) == ["switch"]:
                indirect_connection["switch"].extend(list(connection["switch"]))
    return indirect_connection


def get_all_switches_from_bus_ids(
    network: pp.pandapowerNet,
    bus_ids: list[int] | pd.Index,
    only_closed_switches: bool = True,
) -> pd.DataFrame:
    """Get all switches connected to a list of buses."""
    connected = pp.toolbox.get_connected_elements_dict(
        network,
        bus_ids,
        respect_switches=only_closed_switches,
        include_empty_lists=True,
    )
    return network.switch[network.switch.index.isin(connected["switch"])]


def get_closed_switch(
    switches: pd.DataFrame,
    column: str,
    column_ids: Iterable[Union[str, int, float, None]],
) -> pd.DataFrame:
    """Get the closed switch rows filtered by one identifier column."""
    return switches[(switches[column].isin(column_ids)) & (switches.closed)]


def get_substation_buses_from_bus_id(
    network: pp.pandapowerNet,
    start_bus_id: int,
    only_closed_switches: bool = False,
) -> set[int]:
    """Get all buses of a substation from a start bus id."""
    station_buses = {start_bus_id}
    len_station = len(station_buses)
    len_update = 0
    break_counter = 0
    max_loop_count = 25
    while len_station != len_update:
        len_station = len(station_buses)
        update_bus = pp.toolbox.get_connected_buses(
            network,
            station_buses,
            consider="s",
            respect_switches=only_closed_switches,
        )
        station_buses.update(update_bus)
        len_update = len(station_buses)
        break_counter += 1
        if break_counter > max_loop_count:
            raise RuntimeError(
                "Infinite loop detected, please check the network model. "
                + f"Substation: {network.bus.loc[start_bus_id, 'name']}, with bus_id: {start_bus_id}"
            )
    return station_buses


def add_substation_column_to_bus(
    network: pp.pandapowerNet,
    substation_col: Optional[str] = "substat",
    get_name_col: Optional[str] = "name",
    only_closed_switches: bool = False,
) -> None:
    """Add a substation column to the bus dataframe in-place."""
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


def get_station_bus_df(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
) -> pd.DataFrame:
    """Get the bus dataframe by either station name or station bus index."""
    bus_df = network.bus
    if station_name is not None and station_bus_index is None:
        bus_df = bus_df[bus_df[station_col] == station_name]
    elif station_bus_index is not None and station_name is None:
        station_bus_index_list = [station_bus_index] if isinstance(station_bus_index, int) else station_bus_index
        bus_df = bus_df.loc[station_bus_index_list]
    else:
        raise ValueError("Either station_name or station_bus_index needs to be set.")
    return bus_df


def get_busses_from_station(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
    foreign_key: str = "equipment",
) -> pd.DataFrame:
    """Get the buses from one station."""
    bus_df = get_station_bus_df(
        network=network,
        station_name=station_name,
        station_col=station_col,
        station_bus_index=station_bus_index,
    )
    bus_df["grid_model_id"] = bus_df.index.astype(str) + SEPARATOR + "bus"
    bus_df["bus_breaker_bus_id"] = bus_df["grid_model_id"]
    bus_df["bus_branch_bus_id"] = bus_df["grid_model_id"]
    bus_df["int_id"] = bus_df.index
    if foreign_key in bus_df.columns:
        bus_df["name"] = bus_df[foreign_key].astype(str)
    station_busses = bus_df[
        ["grid_model_id", "type", "name", "int_id", "in_service", "bus_breaker_bus_id", "bus_branch_bus_id"]
    ]
    station_busses["name"] = station_busses["name"].astype(str)
    return station_busses


def get_coupler_from_station(  # noqa: C901
    network: pp.pandapowerNet,
    station_buses: pd.DataFrame,
    foreign_key: str = "equipment",
) -> pd.DataFrame:
    """Get the coupler elements from one station."""
    station_switches = get_all_switches_from_bus_ids(network=network, bus_ids=station_buses.index)
    bus_type_b = station_buses[station_buses["type"] == "b"]
    busbar_combinations = [(bus_1, bus_2) for i, bus_1 in enumerate(bus_type_b.index) for bus_2 in bus_type_b.index[i + 1 :]]
    switch_ids = []
    switch_bus = []
    for bus_1, bus_2 in busbar_combinations:
        direct_connection = pp.toolbox.get_connecting_branches(network, [bus_1], [bus_2])
        if list(direct_connection.keys()) == ["switch"]:
            switch_ids.append(next(iter(direct_connection["switch"])))
            switch_bus.append((bus_1, bus_2))
        else:
            indirect_connection = get_indirect_connected_switch(
                network,
                bus_1,
                bus_2,
                only_closed_switches=False,
                consider_three_buses=True,
            )
            if list(indirect_connection.keys()) == ["switch"]:
                for switch_id in indirect_connection["switch"]:
                    switch_ids.append(switch_id)
                    switch_bus.append((bus_1, bus_2))
            else:
                raise ValueError(
                    f"Busbars {bus_1} and {bus_2} are not or not only connected by a switch. "
                    + f"Element: {indirect_connection}. Busbar:{bus_type_b.iloc[0].to_dict()}"
                )

    station_switches_cb = station_switches[station_switches.index.isin(switch_ids)]
    if len(switch_ids) != len(station_switches_cb):
        raise ValueError(f"switch_id {switch_ids} does not match station_switches_CB {station_switches_cb.index}")
    if not all(station_switches_cb["type"] == "CB"):
        raise ValueError(
            f"switches {station_switches_cb.index} are not of type CB, but {station_switches_cb['type'].unique()}"
        )

    for index, switch_id in enumerate(switch_ids):
        bus_1, bus_2 = switch_bus[index]
        station_switches_cb.at[switch_id, "element"] = bus_1
        station_switches_cb.at[switch_id, "bus"] = bus_2

    station_switches_cb["closed"] = ~station_switches_cb["closed"]
    station_switches_cb.rename(columns={"closed": "open"}, inplace=True)
    busbar_grid_model_id_by_int_id = station_buses["grid_model_id"].to_dict()
    station_switches_cb["coupler_bay"] = None
    for switch_id in station_switches_cb.index:
        from_busbar_int_id = int(station_switches_cb.at[switch_id, "bus"])
        to_busbar_int_id = int(station_switches_cb.at[switch_id, "element"])
        station_switches_cb.at[switch_id, "coupler_bay"] = CouplerBay(
            coupler_breaker_ids=[str(switch_id) + SEPARATOR + "switch"],
            coupler_disconnector_ids=[],
            from_busbar_ids=[str(busbar_grid_model_id_by_int_id[from_busbar_int_id])],
            to_busbar_ids=[str(busbar_grid_model_id_by_int_id[to_busbar_int_id])],
            from_busbar_disconnector_ids={},
            to_busbar_disconnector_ids={},
        ).model_dump()
    station_switches_cb = station_switches_cb.rename(columns={"bus": "busbar_from_id", "element": "busbar_to_id"})
    station_switches_cb["grid_model_id"] = station_switches_cb.index.astype(str) + SEPARATOR + "switch"
    if "in_service" not in station_switches_cb.columns:
        station_switches_cb["in_service"] = True
    if foreign_key in station_switches_cb.columns:
        station_switches_cb["name"] = station_switches_cb[foreign_key]
    return station_switches_cb[
        ["grid_model_id", "type", "name", "busbar_from_id", "busbar_to_id", "open", "in_service", "coupler_bay"]
    ]


def get_asset_connection_path_to_busbars(
    network: pp.pandapowerNet,
    station_grid_model_id: str,
    asset_grid_model_id: str,
    asset_bus: int,
    station_buses: pd.DataFrame,
    save_col_name: str = "equipment",
) -> AssetBay:
    """Get the asset connection path to busbars."""
    station_switches = get_all_switches_from_bus_ids(
        network=network,
        bus_ids=station_buses.index,
        only_closed_switches=False,
    )
    station_switches = station_switches[(station_switches["et"] == "b")]

    bus_1_element = station_buses[station_buses.index == asset_bus]
    bus_1 = asset_bus
    assert len(bus_1_element) == 1, f"Expected one bus with index {asset_bus}, got {len(bus_1_element)}"
    assert bus_1_element.type.iloc[0] == "n", f"Expected bus.type 'n', got {bus_1_element.type.iloc[0]}"

    asset_disconnector = station_switches[(station_switches.bus == bus_1) | (station_switches.element == bus_1)]
    assert len(asset_disconnector) == 1, (
        f"Expected one asset disconnector connected to bus {bus_1}, got {len(asset_disconnector)}"
    )
    assert asset_disconnector.et.iloc[0] == "b", f"Expected bus-bus switch, got {asset_disconnector.et.iloc[0]}"

    if asset_disconnector.type.iloc[0] == "CB":
        asset_disconnector = None
        bus_2 = bus_1
        condition_not_bus_1 = np.ones(len(station_switches), dtype=bool)
    else:
        assert asset_disconnector.type.iloc[0] == "DS", f"Expected switch type DS, got {asset_disconnector.type.iloc[0]}"
        bus_2 = asset_disconnector.element.iloc[0]
        if bus_2 == bus_1:
            bus_2 = asset_disconnector.bus.iloc[0]
        condition_not_bus_1 = (station_switches.bus != bus_1) & (station_switches.element != bus_1)

    condition_bus_2 = (station_switches.bus == bus_2) | (station_switches.element == bus_2)
    circuit_breaker = station_switches[condition_not_bus_1 & condition_bus_2]
    assert len(circuit_breaker) == 1
    assert circuit_breaker.et.iloc[0] == "b"
    assert circuit_breaker.type.iloc[0] == "CB"

    bus_3 = circuit_breaker.element.iloc[0]
    if bus_3 == bus_2:
        bus_3 = circuit_breaker.bus.iloc[0]

    condition_not_bus_2 = (station_switches.bus != bus_2) & (station_switches.element != bus_2)
    condition_bus_3 = (station_switches.bus == bus_3) | (station_switches.element == bus_3)
    busbar_disconnectors = station_switches[condition_not_bus_2 & condition_bus_3]
    assert len(busbar_disconnectors) != 0
    assert all(busbar_disconnectors.et == "b")
    assert all(busbar_disconnectors.type == "DS")

    final_buses = {}
    for _, busbar_disconnector in busbar_disconnectors.iterrows():
        final_bus = busbar_disconnector.element
        if final_bus == bus_3:
            final_bus = busbar_disconnector.bus
        final_bus_element = station_buses[station_buses.index == final_bus]
        assert len(final_bus_element) != 0
        assert final_bus_element.type.iloc[0] == "b"
        final_buses[f"{final_bus}{SEPARATOR}bus"] = busbar_disconnector[save_col_name]

    return AssetBay(
        asset_bay_id=build_asset_bay_id(station_grid_model_id, asset_grid_model_id),
        asset_disconnector_grid_model_id=asset_disconnector[save_col_name].iloc[0]
        if asset_disconnector is not None
        else None,
        breaker_grid_model_id=circuit_breaker[save_col_name].iloc[0],
        busbar_disconnector_grid_model_id=final_buses,
    )


def _build_direct_busbar_asset_bay(
    station_grid_model_id: str,
    asset_grid_model_id: str,
    busbar_grid_model_id: str,
) -> AssetBay:
    """Build a deterministic synthetic asset bay for a direct busbar connection."""
    asset_bay_id = build_asset_bay_id(station_grid_model_id, asset_grid_model_id)
    return AssetBay(
        asset_bay_id=asset_bay_id,
        asset_disconnector_grid_model_id=None,
        breaker_grid_model_id=f"{asset_bay_id}::breaker",
        busbar_disconnector_grid_model_id={
            busbar_grid_model_id: f"{asset_bay_id}::busbar_disconnector::{busbar_grid_model_id}"
        },
    )


def get_branch_from_bus_ids(
    branch_df: pd.DataFrame,
    branch_type: str,
    bus_ids: List[int] | pd.Index,
    bus_types: List[Tuple[str, Optional[str], str]],
) -> pd.DataFrame:
    """Get the branch rows connected to the given bus ids."""
    branch_df["branch_end"] = None
    branch_df_col_list = []
    for bus_col_name, pydantic_type, postfix_gridmodel_id in bus_types:
        if bus_col_name in branch_df.columns:
            branch_df_col = branch_df[branch_df[bus_col_name].isin(bus_ids)].copy()
            branch_df_col.loc[branch_df_col[bus_col_name].isin(bus_ids), "branch_end"] = pydantic_type
            branch_df_col.rename(columns={bus_col_name: "bus_int_id"}, inplace=True)
            branch_df_col["grid_model_id"] = branch_df_col.index.astype(str) + SEPARATOR + branch_type
            if branch_type != "trafo":
                branch_df_col["grid_model_id"] = branch_df_col["grid_model_id"] + postfix_gridmodel_id
                branch_df_col["type"] = branch_type + postfix_gridmodel_id
            else:
                branch_df_col["type"] = branch_type
            branch_df_col_list.append(branch_df_col)
    if len(branch_df_col_list) == 0:
        raise ValueError(
            f"bus column not found for branch_type: '{branch_type}', "
            f"using bus_type: '{bus_types}' in columns: '{branch_df.columns}'"
        )
    return pd.concat(branch_df_col_list)


def get_branches_from_station(  # noqa: C901, PLR0912
    network: pp.pandapowerNet,
    station_buses: pd.DataFrame,
    branch_types: Optional[List[str]] = None,
    bus_types: Optional[List[Tuple[str, Optional[str], str]]] = None,
    foreign_key: str = "equipment",
) -> Tuple[pd.DataFrame, np.ndarray, List[AssetBay | None]]:
    """Get switchable branch and injection assets for one station."""
    station_grid_model_id = str(station_buses["grid_model_id"].values[0])

    if branch_types is None:
        branch_types = ["line", "trafo", "trafo3w", "load", "gen", "sgen", "impedance", "shunt"]
    if bus_types is None:
        bus_types = [
            ("bus", None, ""),
            ("from_bus", "from", ""),
            ("to_bus", "to", ""),
            ("hv_bus", "hv", "_hv"),
            ("lv_bus", "lv", "_lv"),
            ("mv_bus", "mv", "_mv"),
        ]

    switch_identifier_col = foreign_key
    if switch_identifier_col not in network.switch.columns:
        network.switch["grid_model_id"] = network.switch.index.astype(str) + SEPARATOR + "switch"
        switch_identifier_col = "grid_model_id"

    bus_ids = station_buses.index
    bus_type_b = station_buses[station_buses["type"] == "b"]
    branch_data = []
    asset_connection_list = []
    for branch_type in branch_types:
        if not hasattr(network, branch_type):
            raise ValueError(f"Branch type {branch_type} not found in pandapower network")

        branch_df = getattr(network, branch_type)
        branch_df_all_busses = get_branch_from_bus_ids(
            branch_df=branch_df,
            branch_type=branch_type,
            bus_ids=bus_ids,
            bus_types=bus_types,
        )
        for index, branch in branch_df_all_busses.iterrows():
            if branch["bus_int_id"] in bus_ids:
                asset_bus = branch["bus_int_id"]
            else:
                raise ValueError(f"Branch {index} is not connected to the station busses {bus_ids}")

            if branch["bus_int_id"] not in bus_type_b.index:
                asset_connection = get_asset_connection_path_to_busbars(
                    network=network,
                    station_grid_model_id=station_grid_model_id,
                    asset_grid_model_id=(
                        str(branch[foreign_key]) if foreign_key in branch.index else str(branch["grid_model_id"])
                    ),
                    asset_bus=asset_bus,
                    station_buses=station_buses,
                    save_col_name=switch_identifier_col,
                )
                final_bus_dict = asset_connection.busbar_disconnector_grid_model_id
                closed_busbar_disconnectors = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=final_bus_dict.values(),
                )
                closed_breakers = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=[asset_connection.breaker_grid_model_id],
                )
                closed_asset_disconnectors = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=[asset_connection.asset_disconnector_grid_model_id],
                )
                if (
                    len(closed_busbar_disconnectors) == 0
                    or len(closed_breakers) == 0
                    or (
                        len(closed_asset_disconnectors) == 0
                        and asset_connection.asset_disconnector_grid_model_id is not None
                    )
                ):
                    logger.warning(
                        "No closed switch found (Element is disconnected and will be dropped) for "
                        + f"element_type:{branch_type} element: {branch.to_dict()}."
                    )
                    branch_df_all_busses.loc[index, "bus_int_id"] = -1
                else:
                    if len(closed_busbar_disconnectors) > 1:
                        logger.warning(
                            f"Expected one closed switch for element_type:{branch_type} element: {branch.to_dict()}, "
                            + f"got {len(closed_busbar_disconnectors)} switches: "
                            + f"{closed_busbar_disconnectors.to_dict()}. Using the first one."
                        )
                        closed_busbar_disconnectors = closed_busbar_disconnectors.iloc[[0]]
                    final_bus = next(
                        bus_id
                        for bus_id in final_bus_dict
                        if final_bus_dict[bus_id] == closed_busbar_disconnectors[switch_identifier_col].values[0]
                    )
                    branch_df_all_busses.loc[index, "bus_int_id"] = int(final_bus.split(SEPARATOR)[0])
                    asset_connection_list.append(asset_connection)
            else:
                asset_connection_list.append(
                    _build_direct_busbar_asset_bay(
                        station_grid_model_id=station_grid_model_id,
                        asset_grid_model_id=(
                            str(branch[foreign_key]) if foreign_key in branch.index else str(branch["grid_model_id"])
                        ),
                        busbar_grid_model_id=f"{asset_bus}{SEPARATOR}bus",
                    )
                )
        branch_df_all_busses = branch_df_all_busses[branch_df_all_busses["bus_int_id"] != -1]
        if "in_service" not in branch_df_all_busses.columns:
            branch_df_all_busses["in_service"] = True
        branch_df_all_busses = branch_df_all_busses[
            ["grid_model_id", "type", "name", "bus_int_id", "branch_end", "in_service"]
        ]
        branch_data.append(branch_df_all_busses)

    station_branches = pd.concat(branch_data)
    switching_matrix = get_asset_switching_table(station_buses=bus_type_b, station_elements=station_branches)
    station_branches = station_branches[["grid_model_id", "type", "name", "branch_end", "in_service"]]
    return station_branches, switching_matrix, asset_connection_list


def get_parameter_from_station(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
    parameter: Literal["vn_kv", "zone"] = "vn_kv",
) -> Union[float, int, str]:
    """Get a station parameter and verify that it is unique within the station."""
    bus_df = get_station_bus_df(
        network=network,
        station_name=station_name,
        station_col=station_col,
        station_bus_index=station_bus_index,
    )
    if parameter not in bus_df.columns:
        raise ValueError(f"parameter '{parameter}' not found in bus_df with columns {bus_df.columns}")
    if len(bus_df[parameter].unique()) != 1:
        raise ValueError(f"parameter '{parameter}' is not unique for station {station_name}: {bus_df[parameter].unique()}")
    return bus_df[parameter].unique()[0]
