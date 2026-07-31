# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to translate the pandapower model to the asset topology model.

File: asset_topology.py
Author:  Benjamin Petrick
Created: 2024-10-01
"""

from string import ascii_lowercase

import numpy as np
import pandapower as pp
import pandas as pd
import structlog
from beartype.typing import Any, List, Literal, Optional, Tuple, Union, get_args
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_grid_helpers.pandapower.station_extraction import (
    get_all_switches_from_bus_ids,
    get_closed_switch,
    get_indirect_connected_switch,
    get_substation_buses_from_bus_id,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    get_asset_switching_table,
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
)
from toop_engine_interfaces.asset_topology.asset_topology import MasterStation, TopologyMasterData
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    InjectionAsset,
    build_asset_bay_id,
)
from toop_engine_interfaces.asset_topology.materialized_topology import StationAssetConnection
from toop_engine_interfaces.asset_topology.topology_conversion import validate_complete_master_data

logger = structlog.get_logger(__name__)


def _get_asset_type(asset_payload: dict[str, Any]) -> str | None:
    """Return the explicit asset type field from a station asset payload."""
    raw_asset_type = asset_payload.get("asset_type", asset_payload.get("type"))
    if raw_asset_type is None or pd.isna(raw_asset_type):
        return None
    return str(raw_asset_type)


def _get_optional_asset_name(asset_payload: dict[str, Any]) -> str | None:
    """Return a sanitized optional asset name from a station asset payload."""
    raw_name = asset_payload.get("name")
    if raw_name is None or pd.isna(raw_name):
        return None
    return str(raw_name)


def _build_canonical_asset(asset_payload: dict[str, Any]) -> BranchAsset | InjectionAsset:
    """Build a canonical branch or injection asset from one prepared payload."""
    asset_type = _get_asset_type(asset_payload)
    asset_kwargs = {
        "grid_model_id": str(asset_payload["grid_model_id"]),
        "asset_type": asset_type,
        "name": _get_optional_asset_name(asset_payload),
    }
    if asset_type in get_args(AssetBranchType):
        return BranchAsset(**asset_kwargs)
    if asset_type in get_args(AssetInjectionType):
        return InjectionAsset(**asset_kwargs)
    raise ValueError(f"Unsupported asset_type {asset_type!r} for asset {asset_kwargs['grid_model_id']}")


def _structural_station_suffix(group_index: int) -> str:
    """Return a deterministic alphabetic suffix for structural station ids.

    Parameters
    ----------
    group_index : int
        Zero-based structural group index inside one station.

    Returns
    -------
    str
        Alphabetic suffix such as ``a``, ``b``, ..., ``aa``.
    """
    suffix = ""
    remaining_index = group_index
    while True:
        remaining_index, char_index = divmod(remaining_index, len(ascii_lowercase))
        suffix = ascii_lowercase[char_index] + suffix
        if remaining_index == 0:
            return suffix
        remaining_index -= 1


def _build_station_bus_group_id(station_buses: pd.DataFrame, group_index: int = 0) -> str:
    """Build a stable station-view identifier for one pandapower station group.

    Parameters
    ----------
    station_buses : pd.DataFrame
        Bus rows belonging to one pandapower station view.
    group_index : int, default=0
        Deterministic index of the structural bus group within the station.

    Returns
    -------
    str
        Synthetic bus-group identifier derived from the representative station bus id.
    """
    representative_bus_id = str(station_buses.sort_index()["grid_model_id"].iloc[0])
    return f"{representative_bus_id}_{_structural_station_suffix(group_index)}"


def _get_structural_station_bus_groups(station_bus_ids: list[int], network: pp.pandapowerNet) -> list[list[int]]:
    """Split one pandapower station into structural bus groups via switch connectivity.

    Open and closed bus-bus switches both retain structural connectivity. Runtime state is
    materialized later; canonical master data keeps the full station structure.

    Parameters
    ----------
    station_bus_ids : list[int]
        Pandapower bus indices that belong to one logical station.
    network : pp.pandapowerNet
        Source pandapower network.

    Returns
    -------
    list[list[int]]
        Deterministic structural bus groups represented as pandapower bus indices.
    """
    remaining_bus_ids = set(station_bus_ids)
    if not remaining_bus_ids:
        return []

    structural_groups: list[list[int]] = []
    while remaining_bus_ids:
        seed_bus_id = min(remaining_bus_ids)
        structural_group = sorted(get_substation_buses_from_bus_id(network, seed_bus_id, only_closed_switches=False))
        structural_group = [
            bus_id for bus_id in structural_group if bus_id in remaining_bus_ids or bus_id in station_bus_ids
        ]
        structural_groups.append([bus_id for bus_id in structural_group if bus_id in station_bus_ids])
        remaining_bus_ids -= set(structural_group)

    structural_groups.sort()
    return [group for group in structural_groups if group]


def _register_unique_payload(
    payloads_by_id: dict[str, object],
    payload_id: str,
    payload: object,
    payload_kind: str,
) -> None:
    """Register one topology-owned payload and reject conflicting duplicates.

    Parameters
    ----------
    payloads_by_id : dict[str, object]
        Mutable mapping of already registered payloads.
    payload_id : str
        Canonical identifier of the payload.
    payload : object
        Payload to register.
    payload_kind : str
        Human-readable payload kind used in validation errors.

    Raises
    ------
    ValueError
        If the same payload id is reused with conflicting content.
    """
    existing_payload = payloads_by_id.get(payload_id)
    if existing_payload is None:
        payloads_by_id[payload_id] = payload
        return
    if existing_payload != payload:
        raise ValueError(f"Conflicting {payload_kind} payload for id {payload_id}")


def _build_direct_busbar_asset_bay(
    station_grid_model_id: str,
    asset_grid_model_id: str,
    busbar_grid_model_id: str,
) -> AssetBay:
    """Build a deterministic synthetic asset bay for a direct busbar connection."""
    asset_bay_id = build_asset_bay_id(station_grid_model_id, asset_grid_model_id)
    return AssetBay(
        asset_bay_id=asset_bay_id,
        sl_switch_grid_model_id=None,
        dv_switch_grid_model_id=f"{asset_bay_id}::dv",
        sr_switch_grid_model_id={busbar_grid_model_id: f"{asset_bay_id}::sr::{busbar_grid_model_id}"},
    )


def _build_station_assets_and_connections(
    station_branches: pd.DataFrame,
    asset_connection_path: list[AssetBay | None],
) -> tuple[
    list[BranchAsset],
    list[InjectionAsset],
    list[StationAssetConnection],
    list[StationAssetConnection],
    list[bool],
    list[AssetBay],
]:
    """Build canonical assets and station-local connections aligned with branch rows.

    Parameters
    ----------
    station_branches : pd.DataFrame
        Station-local asset rows aligned with the switching matrix columns.
    asset_connection_path : list[AssetBay | None]
        Station-local asset-bay payloads aligned with ``station_branches``.

    Returns
    -------
    tuple[
        list[BranchAsset],
        list[InjectionAsset],
        list[StationAssetConnection],
        list[StationAssetConnection],
        list[bool],
        list[AssetBay],
    ]
        Canonical branch assets, canonical injection assets, aligned station-local
        references, branch mask, and referenced asset bays.
    """
    switchable_assets = [
        _build_canonical_asset(asset_payload) for asset_payload in station_branches.to_dict(orient="records")
    ]
    asset_terminals = (
        station_branches["branch_end"].tolist()
        if "branch_end" in station_branches.columns
        else [None] * len(station_branches)
    )
    branch_mask = [isinstance(asset, BranchAsset) for asset in switchable_assets]
    branch_assets: list[BranchAsset] = []
    injection_assets: list[InjectionAsset] = []
    branch_connections: list[StationAssetConnection] = []
    injection_connections: list[StationAssetConnection] = []
    asset_bays: list[AssetBay] = []
    for asset, asset_terminal, asset_bay, is_branch in zip(
        switchable_assets,
        asset_terminals,
        asset_connection_path,
        branch_mask,
        strict=True,
    ):
        if asset_bay is not None:
            asset_bays.append(asset_bay.model_copy(deep=True))
        connection = StationAssetConnection(
            asset_id=asset.grid_model_id,
            branch_end=asset_terminal,
            asset_bay_id=asset_bay.asset_bay_id if asset_bay is not None else None,
        )
        if is_branch:
            branch_assets.append(asset.model_copy(deep=True))
            branch_connections.append(connection)
        else:
            injection_assets.append(asset.model_copy(deep=True))
            injection_connections.append(connection)
    return branch_assets, injection_assets, branch_connections, injection_connections, branch_mask, asset_bays


def _build_master_station_from_station_id(
    network: pp.pandapowerNet,
    station_id_list: list[int],
    group_index: int = 0,
    foreign_key: str = "equipment",
) -> tuple[MasterStation, list[BranchAsset], list[InjectionAsset], list[AssetBay]]:
    """Build one canonical master station plus topology-owned payloads.

    Parameters
    ----------
    network : pp.pandapowerNet
        Source pandapower network.
    station_id_list : list[int]
        Pandapower bus indices belonging to the structural station group.
    group_index : int, default=0
        Structural group index inside the original station.
    foreign_key : str, default="equipment"
        Column name used as the preferred human-readable identifier.

    Returns
    -------
    tuple[MasterStation, list[BranchAsset], list[InjectionAsset], list[AssetBay]]
        Canonical station plus the topology-owned payloads it references.
    """
    station_buses = get_busses_from_station(network, station_bus_index=station_id_list, foreign_key=foreign_key)
    coupler_elements = get_coupler_from_station(network, station_buses, foreign_key=foreign_key)
    station_branches, switching_matrix, asset_connection_path = get_branches_from_station(
        network,
        station_buses,
        foreign_key=foreign_key,
    )
    voltage_level_float = get_parameter_from_station(network=network, station_bus_index=station_id_list, parameter="vn_kv")
    station_identity = _get_station_identity(
        station_buses=station_buses,
        voltage_level=voltage_level_float,
        group_index=group_index,
    )
    branch_assets, injection_assets, branch_connections, injection_connections, branch_mask, asset_bays = (
        _build_station_assets_and_connections(
            station_branches=station_branches,
            asset_connection_path=asset_connection_path,
        )
    )
    master_station = MasterStation(
        bus_group_id=str(station_identity["bus_group_id"]),
        name=str(station_identity["name"]),
        voltage_level=float(station_identity["voltage_level"]),
        busbars=get_list_of_busbars_from_df(station_buses[station_buses["type"] == "b"]),
        couplers=get_list_of_coupler_from_df(coupler_elements),
        branch_connections=branch_connections,
        injection_connections=injection_connections,
        branch_connectivity=np.ones_like(switching_matrix[:, branch_mask], dtype=bool),
        injection_connectivity=np.ones_like(switching_matrix[:, [not is_branch for is_branch in branch_mask]], dtype=bool),
    )
    return master_station, branch_assets, injection_assets, asset_bays


def get_asset_topology_master_data_from_network(
    network: pp.pandapowerNet,
    topology_id: str,
    grid_model_file: str,
    station_id_list: List[List[int]],
    foreign_key: str = "equipment",
) -> TopologyMasterData:
    """Return canonical asset-topology master data derived from a pandapower network.

    Parameters
    ----------
    network : pp.pandapowerNet
        Source pandapower network.
    topology_id : str
        Identifier to store on the resulting master data.
    grid_model_file : str
        Source grid-model file name stored in the master data.
    station_id_list : list[list[int]]
        Station definitions as lists of pandapower bus indices.
    foreign_key : str, default="equipment"
        Column name used as the preferred human-readable identifier.

    Returns
    -------
    TopologyMasterData
        Canonical master data split into structural station groups.
    """
    master_stations: list[MasterStation] = []
    branch_assets_by_id: dict[str, BranchAsset] = {}
    injection_assets_by_id: dict[str, InjectionAsset] = {}
    asset_bays_by_id: dict[str, AssetBay] = {}

    for station_ids in station_id_list:
        structural_groups = _get_structural_station_bus_groups(list(station_ids), network)
        for group_index, structural_group in enumerate(structural_groups):
            master_station, branch_assets, injection_assets, asset_bays = _build_master_station_from_station_id(
                network=network,
                station_id_list=structural_group,
                group_index=group_index,
                foreign_key=foreign_key,
            )
            master_stations.append(master_station)
            for asset in branch_assets:
                _register_unique_payload(branch_assets_by_id, asset.grid_model_id, asset, "branch asset")
            for asset in injection_assets:
                _register_unique_payload(injection_assets_by_id, asset.grid_model_id, asset, "injection asset")
            for asset_bay in asset_bays:
                if asset_bay.asset_bay_id is None:
                    continue
                _register_unique_payload(asset_bays_by_id, asset_bay.asset_bay_id, asset_bay, "asset bay")

    master_data = TopologyMasterData(
        topology_id=topology_id,
        grid_model_file=grid_model_file,
        stations=master_stations,
        branch_assets=list(branch_assets_by_id.values()),
        injection_assets=list(injection_assets_by_id.values()),
        asset_bays=list(asset_bays_by_id.values()),
    )
    validate_complete_master_data(master_data)
    return master_data


def get_busses_from_station(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
    foreign_key: str = "equipment",
) -> pd.DataFrame:
    """Get the busses from a station_name.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_name: Optional[Union[str, int, float]]
        Station id for which the busses should be retrieved.
    station_col: str
        Column name in the bus DataFrame that contains the station_name.
    station_bus_index: Optional[Union[list[int], int]]
        List of bus indices for which the busses should be retrieved.
    foreign_key: str
        Defines the column name that is used as the foreign_key.

    Returns
    -------
    station_busses: pd.DataFrame
        DataFrame with the busses of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.
        Note: the function does not check for types e.g. 'n' (Node) or 'b' (Busbar).

    Raises
    ------
    ValueError:
        If station_name and station_bus_index are None.

    """
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
    # equipment col is the foreign key (unique) in powerfactory
    if foreign_key in bus_df.columns:
        bus_df["name"] = bus_df[foreign_key].astype(str)
    station_busses = bus_df[
        ["grid_model_id", "type", "name", "int_id", "in_service", "bus_breaker_bus_id", "bus_branch_bus_id"]
    ]
    station_busses["name"] = station_busses["name"].astype(str)

    if foreign_key in station_busses.columns:
        # force behavior: the index of the DataFrame is the internal id of the bus
        assert all(network.bus.loc[station_busses.index, foreign_key] == station_busses["name"])
    return station_busses


# TODO: replace by networkX_logic_modules
def get_coupler_from_station(
    network: pp.pandapowerNet,
    station_buses: pd.DataFrame,
    foreign_key: str = "equipment",
) -> pd.DataFrame:
    """Get the coupler elements from a station_name.

    This function expects couplers between all busbars of type 'b' (Busbar).
    Merge cross couplers before using this function.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_buses: pd.DataFrame
        DataFrame with the busses of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: The index of the DataFrame is the internal id of the bus.
        Note: The function expects all nodes of the station, inlc. type 'n' (Node) and 'b' (Busbar).
    foreign_key: str
        Defines the column name that is used as the foreign_key.

    Returns
    -------
    station_switches_CB: pd.DataFrame
        DataFrame with the coupler elements of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.

    Raises
    ------
    ValueError:
        If no busbar coupler is found between the busbars.
        If the station_busses are not of type 'b' (Busbar).
        If the coupler elements are not of type 'CB' (Circuit Breaker).
    """
    station_switches = get_all_switches_from_bus_ids(network=network, bus_ids=station_buses.index)
    # get all busbars of type 'b'
    bus_type_b = station_buses[station_buses["type"] == "b"]
    # create a list of all possible busbar combinations
    busbar_combinations = [(bus_1, bus_2) for i, bus_1 in enumerate(bus_type_b.index) for bus_2 in bus_type_b.index[i + 1 :]]
    switch_ids = []
    switch_bus = []
    # iterate over all busbar combinations and check if they are connected by a switch
    for bus_1, bus_2 in busbar_combinations:
        # check if the busbars are directly connected by a switch
        direct_connection = pp.toolbox.get_connecting_branches(network, [bus_1], [bus_2])
        if list(direct_connection.keys()) == ["switch"]:
            switch_ids.append(next(iter(direct_connection["switch"])))
            switch_bus.append((bus_1, bus_2))
        else:
            # check if the busbars are indirectly connected by a switch
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

    # verify that the switches are of type CB
    station_switches_cb = station_switches[station_switches.index.isin(switch_ids)]
    if len(switch_ids) != len(station_switches_cb):
        raise ValueError(f"switch_id {switch_ids} does not match station_switches_CB {station_switches_cb.index}")
    if not all(station_switches_cb["type"] == "CB"):
        raise ValueError(
            f"switches {station_switches_cb.index} are not of type CB, but {station_switches_cb['type'].unique()}"
        )

    # modify the station_switches_CB DataFrame to match the pydantic model
    # -> rename bus ids to match the busbars of the station
    for index, switch_id in enumerate(switch_ids):
        bus_1, bus_2 = switch_bus[index]
        station_switches_cb.at[switch_id, "element"] = bus_1
        station_switches_cb.at[switch_id, "bus"] = bus_2

    # translate closed to open convention
    station_switches_cb["closed"] = ~station_switches_cb["closed"]
    station_switches_cb.rename(columns={"closed": "open"}, inplace=True)
    # rename columns to pydantic model
    station_switches_cb = station_switches_cb.rename(columns={"bus": "busbar_from_id", "element": "busbar_to_id"})
    station_switches_cb["grid_model_id"] = station_switches_cb.index.astype(str) + SEPARATOR + "switch"
    if "in_service" not in station_switches_cb.columns:
        station_switches_cb["in_service"] = True
    # equipment col is the foreign key (unique) in powerfactory
    if foreign_key in station_switches_cb.columns:
        station_switches_cb["name"] = station_switches_cb[foreign_key]
    station_switches_cb = station_switches_cb[
        [
            "grid_model_id",
            "type",
            "name",
            "busbar_from_id",
            "busbar_to_id",
            "open",
            "in_service",
        ]
    ]
    return station_switches_cb


# TODO: replace by networkX_logic_modules
def get_branches_from_station(  # noqa: PLR0912, C901
    network: pp.pandapowerNet,
    station_buses: pd.DataFrame,
    branch_types: Optional[List[str]] = None,
    bus_types: Optional[List[Tuple[str, Optional[str], str]]] = None,
    foreign_key: str = "equipment",
) -> Tuple[pd.DataFrame, np.ndarray, List[AssetBay | None]]:
    """Get the branches from a station_buses index.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_buses: pd.DataFrame
        DataFrame with one or multiple busses to get the bus id from.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.
    branch_types: List[str]
        List of branch types that should be retrieved.
        It needs to be an attribute of the pandapower network.
    bus_types: List[Tuple[str, Optional[str], str]]
        List of the tuple(name_of_column, pydantic_type, postfix_gridmodel_id).
        This list is used to identify the bus columns in the branch types.
    foreign_key: str
        Defines the column name that is used as the foreign_key/unique identifier.

    Returns
    -------
    station_branches: pd.DataFrame
        DataFrame with the branches of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.
    switching_matrix: np.ndarray
        DataFrame with the switching matrix of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.

    Raises
    ------
    ValueError:
        If the branch type is not found in the pp.pandapowerNet.
            Note: trying to call an attribute with an empty DataFrame will not
            raise an error. e.g. pp.networks.case4gs()["impedance"].empty == True
            will not raise an error. But calling an "not_existing_attribute" will raise an error.
        If any bus name from bus_from_to_names is not found in the branch type.
    """
    station_grid_model_id = str(station_buses["grid_model_id"].values[0])

    if branch_types is None:
        branch_types = [
            "line",
            "trafo",
            "trafo3w",
            "load",
            "gen",
            "sgen",
            "impedance",
            "shunt",
        ]
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
    # get all branches
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

        # get connection path to busbars
        for index, branch in branch_df_all_busses.iterrows():
            if branch["bus_int_id"] in bus_ids:
                asset_bus = branch["bus_int_id"]
            else:
                raise ValueError(f"Branch {index} is not connected to the station busses {bus_ids}")

            if branch["bus_int_id"] not in bus_type_b.index:
                # asset is not directly connected to a busbar -> get connection path
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

                # change bus_int_id to the final busbar
                final_bus_dict = asset_connection.sr_switch_grid_model_id
                closed_sr_switches = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=final_bus_dict.values(),
                )
                closed_dv_switches = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=[asset_connection.dv_switch_grid_model_id],
                )
                closed_sl_switches = get_closed_switch(
                    network.switch,
                    column=switch_identifier_col,
                    column_ids=[asset_connection.sl_switch_grid_model_id],
                )
                if (
                    (len(closed_sr_switches) == 0)
                    or (len(closed_dv_switches) == 0)
                    or ((len(closed_sl_switches) == 0) and (asset_connection.sl_switch_grid_model_id is not None))
                ):
                    logger.warning(
                        "No closed switch found (Element is disconnected and will be dropped) for "
                        + f"element_type:{branch_type} element: {branch.to_dict()}."
                    )
                    # if asset is not connected -> -1
                    branch_df_all_busses.loc[index, "bus_int_id"] = -1
                    # do not append asset_connection -> asset is disconnected and will be dropped
                else:
                    if len(closed_sr_switches) > 1:
                        logger.warning(
                            f"Expected one closed switch for element_type:{branch_type} element: {branch.to_dict()}, "
                            + f"got {len(closed_sr_switches)} switches: {closed_sr_switches.to_dict()}. Using the first one."
                        )
                        closed_sr_switches = closed_sr_switches.iloc[[0]]
                    final_bus = [  # noqa: RUF015
                        i for i in final_bus_dict if final_bus_dict[i] == closed_sr_switches[switch_identifier_col].values[0]
                    ][0]
                    branch_df_all_busses.loc[index, "bus_int_id"] = int(final_bus.split(SEPARATOR)[0])
                    asset_connection_list.append(asset_connection)
            else:
                # asset is directly connected to a busbar
                # -> bus_int_id is already the final busbar
                asset_connection_list.append(
                    _build_direct_busbar_asset_bay(
                        station_grid_model_id=station_grid_model_id,
                        asset_grid_model_id=(
                            str(branch[foreign_key]) if foreign_key in branch.index else str(branch["grid_model_id"])
                        ),
                        busbar_grid_model_id=f"{asset_bus}{SEPARATOR}bus",
                    )
                )
        # drop all assets that are not connected to the busbars
        branch_df_all_busses = branch_df_all_busses[branch_df_all_busses["bus_int_id"] != -1]
        if "in_service" not in branch_df_all_busses.columns:
            branch_df_all_busses["in_service"] = True

        # get columns for pydantic model
        branch_df_all_busses = branch_df_all_busses[
            ["grid_model_id", "type", "name", "bus_int_id", "branch_end", "in_service"]
        ]
        branch_data.append(branch_df_all_busses)

    station_branches = pd.concat(branch_data)
    # create switching matrix
    switching_matrix = get_asset_switching_table(station_buses=bus_type_b, station_elements=station_branches)
    # keep only relevant columns
    station_branches = station_branches[["grid_model_id", "type", "name", "branch_end", "in_service"]]
    return station_branches, switching_matrix, asset_connection_list


def get_parameter_from_station(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
    parameter: Literal["vn_kv", "zone"] = "vn_kv",
) -> Union[float, int, str]:
    """Get the voltage level from a station_name.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_name: Optional[Union[str, int, float]]
        Station id for which the busses should be retrieved.
    station_col: str
        Column name in the bus DataFrame that contains the station_name.
    station_bus_index: Optional[Union[list[int], int]]
        List of bus indices for which the busses should be retrieved.
    parameter: Literal["vn_kn", "zone"]
        Parameter that should be retrieved.

    Returns
    -------
    parameter: Union[float, int, str]
        Parameter value.

    Raises
    ------
    ValueError:
        If station_name and station_bus_index are None.
        If the parameter is not found in the bus_df.
        If the voltage level is not unique for the station_name.
    """
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
    parameter = bus_df[parameter].unique()[0]
    return parameter


def _get_station_identity(
    station_buses: pd.DataFrame,
    voltage_level: float | int | str,
    group_index: int = 0,
) -> dict[str, str | float | int]:
    """Extract the station identity fields used to build the canonical station.

    Parameters
    ----------
    station_buses : pd.DataFrame
        Station bus rows used to derive the grid model id and display name.
    voltage_level : float | int | str
        Resolved station voltage level.
    group_index : int, default=0
        Deterministic index of the structural bus group within the station.

    Returns
    -------
    dict[str, str | float | int]
        Minimal station identity payload containing ``bus_group_id``, ``name``, and
        ``voltage_level``.
    """
    station_buses = station_buses.sort_index()
    return {
        "bus_group_id": _build_station_bus_group_id(station_buses, group_index=group_index),
        "name": station_buses["name"].iloc[0],
        "voltage_level": voltage_level,
    }


def get_station_bus_df(
    network: pp.pandapowerNet,
    station_name: Optional[Union[str, int, float]] = None,
    station_col: str = "substat",
    station_bus_index: Optional[Union[list[int], int]] = None,
) -> pd.DataFrame:
    """Get the bus df by either station_name or station_bus_index.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_name: Optional[Union[str, int, float]]
        Station id for which the busses should be retrieved.
    station_col: str
        Column name in the bus DataFrame that contains the station_name.
        Note: Pandapower does not have a station column, so this is a custom column.
    station_bus_index: Optional[Union[list[int], int]]
        List of bus indices for which the busses should be retrieved.

    Returns
    -------
    bus_df: pd.DataFrame

    Raises
    ------
    ValueError:
        If station_name and station_bus_index are None.
    """
    bus_df = network.bus
    if station_name is not None and station_bus_index is None:
        bus_df = bus_df[bus_df[station_col] == station_name]
    elif station_bus_index is not None and station_name is None:
        if isinstance(station_bus_index, int):
            station_bus_index_list = [station_bus_index]
        else:
            station_bus_index_list = station_bus_index
        bus_df = bus_df.loc[station_bus_index_list]
    else:
        raise ValueError("Either station_name or station_bus_index needs to be set.")

    return bus_df


# TODO: replace by networkX_logic_modules
def get_asset_connection_path_to_busbars(  # noqa: PLR0915
    network: pp.pandapowerNet,
    station_grid_model_id: str,
    asset_grid_model_id: str,
    asset_bus: int,
    station_buses: pd.DataFrame,
    save_col_name: str = "equipment",
) -> AssetBay:
    """Get the asset connection path to busbars.

    Parameters
    ----------
    network: pp.pandapowerNet
        pandapower network object
    station_grid_model_id: str
        Station identifier owning the asset bay.
    asset_grid_model_id: str
        Asset identifier for which the asset bay is constructed.
    asset_bus: int
        Asset bus id for which the connection path should be retrieved.
    station_buses: pd.DataFrame
        DataFrame with the busses of the station_name.
        Note: The DataFrame columns are the same as in the pydantic model.
    save_col_name: str
        Column name that is used as the foreign_key/unique identifier.

    Returns
    -------
    asset_connection: AssetConnectionPath
        AssetConnectionPath object.

    """
    station_switches = get_all_switches_from_bus_ids(
        network=network, bus_ids=station_buses.index, only_closed_switches=False
    )
    station_switches = station_switches[(station_switches["et"] == "b")]
    # ---------- bus 1 - asset bus
    # entry point for search
    bus_1_element = station_buses[station_buses.index == asset_bus]
    bus_1 = asset_bus
    # check if bus exists and if a bus of type 'n' (Node)
    assert len(bus_1_element) == 1, f"Expected one bus with index {asset_bus}, got {len(bus_1_element)}"
    assert bus_1_element.type.iloc[0] == "n", f"Expected bus.type 'n', got {bus_1_element.type.iloc[0]}"

    # get all switches connected to the bus
    sl_disconnector = station_switches[(station_switches.bus == bus_1) | (station_switches.element == bus_1)]
    assert len(sl_disconnector) == 1, f"Expected one switch for SL connected to bus {bus_1}, got {len(sl_disconnector)}"
    assert sl_disconnector.et.iloc[0] == "b", f"Expected bus-bus switch, got {sl_disconnector.et.iloc[0]}"

    # check if switch is a disconnector or a circuit breaker
    # bus 1 / SL Switch is optional, so it can be a disconnector or a circuit breaker
    if sl_disconnector.type.iloc[0] == "CB":
        sl_disconnector = None
        bus_2 = bus_1
        condition_not_bus_1 = np.ones(len(station_switches), dtype=bool)
    else:
        assert sl_disconnector.type.iloc[0] == "DS", f"Expected switch type DS, got {sl_disconnector.type.iloc[0]}"
        bus_2 = sl_disconnector.element.iloc[0]
        if bus_2 == bus_1:
            bus_2 = sl_disconnector.bus.iloc[0]
        condition_not_bus_1 = (station_switches.bus != bus_1) & (station_switches.element != bus_1)

    # ---------- bus 2 - circuit breaker bus

    # get circuit breaker connected to the bus

    condition_bus_2 = (station_switches.bus == bus_2) | (station_switches.element == bus_2)
    circuit_breaker = station_switches[condition_not_bus_1 & condition_bus_2]
    assert len(circuit_breaker) == 1, (
        f"Expected one circuit breaker connected to bus {bus_2}, got {len(circuit_breaker)}, CB: {circuit_breaker.to_dict()}"
    )
    assert circuit_breaker.et.iloc[0] == "b", (
        f"Expected bus-bus switch, got {circuit_breaker.et.iloc[0]}, CB: {circuit_breaker.to_dict()}"
    )
    assert circuit_breaker.type.iloc[0] == "CB", (
        f"Expected switch type CB, got {circuit_breaker.type.iloc[0]}, CB: {circuit_breaker.to_dict()}"
    )
    bus_2_element = station_buses[station_buses.index == bus_2]
    assert len(bus_2_element) == 1, (
        f"Expected one bus with index {bus_2}, got {len(bus_2_element)}, CB: {circuit_breaker.to_dict()}"
    )
    assert bus_2_element.type.iloc[0] == "n", (
        f"Expected bus.type 'n', got {bus_2_element.type.iloc[0]}, CB: {circuit_breaker.to_dict()}"
    )

    # ---------- bus 3 - busbar section bus
    bus_3 = circuit_breaker.element.iloc[0]
    if bus_3 == bus_2:
        bus_3 = circuit_breaker.bus.iloc[0]

    # get sr disconnector to the final busbar
    condition_not_bus_2 = (station_switches.bus != bus_2) & (station_switches.element != bus_2)
    condition_bus_3 = (station_switches.bus == bus_3) | (station_switches.element == bus_3)
    sr_disconnectors = station_switches[condition_not_bus_2 & condition_bus_3]

    assert len(sr_disconnectors) != 0, (
        f"Expected one ore more switches for DS connected to the bus {bus_3}, got {len(sr_disconnectors)}"
    )
    assert all(sr_disconnectors.et == "b"), f"Expected bus-bus switch, got {sr_disconnectors.et.to_list()}"
    assert all(sr_disconnectors.type == "DS"), f"Expected switch type DS, got {sr_disconnectors.type.to_list()}"
    bus_3_element = station_buses[station_buses.index == bus_3]
    assert len(bus_3_element) == 1, f"Expected one bus with index {bus_3}, got {len(bus_3_element)}"
    assert bus_3_element.type.iloc[0] == "n", f"Expected bus.type 'n', got {bus_3_element.type.iloc[0]}"

    # get final busbars by iterating over all sr disconnectors
    final_buses = {}
    for _, sr_disconnector in sr_disconnectors.iterrows():
        final_bus = sr_disconnector.element
        if final_bus == bus_3:
            final_bus = sr_disconnector.bus
        final_bus_element = station_buses[station_buses.index == final_bus]
        assert len(final_bus_element) != 0, f"Expected one bus with index {final_bus}, got {len(final_bus_element)}"
        assert final_bus_element.type.iloc[0] == "b", f"Expected bus.type 'b', got {final_bus_element.type.iloc[0]}"
        final_buses[f"{final_bus}{SEPARATOR}bus"] = sr_disconnector[save_col_name]

    if sl_disconnector is not None:
        sl_switch_grid_model_id = sl_disconnector[save_col_name].iloc[0]
    else:
        sl_switch_grid_model_id = None

    asset_connection = AssetBay(
        asset_bay_id=build_asset_bay_id(station_grid_model_id, asset_grid_model_id),
        sl_switch_grid_model_id=sl_switch_grid_model_id,
        dv_switch_grid_model_id=circuit_breaker[save_col_name].iloc[0],
        sr_switch_grid_model_id=final_buses,
    )

    return asset_connection


def get_branch_from_bus_ids(
    branch_df: pd.DataFrame,
    branch_type: str,
    bus_ids: List[int] | pd.Index,
    bus_types: List[Tuple[str, Optional[str], str]],
) -> pd.DataFrame:
    """Get the branches based on branch_type and bus_ids.

    Parameters
    ----------
    branch_df: pd.DataFrame
        DataFrame with one or multiple busses to get the bus id from.
        Note: The DataFrame columns are the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.
    branch_type: str
        Branch type that should be retrieved.
        It needs to be an attribute of the pandapower network.
    bus_ids: List[int]
        List of bus indices for which the busses should be retrieved.
    bus_types: List[Tuple[str, Optional[str], str]]
        List of the tuple(name_of_column, pydantic_type, postfix_gridmodel_id).
        This list is used to identify the bus columns in the branch types.

    Returns
    -------
    branch_df_all_busses: pd.DataFrame
        DataFrame with the branches of the station_name.
        Note: The DataFrame columns NOT yet the same as in the pydantic model.
        Note: the index of the DataFrame is the internal id of the bus.

    Raises
    ------
    ValueError:
        If branch_type is not found in the pp.pandapowerNet.

    """
    branch_df["branch_end"] = None
    # rename bus columns to bus_int_id
    # get all elements from bus columns
    branch_df_col_list = []
    for bus_col_name, pydantic_type, postfix_gridmodel_id in bus_types:
        if bus_col_name in branch_df.columns:
            branch_df_col = branch_df[branch_df[bus_col_name].isin(bus_ids)].copy()
            branch_df_col.loc[branch_df_col[bus_col_name].isin(bus_ids), "branch_end"] = pydantic_type
            branch_df_col.rename(columns={bus_col_name: "bus_int_id"}, inplace=True)
            # get grid_model_id from index and branch_type
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
            + f"using bus_type: '{bus_types}' in columns: '{branch_df.columns}'"
        )
    branch_df_all_busses = pd.concat(branch_df_col_list)
    return branch_df_all_busses
