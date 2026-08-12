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
from beartype.typing import Any, List, get_args
from toop_engine_grid_helpers.pandapower.station_extraction import (
    get_branches_from_station,
    get_busses_from_station,
    get_parameter_from_station,
    get_substation_buses_from_bus_id,
)
from toop_engine_grid_helpers.pandapower.station_extraction import (
    get_coupler_from_station as _get_coupler_from_station,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
)
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    InjectionAsset,
    build_asset_bay_id,
)
from toop_engine_interfaces.asset_topology.topology_conversion import validate_complete_master_asset_topology

logger = structlog.get_logger(__name__)


def get_coupler_from_station(
    network: pp.pandapowerNet,
    station_buses: pd.DataFrame,
    foreign_key: str = "equipment",
) -> pd.DataFrame:
    """Return the legacy public coupler dataframe shape for pandapower callers.

    The canonical builder inside this module still consumes the full helper payload,
    including ``coupler_bay`` metadata, but the long-standing public helper exposed by
    this module historically returned the leaner dataframe without that column.
    """
    coupler_df = _get_coupler_from_station(network=network, station_buses=station_buses, foreign_key=foreign_key)
    if "coupler_bay" in coupler_df.columns:
        return coupler_df.drop(columns=["coupler_bay"])
    return coupler_df


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
        asset_disconnector_grid_model_id=None,
        breaker_grid_model_id=f"{asset_bay_id}::breaker",
        busbar_disconnector_grid_model_id={
            busbar_grid_model_id: f"{asset_bay_id}::busbar_disconnector::{busbar_grid_model_id}"
        },
    )


def _build_station_assets_and_connections(
    station_branches: pd.DataFrame,
    asset_connection_path: list[AssetBay | None],
) -> tuple[
    list[BranchAsset],
    list[InjectionAsset],
    list[BusGroupAssetConnection],
    list[BusGroupAssetConnection],
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
    branch_connections: list[BusGroupAssetConnection] = []
    injection_connections: list[BusGroupAssetConnection] = []
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
        connection = BusGroupAssetConnection(
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


def _build_master_bus_group_from_station_id(
    network: pp.pandapowerNet,
    station_id_list: list[int],
    group_index: int = 0,
    foreign_key: str = "equipment",
) -> tuple[MasterBusGroup, list[BranchAsset], list[InjectionAsset], list[AssetBay]]:
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
    tuple[MasterBusGroup, list[BranchAsset], list[InjectionAsset], list[AssetBay]]
        Canonical station plus the topology-owned payloads it references.
    """
    station_buses = get_busses_from_station(network, station_bus_index=station_id_list, foreign_key=foreign_key)
    coupler_elements = _get_coupler_from_station(network, station_buses, foreign_key=foreign_key)
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
    master_station = MasterBusGroup(
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


def get_master_asset_topology_from_network(
    network: pp.pandapowerNet,
    topology_id: str,
    grid_model_file: str,
    station_id_list: List[List[int]],
    foreign_key: str = "equipment",
) -> MasterAssetTopology:
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
    MasterAssetTopology
        Canonical master data split into structural station groups.
    """
    master_stations: list[MasterBusGroup] = []
    branch_assets_by_id: dict[str, BranchAsset] = {}
    injection_assets_by_id: dict[str, InjectionAsset] = {}
    asset_bays_by_id: dict[str, AssetBay] = {}

    for station_ids in station_id_list:
        structural_groups = _get_structural_station_bus_groups(list(station_ids), network)
        for group_index, structural_group in enumerate(structural_groups):
            master_station, branch_assets, injection_assets, asset_bays = _build_master_bus_group_from_station_id(
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

    master_data = MasterAssetTopology(
        topology_id=topology_id,
        grid_model_file=grid_model_file,
        stations=master_stations,
        branch_assets=list(branch_assets_by_id.values()),
        injection_assets=list(injection_assets_by_id.values()),
        asset_bays=list(asset_bays_by_id.values()),
    )
    validate_complete_master_asset_topology(master_data)
    return master_data


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
