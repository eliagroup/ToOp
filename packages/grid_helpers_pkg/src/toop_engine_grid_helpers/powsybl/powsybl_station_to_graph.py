# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert a pypowsybl network to a NetworkGraph."""

from dataclasses import dataclass
from string import ascii_lowercase

import networkx as nx
import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import Any, get_args
from pydantic import ValidationError
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.network_graph.data_classes import (
    SWITCH_TYPES,
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SubstationInformation,
    SwitchSchema,
)
from toop_engine_grid_helpers.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_grid_helpers.network_graph.graph_to_asset_topo import (
    get_asset_bay,
    get_busbar_df,
    get_coupler_df,
    get_station_asset_connectivity_table,
    get_switchable_asset,
)
from toop_engine_grid_helpers.network_graph.network_graph import (
    generate_graph,
    get_busbar_connection_info,
    get_edge_connection_info,
)
from toop_engine_grid_helpers.network_graph.network_graph_data import add_graph_specific_data
from toop_engine_grid_helpers.network_graph.network_graph_helper_functions import (
    remove_suffix_from_switchable_assets,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    _get_busbar_sections_with_in_service,
    get_all_element_names,
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
)
from toop_engine_grid_helpers.powsybl.powsybl_helpers import get_voltage_level_with_region
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
)
from toop_engine_interfaces.asset_topology.topology_conversion import validate_complete_master_asset_topology
from toop_engine_interfaces.messages.preprocess.preprocess_commands import CgmesImporterParameters
from toop_engine_interfaces.network_masks import NetworkMasks

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


@dataclass
class _StructuralStationContext:
    """Cached node-breaker data reused across structural station views."""

    station_info: SubstationInformation
    graph_data: NetworkGraphData
    graph: nx.Graph
    busbar_df: pd.DataFrame
    full_busbar_connection_info: dict[str, object]
    edge_connection_info: dict[str, object]


@dataclass
class _NodeBreakerNetworkContext:
    """Cached network-wide node-breaker data reused across voltage levels."""

    all_names_df: pd.Series
    asset_in_service: pd.Series
    boundary_line_tie_ids: pd.Series
    bus_breaker_view_buses_df: pd.DataFrame
    busbar_sections_names_df: pd.DataFrame


@dataclass
class _StructuralStationView:
    """Structural station view paired with its cached voltage-level context."""

    structural_station_id: str
    selected_busbar_ids: set[str]
    station_context: _StructuralStationContext


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


def _build_structural_station_context(
    network: Network,
    station_info: SubstationInformation,
    network_context: _NodeBreakerNetworkContext | None = None,
) -> _StructuralStationContext:
    """Build cached node-breaker data for one voltage level."""
    graph_data = node_breaker_topology_to_graph_data(
        network,
        substation_info=station_info,
        network_context=network_context,
    )
    graph = get_node_breaker_topology_graph(graph_data)
    busbar_df = get_busbar_df(nodes_df=graph_data.nodes, substation_id=station_info.name)
    full_busbar_connection_info = get_busbar_connection_info(graph=graph)
    edge_connection_info = get_edge_connection_info(graph=graph)
    return _StructuralStationContext(
        station_info=station_info,
        graph_data=graph_data,
        graph=graph,
        busbar_df=busbar_df,
        full_busbar_connection_info=full_busbar_connection_info,
        edge_connection_info=edge_connection_info,
    )


def _get_station_topology_frames(
    network: Network,
    selected_busbar_ids: set[str],
    station_grid_model_id: str,
    station_info: SubstationInformation | None = None,
    station_context: _StructuralStationContext | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    object,
    pd.DataFrame,
    dict[str, AssetBay],
    list[str],
]:
    """Collect station-local frames needed to derive canonical master data.

    Parameters
    ----------
    network : Network
        Source powsybl network used to build a cached context when needed.
    station_info : SubstationInformation | None
        Voltage-level metadata used when callers do not provide ``station_context``.
    station_context : _StructuralStationContext | None
        Cached node-breaker data for the voltage level currently being processed.
    selected_busbar_ids : set[str]
        Structural busbar ids that belong to the station group.
    station_grid_model_id : str
        Canonical bus-group id to assign to the resulting station view.

    Returns
    -------
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        object,
        pd.DataFrame,
        dict[str, AssetBay],
        list[str],
    ]
        Filtered busbars, filtered couplers, restricted busbar connection info,
        switchable asset rows, asset bays keyed by lookup id, and collected asset-bay logs.
    """
    if station_context is None:
        if station_info is None:
            raise ValueError("station_info must be provided when station_context is None")
        station_context = _build_structural_station_context(network=network, station_info=station_info)

    substation_id = station_context.station_info.name
    graph_data = station_context.graph_data
    graph = station_context.graph

    busbar_df, selected_busbar_ids, busbar_connection_info = _get_station_busbar_view_from_group(
        full_busbar_df=station_context.busbar_df,
        selected_busbar_ids=selected_busbar_ids,
        full_busbar_connection_info=station_context.full_busbar_connection_info,
        substation_id=substation_id,
    )

    coupler_df = get_coupler_df(
        switches_df=graph_data.switches, busbar_df=busbar_df, substation_id=substation_id, graph=graph
    )
    if {"busbar_from_id", "busbar_to_id"}.issubset(coupler_df.columns):
        coupler_df = coupler_df.dropna(subset=["busbar_from_id", "busbar_to_id"]).copy()
    else:
        coupler_df = coupler_df.iloc[0:0].copy()
    if not coupler_df.empty:
        coupler_df[["busbar_from_id", "busbar_to_id"]] = coupler_df[["busbar_from_id", "busbar_to_id"]].astype(int)

    switchable_assets_df = get_switchable_asset(busbar_connection_info, graph_data.node_assets, graph_data.branches)
    connected_asset_ids = {
        asset_grid_model_id
        for connection_info in busbar_connection_info.values()
        for asset_grid_model_id in connection_info.connectable_assets
    }
    switchable_assets_df = switchable_assets_df[switchable_assets_df["grid_model_id"].isin(connected_asset_ids)].reset_index(
        drop=True
    )

    asset_bays_by_asset_id, station_logs = _get_station_asset_bays(
        switches_df=graph_data.switches,
        switchable_assets_df=switchable_assets_df,
        busbar_df=busbar_df,
        edge_connection_info=station_context.edge_connection_info,
        station_grid_model_id=station_grid_model_id,
        selected_busbar_ids=selected_busbar_ids,
    )
    return busbar_df, coupler_df, busbar_connection_info, switchable_assets_df, asset_bays_by_asset_id, station_logs


def _build_station_connectivity_by_asset_type(
    busbar_connection_info: object,
    busbar_df: pd.DataFrame,
    switchable_assets_df: pd.DataFrame,
    branch_mask: list[bool],
) -> tuple[object, object]:
    """Build branch and injection connectivity tables for one station view.

    Parameters
    ----------
    busbar_connection_info : object
        Restricted busbar connection info for the station-local view.
    busbar_df : pd.DataFrame
        Filtered station-local busbar view.
    switchable_assets_df : pd.DataFrame
        Station-local asset rows aligned with the connectivity matrix columns.
    branch_mask : list[bool]
        Boolean mask indicating which asset columns are branch assets.

    Returns
    -------
    tuple[object, object]
        Branch connectivity and injection connectivity tables derived from the
        station-local connection matrix.
    """
    asset_connectivity = get_station_asset_connectivity_table(
        busbar_connection_info,
        busbar_df=busbar_df,
        switchable_assets_df=switchable_assets_df,
    )
    return asset_connectivity[:, branch_mask], asset_connectivity[:, [not is_branch for is_branch in branch_mask]]


def _expand_busbars_connected_via_switches(
    seed_busbar_ids: set[str], full_busbar_connection_info: dict[str, object], allowed_busbar_ids: set[str]
) -> set[str]:
    """Expand a bus view with busbars reachable through switchable couplers.

    Parameters
    ----------
    seed_busbar_ids : set[str]
        Initial busbar ids that seed the search.
    full_busbar_connection_info : dict[str, object]
        Mapping of busbar ids to connection metadata exposing ``connectable_busbars``.
    allowed_busbar_ids : set[str]
        Busbar ids that may be included in the expanded view.

    Returns
    -------
    set[str]
        Expanded set of reachable busbar ids restricted to ``allowed_busbar_ids``.
    """
    selected_busbar_ids = set(seed_busbar_ids)
    frontier = set(seed_busbar_ids)

    while frontier:
        busbar_id = frontier.pop()
        connection_info = full_busbar_connection_info.get(busbar_id)
        if connection_info is None:
            continue
        connected_busbar_ids = set(connection_info.connectable_busbars) & allowed_busbar_ids
        new_busbar_ids = connected_busbar_ids - selected_busbar_ids
        selected_busbar_ids.update(new_busbar_ids)
        frontier.update(new_busbar_ids)

    return selected_busbar_ids


def _get_structural_busbar_groups(
    graph: nx.Graph,
    allowed_busbar_ids: set[str],
) -> list[set[str]]:
    """Group busbars by switch-only structural connectivity.

    Parameters
    ----------
    graph : nx.Graph
        Node-breaker graph for the voltage level.
    allowed_busbar_ids : set[str]
        Busbar ids that are eligible for the current station view.

    Returns
    -------
    list[set[str]]
        Deterministic structural busbar groups independent of runtime switch state.
    """
    switch_types = set(get_args(SWITCH_TYPES))
    structural_graph = nx.Graph()
    structural_graph.add_nodes_from(graph.nodes(data=True))
    structural_graph.add_edges_from(
        (from_node, to_node)
        for from_node, to_node, edge_data in graph.edges(data=True)
        if edge_data.get("asset_type") in switch_types
    )

    busbar_node_ids = {
        node_id
        for node_id, node_data in structural_graph.nodes(data=True)
        if node_data.get("node_type") == "busbar" and node_data.get("grid_model_id") in allowed_busbar_ids
    }
    structural_groups: list[set[str]] = []
    for component in nx.connected_components(structural_graph):
        component_busbar_ids = {
            str(structural_graph.nodes[node_id]["grid_model_id"]) for node_id in component if node_id in busbar_node_ids
        }
        if component_busbar_ids:
            structural_groups.append(component_busbar_ids)

    structural_groups.sort(key=sorted)
    return structural_groups


def _structural_station_suffix(group_index: int) -> str:
    """Return a deterministic alphabetic suffix for structural station ids.

    Parameters
    ----------
    group_index : int
        Zero-based structural group index.

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


def _get_structural_station_views(
    network: Network,
    relevant_voltage_level_with_region: pd.DataFrame,
) -> list[_StructuralStationView]:
    """Enumerate structural station views with synthetic ids and busbar groups.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    relevant_voltage_level_with_region : pd.DataFrame
        Relevant voltage-level rows including naming and region metadata.

    Returns
    -------
    list[_StructuralStationView]
        Canonical station ids, structural busbar groups, and the cached
        voltage-level context they reuse.
    """
    structural_station_views: list[_StructuralStationView] = []
    voltage_level_rows = relevant_voltage_level_with_region.drop_duplicates(subset=["voltage_level_id"])
    network_context = _build_node_breaker_network_context(network)

    for _bus_id, row in voltage_level_rows.iterrows():
        station_info = SubstationInformation(
            name=row["name"],
            region=row["region"],
            nominal_v=row["nominal_v"],
            voltage_level_id=row["voltage_level_id"],
        )
        station_context = _build_structural_station_context(
            network=network,
            station_info=station_info,
            network_context=network_context,
        )
        structural_groups = _get_structural_busbar_groups(
            graph=station_context.graph,
            allowed_busbar_ids=set(station_context.busbar_df["grid_model_id"]),
        )

        for group_index, structural_group in enumerate(structural_groups):
            structural_station_id = f"{station_info.voltage_level_id}_{_structural_station_suffix(group_index)}"
            structural_station_views.append(
                _StructuralStationView(
                    structural_station_id=structural_station_id,
                    selected_busbar_ids=structural_group,
                    station_context=station_context,
                )
            )

    return structural_station_views


def _get_station_busbar_view(
    graph: nx.Graph,
    graph_data: NetworkGraphData,
    bus_id: str,
    substation_id: str,
) -> tuple[pd.DataFrame, set[str], dict[str, object]]:
    """Build the station-local busbar view and matching connection metadata.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the node-breaker topology.
    graph_data : NetworkGraphData
        Graph data containing the station nodes used to derive busbars.
    bus_id : str
        Bus-branch bus identifier for the station view to build.
    substation_id : str
        Substation name used when deriving station-local busbars.

    Returns
    -------
    tuple[pd.DataFrame, set[str], dict[str, object]]
        The filtered busbar DataFrame with reassigned local integer ids, the selected
        busbar ids, and the busbar connection info restricted to that station view.
    """
    busbar_df = get_busbar_df(nodes_df=graph_data.nodes, substation_id=substation_id)
    seed_busbar_ids = set(busbar_df.loc[busbar_df["bus_branch_bus_id"] == bus_id, "grid_model_id"])
    full_busbar_connection_info = get_busbar_connection_info(graph=graph)
    selected_busbar_ids = _expand_busbars_connected_via_switches(
        seed_busbar_ids=seed_busbar_ids,
        full_busbar_connection_info=full_busbar_connection_info,
        allowed_busbar_ids=set(busbar_df["grid_model_id"]),
    )
    busbar_df = busbar_df[busbar_df["grid_model_id"].isin(selected_busbar_ids)].copy().reset_index(drop=True)
    busbar_df["int_id"] = busbar_df.index
    if busbar_df.empty:
        raise ValueError(f"No busbars found for bus_id {bus_id} in substation {substation_id}")

    selected_busbar_ids = set(busbar_df["grid_model_id"])
    busbar_connection_info = {
        busbar_grid_model_id: connection_info
        for busbar_grid_model_id, connection_info in full_busbar_connection_info.items()
        if busbar_grid_model_id in selected_busbar_ids
    }
    return busbar_df, selected_busbar_ids, busbar_connection_info


def _get_station_busbar_view_from_group(
    full_busbar_df: pd.DataFrame,
    selected_busbar_ids: set[str],
    full_busbar_connection_info: dict[str, object],
    substation_id: str,
) -> tuple[pd.DataFrame, set[str], dict[str, object]]:
    """Build the station-local busbar view directly from a structural busbar group.

    Parameters
    ----------
    full_busbar_df : pd.DataFrame
        Cached busbar DataFrame for the voltage level.
    selected_busbar_ids : set[str]
        Structural busbar ids belonging to the current station group.
    full_busbar_connection_info : dict[str, object]
        Cached busbar-connection metadata for the voltage level.
    substation_id : str
        Human-readable substation identifier used for error messages.

    Returns
    -------
    tuple[pd.DataFrame, set[str], dict[str, object]]
        Filtered busbar DataFrame, retained busbar ids, and restricted connection info.
    """
    busbar_df = full_busbar_df[full_busbar_df["grid_model_id"].isin(selected_busbar_ids)].copy().reset_index(drop=True)
    busbar_df["int_id"] = busbar_df.index
    if busbar_df.empty:
        raise ValueError(f"No busbars found for selected group in substation {substation_id}")

    selected_busbar_ids = set(busbar_df["grid_model_id"])
    busbar_connection_info = {
        busbar_grid_model_id: connection_info
        for busbar_grid_model_id, connection_info in full_busbar_connection_info.items()
        if busbar_grid_model_id in selected_busbar_ids
    }
    return busbar_df, selected_busbar_ids, busbar_connection_info


def _get_station_asset_bays(
    switches_df: pd.DataFrame,
    switchable_assets_df: pd.DataFrame,
    busbar_df: pd.DataFrame,
    edge_connection_info: dict[str, object],
    station_grid_model_id: str,
    selected_busbar_ids: set[str],
) -> tuple[dict[str, AssetBay], list[str]]:
    """Build station-local asset bays filtered to the selected busbar view.

    Parameters
    ----------
    switches_df : pd.DataFrame
        Switch rows for the station topology.
    switchable_assets_df : pd.DataFrame
        Asset rows aligned with the station-local switching table.
    busbar_df : pd.DataFrame
        Filtered station-local busbar view.
    edge_connection_info : dict[str, object]
        Edge metadata used by ``get_asset_bay(...)`` to derive bay paths.
    station_grid_model_id : str
        Structural station identifier owning the station-local asset bays.
    selected_busbar_ids : set[str]
        Busbar ids kept in the station-local busbar view.

    Returns
    -------
    tuple[dict[str, AssetBay], list[str]]
        Asset bays keyed by asset grid model id and the collected station log messages.
    """
    station_logs: list[str] = []
    asset_bays_by_asset_id: dict[str, AssetBay] = {}
    for asset_grid_model_id in switchable_assets_df["grid_model_id"].to_list():
        asset_bay, logs = get_asset_bay(
            switches_df,
            station_grid_model_id=station_grid_model_id,
            asset_grid_model_id=asset_grid_model_id,
            busbar_df=busbar_df,
            edge_connection_info=edge_connection_info,
        )
        station_logs.extend(logs)
        if asset_bay is None:
            continue

        station_local_busbar_disconnectors = {
            busbar_grid_model_id: switch_grid_model_id
            for busbar_grid_model_id, switch_grid_model_id in asset_bay.busbar_disconnector_grid_model_id.items()
            if busbar_grid_model_id in selected_busbar_ids
        }
        if len(station_local_busbar_disconnectors) == 0:
            continue

        asset_bays_by_asset_id[asset_grid_model_id] = asset_bay.model_copy(
            update={"busbar_disconnector_grid_model_id": station_local_busbar_disconnectors},
            deep=True,
        )

    return asset_bays_by_asset_id, station_logs


def _get_station_asset_connections(
    network: Network,
    station_info: SubstationInformation,
    busbar_df: pd.DataFrame,
    switchable_assets_df: pd.DataFrame,
    asset_bays_by_asset_id: dict[str, AssetBay],
    branches: pd.DataFrame | None = None,
) -> tuple[
    list[BranchAsset],
    list[InjectionAsset],
    list[BusGroupAssetConnection],
    list[BusGroupAssetConnection],
    list[bool],
    list[AssetBay],
]:
    """Build canonical assets and station-local connections aligned with station tables.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    station_info : SubstationInformation
        Metadata for the voltage level owning the station view.
    busbar_df : pd.DataFrame
        Filtered station-local busbar view.
    switchable_assets_df : pd.DataFrame
        Station-local asset rows aligned with the switching table.
    asset_bays_by_asset_id : dict[str, AssetBay]
        Station-local asset bays keyed by lookup id.
    branches : pd.DataFrame | None
        Optional preloaded global branch table used to infer station-local branch ends.

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
    asset_bay_lookup_ids = switchable_assets_df["grid_model_id"].to_list()
    assets = [_build_canonical_asset(asset_payload) for asset_payload in switchable_assets_df.to_dict(orient="records")]
    remove_suffix_from_switchable_assets(assets)
    if branches is None:
        branches = network.get_branches(attributes=["voltage_level1_id", "voltage_level2_id", "bus1_id", "bus2_id"])
    local_bus_ids = {bus_id for bus_id in busbar_df["bus_branch_bus_id"].dropna().tolist() if bus_id}

    branch_mask = [isinstance(asset, BranchAsset) for asset in assets]
    branch_assets: list[BranchAsset] = []
    injection_assets: list[InjectionAsset] = []
    branch_connections: list[BusGroupAssetConnection] = []
    injection_connections: list[BusGroupAssetConnection] = []
    asset_bays: list[AssetBay] = []
    for asset, asset_bay_lookup_id, is_branch in zip(assets, asset_bay_lookup_ids, branch_mask, strict=True):
        branch_end = _infer_branch_end(
            asset_grid_model_id=asset.grid_model_id,
            asset_bay_lookup_id=asset_bay_lookup_id,
            station_info=station_info,
            local_bus_ids=local_bus_ids,
            branches=branches,
        )
        asset_bay = asset_bays_by_asset_id.get(asset_bay_lookup_id)
        if asset_bay is not None:
            asset_bays.append(asset_bay.model_copy(deep=True))
        connection = BusGroupAssetConnection(
            asset_id=asset.grid_model_id,
            branch_end=branch_end,
            asset_bay_id=asset_bay.asset_bay_id if asset_bay is not None else None,
        )
        if is_branch:
            branch_assets.append(asset.model_copy(deep=True))
            branch_connections.append(connection)
        else:
            injection_assets.append(asset.model_copy(deep=True))
            injection_connections.append(connection)

    return branch_assets, injection_assets, branch_connections, injection_connections, branch_mask, asset_bays


def _infer_branch_end(
    asset_grid_model_id: str,
    asset_bay_lookup_id: str,
    station_info: SubstationInformation,
    local_bus_ids: set[str],
    branches: pd.DataFrame,
) -> str | None:
    """Infer canonical branch_end metadata for one station-local branch occurrence.

    Parameters
    ----------
    asset_grid_model_id : str
        Canonical branch asset id without any duplicated lookup suffix.
    asset_bay_lookup_id : str
        Station-local lookup id which may still carry duplicated lookup suffixes.
    station_info : SubstationInformation
        Structural station information for the current materialized station.
    local_bus_ids : set[str]
        Local bus ids visible in the current station view.
    branches : pd.DataFrame
        Live powsybl branch table indexed by canonical branch id.

    Returns
    -------
    str | None
        Canonical branch end for the station-local occurrence, or ``None`` if no unambiguous
        orientation can be derived from the available source data.
    """
    branch_end_from_lookup_id = _infer_branch_end_from_lookup_id(asset_bay_lookup_id)
    if branch_end_from_lookup_id is not None:
        return branch_end_from_lookup_id
    if asset_grid_model_id not in branches.index:
        return None

    branch_entry = branches.loc[asset_grid_model_id]
    if branch_entry["voltage_level1_id"] != branch_entry["voltage_level2_id"]:
        return _infer_inter_voltage_level_branch_end(branch_entry, station_info)
    return _infer_same_voltage_level_branch_end(branch_entry, local_bus_ids)


def _infer_branch_end_from_lookup_id(asset_bay_lookup_id: str) -> str | None:
    """Infer a branch end from the duplicated station-local lookup suffix.

    Parameters
    ----------
    asset_bay_lookup_id : str
        Station-local lookup id that may carry `_FROM` or `_TO` suffixes.

    Returns
    -------
    str | None
        `from`, `to`, or ``None`` if no explicit suffix is present.
    """
    if asset_bay_lookup_id.endswith("_FROM"):
        return "from"
    if asset_bay_lookup_id.endswith("_TO"):
        return "to"
    return None


def _infer_inter_voltage_level_branch_end(branch_entry: pd.Series, station_info: SubstationInformation) -> str | None:
    """Infer branch end for branches spanning two voltage levels.

    Parameters
    ----------
    branch_entry : pd.Series
        Canonical branch table row.
    station_info : SubstationInformation
        Metadata for the current structural station view.

    Returns
    -------
    str | None
        Canonical branch end relative to the current station, if inferable.
    """
    if branch_entry["voltage_level1_id"] == station_info.voltage_level_id:
        return "from"
    if branch_entry["voltage_level2_id"] == station_info.voltage_level_id:
        return "to"
    return None


def _infer_same_voltage_level_branch_end(branch_entry: pd.Series, local_bus_ids: set[str]) -> str | None:
    """Infer branch end for same-voltage-level branches from local bus visibility.

    Parameters
    ----------
    branch_entry : pd.Series
        Canonical branch table row.
    local_bus_ids : set[str]
        Runtime-visible bus ids for the station-local view.

    Returns
    -------
    str | None
        Canonical branch end relative to the current station, if inferable.
    """
    bus1_local = branch_entry["bus1_id"] in local_bus_ids
    bus2_local = branch_entry["bus2_id"] in local_bus_ids
    if bus1_local == bus2_local:
        return None
    return "from" if bus1_local else "to"


def _build_master_bus_group_from_busbar_group(
    network: Network,
    selected_busbar_ids: set[str],
    station_grid_model_id: str,
    station_info: SubstationInformation | None = None,
    station_context: _StructuralStationContext | None = None,
    branches: pd.DataFrame | None = None,
) -> tuple[MasterBusGroup, list[BranchAsset], list[InjectionAsset], list[AssetBay]]:
    """Build one canonical master station for a structural busbar group.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    station_info : SubstationInformation | None
        Voltage-level metadata used when callers do not provide ``station_context``.
    station_context : _StructuralStationContext | None
        Cached node-breaker data for the voltage level owning the station group.
    selected_busbar_ids : set[str]
        Structural busbar ids belonging to the station group.
    station_grid_model_id : str
        Canonical bus-group id for the resulting station.
    branches : pd.DataFrame | None
        Optional preloaded global branch table used to infer station-local branch ends.

    Returns
    -------
    tuple[MasterBusGroup, list[BranchAsset], list[InjectionAsset], list[AssetBay]]
        Canonical station plus the topology-owned payloads it references.
    """
    if station_context is None:
        if station_info is None:
            raise ValueError("station_info must be provided when station_context is None")
        station_context = _build_structural_station_context(network=network, station_info=station_info)

    station_info = station_context.station_info
    busbar_df, coupler_df, busbar_connection_info, switchable_assets_df, asset_bays_by_asset_id, _station_logs = (
        _get_station_topology_frames(
            network=network,
            station_info=station_info,
            station_context=station_context,
            selected_busbar_ids=selected_busbar_ids,
            station_grid_model_id=station_grid_model_id,
        )
    )
    busbars = get_list_of_busbars_from_df(busbar_df)
    couplers = get_list_of_coupler_from_df(coupler_df)
    branch_assets, injection_assets, branch_connections, injection_connections, branch_mask, asset_bays = (
        _get_station_asset_connections(
            network=network,
            station_info=station_info,
            busbar_df=busbar_df,
            switchable_assets_df=switchable_assets_df,
            asset_bays_by_asset_id=asset_bays_by_asset_id,
            branches=branches,
        )
    )
    branch_connectivity, injection_connectivity = _build_station_connectivity_by_asset_type(
        busbar_connection_info=busbar_connection_info,
        busbar_df=busbar_df,
        switchable_assets_df=switchable_assets_df,
        branch_mask=branch_mask,
    )

    master_station = MasterBusGroup(
        bus_group_id=station_grid_model_id,
        voltage_level_id=station_info.voltage_level_id,
        name=station_info.name,
        region=station_info.region,
        voltage_level=float(station_info.nominal_v),
        busbars=[
            Busbar(
                grid_model_id=busbar.grid_model_id,
                busbar_type=busbar.busbar_type,
                name=busbar.name,
                int_id=busbar.int_id,
                bus_breaker_bus_id=busbar.bus_breaker_bus_id,
            )
            for busbar in busbars
        ],
        couplers=[
            BusbarCoupler(
                grid_model_id=coupler.grid_model_id,
                coupler_type=coupler.coupler_type,
                name=coupler.name,
                asset_bay=coupler.asset_bay.model_copy(deep=True) if coupler.asset_bay is not None else None,
                coupler_bay=coupler.coupler_bay.model_copy(deep=True) if coupler.coupler_bay is not None else None,
            )
            for coupler in couplers
        ],
        branch_connections=branch_connections,
        injection_connections=injection_connections,
        branch_connectivity=branch_connectivity,
        injection_connectivity=injection_connectivity,
    )

    return master_station, branch_assets, injection_assets, asset_bays


def normalize_nullable_bool(series: pd.Series, default: bool) -> pd.Series:
    """Normalize nullable or object-backed boolean-like series without pandas downcast warnings."""
    return series.astype("boolean").fillna(default).astype(bool)


def _build_node_breaker_network_context(net: Network) -> _NodeBreakerNetworkContext:
    """Build global node-breaker inputs reused across all voltage levels of one network."""
    all_names_df = get_all_element_names(net, line_trafo_name_col="name")
    branches_df = net.get_branches(attributes=["connected1", "connected2", "bus1_id", "bus2_id"])
    boundary_line_tie_ids = net.get_boundary_lines(attributes=["tie_line_id"])["tie_line_id"]
    injections_df = net.get_injections(attributes=["connected", "bus_id"])
    buses_df = net.get_buses(attributes=["connected_component"])
    bus_breaker_view_buses_df = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    busbar_sections_names_df = _get_busbar_sections_with_in_service(
        network=net,
        attributes=["name", "bus_id", "in_service"],
    )

    in_main_connected_component = buses_df["connected_component"].fillna(0).eq(0)
    branch_in_service = (
        normalize_nullable_bool(branches_df["connected1"], default=False)
        & normalize_nullable_bool(branches_df["connected2"], default=False)
        & normalize_nullable_bool(branches_df["bus1_id"].map(in_main_connected_component), default=False)
        & normalize_nullable_bool(branches_df["bus2_id"].map(in_main_connected_component), default=False)
    ).rename("in_service")
    injection_in_service = (
        normalize_nullable_bool(injections_df["connected"], default=False)
        & normalize_nullable_bool(injections_df["bus_id"].map(in_main_connected_component), default=False)
    ).rename("in_service")

    return _NodeBreakerNetworkContext(
        all_names_df=all_names_df,
        asset_in_service=pd.concat([branch_in_service, injection_in_service]),
        boundary_line_tie_ids=boundary_line_tie_ids,
        bus_breaker_view_buses_df=bus_breaker_view_buses_df,
        busbar_sections_names_df=busbar_sections_names_df,
    )


def node_breaker_topology_to_graph_data(
    net: Network,
    substation_info: SubstationInformation,
    network_context: _NodeBreakerNetworkContext | None = None,
) -> NetworkGraphData:
    """Convert a node breaker topology to a NetworkGraph.

    This function is WIP.

    Parameters
    ----------
    net : Network
        The network to convert.
    substation_info : SubstationInformation
        The substation information to retrieve the node breaker topology.
    network_context : _NodeBreakerNetworkContext | None
        The network context to use for the conversion. If None, a new context will be built

    Returns
    -------
    NetworkGraphData.
    """
    if network_context is None:
        network_context = _build_node_breaker_network_context(net)

    nbt = net.get_node_breaker_topology(substation_info.voltage_level_id)
    bbt = net.get_bus_breaker_topology(substation_info.voltage_level_id)

    switches_df = get_switches(switches_df=nbt.switches)
    nodes_df = get_nodes(
        busbar_sections_names_df=network_context.busbar_sections_names_df,
        nodes_df=nbt.nodes,
        bus_breaker_elements_df=bbt.elements,
        bus_breaker_view_buses_df=network_context.bus_breaker_view_buses_df,
        switches_df=switches_df,
        substation_info=substation_info,
    )
    helper_branches = get_helper_branches(internal_connections_df=nbt.internal_connections)
    node_assets_df = get_node_assets(
        nodes_df=nodes_df,
        all_names_df=network_context.all_names_df,
        asset_in_service=network_context.asset_in_service,
        boundary_line_tie_ids=network_context.boundary_line_tie_ids,
    )

    graph_data = NetworkGraphData(
        nodes=nodes_df,
        switches=switches_df,
        helper_branches=helper_branches,
        node_assets=node_assets_df,
    )
    add_graph_specific_data(graph_data)
    return graph_data


def get_node_breaker_topology_graph(network_graph_data: NetworkGraphData) -> nx.Graph:
    """Get the network graph from the NetworkGraphData and run default filter.

    Parameters
    ----------
    network_graph_data : NetworkGraphData
        The NetworkGraphData containing the nodes, switches, branches and node_assets.

    Returns
    -------
    nx.Graph
        The network graph.
    """
    graph = generate_graph(network_graph_data)
    run_default_filter_strategy(graph=graph)
    return graph


@pa.check_types
def get_switches(switches_df: pd.DataFrame) -> pat.DataFrame[SwitchSchema]:
    """Get switches from a node breaker topology.

    Get the switches from a node breaker topology, rename and retype for the NetworkGraph.

    Parameters
    ----------
    switches_df : pd.DataFrame
        The switches DataFrame from the node NodeBreakerTopology.

    Returns
    -------
    switches_df : pat.DataFrame[SwitchSchema]
        The switches as a DataFrame, with renamed columns for the NetworkGraph.
    """
    switches_df.reset_index(inplace=True)
    switches_df.rename(
        columns={
            "id": "grid_model_id",
            "name": "foreign_id",
            "kind": "asset_type",
            "node1": "from_node",
            "node2": "to_node",
        },
        inplace=True,
    )
    switches_df.fillna({"foreign_id": ""}, inplace=True)
    cond = switches_df["foreign_id"] == ""
    switches_df.loc[cond, "foreign_id"] = switches_df.loc[cond, "grid_model_id"]
    switches_df["from_node"] = switches_df["from_node"].astype(int)
    switches_df["to_node"] = switches_df["to_node"].astype(int)
    # TODO: might need to be changed once there is more information about the in_service state
    switches_df["in_service"] = True
    return switches_df


@pa.check_types
def get_nodes(
    busbar_sections_names_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    bus_breaker_elements_df: pd.DataFrame,
    bus_breaker_view_buses_df: pd.DataFrame,
    switches_df: pd.DataFrame,
    substation_info: SubstationInformation,
) -> pat.DataFrame[NodeSchema]:
    """Get nodes from a node breaker topology.

    Get the nodes from a node breaker topology, rename and retype for the NetworkGraph.
    Adds additional information to the nodes.

    Parameters
    ----------
    busbar_sections_names_df : pd.DataFrame
        The busbar sections names.
        from _get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])
    nodes_df : pd.DataFrame
        The nodes DataFrame from the net.get_node_breaker_topology(voltage_level_id).nodes
    bus_breaker_elements_df : pd.DataFrame
        The elements DataFrame from the bus-breaker topology of the same voltage level.
    bus_breaker_view_buses_df : pd.DataFrame
        The bus breaker view buses DataFrame from the pypowsybl network.
    switches_df : pd.DataFrame
        The switches DataFrame from the node NodeBreakerTopology.
    substation_info : SubstationInformation
        The substation information to add as node information.

    Returns
    -------
    nodes_df : pat.DataFrame[NodeSchema]
        The nodes as a DataFrame, with renamed columns for the NetworkGraph.
    """
    nodes_df = nodes_df.merge(busbar_sections_names_df, left_on="connectable_id", right_index=True, how="left")
    busbar_bus_ids = bus_breaker_elements_df.loc[bus_breaker_elements_df["type"] == "BUSBAR_SECTION", ["bus_id"]].rename(
        columns={"bus_id": "bus_breaker_bus_id"}
    )
    nodes_df = nodes_df.merge(busbar_bus_ids, left_on="connectable_id", right_index=True, how="left")
    if "bus_id" not in nodes_df.columns:
        nodes_df = nodes_df.merge(bus_breaker_view_buses_df, left_on="bus_breaker_bus_id", right_index=True, how="left")
    nodes_df["grid_model_id"] = ""
    nodes_df["node_type"] = "node"
    nodes_df["substation_id"] = substation_info.name
    nodes_df["system_operator"] = substation_info.region
    nodes_df["voltage_level"] = int(substation_info.nominal_v)
    nodes_df["helper_node"] = False

    nodes_df.rename(columns={"name": "foreign_id"}, inplace=True)
    nodes_df.index = nodes_df.index.astype(int)
    cond_busbar = nodes_df["connectable_type"] == "BUSBAR_SECTION"
    nodes_df.loc[cond_busbar, "node_type"] = "busbar"
    nodes_df.loc[cond_busbar, "grid_model_id"] = nodes_df.loc[cond_busbar, "connectable_id"]
    cond_helper_node = (nodes_df["connectable_type"] == "") & (
        ~nodes_df.index.isin(switches_df["from_node"].to_list() + switches_df["to_node"].to_list())
    )
    nodes_df.loc[cond_helper_node, "helper_node"] = True
    nodes_df.fillna({"foreign_id": ""}, inplace=True)
    cond = nodes_df["foreign_id"] == ""
    nodes_df.loc[cond, "foreign_id"] = nodes_df.loc[cond, "grid_model_id"]
    nodes_df["helper_node"] = nodes_df["helper_node"].astype("boolean").fillna(False).astype(bool)
    nodes_df["in_service"] = nodes_df["in_service"].astype("boolean").fillna(True).astype(bool)
    return nodes_df


@pa.check_types
def get_helper_branches(internal_connections_df: pd.DataFrame) -> pat.DataFrame[HelperBranchSchema]:
    """Get helper branches from a node breaker topology.

    Get the helper branches from a node breaker topology, rename and retype for the NetworkGraph.

    Parameters
    ----------
    internal_connections_df : pd.DataFrame
        The internal connections DataFrame from the node NodeBreakerTopology.

    Returns
    -------
    helper_branches : pat.DataFrame[HelperBranchSchema]
        The helper branches as a DataFrame, with renamed columns for the NetworkGraph.
    """
    helper_branches = internal_connections_df
    helper_branches.rename(columns={"node1": "from_node", "node2": "to_node"}, inplace=True)
    helper_branches["from_node"] = helper_branches["from_node"].astype(int)
    helper_branches["to_node"] = helper_branches["to_node"].astype(int)
    # helper branches have no grid model id, but are needed for consistency of edges
    helper_branches["grid_model_id"] = ""
    # all helper branches are in service
    helper_branches["in_service"] = True
    return helper_branches


def get_node_assets(
    nodes_df: pd.DataFrame,
    all_names_df: pd.Series,
    asset_in_service: pd.Series,
    boundary_line_tie_ids: pd.Series,
) -> pat.DataFrame[NodeAssetSchema]:
    """Get node assets from a node breaker topology.

    Get the node assets from a node breaker topology, rename and retype for the NetworkGraph.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        The nodes DataFrame from the node NodeBreakerTopology.
    all_names_df : pd.Series
        The names of all elements in the network.
    asset_in_service : pd.Series
        Boolean service state by connectable id derived from the Powsybl network model.
    boundary_line_tie_ids : pd.Series
        Mapping from boundary line id to tie line id for paired boundary lines.

    Returns
    -------
    node_assets_df : pat.DataFrame[NodeAssetSchema]
        The node assets as a DataFrame, with renamed columns for the NetworkGraph
    """
    node_assets_df = nodes_df[
        (nodes_df["connectable_type"] != "") & (nodes_df["connectable_type"] != "BUSBAR_SECTION")
    ].copy()
    node_assets_df["grid_model_id"] = node_assets_df["connectable_id"]
    node_assets_df.reset_index(inplace=True, drop=False)
    node_assets_df["node"] = node_assets_df["node"].astype(int)
    node_assets_df.drop(columns=["foreign_id"], inplace=True)
    node_assets_df = node_assets_df.merge(all_names_df, how="left", left_on="grid_model_id", right_index=True)
    node_assets_df.rename(columns={"connectable_type": "asset_type", "name": "foreign_id"}, inplace=True)
    node_assets_df.fillna({"foreign_id": ""}, inplace=True)
    tie_line_ids = node_assets_df["grid_model_id"].map(boundary_line_tie_ids).fillna("")
    paired_boundary_lines = (node_assets_df["asset_type"] == "BOUNDARY_LINE") & tie_line_ids.ne("")
    node_assets_df.loc[paired_boundary_lines, "grid_model_id"] = tie_line_ids[paired_boundary_lines]
    node_assets_df.loc[paired_boundary_lines, "asset_type"] = "TIE_LINE"
    node_assets_df.loc[paired_boundary_lines, "foreign_id"] = (
        node_assets_df.loc[paired_boundary_lines, "grid_model_id"].map(all_names_df).fillna("")
    )
    node_assets_df = node_assets_df[["grid_model_id", "foreign_id", "node", "asset_type"]]
    node_assets_df["in_service"] = normalize_nullable_bool(
        node_assets_df["grid_model_id"].map(asset_in_service), default=True
    )
    return node_assets_df


def _master_asset_topology_from_structural_station_views(
    network: Network,
    relevant_voltage_level_with_region: pd.DataFrame,
    importer_parameters: CgmesImporterParameters,
) -> MasterAssetTopology:
    """Build canonical master data from structural station views.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    relevant_voltage_level_with_region : pd.DataFrame
        Relevant voltage-level rows enriched with naming and region metadata.
    importer_parameters : CgmesImporterParameters
        Import configuration providing the topology identifier and source file name.

    Returns
    -------
    MasterAssetTopology
        Canonical master data assembled directly from structural station groups.
    """
    master_stations: list[MasterBusGroup] = []
    branch_assets_by_id: dict[str, BranchAsset] = {}
    injection_assets_by_id: dict[str, InjectionAsset] = {}
    asset_bays_by_id: dict[str, AssetBay] = {}
    branches = network.get_branches(attributes=["voltage_level1_id", "voltage_level2_id", "bus1_id", "bus2_id"])

    for structural_station_view in _get_structural_station_views(
        network=network,
        relevant_voltage_level_with_region=relevant_voltage_level_with_region,
    ):
        station_info = structural_station_view.station_context.station_info
        try:
            master_station, branch_assets, injection_assets, asset_bays = _build_master_bus_group_from_busbar_group(
                network=network,
                selected_busbar_ids=structural_station_view.selected_busbar_ids,
                station_grid_model_id=structural_station_view.structural_station_id,
                station_context=structural_station_view.station_context,
                branches=branches,
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
        except ValidationError as error:
            logger.warning(
                f"ValidationError while building master station: {station_info} with error: {error}. "
                "Consider checking the Station or adding to ignore list."
            )
        except KeyError as error:
            logger.warning(
                f"KeyError while building master station: {station_info} with error: {error}. "
                "Consider checking the Station or adding to ignore list. "
                "Likely a maintenance busbar present - currently working."
            )
        except ValueError as error:
            logger.warning(
                f"ValueError while building master station: {station_info} with error: {error}. "
                "Consider checking the Station or adding to ignore list."
            )

    master_data = MasterAssetTopology(
        topology_id=importer_parameters.grid_model_file.name,
        grid_model_file=str(importer_parameters.grid_model_file.name),
        stations=master_stations,
        branch_assets=list(branch_assets_by_id.values()),
        injection_assets=list(injection_assets_by_id.values()),
        asset_bays=list(asset_bays_by_id.values()),
    )
    validate_complete_master_asset_topology(master_data)
    return master_data


def get_node_breaker_master_asset_topology(
    network: Network,
    network_masks: "NetworkMasks",
    importer_parameters: CgmesImporterParameters,
) -> MasterAssetTopology:
    """Return canonical master data derived directly from node-breaker graph extraction.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    network_masks : NetworkMasks
        Precomputed masks defining the relevant stations.
    importer_parameters : CgmesImporterParameters
        Import configuration providing the topology identifier and source file name.

    Returns
    -------
    MasterAssetTopology
        Canonical node-breaker station master data.
    """
    relevant_voltage_level_with_region = get_relevant_voltage_levels(network=network, network_masks=network_masks)
    return _master_asset_topology_from_structural_station_views(
        network=network,
        relevant_voltage_level_with_region=relevant_voltage_level_with_region,
        importer_parameters=importer_parameters,
    )


def get_relevant_voltage_levels(network: Network, network_masks: "NetworkMasks") -> pd.DataFrame:
    """Get all relevant voltage level from the network.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    network_masks: NetworkMasks
        Object with a relevant_subs mask aligned to the powsybl bus table

    Returns
    -------
    relevant_voltage_level_with_region_and_bus_id: pd.DataFrame
        DataFrame with the relevant voltage level and region information with bus_id as index.
    """
    attributes = ["name", "substation_id", "nominal_v", "high_voltage_limit", "low_voltage_limit", "region", "topology_kind"]
    voltage_levels = get_voltage_level_with_region(network, attributes=attributes)
    busses = network.get_buses()
    relevant_voltage_levels = busses[network_masks.relevant_subs]["voltage_level_id"]
    busbar_sections = network.get_busbar_sections(attributes=["bus_id"])
    busbar_outage_bus_ids = pd.Index(busbar_sections[network_masks.busbar_for_nminus1]["bus_id"].unique())
    relevant_busbar_buses = busses.loc[busses.index.intersection(busbar_outage_bus_ids)]
    relevant_voltage_levels = pd.concat(
        [relevant_voltage_levels, relevant_busbar_buses["voltage_level_id"]]
    ).drop_duplicates()
    relevant_voltage_level_with_region = voltage_levels[voltage_levels.index.isin(relevant_voltage_levels)]
    relevant_voltage_level_with_region_and_bus_id = relevant_voltage_level_with_region.merge(
        relevant_voltage_levels, left_index=True, right_on="voltage_level_id", how="left"
    )
    return relevant_voltage_level_with_region_and_bus_id
