# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert a pypowsybl network to a NetworkGraph."""

import datetime

import networkx as nx
import numpy as np
import pandas as pd
import pandera.typing as pat
import structlog
from pandera.errors import SchemaError
from pydantic import ValidationError
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    get_all_element_names,
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
    get_list_of_switchable_assets_from_df,
)
from toop_engine_importer.network_graph.data_classes import (
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SubstationInformation,
    SwitchSchema,
)
from toop_engine_importer.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_importer.network_graph.graph_to_asset_topo import (
    get_asset_bay,
    get_busbar_df,
    get_coupler_df,
    get_station_connection_tables,
    get_switchable_asset,
    remove_double_connections,
)
from toop_engine_importer.network_graph.network_graph import (
    generate_graph,
    get_busbar_connection_info,
    get_edge_connection_info,
)
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data
from toop_engine_importer.network_graph.network_graph_helper_functions import (
    remove_suffix_from_switchable_assets,
)
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import (
    get_busbar_sections_with_in_service,
    get_voltage_level_with_region,
)
from toop_engine_importer.pypowsybl_import.powsybl_masks import NetworkMasks
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.messages.preprocess.preprocess_commands import CgmesImporterParameters, DoubleConnectionHandling

logger = structlog.get_logger(__name__)


def _extract_busbar_connection_switch_updates(
    busbars: list,
    assets: list,
    asset_switching_table: np.ndarray,
) -> dict[str, bool]:
    """Derive switch open states from a normalized asset switching table.

    Parameters
    ----------
    busbars : list
        Busbars belonging to the station.
    assets : list
        Switchable assets belonging to the station.
    asset_switching_table : np.ndarray
        Normalized asset switching table.

    Returns
    -------
    dict[str, bool]
        Mapping from switch id to desired open state.
    """
    switch_updates: dict[str, bool] = {}
    for asset_index, asset in enumerate(assets):
        if asset.asset_bay is None:
            continue
        target_busbars = np.flatnonzero(asset_switching_table[:, asset_index])
        if target_busbars.size != 1:
            continue

        target_busbar_id = busbars[int(target_busbars[0])].grid_model_id
        for busbar_id, switch_id in asset.asset_bay.sr_switch_grid_model_id.items():
            switch_updates[switch_id] = busbar_id != target_busbar_id
    return switch_updates


def _build_station_components(network: Network, substation_id: str, station_info: SubstationInformation) -> dict:
    """Build station model components and selector-switch fixes from one station traversal."""
    station_logs = []
    graph_data = node_breaker_topology_to_graph_data(network, substation_info=station_info)
    graph = get_node_breaker_topology_graph(graph_data)

    busbar_df = get_busbar_df(nodes_df=graph_data.nodes, substation_id=substation_id)
    coupler_df = get_coupler_df(
        switches_df=graph_data.switches, busbar_df=busbar_df, substation_id=substation_id, graph=graph
    )
    busbar_connection_info = get_busbar_connection_info(graph=graph)
    edge_connection_info = get_edge_connection_info(graph=graph)
    switchable_assets_df = get_switchable_asset(busbar_connection_info, graph_data.node_assets, graph_data.branches)
    asset_bay_dict = {}
    for asset_grid_model_id in switchable_assets_df["grid_model_id"].to_list():
        asset_bay, logs = get_asset_bay(
            graph_data.switches,
            asset_grid_model_id=asset_grid_model_id,
            busbar_df=busbar_df,
            edge_connection_info=edge_connection_info,
        )
        station_logs.extend(logs)
        if asset_bay is None:
            continue
        asset_bay_dict[asset_grid_model_id] = asset_bay

    asset_connectivity, asset_switching_table, busbar_connectivity, busbar_switching_table = get_station_connection_tables(
        busbar_connection_info, busbar_df=busbar_df, switchable_assets_df=switchable_assets_df
    )
    had_double_connections = not np.all(asset_switching_table.sum(axis=0) <= 1)
    busbar_priority = asset_switching_table.sum(axis=1).astype(int)
    asset_switching_table = remove_double_connections(
        asset_switching_table,
        busbar_priority=busbar_priority,
        substation_id=substation_id,
    )
    busbars = get_list_of_busbars_from_df(busbar_df)
    couplers = get_list_of_coupler_from_df(coupler_df)
    assets = get_list_of_switchable_assets_from_df(station_branches=switchable_assets_df, asset_bay_dict=asset_bay_dict)
    remove_suffix_from_switchable_assets(assets)
    switch_updates = _extract_busbar_connection_switch_updates(busbars, assets, asset_switching_table)

    return {
        "busbars": busbars,
        "couplers": couplers,
        "assets": assets,
        "asset_switching_table": asset_switching_table,
        "asset_connectivity": asset_connectivity,
        "busbar_switching_table": busbar_switching_table,
        "busbar_connectivity": busbar_connectivity,
        "model_log": station_logs,
        "switch_updates": switch_updates,
        "had_double_connections": had_double_connections,
    }


def node_breaker_topology_to_graph_data(net: Network, substation_info: SubstationInformation) -> NetworkGraphData:
    """Convert a node breaker topology to a NetworkGraph.

    This function is WIP.

    Parameters
    ----------
    net : Network
        The network to convert.
    substation_info : SubstationInformation
        The substation information to retrieve the node breaker topology.

    Returns
    -------
    NetworkGraphData.
    """
    all_names_df = get_all_element_names(net, line_trafo_name_col="name")
    branches_df = net.get_branches(attributes=["connected1", "connected2", "bus1_id", "bus2_id"])
    injections_df = net.get_injections(attributes=["connected", "bus_id"])
    buses_df = net.get_buses(attributes=["connected_component"])
    in_main_connected_component = buses_df["connected_component"].fillna(0).eq(0)

    asset_in_service = pd.concat(
        [
            (
                branches_df["connected1"].fillna(False)
                & branches_df["connected2"].fillna(False)
                & branches_df["bus1_id"].map(in_main_connected_component).fillna(False)
                & branches_df["bus2_id"].map(in_main_connected_component).fillna(False)
            ).rename("in_service"),
            (
                injections_df["connected"].fillna(False)
                & injections_df["bus_id"].map(in_main_connected_component).fillna(False)
            ).rename("in_service"),
        ]
    )
    nbt = net.get_node_breaker_topology(substation_info.voltage_level_id)
    bbt = net.get_bus_breaker_topology(substation_info.voltage_level_id)

    switches_df = get_switches(switches_df=nbt.switches)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])
    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        bus_breaker_elements_df=bbt.elements,
        switches_df=switches_df,
        substation_info=substation_info,
    )
    helper_branches = get_helper_branches(internal_connections_df=nbt.internal_connections)
    node_assets_df = get_node_assets(
        nodes_df=nodes_df,
        all_names_df=all_names_df,
        asset_in_service=asset_in_service,
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


def get_nodes(
    busbar_sections_names_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    bus_breaker_elements_df: pd.DataFrame,
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
        from get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])
    nodes_df : pd.DataFrame
        The nodes DataFrame from the net.get_node_breaker_topology(voltage_level_id).nodes
    bus_breaker_elements_df : pd.DataFrame
        The elements DataFrame from the bus-breaker topology of the same voltage level.
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
    nodes_df.loc[cond_busbar, "bus_id"] = nodes_df.loc[cond_busbar, "bus_breaker_bus_id"]
    cond_helper_node = (nodes_df["connectable_type"] == "") & (
        ~nodes_df.index.isin(switches_df["from_node"].to_list() + switches_df["to_node"].to_list())
    )
    nodes_df.loc[cond_helper_node, "helper_node"] = True
    nodes_df.fillna({"foreign_id": ""}, inplace=True)
    cond = nodes_df["foreign_id"] == ""
    nodes_df.loc[cond, "foreign_id"] = nodes_df.loc[cond, "grid_model_id"]
    nodes_df["helper_node"] = nodes_df["helper_node"].astype("boolean").fillna(False).astype(bool)
    nodes_df["in_service"] = nodes_df["in_service"].astype("boolean").fillna(True).astype(bool)
    nodes_df.drop(columns=["bus_breaker_bus_id"], inplace=True)

    return NodeSchema.validate(nodes_df)


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
    node_assets_df = node_assets_df[["grid_model_id", "foreign_id", "node", "asset_type"]]
    node_assets_df["in_service"] = node_assets_df["grid_model_id"].map(asset_in_service).fillna(True).astype(bool)
    return node_assets_df


def get_station(network: Network, bus_id: str, station_info: SubstationInformation) -> Station:
    """Get the station from a pypowsybl network.

    Parameters
    ----------
    network : Network
        The powsybl network.
    bus_id : str
        bus id of the station.
        (the substation grid_model_id)
    station_info : SubstationInformation
        The substation information.

    Returns
    -------
    station : Station
        The station as a AssetTopology.
    """
    substation_id = station_info.name
    station_components = _build_station_components(network, substation_id, station_info)

    station = Station(
        grid_model_id=bus_id,
        name=substation_id,
        region=station_info.region,
        voltage_level=int(station_info.nominal_v),
        busbars=station_components["busbars"],
        couplers=station_components["couplers"],
        assets=station_components["assets"],
        asset_switching_table=station_components["asset_switching_table"],
        asset_connectivity=station_components["asset_connectivity"],
        busbar_switching_table=station_components["busbar_switching_table"],
        busbar_connectivity=station_components["busbar_connectivity"],
        model_log=station_components["model_log"],
    )
    return station


def get_station_with_busbar_connection_fixes(
    network: Network, bus_id: str, station_info: SubstationInformation
) -> tuple[Station, dict[str, bool]]:
    """Get a station and the selector-switch fixes implied by its normalized switching table.

    Parameters
    ----------
    network : Network
        The powsybl network.
    bus_id : str
        Bus id of the station.
    station_info : SubstationInformation
        The substation information.

    Returns
    -------
    tuple[Station, dict[str, bool]]
        The station together with desired open states for selector switches.
    """
    station_components = _build_station_components(network, station_info.name, station_info)
    station = Station(
        grid_model_id=bus_id,
        name=station_info.name,
        region=station_info.region,
        voltage_level=int(station_info.nominal_v),
        busbars=station_components["busbars"],
        couplers=station_components["couplers"],
        assets=station_components["assets"],
        asset_switching_table=station_components["asset_switching_table"],
        asset_connectivity=station_components["asset_connectivity"],
        busbar_switching_table=station_components["busbar_switching_table"],
        busbar_connectivity=station_components["busbar_connectivity"],
        model_log=station_components["model_log"],
    )
    return station, station_components["switch_updates"]


def get_station_list(network: Network, relevant_voltage_level_with_region: pd.DataFrame) -> list[Station]:
    """Get the station list from a pypowsybl network.

    Note: include only wanted voltage levels and regions.

    Parameters
    ----------
    network : Network
        The powsybl network.
    relevant_voltage_level_with_region : pd.DataFrame
        DataFrame with the voltage level and region information.

    Returns
    -------
    station_list : list[Station]
        The station list as a AssetTopology.
    """
    station_list = []
    for bus_id, row in relevant_voltage_level_with_region.iterrows():
        station_info = SubstationInformation(
            name=row["name"],
            region=row["region"],
            nominal_v=row["nominal_v"],
            voltage_level_id=row["voltage_level_id"],
        )
        try:
            station = get_station(network=network, bus_id=bus_id, station_info=station_info)
            station_list.append(station)
        except ValidationError as e:
            logger.warning(
                f"ValidationError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list."
            )
        except KeyError as e:
            logger.warning(
                f"KeyError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list. "
                "Likely a maintenance busbar present - Currently working."
            )
        except ValueError as e:
            logger.warning(
                f"ValueError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list."
            )

    return station_list


def get_station_list_with_busbar_connection_fixes(
    network: Network, relevant_voltage_level_with_region: pd.DataFrame
) -> tuple[list[Station], dict[str, bool]]:
    """Get stations and aggregated selector-switch fixes from a pypowsybl network.

    Parameters
    ----------
    network : Network
        The powsybl network.
    relevant_voltage_level_with_region : pd.DataFrame
        DataFrame with the voltage level and region information.

    Returns
    -------
    tuple[list[Station], dict[str, bool]]
        Stations and desired switch open states inferred from their normalized switching tables.
    """
    station_list = []
    switch_updates: dict[str, bool] = {}
    for bus_id, row in relevant_voltage_level_with_region.iterrows():
        station_info = SubstationInformation(
            name=row["name"],
            region=row["region"],
            nominal_v=row["nominal_v"],
            voltage_level_id=row["voltage_level_id"],
        )
        try:
            station, station_switch_updates = get_station_with_busbar_connection_fixes(
                network=network,
                bus_id=bus_id,
                station_info=station_info,
            )
            station_list.append(station)
            switch_updates.update(station_switch_updates)
        except ValidationError as e:
            logger.warning(
                f"ValidationError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list."
            )
        except KeyError as e:
            logger.warning(
                f"KeyError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list. "
                "Likely a maintenance busbar present - Currently working."
            )
        except ValueError as e:
            logger.warning(
                f"ValueError while getting station: {station_info} with error: {e}. "
                "Consider checking the Station or adding to ignore list."
            )

    return station_list, switch_updates


def get_busbar_connection_fixes(network: Network, relevant_voltage_level_with_region: pd.DataFrame) -> dict[str, bool]:
    """Get selector-switch fixes without requiring successful station model validation."""
    switch_updates: dict[str, bool] = {}
    for _, row in relevant_voltage_level_with_region.iterrows():
        station_info = SubstationInformation(
            name=row["name"],
            region=row["region"],
            nominal_v=row["nominal_v"],
            voltage_level_id=row["voltage_level_id"],
        )
        try:
            station_components = _build_station_components(network, station_info.name, station_info)
            switch_updates.update(station_components["switch_updates"])
        except (ValidationError, KeyError, ValueError, SchemaError):
            logger.warning(
                f"Could not derive selector switch fixes for station {station_info}. Continuing without station-level fixes."
            )
    return switch_updates


def get_stations_with_double_connections(network: Network, relevant_voltage_level_with_region: pd.DataFrame) -> set[str]:
    """Identify stations whose assets are connected to multiple busbars."""
    stations_with_double_connections: set[str] = set()
    for _, row in relevant_voltage_level_with_region.iterrows():
        station_info = SubstationInformation(
            name=row["name"],
            region=row["region"],
            nominal_v=row["nominal_v"],
            voltage_level_id=row["voltage_level_id"],
        )
        try:
            station_components = _build_station_components(network, station_info.name, station_info)
            if station_components["had_double_connections"]:
                stations_with_double_connections.add(station_info.name)
        except (ValidationError, KeyError, ValueError, SchemaError):
            logger.warning(
                f"Could not inspect double connections for station {station_info}. Continuing without station-level classification."
            )
    return stations_with_double_connections


def filter_relevant_voltage_levels_for_double_connections(
    relevant_voltage_level_with_region: pd.DataFrame,
    ignored_station_ids: set[str],
    handling: DoubleConnectionHandling,
) -> pd.DataFrame:
    """Filter relevant voltage levels according to the configured double-connection handling."""
    if handling != "ignore_station" or not ignored_station_ids:
        return relevant_voltage_level_with_region
    return relevant_voltage_level_with_region[~relevant_voltage_level_with_region["name"].isin(ignored_station_ids)]


def get_topology_with_busbar_connection_fixes(
    network: Network, network_masks: NetworkMasks, importer_parameters: CgmesImporterParameters
) -> tuple[Topology, dict[str, bool]]:
    """Get the topology model together with selector-switch fixes for double connections.

    Parameters
    ----------
    network : Network
        Pypowsybl network object.
    network_masks : NetworkMasks
        NetworkMasks object with the relevant voltage levels.
    importer_parameters : CgmesImporterParameters
        Import parameters.

    Returns
    -------
    tuple[Topology, dict[str, bool]]
        Topology and desired switch open states inferred during station creation.
    """
    relevant_voltage_level_with_region = get_relevant_voltage_levels(network=network, network_masks=network_masks)
    stations_with_double_connections = get_stations_with_double_connections(network, relevant_voltage_level_with_region)
    if importer_parameters.double_connection_handling == "raise" and stations_with_double_connections:
        raise ValueError("Double connections detected in stations: " + ", ".join(sorted(stations_with_double_connections)))
    relevant_voltage_level_with_region = filter_relevant_voltage_levels_for_double_connections(
        relevant_voltage_level_with_region,
        ignored_station_ids=stations_with_double_connections,
        handling=importer_parameters.double_connection_handling,
    )
    station_list, switch_updates = get_station_list_with_busbar_connection_fixes(
        network=network,
        relevant_voltage_level_with_region=relevant_voltage_level_with_region,
    )
    if importer_parameters.double_connection_handling != "choose_first_busbar":
        switch_updates = {}
    grid_model_file = str(importer_parameters.grid_model_file.name)
    topology_id = importer_parameters.grid_model_file.name
    timestamp = datetime.datetime.now()

    topology = Topology(
        topology_id=topology_id,
        grid_model_file=grid_model_file,
        stations=station_list,
        timestamp=timestamp,
    )
    return topology, switch_updates


def get_relevant_voltage_levels(network: Network, network_masks: NetworkMasks) -> pd.DataFrame:
    """Get all relevant voltage level from the network.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    network_masks: NetworkMasks
        NetworkMasks object with the relevant_subs mask

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


def get_topology(network: Network, network_masks: NetworkMasks, importer_parameters: CgmesImporterParameters) -> Topology:
    """Get the pydantic topology model from the network.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    network_masks: NetworkMasks
        NetworkMasks object with the relevant voltage levels.
    importer_parameters: UcteImporterParameters
        UCTE importer parameters

    Returns
    -------
    topology: Topology
        Topology object, including all relevant stations
    """
    topology, _ = get_topology_with_busbar_connection_fixes(network, network_masks, importer_parameters)
    return topology
