# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to translate the powsybl model to the asset topology model.

File: asset_topology.py
Author:  Benjamin Petrick
Created: 2024-09-18
"""

import datetime

import numpy as np
import pandas as pd
from beartype.typing import Optional, TypeVar, Union
from jaxtyping import Bool
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import change_dangling_to_tie, get_voltage_level_with_region
from toop_engine_interfaces.asset_topology import (
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
    MaterializedStation,
    RawStation,
    StationAssetConnection,
    SwitchableAsset,
    Topology,
    normalize_switchable_asset_payload,
)

SwitchableAssetType = TypeVar("SwitchableAssetType", bound=SwitchableAsset)


def get_all_element_names(network: Network, line_trafo_name_col: str = "elementName") -> pd.Series:
    """Get the names of all injections and branches in the network.

    For trafo and line -> elementName
    For the rest -> name

    Parameters
    ----------
    network: Network
        pypowsybl network object
    line_trafo_name_col: str
        Column name for the element names of lines and trafos

    Returns
    -------
    all_names: pd.Series
        Series with the names of all injections and branches in the network and their ids as index
    """
    line_names = network.get_lines(attributes=[line_trafo_name_col])[line_trafo_name_col]
    trafo_names = network.get_2_windings_transformers(attributes=[line_trafo_name_col])[line_trafo_name_col]
    trafo_3w_names = network.get_3_windings_transformers(attributes=["name"]).name
    shunt_compensator_names = network.get_shunt_compensators(attributes=["name"]).name
    generator_names = network.get_generators(attributes=["name"]).name
    load_names = network.get_loads(attributes=["name"]).name
    dangling_line_names = network.get_boundary_lines(attributes=["name"]).name
    tie_line_names = network.get_tie_lines(attributes=["name"]).name
    all_names = pd.concat(
        [
            line_names,
            trafo_names,
            trafo_3w_names,
            generator_names,
            load_names,
            dangling_line_names,
            tie_line_names,
            shunt_compensator_names,
        ]
    )
    return all_names


def get_asset_switching_table(station_buses: pd.DataFrame, station_elements: pd.DataFrame) -> np.ndarray:
    """Get the asset switching table, which holds the switching of each asset to each busbar.

    Parameters
    ----------
    station_buses: pd.DataFrame
        DataFrame with the station busbars
        Note: The DataFrame is expected be sorted by its "int_id".
    station_elements: pd.DataFrame
        DataFrame with the injections and branches at the station
        Note: The DataFrame is expected to have a column "bus_int_id" which holds the busbar id for each asset.

    Returns
    -------
    switching_matrix: np.ndarray
        Switching matrix with the shape (n_bus, n_asset) where n_bus is the number of busbars
        and n_asset is the number of assets.
    """
    assert station_buses["int_id"].is_monotonic_increasing, "station_buses not sorted"
    station_buses_ranged = station_buses.copy().reset_index(drop=True)

    n_bus = station_buses.shape[0]
    n_asset = station_elements.shape[0]
    switching_matrix = np.zeros((n_bus, n_asset), dtype=bool)

    for asset_idx, bus_id in enumerate(station_elements["bus_int_id"]):
        # if asset is not connected -> -1
        if bus_id != -1:
            bus_idx = station_buses_ranged[station_buses_ranged["int_id"] == bus_id].index[0]
            switching_matrix[bus_idx, asset_idx] = True

    return switching_matrix


def get_list_of_coupler_from_df(coupler_elements: pd.DataFrame) -> list[BusbarCoupler]:
    """Get the list of coupler elements from the DataFrame.

    Parameters
    ----------
    coupler_elements: pd.DataFrame
        DataFrame with the coupler elements
        Note: datatype of columns is expected to be the same as in the pydantic model.

    Returns
    -------
    coupler_list: list[BusbarCoupler]
        List of coupler elements.
    """
    coupler_dict = coupler_elements.to_dict(orient="records")
    coupler_list = [BusbarCoupler(**coupler) for coupler in coupler_dict]
    return coupler_list


def get_list_of_busbars_from_df(station_buses: pd.DataFrame) -> list[Busbar]:
    """Get the list of busbars from the DataFrame.

    Parameters
    ----------
    station_buses: pd.DataFrame
        DataFrame with the busbars
        Note: datatype of columns is expected to be the same as in the pydantic model.

    Returns
    -------
    busbar_list: list[Busbar]
        List of busbars.
    """
    busbar_dict = station_buses.to_dict(orient="records")
    busbar_list = [Busbar(**busbar) for busbar in busbar_dict]

    return busbar_list


def _dedupe_assets_by_id(assets: list[SwitchableAssetType]) -> list[SwitchableAssetType]:
    """Deduplicate topology-owned assets by grid model id while preserving first-seen order.

    Parameters
    ----------
    assets : list[SwitchableAssetType]
        Topology-owned assets collected across stations.

    Returns
    -------
    list[SwitchableAssetType]
        One deep-copied asset per grid model id, ordered by first appearance.
    """
    deduped_assets: dict[str, SwitchableAssetType] = {}
    for asset in assets:
        deduped_assets.setdefault(asset.grid_model_id, asset.model_copy(deep=True))
    return list(deduped_assets.values())


def _get_station_asset_inputs_from_topology(
    station_topology_elements: pd.DataFrame,
    station_buses: pd.DataFrame,
    dangling_lines: pd.DataFrame,
    element_names: pd.Series,
) -> tuple[pd.DataFrame, list[str | None], np.ndarray, np.ndarray]:
    """Build the shared station asset inputs used by the split extraction helpers.

    Parameters
    ----------
    station_topology_elements : pd.DataFrame
        Raw powsybl topology elements for one station.
    station_buses : pd.DataFrame
        Formatted station busbars with local integer ids.
    dangling_lines : pd.DataFrame
        Network dangling-line metadata used to normalize tie lines.
    element_names : pd.Series
        Asset names keyed by grid model id.

    Returns
    -------
    station_elements : pd.DataFrame
        Normalized station asset rows used for asset materialization.
    asset_terminals : list[str | None]
        Terminal labels aligned with `station_elements`.
    switching_matrix : np.ndarray
        Station-local switching matrix aligned with `station_elements`.
    asset_connectivity : np.ndarray
        Default connectivity matrix aligned with `station_elements`.
    """
    station_elements, switching_matrix = get_asset_info_from_topology(
        station_topology_elements,
        station_buses,
        dangling_lines,
        element_names,
    )
    asset_connectivity = np.ones_like(switching_matrix, dtype=bool)
    asset_terminals = (
        station_elements["branch_end"].tolist()
        if "branch_end" in station_elements.columns
        else [None] * len(station_elements)
    )
    return station_elements, asset_terminals, switching_matrix, asset_connectivity


def _get_branch_station_assets_from_df(
    station_elements: pd.DataFrame,
    asset_terminals: list[str | None],
    switching_matrix: np.ndarray,
    asset_connectivity: np.ndarray,
) -> tuple[list[BranchAsset], list[str | None], np.ndarray, np.ndarray]:
    """Return branch station assets and aligned arrays from prepared station inputs.

    Parameters
    ----------
    station_elements : pd.DataFrame
        Normalized station asset rows.
    asset_terminals : list[str | None]
        Terminal labels aligned with `station_elements`.
    switching_matrix : np.ndarray
        Station-local switching matrix aligned with `station_elements`.
    asset_connectivity : np.ndarray
        Connectivity matrix aligned with `station_elements`.

    Returns
    -------
    branch_assets : list[BranchAsset]
        Materialized branch assets for the station.
    branch_terminals : list[str | None]
        Terminal labels aligned with `branch_assets`.
    branch_switching_table : np.ndarray
        Branch-only switching table.
    branch_connectivity : np.ndarray
        Branch-only connectivity table.
    """
    normalized_assets = [
        normalize_switchable_asset_payload(switchable_asset)
        for switchable_asset in station_elements.to_dict(orient="records")
    ]
    branch_mask = np.asarray([asset.is_branch() is not False for asset in normalized_assets], dtype=bool)
    branch_assets = [
        asset if isinstance(asset, BranchAsset) else BranchAsset.model_validate(asset.model_dump())
        for asset, is_branch in zip(normalized_assets, branch_mask, strict=True)
        if is_branch
    ]
    branch_terminals = [terminal for terminal, is_branch in zip(asset_terminals, branch_mask, strict=True) if is_branch]
    branch_switching_table = switching_matrix[:, branch_mask]
    branch_connectivity = asset_connectivity[:, branch_mask]

    return branch_assets, branch_terminals, branch_switching_table, branch_connectivity


def _get_injection_station_assets_from_df(
    station_elements: pd.DataFrame,
    asset_terminals: list[str | None],
    switching_matrix: np.ndarray,
    asset_connectivity: np.ndarray,
) -> tuple[list[InjectionAsset], list[str | None], np.ndarray, np.ndarray]:
    """Return injection station assets and aligned arrays from prepared station inputs.

    Parameters
    ----------
    station_elements : pd.DataFrame
        Normalized station asset rows.
    asset_terminals : list[str | None]
        Terminal labels aligned with `station_elements`.
    switching_matrix : np.ndarray
        Station-local switching matrix aligned with `station_elements`.
    asset_connectivity : np.ndarray
        Connectivity matrix aligned with `station_elements`.

    Returns
    -------
    injection_assets : list[InjectionAsset]
        Materialized injection assets for the station.
    injection_terminals : list[str | None]
        Terminal labels aligned with `injection_assets`.
    injection_switching_table : np.ndarray
        Injection-only switching table.
    injection_connectivity : np.ndarray
        Injection-only connectivity table.
    """
    normalized_assets = [
        normalize_switchable_asset_payload(switchable_asset)
        for switchable_asset in station_elements.to_dict(orient="records")
    ]
    injection_mask = np.asarray([asset.is_branch() is False for asset in normalized_assets], dtype=bool)
    injection_assets = [
        asset if isinstance(asset, InjectionAsset) else InjectionAsset.model_validate(asset.model_dump())
        for asset, is_injection in zip(normalized_assets, injection_mask, strict=True)
        if is_injection
    ]
    injection_terminals = [
        terminal for terminal, is_injection in zip(asset_terminals, injection_mask, strict=True) if is_injection
    ]
    injection_switching_table = switching_matrix[:, injection_mask]
    injection_connectivity = asset_connectivity[:, injection_mask]

    return injection_assets, injection_terminals, injection_switching_table, injection_connectivity


def _get_single_topology_kind(buses_with_substation_and_voltage: pd.DataFrame) -> str:
    """Return the shared topology kind for all relevant stations or raise on mixed input.

    Parameters
    ----------
    buses_with_substation_and_voltage : pd.DataFrame
        Relevant station rows including the `topology_kind` column.

    Returns
    -------
    str
        The single topology kind shared by all relevant stations.

    Raises
    ------
    ValueError
        If the relevant stations contain a mix of topology kinds.
    """
    topology_kinds = buses_with_substation_and_voltage["topology_kind"].unique()
    if len(topology_kinds) != 1:
        raise ValueError(
            "Relevant stations must be either of kind NODE_BREAKER or BUS_BREAKER, a mix of both is not permitted"
        )
    return str(topology_kinds[0])


def get_bus_info_from_topology(station_buses: pd.DataFrame, bus_id: str) -> pd.DataFrame:
    """Get the info for all busbars that are part of the bus.

    Parameters
    ----------
    station_buses: pd.DataFrame
        Dataframe with all busbars of the station and which bus they are connected to.
        Comes from BusBreakerTopology.buses
    bus_id: str
        Bus id for which the buses should be retrieved.

    Returns
    -------
    station_buses: pd.DataFrame
        DataFrame with the busbars of the specified bus.
        Note: The DataFrame columns are the same as in the pydantic model.
    """
    station_buses = station_buses[station_buses["bus_id"] == bus_id].copy()
    # for UCTE model: if not in service, asset will not appear
    station_buses["in_service"] = True

    # get bus df
    station_buses = (
        station_buses.sort_index().reset_index().reset_index().rename(columns={"index": "int_id", "id": "grid_model_id"})
    )
    station_buses = station_buses[["grid_model_id", "name", "int_id", "in_service"]]

    return station_buses


def get_coupler_info_from_topology(
    station_switches: pd.DataFrame, switches_df: pd.DataFrame, station_buses: pd.DataFrame
) -> pd.DataFrame:
    """Get the coupler elements that are connected to the busbars of the station.

    Parameters
    ----------
    station_switches: pd.DataFrame
        Dataframe of all switches at the station and which busbars they connect
        Comes from BusBreakerTopology.switches
    switches_df: pd.DataFrame
        DataFrame of all switches in the network
    station_buses: pd.DataFrame
        Formatted dataframe of all busbars at the station.

    Returns
    -------
    coupler_elements: pd.DataFrame
        DataFrame with the coupler elements of the station.
        Note: The DataFrame columns are the same as in the pydantic model.
    """
    # get the coupler elements information
    coupler_elements = station_switches.merge(switches_df, how="left", left_index=True, right_index=True)
    # for UCTE model: if not in service, asset will not appear
    coupler_elements["in_service"] = True
    coupler_elements.reset_index(inplace=True)
    # rename the columns to match the pydantic model
    coupler_elements.rename(
        columns={
            "kind": "coupler_type",
            "bus1_id": "busbar_from_id",
            "bus2_id": "busbar_to_id",
            "id": "grid_model_id",
        },
        inplace=True,
    )
    # get the busbar ids
    merged_df = pd.merge(
        coupler_elements,
        station_buses,
        left_on="busbar_from_id",
        right_on="grid_model_id",
        how="left",
    )
    coupler_elements["busbar_from_id"] = merged_df["int_id"]
    merged_df = pd.merge(
        coupler_elements,
        station_buses,
        left_on="busbar_to_id",
        right_on="grid_model_id",
        how="left",
    )
    coupler_elements["busbar_to_id"] = merged_df["int_id"]

    return coupler_elements.dropna()


def get_name_of_station_elements(station_elements: pd.DataFrame, element_names: pd.Series) -> pd.DataFrame:
    """Attach the name of the elements to the station elements.

    Parameters
    ----------
    station_elements: pd.DataFrame
        DataFrame with the station elements
        Comes from BusBreakerTopology.elements
    element_names: pd.Series
        Series with the names of all injections and branches in the network and their ids as index

    Returns
    -------
    station_elements: pd.DataFrame
        DataFrame with the names of the elements attached
    """
    station_elements["name"] = element_names
    return station_elements


def get_asset_info_from_topology(
    station_elements: pd.DataFrame, station_buses: pd.DataFrame, dangling_lines: pd.DataFrame, element_names: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """Get the asset information of all elements at the station.

    Parameters
    ----------
    station_elements: pd.DataFrame
        DataFrame with the station elements
        Comes from BusBreakerTopology.elements
    station_buses: pd.DataFrame
        DataFrame with the busbars of the station
    dangling_lines: pd.DataFrame^
        DataFrame of all dangling lines in the network with column "tie_line_id"
    element_names: pd.Series
        Series with the names of all injections and branches in the network and their ids as index

    Returns
    -------
    station_elements: pd.DataFrame
        DataFrame with the asset information
    switching_matrix: np.ndarray
        Switching matrix with the shape (n_bus, n_asset) where n_bus is the number of busbars
        and n_asset is the number of assets. True, where the asset is connected to the busbar.
    """
    # check for TIE_LINE
    station_elements = change_dangling_to_tie(dangling_lines, station_elements)
    # get the name for the branches
    station_elements = get_name_of_station_elements(station_elements, element_names)
    # for UCTE model: if not in service, asset will not appear
    station_elements["in_service"] = True
    station_elements.reset_index(inplace=True)
    station_elements.rename(columns={"id": "grid_model_id"}, inplace=True)
    # get the busbar ids for switching matrix
    merged_df = pd.merge(
        station_elements,
        station_buses,
        left_on="bus_id",
        right_on="grid_model_id",
        how="left",
    )
    station_elements["bus_int_id"] = merged_df["int_id"].fillna(-1).astype(int)

    station_elements = station_elements[station_elements["type"] != "BUSBAR_SECTION"]
    # TODO: change selection to bus_id -> keep disconnected assets
    # currently disconnected assets are not shown in the topology
    station_elements = station_elements[station_elements["bus_id"].isin(station_buses["grid_model_id"])]
    switching_matrix = get_asset_switching_table(station_buses=station_buses, station_elements=station_elements)
    # get columns for pydantic model
    station_elements = station_elements.rename(columns={"type": "asset_type"})
    station_elements = station_elements[["grid_model_id", "asset_type", "name", "in_service"]].reset_index(drop=True)
    return station_elements, switching_matrix


def get_relevant_network_data(
    network: Network, relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Get the relevant data from the network that is required for all stations.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]]
        The relevant stations to be included in the resulting topology. Either as a boolean mask over all buses in
        network.get_buses or as a list of bus ids in network.get_buses()

    Returns
    -------
    buses_with_substation_and_voltage: pd.DataFrame
        DataFrame with the relevant buses, substation id and voltage level
    switches: pd.DataFrame
        DataFrame with all the switches in the network. Includes the column "name"
    dangling_lines: pd.DataFrame
        DataFrame with all the dangling lines in the network. Includes the column "tie_line_id"
    element_names: pd.Series
        Series with the names of all injections and branches in the network and their ids as index
    """
    relevant_buses = network.get_buses(attributes=["voltage_level_id"]).loc[relevant_stations]
    voltage_level_df = get_voltage_level_with_region(network, attributes=["substation_id", "nominal_v", "topology_kind"])
    buses_with_substation_and_voltage = relevant_buses.merge(voltage_level_df, left_on="voltage_level_id", right_index=True)
    topology_kind = _get_single_topology_kind(buses_with_substation_and_voltage)
    if topology_kind == "BUS_BREAKER":
        if "elementName" in network.get_lines(all_attributes=True).columns:
            # For UCTE models, the name is stored in elementName
            element_name_col = "elementName"
        else:
            # All other grid files use normally "name"
            element_name_col = "name"
    elif topology_kind == "NODE_BREAKER":
        element_name_col = "name"
    else:
        raise ValueError(
            "Relevant stations must be either of kind NODE_BREAKER or BUS_BREAKER, a mix of both is not permitted"
        )
    buses_with_substation_and_voltage.drop(columns=["topology_kind"], inplace=True)

    element_names = get_all_element_names(network, line_trafo_name_col=element_name_col)
    switches = network.get_switches(attributes=["name"])
    dangling_lines = network.get_boundary_lines(attributes=["tie_line_id"])
    return buses_with_substation_and_voltage, switches, dangling_lines, element_names


def get_relevant_stations(
    network: Network, relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]]
) -> list[RawStation]:
    """Get all relevant stations from the network.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]]
        The relevant stations to be included in the resulting topology. Either as a boolean mask over all buses in
        network.get_buses or as a list of bus ids in network.get_buses()

    Returns
    -------
    station: list[RawStation]
        List of all lean topology stations of the relevant buses in the network
    """
    # Load relevant data once
    buses_with_substation_and_voltage, switches, dangling_lines, element_names = get_relevant_network_data(
        network=network,
        relevant_stations=relevant_stations,
    )

    # Calculate the pydantic station for each relevant bus
    raw_stations, _, _ = get_raw_stations_and_assets(
        network, buses_with_substation_and_voltage, switches, dangling_lines, element_names
    )
    return raw_stations


def get_raw_stations_and_assets(
    network: Network,
    buses_with_substation_and_voltage: pd.DataFrame,
    switches: pd.DataFrame,
    dangling_lines: pd.DataFrame,
    element_names: pd.Series,
) -> tuple[list[RawStation], list[BranchAsset], list[InjectionAsset]]:
    """Build raw topology stations and topology-owned assets from the relevant buses.

    Parameters
    ----------
    network : Network
        pypowsybl network object
    buses_with_substation_and_voltage : pd.DataFrame
        DataFrame with the relevant buses, substation id and voltage level
    switches : pd.DataFrame
        DataFrame with all the switches in the network. Includes the column "name"
    dangling_lines : pd.DataFrame
        DataFrame with all the dangling lines in the network. Includes the column "tie_line_id"
    element_names : pd.Series
        Series with the names of all injections and branches in the network and their ids as index

    Returns
    -------
    station_list : list[RawStation]
        List of all lean topology stations of the relevant buses in the network
    branch_assets : list[BranchAsset]
        Deduplicated topology-owned branch assets referenced by the stations
    injection_assets : list[InjectionAsset]
        Deduplicated topology-owned injection assets referenced by the stations
    """
    station_list: list[RawStation] = []
    all_branch_assets: list[BranchAsset] = []
    all_injection_assets: list[InjectionAsset] = []
    for bus_id, bus_info in buses_with_substation_and_voltage.iterrows():
        station_topology = network.get_bus_breaker_topology(bus_info.voltage_level_id)
        station_buses = get_bus_info_from_topology(station_topology.buses, bus_id)
        coupler_elements = get_coupler_info_from_topology(station_topology.switches, switches, station_buses)
        station_elements, asset_terminals, switching_matrix, asset_connectivity = _get_station_asset_inputs_from_topology(
            station_topology.elements,
            station_buses,
            dangling_lines,
            element_names,
        )
        (
            branch_assets,
            branch_terminals,
            branch_switching_table,
            branch_connectivity,
        ) = _get_branch_station_assets_from_df(
            station_elements,
            asset_terminals,
            switching_matrix,
            asset_connectivity,
        )
        (
            injection_assets,
            injection_terminals,
            injection_switching_table,
            injection_connectivity,
        ) = _get_injection_station_assets_from_df(
            station_elements,
            asset_terminals,
            switching_matrix,
            asset_connectivity,
        )
        all_branch_assets.extend(branch_assets)
        all_injection_assets.extend(injection_assets)

        station = RawStation(
            grid_model_id=bus_id,
            name=bus_info.substation_id,
            region=bus_info.voltage_level_id[0:2],
            voltage_level=bus_info.nominal_v,
            busbars=get_list_of_busbars_from_df(station_buses),
            couplers=get_list_of_coupler_from_df(coupler_elements),
            branch_connections=[
                StationAssetConnection(asset_id=asset.grid_model_id, terminal=asset_terminal, asset_bay_id=None)
                for asset, asset_terminal in zip(branch_assets, branch_terminals, strict=True)
            ],
            injection_connections=[
                StationAssetConnection(asset_id=asset.grid_model_id, terminal=asset_terminal, asset_bay_id=None)
                for asset, asset_terminal in zip(injection_assets, injection_terminals, strict=True)
            ],
            branch_switching_table=branch_switching_table,
            injection_switching_table=injection_switching_table,
            branch_connectivity=branch_connectivity,
            injection_connectivity=injection_connectivity,
        )
        station_list.append(station)
    return (
        station_list,
        _dedupe_assets_by_id(all_branch_assets),
        _dedupe_assets_by_id(all_injection_assets),
    )


def get_topology(
    network: Network,
    relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]],
    topology_id: str,
    grid_model_file: Optional[str] = None,
) -> Topology:
    """Get the pydantic topology model from the network.

    Parameters
    ----------
    network: Network
        pypowsybl network object
    relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]]
        The relevant stations to be included in the resulting topology. Either as a boolean mask over all buses in
        network.get_buses or as a list of bus ids in network.get_buses()
    topology_id: str
        Id of the topology to set in the asset topology
    grid_model_file: Optional[str]
        Path to the grid model file to set in the asset topology

    Returns
    -------
    topology: Topology
        Topology object, including all relevant stations
    """
    buses_with_substation_and_voltage, switches, dangling_lines, element_names = get_relevant_network_data(
        network=network,
        relevant_stations=relevant_stations,
    )
    raw_stations, branch_assets, injection_assets = get_raw_stations_and_assets(
        network, buses_with_substation_and_voltage, switches, dangling_lines, element_names
    )
    timestamp = datetime.datetime.now()

    return Topology(
        topology_id=topology_id,
        grid_model_file=grid_model_file,
        raw_stations=raw_stations,
        branch_assets=branch_assets,
        injection_assets=injection_assets,
        asset_bays=[],
        timestamp=timestamp,
    )


def get_raw_stations_and_assets_bus_breaker(
    net: Network,
) -> tuple[list[RawStation], list[BranchAsset], list[InjectionAsset]]:
    """Convert a bus-breaker topology grid to raw topology stations and topology-owned assets.

    This is mainly used for fixture and test-grid extraction.

    Parameters
    ----------
    net : Network
        The bus/breaker powsybl network to convert

    Returns
    -------
    stations : list[RawStation]
        List of all lean topology stations in the network
    branch_assets : list[BranchAsset]
        Deduplicated topology-owned branch assets referenced by the raw stations
    injection_assets : list[InjectionAsset]
        Deduplicated topology-owned injection assets referenced by the raw stations
    """
    all_switches = net.get_switches(all_attributes=True)
    all_branches = net.get_branches(all_attributes=True)
    all_injections = net.get_injections(all_attributes=True)
    all_breaker_buses = net.get_bus_breaker_view_buses(all_attributes=True)

    stations: list[RawStation] = []
    all_branch_assets: list[BranchAsset] = []
    all_injection_assets: list[InjectionAsset] = []
    for bus_id, bus_row in net.get_buses().iterrows():
        local_buses = all_breaker_buses[all_breaker_buses["bus_id"] == bus_id]
        local_switches = all_switches[
            (
                all_switches["bus_breaker_bus1_id"].isin(local_buses.index)
                | all_switches["bus_breaker_bus2_id"].isin(local_buses.index)
            )
        ]
        from_branches = all_branches[
            (all_branches["bus_breaker_bus1_id"].isin(local_buses.index) & all_branches["connected1"])
        ]
        to_branches = all_branches[
            (all_branches["bus_breaker_bus2_id"].isin(local_buses.index) & all_branches["connected2"])
        ]
        injections = all_injections[
            (all_injections["bus_breaker_bus_id"].isin(local_buses.index) & all_injections["connected"])
        ]
        busbar_mapper = {grid_model_id: index for index, grid_model_id in enumerate(local_buses.index)}

        busbars = [
            Busbar(grid_model_id=grid_model_id, int_id=busbar_mapper[grid_model_id]) for grid_model_id in local_buses.index
        ]
        couplers = [
            BusbarCoupler(
                grid_model_id=grid_model_id,
                coupler_type=switch.kind,
                busbar_from_id=busbar_mapper[switch.bus_breaker_bus1_id],
                busbar_to_id=busbar_mapper[switch.bus_breaker_bus2_id],
                open=switch.open,
            )
            for grid_model_id, switch in local_switches.iterrows()
        ]
        from_branch_assets = [
            BranchAsset(grid_model_id=grid_model_id, asset_type=branch.type)
            for grid_model_id, branch in from_branches.iterrows()
        ]
        to_branch_assets = [
            BranchAsset(grid_model_id=grid_model_id, asset_type=branch.type)
            for grid_model_id, branch in to_branches.iterrows()
        ]
        injection_assets = [
            InjectionAsset(grid_model_id=grid_model_id, asset_type=injection.type)
            for grid_model_id, injection in injections.iterrows()
        ]

        branch_assets = from_branch_assets + to_branch_assets
        branch_terminals = ["from"] * len(from_branch_assets) + ["to"] * len(to_branch_assets)
        branch_bus_index = [busbar_mapper[branch.bus_breaker_bus1_id] for branch in from_branches.itertuples()] + [
            busbar_mapper[branch.bus_breaker_bus2_id] for branch in to_branches.itertuples()
        ]
        injection_bus_index = [busbar_mapper[injection.bus_breaker_bus_id] for injection in injections.itertuples()]

        branch_switching_table = np.zeros((len(busbars), len(branch_assets)), dtype=bool)
        for asset_index, idx in enumerate(branch_bus_index):
            branch_switching_table[idx, asset_index] = True

        injection_switching_table = np.zeros((len(busbars), len(injection_assets)), dtype=bool)
        for asset_index, idx in enumerate(injection_bus_index):
            injection_switching_table[idx, asset_index] = True

        all_branch_assets.extend(branch_assets)
        all_injection_assets.extend(injection_assets)

        station = RawStation(
            grid_model_id=bus_id,
            name=bus_row.name,
            busbars=busbars,
            couplers=couplers,
            branch_connections=[
                StationAssetConnection(asset_id=asset.grid_model_id, terminal=asset_terminal, asset_bay_id=None)
                for asset, asset_terminal in zip(branch_assets, branch_terminals, strict=True)
            ],
            injection_connections=[
                StationAssetConnection(asset_id=asset.grid_model_id, terminal=None, asset_bay_id=None)
                for asset in injection_assets
            ],
            branch_switching_table=branch_switching_table,
            injection_switching_table=injection_switching_table,
        )
        stations.append(station)
    return stations, _dedupe_assets_by_id(all_branch_assets), _dedupe_assets_by_id(all_injection_assets)


# TODO: refactor due to C901
def assert_station_in_network(  # noqa: C901
    net: Network,
    station: MaterializedStation,
    couplers_strict: bool = True,
    assets_strict: bool = True,
    busbars_strict: bool = True,
) -> None:
    """Check if an asset topology station and all assets/busbars are actually in the station in the grid

    This only checks subsets, i.e. if all asset in the asset topology are also in the grid. If there are more assets in the
    grid, this will not raise by default. You can enable strict equality checking by setting ..._strict to True.

    Parameters
    ----------
    net: Network
        The powsybl network to check the station in
    station: Station
        The asset topology station to check
    couplers_strict: bool
        If you opt out of strict coupler checking, it will only be checked if all couplers in the station are present in the
        grid, not vice versa.
    assets_strict: bool
        If you opt out of strict asset checking, it will only be checked if all assets in the station are present in the
        grid, not vice versa.
    busbars_strict: bool
        If you opt out of strict busbar checking, it will only be checked if all busbars in the station are present in the
        grid, not vice versa.

    Raises
    ------
    ValueError
        If the station or any of the assets/busbars are not in the network
    """
    buses_df = net.get_buses(attributes=["voltage_level_id"])
    if station.grid_model_id not in buses_df.index:
        raise ValueError(f"Station {station.grid_model_id} not found in the network")

    bus_breaker_topo = net.get_bus_breaker_topology(buses_df.loc[station.grid_model_id]["voltage_level_id"])
    station_connections = [*station.branch_connections, *station.injection_connections]

    for asset_connection in station_connections:
        asset = asset_connection.asset
        if asset.grid_model_id not in bus_breaker_topo.elements.index:
            raise ValueError(f"Asset {asset.grid_model_id} not found in the station elements: {bus_breaker_topo.elements}")
    if assets_strict and len(bus_breaker_topo.elements) != len(station_connections):
        raise ValueError(f"Asset count mismatch: {len(bus_breaker_topo.elements)} != {len(station_connections)}")

    for busbar in station.busbars:
        if busbar.grid_model_id not in bus_breaker_topo.buses.index:
            raise ValueError(f"Busbar {busbar.grid_model_id} not found in the station buses: {bus_breaker_topo.buses}")
    if busbars_strict and len(bus_breaker_topo.buses) != len(station.busbars):
        raise ValueError(f"Busbar count mismatch: {len(bus_breaker_topo.buses)} != {len(station.busbars)}")

    for coupler in station.couplers:
        if coupler.grid_model_id not in bus_breaker_topo.switches.index:
            raise ValueError(
                f"Coupler {coupler.grid_model_id} not found in the station switches: {bus_breaker_topo.switches}"
            )
    if couplers_strict and len(bus_breaker_topo.switches) != len(station.couplers):
        raise ValueError(f"Coupler count mismatch: {len(bus_breaker_topo.switches)} != {len(station.couplers)}")
