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

from string import ascii_lowercase

import numpy as np
import pandas as pd
import structlog
from beartype.typing import Any, Optional, TypeVar, Union, get_args
from jaxtyping import Bool
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import get_voltage_level_with_region
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    CouplerBay,
    InjectionAsset,
    SwitchableAsset,
)
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeBusGroup,
)
from toop_engine_interfaces.asset_topology.topology_conversion import (
    RuntimeSwitchingState,
    materialize_runtime_bus_group_from_runtime_state,
)

SwitchableAssetType = TypeVar("SwitchableAssetType", bound=SwitchableAsset)
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
    """Return a deterministic alphabetic suffix for structural station ids."""
    suffix = ""
    remaining_index = group_index
    while True:
        remaining_index, char_index = divmod(remaining_index, len(ascii_lowercase))
        suffix = ascii_lowercase[char_index] + suffix
        if remaining_index == 0:
            return suffix
        remaining_index -= 1


def _build_structural_station_id(voltage_level_id: str, group_index: int) -> str:
    """Build a deterministic station id for one structural bus group within a voltage level."""
    return f"{voltage_level_id}_{_structural_station_suffix(group_index)}"


def _get_bus_breaker_structural_bus_groups(
    station_topology_buses: pd.DataFrame,
    station_topology_switches: pd.DataFrame,
) -> list[set[str]]:
    """Group bus-breaker buses by switch topology, independent of runtime open state.

    Parameters
    ----------
    station_topology_buses : pd.DataFrame
        Bus-breaker topology buses for one voltage level.
    station_topology_switches : pd.DataFrame
        Bus-breaker switches connecting the buses.

    Returns
    -------
    list[set[str]]
        Deterministic structural groups of bus-breaker bus ids.
    """
    bus_ids = station_topology_buses.index.tolist()
    remaining_bus_ids = set(bus_ids)
    if not remaining_bus_ids:
        return []

    adjacency: dict[str, set[str]] = {bus_id: set() for bus_id in bus_ids}
    for switch in station_topology_switches.itertuples():
        bus1_id = switch.bus1_id
        bus2_id = switch.bus2_id
        if bus1_id not in adjacency or bus2_id not in adjacency:
            continue
        adjacency[bus1_id].add(bus2_id)
        adjacency[bus2_id].add(bus1_id)

    structural_groups: list[set[str]] = []
    while remaining_bus_ids:
        start_bus_id = min(remaining_bus_ids)
        structural_group = {start_bus_id}
        frontier = [start_bus_id]
        while frontier:
            current_bus_id = frontier.pop()
            for connected_bus_id in adjacency[current_bus_id]:
                if connected_bus_id in structural_group:
                    continue
                structural_group.add(connected_bus_id)
                frontier.append(connected_bus_id)
        structural_groups.append(structural_group)
        remaining_bus_ids -= structural_group

    structural_groups.sort(key=sorted)
    return structural_groups


def _get_bus_breaker_station_bus_info_from_group(
    station_buses: pd.DataFrame,
    selected_busbar_ids: set[str],
) -> pd.DataFrame:
    """Return formatted station busbars for one structural bus-breaker group.

    Parameters
    ----------
    station_buses : pd.DataFrame
        Raw bus-breaker topology bus rows for one voltage level.
    selected_busbar_ids : set[str]
        Structural bus ids belonging to the current station group.

    Returns
    -------
    pd.DataFrame
        Formatted busbar rows ready for conversion into canonical station busbars.
    """
    sorted_busbar_ids = sorted(selected_busbar_ids)
    if not sorted_busbar_ids:
        return pd.DataFrame(
            columns=["grid_model_id", "name", "int_id", "in_service", "bus_breaker_bus_id", "bus_branch_bus_id"]
        )

    bus_names: list[str | None] = []
    bus_branch_bus_ids: list[str] = []
    for _busbar_int_id, busbar_id in enumerate(sorted_busbar_ids):
        bus_entry = station_buses.loc[busbar_id]
        bus_names.append(bus_entry["name"])
        bus_branch_bus_ids.append(str(bus_entry["bus_id"]))

    group_station_bus_info = pd.DataFrame(
        {
            "grid_model_id": sorted_busbar_ids,
            "name": bus_names,
            "int_id": np.arange(len(sorted_busbar_ids), dtype=int),
            "in_service": True,
            "bus_breaker_bus_id": sorted_busbar_ids,
            "bus_branch_bus_id": bus_branch_bus_ids,
        }
    )
    return group_station_bus_info


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
    n_bus = station_buses.shape[0]
    n_asset = station_elements.shape[0]
    switching_matrix = np.zeros((n_bus, n_asset), dtype=bool)

    if n_asset == 0 or n_bus == 0:
        return switching_matrix

    bus_positions_by_int_id = {
        int(bus_int_id): position for position, bus_int_id in enumerate(station_buses["int_id"].to_list())
    }
    bus_indices = station_elements["bus_int_id"].to_numpy(dtype=int, copy=False)
    connected_asset_mask = np.array(
        [bus_int_id in bus_positions_by_int_id and bus_int_id != -1 for bus_int_id in bus_indices], dtype=bool
    )
    if np.any(connected_asset_mask):
        row_positions = np.array(
            [bus_positions_by_int_id[int(bus_int_id)] for bus_int_id in bus_indices[connected_asset_mask]]
        )
        switching_matrix[
            row_positions,
            np.flatnonzero(connected_asset_mask),
        ] = True

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
    coupler_list: list[BusbarCoupler] = []
    for coupler in coupler_elements.itertuples(index=False):
        coupler_bay = getattr(coupler, "coupler_bay", None)
        if coupler_bay is not None and not isinstance(coupler_bay, CouplerBay):
            coupler_bay = CouplerBay.model_validate(coupler_bay)
        coupler_type = getattr(coupler, "coupler_type", getattr(coupler, "type", None))
        coupler_list.append(
            BusbarCoupler.model_construct(
                grid_model_id=str(coupler.grid_model_id),
                name=coupler.name,
                coupler_type=coupler_type,
                coupler_bay=coupler_bay.model_copy(deep=True) if coupler_bay is not None else None,
            )
        )
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
    busbar_list: list[Busbar] = []
    for busbar in station_buses.itertuples(index=False):
        busbar_type = getattr(busbar, "busbar_type", None)
        busbar_list.append(
            Busbar.model_construct(
                grid_model_id=str(busbar.grid_model_id),
                busbar_type=busbar_type,
                name=busbar.name,
                int_id=int(busbar.int_id),
                bus_breaker_bus_id=getattr(busbar, "bus_breaker_bus_id", None),
            )
        )

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
) -> tuple[pd.DataFrame, list[BranchAsset | InjectionAsset], list[str | None], np.ndarray, np.ndarray]:
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
    normalized_assets : list[BranchAsset | InjectionAsset]
        Canonical station assets aligned with `station_elements`.
    asset_branch_ends : list[str | None]
        Branch-end labels aligned with `station_elements`.
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
    normalized_assets: list[BranchAsset | InjectionAsset] = []
    asset_branch_ends: list[str | None] = []
    include_branch_end = "branch_end" in station_elements.columns
    for station_element in station_elements.itertuples(index=False):
        asset_type = station_element.asset_type
        asset_kwargs = {
            "grid_model_id": str(station_element.grid_model_id),
            "asset_type": asset_type,
            "name": station_element.name,
        }
        if asset_type in get_args(AssetBranchType):
            normalized_assets.append(BranchAsset.model_construct(**asset_kwargs))
        elif asset_type in get_args(AssetInjectionType):
            normalized_assets.append(InjectionAsset.model_construct(**asset_kwargs))
        else:
            raise ValueError(f"Unsupported asset_type {asset_type!r} for asset {asset_kwargs['grid_model_id']}")
        asset_branch_ends.append(station_element.branch_end if include_branch_end else None)

    asset_connectivity = np.ones(switching_matrix.shape, dtype=bool)
    return station_elements, normalized_assets, asset_branch_ends, switching_matrix, asset_connectivity


def _get_branch_station_assets_from_df(
    normalized_assets: list[BranchAsset | InjectionAsset] | pd.DataFrame,
    asset_branch_ends: list[str | None],
    switching_matrix: np.ndarray,
    asset_connectivity: np.ndarray,
) -> tuple[list[BranchAsset], list[str | None], np.ndarray, np.ndarray]:
    """Return branch station assets and aligned arrays from prepared station inputs.

    Parameters
    ----------
    normalized_assets : list[BranchAsset | InjectionAsset]
        Canonical station assets aligned with the switching arrays.
    asset_branch_ends : list[str | None]
        Branch-end labels aligned with `station_elements`.
    switching_matrix : np.ndarray
        Station-local switching matrix aligned with `station_elements`.
    asset_connectivity : np.ndarray
        Connectivity matrix aligned with `station_elements`.

    Returns
    -------
    branch_assets : list[BranchAsset]
        Materialized branch assets for the station.
    branch_ends : list[str | None]
        Branch-end labels aligned with `branch_assets`.
    branch_switching_table : np.ndarray
        Branch-only switching table.
    branch_connectivity : np.ndarray
        Branch-only connectivity table.
    """
    if isinstance(normalized_assets, pd.DataFrame):
        normalized_assets = [
            _build_canonical_asset(asset_payload) for asset_payload in normalized_assets.to_dict(orient="records")
        ]

    branch_mask = np.asarray([isinstance(asset, BranchAsset) for asset in normalized_assets], dtype=bool)
    branch_assets = [asset for asset, is_branch in zip(normalized_assets, branch_mask, strict=True) if is_branch]
    branch_ends = [branch_end for branch_end, is_branch in zip(asset_branch_ends, branch_mask, strict=True) if is_branch]
    branch_switching_table = switching_matrix[:, branch_mask]
    branch_connectivity = asset_connectivity[:, branch_mask]

    return branch_assets, branch_ends, branch_switching_table, branch_connectivity


def _get_injection_station_assets_from_df(
    normalized_assets: list[BranchAsset | InjectionAsset] | pd.DataFrame,
    asset_branch_ends: list[str | None],
    switching_matrix: np.ndarray,
    asset_connectivity: np.ndarray,
) -> tuple[list[InjectionAsset], list[str | None], np.ndarray, np.ndarray]:
    """Return injection station assets and aligned arrays from prepared station inputs.

    Parameters
    ----------
    normalized_assets : list[BranchAsset | InjectionAsset]
        Canonical station assets aligned with the switching arrays.
    asset_branch_ends : list[str | None]
        Branch-end labels aligned with `station_elements`.
    switching_matrix : np.ndarray
        Station-local switching matrix aligned with `station_elements`.
    asset_connectivity : np.ndarray
        Connectivity matrix aligned with `station_elements`.

    Returns
    -------
    injection_assets : list[InjectionAsset]
        Materialized injection assets for the station.
    injection_branch_ends : list[str | None]
        Branch-end labels aligned with `injection_assets`.
    injection_switching_table : np.ndarray
        Injection-only switching table.
    injection_connectivity : np.ndarray
        Injection-only connectivity table.
    """
    if isinstance(normalized_assets, pd.DataFrame):
        normalized_assets = [
            _build_canonical_asset(asset_payload) for asset_payload in normalized_assets.to_dict(orient="records")
        ]

    injection_mask = np.asarray([isinstance(asset, InjectionAsset) for asset in normalized_assets], dtype=bool)
    injection_assets = [asset for asset, is_injection in zip(normalized_assets, injection_mask, strict=True) if is_injection]
    injection_branch_ends = [
        branch_end for branch_end, is_injection in zip(asset_branch_ends, injection_mask, strict=True) if is_injection
    ]
    injection_switching_table = switching_matrix[:, injection_mask]
    injection_connectivity = asset_connectivity[:, injection_mask]

    return injection_assets, injection_branch_ends, injection_switching_table, injection_connectivity


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
    station_buses["bus_breaker_bus_id"] = station_buses.index.astype(str)
    station_buses["bus_branch_bus_id"] = str(bus_id)

    # get bus df
    station_buses = (
        station_buses.sort_index().reset_index().reset_index().rename(columns={"index": "int_id", "id": "grid_model_id"})
    )
    station_buses = station_buses[
        ["grid_model_id", "name", "int_id", "in_service", "bus_breaker_bus_id", "bus_branch_bus_id"]
    ]

    return station_buses


def get_coupler_info_from_topology(
    station_switches: pd.DataFrame, switch_names: pd.Series | pd.DataFrame, station_buses: pd.DataFrame
) -> pd.DataFrame:
    """Get the coupler elements that are connected to the busbars of the station.

    Parameters
    ----------
    station_switches: pd.DataFrame
        Dataframe of all switches at the station and which busbars they connect
        Comes from BusBreakerTopology.switches
    switch_names: pd.Series
        Switch names indexed by grid-model id.
    station_buses: pd.DataFrame
        Formatted dataframe of all busbars at the station.

    Returns
    -------
    coupler_elements: pd.DataFrame
        DataFrame with the coupler elements of the station.
        Note: The DataFrame columns are the same as in the pydantic model.
    """
    busbar_int_id_by_grid_model_id = dict(
        zip(station_buses["grid_model_id"].tolist(), station_buses["int_id"].tolist(), strict=True)
    )
    if isinstance(switch_names, pd.DataFrame):
        switch_names = switch_names["name"]

    coupler_rows: list[dict[str, Any]] = []
    for station_switch in station_switches.itertuples():
        if hasattr(station_switch, "retained") and not bool(station_switch.retained):
            continue
        busbar_from_grid_model_id = str(station_switch.bus1_id)
        busbar_to_grid_model_id = str(station_switch.bus2_id)
        busbar_from_id = busbar_int_id_by_grid_model_id.get(station_switch.bus1_id)
        busbar_to_id = busbar_int_id_by_grid_model_id.get(station_switch.bus2_id)
        if busbar_from_id is None or busbar_to_id is None:
            continue
        coupler_rows.append(
            {
                "grid_model_id": station_switch.Index,
                "name": switch_names.get(station_switch.Index),
                "coupler_type": station_switch.kind,
                "in_service": True,
                "open": bool(station_switch.open),
                "busbar_from_id": busbar_from_id,
                "busbar_to_id": busbar_to_id,
                "coupler_bay": {
                    "coupler_breaker_ids": [station_switch.Index],
                    "coupler_disconnector_ids": [],
                    "from_busbar_ids": [busbar_from_grid_model_id],
                    "to_busbar_ids": [busbar_to_grid_model_id],
                    "from_busbar_disconnector_ids": {},
                    "to_busbar_disconnector_ids": {},
                },
            }
        )

    return pd.DataFrame(
        coupler_rows,
        columns=[
            "grid_model_id",
            "busbar_from_id",
            "busbar_to_id",
            "open",
            "coupler_type",
            "name",
            "in_service",
            "coupler_bay",
        ],
    )


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
    station_elements["name"] = station_elements.index.map(element_names)
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
    busbar_int_id_by_grid_model_id = dict(
        zip(station_buses["grid_model_id"].tolist(), station_buses["int_id"].tolist(), strict=True)
    )
    dangling_tie_line_ids = dangling_lines["tie_line_id"].to_dict()
    include_branch_end = "branch_end" in station_elements.columns

    normalized_rows: list[dict[str, Any]] = []
    normalized_rows_boundary_lines: list[dict[str, Any]] = []
    bus_indices: list[int] = []
    bus_indices_boundary_lines: list[int] = []
    for station_element in station_elements.itertuples():
        grid_model_id = getattr(station_element, "grid_model_id", station_element.Index)
        asset_type = station_element.type
        is_boundary_line = asset_type == "BOUNDARY_LINE"

        if is_boundary_line:
            tie_line_id = dangling_tie_line_ids.get(grid_model_id)
            if tie_line_id is not None and tie_line_id != "":
                grid_model_id = tie_line_id
                asset_type = "TIE_LINE"

        if asset_type == "BUSBAR_SECTION":
            continue

        bus_int_id = busbar_int_id_by_grid_model_id.get(station_element.bus_id)
        if bus_int_id is None:
            continue

        normalized_row: dict[str, Any] = {
            "grid_model_id": grid_model_id,
            "asset_type": asset_type,
            "name": element_names.get(grid_model_id),
            "in_service": True,
        }
        if include_branch_end:
            normalized_row["branch_end"] = station_element.branch_end
        if is_boundary_line:
            normalized_rows_boundary_lines.append(normalized_row)
            bus_indices_boundary_lines.append(bus_int_id)
        else:
            normalized_rows.append(normalized_row)
            bus_indices.append(bus_int_id)

    normalized_rows.extend(normalized_rows_boundary_lines)
    bus_indices.extend(bus_indices_boundary_lines)

    switching_matrix = np.zeros((station_buses.shape[0], len(normalized_rows)), dtype=bool)
    if bus_indices:
        switching_matrix[np.asarray(bus_indices, dtype=int), np.arange(len(bus_indices), dtype=int)] = True

    station_element_columns = ["grid_model_id", "asset_type", "name", "in_service"]
    if include_branch_end:
        station_element_columns.append("branch_end")
    return pd.DataFrame(normalized_rows, columns=station_element_columns), switching_matrix


def _infer_branch_end_from_branch_table(
    asset_grid_model_id: str,
    station_voltage_level_id: str,
    local_bus_ids: set[str],
    branches: pd.DataFrame,
) -> str | None:
    """Infer canonical branch-end metadata for one bus-breaker station-local branch occurrence.

    Parameters
    ----------
    asset_grid_model_id : str
        Canonical branch asset id.
    station_voltage_level_id : str
        Voltage-level id of the station currently being built.
    local_bus_ids : set[str]
        Bus ids visible in the current structural station group.
    branches : pd.DataFrame
        Branch table indexed by canonical branch id.

    Returns
    -------
    str | None
        Canonical branch end relative to the station, if inferable.
    """
    if asset_grid_model_id not in branches.index:
        return None

    branch_entry = branches.loc[asset_grid_model_id]
    same_voltage_level = branch_entry["voltage_level1_id"] == branch_entry["voltage_level2_id"]
    if not same_voltage_level:
        if branch_entry["voltage_level1_id"] == station_voltage_level_id:
            return "from"
        if branch_entry["voltage_level2_id"] == station_voltage_level_id:
            return "to"
        return None

    bus1_local = branch_entry["bus1_id"] in local_bus_ids
    bus2_local = branch_entry["bus2_id"] in local_bus_ids
    if bus1_local == bus2_local:
        return None
    return "from" if bus1_local else "to"


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


def get_bus_breaker_master_asset_topology(
    network: Network,
    relevant_stations: Union[list[str], Bool[np.ndarray, " n_buses"]],
    topology_id: str,
    grid_model_file: Optional[str] = None,
) -> MasterAssetTopology:
    """Return canonical topology master data derived from the current bus-breaker structure.

    Parameters
    ----------
    network : Network
        Source powsybl network.
    relevant_stations : Union[list[str], Bool[np.ndarray, " n_buses"]]
        Relevant stations as bus ids or as a boolean mask over ``network.get_buses()``.
    topology_id : str
        Identifier to store on the resulting master data.
    grid_model_file : Optional[str], optional
        Source grid-model file name stored in the master data.

    Returns
    -------
    MasterAssetTopology
        Canonical master data grouped by structural bus-breaker station views.
    """
    buses_with_substation_and_voltage, switches, dangling_lines, element_names = get_relevant_network_data(
        network=network,
        relevant_stations=relevant_stations,
    )
    master_stations: list[MasterBusGroup] = []
    topology_branch_assets: list[BranchAsset] = []
    topology_injection_assets: list[InjectionAsset] = []
    branches = network.get_branches(attributes=["voltage_level1_id", "voltage_level2_id", "bus1_id", "bus2_id"])
    for voltage_level_id, voltage_level_rows in buses_with_substation_and_voltage.groupby("voltage_level_id", sort=False):
        station_topology = network.get_bus_breaker_topology(voltage_level_id)
        structural_groups = _get_bus_breaker_structural_bus_groups(
            station_topology_buses=station_topology.buses,
            station_topology_switches=station_topology.switches,
        )
        relevant_bus_ids = {str(bus_id) for bus_id in voltage_level_rows.index}
        representative_row = voltage_level_rows.iloc[0]

        for group_index, structural_group in enumerate(structural_groups):
            station_buses = _get_bus_breaker_station_bus_info_from_group(
                station_buses=station_topology.buses,
                selected_busbar_ids=structural_group,
            )
            local_bus_ids = set(station_buses["bus_branch_bus_id"])
            if relevant_bus_ids.isdisjoint(local_bus_ids):
                continue

            coupler_elements = get_coupler_info_from_topology(station_topology.switches, switches["name"], station_buses)
            _station_elements, normalized_assets, asset_terminals, switching_matrix, asset_connectivity = (
                _get_station_asset_inputs_from_topology(
                    station_topology.elements,
                    station_buses,
                    dangling_lines,
                    element_names,
                )
            )
            (
                station_branch_assets,
                branch_terminals,
                _branch_switching_table,
                branch_connectivity,
            ) = _get_branch_station_assets_from_df(
                normalized_assets,
                asset_terminals,
                switching_matrix,
                asset_connectivity,
            )
            branch_terminals = [
                branch_terminal
                if branch_terminal is not None
                else _infer_branch_end_from_branch_table(
                    asset_grid_model_id=asset.grid_model_id,
                    station_voltage_level_id=voltage_level_id,
                    local_bus_ids=local_bus_ids,
                    branches=branches,
                )
                for asset, branch_terminal in zip(station_branch_assets, branch_terminals, strict=True)
            ]
            (
                station_injection_assets,
                injection_terminals,
                _injection_switching_table,
                injection_connectivity,
            ) = _get_injection_station_assets_from_df(
                normalized_assets,
                asset_terminals,
                switching_matrix,
                asset_connectivity,
            )

            station_busbars = get_list_of_busbars_from_df(station_buses)
            station_couplers = get_list_of_coupler_from_df(coupler_elements)

            topology_branch_assets.extend(station_branch_assets)
            topology_injection_assets.extend(station_injection_assets)
            master_stations.append(
                MasterBusGroup(
                    bus_group_id=_build_structural_station_id(voltage_level_id, group_index),
                    voltage_level_id=voltage_level_id,
                    name=representative_row.substation_id,
                    region=str(voltage_level_id)[0:2],
                    voltage_level=representative_row.nominal_v,
                    busbars=station_busbars,
                    couplers=station_couplers,
                    branch_connections=[
                        BusGroupAssetConnection(asset_id=asset.grid_model_id, branch_end=asset_terminal, asset_bay_id=None)
                        for asset, asset_terminal in zip(station_branch_assets, branch_terminals, strict=True)
                    ],
                    injection_connections=[
                        BusGroupAssetConnection(asset_id=asset.grid_model_id, branch_end=asset_terminal, asset_bay_id=None)
                        for asset, asset_terminal in zip(station_injection_assets, injection_terminals, strict=True)
                    ],
                    branch_connectivity=branch_connectivity,
                    injection_connectivity=injection_connectivity,
                )
            )

    master_data = MasterAssetTopology(
        topology_id=topology_id,
        grid_model_file=grid_model_file,
        stations=master_stations,
        branch_assets=_dedupe_assets_by_id(topology_branch_assets),
        injection_assets=_dedupe_assets_by_id(topology_injection_assets),
    )
    return master_data


def _get_busbar_sections_with_in_service(network: Network, attributes: Optional[list[str]] = None) -> pd.DataFrame:
    """Return busbar sections with an inferred in_service flag from live network state."""
    if attributes is None:
        attributes = ["in_service"]

    attributes_merge = [attribute for attribute in attributes if attribute != "in_service"]
    if "bus_id" not in attributes_merge:
        attributes_merge.append("bus_id")
    if "connected" not in attributes_merge:
        attributes_merge.append("connected")

    busbar_sections = network.get_busbar_sections(attributes=attributes_merge)
    if busbar_sections.empty:
        bus_breaker_view_buses = network.get_bus_breaker_view_buses(all_attributes=True)
        if bus_breaker_view_buses.empty:
            return busbar_sections.reindex(columns=attributes)

        bus_breaker_view_buses = bus_breaker_view_buses.copy()
        bus_breaker_view_buses["in_service"] = bus_breaker_view_buses["connected_component"] == 0
        return bus_breaker_view_buses.reindex(columns=attributes)

    busbar_sections["in_service"] = True
    buses = network.get_buses(attributes=["connected_component"])
    busbar_sections = busbar_sections.merge(
        buses,
        left_on="bus_id",
        right_index=True,
        how="left",
        suffixes=("", "_bus"),
    ).set_index(busbar_sections.index)
    busbar_sections.loc[(busbar_sections["connected_component"] != 0) | ~busbar_sections["connected"], "in_service"] = False

    return busbar_sections[attributes]


def _connected_bus_id(bus_id: object, is_connected: object) -> str | None:
    """Return a normalized connected bus id or ``None`` for disconnected assets."""
    if not bool(is_connected) or pd.isna(bus_id) or bus_id == "":
        return None
    return str(bus_id)


def _get_station_switch_ids(
    station: MasterBusGroup,
    asset_bay_map: dict[str | None, object],
) -> set[str]:
    """Collect all switch ids that influence one station runtime overlay."""
    station_switch_ids: set[str] = set()
    for coupler in station.couplers:
        if coupler.coupler_bay is None:
            continue
        station_switch_ids.update(coupler.coupler_bay.coupler_breaker_ids)
        station_switch_ids.update(coupler.coupler_bay.coupler_disconnector_ids)
        station_switch_ids.update(coupler.coupler_bay.from_busbar_disconnector_ids.values())
        station_switch_ids.update(coupler.coupler_bay.to_busbar_disconnector_ids.values())

    for asset_connection in [*station.branch_connections, *station.injection_connections]:
        if asset_connection.asset_bay_id is None:
            continue
        asset_bay = asset_bay_map.get(asset_connection.asset_bay_id)
        if asset_bay is None:
            continue
        if asset_bay.asset_disconnector_grid_model_id is not None:
            station_switch_ids.add(asset_bay.asset_disconnector_grid_model_id)
        station_switch_ids.add(asset_bay.dv_switch_grid_model_id)
        station_switch_ids.update(asset_bay.busbar_disconnector_grid_model_id.values())
    return station_switch_ids


def _get_station_busbar_runtime_state(
    station: MasterBusGroup,
    busbar_bus_id_by_id: dict[str, object],
    busbar_in_service_by_id: dict[str, object],
) -> tuple[dict[str, str], set[str]]:
    """Build station-local busbar ids and out-of-service flags from live network state."""
    busbar_bus_branch_bus_ids = {
        busbar.grid_model_id: str(busbar_bus_id_by_id[busbar.grid_model_id])
        for busbar in station.busbars
        if busbar.grid_model_id in busbar_bus_id_by_id and pd.notna(busbar_bus_id_by_id[busbar.grid_model_id])
    }
    busbar_out_of_service_ids = {
        busbar.grid_model_id
        for busbar in station.busbars
        if not bool(busbar_in_service_by_id.get(busbar.grid_model_id, True))
    }
    return busbar_bus_branch_bus_ids, busbar_out_of_service_ids


def _get_branch_current_bus_ids(
    station: MasterBusGroup,
    branches: pd.DataFrame,
) -> list[str | None]:
    """Resolve current bus ids for each branch connection of a station."""
    branch_current_bus_ids: list[str | None] = []
    for asset_connection in station.branch_connections:
        branch_row = branches.loc[asset_connection.asset_id]
        if asset_connection.branch_end == "from":
            branch_current_bus_ids.append(
                _connected_bus_id(branch_row.get("bus_breaker_bus1_id", branch_row["bus1_id"]), branch_row["connected1"])
            )
        elif asset_connection.branch_end == "to":
            branch_current_bus_ids.append(
                _connected_bus_id(branch_row.get("bus_breaker_bus2_id", branch_row["bus2_id"]), branch_row["connected2"])
            )
        else:
            branch_current_bus_ids.append(None)
    return branch_current_bus_ids


def _get_injection_current_bus_ids(
    station: MasterBusGroup,
    injections: pd.DataFrame,
) -> list[str | None]:
    """Resolve current bus ids for each injection connection of a station."""
    return [
        _connected_bus_id(
            injections.loc[asset_connection.asset_id].get(
                "bus_breaker_bus_id", injections.loc[asset_connection.asset_id]["bus_id"]
            ),
            injections.loc[asset_connection.asset_id]["connected"],
        )
        for asset_connection in station.injection_connections
    ]


def _get_runtime_branch_asset_map(
    master_data: MasterAssetTopology,
    branches: pd.DataFrame,
    buses: pd.DataFrame,
) -> dict[str, BranchAsset]:
    """Build runtime branch assets with an in-service flag derived from the live network state."""
    in_main_connected_component = buses["connected_component"].fillna(0).eq(0)
    branch_in_service = (
        branches["connected1"].fillna(False)
        & branches["connected2"].fillna(False)
        & branches["bus1_id"].map(in_main_connected_component).fillna(False)
        & branches["bus2_id"].map(in_main_connected_component).fillna(False)
    )
    runtime_in_service_by_id = branch_in_service.to_dict()
    return {
        asset.grid_model_id: RuntimeBranchAsset(
            grid_model_id=asset.grid_model_id,
            asset_type=asset.asset_type,
            name=asset.name,
            in_service=bool(runtime_in_service_by_id.get(asset.grid_model_id, False)),
        )
        for asset in master_data.branch_assets
    }


def _get_runtime_injection_asset_map(
    master_data: MasterAssetTopology,
    injections: pd.DataFrame,
    buses: pd.DataFrame,
) -> dict[str, InjectionAsset]:
    """Build runtime injection assets with an in-service flag derived from the live network state."""
    in_main_connected_component = buses["connected_component"].fillna(0).eq(0)
    injection_in_service = injections["connected"].fillna(False) & injections["bus_id"].map(
        in_main_connected_component
    ).fillna(False)
    runtime_in_service_by_id = injection_in_service.to_dict()
    return {
        asset.grid_model_id: RuntimeInjectionAsset(
            grid_model_id=asset.grid_model_id,
            asset_type=asset.asset_type,
            name=asset.name,
            in_service=bool(runtime_in_service_by_id.get(asset.grid_model_id, False)),
        )
        for asset in master_data.injection_assets
    }


def _build_runtime_switching_state(
    station: MasterBusGroup,
    switch_open_by_id: dict[str, object],
    busbar_bus_id_by_id: dict[str, object],
    busbar_in_service_by_id: dict[str, object],
    branches: pd.DataFrame,
    injections: pd.DataFrame,
    asset_bay_map: dict[str | None, AssetBay],
) -> RuntimeSwitchingState:
    """Build the compact runtime overlay inputs for one station.

    Parameters
    ----------
    station : MasterBusGroup
        Canonical station definition.
    switch_open_by_id : dict[str, object]
        Runtime switch open-state mapping.
    busbar_bus_id_by_id : dict[str, object]
        Runtime mapping from busbar id to current bus id.
    busbar_in_service_by_id : dict[str, object]
        Runtime mapping from busbar id to in-service flag.
    branches : pd.DataFrame
        Runtime branch table containing bus assignment columns.
    injections : pd.DataFrame
        Runtime injection table containing bus assignment columns.
    asset_bay_map : dict[str | None, AssetBay]
        Canonical asset bays keyed by asset-bay id.

    Returns
    -------
    RuntimeSwitchingState
        Compact runtime overlay for reconstructing one materialized station.
    """
    station_switch_ids = _get_station_switch_ids(station=station, asset_bay_map=asset_bay_map)
    busbar_bus_branch_bus_ids, busbar_out_of_service_ids = _get_station_busbar_runtime_state(
        station=station,
        busbar_bus_id_by_id=busbar_bus_id_by_id,
        busbar_in_service_by_id=busbar_in_service_by_id,
    )
    open_coupler_ids = {
        coupler.grid_model_id for coupler in station.couplers if bool(switch_open_by_id.get(coupler.grid_model_id, False))
    }
    open_switch_ids = {switch_id for switch_id in station_switch_ids if bool(switch_open_by_id.get(switch_id, False))}
    return RuntimeSwitchingState(
        busbar_bus_branch_bus_ids=busbar_bus_branch_bus_ids,
        branch_current_bus_ids=_get_branch_current_bus_ids(station=station, branches=branches),
        injection_current_bus_ids=_get_injection_current_bus_ids(station=station, injections=injections),
        busbar_out_of_service_ids=busbar_out_of_service_ids,
        open_coupler_ids=open_coupler_ids,
        out_of_service_coupler_ids=set(),
        open_switch_ids=open_switch_ids,
    )


def materialize_runtime_bus_groups_from_network_state(
    network: Network,
    master_data: MasterAssetTopology,
) -> list[RuntimeBusGroup]:
    """Materialize station snapshots from canonical master data and current runtime state.

    Parameters
    ----------
    network : Network
        Source powsybl network in its current runtime state.
    master_data : MasterAssetTopology
        Canonical master data to materialize.

    Returns
    -------
    list[RuntimeBusGroup]
        Runtime station snapshots that could be reconstructed successfully.
    """
    switches = network.get_switches(all_attributes=True)
    branches = network.get_branches(
        attributes=["bus1_id", "bus2_id", "bus_breaker_bus1_id", "bus_breaker_bus2_id", "connected1", "connected2"]
    )
    injections = network.get_injections(attributes=["bus_id", "bus_breaker_bus_id", "connected"])
    buses = network.get_buses(attributes=["connected_component"])
    switch_open_by_id = switches["open"].to_dict()
    busbar_sections = _get_busbar_sections_with_in_service(network=network, attributes=["in_service", "bus_id"])
    busbar_in_service_by_id = busbar_sections["in_service"].to_dict()
    busbar_bus_id_by_id = busbar_sections["bus_id"].to_dict()
    branch_asset_map = _get_runtime_branch_asset_map(master_data=master_data, branches=branches, buses=buses)
    injection_asset_map = _get_runtime_injection_asset_map(master_data=master_data, injections=injections, buses=buses)
    asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in master_data.asset_bays}

    materialized_stations: list[RuntimeBusGroup] = []
    for station in master_data.stations:
        try:
            runtime_switching_state = _build_runtime_switching_state(
                station=station,
                switch_open_by_id=switch_open_by_id,
                busbar_bus_id_by_id=busbar_bus_id_by_id,
                busbar_in_service_by_id=busbar_in_service_by_id,
                branches=branches,
                injections=injections,
                asset_bay_map=asset_bay_map,
            )
            materialized_stations.append(
                materialize_runtime_bus_group_from_runtime_state(
                    station=station,
                    branch_asset_map=branch_asset_map,
                    injection_asset_map=injection_asset_map,
                    asset_bay_map=asset_bay_map,
                    runtime_switching_state=runtime_switching_state,
                )
            )
        except Exception as error:
            logger.warning(
                f"Dropped runtime station {station.bus_group_id} during direct network-state materialization: {error}",
                station_id=station.bus_group_id,
                error_type=type(error).__name__,
            )

    return materialized_stations


def get_stations_and_assets_bus_breaker(
    net: Network,
) -> tuple[list[RuntimeBusGroup], list[BranchAsset], list[InjectionAsset]]:
    """Convert a bus-breaker topology grid to runtime-aware stations and topology-owned assets.

    This is mainly used for fixture and test-grid extraction.

    Parameters
    ----------
    net : Network
        The bus/breaker powsybl network to convert

    Returns
    -------
    stations : list[RuntimeBusGroup]
        List of all runtime-aware stations in the network.
    branch_assets : list[BranchAsset]
        Deduplicated topology-owned branch assets referenced by the stations.
    injection_assets : list[InjectionAsset]
        Deduplicated topology-owned injection assets referenced by the stations.
    """
    all_switches = net.get_switches(all_attributes=True)
    all_branches = net.get_branches(all_attributes=True)
    all_injections = net.get_injections(all_attributes=True)
    all_breaker_buses = net.get_bus_breaker_view_buses(all_attributes=True)

    stations: list[RuntimeBusGroup] = []
    all_branch_assets: list[BranchAsset] = []
    all_injection_assets: list[InjectionAsset] = []
    voltage_levels = get_voltage_level_with_region(net, attributes=["substation_id", "nominal_v", "topology_kind"])

    for voltage_level_id, voltage_level_row in voltage_levels[voltage_levels["topology_kind"] == "BUS_BREAKER"].iterrows():
        station_topology = net.get_bus_breaker_topology(voltage_level_id)
        structural_groups = _get_bus_breaker_structural_bus_groups(
            station_topology_buses=station_topology.buses,
            station_topology_switches=station_topology.switches,
        )

        for group_index, structural_group in enumerate(structural_groups):
            local_buses = all_breaker_buses[all_breaker_buses.index.isin(structural_group)]
            if local_buses.empty:
                continue
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
                RuntimeBusbar(
                    grid_model_id=grid_model_id,
                    int_id=busbar_mapper[grid_model_id],
                    bus_breaker_bus_id=str(grid_model_id),
                    bus_branch_bus_id=str(local_buses.loc[grid_model_id]["bus_id"]),
                )
                for grid_model_id in local_buses.index
            ]
            couplers = [
                RuntimeBusbarCoupler(
                    grid_model_id=grid_model_id,
                    coupler_type=switch.kind,
                    busbar_from_id=busbar_mapper[switch.bus_breaker_bus1_id],
                    busbar_to_id=busbar_mapper[switch.bus_breaker_bus2_id],
                    open=switch.open,
                )
                for grid_model_id, switch in local_switches.iterrows()
                if busbar_mapper[switch.bus_breaker_bus1_id] != busbar_mapper[switch.bus_breaker_bus2_id]
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

            station = RuntimeBusGroup(
                bus_group_id=_build_structural_station_id(voltage_level_id, group_index),
                voltage_level_id=voltage_level_id,
                name=voltage_level_row.substation_id,
                region=str(voltage_level_id)[0:2],
                voltage_level=voltage_level_row.nominal_v,
                busbars=busbars,
                couplers=couplers,
                branch_connections=[
                    RuntimeAssetConnection(
                        asset=RuntimeBranchAsset.model_validate(asset.model_dump()),
                        branch_end=asset_terminal,
                        asset_bay=None,
                    )
                    for asset, asset_terminal in zip(branch_assets, branch_terminals, strict=True)
                ],
                injection_connections=[
                    RuntimeAssetConnection(
                        asset=RuntimeInjectionAsset.model_validate(asset.model_dump()),
                        branch_end=None,
                        asset_bay=None,
                    )
                    for asset in injection_assets
                ],
                branch_switching_table=branch_switching_table,
                injection_switching_table=injection_switching_table,
                branch_connectivity=np.ones_like(branch_switching_table, dtype=bool),
                injection_connectivity=np.ones_like(injection_switching_table, dtype=bool),
            )
            stations.append(station)
    return stations, _dedupe_assets_by_id(all_branch_assets), _dedupe_assets_by_id(all_injection_assets)


# TODO: refactor due to C901
def assert_station_in_network(  # noqa: C901
    net: Network,
    station: RuntimeBusGroup,
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
    if station.voltage_level_id is None:
        if "_" not in station.bus_group_id:
            raise ValueError(f"Station {station.bus_group_id} is missing voltage_level_id")
        voltage_level_id = station.bus_group_id.rsplit("_", 1)[0]
    else:
        voltage_level_id = station.voltage_level_id

    bus_breaker_topo = net.get_bus_breaker_topology(voltage_level_id)
    station_connections = [*station.branch_connections, *station.injection_connections]

    for asset_connection in station_connections:
        asset = asset_connection.asset
        if asset.grid_model_id not in bus_breaker_topo.elements.index:
            raise ValueError(f"Asset {asset.grid_model_id} not found in the station elements: {bus_breaker_topo.elements}")
    if assets_strict and len(bus_breaker_topo.elements) != len(station_connections):
        raise ValueError(f"Asset count mismatch: {len(bus_breaker_topo.elements)} != {len(station_connections)}")

    for busbar in station.busbars:
        if busbar.bus_breaker_bus_id not in bus_breaker_topo.buses.index:
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
