# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Common helper functions for asset topology manipulation."""

import itertools
from numbers import Integral
from pathlib import Path

import networkx as nx
import numpy as np
from beartype.typing import Literal, Optional, Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_interfaces.asset_topology.applied_topology import AppliedStation, RealizedTopology
from toop_engine_interfaces.asset_topology.asset_topology import (
    Topology,
    copy_topology_with_updates,
)
from toop_engine_interfaces.asset_topology.assets import AssetBay, Busbar, BusbarCoupler, SwitchableAsset
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs


def electrical_components(station: MaterializedStation, min_num_assets: int = 1) -> list[list[int]]:
    """Compute the electrical components of a station.

    A set of busbars is considered a separate electrical component if it is not connected through a
    closed coupler to other busbars and there are at least two assets connected to the component.

    Parameters
    ----------
    station : MaterializedStation
        Station to analyze.
    min_num_assets : int, optional
        Minimum number of connected assets required for a component to be kept.

    Returns
    -------
    list[list[int]]
        Busbar index groups, where each inner list references positions in
        `station.busbars`.
    """
    n_connections_per_bus = station.branch_switching_table.sum(axis=1) + station.injection_switching_table.sum(axis=1)

    int_id_mapper = {busbar.int_id: i for i, busbar in enumerate(station.busbars)}

    graph = nx.Graph()
    graph.add_nodes_from(
        [(busbar.int_id, {"degree": degree}) for busbar, degree in zip(station.busbars, n_connections_per_bus, strict=True)]
    )
    graph.add_edges_from(
        [(coupler.busbar_from_id, coupler.busbar_to_id) for coupler in station.couplers if not coupler.open]
    )

    components = nx.connected_components(graph)
    # Filter out busbars with no assets connected to them
    components = [
        list(component)
        for component in components
        if sum(graph.nodes[busbar]["degree"] for busbar in component) >= min_num_assets
    ]
    # Map int ids
    components = [[int_id_mapper[busbar] for busbar in component] for component in components]

    return components


def number_of_splits(station: MaterializedStation) -> int:
    """Compute the number of electrical components in a station.

    A set of busbars is considered a separate electrical component if it is not connected through a
    closed coupler to other busbars and there are at least two assets connected to the component.

    Parameters
    ----------
    station : MaterializedStation
        Station to analyze.

    Returns
    -------
    int
        The number of electrical components in the station.
    """
    station = filter_out_of_service(station)

    components = electrical_components(station, min_num_assets=2)
    return len(components)


def remove_busbar(station: MaterializedStation, grid_model_id: str) -> MaterializedStation:
    """Remove a busbar with a specific grid_model_id from the station.

    This will
    - remove the busbar from the list of busbars
    - remove all couplers that are connected to the busbar at either end
    - remove all asset bay entries that are connected to the busbar
    - remove the line from the asset switching table
    - remove the line from the asset connectivity table

    Parameters
    ----------
    station : MaterializedStation
        Station to modify.
    grid_model_id : str
        Grid model identifier of the busbar to remove.

    Returns
    -------
    MaterializedStation
        Copy of `station` with the busbar removed.
    """
    # Store the index and int_id of the dropped busbar
    index = [b.grid_model_id for b in station.busbars].index(grid_model_id)
    int_id = station.busbars[index].int_id

    busbars = [b for b in station.busbars if b.grid_model_id != grid_model_id]
    couplers = [c for c in station.couplers if int_id not in (c.busbar_from_id, c.busbar_to_id)]
    branch_switching_table = np.delete(station.branch_switching_table, index, axis=0)
    injection_switching_table = np.delete(station.injection_switching_table, index, axis=0)
    branch_connectivity = (
        np.delete(station.branch_connectivity, index, axis=0) if station.branch_connectivity is not None else None
    )
    injection_connectivity = (
        np.delete(station.injection_connectivity, index, axis=0) if station.injection_connectivity is not None else None
    )

    def filter_sr_keys(asset_bay: Optional[AssetBay]) -> Optional[AssetBay]:
        """Drop SR switch references to the removed busbar.

        Parameters
        ----------
        asset_bay : Optional[AssetBay]
            Asset bay whose SR switch references should be filtered.

        Returns
        -------
        Optional[AssetBay]
            Updated asset bay without references to the removed busbar.
        """
        if asset_bay is None:
            return None
        return asset_bay.model_copy(
            update={
                "sr_switch_grid_model_id": {
                    busbar_id: foreign_id
                    for busbar_id, foreign_id in asset_bay.sr_switch_grid_model_id.items()
                    if busbar_id != grid_model_id
                }
            }
        )

    branch_connections = [
        asset_connection.model_copy(update={"asset_bay": filter_sr_keys(asset_connection.asset_bay)})
        for asset_connection in station.branch_connections
    ]
    injection_connections = [
        asset_connection.model_copy(update={"asset_bay": filter_sr_keys(asset_connection.asset_bay)})
        for asset_connection in station.injection_connections
    ]

    # Create a new station object with the modified busbars, couplers, and asset switching table
    new_station = station.model_copy(
        update={
            "busbars": busbars,
            "couplers": couplers,
            "branch_connections": branch_connections,
            "injection_connections": injection_connections,
            "branch_switching_table": branch_switching_table,
            "injection_switching_table": injection_switching_table,
            "branch_connectivity": branch_connectivity,
            "injection_connectivity": injection_connectivity,
        }
    )
    return new_station


def filter_out_of_service_assets(station: MaterializedStation) -> MaterializedStation:
    """Filter out-of-service assets from the station.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.

    Returns
    -------
    MaterializedStation
        Copy of `station` without out-of-service asset connections.
    """
    if all(asset_connection.asset.in_service for asset_connection in station.branch_connections) and all(
        asset_connection.asset.in_service for asset_connection in station.injection_connections
    ):
        return station

    branch_mask = [asset_connection.asset.in_service for asset_connection in station.branch_connections]
    injection_mask = [asset_connection.asset.in_service for asset_connection in station.injection_connections]
    branch_connections = [
        asset_connection
        for asset_connection, in_service in zip(station.branch_connections, branch_mask, strict=True)
        if in_service
    ]
    injection_connections = [
        asset_connection
        for asset_connection, in_service in zip(station.injection_connections, injection_mask, strict=True)
        if in_service
    ]

    return station.model_copy(
        update={
            "branch_connections": branch_connections,
            "injection_connections": injection_connections,
            "branch_switching_table": station.branch_switching_table[:, branch_mask],
            "injection_switching_table": station.injection_switching_table[:, injection_mask],
            "branch_connectivity": (
                station.branch_connectivity[:, branch_mask] if station.branch_connectivity is not None else None
            ),
            "injection_connectivity": (
                station.injection_connectivity[:, injection_mask] if station.injection_connectivity is not None else None
            ),
        }
    )


def filter_out_of_service_busbars(station: MaterializedStation) -> MaterializedStation:
    """Filter out-of-service busbars from the station.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.

    Returns
    -------
    MaterializedStation
        Copy of `station` without out-of-service busbars.
    """
    deleted_busbar_ids = [busbar.grid_model_id for busbar in station.busbars if not busbar.in_service]

    for busbar in deleted_busbar_ids:
        station = remove_busbar(station, busbar)

    return station


def filter_out_of_service_couplers(station: MaterializedStation) -> MaterializedStation:
    """Filter out-of-service couplers from the station.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.

    Returns
    -------
    MaterializedStation
        Copy of `station` without out-of-service couplers.
    """
    if all(coupler.in_service for coupler in station.couplers):
        return station

    return station.model_copy(
        update={
            "couplers": [coupler for coupler in station.couplers if coupler.in_service],
        }
    )


def filter_out_of_service(station: MaterializedStation) -> MaterializedStation:
    """Filter out-of-service assets, busbars and couplers from the station.

    The return value will be a new station object with all out-of-service elements removed.
    Note that the busbars are not reindexed, so the busbar ids will be the same as in the original
    station with missing elements. If you expect a continuous range of busbar ids, you
    should call reindex_busbars after this function.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.

    Returns
    -------
    MaterializedStation
        Copy of `station` without out-of-service assets, busbars, or couplers.
    """
    station = filter_out_of_service_couplers(station)
    station = filter_out_of_service_assets(station)
    station = filter_out_of_service_busbars(station)

    # Validate the new station object
    MaterializedStation.model_validate(station)

    return station


def filter_duplicate_couplers(
    station: MaterializedStation, retain_type_hierarchy: Optional[list[str]] = None
) -> tuple[MaterializedStation, list[BusbarCoupler]]:
    """Filter out duplicate couplers.

    Two couplers are duplicates if they connect the same busbar pair, regardless of order.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.
    retain_type_hierarchy : Optional[list[str]], optional
        Optional priority order for retained coupler types. Earlier entries have
        higher priority.

    Returns
    -------
    MaterializedStation
        Station with duplicate couplers removed.
    list[BusbarCoupler]
        Removed duplicate couplers.
    """
    # A dict with the coupler representation as key and a list of indices of the couplers
    coupler_dict: dict[tuple[int, int], list[int]] = {}
    for index, coupler in enumerate(station.couplers):
        coupler_repr = (
            min(coupler.busbar_from_id, coupler.busbar_to_id),
            max(coupler.busbar_from_id, coupler.busbar_to_id),
        )
        coupler_dict[coupler_repr] = [*coupler_dict.get(coupler_repr, []), index]

    kept_couplers = []
    removed_couplers = []
    for _coupler_repr, index in coupler_dict.items():
        # If there is only one or we don't have to sort, take the first and remove all the others
        if len(index) == 1 or retain_type_hierarchy is None:
            sorted_couplers = [station.couplers[i] for i in index]
        # We have to sort by type hierarchy
        else:
            # Sort the couplers by their type in the hierarchy, if the type is not in the hierarchy, it will be at the end
            sorted_couplers = sorted(
                (station.couplers[i] for i in index),
                key=lambda c: (
                    retain_type_hierarchy.index(c.coupler_type)
                    if c.coupler_type in retain_type_hierarchy
                    else len(retain_type_hierarchy)
                ),
            )
        # Keep the first coupler and remove the others
        kept_couplers.append(sorted_couplers[0])
        removed_couplers.extend(sorted_couplers[1:])

    if len(removed_couplers) == 0:
        return station, removed_couplers

    return (
        station.model_copy(update={"couplers": kept_couplers}),
        removed_couplers,
    )


def filter_disconnected_busbars(
    station: MaterializedStation, respect_coupler_open: bool = False
) -> tuple[MaterializedStation, list[Busbar]]:
    """Remove busbars that can not get connected by any coupler.

    This creates a graph of the busbars and couplers and returns only the largest connected component. The size
    of a component is determined by the number of assets connected to it.
    Open and closed couplers are treated the same if respect_coupler_open is False, i.e. a busbar connected
    by only open couplers is considered connected.
    Busbars connected by out-of-service couplers are always considered disconnected.

    This also means that elements which are only connected to the disconnected busbars are
    effectively disconnected, and it looks like they are subject to transmission line switching.

    Note that this function does not reindex the busbars, so the busbar ids will be the same as in
    the original station with missing elements. If you expect a continuous range of busbar ids, you
    should call reindex_busbars after this function.

    Parameters
    ----------
    station : MaterializedStation
        Station that may contain disconnected busbars.
    respect_coupler_open : bool, optional
        If `True`, only closed couplers contribute to connectivity.

    Returns
    -------
    MaterializedStation
        Station with disconnected busbars removed.
    list[Busbar]
        Removed busbars.
    """
    couplers = [
        coupler for coupler in station.couplers if (not respect_coupler_open or not coupler.open) and coupler.in_service
    ]
    graph = nx.Graph()
    num_assets_per_busbar = station.branch_switching_table.sum(axis=1) + station.injection_switching_table.sum(axis=1)
    graph.add_nodes_from(
        [
            (busbar.int_id, {"num_assets": num_assets})
            for (busbar, num_assets) in zip(station.busbars, num_assets_per_busbar, strict=True)
        ]
    )
    graph.add_edges_from([(coupler.busbar_from_id, coupler.busbar_to_id) for coupler in couplers])

    components = list(nx.connected_components(graph))
    if len(components) == 1:
        return station, []

    # Order components by the number of assets connected to them
    components.sort(key=lambda x: sum(graph.nodes[busbar]["num_assets"] for busbar in x), reverse=True)

    removed_busbars = [busbar for busbar in station.busbars if busbar.int_id not in components[0]]

    for busbar in removed_busbars:
        station = remove_busbar(station, busbar.grid_model_id)

    return station, removed_busbars


def reindex_busbars(station: MaterializedStation) -> MaterializedStation:
    """Reindex the int-ids of the busbars in the station.

    This might be necessary after filder_disconnected_busbars or filter_out_of_service have been called.

    Parameters
    ----------
    station : MaterializedStation
        Station with potentially non-contiguous busbar ids.

    Returns
    -------
    MaterializedStation
        Copy of `station` with contiguous busbar `int_id` values starting at 0.
    """
    busbar_mapping = {busbar.int_id: i for i, busbar in enumerate(station.busbars)}
    new_busbars = [busbar.model_copy(update={"int_id": i}) for i, busbar in enumerate(station.busbars)]
    new_couplers = [
        coupler.model_copy(
            update={
                "busbar_from_id": busbar_mapping[coupler.busbar_from_id],
                "busbar_to_id": busbar_mapping[coupler.busbar_to_id],
            }
        )
        for coupler in station.couplers
    ]

    station = station.model_copy(update={"busbars": new_busbars, "couplers": new_couplers})
    MaterializedStation.model_validate(station)
    return station


def filter_assets_by_type(
    station: MaterializedStation, assets_allowed: set[str], allow_none_type: bool = False
) -> tuple[MaterializedStation, list[SwitchableAsset]]:
    """Filter assets by type.

    Removes all assets that have a type which is not in the set of allowed types.

    Parameters
    ----------
    station : MaterializedStation
        Station to filter.
    assets_allowed : set[str]
        Allowed asset types.
    allow_none_type : bool, optional
        If `True`, assets without an `asset_type` are retained.

    Returns
    -------
    MaterializedStation
        Station with disallowed assets removed.
    list[SwitchableAsset]
        Removed assets.
    """
    branch_mask = [
        (asset_connection.asset.asset_type in assets_allowed)
        or (allow_none_type and asset_connection.asset.asset_type is None)
        for asset_connection in station.branch_connections
    ]
    injection_mask = [
        (asset_connection.asset.asset_type in assets_allowed)
        or (allow_none_type and asset_connection.asset.asset_type is None)
        for asset_connection in station.injection_connections
    ]
    if all(branch_mask) and all(injection_mask):
        return station, []

    removed_assets = [
        asset_connection.asset
        for asset_connection, mask in zip(station.branch_connections, branch_mask, strict=True)
        if not mask
    ] + [
        asset_connection.asset
        for asset_connection, mask in zip(station.injection_connections, injection_mask, strict=True)
        if not mask
    ]
    branch_connections = [
        asset_connection for asset_connection, mask in zip(station.branch_connections, branch_mask, strict=True) if mask
    ]
    injection_connections = [
        asset_connection
        for asset_connection, mask in zip(station.injection_connections, injection_mask, strict=True)
        if mask
    ]

    new_station = station.model_copy(
        update={
            "branch_connections": branch_connections,
            "injection_connections": injection_connections,
            "branch_switching_table": station.branch_switching_table[:, branch_mask],
            "injection_switching_table": station.injection_switching_table[:, injection_mask],
            "branch_connectivity": (
                station.branch_connectivity[:, branch_mask] if station.branch_connectivity is not None else None
            ),
            "injection_connectivity": (
                station.injection_connectivity[:, injection_mask] if station.injection_connectivity is not None else None
            ),
        }
    )
    return new_station, removed_assets


def find_multi_connected_without_coupler(
    station: MaterializedStation,
) -> list[tuple[Integral, Integral, Integral]]:
    """Find assets that bridge multiple busbars without an intervening coupler.

    These cases can cause problems in downstream processing.

    Parameters
    ----------
    station : MaterializedStation
        Station to inspect.

    Returns
    -------
    list[tuple[Integral, Integral, Integral]]
        Tuples of `(asset_index, lower_busbar_index, upper_busbar_index)` for
        multi-connected assets that bridge busbars without a coupler.
    """
    multi_connected_without_coupler = []

    def _append_multi_connected(asset_table: np.ndarray, asset_index_offset: int) -> None:
        """Collect multi-busbar asset bridges that have no coupler between them.

        Parameters
        ----------
        asset_table : np.ndarray
            Switching table to inspect.
        asset_index_offset : int
            Offset used to map local asset indices to station-wide asset indices.
        """
        for local_asset_idx in np.flatnonzero(asset_table.sum(axis=0) > 1):
            busbars_bridged = np.flatnonzero(asset_table[:, local_asset_idx]).tolist()
            for bus1_idx, bus2_idx in itertools.combinations(busbars_bridged, 2):
                smaller_bus_idx = min(bus1_idx, bus2_idx)
                larger_bus_idx = max(bus1_idx, bus2_idx)

                if not any(
                    (
                        coupler.busbar_from_id == station.busbars[smaller_bus_idx].int_id
                        and coupler.busbar_to_id == station.busbars[larger_bus_idx].int_id
                    )
                    or (
                        coupler.busbar_from_id == station.busbars[larger_bus_idx].int_id
                        and coupler.busbar_to_id == station.busbars[smaller_bus_idx].int_id
                    )
                    for coupler in station.couplers
                ):
                    multi_connected_without_coupler.append(
                        (
                            int(local_asset_idx + asset_index_offset),
                            smaller_bus_idx,
                            larger_bus_idx,
                        )
                    )

    _append_multi_connected(station.branch_switching_table, 0)
    _append_multi_connected(station.injection_switching_table, len(station.branch_connections))

    return multi_connected_without_coupler


def fix_multi_connected_without_coupler(
    station: MaterializedStation,
) -> tuple[MaterializedStation, list[tuple[SwitchableAsset, Busbar, Busbar]]]:
    """Remove one connection for unsupported multi-connected assets.

    Will always remove the connection to the busbar with the lower index.

    Parameters
    ----------
    station : MaterializedStation
        Station to fix.

    Returns
    -------
    MaterializedStation
        Station with unsupported multi-connected assets reduced to one busbar.
    list[tuple[SwitchableAsset, Busbar, Busbar]]
        Tuples of the affected asset and the two busbars it previously bridged.
    """
    multi_connected_without_coupler = find_multi_connected_without_coupler(station)
    if not multi_connected_without_coupler:
        return station, []

    branch_switching_table = np.copy(station.branch_switching_table)
    injection_switching_table = np.copy(station.injection_switching_table)
    diff = []
    n_branch = len(station.branch_connections)
    for asset_idx, bus1_idx, bus2_idx in multi_connected_without_coupler:
        if asset_idx < n_branch:
            branch_switching_table[bus1_idx, asset_idx] = 0
            asset = station.branch_connections[asset_idx].asset
        else:
            injection_idx = asset_idx - n_branch
            injection_switching_table[bus1_idx, injection_idx] = 0
            asset = station.injection_connections[injection_idx].asset
        diff.append(
            (
                asset,
                station.busbars[bus1_idx],
                station.busbars[bus2_idx],
            )
        )

    return station.model_copy(
        update={
            "branch_connections": station.branch_connections,
            "injection_connections": station.injection_connections,
            "branch_switching_table": branch_switching_table,
            "injection_switching_table": injection_switching_table,
            "branch_connectivity": station.branch_connectivity,
            "injection_connectivity": station.injection_connectivity,
        }
    ), diff


def has_transmission_line_switching(station: MaterializedStation) -> bool:
    """Check if the switching table contains transmission line switching.

    Transmission line switching is defined as disconnecting an asset from all busbars on purpose as
    a remedial action. Out-of-service assets are not considered irrespective of the switching table.

    Parameters
    ----------
    station : MaterializedStation
        Station to inspect.

    Returns
    -------
    bool
        True if the switching table contains transmission line switching, False otherwise.
    """
    branch_in_service = np.array(
        [asset_connection.asset.in_service for asset_connection in station.branch_connections], dtype=bool
    )
    injection_in_service = np.array(
        [asset_connection.asset.in_service for asset_connection in station.injection_connections], dtype=bool
    )
    return bool(
        np.any((station.branch_switching_table.sum(axis=0) == 0) & branch_in_service)
        or np.any((station.injection_switching_table.sum(axis=0) == 0) & injection_in_service)
    )


def find_busbars_for_coupler(busbars: list[Busbar], coupler: BusbarCoupler) -> tuple[Busbar, Busbar]:
    """Find the from-side and to-side busbars for a coupler.

    Matching is based on busbar `int_id` values.

    Parameters
    ----------
    busbars : list[Busbar]
        Busbars to search.
    coupler : BusbarCoupler
        Coupler whose endpoints should be resolved.

    Returns
    -------
    Busbar
        From-side busbar.
    Busbar
        To-side busbar.

    Raises
    ------
    ValueError
        If any of the busbars for the coupler are not found. This should never happen as the station validator should
        capture such a scenario.
    """
    try:
        busbar_from = next(busbar for busbar in busbars if busbar.int_id == coupler.busbar_from_id)
        busbar_to = next(busbar for busbar in busbars if busbar.int_id == coupler.busbar_to_id)
        return busbar_from, busbar_to
    except StopIteration as e:
        raise ValueError(f"Busbars for coupler {coupler.grid_model_id} not found") from e


def merge_couplers(
    original: list[BusbarCoupler],
    new: list[BusbarCoupler],
    busbar_mapping: dict[int, int],
) -> tuple[list[BusbarCoupler], list[BusbarCoupler]]:
    """Merge an updated list of couplers into the original list.

    The processed list may be a subset of the original list because preprocessing can remove
    duplicate couplers. If the processed list still contains duplicates, the merged state is open
    only if all duplicates are open.

    Parameters
    ----------
    original : list[BusbarCoupler]
        Original couplers from the unprocessed station.
    new : list[BusbarCoupler]
        Couplers from the processed station.
    busbar_mapping : dict[int, int]
        Mapping from original busbar ids to the processed station busbar ids.

    Returns
    -------
    list[BusbarCoupler]
        Updated couplers.
    list[BusbarCoupler]
        Couplers whose open state changed.
    """
    # Store the couplers in a dict for easier access
    target_state = {}
    for coupler in new:
        key = (coupler.busbar_from_id, coupler.busbar_to_id)
        # There can be a case with multiple couplers
        # In that case, the coupler is open if all are open
        # If no other coupler is present, the coupler state is just copied
        target_state[key] = coupler.open and target_state.get(key, True)

    new_couplers = []
    diff = []
    for coupler in original:
        key = (
            busbar_mapping[coupler.busbar_from_id],
            busbar_mapping[coupler.busbar_to_id],
        )
        if key in target_state and target_state[key] != coupler.open:
            new_coupler = coupler.model_copy(update={"open": target_state[key]})
            new_couplers.append(new_coupler)
            diff.append(new_coupler)
        else:
            new_couplers.append(coupler)

    return new_couplers, diff


def merge_stations(
    original: list[MaterializedStation],
    new: list[MaterializedStation],
    missing_station_behavior: Literal["append", "raise"] = "append",
) -> tuple[list[MaterializedStation], list[tuple[str, BusbarCoupler]], list[tuple[str, int, int, bool]]]:
    """Merge a list of changed stations into a list of original stations.

    Stations with matching `grid_model_id` values are merged with `merge_station`.
    Coupler and reassignment diffs are concatenated across all stations.

    Parameters
    ----------
    original : list[MaterializedStation]
        Original stations.
    new : list[MaterializedStation]
        Updated stations to merge into `original`.
    missing_station_behavior : Literal["append", "raise"], optional
        Behavior when a station exists only in `new`.

    Returns
    -------
    list[MaterializedStation]
        Merged station list.
    list[tuple[str, BusbarCoupler]]
        Coupler diffs as `(station_id, coupler)` tuples.
    list[tuple[str, int, int, bool]]
        Asset reassignment diffs as `(station_id, asset_index, busbar_index, connected)` tuples.
    """
    new_stations_found = []
    updated_station_list = []
    coupler_diff = []
    reassignment_diff = []
    for station in original:
        found = False
        for new_station in new:
            if station.grid_model_id == new_station.grid_model_id:
                updated_station, coupler_diff_local, reassignment_diff_local = merge_station(station, new_station)
                updated_station_list.append(updated_station)
                new_stations_found.append(new_station.grid_model_id)
                coupler_diff.extend([(station.grid_model_id, coupler) for coupler in coupler_diff_local])
                reassignment_diff.extend(
                    [
                        (station.grid_model_id, asset_idx, busbar_idx, bool(connected))
                        for asset_idx, busbar_idx, connected in reassignment_diff_local
                    ]
                )
                found = True
                break
        if not found:
            updated_station_list.append(station)

    # Check if there are new stations that were not found in the original list
    for new_station in new:
        if new_station.grid_model_id not in new_stations_found:
            if missing_station_behavior == "append":
                updated_station_list.append(new_station)
            else:
                raise ValueError(f"Station {new_station.grid_model_id} was not found in the original list")

    return updated_station_list, coupler_diff, reassignment_diff


# TODO: refactor due to C901
def merge_station(
    original: MaterializedStation, new: MaterializedStation
) -> tuple[MaterializedStation, list[BusbarCoupler], list[tuple[int, int, bool]]]:
    """Merge all the changes from the new station into the original station.

    Matching assets, couplers, and busbars are updated from `new` when they can be mapped back
    onto `original`. Elements that cannot be matched are left unchanged.

    If all in-service elements of original are also in new, then the returned substation will be
    electrically equivalent to the new substation. If this is not the case, the returned substation
    has all possible changes applied, but there are cases in which this is not electrically
    equivalent to the new substation.

    Use `compare_stations` to inspect structural differences between both stations.

    Parameters
    ----------
    original : MaterializedStation
        Original station to update.
    new : MaterializedStation
        Station that provides the target changes.

    Returns
    -------
    MaterializedStation
        Copy of `original` with all mergeable changes from `new` applied.
    list[BusbarCoupler]
        Couplers whose open state changed.
    list[tuple[int, int, bool]]
        Asset switching diff as `(asset_index, busbar_index, connected)` tuples.
    """
    if original == new:
        return original, [], []

    # Find a mapping from old busbars to new busbars. We just need an index mapping for copying the
    # switching table. Also store the missed busbars.
    busbar_mapping = {}
    busbar_int_id_mapping = {}
    max_busbar_id = max(busbar.int_id for busbar in new.busbars)
    branch_asset_mapping, injection_asset_mapping, new_couplers, coupler_diff = map_busbars_and_assets(
        original, new, busbar_mapping, busbar_int_id_mapping, max_busbar_id
    )

    # Loop through the switching table and copy the values over for which there is a mapping
    new_branch_switching_table = original.branch_switching_table.copy()
    new_injection_switching_table = original.injection_switching_table.copy()
    station, asset_diff = update_asset_switching_table(
        original,
        new,
        busbar_mapping,
        branch_asset_mapping,
        injection_asset_mapping,
        new_couplers,
        new_branch_switching_table,
        new_injection_switching_table,
    )
    return station, coupler_diff, asset_diff


def update_asset_switching_table(
    original_station: MaterializedStation,
    new_station: MaterializedStation,
    busbar_mapping: dict[int, int],
    branch_asset_mapping: dict[int, int],
    injection_asset_mapping: dict[int, int],
    new_couplers: list[BusbarCoupler],
    new_branch_switching_table: np.ndarray,
    new_injection_switching_table: np.ndarray,
) -> tuple[MaterializedStation, list[tuple[int, int, bool]]]:
    """Update switching tables and couplers using a mapped target station.

    This helper copies branch and injection switching states from `new_station` into
    `original_station` for all mapped busbars and assets. It also replaces the
    coupler list with `new_couplers` and records the resulting switching diff.

    Parameters
    ----------
    original_station : MaterializedStation
        Original station that receives the mapped switching updates.
    new_station : MaterializedStation
        Station that provides the target switching states.
    busbar_mapping : dict[int, int]
        Mapping from original busbar indices to busbar indices in `new_station`.
    branch_asset_mapping : dict[int, int]
        Mapping from original branch connection indices to branch indices in `new_station`.
    injection_asset_mapping : dict[int, int]
        Mapping from original injection connection indices to injection indices in `new_station`.
    new_couplers : list[BusbarCoupler]
        Couplers to store on the returned station.
    new_branch_switching_table : np.ndarray
        Branch switching table buffer for the returned station. Modified in place.
    new_injection_switching_table : np.ndarray
        Injection switching table buffer for the returned station. Modified in place.

    Returns
    -------
    MaterializedStation
        Copy of `original_station` with updated switching tables and couplers.
    list[tuple[int, int, bool]]
        Switching diff as `(asset_index, busbar_index, connected)` tuples.
    """
    asset_diff = []
    for busbar_idx, mapped_busbar_idx in busbar_mapping.items():
        for branch_idx, mapped_branch_idx in branch_asset_mapping.items():
            if (
                new_branch_switching_table[busbar_idx, branch_idx]
                != new_station.branch_switching_table[mapped_busbar_idx, mapped_branch_idx]
            ):
                asset_diff.append(
                    (
                        branch_idx,
                        busbar_idx,
                        bool(new_station.branch_switching_table[mapped_busbar_idx, mapped_branch_idx]),
                    )
                )
                new_branch_switching_table[busbar_idx, branch_idx] = new_station.branch_switching_table[
                    mapped_busbar_idx, mapped_branch_idx
                ]

        for injection_idx, mapped_injection_idx in injection_asset_mapping.items():
            if (
                new_injection_switching_table[busbar_idx, injection_idx]
                != new_station.injection_switching_table[mapped_busbar_idx, mapped_injection_idx]
            ):
                asset_diff.append(
                    (
                        injection_idx + len(original_station.branch_connections),
                        busbar_idx,
                        bool(new_station.injection_switching_table[mapped_busbar_idx, mapped_injection_idx]),
                    )
                )
                new_injection_switching_table[busbar_idx, injection_idx] = new_station.injection_switching_table[
                    mapped_busbar_idx, mapped_injection_idx
                ]

    original_station = original_station.model_copy(
        update={
            "couplers": new_couplers,
            "branch_connections": original_station.branch_connections,
            "injection_connections": original_station.injection_connections,
            "branch_switching_table": new_branch_switching_table,
            "injection_switching_table": new_injection_switching_table,
            "branch_connectivity": original_station.branch_connectivity,
            "injection_connectivity": original_station.injection_connectivity,
        }
    )
    return original_station, asset_diff


def map_busbars_and_assets(
    original_station: MaterializedStation,
    new_station: MaterializedStation,
    busbar_mapping: dict[int, int],
    busbar_int_id_mapping: dict[int, int],
    max_busbar_id: int,
) -> tuple[dict[int, int], dict[int, int], list[BusbarCoupler], list[BusbarCoupler]]:
    """Build busbar and asset mappings between two materialized stations.

    This helper matches busbars and asset connections by `grid_model_id`, updates the
    provided busbar mapping dictionaries in place, and prepares merged coupler state
    for `merge_station`.

    Parameters
    ----------
    original_station : MaterializedStation
        Original station whose busbars and asset connections should be matched.
    new_station : MaterializedStation
        Target station that provides the reference busbars, assets, and couplers.
    busbar_mapping : dict[int, int]
        Mapping from original busbar indices to busbar indices in `new_station`.
        Modified in place.
    busbar_int_id_mapping : dict[int, int]
        Mapping from original busbar `int_id` values to corresponding `int_id` values
        in `new_station`. Missing busbars are assigned synthetic ids beyond
        `max_busbar_id`. Modified in place.
    max_busbar_id : int
        Maximum busbar `int_id` present in `new_station`.

    Returns
    -------
    dict[int, int]
        Mapping from original branch connection indices to branch indices in `new_station`.
    dict[int, int]
        Mapping from original injection connection indices to injection indices in `new_station`.
    list[BusbarCoupler]
        Coupler list to apply to the original station.
    list[BusbarCoupler]
        Couplers whose open state changes during the merge.
    """

    def _build_index_by_grid_model_id(elements: list[Union[Busbar, SwitchableAsset]]) -> dict[str, int]:
        """Map each element grid model id to its position in the input list.

        Parameters
        ----------
        elements : list[Union[Busbar, SwitchableAsset]]
            Elements that expose a `grid_model_id` attribute.

        Returns
        -------
        dict[str, int]
            Mapping from `grid_model_id` to element position.
        """
        return {element.grid_model_id: index for index, element in enumerate(elements)}

    def _map_asset_connections(
        original_connections: list[MaterializedAssetConnection],
        new_connections: list[MaterializedAssetConnection],
    ) -> dict[int, int]:
        """Map original asset connection indices to matching target connection indices.

        Parameters
        ----------
        original_connections : list[MaterializedAssetConnection]
            Original asset connections to map.
        new_connections : list[MaterializedAssetConnection]
            Target asset connections that define the mapped indices.

        Returns
        -------
        dict[int, int]
            Mapping from original connection index to target connection index.
        """
        new_asset_indices = _build_index_by_grid_model_id([asset_connection.asset for asset_connection in new_connections])
        return {
            index: new_asset_indices[asset_connection.asset.grid_model_id]
            for index, asset_connection in enumerate(original_connections)
            if asset_connection.asset.grid_model_id in new_asset_indices
        }

    new_busbar_indices = _build_index_by_grid_model_id(new_station.busbars)
    new_busbar_int_ids = {busbar.grid_model_id: busbar.int_id for busbar in new_station.busbars}
    for index, busbar in enumerate(original_station.busbars):
        mapped_busbar_index = new_busbar_indices.get(busbar.grid_model_id)
        if mapped_busbar_index is None:
            # Make sure to not accidentally map to an existing busbar
            busbar_int_id_mapping[busbar.int_id] = busbar.int_id + max_busbar_id + 1
            continue

        busbar_mapping[index] = mapped_busbar_index
        busbar_int_id_mapping[busbar.int_id] = new_busbar_int_ids[busbar.grid_model_id]

    branch_asset_mapping = _map_asset_connections(original_station.branch_connections, new_station.branch_connections)
    injection_asset_mapping = _map_asset_connections(
        original_station.injection_connections,
        new_station.injection_connections,
    )

    # Merge couplers
    new_couplers, coupler_diff = merge_couplers(
        original_station.couplers, new_station.couplers, busbar_mapping=busbar_int_id_mapping
    )
    return branch_asset_mapping, injection_asset_mapping, new_couplers, coupler_diff


def compare_stations(
    a: MaterializedStation, b: MaterializedStation
) -> tuple[
    list[BusbarCoupler],
    list[BusbarCoupler],
    list[Busbar],
    list[Busbar],
    list[SwitchableAsset],
    list[SwitchableAsset],
]:
    """Compare two stations and return the missing elements.

    It uses grid_model_ids to compare the assets, busbars and couplers. It does not consider
    different switching states or coupler states, but just checks if all the elements are also in
    the other station.

    Parameters
    ----------
    a : MaterializedStation
        First station to compare.
    b : MaterializedStation
        Second station to compare.

    Returns
    -------
    list[BusbarCoupler]
        Couplers present in `a` but not in `b`.
    list[BusbarCoupler]
        Couplers present in `b` but not in `a`.
    list[Busbar]
        Busbars present in `a` but not in `b`.
    list[Busbar]
        Busbars present in `b` but not in `a`.
    list[SwitchableAsset]
        Assets present in `a` but not in `b`.
    list[SwitchableAsset]
        Assets present in `b` but not in `a`.
    """
    a_busbars = set(busbar.grid_model_id for busbar in a.busbars)
    b_busbars = set(busbar.grid_model_id for busbar in b.busbars)

    a_couplers = set(coupler.grid_model_id for coupler in a.couplers)
    b_couplers = set(coupler.grid_model_id for coupler in b.couplers)

    a_assets = {
        *(asset_connection.asset.grid_model_id for asset_connection in a.branch_connections),
        *(asset_connection.asset.grid_model_id for asset_connection in a.injection_connections),
    }
    b_assets = {
        *(asset_connection.asset.grid_model_id for asset_connection in b.branch_connections),
        *(asset_connection.asset.grid_model_id for asset_connection in b.injection_connections),
    }

    return (
        [coupler for coupler in a.couplers if coupler.grid_model_id not in b_couplers],
        [coupler for coupler in b.couplers if coupler.grid_model_id not in a_couplers],
        [busbar for busbar in a.busbars if busbar.grid_model_id not in b_busbars],
        [busbar for busbar in b.busbars if busbar.grid_model_id not in a_busbars],
        [
            *(
                asset_connection.asset
                for asset_connection in a.branch_connections
                if asset_connection.asset.grid_model_id not in b_assets
            ),
            *(
                asset_connection.asset
                for asset_connection in a.injection_connections
                if asset_connection.asset.grid_model_id not in b_assets
            ),
        ],
        [
            *(
                asset_connection.asset
                for asset_connection in b.branch_connections
                if asset_connection.asset.grid_model_id not in a_assets
            ),
            *(
                asset_connection.asset
                for asset_connection in b.injection_connections
                if asset_connection.asset.grid_model_id not in a_assets
            ),
        ],
    )


def load_asset_topology_fs(
    filesystem: AbstractFileSystem,
    file_path: Union[str, Path],
) -> Topology:
    """Load an asset topology from a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        File system to load from.
    file_path : Union[str, Path]
        Path to the JSON file containing the topology.

    Returns
    -------
    Topology
        The loaded asset topology.
    """
    return load_pydantic_model_fs(
        filesystem=filesystem,
        file_path=file_path,
        model_class=Topology,
    )


def load_asset_topology(filename: Union[str, Path]) -> Topology:
    """Load an asset topology from a file.

    Parameters
    ----------
    filename : Union[str, Path]
        File name to load the topology from.

    Returns
    -------
    Topology
        Loaded topology.
    """
    return load_asset_topology_fs(
        filesystem=LocalFileSystem(),
        file_path=filename,
    )


def save_asset_topology_fs(filesystem: AbstractFileSystem, filename: Union[str, Path], asset_topology: Topology) -> None:
    """Save an asset topology to a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        File system to save to.
    filename : Union[str, Path]
        File name to save the topology to.
    asset_topology : Topology
        Topology to save.
    """
    with filesystem.open(str(filename), "w", encoding="utf-8") as file:
        file.write(asset_topology.model_dump_json(indent=2))


def save_asset_topology(filename: Union[str, Path], asset_topology: Topology) -> None:
    """Save an asset topology to a file.

    Parameters
    ----------
    filename : Union[str, Path]
        File name to save the topology to.
    asset_topology : Topology
        Topology to save.
    """
    save_asset_topology_fs(LocalFileSystem(), filename, asset_topology)


def accumulate_diffs(
    realized_stations: list[AppliedStation],
) -> tuple[
    list[tuple[str, BusbarCoupler]],
    list[tuple[str, int, int, bool]],
    list[tuple[str, int, int, bool]],
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    """Accumulate the diffs of the realized stations into the format of realized topology.

    Parameters
    ----------
    realized_stations : list[AppliedStation]
        Realized stations to accumulate.

    Returns
    -------
    list[tuple[str, BusbarCoupler]]
        Accumulated coupler diff.
    list[tuple[str, int, int, bool]]
        Accumulated branch reassignment diff.
    list[tuple[str, int, int, bool]]
        Accumulated injection reassignment diff.
    list[tuple[str, int]]
        Accumulated branch disconnection diff.
    list[tuple[str, int]]
        Accumulated injection disconnection diff.
    """
    coupler_diff = []
    branch_reassignment_diff = []
    injection_reassignment_diff = []
    branch_disconnection_diff = []
    injection_disconnection_diff = []
    for station in realized_stations:
        s_id = station.station.grid_model_id
        coupler_diff.extend([(s_id, coupler) for coupler in station.coupler_diff])
        branch_reassignment_diff.extend(
            [(s_id, asset_idx, bus_idx, connected) for (asset_idx, bus_idx, connected) in station.branch_reassignment_diff]
        )
        injection_reassignment_diff.extend(
            [
                (s_id, asset_idx, bus_idx, connected)
                for (asset_idx, bus_idx, connected) in station.injection_reassignment_diff
            ]
        )
        branch_disconnection_diff.extend([(s_id, asset_idx) for asset_idx in station.branch_disconnection_diff])
        injection_disconnection_diff.extend([(s_id, asset_idx) for asset_idx in station.injection_disconnection_diff])

    return (
        coupler_diff,
        branch_reassignment_diff,
        injection_reassignment_diff,
        branch_disconnection_diff,
        injection_disconnection_diff,
    )


def station_diff(
    start_station: MaterializedStation,
    target_station: MaterializedStation,
) -> AppliedStation:
    """Compute the diff between two stations.

    The same station must be described by both inputs, i.e. the assets, busbars and couplers (except for their open state)
    must be the same.

    Parameters
    ----------
    start_station : MaterializedStation
        Starting station.
    target_station : MaterializedStation
        Target station.

    Returns
    -------
    AppliedStation
        Realized station containing the target station and all derived diffs.
    """
    assert [s.asset.grid_model_id for s in start_station.branch_connections] == [
        s.asset.grid_model_id for s in target_station.branch_connections
    ], "Branch assets do not match"
    assert [s.asset.grid_model_id for s in start_station.injection_connections] == [
        s.asset.grid_model_id for s in target_station.injection_connections
    ], "Injection assets do not match"
    assert [b.grid_model_id for b in start_station.busbars] == [b.grid_model_id for b in target_station.busbars], (
        "Busbars do not match"
    )
    assert [b.grid_model_id for b in start_station.couplers] == [b.grid_model_id for b in target_station.couplers], (
        "Couplers do not match"
    )

    def _table_diff(
        start_table: np.ndarray,
        target_table: np.ndarray,
        asset_kind: str,
    ) -> tuple[list[tuple[int, int, bool]], list[int]]:
        """Compute reassignment and disconnection diffs for one asset table.

        Parameters
        ----------
        start_table : np.ndarray
            Initial switching table.
        target_table : np.ndarray
            Target switching table.
        asset_kind : str
            Asset kind used in error messages.

        Returns
        -------
        list[tuple[int, int, bool]]
            Reassignment diff as `(asset_index, busbar_index, connected)` tuples.
        list[int]
            Asset indices that become disconnected.
        """
        reassignment_diff_local: list[tuple[int, int, bool]] = []
        disconnection_diff_local: list[int] = []
        for asset_index in range(start_table.shape[1]):
            target_disconnected = ~np.any(target_table[:, asset_index])
            start_disconnected = ~np.any(start_table[:, asset_index])

            if start_disconnected and not target_disconnected:
                raise NotImplementedError(
                    "Reconnections are not supported yet, there is no diff for that"
                    + f" ({asset_kind} {asset_index} in station {start_station.grid_model_id})"
                )

            if target_disconnected and not start_disconnected:
                disconnection_diff_local.append(int(asset_index))

            if target_disconnected:
                continue

            xor_diff = np.logical_xor(start_table[:, asset_index], target_table[:, asset_index])
            for busbar_index in np.flatnonzero(xor_diff):
                reassignment_diff_local.append(
                    (int(asset_index), int(busbar_index), bool(target_table[busbar_index, asset_index]))
                )
        return reassignment_diff_local, disconnection_diff_local

    branch_reassignment_diff, branch_disconnection_diff = _table_diff(
        start_station.branch_switching_table,
        target_station.branch_switching_table,
        "branch",
    )
    injection_reassignment_diff, injection_disconnection_diff = _table_diff(
        start_station.injection_switching_table,
        target_station.injection_switching_table,
        "injection",
    )

    coupler_diff = []
    for start_coupler, target_coupler in zip(start_station.couplers, target_station.couplers, strict=True):
        if start_coupler.open != target_coupler.open:
            coupler_diff.append(target_coupler)

    return AppliedStation(
        station=target_station,
        coupler_diff=coupler_diff,
        branch_reassignment_diff=branch_reassignment_diff,
        injection_reassignment_diff=injection_reassignment_diff,
        branch_disconnection_diff=branch_disconnection_diff,
        injection_disconnection_diff=injection_disconnection_diff,
    )


def topology_diff(
    start_topo: Topology,
    target_topo: Topology,
) -> RealizedTopology:
    """Compute the difference between two topologies.

    Parameters
    ----------
    start_topo : Topology
        Starting topology.
    target_topo : Topology
        Target topology.

    Returns
    -------
    RealizedTopology
        Realized topology containing the target topology and all diffs from the start topology.
    """
    realized_stations = [
        station_diff(start_station, target_station)
        for (start_station, target_station) in zip(
            start_topo.materialize_stations(), target_topo.materialize_stations(), strict=True
        )
    ]
    (
        coupler_diff,
        branch_reassignment_diff,
        injection_reassignment_diff,
        branch_disconnection_diff,
        injection_disconnection_diff,
    ) = accumulate_diffs(realized_stations)
    return RealizedTopology(
        topology=target_topo,
        coupler_diff=coupler_diff,
        branch_reassignment_diff=branch_reassignment_diff,
        injection_reassignment_diff=injection_reassignment_diff,
        branch_disconnection_diff=branch_disconnection_diff,
        injection_disconnection_diff=injection_disconnection_diff,
    )


def order_station_assets(
    station: MaterializedStation, asset_ids: list[str]
) -> tuple[MaterializedStation, list[str], list[str]]:
    """Order station assets according to a list of asset ids.

    Asset ids not present in the station are reported in `not_found`.
    Assets omitted from `asset_ids` are dropped and reported in `ignored`.

    Parameters
    ----------
    station : MaterializedStation
        Station to reorder.
    asset_ids : list[str]
        Asset ids in the desired order.

    Returns
    -------
    MaterializedStation
        Station with reordered assets.
    list[str]
        Asset ids that were not found in the station.
    list[str]
        Asset ids ignored because they were not present in `asset_ids`.
    """
    branch_lookup = {
        asset_connection.asset.grid_model_id: (index, asset_connection)
        for index, asset_connection in enumerate(station.branch_connections)
    }
    injection_lookup = {
        asset_connection.asset.grid_model_id: (index, asset_connection)
        for index, asset_connection in enumerate(station.injection_connections)
    }
    new_branch_connections = []
    new_injection_connections = []
    not_found = []
    branch_positions = []
    injection_positions = []
    for asset_id in asset_ids:
        if asset_id in branch_lookup:
            pos, asset_connection = branch_lookup[asset_id]
            new_branch_connections.append(asset_connection)
            branch_positions.append(pos)
        elif asset_id in injection_lookup:
            pos, asset_connection = injection_lookup[asset_id]
            new_injection_connections.append(asset_connection)
            injection_positions.append(pos)
        else:
            not_found.append(asset_id)

    ignored = [
        asset_connection.asset.grid_model_id
        for index, asset_connection in enumerate(station.branch_connections)
        if index not in branch_positions
    ] + [
        asset_connection.asset.grid_model_id
        for index, asset_connection in enumerate(station.injection_connections)
        if index not in injection_positions
    ]

    station = station.model_copy(
        update={
            "branch_connections": new_branch_connections,
            "injection_connections": new_injection_connections,
            "branch_switching_table": station.branch_switching_table[:, branch_positions],
            "injection_switching_table": station.injection_switching_table[:, injection_positions],
            "branch_connectivity": (
                station.branch_connectivity[:, branch_positions] if station.branch_connectivity is not None else None
            ),
            "injection_connectivity": (
                station.injection_connectivity[:, injection_positions]
                if station.injection_connectivity is not None
                else None
            ),
        }
    )
    MaterializedStation.model_validate(station)
    return station, not_found, ignored


def order_topology(topology: Topology, station_ids: list[str]) -> tuple[Topology, list[str]]:
    """Order topology stations according to a list of ids.

    Station ids not present in the topology are reported in `not_found`.
    Stations omitted from `station_ids` are dropped.

    Parameters
    ----------
    topology : Topology
        Topology to reorder.
    station_ids : list[str]
        Station ids in the desired order.

    Returns
    -------
    Topology
        Topology with reordered stations.
    list[str]
        Station ids that were not found in the topology.
    """
    new_stations = []
    not_found = []
    for relevant_node in station_ids:
        found = False
        for station in topology.raw_stations:
            if station.grid_model_id == relevant_node:
                new_stations.append(station)
                found = True
                break
        if not found:
            not_found.append(relevant_node)

    topology = copy_topology_with_updates(
        topology,
        new_stations,
        topology.asset_bays,
        branch_assets=topology.branch_assets,
        injection_assets=topology.injection_assets,
    )
    return topology, not_found


def _coupler_connects_same_busbars(coupler: BusbarCoupler, other_coupler: BusbarCoupler) -> bool:
    """Return whether two couplers connect the same busbar pair.

    Parameters
    ----------
    coupler : BusbarCoupler
        Reference coupler.
    other_coupler : BusbarCoupler
        Coupler to compare against `coupler`.

    Returns
    -------
    bool
        `True` if both couplers connect the same busbar pair.
    """
    return (
        other_coupler.busbar_from_id == coupler.busbar_from_id and other_coupler.busbar_to_id == coupler.busbar_to_id
    ) or (other_coupler.busbar_to_id == coupler.busbar_from_id and other_coupler.busbar_from_id == coupler.busbar_to_id)


def _validate_coupler_can_be_fused(station: MaterializedStation, coupler: BusbarCoupler) -> None:
    """Raise if the coupler has a parallel coupler on the same busbar pair.

    Parameters
    ----------
    station : MaterializedStation
        Station containing the coupler.
    coupler : BusbarCoupler
        Coupler that should be fused.

    Raises
    ------
    ValueError
        If more than one coupler connects the same busbar pair.
    """
    parallel_couplers = [
        other_coupler for other_coupler in station.couplers if _coupler_connects_same_busbars(coupler, other_coupler)
    ]
    if len(parallel_couplers) > 1:
        raise ValueError(
            f"Coupler {coupler.grid_model_id} has parallel couplers in station {station.grid_model_id}, "
            "cannot fuse parallel couplers with the same busbars"
        )


def _resolve_fused_busbars(
    station: MaterializedStation,
    coupler: BusbarCoupler,
    copy_info_from: bool,
) -> tuple[int, int, int, int, Busbar, Busbar]:
    """Resolve busbar indices and busbar objects used during coupler fusion.

    Parameters
    ----------
    station : MaterializedStation
        Station containing the coupler.
    coupler : BusbarCoupler
        Coupler whose adjacent busbars should be resolved.
    copy_info_from : bool
        Whether metadata should be kept from the from-side busbar.

    Returns
    -------
    int
        From-side busbar index.
    int
        To-side busbar index.
    int
        Kept busbar index.
    int
        Removed busbar index.
    Busbar
        Kept busbar.
    Busbar
        Removed busbar.
    """
    busbar_from, busbar_to = find_busbars_for_coupler(station.busbars, coupler)
    busbar_index_by_int_id = {busbar.int_id: index for index, busbar in enumerate(station.busbars)}
    busbar_from_index = busbar_index_by_int_id[busbar_from.int_id]
    busbar_to_index = busbar_index_by_int_id[busbar_to.int_id]
    keep_busbar_index = busbar_from_index if copy_info_from else busbar_to_index
    remove_busbar_index = busbar_to_index if copy_info_from else busbar_from_index
    keep_busbar = station.busbars[keep_busbar_index]
    remove_busbar = station.busbars[remove_busbar_index]
    return busbar_from_index, busbar_to_index, keep_busbar_index, remove_busbar_index, keep_busbar, remove_busbar


def _merge_and_delete_fused_busbar_row(
    table: Optional[np.ndarray],
    busbar_from_index: int,
    busbar_to_index: int,
    keep_busbar_index: int,
    remove_busbar_index: int,
) -> Optional[np.ndarray]:
    """Merge fused busbar rows and delete the removed busbar row.

    Parameters
    ----------
    table : Optional[np.ndarray]
        Switching or connectivity table to update.
    busbar_from_index : int
        From-side busbar index.
    busbar_to_index : int
        To-side busbar index.
    keep_busbar_index : int
        Busbar index that should remain in the table.
    remove_busbar_index : int
        Busbar index that should be removed from the table.

    Returns
    -------
    Optional[np.ndarray]
        Updated table, or `None` if the input table is `None`.
    """
    if table is None:
        return None
    merged_table = np.copy(table)
    merged_table[keep_busbar_index] = merged_table[busbar_from_index] | merged_table[busbar_to_index]
    return np.delete(merged_table, remove_busbar_index, axis=0)


def _replace_sr_keys_for_fused_busbar(
    asset_bay: Optional[AssetBay],
    keep_busbar: Busbar,
    remove_busbar: Busbar,
) -> Optional[AssetBay]:
    """Rewrite SR switch busbar keys from the removed busbar to the kept one.

    Parameters
    ----------
    asset_bay : Optional[AssetBay]
        Asset bay whose SR switch mapping should be updated.
    keep_busbar : Busbar
        Busbar that remains after fusion.
    remove_busbar : Busbar
        Busbar that is removed during fusion.

    Returns
    -------
    Optional[AssetBay]
        Updated asset bay, or `None` if no asset bay is present.
    """
    if asset_bay is None:
        return None

    new_sr_switch_grid_model_id = {}
    for key, foreign_id in asset_bay.sr_switch_grid_model_id.items():
        if key == remove_busbar.grid_model_id and keep_busbar.grid_model_id in asset_bay.sr_switch_grid_model_id:
            continue
        mapped_key = keep_busbar.grid_model_id if key == remove_busbar.grid_model_id else key
        new_sr_switch_grid_model_id[mapped_key] = foreign_id

    return asset_bay.model_copy(update={"sr_switch_grid_model_id": new_sr_switch_grid_model_id})


def _update_asset_bays_for_fused_busbar(
    asset_connections: list[MaterializedAssetConnection],
    keep_busbar: Busbar,
    remove_busbar: Busbar,
) -> list[MaterializedAssetConnection]:
    """Apply the fused-busbar SR key rewrite to each asset connection.

    Parameters
    ----------
    asset_connections : list[MaterializedAssetConnection]
        Asset connections to update.
    keep_busbar : Busbar
        Busbar that remains after fusion.
    remove_busbar : Busbar
        Busbar that is removed during fusion.

    Returns
    -------
    list[MaterializedAssetConnection]
        Updated asset connections.
    """
    return [
        asset_connection.model_copy(
            update={"asset_bay": _replace_sr_keys_for_fused_busbar(asset_connection.asset_bay, keep_busbar, remove_busbar)}
        )
        for asset_connection in asset_connections
    ]


def _replace_coupler_busbar_id_for_fusion(
    coupler: BusbarCoupler,
    keep_busbar: Busbar,
    remove_busbar: Busbar,
) -> BusbarCoupler:
    """Rewrite coupler endpoints that still reference the removed busbar.

    Parameters
    ----------
    coupler : BusbarCoupler
        Coupler to update.
    keep_busbar : Busbar
        Busbar that remains after fusion.
    remove_busbar : Busbar
        Busbar that is removed during fusion.

    Returns
    -------
    BusbarCoupler
        Updated coupler with rewritten endpoints when needed.
    """
    if coupler.busbar_from_id == remove_busbar.int_id:
        return coupler.model_copy(update={"busbar_from_id": keep_busbar.int_id})
    if coupler.busbar_to_id == remove_busbar.int_id:
        return coupler.model_copy(update={"busbar_to_id": keep_busbar.int_id})
    return coupler


def fuse_coupler(
    station: MaterializedStation,
    coupler_grid_model_id: str,
    copy_info_from: bool = True,
) -> MaterializedStation:
    """Fuse a coupler by merging its adjacent busbars.

    Parameters
    ----------
    station : MaterializedStation
        Station containing the coupler to fuse. The coupler open state is ignored.
    coupler_grid_model_id : str
        Grid model identifier of the coupler to fuse.
    copy_info_from : bool, optional
        If `True`, keep metadata from the busbar on the coupler's from-side.
        If `False`, keep metadata from the to-side busbar.

    Returns
    -------
    MaterializedStation
        Station with the coupler removed and both busbars merged.

    Raises
    ------
    ValueError
        If the coupler is missing or if parallel couplers connect the same busbar pair.
    """
    coupler = next((c for c in station.couplers if c.grid_model_id == coupler_grid_model_id), None)
    if coupler is None:
        raise ValueError(f"Coupler {coupler_grid_model_id} not found in station {station.grid_model_id}")

    _validate_coupler_can_be_fused(station, coupler)
    busbar_from_index, busbar_to_index, keep_busbar_index, remove_busbar_index, keep_busbar, remove_busbar = (
        _resolve_fused_busbars(station, coupler, copy_info_from)
    )

    new_branch_switching_table = _merge_and_delete_fused_busbar_row(
        station.branch_switching_table,
        busbar_from_index,
        busbar_to_index,
        keep_busbar_index,
        remove_busbar_index,
    )
    new_injection_switching_table = _merge_and_delete_fused_busbar_row(
        station.injection_switching_table,
        busbar_from_index,
        busbar_to_index,
        keep_busbar_index,
        remove_busbar_index,
    )
    new_branch_connectivity = _merge_and_delete_fused_busbar_row(
        station.branch_connectivity,
        busbar_from_index,
        busbar_to_index,
        keep_busbar_index,
        remove_busbar_index,
    )
    new_injection_connectivity = _merge_and_delete_fused_busbar_row(
        station.injection_connectivity,
        busbar_from_index,
        busbar_to_index,
        keep_busbar_index,
        remove_busbar_index,
    )

    new_busbars = [busbar for index, busbar in enumerate(station.busbars) if index != remove_busbar_index]
    new_couplers = [
        _replace_coupler_busbar_id_for_fusion(other_coupler, keep_busbar, remove_busbar)
        for other_coupler in station.couplers
        if other_coupler.grid_model_id != coupler_grid_model_id
    ]
    new_branch_connections = _update_asset_bays_for_fused_busbar(station.branch_connections, keep_busbar, remove_busbar)
    new_injection_connections = _update_asset_bays_for_fused_busbar(
        station.injection_connections,
        keep_busbar,
        remove_busbar,
    )

    station = station.model_copy(
        update={
            "busbars": new_busbars,
            "branch_connections": new_branch_connections,
            "injection_connections": new_injection_connections,
            "branch_switching_table": new_branch_switching_table,
            "injection_switching_table": new_injection_switching_table,
            "branch_connectivity": new_branch_connectivity,
            "injection_connectivity": new_injection_connectivity,
            "couplers": new_couplers,
        }
    )
    MaterializedStation.model_validate(station)
    return station


def fuse_all_couplers_with_type(
    station: MaterializedStation,
    coupler_type: str,
    copy_info_from: bool = True,
) -> tuple[MaterializedStation, list[BusbarCoupler]]:
    """Fuse all couplers of a given type in a station.

    Duplicate couplers are filtered after each fusion step so chained merges do not
    leave residual parallel couplers between already fused busbars.

    Parameters
    ----------
    station : MaterializedStation
        Station containing the couplers to fuse.
    coupler_type : str
        Coupler type to match against `coupler.coupler_type`.
    copy_info_from : bool, optional
        If `True`, keep metadata from the from-side busbar for each fusion.
        If `False`, keep metadata from the to-side busbar.

    Returns
    -------
    MaterializedStation
        Station after all matching couplers have been fused.
    list[BusbarCoupler]
        Couplers removed either by fusion or duplicate filtering.
    """
    fused_couplers = []
    while True:
        coupler = next(
            (c for c in station.couplers if (c.coupler_type is not None and c.coupler_type == coupler_type)), None
        )
        if coupler is None:
            break

        station = fuse_coupler(station, coupler.grid_model_id, copy_info_from=copy_info_from)
        fused_couplers.append(coupler)

        # We want to retain the coupler type that we filter for and remove all other couplers - this way we can
        # make sure that all parallel couplers are removed. Otherwise it would depend on the order of the couplers
        # whether there would be a residual coupler left between the potentially fused busbars.
        station, removed = filter_duplicate_couplers(station, retain_type_hierarchy=[coupler_type])
        fused_couplers.extend(removed)

    return station, fused_couplers


def find_station_by_id(stations: list[MaterializedStation], station_id: str) -> MaterializedStation:
    """Find a station by its grid_model_id in a list of stations.

    Parameters
    ----------
    stations : list[MaterializedStation]
        Stations to search.
    station_id : str
        Grid model identifier of the station to find.

    Returns
    -------
    MaterializedStation
        Matching station.

    Raises
    ------
    ValueError
        If no station with the requested identifier is present.
    """
    for station in stations:
        if station.grid_model_id == station_id:
            return station
    raise ValueError(f"Station {station_id} not found in the list")
