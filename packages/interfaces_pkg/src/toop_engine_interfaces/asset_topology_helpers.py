"""Collects some common helper functions for asset topology manipulation."""

import itertools
from pathlib import Path

import networkx as nx
import numpy as np
from beartype.typing import Literal, Optional, Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    RealizedStation,
    RealizedTopology,
    Station,
    SwitchableAsset,
    Topology,
)


def electrical_components(station: Station, min_num_assets: int = 1) -> list[list[int]]:
    """Compute the electrical components of a station.

    A set of busbars is considered a separate electrical component if it is not connected through a
    closed coupler to other busbars and there are at least two assets connected to the component.

    Parameters
    ----------
    station : Station
        The station object to analyze.
    min_num_assets : int, optional
        The minimum number of assets connected to a component to be considered a valid component, by default 1

    Returns
    -------
    list[list[int]]
        A list of lists, where each inner list contains the indices of the busbars in the component indexing into the list
        of all busbars in the station.
    """
    n_connections_per_bus = station.asset_switching_table.sum(axis=1)

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


def number_of_splits(station: Station) -> int:
    """Compute the number of electrical components that are present in a station.

    A set of busbars is considered a separate electrical component if it is not connected through a
    closed coupler to other busbars and there are at least two assets connected to the component.

    Parameters
    ----------
    station : Station
        The station object to analyze.

    Returns
    -------
    int
        The number of electrical components in the station.
    """
    station = filter_out_of_service(station)

    components = electrical_components(station, min_num_assets=2)
    return len(components)


def remove_busbar(station: Station, grid_model_id: str) -> Station:
    """Remove a busbar with a specific grid_model_id from the station.

    This will
    - remove the busbar from the list of busbars
    - remove all couplers that are connected to the busbar at either end
    - remove all asset bay entries that are connected to the busbar
    - remove the line from the asset switching table
    - remove the line from the asset connectivity table

    Parameters
    ----------
    station : Station
        The station object to modify.
    grid_model_id : str
        The grid_model_id of the busbar to remove.

    Returns
    -------
    Station
        The modified station object with the busbar removed.
    """
    # Store the index and int_id of the dropped busbar
    index = [b.grid_model_id for b in station.busbars].index(grid_model_id)
    int_id = station.busbars[index].int_id

    busbars = [b for b in station.busbars if b.grid_model_id != grid_model_id]
    couplers = [c for c in station.couplers if int_id not in (c.busbar_from_id, c.busbar_to_id)]
    asset_switching_table = np.delete(station.asset_switching_table, index, axis=0)
    asset_connectivity = (
        np.delete(station.asset_connectivity, index, axis=0) if station.asset_connectivity is not None else None
    )

    def filter_sr_keys(asset: SwitchableAsset) -> SwitchableAsset:
        """Filter out the sr keys from the asset bay"""
        if asset.asset_bay is None:
            return asset
        return asset.model_copy(
            update={
                "asset_bay": asset.asset_bay.model_copy(
                    update={
                        "sr_switch_grid_model_id": {
                            busbar_id: foreign_id
                            for busbar_id, foreign_id in asset.asset_bay.sr_switch_grid_model_id.items()
                            if busbar_id != grid_model_id
                        }
                    }
                )
            }
        )

    assets = [filter_sr_keys(a) for a in station.assets]

    # Create a new station object with the modified busbars, couplers, and asset switching table
    new_station = station.model_copy(
        update={
            "busbars": busbars,
            "couplers": couplers,
            "assets": assets,
            "asset_switching_table": asset_switching_table,
            "asset_connectivity": asset_connectivity,
        }
    )
    return new_station


def filter_out_of_service_assets(station: Station) -> Station:
    """Filter out-of-service assets from the station.

    Parameters
    ----------
    station : Station
        The station object to filter.

    Returns
    -------
    Station
        The new station object with all out-of-service assets removed.
    """
    if all(asset.in_service for asset in station.assets):
        return station

    in_service_assets = [asset.in_service for asset in station.assets]

    return station.model_copy(
        update={
            "assets": [asset for asset in station.assets if asset.in_service],
            "asset_switching_table": station.asset_switching_table[:, in_service_assets],
            "asset_connectivity": station.asset_connectivity[:, in_service_assets]
            if station.asset_connectivity is not None
            else None,
        }
    )


def filter_out_of_service_busbars(station: Station) -> Station:
    """Filter out-of-service busbars from the station.

    Parameters
    ----------
    station : Station
        The station object to filter.

    Returns
    -------
    Station
        The new station object with all out-of-service busbars removed.
    """
    deleted_busbar_ids = [busbar.grid_model_id for busbar in station.busbars if not busbar.in_service]

    for busbar in deleted_busbar_ids:
        station = remove_busbar(station, busbar)

    return station


def filter_out_of_service_couplers(station: Station) -> Station:
    """Filter out-of-service couplers from the station.

    Parameters
    ----------
    station : Station
        The station object to filter.

    Returns
    -------
    Station
        The new station object with all out-of-service couplers removed.
    """
    if all(coupler.in_service for coupler in station.couplers):
        return station

    return station.model_copy(
        update={
            "couplers": [coupler for coupler in station.couplers if coupler.in_service],
        }
    )


def filter_out_of_service(station: Station) -> Station:
    """Filter out-of-service assets, busbars and couplers from the station.

    The return value will be a new station object with all out-of-service elements removed.
    Note that the busbars are not reindexed, so the busbar ids will be the same as in the original
    station with missing elements. If you expect a continuous range of busbar ids, you
    should call reindex_busbars after this function.

    Parameters
    ----------
    station : Station
        The station object to filter.

    Returns
    -------
    Station
        The new station object with all out-of-service assets removed.
    """
    station = filter_out_of_service_couplers(station)
    station = filter_out_of_service_assets(station)
    station = filter_out_of_service_busbars(station)

    # Validate the new station object
    Station.model_validate(station)

    return station


def filter_duplicate_couplers(
    station: Station, retain_type_hierarchy: Optional[list[str]] = None
) -> tuple[Station, list[BusbarCoupler]]:
    """Filter out duplicate couplers

    Two couplers are considered duplicates if they connect the same busbars, regardless of their
    order. If a duplicate coupler is found, only the first one is kept.

    Parameters
    ----------
    station : Station
        The station object to filter.
    retain_type_hierarchy : Optional[list[str]], optional
        If provided, not the first coupler is kept but the one with the highest type in the hierarchy list. Highest means
        that the type is the first in the list. If not provided, the first coupler is kept, by default None

    Returns
    -------
    Station
        The new station object with duplicate couplers removed.
    list[BusbarCoupler]
        The list of removed couplers.
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
                key=lambda c: retain_type_hierarchy.index(c.type)
                if c.type in retain_type_hierarchy
                else len(retain_type_hierarchy),
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


def filter_disconnected_busbars(station: Station, respect_coupler_open: bool = False) -> tuple[Station, list[Busbar]]:
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
    station : Station
        The station object potentially with disconnected busbars.
    respect_coupler_open : bool, optional
        If True, only closed couplers are considered connected, if False, all couplers are
        considered connected, by default False

    Returns
    -------
    Station
        The new station object with disconnected busbars removed.
    list[Busbar]
        The list of removed busbars.
    """
    couplers = [
        coupler for coupler in station.couplers if (not respect_coupler_open or not coupler.open) and coupler.in_service
    ]
    graph = nx.Graph()
    num_assets_per_busbar = station.asset_switching_table.sum(axis=1)
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


def reindex_busbars(station: Station) -> Station:
    """Reindex the int-ids of the busbars in the station

    This might be necessary after filder_disconnected_busbars or filter_out_of_service have been called.

    Parameters
    ----------
    station : Station
        The station object with possible inconsistent busbar ids.

    Returns
    -------
    Station
        The new station object with reindexed busbars, where int-ids are continuous and start at 0.
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
    Station.model_validate(station)
    return station


def filter_assets_by_type(
    station: Station, assets_allowed: set[str], allow_none_type: bool = False
) -> tuple[Station, list[SwitchableAsset]]:
    """Filter assets by type

    Removes all assets that have a type which is not in the set of allowed types.

    Parameters
    ----------
    station : Station
        The station object to filter.
    assets_allowed : set[str]
        The set of asset types that are allowed.
    allow_none_type : bool, optional
        If True, assets without a type are allowed, by default False

    Returns
    -------
    Station
        The new station object with assets of the wrong type removed.
    list[SwitchableAsset]
        The list of removed assets.
    """
    asset_mask = [(asset.type in assets_allowed) or (allow_none_type and asset.type is None) for asset in station.assets]
    if all(asset_mask):
        return station, []

    removed_assets = [asset for asset, mask in zip(station.assets, asset_mask, strict=True) if not mask]
    kept_assets = [asset for asset, mask in zip(station.assets, asset_mask, strict=True) if mask]

    new_station = station.model_copy(
        update={
            "assets": kept_assets,
            "asset_switching_table": station.asset_switching_table[:, asset_mask],
            "asset_connectivity": station.asset_connectivity[:, asset_mask]
            if station.asset_connectivity is not None
            else None,
        }
    )
    return new_station, removed_assets


def find_multi_connected_without_coupler(
    station: Station,
) -> list[tuple[int, int, int]]:
    """Find assets bridging multiple busbars without a coupler in between

    These cases can cause a bug in the downstream functions

    Parameters
    ----------
    station : Station
        The station object to check.

    Returns
    -------
    list[tuple[int, int, int]]
        A list of tuples containing the index of the multi-connected asset and the indices of the
        two busbars it is bridging.
        Only returns bridges that don't have a coupler in between.
        The first busbar index will always be lower than the second, hence a routine which always
        removes the first doesn't run into double removals if an asset appears multiple times.
    """
    multi_connected_without_coupler = []
    for asset_idx in np.flatnonzero(station.asset_switching_table.sum(axis=0) > 1):
        busbars_bridged = np.flatnonzero(station.asset_switching_table[:, asset_idx]).tolist()
        for bus1_idx, bus2_idx in itertools.combinations(busbars_bridged, 2):
            # bus1_idx shall always be the smaller of the two
            if bus1_idx > bus2_idx:
                smaller_bus_idx, larger_bus_idx = bus2_idx, bus1_idx
            else:
                smaller_bus_idx, larger_bus_idx = bus1_idx, bus2_idx

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
                        asset_idx,
                        smaller_bus_idx,
                        larger_bus_idx,
                    )
                )

    return multi_connected_without_coupler


def fix_multi_connected_without_coupler(
    station: Station,
) -> tuple[Station, list[tuple[SwitchableAsset, Busbar, Busbar]]]:
    """Remove one connection for multi-connected assets without a coupler in between

    Will always remove the connection to the busbar with the lower index.

    Parameters
    ----------
    station : Station
        The station object to fix

    Returns
    -------
    Station
        The new station object with the multi-connected assets fixed.
    list[tuple[SwitchableAsset, Busbar, Busbar]]
        A list of tuples containing the previously multi-connected assets and the busbars they were
        connected to. The first busbar in the tuple is the one that was disconnected.
    """
    multi_connected_without_coupler = find_multi_connected_without_coupler(station)
    if not multi_connected_without_coupler:
        return station, []

    new_asset_switching_table = np.copy(station.asset_switching_table)
    diff = []
    for asset_idx, bus1_idx, bus2_idx in multi_connected_without_coupler:
        new_asset_switching_table[bus1_idx, asset_idx] = 0
        diff.append(
            (
                station.assets[asset_idx],
                station.busbars[bus1_idx],
                station.busbars[bus2_idx],
            )
        )

    return station.model_copy(update={"asset_switching_table": new_asset_switching_table}), diff


def has_transmission_line_switching(station: Station) -> bool:
    """Check if the switching table contains transmission line switching

    Transmission line switching is defined as disconnecting an asset from all busbars on purpose as
    a remedial action. Out-of-service assets are not considered irrespective of the switching table.

    Parameters
    ----------
    station : Station
        The station object to check.

    Returns
    -------
    bool
        True if the switching table contains transmission line switching, False otherwise.
    """
    in_service = np.array([asset.in_service for asset in station.assets])
    return np.any((station.asset_switching_table.sum(axis=0) == 0) & in_service).item()


def find_busbars_for_coupler(busbars: list[Busbar], coupler: BusbarCoupler) -> tuple[Busbar, Busbar]:
    """Find the from and two busbars for a coupler

    Finds based on the int_id of the busbars.

    Parameters
    ----------
    busbars : list[Busbar]
        The list of busbars to search in
    coupler : BusbarCoupler
        The coupler to search for

    Returns
    -------
    Busbar
        The from busbar
    Busbar
        The to busbar

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
    """Merge an updated list of couplers into the original list

    Due to preprocessing actions, the new list might contain a subset of the original couplers.
    Especially duplicate couplers were removed, so if the original list contains duplicates, both
    couplers need to be switched. It assumes that the busbar ids are mapped through the busbar
    mapping

    If the new list contains duplicates, the couplers are counted as open if all duplicates are open
    and as closed otherwise

    Parameters
    ----------
    original : list[BusbarCoupler]
        The original list of couplers of the unprocessed station
    new : list[BusbarCoupler]
        The new list of couplers of the processed station after applying the topology
    busbar_mapping : dict[int, int]
        The mapping from the original busbar indices to the new busbar indices

    Returns
    -------
    list[BusbarCoupler]
        The updated list of couplers
    list[BusbarCoupler]
        The coupler diff in this station, i.e. which couplers have been switched. Stores the
        new state of the couplers, i.e. which state the coupler has been switched to
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
    original: list[Station],
    new: list[Station],
    missing_station_behavior: Literal["append", "raise"] = "append",
) -> tuple[list[Station], list[tuple[str, BusbarCoupler]], list[tuple[str, int, int, bool]]]:
    """Merge a list of changed stations into a list of original stations

    All stations with a grid_model_id that is present in the new list will be updated using
    merge_station. The coupler+reassignment diffs will be concatenated. If a new station is not
    present in the original list, it will be appended to the end of the list if
    missing_station_behavior is "append".

    Parameters
    ----------
    original : list[Station]
        The original list of stations
    new : list[Station]
        The list of changed stations
    missing_station_behavior : Literal["append", "raise"], optional
        What to do if a station is not found in the original list, by default "append"

    Returns
    -------
    list[Station]
        The updated list of stations
    list[tuple[str, BusbarCoupler]]
        The coupler diff that has been switched, consisting of tuples with the station grid_model_id
        and the coupler that has been switched
    list[tuple[str, int, int, bool]]
        The reassignment diff that has been switched, consisting of tuples with the station grid_model_id,
        the asset index that was affected, which busbar index was affected and whether the asset was
        connected (True) or disconnected (False) to that bus
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
def merge_station(original: Station, new: Station) -> tuple[Station, list[BusbarCoupler], list[tuple[int, int, bool]]]:  # noqa: PLR0912, C901
    """Merge all the changes from the new station into the original station

    This will overwrite all assets, couplers and busbars in the original station with the ones from
    the new station if the couplers are also found in the new station. Things that are not found in
    the new station will be left untouched. Assets that are in the new station but not in the
    original will also be left untouched.

    If all in-service elements of original are also in new, then the returned substation will be
    electrically equivalent to the new substation. If this is not the case, the returned substation
    has all possible changes applied, but there are cases in which this is not electrically
    equivalent to the new substation.

    You can use asset_topology_helpers.compare_stations to check which elements are different
    in the two stations and infer the differences.

    Parameters
    ----------
    original : Station
        The original station that should be modified
    new : Station
        The new station that contains the changes that should be merged into the original station

    Returns
    -------
    Station
        The modified original station, with all the changes from the new station merged in
    list[BusbarCoupler]
        The coupler diff that has been switched
    list[tuple[int, int, bool]]
        The asset diff that has been switched. Each tuple contains the asset index that was
        affected, which busbar index was affected and whether the asset was connected (True) or
        disconnected (False) to that bus
    """
    if original == new:
        return original, [], []

    # Find a mapping from old busbars to new busbars. We just need an index mapping for copying the
    # switching table. Also store the missed busbars.
    busbar_mapping = {}
    busbar_int_id_mapping = {}
    max_busbar_id = max(busbar.int_id for busbar in new.busbars)
    for i, busbar in enumerate(original.busbars):
        found = False
        for j, other in enumerate(new.busbars):
            if busbar.grid_model_id == other.grid_model_id:
                busbar_mapping[i] = j
                busbar_int_id_mapping[busbar.int_id] = other.int_id
                found = True
                break
        if not found:
            # Make sure to not accidentally map to an existing busbar
            busbar_int_id_mapping[busbar.int_id] = busbar.int_id + max_busbar_id + 1

    # Same for the switchable assets
    asset_mapping = {}
    for i, asset in enumerate(original.assets):
        for j, other in enumerate(new.assets):
            if asset.grid_model_id == other.grid_model_id:
                asset_mapping[i] = j
                break

    # Merge couplers
    new_couplers, coupler_diff = merge_couplers(original.couplers, new.couplers, busbar_mapping=busbar_int_id_mapping)

    # Loop through the switching table and copy the values over for which there is a mapping
    asset_diff = []
    new_asset_switching_table = original.asset_switching_table.copy()
    for busbar_idx in range(original.asset_switching_table.shape[0]):
        if busbar_idx in busbar_mapping:
            mapped_busbar_idx = busbar_mapping[busbar_idx]
            for asset_idx in range(original.asset_switching_table.shape[1]):
                if asset_idx in asset_mapping:
                    mapped_asset_idx = asset_mapping[asset_idx]

                    if (
                        new_asset_switching_table[busbar_idx, asset_idx]
                        != new.asset_switching_table[mapped_busbar_idx, mapped_asset_idx]
                    ):
                        asset_diff.append(
                            (
                                asset_idx,
                                busbar_idx,
                                new.asset_switching_table[mapped_busbar_idx, mapped_asset_idx],
                            )
                        )

                        new_asset_switching_table[busbar_idx, asset_idx] = new.asset_switching_table[
                            mapped_busbar_idx, mapped_asset_idx
                        ]

    station = original.model_copy(update={"couplers": new_couplers, "asset_switching_table": new_asset_switching_table})
    return station, coupler_diff, asset_diff


def compare_stations(
    a: Station, b: Station
) -> tuple[
    list[BusbarCoupler],
    list[BusbarCoupler],
    list[Busbar],
    list[Busbar],
    list[SwitchableAsset],
    list[SwitchableAsset],
]:
    """Compare two stations and return the missing elements

    It uses grid_model_ids to compare the assets, busbars and couplers. It does not consider
    different switching states or coupler states, but just checks if all the elements are also in
    the other station.

    Parameters
    ----------
    a : Station
        The first station to compare.
    b : Station
        The second station to compare.

    Returns
    -------
    list[BusbarCoupler]
        The couplers that are in a but not in b.
    list[BusbarCoupler]
        The couplers that are in b but not in a.
    list[Busbar]
        The busbars that are in a but not in b.
    list[Busbar]
        The busbars that are in b but not in a.
    list[SwitchableAsset]
        The assets that are in a but not in b.
    list[SwitchableAsset]
        The assets that are in b but not in a.
    """
    a_busbars = set(busbar.grid_model_id for busbar in a.busbars)
    b_busbars = set(busbar.grid_model_id for busbar in b.busbars)

    a_couplers = set(coupler.grid_model_id for coupler in a.couplers)
    b_couplers = set(coupler.grid_model_id for coupler in b.couplers)

    a_assets = set(asset.grid_model_id for asset in a.assets)
    b_assets = set(asset.grid_model_id for asset in b.assets)

    return (
        [coupler for coupler in a.couplers if coupler.grid_model_id not in b_couplers],
        [coupler for coupler in b.couplers if coupler.grid_model_id not in a_couplers],
        [busbar for busbar in a.busbars if busbar.grid_model_id not in b_busbars],
        [busbar for busbar in b.busbars if busbar.grid_model_id not in a_busbars],
        [asset for asset in a.assets if asset.grid_model_id not in b_assets],
        [asset for asset in b.assets if asset.grid_model_id not in a_assets],
    )


def load_asset_topology_fs(filesystem: AbstractFileSystem, filename: Union[str, Path]) -> Topology:
    """Load an asset topology from a file system

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to load the asset topology from
    filename : Union[str, Path]
        The filename to load the asset topology from

    Returns
    -------
    Topology
        The loaded asset topology
    """
    with filesystem.open(str(filename), "r", encoding="utf-8") as file:
        asset_topology = Topology.model_validate_json(file.read())
    return asset_topology


def load_asset_topology(filename: Union[str, Path]) -> Topology:
    """Load an asset topology from a file

    Parameters
    ----------
    filename : Union[str, Path]
        The filename to load the asset topology from

    Returns
    -------
    Topology
        The loaded asset topology
    """
    return load_asset_topology_fs(LocalFileSystem(), filename)


def save_asset_topology_fs(filesystem: AbstractFileSystem, filename: Union[str, Path], asset_topology: Topology) -> None:
    """Save an asset topology to a file system

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to save the asset topology to
    filename : Union[str, Path]
        The filename to save the asset topology to
    asset_topology: Topology
        The asset topology to save
    """
    with filesystem.open(str(filename), "w", encoding="utf-8") as file:
        file.write(asset_topology.model_dump_json(indent=2))


def save_asset_topology(filename: Union[str, Path], asset_topology: Topology) -> None:
    """Save an asset topology to a file

    Parameters
    ----------
    filename : Union[str, Path]
        The filename to save the asset topology to
    asset_topology: Topology
        The asset topology to save
    """
    save_asset_topology_fs(LocalFileSystem(), filename, asset_topology)


def get_connected_assets(station: Station, busbar_index: int) -> list[SwitchableAsset]:
    """
    Get the assets connected to a specific busbar in a station.

    Parameters
    ----------
    station : Station
        The station object containing the switching table and assets.
    busbar_index : int
        The index of the busbar in the switching table.

    Returns
    -------
    list of SwitchableAsset
        A list of assets connected to the specified busbar.
    """
    connected_asset_indices = np.where(station.asset_switching_table[busbar_index])[0]
    return [station.assets[i] for i in connected_asset_indices if station.assets[i].in_service]


def accumulate_diffs(
    realized_stations: list[RealizedStation],
) -> tuple[
    list[tuple[str, BusbarCoupler]],
    list[tuple[str, int, int, bool]],
    list[tuple[str, int]],
]:
    """Accumulate the diffs of the realized stations into the format of realized topology

    Parameters
    ----------
    realized_stations : list[RealizedStation]
        The realized stations to accumulate

    Returns
    -------
    list[tuple[str, BusbarCoupler]]
        The accumulated coupler diff
    list[tuple[str, int, int, bool]]
        The accumulated reassignment diff
    list[tuple[str, int]]
        The accumulated disconnection diff
    """
    coupler_diff = []
    reassignment_diff = []
    disconnection_diff = []
    for station in realized_stations:
        s_id = station.station.grid_model_id
        coupler_diff.extend([(s_id, coupler) for coupler in station.coupler_diff])
        reassignment_diff.extend(
            [(s_id, asset_idx, bus_idx, connected) for (asset_idx, bus_idx, connected) in station.reassignment_diff]
        )
        disconnection_diff.extend([(s_id, asset_idx) for asset_idx in station.disconnection_diff])

    return coupler_diff, reassignment_diff, disconnection_diff


def station_diff(
    start_station: Station,
    target_station: Station,
) -> RealizedStation:
    """Compute the diff between two stations

    The same station must be described by both inputs, i.e. the assets, busbars and couplers (except for their open state)
    must be the same.

    Parameters
    ----------
    start_station : Station
        The starting station from which to start the diff
    target_station : Station
        The ending station to which the diff shall lead when applied to the starting station

    Returns
    -------
    RealizedStation
        The realized station containing the target station and the coupler, reassignment and disconnection diffs
    """
    assert [s.grid_model_id for s in start_station.assets] == [s.grid_model_id for s in target_station.assets], (
        "Assets do not match"
    )
    assert [b.grid_model_id for b in start_station.busbars] == [b.grid_model_id for b in target_station.busbars], (
        "Busbars do not match"
    )
    assert [b.grid_model_id for b in start_station.couplers] == [b.grid_model_id for b in target_station.couplers], (
        "Couplers do not match"
    )

    reassignment_diff = []
    disconnection_diff = []
    for asset_index in range(len(start_station.assets)):
        target_disconnected = ~np.any(target_station.asset_switching_table[:, asset_index])
        start_disconnected = ~np.any(start_station.asset_switching_table[:, asset_index])

        if start_disconnected and not target_disconnected:
            raise NotImplementedError(
                "Reconnections are not supported yet, there is no diff for that"
                + f" (asset {asset_index} in station {start_station.grid_model_id})"
            )

        if target_disconnected and not start_disconnected:
            # if the asset is connected in the target station but disconnected in the starting station
            # we need to add it to the diff
            disconnection_diff.append(asset_index)

        # No reassignment diff if the asset is disconnected
        if target_disconnected:
            continue

        xor_diff = np.logical_xor(
            start_station.asset_switching_table[:, asset_index],
            target_station.asset_switching_table[:, asset_index],
        )
        for busbar_index in np.flatnonzero(xor_diff):
            # If the start and end switching entries differ, add it to the reassignment diff
            reassignment_diff.append(
                (asset_index, busbar_index, target_station.asset_switching_table[busbar_index, asset_index])
            )

    coupler_diff = []
    for start_coupler, target_coupler in zip(start_station.couplers, target_station.couplers, strict=True):
        if start_coupler.open != target_coupler.open:
            coupler_diff.append(target_coupler)

    return RealizedStation(
        station=target_station,
        coupler_diff=coupler_diff,
        reassignment_diff=reassignment_diff,
        disconnection_diff=disconnection_diff,
    )


def topology_diff(
    start_topo: Topology,
    target_topo: Topology,
) -> RealizedTopology:
    """Compute the difference between two topologies

    Parameters
    ----------
    start_topo : Topology
        The starting topology
    target_topo : Topology
        The targeted topology.

    Returns
    -------
    RealizedTopology
        The realized topology containing the target topology and the coupler, reassignment and disconnection diffs that lead
        from the starting topology to the target topology.
    """
    realized_stations = [
        station_diff(start_station, target_station)
        for (start_station, target_station) in zip(start_topo.stations, target_topo.stations, strict=True)
    ]
    coupler_diff, reassignment_diff, disconnection_diff = accumulate_diffs(realized_stations)
    return RealizedTopology(
        topology=target_topo,
        coupler_diff=coupler_diff,
        reassignment_diff=reassignment_diff,
        disconnection_diff=disconnection_diff,
    )


def order_station_assets(station: Station, asset_ids: list[str]) -> tuple[Station, list[str]]:
    """Orders the assets in a station according to a list of asset ids.

    If an asset is not found in the station, it will be added to the not_found list.
    If an asset is not present in the asset_ids list, it will be dropped from the station.

    Parameters
    ----------
    station : Station
        The station to order.
    asset_ids : list[str]
        A list of asset ids. The assets will be ordered according to the grid_model_id.

    Returns
    -------
    Station
        The ordered station
    list[str]
        A list of asset ids that were not found in the station
    list[str]
        A list of asset ids that were ignored, i.e. not present in the asset_ids list
    """
    new_assets = []
    not_found = []
    old_positions = []
    for asset_id in asset_ids:
        found = False
        for pos, asset in enumerate(station.assets):
            if asset.grid_model_id == asset_id:
                new_assets.append(asset)
                old_positions.append(pos)
                found = True
                break
        if not found:
            not_found.append(asset_id)

    ignored = [asset.grid_model_id for index, asset in enumerate(station.assets) if index not in old_positions]

    asset_switching_table = station.asset_switching_table[:, old_positions]
    asset_connectivity = station.asset_connectivity[:, old_positions] if station.asset_connectivity is not None else None
    station = station.model_copy(
        update={
            "assets": new_assets,
            "asset_switching_table": asset_switching_table,
            "asset_connectivity": asset_connectivity,
        }
    )
    Station.model_validate(station)
    return station, not_found, ignored


def order_topology(topology: Topology, station_ids: list[str]) -> tuple[Topology, list[str]]:
    """Orders the stations in a topology according to a list of ids.

    If a station is not found in the topology, it will be added to the not_found list.
    If a station is not present in the station_ids list, it will be dropped.

    Parameters
    ----------
    topology : Topology
        The topology to order.
    station_ids : list[str]
        A list of station ids. The stations will be ordered according to the grid_model_id.

    Returns
    -------
    Topology
        The ordered topology
    list[str]
        A list of station ids that were not found in the topology
    """
    new_stations = []
    not_found = []
    for relevant_node in station_ids:
        found = False
        for station in topology.stations:
            if station.grid_model_id == relevant_node:
                new_stations.append(station)
                found = True
                break
        if not found:
            not_found.append(relevant_node)

    topology = topology.model_copy(update={"stations": new_stations})
    return topology, not_found


def fuse_coupler(
    station: Station,
    coupler_grid_model_id: str,
    copy_info_from: bool = True,
) -> Station:
    """Fuses a coupler by merging the adjacent busbars into one busbar.

    Parameters
    ----------
    station : Station
        The station with the coupler to fuse. Assumes that the coupler is in the station. The open/closed state of the
        coupler is disregarded, i.e. the coupler will be fused regardless of its state.
    coupler_grid_model_id : str
        The grid_model_id of the coupler to fuse
    copy_info_from : bool, optional
        Whether the new busbar retains the information (grid_model_id, etc) of the busbar on the from side of the coupler,
        by default True. If False, the busbar on the to side of the coupler is used.

    Returns
    -------
    Station
        The station with the coupler fused. The coupler is removed and the busbars are merged.
    """
    coupler = next((c for c in station.couplers if c.grid_model_id == coupler_grid_model_id), None)
    if coupler is None:
        raise ValueError(f"Coupler {coupler_grid_model_id} not found in station {station.grid_model_id}")

    busbar_from_index = next((index for index, b in enumerate(station.busbars) if b.int_id == coupler.busbar_from_id), None)
    busbar_to_index = next((index for index, b in enumerate(station.busbars) if b.int_id == coupler.busbar_to_id), None)

    assert busbar_from_index is not None, f"Busbar {coupler.busbar_from_id} not found in station {station.grid_model_id}"
    assert busbar_to_index is not None, f"Busbar {coupler.busbar_to_id} not found in station {station.grid_model_id}"

    if (
        len(
            [
                c
                for c in station.couplers
                if (c.busbar_from_id == coupler.busbar_from_id and c.busbar_to_id == coupler.busbar_to_id)
                or (c.busbar_to_id == coupler.busbar_from_id and c.busbar_from_id == coupler.busbar_to_id)
            ]
        )
        > 1
    ):
        raise ValueError(
            f"Coupler {coupler_grid_model_id} has parallel couplers in station {station.grid_model_id}, "
            "cannot fuse parallel couplers with the same busbars"
        )

    switching_row = station.asset_switching_table[busbar_from_index] | station.asset_switching_table[busbar_to_index]
    connectivity_row = station.asset_connectivity[busbar_from_index] | station.asset_connectivity[busbar_to_index]

    # Remove either the from or the to busbar
    busbar_index_to_remove = busbar_to_index if copy_info_from else busbar_from_index
    busbar_index_to_keep = busbar_from_index if copy_info_from else busbar_to_index

    new_switching_table = np.copy(station.asset_switching_table)
    new_switching_table[busbar_index_to_keep] = switching_row
    new_switching_table = np.delete(new_switching_table, busbar_index_to_remove, axis=0)

    if station.asset_connectivity is not None:
        new_connectivity_table = np.copy(station.asset_connectivity)
        new_connectivity_table[busbar_index_to_keep] = connectivity_row
        new_connectivity_table = np.delete(new_connectivity_table, busbar_index_to_remove, axis=0)
    else:
        new_connectivity_table = None

    busbar_to_remove = station.busbars[busbar_index_to_remove]
    busbar_to_keep = station.busbars[busbar_index_to_keep]

    def _replace_sr_keys(asset: SwitchableAsset) -> SwitchableAsset:
        """Update the sr switch asset if it is connected to the removed busbar."""
        if asset.asset_bay is None:
            return asset
        if (
            busbar_to_remove.grid_model_id in asset.asset_bay.sr_switch_grid_model_id.keys()
            and busbar_to_keep.grid_model_id in asset.asset_bay.sr_switch_grid_model_id.keys()
        ):
            # If both busbars are present, we need to remove the one that is not kept
            new_sr_switch_grid_model_id = {
                key: foreign_id
                for (key, foreign_id) in asset.asset_bay.sr_switch_grid_model_id.items()
                if key != busbar_to_remove.grid_model_id
            }
        else:
            # If the target busbar is not present, we change the key
            new_sr_switch_grid_model_id = {
                (busbar_to_keep.grid_model_id if key == busbar_to_remove.grid_model_id else key): foreign_id
                for (key, foreign_id) in asset.asset_bay.sr_switch_grid_model_id.items()
            }

        return asset.model_copy(
            update={"asset_bay": asset.asset_bay.model_copy(update={"sr_switch_grid_model_id": new_sr_switch_grid_model_id})}
        )

    def _replace_int_id(coupler: BusbarCoupler) -> BusbarCoupler:
        """Update coupler int-ids that are pointing to the removed busbar."""
        if coupler.busbar_from_id == busbar_to_remove.int_id:
            return coupler.model_copy(update={"busbar_from_id": busbar_to_keep.int_id})
        if coupler.busbar_to_id == busbar_to_remove.int_id:
            return coupler.model_copy(update={"busbar_to_id": busbar_to_keep.int_id})
        return coupler

    new_busbars = [b for i, b in enumerate(station.busbars) if i != busbar_index_to_remove]
    new_couplers = [_replace_int_id(c) for c in station.couplers if c.grid_model_id != coupler_grid_model_id]
    new_assets = [_replace_sr_keys(a) for a in station.assets]

    station = station.model_copy(
        update={
            "busbars": new_busbars,
            "asset_switching_table": new_switching_table,
            "asset_connectivity": new_connectivity_table,
            "assets": new_assets,
            "couplers": new_couplers,
        }
    )
    Station.model_validate(station)
    return station


def fuse_all_couplers_with_type(
    station: Station,
    coupler_type: str,
    copy_info_from: bool = True,
) -> tuple[Station, list[BusbarCoupler]]:
    """Fuses all couplers of a specific type in a station.

    This will also filter all duplicate couplers, as there might be an edge case in which a triangle of busbars is not
    properly merged otherwise.

    Parameters
    ----------
    station : Station
        The station with the couplers to fuse
    coupler_type : str
        The type of coupler to fuse, will match the coupler.type attribute. If coupler.type is None, it will never match.
    copy_info_from : bool, optional
        Whether the new busbar retains the information (grid_model_id, etc) of the busbar on the from side of the coupler,
        by default True. If False, the busbar on the to side of the coupler is used.

    Returns
    -------
    Station
        The station with the couplers fused that matched the type. The couplers are removed and the busbars are merged.
    list[BusbarCoupler]
        The couplers that were fused or removed due to being parallel and are no longer present in the station.
    """
    fused_couplers = []
    while True:
        coupler = next((c for c in station.couplers if (c.type is not None and c.type == coupler_type)), None)
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


def find_station_by_id(stations: list[Station], station_id: str) -> Station:
    """Find a station by its grid_model_id in a list of stations.

    Parameters
    ----------
    stations : list[Station]
        The list of stations to search in.
    station_id : str
        The grid_model_id of the station to find.

    Returns
    -------
    Station
        The station with the given grid_model_id.

    Raises
    ------
    ValueError
        If the station is not found in the list.
    """
    for station in stations:
        if station.grid_model_id == station_id:
            return station
    raise ValueError(f"Station {station_id} not found in the list")
