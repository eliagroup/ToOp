# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains preprocessing routines to be able to compute the switching distance.

The switching distance is evaluated between a topology and a reference topology.
"""

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import logbook
import networkx as nx
import numpy as np
from beartype.typing import Literal, NamedTuple, Optional, Sequence
from jaxtyping import Array, Bool, Int
from networkx.algorithms.components import (
    connected_components,
    number_connected_components,
)
from toop_engine_dc_solver.preprocess.helpers.switching_distance import hamming_distance
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    Station,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import (
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    fix_multi_connected_without_coupler,
    fuse_all_couplers_with_type,
    order_station_assets,
    order_topology,
)

logger = logbook.Logger(__name__)


class OptimalSeparationSetInfo(NamedTuple):
    """Tuple that holds information about the possible 2-node separations in a Station"""

    separation_set: Bool[np.ndarray, " n_configurations 2 n_assets"]
    """The separation set of busbars in the station. Each row corresponds to a possible two-way split of
    the station obtained by opening some couplers. This is the optimized table, i.e. equivalent
    configurations have been purged and the table is in the format where busbar A and B are joined,
    i.e. B is the inverse of A."""

    coupler_states: Bool[np.ndarray, " n_configurations n_couplers"]
    """Bool[np.ndarray, " n_configurations n_couplers"]
    A table of coupler states. Each row corresponds to a configuration, and each column
    corresponds to a coupler. The value in the table is True if the coupler is open, and False
    if it is closed."""

    coupler_distance: Int[Array, " n_configurations"]
    """The hamming distance between the coupler states of the station and the coupler states of the
    configuration in number of switches changed."""

    busbar_a: list[set[int]]
    """A list of length n_configurations containing sets, each set contains the busbars that are
    considered busbar A in the configurations table. The integers in the set correspond to the
    int_ids of the busbars in the station."""


def make_separation_set(
    station: Station,
) -> tuple[
    Bool[np.ndarray, " n_configurations 2 n_assets"],
    Bool[np.ndarray, " n_configurations n_couplers"],
    list[set[int]],
]:
    """Translate a switching table from the station to a table of possible electrical two-way configurations.

    This entails taking each possible separation of physical busbars to two electrical busbars and
    enumerating the resulting electrical configurations.

    TODO there is a bug where a separation might not allow an asset to be connected to both busbars A and B but only one of
    them. This should be caught somehow.

    Parameters
    ----------
    station : Station
        The station object containing the switching table. This assumes that the station has
        undergone the following preprocessing steps:
        - Out-of-service assets have been removed from the station using
            asset_topology.filter_out_of_service
        - Duplicate couplers have been removed from the station using
            asset_topology.filter_duplicate_couplers
        - Disconnected busbars have been removed from the station using
            asset_topology.filter_disconnected_busbars
        - Multi-connected assets have a coupler parallel to them, you can use
            asset_topology.create_couplers_for_multi_connected_assets to create these couplers.
            Alternatively, you can convert multi-connected assets to single-connected assets using
            asset_topology.select_one_for_multi_connected_assets.

    Returns
    -------
    Bool[np.ndarray, " n_configurations 2 n_assets"]
        A table of electrical configurations. Each row corresponds to a configuration, and each
        column corresponds to an asset. The second dimension represents busbar A and B. A true value
        means that for this configuration, on this busbar, the asset is connected.
    Bool[np.ndarray, " n_configurations n_couplers"]
        A table of coupler states. Each row corresponds to a configuration, and each column
        corresponds to a coupler. The value in the table is True if the coupler is open, and False
        if it is closed.
    list[set[int]]
        A list of length n_configurations containing sets, each set contains the busbars that are
        considered busbar A in the configurations table. This is represented as indices into the
        list of busbars, not using the busbar int_ids.

    Raises
    ------
    ValueError
        If the number of couplers in the station is larger than 20. This is an arbitrary limit to
        prevent excessive computation time.
    """
    assert all(busbar.in_service for busbar in station.busbars)
    assert all(coupler.in_service for coupler in station.couplers)
    assert all(asset.in_service for asset in station.assets)
    if not len(station.couplers):
        return (np.zeros((0, 2, len(station.assets)), dtype=bool), np.zeros((0, 0), dtype=bool), [])
    # Go through all possible coupler assignments and find all the configurations in which there are
    # exactly two electrical busbars.
    switching_map = {
        busbar_index: station.asset_switching_table[busbar_index] for busbar_index, _ in enumerate(station.busbars)
    }
    busbar_int_id_map = {busbar.int_id: busbar_index for busbar_index, busbar in enumerate(station.busbars)}
    configurations = []
    coupler_configurations = []
    busbar_matchings = []

    if len(station.couplers) > 20:
        raise ValueError(
            f"Unrealistic number of couplers ({len(station.couplers)}) found in the station {station.grid_model_id}"
        )

    for coupler_open in itertools.product([True, False], repeat=len(station.couplers)):
        graph = nx.Graph()
        graph.add_nodes_from(list(range(len(station.busbars))))
        graph.add_edges_from(
            [
                (busbar_int_id_map[coupler.busbar_from_id], busbar_int_id_map[coupler.busbar_to_id])
                for coupler, open in zip(station.couplers, coupler_open, strict=True)
                if not open
            ]
        )

        # If all couplers are closed, we can check if the substation was fully connected originally
        # This is required for this function to make sense
        if not any(coupler_open):
            assert number_connected_components(graph) == 1, (
                "Some busbars are isolated in the station - has the station been simplified?"
            )
            continue

        components = list(connected_components(graph))
        if len(components) != 2:
            continue

        # Extract the switching states for the two busbars
        # We can't say in advance which of the two busbars will be closer to an eventual electrical
        # switching candidate, hence we have to store both configurations
        busbar_a_switchings = np.any(
            np.stack([switching_map[busbar_index] for busbar_index in components[0]], axis=0),
            axis=0,
        )
        busbar_b_switchings = np.any(
            np.stack([switching_map[busbar_index] for busbar_index in components[1]], axis=0),
            axis=0,
        )

        configurations.append(np.stack([busbar_a_switchings, busbar_b_switchings], axis=0))
        coupler_configurations.append(np.array(coupler_open))
        busbar_matchings.append(components[0])

    return (
        np.stack(configurations, axis=0),
        np.stack(coupler_configurations, axis=0),
        busbar_matchings,
    )


def identify_unnecessary_configurations(
    configurations: Bool[np.ndarray, " n_configurations n_assets"],
    clip_hamming_distance: int = 0,
) -> Bool[np.ndarray, " n_configurations"]:
    """Identify configurations that are equivalent to others.

    Two configurations are considered equivalent if their hamming distance is less or equal to the
    clip_hamming_distance. This function will return a mask of the configurations that are to be
    kept.

    Parameters
    ----------
    configurations : Bool[np.ndarray, " n_configurations n_assets"]
        The configurations table to be filtered.
    clip_hamming_distance : int, optional
        The maximum hamming distance between two configurations for them to be considered
        equivalent, by default 0 (exact match).

    Returns
    -------
    Bool[np.ndarray, " n_configurations"]
        A mask of the configurations that are to be kept.
    """
    assert len(configurations.shape) == 2, "The configurations should be reshaped to two dimensions"
    # Compute the hamming distance between all pairs of configurations
    hamming_distances = np.sum(np.logical_xor(configurations[:, None], configurations[None, :]), axis=-1)
    # Also compute the hamming distance to their inverse
    hamming_distances_inv = np.sum(np.logical_xor(configurations[:, None], ~configurations[None, :]), axis=-1)
    hamming_distances = np.minimum(hamming_distances, hamming_distances_inv)

    # Put the diagonal to a value higher than the clip_hamming_distance so we don't exclude
    # a configuration based on the distance to itself
    np.fill_diagonal(hamming_distances, clip_hamming_distance + 1)

    # Find the configurations that are equivalent to others
    # Make sure to keep at least one configuration for each equivalence class
    mask = np.ones(len(configurations), dtype=bool)
    for i in range(len(configurations)):
        if mask[i]:
            mask = mask & (hamming_distances[i] > clip_hamming_distance)

    return mask


def match_topology_to_network_data(
    topology: Topology,
    branches_at_nodes: list[Int[np.ndarray, " n_branches_at_node"]],
    injections_at_nodes: list[Int[np.ndarray, " n_injections_at_node"]],
    branch_ids: list[str],
    injection_ids: list[str],
    relevant_node_ids: list[str],
    filter_assets: Literal["branch", "injection", "both"] = "both",
) -> Topology:
    """Match a topology to the grid in the network data

    This function will order the station in the topology according to the relevant_node_ids and
    order the assets in the stations according to the branch_ids and injection_ids.

    If stations or assets are missing, it will raise an error.

    Parameters
    ----------
    topology : Topology
        The topology to match, coming from the import process
    branches_at_nodes : list[Int[np.ndarray, " n_branches_at_node"]]
        The branch indices at each node in the network data, indexing into branch_ids
    injections_at_nodes : list[Int[np.ndarray, " n_injections_at_node"]]
        The injection indices at each node in the network data, indexing into injection_ids
    branch_ids : list[str]
        The branch ids in the network data, matching the grid_model_id of the assets in the station
    injection_ids : list[str]
        The injection ids in the network data, matching the grid_model_id of the assets in the station
    relevant_node_ids : list[str]
        The node ids in the network data, matching the grid_model_id of the stations in the topology
    filter_assets : Literal["branch", "injection", "both"], optional
        Whether to return all assets, only branches, or only injections. By default "both"

    Returns
    -------
    Topology
        The topology with the stations and assets ordered according to the network data
    """
    # Order the stations according to the relevant nodes
    topology, not_found = order_topology(topology, relevant_node_ids)
    if not_found:
        raise ValueError(f"The following nodes were not found in the topology: {not_found}")

    keep_branches = filter_assets in ["branch", "both"]
    keep_injections = filter_assets in ["injection", "both"]

    # Order the assets in the stations according to the branch and injection ids
    new_stations = []
    for station, branches_at_node, injections_at_node in zip(
        topology.stations, branches_at_nodes, injections_at_nodes, strict=True
    ):
        branch_ids_local = [branch_ids[i] for i in branches_at_node] if keep_branches else []
        injection_ids_local = [injection_ids[i] for i in injections_at_node] if keep_injections else []
        new_station, not_found, _ignored = order_station_assets(station, branch_ids_local + injection_ids_local)
        if not_found:
            raise ValueError(f"The following assets were not found in the station {station.grid_model_id}: {not_found}")
        new_stations.append(new_station)

    return topology.model_copy(update={"stations": new_stations})


@dataclass
class StationProblems:
    """Holds the potential non-fatal problems of preprocessing a station in preprocess_station."""

    assets_ignored: Optional[list[str]] = None
    """Assets that were in the station but not in the network data."""

    duplicate_couplers: Optional[list[BusbarCoupler]] = None
    """There were duplicate couplers found in the station."""

    disconnected_busbars: Optional[list[Busbar]] = None
    """There were disconnected busbars found in the station."""

    multi_connected_assets: Optional[list[tuple[SwitchableAsset, Busbar, Busbar]]] = None
    """There were multi-connected assets without a coupler found in the station."""

    def __bool__(self) -> bool:
        """Check if any of the problems are present.

        Returns
        -------
        bool
            True if any of the problems are present, False otherwise.
        """
        return any(getattr(self, field) is not None for field in StationProblems.__annotations__)


def prepare_for_separation_set(
    station: Station, branch_ids: list[str], injection_ids: list[str], close_couplers: bool = False
) -> tuple[Station, StationProblems]:
    """Prepare a Station so it can be used in make_separation_set.

    This function will:
    - Close all open couplers if close_couplers is True
    - Order the assets in the station according to the branch_ids and injection_ids. The convention is to put the branches
    first and then the injections.
    - Remove out-of-service assets
    - Remove duplicate couplers
    - Fuse couplers of type DISCONNECTOR
    - Remove disconnected busbars
    - Select an arbitraty bus for multi-connected assets without a coupler

    Parameters
    ----------
    station : Station
        The station to prepare
    branch_ids : list[str]
        The branch ids to order the assets by
    injection_ids : list[str]
        The injection ids to order the assets by
    close_couplers : bool, optional
        Whether to close all open couplers, by default True.

    Returns
    -------
    Station
        The prepared station
    StationProblems
        A dataclass containing the problems that were fixed in the station.
    """
    if close_couplers:
        station = station.model_copy(
            update={"couplers": [coupler.model_copy(update={"open": False}) for coupler in station.couplers]}
        )
    station, not_found, ignored = order_station_assets(station, branch_ids + injection_ids)
    if not_found:
        raise ValueError(
            f"The following assets were not found in the station {station.grid_model_id}/{station.name}: "
            f"{not_found} - this station can not be switched."
        )

    station = filter_out_of_service(station)
    station, disconnected_busbars = filter_disconnected_busbars(station, respect_coupler_open=True)
    station, duplicate_couplers = filter_duplicate_couplers(station, retain_type_hierarchy=["DISCONNECTOR", "BREAKER"])
    station, _fused_couplers = fuse_all_couplers_with_type(station, coupler_type="DISCONNECTOR")
    station, fixed_assets = fix_multi_connected_without_coupler(station)

    if not station.couplers:
        raise ValueError(
            f"Station {station.grid_model_id}/{station.name} has no couplers left after preprocessing. "
            "this station can not be switched.."
        )

    problems = StationProblems(
        duplicate_couplers=duplicate_couplers if duplicate_couplers else None,
        disconnected_busbars=disconnected_busbars if disconnected_busbars else None,
        multi_connected_assets=fixed_assets if fixed_assets else None,
        assets_ignored=ignored if ignored else None,
    )

    if problems:
        station = station.model_copy(
            update={"model_log": (station.model_log or []) + [f"Problems during simplification: {problems}"]}
        )

    Station.model_validate(station)

    return station, problems


def make_optimal_separation_set(
    station: Station,
    clip_hamming_distance: int = 0,
    clip_at_size: int = 100,
) -> OptimalSeparationSetInfo:
    """Build the configurations table for a station and optimize it using clipping and hamming distance.

    This function will:
    - Create the busbar configuration table using make_separation_set
    - Simplify the configuration table using identify_unnecessary_configurations

    Parameters
    ----------
    station : Station
        The station to preprocess. It is assumed that prepare_for_separation_set has been called on the
        station.
    clip_hamming_distance : int, optional
        If a large configuration table comes out of a substation, the table size can be reduced
        by removing configurations that are close to each other. This parameter sets the definition
        of close in terms of hamming distance, by default 0 (no reduction).
    clip_at_size : int, optional
        By what size a table is considered large. If the table is larger than this size, the
        clip_hamming_distance will be used to reduce the table size, by default 100. If a table is
        smaller, no reduction will be applied.

    Returns
    -------
    OptimalSeparationSetInfo
        A tuple containing the optimized separation set information.
        The separation_set itself, the coupler states, the coupler distances and the busbar A matchings.
    """
    configuration_table, coupler_states, busbar_matchings = make_separation_set(station)
    clip_hamming_distance = 0 if configuration_table.shape[0] < clip_at_size else clip_hamming_distance
    config_mask = identify_unnecessary_configurations(configuration_table[:, 0, :], clip_hamming_distance)
    configuration_table = configuration_table[config_mask]
    coupler_states = coupler_states[config_mask]
    busbar_matchings = [x for (x, mask) in zip(busbar_matchings, config_mask, strict=True) if mask]

    coupler_distances = hamming_distance(
        jnp.array(coupler_states), jnp.array([coupler.open for coupler in station.couplers], dtype=bool)
    )

    return OptimalSeparationSetInfo(
        separation_set=configuration_table,
        coupler_states=coupler_states,
        coupler_distance=coupler_distances,
        busbar_a=busbar_matchings,
    )


def pad_configurations_table(
    configuration_table: Bool[Array, " n_configurations n_assets"],
    coupler_distances: Int[Array, " n_configurations"],
    ignore_assets: Bool[Array, " n_assets"],
    max_n_assets: int,
    max_n_configurations: int,
) -> tuple[
    Bool[Array, " max_n_configurations max_n_assets"],
    Int[Array, " max_n_configurations"],
    Bool[Array, " max_n_assets"],
]:
    """Pad a configuration table to a fixed size.

    The values are padded with zeros along the asset dimension, but are repeated along the
    configuration dimension.

    Parameters
    ----------
    configuration_table : Bool[Array, " n_configurations n_assets"]
        The configuration table to pad.
    coupler_distances : Int[Array, " n_configurations"]
        The coupler distances to pad.
    ignore_assets : Bool[Array, " n_assets"]
        Which assets shall not be counted in the hamming distance. True to ignore.
    max_n_assets : int
        The maximum number of assets in the table.
    max_n_configurations : int
        The maximum number of configurations in the table.

    Returns
    -------
    Bool[Array, " max_n_configurations max_n_assets"]
        The padded configuration table, padded with False along the asset dimension and repeated
        along the configuration dimension.
    Int[Array, " max_n_configurations"]
        The padded coupler distances, repeated the same way as the configuration table.
    Bool[Array, " max_n_assets"]
        The padded ignore assets, padded with True
    """
    configuration_table = jnp.pad(
        configuration_table,
        (
            (0, 0),
            (0, max_n_assets - configuration_table.shape[1]),
        ),
        mode="constant",
        constant_values=False,
    )
    configuration_table = jnp.pad(
        configuration_table,
        (
            (0, max_n_configurations - configuration_table.shape[0]),
            (0, 0),
        ),
        mode="edge",
    )

    coupler_distances = jnp.pad(
        coupler_distances,
        (0, max_n_configurations - coupler_distances.shape[0]),
        mode="edge",
    )

    ignore_assets = jnp.pad(
        ignore_assets,
        (0, max_n_assets - ignore_assets.shape[0]),
        mode="constant",
        constant_values=True,
    )

    return (
        configuration_table,
        coupler_distances,
        ignore_assets,
    )


def add_missing_asset_topology_branch_info(
    asset_topology: Topology,
    branch_ids: Sequence[str],
    branch_names: Sequence[str],
    branch_types: Sequence[str],
    branch_from_nodes: Sequence[str],
    overwrite_if_present: bool = True,
) -> Topology:
    """Add name, type and direction info to the asset topology

    These fields are optional and are not necessarily present in the asset topology. This function
    will update the topology to have these fields. If the fields are already present, they will be
    overwritten if overwrite_if_present is True.

    Parameters
    ----------
    asset_topology : Topology
        The asset topology to update
    branch_ids : Sequence[str]
        The branch ids of the branches in the grid, should match the grid_model_id in the assets
    branch_names : Sequence[str]
        The names of the branches, should have the same length as branch_ids
    branch_types : Sequence[str]
        The types of the branches, should have the same length as branch_ids
    branch_from_nodes : Sequence[str]
        The from node grid_model_id of each branch, should have the same length as branch_ids. If
        the station name is equal to the entry in from_nodes, the branch-end will be saved as "from"
        If it's not found, the branch-end will be saved as "to" without checking the to_nodes
    overwrite_if_present : bool, optional
        Whether to overwrite the fields if they are already present, by default True

    Returns
    -------
    Topology
        The updated asset topology
    """
    # Faster lookup of the position of the branch
    branch_id_lookup = {branch_id: i for i, branch_id in enumerate(branch_ids)}

    new_stations = []
    for station in asset_topology.stations:
        new_assets = []
        for asset in station.assets:
            index = branch_id_lookup.get(asset.grid_model_id, None)
            if index is not None:
                branch_end = "from" if branch_from_nodes[index] == station.grid_model_id else "to"
                new_assets.append(
                    asset.model_copy(
                        update={
                            "name": (branch_names[index] if overwrite_if_present or asset.name is None else asset.name),
                            "type": (branch_types[index] if overwrite_if_present or asset.type is None else asset.type),
                            "branch_end": (
                                branch_end if overwrite_if_present or asset.branch_end is None else asset.branch_end
                            ),
                        }
                    )
                )
            else:
                new_assets.append(asset)
        new_stations.append(station.model_copy(update={"assets": new_assets}))

    return asset_topology.model_copy(update={"stations": new_stations})


def add_missing_asset_topology_injection_info(
    asset_topology: Topology,
    injection_ids: Sequence[str],
    injection_names: Sequence[str],
    injection_types: Sequence[str],
    overwrite_if_present: bool = True,
) -> Topology:
    """Add name info to the asset topology

    These fields are optional and are not necessarily present in the asset topology. This function
    will update the topology to have these fields. If the fields are already present, they will be
    overwritten if overwrite_if_present is True.

    Parameters
    ----------
    asset_topology : Topology
        The asset topology to update
    injection_ids : Sequence[str]
        The injection ids of the injections in the grid, should match the grid_model_id in the assets
    injection_names : Sequence[str]
        The names of the injections, should have the same length as injection_ids
    injection_types : Sequence[str]
        The types of the injections, should have the same length as injection_ids
    overwrite_if_present : bool, optional
        Whether to overwrite the fields if they are already present, by default True

    Returns
    -------
    Topology
        The updated asset topology
    """
    # Faster lookup of the position of the injection
    injection_id_lookup = {injection_id: i for i, injection_id in enumerate(injection_ids)}

    new_stations = []
    for station in asset_topology.stations:
        new_assets = []
        for asset in station.assets:
            index = injection_id_lookup.get(asset.grid_model_id, None)
            if index is not None:
                new_assets.append(
                    asset.model_copy(
                        update={
                            "name": (injection_names[index] if overwrite_if_present or asset.name is None else asset.name),
                            "type": (injection_types[index] if overwrite_if_present or asset.type is None else asset.type),
                        }
                    )
                )
            else:
                new_assets.append(asset)
        new_stations.append(station.model_copy(update={"assets": new_assets}))

    return asset_topology.model_copy(update={"stations": new_stations})
