# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides helper functions to realize an assignment into an asset topology

This entire document is deprecated and is only used for keeping busbar tests running. It will be removed in the future.

"""

import jax.numpy as jnp
import logbook
import numpy as np
from beartype.typing import Literal, Optional
from jaxtyping import ArrayLike, Bool
from toop_engine_dc_solver.preprocess.helpers.switching_distance import per_station_switching_distance
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    make_separation_set,
)
from toop_engine_interfaces.asset_topology import Station

logger = logbook.Logger(__name__)


# TODO: A different rule could take precedent here. I would first check for parallel lines/trafos
# and their location. If you have a parallel line/trafo, putting the asset on another busbar should be more
# important than evening out the number of elements.
def determine_asset_assignment(
    target_electrical_busbar: list[int],
    asset_switching_table: Bool[np.ndarray, " n_busbars n_assets"],
) -> int:
    """Determine the busbar to which the asset should be assigned.

    The assets will be assigned to the busbar with the least number of connected assets. If there
    are multiple busbars with the same number of connected assets, the first busbar
    will be selected.

    Parameters
    ----------
    target_electrical_busbar : list of int
        List of busbars to which the assets can be assigned. The integers in the list represent
        the indices of the busbars in the station.
    asset_switching_table : NDArray
        A 2D array where each row corresponds to a busbar and each column corresponds
        to an asset. The value is True if the asset is connected to the busbar, otherwise False.

    Returns
    -------
    int
        Index of the busbar to which the asset should be assigned.
    """
    # get asset distribution on target_electrical_busbar
    num_connected_assets = asset_switching_table[target_electrical_busbar, :].sum(axis=1)
    # get the bubsbar with the least number of connected assets
    busbar_to_assign = target_electrical_busbar[np.argmin(num_connected_assets)]
    return busbar_to_assign


def realize_single_asset_assignment(
    asset_switching_table: Bool[np.ndarray, " n_busbars n_assets"],
    target_assignment: bool,
    busbar_a: list[int],
    busbar_b: list[int],
    asset_index: int,
    asset_connectivity: Optional[Bool[np.ndarray, " n_busbars n_assets"]] = None,
) -> Optional[Bool[np.ndarray, " n_busbars"]]:
    """Realize a single asset assignment to be physically implemented in the station.

    If an asset is on the busbar and should be on the busbar, nothing to do
    If an asset is on the busbar and should not be on the busbar, we have to remove *all* connections
    If an asset is not on the busbar and should be on the busbar, we have to add *one* connection
    If an asset is not on the busbar and should not be on the busbar, nothing to do

    Parameters
    ----------
    asset_switching_table : Bool[np.ndarray, " n_busbars"]
        The current switching state of the asset.
    target_assignment : bool
        The target switching state of the asset, meaning should the asset be electrically connected
        to busbar A (True) or busbar B (False).
    busbar_a : list[int]
        The busbars that are considered busbar A in the configurations table. The integers in the
        list correspond to the indices (not int_ids) of the busbars in the station.
    busbar_b : list[int]
        The busbars that are considered busbar B in the configurations table. The integers in the
        list correspond to the indices (not int_ids) of the busbars in the station.
    asset_index : int
        The index of the asset in the asset_switching_table.
    asset_connectivity : Bool[np.ndarray, " n_busbars n_assets"], optional
        The asset connectivity table. This is required when the function shall respect the feasibility of the realized
        assignment. If True, the function may return None if the realized assignment is not feasible. If False, the function
        will realize the assignment without checking feasibility. By default False.

    Returns
    -------
    Optional[Bool[np.ndarray, " n_busbars"]]
        The realized switching state of the asset. If the assignment is not feasible, None is returned.
    """
    if asset_connectivity is None:
        asset_connectivity = np.ones_like(asset_switching_table, dtype=bool)

    current_asset_switching = asset_switching_table[:, asset_index]
    new_asset_switching = current_asset_switching.copy()

    # Only consider the busbars that the asset can be connected to
    busbar_a = [busbar for busbar in busbar_a if asset_connectivity[busbar, asset_index]]
    busbar_b = [busbar for busbar in busbar_b if asset_connectivity[busbar, asset_index]]

    is_on_a = np.any(current_asset_switching[busbar_a]).item()
    is_on_b = np.any(current_asset_switching[busbar_b]).item()
    should_be_on_a = not target_assignment
    should_be_on_b = target_assignment

    if (not busbar_a and should_be_on_a) or (not busbar_b and should_be_on_b):
        # The target configuration is not feasible
        return None

    def handle_busbar(should_be_on: bool, is_on: bool, busbar: list[int]) -> None:
        """Handle the busbar assignment for the asset.

        Note: The function modifies the new_asset_switching array in place.

        Parameters
        ----------
        should_be_on : bool
            Should the asset be connected to the busbar?
        is_on : bool
            Is the asset connected to the busbar?
        busbar : list[int]
            The busbars that are considered busbar A or B in the configurations table. The integers
            in the list correspond to the int_ids of the busbars in the station.
        """
        if is_on is True and should_be_on is False:
            new_asset_switching[busbar] = False
        elif is_on is False and should_be_on is True:
            # There can be different strategies to assign the asset to the busbar.
            # We chose a simple strategy to assign the asset to the busbar with the least number of connected assets.
            bus_i = determine_asset_assignment(busbar, asset_switching_table)
            new_asset_switching[bus_i] = True

    handle_busbar(should_be_on_a, is_on_a, busbar_a)
    handle_busbar(should_be_on_b, is_on_b, busbar_b)

    return new_asset_switching


def realize_single_station_assignment(
    station: Station,
    configuration_table: Bool[np.ndarray, " n_configurations 2 n_assets"],
    coupler_states: Bool[np.ndarray, " n_configurations n_couplers"],
    busbar_matchings: list[set[int]],
    target_configuration: Bool[np.ndarray, " n_assets"],
    target_ignores: Optional[Bool[np.ndarray, " n_assets"]],
) -> tuple[Optional[Station], list[int], int]:
    """Realize a electric target to be physically implemented in the station.

    This fundamentally involves two steps, at first finding the closest busbar mapping and then
    taking the least steps to implement that mapping. A mapping is basically the assignment which
    physical busbars in the substation will be busbar A and which ones will be busbar B. The options
    are passed in through the configuration_table, and you can obtain those tables through
    make_configurations_table. Using the slower but more precise station_switching_distance the
    routine will find the best configuration to realize the target configuration. Then, a second
    part will diff the current configuration with the target configuration and apply the necessary
    changes to the station.

    Parameters
    ----------
    station : Station
        The station object containing the original asset switching before the optimization. Out-
        of-service assets should be removed from the station before calling this function, as the
        behaviour is undefined.
    configuration_table : Bool[np.ndarray, " n_configurations 2 n_assets"]
        The configuration table of the unsplit station. Will be used to find the closest busbar
        mapping. Make sure to use the numpy not the jax table, i.e. the one where both busbar a and b
        are present.
    coupler_states : Bool[np.ndarray, " n_configurations n_couplers"]
        The coupler states for each configuration in the configuration table.
    busbar_matchings : list[set[int]]
        A list of length n_configurations containing sets, each set contains the busbars that are
        considered busbar A in the configurations table.
    target_configuration : Bool[np.ndarray, " n_assets"]
        The target configuration to be realized. True means an asset is connected to busbar A, False
        means it is connected to busbar B.
    target_ignores : Optional[Bool[np.ndarray, " n_assets"]]
        The target ignores for assets that should be left as-is. True means the asset should be
        ignored, False means it should be considered for reassignment. If none, all assets are
        considered for reassignment.

    Returns
    -------
    Optional[Station]
        The realized station object. If the assignment is not feasible, None is returned.
    list[int]
        The busbar indices that are considered busbar A in the realized configuration.
    int
        The switching distance between the realized configuration and the target configuration.
    """
    assert len(configuration_table.shape) == 3
    assert configuration_table.shape[1] == 2

    current_coupler_state = [c.open for c in station.couplers]

    if not target_configuration.any():
        # the station has to be left unsplit.
        modified_station = station
        busbar_a = [index for index in range(len(station.busbars))]
        reassignment_distance = 0
        return modified_station, busbar_a, reassignment_distance

    # Find the best configuration, i.e. the best way to open the couplers to produce a two way
    # split such that this two way split is as close as possible to the target configuration
    (
        best_configuration,
        invert,
        reassignment_distance,
        _coupler_distance,
    ) = per_station_switching_distance(
        target_configuration=jnp.array(target_configuration),
        current_coupler_state=jnp.array(current_coupler_state),
        separation_set=jnp.array(configuration_table),
        coupler_states=jnp.array(coupler_states),
        ignore_assets=jnp.array(target_ignores) if target_ignores is not None else None,
    )

    # Replace the coupler states with the chosen configuration
    # Also save a diff in case that's interesting somewhere
    new_coupler_state = coupler_states[best_configuration]
    new_couplers = [
        coupler.model_copy(update={"open": bool(state)}) for coupler, state in zip(station.couplers, new_coupler_state)
    ]

    # Find out which busbars are busbar A and B
    busbar_a = busbar_matchings[int(best_configuration)]
    busbar_b = set(index for index in range(len(station.busbars))) - busbar_a
    busbar_a, busbar_b = list(busbar_a), list(busbar_b)

    if invert:
        busbar_a, busbar_b = busbar_b, busbar_a

    # Go through the assets and assign them to the correct busbars
    new_asset_switching_table = station.asset_switching_table.copy()
    for asset_i in range(len(station.assets)):
        if target_ignores is not None and target_ignores[asset_i]:
            # We ignore this asset and leave it as it is
            continue

        new_asset_switching = realize_single_asset_assignment(
            new_asset_switching_table,
            bool(target_configuration[asset_i]),
            busbar_a,
            busbar_b,
            asset_index=asset_i,
            asset_connectivity=station.asset_connectivity,
        )

        # break when the branch action is not feasible
        if new_asset_switching is None:
            return None, busbar_a, int(reassignment_distance)

        new_asset_switching_table[:, asset_i] = new_asset_switching

    new_station = station.model_copy(
        update={
            "asset_switching_table": new_asset_switching_table,
            "couplers": new_couplers,
        }
    )
    Station.model_validate(new_station)

    return new_station, busbar_a, int(reassignment_distance)


def realise_bus_split_single_station(
    branch_ids_local: list[str],
    branch_topology_local: Bool[ArrayLike, " n_branches_at_node"],
    injection_ids_local: list[str],
    injection_topology_local: Bool[np.ndarray, " n_injections_at_node"],
    station: Station,
    missing_element_behavior: Literal["raise", "leave"] = "leave",
) -> tuple[Optional[Station], list[int], int]:
    """Realizes the bus split for a single station.

    In this method, the switching table of the input Station is modified as per input
    branch and injection topology. An outer realize_bus_splits method calls this method
    iteratively to process a list of stations.

    Parameters
    ----------
    branch_ids_local : list of str
        List of branch IDs local to the station.
    branch_topology_local : Bool[np.ndarray, " n_branches_at_node"]
        Boolean array representing the topology of branches at the node.
    injection_ids_local : list of str
        List of injection IDs local to the station.
    injection_topology_local : Bool[np.ndarray, " n_injections_at_node"]
        Boolean array representing the topology of injections at the node.
    station : Station
        The station object containing assets to be processed.
    missing_element_behavior : {'raise', 'leave'}
        Behavior when an asset is missing in the local IDs. If 'raise', an error is raised.
        If 'leave', the asset is ignored and left as it is.

    Returns
    -------
    Optional[Station]
        A new station object with the updated asset topology based on the provided branch
        and injection topologies. If the assignment is not feasible, None is returned.
    list[int]
        The busbar indices that are considered busbar A in the realized configuration.

    int
        The switching distance between the realized configuration and the target configuration.

    Raises
    ------
    ValueError
        If an asset in the station is not found in either the branch or injection local IDs
        and `missing_element_behavior` is set to 'raise'.
    """
    assert len(branch_ids_local) == len(branch_topology_local)
    assert len(injection_ids_local) == len(injection_topology_local)

    asset_assignment = []
    ignore_assets = []
    for asset in station.assets:
        # Find the asset in either the branches or injections to map from the separate branch
        # and injection topologies to a unified asset topology
        if asset.grid_model_id in branch_ids_local:
            asset_assignment.append(branch_topology_local[branch_ids_local.index(asset.grid_model_id)])
            ignore_assets.append(False)
        elif asset.grid_model_id in injection_ids_local:
            asset_assignment.append(injection_topology_local[injection_ids_local.index(asset.grid_model_id)])
            ignore_assets.append(False)
        elif missing_element_behavior == "leave":
            # We want to ignore this asset and leave it as it is
            # Hence we set ignore to true and the assignment to any value
            ignore_assets.append(True)
            asset_assignment.append(False)
        else:
            raise ValueError(f"Could not find asset {asset.grid_model_id} in the station {station.grid_model_id}")

    configuration_table, coupler_states, busbar_matchings = make_separation_set(station)

    new_station, busbar_a, reassignment_distance = realize_single_station_assignment(
        station=station,
        configuration_table=configuration_table,
        coupler_states=coupler_states,
        busbar_matchings=busbar_matchings,
        target_configuration=np.array(asset_assignment),
        target_ignores=np.array(ignore_assets),
    )

    return new_station, busbar_a, reassignment_distance
