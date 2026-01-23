# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains functions to realise branch actions to physical topology"""

from dataclasses import replace

import logbook
import numpy as np
from beartype.typing import Literal
from jaxtyping import Float
from toop_engine_dc_solver.postprocess.realize_assignment import (
    realise_ba_to_physical_topo_per_station_jax,
)
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    get_relevant_stations,
)
from toop_engine_interfaces.asset_topology import Station
from toop_engine_interfaces.asset_topology_helpers import get_connected_assets

logger = logbook.Logger(__name__)


def enumerate_station_realisations(
    network_data: NetworkData,
    choice_heuristic: Literal["first", "least_connected_busbar", "most_connected_busbar"] = "least_connected_busbar",
) -> NetworkData:
    """Find a physical station realization for every branch action in the branch action set.

    This method enumerates the following as per the branch_action_set:
    1. Realisations of stations in the network data
    2. Updates the branch action set with infeasible actions removed
    3. Busbar A mappings. This is a list of physical busbars that are mapped to electrical busbar A.
    This will be used to spread the injections to different busbars.
    4. Switching distance: The number of reassignments needed to reach the target configuration while
    taking into account only the branch_actions.

    Parameters
    ----------
    network_data : NetworkData
        An instance of NetworkData containing the asset topology and branch action set.
    choice_heuristic : Literal["first", "least_connected_busbar", "most_connected_busbar"]
        A heuristic to choose the busbar for the action realization.

    Returns
    -------
    NetworkData
        The updated NetworkData instance with realized stations and updated branch action set.


    Raises
    ------
    AssertionError
        If `network_data.simplified_asset_topology`, `network_data.branch_action_set`
        or `network_data.separation_sets_info` is not provided.

    Notes
    -----
    There can be certain branch_actions which are not feasible due to grid constraints. Such constraints are documented
    in the asset_connectivity matrix of the Station object. This function iterates over the branch action set and
    attempts to realize each action on the given station. If an action is not feasible due to grid constraints, it is
    removed from the action set.
    """
    assert network_data.simplified_asset_topology is not None, "Simplified asset topology is not provided"
    assert network_data.branch_action_set is not None, "Branch action set is not provided"
    assert network_data.separation_sets_info is not None, "Separation set info is not provided, please compute it first"
    branch_action_set = network_data.branch_action_set.copy()
    all_rel_realised_stations = []
    all_rel_subs_busbar_a_mappings = []
    all_rel_subs_reassignment_distances = []

    for index, (station, local_branch_action_set, separation_set_info) in enumerate(
        zip(
            network_data.simplified_asset_topology.stations,
            branch_action_set,
            network_data.separation_sets_info,
            strict=True,
        )
    ):
        (realised_stations, local_updated_branch_action_set, local_busbar_a_mappings, local_reassignment_distances) = (
            realise_ba_to_physical_topo_per_station_jax(
                local_branch_action_set=local_branch_action_set,
                station=station,
                separation_set_info=separation_set_info,
                choice_heuristic=choice_heuristic,
                validate=True,
            )
        )
        all_rel_realised_stations.append(realised_stations)
        all_rel_subs_busbar_a_mappings.append(local_busbar_a_mappings)
        all_rel_subs_reassignment_distances.append(np.array(local_reassignment_distances, dtype=int))
        if not np.array_equal(local_branch_action_set, local_updated_branch_action_set):
            branch_action_set[index] = local_updated_branch_action_set

    network_data = replace(
        network_data,
        branch_action_set=branch_action_set,
        realised_stations=all_rel_realised_stations,
        busbar_a_mappings=all_rel_subs_busbar_a_mappings,
        branch_action_set_switching_distance=all_rel_subs_reassignment_distances,
    )

    return network_data


def get_injections_on_physical_bb(
    network_data: NetworkData, sub: Station, busbar_index: int
) -> Float[np.ndarray, " n_timesteps"]:
    """Calculate the total connected injections in megawatts (MW) to a given physical busbar inside a substation.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing MW injections and injection IDs.
    sub : Station
        The substation object containing assets.
    busbar_index : int
        The index of the busbar within the substation.

    Returns
    -------
    Float[np.ndarray, " n_timesteps"]
        The total connected injections in MW for the specified busbar index over all timesteps.
    """
    connected_assets = get_connected_assets(sub, busbar_index)
    connected_injection_ids = [
        asset.grid_model_id for asset in connected_assets if asset.in_service and not asset.is_branch()
    ]

    # Certain IDs in the connected_injection_ids may not be present in the network_data.injection_ids.
    # TODO: FInd out why?
    connected_injections_mw = [
        network_data.mw_injections[:, network_data.injection_ids.index(id)]
        for id in connected_injection_ids
        if id in network_data.injection_ids
    ]
    if len(connected_injections_mw) > 0:
        total_injection = np.sum(connected_injections_mw, axis=0)
        return total_injection
    return np.zeros(network_data.mw_injections.shape[0])


def get_injections_on_electrical_busbar(
    network_data: NetworkData, sub: Station, busbar_mapping: list[int]
) -> Float[np.ndarray, " n_timesteps"]:
    """Calculate the total injections on an electrical busbar.

    This is equal to the sum of injections on all the physical busbars mapped to the electrical busbar.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing MW injections.
    sub : Station
        The station object representing the substation.
    busbar_mapping : list[int]
        A list of busbar indices to map the injections.

    Returns
    -------
    Float[np.ndarray, " n_timesteps"]
        An array of total injections over the specified timesteps connected to the input electrical busbar.
    """
    total_injections = np.zeros(network_data.mw_injections.shape[0])
    for busbar_index in busbar_mapping:
        total_injections += get_injections_on_physical_bb(network_data, sub, busbar_index)
    return total_injections


def enumerate_spreaded_nodal_injections_for_rel_subs(
    network_data: NetworkData,
) -> list[list[Float[np.ndarray, " 2 n_timesteps"]]]:
    """Get the nodal_injections for the relevant nodes in the network corresponding to the busbar_A_mappings.

    This function is used as a sanity check for testing purposes.

    In the initial nodal_injections vector, all the injections are assigned to busbar_a.
    This function spreads the injections to different busbars as per the busbar_a_mappings.
    For example, if in a station with three busbars, the busbar_a_mappings =[2]
    and there are three physical busbars with the following injections (in MW):
    busbar_1 = [10, 3]
    busbar_2 = [-5]
    busbar_3 = [2],
    The default nodal_injections vector has [10] for busbar_a and [0] for busbar_b.
    After spreading, the nodal_injections vector for busbar_a and busbar_b, will be:
    busbar_a = [-5]
    busbar_b = [15]

    Parameters
    ----------
    network_data : NetworkData
        The network data containing busbar mappings and other relevant information.

    Returns
    -------
    list[list[Float[np.ndarray, " 2 n_timesteps"]]]
        The nodal injection combinations

    Raises
    ------
    AssertionError
        If busbar_a_mappings is not provided in the network_data.

    Note
    ------
    This function is used in tests to check if the nodal injections are spreaded correctly.

    """
    assert network_data.busbar_a_mappings is not None, "Busbar A mappings are not provided"

    rel_stations = get_relevant_stations(network_data)
    rel_subs_busbar_a_mappings = network_data.busbar_a_mappings

    nodal_injection_combis = []
    for sub, local_busbar_a_mappings in zip(rel_stations, rel_subs_busbar_a_mappings, strict=True):
        busbar_indices = set(index for index in range(len(sub.busbars)))
        local_injection_combis = []
        for busbar_a_mappings in local_busbar_a_mappings:
            busbar_b_mappings = list(busbar_indices - set(busbar_a_mappings))
            busbar_a_injections = get_injections_on_electrical_busbar(network_data, sub, busbar_a_mappings)
            busbar_b_injections = get_injections_on_electrical_busbar(network_data, sub, busbar_b_mappings)
            local_injection_combis.append(np.vstack((busbar_a_injections, busbar_b_injections)))
        nodal_injection_combis.append(local_injection_combis)

    return nodal_injection_combis
