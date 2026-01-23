# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess import compute_separaration_set_for_stations
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import (
    enumerate_spreaded_nodal_injections_for_rel_subs,
    enumerate_station_realisations,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import ReassignmentLimits


def test_enumerate_station_realisations(network_data_test_grid: NetworkData):
    network_data = compute_separaration_set_for_stations(network_data_test_grid)
    network_data = enumerate_station_realisations(network_data)
    assert len(network_data.realised_stations) == len(network_data.branch_action_set), (
        "The number of realised stations should be equal to the number of branch action sets. They equal to the number of relevant stations."
    )
    assert np.all(
        [
            len(network_data.realised_stations[i]) == network_data.branch_action_set[i].shape[0]
            for i in range(len(network_data.relevant_nodes))
        ]
    ), "There should be as many realisations of a particular station as there are branch action sets for that station."


def test_enumerate_station_realisations_limit_physical_reassignments(network_data_test_grid: NetworkData):
    network_data = compute_separation_set_for_stations(network_data_test_grid)
    network_data_1 = enumerate_station_realisations(network_data, reassignment_limits=ReassignmentLimits(global_limit=2))

    assert network_data_1.branch_action_set is not None, "Branch action set should not be None"
    assert network_data_1.branch_action_set_switching_distance is not None, (
        "Branch action set switching distance should not be None"
    )
    assert np.all(np.array(network_data_1.branch_action_set_switching_distance) <= 2), (
        "All switching distances should be less than or equal to the global limit of 2"
    )
    assert len(network_data_1.branch_action_set) == len(network_data_1.branch_action_set_switching_distance), (
        "Branch action set length should match the length of switching distances"
    )
    assert len(network_data_1.realised_stations) == len(network_data_1.branch_action_set), (
        "The number of realised stations should be equal to the number of branch action sets. They equal to the number of relevant stations."
    )
    assert np.all(
        [
            len(network_data_1.realised_stations[i]) == network_data_1.branch_action_set[i].shape[0]
            for i in range(len(network_data_1.relevant_nodes))
        ]
    ), "There should be as many realisations of a particular station as there are branch action sets for that station."

    # Make sure that setting a station-specific limit works
    relevant_ids = np.array(network_data_1.node_ids)[network_data_1.relevant_node_mask]
    limit_for_first_sub = 1
    assert np.any(network_data.branch_action_set_switching_distance[0].max() > limit_for_first_sub), (
        "At least one action for the first sub should have a switching distance greater than limit_for_first_sub"
    )
    network_data_2 = enumerate_station_realisations(
        network_data, reassignment_limits=ReassignmentLimits(station_specific_limits={relevant_ids[0]: limit_for_first_sub})
    )
    assert np.all(np.array(network_data_2.branch_action_set_switching_distance) <= limit_for_first_sub), (
        "The switching distances for the first sub should now be less than or equal to limit_for_first_sub"
    )

    # Make sure station specific limits override global limits
    limit_for_first_sub = 2
    global_limit = 1
    network_data_3 = enumerate_station_realisations(
        network_data,
        reassignment_limits=ReassignmentLimits(
            global_limit=global_limit, station_specific_limits={relevant_ids[0]: limit_for_first_sub}
        ),
    )
    assert np.all(np.array(network_data_3.branch_action_set_switching_distance[0]) <= limit_for_first_sub), (
        "The switching distances for the first sub should be less than or equal to limit_for_first_sub"
    )
    assert np.any(np.array(network_data_3.branch_action_set_switching_distance[0]) > global_limit), (
        "At least one action for the first sub should have a switching distance greater than the global limit"
    )


def test_enumerate_station_realisations_oberrhein(network_data_preprocessed: NetworkData):
    # network_data = enumerate_station_realisations(network_data_preprocessed)
    network_data = network_data_preprocessed
    assert len(network_data.realised_stations) == len(network_data.branch_action_set), (
        "The number of realised stations should be equal to the number of branch action sets. They equal to the number of relevant stations."
    )
    assert np.all(
        [
            len(network_data.realised_stations[i]) == network_data.branch_action_set[i].shape[0]
            for i in range(len(network_data.relevant_nodes))
        ]
    ), "There should be as many realisations of a particular station as there are branch action sets for that station."


def test_enumerate_spreaded_nodal_injections_for_rel_subs(network_data_preprocessed: NetworkData):
    nodal_injection_combis = enumerate_spreaded_nodal_injections_for_rel_subs(network_data_preprocessed)

    assert len(nodal_injection_combis) == len(network_data_preprocessed.relevant_nodes), (
        "The length of nodal_injection_combis = len of relevant nodes."
    )
    for nodal_index, local_nodal_injection_combis in enumerate(nodal_injection_combis):
        assert len(local_nodal_injection_combis) == len(network_data_preprocessed.branch_action_set[nodal_index]), (
            "The number of injection combis should be equal to the number of branch action sets for the station."
        )
        local_nodal_injection = network_data_preprocessed.nodal_injection[
            :, network_data_preprocessed.relevant_nodes[nodal_index]
        ]
        for nodal_injection in local_nodal_injection_combis:
            assert nodal_injection.shape[0] == 2, (
                "each injection combi should have two elements, one for busbar_a and one for busbar_b."
            )
            assert nodal_injection.shape[1] == network_data_preprocessed.mw_injections.shape[0], (
                "The injection vector should have the same length as the number of timesteps."
            )
            nodal_injection_sum = np.sum(nodal_injection, axis=0)
            assert np.allclose(nodal_injection_sum, local_nodal_injection), (
                "The sum of all local nodal injections should be equal to the nodal injection at the node."
            )
