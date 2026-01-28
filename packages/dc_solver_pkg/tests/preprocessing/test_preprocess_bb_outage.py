# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from collections import Counter
from dataclasses import replace

import numpy as np
from tests.deprecated.assignment import realise_bus_split_single_station
from toop_engine_dc_solver.preprocess.network_data import NetworkData, map_branch_injection_ids
from toop_engine_dc_solver.preprocess.preprocess import compute_separation_set_for_stations
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import (
    extract_busbar_outage_data,
    extract_outage_index_injection_from_asset,
    get_articulation_nodes,
    get_branch_injection_outages_for_rel_subs,
    get_busbar_branches_map,
    get_busbar_index,
    get_modified_stations,
    get_non_rel_articulation_nodes,
    get_rel_articulation_nodes,
    get_rel_non_rel_sub_bb_maps,
    get_relevant_stations,
    get_total_injection_along_stub_branch,
    update_network_data_with_non_rel_bb_outages,
)
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import enumerate_station_realisations
from toop_engine_interfaces.asset_topology import Busbar, Station, SwitchableAsset


def test_get_total_injection_along_stub_branch(network_data: NetworkData):
    # 0 - 1 - 2 - 3 - 4
    # Create a mock NetworkData object
    network_data_dummy = replace(
        network_data,
        from_nodes=np.array([0, 1, 2, 3]),
        to_nodes=np.array([1, 2, 3, 4]),
        nodal_injection=np.array([[10, 20, 30, 40, 50], [15, 25, 35, 45, 55]], dtype=float),
    )

    # Test case 1: Stub branch index 0, current node index 0
    result = get_total_injection_along_stub_branch(0, 0, network_data_dummy)
    expected_result = np.array([140, 160])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 2: Stub branch index 1, current node index 1
    result = get_total_injection_along_stub_branch(1, 1, network_data_dummy)
    expected_result = np.array([120, 135])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 3: Stub branch index 0, current node index 1
    result = get_total_injection_along_stub_branch(0, 1, network_data_dummy)
    expected_result = np.array([10, 15])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 5: Stub branch index 0, current node index 0 for the following network
    # 0 - 1 - 2 - 3 - 4
    #     \ - 5

    # Create a mock NetworkData object
    network_data_dummy = replace(
        network_data,
        from_nodes=np.array([0, 1, 2, 3, 1]),
        to_nodes=np.array([1, 2, 3, 4, 5]),
        nodal_injection=np.array([[10, 20, 30, 40, 50, -10], [15, 25, 35, 45, 55, -10]], dtype=float),
    )
    result = get_total_injection_along_stub_branch(0, 0, network_data_dummy)
    expected_result = np.array([130, 150])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_extract_outage_index_injection_from_asset(network_data: NetworkData):
    # Create mock SwitchableAsset objects
    asset1 = SwitchableAsset(grid_model_id="branch_01", in_service=True, branch_end="from", type="line")
    asset2 = SwitchableAsset(grid_model_id="branch_12", in_service=False, branch_end="to", type="line")
    asset3 = SwitchableAsset(grid_model_id="branch_23", in_service=True, branch_end="from", type="line")
    # asset4 = SwitchableAsset(
    #     grid_model_id="branch_02", in_service=True, branch_end="from"
    # )
    # asset5 = SwitchableAsset(
    #     grid_model_id="branch_03", in_service=True, branch_end="from"
    # )
    # asset6 = SwitchableAsset(
    #     grid_model_id="injection_node_0", in_service=True, branch_end=None
    # )
    asset7 = SwitchableAsset(
        grid_model_id="injection_node_2",
        in_service=True,
        branch_end=None,
        type="GENERATOR",
    )
    # asset8 = SwitchableAsset(
    #     grid_model_id="injection_node_1", in_service=True, branch_end=None
    # )

    # Create a mock NetworkData object

    """
    Network topology:
    |--------------------------> 3(0)
    0(10) -> 1(50) -/> 2(-10) -> 3(0)
    | -------------> 2(-10)

    2 is a relevant unsplit station
    asset1 is a stub branch
    """
    network_data_dummy = replace(
        network_data,
        from_nodes=np.array([0, 2, 0, 0]),
        to_nodes=np.array([1, 3, 2, 3]),
        nodal_injection=np.array([[10, 50, -10, 0, 0]], dtype=float),
        node_ids=["node_0", "node_1", "node_2", "node_3", "node_2"],
        branch_ids=["branch_01", "branch_23", "branch_02", "branch_03"],
        bridging_branch_mask=np.array([True, False, False, False]),
        injection_ids=["injection_node_0", "injection_node_2", "injection_node_1"],
        mw_injections=np.array([[10, -10, 50]], dtype=float),
        asset_topology=None,
        split_multi_outage_branches=None,
    )

    # Test case 1: Process a branch (asset_3) that is in service and not a stub branch
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection = extract_outage_index_injection_from_asset(asset3, network_data_dummy, 2, {})
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([0])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [1], f"Expected [1], but got {connected_branches_to_outage}"

    # Test case 2: Process an injection (asset_7) to a relevant substation (node 2) that is in service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection = extract_outage_index_injection_from_asset(asset7, network_data_dummy, 2, stub_power_map={})
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([-10])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [-10], but got {connected_branches_to_outage}"

    # Test case 3: Process a branch (asset_2) that is out of service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection = extract_outage_index_injection_from_asset(asset2, network_data_dummy, 1, {})
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [], but got {connected_branches_to_outage}"

    # Test case 4: Process a stub branch that is in service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection = extract_outage_index_injection_from_asset(asset1, network_data_dummy, 0, {})
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([50])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [], but got {connected_branches_to_outage}"


def test_extract_busbar_outage_data(network_data_preprocessed: NetworkData):
    # Create mock SwitchableAsset objects
    asset1 = SwitchableAsset(grid_model_id="branch_01", in_service=True, branch_end="from", type="line")
    asset2 = SwitchableAsset(grid_model_id="branch_12", in_service=False, branch_end="to", type="line")
    asset3 = SwitchableAsset(grid_model_id="branch_23", in_service=True, branch_end="from", type="line")
    asset4 = SwitchableAsset(grid_model_id="branch_02", in_service=True, branch_end="from", type="line")
    asset5 = SwitchableAsset(grid_model_id="branch_03", in_service=True, branch_end="from", type="line")
    asset6 = SwitchableAsset(
        grid_model_id="injection_node_0",
        in_service=True,
        branch_end=None,
        type="GENERATOR",
    )
    asset7 = SwitchableAsset(
        grid_model_id="injection_node_2",
        in_service=True,
        branch_end=None,
        type="GENERATOR",
    )
    # asset8 = SwitchableAsset(
    #     grid_model_id="injection_node_1", in_service=True, branch_end=None
    # )

    # Create a mock NetworkData object

    """
    Network topology:
    |--------------------------> 3(0)
    0(10) -> 1(50)  2(-10) -> 3(0)
    |-------------> 2(-10)

    asset1 (branch_01) is a stub branch
    2 is a relevant unsplit station
    """
    network_data_dummy = replace(
        network_data_preprocessed,
        from_nodes=np.array([0, 2, 0, 0]),
        to_nodes=np.array([1, 3, 2, 3]),
        nodal_injection=np.array([[10, 50, -10, 0, 0]], dtype=float),
        node_ids=["node_0", "node_1", "node_2", "node_3"],
        branch_ids=["branch_01", "branch_23", "branch_02", "branch_03"],
        bridging_branch_mask=np.array([True, False, False, False]),
        injection_ids=["injection_node_0", "injection_node_2", "injection_node_1"],
        mw_injections=np.array([[10, -10, 50]], dtype=float),
        relevant_node_mask=np.array([False, False, False, False]),
        asset_topology=None,
        split_multi_outage_branches=None,
    )

    # Create a mock Station object
    busbar_0 = Busbar(
        grid_model_id="busbar_0",
        int_id=0,
    )
    busbar_1 = Busbar(
        grid_model_id="busbar_1",
        int_id=1,
    )
    station = Station(
        grid_model_id="node_2",
        busbars=[busbar_0, busbar_1],
        couplers=[],
        assets=[asset2, asset3, asset4, asset7],
        asset_switching_table=np.array(
            [
                [True, False, True, False],  # Busbar 0
                [False, True, False, True],  # Busbar 1
            ],
            dtype=bool,
        ),
    )

    # Test case 1: Outage busbar_1 of station 2 when all the assets are not connected to
    # the same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage = extract_busbar_outage_data(
        station, "busbar_0", network_data_dummy, {}
    )
    multi_branch_outages.append(list(set(branch_indices_to_outage)))
    multi_injection_outages.append(nodal_injection_to_outage.tolist())
    multi_node_outages.append(node_index_to_outage)

    expected_multi_branch_outages = [[2]]
    expected_multi_injection_outages = [[0]]

    assert multi_branch_outages == expected_multi_branch_outages, (
        f"Expected {expected_multi_branch_outages}, but got {multi_branch_outages}"
    )
    assert np.allclose(multi_injection_outages[0], expected_multi_injection_outages[0]), (
        f"Expected {expected_multi_injection_outages}, but got {multi_injection_outages}"
    )

    # Test case 2: Outage busbar_0 of node 2 when all the assets are not connected to same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage = extract_busbar_outage_data(
        station, "busbar_1", network_data_dummy, {}
    )

    multi_branch_outages.append(list(set(branch_indices_to_outage)))
    multi_injection_outages.append(nodal_injection_to_outage.tolist())
    multi_node_outages.append(node_index_to_outage)

    expected_multi_branch_outages = [[1]]
    expected_multi_injection_outages = [[-10]]

    assert multi_branch_outages == expected_multi_branch_outages, (
        f"Expected {expected_multi_branch_outages}, but got {multi_branch_outages}"
    )
    assert np.allclose(multi_injection_outages[0], expected_multi_injection_outages[0]), (
        f"Expected {expected_multi_injection_outages}, but got {multi_injection_outages}"
    )

    # Test case 3: Outage busbar where all the assets are connected to the same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage = extract_busbar_outage_data(
        station, "busbar_1", network_data_dummy, {}
    )

    multi_branch_outages.append(list(set(branch_indices_to_outage)))
    multi_injection_outages.append(nodal_injection_to_outage.tolist())
    multi_node_outages.append(node_index_to_outage)

    expected_multi_injection_outages = [[-10]]

    # There are 2 branches and 1 injection (asset7(-10)) connected to busbar_1 of node_2. However,
    # if all the branches are disconnected, then it would lead to grid splitting. Hence,
    # only 1 branch (either asset3 or asset4) should be outaged.
    assert len(multi_branch_outages) == 1, f"Expected {1} branch outage, but got {len(multi_branch_outages)}"
    assert np.allclose(multi_injection_outages[0], expected_multi_injection_outages[0]), (
        f"Expected {expected_multi_injection_outages}, but got {multi_injection_outages}"
    )

    # Test case 4: Outage a node (node 0) with stub branch (asset_1)
    # Create a mock Station object for node_0. Node_0 is a non relevant unsplit station. Therefore, there is just 1 busbar

    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []

    busbar_0 = Busbar(
        grid_model_id="busbar_0",
        int_id=0,
    )
    station = Station(
        grid_model_id="node_0",
        busbars=[busbar_0],
        couplers=[],
        assets=[asset1, asset4, asset5, asset6],
        asset_switching_table=np.array(
            [
                [True, True, True, True],  # Busbar 0
            ],
            dtype=bool,
        ),
    )

    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage = extract_busbar_outage_data(
        station, "busbar_0", network_data_dummy, {}
    )
    multi_branch_outages.append(list(set(branch_indices_to_outage)))
    multi_injection_outages.append(nodal_injection_to_outage.tolist())
    multi_node_outages.append(node_index_to_outage)

    # In this case, there are three branches connected to node_0 (asset1 (stub branch), asset4, asset5) and one injection (asset6).
    # The stub branch (asset1) can't be disconncted as it will lead to isolation of node_1.
    len_expected_multi_branch_outages = 2

    # As asset1 is a stub branch, the injection of node_0 (10) + the injection of node_1 (50) should be outaged
    expected_multi_injection_outages = [[60]]
    expected_node_outage_indices = [0]
    expected_multi_branch_outages = [2, 3]
    assert len(multi_branch_outages[0]) == len_expected_multi_branch_outages, (
        f"Expected {len_expected_multi_branch_outages} branch outage, but got {len(multi_branch_outages)}"
    )

    assert multi_branch_outages[0] == expected_multi_branch_outages, (
        f"Expected {expected_multi_branch_outages}, but got {multi_branch_outages}"
    )
    assert np.allclose(multi_injection_outages[0], expected_multi_injection_outages[0]), (
        f"Expected {expected_multi_injection_outages}, but got {multi_injection_outages}"
    )
    assert multi_node_outages == expected_node_outage_indices, (
        f"Expected {expected_node_outage_indices}, but got {multi_node_outages}"
    )


def test_update_network_data_with_non_rel_bb_outages(network_data_preprocessed: NetworkData):
    outage_station_busbars_map = {"8%%bus": ["8%%bus"], "71%%bus": ["71%%bus"]}
    rel_bb_map, non_rel_bb_map = get_rel_non_rel_sub_bb_maps(
        network_data_preprocessed, outage_station_busbars_map=outage_station_busbars_map
    )
    updated_net_data = update_network_data_with_non_rel_bb_outages(network_data_preprocessed, non_rel_bb_map)

    # Test case 1: Check if the function returns the correct number of multi-branch outages
    assert len(updated_net_data.non_rel_bb_outage_br_indices) == len(non_rel_bb_map), (
        f"Expected {len(non_rel_bb_map)} multi-branch outages, but got {len(non_rel_bb_map.non_rel_bb_outage_br_indices)}"
    )

    # Test case 2: Check if the function returns the node_index for each of multi-injection outages
    assert len(updated_net_data.non_rel_bb_outage_deltap) == len(updated_net_data.non_rel_bb_outage_nodal_indices), (
        "Expected the number of multi-injection outages to be equal to the number of nodes to be outaged"
    )

    # Test case 3: Check if the branches to be outaged are valid and connected to the busbar
    for branch_outages, station_id in zip(updated_net_data.non_rel_bb_outage_br_indices, non_rel_bb_map):
        for station in updated_net_data.asset_topology.stations:
            if station.grid_model_id == station_id:
                break

        for busbar_id in non_rel_bb_map[station_id]:
            busbar_index = get_busbar_index(station, busbar_id)
            for branch_index in branch_outages:
                branch_id = updated_net_data.branch_ids[branch_index]
                # get asset_index of the branch
                for asset_index, asset in enumerate(station.assets):
                    if asset.grid_model_id == branch_id:
                        break

                assert station.asset_switching_table[busbar_index, asset_index], (
                    f"Branch {branch_id} is not connected to busbar {busbar_id}"
                )


def test_get_branch_injection_outages_for_rel_subs(
    network_data_preprocessed: NetworkData,
):
    network_data_preprocessed = compute_separation_set_for_stations(network_data_preprocessed)
    network_data_preprocessed = enumerate_station_realisations(network_data_preprocessed)
    # 71%%bus is a relevant node
    rel_station_busbars_map = {
        "71%%bus": ["71%%bus_a", "71%%bus_b"],
        "157%%bus": ["157%%bus_a"],
    }
    outage_data_branch_indices, outage_data_deltap, outage_data_nodal_index = get_branch_injection_outages_for_rel_subs(
        network_data_preprocessed, rel_station_busbars_map
    )

    # Test case 1: Check if the function returns the correct number of outage data sets
    assert len(outage_data_branch_indices) == len(network_data_preprocessed.relevant_nodes), (
        f"Expected {len(network_data_preprocessed.relevant_nodes)} outage data sets, "
        f"but got {len(outage_data_branch_indices)}"
    )
    assert len(outage_data_deltap) == len(network_data_preprocessed.relevant_nodes), (
        f"Expected {len(network_data_preprocessed.relevant_nodes)} outage data sets, but got {len(outage_data_deltap)}"
    )
    assert len(outage_data_nodal_index) == len(network_data_preprocessed.relevant_nodes), (
        f"Expected {len(network_data_preprocessed.relevant_nodes)} outage data sets, but got {len(outage_data_nodal_index)}"
    )

    # Test case 2: Check if the second and third dimensions of outage_data_deltap and outage_data_nodal_index are the same
    assert all(len(outage_data_deltap[i]) == len(outage_data_nodal_index[i]) for i in range(len(outage_data_deltap))), (
        f"Expected the first dimension of outage_data_deltap and outage_data_nodal_index to be the same, "
        f"but got {len(outage_data_deltap)} and {len(outage_data_nodal_index)}"
    )
    assert all(
        len(outage_data_deltap[i][j]) == len(outage_data_nodal_index[i][j])
        for i in range(len(outage_data_deltap))
        for j in range(len(outage_data_deltap[i]))
    ), (
        f"Expected the second dimension of outage_data_deltap and outage_data_nodal_index to be the same, "
        f"but got {[[len(outage_data_deltap[i][j]) for j in range(len(outage_data_deltap[i]))] for i in range(len(outage_data_deltap))]} and "
        f"{[[len(outage_data_nodal_index[i][j]) for j in range(len(outage_data_nodal_index[i]))] for i in range(len(outage_data_nodal_index))]}"
    )

    rel_stations = get_relevant_stations(network_data_preprocessed)
    rel_station_index = 0
    branch_ids_mapped, _ = map_branch_injection_ids(network_data_preprocessed)
    branch_actions = network_data_preprocessed.branch_action_set
    for station_combis in outage_data_branch_indices:
        for combi_index, busbar_outages in enumerate(station_combis):
            modified_station, _, _ = realise_bus_split_single_station(
                branch_ids_local=branch_ids_mapped[rel_station_index],
                branch_topology_local=branch_actions[rel_station_index][combi_index],
                injection_ids_local=[],
                injection_topology_local=np.array([], dtype=bool),
                station=rel_stations[rel_station_index],
            )

            busbar_branches_map = get_busbar_branches_map(modified_station, network_data_preprocessed)

            for br_indices in busbar_outages:
                if len(br_indices) > 0:
                    # Test case 4: Check if the branches to be outaged are connected to a single physical busbar of the station
                    # get modified station object for the given combinattion.
                    match_found = 0
                    for connected_branches in busbar_branches_map.values():
                        if set(br_indices).issubset(set(connected_branches)):
                            match_found += 1

                    assert match_found == 1, (
                        "Expected br_indices to be a subset of only one of the busbars' connected branches"
                    )
        rel_station_index += 1

    # Test case 5: Check that the there should be 0 combis for 3rd relevant node, 2 busbar outage data for 1st rel node and
    # 1 busbar outage data for 2nd rel node
    assert len(outage_data_branch_indices[0][0]) == 2, (
        f"Expected 2 busbar outage data for 1st rel node, but got {len(outage_data_branch_indices[0])}"
    )
    assert len(outage_data_branch_indices[1][0]) == 2, (
        f"Expected 2 busbar outage data for 2nd rel node, but got {len(outage_data_branch_indices[1])}"
    )
    assert len(outage_data_branch_indices[1][0][0]) == 0 or len(outage_data_branch_indices[1][0][1]) == 0
    assert len(outage_data_branch_indices[2]) == 0, (
        f"Expected 0 combis data for 3rd rel node, but got {len(outage_data_branch_indices[2])}"
    )


def test_get_modified_stations(network_data_preprocessed: NetworkData):
    # '157%%bus' is a rel sub with 2 busbars; This has 5 branches connected to it and 2 injections -
    # 1 generator and 1 load; 1 closed coupler
    # switching_table:
    # array([[False,  True, False, False, False,  True, False],
    #        [True, False,  True,  True,  True, False,  True]]
    monitored_station = network_data_preprocessed.asset_topology.stations[156]
    outage_stations = [monitored_station.grid_model_id]
    branch_actions_all_rel_sub = network_data_preprocessed.branch_action_set
    modified_stations_br = get_modified_stations(network_data=network_data_preprocessed, stations_to_outage=outage_stations)

    # Test Case 1: There should be no combinations for stations 0 and station 2
    assert len(modified_stations_br[0]) == 0, (
        f"Expected 0 combinations for station 0 branch actions, but got {len(modified_stations_br[0])}"
    )
    assert len(modified_stations_br[2]) == 0, (
        f"Expected 0 combinations for station 2 branch actions, but got {len(modified_stations_br[0])}"
    )
    assert len(modified_stations_br[1]) == len(branch_actions_all_rel_sub[1]), (
        f"Expected {len(branch_actions_all_rel_sub[1])} combinations for station 1 branch actions, but got {len(modified_stations_br[1])}"
    )

    # Test Case 2: The switching table of station 1 should be according to the branch_actions_all_rel_sub[1].
    # Also, the configuration of the injections should not change.
    res = []
    for action_index, action in enumerate(branch_actions_all_rel_sub[1]):
        if not action.any():
            res.append(
                np.all(
                    modified_stations_br[1][action_index].asset_switching_table == monitored_station.asset_switching_table
                )
            )
        else:
            res.append(
                np.all(
                    modified_stations_br[1][action_index].asset_switching_table[:, 0 : len(action)] == action, axis=1
                ).any()
            )
    assert np.sum(res) == len(modified_stations_br[1]), (
        "Some branch actions didn't execute properly as a result, the modified switching table is not as expected"
    )
    assert np.all(
        [
            np.all(
                modified_stations_br[1][i].asset_switching_table[:, len(branch_actions_all_rel_sub[1][i]) :]
                == monitored_station.asset_switching_table[:, len(branch_actions_all_rel_sub[1][i]) :]
            )
            for i in range(len(modified_stations_br[1]))
        ]
    ), (
        "The injection configuration in the switching table for the modified station should be the same as the original station"
    )


def test_get_articulation_nodes():
    # Test case 1: Simple graph with one articulation node
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    result = get_articulation_nodes(nodes, edges)
    expected_result = [1]
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Test case 2: Simple graph with teo articulation nodes
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    result = get_articulation_nodes(nodes, edges)
    expected_result = [1, 2]
    assert Counter(result) == Counter(expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 3: Graph with multiple articulation nodes
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
    result = get_articulation_nodes(nodes, edges)
    expected_result = [1, 3]
    assert Counter(result) == Counter(expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 4: Graph with no edges
    nodes = [0, 1, 2, 3]
    edges = []
    result = get_articulation_nodes(nodes, edges)
    expected_result = []
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Test case 5: Graph with a single edge
    nodes = [0, 1]
    edges = [(0, 1)]
    result = get_articulation_nodes(nodes, edges)
    expected_result = []
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Test case 6: Graph with a cycle (no articulation nodes)
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result = get_articulation_nodes(nodes, edges)
    expected_result = []
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_get_non_rel_bridge_busbars(network_data_test_grid: NetworkData):
    outage_map = {
        "VL2_0": ["BBS2_1", "BBS2_2", "BBS2_3"],
    }
    non_rel_busbar_outage_map = get_non_rel_articulation_nodes(outage_map, network_data_test_grid)
    expected_map = {
        "VL2_0": ["BBS2_1", "BBS2_3"],
    }
    assert non_rel_busbar_outage_map == expected_map, f"Expected {expected_map}, but got {non_rel_busbar_outage_map}"


def test_get_rel_bridge_busbars(mock_station: Station):
    articulation_nodes = get_rel_articulation_nodes([mock_station], [[[2, 3, 4]]])
    assert articulation_nodes == [[[3]]], f"Expected [[[3]]], but got {articulation_nodes}"

    articulation_nodes = get_rel_articulation_nodes([mock_station], [[[2, 3, 4], [2, 3, 4]]])
    assert articulation_nodes == [[[3], [3]]], f"Expected [[[3], [3]]], but got {articulation_nodes}"
