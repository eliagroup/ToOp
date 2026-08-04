# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from collections import Counter
from dataclasses import replace

import numpy as np
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess import compute_separation_set_for_stations
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import (
    _get_busbar_outage_node_index,
    _traverse_stub_branch_subtree,
    extract_busbar_outage_data,
    extract_outage_index_injection_from_asset,
    get_all_rel_bb_outage_data,
    get_articulation_nodes,
    get_branch_injection_outages_for_rel_subs,
    get_busbar_branches_map,
    get_busbar_index,
    get_modified_stations,
    get_non_rel_articulation_nodes,
    get_rel_articulation_nodes,
    get_rel_non_rel_sub_bb_maps,
    update_network_data_with_non_rel_bb_outages,
)
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import enumerate_station_realisations
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeAssetTopology,
    RuntimeBusGroup,
)


def _combined_asset_connections(station: RuntimeBusGroup) -> list[RuntimeAssetConnection]:
    return [*station.branch_connections, *station.injection_connections]


def _combined_asset_switching_table(station: RuntimeBusGroup) -> np.ndarray:
    return np.concatenate([station.branch_switching_table, station.injection_switching_table], axis=1)


def build_runtime_bus_group(
    grid_model_id: str,
    busbars: list[RuntimeBusbar],
    couplers: list[RuntimeBusbarCoupler],
    branch_assets: list[RuntimeBranchAsset],
    injection_assets: list[RuntimeInjectionAsset],
    branch_switching_table: np.ndarray,
    injection_switching_table: np.ndarray,
    branch_connectivity: np.ndarray | None = None,
    injection_connectivity: np.ndarray | None = None,
) -> RuntimeBusGroup:
    return RuntimeBusGroup(
        bus_group_id=grid_model_id,
        busbars=busbars,
        couplers=couplers,
        branch_connections=[RuntimeAssetConnection(asset=asset) for asset in branch_assets],
        injection_connections=[RuntimeAssetConnection(asset=asset) for asset in injection_assets],
        branch_switching_table=branch_switching_table,
        injection_switching_table=injection_switching_table,
        branch_connectivity=branch_connectivity,
        injection_connectivity=injection_connectivity,
    )


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
    result, _ = _traverse_stub_branch_subtree(0, 0, network_data_dummy)
    expected_result = np.array([140, 160])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 2: Stub branch index 1, current node index 1
    result, _ = _traverse_stub_branch_subtree(1, 1, network_data_dummy)
    expected_result = np.array([120, 135])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test case 3: Stub branch index 0, current node index 1
    result, _ = _traverse_stub_branch_subtree(0, 1, network_data_dummy)
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
    result, _ = _traverse_stub_branch_subtree(0, 0, network_data_dummy)
    expected_result = np.array([130, 150])
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_extract_outage_index_injection_from_asset(network_data: NetworkData):
    # Create mock SwitchableAsset objects
    asset1 = RuntimeBranchAsset(grid_model_id="branch_01", in_service=True, asset_type="line")
    asset2 = RuntimeBranchAsset(grid_model_id="branch_12", in_service=False, asset_type="line")
    asset3 = RuntimeBranchAsset(grid_model_id="branch_23", in_service=True, asset_type="line")
    # asset4 = SwitchableAsset(
    #     grid_model_id="branch_02", in_service=True, branch_end="from"
    # )
    # asset5 = SwitchableAsset(
    #     grid_model_id="branch_03", in_service=True, branch_end="from"
    # )
    # asset6 = SwitchableAsset(
    #     grid_model_id="injection_node_0", in_service=True, branch_end=None
    # )
    asset7 = RuntimeInjectionAsset(
        grid_model_id="injection_node_2",
        in_service=True,
        asset_type="GENERATOR",
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
        split_multi_outage_branches=None,
    )

    # Test case 1: Process a branch (asset_3) that is in service and not a stub branch
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection, zero_flow_branch_indices = extract_outage_index_injection_from_asset(
        asset3, network_data_dummy, 2, {}
    )
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([0])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [1], f"Expected [1], but got {connected_branches_to_outage}"
    assert zero_flow_branch_indices == [], f"Expected [], but got {zero_flow_branch_indices}"
    # Test case 2: Process an injection (asset_7) to a relevant substation (node 2) that is in service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection, zero_flow_branch_indices = extract_outage_index_injection_from_asset(
        asset7, network_data_dummy, 2, stub_power_map={}
    )
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([-10])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [-10], but got {connected_branches_to_outage}"
    assert zero_flow_branch_indices == [], f"Expected [], but got {zero_flow_branch_indices}"

    # Test case 3: Process a branch (asset_2) that is out of service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection, zero_flow_branch_indices = extract_outage_index_injection_from_asset(
        asset2, network_data_dummy, 1, {}
    )
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [], but got {connected_branches_to_outage}"
    assert zero_flow_branch_indices == [], f"Expected [], but got {zero_flow_branch_indices}"

    # Test case 4: Process a stub branch that is in service
    nodal_injection_to_outage = np.zeros(network_data_dummy.nodal_injection.shape[0], float)
    connected_branches_to_outage = []
    branch_index, injection, zero_flow_branch_indices = extract_outage_index_injection_from_asset(
        asset1, network_data_dummy, 0, {}
    )
    if branch_index is not None:
        connected_branches_to_outage.append(branch_index)
    nodal_injection_to_outage += injection

    expected_busbar_nodal_injection_removal = np.array([50])
    assert np.allclose(nodal_injection_to_outage, expected_busbar_nodal_injection_removal), (
        f"Expected {expected_busbar_nodal_injection_removal}, but got {nodal_injection_to_outage}"
    )
    assert connected_branches_to_outage == [], f"Expected [], but got {connected_branches_to_outage}"
    assert zero_flow_branch_indices == [0], f"Expected [0], but got {zero_flow_branch_indices}"


def test_extract_outage_index_injection_from_asset_uses_bridge_direction_for_stub_side_busbar(
    network_data: NetworkData,
) -> None:
    asset = RuntimeBranchAsset(grid_model_id="branch_01", in_service=True, asset_type="line")
    network_data_dummy = replace(
        network_data,
        from_nodes=np.array([0, 1, 2]),
        to_nodes=np.array([1, 2, 3]),
        nodal_injection=np.array([[10, 20, 30, 40]], dtype=float),
        node_ids=["node_0", "node_1", "node_2", "node_3"],
        branch_ids=["branch_01", "branch_12", "branch_23"],
        bridging_branch_mask=np.array([True, False, False]),
        bridge_mainland_node_indices=np.array([1, -1, -1]),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        split_multi_outage_branches=None,
    )

    branch_index, injection, zero_flow_branch_indices = extract_outage_index_injection_from_asset(
        asset,
        network_data_dummy,
        nodal_index_for_busbar=0,
        stub_power_map={},
    )

    assert branch_index is None
    assert np.allclose(injection, np.array([10]))
    assert zero_flow_branch_indices == [0]


def test_get_busbar_outage_node_index_falls_back_to_busbar_bus_id(network_data: NetworkData) -> None:
    station = build_runtime_bus_group(
        grid_model_id="ab0e0e4f-10e5-411a-bf4e-6232f521985e_1",
        busbars=[
            RuntimeBusbar(
                grid_model_id="38861c44-57c7-5778-459d-bb997f25d415",
                int_id=0,
                bus_branch_bus_id="ab0e0e4f-10e5-411a-bf4e-6232f521985e_2",
                bus_breaker_bus_id="ab0e0e4f-10e5-411a-bf4e-6232f521985e_2",
            ),
            RuntimeBusbar(
                grid_model_id="5fd53822-fbb9-f012-5cbb-ac2d2d6aaf6c",
                int_id=1,
                bus_branch_bus_id="ab0e0e4f-10e5-411a-bf4e-6232f521985e_1",
                bus_breaker_bus_id="ab0e0e4f-10e5-411a-bf4e-6232f521985e_1",
            ),
        ],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((2, 0), dtype=bool),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
        branch_connectivity=np.zeros((2, 0), dtype=bool),
        injection_connectivity=np.zeros((2, 0), dtype=bool),
    )
    network_data_dummy = replace(
        network_data,
        node_ids=["ab0e0e4f-10e5-411a-bf4e-6232f521985e_2"],
    )

    node_index = _get_busbar_outage_node_index(station, 0, network_data_dummy, branch_action_combi_index=None)

    assert node_index == 0


def test_get_busbar_outage_node_index_falls_back_when_station_lookup_is_ambiguous(
    network_data: NetworkData,
) -> None:
    station = build_runtime_bus_group(
        grid_model_id="station_0",
        busbars=[
            RuntimeBusbar(
                grid_model_id="busbar_0",
                int_id=0,
                bus_branch_bus_id="node_0",
                bus_breaker_bus_id="node_0",
            ),
            RuntimeBusbar(
                grid_model_id="busbar_1",
                int_id=1,
                bus_branch_bus_id="node_1",
                bus_breaker_bus_id="node_1",
            ),
        ],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((2, 0), dtype=bool),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
        branch_connectivity=np.zeros((2, 0), dtype=bool),
        injection_connectivity=np.zeros((2, 0), dtype=bool),
    )
    network_data_dummy = replace(
        network_data,
        node_ids=["station_0", "node_0", "node_1"],
        relevant_node_mask=np.array([False, False, False], dtype=bool),
    )

    node_index = _get_busbar_outage_node_index(station, 0, network_data_dummy, branch_action_combi_index=0)

    assert node_index == 1


def test_extract_busbar_outage_data(network_data_preprocessed: NetworkData):
    # Create mock SwitchableAsset objects
    asset1 = RuntimeBranchAsset(grid_model_id="branch_01", in_service=True, asset_type="line")
    asset2 = RuntimeBranchAsset(grid_model_id="branch_12", in_service=False, asset_type="line")
    asset3 = RuntimeBranchAsset(grid_model_id="branch_23", in_service=True, asset_type="line")
    asset4 = RuntimeBranchAsset(grid_model_id="branch_02", in_service=True, asset_type="line")
    asset5 = RuntimeBranchAsset(grid_model_id="branch_03", in_service=True, asset_type="line")
    asset7 = RuntimeInjectionAsset(
        grid_model_id="injection_node_2",
        in_service=True,
        asset_type="GENERATOR",
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
        split_multi_outage_branches=None,
    )

    # Create a mock Station object
    busbar_0 = RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")
    busbar_1 = RuntimeBusbar(grid_model_id="busbar_1", int_id=1, bus_branch_bus_id="node_2", bus_breaker_bus_id="node_2")
    station = RuntimeBusGroup(
        bus_group_id="node_2",
        busbars=[busbar_0, busbar_1],
        couplers=[],
        branch_connections=[RuntimeAssetConnection(asset=asset) for asset in [asset2, asset3, asset4]],
        injection_connections=[RuntimeAssetConnection(asset=asset7)],
        branch_switching_table=np.array(
            [
                [True, False, True],  # Busbar 0
                [False, True, False],  # Busbar 1
            ],
            dtype=bool,
        ),
        injection_switching_table=np.array(
            [
                [False],
                [True],
            ],
            dtype=bool,
        ),
    )

    # Test case 1: Outage busbar_1 of station 2 when all the assets are not connected to
    # the same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(station, "busbar_0", network_data_dummy, {})
    )
    multi_branch_outages.append(branch_indices_to_outage)
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
    assert zero_flow_branch_indices == []

    # Test case 2: Outage busbar_0 of node 2 when all the assets are not connected to same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(station, "busbar_1", network_data_dummy, {})
    )

    multi_branch_outages.append(branch_indices_to_outage)
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
    assert zero_flow_branch_indices == []

    # Test case 3: Outage busbar where all the assets are connected to the same busbar
    multi_branch_outages = []
    multi_injection_outages = []
    multi_node_outages = []
    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(station, "busbar_1", network_data_dummy, {})
    )

    multi_branch_outages.append(branch_indices_to_outage)
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
    assert zero_flow_branch_indices == []

    # Test case 4: Outage a node (node 0) with stub branch (asset_1)
    # Create a mock Station object for node_0. Node_0 is a non relevant unsplit station. Therefore, there is just 1 busbar


def test_extract_busbar_outage_data_extends_over_double_connections(network_data_preprocessed: NetworkData) -> None:
    network_data_dummy = replace(
        network_data_preprocessed,
        from_nodes=np.array([0, 0]),
        to_nodes=np.array([1, 1]),
        nodal_injection=np.array([[0.0, 0.0]], dtype=float),
        node_ids=["node_0", "node_1"],
        branch_ids=["branch_local", "branch_shared"],
        bridging_branch_mask=np.array([False, False]),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        relevant_node_mask=np.array([False, False]),
        asset_topology=None,
        split_multi_outage_branches=None,
    )

    busbar_0 = RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")
    busbar_1 = RuntimeBusbar(grid_model_id="busbar_1", int_id=1, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")
    station = build_runtime_bus_group(
        grid_model_id="node_a",
        busbars=[busbar_0, busbar_1],
        couplers=[],
        branch_assets=[
            RuntimeBranchAsset(grid_model_id="branch_local", in_service=True, asset_type="line"),
            RuntimeBranchAsset(grid_model_id="branch_shared", in_service=True, asset_type="line"),
        ],
        injection_assets=[],
        branch_switching_table=np.array(
            [
                [True, True],
                [False, True],
            ],
            dtype=bool,
        ),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
    )

    branch_indices_to_outage, nodal_injection_to_outage, _node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(station, "busbar_0", network_data_dummy, {})
    )

    assert branch_indices_to_outage == [0, 1], f"Expected both connected branches to outage, got {branch_indices_to_outage}"
    assert np.allclose(nodal_injection_to_outage, np.array([0.0]))
    assert zero_flow_branch_indices == []


def test_extract_busbar_outage_data_uses_realized_station_topology_for_relevant_case(
    network_data_preprocessed: NetworkData,
) -> None:
    physical_station = build_runtime_bus_group(
        grid_model_id="node_0",
        busbars=[
            RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0"),
            RuntimeBusbar(grid_model_id="busbar_1", int_id=1, bus_branch_bus_id="node_1", bus_breaker_bus_id="node_1"),
        ],
        couplers=[],
        branch_assets=[
            RuntimeBranchAsset(grid_model_id="branch_left", in_service=True, asset_type="line"),
            RuntimeBranchAsset(grid_model_id="branch_right", in_service=True, asset_type="line"),
        ],
        injection_assets=[],
        branch_switching_table=np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        ),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
    )
    realized_station = build_runtime_bus_group(
        grid_model_id="node_0",
        busbars=[
            RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0"),
            RuntimeBusbar(grid_model_id="busbar_1", int_id=1, bus_branch_bus_id="node_1", bus_breaker_bus_id="node_1"),
        ],
        couplers=[],
        branch_assets=[
            RuntimeBranchAsset(grid_model_id="branch_left", in_service=True, asset_type="line"),
            RuntimeBranchAsset(grid_model_id="branch_right", in_service=True, asset_type="line"),
        ],
        injection_assets=[],
        branch_switching_table=np.array(
            [
                [True, True],
                [False, False],
            ],
            dtype=bool,
        ),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
    )
    network_data_dummy = replace(
        network_data_preprocessed,
        from_nodes=np.array([0, 0]),
        to_nodes=np.array([1, 1]),
        nodal_injection=np.array([[0.0, 0.0]], dtype=float),
        node_ids=["node_0", "node_1"],
        branch_ids=["branch_left", "branch_right"],
        bridging_branch_mask=np.array([False, False]),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        relevant_node_mask=np.array([False, False]),
        asset_topology=network_data_preprocessed.asset_topology.model_copy(update={"stations": [physical_station]}),
        split_multi_outage_branches=None,
    )

    branch_indices_to_outage, nodal_injection_to_outage, _node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(realized_station, "busbar_0", network_data_dummy, {}, branch_action_combi_index=0)
    )

    assert branch_indices_to_outage == [0, 1]
    assert np.allclose(nodal_injection_to_outage, np.array([0.0]))
    assert zero_flow_branch_indices == []


def test_extract_busbar_outage_data_preserves_branch_order(network_data_preprocessed: NetworkData) -> None:
    network_data_dummy = replace(
        network_data_preprocessed,
        from_nodes=np.array([0, 0, 1]),
        to_nodes=np.array([1, 2, 2]),
        nodal_injection=np.array([[0.0, 0.0, 0.0]], dtype=float),
        node_ids=["node_0", "node_1", "node_2"],
        branch_ids=["branch_a", "branch_b", "branch_c"],
        bridging_branch_mask=np.array([False, False, False]),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        relevant_node_mask=np.array([False, False, False]),
        asset_topology=None,
        split_multi_outage_branches=None,
    )

    busbar_0 = RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")
    station = build_runtime_bus_group(
        grid_model_id="node_0",
        busbars=[busbar_0],
        couplers=[],
        branch_assets=[
            RuntimeBranchAsset(grid_model_id="branch_b", in_service=True, asset_type="line"),
            RuntimeBranchAsset(grid_model_id="branch_a", in_service=True, asset_type="line"),
            RuntimeBranchAsset(grid_model_id="branch_b", in_service=True, asset_type="line"),
        ],
        injection_assets=[],
        branch_switching_table=np.array([[True, True, True]], dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )

    branch_indices_to_outage, _, _, _ = extract_busbar_outage_data(station, "busbar_0", network_data_dummy, {})

    assert branch_indices_to_outage == [1, 0], f"Expected branch order [1, 0], but got {branch_indices_to_outage}"

    # This test only checks that branch order survives deduplication.


def test_extract_busbar_outage_data_handles_non_rel_stub_branch_compensation(
    network_data_preprocessed: NetworkData,
) -> None:
    # Topology under test:
    # |--------------------------> 3(0)
    # 0(10) -> 1(50)   2(-10) -> 3(0)
    # |-------------> 2(-10)
    #
    # node_0 is a non-relevant unsplit station with one physical busbar. When that
    # busbar is outaged, the bridge branch branch_01 cannot be opened directly because
    # it would isolate node_1 from the reduced network. The preprocessing therefore
    # keeps the bridge branch physically connected, compensates the disconnected stub
    # subtree via delta-p, and marks the bridge-fed subtree branches as zero-flow.
    asset1 = RuntimeBranchAsset(grid_model_id="branch_01", in_service=True, asset_type="line")
    asset4 = RuntimeBranchAsset(grid_model_id="branch_02", in_service=True, asset_type="line")
    asset5 = RuntimeBranchAsset(grid_model_id="branch_03", in_service=True, asset_type="line")
    asset6 = RuntimeInjectionAsset(grid_model_id="injection_node_0", in_service=True, asset_type="GENERATOR")
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

    busbar_0 = RuntimeBusbar(
        grid_model_id="busbar_0",
        int_id=0,
        bus_branch_bus_id="node_0",
    )
    station = RuntimeBusGroup(
        bus_group_id="node_a",
        busbars=[busbar_0],
        couplers=[],
        branch_connections=[RuntimeAssetConnection(asset=asset) for asset in [asset1, asset4, asset5]],
        injection_connections=[RuntimeAssetConnection(asset=asset6)],
        branch_switching_table=np.array(
            [
                [True, True, True],
            ],
            dtype=bool,
        ),
        injection_switching_table=np.array(
            [
                [True],
            ],
            dtype=bool,
        ),
    )

    branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage, zero_flow_branch_indices = (
        extract_busbar_outage_data(station, "busbar_0", network_data_dummy, {})
    )

    # branch_02 and branch_03 are the non-bridge branches directly connected to the
    # outaged busbar, so they are removed explicitly and must preserve this order.
    assert branch_indices_to_outage == [2, 3], f"Expected branch order [2, 3], but got {branch_indices_to_outage}"
    # The compensated outage injection is the local injection at node_0 (10) plus the
    # disconnected stub subtree injection behind branch_01 at node_1 (50) -> 60 total.
    assert np.allclose(nodal_injection_to_outage, np.array([60])), (
        f"Expected compensated outage injection [60], but got {nodal_injection_to_outage}"
    )
    # node_0 is the nodal index that represents this non-split physical busbar outage.
    assert node_index_to_outage == 0, f"Expected node outage index 0, but got {node_index_to_outage}"
    # branch_01 remains in the reduced topology but only the bridge-fed stub itself should be
    # forced to zero. The other branches are explicit outages and must not be reclassified as
    # zero-flow compensation branches.
    assert zero_flow_branch_indices == [0], f"Expected zero-flow branch indices [0], but got {zero_flow_branch_indices}"


def test_get_all_rel_bb_outage_data_preserves_physical_busbar_slots_for_out_of_service_busbars(
    network_data: NetworkData,
) -> None:
    station = build_runtime_bus_group(
        grid_model_id="node_a",
        busbars=[
            RuntimeBusbar(grid_model_id="busbar_0", int_id=0, in_service=True, bus_branch_bus_id="node_0"),
            RuntimeBusbar(
                grid_model_id="busbar_1", int_id=1, in_service=False, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0"
            ),
        ],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((2, 0), dtype=bool),
        injection_switching_table=np.zeros((2, 0), dtype=bool),
    )
    network_data_dummy = replace(
        network_data,
        node_ids=["node_0"],
        nodal_injection=np.zeros((1, 1), dtype=float),
        relevant_node_mask=np.array([True], dtype=bool),
        branch_ids=[],
        bridging_branch_mask=np.array([], dtype=bool),
        from_nodes=np.array([], dtype=int),
        to_nodes=np.array([], dtype=int),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        asset_topology=None,
    )

    outage_data = get_all_rel_bb_outage_data([[station]], network_data_dummy, {"busbar_0", "busbar_1"})

    assert len(outage_data[0][0]) == 2
    assert outage_data[0][0][1] is None


def test_get_all_rel_bb_outage_data_preserves_busbars_without_articulation_filtering(
    network_data: NetworkData,
) -> None:
    station = build_runtime_bus_group(
        grid_model_id="node_a",
        busbars=[
            RuntimeBusbar(grid_model_id="busbar_0", int_id=0, in_service=True, bus_branch_bus_id="node_0"),
            RuntimeBusbar(grid_model_id="busbar_1", int_id=1, in_service=True, bus_branch_bus_id="node_0"),
            RuntimeBusbar(grid_model_id="busbar_2", int_id=2, in_service=True, bus_branch_bus_id="node_0"),
        ],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((3, 0), dtype=bool),
        injection_switching_table=np.zeros((3, 0), dtype=bool),
    )
    network_data_dummy = replace(
        network_data,
        node_ids=["node_0"],
        nodal_injection=np.zeros((1, 1), dtype=float),
        relevant_node_mask=np.array([True], dtype=bool),
        branch_ids=[],
        bridging_branch_mask=np.array([], dtype=bool),
        from_nodes=np.array([], dtype=int),
        to_nodes=np.array([], dtype=int),
        injection_ids=[],
        mw_injections=np.zeros((1, 0), dtype=float),
        asset_topology=None,
    )

    outage_data = get_all_rel_bb_outage_data(
        [[station]],
        network_data_dummy,
        {"busbar_0", "busbar_1", "busbar_2"},
    )

    assert len(outage_data[0][0]) == 3
    assert outage_data[0][0][0] is not None
    assert outage_data[0][0][1] is not None
    assert outage_data[0][0][2] is not None


def test_update_network_data_with_non_rel_bb_outages(network_data_preprocessed: NetworkData):
    outage_station_busbars_map = {"8%%bus": ["8%%bus"], "71%%bus": ["71%%bus"]}
    rel_bb_map, non_rel_bb_map = get_rel_non_rel_sub_bb_maps(
        network_data_preprocessed, outage_station_busbars_map=outage_station_busbars_map
    )
    updated_net_data = update_network_data_with_non_rel_bb_outages(network_data_preprocessed, non_rel_bb_map)
    assert updated_net_data.simplified_asset_topology is not None
    stations = updated_net_data.asset_topology.stations
    expected_outage_count = sum(
        len(busbar_ids)
        for station_id, busbar_ids in non_rel_bb_map.items()
        if any(station.bus_group_id == station_id for station in stations)
    )

    # Test case 1: Check if the function returns the correct number of multi-branch outages
    assert len(updated_net_data.non_rel_bb_outage_br_indices) == expected_outage_count, (
        f"Expected {expected_outage_count} multi-branch outages, but got {len(updated_net_data.non_rel_bb_outage_br_indices)}"
    )

    # Test case 2: Check if the function returns the node_index for each of multi-injection outages
    assert len(updated_net_data.non_rel_bb_outage_deltap) == len(updated_net_data.non_rel_bb_outage_nodal_indices), (
        "Expected the number of multi-injection outages to be equal to the number of nodes to be outaged"
    )

    # Test case 3: Check if the branches to be outaged are valid and connected to the busbar
    branch_outages_iter = iter(updated_net_data.non_rel_bb_outage_br_indices)
    for station in stations:
        if station.bus_group_id not in non_rel_bb_map:
            continue
        for busbar_id in non_rel_bb_map[station.bus_group_id]:
            branch_outages = next(branch_outages_iter)
            busbar_index = get_busbar_index(station, busbar_id)
            for branch_index in branch_outages:
                branch_id = updated_net_data.branch_ids[branch_index]
                # get asset_index of the branch
                for asset_index, asset_connection in enumerate(_combined_asset_connections(station)):
                    if asset_connection.asset.grid_model_id == branch_id:
                        break

                assert _combined_asset_switching_table(station)[busbar_index, asset_index], (
                    f"Branch {branch_id} is not connected to busbar {busbar_id}"
                )


def test_get_branch_injection_outages_for_rel_subs(
    network_data_preprocessed: NetworkData,
):
    network_data_preprocessed = compute_separation_set_for_stations(network_data_preprocessed)
    network_data_preprocessed = enumerate_station_realisations(network_data_preprocessed)
    relevant_station_ids = [station_combis[0].bus_group_id for station_combis in network_data_preprocessed.realised_stations]
    assert "71%%bus" in relevant_station_ids

    assert network_data_preprocessed.asset_topology is not None
    monitored_station = next(
        station for station in network_data_preprocessed.asset_topology.stations if station.bus_group_id == "71%%bus"
    )
    non_outaged_station = next(
        station for station in network_data_preprocessed.asset_topology.stations if station.bus_group_id == "157%%bus"
    )

    rel_station_busbars_map = {
        "71%%bus": [busbar.grid_model_id for busbar in monitored_station.busbars],
        "157%%bus": [non_outaged_station.busbars[0].grid_model_id],
    }
    outage_data_branch_indices, outage_data_deltap, outage_data_nodal_index, outage_data_zero_flow_branches = (
        get_branch_injection_outages_for_rel_subs(network_data_preprocessed, rel_station_busbars_map)
    )
    monitored_station_index = relevant_station_ids.index("71%%bus")
    ignored_station_ids = set(relevant_station_ids) - set(rel_station_busbars_map)

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
    assert len(outage_data_zero_flow_branches) == len(network_data_preprocessed.relevant_nodes), (
        f"Expected {len(network_data_preprocessed.relevant_nodes)} zero-flow outage data sets, but got {len(outage_data_zero_flow_branches)}"
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

    assert network_data_preprocessed.realised_stations is not None
    rel_station_index = 0
    for station_combis in outage_data_branch_indices:
        for combi_index, busbar_outages in enumerate(station_combis):
            modified_station = network_data_preprocessed.realised_stations[rel_station_index][combi_index]

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

    # Test case 5: The monitored station keeps both busbar outages, and stations outside the
    # requested outage map remain empty.
    assert len(outage_data_branch_indices[monitored_station_index][0]) == 2, (
        "Expected 2 busbar outage data for the monitored relevant station, "
        f"but got {len(outage_data_branch_indices[monitored_station_index][0])}"
    )
    assert all(nodal_index is not None for nodal_index in outage_data_nodal_index[monitored_station_index][0]), (
        "Expected nodal indices for all selected busbar outages of the remaining relevant station"
    )
    for station_id, station_outages in zip(relevant_station_ids, outage_data_branch_indices, strict=True):
        if station_id in ignored_station_ids:
            assert station_outages == [], f"Expected no outage data for non-selected station {station_id}"


def test_get_modified_stations(network_data_preprocessed: NetworkData):
    # '71%%bus' is the monitored relevant substation for this test.
    # 1 generator and 1 load; 1 closed coupler
    # switching_table:
    # array([[False,  True, False, False, False,  True, False],
    #        [True, False,  True,  True,  True, False,  True]]
    assert network_data_preprocessed.asset_topology is not None
    monitored_station = next(
        station for station in network_data_preprocessed.asset_topology.stations if station.bus_group_id == "71%%bus"
    )
    outage_stations = [monitored_station.bus_group_id]
    branch_actions_all_rel_sub = network_data_preprocessed.branch_action_set
    modified_stations_br = get_modified_stations(network_data=network_data_preprocessed, stations_to_outage=outage_stations)
    relevant_station_ids = [station_combis[0].bus_group_id for station_combis in network_data_preprocessed.realised_stations]
    monitored_station_index = relevant_station_ids.index(monitored_station.bus_group_id)

    assert len(modified_stations_br) == len(network_data_preprocessed.realised_stations), (
        "Expected modified stations to preserve the relevant-station outer dimension"
    )
    for station_id, station_combis in zip(relevant_station_ids, modified_stations_br, strict=True):
        if station_id != monitored_station.bus_group_id:
            assert station_combis == [], f"Expected no modified station combinations for {station_id}"

    # Test Case 1: The monitored station should keep all branch action combinations.
    assert len(modified_stations_br[monitored_station_index]) == len(branch_actions_all_rel_sub[monitored_station_index]), (
        "Expected the monitored station to keep all branch action combinations, "
        f"but got {len(modified_stations_br[monitored_station_index])} instead of "
        f"{len(branch_actions_all_rel_sub[monitored_station_index])}"
    )

    # Test Case 2: The switching table of the monitored station should be according to its branch actions.
    # Also, the configuration of the injections should not change.
    res = []
    for action_index, action in enumerate(branch_actions_all_rel_sub[monitored_station_index]):
        if not action.any():
            res.append(
                np.all(
                    _combined_asset_switching_table(modified_stations_br[monitored_station_index][action_index])
                    == _combined_asset_switching_table(monitored_station)
                )
            )
        else:
            res.append(
                np.all(
                    modified_stations_br[monitored_station_index][action_index].branch_switching_table[:, 0 : len(action)]
                    == action,
                    axis=1,
                ).any()
            )
    assert np.sum(res) == len(modified_stations_br[monitored_station_index]), (
        "Some branch actions didn't execute properly as a result, the modified switching table is not as expected"
    )
    assert np.all(
        [
            np.all(
                modified_stations_br[monitored_station_index][i].injection_switching_table
                == monitored_station.injection_switching_table
            )
            for i in range(len(modified_stations_br[monitored_station_index]))
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
        "VL2_a": ["BBS2_1", "BBS2_2", "BBS2_3"],
    }
    non_rel_busbar_outage_map = get_non_rel_articulation_nodes(outage_map, network_data_test_grid)
    expected_map = {
        "VL2_a": ["BBS2_1", "BBS2_3"],
    }
    assert non_rel_busbar_outage_map == expected_map, f"Expected {expected_map}, but got {non_rel_busbar_outage_map}"


def test_exclude_island_busbars_from_outage_map(network_data: NetworkData) -> None:
    slack_station = build_runtime_bus_group(
        grid_model_id="station_0",
        busbars=[RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    mainland_station = build_runtime_bus_group(
        grid_model_id="station_1",
        busbars=[RuntimeBusbar(grid_model_id="busbar_1", int_id=0, bus_branch_bus_id="node_1", bus_breaker_bus_id="node_1")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station = build_runtime_bus_group(
        grid_model_id="station_2",
        busbars=[RuntimeBusbar(grid_model_id="busbar_2", int_id=0, bus_branch_bus_id="node_2", bus_breaker_bus_id="node_2")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_leaf_station = build_runtime_bus_group(
        grid_model_id="station_3",
        busbars=[RuntimeBusbar(grid_model_id="busbar_3", int_id=0, bus_branch_bus_id="node_3", bus_breaker_bus_id="node_3")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )

    network_data_dummy = replace(
        network_data,
        node_ids=["node_0", "node_1", "node_2", "node_3"],
        from_nodes=np.array([0, 1, 2]),
        to_nodes=np.array([1, 2, 3]),
        branch_ids=["branch_01", "branch_12", "branch_23"],
        bridging_branch_mask=np.array([False, True, False]),
        slack=0,
        asset_topology=RuntimeAssetTopology(
            stations=[slack_station, mainland_station, island_station, island_leaf_station],
        ),
    )

    outage_map = {
        "station_0": ["busbar_0"],
        "station_1": ["busbar_1"],
        "station_2": ["busbar_2"],
        "station_3": ["busbar_3"],
    }

    filtered_outage_map = exclude_island_busbars_from_outage_map(outage_map, network_data_dummy)

    assert filtered_outage_map == {
        "station_0": ["busbar_0"],
        "station_1": ["busbar_1"],
    }


def test_exclude_island_busbars_from_outage_map_uses_largest_component_when_slack_is_isolated(
    network_data: NetworkData,
) -> None:
    isolated_slack_station = build_runtime_bus_group(
        grid_model_id="station_0",
        busbars=[RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    mainland_station_a = build_runtime_bus_group(
        grid_model_id="station_1",
        busbars=[RuntimeBusbar(grid_model_id="busbar_1", int_id=0, bus_branch_bus_id="node_1", bus_breaker_bus_id="node_1")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    mainland_station_b = build_runtime_bus_group(
        grid_model_id="station_2",
        busbars=[RuntimeBusbar(grid_model_id="busbar_2", int_id=0, bus_branch_bus_id="node_2", bus_breaker_bus_id="node_2")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station = build_runtime_bus_group(
        grid_model_id="station_3",
        busbars=[RuntimeBusbar(grid_model_id="busbar_3", int_id=0, bus_branch_bus_id="node_3", bus_breaker_bus_id="node_3")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )

    network_data_dummy = replace(
        network_data,
        node_ids=["node_0", "node_1", "node_2", "node_3"],
        from_nodes=np.array([1, 2]),
        to_nodes=np.array([2, 3]),
        branch_ids=["branch_12", "branch_23"],
        bridging_branch_mask=np.array([False, True]),
        slack=0,
        asset_topology=RuntimeAssetTopology(
            stations=[isolated_slack_station, mainland_station_a, mainland_station_b, island_station],
        ),
    )

    outage_map = {
        "station_0": ["busbar_0"],
        "station_1": ["busbar_1"],
        "station_2": ["busbar_2"],
        "station_3": ["busbar_3"],
    }

    filtered_outage_map = exclude_island_busbars_from_outage_map(outage_map, network_data_dummy)

    assert filtered_outage_map == {
        "station_1": ["busbar_1"],
        "station_2": ["busbar_2"],
    }


def test_exclude_island_busbars_from_outage_map_prefers_component_with_more_monitored_branches(
    network_data: NetworkData,
) -> None:
    mainland_station_a = build_runtime_bus_group(
        grid_model_id="station_0",
        busbars=[RuntimeBusbar(grid_model_id="busbar_0", int_id=0, bus_branch_bus_id="node_0", bus_breaker_bus_id="node_0")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    mainland_station_b = build_runtime_bus_group(
        grid_model_id="station_1",
        busbars=[RuntimeBusbar(grid_model_id="busbar_1", int_id=0, bus_branch_bus_id="node_1", bus_breaker_bus_id="node_1")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    mainland_station_c = build_runtime_bus_group(
        grid_model_id="station_2",
        busbars=[RuntimeBusbar(grid_model_id="busbar_2", int_id=0, bus_branch_bus_id="node_2", bus_breaker_bus_id="node_2")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station_a = build_runtime_bus_group(
        grid_model_id="station_3",
        busbars=[RuntimeBusbar(grid_model_id="busbar_3", int_id=0, bus_branch_bus_id="node_3", bus_breaker_bus_id="node_3")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station_b = build_runtime_bus_group(
        grid_model_id="station_4",
        busbars=[RuntimeBusbar(grid_model_id="busbar_4", int_id=0, bus_branch_bus_id="node_4", bus_breaker_bus_id="node_4")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station_c = build_runtime_bus_group(
        grid_model_id="station_5",
        busbars=[RuntimeBusbar(grid_model_id="busbar_5", int_id=0, bus_branch_bus_id="node_5", bus_breaker_bus_id="node_5")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )
    island_station_d = build_runtime_bus_group(
        grid_model_id="station_6",
        busbars=[RuntimeBusbar(grid_model_id="busbar_6", int_id=0, bus_branch_bus_id="node_6", bus_breaker_bus_id="node_6")],
        couplers=[],
        branch_assets=[],
        injection_assets=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
    )

    network_data_dummy = replace(
        network_data,
        node_ids=["node_0", "node_1", "node_2", "node_3", "node_4", "node_5", "node_6"],
        from_nodes=np.array([0, 1, 2, 1, 3, 4, 5]),
        to_nodes=np.array([1, 2, 0, 3, 4, 5, 6]),
        branch_ids=["branch_01", "branch_12", "branch_20", "bridge_13", "branch_34", "branch_45", "branch_56"],
        bridging_branch_mask=np.array([False, False, False, True, False, False, False]),
        monitored_branch_mask=np.array([True, True, True, False, False, False, False]),
        slack=0,
        asset_topology=RuntimeAssetTopology(
            stations=[
                mainland_station_a,
                mainland_station_b,
                mainland_station_c,
                island_station_a,
                island_station_b,
                island_station_c,
                island_station_d,
            ],
        ),
    )

    outage_map = {
        "station_0": ["busbar_0"],
        "station_1": ["busbar_1"],
        "station_2": ["busbar_2"],
        "station_3": ["busbar_3"],
        "station_4": ["busbar_4"],
        "station_5": ["busbar_5"],
        "station_6": ["busbar_6"],
    }

    filtered_outage_map = exclude_island_busbars_from_outage_map(outage_map, network_data_dummy)

    assert filtered_outage_map == {
        "station_0": ["busbar_0"],
        "station_1": ["busbar_1"],
        "station_2": ["busbar_2"],
    }


def test_get_rel_bridge_busbars(mock_station: RuntimeBusGroup):
    articulation_nodes = get_rel_articulation_nodes([mock_station], [[[2, 3, 4]]])
    assert articulation_nodes == [[[3]]], f"Expected [[[3]]], but got {articulation_nodes}"

    articulation_nodes = get_rel_articulation_nodes([mock_station], [[[2, 3, 4], [2, 3, 4]]])
    assert articulation_nodes == [[[3], [3]]], f"Expected [[[3], [3]]], but got {articulation_nodes}"
