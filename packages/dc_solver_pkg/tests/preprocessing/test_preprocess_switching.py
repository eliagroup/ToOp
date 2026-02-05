# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path
from typing import get_args

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    add_missing_asset_topology_branch_info,
    add_missing_asset_topology_injection_info,
    identify_unnecessary_configurations,
    make_optimal_separation_set,
    make_separation_set,
    match_topology_to_network_data,
    prepare_for_separation_set,
)
from toop_engine_interfaces.asset_topology import (
    AssetBranchType,
    AssetInjectionType,
    Busbar,
    BusbarCoupler,
    Station,
    SwitchableAsset,
)


def test_match_topology_to_network_data(network_data_preprocessed: NetworkData) -> None:
    topology = network_data_preprocessed.asset_topology
    network_data = network_data_preprocessed
    relevant_node_ids = [node for (node, mask) in zip(network_data.node_ids, network_data.relevant_node_mask) if mask]
    topology = match_topology_to_network_data(
        topology=topology,
        branches_at_nodes=network_data.branches_at_nodes,
        injections_at_nodes=network_data.injection_idx_at_nodes,
        branch_ids=network_data.branch_ids,
        injection_ids=network_data.injection_ids,
        relevant_node_ids=relevant_node_ids,
    )
    assert [station.grid_model_id for station in topology.stations] == relevant_node_ids
    for station, branches_at_node, injections_at_node in zip(
        topology.stations,
        network_data.branches_at_nodes,
        network_data.injection_idx_at_nodes,
    ):
        assert len(station.assets) == len(branches_at_node) + len(injections_at_node)

    # Works with a filter in place
    topology2 = match_topology_to_network_data(
        topology=topology,
        branches_at_nodes=network_data.branches_at_nodes,
        injections_at_nodes=network_data.injection_idx_at_nodes,
        branch_ids=network_data.branch_ids,
        injection_ids=network_data.injection_ids,
        relevant_node_ids=relevant_node_ids,
        filter_assets="branch",
    )
    for station, branches_at_node in zip(topology2.stations, network_data.branches_at_nodes):
        assert len(station.assets) == len(branches_at_node)

    topology3 = match_topology_to_network_data(
        topology=topology,
        branches_at_nodes=network_data.branches_at_nodes,
        injections_at_nodes=network_data.injection_idx_at_nodes,
        branch_ids=network_data.branch_ids,
        injection_ids=network_data.injection_ids,
        relevant_node_ids=relevant_node_ids,
        filter_assets="injection",
    )
    for station, injections_at_node in zip(topology3.stations, network_data.injection_idx_at_nodes):
        assert len(station.assets) == len(injections_at_node)


def test_make_configurations_table():
    station = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
            Busbar(int_id=4, grid_model_id="busbar4"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            BusbarCoupler(busbar_from_id=3, busbar_to_id=4, open=False, grid_model_id="coupler3"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_switching_table=np.array(
            [
                [True, False, False, False, True],
                [False, False, False, True, True],
                [False, False, True, False, True],
                [False, True, False, False, True],
            ]
        ),
        grid_model_id="station1",
    )

    preprocessed_station, problems = prepare_for_separation_set(
        station, branch_ids=["line1", "line2", "line3", "line4", "line5"], injection_ids=[]
    )
    assert len(problems.multi_connected_assets) == 3
    assert problems.disconnected_busbars is None
    assert problems.duplicate_couplers is None
    assert preprocessed_station.busbars == station.busbars
    assert preprocessed_station.assets == station.assets
    assert len(preprocessed_station.couplers) == 3
    assert all([not coupler.open for coupler in preprocessed_station.couplers])

    configurations_table, coupler_states, assignment = make_separation_set(preprocessed_station)

    assert configurations_table.shape == (3, 2, 5)
    assert coupler_states.shape == (3, 3)
    assert len(assignment) == 3

    table = preprocessed_station.asset_switching_table

    # Coupler 3 open
    config_1_a = table[0] | table[1] | table[2]
    config_1_b = table[3]
    config_1 = np.stack([config_1_a, config_1_b], axis=0)
    # Coupler 2 open
    config_2_a = table[0] | table[1]
    config_2_b = table[2] | table[3]
    config_2 = np.stack([config_2_a, config_2_b], axis=0)
    # Coupler 1 open
    config_3_a = table[0]
    config_3_b = table[1] | table[2] | table[3]
    config_3 = np.stack([config_3_a, config_3_b], axis=0)

    # We expect to find all configurations in the table
    assert any(np.array_equal(config, config_1) for config in configurations_table)
    assert any(np.array_equal(config, config_2) for config in configurations_table)
    assert any(np.array_equal(config, config_3) for config in configurations_table)

    configurations_table = configurations_table.reshape(6, 5)
    assert any(np.array_equal(config, config_1_a) for config in configurations_table)
    assert any(np.array_equal(config, config_1_b) for config in configurations_table)
    assert any(np.array_equal(config, config_2_a) for config in configurations_table)
    assert any(np.array_equal(config, config_2_b) for config in configurations_table)
    assert any(np.array_equal(config, config_3_a) for config in configurations_table)
    assert any(np.array_equal(config, config_3_b) for config in configurations_table)

    # We expect exactly one coupler to be open in each configuration
    assert np.all(np.sum(coupler_states, axis=1) == 1)
    # It's a different coupler each time
    assert np.all(np.sum(coupler_states, axis=0) == 1)


def test_identify_unnecessary_combinations() -> None:
    configurations = np.array(
        [
            [True, False, True, False, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, True, False, False, True],
            [True, False, True, False, True],
        ]
    )

    mask = identify_unnecessary_configurations(configurations, clip_hamming_distance=0)

    assert np.array_equal(mask, [True, True, True, True, False])

    mask = identify_unnecessary_configurations(configurations, clip_hamming_distance=1)

    assert np.array_equal(mask, [True, True, False, True, False])

    mask = identify_unnecessary_configurations(configurations, clip_hamming_distance=2)

    assert np.array_equal(mask, [True, False, False, False, False])


def test_preprocess_station() -> None:
    station = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            BusbarCoupler(busbar_from_id=3, busbar_to_id=1, open=False, grid_model_id="coupler3"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_switching_table=jnp.array(
            [
                [True, True, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ]
        ),
        grid_model_id="station1",
    )

    configurations_table, coupler_states, coupler_distance, busbar_a = make_optimal_separation_set(station)
    n_configs = configurations_table.shape[0]

    assert configurations_table.shape == (n_configs, 2, 5)
    assert coupler_states.shape == (n_configs, 3)
    assert coupler_distance.shape == (n_configs,)
    assert len(busbar_a) == n_configs
    assert all([len(a) <= 2 for a in busbar_a])


def test_add_missing_asset_topology_branch_info(network_data: NetworkData) -> None:
    num_assets_before = sum(len(station.assets) for station in network_data.asset_topology.stations)

    topo = add_missing_asset_topology_branch_info(
        asset_topology=network_data.asset_topology,
        branch_ids=network_data.branch_ids,
        branch_names=network_data.branch_names,
        branch_types=network_data.branch_types,
        branch_from_nodes=[network_data.node_ids[i] for i in network_data.from_nodes],
        overwrite_if_present=True,
    )

    from_ends = 0
    to_ends = 0
    for station in topo.stations:
        for asset in station.assets:
            if asset.grid_model_id in network_data.branch_ids:
                assert asset.name in network_data.branch_names
                assert asset.type in network_data.branch_types
                assert asset.branch_end in ["from", "to"]
                if asset.branch_end == "from":
                    from_ends += 1
                else:
                    to_ends += 1

    assert from_ends > 0
    assert to_ends > 0

    num_assets_after = sum(len(station.assets) for station in topo.stations)
    assert num_assets_before == num_assets_after


def test_add_missing_asset_topology_injection_info(network_data: NetworkData) -> None:
    num_assets_before = sum(len(station.assets) for station in network_data.asset_topology.stations)

    topo = add_missing_asset_topology_injection_info(
        asset_topology=network_data.asset_topology,
        injection_ids=network_data.injection_ids,
        injection_names=network_data.injection_names,
        injection_types=network_data.injection_types,
        overwrite_if_present=True,
    )

    for station in topo.stations:
        for asset in station.assets:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name in network_data.injection_names
                assert asset.type in network_data.injection_types

    num_assets_after = sum(len(station.assets) for station in topo.stations)
    assert num_assets_before == num_assets_after

    # Test with overwrite_if_present=False
    topo = add_missing_asset_topology_injection_info(
        asset_topology=topo,
        injection_ids=network_data.injection_ids,
        injection_names=["new_name"] * len(network_data.injection_ids),
        injection_types=["new_type"] * len(network_data.injection_ids),
        overwrite_if_present=False,
    )

    for station in topo.stations:
        for asset in station.assets:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name in network_data.injection_names
                assert asset.type in network_data.injection_types

    topo = add_missing_asset_topology_injection_info(
        asset_topology=topo,
        injection_ids=network_data.injection_ids,
        injection_names=["new_name"] * len(network_data.injection_ids),
        injection_types=["new_type"] * len(network_data.injection_ids),
        overwrite_if_present=True,
    )

    for station in topo.stations:
        for asset in station.assets:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name == "new_name"
                assert asset.type == "new_type"


def test_prepare_for_separation_set_node_breaker_test_station():
    file = Path(__file__).parent.parent / "files" / "test_station.json"
    with open(file, "r") as f:
        station = Station.model_validate_json(f.read())

    x = nx.Graph()
    for busbar in station.busbars:
        x.add_node(busbar.int_id)
    for coupler in station.couplers:
        if coupler.type != "DISCONNECTOR":
            continue
        x.add_edge(coupler.busbar_from_id, coupler.busbar_to_id)
    connected_components = list(nx.connected_components(x))
    assert len(connected_components) > 1

    with pytest.raises(ValueError, match="no couplers left after preprocessing"):
        prepare_for_separation_set(
            station,
            branch_ids=[asset.grid_model_id for asset in station.assets if asset.type in get_args(AssetBranchType)],
            injection_ids=[asset.grid_model_id for asset in station.assets if asset.type in get_args(AssetInjectionType)],
        )

    station = station.model_copy(
        update={
            "busbars": [b.model_copy(update={"in_service": True}) for b in station.busbars],
        }
    )

    preprocessed_station, problems = prepare_for_separation_set(
        station,
        branch_ids=[asset.grid_model_id for asset in station.assets if asset.type in get_args(AssetBranchType)],
        injection_ids=[asset.grid_model_id for asset in station.assets if asset.type in get_args(AssetInjectionType)],
        close_couplers=True,
    )
    assert len(preprocessed_station.couplers)
