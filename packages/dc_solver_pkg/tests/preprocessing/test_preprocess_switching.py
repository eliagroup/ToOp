# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from pathlib import Path

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from beartype.typing import get_args
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    add_missing_asset_topology_branch_info,
    add_missing_asset_topology_injection_info,
    identify_unnecessary_configurations,
    make_optimal_separation_set,
    make_separation_set,
    prepare_for_separation_set,
)
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType
from toop_engine_interfaces.asset_topology.assets import Busbar, BusbarCoupler, SwitchableAsset
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation


def _combined_asset_connections(station: MaterializedStation) -> list[MaterializedAssetConnection]:
    return [*station.branch_connections, *station.injection_connections]


def _combined_asset_switching_table(station: MaterializedStation) -> np.ndarray:
    return np.concatenate([station.branch_switching_table, station.injection_switching_table], axis=1)


def build_materialized_station(
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    assets: list[SwitchableAsset],
    asset_switching_table: np.ndarray,
) -> MaterializedStation:
    return MaterializedStation(
        grid_model_id=grid_model_id,
        busbars=busbars,
        couplers=couplers,
        branch_connections=[MaterializedAssetConnection(asset=asset) for asset in assets],
        injection_connections=[],
        branch_switching_table=asset_switching_table,
        injection_switching_table=np.zeros((asset_switching_table.shape[0], 0), dtype=bool),
    )


def test_make_configurations_table():
    station = build_materialized_station(
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
    assert preprocessed_station.branch_connections == station.branch_connections
    assert len(preprocessed_station.couplers) == 3
    assert all([not coupler.open for coupler in preprocessed_station.couplers])

    configurations_table, coupler_states, assignment = make_separation_set(preprocessed_station)

    assert configurations_table.shape == (3, 2, 5)
    assert coupler_states.shape == (3, 3)
    assert len(assignment) == 3

    table = preprocessed_station.branch_switching_table

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
    station = build_materialized_station(
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
    separation_set = make_optimal_separation_set(station)
    configurations_table, coupler_states, coupler_distance, busbar_a = (
        separation_set.separation_set,
        separation_set.coupler_states,
        separation_set.coupler_distance,
        separation_set.busbar_a,
    )
    n_configs = configurations_table.shape[0]

    assert configurations_table.shape == (n_configs, 2, 5)
    assert coupler_states.shape == (n_configs, 3)
    assert coupler_distance.shape == (n_configs,)
    assert len(busbar_a) == n_configs
    assert all([len(a) <= 2 for a in busbar_a])


def test_add_missing_asset_topology_branch_info(network_data: NetworkData) -> None:
    num_assets_before = sum(
        len(station.branch_connections) for station in network_data.asset_topology.materialize_stations()
    )

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
    for station in topo.materialize_stations():
        for asset_connection in station.branch_connections:
            asset = asset_connection.asset
            if asset.grid_model_id in network_data.branch_ids:
                assert asset.name in network_data.branch_names
                assert asset.asset_type in network_data.branch_types
                assert asset_connection.terminal in ["from", "to"]
                if asset_connection.terminal == "from":
                    from_ends += 1
                else:
                    to_ends += 1

    assert from_ends > 0
    assert to_ends > 0

    num_assets_after = sum(len(station.branch_connections) for station in topo.materialize_stations())
    assert num_assets_before == num_assets_after


def test_add_missing_asset_topology_injection_info(network_data: NetworkData) -> None:
    num_assets_before = sum(
        len(_combined_asset_connections(station)) for station in network_data.asset_topology.materialize_stations()
    )
    replacement_injection_type = "load"

    topo = add_missing_asset_topology_injection_info(
        asset_topology=network_data.asset_topology,
        injection_ids=network_data.injection_ids,
        injection_names=network_data.injection_names,
        injection_types=network_data.injection_types,
        overwrite_if_present=True,
    )

    for station in topo.materialize_stations():
        for asset in [asset_connection.asset for asset_connection in _combined_asset_connections(station)]:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name in network_data.injection_names
                assert asset.asset_type in network_data.injection_types

    num_assets_after = sum(len(_combined_asset_connections(station)) for station in topo.materialize_stations())
    assert num_assets_before == num_assets_after

    # Test with overwrite_if_present=False
    topo = add_missing_asset_topology_injection_info(
        asset_topology=topo,
        injection_ids=network_data.injection_ids,
        injection_names=["new_name"] * len(network_data.injection_ids),
        injection_types=[replacement_injection_type] * len(network_data.injection_ids),
        overwrite_if_present=False,
    )

    for station in topo.materialize_stations():
        for asset in [asset_connection.asset for asset_connection in _combined_asset_connections(station)]:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name in network_data.injection_names
                assert asset.asset_type in network_data.injection_types

    topo = add_missing_asset_topology_injection_info(
        asset_topology=topo,
        injection_ids=network_data.injection_ids,
        injection_names=["new_name"] * len(network_data.injection_ids),
        injection_types=[replacement_injection_type] * len(network_data.injection_ids),
        overwrite_if_present=True,
    )

    for station in topo.materialize_stations():
        for asset in [asset_connection.asset for asset_connection in _combined_asset_connections(station)]:
            if asset.grid_model_id in network_data.injection_ids:
                assert asset.name == "new_name"
                assert asset.asset_type == replacement_injection_type


def test_prepare_for_separation_set_node_breaker_test_station():
    file = Path(__file__).parent.parent / "files" / "test_station.json"
    with open(file, "r") as f:
        station_data = json.load(f)

    station_data["station_type"] = station_data.pop("type")
    for busbar in station_data["busbars"]:
        busbar["busbar_type"] = busbar.pop("type")
    for coupler in station_data["couplers"]:
        coupler["coupler_type"] = coupler.pop("type")

    assets = station_data.pop("assets")
    for asset in assets:
        asset["asset_type"] = asset.pop("type")
    branch_mask = [asset["asset_type"] in get_args(AssetBranchType) for asset in assets]
    injection_mask = [asset["asset_type"] in get_args(AssetInjectionType) for asset in assets]
    asset_terminals = station_data.pop("asset_terminals", [None] * len(assets))
    asset_bays = station_data.pop("asset_bays", [None] * len(assets))
    combined_switching_table = np.asarray(station_data.pop("asset_switching_table"), dtype=bool)
    station_data["branch_connections"] = [
        {"asset": asset, "terminal": asset_terminal, "asset_bay": asset_bay}
        for asset, asset_terminal, asset_bay in zip(assets, asset_terminals, asset_bays, strict=True)
        if asset["asset_type"] in get_args(AssetBranchType)
    ]
    station_data["injection_connections"] = [
        {"asset": asset, "terminal": asset_terminal, "asset_bay": asset_bay}
        for asset, asset_terminal, asset_bay in zip(assets, asset_terminals, asset_bays, strict=True)
        if asset["asset_type"] in get_args(AssetInjectionType)
    ]
    station_data["branch_switching_table"] = combined_switching_table[:, branch_mask]
    station_data["injection_switching_table"] = combined_switching_table[:, injection_mask]
    station = MaterializedStation.model_validate(station_data)

    x = nx.Graph()
    for busbar in station.busbars:
        x.add_node(busbar.int_id)
    for coupler in station.couplers:
        if coupler.coupler_type != "DISCONNECTOR":
            continue
        x.add_edge(coupler.busbar_from_id, coupler.busbar_to_id)
    connected_components = list(nx.connected_components(x))
    assert len(connected_components) > 1

    with pytest.raises(ValueError, match="no couplers left after preprocessing"):
        prepare_for_separation_set(
            station,
            branch_ids=[
                asset_connection.asset.grid_model_id
                for asset_connection in _combined_asset_connections(station)
                if asset_connection.asset.asset_type in get_args(AssetBranchType)
            ],
            injection_ids=[
                asset_connection.asset.grid_model_id
                for asset_connection in _combined_asset_connections(station)
                if asset_connection.asset.asset_type in get_args(AssetInjectionType)
            ],
        )

    station = station.model_copy(
        update={
            "busbars": [b.model_copy(update={"in_service": True}) for b in station.busbars],
        }
    )

    preprocessed_station, problems = prepare_for_separation_set(
        station,
        branch_ids=[
            asset_connection.asset.grid_model_id
            for asset_connection in _combined_asset_connections(station)
            if asset_connection.asset.asset_type in get_args(AssetBranchType)
        ],
        injection_ids=[
            asset_connection.asset.grid_model_id
            for asset_connection in _combined_asset_connections(station)
            if asset_connection.asset.asset_type in get_args(AssetInjectionType)
        ],
        close_couplers=True,
    )
    assert len(preprocessed_station.couplers)
