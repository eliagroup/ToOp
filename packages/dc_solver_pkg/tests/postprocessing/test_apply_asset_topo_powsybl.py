# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import (
    apply_node_breaker_topology,
    apply_single_asset_bus_branch,
    apply_station,
    apply_station_bus_branch,
    apply_topology_bus_branch,
    find_asset,
)
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl
from toop_engine_interfaces.asset_topology import (
    Topology,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_find_asset(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    station = topology.stations[0]
    bus_id = station.grid_model_id
    vl_id = net.get_buses().loc[bus_id]["voltage_level_id"]

    for elem in station.assets:
        is_branch, connected, direction, bus_breaker_id = find_asset(
            net=net,
            elem_id=elem.grid_model_id,
            voltage_level_id=vl_id,
            bus_id=bus_id,
        )

        assert is_branch == elem.is_branch()
        assert connected
        if is_branch:
            brh = net.get_branches(all_attributes=True).loc[elem.grid_model_id]
            if direction:
                assert brh["bus1_id"] == bus_id
                assert brh["bus_breaker_bus1_id"] == bus_breaker_id
            else:
                assert brh["bus2_id"] == bus_id
                assert brh["bus_breaker_bus2_id"] == bus_breaker_id
        else:
            inj = net.get_injections(all_attributes=True).loc[elem.grid_model_id]
            assert inj["bus_id"] == bus_id
            assert inj["bus_breaker_bus_id"] == bus_breaker_id

    for elem in topology.stations[10].assets:
        if elem.is_branch():
            with pytest.raises(ValueError, match=f"Branch {elem.grid_model_id} not found in the station"):
                find_asset(
                    net=net,
                    elem_id=elem.grid_model_id,
                    voltage_level_id=vl_id,
                    bus_id=bus_id,
                )
        else:
            with pytest.raises(ValueError, match=f"Element {elem.grid_model_id} not found in the station."):
                find_asset(
                    net=net,
                    elem_id=elem.grid_model_id,
                    voltage_level_id=vl_id,
                    bus_id=bus_id,
                )


def test_apply_single_asset_bus_branch_reassign(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    for asset_index in range(len(topology.stations[0].assets)):
        station = topology.stations[0].model_copy(deep=True)
        # Reassign the first asset to the second bus
        assert len(station.busbars) == 2
        station.asset_switching_table[:, asset_index] = ~station.asset_switching_table[:, asset_index]
        status, reassignments = apply_single_asset_bus_branch(
            net=net,
            station=station,
            asset_index=asset_index,
        )
        assert status == "reassigned"
        assert len(reassignments) == 2
        for re_asset_index, bus_id, connected in reassignments:
            assert re_asset_index == asset_index
            assert connected == station.asset_switching_table[bus_id, asset_index]

        new_bus = [
            bus
            for (bus, switching) in zip(station.busbars, station.asset_switching_table[:, asset_index].tolist(), strict=True)
            if switching
        ]
        assert len(new_bus) == 1
        new_bus = new_bus[0]
        if station.assets[asset_index].is_branch():
            brh = net.get_branches(all_attributes=True).loc[station.assets[asset_index].grid_model_id]
            assert brh["bus_breaker_bus1_id"] == new_bus.grid_model_id or brh["bus_breaker_bus2_id"] == new_bus.grid_model_id
        else:
            inj = net.get_injections(all_attributes=True).loc[station.assets[asset_index].grid_model_id]
            assert inj["bus_breaker_bus_id"] == new_bus.grid_model_id


def test_apply_single_asset_bus_branch_disconnect(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    for asset_index in range(len(topology.stations[0].assets)):
        station = topology.stations[0].model_copy(deep=True)
        # Disconnect the first asset
        assert len(station.busbars) == 2
        station.asset_switching_table[:, asset_index] = False
        status, reassignments = apply_single_asset_bus_branch(
            net=net,
            station=station,
            asset_index=asset_index,
        )
        assert status == "disconnected"
        assert len(reassignments) == 0

        if station.assets[asset_index].is_branch():
            brh = net.get_branches(all_attributes=True).loc[station.assets[asset_index].grid_model_id]
            assert not brh["connected1"] and not brh["connected2"]
        else:
            inj = net.get_injections(all_attributes=True).loc[station.assets[asset_index].grid_model_id]
            assert not inj["connected"]


def test_apply_single_asset_bus_branch_nothing(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    for asset_index in range(len(topology.stations[0].assets)):
        status, reassignments = apply_single_asset_bus_branch(
            net=net,
            station=topology.stations[0],
            asset_index=asset_index,
        )
        assert status == "nothing"
        assert len(reassignments) == 0


def test_apply_station_bus_branch_reassign(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Try do nothing
    station = topology.stations[0].model_copy(deep=True)
    realized_station = apply_station_bus_branch(
        net=net,
        station=station,
    )
    assert len(realized_station.coupler_diff) == 0
    assert len(realized_station.reassignment_diff) == 0
    assert len(realized_station.disconnection_diff) == 0

    realized_station_2 = apply_station(net=net, station=station)
    assert realized_station_2 == realized_station

    # Try with disconnections and reassignments
    assert np.array_equal(station.asset_switching_table, np.array([[True, True, True], [False, False, False]]))

    station.asset_switching_table = np.array([[True, False, False], [False, True, False]])
    realized_station = apply_station_bus_branch(
        net=net,
        station=station,
    )
    assert len(realized_station.coupler_diff) == 0
    assert len(realized_station.reassignment_diff) == 2
    assert len(realized_station.disconnection_diff) == 1


def test_apply_station_bus_branch_coupler(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    station = topology.stations[0].model_copy(deep=True)
    station.asset_switching_table = np.array([[True, False, True], [False, True, False]])
    station.couplers[0].open = True

    realized_station = apply_station_bus_branch(
        net=net,
        station=station,
    )
    assert len(realized_station.coupler_diff) == 1
    assert realized_station.coupler_diff[0] == station.couplers[0]
    assert len(realized_station.reassignment_diff) == 2
    assert len(realized_station.disconnection_diff) == 0


def test_apply_node_breaker_topology(basic_node_breaker_topology: Topology) -> None:
    net = basic_node_breaker_network_powsybl()

    switch_update_df = apply_node_breaker_topology(net, basic_node_breaker_topology)
    assert isinstance(switch_update_df, pd.DataFrame)

    sw = net.get_switches()
    # busbar coupler open
    assert sw.loc["VL4_BREAKER", "open"]
    # switched line L4
    assert sw.loc["L42_DISCONNECTOR_3_0", "open"]
    assert not sw.loc["L42_DISCONNECTOR_3_1", "open"]
    assert not sw.loc["L42_BREAKER", "open"]
    # line not switched L5, diff was as same as the current state
    assert sw.loc["L52_DISCONNECTOR_5_0", "open"]
    assert not sw.loc["L52_DISCONNECTOR_5_1", "open"]
    assert not sw.loc["L52_BREAKER", "open"]
    # line disconnected L8 -> disconnector stay, breaker open
    assert not sw.loc["L82_DISCONNECTOR_7_0", "open"]
    assert sw.loc["L82_DISCONNECTOR_7_1", "open"]
    assert sw.loc["L82_BREAKER", "open"]

    net = basic_node_breaker_network_powsybl()
    switch_update_df_2 = apply_station(net, basic_node_breaker_topology.stations[0])
    assert isinstance(switch_update_df_2, pd.DataFrame)
    assert switch_update_df_2.isin(switch_update_df).all().all()


def test_apply_topology_bus_branch_do_nothing(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Try do nothing
    realized_topology = apply_topology_bus_branch(
        net=net,
        topology=topology,
    )
    assert len(realized_topology.coupler_diff) == 0
    assert len(realized_topology.reassignment_diff) == 0
    assert len(realized_topology.disconnection_diff) == 0


def test_apply_topology_bus_branch_reassign(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    np.random.seed(0)

    # Try with disconnections and reassignments
    # Just randomize the switching tables, we'll have both disconnections and reassignments then
    for station in topology.stations:
        station.asset_switching_table = np.random.randint(0, 2, size=station.asset_switching_table.shape, dtype=bool)

    realized_topology = apply_topology_bus_branch(
        net=net,
        topology=topology,
    )
    assert len(realized_topology.coupler_diff) == 0
    assert len(realized_topology.reassignment_diff) > 0
    assert len(realized_topology.disconnection_diff) > 0
