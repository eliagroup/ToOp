# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import pandapower as pp
from toop_engine_dc_solver.postprocess.apply_asset_topo_pandapower import apply_station, apply_topology
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_apply_station(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    with open(case14_data_folder / PREPROCESSING_PATHS["asset_topology_file_path"]) as f:
        asset_topo = Topology.model_validate_json(f.read())

    # Make sure we have valid busbar ids
    # Currently only one bus exists in the station, so we expect the method to create the coupler and the missing busbar.
    station = asset_topo.stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    # Apply the station topology
    apply_diff, realized_station = apply_station(net, station)

    assert len(apply_diff.busbars_created) == 1
    assert len(apply_diff.switches_created) == 1
    assert len(apply_diff.busbars_deleted) == 0
    assert len(apply_diff.switches_deleted) == 0
    assert len(realized_station.disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 0
    assert len(realized_station.reassignment_diff)


def test_apply_station_existing_buses(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    with open(case14_data_folder / PREPROCESSING_PATHS["asset_topology_file_path"]) as f:
        asset_topo = Topology.model_validate_json(f.read())

    station = asset_topo.stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    net.bus.loc[15] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus B"}
    net.switch.loc[1] = {"closed": False, "bus": 1, "element": 15, "et": "b", "name": "Switch B"}

    (apply_diff, realized_station) = apply_station(net, station)
    assert len(apply_diff.busbars_created) == 0
    assert len(apply_diff.switches_created) == 0
    assert len(apply_diff.busbars_deleted) == 0
    assert len(apply_diff.switches_deleted) == 0
    assert len(realized_station.disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 1
    assert len(realized_station.reassignment_diff)


def test_apply_station_extra_busbar(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    with open(case14_data_folder / PREPROCESSING_PATHS["asset_topology_file_path"]) as f:
        asset_topo = Topology.model_validate_json(f.read())

    station = asset_topo.stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    net.bus.loc[15] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus B"}
    net.bus.loc[16] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus C"}
    net.switch.loc[1] = {"closed": False, "bus": 1, "element": 15, "et": "b", "name": "Switch B"}
    net.switch.loc[2] = {"closed": False, "bus": 1, "element": 16, "et": "b", "name": "Switch C"}

    (apply_diff, realized_station) = apply_station(net, station)
    assert len(apply_diff.busbars_created) == 0
    assert len(apply_diff.switches_created) == 0
    assert apply_diff.busbars_deleted == [16]
    assert apply_diff.switches_deleted == [2]
    assert len(realized_station.disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 1
    assert len(realized_station.reassignment_diff)


def test_apply_topology(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    with open(case14_data_folder / PREPROCESSING_PATHS["asset_topology_file_path"]) as f:
        asset_topo = Topology.model_validate_json(f.read())

    # Apply the topology
    apply_diff, realized_topology = apply_topology(net, asset_topo)

    for station_id, local_apply_diff in apply_diff:
        assert len(local_apply_diff.busbars_created) == 1
        assert len(local_apply_diff.switches_created) == 1
        assert len(local_apply_diff.busbars_deleted) == 0
        assert len(local_apply_diff.switches_deleted) == 0

    assert len(realized_topology.disconnection_diff) == 0
    assert len(realized_topology.coupler_diff) == 0
    assert len(realized_topology.reassignment_diff)
    assert realized_topology.topology == asset_topo
