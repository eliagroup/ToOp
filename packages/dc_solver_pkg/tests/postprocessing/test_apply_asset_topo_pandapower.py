# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import pandapower as pp
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.postprocess.apply_asset_topo_pandapower import apply_bus_group, apply_topology_bus_groups
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_apply_station(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runtime_topology = PandaPowerBackend(DirFileSystem(str(case14_data_folder))).get_runtime_asset_topology()
    assert runtime_topology is not None
    topology_stations = runtime_topology.bus_groups

    # Make sure we have valid busbar ids
    # Currently only one bus exists in the station, so we expect the method to create the coupler and the missing busbar.
    station = topology_stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    # Apply the station topology
    apply_diff, realized_station = apply_bus_group(net, station)

    assert len(apply_diff.busbars_created) == 1
    assert len(apply_diff.switches_created) == 1
    assert len(apply_diff.busbars_deleted) == 0
    assert len(apply_diff.switches_deleted) == 0
    assert len(realized_station.branch_disconnection_diff) == 0
    assert len(realized_station.injection_disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 0
    assert len(realized_station.branch_reassignment_diff) == 0
    assert len(realized_station.injection_reassignment_diff) == 0


def test_apply_station_existing_buses(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runtime_topology = PandaPowerBackend(DirFileSystem(str(case14_data_folder))).get_runtime_asset_topology()
    assert runtime_topology is not None
    topology_stations = runtime_topology.bus_groups

    station = topology_stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    net.bus.loc[15] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus B"}
    net.switch.loc[1] = {"closed": False, "bus": 1, "element": 15, "et": "b", "name": "Switch B"}

    (apply_diff, realized_station) = apply_bus_group(net, station)
    assert len(apply_diff.busbars_created) == 0
    assert len(apply_diff.switches_created) == 0
    assert len(apply_diff.busbars_deleted) == 0
    assert len(apply_diff.switches_deleted) == 0
    assert len(realized_station.branch_disconnection_diff) == 0
    assert len(realized_station.injection_disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 1
    assert len(realized_station.branch_reassignment_diff) == 0
    assert len(realized_station.injection_reassignment_diff) == 0


def test_apply_station_extra_busbar(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runtime_topology = PandaPowerBackend(DirFileSystem(str(case14_data_folder))).get_runtime_asset_topology()
    assert runtime_topology is not None
    topology_stations = runtime_topology.bus_groups

    station = topology_stations[0].model_copy()
    station.busbars[0].grid_model_id = f"1{SEPARATOR}bus"
    station.busbars[1].grid_model_id = f"15{SEPARATOR}bus"
    station.couplers[0].grid_model_id = f"1{SEPARATOR}switch"

    net.bus.loc[15] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus B"}
    net.bus.loc[16] = {"vn_kv": net.bus.loc[1, "vn_kv"], "in_service": True, "name": "Bus C"}
    net.switch.loc[1] = {"closed": False, "bus": 1, "element": 15, "et": "b", "name": "Switch B"}
    net.switch.loc[2] = {"closed": False, "bus": 1, "element": 16, "et": "b", "name": "Switch C"}

    (apply_diff, realized_station) = apply_bus_group(net, station)
    assert len(apply_diff.busbars_created) == 0
    assert len(apply_diff.switches_created) == 0
    assert apply_diff.busbars_deleted == [16]
    assert apply_diff.switches_deleted == [2]
    assert len(realized_station.branch_disconnection_diff) == 0
    assert len(realized_station.injection_disconnection_diff) == 0
    assert len(realized_station.coupler_diff) == 1
    assert len(realized_station.branch_reassignment_diff) == 0
    assert len(realized_station.injection_reassignment_diff) == 0


def test_apply_topology(case14_data_folder: Path) -> None:
    net = pp.from_json(case14_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runtime_topology = PandaPowerBackend(DirFileSystem(str(case14_data_folder))).get_runtime_asset_topology()
    assert runtime_topology is not None
    topology_stations = runtime_topology.bus_groups

    # Apply the topology
    apply_diff, realized_topology = apply_topology_bus_groups(net, topology_stations)

    for station_id, local_apply_diff in apply_diff:
        assert len(local_apply_diff.busbars_created) == 1
        assert len(local_apply_diff.switches_created) == 1
        assert len(local_apply_diff.busbars_deleted) == 0
        assert len(local_apply_diff.switches_deleted) == 0

    assert len(realized_topology.branch_disconnection_diff) == 0
    assert len(realized_topology.injection_disconnection_diff) == 0
    assert len(realized_topology.coupler_diff) == 0
    assert len(realized_topology.branch_reassignment_diff) == 0
    assert len(realized_topology.injection_reassignment_diff) == 0
    assert realized_topology.master_data is None
