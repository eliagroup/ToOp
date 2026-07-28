# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
from pathlib import Path

import numpy as np
import pandapower as pp
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from pydantic import BaseModel
from tests.network_data_pickle import load_network_data, save_network_data
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_network_data_from_interface,
    get_relevant_stations,
    map_branch_injection_ids,
    map_runtime_stations_by_node_id,
)
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_interfaces.asset_topology.asset_topology import RuntimeAssetTopology
from toop_engine_interfaces.asset_topology.assets import Busbar
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def _assert_roundtrip_equal(original: object, loaded: object) -> None:
    """Recursively compare serialized and deserialized network-data payloads."""
    if isinstance(original, np.ndarray):
        assert isinstance(loaded, np.ndarray)
        assert np.array_equal(original, loaded)
        return

    if isinstance(original, BaseModel):
        assert isinstance(loaded, type(original))
        for field_name in original.model_fields:
            _assert_roundtrip_equal(getattr(original, field_name), getattr(loaded, field_name))
        return

    if isinstance(original, list):
        assert isinstance(loaded, list)
        assert len(original) == len(loaded)
        for original_item, loaded_item in zip(original, loaded, strict=True):
            _assert_roundtrip_equal(original_item, loaded_item)
        return

    assert original == loaded


def test_extract_network_data(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = extract_network_data_from_interface(backend)
    assert network_data is not None

    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    n_branch = (
        net.line.in_service.sum()
        + net.trafo.in_service.sum()
        + net.trafo3w.in_service.sum() * 3
        + net.impedance.in_service.sum()
        + net.xward.in_service.sum()
    )
    n_node = net.bus.in_service.sum() + net.trafo3w.in_service.sum() + net.xward.in_service.sum()
    n_injection = (
        net.load.in_service.sum()
        + net.gen.in_service.sum()
        + net.sgen.in_service.sum()
        + net.xward.in_service.sum() * 3
        + net.ward.in_service.sum() * 2
        + net.shunt.in_service.sum()
        + net.dcline.in_service.sum() * 2
    )
    n_timestep = 1

    assert len(network_data.branch_ids) == (n_branch)
    assert len(network_data.node_ids) == (n_node)
    assert len(network_data.branch_names) == (n_branch)
    assert len(network_data.node_names) == (n_node)
    assert network_data.from_nodes.shape == (n_branch,)
    assert np.all(network_data.from_nodes >= 0)
    assert np.all(network_data.from_nodes < n_node)
    assert network_data.to_nodes.shape == (n_branch,)
    assert np.all(network_data.to_nodes >= 0)
    assert np.all(network_data.to_nodes < n_node)
    assert network_data.susceptances.shape == (n_branch,)
    assert np.all(network_data.susceptances != 0)
    assert network_data.phase_shift_mask.shape == (n_branch,)
    assert len(network_data.branch_types) == (n_branch)
    assert network_data.max_mw_flows.shape == (n_timestep, n_branch)
    assert np.all(network_data.max_mw_flows > 0)
    assert network_data.monitored_branch_mask.shape == (n_branch,)
    assert network_data.disconnectable_branch_mask.shape == (n_branch)
    assert network_data.outaged_branch_mask.shape == (n_branch,)
    assert network_data.relevant_node_mask.shape == (n_node,)
    assert len(network_data.injection_ids) == (n_injection)
    assert network_data.injection_nodes.shape == (n_injection,)
    assert len(network_data.injection_names) == (n_injection)
    assert network_data.mw_injections.shape == (n_timestep, n_injection)
    assert network_data.outaged_injection_mask.shape == (n_injection,)


def test_load_save(network_data: NetworkData, tmp_path: str) -> None:
    filename = os.path.join(tmp_path, "test.pkl")
    save_network_data(filename, network_data)
    network_data_loaded = load_network_data(filename)
    for key in network_data.__dict__.keys():
        assert type(getattr(network_data, key)) is type(getattr(network_data_loaded, key)), (
            f"type of {key} differs between save and load"
        )
        _assert_roundtrip_equal(getattr(network_data, key), getattr(network_data_loaded, key))


def test_get_relevant_stations(network_data_preprocessed: NetworkData) -> None:
    rel_stations = get_relevant_stations(network_data_preprocessed)
    assert len(rel_stations) == len(network_data_preprocessed.relevant_nodes)
    for station in rel_stations:
        assert network_data_preprocessed.node_ids.index(station.bus_group_id) in network_data_preprocessed.relevant_nodes


def test_get_relevant_stations_requires_runtime_stations_when_master_data_exists(
    node_breaker_grid_imported_data_folder: Path,
) -> None:
    """Verify that relevant-station lookup rejects missing runtime stations when master data exists."""
    filesystem_dir = DirFileSystem(str(node_breaker_grid_imported_data_folder))
    backend = PowsyblBackend(filesystem_dir)
    network_data = extract_network_data_from_interface(backend)
    network_data = NetworkData(**{**network_data.__dict__, "asset_topology": None})

    with pytest.raises(AssertionError, match="Missing runtime asset-topology stations"):
        get_relevant_stations(network_data)


def test_get_relevant_stations_requires_complete_runtime_station_coverage(
    network_data_preprocessed: NetworkData,
) -> None:
    """Verify that relevant-station lookup rejects incomplete runtime-station coverage."""
    assert network_data_preprocessed.asset_topology is not None
    relevant_node_ids = {
        node_id
        for node_id, is_relevant in zip(
            network_data_preprocessed.node_ids,
            network_data_preprocessed.relevant_node_mask,
            strict=True,
        )
        if is_relevant
    }

    removed_station_id = next(
        station.bus_group_id
        for station in network_data_preprocessed.asset_topology.stations
        if station.bus_group_id in relevant_node_ids
    )
    incomplete_stations = [
        station
        for station in network_data_preprocessed.asset_topology.stations
        if station.bus_group_id != removed_station_id
    ]
    network_data = NetworkData(
        **{
            **network_data_preprocessed.__dict__,
            "asset_topology": RuntimeAssetTopology(
                stations=incomplete_stations,
                circuit_groups=network_data_preprocessed.asset_topology.circuit_groups,
            ),
        }
    )

    with pytest.raises(ValueError, match="Missing runtime asset-topology stations for relevant station extraction"):
        get_relevant_stations(network_data)


def test_map_runtime_stations_by_node_id_prefers_explicit_bus_ids() -> None:
    """Verify that inactive sibling groups do not steal a live electrical bus via fallback aliases."""
    active_station = MaterializedStation(
        bus_group_id="voltage_level_a",
        voltage_level_id="voltage_level",
        busbars=[Busbar(grid_model_id="bbs_a", int_id=0, bus_branch_bus_id="voltage_level_0")],
        couplers=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
        branch_connectivity=np.zeros((1, 0), dtype=bool),
        injection_connectivity=np.zeros((1, 0), dtype=bool),
        branch_connections=[],
        injection_connections=[],
    )
    detached_station_b = MaterializedStation(
        bus_group_id="voltage_level_b",
        voltage_level_id="voltage_level",
        busbars=[Busbar(grid_model_id="bbs_b", int_id=0, bus_branch_bus_id="")],
        couplers=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
        branch_connectivity=np.zeros((1, 0), dtype=bool),
        injection_connectivity=np.zeros((1, 0), dtype=bool),
        branch_connections=[],
        injection_connections=[],
    )
    detached_station_c = MaterializedStation(
        bus_group_id="voltage_level_c",
        voltage_level_id="voltage_level",
        busbars=[Busbar(grid_model_id="bbs_c", int_id=0, bus_branch_bus_id="")],
        couplers=[],
        branch_switching_table=np.zeros((1, 0), dtype=bool),
        injection_switching_table=np.zeros((1, 0), dtype=bool),
        branch_connectivity=np.zeros((1, 0), dtype=bool),
        injection_connectivity=np.zeros((1, 0), dtype=bool),
        branch_connections=[],
        injection_connections=[],
    )

    stations_by_node_id = map_runtime_stations_by_node_id(
        [active_station, detached_station_b, detached_station_c],
        node_ids=["voltage_level_0"],
    )

    assert stations_by_node_id["voltage_level_0"].bus_group_id == "voltage_level_a"


def test_map_branch_injection_ids(network_data_preprocessed: NetworkData) -> None:
    branch_ids_mapped, injection_ids_mapped = map_branch_injection_ids(network_data_preprocessed)
    assert len(branch_ids_mapped) == len(network_data_preprocessed.relevant_nodes)
    assert len(injection_ids_mapped) == len(network_data_preprocessed.relevant_nodes)
    for sub_idx, branches_local in enumerate(branch_ids_mapped):
        assert len(branches_local) == len(network_data_preprocessed.branches_at_nodes[sub_idx])
        assert np.all(
            np.array(network_data_preprocessed.branch_ids)[network_data_preprocessed.branches_at_nodes[sub_idx]]
            == branches_local
        )

    for sub_idx, injections_local in enumerate(injection_ids_mapped):
        assert len(injections_local) == len(network_data_preprocessed.injection_idx_at_nodes[sub_idx])
        assert np.all(
            np.array(network_data_preprocessed.injection_ids)[network_data_preprocessed.injection_idx_at_nodes[sub_idx]]
            == injections_local
        )


def test_extract_action_set_requires_realised_stations(
    network_data_preprocessed: NetworkData,
) -> None:
    """Verify that action-set extraction requires realized station payloads."""
    network_data_missing_realisations = NetworkData(**{**network_data_preprocessed.__dict__, "realised_stations": None})
    with pytest.raises(AssertionError, match="No realised stations in network data"):
        extract_action_set(network_data_missing_realisations)


def test_contingency_ids(network_data_preprocessed: NetworkData) -> None:
    contingency_ids = network_data_preprocessed.contingency_ids
    assert len(
        contingency_ids
    ) == network_data_preprocessed.outaged_branch_mask.sum() + network_data_preprocessed.outaged_injection_mask.sum() + len(
        network_data_preprocessed.multi_outage_ids
    )
