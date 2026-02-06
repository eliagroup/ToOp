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
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_network_data_from_interface,
    get_monitored_node_ids,
    get_relevant_stations,
    load_network_data,
    map_branch_injection_ids,
    save_network_data,
)
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


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
        if isinstance(getattr(network_data, key), np.ndarray):
            assert np.array_equal(getattr(network_data, key), getattr(network_data_loaded, key))
        elif (
            isinstance(getattr(network_data, key), list)
            and len(getattr(network_data, key))
            and isinstance(getattr(network_data, key)[0], np.ndarray)
        ):
            for i in range(len(getattr(network_data, key))):
                assert np.array_equal(getattr(network_data, key)[i], getattr(network_data_loaded, key)[i])
        else:
            assert getattr(network_data, key) == getattr(network_data_loaded, key)


def test_get_monitored_node_ids(network_data: NetworkData) -> None:
    relevant_node_ids = list(set(np.array(network_data.node_ids)[network_data.relevant_node_mask].tolist()))
    relevant_node_ids.sort()
    monitored_branch_end_ids = set(
        np.array(network_data.node_ids)[network_data.from_nodes[network_data.monitored_branch_mask]].tolist()
    )
    monitored_branch_end_ids.update(
        np.array(network_data.node_ids)[network_data.to_nodes[network_data.monitored_branch_mask]].tolist()
    )
    monitored_branch_end_ids = list(monitored_branch_end_ids)
    monitored_branch_end_ids.sort()
    union = list(set(relevant_node_ids).union(set(monitored_branch_end_ids)))
    union.sort()

    # Test with both include_relevant_nodes and include_monitored_branch_end set to True
    monitored_node_ids = get_monitored_node_ids(network_data, True, True)
    assert isinstance(monitored_node_ids, list)
    assert all(isinstance(node_id, str) for node_id in monitored_node_ids)
    assert monitored_node_ids == union

    # Test with include_relevant_nodes set to True and include_monitored_branch_end set to False
    monitored_node_ids = get_monitored_node_ids(network_data, True, False)
    assert isinstance(monitored_node_ids, list)
    assert monitored_node_ids == relevant_node_ids

    # Test with include_relevant_nodes set to False and include_monitored_branch_end set to True
    monitored_node_ids = get_monitored_node_ids(network_data, False, True)
    assert isinstance(monitored_node_ids, list)
    assert monitored_node_ids == monitored_branch_end_ids

    # Test with both include_relevant_nodes and include_monitored_branch_end set to False
    monitored_node_ids = get_monitored_node_ids(network_data, False, False)
    assert monitored_node_ids == []


def test_get_relevant_stations(network_data_preprocessed: NetworkData) -> None:
    rel_stations = get_relevant_stations(network_data_preprocessed)
    assert len(rel_stations) == len(network_data_preprocessed.relevant_nodes)
    for station in rel_stations:
        assert network_data_preprocessed.node_ids.index(station.grid_model_id) in network_data_preprocessed.relevant_nodes


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


def test_contingency_ids(network_data_preprocessed: NetworkData) -> None:
    contingency_ids = network_data_preprocessed.contingency_ids
    assert len(
        contingency_ids
    ) == network_data_preprocessed.outaged_branch_mask.sum() + network_data_preprocessed.outaged_injection_mask.sum() + len(
        network_data_preprocessed.multi_outage_ids
    )
