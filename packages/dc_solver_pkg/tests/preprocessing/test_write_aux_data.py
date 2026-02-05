# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case30_with_psts
from toop_engine_dc_solver.postprocess.write_aux_data import write_aux_data
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, extract_nminus1_definition
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.stored_action_set import load_action_set


def test_extract_data_compare_to_jax(network_data_preprocessed: NetworkData) -> None:
    nminus1_definition = extract_nminus1_definition(network_data_preprocessed)
    action_set = extract_action_set(network_data_preprocessed)

    # Compare with the processed jax data
    static_information = convert_to_jax(network_data_preprocessed)
    assert len(action_set.local_actions) == len(static_information.dynamic_information.action_set)
    mon_branches = [el for el in nminus1_definition.monitored_elements if el.kind == "branch"]
    assert len(mon_branches) == static_information.n_branches_monitored
    assert len(nminus1_definition.contingencies) == static_information.n_nminus1_cases + 1
    assert nminus1_definition.contingencies[0].id == "BASECASE"
    assert len(nminus1_definition.contingencies[0].elements) == 0
    assert len(action_set.disconnectable_branches) == static_information.dynamic_information.disconnectable_branches.shape[0]
    assert len(action_set.pst_ranges) == static_information.dynamic_information.n_controllable_pst


def test_extract_data_compare_to_network_data(network_data_preprocessed: NetworkData) -> None:
    # Test the extraction of the N-1 definition
    n_minus_1_definition = extract_nminus1_definition(network_data_preprocessed)
    n_contingencies = len(n_minus_1_definition.contingencies)

    n_multi_outages = len(network_data_preprocessed.multi_outage_ids)
    n_branch_outages = network_data_preprocessed.outaged_branch_mask.sum()
    n_inj_outages = network_data_preprocessed.outaged_injection_mask.sum()

    n_expected_contingencies = (
        1  # Base case
        + n_branch_outages  # Single branch outages
        + n_multi_outages  # Multi outages
        + n_inj_outages  # Injection outages
    )
    assert n_contingencies == n_expected_contingencies, "Number of contingencies does not match expected count"

    n_monitored_elements = len(n_minus_1_definition.monitored_elements)

    n_monitored_branches = network_data_preprocessed.monitored_branch_mask.sum()
    n_monitored_nodes = sum(
        [len(station.busbars) for station in network_data_preprocessed.simplified_asset_topology.stations]
    )
    n_monitored_switches = sum(
        [len(station.couplers) for station in network_data_preprocessed.simplified_asset_topology.stations]
    )
    n_expected_monitored_elements = n_monitored_branches + n_monitored_nodes + n_monitored_switches
    assert n_monitored_elements == n_expected_monitored_elements, (
        "Number of monitored elements does not match expected count"
    )
    assert n_minus_1_definition.loadflow_parameters.distributed_slack == network_data_preprocessed.metadata.get(
        "distributed_slack", True
    )


def test_write_aux_data(network_data_preprocessed: NetworkData, tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("test_write_aux_data")
    write_aux_data(tmp_path, network_data_preprocessed)
    assert (tmp_path / PREPROCESSING_PATHS["action_set_file_path"]).exists()
    assert (tmp_path / PREPROCESSING_PATHS["nminus1_definition_file_path"]).exists()

    action_set = load_action_set(tmp_path / PREPROCESSING_PATHS["action_set_file_path"])
    nminus1_definition = load_nminus1_definition(tmp_path / PREPROCESSING_PATHS["nminus1_definition_file_path"])

    assert len(action_set.local_actions)
    assert len(action_set.disconnectable_branches)
    assert len(nminus1_definition.contingencies)
    assert len(nminus1_definition.monitored_elements)


def test_write_aux_data_pst_ranges(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("test_write_aux_data_pst_ranges")
    case30_with_psts(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)

    write_aux_data(tmp_path, network_data)
    action_set = load_action_set(tmp_path / PREPROCESSING_PATHS["action_set_file_path"])

    assert len(action_set.local_actions) == sum(len(x) for x in network_data.branch_action_set)
    assert len(action_set.pst_ranges) == network_data.controllable_phase_shift_mask.sum()
