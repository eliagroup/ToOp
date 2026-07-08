# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import replace

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case30_with_psts_pandapower
from toop_engine_dc_solver.postprocess.write_aux_data import write_aux_data
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_busbar_outage_ids,
    extract_nminus1_definition,
)
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.stored_action_set import load_action_set


def test_extract_data_compare_to_jax(network_data_preprocessed: NetworkData) -> None:
    nminus1_definition = extract_nminus1_definition(network_data_preprocessed)
    action_set = extract_action_set(network_data_preprocessed)
    busbar_outage_ids = extract_busbar_outage_ids(network_data_preprocessed)
    busbar_contingencies = [
        contingency
        for contingency in nminus1_definition.contingencies
        if contingency.elements and contingency.elements[0].kind == "bus"
    ]

    # Compare with the processed jax data
    static_information = convert_to_jax(network_data_preprocessed, preprocess_bb_outages=True)
    assert len(action_set.local_actions) == len(static_information.dynamic_information.action_set)
    mon_branches = [el for el in nminus1_definition.monitored_elements if el.kind == "branch"]
    assert len(mon_branches) == static_information.n_branches_monitored
    assert [contingency.id for contingency in busbar_contingencies] == busbar_outage_ids
    assert static_information.dynamic_information.bb_outage_contingency_ids == busbar_outage_ids
    assert static_information.dynamic_information.n_bb_outages == len(busbar_outage_ids)
    assert len(nminus1_definition.contingencies) == static_information.n_nminus1_cases + len(busbar_outage_ids) + 1
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
    n_busbar_outages = len(extract_busbar_outage_ids(network_data_preprocessed))

    n_expected_contingencies = (
        1  # Base case
        + n_branch_outages  # Single branch outages
        + n_multi_outages  # Multi outages
        + n_inj_outages  # Injection outages
        + n_busbar_outages  # Exported busbar outages
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


def test_write_aux_data(network_data_preprocessed: NetworkData, tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("test_write_aux_data")
    write_aux_data(tmp_path, network_data_preprocessed)
    # assert (tmp_path / PREPROCESSING_PATHS["action_set_file_path"]).exists()
    assert (tmp_path / PREPROCESSING_PATHS["nminus1_definition_file_path"]).exists()

    action_set = load_action_set(
        tmp_path / PREPROCESSING_PATHS["action_set_file_path"],
        tmp_path / PREPROCESSING_PATHS["action_set_diff_path"],
    )
    nminus1_definition = load_nminus1_definition(tmp_path / PREPROCESSING_PATHS["nminus1_definition_file_path"])

    assert len(action_set.local_actions)
    assert len(action_set.disconnectable_branches)
    assert len(nminus1_definition.contingencies)
    assert len(nminus1_definition.monitored_elements)


def test_write_aux_data_pst_ranges(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("test_write_aux_data_pst_ranges")
    case30_with_psts_pandapower(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)

    write_aux_data(tmp_path, network_data)
    action_set = load_action_set(
        tmp_path / PREPROCESSING_PATHS["action_set_file_path"],
        tmp_path / PREPROCESSING_PATHS["action_set_diff_path"],
    )

    assert len(action_set.local_actions) == sum(len(x) for x in network_data.branch_action_set)
    assert len(action_set.pst_ranges) == network_data.controllable_phase_shift_mask.sum()


def test_write_aux_data_persists_pst_groups_and_clipped_ranges(
    network_data_preprocessed: NetworkData,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tmp_path = tmp_path_factory.mktemp("test_write_aux_data_parallel_pst_groups")

    controllable_phase_shift_mask = network_data_preprocessed.controllable_phase_shift_mask.copy()
    controllable_phase_shift_mask[:] = False
    controllable_phase_shift_mask[:2] = True

    phase_shift_mask = network_data_preprocessed.phase_shift_mask.copy()
    phase_shift_mask[:2] = True

    branch_ids = network_data_preprocessed.branch_ids.copy()
    branch_names = network_data_preprocessed.branch_names.copy()
    branch_types = network_data_preprocessed.branch_types.copy()
    branch_ids[:2] = ["PST1", "PST2"]
    branch_names[:2] = ["PST1", "PST2"]
    branch_types[:2] = ["TWO_WINDINGS_TRANSFORMER", "TWO_WINDINGS_TRANSFORMER"]

    network_data = replace(
        network_data_preprocessed,
        controllable_phase_shift_mask=controllable_phase_shift_mask,
        phase_shift_mask=phase_shift_mask,
        phase_shift_taps=[np.array([1.0, 2.0]), np.array([10.0, 11.0])],
        phase_shift_starting_tap_idx=np.array([1, 1]),
        phase_shift_low_tap=np.array([5, 5]),
        phase_shift_linearity=np.array([True, True]),
        parallel_pst_group_mask=np.array([[True, True]], dtype=bool),
        parallel_pst_group_ids=["shared_group"],
        branch_ids=branch_ids,
        branch_names=branch_names,
        branch_types=branch_types,
    )

    write_aux_data(tmp_path, network_data)
    action_set = load_action_set(
        tmp_path / PREPROCESSING_PATHS["action_set_file_path"],
        tmp_path / PREPROCESSING_PATHS["action_set_diff_path"],
    )

    assert [pst_range.pst_group for pst_range in action_set.pst_ranges] == ["shared_group", "shared_group"]
    assert [pst_range.low_tap for pst_range in action_set.pst_ranges] == [5, 5]
    assert [pst_range.high_tap for pst_range in action_set.pst_ranges] == [7, 7]
    assert [pst_range.starting_tap for pst_range in action_set.pst_ranges] == [6, 6]
