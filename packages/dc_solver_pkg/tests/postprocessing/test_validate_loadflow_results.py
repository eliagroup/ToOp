# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pypowsybl
import pytest
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import ActionIndexComputations, StaticInformation
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_dc_solver.postprocess.validate_loadflow_results import (
    validate_loadflow_results,
)
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_nminus1_definition,
    load_network_data,
)
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars


def test_validate_loadflow_results_unsplit(preprocessed_powsybl_data_folder: Path) -> None:
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    action_set = extract_action_set(network_data)
    nminus1_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner()
    runner.replace_grid(net)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )

    unsplit_lfs = runner.run_dc_loadflow([], [])

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=unsplit_lfs,
        actions=[],
        disconnections=[],
    )


def test_validate_loadflow_results(preprocessed_powsybl_data_folder: Path) -> None:
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])

    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    action_set = extract_action_set(network_data)
    nminus1_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner()
    runner.replace_grid(net)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        optim_res = json.load(f)

    actions = optim_res["best_topos"][0]["actions"]
    disconnections = optim_res["best_topos"][0]["disconnection"]

    # Compute loadflows
    lfs = runner.run_dc_loadflow(actions, disconnections)

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=lfs,
        actions=actions,
        disconnections=disconnections,
    )


@pytest.mark.parametrize("topo_idx", range(10))
def test_lf_results_for_overlapping_branch_masks(
    powsybl_data_folder: Path, overlapping_branch_data: tuple[NetworkData, StaticInformation, list[dict]], topo_idx: int
) -> None:
    """
    Test that the branch masks for the N-0 and N-1 cases do not overlap.
    """
    network_data, static_information, best_actions = overlapping_branch_data

    actions = best_actions[topo_idx]["actions"]
    disconnections = best_actions[topo_idx]["disconnection"]

    (n_0_solver, n_1_solver), success_solver = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=jnp.array(disconnections)[None],
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    if not all(success_solver):
        pytest.skip("Solver did not converge for all actions")
    n_0_solver = n_0_solver[0, 0]
    n_1_solver = n_1_solver[0, 0]

    runner = PowsyblRunner()
    runner.load_base_grid(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    action_set = extract_action_set(network_data)
    runner.store_action_set(action_set)
    ref = runner.run_dc_loadflow(actions, disconnections)
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)

    assert n_0_ref.shape == n_0_solver.shape
    assert n_1_ref.shape == n_1_solver.shape

    assert np.allclose(np.abs(n_0_ref), np.abs(n_0_solver))
    assert np.allclose(np.abs(n_1_ref), np.abs(n_1_solver))


@pytest.mark.parametrize("topo_idx", range(10))
def test_lf_results_for_non_overlapping_branch_masks(
    powsybl_data_folder: Path, non_overlapping_branch_data: tuple[NetworkData, StaticInformation, list[dict]], topo_idx: int
) -> None:
    network_data, static_information, best_actions = non_overlapping_branch_data

    actions = best_actions[topo_idx]["actions"]
    disconnections = best_actions[topo_idx]["disconnection"]

    (n_0_solver, n_1_solver), success_solver = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=jnp.array(disconnections)[None],
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    if not any(success_solver):
        pytest.skip("Solver did not converge for all actions")
    n_0_solver = n_0_solver[0, 0]
    n_1_solver = n_1_solver[0, 0]

    runner = PowsyblRunner()
    runner.load_base_grid(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    action_set = extract_action_set(network_data)
    runner.store_action_set(action_set)
    ref = runner.run_dc_loadflow(actions, disconnections)
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)

    # Since there is no overlap between the branch masks, there should be no columns/rows that have to be zeroed out
    assert n_0_ref.shape == n_0_solver.shape
    assert n_1_ref.shape == n_1_solver.shape

    assert np.allclose(np.abs(n_0_ref), np.abs(n_0_solver))
    assert np.allclose(np.abs(n_1_ref), np.abs(n_1_solver))


@pytest.mark.parametrize("topo_idx", range(10))
def test_lf_results_for_overlapping_monitored_and_disconnected_branch_data(
    powsybl_data_folder: Path,
    overlapping_monitored_and_disconnected_branch_data: tuple[NetworkData, StaticInformation, list[dict]],
    topo_idx: int,
) -> None:
    ## Test N0
    network_data, static_information, best_actions = overlapping_monitored_and_disconnected_branch_data

    actions = best_actions[topo_idx]["actions"]
    disconnections = best_actions[topo_idx]["disconnection"]

    (n_0_solver, n_1_solver), success_solver = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=jnp.array(disconnections)[None],
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    if not any(success_solver):
        pytest.skip("Solver did not converge for all actions")
    n_0_solver = n_0_solver[0, 0]
    n_1_solver = n_1_solver[0, 0]

    n_monitored_branches = network_data.monitored_branch_mask.sum()
    n_contingencies = (
        network_data.outaged_branch_mask.sum()
        + network_data.outaged_injection_mask.sum()
        + len(network_data.multi_outage_names)
    )
    expected_n_0_shape = (n_monitored_branches,)
    expected_n_1_shape = (n_contingencies, n_monitored_branches)

    assert n_0_solver.shape == expected_n_0_shape
    assert n_1_solver.shape == expected_n_1_shape

    runner = PowsyblRunner()
    runner.load_base_grid(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    action_set = extract_action_set(network_data)
    runner.store_action_set(action_set)
    ref = runner.run_dc_loadflow(actions, disconnections)
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)

    assert n_0_ref.shape == expected_n_0_shape
    assert n_1_ref.shape == expected_n_1_shape

    assert np.allclose(np.abs(n_0_ref), np.abs(n_0_solver))
    assert np.allclose(np.abs(n_1_ref), np.abs(n_1_solver))
