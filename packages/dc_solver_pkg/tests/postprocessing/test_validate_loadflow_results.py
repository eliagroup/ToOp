# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from dataclasses import replace
from pathlib import Path

import pypowsybl
import pytest
from tests.network_data_pickle import load_network_data
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import StaticInformation
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
    load_lf_params,
)
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)


def _load_validation_inputs(
    preprocessed_powsybl_data_folder: Path,
) -> tuple[NetworkData, StaticInformation, PowsyblRunner, object]:
    network_data = load_network_data(preprocessed_powsybl_data_folder / "network_data.pkl")
    nminus1_definition = extract_nminus1_definition(network_data)
    action_set = extract_action_set(network_data)
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )

    lf_params = load_lf_params(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    runner = PowsyblRunner(lf_params=lf_params)
    runner.replace_grid(net)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    return network_data, static_information, runner, nminus1_definition


def _run_and_validate_loadflow_results(
    powsybl_data_folder: Path,
    network_data: NetworkData,
    static_information: StaticInformation,
    actions: list[int],
    disconnections: list[int],
) -> None:
    lf_params = load_lf_params(powsybl_data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    runner = PowsyblRunner(lf_params=lf_params)
    runner.load_base_grid(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    action_set = extract_action_set(network_data)
    runner.store_action_set(action_set)
    loadflow_results = runner.run_dc_loadflow(actions, disconnections)
    active_topology_network = runner.build_topology_network(actions, disconnections)

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=loadflow_results,
        actions=actions,
        disconnections=disconnections,
        active_topology_network=active_topology_network,
    )


def test_validate_loadflow_results_unsplit(preprocessed_powsybl_data_folder: Path) -> None:
    _network_data, static_information, runner, nminus1_definition = _load_validation_inputs(preprocessed_powsybl_data_folder)

    unsplit_lfs = runner.run_dc_loadflow([], [])

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=unsplit_lfs,
        actions=[],
        disconnections=[],
        active_topology_network=runner.build_topology_network([], []),
    )


def test_validate_loadflow_results(preprocessed_powsybl_data_folder: Path) -> None:
    network_data, static_information, runner, nminus1_definition = _load_validation_inputs(preprocessed_powsybl_data_folder)
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
        active_topology_network=runner.build_topology_network(actions, disconnections),
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

    _run_and_validate_loadflow_results(
        powsybl_data_folder=powsybl_data_folder,
        network_data=network_data,
        static_information=static_information,
        actions=actions,
        disconnections=disconnections,
    )


@pytest.mark.parametrize("topo_idx", range(10))
def test_lf_results_for_non_overlapping_branch_masks(
    powsybl_data_folder: Path, non_overlapping_branch_data: tuple[NetworkData, StaticInformation, list[dict]], topo_idx: int
) -> None:
    network_data, static_information, best_actions = non_overlapping_branch_data

    actions = best_actions[topo_idx]["actions"]
    disconnections = best_actions[topo_idx]["disconnection"]

    _run_and_validate_loadflow_results(
        powsybl_data_folder=powsybl_data_folder,
        network_data=network_data,
        static_information=static_information,
        actions=actions,
        disconnections=disconnections,
    )


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

    _run_and_validate_loadflow_results(
        powsybl_data_folder=powsybl_data_folder,
        network_data=network_data,
        static_information=static_information,
        actions=actions,
        disconnections=disconnections,
    )
