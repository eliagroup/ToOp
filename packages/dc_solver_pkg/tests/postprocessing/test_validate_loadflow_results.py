# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
from fsspec.implementations.dirfs import DirFileSystem
from tests.network_data_pickle import load_network_data
from toop_engine_dc_solver.example_grids import case30_with_psts_powsybl
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_dc_solver.postprocess.validate_loadflow_results import (
    LoadflowValidationParameters,
    validate_loadflow_results,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax, load_grid
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_busbar_outage_ids,
    extract_nminus1_definition,
    get_relevant_stations,
    load_lf_params,
)
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_dc_solver.preprocess.preprocess import PreprocessParameters, preprocess
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


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
        active_topology_network=active_topology_network,
        actions=actions,
        disconnections=disconnections,
    )


def test_validate_loadflow_results_unsplit(preprocessed_powsybl_data_folder: Path) -> None:
    _network_data, static_information, runner, nminus1_definition = _load_validation_inputs(preprocessed_powsybl_data_folder)

    unsplit_lfs = runner.run_dc_loadflow([], [])

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=unsplit_lfs,
        active_topology_network=runner.build_topology_network([], []),
        actions=[],
        disconnections=[],
    )


def test_validate_loadflow_results_unsplit_complex_grid_with_busbar_outages(
    create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path: Path,
) -> None:
    data_folder = create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path
    excluded_busbar_ids = {}
    # this list makes sure some edge cases are for sure in the outage list
    minimal_expected_busbar_ids_list = {
        "VL_2W_MV_HV_HV_1_1",
        "VL_2W_MV_HV_HV_1_2",
        "VL_2W_MV_HV_HV_2_1",
        "VL_2W_MV_HV_HV_2_2",
        "VL_3W_HV_1_1",
        "VL_MV_1_1",
    }

    network_data, static_information, runner, nminus1_definition = _load_validation_inputs(data_folder)
    unsplit_lfs = runner.run_dc_loadflow([], [])
    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=unsplit_lfs,
        active_topology_network=runner.build_topology_network([], []),
        actions=[],
        disconnections=[],
    )

    lf_params = load_lf_params(data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    backend = PowsyblBackend(DirFileSystem(str(data_folder)), lf_params=lf_params)
    bb_network_data = preprocess(backend, parameters=PreprocessParameters(preprocess_bb_outages=True))

    busbar_outage_ids = extract_busbar_outage_ids(bb_network_data)
    assert busbar_outage_ids
    assert minimal_expected_busbar_ids_list.issubset(set(busbar_outage_ids))

    selected_busbar_ids = [
        busbar.grid_model_id
        for station in get_relevant_stations(bb_network_data)
        for busbar in station.busbars
        if busbar.grid_model_id in busbar_outage_ids and busbar.grid_model_id not in excluded_busbar_ids
    ]
    assert selected_busbar_ids

    bb_static_information = convert_to_jax(
        bb_network_data,
        preprocess_bb_outages=True,
    )
    bb_static_information = replace(
        bb_static_information,
        solver_config=replace(
            bb_static_information.solver_config,
            batch_size_bsdf=1,
            enable_bb_outages=True,
            bb_outage_as_nminus1=True,
        ),
        dynamic_information=replace(
            bb_static_information.dynamic_information,
            bb_outage_baseline_analysis=None,
        ),
    )

    (n_0, n_1), success = run_solver_symmetric(
        default_topology(bb_static_information.solver_config),
        None,
        None,
        bb_static_information.dynamic_information,
        bb_static_information.solver_config,
        lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    n_0 = np.abs(n_0[0, 0])
    n_1 = np.abs(n_1[0, 0])
    assert np.all(success)

    full_nminus1_definition = extract_nminus1_definition(bb_network_data)
    busbar_contingency_ids = {
        contingency.id
        for contingency in full_nminus1_definition.contingencies
        if any(element.kind == "bus" for element in contingency.elements)
    }
    assert set(selected_busbar_ids).issubset(busbar_contingency_ids)
    assert minimal_expected_busbar_ids_list.issubset(busbar_contingency_ids)
    contingency_order = [
        contingency.id for contingency in full_nminus1_definition.contingencies if not contingency.is_basecase()
    ]
    selected_row_indices = [contingency_order.index(busbar_id) for busbar_id in selected_busbar_ids]

    busbar_runner = PowsyblRunner(lf_params=lf_params)
    busbar_runner.load_base_grid(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    busbar_runner.store_action_set(extract_action_set(bb_network_data))
    selected_contingencies = [
        next(contingency for contingency in full_nminus1_definition.contingencies if contingency.id == busbar_id)
        for busbar_id in selected_busbar_ids
    ]
    busbar_nminus1_definition = Nminus1Definition(
        monitored_elements=full_nminus1_definition.monitored_elements,
        contingencies=[full_nminus1_definition.contingencies[0], *selected_contingencies],
    )
    busbar_runner.store_nminus1_definition(busbar_nminus1_definition)

    ref_result = busbar_runner.run_dc_loadflow([], [])
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref_result, busbar_nminus1_definition, 0)
    n_0_ref = np.abs(n_0_ref)
    n_1_ref = np.abs(n_1_ref)

    assert np.allclose(n_0, n_0_ref, atol=1e-5, rtol=1e-5)
    assert success_ref.shape == (len(selected_busbar_ids),)
    assert np.all(success_ref)
    assert np.allclose(n_1[selected_row_indices], n_1_ref, atol=1e-5, rtol=1e-5)


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
        active_topology_network=runner.build_topology_network(actions, disconnections),
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


def test_validate_loadflows_with_psts(tmp_path: Path) -> None:
    case30_with_psts_powsybl(tmp_path)

    _stats, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(tmp_path)),
        pandapower=False,
        lf_params=CGMES_DISTRIBUTED_SLACK,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    nminus1_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner(lf_params=load_lf_params(tmp_path / PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    runner.load_base_grid(tmp_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_nminus1_definition(nminus1_definition)
    runner.store_action_set(extract_action_set(network_data))

    # TODO find out why the - is needed...
    pst_setpoints = [-1, -2]
    wrong_pst_setpoints = [-2, -3]

    # Compute loadflows, should match
    lfs = runner.run_dc_loadflow([], [], pst_setpoints)
    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=lfs,
        active_topology_network=runner.build_topology_network([], [], pst_setpoints),
        actions=[],
        disconnections=[],
        pst_setpoints=pst_setpoints,
    )

    with pytest.raises(AssertionError):
        validate_loadflow_results(
            static_information=static_information,
            nminus1_definition=nminus1_definition,
            loadflows=lfs,
            active_topology_network=runner.build_topology_network([], [], pst_setpoints),
            actions=[],
            disconnections=[],
            pst_setpoints=wrong_pst_setpoints,
        )


def test_validate_loadflows_with_nonlinear_psts(
    create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path: Path,
) -> None:
    powsybl_data_folder = create_complex_grid_battery_hvdc_svc_3w_trafo_linear_0_0_data_path

    _stats, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(powsybl_data_folder)),
        pandapower=False,
        lf_params=CGMES_DISTRIBUTED_SLACK,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    nminus1_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner(
        lf_params=load_lf_params(powsybl_data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )
    runner.load_base_grid(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_nminus1_definition(nminus1_definition)
    runner.store_action_set(extract_action_set(network_data))

    di = static_information.dynamic_information
    assert di.nodal_injection_information is not None
    pst_n_taps = di.nodal_injection_information.pst_n_taps.astype(int)
    low_tap = di.nodal_injection_information.grid_model_low_tap.astype(int)
    min_tap_setpoints = low_tap
    max_tap_setpoints = low_tap + pst_n_taps - 1
    neutral_tap_setpoints = (di.nodal_injection_information.starting_tap_idx + low_tap).astype(int)
    midpoint_tap_setpoints = (low_tap + ((pst_n_taps - 1) // 2)).astype(int)
    tap_scenarios = {
        "min": min_tap_setpoints,
        "max": max_tap_setpoints,
        "neutral": neutral_tap_setpoints,
        "midpoint": midpoint_tap_setpoints,
    }

    assert neutral_tap_setpoints.shape[0] == di.nodal_injection_information.pst_n_taps.shape[0]
    # make sure the pst are set
    # hardcoded 2 = two psts are controllable and non linear in the test grid
    # at the time of writing this test, additionally two pst are outside of the control area due to the settings of this test
    # see cgmes import parameter in def complex_grid_battery_hvdc_svc_3w_trafo_data_folder()
    # this should not dynamically be tested against the network mask, to be sure there are non linear pst in the test grid
    assert neutral_tap_setpoints.shape[0] == 2
    # check that the psts are non linear
    assert ~np.isclose(runner.net.get_phase_tap_changer_steps()["x"].sum(), 0.0)
    assert ~np.isclose(runner.net.get_phase_tap_changer_steps()["rho"].min(), 1.0) | ~np.isclose(
        runner.net.get_phase_tap_changer_steps()["rho"].max(), 1.0
    )

    for pst_taps in tap_scenarios.values():
        pst_setpoints = pst_taps.tolist()
        lfs = runner.run_dc_loadflow([], [], pst_setpoints)
        validate_loadflow_results(
            static_information=static_information,
            nminus1_definition=nminus1_definition,
            loadflows=lfs,
            active_topology_network=runner.build_topology_network([], [], pst_setpoints),
            actions=[],
            disconnections=[],
            pst_setpoints=pst_setpoints,
            validation_parameters=LoadflowValidationParameters(atol=1e-9, rtol=1e-9),
        )

    tap_span = pst_n_taps - 1
    assert bool(jnp.any(tap_span > 0))
    modifiable_idx = int(jnp.argmax(tap_span))
    wrong_pst_setpoints = neutral_tap_setpoints.tolist()
    if neutral_tap_setpoints[modifiable_idx] < max_tap_setpoints[modifiable_idx]:
        wrong_pst_setpoints[modifiable_idx] += 1
    else:
        wrong_pst_setpoints[modifiable_idx] -= 1

    neutral_setpoints = neutral_tap_setpoints.tolist()
    lfs_neutral = runner.run_dc_loadflow([], [], neutral_setpoints)

    with pytest.raises(AssertionError):
        validate_loadflow_results(
            static_information=static_information,
            nminus1_definition=nminus1_definition,
            loadflows=lfs_neutral,
            active_topology_network=runner.build_topology_network([], [], neutral_setpoints),
            actions=[],
            disconnections=[],
            pst_setpoints=wrong_pst_setpoints,
        )
