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
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_dc_solver.postprocess.validate_loadflow_results import (
    LoadflowValidationParameters,
    validate_loadflow_results,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_busbar_outage_ids,
    extract_nminus1_definition,
    load_lf_params,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    PreprocessParameters,
    ReassignmentLimits,
)
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import update_static_information
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    LoadflowSolverParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _get_preprocess_parameters() -> PreprocessParameters:
    return PreprocessParameters(
        preprocess_bb_outages=True,
        filter_disconnectable_branches_processes=8,
        double_limit_n0=0.95,
        double_limit_n1=0.95,
        ac_dc_interpolation=0.0,
        electrical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=0),
        physical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=0),
    )


def _get_complex_grid_preprocess_parameters() -> PreprocessParameters:
    return PreprocessParameters(
        preprocess_bb_outages=True,
        electrical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=0),
        physical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=0),
    )


def _get_ga_parameters() -> BatchedMEParameters:
    return BatchedMEParameters(
        enable_bb_outage=True,
        bb_outage_as_nminus1=True,
        clip_bb_outage_penalty=True,
        bb_outage_more_islands_penalty=5.0,
        runtime_seconds=300,
        iterations_per_epoch=500,
        random_topo_prob=0.05,
        mutation_repetition=1,
        n_subs_mutated_lambda=2.0,
        add_split_prob=0.2,
        change_split_prob=0.5,
        remove_split_prob=0.2,
        add_disconnection_prob=0.25,
        change_disconnection_prob=0.5,
        remove_disconnection_prob=0.25,
        pst_mutation_sigma=3.0,
        pst_mutation_probability=0.2,
        pst_reset_probability=0.05,
        enable_nodal_inj_optim=True,
        enable_parallel_pst_group_optim=False,
        target_metrics=(
            ("overload_energy_limited_n_1", 1.0),
            ("critical_branch_count_n_0", 200.0),
            ("critical_branch_count_n_1", 50.0),
            ("pst_switching_distance", 0.01),
            ("pst_activated", 0.1),
        ),
        observed_metrics=(
            "max_flow_n_0",
            "max_flow_n_1",
            "overload_energy_n_0",
            "overload_energy_n_1",
            "overload_energy_limited_n_0",
            "overload_energy_limited_n_1",
            "split_subs",
            "switching_distance",
            "disconnected_branches",
            "critical_branch_count_n_0",
            "critical_branch_count_n_1",
            "pst_switching_distance",
            "pst_activated",
            "pst_switching_distance_squared",
        ),
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=4),
            DescriptorDef(metric="disconnected_branches", num_cells=3),
            DescriptorDef(metric="pst_activated", num_cells=10),
        ),
    )


def _get_loadflow_solver_parameters() -> LoadflowSolverParameters:
    return LoadflowSolverParameters(batch_size=64, max_num_disconnections=2, max_num_splits=3)


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


def _reload_complex_grid_with_busbar_outages(
    complex_grid_data_folder: Path,
) -> tuple[NetworkData, StaticInformation, object]:
    lf_params = load_lf_params(complex_grid_data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    _info, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(complex_grid_data_folder)),
        pandapower=False,
        parameters=_get_complex_grid_preprocess_parameters(),
        lf_params=lf_params,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    return network_data, static_information, lf_params


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


def test_validate_loadflow_results_including_busbar_outages(
    test_grid_folder_path: Path,
    network_data_test_grid: NetworkData,
    jax_inputs_test_grid: tuple,
) -> None:
    topo_indices, static_information = jax_inputs_test_grid
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    nminus1_definition = extract_nminus1_definition(network_data_test_grid)

    runner = PowsyblRunner(
        lf_params=load_lf_params(test_grid_folder_path / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )
    runner.load_base_grid(test_grid_folder_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_nminus1_definition(nminus1_definition)
    runner.store_action_set(extract_action_set(network_data_test_grid))

    actions = topo_indices.action[0].tolist()
    disconnections: list[int] = []
    loadflow_results = runner.run_dc_loadflow(actions, disconnections)

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=loadflow_results,
        active_topology_network=runner.build_topology_network(actions, disconnections),
        actions=actions,
        disconnections=disconnections,
    )


def test_validate_loadflow_results_complex_grid_busbar_outages_repro(
    complex_grid_battery_hvdc_svc_3w_trafo_linear_1_1_data_folder: Path,
) -> None:
    network_data, static_information, lf_params = _reload_complex_grid_with_busbar_outages(
        complex_grid_battery_hvdc_svc_3w_trafo_linear_1_1_data_folder
    )
    ga_parameters = _get_ga_parameters()
    loadflow_solver_parameters = _get_loadflow_solver_parameters()

    busbar_outage_ids = extract_busbar_outage_ids(network_data)
    assert busbar_outage_ids

    (static_information,) = update_static_information(
        static_informations=(static_information,),
        batch_size=loadflow_solver_parameters.batch_size,
        enable_nodal_inj_optim=ga_parameters.enable_nodal_inj_optim,
        enable_parallel_pst_group_optim=ga_parameters.enable_parallel_pst_group_optim,
        enable_bb_outage=ga_parameters.enable_bb_outage,
        bb_outage_as_nminus1=ga_parameters.bb_outage_as_nminus1,
        clip_bb_outage_penalty=ga_parameters.clip_bb_outage_penalty,
        bb_outage_more_islands_penalty=ga_parameters.bb_outage_more_islands_penalty,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )

    assert static_information.dynamic_information.n_bb_outages == len(busbar_outage_ids)
    assert static_information.dynamic_information.bb_outage_contingency_ids == tuple(busbar_outage_ids)

    nminus1_definition = extract_nminus1_definition(network_data)
    runner = PowsyblRunner(lf_params=lf_params)
    runner.load_base_grid(
        complex_grid_battery_hvdc_svc_3w_trafo_linear_1_1_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    )
    runner.store_nminus1_definition(nminus1_definition)
    runner.store_action_set(extract_action_set(network_data))

    validate_loadflow_results(
        static_information=static_information,
        nminus1_definition=nminus1_definition,
        loadflows=runner.run_dc_loadflow([], []),
        active_topology_network=runner.build_topology_network([], []),
        actions=[],
        disconnections=[],
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
