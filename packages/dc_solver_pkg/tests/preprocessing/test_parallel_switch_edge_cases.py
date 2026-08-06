# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import pytest
from pypowsybl.network import Network
from toop_engine_dc_solver.example_grids import parallel_switch_edge_cases_node_breaker_folder
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.postprocess.validate_loadflow_results import validate_loadflow_results
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_nminus1_definition,
)
from toop_engine_grid_helpers.powsybl.example_grids import (
    PARALLEL_SWITCH_EDGE_CASES,
    parallel_switch_edge_cases_node_breaker_network,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet

PreprocessedParallelSwitchCases = tuple[
    NetworkData,
    StaticInformation,
    ActionSet,
    Nminus1Definition,
    PowsyblRunner,
]


@pytest.fixture(scope="module")
def parallel_switch_edge_case_grid() -> Network:
    return parallel_switch_edge_cases_node_breaker_network()


@pytest.fixture(scope="module")
def preprocessed_parallel_switch_edge_cases(
    tmp_path_factory: pytest.TempPathFactory,
) -> PreprocessedParallelSwitchCases:
    folder = tmp_path_factory.mktemp("parallel_switch_edge_cases")
    network_data = parallel_switch_edge_cases_node_breaker_folder(folder)
    static_information = load_static_information(folder / PREPROCESSING_PATHS["static_information_file_path"])
    action_set = extract_action_set(network_data)
    nminus1_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner(lf_params=CGMES_DISTRIBUTED_SLACK)
    runner.load_base_grid(folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    return network_data, static_information, action_set, nminus1_definition, runner


def test_parallel_switch_edge_case_grid_has_four_relevant_substations(
    preprocessed_parallel_switch_edge_cases: PreprocessedParallelSwitchCases,
) -> None:
    network_data, _static_information, _action_set, _nminus1_definition, _runner = preprocessed_parallel_switch_edge_cases
    relevant_node_names = {
        node_name
        for node_name, relevant in zip(network_data.node_names, network_data.relevant_node_mask, strict=True)
        if relevant
    }
    assert relevant_node_names == {f"{voltage_level_id}_0" for voltage_level_id in PARALLEL_SWITCH_EDGE_CASES}


@pytest.mark.parametrize(
    ("voltage_level_id", "breaker_open", "disconnector_open"),
    [(voltage_level_id, *states) for voltage_level_id, states in PARALLEL_SWITCH_EDGE_CASES.items()],
    ids=PARALLEL_SWITCH_EDGE_CASES,
)
def test_parallel_switch_edge_case_preprocessing(
    preprocessed_parallel_switch_edge_cases: PreprocessedParallelSwitchCases,
    voltage_level_id: str,
    breaker_open: bool,
    disconnector_open: bool,
) -> None:
    _network_data, _static_information, action_set, _nminus1_definition, _runner = preprocessed_parallel_switch_edge_cases
    station_id = f"{voltage_level_id}_a"
    starting_station = next(station for station in action_set.starting_stations if station.bus_group_id == station_id)
    couplers_by_id = {coupler.grid_model_id: coupler for coupler in starting_station.couplers}

    assert couplers_by_id[f"{voltage_level_id}_BREAKER"].open is breaker_open
    assert couplers_by_id[f"{voltage_level_id}_DISCONNECTOR_0_1"].open is disconnector_open

    action_indices = [
        action_index for action_index, station in enumerate(action_set.local_actions) if station.bus_group_id == station_id
    ]
    assert 1 <= len(action_indices) <= 4


@pytest.mark.parametrize("voltage_level_id", PARALLEL_SWITCH_EDGE_CASES, ids=PARALLEL_SWITCH_EDGE_CASES)
def test_parallel_switch_edge_case_actions_match_powsybl(
    preprocessed_parallel_switch_edge_cases: PreprocessedParallelSwitchCases,
    voltage_level_id: str,
) -> None:
    _network_data, static_information, action_set, nminus1_definition, runner = preprocessed_parallel_switch_edge_cases
    action_indices = [
        action_index
        for action_index, station in enumerate(action_set.local_actions)
        if station.bus_group_id == f"{voltage_level_id}_a"
    ]

    for action_index in action_indices:
        actions = [action_index]
        loadflows = runner.run_dc_loadflow(actions, [])
        validate_loadflow_results(
            static_information=static_information,
            nminus1_definition=nminus1_definition,
            loadflows=loadflows,
            active_topology_network=runner.build_topology_network(actions, []),
            actions=actions,
            disconnections=[],
        )
