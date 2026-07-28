# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pypowsybl
import pytest
from tests.network_data_pickle import load_network_data
from toop_engine_contingency_analysis.pypowsybl import run_contingency_analysis_powsybl
from toop_engine_dc_solver.export.export import (
    get_changing_switches_from_action_set,
    get_changing_switches_from_actions,
)
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_stations
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.network_data import extract_action_set, extract_nminus1_definition, load_lf_params
from toop_engine_interfaces.folder_structure import OUTPUT_FILE_NAMES, POSTPROCESSING_PATHS, PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def test_get_changing_switches_from_actions_matches_network_diff(
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
):
    net = basic_node_breaker_grid_v1
    topology_stations = basic_node_breaker_topology
    target_station = topology_stations[0]
    changed_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )
    target_station = topology_stations[0]
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    disconnections = [GridElement(id="L8", name="", type="LINE", kind="branch")]

    expected = get_changing_switches_from_stations(network=net, stations=[target_station])
    result = get_changing_switches_from_actions(
        changed_stations=[changed_station],
        simplified_starting_stations=[starting_station],
        disconnections=disconnections,
        full_starting_stations=[starting_station],
    )

    SwitchUpdateSchema.validate(result)
    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_get_changing_switches_from_action_set_matches_expanded_inputs(
    basic_node_breaker_topology,
) -> None:
    topology_stations = basic_node_breaker_topology
    target_station = topology_stations[0]
    changed_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )

    base_station = topology_stations[0]
    starting_station = base_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in base_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    starting_stations = [starting_station]
    disconnection_elements = [GridElement(id="L8", name="", type="LINE", kind="branch")]
    action_set = ActionSet.model_construct(
        starting_stations=starting_stations,
        simplified_starting_stations=starting_stations,
        connectable_branches=[],
        disconnectable_branches=disconnection_elements,
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[changed_station],
    )

    result = get_changing_switches_from_action_set(
        action_set=action_set,
        actions=[0],
        disconnections=[0],
    )
    expected = get_changing_switches_from_actions(
        changed_stations=[changed_station],
        simplified_starting_stations=starting_stations,
        disconnections=disconnection_elements,
        full_starting_stations=starting_stations,
    )

    SwitchUpdateSchema.validate(result)
    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))


@pytest.mark.parametrize(
    ("actions", "disconnections", "expected_message"),
    [
        ([1], [], "Action index 1 is out of bounds for the action set"),
        ([-1], [], "Action index -1 is out of bounds for the action set"),
        ([], [1], "Disconnection index 1 is out of bounds for the action set"),
        ([], [-1], "Disconnection index -1 is out of bounds for the action set"),
    ],
)
def test_get_changing_switches_from_action_set_validates_indices(
    basic_node_breaker_topology,
    actions: list[int],
    disconnections: list[int],
    expected_message: str,
) -> None:
    starting_stations = basic_node_breaker_topology
    action_set = ActionSet.model_construct(
        starting_stations=starting_stations,
        simplified_starting_stations=starting_stations,
        connectable_branches=[],
        disconnectable_branches=[GridElement(id="L8", name="", type="LINE", kind="branch")],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[starting_stations[0]],
    )

    with pytest.raises(ValueError, match=expected_message):
        get_changing_switches_from_action_set(
            action_set=action_set,
            actions=actions,
            disconnections=disconnections,
        )


def test_switch_updates_match_runner_on_node_breaker_grid(
    node_breaker_grid_preprocessed_data_folder: Path,
) -> None:
    data_folder = node_breaker_grid_preprocessed_data_folder
    base_net = pypowsybl.network.load(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(data_folder / "network_data.pkl")
    action_set = extract_action_set(network_data)
    nminus1_definition = extract_nminus1_definition(network_data)
    lf_params = load_lf_params(data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])

    runner = PowsyblRunner(lf_params=lf_params)
    runner.replace_grid(deepcopy(base_net))
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    post_process_file_path = (
        data_folder / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        optim_res = json.load(f)

    for topology in optim_res["best_topos"][:3]:
        actions = topology["actions"]
        changed_stations = [action_set.local_actions[action] for action in actions]

        switch_updates = get_changing_switches_from_actions(
            changed_stations=changed_stations,
            simplified_starting_stations=action_set.get_simplified_starting_stations(),
            disconnections=[],
        )

        switch_update_df = switch_updates.rename(columns={"grid_model_id": "id"}).set_index("id")
        net_with_switch_updates = deepcopy(base_net)
        net_with_switch_updates.update_switches(switch_update_df)

        direct_result = run_contingency_analysis_powsybl(
            net=net_with_switch_updates,
            n_minus_1_definition=nminus1_definition,
            job_id="",
            timestep=0,
            method="dc",
            polars=True,
            lf_params=lf_params,
        )
        runner_result = runner.run_dc_loadflow(actions, [])

        assert runner.get_last_action_info() is not None
        assert direct_result == runner_result
