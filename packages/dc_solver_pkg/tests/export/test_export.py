# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import numpy as np
import pytest
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_contingency_analysis_powsybl,
)
from toop_engine_dc_solver.export.export import (
    get_changing_switches_from_action_set,
    get_changing_switches_from_actions,
)
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_stations
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import to_simplified_bus_group
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def simplify_station(station):
    return to_simplified_bus_group(station)


def simplify_stations(stations):
    return [simplify_station(station) for station in stations]


def test_get_changing_switches_from_actions_matches_network_diff(
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
):
    net = basic_node_breaker_grid_v1
    topology_stations = simplify_stations(basic_node_breaker_topology)
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
    topology_stations = simplify_stations(basic_node_breaker_topology)
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
    starting_stations = simplify_stations(basic_node_breaker_topology)
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
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
) -> None:
    base_net = deepcopy(basic_node_breaker_grid_v1)
    topology_stations = simplify_stations(basic_node_breaker_topology)
    target_station = topology_stations[0]
    changed_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )
    action_set = ActionSet.model_construct(
        starting_stations=topology_stations,
        simplified_starting_stations=topology_stations,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[changed_station],
    )
    nminus1_definition = get_full_nminus1_definition_powsybl(base_net)
    lf_params = CGMES_DISTRIBUTED_SLACK

    runner = PowsyblRunner(lf_params=lf_params)
    runner.replace_grid(deepcopy(base_net))
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    actions = [0]
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
