# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from unittest.mock import patch

import numpy as np
from toop_engine_dc_solver.export.disconnection_switch_updates import (
    apply_disconnections_to_station_switching_table,
    get_changing_switches_from_disconnections,
    get_disconnected_asset_ids,
    get_station_ids_affected_by_disconnections,
)
from toop_engine_dc_solver.export.export import get_changing_switches_from_actions
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_topology
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def test_get_disconnected_asset_ids_and_affected_stations(basic_node_breaker_topology):
    disconnected_asset_ids = get_disconnected_asset_ids(
        disconnections=[GridElement(id="L8", name="", type="LINE", kind="branch")],
        starting_topology=basic_node_breaker_topology,
    )

    affected_station_ids = get_station_ids_affected_by_disconnections(
        starting_topology=basic_node_breaker_topology,
        disconnected_asset_ids=disconnected_asset_ids,
    )

    assert disconnected_asset_ids == {"L8"}
    assert affected_station_ids == {basic_node_breaker_topology.stations[0].grid_model_id}


def test_apply_disconnections_to_station_switching_table(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    switching_table = np.array([[True, False, True], [False, True, False]], dtype=bool)

    result = apply_disconnections_to_station_switching_table(
        station=station,
        switching_table=switching_table,
        disconnected_asset_ids={"L8"},
    )

    assert result.tolist() == [[True, False, False], [False, True, False]]


def test_get_changing_switches_from_actions_warns_for_unrepresentable_disconnection(
    basic_node_breaker_topology,
):
    starting_topology = basic_node_breaker_topology.model_copy(update={"stations": []})
    disconnections = [GridElement(id="L8", name="Line 8", type="LINE", kind="branch")]

    with patch("toop_engine_dc_solver.export.disconnection_switch_updates.logger.warning") as warning_mock:
        result = get_changing_switches_from_actions(
            changed_stations=[],
            starting_topology=starting_topology,
            disconnections=disconnections,
        )

    assert result.empty
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[0].startswith("Disconnected asset cannot be represented")
    assert warning_mock.call_args.kwargs == {
        "disconnection_id": "L8",
        "disconnection_name": "Line 8",
        "disconnection_type": "LINE",
        "available_station_ids": [],
    }


def test_get_changing_switches_from_disconnections_matches_network_diff(
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
):
    net = basic_node_breaker_grid_v1
    target_station = basic_node_breaker_topology.stations[0]
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "asset_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    target_topology = basic_node_breaker_topology.model_copy(update={"stations": [target_station]})
    starting_topology = target_topology.model_copy(update={"stations": [starting_station]})
    disconnections = [GridElement(id="L8", name="", type="LINE", kind="branch")]

    expected = get_changing_switches_from_topology(network=net, target_topology=target_topology)
    result = get_changing_switches_from_disconnections(
        starting_topology=starting_topology,
        disconnections=disconnections,
    )

    SwitchUpdateSchema.validate(result)
    expected_disconnection_switches = expected.loc[expected["grid_model_id"] == "L82_BREAKER"]
    assert result.reset_index(drop=True).equals(expected_disconnection_switches.reset_index(drop=True))


def test_get_changing_switches_from_actions_warns_on_overlapping_switch_updates(
    basic_node_breaker_topology,
):
    target_station = basic_node_breaker_topology.stations[0]
    changed_station = target_station.model_copy(
        update={
            "asset_switching_table": np.array([[False, False, False], [True, True, False]], dtype=bool),
        }
    )
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "asset_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    starting_topology = basic_node_breaker_topology.model_copy(update={"stations": [starting_station]})
    disconnections = [GridElement(id="L8", name="", type="LINE", kind="branch")]

    with patch("toop_engine_dc_solver.export.export.logger.warning") as warning_mock:
        result = get_changing_switches_from_actions(
            changed_stations=[changed_station],
            starting_topology=starting_topology,
            disconnections=disconnections,
        )

    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[0].startswith("Action and disconnection switch updates overlap")
    assert warning_mock.call_args.kwargs == {"overlapping_switch_ids": ["L82_BREAKER"]}
    assert result["grid_model_id"].tolist().count("L82_BREAKER") == 1
