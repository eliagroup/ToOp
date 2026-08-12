# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from unittest.mock import patch

import numpy as np
from toop_engine_dc_solver.export.disconnection_switch_updates import (
    get_changing_switches_from_disconnections,
    get_disconnected_asset_ids,
)
from toop_engine_dc_solver.export.export import get_changing_switches_from_actions
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_stations
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def test_get_disconnected_asset_ids_returns_asset_bays_by_disconnection_id(basic_node_breaker_topology) -> None:
    starting_stations = basic_node_breaker_topology
    disconnections = [
        GridElement(id="L8", name="", type="LINE", kind="branch"),
        GridElement(id="missing", name="", type="LINE", kind="branch"),
    ]

    result = get_disconnected_asset_ids(
        stations=starting_stations,
        disconnections=disconnections,
    )

    assert set(result) == {"L8", "missing"}
    assert [asset_bay.breaker_grid_model_id for asset_bay in result["L8"]] == ["L82_BREAKER"]
    assert result["missing"] == []


def test_get_changing_switches_from_actions_warns_for_unrepresentable_disconnection(
    basic_node_breaker_topology,
) -> None:
    disconnections = [GridElement(id="L8", name="Line 8", type="LINE", kind="branch")]

    with patch("toop_engine_dc_solver.export.disconnection_switch_updates.logger.warning") as warning_mock:
        result = get_changing_switches_from_actions(
            changed_stations=[],
            simplified_starting_stations=[],
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
) -> None:
    net = basic_node_breaker_grid_v1
    topology_stations = basic_node_breaker_topology
    target_station = topology_stations[0]
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    target_station = topology_stations[0]
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    disconnections = [GridElement(id="L8", name="", type="LINE", kind="branch")]

    expected = get_changing_switches_from_stations(network=net, stations=[target_station])
    result = get_changing_switches_from_disconnections(
        starting_stations=[starting_station],
        disconnections=disconnections,
    )

    SwitchUpdateSchema.validate(result)
    expected_disconnection_switches = expected.loc[expected["grid_model_id"] == "L82_BREAKER"]
    assert result.reset_index(drop=True).equals(expected_disconnection_switches.reset_index(drop=True))


def test_get_changing_switches_from_actions_warns_on_overlapping_switch_updates(
    basic_node_breaker_topology,
) -> None:
    topology_stations = basic_node_breaker_topology
    target_station = topology_stations[0]
    changed_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False, False], [True, True, False]], dtype=bool),
        }
    )
    starting_station = target_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in target_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    base_station = topology_stations[0]
    starting_station = base_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in base_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    disconnections = [GridElement(id="L8", name="", type="LINE", kind="branch")]

    with patch("toop_engine_dc_solver.export.export.logger.warning") as warning_mock:
        result = get_changing_switches_from_actions(
            changed_stations=[changed_station],
            simplified_starting_stations=[starting_station],
            disconnections=disconnections,
            full_starting_stations=[starting_station],
        )

    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[0].startswith("Action and disconnection switch updates overlap")
    assert warning_mock.call_args.kwargs == {"overlapping_switch_ids": ["L82_BREAKER"]}
    assert result["grid_model_id"].tolist().count("L82_BREAKER") == 1
