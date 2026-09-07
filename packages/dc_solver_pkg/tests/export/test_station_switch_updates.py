# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from pydantic import ValidationError
from toop_engine_dc_solver.export.station_switch_updates import (
    _get_asset_switch_diffs,
    _get_coupler_switch_diffs,
    _resolve_changed_bus_groups,
    get_changing_switches_from_changed_bus_groups,
)
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_bus_groups
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import to_simplified_bus_group
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def test_resolve_changed_bus_groups_preserves_topology_order(basic_node_breaker_topology):
    starting_bus_groups = [to_simplified_bus_group(bus_group) for bus_group in basic_node_breaker_topology]
    changed_bus_group = starting_bus_groups[0]

    starting_lookup, changed_lookup, ordered_bus_group_ids = _resolve_changed_bus_groups(
        changed_bus_groups=[changed_bus_group],
        starting_bus_groups=starting_bus_groups,
    )

    assert list(starting_lookup) == [changed_bus_group.bus_group_id]
    assert list(changed_lookup) == [changed_bus_group.bus_group_id]
    assert ordered_bus_group_ids == [changed_bus_group.bus_group_id]


def test_get_coupler_switch_diffs(basic_node_breaker_topology):
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    station = starting_stations[0]
    starting_station = station.model_copy(
        update={"couplers": [coupler.model_copy(update={"open": False}) for coupler in station.couplers]}
    )
    changed_station = station

    result = _get_coupler_switch_diffs(
        changed_bus_group=changed_station,
        starting_bus_group=starting_station,
    )

    assert result == [{"grid_model_id": "VL4_BREAKER", "open": True}]


def test_get_asset_switch_diffs(basic_node_breaker_topology):
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    target_station = starting_stations[0]
    starting_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    changed_station = target_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )

    result = _get_asset_switch_diffs(
        changed_bus_group=changed_station,
        starting_bus_group=starting_station,
    )

    assert result == [
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
    ]


def test_get_asset_switch_diffs_requires_matching_switching_table_shape(basic_node_breaker_topology):
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    station = starting_stations[0]
    with pytest.raises(ValidationError, match="branch_switching_table shape"):
        station.model_copy(
            update={
                "branch_switching_table": np.array([[False, False, True]], dtype=bool),
            }
        )


def test_get_asset_switch_diffs_requires_matching_asset_order(basic_node_breaker_topology):
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    station = starting_stations[0]
    reordered_asset_connections = [
        station.branch_connections[1],
        station.branch_connections[0],
        station.branch_connections[2],
    ]
    changed_station = station.model_copy(update={"branch_connections": reordered_asset_connections})

    with pytest.raises(ValueError, match=r"Use ActionSet.get_simplified_starting_bus_groups\(\) as input"):
        _get_asset_switch_diffs(
            changed_bus_group=changed_station,
            starting_bus_group=station,
        )


def test_get_asset_switch_diffs_allows_multiple_active_busbars(basic_node_breaker_topology):
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    station = starting_stations[0]
    starting_station = station.model_copy(
        update={
            "branch_switching_table": np.array([[True, False, False], [True, True, False]], dtype=bool),
        }
    )
    changed_station = station.model_copy(
        update={
            "branch_switching_table": np.array([[True, False, False], [False, True, False]], dtype=bool),
        }
    )

    result = _get_asset_switch_diffs(
        changed_bus_group=changed_station,
        starting_bus_group=starting_station,
    )

    assert result == [{"grid_model_id": "L42_DISCONNECTOR_3_1", "open": True}]


def test_get_changing_switches_from_changed_stations_matches_network_diff(
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
):
    net = basic_node_breaker_grid_v1
    starting_stations = [to_simplified_bus_group(station) for station in basic_node_breaker_topology]
    station = starting_stations[0]
    starting_station = station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    changed_station = starting_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": True}) for coupler in starting_station.couplers],
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )
    base_station = starting_stations[0]
    target_station = base_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": True}) for coupler in base_station.couplers],
            "branch_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )
    starting_station = base_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": False}) for coupler in base_station.couplers],
            "branch_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    expected = get_changing_switches_from_bus_groups(network=net, bus_groups=[target_station])
    result = get_changing_switches_from_changed_bus_groups(
        changed_bus_groups=[changed_station],
        starting_bus_groups=[starting_station],
    )

    SwitchUpdateSchema.validate(result)
    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))
