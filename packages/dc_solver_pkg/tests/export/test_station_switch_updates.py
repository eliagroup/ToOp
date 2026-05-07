# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from toop_engine_dc_solver.export.station_switch_updates import (
    _get_asset_switch_diffs,
    _get_coupler_switch_diffs,
    _resolve_changed_stations,
    get_changing_switches_from_changed_stations,
)
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import get_changing_switches_from_topology
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema


def test_resolve_changed_stations_preserves_topology_order(basic_node_breaker_topology):
    changed_station = basic_node_breaker_topology.stations[0]

    starting_lookup, changed_lookup, ordered_station_ids = _resolve_changed_stations(
        changed_stations=[changed_station],
        starting_topology=basic_node_breaker_topology,
    )

    assert list(starting_lookup) == [changed_station.grid_model_id]
    assert list(changed_lookup) == [changed_station.grid_model_id]
    assert ordered_station_ids == [changed_station.grid_model_id]


def test_get_coupler_switch_diffs(basic_node_breaker_topology):
    starting_station = basic_node_breaker_topology.stations[0].model_copy(
        update={
            "couplers": [
                coupler.model_copy(update={"open": False}) for coupler in basic_node_breaker_topology.stations[0].couplers
            ]
        }
    )
    changed_station = basic_node_breaker_topology.stations[0]

    result = _get_coupler_switch_diffs(
        changed_station=changed_station,
        starting_station=starting_station,
    )

    assert result == [{"grid_model_id": "VL4_BREAKER", "open": True}]


def test_get_asset_switch_diffs(basic_node_breaker_topology):
    target_station = basic_node_breaker_topology.stations[0]
    starting_station = target_station.model_copy(
        update={
            "asset_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    changed_station = target_station.model_copy(
        update={
            "asset_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )

    result = _get_asset_switch_diffs(
        changed_station=changed_station,
        starting_station=starting_station,
    )

    assert result == [
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
    ]


def test_get_asset_switch_diffs_requires_matching_switching_table_shape(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    changed_station = station.model_copy(
        update={
            "asset_switching_table": np.array([[False, False, True]], dtype=bool),
        }
    )

    with pytest.raises(ValueError, match="asset switching table shape does not match"):
        _get_asset_switch_diffs(
            changed_station=changed_station,
            starting_station=station,
        )


def test_get_asset_switch_diffs_requires_matching_asset_order(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    reordered_assets = [station.assets[1], station.assets[0], station.assets[2]]
    changed_station = station.model_copy(update={"assets": reordered_assets})

    with pytest.raises(ValueError, match="Use ActionSet.simplified_starting_topology as input"):
        _get_asset_switch_diffs(
            changed_station=changed_station,
            starting_station=station,
        )


def test_get_asset_switch_diffs_allows_multiple_active_busbars(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    starting_station = station.model_copy(
        update={
            "asset_switching_table": np.array([[True, False, False], [True, True, False]], dtype=bool),
        }
    )
    changed_station = station.model_copy(
        update={
            "asset_switching_table": np.array([[True, False, False], [False, True, False]], dtype=bool),
        }
    )

    result = _get_asset_switch_diffs(
        changed_station=changed_station,
        starting_station=starting_station,
    )

    assert result == [{"grid_model_id": "L42_DISCONNECTOR_3_1", "open": True}]


def test_get_changing_switches_from_changed_stations_matches_network_diff(
    basic_node_breaker_grid_v1,
    basic_node_breaker_topology,
):
    net = basic_node_breaker_grid_v1
    starting_station = basic_node_breaker_topology.stations[0].model_copy(
        update={
            "couplers": [
                coupler.model_copy(update={"open": False}) for coupler in basic_node_breaker_topology.stations[0].couplers
            ],
            "asset_switching_table": np.array([[True, False, True], [False, True, False]], dtype=bool),
        }
    )
    changed_station = starting_station.model_copy(
        update={
            "couplers": [coupler.model_copy(update={"open": True}) for coupler in starting_station.couplers],
            "asset_switching_table": np.array([[False, False, True], [True, True, False]], dtype=bool),
        }
    )
    target_topology = basic_node_breaker_topology.model_copy(update={"stations": [changed_station]})
    starting_topology = target_topology.model_copy(update={"stations": [starting_station]})

    expected = get_changing_switches_from_topology(network=net, target_topology=target_topology)
    result = get_changing_switches_from_changed_stations(
        changed_stations=[changed_station],
        starting_topology=starting_topology,
    )

    SwitchUpdateSchema.validate(result)
    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))
