# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from toop_engine_interfaces.asset_topology import Busbar, BusbarCoupler, Station, SwitchableAsset, Topology
from toop_engine_interfaces.stored_action_set import (
    ActionSet,
    StationDiffArray,
    compress_actions_to_station_diffs,
    expand_station_diffs,
    load_action_set,
    load_station_diff_io,
    random_actions,
    save_action_set,
    store_station_diff_io,
)


class DummyStation:
    def __init__(self, grid_model_id):
        self.grid_model_id = str(grid_model_id)


@pytest.fixture
def action_set_multiple_subs() -> ActionSet:
    # 3 substations, each with 2 actions
    local_actions = [
        DummyStation(1),
        DummyStation(1),
        DummyStation(2),
        DummyStation(2),
        DummyStation(3),
        DummyStation(3),
    ]
    return ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=local_actions,
        global_actions=[],
    )


def test_random_actions_no_duplicates(action_set_multiple_subs: ActionSet):
    rng = np.random.default_rng(123)
    n_split_subs = 3
    result = random_actions(action_set_multiple_subs, rng, n_split_subs)
    assert len(result) == n_split_subs
    # Each index should correspond to a different substation
    chosen_subs = [action_set_multiple_subs.local_actions[i].grid_model_id for i in result]
    assert len(set(chosen_subs)) == len(chosen_subs)


def test_random_actions_clips_to_available_subs(action_set_multiple_subs: ActionSet):
    rng = np.random.default_rng(7)
    n_split_subs = 10  # more than available substations
    result = random_actions(action_set_multiple_subs, rng, n_split_subs)
    assert len(result) == 3  # only 3 substations available


def test_random_actions_empty_local_actions():
    rng = np.random.default_rng(0)
    action_set = ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[],
        global_actions=[],
    )
    result = random_actions(action_set, rng, 2)
    assert result == []


def test_random_actions_single_substation():
    rng = np.random.default_rng(0)
    local_actions = [DummyStation(42), DummyStation(42)]
    action_set = ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=local_actions,
        global_actions=[],
    )
    result = random_actions(action_set, rng, 1)
    assert len(result) == 1
    # The only possible indices are 0 or 1
    assert result[0] in [0, 1]


def test_store_and_load_station_diff_io_random_roundtrip():
    rng = np.random.default_rng(1234)

    station_diffs = []
    for station_idx in range(4):
        n_actions = int(rng.integers(1, 8))
        n_couplers = int(rng.integers(1, 6))
        n_busbars = int(rng.integers(1, 6))
        n_assets = int(rng.integers(1, 10))
        station_diffs.append(
            StationDiffArray(
                grid_model_id=f"station_{station_idx}",
                coupler_open=rng.integers(0, 2, size=(n_actions, n_couplers), dtype=np.uint8).astype(bool),
                switching_table=rng.integers(0, 2, size=(n_actions, n_busbars, n_assets), dtype=np.uint8).astype(bool),
            )
        )

    buffer = io.BytesIO()
    store_station_diff_io(buffer, station_diffs)
    buffer.seek(0)

    loaded = load_station_diff_io(buffer)

    loaded_by_id = {station_diff.grid_model_id: station_diff for station_diff in loaded}
    assert set(loaded_by_id) == {station_diff.grid_model_id for station_diff in station_diffs}

    for original in station_diffs:
        result = loaded_by_id[original.grid_model_id]
        assert result.grid_model_id == original.grid_model_id
        assert result.coupler_open.dtype == bool
        assert result.switching_table.dtype == bool
        assert np.array_equal(result.coupler_open, original.coupler_open)
        assert np.array_equal(result.switching_table, original.switching_table)


def test_store_and_load_station_diff_io_empty_list():
    buffer = io.BytesIO()
    store_station_diff_io(buffer, [])
    buffer.seek(0)

    loaded = load_station_diff_io(buffer)

    assert loaded == []


def test_compress_and_expand_station_diffs_random_roundtrip():
    rng = np.random.default_rng(20260313)

    starting_stations: list[Station] = []
    actions: list[Station] = []
    expected_by_station: dict[str, list[Station]] = {}

    n_stations = 4
    for station_idx in range(n_stations):
        grid_model_id = f"station_{station_idx}"
        n_busbars = int(rng.integers(1, 5))
        n_assets = int(rng.integers(1, 7))
        n_couplers = int(rng.integers(1, 6))
        n_actions = int(rng.integers(1, 6))

        starting_couplers = [
            BusbarCoupler.model_construct(
                grid_model_id=f"{grid_model_id}_coupler_{coupler_idx}",
                type=None,
                name=None,
                busbar_from_id=0,
                busbar_to_id=0,
                open=bool(rng.integers(0, 2)),
                in_service=True,
                asset_bay=None,
            )
            for coupler_idx in range(n_couplers)
        ]

        starting_station = Station.model_construct(
            grid_model_id=grid_model_id,
            name=None,
            type=None,
            region=None,
            voltage_level=None,
            busbars=[],
            couplers=starting_couplers,
            assets=[],
            asset_switching_table=rng.integers(0, 2, size=(n_busbars, n_assets), dtype=np.uint8).astype(bool),
            asset_connectivity=None,
            model_log=None,
        )
        starting_stations.append(starting_station)

        station_actions: list[Station] = []
        for _ in range(n_actions):
            couplers = [
                coupler.model_copy(update={"open": bool(rng.integers(0, 2))}) for coupler in starting_station.couplers
            ]
            switching_table = rng.integers(0, 2, size=(n_busbars, n_assets), dtype=np.uint8).astype(bool)
            station_actions.append(
                starting_station.model_copy(
                    update={
                        "couplers": couplers,
                        "asset_switching_table": switching_table,
                    }
                )
            )

        expected_by_station[grid_model_id] = station_actions
        actions.extend(station_actions)

    starting_topology = Topology.model_construct(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        stations=starting_stations,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    station_diffs = compress_actions_to_station_diffs(starting_topology, actions)
    expanded_actions = expand_station_diffs(starting_topology, station_diffs)

    result_by_station: dict[str, list[Station]] = {grid_model_id: [] for grid_model_id in expected_by_station}
    for action in expanded_actions:
        result_by_station[action.grid_model_id].append(action)

    assert set(result_by_station) == set(expected_by_station)

    for grid_model_id, expected_actions in expected_by_station.items():
        result_actions = result_by_station[grid_model_id]
        assert len(result_actions) == len(expected_actions)

        for expected_action, result_action in zip(expected_actions, result_actions):
            expected_coupler_open = [coupler.open for coupler in expected_action.couplers]
            result_coupler_open = [coupler.open for coupler in result_action.couplers]
            assert result_coupler_open == expected_coupler_open
            assert np.array_equal(result_action.asset_switching_table, expected_action.asset_switching_table)


def test_compress_station_diffs_raises_on_non_diff_hypothesis_change():
    starting_station = Station.model_construct(
        grid_model_id="station_x",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=[],
        couplers=[
            BusbarCoupler.model_construct(
                grid_model_id="station_x_coupler_0",
                type=None,
                name=None,
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                in_service=True,
                asset_bay=None,
            )
        ],
        assets=[
            SwitchableAsset.model_construct(
                grid_model_id="station_save_load_asset_1",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay=None,
            ),
            SwitchableAsset.model_construct(
                grid_model_id="station_save_load_asset_2",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay=None,
            ),
        ],
        asset_switching_table=np.array([[True, False], [False, True]], dtype=bool),
        asset_connectivity=None,
        model_log=None,
    )

    starting_topology = Topology.model_construct(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        stations=[starting_station],
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    valid_action = starting_station.model_copy(
        update={
            "couplers": [
                starting_station.couplers[0].model_copy(update={"open": True}),
            ],
            "asset_switching_table": np.array([[False, True]], dtype=bool),
        }
    )

    invalid_action = starting_station.model_copy(
        update={
            "couplers": [
                starting_station.couplers[0].model_copy(update={"busbar_from_id": 99}),
            ],
            "asset_switching_table": np.array([[True, True]], dtype=bool),
        }
    )

    actions = [valid_action, invalid_action]

    with pytest.raises(ValueError, match="coupler structure|fields other than coupler open states"):
        compress_actions_to_station_diffs(
            starting_topology=starting_topology,
            actions=actions,
            validate_diff_hypothesis=True,
        )


def test_save_and_load_action_set_split_files_roundtrip(tmp_path: Path):
    starting_station = Station.model_construct(
        grid_model_id="station_save_load",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=[
            Busbar.model_construct(
                grid_model_id="station_save_load_busbar_1",
                type=None,
                name=None,
                int_id=1,
                in_service=True,
                bus_branch_bus_id=None,
            ),
            Busbar.model_construct(
                grid_model_id="station_save_load_busbar_2",
                type=None,
                name=None,
                int_id=2,
                in_service=True,
                bus_branch_bus_id=None,
            ),
        ],
        couplers=[
            BusbarCoupler.model_construct(
                grid_model_id="station_save_load_coupler_0",
                type=None,
                name=None,
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                in_service=True,
                asset_bay=None,
            )
        ],
        assets=[
            SwitchableAsset.model_construct(
                grid_model_id="station_save_load_asset_1",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay=None,
            ),
            SwitchableAsset.model_construct(
                grid_model_id="station_save_load_asset_2",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay=None,
            ),
        ],
        asset_switching_table=np.array([[True, False], [False, True]], dtype=bool),
        asset_connectivity=None,
        model_log=None,
    )

    starting_topology = Topology.model_construct(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        stations=[starting_station],
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    local_action = starting_station.model_copy(
        update={
            "couplers": [starting_station.couplers[0].model_copy(update={"open": True})],
            "asset_switching_table": np.array([[False, True], [True, False]], dtype=bool),
        }
    )
    action_set = ActionSet.model_construct(
        starting_topology=starting_topology,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[local_action],
    )

    json_file = tmp_path / "action_set.json"
    diff_file = tmp_path / "action_set.hdf5"
    save_action_set(json_file, diff_file, action_set)

    assert json_file.exists()
    assert diff_file.exists()

    loaded_action_set = load_action_set(json_file, diff_file)

    assert len(loaded_action_set.local_actions) == 1
    loaded_action = loaded_action_set.local_actions[0]
    assert loaded_action.grid_model_id == local_action.grid_model_id
    assert [c.open for c in loaded_action.couplers] == [c.open for c in local_action.couplers]
    assert np.array_equal(loaded_action.asset_switching_table, local_action.asset_switching_table)
