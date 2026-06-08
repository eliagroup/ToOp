# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from pydantic import ValidationError
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    MaterializedStation,
    RawStation,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.stored_action_set import (
    ActionSet,
    StationDiffArray,
    compress_actions_to_station_diffs,
    expand_station_diffs,
    load_action_set,
    load_station_diff_fs,
    random_actions,
    save_action_set,
    store_station_diff_fs,
    validate_actions_grouped,
)


def build_raw_station(
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    asset_ids: list[str],
    asset_switching_table: np.ndarray,
    asset_terminals: list[str | None] | None = None,
    asset_bay_ids: list[str | None] | None = None,
) -> RawStation:
    """Build a raw station from explicit raw-topology fields.

    Parameters
    ----------
    grid_model_id : str
        Identifier of the station in the grid model.
    busbars : list[Busbar]
        Busbars belonging to the station.
    couplers : list[BusbarCoupler]
        Couplers belonging to the station.
    asset_ids : list[str]
        Grid model ids of the assets connected to the station.
    asset_switching_table : np.ndarray
        Busbar-to-asset switching matrix for the station.
    asset_terminals : list[str | None] | None, optional
        Optional branch-end metadata aligned with ``asset_ids``.
    asset_bay_ids : list[str | None] | None, optional
        Optional asset-bay metadata aligned with ``asset_ids``.

    Returns
    -------
    RawStation
        Raw station representation suitable for topology construction in tests.
    """
    return RawStation(
        grid_model_id=grid_model_id,
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        asset_ids=asset_ids,
        asset_terminals=asset_terminals if asset_terminals is not None else [None] * len(asset_ids),
        asset_bay_ids=asset_bay_ids if asset_bay_ids is not None else [None] * len(asset_ids),
        asset_switching_table=asset_switching_table,
        asset_connectivity=None,
        model_log=None,
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
    )
    result = random_actions(action_set, rng, 1)
    assert len(result) == 1
    # The only possible indices are 0 or 1
    assert result[0] in [0, 1]


def test_store_and_load_station_diff_io_random_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(1234)
    filesystem = DirFileSystem(str(tmp_path))
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

    store_station_diff_fs(filesystem, station_diffs, "station_diffs.hdf5")

    loaded = load_station_diff_fs(filesystem, "station_diffs.hdf5")

    loaded_by_id = {station_diff.grid_model_id: station_diff for station_diff in loaded}
    assert set(loaded_by_id) == {station_diff.grid_model_id for station_diff in station_diffs}

    for original in station_diffs:
        result = loaded_by_id[original.grid_model_id]
        assert result.grid_model_id == original.grid_model_id
        assert result.coupler_open.dtype == bool
        assert result.switching_table.dtype == bool
        assert np.array_equal(result.coupler_open, original.coupler_open)
        assert np.array_equal(result.switching_table, original.switching_table)


def test_store_and_load_station_diff_io_empty_list(tmp_path: Path):
    filesystem = DirFileSystem(str(tmp_path))
    store_station_diff_fs(filesystem, [], "station_diffs.hdf5")
    loaded = load_station_diff_fs(filesystem, "station_diffs.hdf5")
    assert loaded == []


def test_station_diff_array_raises_for_mismatched_action_count() -> None:
    with pytest.raises(ValueError, match="same n_actions dimension"):
        StationDiffArray(
            grid_model_id="station_1",
            coupler_open=np.zeros((5, 1), dtype=bool),
            switching_table=np.zeros((10, 2, 7), dtype=bool),
        )


def test_store_and_load_station_diff_io_supports_different_station_action_counts(tmp_path: Path) -> None:
    filesystem = DirFileSystem(str(tmp_path))
    station_diffs = [
        StationDiffArray(
            grid_model_id="station_1",
            coupler_open=np.zeros((5, 1), dtype=bool),
            switching_table=np.zeros((5, 2, 7), dtype=bool),
        ),
        StationDiffArray(
            grid_model_id="station_2",
            coupler_open=np.zeros((10, 2), dtype=bool),
            switching_table=np.zeros((10, 3, 4), dtype=bool),
        ),
    ]

    store_station_diff_fs(filesystem, station_diffs, "station_diffs.hdf5")
    loaded = load_station_diff_fs(filesystem, "station_diffs.hdf5")

    assert [(station_diff.grid_model_id, station_diff.coupler_open.shape) for station_diff in loaded] == [
        ("station_1", (5, 1)),
        ("station_2", (10, 2)),
    ]
    assert [station_diff.switching_table.shape for station_diff in loaded] == [(5, 2, 7), (10, 3, 4)]


def test_store_and_load_station_diff_io_preserves_station_order(tmp_path: Path) -> None:
    filesystem = DirFileSystem(str(tmp_path))
    station_diffs = [
        StationDiffArray(
            grid_model_id="station_10",
            coupler_open=np.zeros((1, 1), dtype=bool),
            switching_table=np.zeros((1, 1, 1), dtype=bool),
        ),
        StationDiffArray(
            grid_model_id="station_2",
            coupler_open=np.zeros((1, 1), dtype=bool),
            switching_table=np.zeros((1, 1, 1), dtype=bool),
        ),
        StationDiffArray(
            grid_model_id="station_1",
            coupler_open=np.zeros((1, 1), dtype=bool),
            switching_table=np.zeros((1, 1, 1), dtype=bool),
        ),
    ]

    store_station_diff_fs(filesystem, station_diffs, "station_diffs.hdf5")
    loaded = load_station_diff_fs(filesystem, "station_diffs.hdf5")

    assert [station_diff.grid_model_id for station_diff in loaded] == [
        "station_10",
        "station_2",
        "station_1",
    ]


def test_validate_actions_grouped_accepts_grouped_actions():
    station_s1 = MaterializedStation.model_construct(
        grid_model_id="s1",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=[],
        couplers=[],
        assets=[],
        asset_switching_table=np.zeros((0, 0), dtype=bool),
        asset_connectivity=None,
        model_log=None,
    )
    station_s2 = station_s1.model_copy(update={"grid_model_id": "s2"})
    station_s3 = station_s1.model_copy(update={"grid_model_id": "s3"})

    actions = [station_s1, station_s1, station_s2, station_s3, station_s3]
    validate_actions_grouped(actions)


def test_validate_actions_grouped_raises_for_non_grouped_actions():
    station_s1 = MaterializedStation.model_construct(
        grid_model_id="s1",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=[],
        couplers=[],
        assets=[],
        asset_switching_table=np.zeros((0, 0), dtype=bool),
        asset_connectivity=None,
        model_log=None,
    )
    station_s2 = station_s1.model_copy(update={"grid_model_id": "s2"})

    actions = [station_s1, station_s2, station_s1]
    with pytest.raises(ValueError, match="not grouped by station"):
        validate_actions_grouped(actions)


def test_action_set_model_validator_rejects_non_grouped_local_actions():
    busbars = [
        Busbar.model_construct(
            grid_model_id="station_a_busbar_0",
            type=None,
            name=None,
            int_id=0,
            in_service=True,
            bus_branch_bus_id=None,
        )
    ]
    assets = [
        SwitchableAsset.model_construct(
            grid_model_id="station_a_asset_0",
            type=None,
            name=None,
            in_service=True,
            branch_end=None,
            asset_bay_id=None,
        )
    ]

    station_a = MaterializedStation.model_construct(
        grid_model_id="station_a",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=[],
        assets=assets,
        asset_switching_table=np.zeros((1, 1), dtype=bool),
        asset_connectivity=None,
        model_log=None,
    )
    station_b = station_a.model_copy(
        update={
            "grid_model_id": "station_b",
            "assets": [station_a.assets[0].model_copy(update={"grid_model_id": "station_b_asset_0"})],
        }
    )

    starting_topology = Topology(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        raw_stations=[
            RawStation(
                grid_model_id="station_a",
                name=None,
                type=None,
                region=None,
                voltage_level=None,
                busbars=busbars,
                couplers=[],
                asset_ids=["station_a_asset_0"],
                asset_terminals=[None],
                asset_bay_ids=[None],
                asset_switching_table=np.zeros((1, 1), dtype=bool),
                asset_connectivity=None,
                model_log=None,
            ),
            RawStation(
                grid_model_id="station_b",
                name=None,
                type=None,
                region=None,
                voltage_level=None,
                busbars=busbars,
                couplers=[],
                asset_ids=["station_b_asset_0"],
                asset_terminals=[None],
                asset_bay_ids=[None],
                asset_switching_table=np.zeros((1, 1), dtype=bool),
                asset_connectivity=None,
                model_log=None,
            ),
        ],
        assets=station_a.assets + station_b.assets,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    non_grouped_local_actions = [station_a, station_b, station_a]
    with pytest.raises(ValidationError, match="not grouped by station"):
        ActionSet(
            starting_topology=starting_topology,
            simplified_starting_topology=starting_topology,
            connectable_branches=[],
            disconnectable_branches=[],
            pst_ranges=[],
            hvdc_ranges=[],
            local_actions=non_grouped_local_actions,
        )


def test_compress_and_expand_station_diffs_random_roundtrip():
    rng = np.random.default_rng(20260313)

    starting_stations: list[MaterializedStation] = []
    starting_raw_stations: list[RawStation] = []
    starting_assets: list[SwitchableAsset] = []
    actions: list[MaterializedStation] = []
    expected_by_station: dict[str, list[MaterializedStation]] = {}

    n_stations = 4
    for station_idx in range(n_stations):
        grid_model_id = f"station_{station_idx}"
        n_busbars = int(rng.integers(2, 5))
        n_assets = int(rng.integers(1, 7))
        n_couplers = int(rng.integers(1, 6))
        n_actions = int(rng.integers(1, 6))

        busbars = [
            Busbar.model_construct(
                grid_model_id=f"{grid_model_id}_busbar_{busbar_idx}",
                type=None,
                name=None,
                int_id=busbar_idx,
                in_service=True,
                bus_branch_bus_id=None,
            )
            for busbar_idx in range(n_busbars)
        ]
        assets = [
            SwitchableAsset.model_construct(
                grid_model_id=f"{grid_model_id}_asset_{asset_idx}",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay_id=None,
            )
            for asset_idx in range(n_assets)
        ]

        starting_couplers = [
            BusbarCoupler.model_construct(
                grid_model_id=f"{grid_model_id}_coupler_{coupler_idx}",
                type=None,
                name=None,
                busbar_from_id=coupler_idx % n_busbars,
                busbar_to_id=(coupler_idx + 1) % n_busbars,
                open=bool(rng.integers(0, 2)),
                in_service=True,
                asset_bay=None,
            )
            for coupler_idx in range(n_couplers)
        ]

        starting_switching_table = rng.integers(0, 2, size=(n_busbars, n_assets), dtype=np.uint8).astype(bool)
        starting_raw_stations.append(
            build_raw_station(
                grid_model_id,
                busbars,
                starting_couplers,
                [asset.grid_model_id for asset in assets],
                starting_switching_table,
            )
        )
        starting_assets.extend(assets)

        starting_station = MaterializedStation.model_construct(
            grid_model_id=grid_model_id,
            name=None,
            type=None,
            region=None,
            voltage_level=None,
            busbars=busbars,
            couplers=starting_couplers,
            assets=assets,
            asset_switching_table=starting_switching_table,
            asset_connectivity=None,
            model_log=None,
        )
        starting_stations.append(starting_station)

        station_actions: list[MaterializedStation] = []
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

    starting_topology = Topology(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        raw_stations=starting_raw_stations,
        assets=starting_assets,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    station_diffs = compress_actions_to_station_diffs(starting_topology, actions)
    expanded_actions = expand_station_diffs(starting_topology, station_diffs)

    result_by_station: dict[str, list[MaterializedStation]] = {grid_model_id: [] for grid_model_id in expected_by_station}
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
    busbars = [
        Busbar.model_construct(
            grid_model_id="station_x_busbar_1",
            type=None,
            name=None,
            int_id=1,
            in_service=True,
            bus_branch_bus_id=None,
        ),
        Busbar.model_construct(
            grid_model_id="station_x_busbar_2",
            type=None,
            name=None,
            int_id=2,
            in_service=True,
            bus_branch_bus_id=None,
        ),
    ]
    couplers = [
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
    ]
    assets = [
        SwitchableAsset.model_construct(
            grid_model_id="station_save_load_asset_1",
            type=None,
            name=None,
            in_service=True,
            branch_end=None,
            asset_bay_id=None,
        ),
        SwitchableAsset.model_construct(
            grid_model_id="station_save_load_asset_2",
            type=None,
            name=None,
            in_service=True,
            branch_end=None,
            asset_bay_id=None,
        ),
    ]
    asset_switching_table = np.array([[True, False], [False, True]], dtype=bool)

    starting_topology = Topology(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        raw_stations=[
            build_raw_station(
                "station_x",
                busbars,
                couplers,
                [asset.grid_model_id for asset in assets],
                asset_switching_table,
            )
        ],
        assets=assets,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    starting_station = MaterializedStation.model_construct(
        grid_model_id="station_x",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=None,
        model_log=None,
    )

    valid_action = starting_station.model_copy(
        update={
            "couplers": [
                starting_station.couplers[0].model_copy(update={"open": True}),
            ],
            "asset_switching_table": np.array([[False, True], [True, False]], dtype=bool),
        }
    )

    invalid_action = starting_station.model_copy(
        update={
            "couplers": [
                starting_station.couplers[0].model_copy(update={"busbar_from_id": 99}),
            ],
            "asset_switching_table": np.array([[True, True], [False, False]], dtype=bool),
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
    busbars = [
        Busbar.model_construct(
            grid_model_id="busbar1",
            type=None,
            name=None,
            int_id=1,
            in_service=True,
            bus_branch_bus_id=None,
        ),
        Busbar.model_construct(
            grid_model_id="busbar2",
            type=None,
            name=None,
            int_id=2,
            in_service=True,
            bus_branch_bus_id=None,
        ),
    ]
    couplers = [
        BusbarCoupler.model_construct(
            grid_model_id="coupler1",
            type=None,
            name=None,
            busbar_from_id=1,
            busbar_to_id=2,
            open=False,
            in_service=True,
            asset_bay=None,
        )
    ]
    assets = [
        SwitchableAsset.model_construct(
            grid_model_id="asset1",
            type=None,
            name=None,
            in_service=True,
            branch_end=None,
            asset_bay_id=None,
        ),
        SwitchableAsset.model_construct(
            grid_model_id="asset2",
            type=None,
            name=None,
            in_service=True,
            branch_end=None,
            asset_bay_id=None,
        ),
    ]
    asset_switching_table = np.array([[True, False], [False, True]], dtype=bool)

    starting_topology = Topology(
        topology_id="starting_topology",
        grid_model_file=None,
        name=None,
        raw_stations=[
            build_raw_station(
                "station1",
                busbars,
                couplers,
                [asset.grid_model_id for asset in assets],
                asset_switching_table,
            )
        ],
        assets=assets,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    starting_station = MaterializedStation.model_construct(
        grid_model_id="station1",
        name=None,
        type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=None,
        model_log=None,
    )

    local_action = starting_station.model_copy(
        update={
            "couplers": [starting_station.couplers[0].model_copy(update={"open": True})],
            "asset_switching_table": np.array([[False, True], [True, False]], dtype=bool),
        }
    )
    action_set = ActionSet.model_construct(
        starting_topology=starting_topology,
        simplified_starting_topology=starting_topology,
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
    assert np.array_equal(
        np.asarray(loaded_action.asset_switching_table),
        np.asarray(local_action.asset_switching_table),
    )
