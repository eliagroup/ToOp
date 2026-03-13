import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from toop_engine_interfaces.asset_topology import Busbar, BusbarCoupler, Station, SwitchableAsset, Topology
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set, load_action_set_fs, save_action_set


def _build_large_random_action_set(
    rng: np.random.Generator,
    n_actions: int,
    n_stations: int,
    avg_assets_per_station: int,
    couplers_per_station: int,
) -> ActionSet:
    """Build a large random action set with grouped actions per station.

    The generated local actions follow the station-diff hypothesis, i.e. they only
    alter coupler open states and asset switching table values.
    """
    if n_actions % n_stations != 0:
        raise ValueError("n_actions must be divisible by n_stations.")

    actions_per_station = n_actions // n_stations

    # Deterministic profile with exact mean asset count equal to avg_assets_per_station.
    asset_counts = np.array(
        [
            avg_assets_per_station - 1,
            avg_assets_per_station,
            avg_assets_per_station + 1,
            avg_assets_per_station,
            avg_assets_per_station,
        ]
    )
    asset_counts = np.tile(asset_counts, n_stations // len(asset_counts))

    stations: list[Station] = []
    local_actions: list[Station] = []

    for station_idx in range(n_stations):
        grid_model_id = f"station_{station_idx:03d}"
        n_assets = int(asset_counts[station_idx])
        n_busbars = 2

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

        couplers = [
            BusbarCoupler.model_construct(
                grid_model_id=f"{grid_model_id}_coupler_{coupler_idx}",
                type=None,
                name=None,
                busbar_from_id=0,
                busbar_to_id=1,
                open=bool(rng.integers(0, 2)),
                in_service=True,
                asset_bay=None,
            )
            for coupler_idx in range(couplers_per_station)
        ]

        assets = [
            SwitchableAsset.model_construct(
                grid_model_id=f"{grid_model_id}_asset_{asset_idx}",
                type=None,
                name=None,
                in_service=True,
                branch_end=None,
                asset_bay=None,
            )
            for asset_idx in range(n_assets)
        ]

        starting_station = Station.model_construct(
            grid_model_id=grid_model_id,
            name=None,
            type=None,
            region=None,
            voltage_level=None,
            busbars=busbars,
            couplers=couplers,
            assets=assets,
            asset_switching_table=rng.integers(0, 2, size=(n_busbars, n_assets), dtype=np.uint8).astype(bool),
            asset_connectivity=None,
            model_log=None,
        )
        stations.append(starting_station)

        base_switching_table = np.asarray(starting_station.asset_switching_table, dtype=bool)
        for _ in range(actions_per_station):
            action_couplers = [
                coupler.model_copy(update={"open": bool(rng.integers(0, 2))}) for coupler in starting_station.couplers
            ]
            switching_table = base_switching_table.copy()
            flat_switching_table = switching_table.reshape(-1)
            n_flips = min(3, flat_switching_table.size)
            if n_flips > 0:
                flip_indices = rng.choice(flat_switching_table.size, size=n_flips, replace=False)
                flat_switching_table[flip_indices] = ~flat_switching_table[flip_indices]

            local_actions.append(
                starting_station.model_copy(
                    update={
                        "couplers": action_couplers,
                        "asset_switching_table": switching_table,
                    }
                )
            )

    starting_topology = Topology.model_construct(
        topology_id="performance_starting_topology",
        grid_model_file=None,
        name=None,
        stations=stations,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )

    return ActionSet(
        starting_topology=starting_topology,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=local_actions,
    )


@pytest.mark.performance
@pytest.mark.timeout(180)
def test_stored_action_set_large_performance(tmp_path: Path, record_property) -> None:
    """Benchmark large action-set serialization for new split format vs legacy JSON.

    Configuration:
    - 100,000 local action entries
    - 100 stations
    - Average 15 assets per station
    - 3 couplers per station
    """
    n_actions = 100_000
    n_stations = 100
    avg_assets_per_station = 15
    couplers_per_station = 3

    rng = np.random.default_rng(42)
    action_set = _build_large_random_action_set(
        rng=rng,
        n_actions=n_actions,
        n_stations=n_stations,
        avg_assets_per_station=avg_assets_per_station,
        couplers_per_station=couplers_per_station,
    )

    # Sanity checks for requested test configuration.
    assert len(action_set.starting_topology.stations) == n_stations
    assert len(action_set.local_actions) == n_actions
    mean_assets = float(np.mean([len(station.assets) for station in action_set.starting_topology.stations]))
    assert mean_assets == avg_assets_per_station
    assert all(len(station.couplers) == couplers_per_station for station in action_set.starting_topology.stations)

    old_file = tmp_path / "action_set_legacy.json"
    new_json = tmp_path / "action_set_split.json"
    new_diff = tmp_path / "action_set_split_diff.hdf5"

    t0 = time.perf_counter()
    save_pydantic_model_fs(filesystem=LocalFileSystem(), file_path=old_file, pydantic_model=action_set)
    old_save_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    save_action_set(json_file_path=new_json, diff_file_path=new_diff, action_set=action_set)
    new_save_seconds = time.perf_counter() - t0

    old_size_bytes = old_file.stat().st_size
    new_json_size_bytes = new_json.stat().st_size
    new_hdf5_size_bytes = new_diff.stat().st_size
    new_size_bytes = new_json_size_bytes + new_hdf5_size_bytes

    t0 = time.perf_counter()
    loaded_old = ActionSet.model_validate_json(old_file.read_text(encoding="utf-8"))
    old_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    loaded_new = load_action_set(new_json, new_diff)
    new_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    loaded_new_metadata_only = load_action_set_fs(LocalFileSystem(), new_json, diff_file_path=None)
    new_load_metadata_only_seconds = time.perf_counter() - t0

    assert len(loaded_old.local_actions) == n_actions
    assert len(loaded_new.local_actions) == n_actions
    assert loaded_new_metadata_only.local_actions == []

    # The key deterministic win of station-diff storage is total file size reduction.
    assert new_size_bytes < old_size_bytes
    assert new_load_seconds < old_load_seconds

    record_property("legacy_json_size_bytes", old_size_bytes)
    record_property("split_json_size_bytes", new_json_size_bytes)
    record_property("split_hdf5_size_bytes", new_hdf5_size_bytes)
    record_property("split_total_size_bytes", new_size_bytes)
    record_property("size_ratio_split_over_legacy", new_size_bytes / old_size_bytes)
    record_property("legacy_save_seconds", old_save_seconds)
    record_property("split_save_seconds", new_save_seconds)
    record_property("legacy_load_seconds", old_load_seconds)
    record_property("split_full_load_seconds", new_load_seconds)
    record_property("split_metadata_only_load_seconds", new_load_metadata_only_seconds)
    record_property("split_over_legacy_save_ratio", new_save_seconds / old_save_seconds)
    record_property("split_over_legacy_full_load_ratio", new_load_seconds / old_load_seconds)
