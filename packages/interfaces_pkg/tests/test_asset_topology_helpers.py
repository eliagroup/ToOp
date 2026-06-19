# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime

import numpy as np
import pytest
from toop_engine_interfaces.asset_topology import (
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
    MaterializedAssetConnection,
    MaterializedStation,
    RawStation,
    StationAssetConnection,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import (
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    find_busbars_for_coupler,
    fuse_all_couplers_with_type,
    fuse_coupler,
    has_transmission_line_switching,
    merge_stations,
    order_station_assets,
    order_topology,
    reindex_busbars,
    station_diff,
    topology_diff,
)


def build_raw_station(
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    asset_ids: list[str],
    asset_switching_table: np.ndarray,
    injection_asset_ids: list[str] | None = None,
    injection_switching_table: np.ndarray | None = None,
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
    resolved_injection_asset_ids = injection_asset_ids if injection_asset_ids is not None else []
    resolved_injection_switching_table = (
        injection_switching_table
        if injection_switching_table is not None
        else np.zeros((len(busbars), len(resolved_injection_asset_ids)), dtype=bool)
    )

    return RawStation(
        grid_model_id=grid_model_id,
        name=None,
        station_type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        branch_connections=[
            StationAssetConnection(asset_id=asset_id, terminal=asset_terminal, asset_bay_id=asset_bay_id)
            for asset_id, asset_terminal, asset_bay_id in zip(
                asset_ids,
                asset_terminals if asset_terminals is not None else [None] * len(asset_ids),
                asset_bay_ids if asset_bay_ids is not None else [None] * len(asset_ids),
                strict=True,
            )
        ],
        injection_connections=[
            StationAssetConnection(asset_id=asset_id, terminal=None, asset_bay_id=None)
            for asset_id in resolved_injection_asset_ids
        ],
        branch_switching_table=asset_switching_table,
        injection_switching_table=resolved_injection_switching_table,
        branch_connectivity=None,
        injection_connectivity=None,
        model_log=None,
    )


def build_materialized_station(
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    assets: list[SwitchableAsset],
    asset_switching_table: np.ndarray,
    asset_connectivity: np.ndarray | None = None,
    injection_assets: list[SwitchableAsset] | None = None,
    injection_switching_table: np.ndarray | None = None,
    injection_connectivity: np.ndarray | None = None,
) -> MaterializedStation:
    """Build a materialized station using explicit asset connections."""
    resolved_injection_assets = injection_assets if injection_assets is not None else []
    resolved_injection_switching_table = (
        injection_switching_table
        if injection_switching_table is not None
        else np.zeros((len(busbars), len(resolved_injection_assets)), dtype=bool)
    )
    resolved_injection_connectivity = injection_connectivity if injection_connectivity is not None else None
    branch_assets = [
        asset if isinstance(asset, BranchAsset) else BranchAsset.model_validate(asset.model_dump()) for asset in assets
    ]
    normalized_injection_assets = [
        asset if isinstance(asset, InjectionAsset) else InjectionAsset.model_validate(asset.model_dump())
        for asset in resolved_injection_assets
    ]
    return MaterializedStation(
        grid_model_id=grid_model_id,
        name=None,
        station_type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        branch_connections=[MaterializedAssetConnection(asset=asset) for asset in branch_assets],
        injection_connections=[MaterializedAssetConnection(asset=asset) for asset in normalized_injection_assets],
        branch_switching_table=asset_switching_table,
        injection_switching_table=resolved_injection_switching_table,
        branch_connectivity=asset_connectivity,
        injection_connectivity=resolved_injection_connectivity,
        model_log=None,
    )


def build_test_topology(topology_id: str, raw_stations: list[RawStation], assets: list[SwitchableAsset]) -> Topology:
    """Build a topology from raw stations and topology-owned assets.

    Parameters
    ----------
    topology_id : str
        Identifier of the topology.
    raw_stations : list[RawStation]
        Raw stations that define the topology layout.
    assets : list[SwitchableAsset]
        Topology-owned assets referenced by the raw stations.

    Returns
    -------
    Topology
        Topology instance for helper-focused tests.
    """
    branch_assets = [
        asset if isinstance(asset, BranchAsset) else BranchAsset.model_validate(asset.model_dump())
        for asset in assets
        if not isinstance(asset, InjectionAsset)
    ]
    injection_assets = [
        asset if isinstance(asset, InjectionAsset) else InjectionAsset.model_validate(asset.model_dump())
        for asset in assets
        if isinstance(asset, InjectionAsset)
    ]

    return Topology(
        topology_id=topology_id,
        grid_model_file=None,
        name=None,
        raw_stations=raw_stations,
        branch_assets=branch_assets,
        injection_assets=injection_assets,
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )


def test_merge_stations():
    station1 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[False, False, True], [True, True, False]]),
        grid_model_id="station2",
    )

    original_stations = [station1, station2]
    new_stations = [
        station1.model_copy(
            update={
                "couplers": [
                    BusbarCoupler(
                        busbar_from_id=1,
                        busbar_to_id=2,
                        open=True,
                        grid_model_id="coupler1",
                    )
                ]
            }
        ),
        station2.model_copy(update={"branch_switching_table": np.array([[False, True, True], [True, False, False]])}),
    ]

    updated_stations, coupler_diff, reassignment_diff = merge_stations(original_stations, new_stations, "raise")

    assert len(updated_stations) == 2
    assert updated_stations[0].grid_model_id == "station1"
    assert updated_stations[0].couplers[0].open
    assert coupler_diff == [("station1", new_stations[0].couplers[0])]

    assert updated_stations[1].grid_model_id == "station2"
    assert np.array_equal(
        updated_stations[1].branch_switching_table,
        np.array([[False, True, True], [True, False, False]]),
    )
    assert set(reassignment_diff) == set([("station2", 1, 0, True), ("station2", 1, 1, False)])


def test_merge_stations_append_behavior():
    station1 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[False, False, True], [True, True, False]]),
        grid_model_id="station2",
    )

    original_stations = [station1]
    new_stations = [
        station1.model_copy(
            update={
                "couplers": [
                    BusbarCoupler(
                        busbar_from_id=1,
                        busbar_to_id=2,
                        open=True,
                        grid_model_id="coupler1",
                    )
                ]
            }
        ),
        station2.model_copy(update={"branch_switching_table": np.array([[False, True, True], [True, False, False]])}),
    ]

    updated_stations, coupler_diff, reassignment_diff = merge_stations(original_stations, new_stations, "append")

    assert len(updated_stations) == 2
    assert updated_stations[0].grid_model_id == "station1"
    assert updated_stations[0].couplers[0].open
    assert coupler_diff == [("station1", new_stations[0].couplers[0])]

    assert updated_stations[1].grid_model_id == "station2"
    assert np.array_equal(
        updated_stations[1].branch_switching_table,
        np.array([[False, True, True], [True, False, False]]),
    )
    assert reassignment_diff == []


def test_merge_stations_raise_behavior():
    station1 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[False, False, True], [True, True, False]]),
        grid_model_id="station2",
    )

    original_stations = [station1]
    new_stations = [station2]

    with pytest.raises(ValueError, match="Station station2 was not found in the original list"):
        merge_stations(original_stations, new_stations, "raise")


def test_merge_stations_with_new_station_append():
    station1 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[False, False, True], [True, True, False]]),
        grid_model_id="station2",
    )

    original_stations = [station1]
    new_stations = [station2]

    updated_stations, coupler_diff, reassignment_diff = merge_stations(original_stations, new_stations, "append")

    assert len(updated_stations) == 2
    assert updated_stations[0].grid_model_id == "station1"
    assert updated_stations[1].grid_model_id == "station2"
    assert coupler_diff == []
    assert reassignment_diff == []


def test_get_connected_assets():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1", in_service=True),
            SwitchableAsset(grid_model_id="line2", in_service=False),
            SwitchableAsset(grid_model_id="line3", in_service=True),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )

    # Test for busbar 0
    connected_assets_busbar_0 = station.get_connected_assets(0)
    assert len(connected_assets_busbar_0) == 2
    assert connected_assets_busbar_0[0].grid_model_id == "line1"
    assert connected_assets_busbar_0[1].grid_model_id == "line3"

    # Test for busbar 1
    connected_assets_busbar_1 = station.get_connected_assets(1)
    assert len(connected_assets_busbar_1) == 0

    # Test with no assets in service
    station.branch_connections[0].asset.in_service = False
    station.branch_connections[2].asset.in_service = False
    connected_assets_busbar_0 = station.get_connected_assets(0)
    assert len(connected_assets_busbar_0) == 0


def test_get_connected_assets_raw_station_injection_scope():
    station = RawStation(
        grid_model_id="station1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[],
        branch_connections=[StationAssetConnection(asset_id="line1", terminal="from", asset_bay_id=None)],
        injection_connections=[StationAssetConnection(asset_id="gen1", terminal=None, asset_bay_id=None)],
        branch_switching_table=np.array([[True], [False]]),
        injection_switching_table=np.array([[False], [True]]),
    )

    connected_assets_busbar_1 = station.get_connected_assets(
        1,
        topology_assets=[InjectionAsset(grid_model_id="gen1", in_service=True)],
        asset_scope="injection",
    )

    assert len(connected_assets_busbar_1) == 1
    assert connected_assets_busbar_1[0].grid_model_id == "gen1"


def test_find_busbars_for_coupler():
    busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
        Busbar(int_id=3, grid_model_id="busbar3"),
    ]

    # Test case: Valid coupler with matching busbars
    coupler = BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1")
    busbar_from, busbar_to = find_busbars_for_coupler(busbars, coupler)
    assert busbar_from.int_id == 1
    assert busbar_to.int_id == 2

    # Test case: Coupler with non-existent busbar_from_id
    invalid_coupler_from = BusbarCoupler(busbar_from_id=4, busbar_to_id=2, open=False, grid_model_id="coupler2")
    with pytest.raises(ValueError, match="Busbars for coupler coupler2 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_from)

    # Test case: Coupler with non-existent busbar_to_id
    invalid_coupler_to = BusbarCoupler(busbar_from_id=1, busbar_to_id=5, open=False, grid_model_id="coupler3")
    with pytest.raises(ValueError, match="Busbars for coupler coupler3 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_to)

    # Test case: Coupler with both busbar_from_id and busbar_to_id non-existent
    invalid_coupler_both = BusbarCoupler(busbar_from_id=6, busbar_to_id=7, open=False, grid_model_id="coupler4")
    with pytest.raises(ValueError, match="Busbars for coupler coupler4 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_both)


def test_station_diff_no_changes():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )

    realized_station = station_diff(station, station)

    assert realized_station.station == station
    assert realized_station.coupler_diff == []
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_coupler_state_change():
    start_station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )
    target_station = start_station.model_copy(
        update={
            "couplers": [
                BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
            ]
        }
    )

    realized_station = station_diff(start_station, target_station)

    assert realized_station.station == target_station
    assert realized_station.coupler_diff == [target_station.couplers[0]]
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_asset_reassignment():
    start_station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )
    target_station = start_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, True], [True, False]]),
        }
    )

    realized_station = station_diff(start_station, target_station)

    assert realized_station.station == target_station
    assert realized_station.coupler_diff == []
    assert set(realized_station.branch_reassignment_diff) == set([(0, 0, False), (0, 1, True), (1, 0, True), (1, 1, False)])
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_asset_disconnection():
    start_station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )
    target_station = start_station.model_copy(
        update={
            "branch_switching_table": np.array([[False, False], [False, True]]),
        }
    )

    realized_station = station_diff(start_station, target_station)

    assert realized_station.station == target_station
    assert realized_station.coupler_diff == []
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == [0]
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_unsupported_reconnection():
    start_station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[False, False], [False, True]]),
        grid_model_id="station1",
    )
    target_station = start_station.model_copy(
        update={
            "branch_switching_table": np.array([[True, False], [False, True]]),
        }
    )

    with pytest.raises(NotImplementedError, match="Reconnections are not supported yet"):
        station_diff(start_station, target_station)


def test_topology_diff() -> None:
    station1_busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
    ]
    station1_assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
    ]
    station1_couplers = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    start_station_1 = build_materialized_station(
        busbars=station1_busbars,
        couplers=station1_couplers,
        assets=station1_assets,
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )
    # We perform reassignments and open the coupler
    target_station_1 = start_station_1.model_copy(
        update={
            "branch_switching_table": np.array([[False, True], [True, False]]),
            "couplers": [start_station_1.couplers[0].model_copy(update={"open": True})],
        }
    )

    station2_busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
        Busbar(int_id=3, grid_model_id="busbar3"),
    ]
    station2_assets = [
        SwitchableAsset(grid_model_id="station2_line1"),
        SwitchableAsset(grid_model_id="station2_line2"),
    ]
    station2_couplers = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
    ]
    start_station_2 = build_materialized_station(
        busbars=station2_busbars,
        couplers=station2_couplers,
        assets=station2_assets,
        asset_switching_table=np.array([[True, False], [False, True], [False, False]]),
        grid_model_id="station2",
    )
    # We disconnect an asset
    target_station_2 = start_station_2.model_copy(
        update={
            "branch_switching_table": np.array([[False, False], [False, True], [False, False]]),
        }
    )

    start_topology = build_test_topology(
        "topology1",
        raw_stations=[
            build_raw_station(
                "station1",
                station1_busbars,
                station1_couplers,
                ["line1", "line2"],
                np.array([[True, False], [False, True]]),
            ),
            build_raw_station(
                "station2",
                station2_busbars,
                station2_couplers,
                ["station2_line1", "station2_line2"],
                np.array([[True, False], [False, True], [False, False]]),
            ),
        ],
        assets=station1_assets + station2_assets,
    )
    target_topology = build_test_topology(
        "topology1",
        raw_stations=[
            build_raw_station(
                "station1",
                station1_busbars,
                [station1_couplers[0].model_copy(update={"open": True})],
                ["line1", "line2"],
                np.array([[False, True], [True, False]]),
            ),
            build_raw_station(
                "station2",
                station2_busbars,
                station2_couplers,
                ["station2_line1", "station2_line2"],
                np.array([[False, False], [False, True], [False, False]]),
            ),
        ],
        assets=station1_assets + station2_assets,
    )

    realized_topo = topology_diff(start_topology, target_topology)

    assert realized_topo.topology == target_topology
    assert realized_topo.coupler_diff == [("station1", target_station_1.couplers[0])]
    assert set(realized_topo.branch_reassignment_diff) == set(
        [("station1", 0, 0, False), ("station1", 0, 1, True), ("station1", 1, 0, True), ("station1", 1, 1, False)]
    )
    assert realized_topo.injection_reassignment_diff == []
    assert realized_topo.branch_disconnection_diff == [("station2", 0)]
    assert realized_topo.injection_disconnection_diff == []


def test_filter_out_of_service():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", in_service=True),
            Busbar(int_id=2, grid_model_id="busbar2", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1", in_service=True),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=1, open=False, grid_model_id="coupler2", in_service=False),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1", in_service=True),
            SwitchableAsset(grid_model_id="line2", in_service=False),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )

    filtered_station = filter_out_of_service(station)

    assert len(filtered_station.busbars) == 1
    assert filtered_station.busbars[0].int_id == 1

    assert len(filtered_station.couplers) == 0

    assert len(filtered_station.branch_connections) == 1
    assert filtered_station.branch_connections[0].asset.grid_model_id == "line1"

    assert filtered_station.branch_switching_table.shape == (1, 1)


def test_filter_disconnected_busbars():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_busbars = filter_disconnected_busbars(station)

    assert len(filtered_station.busbars) == 2
    assert {busbar.int_id for busbar in filtered_station.busbars} == {1, 2}

    assert len(removed_busbars) == 1
    assert removed_busbars[0].int_id == 3

    assert filtered_station.branch_switching_table.shape == (2, 1)


def test_filter_disconnected_busbars_open_coupler():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_busbars = filter_disconnected_busbars(station, respect_coupler_open=False)

    assert len(removed_busbars) == 0
    assert filtered_station == station

    filtered_station, removed_busbars = filter_disconnected_busbars(station, respect_coupler_open=True)
    assert len(filtered_station.busbars) == 2
    assert {busbar.int_id for busbar in filtered_station.busbars} == {1, 2}
    assert len(removed_busbars) == 1
    assert removed_busbars[0].int_id == 3
    assert filtered_station.branch_switching_table.shape == (2, 1)


def test_reindex_busbars():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=10, grid_model_id="busbar1"),
            Busbar(int_id=20, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=10, busbar_to_id=20, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    reindexed_station = reindex_busbars(station)

    assert len(reindexed_station.busbars) == 2
    assert reindexed_station.busbars[0].int_id == 0
    assert reindexed_station.busbars[1].int_id == 1

    assert len(reindexed_station.couplers) == 1
    assert reindexed_station.couplers[0].busbar_from_id == 0
    assert reindexed_station.couplers[0].busbar_to_id == 1


def test_has_transmission_line_switching():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[],
        assets=[
            SwitchableAsset(grid_model_id="line1", in_service=True),
            SwitchableAsset(grid_model_id="line2", in_service=False),
        ],
        asset_switching_table=np.array([[False, False], [False, False]]),
        grid_model_id="station1",
    )

    assert has_transmission_line_switching(station) is True

    station.branch_switching_table = np.array([[True, False], [False, True]])
    assert has_transmission_line_switching(station) is False


def test_order_station_assets() -> None:
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            BusbarCoupler(busbar_from_id=3, busbar_to_id=1, open=False, grid_model_id="coupler3"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
        ],
        asset_switching_table=np.array(
            [
                [True, True, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ]
        ),
        grid_model_id="station1",
    )

    desired_order = ["line5", "line4", "line3", "line1"]
    ordered, not_found, ignored = order_station_assets(station, desired_order)

    assert not_found == []
    assert ignored == ["line2"]
    assert len(ordered.branch_connections) == len(desired_order)
    assert [asset_connection.asset.grid_model_id for asset_connection in ordered.branch_connections] == desired_order
    assert np.array_equal(
        ordered.branch_switching_table,
        np.array(
            [
                [False, False, True, True],
                [False, True, False, False],
                [True, False, False, False],
            ]
        ),
    )

    desired_order = ["line5", "line4", "pink_unicorn", "line3", "line1"]
    ordered, not_found, ignored = order_station_assets(station, desired_order)

    assert not_found == ["pink_unicorn"]
    assert ignored == ["line2"]
    assert len(ordered.branch_connections) == len(desired_order) - 1
    assert [asset_connection.asset.grid_model_id for asset_connection in ordered.branch_connections] == [
        "line5",
        "line4",
        "line3",
        "line1",
    ]


def test_order_topology() -> None:
    busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
    ]
    couplers = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    switching_table = np.array(
        [
            [True, False, True, False, False],
            [False, False, False, True, True],
        ]
    )
    raw_stations = [
        build_raw_station("station1", busbars, couplers, ["line1", "line2", "line3", "line4", "line5"], switching_table),
        build_raw_station(
            "station2",
            busbars,
            couplers,
            ["station2_line1", "station2_line2", "station2_line3", "station2_line4", "station2_line5"],
            switching_table,
        ),
        build_raw_station(
            "station3",
            busbars,
            couplers,
            ["station3_line1", "station3_line2", "station3_line3", "station3_line4", "station3_line5"],
            switching_table,
        ),
        build_raw_station(
            "station4",
            busbars,
            couplers,
            ["station4_line1", "station4_line2", "station4_line3", "station4_line4", "station4_line5"],
            switching_table,
        ),
    ]
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
        SwitchableAsset(grid_model_id="line4"),
        SwitchableAsset(grid_model_id="line5"),
        SwitchableAsset(grid_model_id="station2_line1"),
        SwitchableAsset(grid_model_id="station2_line2"),
        SwitchableAsset(grid_model_id="station2_line3"),
        SwitchableAsset(grid_model_id="station2_line4"),
        SwitchableAsset(grid_model_id="station2_line5"),
        SwitchableAsset(grid_model_id="station3_line1"),
        SwitchableAsset(grid_model_id="station3_line2"),
        SwitchableAsset(grid_model_id="station3_line3"),
        SwitchableAsset(grid_model_id="station3_line4"),
        SwitchableAsset(grid_model_id="station3_line5"),
        SwitchableAsset(grid_model_id="station4_line1"),
        SwitchableAsset(grid_model_id="station4_line2"),
        SwitchableAsset(grid_model_id="station4_line3"),
        SwitchableAsset(grid_model_id="station4_line4"),
        SwitchableAsset(grid_model_id="station4_line5"),
    ]
    topology = build_test_topology(
        "topo-popo",
        raw_stations=raw_stations,
        assets=assets,
    )

    ordered, not_found = order_topology(topology, ["station4", "station2", "station1", "station3"])
    station_ids = [station.grid_model_id for station in ordered.raw_stations]
    assert station_ids == ["station4", "station2", "station1", "station3"]
    assert not_found == []

    ordered, not_found = order_topology(topology, ["station4", "station2", "station1", "station3", "station5"])
    station_ids = [station.grid_model_id for station in ordered.raw_stations]
    assert station_ids == ["station4", "station2", "station1", "station3"]
    assert not_found == ["station5"]

    ordered, not_found = order_topology(topology, ["station4", "station2", "station1"])
    station_ids = [station.grid_model_id for station in ordered.raw_stations]
    assert station_ids == ["station4", "station2", "station1"]
    assert not_found == []


def test_fuse_coupler():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
            [[True, False, False, False], [False, True, True, False], [False, False, False, True]]
        ),
        asset_connectivity=np.array([[True, False, False, False], [False, True, True, False], [False, False, False, True]]),
        grid_model_id="station1",
    )

    # Test fusing the coupler with copy_info_from=True
    fused_station = fuse_coupler(station, "coupler1", copy_info_from=True)

    assert len(fused_station.busbars) == 2
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar3"
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "coupler2"
    expected_switching_table = np.array([[True, True, True, False], [False, False, False, True]])
    assert np.array_equal(fused_station.branch_switching_table, expected_switching_table)
    assert np.array_equal(fused_station.branch_connectivity, expected_switching_table)

    # Test fusing the coupler with copy_info_from=False
    fused_station = fuse_coupler(station, "coupler1", copy_info_from=False)

    assert len(fused_station.busbars) == 2
    assert fused_station.busbars[0].grid_model_id == "busbar2"
    assert fused_station.busbars[1].grid_model_id == "busbar3"
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "coupler2"
    assert np.array_equal(fused_station.branch_switching_table, expected_switching_table)
    assert np.array_equal(fused_station.branch_connectivity, expected_switching_table)

    # Test fusing the other coupler

    fused_station = fuse_coupler(station, "coupler2", copy_info_from=True)
    assert len(fused_station.busbars) == 2
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar2"
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "coupler1"
    expected_switching_table = np.array([[True, False, False, False], [False, True, True, True]])
    assert np.array_equal(fused_station.branch_switching_table, expected_switching_table)
    assert np.array_equal(fused_station.branch_connectivity, expected_switching_table)

    fused_station = fuse_coupler(station, "coupler2", copy_info_from=False)
    assert len(fused_station.busbars) == 2
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar3"
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "coupler1"
    expected_switching_table = np.array([[True, False, False, False], [False, True, True, True]])
    assert np.array_equal(fused_station.branch_switching_table, expected_switching_table)
    assert np.array_equal(fused_station.branch_connectivity, expected_switching_table)

    # Test fusing a non-existent coupler
    with pytest.raises(ValueError, match="Coupler invalid_coupler not found in station station1"):
        fuse_coupler(station, "invalid_coupler")


def test_fuse_all_couplers_with_type():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
            Busbar(int_id=4, grid_model_id="busbar4"),
            Busbar(int_id=5, grid_model_id="busbar5"),
        ],
        couplers=[
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="BREAKER",
            ),
            BusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=3,
                open=False,
                grid_model_id="coupler2",
                coupler_type="BREAKER",
            ),
            BusbarCoupler(
                busbar_from_id=3,
                busbar_to_id=4,
                open=False,
                grid_model_id="coupler3",
                coupler_type="DISCONNECTOR",
            ),
            BusbarCoupler(
                busbar_from_id=4,
                busbar_to_id=5,
                open=False,
                grid_model_id="coupler4",
                coupler_type="BREAKER",
            ),
            BusbarCoupler(
                busbar_from_id=5,
                busbar_to_id=4,
                open=False,
                grid_model_id="coupler5",
                coupler_type="DISCONNECTOR",
            ),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
                [False, False, False],
            ]
        ),
        asset_connectivity=np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
                [False, False, False],
            ]
        ),
        grid_model_id="station1",
    )

    # Test fusing all couplers of type BREAKER
    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "BREAKER", copy_info_from=True)

    assert len(fused_station.busbars) == 2
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar4"
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "coupler3"
    assert len(fused_couplers) == 4
    assert {coupler.grid_model_id for coupler in fused_couplers} == {"coupler1", "coupler2", "coupler4", "coupler5"}
    expected_switching_table = np.array(
        [
            [True, True, True],
            [False, False, False],
        ]
    )
    assert np.array_equal(fused_station.branch_switching_table, expected_switching_table)
    assert np.array_equal(fused_station.branch_connectivity, expected_switching_table)

    # Test fusing all couplers of type DISCONNECTOR
    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "DISCONNECTOR", copy_info_from=True)

    assert len(fused_station.busbars) == 3
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar2"
    assert fused_station.busbars[2].grid_model_id in {"busbar3", "busbar4", "busbar5"}
    assert len(fused_station.couplers) == 2
    assert {coupler.grid_model_id for coupler in fused_station.couplers} == {"coupler1", "coupler2"}
    assert {coupler.grid_model_id for coupler in fused_couplers} == {"coupler3", "coupler4", "coupler5"}

    # Test fusing couplers of a non-existent type
    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "NONEXISTENTTYPE", copy_info_from=True)

    assert fused_station == station
    assert len(fused_couplers) == 0


def test_filter_duplicate_couplers():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=1, open=False, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_couplers = filter_duplicate_couplers(station)

    assert len(filtered_station.couplers) == 1
    assert filtered_station.couplers[0].grid_model_id == "coupler1"

    assert len(removed_couplers) == 1
    assert removed_couplers[0].grid_model_id == "coupler2"


def test_filter_duplicate_couplers_with_type_hierarchy():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="DISCONNECTOR",
            ),
            BusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler2",
                coupler_type="BREAKER",
            ),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    # Test with type hierarchy where BREAKER is preferred
    filtered_station, removed_couplers = filter_duplicate_couplers(
        station, retain_type_hierarchy=["BREAKER", "DISCONNECTOR"]
    )

    assert len(filtered_station.couplers) == 1
    assert filtered_station.couplers[0].grid_model_id == "coupler2"
    assert filtered_station.couplers[0].coupler_type == "BREAKER"

    assert len(removed_couplers) == 1
    assert removed_couplers[0].grid_model_id == "coupler1"
    assert removed_couplers[0].coupler_type == "DISCONNECTOR"

    # Test with reversed hierarchy where DISCONNECTOR is preferred
    filtered_station, removed_couplers = filter_duplicate_couplers(
        station, retain_type_hierarchy=["DISCONNECTOR", "BREAKER"]
    )

    assert len(filtered_station.couplers) == 1
    assert filtered_station.couplers[0].grid_model_id == "coupler1"
    assert filtered_station.couplers[0].coupler_type == "DISCONNECTOR"

    assert len(removed_couplers) == 1
    assert removed_couplers[0].grid_model_id == "coupler2"
    assert removed_couplers[0].coupler_type == "BREAKER"


def test_filter_duplicate_couplers_no_duplicates():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_couplers = filter_duplicate_couplers(station)

    assert filtered_station == station
    assert len(removed_couplers) == 0


def test_filter_duplicate_couplers_with_unknown_type():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="KNOWN",
            ),
            BusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler2",
                coupler_type="UNKNOWN",
            ),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    # Test with type hierarchy that doesn't include UNKNOWN
    filtered_station, removed_couplers = filter_duplicate_couplers(station, retain_type_hierarchy=["KNOWN"])

    assert len(filtered_station.couplers) == 1
    assert filtered_station.couplers[0].grid_model_id == "coupler1"
    assert filtered_station.couplers[0].coupler_type == "KNOWN"

    assert len(removed_couplers) == 1
    assert removed_couplers[0].grid_model_id == "coupler2"
    assert removed_couplers[0].coupler_type == "UNKNOWN"


def test_filter_duplicate_couplers_multiple_duplicates():
    station = build_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="TYPE_A",
            ),
            BusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler2",
                coupler_type="TYPE_B",
            ),
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=True,
                grid_model_id="coupler3",
                coupler_type="TYPE_C",
            ),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
        ],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    # With hierarchy
    filtered_station, removed_couplers = filter_duplicate_couplers(
        station, retain_type_hierarchy=["TYPE_B", "TYPE_C", "TYPE_A"]
    )

    assert len(filtered_station.couplers) == 1
    assert filtered_station.couplers[0].grid_model_id == "coupler2"
    assert filtered_station.couplers[0].coupler_type == "TYPE_B"

    assert len(removed_couplers) == 2
    assert set(c.grid_model_id for c in removed_couplers) == {"coupler1", "coupler3"}
