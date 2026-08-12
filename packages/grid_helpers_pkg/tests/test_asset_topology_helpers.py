# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import numpy as np
import pytest
from toop_engine_grid_helpers.asset_topology_helpers import (
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
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    Busbar,
    InjectionAsset,
    SwitchableAsset,
)
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeBusGroup,
)


def build_runtime_bus_group(
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[RuntimeBusbarCoupler],
    assets: list[SwitchableAsset],
    asset_switching_table: np.ndarray,
    asset_connectivity: np.ndarray | None = None,
    injection_assets: list[SwitchableAsset] | None = None,
    injection_switching_table: np.ndarray | None = None,
    injection_connectivity: np.ndarray | None = None,
) -> RuntimeBusGroup:
    """Build a materialized station using explicit asset connections."""
    resolved_injection_assets = injection_assets if injection_assets is not None else []
    resolved_injection_switching_table = (
        injection_switching_table
        if injection_switching_table is not None
        else np.zeros((len(busbars), len(resolved_injection_assets)), dtype=bool)
    )
    resolved_injection_connectivity = injection_connectivity if injection_connectivity is not None else None
    branch_assets = [
        asset if isinstance(asset, RuntimeBranchAsset) else RuntimeBranchAsset.model_validate(asset.model_dump())
        for asset in assets
    ]
    normalized_injection_assets = [
        asset if isinstance(asset, RuntimeInjectionAsset) else RuntimeInjectionAsset.model_validate(asset.model_dump())
        for asset in resolved_injection_assets
    ]
    return RuntimeBusGroup(
        bus_group_id=grid_model_id,
        name=None,
        station_type=None,
        region=None,
        voltage_level=None,
        busbars=busbars,
        couplers=couplers,
        branch_connections=[RuntimeAssetConnection(asset=asset) for asset in branch_assets],
        injection_connections=[RuntimeAssetConnection(asset=asset) for asset in normalized_injection_assets],
        branch_switching_table=asset_switching_table,
        injection_switching_table=resolved_injection_switching_table,
        branch_connectivity=asset_connectivity,
        injection_connectivity=resolved_injection_connectivity,
        model_log=None,
    )


def build_test_topology(
    topology_id: str, stations: list[RuntimeBusGroup], assets: list[SwitchableAsset]
) -> tuple[MasterAssetTopology, list[RuntimeBusGroup]]:
    """Build topology master data with matching runtime stations.

    Parameters
    ----------
    topology_id : str
        Identifier of the topology.
    stations : list[RuntimeBusGroup]
        Runtime stations that define the topology layout.
    assets : list[SwitchableAsset]
        Topology-owned assets referenced by the raw stations.

    Returns
    -------
    tuple[MasterAssetTopology, list[RuntimeBusGroup]]
        Canonical master data and matching runtime stations for helper-focused tests.
    """
    del assets

    master_stations: list[MasterBusGroup] = []
    branch_assets_by_id: dict[str, BranchAsset] = {}
    injection_assets_by_id: dict[str, InjectionAsset] = {}
    asset_bays_by_id: dict[str, AssetBay] = {}
    for station in stations:
        station_branch_connections = []
        for asset_connection in station.branch_connections:
            branch_asset = asset_connection.asset.model_copy(update={"in_service": True}, deep=True)
            assert isinstance(branch_asset, BranchAsset)
            branch_assets_by_id[branch_asset.grid_model_id] = branch_asset
            asset_bay_id = asset_connection.asset_bay.asset_bay_id if asset_connection.asset_bay is not None else None
            if asset_connection.asset_bay is not None and asset_bay_id is not None:
                asset_bays_by_id[asset_bay_id] = asset_connection.asset_bay.model_copy(deep=True)
            station_branch_connections.append(
                BusGroupAssetConnection(
                    asset_id=branch_asset.grid_model_id,
                    branch_end=asset_connection.branch_end,
                    asset_bay_id=asset_bay_id,
                )
            )

        station_injection_connections = []
        for asset_connection in station.injection_connections:
            injection_asset = asset_connection.asset.model_copy(update={"in_service": True}, deep=True)
            assert isinstance(injection_asset, InjectionAsset)
            injection_assets_by_id[injection_asset.grid_model_id] = injection_asset
            asset_bay_id = asset_connection.asset_bay.asset_bay_id if asset_connection.asset_bay is not None else None
            if asset_connection.asset_bay is not None and asset_bay_id is not None:
                asset_bays_by_id[asset_bay_id] = asset_connection.asset_bay.model_copy(deep=True)
            station_injection_connections.append(
                BusGroupAssetConnection(
                    asset_id=injection_asset.grid_model_id,
                    branch_end=asset_connection.branch_end,
                    asset_bay_id=asset_bay_id,
                )
            )

        is_bus_branch_model = all(
            asset_connection.asset_bay_id is None
            for asset_connection in [*station_branch_connections, *station_injection_connections]
        )
        branch_switching_table = np.asarray(station.branch_switching_table, dtype=bool)
        if is_bus_branch_model:
            branch_connectivity = np.ones_like(branch_switching_table, dtype=bool)
        else:
            branch_connectivity = np.array(
                station.branch_connectivity if station.branch_connectivity is not None else branch_switching_table,
                dtype=bool,
                copy=True,
            )
            for asset_index, asset_connection in enumerate(station_branch_connections):
                if asset_connection.asset_bay_id is None and branch_switching_table[:, asset_index].sum() == 1:
                    branch_connectivity[:, asset_index] = branch_switching_table[:, asset_index]

        injection_switching_table = np.asarray(station.injection_switching_table, dtype=bool)
        if is_bus_branch_model:
            injection_connectivity = np.ones_like(injection_switching_table, dtype=bool)
        else:
            injection_connectivity = np.array(
                station.injection_connectivity if station.injection_connectivity is not None else injection_switching_table,
                dtype=bool,
                copy=True,
            )
            for asset_index, asset_connection in enumerate(station_injection_connections):
                if asset_connection.asset_bay_id is None and injection_switching_table[:, asset_index].sum() == 1:
                    injection_connectivity[:, asset_index] = injection_switching_table[:, asset_index]

        master_stations.append(
            MasterBusGroup(
                bus_group_id=station.bus_group_id,
                name=station.name,
                station_type=station.station_type,
                region=station.region,
                voltage_level=station.voltage_level,
                busbars=[busbar.model_copy(update={"in_service": True}, deep=True) for busbar in station.busbars],
                couplers=[
                    coupler.model_copy(update={"open": False, "in_service": True}, deep=True) for coupler in station.couplers
                ],
                branch_connections=station_branch_connections,
                injection_connections=station_injection_connections,
                branch_connectivity=branch_connectivity,
                injection_connectivity=injection_connectivity,
            )
        )

    return (
        MasterAssetTopology(
            topology_id=topology_id,
            bus_groups=master_stations,
            branch_assets=list(branch_assets_by_id.values()),
            injection_assets=list(injection_assets_by_id.values()),
            asset_bays=list(asset_bays_by_id.values()),
        ),
        stations,
    )


def test_merge_stations():
    station1 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
                    RuntimeBusbarCoupler(
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
    assert updated_stations[0].bus_group_id == "station1"
    assert updated_stations[0].couplers[0].open
    assert coupler_diff == [("station1", new_stations[0].couplers[0])]

    assert updated_stations[1].bus_group_id == "station2"
    assert np.array_equal(
        updated_stations[1].branch_switching_table,
        np.array([[False, True, True], [True, False, False]]),
    )
    assert set(reassignment_diff) == set([("station2", 1, 0, True), ("station2", 1, 1, False)])


def test_merge_stations_append_behavior():
    station1 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
                    RuntimeBusbarCoupler(
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
    assert updated_stations[0].bus_group_id == "station1"
    assert updated_stations[0].couplers[0].open
    assert coupler_diff == [("station1", new_stations[0].couplers[0])]

    assert updated_stations[1].bus_group_id == "station2"
    assert np.array_equal(
        updated_stations[1].branch_switching_table,
        np.array([[False, True, True], [True, False, False]]),
    )
    assert reassignment_diff == []


def test_merge_stations_raise_behavior():
    station1 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
    station1 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station1",
    )
    station2 = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
    assert updated_stations[0].bus_group_id == "station1"
    assert updated_stations[1].bus_group_id == "station2"
    assert coupler_diff == []
    assert reassignment_diff == []


def test_get_connected_assets():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            RuntimeBranchAsset(grid_model_id="line1", in_service=True),
            RuntimeBranchAsset(grid_model_id="line2", in_service=False),
            RuntimeBranchAsset(grid_model_id="line3", in_service=True),
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


def test_find_busbars_for_coupler():
    busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
        Busbar(int_id=3, grid_model_id="busbar3"),
    ]

    # Test case: Valid coupler with matching busbars
    coupler = RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1")
    busbar_from, busbar_to = find_busbars_for_coupler(busbars, coupler)
    assert busbar_from.int_id == 1
    assert busbar_to.int_id == 2

    # Test case: Coupler with non-existent busbar_from_id
    invalid_coupler_from = RuntimeBusbarCoupler(busbar_from_id=4, busbar_to_id=2, open=False, grid_model_id="coupler2")
    with pytest.raises(ValueError, match="Busbars for coupler coupler2 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_from)

    # Test case: Coupler with non-existent busbar_to_id
    invalid_coupler_to = RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=5, open=False, grid_model_id="coupler3")
    with pytest.raises(ValueError, match="Busbars for coupler coupler3 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_to)

    # Test case: Coupler with both busbar_from_id and busbar_to_id non-existent
    invalid_coupler_both = RuntimeBusbarCoupler(busbar_from_id=6, busbar_to_id=7, open=False, grid_model_id="coupler4")
    with pytest.raises(ValueError, match="Busbars for coupler coupler4 not found"):
        find_busbars_for_coupler(busbars, invalid_coupler_both)


def test_station_diff_no_changes():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )

    realized_station = station_diff(station, station)

    assert realized_station.bus_group == station
    assert realized_station.coupler_diff == []
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_coupler_state_change():
    start_station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
                RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
            ]
        }
    )

    realized_station = station_diff(start_station, target_station)

    assert realized_station.bus_group == target_station
    assert realized_station.coupler_diff == [target_station.couplers[0]]
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_asset_reassignment():
    start_station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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

    assert realized_station.bus_group == target_station
    assert realized_station.coupler_diff == []
    assert set(realized_station.branch_reassignment_diff) == set([(0, 0, False), (0, 1, True), (1, 0, True), (1, 1, False)])
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == []
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_asset_disconnection():
    start_station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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

    assert realized_station.bus_group == target_station
    assert realized_station.coupler_diff == []
    assert realized_station.branch_reassignment_diff == []
    assert realized_station.injection_reassignment_diff == []
    assert realized_station.branch_disconnection_diff == [0]
    assert realized_station.injection_disconnection_diff == []


def test_station_diff_unsupported_reconnection():
    start_station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
        RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    start_station_1 = build_runtime_bus_group(
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
        RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
    ]
    start_station_2 = build_runtime_bus_group(
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

    start_master_data, start_stations = build_test_topology(
        "topology1",
        stations=[start_station_1, start_station_2],
        assets=station1_assets + station2_assets,
    )
    target_master_data, target_stations = build_test_topology(
        "topology1",
        stations=[target_station_1, target_station_2],
        assets=station1_assets + station2_assets,
    )

    realized_topo = topology_diff(start_stations, target_stations, master_data=target_master_data)

    assert realized_topo.master_data == target_master_data
    assert realized_topo.bus_groups == target_stations
    assert realized_topo.coupler_diff == [("station1", target_station_1.couplers[0])]
    assert set(realized_topo.branch_reassignment_diff) == set(
        [("station1", 0, 0, False), ("station1", 0, 1, True), ("station1", 1, 0, True), ("station1", 1, 1, False)]
    )
    assert realized_topo.injection_reassignment_diff == []
    assert realized_topo.branch_disconnection_diff == [("station2", 0)]
    assert realized_topo.injection_disconnection_diff == []


def test_filter_out_of_service():
    station = build_runtime_bus_group(
        busbars=[
            RuntimeBusbar(int_id=1, grid_model_id="busbar1", in_service=True),
            RuntimeBusbar(int_id=2, grid_model_id="busbar2", in_service=False),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1", in_service=True),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=1, open=False, grid_model_id="coupler2", in_service=False),
        ],
        assets=[
            RuntimeBranchAsset(grid_model_id="line1", in_service=True),
            RuntimeBranchAsset(grid_model_id="line2", in_service=False),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=10, grid_model_id="busbar1"),
            Busbar(int_id=20, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=10, busbar_to_id=20, open=False, grid_model_id="coupler1"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[],
        assets=[
            RuntimeBranchAsset(grid_model_id="line1", in_service=True),
            RuntimeBranchAsset(grid_model_id="line2", in_service=False),
        ],
        asset_switching_table=np.array([[False, False], [False, False]]),
        grid_model_id="station1",
    )

    assert has_transmission_line_switching(station) is True

    station.branch_switching_table = np.array([[True, False], [False, True]])
    assert has_transmission_line_switching(station) is False


def test_order_station_assets() -> None:
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            RuntimeBusbarCoupler(busbar_from_id=3, busbar_to_id=1, open=False, grid_model_id="coupler3"),
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
        RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    switching_table = np.array(
        [
            [True, False, True, False, False],
            [False, False, False, True, True],
        ]
    )
    stations = [
        build_runtime_bus_group(
            "station1",
            busbars,
            couplers,
            [
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
                SwitchableAsset(grid_model_id="line4"),
                SwitchableAsset(grid_model_id="line5"),
            ],
            switching_table,
        ),
        build_runtime_bus_group(
            "station2",
            busbars,
            couplers,
            [
                SwitchableAsset(grid_model_id="station2_line1"),
                SwitchableAsset(grid_model_id="station2_line2"),
                SwitchableAsset(grid_model_id="station2_line3"),
                SwitchableAsset(grid_model_id="station2_line4"),
                SwitchableAsset(grid_model_id="station2_line5"),
            ],
            switching_table,
        ),
        build_runtime_bus_group(
            "station3",
            busbars,
            couplers,
            [
                SwitchableAsset(grid_model_id="station3_line1"),
                SwitchableAsset(grid_model_id="station3_line2"),
                SwitchableAsset(grid_model_id="station3_line3"),
                SwitchableAsset(grid_model_id="station3_line4"),
                SwitchableAsset(grid_model_id="station3_line5"),
            ],
            switching_table,
        ),
        build_runtime_bus_group(
            "station4",
            busbars,
            couplers,
            [
                SwitchableAsset(grid_model_id="station4_line1"),
                SwitchableAsset(grid_model_id="station4_line2"),
                SwitchableAsset(grid_model_id="station4_line3"),
                SwitchableAsset(grid_model_id="station4_line4"),
                SwitchableAsset(grid_model_id="station4_line5"),
            ],
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
    _, topology_stations = build_test_topology(
        "topo-popo",
        stations=stations,
        assets=assets,
    )

    ordered, not_found = order_topology(topology_stations, ["station4", "station2", "station1", "station3"])
    station_ids = [station.bus_group_id for station in ordered]
    assert station_ids == ["station4", "station2", "station1", "station3"]
    assert not_found == []

    ordered, not_found = order_topology(topology_stations, ["station4", "station2", "station1", "station3", "station5"])
    station_ids = [station.bus_group_id for station in ordered]
    assert station_ids == ["station4", "station2", "station1", "station3"]
    assert not_found == ["station5"]

    ordered, not_found = order_topology(topology_stations, ["station4", "station2", "station1"])
    station_ids = [station.bus_group_id for station in ordered]
    assert station_ids == ["station4", "station2", "station1"]
    assert not_found == []


def test_order_topology_with_runtime_bus_groups() -> None:
    """Verify topology ordering when the input consists of runtime materialized stations."""
    busbars = [Busbar(int_id=1, grid_model_id="busbar1")]
    stations = [
        build_runtime_bus_group(
            "station1",
            busbars,
            [],
            [SwitchableAsset(grid_model_id="line1")],
            np.array([[True]]),
        ),
        build_runtime_bus_group(
            "station2",
            busbars,
            [],
            [SwitchableAsset(grid_model_id="line2")],
            np.array([[True]]),
        ),
        build_runtime_bus_group(
            "station3",
            busbars,
            [],
            [SwitchableAsset(grid_model_id="line3")],
            np.array([[True]]),
        ),
    ]
    topology_stations = stations

    ordered, not_found = order_topology(topology_stations, ["station3", "station1", "station4"])

    assert [station.bus_group_id for station in ordered] == ["station3", "station1"]
    assert not_found == ["station4"]


def test_fuse_coupler():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
            Busbar(int_id=4, grid_model_id="busbar4"),
            Busbar(int_id=5, grid_model_id="busbar5"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="BREAKER",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=3,
                open=False,
                grid_model_id="coupler2",
                coupler_type="BREAKER",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=3,
                busbar_to_id=4,
                open=False,
                grid_model_id="coupler3",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=4,
                busbar_to_id=5,
                open=False,
                grid_model_id="coupler4",
                coupler_type="BREAKER",
            ),
            RuntimeBusbarCoupler(
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

    assert len(fused_station.busbars) == 4
    assert fused_station.busbars[0].grid_model_id == "busbar1"
    assert fused_station.busbars[1].grid_model_id == "busbar2"
    assert fused_station.busbars[2].grid_model_id == "busbar3"
    assert fused_station.busbars[3].grid_model_id == "busbar5"
    assert len(fused_station.couplers) == 3
    assert {coupler.grid_model_id for coupler in fused_station.couplers} == {"coupler1", "coupler2", "coupler4"}
    assert {coupler.grid_model_id for coupler in fused_couplers} == {"coupler3", "coupler5"}

    # Test fusing couplers of a non-existent type
    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "NONEXISTENTTYPE", copy_info_from=True)

    assert fused_station == station
    assert len(fused_couplers) == 0


def test_fuse_all_couplers_with_type_keeps_breaker_over_parallel_disconnector() -> None:
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="disconnector1",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=3,
                open=False,
                grid_model_id="breaker1",
                coupler_type="BREAKER",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=3,
                open=False,
                grid_model_id="disconnector2",
                coupler_type="DISCONNECTOR",
            ),
        ],
        assets=[SwitchableAsset(grid_model_id="line1")],
        asset_switching_table=np.array([[True], [False], [False]]),
        asset_connectivity=np.array([[True], [False], [False]]),
        grid_model_id="station1",
    )

    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "DISCONNECTOR", copy_info_from=True)

    assert len(fused_station.busbars) == 2
    assert len(fused_station.couplers) == 1
    assert fused_station.couplers[0].grid_model_id == "breaker1"
    assert fused_station.couplers[0].coupler_type == "BREAKER"
    assert {coupler.grid_model_id for coupler in fused_couplers} == {"disconnector1", "disconnector2"}


def test_fuse_all_couplers_with_type_skips_open_disconnectors() -> None:
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=True,
                grid_model_id="disconnector_open",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=3,
                open=False,
                grid_model_id="disconnector_closed",
                coupler_type="DISCONNECTOR",
            ),
        ],
        assets=[SwitchableAsset(grid_model_id="line1")],
        asset_switching_table=np.array([[True], [False], [False]]),
        asset_connectivity=np.array([[True], [False], [False]]),
        grid_model_id="station1",
    )

    fused_station, fused_couplers = fuse_all_couplers_with_type(station, "DISCONNECTOR", copy_info_from=True)

    assert len(fused_station.busbars) == 2
    assert {coupler.grid_model_id for coupler in fused_station.couplers} == {"disconnector_open"}
    assert fused_station.couplers[0].open is True
    assert {coupler.grid_model_id for coupler in fused_couplers} == {"disconnector_closed"}


def test_filter_duplicate_couplers():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=1, open=False, grid_model_id="coupler2"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
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


def test_filter_duplicate_couplers_preserves_closed_connectivity_before_type_hierarchy():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=True,
                grid_model_id="open_disconnector",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="closed_breaker",
                coupler_type="BREAKER",
            ),
        ],
        assets=[SwitchableAsset(grid_model_id="line1")],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_couplers = filter_duplicate_couplers(
        station,
        retain_type_hierarchy=["DISCONNECTOR", "BREAKER"],
    )

    assert [coupler.grid_model_id for coupler in filtered_station.couplers] == ["closed_breaker"]
    assert [coupler.grid_model_id for coupler in removed_couplers] == ["open_disconnector"]


def test_filter_duplicate_couplers_can_preserve_closed_parallel_switches():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="parallel_disconnector",
                coupler_type="DISCONNECTOR",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="parallel_breaker",
                coupler_type="BREAKER",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=1,
                open=True,
                grid_model_id="open_parallel_breaker",
                coupler_type="BREAKER",
            ),
        ],
        assets=[SwitchableAsset(grid_model_id="line1")],
        asset_switching_table=np.array([[True], [False]]),
        grid_model_id="station1",
    )

    filtered_station, removed_couplers = filter_duplicate_couplers(
        station,
        retain_type_hierarchy=["BREAKER", "DISCONNECTOR"],
        preserve_closed_parallel=True,
    )

    assert {coupler.grid_model_id for coupler in filtered_station.couplers} == {
        "parallel_disconnector",
        "parallel_breaker",
    }
    assert [coupler.grid_model_id for coupler in removed_couplers] == ["open_parallel_breaker"]


def test_filter_duplicate_couplers_no_duplicates():
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            RuntimeBusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            RuntimeBusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="KNOWN",
            ),
            RuntimeBusbarCoupler(
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
    station = build_runtime_bus_group(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                grid_model_id="coupler1",
                coupler_type="TYPE_A",
            ),
            RuntimeBusbarCoupler(
                busbar_from_id=2,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler2",
                coupler_type="TYPE_B",
            ),
            RuntimeBusbarCoupler(
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
