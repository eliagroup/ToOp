# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from toop_engine_grid_helpers.asset_topology_helpers import (
    filter_assets_by_type,
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    fix_multi_connected_without_coupler,
    has_transmission_line_switching,
    load_asset_topology_stations,
    save_asset_topology_master_data,
    save_asset_topology_stations,
)
from toop_engine_interfaces.asset_topology.asset_topology import (
    MasterStation,
    RuntimeAssetTopology,
    TopologyMasterData,
    get_asset_bay_ids_for_asset,
    validate_runtime_station_asset_references,
)
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
    SwitchableAsset,
    build_asset_bay_id,
    normalize_switchable_asset_payload,
)
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import StationAssetConnection
from toop_engine_interfaces.asset_topology.topology_conversion import (
    RuntimeSwitchingState,
    materialize_station_from_runtime_state,
)


def materialized_asset_connections(
    assets: list[SwitchableAsset],
    terminals: list[str | None] | None = None,
    asset_bays: list[AssetBay | None] | None = None,
) -> list[MaterializedAssetConnection]:
    if terminals is None:
        terminals = [None] * len(assets)
    if asset_bays is None:
        asset_bays = [None] * len(assets)
    return [
        MaterializedAssetConnection(asset=asset, branch_end=terminal, asset_bay=asset_bay)
        for asset, terminal, asset_bay in zip(assets, terminals, asset_bays, strict=True)
    ]


def make_materialized_station(
    *,
    bus_group_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    branch_assets: list[SwitchableAsset],
    branch_switching_table: np.ndarray,
    branch_connectivity: np.ndarray | None = None,
    branch_terminals: list[str | None] | None = None,
    branch_asset_bays: list[AssetBay | None] | None = None,
    injection_assets: list[SwitchableAsset] | None = None,
    injection_switching_table: np.ndarray | None = None,
    injection_connectivity: np.ndarray | None = None,
    injection_terminals: list[str | None] | None = None,
    injection_asset_bays: list[AssetBay | None] | None = None,
) -> MaterializedStation:
    n_busbars = len(busbars)
    resolved_injection_assets = injection_assets or []
    resolved_injection_switching = injection_switching_table
    if resolved_injection_switching is None:
        resolved_injection_switching = np.zeros((n_busbars, len(resolved_injection_assets)), dtype=bool)

    return MaterializedStation(
        bus_group_id=bus_group_id,
        busbars=busbars,
        couplers=couplers,
        branch_connections=materialized_asset_connections(branch_assets, branch_terminals, branch_asset_bays),
        injection_connections=materialized_asset_connections(
            resolved_injection_assets,
            injection_terminals,
            injection_asset_bays,
        ),
        branch_switching_table=branch_switching_table,
        injection_switching_table=resolved_injection_switching,
        branch_connectivity=branch_connectivity,
        injection_connectivity=injection_connectivity,
    )


def build_reference_master_data(
    *,
    topology_id: str,
    stations: list[MaterializedStation] | None = None,
    branch_assets: list[BranchAsset] | None = None,
    injection_assets: list[InjectionAsset] | None = None,
    asset_bays: list[AssetBay] | None = None,
    grid_model_file: str | None = None,
    name: str | None = None,
) -> TopologyMasterData:
    """Build canonical reference master data for runtime-station-based interface tests."""
    has_runtime_stations = bool(stations)
    if stations:
        master_stations: list[MasterStation] = []
        branch_assets_by_id: dict[str, BranchAsset] = {}
        injection_assets_by_id: dict[str, InjectionAsset] = {}
        asset_bays_by_id: dict[str, AssetBay] = {}
        for station in stations:
            station_branch_connections: list[StationAssetConnection] = []
            station_injection_connections: list[StationAssetConnection] = []

            for asset_connection in station.branch_connections:
                asset = normalize_switchable_asset_payload(asset_connection.asset.model_dump(round_trip=True))
                asset_bay = asset_connection.asset_bay
                asset_bay_id: str | None = None
                if asset_bay is not None and asset_bay.asset_bay_id is not None:
                    asset_bays_by_id[asset_bay.asset_bay_id] = asset_bay.model_copy(deep=True)
                    asset_bay_id = asset_bay.asset_bay_id

                branch_asset = asset if isinstance(asset, BranchAsset) else BranchAsset.model_validate(asset.model_dump())
                branch_assets_by_id[branch_asset.grid_model_id] = branch_asset.model_copy(
                    update={"in_service": True},
                    deep=True,
                )
                station_branch_connections.append(
                    StationAssetConnection(
                        asset_id=branch_asset.grid_model_id,
                        branch_end=asset_connection.branch_end,
                        asset_bay_id=asset_bay_id,
                    )
                )

            for asset_connection in station.injection_connections:
                asset = normalize_switchable_asset_payload(asset_connection.asset.model_dump(round_trip=True))
                asset_bay = asset_connection.asset_bay
                asset_bay_id: str | None = None
                if asset_bay is not None and asset_bay.asset_bay_id is not None:
                    asset_bays_by_id[asset_bay.asset_bay_id] = asset_bay.model_copy(deep=True)
                    asset_bay_id = asset_bay.asset_bay_id

                injection_asset = (
                    asset if isinstance(asset, InjectionAsset) else InjectionAsset.model_validate(asset.model_dump())
                )
                injection_assets_by_id[injection_asset.grid_model_id] = injection_asset.model_copy(
                    update={"in_service": True},
                    deep=True,
                )
                station_injection_connections.append(
                    StationAssetConnection(
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
                    (
                        station.injection_connectivity
                        if station.injection_connectivity is not None
                        else injection_switching_table
                    ),
                    dtype=bool,
                    copy=True,
                )
                for asset_index, asset_connection in enumerate(station_injection_connections):
                    if asset_connection.asset_bay_id is None and injection_switching_table[:, asset_index].sum() == 1:
                        injection_connectivity[:, asset_index] = injection_switching_table[:, asset_index]

            master_stations.append(
                MasterStation(
                    bus_group_id=station.bus_group_id,
                    voltage_level_id=station.voltage_level_id,
                    name=station.name,
                    station_type=station.station_type,
                    region=station.region,
                    voltage_level=station.voltage_level,
                    busbars=[busbar.model_copy(update={"in_service": True}, deep=True) for busbar in station.busbars],
                    couplers=[
                        coupler.model_copy(update={"open": False, "in_service": True}, deep=True)
                        for coupler in station.couplers
                    ],
                    branch_connections=station_branch_connections,
                    injection_connections=station_injection_connections,
                    branch_connectivity=branch_connectivity,
                    injection_connectivity=injection_connectivity,
                )
            )

        master_data = TopologyMasterData(
            topology_id=topology_id,
            grid_model_file=grid_model_file,
            name=name,
            stations=master_stations,
            branch_assets=list(branch_assets_by_id.values()),
            injection_assets=list(injection_assets_by_id.values()),
            asset_bays=list(asset_bays_by_id.values()),
        )
    else:
        master_data = TopologyMasterData(
            topology_id=topology_id,
            grid_model_file=grid_model_file,
            name=name,
            stations=[],
            branch_assets=[],
            injection_assets=[],
            asset_bays=[],
        )

    branch_assets_by_id = {asset.grid_model_id: asset.model_copy(deep=True) for asset in master_data.branch_assets}
    for asset in branch_assets or []:
        if has_runtime_stations and asset.grid_model_id in branch_assets_by_id:
            continue
        branch_assets_by_id[asset.grid_model_id] = asset.model_copy(deep=True)

    injection_assets_by_id = {asset.grid_model_id: asset.model_copy(deep=True) for asset in master_data.injection_assets}
    for asset in injection_assets or []:
        if has_runtime_stations and asset.grid_model_id in injection_assets_by_id:
            continue
        injection_assets_by_id[asset.grid_model_id] = asset.model_copy(deep=True)

    asset_bays_by_id = {asset_bay.asset_bay_id: asset_bay.model_copy(deep=True) for asset_bay in master_data.asset_bays}
    for asset_bay in asset_bays or []:
        if has_runtime_stations and asset_bay.asset_bay_id in asset_bays_by_id:
            continue
        asset_bays_by_id[asset_bay.asset_bay_id] = asset_bay.model_copy(deep=True)

    return master_data.model_copy(
        update={
            "branch_assets": list(branch_assets_by_id.values()),
            "injection_assets": list(injection_assets_by_id.values()),
            "asset_bays": list(asset_bays_by_id.values()),
        }
    )


def realize_reference_topology(
    reference_topology: TopologyMasterData, stations: list[MaterializedStation]
) -> tuple[TopologyMasterData, list[MaterializedStation]]:
    """Combine reference master data with runtime stations and validate their references."""
    effective_master_data = build_reference_master_data(
        topology_id=reference_topology.topology_id,
        stations=stations,
        branch_assets=reference_topology.branch_assets,
        injection_assets=reference_topology.injection_assets,
        asset_bays=reference_topology.asset_bays,
        grid_model_file=reference_topology.grid_model_file,
        name=reference_topology.name,
    )
    validate_runtime_station_asset_references(
        stations,
        effective_master_data.branch_assets,
        effective_master_data.injection_assets,
        effective_master_data.asset_bays,
    )
    return (
        effective_master_data,
        stations,
    )


def test_topology_master_data_normalizes_runtime_state() -> None:
    """Verify that canonical master data strips runtime-only outage and switch state."""
    master_data = build_reference_master_data(
        topology_id="topology-master",
        stations=[
            make_materialized_station(
                bus_group_id="station1",
                busbars=[
                    Busbar(int_id=1, grid_model_id="busbar1", in_service=False),
                    Busbar(int_id=2, grid_model_id="busbar2"),
                ],
                couplers=[
                    BusbarCoupler(
                        grid_model_id="coupler1",
                        busbar_from_id=1,
                        busbar_to_id=2,
                        open=True,
                        in_service=False,
                    )
                ],
                branch_assets=[BranchAsset(grid_model_id="line1", in_service=False)],
                branch_switching_table=np.array([[False], [True]], dtype=bool),
                branch_connectivity=np.array([[True], [True]], dtype=bool),
                injection_assets=[InjectionAsset(grid_model_id="load1", in_service=False)],
                injection_switching_table=np.array([[True], [False]], dtype=bool),
                injection_connectivity=np.array([[True], [False]], dtype=bool),
            ).model_copy(update={"voltage_level_id": "VL1"})
        ],
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=False)],
        injection_assets=[InjectionAsset(grid_model_id="load1", in_service=False)],
    )

    assert [asset.in_service for asset in master_data.branch_assets] == [True]
    assert [asset.in_service for asset in master_data.injection_assets] == [True]
    assert master_data.stations[0].voltage_level_id == "VL1"
    assert [busbar.in_service for busbar in master_data.stations[0].busbars] == [True, True]
    assert [coupler.open for coupler in master_data.stations[0].couplers] == [False]
    assert [coupler.in_service for coupler in master_data.stations[0].couplers] == [True]
    assert np.array_equal(master_data.stations[0].branch_connectivity, np.array([[True], [True]], dtype=bool))
    assert np.array_equal(master_data.stations[0].injection_connectivity, np.array([[True], [True]], dtype=bool))


def test_realize_topology_from_runtime_topology_restores_runtime_state() -> None:
    """Verify that runtime station state remains available after canonical realization."""
    timestamp = datetime(2026, 1, 2)
    metrics = {"fitness": 1.0}
    runtime_stations = [
        make_materialized_station(
            bus_group_id="station1",
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1", in_service=False),
                Busbar(int_id=2, grid_model_id="busbar2", in_service=True),
            ],
            couplers=[
                BusbarCoupler(
                    grid_model_id="coupler1",
                    busbar_from_id=1,
                    busbar_to_id=2,
                    open=True,
                    in_service=False,
                )
            ],
            branch_assets=[BranchAsset(grid_model_id="line1", in_service=False)],
            branch_switching_table=np.array([[False], [True]], dtype=bool),
            branch_connectivity=np.array([[True], [True]], dtype=bool),
            injection_assets=[InjectionAsset(grid_model_id="load1", in_service=False)],
            injection_switching_table=np.array([[True], [False]], dtype=bool),
            injection_connectivity=np.array([[True], [False]], dtype=bool),
        )
    ]
    runtime_station = runtime_stations[0]
    master_data = build_reference_master_data(
        topology_id="topology-runtime",
        stations=runtime_stations,
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=False)],
        injection_assets=[InjectionAsset(grid_model_id="load1", in_service=False)],
    )

    assert [asset_connection.asset.in_service for asset_connection in runtime_stations[0].branch_connections] == [False]
    assert [asset_connection.asset.in_service for asset_connection in runtime_stations[0].injection_connections] == [False]
    assert [busbar.in_service for busbar in runtime_stations[0].busbars] == [False, True]
    assert [coupler.open for coupler in runtime_stations[0].couplers] == [True]
    assert [coupler.in_service for coupler in runtime_stations[0].couplers] == [False]
    assert np.array_equal(
        runtime_stations[0].branch_switching_table,
        runtime_station.branch_switching_table,
    )
    assert np.array_equal(
        runtime_stations[0].injection_switching_table,
        runtime_station.injection_switching_table,
    )
    assert timestamp == datetime(2026, 1, 2)
    assert metrics == {"fitness": 1.0}


def test_materialize_station_from_compact_switch_overlay_uses_master_data_for_ambiguous_no_bay_assignment() -> None:
    """Verify that ambiguous compact overlays fall back to master-data connectivity when possible."""
    station = make_materialized_station(
        bus_group_id="station1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", in_service=True, bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", in_service=True, bus_branch_bus_id="bus_id2"),
        ],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=True, name="line-1")],
        branch_switching_table=np.array([[False], [True]], dtype=bool),
        branch_connectivity=np.array([[True], [True]], dtype=bool),
    )
    master_data = build_reference_master_data(
        topology_id="topology-master",
        stations=[station],
        grid_model_file="grid.xiidm",
    )

    rebuilt_station = materialize_station_from_runtime_state(
        station=master_data.stations[0],
        branch_asset_map={asset.grid_model_id: asset for asset in master_data.branch_assets},
        injection_asset_map={asset.grid_model_id: asset for asset in master_data.injection_assets},
        asset_bay_map={asset_bay.asset_bay_id: asset_bay for asset_bay in master_data.asset_bays},
        runtime_switching_state=RuntimeSwitchingState(
            busbar_bus_branch_bus_ids={"busbar1": "bus_id1", "busbar2": "bus_id2"},
            branch_current_bus_ids=["bus_id2"],
            busbar_out_of_service_ids=set(),
            open_coupler_ids=set(),
            out_of_service_coupler_ids=set(),
            open_switch_ids=set(),
        ),
    )

    assert np.array_equal(rebuilt_station.branch_switching_table, station.branch_switching_table)


def test_topology_master_data_keeps_unique_node_breaker_connectivity() -> None:
    """Verify that node-breaker connectivity remains unique in canonical master data."""
    station = make_materialized_station(
        bus_group_id="station1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", in_service=True, bus_branch_bus_id="node1"),
            Busbar(int_id=2, grid_model_id="busbar2", in_service=True, bus_branch_bus_id="node1"),
        ],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=True, name="line-1")],
        branch_switching_table=np.array([[False], [True]], dtype=bool),
        branch_connectivity=np.array([[False], [True]], dtype=bool),
        branch_asset_bays=[
            AssetBay(
                asset_bay_id="bay-line1",
                dv_switch_grid_model_id="dv-line1",
                sr_switch_grid_model_id={"busbar1": "sr-line1-a", "busbar2": "sr-line1-b"},
            )
        ],
    )

    master_data = build_reference_master_data(
        topology_id="topology-master",
        stations=[station],
        grid_model_file="grid.xiidm",
    )

    assert np.array_equal(master_data.stations[0].branch_connectivity, np.array([[False], [True]], dtype=bool))


def test_materialize_station_from_compact_switch_overlay_raises_for_ambiguous_no_bay_assignment_without_connectivity() -> (
    None
):
    """Verify that ambiguous compact overlays raise when no bay or connectivity data exists."""
    station = make_materialized_station(
        bus_group_id="station1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", in_service=True),
            Busbar(int_id=2, grid_model_id="busbar2", in_service=True),
        ],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=True, name="line-1")],
        branch_switching_table=np.array([[False], [True]], dtype=bool),
        branch_connectivity=np.array([[True], [True]], dtype=bool),
    )
    master_data = build_reference_master_data(
        topology_id="topology-master",
        stations=[station],
        grid_model_file="grid.xiidm",
    )
    master_data = master_data.model_copy(
        update={
            "stations": [
                master_data.stations[0].model_copy(update={"branch_connectivity": None}),
            ]
        }
    )

    with pytest.raises(ValueError, match="Missing asset bay and connectivity for branch asset"):
        materialize_station_from_runtime_state(
            station=master_data.stations[0],
            branch_asset_map={asset.grid_model_id: asset for asset in master_data.branch_assets},
            injection_asset_map={asset.grid_model_id: asset for asset in master_data.injection_assets},
            asset_bay_map={asset_bay.asset_bay_id: asset_bay for asset_bay in master_data.asset_bays},
            runtime_switching_state=RuntimeSwitchingState(
                busbar_out_of_service_ids=set(),
                open_coupler_ids=set(),
                out_of_service_coupler_ids=set(),
                open_switch_ids=set(),
            ),
            model_log=["runtime-log"],
        )


def test_materialize_station_from_compact_switch_overlay_restores_materialized_stations() -> None:
    """Verify that compact runtime overlays rebuild the original materialized station state."""
    station = make_materialized_station(
        bus_group_id="station1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", in_service=True),
            Busbar(int_id=2, grid_model_id="busbar2", in_service=True),
        ],
        couplers=[
            BusbarCoupler(
                grid_model_id="coupler1",
                busbar_from_id=1,
                busbar_to_id=2,
                open=True,
                in_service=True,
            )
        ],
        branch_assets=[BranchAsset(grid_model_id="line1", in_service=False, name="line-1")],
        branch_switching_table=np.array([[False], [True]], dtype=bool),
        branch_connectivity=np.array([[True], [True]], dtype=bool),
        branch_terminals=["from"],
        branch_asset_bays=[
            AssetBay(
                asset_bay_id="bay-line1",
                dv_switch_grid_model_id="dv-line1",
                sr_switch_grid_model_id={"busbar1": "sr-line1-a", "busbar2": "sr-line1-b"},
            )
        ],
        injection_assets=[InjectionAsset(grid_model_id="load1", in_service=False, name="load-1")],
        injection_switching_table=np.array([[True], [False]], dtype=bool),
        injection_connectivity=np.array([[True], [True]], dtype=bool),
        injection_asset_bays=[
            AssetBay(
                asset_bay_id="bay-load1",
                dv_switch_grid_model_id="dv-load1",
                sr_switch_grid_model_id={"busbar1": "sr-load1-a", "busbar2": "sr-load1-b"},
            )
        ],
    ).model_copy(update={"name": "Station One", "station_type": "AIS", "region": "BE", "voltage_level": 380.0})
    station = station.model_copy(
        update={
            "busbars": [
                station.busbars[0].model_copy(update={"bus_branch_bus_id": "bus_id1"}),
                station.busbars[1].model_copy(update={"bus_branch_bus_id": "bus_id2"}),
            ]
        }
    )
    station = station.model_copy(update={"model_log": ["runtime-log"]})
    master_data = build_reference_master_data(
        topology_id="topology-master",
        stations=[station],
        grid_model_file="grid.xiidm",
    )

    rebuilt_station = materialize_station_from_runtime_state(
        station=master_data.stations[0],
        branch_asset_map={asset.grid_model_id: asset for asset in master_data.branch_assets},
        injection_asset_map={asset.grid_model_id: asset for asset in master_data.injection_assets},
        asset_bay_map={asset_bay.asset_bay_id: asset_bay for asset_bay in master_data.asset_bays},
        runtime_switching_state=RuntimeSwitchingState(
            busbar_bus_branch_bus_ids={"busbar1": "bus_id1", "busbar2": "bus_id2"},
            busbar_out_of_service_ids=set(),
            open_coupler_ids={"coupler1"},
            out_of_service_coupler_ids=set(),
            open_switch_ids={"sr-line1-a", "sr-load1-b"},
        ),
        model_log=["runtime-log"],
    )

    expected_station = station.model_copy(deep=True)
    expected_station.branch_connections[0].asset.in_service = True
    expected_station.injection_connections[0].asset.in_service = True

    assert rebuilt_station == expected_station
    assert rebuilt_station.name == station.name
    assert rebuilt_station.station_type == station.station_type
    assert rebuilt_station.voltage_level == station.voltage_level
    assert rebuilt_station.model_log == station.model_log
    assert [busbar.bus_branch_bus_id for busbar in rebuilt_station.busbars] == ["bus_id1", "bus_id2"]


def test_station() -> None:
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=assets,
        branch_switching_table=np.array([[True, False, True], [False, True, False]]),
        bus_group_id="station1",
    )
    assert station is not None

    with pytest.raises(ValidationError):
        # Wrong shape of switching table
        station = make_materialized_station(
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
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]).T,
            bus_group_id="station1",
        )

    with pytest.raises(ValidationError):
        # Coupler references non-existing busbar
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1"),
                Busbar(int_id=2, grid_model_id="busbar2"),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=1,
                    busbar_to_id=3,
                    open=False,
                    grid_model_id="coupler1",
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]),
            bus_group_id="station1",
        )

    with pytest.raises(ValidationError):
        # Coupler references non-existing busbar
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1"),
                Busbar(int_id=2, grid_model_id="busbar2"),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=3,
                    busbar_to_id=1,
                    open=False,
                    grid_model_id="coupler1",
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]),
            bus_group_id="station1",
        )

    with pytest.raises(ValidationError):
        # Busbar int_id is not unique
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1"),
                Busbar(int_id=1, grid_model_id="busbar2"),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=1,
                    busbar_to_id=2,
                    open=False,
                    grid_model_id="coupler1",
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]),
            bus_group_id="station1",
        )

    with pytest.raises(ValidationError):
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1"),
                Busbar(int_id=2, grid_model_id="busbar2"),
                Busbar(int_id=3, grid_model_id="busbar3", in_service=False),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=1,
                    busbar_to_id=2,
                    open=False,
                    grid_model_id="coupler1",
                ),
                BusbarCoupler(
                    busbar_from_id=2,
                    busbar_to_id=3,
                    open=False,
                    grid_model_id="coupler2",
                ),
            ],
            branch_assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2", in_service=False),
                SwitchableAsset(grid_model_id="line3"),
                SwitchableAsset(grid_model_id="line4", in_service=False),
            ],
            branch_switching_table=np.array(
                [
                    [True, False, True, True],
                    [False, True, False, False],
                    [True, False, True, True],
                ]
            ),
            bus_group_id="station1",
        )

    with pytest.raises(ValidationError):
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1"),
                Busbar(int_id=2, grid_model_id="busbar2"),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=1,
                    busbar_to_id=1,
                    open=False,
                    grid_model_id="coupler1",
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]),
            bus_group_id="station1",
        )

    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=assets,
        branch_switching_table=np.array([[True, False, True], [False, True, False]]),
        bus_group_id="VL1_a",
    )
    assert station is not None
    assert not station.is_split()

    split_station = station.model_copy(
        update={
            "busbars": [
                Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
                Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id2"),
            ]
        }
    )
    assert split_station.is_split()

    station_with_empty_bus_id = station.model_copy(
        update={
            "busbars": [
                Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
                Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id=""),
            ]
        }
    )
    assert not station_with_empty_bus_id.is_split()

    with pytest.raises(ValidationError):
        station = make_materialized_station(
            busbars=[
                Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
                Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
            ],
            couplers=[
                BusbarCoupler(
                    busbar_from_id=1,
                    busbar_to_id=1,
                    open=False,
                    grid_model_id="coupler1",
                ),
            ],
            branch_assets=assets,
            branch_switching_table=np.array([[True, False, True], [False, True, False]]),
            bus_group_id="VL1_a",
        )


def test_station_connectivity_tables():
    busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
    ]
    couplers = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    asset_connectivity = np.array([[True, True, True], [True, True, True]])
    grid_model_id = "station1"
    station = make_materialized_station(
        busbars=busbars,
        couplers=couplers,
        branch_assets=assets,
        branch_switching_table=asset_switching_table,
        branch_connectivity=asset_connectivity,
        bus_group_id=grid_model_id,
    )
    assert station is not None

    with pytest.raises(ValidationError):
        # entry in asset_switching_table is not in asset_connectivity
        asset_switching_table = np.array([[True, False, True], [False, True, False]])
        asset_connectivity = np.array([[True, True, True], [True, False, True]])
        station = make_materialized_station(
            busbars=busbars,
            couplers=couplers,
            branch_assets=assets,
            branch_switching_table=asset_switching_table,
            branch_connectivity=asset_connectivity,
            bus_group_id=grid_model_id,
        )

    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    asset_connectivity = np.array([[True, True, True], [True, True, True]])
    station = make_materialized_station(
        busbars=busbars,
        couplers=couplers,
        branch_assets=assets,
        branch_switching_table=asset_switching_table,
        branch_connectivity=asset_connectivity,
        bus_group_id=grid_model_id,
    )
    assert station is not None


def test_topology_station_is_split() -> None:
    station = make_materialized_station(
        bus_group_id="VL1_a",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
        ],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1")],
        branch_switching_table=np.array([[True], [False]]),
    )
    assert not station.is_split()

    split_station = station.model_copy(
        update={
            "busbars": [
                Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
                Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id2"),
            ]
        }
    )
    assert split_station.is_split()

    station_with_empty_bus_id = station.model_copy(
        update={
            "busbars": [
                Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
                Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id=""),
            ]
        }
    )
    assert not station_with_empty_bus_id.is_split()


def test_schema() -> None:
    # Schema generation works
    schema = MaterializedStation.model_json_schema()
    assert schema is not None
    assert "busbars" in schema["properties"]
    assert "couplers" in schema["properties"]
    assert "branch_connections" in schema["properties"]
    assert "branch_switching_table" in schema["properties"]
    assert "injection_connections" in schema["properties"]
    assert "injection_switching_table" in schema["properties"]
    assert "bus_group_id" in schema["properties"]


def test_serialize_station() -> None:
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=assets,
        branch_switching_table=np.array([[True, False, True], [False, True, False]]),
        bus_group_id="station1",
    )

    serialized = station.model_dump_json()
    station2 = MaterializedStation.model_validate_json(serialized)

    assert station == station2


def test_topology_master_data_rejects_duplicate_station_ids() -> None:
    station = MasterStation(
        bus_group_id="station1",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_connections=[],
        injection_connections=[],
        branch_connectivity=np.zeros((1, 0), dtype=bool),
        injection_connectivity=np.zeros((1, 0), dtype=bool),
    )

    with pytest.raises(ValidationError, match="bus_group_id must be unique for topology master data stations"):
        TopologyMasterData(
            topology_id="topology1",
            stations=[station, station.model_copy(update={"name": "duplicate"})],
            branch_assets=[],
            injection_assets=[],
            asset_bays=[],
        )


def test_save_asset_topology_master_data_and_stations() -> None:
    """Verify separate persistence of canonical master data and runtime topology."""
    station1_assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    station1 = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=station1_assets,
        branch_switching_table=np.array([[True, False, True], [False, True, False]]),
        bus_group_id="station1",
    )

    station2_assets = [
        SwitchableAsset(grid_model_id="line4"),
        SwitchableAsset(grid_model_id="line5"),
        SwitchableAsset(grid_model_id="line6"),
    ]
    station2 = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar3"),
            Busbar(int_id=2, grid_model_id="busbar4"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler2"),
        ],
        branch_assets=station2_assets,
        branch_switching_table=np.array([[True, False, True], [False, True, False]]),
        bus_group_id="station2",
    )

    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[
                BranchAsset(grid_model_id="line1"),
                BranchAsset(grid_model_id="line2"),
                BranchAsset(grid_model_id="line3"),
                BranchAsset(grid_model_id="line4"),
                BranchAsset(grid_model_id="line5"),
                BranchAsset(grid_model_id="line6"),
            ],
        ),
        stations=[station1, station2],
    )
    expected_runtime_topology = RuntimeAssetTopology(stations=runtime_stations)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        save_asset_topology_stations(
            tmpdirname / "topology_runtime.json",
            stations=expected_runtime_topology,
        )
        save_asset_topology_master_data(
            tmpdirname / "topology_master_data.json",
            master_data=master_data,
        )
        loaded_runtime_topology = load_asset_topology_stations(tmpdirname / "topology_runtime.json")
        with open(tmpdirname / "topology_master_data.json", "r", encoding="utf-8") as file:
            master_data_payload = json.load(file)

    loaded_master_data_file = TopologyMasterData.model_validate(master_data_payload)

    assert loaded_master_data_file.model_dump(mode="json") == master_data.model_dump(mode="json")
    assert loaded_runtime_topology.model_dump(mode="json") == expected_runtime_topology.model_dump(mode="json")


def test_topology_extracts_assets_and_materializes_stations() -> None:
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="load1"),
    ]
    asset_bays = [
        AssetBay(
            asset_bay_id="station1::line1::bay",
            dv_switch_grid_model_id="dv1",
            sr_switch_grid_model_id={"busbar1": "sr1"},
        ),
        None,
    ]
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_asset_bays=[asset_bays[0]],
        branch_switching_table=np.array([[True], [False]]),
        injection_assets=[InjectionAsset(grid_model_id="load1", asset_type="load")],
        injection_terminals=[None],
        injection_asset_bays=[asset_bays[1]],
        injection_switching_table=np.array([[False], [True]]),
        bus_group_id="station1",
    )

    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            injection_assets=[InjectionAsset(grid_model_id="load1", asset_type="load")],
            asset_bays=[asset_bays[0]],
        ),
        stations=[station],
    )

    assert len(runtime_stations) == 1
    assert runtime_stations[0].model_dump(
        mode="json",
        exclude={"branch_connectivity", "injection_connectivity"},
    ) == station.model_dump(
        mode="json",
        exclude={"branch_connectivity", "injection_connectivity"},
    )
    assert np.array_equal(master_data.stations[0].branch_connectivity, np.array([[True], [False]]))
    assert np.array_equal(master_data.stations[0].injection_connectivity, np.array([[False], [True]]))
    assert [asset.grid_model_id for asset in master_data.branch_assets] == ["line1"]
    assert [asset.grid_model_id for asset in master_data.injection_assets] == ["load1"]
    assert [asset_bay.asset_bay_id for asset_bay in master_data.asset_bays] == ["station1::line1::bay"]

    materialized_station = runtime_stations[0]
    assert materialized_station.model_dump(
        mode="json",
        exclude={"branch_connectivity", "injection_connectivity"},
    ) == station.model_dump(
        mode="json",
        exclude={"branch_connectivity", "injection_connectivity"},
    )
    assert materialized_station.branch_connectivity is None
    assert materialized_station.injection_connectivity is None


def test_topology_from_materialized_stations_keeps_single_canonical_asset_for_two_station_views() -> None:
    asset_from = BranchAsset(grid_model_id="line1", asset_type="line")
    asset_to = BranchAsset(grid_model_id="line1", asset_type="line")

    station_from = make_materialized_station(
        bus_group_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[asset_from],
        branch_terminals=["from"],
        branch_switching_table=np.array([[True]]),
    )
    station_to = make_materialized_station(
        bus_group_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        branch_assets=[asset_to],
        branch_terminals=["to"],
        branch_switching_table=np.array([[True]]),
    )

    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        ),
        stations=[station_from, station_to],
    )

    assert [asset.grid_model_id for asset in master_data.branch_assets] == ["line1"]
    assert [connection.asset.grid_model_id for connection in runtime_stations[0].branch_connections] == ["line1"]
    assert [connection.branch_end for connection in runtime_stations[0].branch_connections] == ["from"]
    assert [connection.asset.grid_model_id for connection in runtime_stations[1].branch_connections] == ["line1"]
    assert [connection.branch_end for connection in runtime_stations[1].branch_connections] == ["to"]
    assert [
        station.model_dump(mode="json", exclude={"branch_connectivity", "injection_connectivity"})
        for station in runtime_stations
    ] == [
        station.model_dump(mode="json", exclude={"branch_connectivity", "injection_connectivity"})
        for station in [station_from, station_to]
    ]
    assert np.array_equal(master_data.stations[0].branch_connectivity, np.array([[True]]))
    assert np.array_equal(master_data.stations[1].branch_connectivity, np.array([[True]]))


def test_topology_from_materialized_stations_normalizes_equivalent_branch_asset_payloads() -> None:
    station_from = make_materialized_station(
        bus_group_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_switching_table=np.array([[True]]),
    )
    station_to = make_materialized_station(
        bus_group_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        branch_assets=[SwitchableAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["to"],
        branch_switching_table=np.array([[True]]),
    )

    master_data, _ = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        ),
        stations=[station_from, station_to],
    )

    assert len(master_data.branch_assets) == 1
    assert isinstance(master_data.branch_assets[0], BranchAsset)
    assert master_data.branch_assets[0].grid_model_id == "line1"


def test_topology_from_materialized_stations_reuses_reference_canonical_assets() -> None:
    reference_topology = build_reference_master_data(
        topology_id="topology1",
        branch_assets=[
            BranchAsset(grid_model_id="line1", asset_type="line"),
            BranchAsset(grid_model_id="line_unused", asset_type="line"),
        ],
        injection_assets=[InjectionAsset(grid_model_id="load1", asset_type="load")],
        asset_bays=[
            AssetBay(
                asset_bay_id="station1::line1::bay",
                dv_switch_grid_model_id="dv1",
                sr_switch_grid_model_id={"busbar1": "sr1"},
            )
        ],
    )
    station = make_materialized_station(
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_asset_bays=[reference_topology.asset_bays[0]],
        branch_switching_table=np.array([[True]]),
        injection_assets=[InjectionAsset(grid_model_id="load1", asset_type="load")],
        injection_switching_table=np.array([[True]]),
        bus_group_id="station1",
    )

    master_data, runtime_stations = realize_reference_topology(reference_topology=reference_topology, stations=[station])

    assert [asset.grid_model_id for asset in master_data.branch_assets] == ["line1", "line_unused"]
    assert [asset.grid_model_id for asset in master_data.injection_assets] == ["load1"]
    assert [asset_bay.asset_bay_id for asset_bay in master_data.asset_bays] == ["station1::line1::bay"]
    assert runtime_stations[0].branch_connections[0].asset.grid_model_id == "line1"


def test_topology_from_materialized_stations_raises_when_reference_assets_are_missing() -> None:
    station = make_materialized_station(
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_switching_table=np.array([[True]]),
        bus_group_id="station1",
    )

    with pytest.raises(ValueError, match="Branch asset grid_model_id line1 referenced by station station1 does not exist"):
        validate_runtime_station_asset_references(stations=[station], branch_assets=[], injection_assets=[], asset_bays=[])


def test_get_asset_bay_ids_for_asset_uses_effective_station_view() -> None:
    """Verify asset-bay lookup against the effective runtime station view."""
    asset_bay = AssetBay(
        asset_bay_id="station1::line1::bay",
        dv_switch_grid_model_id="dv1",
        sr_switch_grid_model_id={"busbar1": "sr1"},
    )
    station = make_materialized_station(
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_asset_bays=[asset_bay],
        branch_switching_table=np.array([[True]]),
        bus_group_id="station1",
    )
    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            injection_assets=[],
            asset_bays=[asset_bay],
        ),
        stations=[station],
    )

    assert get_asset_bay_ids_for_asset(runtime_stations, "line1") == ["station1::line1::bay"]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_station() -> None:
    station_from = make_materialized_station(
        bus_group_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_asset_bays=[
            AssetBay(
                asset_bay_id=build_asset_bay_id("station_from", "line1"),
                dv_switch_grid_model_id="dv_from",
                sr_switch_grid_model_id={"busbar1": "sr_from"},
            )
        ],
        branch_switching_table=np.array([[True]]),
    )
    station_to = make_materialized_station(
        bus_group_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["to"],
        branch_asset_bays=[
            AssetBay(
                asset_bay_id=build_asset_bay_id("station_to", "line1"),
                dv_switch_grid_model_id="dv_to",
                sr_switch_grid_model_id={"busbar2": "sr_to"},
            )
        ],
        branch_switching_table=np.array([[True]]),
    )

    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            asset_bays=[
                station_from.branch_connections[0].asset_bay.model_copy(deep=True),
                station_to.branch_connections[0].asset_bay.model_copy(deep=True),
            ],
        ),
        stations=[station_from, station_to],
    )

    assert sorted(asset_bay.asset_bay_id for asset_bay in master_data.asset_bays) == [
        "station_from::line1::bay",
        "station_to::line1::bay",
    ]
    assert [connection.asset_bay.asset_bay_id for connection in runtime_stations[0].branch_connections] == [
        "station_from::line1::bay"
    ]
    assert [connection.asset_bay.asset_bay_id for connection in runtime_stations[1].branch_connections] == [
        "station_to::line1::bay"
    ]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_occurrence() -> None:
    station = make_materialized_station(
        bus_group_id="station1",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[
            BranchAsset(grid_model_id="line1", asset_type="line"),
            BranchAsset(grid_model_id="line1", asset_type="line"),
        ],
        branch_asset_bays=[
            AssetBay(
                asset_bay_id=build_asset_bay_id("station1", "line1"),
                dv_switch_grid_model_id="dv1",
                sr_switch_grid_model_id={"busbar1": "sr1"},
            ),
            AssetBay(
                asset_bay_id=build_asset_bay_id("station1", "line1", 1),
                dv_switch_grid_model_id="dv2",
                sr_switch_grid_model_id={"busbar1": "sr2"},
            ),
        ],
        branch_switching_table=np.array([[True, True]]),
    )

    master_data, runtime_stations = realize_reference_topology(
        reference_topology=build_reference_master_data(
            topology_id="topology1",
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            asset_bays=[
                station.branch_connections[0].asset_bay.model_copy(deep=True),
                station.branch_connections[1].asset_bay.model_copy(deep=True),
            ],
        ),
        stations=[station],
    )

    assert [connection.asset.grid_model_id for connection in runtime_stations[0].branch_connections] == ["line1", "line1"]
    assert [connection.asset_bay.asset_bay_id for connection in runtime_stations[0].branch_connections] == [
        "station1::line1::bay",
        "station1::line1::bay::1",
    ]
    assert sorted(asset_bay.asset_bay_id for asset_bay in master_data.asset_bays) == [
        "station1::line1::bay",
        "station1::line1::bay::1",
    ]


def test_filter_out_of_service() -> None:
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2", in_service=False),
        SwitchableAsset(grid_model_id="line3"),
        SwitchableAsset(grid_model_id="line4", in_service=False),
    ]
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        branch_assets=assets,
        branch_switching_table=np.array(
            [
                [True, False, True, True],
                [False, True, False, False],
                [True, False, True, True],
            ]
        ),
        bus_group_id="station1",
    )

    station = filter_out_of_service(station)
    assert len(station.busbars) == 2
    assert len(station.couplers) == 1
    assert len(station.branch_connections) == 2
    assert np.array_equal(station.branch_switching_table, np.array([[True, True], [False, False]]))


def test_has_transmission_line_switching() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2", in_service=False),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4", in_service=False),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [True, False, True, False],
            ]
        ),
        bus_group_id="station1",
    )

    assert has_transmission_line_switching(station) is False

    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True, True],
                [False, False, False, False],
                [True, False, True, True],
            ]
        ),
        bus_group_id="station1",
    )

    assert has_transmission_line_switching(station) is True


def test_filter_duplicate_couplers() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=1, open=True, grid_model_id="coupler2"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler3"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler4"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True, True],
                [False, True, False, False],
                [True, False, True, True],
            ]
        ),
        bus_group_id="station1",
    )

    station, removed = filter_duplicate_couplers(station)
    assert len(station.couplers) == 2
    assert station.couplers[0].busbar_from_id == 1
    assert station.couplers[0].busbar_to_id == 2
    assert station.couplers[1].busbar_from_id == 2
    assert station.couplers[1].busbar_to_id == 3

    assert len(removed) == 2
    assert removed[0].busbar_from_id == 2
    assert removed[0].busbar_to_id == 1
    assert removed[1].busbar_from_id == 2
    assert removed[1].busbar_to_id == 3


def test_filter_disconnected_busbars() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True, True],
                [False, True, False, False],
                [True, False, True, True],
            ]
        ),
        bus_group_id="station1",
    )

    station, removed = filter_disconnected_busbars(station)
    assert len(station.busbars) == 2
    assert len(station.couplers) == 1
    assert len(station.branch_connections) == 4
    assert np.array_equal(
        station.branch_switching_table,
        np.array([[True, False, True, True], [False, True, False, False]]),
    )
    assert len(removed) == 1
    assert removed[0].int_id == 3


def test_filter_disconnected_busbars_sort_by_asset_count() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        branch_switching_table=np.array(
            [
                [False, False, False, False],
                [False, True, False, False],
                [True, False, True, True],
            ]
        ),
        bus_group_id="station1",
    )

    station, removed = filter_disconnected_busbars(station)
    assert len(station.busbars) == 1
    assert station.busbars[0].grid_model_id == "busbar3"
    assert len(station.couplers) == 0
    assert len(station.branch_connections) == 4
    assert np.array_equal(
        station.branch_switching_table,
        np.array([[True, False, True, True]]),
    )
    assert len(removed) == 2
    assert removed[0].grid_model_id == "busbar1"
    assert removed[1].grid_model_id == "busbar2"


def test_select_one_for_multi_connected_assets() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
        ],
        branch_assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, False, True],
            ]
        ),
        bus_group_id="station1",
    )

    station, removed = fix_multi_connected_without_coupler(station)
    assert station.branch_switching_table[:, 0].sum() == 1
    assert np.array_equal(
        station.branch_switching_table,
        np.array(
            [
                [False, False, True, False],
                [False, True, False, True],
                [True, False, False, True],
            ]
        ),
    )
    assert len(removed) == 1


def test_filter_assets_by_type() -> None:
    station = make_materialized_station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        branch_assets=[
            BranchAsset(grid_model_id="line1", asset_type="line"),
            BranchAsset(grid_model_id="line2", asset_type="line"),
            SwitchableAsset(grid_model_id="line3", asset_type=None),
        ],
        branch_switching_table=np.array(
            [
                [True, False, True],
                [False, True, False],
            ]
        ),
        injection_assets=[
            InjectionAsset(grid_model_id="gen1", asset_type="gen"),
            InjectionAsset(grid_model_id="load1", asset_type="load"),
        ],
        injection_switching_table=np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        bus_group_id="station1",
    )

    station_filtered, removed = filter_assets_by_type(station, set(["line", "trafo"]))
    assert len(station_filtered.branch_connections) == 2
    assert len(station_filtered.injection_connections) == 0
    assert len(removed) == 3
    assert station_filtered.branch_connections[0].asset.grid_model_id == "line1"
    assert station_filtered.branch_connections[1].asset.grid_model_id == "line2"
    assert station_filtered.branch_switching_table.shape == (2, 2)

    station_filtered, removed = filter_assets_by_type(station, set(["line", "gen"]), allow_none_type=True)
    assert len(station_filtered.branch_connections) == 3
    assert len(station_filtered.injection_connections) == 1
    assert len(removed) == 1
    combined_connections = [*station_filtered.branch_connections, *station_filtered.injection_connections]
    assert combined_connections[0].asset.grid_model_id == "line1"
    assert combined_connections[1].asset.grid_model_id == "line2"
    assert combined_connections[2].asset.grid_model_id == "line3"
    assert combined_connections[3].asset.grid_model_id == "gen1"


def test_asset_bay() -> None:
    # Test valid AssetBay
    path = AssetBay(
        asset_bay_id="station1::line1::bay",
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
    )
    assert path is not None

    path = AssetBay(
        asset_bay_id="station1::line2::bay",
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
        sr_switch_bus_assignment=[1, 2],
    )
    assert path is not None

    # Test AssetBay with missing dv_switch_grid_model_id
    with pytest.raises(ValidationError, match="Field required"):
        path = AssetBay(
            asset_bay_id="station1::line3::bay",
            sl_switch_grid_model_id="sl_switch_1",
            sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
        )

    # Test AssetBay with empty sr_switch_grid_model_id
    with pytest.raises(ValidationError, match="sr_switch_grid_model_id must not be empty"):
        path = AssetBay(
            asset_bay_id="station1::line4::bay",
            sl_switch_grid_model_id="sl_switch_1",
            dv_switch_grid_model_id="dv_switch_1",
            sr_switch_grid_model_id={},
        )

    # Test AssetBay with invalid sr_switch_grid_model_id type
    with pytest.raises(ValidationError):
        path = AssetBay(
            asset_bay_id="station1::line5::bay",
            sl_switch_grid_model_id="sl_switch_1",
            dv_switch_grid_model_id="dv_switch_1",
            sr_switch_grid_model_id={"busbar1": 123},  # Invalid type
        )


def test_station_bay() -> None:
    path = AssetBay(
        asset_bay_id="station1::line2::bay",
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
        sr_switch_bus_assignment=[1, 2],
    )
    busbars = [
        Busbar(int_id=1, grid_model_id="busbar1"),
        Busbar(int_id=2, grid_model_id="busbar2"),
    ]
    couplers = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
    ]
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    grid_model_id = "station1"

    # test valid Station
    station = make_materialized_station(
        busbars=busbars,
        couplers=couplers,
        branch_assets=assets,
        branch_asset_bays=[None, path, None],
        branch_switching_table=asset_switching_table,
        bus_group_id=grid_model_id,
    )
    assert station is not None

    # Test invalid AssetBay -> busbar3 is not in busbars
    path_error = AssetBay(
        asset_bay_id="station1::line2::bay",
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar3": "sr_switch_2"},
    )
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2"),
        SwitchableAsset(grid_model_id="line3"),
    ]
    with pytest.raises(ValidationError, match="busbar_id busbar3 in asset line2 does not exist in busbars"):
        station = make_materialized_station(
            busbars=busbars,
            couplers=couplers,
            branch_assets=assets,
            branch_asset_bays=[None, path_error, None],
            branch_switching_table=asset_switching_table,
            bus_group_id=grid_model_id,
        )


def test_disambiguate_type() -> None:
    asset = normalize_switchable_asset_payload({"grid_model_id": "line", "asset_type": None})
    assert type(asset) is SwitchableAsset

    asset = normalize_switchable_asset_payload({"grid_model_id": "line", "asset_type": "line"})
    assert isinstance(asset, BranchAsset)

    asset = normalize_switchable_asset_payload({"grid_model_id": "gen", "asset_type": "gen"})
    assert isinstance(asset, InjectionAsset)
