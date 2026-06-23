# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from toop_engine_interfaces.asset_topology.asset_topology import (
    RawStation,
    Topology,
)
from toop_engine_interfaces.asset_topology.asset_topology_helpers import (
    filter_assets_by_type,
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    fix_multi_connected_without_coupler,
    has_transmission_line_switching,
    load_asset_topology,
    save_asset_topology,
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
from toop_engine_interfaces.asset_topology.topology_conversion import topology_from_materialized_stations


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
        MaterializedAssetConnection(asset=asset, terminal=terminal, asset_bay=asset_bay)
        for asset, terminal, asset_bay in zip(assets, terminals, asset_bays, strict=True)
    ]


def raw_asset_connections(
    asset_ids: list[str],
    terminals: list[str | None] | None = None,
    asset_bay_ids: list[str | None] | None = None,
) -> list[StationAssetConnection]:
    if terminals is None:
        terminals = [None] * len(asset_ids)
    if asset_bay_ids is None:
        asset_bay_ids = [None] * len(asset_ids)
    return [
        StationAssetConnection(asset_id=asset_id, terminal=terminal, asset_bay_id=asset_bay_id)
        for asset_id, terminal, asset_bay_id in zip(asset_ids, terminals, asset_bay_ids, strict=True)
    ]


def make_materialized_station(
    *,
    grid_model_id: str,
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
        grid_model_id=grid_model_id,
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


def make_raw_station(
    *,
    grid_model_id: str,
    busbars: list[Busbar],
    couplers: list[BusbarCoupler],
    branch_asset_ids: list[str],
    branch_switching_table: np.ndarray,
    branch_connectivity: np.ndarray | None = None,
    branch_terminals: list[str | None] | None = None,
    branch_asset_bay_ids: list[str | None] | None = None,
    injection_asset_ids: list[str] | None = None,
    injection_switching_table: np.ndarray | None = None,
    injection_connectivity: np.ndarray | None = None,
    injection_terminals: list[str | None] | None = None,
    injection_asset_bay_ids: list[str | None] | None = None,
) -> RawStation:
    n_busbars = len(busbars)
    resolved_injection_asset_ids = injection_asset_ids or []
    resolved_injection_switching = injection_switching_table
    if resolved_injection_switching is None:
        resolved_injection_switching = np.zeros((n_busbars, len(resolved_injection_asset_ids)), dtype=bool)

    return RawStation(
        grid_model_id=grid_model_id,
        busbars=busbars,
        couplers=couplers,
        branch_connections=raw_asset_connections(branch_asset_ids, branch_terminals, branch_asset_bay_ids),
        injection_connections=raw_asset_connections(
            resolved_injection_asset_ids,
            injection_terminals,
            injection_asset_bay_ids,
        ),
        branch_switching_table=branch_switching_table,
        injection_switching_table=resolved_injection_switching,
        branch_connectivity=branch_connectivity,
        injection_connectivity=injection_connectivity,
    )


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
        grid_model_id="station1",
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
            grid_model_id="station1",
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
            grid_model_id="station1",
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
            grid_model_id="station1",
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
            grid_model_id="station1",
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
            grid_model_id="station1",
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
            grid_model_id="station1",
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
        grid_model_id="bus_id1",
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
            grid_model_id="bus_id0",
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
        grid_model_id=grid_model_id,
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
            grid_model_id=grid_model_id,
        )

    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    asset_connectivity = np.array([[True, True, True], [True, True, True]])
    station = make_materialized_station(
        busbars=busbars,
        couplers=couplers,
        branch_assets=assets,
        branch_switching_table=asset_switching_table,
        branch_connectivity=asset_connectivity,
        grid_model_id=grid_model_id,
    )
    assert station is not None


def test_topology_station_is_split() -> None:
    station = make_raw_station(
        grid_model_id="bus_id1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
        ],
        couplers=[],
        branch_asset_ids=["line1"],
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
    assert "grid_model_id" in schema["properties"]


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
        grid_model_id="station1",
    )

    serialized = station.model_dump_json()
    station2 = MaterializedStation.model_validate_json(serialized)

    assert station == station2


def test_load_asset_topology() -> None:
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
        grid_model_id="station1",
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
        grid_model_id="station2",
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[
                BranchAsset(grid_model_id="line1"),
                BranchAsset(grid_model_id="line2"),
                BranchAsset(grid_model_id="line3"),
                BranchAsset(grid_model_id="line4"),
                BranchAsset(grid_model_id="line5"),
                BranchAsset(grid_model_id="line6"),
            ],
            timestamp=datetime.now(),
        ),
        stations=[station1, station2],
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        save_asset_topology(tmpdirname / "topology.json", topology)
        loaded_topology = load_asset_topology(tmpdirname / "topology.json")
        assert topology == loaded_topology


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
        grid_model_id="station1",
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            injection_assets=[InjectionAsset(grid_model_id="load1", asset_type="load")],
            asset_bays=[asset_bays[0]],
            timestamp=datetime.now(),
        ),
        stations=[station],
    )

    assert topology.raw_stations == [
        RawStation(
            grid_model_id="station1",
            busbars=station.busbars,
            couplers=station.couplers,
            branch_connections=raw_asset_connections(["line1"], ["from"], ["station1::line1::bay"]),
            injection_connections=raw_asset_connections(["load1"], [None], [None]),
            branch_switching_table=np.array([[True], [False]]),
            injection_switching_table=np.array([[False], [True]]),
        )
    ]
    assert [asset.grid_model_id for asset in topology.branch_assets] == ["line1"]
    assert [asset.grid_model_id for asset in topology.injection_assets] == ["load1"]
    assert [asset_bay.asset_bay_id for asset_bay in topology.asset_bays] == ["station1::line1::bay"]

    materialized_station = topology.materialize_stations()[0]
    assert materialized_station == station


def test_topology_from_materialized_stations_keeps_single_canonical_asset_for_two_station_views() -> None:
    asset_from = BranchAsset(grid_model_id="line1", asset_type="line")
    asset_to = BranchAsset(grid_model_id="line1", asset_type="line")

    station_from = make_materialized_station(
        grid_model_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[asset_from],
        branch_terminals=["from"],
        branch_switching_table=np.array([[True]]),
    )
    station_to = make_materialized_station(
        grid_model_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        branch_assets=[asset_to],
        branch_terminals=["to"],
        branch_switching_table=np.array([[True]]),
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            timestamp=datetime.now(),
        ),
        stations=[station_from, station_to],
    )

    assert [asset.grid_model_id for asset in topology.branch_assets] == ["line1"]
    assert [connection.asset_id for connection in topology.raw_stations[0].branch_connections] == ["line1"]
    assert [connection.terminal for connection in topology.raw_stations[0].branch_connections] == ["from"]
    assert [connection.asset_id for connection in topology.raw_stations[1].branch_connections] == ["line1"]
    assert [connection.terminal for connection in topology.raw_stations[1].branch_connections] == ["to"]
    assert topology.materialize_stations() == [station_from, station_to]


def test_topology_from_materialized_stations_normalizes_equivalent_branch_asset_payloads() -> None:
    station_from = make_materialized_station(
        grid_model_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["from"],
        branch_switching_table=np.array([[True]]),
    )
    station_to = make_materialized_station(
        grid_model_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        branch_assets=[SwitchableAsset(grid_model_id="line1", asset_type="line")],
        branch_terminals=["to"],
        branch_switching_table=np.array([[True]]),
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            timestamp=datetime.now(),
        ),
        stations=[station_from, station_to],
    )

    assert len(topology.branch_assets) == 1
    assert isinstance(topology.branch_assets[0], BranchAsset)
    assert topology.branch_assets[0].grid_model_id == "line1"


def test_topology_from_materialized_stations_reuses_reference_canonical_assets() -> None:
    reference_topology = Topology(
        topology_id="topology1",
        raw_stations=[],
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
        timestamp=datetime.now(),
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
        grid_model_id="station1",
    )

    topology = topology_from_materialized_stations(reference_topology=reference_topology, stations=[station])

    assert [asset.grid_model_id for asset in topology.branch_assets] == ["line1", "line_unused"]
    assert [asset.grid_model_id for asset in topology.injection_assets] == ["load1"]
    assert [asset_bay.asset_bay_id for asset_bay in topology.asset_bays] == ["station1::line1::bay"]
    assert topology.raw_stations[0].branch_connections[0].asset_id == "line1"


def test_topology_from_materialized_stations_raises_when_reference_assets_are_missing() -> None:
    reference_topology = Topology(
        topology_id="topology1",
        raw_stations=[],
        branch_assets=[],
        injection_assets=[],
        asset_bays=[],
        timestamp=datetime.now(),
    )
    station = make_materialized_station(
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
        branch_switching_table=np.array([[True]]),
        grid_model_id="station1",
    )

    with pytest.raises(
        ValidationError, match="Branch asset grid_model_id line1 referenced by station station1 does not exist"
    ):
        topology_from_materialized_stations(reference_topology=reference_topology, stations=[station])


def test_raw_station_model_copy_revalidates_updates() -> None:
    station = make_raw_station(
        grid_model_id="station1",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1"), Busbar(int_id=2, grid_model_id="busbar2")],
        couplers=[],
        branch_asset_ids=["line1"],
        branch_switching_table=np.array([[True], [False]]),
    )

    with pytest.raises(ValidationError, match="branch_switching_table shape"):
        station.model_copy(update={"branch_switching_table": np.array([[True]], dtype=bool)})


def test_raw_station_model_copy_honors_deep_flag_for_nested_models() -> None:
    station = make_raw_station(
        grid_model_id="station1",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1"), Busbar(int_id=2, grid_model_id="busbar2")],
        couplers=[],
        branch_asset_ids=["line1"],
        branch_switching_table=np.array([[True], [False]]),
    )

    shallow_copy = station.model_copy()
    deep_copy = station.model_copy(deep=True)

    assert shallow_copy.busbars[0] is station.busbars[0]
    assert deep_copy.busbars[0] is not station.busbars[0]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_station() -> None:
    station_from = make_materialized_station(
        grid_model_id="station_from",
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
        grid_model_id="station_to",
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

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            asset_bays=[
                station_from.branch_connections[0].asset_bay.model_copy(deep=True),
                station_to.branch_connections[0].asset_bay.model_copy(deep=True),
            ],
            timestamp=datetime.now(),
        ),
        stations=[station_from, station_to],
    )

    assert sorted(asset_bay.asset_bay_id for asset_bay in topology.asset_bays) == [
        "station_from::line1::bay",
        "station_to::line1::bay",
    ]
    assert [connection.asset_bay_id for connection in topology.raw_stations[0].branch_connections] == [
        "station_from::line1::bay"
    ]
    assert [connection.asset_bay_id for connection in topology.raw_stations[1].branch_connections] == [
        "station_to::line1::bay"
    ]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_occurrence() -> None:
    station = make_materialized_station(
        grid_model_id="station1",
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

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            branch_assets=[BranchAsset(grid_model_id="line1", asset_type="line")],
            asset_bays=[
                station.branch_connections[0].asset_bay.model_copy(deep=True),
                station.branch_connections[1].asset_bay.model_copy(deep=True),
            ],
            timestamp=datetime.now(),
        ),
        stations=[station],
    )

    assert [connection.asset_id for connection in topology.raw_stations[0].branch_connections] == ["line1", "line1"]
    assert [connection.asset_bay_id for connection in topology.raw_stations[0].branch_connections] == [
        "station1::line1::bay",
        "station1::line1::bay::1",
    ]
    assert sorted(asset_bay.asset_bay_id for asset_bay in topology.asset_bays) == [
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id="station1",
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
        grid_model_id=grid_model_id,
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
            grid_model_id=grid_model_id,
        )


def test_disambiguate_type() -> None:
    asset = normalize_switchable_asset_payload({"grid_model_id": "line", "asset_type": None})
    assert type(asset) is SwitchableAsset

    asset = normalize_switchable_asset_payload({"grid_model_id": "line", "asset_type": "line"})
    assert isinstance(asset, BranchAsset)

    asset = normalize_switchable_asset_payload({"grid_model_id": "gen", "asset_type": "gen"})
    assert isinstance(asset, InjectionAsset)
