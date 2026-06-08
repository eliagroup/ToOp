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
from toop_engine_interfaces.asset_topology import (
    AssetBay,
    Busbar,
    BusbarCoupler,
    MaterializedStation,
    RawStation,
    SwitchableAsset,
    Topology,
    build_asset_bay_id,
    topology_from_materialized_stations,
)
from toop_engine_interfaces.asset_topology_helpers import (
    filter_assets_by_type,
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    fix_multi_connected_without_coupler,
    has_transmission_line_switching,
    load_asset_topology,
    save_asset_topology,
)


def test_station() -> None:
    station = MaterializedStation(
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
    assert station is not None

    with pytest.raises(ValidationError):
        # Wrong shape of switching table
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]).T,
            grid_model_id="station1",
        )

    with pytest.raises(ValidationError):
        # Coupler references non-existing busbar
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]),
            grid_model_id="station1",
        )

    with pytest.raises(ValidationError):
        # Coupler references non-existing busbar
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]),
            grid_model_id="station1",
        )

    with pytest.raises(ValidationError):
        # Busbar int_id is not unique
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]),
            grid_model_id="station1",
        )

    with pytest.raises(ValidationError):
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2", in_service=False),
                SwitchableAsset(grid_model_id="line3"),
                SwitchableAsset(grid_model_id="line4", in_service=False),
            ],
            asset_switching_table=np.array(
                [
                    [True, False, True, True],
                    [False, True, False, False],
                    [True, False, True, True],
                ]
            ),
            grid_model_id="station1",
        )

    with pytest.raises(ValidationError):
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]),
            grid_model_id="station1",
        )

    station = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
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
        station = MaterializedStation(
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
            assets=[
                SwitchableAsset(grid_model_id="line1"),
                SwitchableAsset(grid_model_id="line2"),
                SwitchableAsset(grid_model_id="line3"),
            ],
            asset_switching_table=np.array([[True, False, True], [False, True, False]]),
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
    station = MaterializedStation(
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=asset_connectivity,
        grid_model_id=grid_model_id,
    )
    assert station is not None

    with pytest.raises(ValidationError):
        # entry in asset_switching_table is not in asset_connectivity
        asset_switching_table = np.array([[True, False, True], [False, True, False]])
        asset_connectivity = np.array([[True, True, True], [True, False, True]])
        station = MaterializedStation(
            busbars=busbars,
            couplers=couplers,
            assets=assets,
            asset_switching_table=asset_switching_table,
            asset_connectivity=asset_connectivity,
            grid_model_id=grid_model_id,
        )

    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    asset_connectivity = np.array([[True, True, True], [True, True, True]])
    station = MaterializedStation(
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=asset_connectivity,
        grid_model_id=grid_model_id,
    )
    assert station is not None


def test_topology_station_is_split() -> None:
    station = RawStation(
        grid_model_id="bus_id1",
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1", bus_branch_bus_id="bus_id1"),
            Busbar(int_id=2, grid_model_id="busbar2", bus_branch_bus_id="bus_id1"),
        ],
        couplers=[],
        asset_ids=["line1"],
        asset_terminals=[None],
        asset_bay_ids=[None],
        asset_switching_table=np.array([[True], [False]]),
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
    assert "assets" in schema["properties"]
    assert "asset_switching_table" in schema["properties"]
    assert "grid_model_id" in schema["properties"]


def test_serialize_station() -> None:
    station = MaterializedStation(
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

    serialized = station.model_dump_json()
    station2 = MaterializedStation.model_validate_json(serialized)

    assert station == station2


def test_load_asset_topology() -> None:
    station1 = MaterializedStation(
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

    station2 = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar3"),
            Busbar(int_id=2, grid_model_id="busbar4"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line4"),
            SwitchableAsset(grid_model_id="line5"),
            SwitchableAsset(grid_model_id="line6"),
        ],
        asset_switching_table=np.array([[True, False, True], [False, True, False]]),
        grid_model_id="station2",
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
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
    station = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(
                grid_model_id="line1",
                asset_bay_id="station1::line1::bay",
            ),
            SwitchableAsset(grid_model_id="load1"),
        ],
        asset_terminals=["from", None],
        asset_bays=[
            AssetBay(
                asset_bay_id="station1::line1::bay",
                dv_switch_grid_model_id="dv1",
                sr_switch_grid_model_id={"busbar1": "sr1"},
            ),
            None,
        ],
        asset_switching_table=np.array([[True, False], [False, True]]),
        grid_model_id="station1",
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            timestamp=datetime.now(),
        ),
        stations=[station],
    )

    assert topology.raw_stations == [
        RawStation(
            grid_model_id="station1",
            busbars=station.busbars,
            couplers=station.couplers,
            asset_ids=["line1", "load1"],
            asset_terminals=["from", None],
            asset_bay_ids=["station1::line1::bay", None],
            asset_switching_table=np.array([[True, False], [False, True]]),
        )
    ]
    assert [asset.grid_model_id for asset in topology.assets] == ["line1", "load1"]
    assert [asset_bay.asset_bay_id for asset_bay in topology.asset_bays] == ["station1::line1::bay"]

    materialized_station = topology.materialize_stations()[0]
    assert materialized_station == station.model_copy(
        update={
            "assets": [
                station.assets[0].model_copy(update={"asset_bay_id": "station1::line1::bay"}),
                station.assets[1],
            ]
        }
    )


def test_topology_from_materialized_stations_keeps_single_canonical_asset_for_two_station_views() -> None:
    asset_from = SwitchableAsset(grid_model_id="line1", type="line")
    asset_to = SwitchableAsset(grid_model_id="line1", type="line")

    station_from = MaterializedStation(
        grid_model_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        assets=[asset_from],
        asset_terminals=["from"],
        asset_switching_table=np.array([[True]]),
    )
    station_to = MaterializedStation(
        grid_model_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        assets=[asset_to],
        asset_terminals=["to"],
        asset_switching_table=np.array([[True]]),
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            timestamp=datetime.now(),
        ),
        stations=[station_from, station_to],
    )

    assert [asset.grid_model_id for asset in topology.assets] == ["line1"]
    assert topology.raw_stations[0].asset_ids == ["line1"]
    assert topology.raw_stations[0].asset_terminals == ["from"]
    assert topology.raw_stations[1].asset_ids == ["line1"]
    assert topology.raw_stations[1].asset_terminals == ["to"]
    assert topology.materialize_stations() == [station_from, station_to]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_station() -> None:
    station_from = MaterializedStation(
        grid_model_id="station_from",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        assets=[
            SwitchableAsset(
                grid_model_id="line1",
                type="line",
                asset_bay_id=build_asset_bay_id("station_from", "line1"),
            )
        ],
        asset_terminals=["from"],
        asset_bays=[
            AssetBay(
                asset_bay_id=build_asset_bay_id("station_from", "line1"),
                dv_switch_grid_model_id="dv_from",
                sr_switch_grid_model_id={"busbar1": "sr_from"},
            )
        ],
        asset_switching_table=np.array([[True]]),
    )
    station_to = MaterializedStation(
        grid_model_id="station_to",
        busbars=[Busbar(int_id=1, grid_model_id="busbar2")],
        couplers=[],
        assets=[
            SwitchableAsset(
                grid_model_id="line1",
                type="line",
                asset_bay_id=build_asset_bay_id("station_to", "line1"),
            )
        ],
        asset_terminals=["to"],
        asset_bays=[
            AssetBay(
                asset_bay_id=build_asset_bay_id("station_to", "line1"),
                dv_switch_grid_model_id="dv_to",
                sr_switch_grid_model_id={"busbar2": "sr_to"},
            )
        ],
        asset_switching_table=np.array([[True]]),
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            timestamp=datetime.now(),
        ),
        stations=[station_from, station_to],
    )

    assert sorted(asset_bay.asset_bay_id for asset_bay in topology.asset_bays) == [
        "station_from::line1::bay",
        "station_to::line1::bay",
    ]
    assert topology.raw_stations[0].asset_bay_ids == ["station_from::line1::bay"]
    assert topology.raw_stations[1].asset_bay_ids == ["station_to::line1::bay"]


def test_topology_from_materialized_stations_scopes_generated_asset_bay_ids_per_occurrence() -> None:
    station = MaterializedStation(
        grid_model_id="station1",
        busbars=[Busbar(int_id=1, grid_model_id="busbar1")],
        couplers=[],
        assets=[
            SwitchableAsset(
                grid_model_id="line1",
                type="line",
                asset_bay_id=build_asset_bay_id("station1", "line1"),
            ),
            SwitchableAsset(
                grid_model_id="line1",
                type="line",
                asset_bay_id=build_asset_bay_id("station1", "line1", 1),
            ),
        ],
        asset_bays=[
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
        asset_switching_table=np.array([[True, True]]),
    )

    topology = topology_from_materialized_stations(
        reference_topology=Topology(
            topology_id="topology1",
            raw_stations=[],
            timestamp=datetime.now(),
        ),
        stations=[station],
    )

    assert topology.raw_stations[0].asset_ids == ["line1", "line1"]
    assert topology.raw_stations[0].asset_bay_ids == ["station1::line1::bay", "station1::line1::bay::1"]
    assert sorted(asset_bay.asset_bay_id for asset_bay in topology.asset_bays) == [
        "station1::line1::bay",
        "station1::line1::bay::1",
    ]


def test_filter_out_of_service() -> None:
    station = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2", in_service=False),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4", in_service=False),
        ],
        asset_switching_table=np.array(
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
    assert len(station.assets) == 2
    assert np.array_equal(station.asset_switching_table, np.array([[True, True], [False, False]]))


def test_has_transmission_line_switching() -> None:
    station = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2", in_service=False),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4", in_service=False),
        ],
        asset_switching_table=np.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [True, False, True, False],
            ]
        ),
        grid_model_id="station1",
    )

    assert has_transmission_line_switching(station) is False

    station = MaterializedStation(
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
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
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
    station = MaterializedStation(
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
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
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
    station = MaterializedStation(
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
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
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
    assert len(station.assets) == 4
    assert np.array_equal(
        station.asset_switching_table,
        np.array([[True, False, True, True], [False, True, False, False]]),
    )
    assert len(removed) == 1
    assert removed[0].int_id == 3


def test_filter_disconnected_busbars_sort_by_asset_count() -> None:
    station = MaterializedStation(
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
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
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
    assert len(station.assets) == 4
    assert np.array_equal(
        station.asset_switching_table,
        np.array([[True, False, True, True]]),
    )
    assert len(removed) == 2
    assert removed[0].grid_model_id == "busbar1"
    assert removed[1].grid_model_id == "busbar2"


def test_select_one_for_multi_connected_assets() -> None:
    station = MaterializedStation(
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
            SwitchableAsset(grid_model_id="line1"),  # Connects busbar 1 and 3
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),  # Connects busbar 2 and 3
        ],
        asset_switching_table=np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, False, True],
            ]
        ),
        grid_model_id="station1",
    )

    station, removed = fix_multi_connected_without_coupler(station)
    assert station.asset_switching_table[:, 0].sum() == 1
    assert np.array_equal(
        station.asset_switching_table,
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
    station = MaterializedStation(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1", type="line"),
            SwitchableAsset(grid_model_id="line2", type="line"),
            SwitchableAsset(grid_model_id="gen1", type="gen"),
            SwitchableAsset(grid_model_id="load1", type="load"),
            SwitchableAsset(grid_model_id="line3", type=None),
        ],
        asset_switching_table=np.array(
            [
                [True, False, True, False, True],
                [False, True, False, True, False],
            ]
        ),
        grid_model_id="station1",
    )

    station_filtered, removed = filter_assets_by_type(station, set(["line", "trafo"]))
    assert len(station_filtered.assets) == 2
    assert len(removed) == 3
    assert station_filtered.assets[0].grid_model_id == "line1"
    assert station_filtered.assets[1].grid_model_id == "line2"
    assert station_filtered.asset_switching_table.shape == (2, 2)

    station_filtered, removed = filter_assets_by_type(station, set(["line", "gen"]), allow_none_type=True)
    assert len(station_filtered.assets) == 4
    assert len(removed) == 1
    assert station_filtered.assets[0].grid_model_id == "line1"
    assert station_filtered.assets[1].grid_model_id == "line2"
    assert station_filtered.assets[2].grid_model_id == "gen1"
    assert station_filtered.assets[3].grid_model_id == "line3"


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
        SwitchableAsset(grid_model_id="line2", asset_bay_id=path.asset_bay_id),
        SwitchableAsset(grid_model_id="line3"),
    ]
    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    grid_model_id = "station1"

    # test valid Station
    station = MaterializedStation(
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_bays=[None, path, None],
        asset_switching_table=asset_switching_table,
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
        SwitchableAsset(grid_model_id="line2", asset_bay_id=path_error.asset_bay_id),
        SwitchableAsset(grid_model_id="line3"),
    ]
    with pytest.raises(ValidationError, match="busbar_id busbar3 in asset line2 does not exist in busbars"):
        station = MaterializedStation(
            busbars=busbars,
            couplers=couplers,
            assets=assets,
            asset_bays=[None, path_error, None],
            asset_switching_table=asset_switching_table,
            grid_model_id=grid_model_id,
        )


def test_disambiguate_type() -> None:
    asset = SwitchableAsset(grid_model_id="line", type=None)
    assert asset.is_branch() is None

    asset = SwitchableAsset(grid_model_id="line", type="line")
    assert asset.is_branch() is True

    asset = SwitchableAsset(grid_model_id="gen", type="gen")
    assert asset.is_branch() is False
