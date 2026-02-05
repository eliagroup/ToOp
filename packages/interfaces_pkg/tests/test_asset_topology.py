# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
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
    Station,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import (
    compare_stations,
    filter_assets_by_type,
    filter_disconnected_busbars,
    filter_duplicate_couplers,
    filter_out_of_service,
    fix_multi_connected_without_coupler,
    has_transmission_line_switching,
    load_asset_topology,
    merge_couplers,
    merge_station,
    number_of_splits,
    save_asset_topology,
)


def test_station() -> None:
    station = Station(
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
        station = Station(
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
        station = Station(
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
        station = Station(
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
        station = Station(
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
        station = Station(
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
        station = Station(
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

    station = Station(
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

    with pytest.raises(ValidationError):
        station = Station(
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
    station = Station(
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
        station = Station(
            busbars=busbars,
            couplers=couplers,
            assets=assets,
            asset_switching_table=asset_switching_table,
            asset_connectivity=asset_connectivity,
            grid_model_id=grid_model_id,
        )

    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    asset_connectivity = np.array([[True, True, True], [True, True, True]])
    station = Station(
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=asset_connectivity,
        grid_model_id=grid_model_id,
    )
    assert station is not None


def test_schema() -> None:
    # Schema generation works
    schema = Station.model_json_schema()
    assert schema is not None
    assert "busbars" in schema["properties"]
    assert "couplers" in schema["properties"]
    assert "assets" in schema["properties"]
    assert "asset_switching_table" in schema["properties"]
    assert "grid_model_id" in schema["properties"]


def test_serialize_station() -> None:
    station = Station(
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
    station2 = Station.model_validate_json(serialized)

    assert station == station2


def test_load_asset_topology() -> None:
    station1 = Station(
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

    station2 = Station(
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

    topology = Topology(
        stations=[station1, station2],
        topology_id="topology1",
        timestamp=datetime.now(),
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        save_asset_topology(tmpdirname / "topology.json", topology)
        loaded_topology = load_asset_topology(tmpdirname / "topology.json")
        assert topology == loaded_topology


def test_filter_out_of_service() -> None:
    station = Station(
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
    station = Station(
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

    station = Station(
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
    station = Station(
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
    station = Station(
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
    station = Station(
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
    station = Station(
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
    station = Station(
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


def test_number_of_splits() -> None:
    station = Station(
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
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line4"),
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

    assert number_of_splits(station) == 1

    station = Station(
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
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, True],
            ]
        ),
        grid_model_id="station1",
    )

    assert number_of_splits(station) == 2

    station = Station(
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
            SwitchableAsset(grid_model_id="line4"),  # This line is isolated on the busbar
        ],
        asset_switching_table=np.array(
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, False, False, True],
            ]
        ),
        grid_model_id="station1",
    )

    assert number_of_splits(station) == 1


def test_compare_stations() -> None:
    original = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
            Busbar(int_id=4, grid_model_id="busbar4", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            BusbarCoupler(
                busbar_from_id=3,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler3",
                in_service=False,
            ),
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=4,
                open=False,
                grid_model_id="coupler4",
                in_service=False,
            ),
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
                [False, False, False, False, False],
            ]
        ),
        grid_model_id="station1",
    )

    new = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
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
            ]
        ),
        grid_model_id="station1",
    )

    (
        coupler_diff_a,
        coupler_diff_b,
        bus_diff_a,
        bus_diff_b,
        asset_diff_a,
        asset_diff_b,
    ) = compare_stations(original, new)

    # A is a superset of B, so all B diffs should be empty
    assert len(coupler_diff_b) == 0
    assert len(bus_diff_b) == 0
    assert len(asset_diff_b) == 0

    assert len(coupler_diff_a) == 3
    assert len(bus_diff_a) == 2
    assert len(asset_diff_a) == 0


def test_merge_station() -> None:
    original = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
            Busbar(int_id=3, grid_model_id="busbar3"),
            Busbar(int_id=4, grid_model_id="busbar4", in_service=False),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
            BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
            BusbarCoupler(
                busbar_from_id=3,
                busbar_to_id=1,
                open=False,
                grid_model_id="coupler3",
                in_service=False,
            ),
            BusbarCoupler(
                busbar_from_id=1,
                busbar_to_id=4,
                open=False,
                grid_model_id="coupler4",
                in_service=False,
            ),
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
                [False, True, True, False, True],
                [False, False, False, False, False],
            ]
        ),
        grid_model_id="station1",
    )

    new = Station(
        busbars=[
            Busbar(int_id=1, grid_model_id="busbar1"),
            Busbar(int_id=2, grid_model_id="busbar2"),
        ],
        couplers=[
            BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        ],
        assets=[
            SwitchableAsset(grid_model_id="line1"),
            SwitchableAsset(grid_model_id="line2"),
            SwitchableAsset(grid_model_id="line3"),
            SwitchableAsset(grid_model_id="line5"),  # Switch 5 and 4
            SwitchableAsset(grid_model_id="line4"),
        ],
        asset_switching_table=np.array(
            [
                [False, True, False, False, True],
                [False, True, False, True, False],
            ]
        ),
        grid_model_id="station1",
    )

    merged, coupler_diff, asset_diff = merge_station(original, new)

    assert len(asset_diff) == np.sum(np.logical_xor(original.asset_switching_table, merged.asset_switching_table))

    assert np.array_equal(
        merged.asset_switching_table,
        np.array(
            [
                [False, True, False, True, False],
                [False, True, False, False, True],
                [False, True, True, False, True],
                [False, False, False, False, False],
            ]
        ),
    )

    assert merged.couplers[0].open

    merged, coupler_diff, asset_diff = merge_station(new, original)

    assert len(asset_diff) == np.sum(np.logical_xor(new.asset_switching_table, merged.asset_switching_table))

    assert np.array_equal(
        merged.asset_switching_table,
        np.array([[True, True, True, False, False], [False, False, False, False, True]]),
    )

    merged, coupler_diff, asset_diff = merge_station(original, original)
    assert merged == original
    assert not coupler_diff
    assert not asset_diff


def test_merge_couplers() -> None:
    couplers_old = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler2"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler3"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler4"),
        BusbarCoupler(busbar_from_id=3, busbar_to_id=4, open=True, grid_model_id="coupler5"),
    ]

    couplers_new = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler3"),
    ]

    busbar_mapping = {1: 1, 2: 2, 3: 3, 4: 4}

    merged, diff = merge_couplers(couplers_old, couplers_new, busbar_mapping)

    assert len(merged) == len(couplers_old)
    assert len(diff) == 3
    assert diff[0] == BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1")
    assert diff[1] == BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler2")
    assert diff[2] == BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler4")

    assert merged[0].open is False
    assert merged[1].open is False
    assert merged[2].open is True
    assert merged[3].open is True
    assert merged[4].open is True


def test_merge_couplers_duplicate() -> None:
    couplers_old = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler2"),
    ]

    couplers_new = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler2"),
    ]

    busbar_mapping = {1: 1, 2: 2}

    merged, diff = merge_couplers(couplers_old, couplers_new, busbar_mapping)

    assert merged == [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler2"),
    ]
    assert len(diff) == 1

    # Conflicting information, in this case we close all couplers
    couplers_new = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler2"),
    ]

    merged, diff = merge_couplers(couplers_old, couplers_new, busbar_mapping)

    assert merged == [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler2"),
    ]
    assert len(diff) == 1


def test_merge_couplers_busbar_mapping() -> None:
    couplers_old = [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=False, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=False, grid_model_id="coupler2"),
    ]

    couplers_new = [
        BusbarCoupler(busbar_from_id=21, busbar_to_id=22, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=22, busbar_to_id=23, open=True, grid_model_id="coupler2"),
    ]

    busbar_mapping = {1: 21, 2: 22, 3: 23}

    merged, diff = merge_couplers(couplers_old, couplers_new, busbar_mapping)

    assert merged == [
        BusbarCoupler(busbar_from_id=1, busbar_to_id=2, open=True, grid_model_id="coupler1"),
        BusbarCoupler(busbar_from_id=2, busbar_to_id=3, open=True, grid_model_id="coupler2"),
    ]

    assert diff == merged


def test_asset_bay() -> None:
    # Test valid AssetBay
    path = AssetBay(
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
    )
    assert path is not None

    path = AssetBay(
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
        sr_switch_bus_assignment=[1, 2],
    )
    assert path is not None

    # Test AssetBay with missing dv_switch_grid_model_id
    with pytest.raises(ValidationError, match="Field required"):
        path = AssetBay(
            sl_switch_grid_model_id="sl_switch_1",
            sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar2": "sr_switch_2"},
        )

    # Test AssetBay with empty sr_switch_grid_model_id
    with pytest.raises(ValidationError, match="sr_switch_grid_model_id must not be empty"):
        path = AssetBay(
            sl_switch_grid_model_id="sl_switch_1",
            dv_switch_grid_model_id="dv_switch_1",
            sr_switch_grid_model_id={},
        )

    # Test AssetBay with invalid sr_switch_grid_model_id type
    with pytest.raises(ValidationError):
        path = AssetBay(
            sl_switch_grid_model_id="sl_switch_1",
            dv_switch_grid_model_id="dv_switch_1",
            sr_switch_grid_model_id={"busbar1": 123},  # Invalid type
        )


def test_station_bay() -> None:
    path = AssetBay(
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
        SwitchableAsset(grid_model_id="line2", asset_bay=path),
        SwitchableAsset(grid_model_id="line3"),
    ]
    asset_switching_table = np.array([[True, False, True], [False, True, False]])
    grid_model_id = "station1"

    # test valid Station
    station = Station(
        busbars=busbars,
        couplers=couplers,
        assets=assets,
        asset_switching_table=asset_switching_table,
        grid_model_id=grid_model_id,
    )
    assert station is not None

    # Test invalid AssetBay -> busbar3 is not in busbars
    path_error = AssetBay(
        sl_switch_grid_model_id="sl_switch_1",
        dv_switch_grid_model_id="dv_switch_1",
        sr_switch_grid_model_id={"busbar1": "sr_switch_1", "busbar3": "sr_switch_2"},
    )
    assets = [
        SwitchableAsset(grid_model_id="line1"),
        SwitchableAsset(grid_model_id="line2", asset_bay=path_error),
        SwitchableAsset(grid_model_id="line3"),
    ]
    with pytest.raises(ValidationError, match="busbar_id busbar3 in asset line2 does not exist in busbars"):
        station = Station(
            busbars=busbars,
            couplers=couplers,
            assets=assets,
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
