# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    Station,
    SwitchableAsset,
)
from toop_engine_interfaces.asset_topology_loadflow import (
    StationWithLF,
    map_loadflow_results_station,
)


def test_map_loadflow_results_station() -> None:
    # Test the function with a simple case
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

    busbar_mapper = {
        "busbar1": (1.0, 2.0),
        "busbar2": (3.0, np.nan),
    }
    asset_mapper = {
        "line1": (5.0, 6.0, 7.0, None),
        "line3": (9.0, 10.0, np.inf, 12.0),
    }

    station_with_lf = map_loadflow_results_station(
        station=station,
        node_extractor=lambda busbar: busbar_mapper[busbar.grid_model_id],
        asset_extractor=lambda asset: asset_mapper[asset.grid_model_id],
    )
    assert station_with_lf.busbars[0].va == 1.0
    assert station_with_lf.busbars[0].vm == 2.0
    assert station_with_lf.busbars[1].va == 3.0
    assert station_with_lf.busbars[1].vm is None
    assert station_with_lf.busbars[2].va is None
    assert station_with_lf.busbars[2].vm is None
    assert station_with_lf.assets[0].p == 5.0
    assert station_with_lf.assets[0].q == 6.0
    assert station_with_lf.assets[0].i == 7.0
    assert station_with_lf.assets[0].i_max is None
    assert station_with_lf.assets[1].p is None
    assert station_with_lf.assets[1].q is None
    assert station_with_lf.assets[1].i is None
    assert station_with_lf.assets[1].i_max is None
    assert station_with_lf.assets[2].p == 9.0
    assert station_with_lf.assets[2].q == 10.0
    assert station_with_lf.assets[2].i is None
    assert station_with_lf.assets[2].i_max == 12.0
    assert station_with_lf.assets[3].p is None
    assert station_with_lf.assets[3].q is None
    assert station_with_lf.assets[3].i is None
    assert station_with_lf.assets[3].i_max is None

    x = station_with_lf.model_dump_json()
    reconstructed = StationWithLF.model_validate_json(x)

    assert reconstructed == station_with_lf
