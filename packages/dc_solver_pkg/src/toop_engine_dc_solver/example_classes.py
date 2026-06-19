# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Example classes for testing purposes."""

import datetime

import numpy as np
from toop_engine_interfaces.asset_topology import (
    AssetBay,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    RawStation,
    StationAssetConnection,
    Topology,
)


def get_basic_node_breaker_topology() -> Topology:
    """Create a Topology for the basic_node_breaker_network_powsybl.

    Based on example_grid.basic_node_breaker_network_powsybl().

    Returns
    -------
        RealizedTopology: A realized topology object for example_grid.basic_node_breaker_network_powsybl().
    """
    return Topology(
        topology_id="test",
        grid_model_file="test_file",
        name=None,
        raw_stations=[
            RawStation(
                grid_model_id="VL4_0",
                name="VLevel4",
                type=None,
                region="BE",
                voltage_level=225.0,
                busbars=[
                    Busbar(grid_model_id="BBS4_1", type="busbar", name="bus1", int_id=0, in_service=True),
                    Busbar(grid_model_id="BBS4_2", type="busbar", name="bus2", int_id=1, in_service=True),
                ],
                couplers=[
                    BusbarCoupler(
                        grid_model_id="VL4_BREAKER",
                        type="busbar_coupler",
                        name="VL4_BREAKER",
                        busbar_from_id=1,
                        busbar_to_id=0,
                        open=True,
                        in_service=True,
                    )
                ],
                branch_connections=[
                    StationAssetConnection(asset_id="L4", terminal=None, asset_bay_id="VL4_0::L4::bay"),
                    StationAssetConnection(asset_id="L5", terminal=None, asset_bay_id="VL4_0::L5::bay"),
                    StationAssetConnection(asset_id="L8", terminal=None, asset_bay_id="VL4_0::L8::bay"),
                ],
                injection_connections=[],
                branch_switching_table=np.array([[False, False, False], [True, True, False]], dtype=bool),
                injection_switching_table=np.zeros((2, 0), dtype=bool),
                branch_connectivity=np.array([[True, True, True], [True, True, True]], dtype=bool),
                injection_connectivity=np.zeros((2, 0), dtype=bool),
            )
        ],
        branch_assets=[
            BranchAsset(grid_model_id="L4", type="LINE", name="", in_service=True),
            BranchAsset(grid_model_id="L5", type="LINE", name="", in_service=True),
            BranchAsset(grid_model_id="L8", type="LINE", name="", in_service=True),
        ],
        injection_assets=[],
        asset_bays=[
            AssetBay(
                asset_bay_id="VL4_0::L4::bay",
                sl_switch_grid_model_id=None,
                dv_switch_grid_model_id="L42_BREAKER",
                sr_switch_grid_model_id={"BBS4_1": "L42_DISCONNECTOR_3_0", "BBS4_2": "L42_DISCONNECTOR_3_1"},
            ),
            AssetBay(
                asset_bay_id="VL4_0::L5::bay",
                sl_switch_grid_model_id=None,
                dv_switch_grid_model_id="L52_BREAKER",
                sr_switch_grid_model_id={"BBS4_1": "L52_DISCONNECTOR_5_0", "BBS4_2": "L52_DISCONNECTOR_5_1"},
            ),
            AssetBay(
                asset_bay_id="VL4_0::L8::bay",
                sl_switch_grid_model_id=None,
                dv_switch_grid_model_id="L82_BREAKER",
                sr_switch_grid_model_id={"BBS4_1": "L82_DISCONNECTOR_7_0", "BBS4_2": "L82_DISCONNECTOR_7_1"},
            ),
        ],
        timestamp=datetime.datetime(2025, 2, 4, 9, 12, 0, 109256),
        asset_setpoints=None,
        metrics=None,
    )
