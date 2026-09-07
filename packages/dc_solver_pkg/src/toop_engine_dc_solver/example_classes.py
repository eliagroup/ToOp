# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Example classes for testing purposes."""

import numpy as np
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
)
from toop_engine_interfaces.asset_topology.assets_runtime import RuntimeBranchAsset, RuntimeBusbar, RuntimeBusbarCoupler
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetConnection, RuntimeBusGroup


def get_basic_node_breaker_topology() -> list[RuntimeBusGroup]:
    """Create runtime stations for the basic node-breaker example grid.

    Based on example_grid.basic_node_breaker_network_powsybl().

    Returns
    -------
    list[RuntimeBusGroup]
        Runtime stations for example_grid.basic_node_breaker_network_powsybl().
    """
    stations = [
        RuntimeBusGroup(
            bus_group_id="VL4_0",
            name="VLevel4",
            station_type=None,
            region="BE",
            voltage_level=225.0,
            busbars=[
                RuntimeBusbar(grid_model_id="BBS4_1", busbar_type="busbar", name="bus1", int_id=0, in_service=True),
                RuntimeBusbar(grid_model_id="BBS4_2", busbar_type="busbar", name="bus2", int_id=1, in_service=True),
            ],
            couplers=[
                RuntimeBusbarCoupler(
                    grid_model_id="VL4_BREAKER",
                    coupler_type="busbar_coupler",
                    name="VL4_BREAKER",
                    busbar_from_id=1,
                    busbar_to_id=0,
                    open=True,
                    in_service=True,
                )
            ],
            branch_connections=[
                RuntimeAssetConnection(
                    asset=RuntimeBranchAsset(grid_model_id="L4", asset_type="LINE", name="", in_service=True),
                    branch_end=None,
                    asset_bay=AssetBay(
                        asset_bay_id="VL4_0::L4::bay",
                        asset_disconnector_grid_model_id=None,
                        breaker_grid_model_id="L42_BREAKER",
                        busbar_disconnector_grid_model_id={
                            "BBS4_1": "L42_DISCONNECTOR_3_0",
                            "BBS4_2": "L42_DISCONNECTOR_3_1",
                        },
                    ),
                ),
                RuntimeAssetConnection(
                    asset=RuntimeBranchAsset(grid_model_id="L5", asset_type="LINE", name="", in_service=True),
                    branch_end=None,
                    asset_bay=AssetBay(
                        asset_bay_id="VL4_0::L5::bay",
                        asset_disconnector_grid_model_id=None,
                        breaker_grid_model_id="L52_BREAKER",
                        busbar_disconnector_grid_model_id={
                            "BBS4_1": "L52_DISCONNECTOR_5_0",
                            "BBS4_2": "L52_DISCONNECTOR_5_1",
                        },
                    ),
                ),
                RuntimeAssetConnection(
                    asset=RuntimeBranchAsset(grid_model_id="L8", asset_type="LINE", name="", in_service=True),
                    branch_end=None,
                    asset_bay=AssetBay(
                        asset_bay_id="VL4_0::L8::bay",
                        asset_disconnector_grid_model_id=None,
                        breaker_grid_model_id="L82_BREAKER",
                        busbar_disconnector_grid_model_id={
                            "BBS4_1": "L82_DISCONNECTOR_7_0",
                            "BBS4_2": "L82_DISCONNECTOR_7_1",
                        },
                    ),
                ),
            ],
            injection_connections=[],
            branch_switching_table=np.array([[False, False, False], [True, True, False]], dtype=bool),
            injection_switching_table=np.zeros((2, 0), dtype=bool),
            branch_connectivity=np.array([[True, True, True], [True, True, True]], dtype=bool),
            injection_connectivity=np.zeros((2, 0), dtype=bool),
        )
    ]
    return stations
