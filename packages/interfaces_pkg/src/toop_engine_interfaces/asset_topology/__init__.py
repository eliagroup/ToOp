# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Public asset topology API.

This package re-exports the main topology data models so downstream imports and
documentation can refer to a stable package-level namespace.
"""

from toop_engine_interfaces.asset_topology.applied_topology import AppliedStation, RealizedTopology
from toop_engine_interfaces.asset_topology.asset_topology import (
    BusGroupAssetConnection,
    CircuitGroup,
    MasterAssetTopology,
    MasterBusGroup,
)
from toop_engine_interfaces.asset_topology.assets import AssetBay, AssetSetpoint, Busbar, BusbarCoupler, SwitchableAsset
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
    RuntimeSwitchableAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeAssetTopology,
    RuntimeBusGroup,
    get_asset_bay_ids_for_asset,
    get_asset_bays_for_asset,
    validate_runtime_station_asset_references,
)

__all__ = [
    "AppliedStation",
    "AssetBay",
    "AssetSetpoint",
    "BusGroupAssetConnection",
    "Busbar",
    "BusbarCoupler",
    "CircuitGroup",
    "MasterAssetTopology",
    "MasterBusGroup",
    "RealizedTopology",
    "RuntimeAssetConnection",
    "RuntimeAssetTopology",
    "RuntimeBranchAsset",
    "RuntimeBusGroup",
    "RuntimeBusbar",
    "RuntimeBusbarCoupler",
    "RuntimeInjectionAsset",
    "RuntimeSwitchableAsset",
    "SwitchableAsset",
    "get_asset_bay_ids_for_asset",
    "get_asset_bays_for_asset",
    "validate_runtime_station_asset_references",
]
