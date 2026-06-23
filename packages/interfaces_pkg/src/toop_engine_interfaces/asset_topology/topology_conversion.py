# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Conversions between materialized stations and canonical topology models."""

from beartype.typing import Optional
from toop_engine_interfaces.asset_topology.asset_topology import Topology
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    InjectionAsset,
    normalize_switchable_asset_payload,
)
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import RawStation, StationAssetConnection


def topology_parts_from_materialized_station(
    station: MaterializedStation,
) -> tuple[RawStation, list[BranchAsset], list[InjectionAsset], list[AssetBay]]:
    """Extract topology-owned payloads from a materialized station.

    Parameters
    ----------
    station : MaterializedStation
        Materialized station to decompose.

    Returns
    -------
    tuple[RawStation, list[BranchAsset], list[InjectionAsset], list[AssetBay]]
        Raw station record plus topology-owned branch assets, injection assets, and asset bays.
    """
    branch_assets: list[BranchAsset] = []
    injection_assets: list[InjectionAsset] = []
    asset_bays: list[AssetBay] = []
    branch_connections: list[StationAssetConnection] = []
    injection_connections: list[StationAssetConnection] = []
    for asset_connection in station.branch_connections:
        asset = normalize_switchable_asset_payload(asset_connection.asset.model_dump(round_trip=True))
        asset_bay = asset_connection.asset_bay
        asset_id = asset.grid_model_id
        asset_bay_id: Optional[str] = None

        if asset_bay is not None:
            asset_bay_id = asset_bay.asset_bay_id
            asset_bays.append(asset_bay.model_copy(deep=True))

        branch_asset = asset if isinstance(asset, BranchAsset) else BranchAsset.model_validate(asset.model_dump())
        branch_assets.append(branch_asset.model_copy(deep=True))
        branch_connections.append(
            StationAssetConnection(
                asset_id=asset_id,
                branch_end=asset_connection.branch_end,
                asset_bay_id=asset_bay_id,
            )
        )

    for asset_connection in station.injection_connections:
        asset = normalize_switchable_asset_payload(asset_connection.asset.model_dump(round_trip=True))
        asset_bay = asset_connection.asset_bay
        asset_id = asset.grid_model_id
        asset_bay_id = asset_bay.asset_bay_id if asset_bay is not None else None

        if asset_bay is not None:
            asset_bays.append(asset_bay.model_copy(deep=True))

        injection_asset = asset if isinstance(asset, InjectionAsset) else InjectionAsset.model_validate(asset.model_dump())
        injection_assets.append(injection_asset.model_copy(deep=True))
        injection_connections.append(
            StationAssetConnection(
                asset_id=asset_id,
                branch_end=asset_connection.branch_end,
                asset_bay_id=asset_bay_id,
            )
        )

    return (
        RawStation(
            grid_model_id=station.grid_model_id,
            name=station.name,
            station_type=station.station_type,
            region=station.region,
            voltage_level=station.voltage_level,
            busbars=station.busbars,
            couplers=station.couplers,
            branch_connections=branch_connections,
            injection_connections=injection_connections,
            branch_switching_table=station.branch_switching_table,
            injection_switching_table=station.injection_switching_table,
            branch_connectivity=station.branch_connectivity,
            injection_connectivity=station.injection_connectivity,
            model_log=station.model_log,
        ),
        branch_assets,
        injection_assets,
        asset_bays,
    )


def topology_from_materialized_stations(reference_topology: Topology, stations: list[MaterializedStation]) -> Topology:
    """Create a topology from materialized stations.

    Parameters
    ----------
    reference_topology : Topology
        Reference topology providing shared metadata.
    stations : list[MaterializedStation]
        Materialized stations to convert.

    Returns
    -------
    Topology
        Topology containing updated raw stations while reusing the reference topology-owned payloads.
    """
    topology_stations: list[RawStation] = []

    for station in stations:
        topology_station, _station_branch_assets, _station_injection_assets, _station_asset_bays = (
            topology_parts_from_materialized_station(station)
        )
        topology_stations.append(topology_station)

    return Topology(
        topology_id=reference_topology.topology_id,
        grid_model_file=reference_topology.grid_model_file,
        name=reference_topology.name,
        raw_stations=topology_stations,
        branch_assets=[asset.model_copy(deep=True) for asset in reference_topology.branch_assets],
        injection_assets=[asset.model_copy(deep=True) for asset in reference_topology.injection_assets],
        asset_bays=[asset_bay.model_copy(deep=True) for asset_bay in reference_topology.asset_bays],
        asset_setpoints=reference_topology.asset_setpoints,
        timestamp=reference_topology.timestamp,
        metrics=reference_topology.metrics,
    )
