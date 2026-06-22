# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Conversions between materialized stations and canonical topology models."""

from beartype.typing import Optional
from toop_engine_interfaces.asset_topology.asset_topology import Topology
from toop_engine_interfaces.asset_topology.assets import AssetBay, SwitchableAsset, normalize_switchable_asset_payload
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import RawStation, StationAssetConnection


def topology_parts_from_materialized_station(
    station: MaterializedStation,
) -> tuple[RawStation, list[SwitchableAsset], list[AssetBay]]:
    """Extract topology-owned payloads from a materialized station.

    Parameters
    ----------
    station : MaterializedStation
        Materialized station to decompose.

    Returns
    -------
    tuple[RawStation, list[SwitchableAsset], list[AssetBay]]
        Raw station record plus topology-owned assets and asset bays derived from the station.
    """
    branch_assets: list[SwitchableAsset] = []
    injection_assets: list[SwitchableAsset] = []
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

        branch_assets.append(asset.model_copy(deep=True))
        branch_connections.append(
            StationAssetConnection(
                asset_id=asset_id,
                terminal=asset_connection.terminal,
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

        injection_assets.append(asset.model_copy(deep=True))
        injection_connections.append(
            StationAssetConnection(
                asset_id=asset_id,
                terminal=asset_connection.terminal,
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
        [*branch_assets, *injection_assets],
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
        Topology containing raw stations and topology-owned payloads derived from ``stations``.
    """
    topology_stations: list[RawStation] = []
    topology_branch_assets: dict[str, SwitchableAsset] = {}
    topology_injection_assets: dict[str, SwitchableAsset] = {}
    topology_asset_bays: dict[str, AssetBay] = {}

    for station in stations:
        topology_station, station_assets, station_asset_bays = topology_parts_from_materialized_station(station)
        topology_stations.append(topology_station)
        for asset in station_assets:
            topology_assets = topology_branch_assets if asset.is_branch() is not False else topology_injection_assets
            existing_asset = topology_assets.get(asset.grid_model_id)
            if existing_asset is None:
                topology_assets[asset.grid_model_id] = asset
            elif existing_asset != asset:
                raise ValueError(f"Conflicting topology asset payload for grid_model_id {asset.grid_model_id}")
        for asset_bay in station_asset_bays:
            existing_asset_bay = topology_asset_bays.get(asset_bay.asset_bay_id)
            if existing_asset_bay is None:
                topology_asset_bays[asset_bay.asset_bay_id] = asset_bay
            elif existing_asset_bay != asset_bay:
                raise ValueError(f"Conflicting topology asset bay payload for asset_bay_id {asset_bay.asset_bay_id}")

    return Topology(
        topology_id=reference_topology.topology_id,
        grid_model_file=reference_topology.grid_model_file,
        name=reference_topology.name,
        raw_stations=topology_stations,
        branch_assets=list(topology_branch_assets.values()),
        injection_assets=list(topology_injection_assets.values()),
        asset_bays=list(topology_asset_bays.values()),
        asset_setpoints=reference_topology.asset_setpoints,
        timestamp=reference_topology.timestamp,
        metrics=reference_topology.metrics,
    )
