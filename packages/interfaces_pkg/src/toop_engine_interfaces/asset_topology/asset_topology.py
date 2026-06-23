# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the data models for the asset topology."""

from datetime import datetime

from beartype.typing import Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    AssetSetpoint,
    BranchAsset,
    InjectionAsset,
    SwitchableAsset,
)
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import (
    RawStation,
    _merged_round_trip_payload,
)


class CircuitGroup(BaseModel):
    """A circuit group represents a set of assets that are connected to each other without power switches.

    This means in case of an outage, the fault current can flow through all assets in the same circuit group,
    triggering  their outage aswell.

    -> All assets in an asset group are outaged together.

    # TODO: This is currently not implemented. Use a graph search to determine these.
    """

    asset_ids: list[str]
    """The grid model ids of the assets in the circuit group.
    This can be used to quickly find the circuit group in case an asset with id x is outaged."""

    asset_bay_ids: list[str]
    """The asset bay ids of the asset bays in the circuit group.
    These can be used to apply the outage effect on the grid by opening the switches in the asset bays."""


class Topology(BaseModel):
    """Topology data describing a single timestep topology.

    A topology includes lean station records in raw_stations, topology-owned canonical assets and
    asset bays, and potentially asset setpoints.
    Use materialize_stations() to reconstruct rich Station objects.
    """

    topology_id: str
    """ The unique identifier of the topology. """

    grid_model_file: Optional[str] = None
    """ The grid model file that represents this timestep. Note that relevant folders might only
    work on the machine they have been created, so some sort of permanent storage server should be
    used to keep these files globally accessible"""

    name: Optional[str] = None
    """ The name of the topology. """

    raw_stations: list[RawStation]
    """The topology-owned station records without embedded asset payloads.

    Each raw station represents one bus-branch bus view of a splitable station.
    """

    circuit_groups: Optional[list[CircuitGroup]] = None
    """The topology-owned circuit groups. The list contains groups of assets that are connected to each
    other without power switches. This means in case of an outage, the fault current can flow through
    all assets in the same circuit group, triggering their outage as well.
    # TODO This is currently not implemented. Use a graph search to determine these."""

    branch_assets: list[BranchAsset] = Field(default_factory=list)
    """The topology-owned canonical branch payloads."""

    injection_assets: list[InjectionAsset] = Field(default_factory=list)
    """The topology-owned canonical injection payloads.

    Station-local branch-end and asset-bay assignment data are stored on raw_stations instead of on
    these canonical payloads.
    """

    asset_bays: list[AssetBay] = Field(default_factory=list)
    """The topology-owned asset bay payloads."""

    asset_setpoints: Optional[list[AssetSetpoint]] = None
    """ The list of asset setpoints in the topology. """

    timestamp: datetime
    """ The timestamp which is represented by this topology during the original optimization. I.e.
     if this timestep was the 5 o clock timestep on the day that was optimized, then this timestamp
      would read 5 o clock. """

    metrics: Optional[dict[str, float]] = None
    """ The metrics of the topology. """

    @field_validator("branch_assets")
    @classmethod
    def check_branch_asset_ids_unique(cls, v: list[BranchAsset]) -> list[BranchAsset]:
        """Check if all topology branch assets have unique grid model ids."""
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology branch assets")
        return v

    @field_validator("injection_assets")
    @classmethod
    def check_injection_asset_ids_unique(cls, v: list[InjectionAsset]) -> list[InjectionAsset]:
        """Check if all topology injection assets have unique grid model ids."""
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology injection assets")
        return v

    @field_validator("asset_bays")
    @classmethod
    def check_asset_bay_ids_unique(cls, v: list[AssetBay]) -> list[AssetBay]:
        """Check if all topology asset bay ids are unique."""
        asset_bay_ids = [asset_bay.asset_bay_id for asset_bay in v]
        if any(asset_bay_id is None for asset_bay_id in asset_bay_ids):
            raise ValueError("All topology asset bays must define asset_bay_id")
        if len(asset_bay_ids) != len(set(asset_bay_ids)):
            raise ValueError("asset_bay_id must be unique for topology asset bays")
        return v

    @model_validator(mode="after")
    def check_station_asset_references(self: "Topology") -> "Topology":
        """Check if all station asset references exist in the topology-owned collections."""
        branch_asset_ids = {asset.grid_model_id for asset in self.branch_assets}
        injection_asset_ids = {asset.grid_model_id for asset in self.injection_assets}
        asset_bay_ids = {asset_bay.asset_bay_id for asset_bay in self.asset_bays}

        for station in self.raw_stations:
            for asset_connection in station.branch_connections:
                asset_id = asset_connection.asset_id
                if asset_id not in branch_asset_ids:
                    raise ValueError(
                        f"Branch asset grid_model_id {asset_id} referenced by station "
                        f"{station.grid_model_id} does not exist in topology assets"
                    )
            for asset_connection in station.injection_connections:
                asset_id = asset_connection.asset_id
                if asset_id not in injection_asset_ids:
                    raise ValueError(
                        f"Injection asset grid_model_id {asset_id} referenced by station "
                        f"{station.grid_model_id} does not exist in topology assets"
                    )
            for asset_connection in [*station.branch_connections, *station.injection_connections]:
                asset_bay_id = asset_connection.asset_bay_id
                if asset_bay_id is not None and asset_bay_id not in asset_bay_ids:
                    raise ValueError(
                        f"asset_bay_id {asset_bay_id} referenced by station "
                        f"{station.grid_model_id} does not exist in topology asset bays"
                    )

        return self

    def materialize_stations(self) -> list[MaterializedStation]:
        """Materialize station-local asset payloads from topology-owned assets and asset bays."""
        branch_asset_map = {asset.grid_model_id: asset for asset in self.branch_assets}
        injection_asset_map = {asset.grid_model_id: asset for asset in self.injection_assets}
        asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in self.asset_bays}
        materialized_stations: list[MaterializedStation] = []

        for station in self.raw_stations:
            station_branch_assets = [
                branch_asset_map[asset_connection.asset_id].model_copy(deep=True)
                for asset_connection in station.branch_connections
            ]
            station_injection_assets = [
                injection_asset_map[asset_connection.asset_id].model_copy(deep=True)
                for asset_connection in station.injection_connections
            ]

            station_branch_asset_bays = [
                asset_bay_map[asset_connection.asset_bay_id].model_copy(deep=True)
                if asset_connection.asset_bay_id is not None
                else None
                for asset_connection in station.branch_connections
            ]
            station_injection_asset_bays = [
                asset_bay_map[asset_connection.asset_bay_id].model_copy(deep=True)
                if asset_connection.asset_bay_id is not None
                else None
                for asset_connection in station.injection_connections
            ]

            materialized_stations.append(
                MaterializedStation(
                    grid_model_id=station.grid_model_id,
                    name=station.name,
                    station_type=station.station_type,
                    region=station.region,
                    voltage_level=station.voltage_level,
                    busbars=station.busbars,
                    couplers=station.couplers,
                    branch_connections=[
                        MaterializedAssetConnection(
                            asset=asset,
                            terminal=asset_connection.terminal,
                            asset_bay=asset_bay,
                        )
                        for asset, asset_connection, asset_bay in zip(
                            station_branch_assets,
                            station.branch_connections,
                            station_branch_asset_bays,
                            strict=True,
                        )
                    ],
                    injection_connections=[
                        MaterializedAssetConnection(
                            asset=asset,
                            terminal=asset_connection.terminal,
                            asset_bay=asset_bay,
                        )
                        for asset, asset_connection, asset_bay in zip(
                            station_injection_assets,
                            station.injection_connections,
                            station_injection_asset_bays,
                            strict=True,
                        )
                    ],
                    branch_switching_table=station.branch_switching_table,
                    injection_switching_table=station.injection_switching_table,
                    branch_connectivity=station.branch_connectivity,
                    injection_connectivity=station.injection_connectivity,
                    model_log=station.model_log,
                )
            )

        return materialized_stations

    def get_asset_bay_ids_for_asset(self, asset_grid_model_id: str) -> list[str]:
        """Return all station-scoped asset bay ids connected to a topology asset.

        Parameters
        ----------
        asset_grid_model_id : str
            Grid model id of the topology-owned asset.

        Returns
        -------
        list[str]
            Ordered unique asset bay ids connected to the asset across all raw stations.
        """
        asset_bay_ids: list[str] = []
        seen_ids: set[str] = set()
        for station in self.raw_stations:
            for asset_connection in [*station.branch_connections, *station.injection_connections]:
                asset_id = asset_connection.asset_id
                asset_bay_id = asset_connection.asset_bay_id
                if asset_id != asset_grid_model_id or asset_bay_id is None or asset_bay_id in seen_ids:
                    continue
                seen_ids.add(asset_bay_id)
                asset_bay_ids.append(asset_bay_id)
        return asset_bay_ids

    def get_asset_bays_for_asset(self, asset_grid_model_id: str) -> list[AssetBay]:
        """Return all station-scoped asset bays connected to a topology asset.

        Parameters
        ----------
        asset_grid_model_id : str
            Grid model id of the topology-owned asset.

        Returns
        -------
        list[AssetBay]
            Ordered unique asset bay payloads connected to the asset across all raw stations.
        """
        asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in self.asset_bays}
        return [asset_bay_map[asset_bay_id] for asset_bay_id in self.get_asset_bay_ids_for_asset(asset_grid_model_id)]

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "Topology":
        """Copy and revalidate the topology."""
        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)


def copy_topology_with_updates(
    reference_topology: Topology,
    raw_stations: list[RawStation],
    asset_bays: list[AssetBay],
    *,
    branch_assets: Optional[list[SwitchableAsset]] = None,
    injection_assets: Optional[list[SwitchableAsset]] = None,
) -> Topology:
    """Create a validated topology copy with updated payloads.

    Parameters
    ----------
    reference_topology : Topology
        Reference topology providing shared metadata.
    raw_stations : list[RawStation]
        Raw stations to set on the copied topology.
    asset_bays : list[AssetBay]
        Topology-owned asset bays to set on the copied topology.
    branch_assets : Optional[list[SwitchableAsset]], optional
        Topology-owned branch assets to set on the copied topology.
    injection_assets : Optional[list[SwitchableAsset]], optional
        Topology-owned injection assets to set on the copied topology.

    Returns
    -------
    Topology
        Validated topology copy with updated topology-owned payloads.
    """
    resolved_branch_assets = branch_assets if branch_assets is not None else reference_topology.branch_assets
    resolved_injection_assets = injection_assets if injection_assets is not None else reference_topology.injection_assets

    return Topology(
        topology_id=reference_topology.topology_id,
        grid_model_file=reference_topology.grid_model_file,
        name=reference_topology.name,
        raw_stations=raw_stations,
        branch_assets=resolved_branch_assets,
        injection_assets=resolved_injection_assets,
        asset_bays=asset_bays,
        asset_setpoints=reference_topology.asset_setpoints,
        timestamp=reference_topology.timestamp,
        metrics=reference_topology.metrics,
    )


class Strategy(BaseModel):
    """Timestep data describing a collection of single timesteps, each represented by a Topology."""

    strategy_id: str
    """ The unique identifier of the strategy. """

    timesteps: list[Topology]
    """ The list of topologies, one for every timestep. """

    name: Optional[str] = None
    """ The name of the strategy. """

    author: Optional[str] = None
    """ The author of the strategy, i.e. who has created it. """

    process_type: Optional[str] = None
    """ The process type that created this topology, e.g. DC-solver, DC+-solver, Human etc. """

    process_parameters: Optional[dict[str, Union[str, float]]] = None
    """ The process parameters that were used to create this topology."""

    date_of_creation: Optional[datetime] = None
    """ The date of creation of this strategy, i.e. when the optimization ran. """

    metadata: Optional[dict[str, Any]] = None
    """ Additional metadata that might be useful for the strategy. """
