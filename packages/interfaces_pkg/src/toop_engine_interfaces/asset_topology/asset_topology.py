# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the data models for the asset topology."""

import numpy as np
from beartype.typing import Any, Iterator, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    AssetSetpoint,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
)
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
from toop_engine_interfaces.asset_topology.station_models import (
    StationAssetConnection,
    StationSwitchingArray,
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


class RuntimeAssetTopology(BaseModel):
    """Runtime topology payload grouped independently from canonical master data.

    The wrapper carries runtime station snapshots and optional runtime-visible
    circuit-group metadata aligned with the same topology view.
    """

    stations: list[MaterializedStation]
    """Runtime station snapshots for the topology view."""

    circuit_groups: Optional[list[CircuitGroup]] = None
    """Optional circuit-group metadata carried alongside the runtime stations."""

    @field_validator("stations")
    @classmethod
    def check_station_ids_unique(cls, v: list[MaterializedStation]) -> list[MaterializedStation]:
        """Validate uniqueness of runtime station identifiers.

        Parameters
        ----------
        v : list[MaterializedStation]
            Runtime stations assigned to the wrapper.

        Returns
        -------
        list[MaterializedStation]
            Validated runtime stations.
        """
        station_ids = [station.bus_group_id for station in v]
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("bus_group_id must be unique for runtime topology stations")
        return v


def _validate_master_station_connectivity(
    station_grid_model_id: str,
    station_name: Optional[str],
    busbar_count: int,
    asset_count: int,
    asset_connectivity: Optional[np.ndarray],
    asset_kind: str,
) -> None:
    """Validate master-station connectivity matrix dimensions.

    Parameters
    ----------
    station_grid_model_id : str
        Stable identifier of the station view.
    station_name : Optional[str]
        Human-readable station name used in validation errors.
    busbar_count : int
        Number of busbars owned by the station.
    asset_count : int
        Number of canonical asset references of the given kind.
    asset_connectivity : Optional[np.ndarray]
        Connectivity matrix to validate.
    asset_kind : str
        Asset kind label used in validation errors.

    Returns
    -------
    None
    """
    if asset_connectivity is not None and asset_connectivity.shape != (busbar_count, asset_count):
        raise ValueError(
            f"{asset_kind}_connectivity shape {asset_connectivity.shape} does not match busbars "
            f"{busbar_count} and {asset_kind} assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )


def iter_station_asset_references(
    stations: list[MaterializedStation],
) -> Iterator[tuple[str, Literal["branch", "injection"], str, str | None]]:
    """Yield normalized runtime station asset references.

    Parameters
    ----------
    stations : list[MaterializedStation]
        Runtime stations whose asset references should be iterated.

    Yields
    ------
    tuple[str, Literal["branch", "injection"], str, str | None]
        Station bus-group id, asset kind, referenced asset id, and optional asset-bay id.
    """
    for station in stations:
        for asset_connection in station.branch_connections:
            asset_bay = asset_connection.asset_bay
            yield (
                station.bus_group_id,
                "branch",
                asset_connection.asset.grid_model_id,
                asset_bay.asset_bay_id if asset_bay is not None else None,
            )
        for asset_connection in station.injection_connections:
            asset_bay = asset_connection.asset_bay
            yield (
                station.bus_group_id,
                "injection",
                asset_connection.asset.grid_model_id,
                asset_bay.asset_bay_id if asset_bay is not None else None,
            )


def _validate_station_asset_references(
    station_asset_references: Iterator[tuple[str, Literal["branch", "injection"], str, str | None]],
    branch_assets: list[BranchAsset],
    injection_assets: list[InjectionAsset],
    asset_bays: list[AssetBay],
) -> None:
    """Validate normalized station asset references against canonical topology collections.

    Parameters
    ----------
    station_asset_references : Iterator[tuple[str, Literal["branch", "injection"], str, str | None]]
        Normalized station references consisting of station id, asset kind, asset id,
        and optional asset-bay id.
    branch_assets : list[BranchAsset]
        Canonical branch assets available to the topology.
    injection_assets : list[InjectionAsset]
        Canonical injection assets available to the topology.
    asset_bays : list[AssetBay]
        Canonical asset bays available to the topology.

    Raises
    ------
    ValueError
        If a station reference points to a missing canonical asset or asset bay.
    """
    branch_asset_ids = {asset.grid_model_id for asset in branch_assets}
    injection_asset_ids = {asset.grid_model_id for asset in injection_assets}
    asset_bay_ids = {asset_bay.asset_bay_id for asset_bay in asset_bays}

    allowed_asset_ids = {
        "branch": branch_asset_ids,
        "injection": injection_asset_ids,
    }
    error_prefix = {
        "branch": "Branch",
        "injection": "Injection",
    }

    for station_id, asset_kind, asset_id, asset_bay_id in station_asset_references:
        if asset_id not in allowed_asset_ids[asset_kind]:
            raise ValueError(
                f"{error_prefix[asset_kind]} asset grid_model_id {asset_id} referenced by station "
                f"{station_id} does not exist in topology assets"
            )
        if asset_bay_id is not None and asset_bay_id not in asset_bay_ids:
            raise ValueError(
                f"asset_bay_id {asset_bay_id} referenced by station {station_id} does not exist in topology asset bays"
            )


def validate_runtime_station_asset_references(
    stations: list[MaterializedStation],
    branch_assets: list[BranchAsset],
    injection_assets: list[InjectionAsset],
    asset_bays: list[AssetBay],
) -> None:
    """Validate runtime station references against canonical topology payloads.

    Parameters
    ----------
    stations : list[MaterializedStation]
        Runtime stations to validate.
    branch_assets : list[BranchAsset]
        Canonical branch assets available to the topology.
    injection_assets : list[InjectionAsset]
        Canonical injection assets available to the topology.
    asset_bays : list[AssetBay]
        Canonical asset bays available to the topology.

    Returns
    -------
    None
    """
    _validate_station_asset_references(
        iter_station_asset_references(stations),
        branch_assets,
        injection_assets,
        asset_bays,
    )


def get_asset_bay_ids_for_asset(stations: list[MaterializedStation], asset_grid_model_id: str) -> list[str]:
    """Return ordered unique asset-bay ids for one asset.

    Parameters
    ----------
    stations : list[MaterializedStation]
        Runtime stations to scan for asset-bay references.
    asset_grid_model_id : str
        Grid-model id of the asset whose asset bays should be collected.

    Returns
    -------
    list[str]
        Ordered unique asset-bay ids referenced by the asset across the stations.
    """
    asset_bay_ids: list[str] = []
    seen_ids: set[str] = set()
    for _, _, asset_id, asset_bay_id in iter_station_asset_references(stations):
        if asset_id != asset_grid_model_id or asset_bay_id is None or asset_bay_id in seen_ids:
            continue
        seen_ids.add(asset_bay_id)
        asset_bay_ids.append(asset_bay_id)
    return asset_bay_ids


def get_asset_bays_for_asset(
    stations: list[MaterializedStation],
    asset_bays: list[AssetBay],
    asset_grid_model_id: str,
) -> list[AssetBay]:
    """Return ordered unique asset-bay payloads for one asset.

    Parameters
    ----------
    stations : list[MaterializedStation]
        Runtime stations to scan for asset-bay references.
    asset_bays : list[AssetBay]
        Canonical asset-bay payloads indexed by asset-bay id.
    asset_grid_model_id : str
        Grid-model id of the asset whose asset bays should be collected.

    Returns
    -------
    list[AssetBay]
        Ordered unique asset-bay payloads referenced by the asset across the stations.
    """
    asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in asset_bays}
    return [asset_bay_map[asset_bay_id] for asset_bay_id in get_asset_bay_ids_for_asset(stations, asset_grid_model_id)]


class MasterStation(BaseModel):
    """Canonical station master data without runtime switching state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bus_group_id: str
    """The unique identifier of the station view or bus group."""

    voltage_level_id: Optional[str] = None
    """The voltage level identifier backing this canonical station view."""

    name: Optional[str] = None
    """The name of the station."""

    station_type: Optional[str] = None
    """The type of the station."""

    region: Optional[str] = None
    """The region of the station."""

    voltage_level: Optional[float] = None
    """The voltage level of the station."""

    busbars: list[Busbar]
    """Canonical busbars owned by the station.

    Runtime outage state is stripped; all busbars are assumed in service in this model.
    """

    couplers: list[BusbarCoupler]
    """Canonical couplers owned by the station.

    Runtime switch state is stripped; all couplers are assumed closed and in service.
    """

    branch_connections: list[StationAssetConnection] = Field(default_factory=list)
    """Station-local canonical branch references aligned with ``branch_connectivity``."""

    injection_connections: list[StationAssetConnection] = Field(default_factory=list)
    """Station-local canonical injection references aligned with ``injection_connectivity``."""

    branch_connectivity: Optional[StationSwitchingArray] = None
    """Physically possible branch-to-busbar assignments for the station."""

    injection_connectivity: Optional[StationSwitchingArray] = None
    """Physically possible injection-to-busbar assignments for the station."""

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "MasterStation":
        """Copy and revalidate the station.

        Parameters
        ----------
        update : Optional[dict[str, Any]], optional
            Field updates to merge into the copied station.
        deep : bool, default=False
            Whether to deep-copy nested structures before validation.

        Returns
        -------
        MasterStation
            Copied and revalidated station instance.
        """
        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    @field_validator("branch_connectivity", "injection_connectivity", mode="before")
    @classmethod
    def normalize_connectivity_tables(cls, v: object | None) -> Optional[np.ndarray]:
        """Normalize connectivity table inputs to boolean arrays.

        Parameters
        ----------
        v : object | None
            Raw connectivity table input.

        Returns
        -------
        Optional[np.ndarray]
            Boolean connectivity table or ``None``.
        """
        if v is None:
            return None
        return np.asarray(v, dtype=bool)

    @field_validator("busbars")
    @classmethod
    def check_busbar_int_ids_unique(cls, v: list[Busbar]) -> list[Busbar]:
        """Validate that station busbar integer ids are unique.

        Parameters
        ----------
        v : list[Busbar]
            Busbars assigned to the station.

        Returns
        -------
        list[Busbar]
            Validated busbars.
        """
        int_ids = [busbar.int_id for busbar in v]
        if len(int_ids) != len(set(int_ids)):
            raise ValueError("busbar int_ids must be unique per station")
        return v

    @field_validator("couplers")
    @classmethod
    def check_coupler_busbars_different(cls, v: list[BusbarCoupler]) -> list[BusbarCoupler]:
        """Validate that couplers connect distinct busbars.

        Parameters
        ----------
        v : list[BusbarCoupler]
            Couplers assigned to the station.

        Returns
        -------
        list[BusbarCoupler]
            Validated couplers.
        """
        for coupler in v:
            if coupler.busbar_from_id == coupler.busbar_to_id:
                raise ValueError(f"Coupler {coupler.grid_model_id} connects the same busbar on both ends")
        return v

    @model_validator(mode="after")
    def check_coupler_busbars_exist(self: "MasterStation") -> "MasterStation":
        """Validate that all coupler busbar references exist on the station.

        Returns
        -------
        MasterStation
            Validated station instance.
        """
        busbar_ids = [busbar.int_id for busbar in self.busbars]
        for coupler in self.couplers:
            if coupler.busbar_from_id not in busbar_ids or coupler.busbar_to_id not in busbar_ids:
                raise ValueError(
                    f"Coupler {coupler.grid_model_id} references non-existing busbars"
                    f" Station_id: {self.bus_group_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def check_asset_reference_alignment(self: "MasterStation") -> "MasterStation":
        """Validate connectivity matrices against canonical asset references.

        Returns
        -------
        MasterStation
            Validated station instance.
        """
        _validate_master_station_connectivity(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_master_station_connectivity(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        return self


class TopologyMasterData(BaseModel):
    """Canonical grid master data without runtime switching or outage state."""

    topology_id: str
    """The unique identifier of the topology master data."""

    grid_model_file: Optional[str] = None
    """The source grid model file the master data was derived from."""

    name: Optional[str] = None
    """The name of the topology master data."""

    stations: list[MasterStation]
    """Canonical stations with asset references and physical connectivity only."""

    circuit_groups: Optional[list[CircuitGroup]] = None
    """Topology-owned circuit groups."""

    branch_assets: list[BranchAsset] = Field(default_factory=list)
    """The canonical branch master data payloads."""

    injection_assets: list[InjectionAsset] = Field(default_factory=list)
    """The canonical injection master data payloads."""

    asset_bays: list[AssetBay] = Field(default_factory=list)
    """The canonical asset-bay payloads."""

    asset_setpoints: Optional[list[AssetSetpoint]] = None
    """Optional topology-owned setpoint payloads."""

    @field_validator("stations")
    @classmethod
    def check_station_ids_unique(cls, v: list[MasterStation]) -> list[MasterStation]:
        """Validate uniqueness of canonical station identifiers.

        Parameters
        ----------
        v : list[MasterStation]
            Canonical stations assigned to the topology master data.

        Returns
        -------
        list[MasterStation]
            Validated canonical stations.
        """
        station_ids = [station.bus_group_id for station in v]
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("bus_group_id must be unique for topology master data stations")
        return v

    @field_validator("branch_assets")
    @classmethod
    def check_branch_asset_ids_unique(cls, v: list[BranchAsset]) -> list[BranchAsset]:
        """Validate uniqueness of canonical branch asset ids.

        Parameters
        ----------
        v : list[BranchAsset]
            Canonical branch assets.

        Returns
        -------
        list[BranchAsset]
            Validated branch assets.
        """
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology branch assets")
        return v

    @field_validator("injection_assets")
    @classmethod
    def check_injection_asset_ids_unique(cls, v: list[InjectionAsset]) -> list[InjectionAsset]:
        """Validate uniqueness of canonical injection asset ids.

        Parameters
        ----------
        v : list[InjectionAsset]
            Canonical injection assets.

        Returns
        -------
        list[InjectionAsset]
            Validated injection assets.
        """
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology injection assets")
        return v

    @field_validator("asset_bays")
    @classmethod
    def check_asset_bay_ids_unique(cls, v: list[AssetBay]) -> list[AssetBay]:
        """Validate uniqueness and presence of canonical asset-bay ids.

        Parameters
        ----------
        v : list[AssetBay]
            Canonical asset-bay payloads.

        Returns
        -------
        list[AssetBay]
            Validated asset-bay payloads.
        """
        asset_bay_ids = [asset_bay.asset_bay_id for asset_bay in v]
        if any(asset_bay_id is None for asset_bay_id in asset_bay_ids):
            raise ValueError("All topology asset bays must define asset_bay_id")
        if len(asset_bay_ids) != len(set(asset_bay_ids)):
            raise ValueError("asset_bay_id must be unique for topology asset bays")
        return v

    @model_validator(mode="after")
    def check_station_asset_references(self: "TopologyMasterData") -> "TopologyMasterData":
        """Validate station asset references against canonical topology collections.

        Returns
        -------
        TopologyMasterData
            Validated topology master data.
        """
        _validate_station_asset_references(
            (
                (station.bus_group_id, "branch", asset_connection.asset_id, asset_connection.asset_bay_id)
                for station in self.stations
                for asset_connection in station.branch_connections
            ),
            self.branch_assets,
            self.injection_assets,
            self.asset_bays,
        )
        _validate_station_asset_references(
            (
                (station.bus_group_id, "injection", asset_connection.asset_id, asset_connection.asset_bay_id)
                for station in self.stations
                for asset_connection in station.injection_connections
            ),
            self.branch_assets,
            self.injection_assets,
            self.asset_bays,
        )

        return self
