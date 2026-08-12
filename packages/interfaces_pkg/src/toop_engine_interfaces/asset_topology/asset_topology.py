# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the data models for the asset topology."""

from copy import deepcopy

import numpy as np
from beartype.typing import Any, Iterator, Literal, Optional, TypeAlias
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology.asset_types import BranchEnd
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    AssetSetpoint,
    BranchAsset,
    Busbar,
    BusbarCoupler,
    InjectionAsset,
)

BusGroupSwitchingArray: TypeAlias = NDArray[Shape["* n_bus, * n_asset"], np.bool_]


def _merged_round_trip_payload(model: BaseModel, update: Optional[dict[str, Any]], *, deep: bool = False) -> dict[str, Any]:
    """Merge model field values and requested updates for revalidation-aware model_copy overrides."""
    payload = {field_name: getattr(model, field_name) for field_name in type(model).model_fields}
    if deep:
        payload = deepcopy(payload)
    if update:
        payload.update(update)
    return payload


class CircuitGroup(BaseModel):
    """A circuit group represents assets connected without power switches.

    All assets inside the same circuit group are treated as jointly outaged.
    """

    asset_ids: list[str]
    """Grid-model ids of the assets contained in the circuit group."""

    asset_bay_ids: list[str]
    """Asset-bay ids whose switches implement the circuit-group outage effect."""


class BusGroupAssetConnection(BaseModel):
    """Station-local association between a switching-table column and a topology asset."""

    asset_id: str
    """Grid model id of the topology-owned asset referenced by this station-local column."""

    branch_end: Optional[BranchEnd] = None
    """Optional branch-end metadata for this station-local asset occurrence."""

    asset_bay_id: Optional[str] = None
    """Optional topology-scoped asset bay identifier for this station-local asset occurrence."""


def _validate_master_bus_group_connectivity(
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


class MasterBusGroup(BaseModel):
    """Canonical station master data without runtime switching state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bus_group_id: str
    """The unique identifier of the station view or bus group."""  # TODO comment structure

    voltage_level_id: Optional[str] = None
    """The voltage level identifier backing this canonical station view."""

    name: Optional[str] = None
    """The name of the station."""  # ADD SUFFIX

    station_type: Optional[str] = None
    """The type of the station."""  # ENUM BUSBREAKER NODE

    region: Optional[str] = None
    """The region of the station."""

    voltage_level: Optional[float] = None
    """The voltage level of the station in kV."""

    busbars: list[Busbar]
    """Canonical busbars owned by the station.

    Runtime outage state is stripped; all busbars are assumed in service in this model.
    """

    couplers: list[BusbarCoupler]
    """Canonical couplers owned by the station.

    Runtime switch state is stripped; all couplers are assumed closed and in service.
    """

    branch_connections: list[BusGroupAssetConnection] = Field(default_factory=list)
    """Station-local canonical branch references aligned with ``branch_connectivity``."""

    injection_connections: list[BusGroupAssetConnection] = Field(default_factory=list)
    """Station-local canonical injection references aligned with ``injection_connectivity``."""

    branch_connectivity: Optional[BusGroupSwitchingArray] = None
    """Physically possible branch-to-busbar assignments for the station."""

    injection_connectivity: Optional[BusGroupSwitchingArray] = None
    """Physically possible injection-to-busbar assignments for the station."""

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "MasterBusGroup":
        """Copy and revalidate the station.

        Parameters
        ----------
        update : Optional[dict[str, Any]], optional
            Field updates to merge into the copied station.
        deep : bool, default=False
            Whether to deep-copy nested structures before validation.

        Returns
        -------
        MasterBusGroup
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

    @model_validator(mode="after")
    def check_asset_reference_alignment(self: "MasterBusGroup") -> "MasterBusGroup":
        """Validate connectivity matrices against canonical asset references.

        Returns
        -------
        MasterBusGroup
            Validated station instance.
        """
        _validate_master_bus_group_connectivity(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_master_bus_group_connectivity(
            station_grid_model_id=self.bus_group_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        return self


class MasterAssetTopology(BaseModel):
    """Canonical grid master data without runtime switching or outage state."""

    topology_id: str
    """The unique identifier of the topology master data."""

    grid_model_file: Optional[str] = None
    """The source grid model file the master data was derived from."""

    name: Optional[str] = None
    """The name of the topology master data."""

    bus_groups: list[MasterBusGroup]
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

    @field_validator("bus_groups")
    @classmethod
    def check_bus_group_ids_unique(cls, v: list[MasterBusGroup]) -> list[MasterBusGroup]:
        """Validate uniqueness of canonical bus-group identifiers.

        Parameters
        ----------
        v : list[MasterBusGroup]
            Canonical bus groups assigned to the topology master data.

        Returns
        -------
        list[MasterBusGroup]
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
    def check_station_asset_references(self: "MasterAssetTopology") -> "MasterAssetTopology":
        """Validate station asset references against canonical topology collections.

        Returns
        -------
        MasterAssetTopology
            Validated topology master data.
        """
        _validate_station_asset_references(
            (
                (station.bus_group_id, "branch", asset_connection.asset_id, asset_connection.asset_bay_id)
                for station in self.bus_groups
                for asset_connection in station.branch_connections
            ),
            self.branch_assets,
            self.injection_assets,
            self.asset_bays,
        )
        _validate_station_asset_references(
            (
                (station.bus_group_id, "injection", asset_connection.asset_id, asset_connection.asset_bay_id)
                for station in self.bus_groups
                for asset_connection in station.injection_connections
            ),
            self.branch_assets,
            self.injection_assets,
            self.asset_bays,
        )

        return self
