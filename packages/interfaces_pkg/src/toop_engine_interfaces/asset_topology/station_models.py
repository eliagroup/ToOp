# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared station models for raw and materialized asset topologies."""

from copy import deepcopy

import numpy as np
from beartype.typing import Any, Literal, Optional, TypeAlias
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology.asset_types import BranchEnd
from toop_engine_interfaces.asset_topology.assets import Busbar, BusbarCoupler, SwitchableAsset

StationSwitchingArray: TypeAlias = NDArray[Shape["* n_bus, * n_asset"], np.bool_]


def _merged_round_trip_payload(model: BaseModel, update: Optional[dict[str, Any]], *, deep: bool = False) -> dict[str, Any]:
    """Merge model field values and requested updates for revalidation-aware model_copy overrides."""
    payload = {field_name: getattr(model, field_name) for field_name in type(model).model_fields}
    if deep:
        payload = deepcopy(payload)
    if update:
        payload.update(update)
    return payload


class StationAssetConnection(BaseModel):
    """Station-local association between a switching-table column and a topology asset."""

    asset_id: str
    """Grid model id of the topology-owned asset referenced by this station-local column."""

    terminal: Optional[BranchEnd] = None
    """Optional branch terminal metadata for this station-local asset occurrence."""

    asset_bay_id: Optional[str] = None
    """Optional topology-scoped asset bay identifier for this station-local asset occurrence."""


def _validate_station_switching_tables(
    station_grid_model_id: str,
    station_name: Optional[str],
    busbar_count: int,
    asset_count: int,
    asset_switching_table: np.ndarray,
    asset_connectivity: Optional[np.ndarray],
    asset_kind: str,
) -> None:
    """Validate switching-table shapes against the station dimensions.

    Parameters
    ----------
    station_grid_model_id : str
        Grid model id of the station being validated.
    station_name : Optional[str]
        Human-readable station name used in validation errors.
    busbar_count : int
        Expected number of busbar rows in the switching tables.
    asset_count : int
        Expected number of asset columns in the switching tables.
    asset_switching_table : np.ndarray
        Current station switching table.
    asset_connectivity : Optional[np.ndarray]
        Optional connectivity mask describing physically allowed assignments.
    asset_kind: str
        The kind of asset being validated, used in error messages (e.g. "branch" or "injection").

    Returns
    -------
    None
        This function returns nothing and raises on invalid shapes.

    Raises
    ------
    ValueError
        If either switching table does not match the expected station dimensions.
    """
    if asset_switching_table.shape != (busbar_count, asset_count):
        raise ValueError(
            f"{asset_kind}_switching_table shape {asset_switching_table.shape} does not match busbars "
            f"{busbar_count} and {asset_kind} assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )

    if asset_connectivity is not None and asset_connectivity.shape != (busbar_count, asset_count):
        raise ValueError(
            f"{asset_kind}_connectivity shape {asset_connectivity.shape} does not match busbars "
            f"{busbar_count} and {asset_kind} assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )


def _validate_station_physical_assignments(
    station_grid_model_id: str,
    station_name: Optional[str],
    asset_switching_table: np.ndarray,
    asset_connectivity: Optional[np.ndarray],
    asset_kind: str,
) -> None:
    """Validate that all current assignments are physically allowed.

    Parameters
    ----------
    station_grid_model_id : str
        Grid model id of the station being validated.
    station_name : Optional[str]
        Human-readable station name used in validation errors.
    asset_switching_table : np.ndarray
        Current station switching table.
    asset_connectivity : Optional[np.ndarray]
        Optional connectivity mask describing physically allowed assignments.
    asset_kind: str
        The kind of asset being validated, used in error messages (e.g. "branch" or "injection").

    Returns
    -------
    None
        This function returns nothing and raises on invalid assignments.

    Raises
    ------
    ValueError
        If the switching table contains assignments forbidden by ``asset_connectivity``.
    """
    if asset_connectivity is not None:
        if np.logical_and(asset_switching_table, np.logical_not(asset_connectivity)).any():
            raise ValueError(
                f"Not all current {asset_kind} assignments are physically allowed "
                f"Station_id: {station_grid_model_id}, Name: {station_name}"
            )


class _StationStructure(BaseModel):
    """Shared station fields and structural validators for station views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_model_id: str
    """The unique identifier of the station.

    Expects the bus-branch model bus_id, not the full voltage level id.

    Included are all assets, busbars and couplers that are connectable via switches.
    Buses in the same station that are connected via branches are excluded in this specific bus.

    This means, that two stations/buses can have the same elements if the station is currently split.
    """

    name: Optional[str] = None
    """The name of the station."""

    station_type: Optional[str] = None
    """The type of the station."""

    region: Optional[str] = None
    """The region of the station."""

    voltage_level: Optional[float] = None
    """The voltage level of the station."""

    busbars: list[Busbar]
    """The list of busbars at the station."""

    couplers: list[BusbarCoupler]
    """The list of couplers at the station."""

    branch_switching_table: StationSwitchingArray
    """Holds the switching of each branch asset to each busbar, shape (n_bus, n_branch_asset).

    An entry is true if the asset is connected to the busbar.
    Note: An asset can be connected to multiple busbars, in which case a closed coupler is assumed
    to be present between these busbars.
    Note: An asset can be connected to none of the busbars. In this case, the asset is intentionally
    disconnected as part of a transmission line switching action. In practice, this usually involves
    a separate switch from the asset-to-busbar couplers, as each asset usually has a switch that
    completely disconnects it from the station. These switches are not modelled here, a
    postprocessing routine needs to do the translation to this physical layout. Do not use
    in_service for intentional disconnections.
    """

    injection_switching_table: StationSwitchingArray
    """Holds the switching of each injection asset to each busbar, shape (n_bus, n_injection_asset)."""

    branch_connectivity: Optional[StationSwitchingArray] = None
    """Holds all physically possible branch layouts, shape (n_bus, n_branch_asset)."""

    injection_connectivity: Optional[StationSwitchingArray] = None
    """Holds all physically possible injection layouts, shape (n_bus, n_injection_asset)."""

    model_log: Optional[list[str]] = None
    """Holds log messages from the model creation process.

    This can be used to store information about the model creation process, e.g. warnings or errors.
    A potential use case is to inform the user about data quality issues e.g. missing the Asset Bay switches.
    """

    @field_validator(
        "branch_switching_table",
        "injection_switching_table",
        "branch_connectivity",
        "injection_connectivity",
        mode="before",
    )
    @classmethod
    def normalize_station_tables(
        cls,
        v: Optional[Any],  # noqa: ANN401
    ) -> Optional[np.ndarray]:
        """Normalize switching and connectivity table inputs to boolean arrays."""
        if v is None:
            return None
        return np.asarray(v, dtype=bool)

    @field_validator("busbars")
    @classmethod
    def check_int_id_unique(cls, v: list[Busbar]) -> list[Busbar]:
        """Check if the int_ids of the busbars are unique."""
        int_ids = [busbar.int_id for busbar in v]
        if len(int_ids) != len(set(int_ids)):
            raise ValueError("busbar int_ids must be unique per station")
        return v

    @field_validator("couplers")
    @classmethod
    def check_coupler_busbars_different(cls, v: list[BusbarCoupler]) -> list[BusbarCoupler]:
        """Check if the couplers do not connect the same busbar on both ends."""
        for coupler in v:
            if coupler.busbar_from_id == coupler.busbar_to_id:
                raise ValueError(f"Coupler {coupler.grid_model_id} connects the same busbar on both ends")
        return v

    @model_validator(mode="after")
    def check_coupler_busbars_exist(self: "_StationStructure") -> "_StationStructure":
        """Check if all coupler busbar ids exist in busbars."""
        busbar_ids = [busbar.int_id for busbar in self.busbars]
        for coupler in self.couplers:
            if coupler.busbar_from_id not in busbar_ids or coupler.busbar_to_id not in busbar_ids:
                raise ValueError(
                    f"Coupler {coupler.grid_model_id} references non-existing busbars"
                    f" Station_id: {self.grid_model_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def check_coupler_references(self: "_StationStructure") -> "_StationStructure":
        """Check if all closed couplers reference in-service busbars."""
        busbar_state_map = {busbar.int_id: busbar.in_service for busbar in self.busbars}
        for coupler in self.couplers:
            if coupler.open or not coupler.in_service:
                continue
            if busbar_state_map[coupler.busbar_from_id] != busbar_state_map[coupler.busbar_to_id]:
                raise ValueError(
                    f"Closed coupler {coupler.grid_model_id} connects out-of-service busbar with in-service busbar."
                    f" Station_id: {self.grid_model_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def check_bus_id(self: "_StationStructure") -> "_StationStructure":
        """Check if station grid_model_id is in the busbar.bus_branch_bus_id."""
        busbar_grid_model_id = [busbar.bus_branch_bus_id for busbar in self.busbars if busbar.bus_branch_bus_id is not None]
        if len(busbar_grid_model_id) > 0 and self.grid_model_id not in busbar_grid_model_id:
            raise ValueError(
                f"Station grid_model_id {self.grid_model_id} does not exist in busbars bus_branch_bus_id"
                f" Station_id: {self.grid_model_id}, Name: {self.name}"
            )

        return self

    def is_split(self) -> bool:
        """Return whether the station view spans more than one non-empty bus-branch bus id."""
        bus_ids = {busbar.bus_branch_bus_id for busbar in self.busbars if busbar.bus_branch_bus_id not in {None, ""}}
        return len(bus_ids) > 1


class RawStation(_StationStructure):
    """Station data stored inside a topology without embedded asset payloads.

    The station identity still refers to a bus-branch model bus_id for one splitable station view.
    Asset membership is expressed through the aligned station-local arrays instead of embedded
    SwitchableAsset payloads.
    """

    branch_connections: list[StationAssetConnection] = Field(default_factory=list)
    """Station-local branch references aligned with ``branch_switching_table``."""

    injection_connections: list[StationAssetConnection] = Field(default_factory=list)
    """Station-local injection references aligned with ``injection_switching_table``."""

    def with_asset_terminals(self, asset_terminals: list[Optional[BranchEnd]]) -> "RawStation":
        """Return a copy with updated branch terminals."""
        if len(asset_terminals) != len(self.branch_connections):
            raise ValueError(
                f"asset_terminals length {len(asset_terminals)} does not match branch_connections length "
                f"{len(self.branch_connections)}"
                f" Station_id: {self.grid_model_id}, Name: {self.name}"
            )

        return self.model_copy(
            update={
                "branch_connections": [
                    asset_connection.model_copy(update={"terminal": asset_terminal})
                    for asset_connection, asset_terminal in zip(self.branch_connections, asset_terminals, strict=True)
                ]
            }
        )

    @model_validator(mode="after")
    def check_asset_reference_alignment(self: "RawStation") -> "RawStation":
        """Check if station-local asset reference arrays are aligned."""
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.branch_connections),
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.branch_switching_table,
            asset_connectivity=self.branch_connectivity,
            asset_kind="branch",
        )
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.injection_connections),
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.injection_switching_table,
            asset_connectivity=self.injection_connectivity,
            asset_kind="injection",
        )
        return self

    def __eq__(self, other: object) -> bool:
        """Check if two topology stations are equal."""
        if not isinstance(other, RawStation):
            return False
        return (
            self.grid_model_id == other.grid_model_id
            and self.name == other.name
            and self.station_type == other.station_type
            and self.region == other.region
            and self.voltage_level == other.voltage_level
            and self.busbars == other.busbars
            and self.couplers == other.couplers
            and self.branch_connections == other.branch_connections
            and self.injection_connections == other.injection_connections
            and np.array_equal(self.branch_switching_table, other.branch_switching_table)
            and np.array_equal(self.injection_switching_table, other.injection_switching_table)
            and (
                np.array_equal(self.branch_connectivity, other.branch_connectivity)
                if (self.branch_connectivity is not None and other.branch_connectivity is not None)
                else self.branch_connectivity == other.branch_connectivity
            )
            and (
                np.array_equal(self.injection_connectivity, other.injection_connectivity)
                if (self.injection_connectivity is not None and other.injection_connectivity is not None)
                else self.injection_connectivity == other.injection_connectivity
            )
            and self.model_log == other.model_log
        )

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "RawStation":
        """Copy and revalidate the station."""
        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    def get_connected_assets(
        self,
        busbar_index: int,
        topology_assets: Optional[list[SwitchableAsset]] = None,
        asset_scope: Literal["all", "branch", "injection"] = "all",
    ) -> list[SwitchableAsset]:
        """Return in-service topology assets connected to one busbar.

        Parameters
        ----------
        busbar_index : int
            Row index into the station switching tables.
        topology_assets : Optional[list[SwitchableAsset]]
            Topology-owned assets used to resolve raw station asset references.
        asset_scope : Literal["all", "branch", "injection"]
            Restrict the lookup to branch or injection connections.

        Returns
        -------
        list[SwitchableAsset]
            Connected in-service assets for the requested busbar and scope.

        Raises
        ------
        ValueError
            If topology assets are missing for a raw station lookup.
        """
        if topology_assets is None:
            raise ValueError("topology_assets must be provided when resolving connected assets for a RawStation")

        asset_map = {asset.grid_model_id: asset for asset in topology_assets}
        if asset_scope == "branch":
            return [
                asset_map[asset_connection.asset_id]
                for asset_connection, is_connected in zip(
                    self.branch_connections,
                    self.branch_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_map[asset_connection.asset_id].in_service
            ]
        if asset_scope == "injection":
            return [
                asset_map[asset_connection.asset_id]
                for asset_connection, is_connected in zip(
                    self.injection_connections,
                    self.injection_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_map[asset_connection.asset_id].in_service
            ]
        return [
            *self.get_connected_assets(busbar_index, topology_assets=topology_assets, asset_scope="branch"),
            *self.get_connected_assets(busbar_index, topology_assets=topology_assets, asset_scope="injection"),
        ]
