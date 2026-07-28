# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared station models and validators for asset topologies."""

from copy import deepcopy

import numpy as np
from beartype.typing import Any, Optional, TypeAlias
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from toop_engine_interfaces.asset_topology.asset_types import BranchEnd
from toop_engine_interfaces.asset_topology.assets import Busbar, BusbarCoupler

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

    branch_end: Optional[BranchEnd] = None
    """Optional branch-end metadata for this station-local asset occurrence."""

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

    bus_group_id: str
    """The unique identifier of the station view or bus group.

    This is a station-view identifier and may be synthetic.
    Runtime electrical bus ids are tracked separately on the busbars.

    Included are all assets, busbars and couplers that are connectable via switches.
    Buses in the same station that are connected via branches are excluded in this specific bus.

    This means, that two stations/buses can have the same elements if the station is currently split.
    """

    voltage_level_id: Optional[str] = None
    """Voltage level identifier backing this station view in the source grid."""

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

    bus_branch_bus_ids: list[str] = Field(default_factory=list)
    """Unique non-empty bus-branch bus ids currently represented by this station view."""

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
        v: object | None,
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

    @field_validator("bus_branch_bus_ids", mode="before")
    @classmethod
    def normalize_bus_branch_bus_ids(cls, v: Optional[list[str]]) -> list[str]:
        """Normalize explicit bus-branch bus ids to a unique sorted list."""
        if v is None:
            return []
        return sorted({bus_id for bus_id in v if bus_id not in {None, ""}})

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
                    f" Station_id: {self.bus_group_id}, Name: {self.name}"
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
                    f" Station_id: {self.bus_group_id}, Name: {self.name}"
                )
        return self

    @model_validator(mode="after")
    def sync_bus_branch_bus_ids(self: "_StationStructure") -> "_StationStructure":
        """Store the unique bus-branch bus ids contained in the station busbars."""
        bus_branch_bus_ids = sorted(
            {busbar.bus_branch_bus_id for busbar in self.busbars if busbar.bus_branch_bus_id not in {None, ""}}
        )
        self.bus_branch_bus_ids = bus_branch_bus_ids if bus_branch_bus_ids else self.bus_branch_bus_ids
        return self

    def is_split(self) -> bool:
        """Return whether the station view spans more than one non-empty bus-branch bus id."""
        return len(self.bus_branch_bus_ids) > 1
