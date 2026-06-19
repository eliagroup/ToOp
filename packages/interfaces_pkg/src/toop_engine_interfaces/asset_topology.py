# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the data models for the asset topology."""

from copy import deepcopy
from datetime import datetime
from enum import Enum

import numpy as np
from beartype.typing import Any, Collection, Literal, Optional, TypeAlias, Union, get_args
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StationSwitchingArray: TypeAlias = NDArray[Shape["* n_bus, * n_asset"], np.bool_]


def _merged_round_trip_payload(model: BaseModel, update: Optional[dict[str, Any]], *, deep: bool = False) -> dict[str, Any]:
    """Merge model field values and requested updates for revalidation-aware model_copy overrides."""
    payload = {field_name: getattr(model, field_name) for field_name in type(model).model_fields}
    if deep:
        payload = deepcopy(payload)
    if update:
        payload.update(update)
    return payload


class PowsyblSwitchValues(Enum):
    """Enum for the switch values in the Powsybl model."""

    OPEN = True
    """ The switch is open, i.e. not connected."""
    CLOSED = False
    """ The switch is closed, i.e. connected."""


BranchEnd: TypeAlias = Literal["from", "to", "hv", "mv", "lv"]
AssetBranchTypePandapower: TypeAlias = Literal[
    "line",
    "trafo",
    "trafo3w_lv",
    "trafo3w_mv",
    "trafo3w_hv",
    "impedance",
    "xward",
]
AssetBranchTypePowsybl: TypeAlias = Literal[
    "LINE",
    "TWO_WINDINGS_TRANSFORMER",
    "TIE_LINE",
]
AssetBranchType: TypeAlias = Literal[AssetBranchTypePandapower, AssetBranchTypePowsybl]

AssetInjectionTypePandapower: TypeAlias = Literal[
    "ext_grid",
    "gen",
    "load",
    "shunt",
    "sgen",
    "ward",
    "ward_load",
    "ward_shunt",
    "xward_load",
    "xward_shunt",
    "dcline_from",
    "dcline_to",
]
AssetInjectionTypePowsybl: TypeAlias = Literal[
    "LOAD",
    "GENERATOR",
    "BOUNDARY_LINE",
    "HVDC_CONVERTER_STATION",
    "STATIC_VAR_COMPENSATOR",
    "SHUNT_COMPENSATOR",
    "BATTERY",
]
AssetInjectionType: TypeAlias = Literal[AssetInjectionTypePandapower, AssetInjectionTypePowsybl]
AssetType: TypeAlias = Literal[AssetBranchType, AssetInjectionType]


class Busbar(BaseModel):
    """Busbar data describing a single busbar a station."""

    grid_model_id: str
    """ The unique identifier of the busbar.
    Corresponds to the busbar's id in the grid model."""

    busbar_type: Optional[str] = None
    """ The type of the busbar, might be useful for finding the busbar later on """

    name: Optional[str] = None
    """ The name of the busbar, might be useful for finding the busbar later on """

    int_id: int
    """ Is used to reference busbars in the couplers. Needs to be unique per station"""

    in_service: bool = True
    """ Whether the busbar is in service. If False, it will be ignored in the switching table"""

    bus_branch_bus_id: Optional[str] = None
    """ The bus_branch_bus_id refers to the bus-branch model bus id.
    There might be a difference between the busbar grid_model_id (a physical busbar)
    and the bus_branch_bus_id from the bus-branch model.
    Use this bus_branch_bus_id to store the bus-branch model bus id.
    Note: the Station grid_model_id also a bus-branch bus_branch_bus_id. This id is the most splitable bus_branch_bus_id.
    Other bus_branch_bus_ids are part of the physical station, but are separated by a coupler or branch."""


class AssetBay(BaseModel):
    """Saves the physical connection from the asset to the substation busbars - a bay (Schaltfeld).

    A line usually has three switches, before it is connected to the busbar.
    Two disconnector switches and one circuit breaker switch.
    A transformer usually has two switches, before it is connected to the busbar.
    One disconnector switch and one circuit breaker switch.

    type: n - node
    type: b - busbar (Sammelschiene)
    type: CB - DV Circuit Breaker / Power Switch (Leistungsschalter)
    type: DS - Disconnector Switch (Trennschalter)

    ------------------ busbar 1 - type: b
          |
          /  type: DS - SR Switch busbar 1   -> used for reassigning the asset to another busbar
          |
    ------|----------- busbar 2 - type: b
      |   |
      /   |  type: DS - SR Switch busbar 2   -> used for reassigning the asset to another busbar
      |   |
    --------- bus_3 - type: n - busbar section bus
        |
        /    type: CB - DV Circuit Breaker / Power Switch -> used for disconnecting the asset from the busbar
        |
    --------- bus_2 - type: n - circuit breaker bus
        |
        /    type: DS - SL Switch (optional) -> not used by the asset
        |
    --------- bus_1 - type: n - asset bus
        ^
        |       Line/Transformer


    """

    asset_bay_id: str
    """Topology-scoped identifier for the asset bay."""

    sl_switch_grid_model_id: Optional[str] = None
    """ The id of the switch, which connects the asset to the circuit breaker node.
    This switch is a disconnector switch. Do not use for anything, leave state as found.
    Default should be closed."""

    dv_switch_grid_model_id: str
    """ This switch is a circuit breaker / power switch.
    Use for disconnecting / reconnecting the asset from the busbar. """

    sr_switch_grid_model_id: dict[str, str]
    """ The ids of the switches, which assign the asset to the busbars.
    key: busbar_grid_model_id e.g. 4%%bus
    value: sr_switch_grid_model_id
    This switch is a disconnector switch. Use for reassigning the asset to another busbar.
    Only one switch should be closed at a time.
    """

    @field_validator("sr_switch_grid_model_id")
    @classmethod
    def check_is_empty(cls, v: dict[str, str]) -> dict[str, str]:
        """Check if the dict is empty.

        Parameters
        ----------
        v : dict[str, str]
            The dictionary of sr_switch_grid_model_id to check.

        Returns
        -------
        dict[str, str]
            The dictionary itself.

        Raises
        ------
        ValueError
            If the dictionary is empty.
        """
        if len(v) == 0:
            raise ValueError("sr_switch_grid_model_id must not be empty")
        return v


class BusbarCoupler(BaseModel):
    """Coupler data describing a single coupler at a station.

    This references only busbar couplers, i.e. couplers connecting two busbars.
    Switches connecting assets to a busbar are represented in the asset_switching_table in the station model.

    Note: A busbar couple is a physical connection between two busbars, this can be also a
    cross coupler. To further specify the connection of an asset to a busbar, the asset connection
    """

    grid_model_id: str
    """ The unique identifier of the coupler.
    Corresponds to the coupler's id in the grid model."""

    coupler_type: Optional[str] = None
    """ The type of the coupler, might be useful for finding the coupler later on """

    name: Optional[str] = None
    """ The name of the coupler, might be useful for finding the coupler later on """

    # TODO: this does not work for a coupler with multiple busbars on one side
    busbar_from_id: int
    """ Is used to determine where the coupler is connected to the busbars on the "from" side.
    Refers to the int_id of the busbar"""

    # TODO: this does not work for a coupler with multiple busbars on one side
    busbar_to_id: int
    """ Is used to determine where the coupler is connected to the busbars on the "to" side.
    Refers to the int_id of the busbar"""

    open: bool
    """ The status of the coupler. True if the coupler is open, False if the coupler is closed.
    TODO: Switch to using the connectivity table instead of this field.
    """

    in_service: bool = True
    """ Whether the coupler is in-service. Out-of-service couplers are assumed to be always open"""

    asset_bay: Optional[AssetBay] = None
    """ The asset bay (Schaltfeld) of the coupler.
    Note: A coupler can have multiple from and to busbars.
    The asset bay sr_switch_grid_model_id is used save the selector switches of the coupler.
    Note: A coupler has never a sl_switch_grid_model_id, the dv_switch_grid_model_id should
    the same as the name of the coupler.

    """


class SwitchableAsset(BaseModel):
    """Asset data describing a single asset at a station.

    An asset can be for instance a transformer, line, generator, load, shunt.
    Note: An asset can be connected to multiple busbars through the switching grid, however if
    this happens a closed coupler between these busbars is assumed. If such couplers are not present,
    they will be created.
    Note: An asset that is out-of-service can be represented, but its switching entries will be
    ignored.
    """

    grid_model_id: str
    """ The unique identifier of the asset.
    Corresponds to the asset's id in the grid model."""

    asset_type: Optional[AssetType] = None
    """ The type of the asset. These refer loosely to the types in the pandapower/powsybl grid
    models. If set, this can be used to disambiguate branches from injections """

    name: Optional[str] = None
    """ The name of the asset, might be useful for finding the asset later on """

    in_service: bool = True
    """ If the element is in service. False means the switching entry for this element will be
    ignored. This shall not be used for elements intentionally disconnected, instead set all zeros
    in the switching table."""

    def is_branch(self) -> Optional[bool]:
        """Return True if the asset is a branch.

        Only works if the type is set. If type is not set this will return None.

        Returns
        -------
        bool
            True if the asset is a branch, False if it is an injection.
        """
        if self.asset_type is None:
            return None
        return self.asset_type in get_args(AssetBranchType)


class BranchAsset(SwitchableAsset):
    """Switchable asset representing a branch-type element."""

    asset_type: Optional[AssetBranchType] = None

    def is_branch(self) -> Optional[bool]:
        """Return branch semantics for branch assets."""
        return None if self.asset_type is None else True


class InjectionAsset(SwitchableAsset):
    """Switchable asset representing an injection-type element."""

    asset_type: Optional[AssetInjectionType] = None

    def is_branch(self) -> Optional[bool]:
        """Return branch semantics for injection assets."""
        return None if self.asset_type is None else False


def normalize_switchable_asset_payload(asset: dict[str, Any]) -> SwitchableAsset:
    """Normalize an asset payload to the matching branch or injection subclass when possible."""
    if isinstance(asset, (BranchAsset, InjectionAsset)):
        return asset

    asset_data = asset.model_dump() if isinstance(asset, SwitchableAsset) else dict(asset)
    if "asset_type" not in asset_data and "type" in asset_data:
        asset_data["asset_type"] = asset_data.pop("type")
    asset_type = asset_data.get("asset_type")
    if asset_type in get_args(AssetBranchType):
        return BranchAsset(**asset_data)
    if asset_type in get_args(AssetInjectionType):
        return InjectionAsset(**asset_data)
    return SwitchableAsset(**asset_data)


class StationAssetConnection(BaseModel):
    """Station-local association between a switching-table column and a topology asset."""

    asset_id: str
    """Grid model id of the topology-owned asset referenced by this station-local column."""

    terminal: Optional[BranchEnd] = None
    """Optional branch terminal metadata for this station-local asset occurrence."""

    asset_bay_id: Optional[str] = None
    """Optional topology-scoped asset bay identifier for this station-local asset occurrence."""


class MaterializedAssetConnection(BaseModel):
    """Station-local association between a switching-table column and a materialized asset payload."""

    asset: SwitchableAsset
    """Station-local asset payload aligned with one switching-table column."""

    terminal: Optional[BranchEnd] = None
    """Optional branch terminal metadata for this station-local asset occurrence."""

    asset_bay: Optional[AssetBay] = None
    """Optional station-local asset bay payload for this station-local asset occurrence."""

    def get_sr_switch(self) -> Optional[dict[str, str]]:
        """Return the sr_switch_grid_model_id dict from the asset bay if it exists."""
        if self.asset_bay is not None:
            return self.asset_bay.sr_switch_grid_model_id
        return None


class AssetSetpoint(BaseModel):
    """Asset data describing a single asset with a setpoint.

    This could for example be a PST or HVDC setpoint.
    Note: The same asset can both be switchable and have a setpoint. In this case, the asset will
    be represented twice.
    """

    grid_model_id: str
    """ The unique identifier of the asset.
    Corresponds to the asset's id in the grid model."""

    asset_type: Optional[str] = None
    """ The type of the asset, might be useful for finding the asset later on """

    name: Optional[str] = None
    """ The name of the asset, might be useful for finding the asset later on """

    setpoint: float
    """ The setpoint of the asset. """


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
    def normalize_station_tables(cls, v: Optional[Collection]) -> Optional[np.ndarray]:
        """Normalize split station tables to boolean numpy arrays."""
        if v is None:
            return None
        return np.asarray(v, dtype=bool)

    @field_validator("busbars")
    @classmethod
    def check_int_id_unique(cls, v: list[Busbar]) -> list[Busbar]:
        """Check if int_id is unique for all busbars."""
        int_ids = [busbar.int_id for busbar in v]
        if len(int_ids) != len(set(int_ids)):
            raise ValueError("int_id must be unique for busbars")
        return v

    @field_validator("couplers")
    @classmethod
    def check_coupler_busbars_different(cls, v: list[BusbarCoupler]) -> list[BusbarCoupler]:
        """Check if busbar_from_id and busbar_to_id are different for all couplers."""
        for coupler in v:
            if coupler.busbar_from_id == coupler.busbar_to_id:
                raise ValueError(f"busbar_from_id and busbar_to_id must be different for coupler {coupler.grid_model_id}")
        return v

    @model_validator(mode="after")
    def check_busbar_exists(self: "_StationStructure") -> "_StationStructure":
        """Check if all busbars in couplers exist in the busbars list."""
        busbar_ids = [busbar.int_id for busbar in self.busbars]
        for coupler in self.couplers:
            if coupler.busbar_from_id not in busbar_ids:
                raise ValueError(
                    f"busbar_from_id {coupler.busbar_from_id} in coupler {coupler.grid_model_id} does not exist in busbars."
                    f" Station_id: {self.grid_model_id}, Name: {self.name}"
                )
            if coupler.busbar_to_id not in busbar_ids:
                raise ValueError(
                    f"busbar_to_id {coupler.busbar_to_id} in coupler {coupler.grid_model_id} does not exist in busbars"
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


class MaterializedStation(_StationStructure):
    """Station data describing a single materialized station.

    The station identity refers to a single bus-branch model bus_id that represents one splitable
    station view.
    A physical substation or voltage level may contain multiple bus-branch model bus_ids.
    The station assets are aligned with the switching tables and describe the assets visible in that
    station view; they are not intended to define a topology-owned canonical asset list.
    """

    branch_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local branch payloads aligned with ``branch_switching_table``."""

    injection_connections: list[MaterializedAssetConnection] = Field(default_factory=list)
    """Station-local injection payloads aligned with ``injection_switching_table``."""

    @model_validator(mode="after")
    def check_asset_shapes(self: "MaterializedStation") -> "MaterializedStation":
        """Check if switching-table-aligned station-local assets match the matrix shapes."""
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

    @model_validator(mode="after")
    def check_asset_bay(self: "MaterializedStation") -> "MaterializedStation":
        """Check if the asset bay bus is in busbars."""
        busbar_grid_model_id = [busbar.grid_model_id for busbar in self.busbars]
        for asset_connection in [*self.branch_connections, *self.injection_connections]:
            asset = asset_connection.asset
            asset_bay = asset_connection.asset_bay
            if asset_bay is not None:
                for busbar_id in asset_bay.sr_switch_grid_model_id.keys():
                    if busbar_id not in busbar_grid_model_id:
                        raise ValueError(
                            f"busbar_id {busbar_id} in asset {asset.grid_model_id} does not exist in busbars"
                            f" Station_id: {self.grid_model_id}, Name: {self.name}"
                        )

        return self

    def __eq__(self, other: object) -> bool:
        """Check if two stations are equal.

        Parameters
        ----------
        other : object
            The other station to compare to.

        Returns
        -------
        bool
            True if the stations are equal, False otherwise.
        """
        if not isinstance(other, MaterializedStation):
            return False
        return (
            self.grid_model_id == other.grid_model_id
            and self.region == other.region
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
        )

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "MaterializedStation":
        """Copy and revalidate the station."""
        payload = _merged_round_trip_payload(self, update, deep=deep)
        return type(self).model_validate(payload)

    def get_connected_assets(
        self,
        busbar_index: int,
        topology_assets: Optional[list[SwitchableAsset]] = None,
        asset_scope: Literal["all", "branch", "injection"] = "all",
    ) -> list[SwitchableAsset]:
        """Return in-service assets connected to one busbar.

        Parameters
        ----------
        busbar_index : int
            Row index into the station switching tables.
        topology_assets : Optional[list[SwitchableAsset]]
            Ignored for materialized stations because payloads are embedded locally.
        asset_scope : Literal["all", "branch", "injection"]
            Restrict the lookup to branch or injection connections.

        Returns
        -------
        list[SwitchableAsset]
            Connected in-service assets for the requested busbar and scope.
        """
        del topology_assets
        if asset_scope == "branch":
            return [
                asset_connection.asset
                for asset_connection, is_connected in zip(
                    self.branch_connections,
                    self.branch_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_connection.asset.in_service
            ]
        if asset_scope == "injection":
            return [
                asset_connection.asset
                for asset_connection, is_connected in zip(
                    self.injection_connections,
                    self.injection_switching_table[busbar_index],
                    strict=True,
                )
                if is_connected and asset_connection.asset.in_service
            ]
        return [
            *self.get_connected_assets(busbar_index, asset_scope="branch"),
            *self.get_connected_assets(busbar_index, asset_scope="injection"),
        ]


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

    branch_assets: list[SwitchableAsset] = Field(default_factory=list)
    """The topology-owned canonical branch payloads."""

    injection_assets: list[SwitchableAsset] = Field(default_factory=list)
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
    def check_branch_asset_ids_unique(cls, v: list[SwitchableAsset]) -> list[SwitchableAsset]:
        """Check if all topology branch assets have unique grid model ids."""
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology branch assets")
        if any(asset.is_branch() is False for asset in v):
            raise ValueError("branch_assets must not contain injection assets")
        return v

    @field_validator("injection_assets")
    @classmethod
    def check_injection_asset_ids_unique(cls, v: list[SwitchableAsset]) -> list[SwitchableAsset]:
        """Check if all topology injection assets have unique grid model ids."""
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology injection assets")
        if any(asset.is_branch() is True for asset in v):
            raise ValueError("injection_assets must not contain branch assets")
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


def build_asset_bay_id(station_grid_model_id: str, asset_grid_model_id: str, occurrence_index: int = 0) -> str:
    """Create a deterministic station-scoped asset bay identifier.

    Parameters
    ----------
    station_grid_model_id : str
        Station identifier owning the asset bay.
    asset_grid_model_id : str
        Asset identifier for which the bay id is created.
    occurrence_index : int, default=0
        Zero-based occurrence index for repeated asset ids within one station.

    Returns
    -------
    str
        Deterministic asset bay identifier scoped to the station.
    """
    base_asset_bay_id = f"{station_grid_model_id}::{asset_grid_model_id}::bay"
    if occurrence_index == 0:
        return base_asset_bay_id
    return f"{base_asset_bay_id}::{occurrence_index}"


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


class AppliedStation(BaseModel):
    """A realized station, including the new station and the changes made to the original station."""

    station: MaterializedStation
    """The realized asset station object"""

    coupler_diff: list[BusbarCoupler]
    """A list of couplers that have been switched."""

    branch_reassignment_diff: list[tuple[int, int, bool]]
    """Branch reassignments as ``(branch_index, busbar_index, connected)`` tuples."""

    injection_reassignment_diff: list[tuple[int, int, bool]]
    """Injection reassignments as ``(injection_index, busbar_index, connected)`` tuples."""

    branch_disconnection_diff: list[int]
    """Branch indices that were disconnected."""

    injection_disconnection_diff: list[int]
    """Injection indices that were disconnected."""


class RealizedTopology(BaseModel):
    """A realized topology, including the new topology and the changes made to the original topology.

    This is similar to AppliedStation but holding information for all stations in the topology.
    The diffs are include a station identifier that shows which station in the topology was affected by the
    diff.
    """

    topology: Topology
    """The realized asset topology object"""

    coupler_diff: list[tuple[str, BusbarCoupler]]
    """A list of couplers that have been switched. Each tuple contains the station grid_model_id
    and the coupler that was switched."""

    branch_reassignment_diff: list[tuple[str, int, int, bool]]
    """Branch reassignments as ``(station_id, branch_index, busbar_index, connected)`` tuples."""

    injection_reassignment_diff: list[tuple[str, int, int, bool]]
    """Injection reassignments as ``(station_id, injection_index, busbar_index, connected)`` tuples."""

    branch_disconnection_diff: list[tuple[str, int]]
    """Branch disconnections as ``(station_id, branch_index)`` tuples."""

    injection_disconnection_diff: list[tuple[str, int]]
    """Injection disconnections as ``(station_id, injection_index)`` tuples."""
