# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the data models for the asset topology."""

from datetime import datetime
from enum import Enum

import numpy as np
from beartype.typing import Any, Literal, Optional, TypeAlias, Union, get_args
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, field_validator, model_validator


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
    "xward",
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

    type: Optional[str] = None
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

    asset_bay_id: Optional[str] = None
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

    type: Optional[str] = None
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

    type: Optional[AssetType] = None
    """ The type of the asset. These refer loosely to the types in the pandapower/powsybl grid
    models. If set, this can be used to disambiguate branches from injections """

    name: Optional[str] = None
    """ The name of the asset, might be useful for finding the asset later on """

    in_service: bool = True
    """ If the element is in service. False means the switching entry for this element will be
    ignored. This shall not be used for elements intentionally disconnected, instead set all zeros
    in the switching table."""

    branch_end: Optional[BranchEnd] = None
    """If the asset was a branch, this can store which end of the branch was connected to the
    station in the original grid model. This can take the values "from", "to", "hv", "mv", "lv",
    where from/to works for lines and hv/mv/lv works for transformers. This should only be set if
    this is needed for the postprocessing, in theory a branch should be identifiable by the branch
    id and the station id. Injection-type assets like generators and loads should not have this set.
    """

    asset_bay: Optional[AssetBay] = None
    """ The asset bay (Schaltfeld) of the asset.
    The connection path is used to determine the physical connection of the asset to the busbar.
    None of these switches will be found in the network model, they are only used for the asset topology."""

    asset_bay_id: Optional[str] = None
    """Topology-scoped identifier for the asset bay associated with this asset."""

    def is_branch(self) -> Optional[bool]:
        """Return True if the asset is a branch.

        Only works if the type is set. If type is not set this will return None.

        Returns
        -------
        bool
            True if the asset is a branch, False if it is an injection.
        """
        if self.type is None:
            return None
        return self.type in get_args(AssetBranchType)


class AssetSetpoint(BaseModel):
    """Asset data describing a single asset with a setpoint.

    This could for example be a PST or HVDC setpoint.
    Note: The same asset can both be switchable and have a setpoint. In this case, the asset will
    be represented twice.
    """

    grid_model_id: str
    """ The unique identifier of the asset.
    Corresponds to the asset's id in the grid model."""

    type: Optional[str] = None
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
            f"asset_switching_table shape {asset_switching_table.shape} does not match busbars "
            f"{busbar_count} and assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )

    if asset_connectivity is not None and asset_connectivity.shape != (busbar_count, asset_count):
        raise ValueError(
            f"asset_connectivity shape {asset_connectivity.shape} does not match busbars "
            f"{busbar_count} and assets {asset_count}"
            f" Station_id: {station_grid_model_id}, Name: {station_name}"
        )


def _validate_station_physical_assignments(
    station_grid_model_id: str,
    station_name: Optional[str],
    asset_switching_table: np.ndarray,
    asset_connectivity: Optional[np.ndarray],
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
                f"Not all current assignments are physically allowed "
                f"Station_id: {station_grid_model_id}, Name: {station_name}"
            )


class _StationStructure(BaseModel):
    """Shared station fields and structural validators for station views."""

    grid_model_id: str
    """The unique identifier of the station.

    Expects the bus-branch model bus_id, not the full voltage level id.
    """

    name: Optional[str] = None
    """The name of the station."""

    type: Optional[str] = None
    """The type of the station."""

    region: Optional[str] = None
    """The region of the station."""

    voltage_level: Optional[float] = None
    """The voltage level of the station."""

    busbars: list[Busbar]
    """The list of busbars at the station."""

    couplers: list[BusbarCoupler]
    """The list of couplers at the station."""

    asset_switching_table: NDArray[Shape[" * bus, * asset"], bool]
    """Holds the switching of each asset to each busbar, shape (n_bus, n_asset).

    An entry is true if the asset is connected to the busbar.
    Note: An asset can be connected to multiple busbars, in which case a closed coupler is assumed
    to be present between these busbars.
    """

    asset_connectivity: Optional[NDArray[Shape[" * bus, * asset"], bool]] = None
    """Holds the all possible layouts of the asset_switching_table, shape (n_bus, n_asset).

    An entry is true if it is possible to connect an asset to the busbar.
    If None, it is assumed that all branches can be connected to all busbars.
    """

    model_log: Optional[list[str]] = None
    """Holds log messages from the model creation process."""

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

    assets: list[SwitchableAsset]
    """The list of assets at the station.

    The order of this list is the same order as the assets in the asset_switching_table.
    This list is station-local and switching-table aligned. Topology-owned canonical assets are
    stored on Topology.assets instead.
    """

    @model_validator(mode="after")
    def check_asset_shapes(self: "MaterializedStation") -> "MaterializedStation":
        """Check if switching-table-aligned station-local assets match the matrix shapes."""
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.assets),
            asset_switching_table=self.asset_switching_table,
            asset_connectivity=self.asset_connectivity,
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.asset_switching_table,
            asset_connectivity=self.asset_connectivity,
        )
        return self

    @model_validator(mode="after")
    def check_asset_bay(self: "MaterializedStation") -> "MaterializedStation":
        """Check if the asset bay bus is in busbars."""
        busbar_grid_model_id = [busbar.grid_model_id for busbar in self.busbars]
        for asset in self.assets:
            if asset.asset_bay is not None:
                for busbar_id in asset.asset_bay.sr_switch_grid_model_id.keys():
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
            and self.assets == other.assets
            and np.array_equal(self.asset_switching_table, other.asset_switching_table)
            and (
                np.array_equal(self.asset_connectivity, other.asset_connectivity)
                if (self.asset_connectivity is not None and other.asset_connectivity is not None)
                else self.asset_connectivity == other.asset_connectivity
            )
        )


class RawStation(_StationStructure):
    """Station data stored inside a topology without embedded asset payloads.

    The station identity still refers to a bus-branch model bus_id for one splitable station view.
    Asset membership is expressed through the aligned station-local arrays instead of embedded
    SwitchableAsset payloads.
    """

    asset_ids: list[str]
    """Asset grid model ids aligned with the switching tables.

    These are station-local references into Topology.assets.
    """

    asset_branch_ends: list[Optional[BranchEnd]]
    """Station-local branch ends aligned with the switching tables."""

    asset_bay_ids: list[Optional[str]]
    """Topology-scoped asset bay identifiers aligned with the switching tables."""

    @model_validator(mode="after")
    def check_asset_reference_alignment(self: "RawStation") -> "RawStation":
        """Check if station-local asset reference arrays are aligned."""
        if len(self.asset_branch_ends) != len(self.asset_ids):
            raise ValueError(
                f"asset_branch_ends length {len(self.asset_branch_ends)} "
                f"does not match asset_ids length {len(self.asset_ids)}"
                f" Station_id: {self.grid_model_id}, Name: {self.name}"
            )
        if len(self.asset_bay_ids) != len(self.asset_ids):
            raise ValueError(
                f"asset_bay_ids length {len(self.asset_bay_ids)} does not match asset_ids length {len(self.asset_ids)}"
                f" Station_id: {self.grid_model_id}, Name: {self.name}"
            )
        _validate_station_switching_tables(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            busbar_count=len(self.busbars),
            asset_count=len(self.asset_ids),
            asset_switching_table=self.asset_switching_table,
            asset_connectivity=self.asset_connectivity,
        )
        _validate_station_physical_assignments(
            station_grid_model_id=self.grid_model_id,
            station_name=self.name,
            asset_switching_table=self.asset_switching_table,
            asset_connectivity=self.asset_connectivity,
        )
        return self

    def __eq__(self, other: object) -> bool:
        """Check if two topology stations are equal."""
        if not isinstance(other, RawStation):
            return False
        return (
            self.grid_model_id == other.grid_model_id
            and self.name == other.name
            and self.type == other.type
            and self.region == other.region
            and self.voltage_level == other.voltage_level
            and self.busbars == other.busbars
            and self.couplers == other.couplers
            and self.asset_ids == other.asset_ids
            and self.asset_branch_ends == other.asset_branch_ends
            and self.asset_bay_ids == other.asset_bay_ids
            and np.array_equal(self.asset_switching_table, other.asset_switching_table)
            and (
                np.array_equal(self.asset_connectivity, other.asset_connectivity)
                if (self.asset_connectivity is not None and other.asset_connectivity is not None)
                else self.asset_connectivity == other.asset_connectivity
            )
            and self.model_log == other.model_log
        )


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

    assets: list[SwitchableAsset] = Field(default_factory=list)
    """The topology-owned canonical asset payloads.

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

    @field_validator("assets")
    @classmethod
    def check_asset_ids_unique(cls, v: list[SwitchableAsset]) -> list[SwitchableAsset]:
        """Check if all topology assets have unique grid model ids."""
        asset_ids = [asset.grid_model_id for asset in v]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("grid_model_id must be unique for topology assets")
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
        asset_ids = {asset.grid_model_id for asset in self.assets}
        asset_bay_ids = {asset_bay.asset_bay_id for asset_bay in self.asset_bays}

        for station in self.raw_stations:
            for asset_id in station.asset_ids:
                if asset_id not in asset_ids:
                    raise ValueError(
                        f"Asset grid_model_id {asset_id} referenced by station "
                        f"{station.grid_model_id} does not exist in topology assets"
                    )
            for asset_bay_id in station.asset_bay_ids:
                if asset_bay_id is not None and asset_bay_id not in asset_bay_ids:
                    raise ValueError(
                        f"asset_bay_id {asset_bay_id} referenced by station "
                        f"{station.grid_model_id} does not exist in topology asset bays"
                    )

        return self

    def materialize_stations(self) -> list[MaterializedStation]:
        """Materialize station-local asset payloads from topology-owned assets and asset bays."""
        asset_map = {asset.grid_model_id: asset for asset in self.assets}
        asset_bay_map = {asset_bay.asset_bay_id: asset_bay for asset_bay in self.asset_bays}
        materialized_stations: list[MaterializedStation] = []

        for station in self.raw_stations:
            station_assets: list[SwitchableAsset] = []
            for asset_id, asset_branch_end, asset_bay_id in zip(
                station.asset_ids, station.asset_branch_ends, station.asset_bay_ids, strict=True
            ):
                asset = asset_map[asset_id]
                asset_bay = asset_bay_map.get(asset_bay_id) if asset_bay_id is not None else None
                station_assets.append(
                    asset.model_copy(
                        update={
                            "branch_end": asset_branch_end,
                            "asset_bay_id": asset_bay_id,
                            "asset_bay": asset_bay.model_copy(deep=True) if asset_bay is not None else None,
                        },
                        deep=True,
                    )
                )

            materialized_stations.append(
                MaterializedStation(
                    grid_model_id=station.grid_model_id,
                    name=station.name,
                    type=station.type,
                    region=station.region,
                    voltage_level=station.voltage_level,
                    busbars=station.busbars,
                    couplers=station.couplers,
                    assets=station_assets,
                    asset_switching_table=station.asset_switching_table,
                    asset_connectivity=station.asset_connectivity,
                    model_log=station.model_log,
                )
            )

        return materialized_stations


def _default_asset_bay_id(station_grid_model_id: str, asset_grid_model_id: str, occurrence_index: int) -> str:
    """Create a deterministic station-scoped asset bay identifier.

    Parameters
    ----------
    station_grid_model_id : str
        Station identifier owning the asset bay.
    asset_grid_model_id : str
        Asset identifier for which the bay id is created.
    occurrence_index : int
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
    assets: list[SwitchableAsset] = []
    asset_bays: list[AssetBay] = []
    asset_ids: list[str] = []
    asset_branch_ends: list[Optional[BranchEnd]] = []
    asset_bay_ids: list[Optional[str]] = []
    station_asset_occurrences: dict[str, int] = {}

    for asset in station.assets:
        asset_id = asset.grid_model_id
        asset_bay_id: Optional[str] = asset.asset_bay_id
        asset_occurrence_index = station_asset_occurrences.get(asset_id, 0)
        station_asset_occurrences[asset_id] = asset_occurrence_index + 1

        if asset.asset_bay is not None:
            asset_bay = asset.asset_bay
            asset_bay_id = asset_bay.asset_bay_id or _default_asset_bay_id(
                station.grid_model_id, asset_id, asset_occurrence_index
            )
            asset_bays.append(asset_bay.model_copy(update={"asset_bay_id": asset_bay_id}, deep=True))

        assets.append(
            asset.model_copy(
                update={
                    "branch_end": None,
                    "asset_bay_id": None,
                    "asset_bay": None,
                },
                deep=True,
            )
        )
        asset_ids.append(asset_id)
        asset_branch_ends.append(asset.branch_end)
        asset_bay_ids.append(asset_bay_id)

    return (
        RawStation(
            grid_model_id=station.grid_model_id,
            name=station.name,
            type=station.type,
            region=station.region,
            voltage_level=station.voltage_level,
            busbars=station.busbars,
            couplers=station.couplers,
            asset_ids=asset_ids,
            asset_branch_ends=asset_branch_ends,
            asset_bay_ids=asset_bay_ids,
            asset_switching_table=station.asset_switching_table,
            asset_connectivity=station.asset_connectivity,
            model_log=station.model_log,
        ),
        assets,
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
    topology_assets: dict[str, SwitchableAsset] = {}
    topology_asset_bays: dict[str, AssetBay] = {}

    for station in stations:
        topology_station, station_assets, station_asset_bays = topology_parts_from_materialized_station(station)
        topology_stations.append(topology_station)
        for asset in station_assets:
            existing_asset = topology_assets.get(asset.grid_model_id)
            if existing_asset is None:
                topology_assets[asset.grid_model_id] = asset
            elif existing_asset != asset:
                raise ValueError(f"Conflicting topology asset payload for grid_model_id {asset.grid_model_id}")
        for asset_bay in station_asset_bays:
            if asset_bay.asset_bay_id is None:
                continue
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
        assets=list(topology_assets.values()),
        asset_bays=list(topology_asset_bays.values()),
        asset_setpoints=reference_topology.asset_setpoints,
        timestamp=reference_topology.timestamp,
        metrics=reference_topology.metrics,
    )


def copy_topology_with_updates(
    reference_topology: Topology,
    raw_stations: list[RawStation],
    assets: list[SwitchableAsset],
    asset_bays: list[AssetBay],
) -> Topology:
    """Create a validated topology copy with updated payloads.

    Parameters
    ----------
    reference_topology : Topology
        Reference topology providing shared metadata.
    raw_stations : list[RawStation]
        Raw stations to set on the copied topology.
    assets : list[SwitchableAsset]
        Topology-owned assets to set on the copied topology.
    asset_bays : list[AssetBay]
        Topology-owned asset bays to set on the copied topology.

    Returns
    -------
    Topology
        Validated topology copy with updated topology-owned payloads.
    """
    return Topology(
        topology_id=reference_topology.topology_id,
        grid_model_file=reference_topology.grid_model_file,
        name=reference_topology.name,
        raw_stations=raw_stations,
        assets=assets,
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
    """A realized station, including the new station and the changes made to the original station"""

    station: MaterializedStation
    """The realized asset station object"""

    coupler_diff: list[BusbarCoupler]
    """A list of couplers that have been switched."""

    reassignment_diff: list[tuple[int, int, bool]]
    """A list of reassignments that have been made. Each tuple contains the asset index that was
    affected (not the asset grid_model_id but the index into the asset_switching_table), the busbar
    index (again the index into the switching table) and whether the asset was connected (True) or
    disconnected (False) to that busbar."""

    disconnection_diff: list[int]
    """A list of disconnections that have been made. Each tuple contains the asset index that was
    disconnected."""


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

    reassignment_diff: list[tuple[str, int, int, bool]]
    """A list of reassignments that have been made. Each tuple contains the station grid_model_id,
    the asset index that was affected (not the asset grid_model_id but the index into the
    asset_switching_table), the busbar index (again the index into the switching table) and whether the
    asset was connected (True) or disconnected (False) to that busbar."""

    disconnection_diff: list[tuple[str, int]]
    """A list of disconnections that have been made. Each tuple contains the station grid_model_id
    and the asset index that was disconnected. This can also include non-relevant stations."""
