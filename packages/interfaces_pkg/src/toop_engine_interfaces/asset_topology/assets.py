# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes that represent Assets in the grid"""

from enum import Enum

from beartype.typing import Any, Optional, get_args
from pydantic import BaseModel, field_validator
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType, AssetType


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


class PowsyblSwitchValues(Enum):
    """Enum for the switch values in the Powsybl model."""

    OPEN = True
    """ The switch is open, i.e. not connected."""
    CLOSED = False
    """ The switch is closed, i.e. connected."""
