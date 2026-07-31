# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Classes that represent Assets in the grid"""

from enum import Enum

from beartype.typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, field_validator
from toop_engine_interfaces.asset_topology.asset_types import AssetBranchType, AssetInjectionType, AssetType


class Busbar(BaseModel):
    """Canonical busbar data describing a physical busbar in a station."""

    model_config = ConfigDict(extra="forbid")

    grid_model_id: str
    """ The unique identifier of the busbar.
    Corresponds to the busbar's id in the grid model."""

    busbar_type: Optional[str] = None
    """ The type of the busbar, might be useful for finding the busbar later on """

    name: Optional[str] = None
    """ The name of the busbar, might be useful for finding the busbar later on """

    int_id: int
    """ Is used to reference busbars in the couplers. Needs to be unique per station"""

    bus_breaker_bus_id: Optional[str] = None
    """Physical bus-breaker bus id backing this busbar section.

    This identifies the bus in the source bus-breaker topology that the physical
    busbar section belongs to. Unlike ``bus_branch_bus_id`` this is not meant to
    reflect runtime regrouping after switching operations.
    """


class RuntimeBusbar(Busbar):
    """Runtime busbar with live service and electrical-bus state."""

    in_service: bool = True
    """Whether the busbar is in service in the current runtime state."""

    bus_branch_bus_id: Optional[str] = None
    """Runtime-only electrical bus id for this physical busbar."""


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


class CouplerBay(BaseModel):
    """Physical switch-path metadata for one busbar coupler."""

    connection_kind: Literal["coupler", "disconnector"] = "coupler"
    """Structural kind of this busbar connection."""

    dv_switch_grid_model_id: str
    """Grid-model id of the coupler power switch itself."""

    from_busbar_grid_model_ids: list[str] = []
    """Directly reachable canonical busbars on the from side."""

    to_busbar_grid_model_ids: list[str] = []
    """Directly reachable canonical busbars on the to side."""

    from_sr_switch_grid_model_id: dict[str, str]
    """Selector switches on the coupler from side keyed by canonical busbar id."""

    to_sr_switch_grid_model_id: dict[str, str]
    """Selector switches on the coupler to side keyed by canonical busbar id."""


class BusbarCoupler(BaseModel):
    """Canonical coupler data describing a physical busbar coupler at a station.

    This references only busbar couplers, i.e. couplers connecting two busbars.
    Switches connecting assets to a busbar are represented in the asset_switching_table in the station model.

    Note: A busbar couple is a physical connection between two busbars, this can be also a
    cross coupler. To further specify the connection of an asset to a busbar, the asset connection
    """

    model_config = ConfigDict(extra="forbid")

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

    asset_bay: Optional[AssetBay] = None
    """ The asset bay (Schaltfeld) of the coupler.
    Note: A coupler can have multiple from and to busbars.
    The asset bay sr_switch_grid_model_id is used save the selector switches of the coupler.
    Note: A coupler has never a sl_switch_grid_model_id, the dv_switch_grid_model_id should
    the same as the name of the coupler.

    """

    coupler_bay: Optional[CouplerBay] = None
    """Side-aware coupler bay metadata used to reconstruct runtime endpoints."""


class RuntimeBusbarCoupler(BusbarCoupler):
    """Runtime coupler with live open and service state."""

    open: bool = False
    """Whether the coupler is open in the current runtime state."""

    in_service: bool = True
    """Whether the coupler is in service in the current runtime state."""


class SwitchableAsset(BaseModel):
    """Canonical asset data describing a single switchable asset.

    An asset can be for instance a transformer, line, generator, load, shunt.
    Note: An asset can be connected to multiple busbars through the switching grid, however if
    this happens a closed coupler between these busbars is assumed. If such couplers are not present,
    they will be created.
    Note: An asset that is out-of-service can be represented, but its switching entries will be
    ignored.
    """

    model_config = ConfigDict(extra="forbid")

    grid_model_id: str
    """ The unique identifier of the asset.
    Corresponds to the asset's id in the grid model."""

    asset_type: Optional[AssetType] = None
    """ The type of the asset. These refer loosely to the types in the pandapower/powsybl grid
    models. If set, this can be used to disambiguate branches from injections """

    name: Optional[str] = None
    """ The name of the asset, might be useful for finding the asset later on """


class BranchAsset(SwitchableAsset):
    """Canonical switchable asset representing a branch-type element."""

    asset_type: Optional[AssetBranchType] = None


class InjectionAsset(SwitchableAsset):
    """Canonical switchable asset representing an injection-type element."""

    asset_type: Optional[AssetInjectionType] = None


class RuntimeSwitchableAsset(SwitchableAsset):
    """Runtime switchable asset with live service state."""

    in_service: bool = True
    """Whether the asset is in service in the current runtime state."""


class RuntimeBranchAsset(BranchAsset, RuntimeSwitchableAsset):
    """Runtime switchable asset representing a branch-type element."""


class RuntimeInjectionAsset(InjectionAsset, RuntimeSwitchableAsset):
    """Runtime switchable asset representing an injection-type element."""


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
