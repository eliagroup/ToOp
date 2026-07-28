# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Typed contracts for electrical circuit group identification."""

import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Optional, TypeAlias
from pydantic import BaseModel, ConfigDict, Field


class EdgeSchema(pa.DataFrameModel):
    """Schema for branch edges in the bus-breaker connectivity graph.

    Used by the rustworkx graph library to represent the bus-breaker topology of the network.
    """

    id_str: pat.Series[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    bus_breaker_id_int_bus1: pat.Series[int]
    bus_breaker_id_int_bus2: pat.Series[int]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class BusBreakerIdSchema(pa.DataFrameModel):
    """Schema for bus-breaker identifiers and their outage-group mapping."""

    id: pat.Index[str]
    bus_breaker_id_int: pat.Series[int]
    electrical_circuit_group: Optional[pat.Series[int]]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class BranchElectricalCircuitGroupSchema(pa.DataFrameModel):
    """Schema for branches enriched with their electrical circuit group."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    electrical_circuit_group: Optional[pat.Series[int]]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class SwitchElectricalCircuitGroupSchema(pa.DataFrameModel):
    """Schema for breaker switches enriched with endpoint circuit groups."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    kind: pat.Series[str]
    electrical_circuit_group_bus1: Optional[pat.Series[int]]
    electrical_circuit_group_bus2: Optional[pat.Series[int]]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class InjectionElectricalCircuitGroupSchema(pa.DataFrameModel):
    """Schema for injections enriched with their electrical circuit group."""

    id: pat.Index[str]
    bus_breaker_bus_id: pat.Series[str]
    type: pat.Series[str]
    electrical_circuit_group: Optional[pat.Series[int]]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class BusbarCouplerSchema(pa.DataFrameModel):
    """Schema for busbar couplers connected between two busbar sections."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    kind: pat.Series[str]
    electrical_circuit_group_bus1: pat.Series[int]
    electrical_circuit_group_bus2: pat.Series[int]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class AssetBreakerSchema(pa.DataFrameModel):
    """Schema for breakers connecting a busbar section to an asset circuit group."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    kind: pat.Series[str]
    electrical_circuit_group_busbar: pat.Series[int]
    asset_circuit_group: pat.Series[int]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class ElectricalCircuitGroup(BaseModel):
    """Outage-group contents for a single electrical circuit group."""

    branches: list[str] = Field(default_factory=list)
    switches: list[str] = Field(default_factory=list)
    injections: list[str] = Field(default_factory=list)
    busbar_section: list[str] = Field(default_factory=list)


class BusbarSectionOutageGroup(BaseModel):
    """Busbar-section outage expansion metadata."""

    primary_circuit_group: int
    busbar_couplers: list[str] = Field(default_factory=list)
    primary_asset_breakers: list[str] = Field(default_factory=list)
    asset_circuit_groups: list[int] = Field(default_factory=list)


class ElectricalCircuitGroupIdentification(BaseModel):
    """Typed result bundle returned by outage-group identification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    circuit_group_map: "ElectricalCircuitGroupMap"
    branches: pd.DataFrame
    switches: pd.DataFrame
    injections: pd.DataFrame


ElectricalCircuitGroupMap: TypeAlias = dict[int, ElectricalCircuitGroup]
BusbarSectionOutageGroups: TypeAlias = dict[str, BusbarSectionOutageGroup]
