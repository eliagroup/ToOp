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

    Used by the networkx graph library to represent the bus-breaker topology of the network.
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


class BusBreakerViewSchema(pa.DataFrameModel):
    """Schema for the reduced bus-breaker view table fetched from pypowsybl."""

    id: pat.Index[str]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class BranchSchema(pa.DataFrameModel):
    """Schema for raw branch rows used by circuit-group identification."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class SwitchSchema(pa.DataFrameModel):
    """Schema for raw switch rows used by circuit-group identification."""

    id: pat.Index[str]
    bus_breaker_bus1_id: pat.Series[str]
    bus_breaker_bus2_id: pat.Series[str]
    kind: pat.Series[str]
    fictitious: pat.Series[bool]

    class Config:
        """Pandera configuration for strict column validation."""

        strict = True


class InjectionSchema(pa.DataFrameModel):
    """Schema for raw injection rows used by circuit-group identification."""

    id: pat.Index[str]
    bus_breaker_bus_id: pat.Series[str]
    type: pat.Series[str]

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


class ElectricalCircuitGroup(BaseModel):
    """Outage-group contents for a single electrical circuit group."""

    branches: list[str] = Field(default_factory=list)
    switches: list[str] = Field(default_factory=list)
    injections: list[str] = Field(default_factory=list)
    busbar_section: list[str] = Field(default_factory=list)


class PreparedCircuitGroupLookupData(BaseModel):
    """Shared preprocessed group mappings reused by circuit-group lookup builders."""

    branch_to_group: dict[str, int] = Field(default_factory=dict)
    busbar_to_primary_group: dict[str, int] = Field(default_factory=dict)
    busbar_to_asset_groups: dict[str, list[int]] = Field(default_factory=dict)
    group_to_branches: dict[int, list[str]] = Field(default_factory=dict)
    group_to_switches: dict[int, list[str]] = Field(default_factory=dict)
    group_to_injections: dict[int, list[str]] = Field(default_factory=dict)
    group_to_busbar_sections: dict[int, list[str]] = Field(default_factory=dict)


class CircuitGroupLookupIndex(BaseModel):
    """Lookup-oriented circuit-group representation keyed by group and busbar section."""

    branch_to_group: dict[str, int] = Field(default_factory=dict)
    busbar_to_primary_group: dict[str, int] = Field(default_factory=dict)
    busbar_to_asset_groups: dict[str, list[int]] = Field(default_factory=dict)
    group_to_branches: dict[int, list[str]] = Field(default_factory=dict)
    group_to_switches: dict[int, list[str]] = Field(default_factory=dict)
    group_to_injections: dict[int, list[str]] = Field(default_factory=dict)
    group_to_busbar_sections: dict[int, list[str]] = Field(default_factory=dict)
    group_to_failing_elements: dict[int, list[str]] = Field(default_factory=dict)
    group_to_failing_switches: dict[int, list[str]] = Field(default_factory=dict)
    busbar_to_failing_elements: dict[str, list[str]] = Field(default_factory=dict)
    busbar_to_failing_switches: dict[str, list[str]] = Field(default_factory=dict)


class ElectricalCircuitGroupIdentification(BaseModel):
    """Typed result bundle returned by outage-group identification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lookup_index: "CircuitGroupLookupIndex"
    branches: pd.DataFrame
    switches: pd.DataFrame
    injections: pd.DataFrame


#: Identifier of one lookup input such as a branch id or busbar-section id.
LookupInputId: TypeAlias = str

#: Ordered failing element ids returned for one lookup input.
FailingElementIds: TypeAlias = list[str]

#: Ordered failing switch ids returned for one lookup input.
FailingSwitchIds: TypeAlias = list[str]

#: Mapping from lookup input id to failing element ids.
FailingElementsByLookupId: TypeAlias = dict[LookupInputId, FailingElementIds]

#: Mapping from lookup input id to failing switch ids.
FailingSwitchesByLookupId: TypeAlias = dict[LookupInputId, FailingSwitchIds]


ElectricalCircuitGroupMap: TypeAlias = dict[int, ElectricalCircuitGroup]
