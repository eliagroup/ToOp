# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Schemas for N-1"""

import dataclasses

import pandera as pa
import pandera.typing as pat
from beartype.typing import Any, Literal, Optional
from networkx.classes import MultiGraph
from pandera.typing import Index, Series
from pydantic import BaseModel, Field
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
)


@dataclasses.dataclass
class SlackAllocationConfig:
    """Carry configuration required for slack allocation per island."""

    net_graph: MultiGraph
    bus_lookup: list[int]
    min_island_size: int = 11


class VADiffInfo(BaseModel):
    """Class to hold information about which switches to monitor for voltage angle difference.

    For each contingency, we need to know the voltage angle difference between the from and to bus of each affected branch.
    This is necessary to determine if we could easily reconnect the outaged branch after the contingency
    using the existing power switches.
    """

    from_bus: int
    """The from side of the branch. The voltage angle difference is calculated as va(from_bus) - va(to_bus)."""
    to_bus: int
    """The to side of the branch. The voltage angle difference is calculated as va(from_bus) - va(to_bus)."""

    power_switches_from: dict[str, str]
    """A mapping from switch unique ids to their names for the from side of the branch."""

    power_switches_to: dict[str, str]
    """A mapping from switch unique ids to their names for the to side of the branch."""


class PandapowerMonitoredElementSchema(pa.DataFrameModel):
    """Schema for a monitored element in the N-1 definition."""

    unique_id: Index[str] = pa.Field(description="The globally unique id of the monitored element.")
    table: Series[str] = pa.Field(description="The type of the monitored element, e.g. 'line', 'bus', 'load', etc.")
    table_id: Series[int] = pa.Field(description="The id of the monitored element in the corresponding table.")
    kind: Series[str] = pa.Field(
        isin=["branch", "bus", "injection", "switch"],
        description="The kind of the monitored element, e.g. 'branch', 'bus' etc.",
    )
    name: Series[str] = pa.Field(description="The name of the monitored element, if available.")


class PandapowerElements(BaseModel):
    """A Pandapower element with its globally unique id, table and table_id."""

    unique_id: str
    """The globally unique id of the element."""
    table: str
    """The type of the element, e.g. 'line', 'bus', 'load', etc."""
    table_id: int
    """The id of the element in the corresponding table."""
    name: str = ""
    """The name of the element, if available."""


class PandapowerContingency(BaseModel):
    """A contingency for Pandapower.

    Adds info about the table and table_id of the outaged elements.
    """

    unique_id: str
    """The globally unique id of the contingency."""
    name: str = ""
    """The name of the contingency, if available."""

    elements: list[PandapowerElements]
    """The elements that are outaged in this contingency."""

    va_diff_info: list[VADiffInfo] = Field(default_factory=list)
    """A mapping from nodes at branches and their closest Circuit breaker switches."""

    def is_basecase(self) -> bool:
        """Check if the contingency is the N-0 base case.

        A base case is defined as a contingency that has no elements in it.
        """
        return len(self.elements) == 0


class PandapowerNMinus1Definition(BaseModel):
    """A Pandapower N-1 definition.

    This is a simplified version of the NMinus1Definition that is used in Pandapower.
    It contains only the necessary information to run an N-1 analysis in Pandapower.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PandapowerContingency]
    """The outages to be considered. Maps contingency id to outaged element ids."""

    missing_contingencies: list[Contingency]
    """A list of contingencies that were not found in the network."""

    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
    """A dictionary mapping the element kind to a list of element ids that are monitored."""

    missing_elements: list[GridElement]
    """A list of monitored elements that were not found in the network."""

    duplicated_grid_elements: list[str] = Field(
        default_factory=list,
        description="A list of ids that were not unique in the grid. This is only relevant for cgmes ids.",
    )

    @property
    def base_case(self) -> Optional[PandapowerContingency]:
        """Get the base case contingency, which is the contingency with no elements in it."""
        for contingency in self.contingencies:
            if contingency.is_basecase():
                return contingency
        return None

    def __getitem__(self, key: str | int | slice) -> "PandapowerNMinus1Definition":
        """Get a subset of the nminus1definition based on the contingencies.

        If a string is given, the contingency id must be in the contingencies list.
        If an integer or slice is given, the case id will be indexed by the integer or slice.
        """
        if isinstance(key, str):
            contingency_ids = [contingency.unique_id for contingency in self.contingencies]
            if key not in contingency_ids:
                raise KeyError(f"Contingency id {key} not in contingencies.")
            index = contingency_ids.index(key)
            index = slice(index, index + 1)
        elif isinstance(key, int):
            index = slice(key, key + 1)
        elif isinstance(key, slice):
            index = key
        else:
            raise TypeError("Key must be a string, int or slice.")

        updated_definition = self.model_copy(
            update={
                "contingencies": self.contingencies[index],
            }
        )
        # pylint: disable=unsubscriptable-object
        return PandapowerNMinus1Definition.model_validate(updated_definition)


class ParallelConfig(BaseModel):
    """Parallel execution settings for contingency analysis."""

    n_processes: int = 1
    batch_size: Optional[int] = None


class ContingencyAnalysisConfig(BaseModel):
    """Configuration for pandapower N-1 contingency analysis."""

    method: Literal["ac", "dc"] = "ac"
    min_island_size: int = 11

    runpp_kwargs: dict[str, Any] | None = None
    polars: bool = False
    apply_outage_grouping: bool = False

    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
