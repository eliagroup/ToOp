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


class PandapowerContingencyGroup(BaseModel):
    """
    Represents a group of contingencies affecting the same set of network components.

    This model aggregates multiple contingencies that map to the same connected
    component signature (i.e., they impact the same parts of the grid). It also
    contains the full set of grid elements belonging to those components.
    """

    contingencies: list[PandapowerContingency]
    """A list of original contingencies that affect the same connected
            component(s) and are therefore grouped together.
    """

    elements: list[PandapowerElements]
    """A list of all grid elements contained in the connected component(s)
            associated with this group. These represent the full outage scope."""

    outage_group_id: str
    """Identifier of the outage group.
            An outage group represents a set of elements that is separated from
            the rest of the grid by circuit breakers. In contingency analysis,
            if one element from such a group is taken out of service, the whole
            outage group is considered disconnected and all elements in that
            group become unavailable together."""


class PandapowerNMinus1Definition(BaseModel):
    """A Pandapower N-1 definition.

    This is a simplified version of the NMinus1Definition that is used in Pandapower.
    It contains only the necessary information to run an N-1 analysis in Pandapower.
    """

    model_config = {"arbitrary_types_allowed": True}

    contingencies: list[PandapowerContingency]
    """The outages to be considered. Maps contingency id to outaged element ids."""

    grouped_contingencies: Optional[list[PandapowerContingencyGroup]] = None
    """Optional grouped outages to be considered."""

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


class ParallelConfig(BaseModel):
    """Parallel execution settings for contingency analysis."""

    n_processes: int = 1
    batch_size: Optional[int] = None


class ContingencyAnalysisConfig(BaseModel):
    """Configuration for pandapower N-1 contingency analysis.

    This configuration controls how the base case and contingency load flows are
    executed, how electrical islands are handled, whether outage grouping is
    applied, and how switch results are mapped and aggregated.
    """

    method: Literal["ac", "dc"] = "ac"
    """Load-flow method used for the base case and contingency calculations.

    - ``"ac"`` runs a full AC power flow using :func:`pandapower.runpp`
    - ``"dc"`` runs a DC approximation using :func:`pandapower.rundcpp`
    """

    min_island_size: int = 11
    """Minimum number of PPCI nodes required for an island to receive a slack bus.

    In pandapower's internal (PPCI) representation, buses connected via closed
    bus-bus switches are merged into a single node. Therefore, this value does
    not count individual buses, but aggregated electrical nodes.

    This parameter is used during slack allocation to decide whether an island
    is large enough to be solved normally.
    """

    runpp_kwargs: dict[str, Any] | None = None
    """Additional keyword arguments forwarded to pandapower load-flow execution.

    These arguments are passed to :func:`pandapower.runpp` for AC or
    :func:`pandapower.rundcpp` for DC calculations.
    """

    polars: bool = False
    """Whether to convert the final result object from pandas to polars format."""

    apply_outage_grouping: bool = False
    """Whether to group contingencies by electrically connected outage scope.

    If enabled, contingencies that lead to the same isolated outage area are
    processed as outage groups and a connectivity result table is included in
    the output.
    """

    switch_traversal_side: Literal["bus", "element"] = "bus"
    """Defines which side of a switch is used when computing switch results.

    This setting affects how elements are associated with monitored switches
    before switch result aggregation.

    A switch connects two bus terminals:
    - ``"bus"``: start traversal from ``sw.bus``
    - ``"element"``: start traversal from ``sw.element``

    The selected side determines which electrically connected buses and
    connected elements contribute to the computed switch results.
    """

    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    """Parallel execution settings for contingency processing.

    Controls the number of worker processes and optional batch sizing for
    distributed execution.
    """
