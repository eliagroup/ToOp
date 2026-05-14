# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Data models exchanged between cascade detection, outage grouping, and simulation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import pandas as pd
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.spps import SppsResult
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    ConvergenceStatus,
    NodeResultSchema,
    SwitchResultsSchema,
)


class CascadeReasonType(str, Enum):
    """Reasons why an element was added to a cascade event log."""

    CASCADE_REASON_CURRENT = "CURRENT_VIOLATION"
    CASCADE_REASON_DISTANCE = "DISTANCE_PROTECTION"
    FAILED_LF = "FAILED_LF"


@dataclass(frozen=True)
class CascadeEvent:
    """One item in the cascade event log."""

    element_mrid: str | None
    """External identifier of the affected grid element, if known."""

    element_id: str | None
    """Globally unique id of the affected grid element, if known."""

    element_name: str | None
    """Human-readable name of the affected grid element, if known."""

    cascade_number: int
    """Cascade step number where the event happened."""

    cascade_reason: str
    """Reason for the event, such as current overload or distance protection."""

    loading: float | None = None
    """Branch loading value that triggered the event, if available."""

    r_ohm: float | None = None
    """Relay resistance value for distance-protection events, if available."""

    x_ohm: float | None = None
    """Relay reactance value for distance-protection events, if available."""

    outage_group_id: str | None = None
    """Identifier of the outage group that produced this event, if available."""

    distance_protection_severity: Optional[str] = None
    """Optional severity label for distance protection events."""

    activated_schemes_per_iter: str | None = None
    """JSON string of SpPS scheme names that activated per inner load-flow iteration."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the event into plain values that can be serialized as JSON.

        Returns
        -------
        dict[str, Any]
            Dictionary with the event fields.
        """
        return {
            "element_mrid": self.element_mrid,
            "element_id": self.element_id,
            "element_name": self.element_name,
            "cascade_number": self.cascade_number,
            "cascade_reason": self.cascade_reason,
            "loading": self.loading,
            "r_ohm": self.r_ohm,
            "x_ohm": self.x_ohm,
            "outage_group_id": self.outage_group_id,
            "distance_protection_severity": self.distance_protection_severity,
            "activated_schemes_per_iter": self.activated_schemes_per_iter,
        }


@dataclass
class CascadeContext:
    """Prepared network information reused during one cascade simulation."""

    switch_characteristics: pd.DataFrame
    """Breaker and relay data used for distance protection checks."""

    bus_couplers_mrids: set[str]
    """External identifiers of switches that couple busbars."""


@dataclass(frozen=True)
class CascadeSppsBranchSwitchResults:
    """Load-flow status and result tables for one cascade step."""

    convergence_status: ConvergenceStatus
    """Whether the load flow converged."""

    spps_result: SppsResult | None
    """Summary of special protection scheme execution, if any."""

    branch_results: pat.DataFrame[BranchResultSchema] | None
    """Branch result table from the step, or None when unavailable."""

    node_results: pat.DataFrame[NodeResultSchema] | None
    """Node result table from the step, or None when unavailable."""

    switch_results: pat.DataFrame[SwitchResultsSchema] | None
    """Switch result table from the step, or None when unavailable."""


@dataclass(frozen=True)
class CascadeTriggers:
    """Initial or next-step conditions that can continue a cascade."""

    tripped_switches: pd.DataFrame
    """Switches whose relay protection should trip."""

    current_overloaded_elements: pd.DataFrame
    """Branches above the current loading threshold."""

    @property
    def empty(self) -> bool:
        """Tell whether there are no switch trips and no current overloads.

        Returns
        -------
        bool
            True when no cascade trigger is present.
        """
        return self.tripped_switches.empty and self.current_overloaded_elements.empty
