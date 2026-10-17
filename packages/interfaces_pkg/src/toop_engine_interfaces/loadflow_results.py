# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines interfaces for loadflow results.

The overall process is that a job is called on a loadflow engine for a grid.
The grid holds some information that is referenced in the results:
- an N-1 definition, which can include multi-contingencys. An N-1 case is uniquely identified by a string descriptor but can
    include multiple failing elements. The string identifier of the N-1 case should be delivered upon loading.
- a number of timesteps, which are uniquely identified by an integer index.
- branches which are uniquely identified by a string descriptor and have either two or three sides.
- nodes which are uniquely identified by a string descriptor and have a type (PV, PQ, REF)
- regulating elements which are uniquely identified by a string descriptor and have a type (generator, regulating
    transformer, SVC, ...)
"""

from enum import Enum

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pat
from beartype.typing import Optional, Union
from pandera.typing import DataFrame, Index, Series
from pydantic import BaseModel, Field
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter


class BranchSide(Enum):
    """The side of a branch."""

    ONE = 1
    """The following side for the types of branches:
    - line: from side
    - 2 winding trafo: high voltage side
    - 3 winding trafo: high voltage side
    - other: from side
    """

    TWO = 2
    """The following side for the types of branches:
    - line: to side
    - 2 winding trafo: low voltage side
    - 3 winding trafo: medium voltage side
    - other: to side
    """

    THREE = 3
    """Only valid for 3 winding transformers, representing the low voltage side."""

    NONE = 4
    """No side specified."""


class RegulatingElementType(Enum):
    """A list of known regulating elements, TODO expand"""

    GENERATOR_Q = "GENERATOR_Q"
    """A generator that is used to control the reactive power output."""

    SLACK_P = "SLACK_P"
    """The active power output of the slack node."""

    SLACK_Q = "SLACK_Q"
    """The reactive power output of the slack node."""

    REGULATING_TRANSFORMER_TAP = "REGULATING_TRANSFORMER_TAP"
    """A regulating transformer that is used to control the tap position."""

    SVC_Q = "SVC_Q"
    """A static var compensator that is used to control the reactive power output."""

    HVDC_CONVERTER_Q = "HVDC_CONVERTER_Q"
    """An HVDC converter station."""

    OTHER = "OTHER"
    """A placeholder for not yet known regulating elements."""


class ConvergenceStatus(Enum):
    """The convergence status of the loadflow in a single timestep/contingency/component"""

    CONVERGED = "CONVERGED"
    """The loadflow converged"""

    FAILED = "FAILED"
    """The loadflow failed to start, e.g. because no slack bus was available"""

    MAX_ITERATION_REACHED = "MAX_ITERATION_REACHED"
    """The maximum number of iterations was reached, i.e. the loadflow did not converge.
    """

    NO_CALCULATION = "NO_CALCULATION"
    """The component was ignored due to other reasons (engine did not support it)"""


class BranchResultSchema(pa.DataFrameModel):
    """A schema for the branch results table.

    This holds i, p and q values for all monitored branches with a multi-index of
    timestep, contingency (CO), branch (CB) and side.

    If no branches are monitored, this is the empty DataFrame.
    # TODO Decide if this should be used for injections aswell
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The contingency that caused this loadflow. For N-0 results,
    the special CO 'BASECASE' without GridElements is used, if its added."""

    element: Index[str] = pa.Field()
    """The branch that these loadflow results correspond to"""

    side: Index[int] = pa.Field()
    """The side of the branch that these results correspond to"""

    i: Series[float] = pa.Field(nullable=True)
    """The current in the branch in A

    This should only be NaN if the branch has no connection to the slack bus.
    """

    p: Series[float] = pa.Field(nullable=True)
    """The active power in the branch in MW

    This should only be NaN if the branch has no connection to the slack bus.
    """

    q: Series[float] = pa.Field(nullable=True)
    """The reactive power in the branch in MVar

    This should only be NaN if the branch has no connection to the slack bus.
    """

    loading: Series[float] = pa.Field(nullable=True)
    """The loading of the branch in % of rated current. This always refers to the permanent/default rating of the
    branch if there are multiple ratings available.
    If no rating is available for the branch, this should be set to NaN.
    If the engine does not support the computation of this value, the column can be omitted.
    """

    element_name: Series[str]
    """The name of the Branch, if available. This is not used for the loadflow computation, but can be used for display
    purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """

    @pa.dataframe_check
    def check_side(cls, dataframe: pd.DataFrame) -> Series[bool]:
        """Validate branch side values on the multi-index."""
        side_values = dataframe.index.get_level_values("side")
        return pd.Series(side_values.isin([side.value for side in BranchSide]), index=dataframe.index)


class NodeResultSchema(pa.DataFrameModel):
    """A schema for the node results table.

    This holds p and q values for all monitored nodes with a multi-index of timestep and contingency.
    If no nodes are monitored, this is the empty DataFrame.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str] = pa.Field()
    """The node that these loadflow results correspond to"""

    vm: Series[float] = pa.Field(nullable=True)
    """The voltage magnitude at the node in kV.

    In DC, this should be the nominal voltage of the node.
    This should only be NaN if the node does not have a connection to the slack bus.
    """

    vm_loading: Series[float] = pa.Field(nullable=True)
    """How close the voltage magnitude is to the max/min voltage limits in percent.
    This is computed as:
    (vm - v_nominal) / (v_max - v_nominal) for vm > v_nominal and
    (vm - v_nominal) / (v_nominal - v_min) for vm < v_nominal."""

    va: Series[float] = pa.Field(nullable=True)
    """The voltage angle at the node in degrees

    This should only be NaN if the node does not have a connection to the slack bus.
    """

    p: Series[float] = pa.Field(nullable=True)
    """The accumulated absolute active power at the node in MW, obtained by summing the absolute active power of all branches
    and injections connected to the node.

    If the engine does not support the computation of this value, the column can be omitted.
    """

    q: Series[float] = pa.Field(nullable=True)
    """The accumulated absolute reactive power at the node in MVar, obtained by summing the absolute reactive power of all
    branches and injections connected to the node

    If the engine does not support the computation of this value, the column can be omitted.
    """

    vm_basecase_deviation: Series[float] = pa.Field(nullable=True)
    """Voltage magnitude deviation from the basecase (N-0) in percent.
        Computed as:
            abs(vm_contingency - vm_basecase) / vm_basecase * 100
        For basecase contingency, the deviation will be 0.0 (basecase vs basecase).
        NaN if no valid basecase voltage exists(basecase not converged).
    """

    element_name: Series[str]
    """The name of the node, if available. This is not used for the loadflow computation, but can be used for display
    purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class ConnectivityResultSchema(pa.DataFrameModel):
    """
    Schema defining the contingency-to-element connectivity mapping.

    Each row represents a relationship between a contingency and an affected
    grid element, based on outage group logic.
    """

    contingency: Index[str] = pa.Field()
    """Global unique identifier of the contingency event.

    Represents the triggering outage (e.g., line, transformer, generator).
    This is the first level of the MultiIndex.
    """

    element: Index[str] = pa.Field()
    """Global unique identifier of a grid element affected by the contingency.

    Each element listed here belongs to the outage group associated with the
    contingency and is therefore considered disconnected when the contingency occurs.
    This is the second level of the MultiIndex.
    """

    outage_group_id: Series[str] = pa.Field()
    """Identifier of the outage group shared by the contingency and element.

    Outage groups represent sets of elements that become de-energized together
    when separated from the rest of the network by circuit breakers.
    Multiple contingencies may map to the same outage group.
    """


class VADiffResultSchema(pa.DataFrameModel):
    """A schema for the voltage angle results.

    Holds information about the voltage angle difference between busses that could be (re)connected by power switches.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str] = pa.Field()
    """The element over which the voltage angle difference is computed. Can be either an open switch or any switch or branch
    under N-1. If under N-1, then element and contingency are the same."""

    va_diff: Series[float] = pa.Field(nullable=True)
    """The voltage angle difference in degrees between the two ends of the element.
    nan if at least one of the ends has no voltage angle (island, out of service)"""

    element_name: Series[str]
    """The name of the Branch or Switch, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class SwitchResultsSchema(pa.DataFrameModel):
    """A schema for the voltage angle results.

    Holds information about the voltage angle difference between busses that could be (re)connected by power switches.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str] = pa.Field()
    """The element over which the voltage angle difference is computed. Can be either an open switch or any switch or branch
    under N-1. If under N-1, then element and contingency are the same."""

    p: Series[float] = pa.Field(nullable=True)
    """The accumulated absolute active power at the node in MW, obtained by summing the absolute active power of all branches
    and injections connected to the node.

    If the engine does not support the computation of this value, the column can be omitted.
    """

    q: Series[float] = pa.Field(nullable=True)
    """The accumulated absolute reactive power at the node in MVar, obtained by summing the absolute reactive power of all
    branches and injections connected to the node

    If the engine does not support the computation of this value, the column can be omitted.
    """

    vm: Series[float] = pa.Field(nullable=True)
    """The voltage magnitude at the node in kV.

    In DC, this should be the nominal voltage of the node.
    This should only be NaN if the node does not have a connection to the slack bus.
    """

    i: Series[float] = pa.Field(nullable=True)
    """The current in the branch in A

    This should only be NaN if the branch has no connection to the slack bus.
    """

    element_name: Series[str]
    """The name of the Branch or Switch, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    side: Series[str] = pa.Field(nullable=True)
    """The measurement side of the switch result.

    - ``"from"``: values measured at the from-bus terminal (taken from ``net.res_switch``).
    - ``"to"``: values measured at the to-bus terminal (taken from ``net.res_switch``).
    - ``null``: result was computed by aggregating branch flows and node injections.
      Switches modelled without impedance have identical electrical conditions on both
      terminals (no voltage drop, no power loss across the switch), so a single
      aggregated value is sufficient and no side distinction is needed.
    """


class SwitchElementMappingSchema(pa.DataFrameModel):
    """Schema for mapping switches to connected elements.

    This table defines which elements are electrically connected to each switch.
    It is used to aggregate branch flows and node injections when computing
    switch-level results.

    The mapping includes both:
    - branch-like elements (lines, trafos, impedances, etc.)
    - buses

    If no switches are mapped, this is an empty DataFrame.
    """

    switch_id: Series[int] = pa.Field(nullable=False)
    """The pandapower index of the switch.

    This identifies the switch for which connected elements are collected and
    used in result aggregation.
    """

    element: Series[str] = pa.Field(nullable=False)
    """The globally unique identifier of the connected element.

    This can represent either:
    - a branch-like element (e.g. "12__line", "5__trafo")
    - a bus (e.g. "3__bus")
    """

    side: Series[float] = pa.Field(nullable=True)
    """The side of the branch element.

    - For branch-like elements:
        Indicates the terminal of the element connected to the bus
        (e.g. BranchSide.ONE, BranchSide.TWO, BranchSide.THREE).
    - For bus entries:
        This value is NaN, since buses do not have sides.
    """


class RegulatingElementResultSchema(pa.DataFrameModel):
    """A schema for the regulating elements.

    A regulating element can either be a branch (a trafo with regulating tap) or a node (a generator, SVC, ...).
    If no regulating elements are monitored, this is the empty DataFrame.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str] = pa.Field()
    """The regulating element that these loadflow results correspond to"""

    value: Series[float] = pa.Field()
    """The value of the regulating element. Depending on the type of the regulating element, this can mean different things.
    """

    regulating_element_type: Series[str] = pa.Field(isin=tuple(item.value for item in RegulatingElementType))
    """The type of the regulating element (generator, regulating transformer, SVC, ...)."""

    element_name: Series[str]
    """The name of the Regulating Element, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class ConvergedSchema(pa.DataFrameModel):
    """A schema for the converged table. This holds the convergence information for each timestep.

    Potentially, multiple islands can exist in the same grid. In this case, the synchronous component needs to be
    distinguished. A synchronous component is a grid area consisting of all nodes and branches that are connected to the same
    slack through AC lines (no HVDC). The largest component must always be called 'MAIN' while the names of the other
    components are arbitrary. Usually only one component is present.
    If no convergence information is available, this is the empty DataFrame.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str] = pa.Field()
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    status: Series[str] = pa.Field(isin=tuple(item.value for item in ConvergenceStatus))
    """Whether the loadflow converged at this timestep/contingency."""

    iteration_count: Series[float] = pa.Field(nullable=True)  # float so its nullable
    """The number of iterations required for the loadflow to converge."""

    warnings: Series[str] = pa.Field(nullable=True)
    """An additional string field that carries warnings or error logs for specific timesteps/contingencys/components."""

    contingency_name: Series[str]
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class SppsResultsSchema(pa.DataFrameModel):
    """SpPS run summaries, one row per (timestep, contingency).

    ``activated_schemes_per_iter`` holds a JSON string encoding of
    ``list[list[str]]`` (outer list = SpPS outer iterations, inner = scheme
    names that fired in that iteration), so the table remains parquet-friendly.
    Use ``json.loads`` to recover the nested structure.
    """

    timestep: Index[int] = pa.Field()
    """Loadflow timestep index for the outage that produced this row."""

    contingency: Index[str] = pa.Field()
    """Globally unique id of the contingency (same value as in other loadflow result tables)."""

    iterations: Series[int] = pa.Field()
    """Number of SpPS iterations that executed (1-based count)."""

    activated_schemes_per_iter: Series[str] = pa.Field()
    """JSON string for ``list[list[str]]`` of scheme names that activated per iteration."""

    max_iterations_reached: Series[bool] = pa.Field()
    """Whether the run stopped because the iteration cap was hit while schemes still fired."""

    power_flow_failed: Series[bool] = pa.Field()
    """Whether a post-action power flow failed in keep_previous mode."""


class CascadeResultSchema(pa.DataFrameModel):
    """A schema for cascade simulation event results.

    This table contains one row per cascade event created after a converged
    contingency load flow. It records why the cascade advanced and which
    element/outage group was affected.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of this cascade event."""

    contingency: Index[str] = pa.Field()
    """Globally unique id of the contingency that started the cascade."""

    cascade_number: Index[int] = pa.Field()
    """Cascade step number where the event happened."""

    element_mrid: Index[str] = pa.Field(nullable=True)
    """External identifier of the affected element, if known."""

    element_id: Series[str] = pa.Field(nullable=True)
    """Globally unique id of the affected element, if known."""

    contingency_outage_id: Series[str] = pa.Field(nullable=True)
    """Identifier of the outage group for the contingency that started the cascade."""

    contingency_name: Series[str] = pa.Field(nullable=True)
    """Human-readable name of the contingency that started the cascade, if known."""

    element_outage_group_id: Series[str] = pa.Field(nullable=True)
    """Stable identifier of the affected element's outage group."""

    element_name: Series[str] = pa.Field(nullable=True)
    """Human-readable name of the affected element, if known."""

    cascade_reason: Series[str] = pa.Field()
    """Reason for the cascade event, such as current overload or distance protection."""

    loading: Series[float] = pa.Field(nullable=True)
    """Branch loading value that triggered the cascade event, if available."""

    r_ohm: Series[float] = pa.Field(nullable=True)
    """Relay resistance value for distance-protection events, if available."""

    x_ohm: Series[float] = pa.Field(nullable=True)
    """Relay reactance value for distance-protection events, if available."""

    distance_protection_severity: Series[str] = pa.Field(nullable=True)
    """Distance-protection severity for relay events. Empty for other event types."""

    activated_schemes_per_iter: Series[str] = pa.Field(nullable=True)
    """JSON string of SpPS scheme names that activated per inner cascade load-flow iteration."""


class ControllerResultSchema(pa.DataFrameModel):
    """A schema for the rows station tap controllers record during a contingency analysis.

    Written into ``net.controller_data`` by every DiscreteTapControl (MCCS pandapower fork) that acts, and
    collected per outage by the contingency analysis. There are two kinds of row:

    - ``step``: one per control step, read *before* that step's tap move.
    - ``final``: one per ``run_control`` call in which the controller acted, holding the settled state
      after its last move.

    Every row type fills every column. Only the method names can be empty, and the step-specific
    ``proposed_increment``, ``actual_increment`` and ``controller_output`` are NaN on a final row.
    """

    timestep: Index[int] = pa.Field()
    """The timestep of the outage the row belongs to."""

    contingency: Index[str] = pa.Field()
    """Globally unique id of the contingency the row belongs to."""

    row_type: Series[str] = pa.Field(isin=["step", "final"])
    """``step`` for one control step, ``final`` for the settled state once the controller stopped."""

    element: Series[str] = pa.Field()
    """Pandapower table of the controlled transformer, e.g. ``trafo``."""

    tap_control_method: Series[str] = pa.Field()
    """How the controller picks its tap move, e.g. ``direct`` or ``tap_step``."""

    tap_step_method: Series[str] = pa.Field(nullable=True)
    """How a tap step maps to a voltage change, e.g. ``tap_step_percent`` or ``characteristic``."""

    tap_direction_method: Series[str] = pa.Field(nullable=True)
    """Which sign convention decides the tap direction, e.g. ``pandapower`` or ``powerfactory``."""

    controlled_bus: Series[int] = pa.Field()
    """Index of the bus whose voltage the controller regulates.

    This is the only column that tells controllers apart; transformers that regulate the same bus share it.
    """

    vn_kv: Series[float] = pa.Field(nullable=True)
    """Nominal voltage of the controlled bus in kV, NaN if the bus could not be read."""

    vm_set_pu: Series[float] = pa.Field()
    """Voltage setpoint of the controller in p.u."""

    vm_lower_pu: Series[float] = pa.Field()
    """Lower edge of the controller's deadband in p.u."""

    vm_upper_pu: Series[float] = pa.Field()
    """Upper edge of the controller's deadband in p.u."""

    tap_min: Series[float] = pa.Field()
    """Lowest tap position the transformer allows."""

    tap_max: Series[float] = pa.Field()
    """Highest tap position the transformer allows."""

    control_step: Series[int] = pa.Field(ge=1)
    """On a step row the number of this step, on a final row how many steps the controller took."""

    vm_start_pu: Series[float] = pa.Field()
    """Voltage at the controlled bus when the controller took its first step in this run, in p.u.

    For a controller that only starts moving in a later iteration this is the state it found then, not the
    outage before any controller acted.
    """

    vm_pu: Series[float] = pa.Field()
    """Voltage at the controlled bus in p.u.: before the tap move on a step row, settled on a final row."""

    in_band: Series[bool] = pa.Field()
    """Whether ``vm_pu`` lies inside the deadband."""

    voltage_violation: Series[float] = pa.Field(ge=0)
    """Distance of ``vm_pu`` past the nearer deadband edge in p.u., 0 inside the band."""

    voltage_region: Series[float] = pa.Field(isin=[-1.0, 0.0, 1.0])
    """Where ``vm_pu`` lies relative to the deadband: -1 below, 0 inside, +1 above."""

    tap_start: Series[float] = pa.Field()
    """Tap position when the controller took its first step in this run."""

    tap_pos_before: Series[float] = pa.Field()
    """Tap position before the move on a step row; on a final row the same as ``tap_start``."""

    tap_pos_after: Series[float] = pa.Field()
    """Tap position after the move on a step row; on a final row the position the controller ended at."""

    proposed_increment: Series[float] = pa.Field(nullable=True)
    """Tap change the control law asked for, before clipping to the tap range. NaN on a final row."""

    actual_increment: Series[float] = pa.Field(nullable=True)
    """Tap change actually applied after clipping to the tap range. NaN on a final row."""

    controller_output: Series[float] = pa.Field(nullable=True)
    """Raw output of the control law. NaN for methods without one and on a final row."""

    tap_limit: Series[str] = pa.Field(nullable=True, isin=["min", "max"])
    """``min`` or ``max`` when ``tap_pos_after`` sits at that end of the tap range, empty otherwise."""

    tap_limit_reached: Series[bool] = pa.Field()
    """Whether ``tap_pos_after`` sits at either end of the tap range."""

    total_tap_changes: Series[float] = pa.Field(ge=0)
    """Sum of the absolute tap changes the controller made in this run so far."""

    hunting_transition_detected: Series[bool] = pa.Field()
    """Whether the voltage crossed from one side of the deadband to the other at this point."""

    hunting_counter: Series[int] = pa.Field(ge=0)
    """How many such crossings the controller has counted in this run."""

    hunting_limit_reached: Series[bool] = pa.Field()
    """Whether the controller stopped because the hunting limit was reached."""

    hunting_detected: Series[bool] = pa.Field()
    """Whether the controller judged its taps to be oscillating."""


LoadflowResultTable = Union[
    pat.DataFrame[NodeResultSchema],
    pat.DataFrame[BranchResultSchema],
    pat.DataFrame[VADiffResultSchema],
    pat.DataFrame[ConnectivityResultSchema],
    pat.DataFrame[SwitchResultsSchema],
    pat.DataFrame[RegulatingElementResultSchema],
    pat.DataFrame[ConvergedSchema],
    pat.DataFrame[SppsResultsSchema],
    pat.DataFrame[CascadeResultSchema],
    pat.DataFrame[ControllerResultSchema],
]


class LoadflowResults(BaseModel):
    """A container for the loadflow results for a computation job."""

    job_id: str
    """The id of the computation job that created these loadflows"""

    branch_results: DataFrame[BranchResultSchema] = None
    """The results for the branches. If no branches are monitored, this is the empty DataFrame.
    Non converging contingencys/timesteps are to be omitted"""

    node_results: DataFrame[NodeResultSchema] = None
    """The results for the nodes. If no nodes are monitored, this is the empty DataFrame."""

    regulating_element_results: DataFrame[RegulatingElementResultSchema] = None
    """The results for the regulating elements. If no regulating elements are monitored, this is the empty DataFrame."""

    converged: DataFrame[ConvergedSchema] = None
    """The convergence information for each timestep and contingency.
    If there were non-converging loadflows for some timesteps/contingencys, these results should be omitted from the other
    tables.
    """

    va_diff_results: DataFrame[VADiffResultSchema] = None
    """The voltage angle difference results for each timestep and contingency.
    Considers the ends of the outaged branch, aswell as all open switches in monitored elements.
    """

    switch_results: DataFrame[SwitchResultsSchema] = None
    """The results for the switches.

    Contains aggregated power flow and injection results per switch for each
    timestep and contingency.

    Switch results are computed by aggregating contributions from all elements
    (branches and buses) electrically connected to one side of the switch.
    This represents the power flowing through the switch.

    If no switches are monitored, this is the empty DataFrame.
    For non-converging contingencies/timesteps, result values are present but set to NaN.
    """

    connectivity_result: DataFrame[ConnectivityResultSchema] = None
    """Connectivity mapping between contingencies and affected grid elements.
        This DataFrame defines which elements become unavailable for each contingency,
        based on outage group logic. Each row represents a (contingency, element) pair,
        indicating that the element is part of the outage group triggered by the
        contingency.
    """

    warnings: Optional[list[str]] = Field(default_factory=list)
    """Global warnings that occured during the computation (e.g. monitored elements/contingencies that were not found)"""

    result_filter: Optional[LoadflowResultFilter] = None
    """The filter policy these results were produced under, or None if they were not filtered.

    Rows the policy dropped are simply absent, so without this a reader cannot tell a contingency that stayed quiet from
    one that was never computed. It is carried through save/load and through conversion to and from polars.
    """

    spps_results: DataFrame[SppsResultsSchema] = None
    """SpPS run summaries, concatenated in single-outage order. When SpPS did not run for a case, that chunk
    contributes no rows. If no job recorded SpPS, this is the empty DataFrame
    (default)."""

    cascade_results: Optional[DataFrame[CascadeResultSchema]] = None
    """Cascade simulation events, one row per event. Empty when cascade simulation is disabled or has no events."""

    controller_results: Optional[DataFrame[ControllerResultSchema]] = None
    """Rows recorded by the station tap controllers, one per control step plus a final row,
    indexed by ``timestep`` and ``contingency``. None when nothing was recorded."""

    def __eq__(self, lf_result: object) -> bool:
        """Compare two LoadflowResults objects for equality.

        Rounds floats to 6 decimal places for comparison.
        This is necessary because floating point arithmetic can lead to small differences in the results.

        Ignores the order of the DataFrames, but checks that the indices are equal.

        Parameters
        ----------
        lf_result : LoadflowResults
            The LoadflowResults object to compare with.

        Returns
        -------
        bool
            True if the two LoadflowResults objects are equal, False otherwise.
        """
        rounding_accuracy = 6

        if not isinstance(lf_result, LoadflowResults):
            return False

        def required_frame_matches(left: pd.DataFrame, right: pd.DataFrame) -> bool:
            """Compare required result frames while ignoring row and column order."""
            if left.shape != right.shape:
                return False
            if not all(left.index.isin(right.index)):
                return False
            if not all(left.columns.isin(right.columns)):
                return False

            ordered_left = left.loc[right.index.drop_duplicates(), right.columns].round(rounding_accuracy)
            return ordered_left.equals(right.round(rounding_accuracy))

        def optional_frame_matches(left: pd.DataFrame | None, right: pd.DataFrame | None) -> bool:
            """Compare optional result frames while treating None and empty frames as equal."""
            if left is None or right is None:
                other = right if left is None else left
                return other is None or other.empty
            return required_frame_matches(left, right)

        return (
            self.job_id == lf_result.job_id
            and self.warnings == lf_result.warnings
            and required_frame_matches(self.branch_results, lf_result.branch_results)
            and required_frame_matches(self.node_results, lf_result.node_results)
            and required_frame_matches(self.regulating_element_results, lf_result.regulating_element_results)
            and required_frame_matches(self.va_diff_results, lf_result.va_diff_results)
            and required_frame_matches(self.converged, lf_result.converged)
            and optional_frame_matches(self.spps_results, lf_result.spps_results)
            and optional_frame_matches(self.cascade_results, lf_result.cascade_results)
            and optional_frame_matches(self.controller_results, lf_result.controller_results)
        )
