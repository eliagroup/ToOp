# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
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

import pandera as pa
import pandera.typing as pat
from beartype.typing import Any, Optional, Self, Union
from pandera.typing import DataFrame, Index, Series
from pydantic import BaseModel, Field


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

    timestep: Index[int]
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str]
    """The contingency that caused this loadflow. For N-0 results,
    the special CO 'BASECASE' without GridElements is used, if its added."""

    element: Index[str]
    """The branch that these loadflow results correspond to"""

    side: Index[int] = pa.Field(isin=[side.value for side in BranchSide])
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

    element_name: Series[str] = pa.Field(default="")
    """The name of the Branch, if available. This is not used for the loadflow computation, but can be used for display
    purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str] = pa.Field(default="")
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class NodeResultSchema(pa.DataFrameModel):
    """A schema for the node results table.

    This holds p and q values for all monitored nodes with a multi-index of timestep and contingency.
    If no nodes are monitored, this is the empty DataFrame.
    """

    timestep: Index[int]
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str]
    """The contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str]
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

    element_name: Series[str] = pa.Field(default="", nullable=True)
    """The name of the node, if available. This is not used for the loadflow computation, but can be used for display
    purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str] = pa.Field(default="", nullable=True)
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class VADiffResultSchema(pa.DataFrameModel):
    """A schema for the voltage angle results.

    Holds information about the voltage angle difference between busses that could be (re)connected by power switches.
    """

    timestep: Index[int]
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str]
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str]
    """The element over which the voltage angle difference is computed. Can be either an open switch or any switch or branch
    under N-1. If under N-1, then element and contingency are the same."""

    va_diff: Series[float] = pa.Field(nullable=True)
    """The voltage angle difference in degrees between the two ends of the element.
    nan if at least one of the ends has no voltage angle (island, out of service)"""

    element_name: Series[str] = pa.Field(default="")
    """The name of the Branch or Switch, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Series[str] = pa.Field(default="")
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


class RegulatingElementResultSchema(pa.DataFrameModel):
    """A schema for the regulating elements.

    A regulating element can either be a branch (a trafo with regulating tap) or a node (a generator, SVC, ...).
    If no regulating elements are monitored, this is the empty DataFrame.
    """

    timestep: Index[int]
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str]
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    element: Index[str]
    """The regulating element that these loadflow results correspond to"""

    value: Series[float]
    """The value of the regulating element. Depending on the type of the regulating element, this can mean different things.
    """

    regulating_element_type: Series[str] = pa.Field(isin=[side.value for side in RegulatingElementType])
    """The type of the regulating element (generator, regulating transformer, SVC, ...)."""

    element_name: Optional[Series[str]] = pa.Field(default="")
    """The name of the Regulating Element, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """
    contingency_name: Optional[Series[str]] = pa.Field(default="")
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

    timestep: Index[int]
    """The timestep of this result. This indexes into the timesteps that were loaded"""

    contingency: Index[str]
    """The critical contingency that caused this loadflow. For N-0 results, the special CO 'BASECASE' is used."""

    status: Series[str] = pa.Field(isin=[side.value for side in ConvergenceStatus])
    """Whether the loadflow converged at this timestep/contingency."""

    iteration_count: Series[float] = pa.Field(nullable=True)  # float so its nullable
    """The number of iterations required for the loadflow to converge."""

    warnings: Series[str] = pa.Field(default="")
    """An additional string field that carries warnings or error logs for specific timesteps/contingencys/components."""

    contingency_name: Series[str] = pa.Field(default="")
    """The name of the contingency, if available. This is not used for the loadflow computation,
    but can be used for display purposes. If no name is available, this should be set to an empty string.
    """


LoadflowResultTable = Union[
    pat.DataFrame[NodeResultSchema],
    pat.DataFrame[BranchResultSchema],
    pat.DataFrame[VADiffResultSchema],
    pat.DataFrame[RegulatingElementResultSchema],
    pat.DataFrame[ConvergedSchema],
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

    warnings: Optional[list[str]] = Field(default_factory=list)
    """Global warnings that occured during the computation (e.g. monitored elements/contingencies that were not found)"""

    additional_information: Optional[list[Any]] = Field(default_factory=list)
    """Additional information that the loadflow solver wants to convey to the user. There is no limitation what can
    be put in here except that it needs to be json serializable."""

    def __eq__(self, lf_result: Self) -> bool:
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

        job_match = self.job_id == lf_result.job_id
        warnings_match = self.warnings == lf_result.warnings
        additional_info_match = self.additional_information == lf_result.additional_information
        simple_checks = job_match and warnings_match and additional_info_match

        # Check shape
        branch_shape_match = self.branch_results.shape == lf_result.branch_results.shape
        node_shape_match = self.node_results.shape == lf_result.node_results.shape
        regulating_element_shape_match = self.regulating_element_results.shape == lf_result.regulating_element_results.shape
        va_diff_shape_match = self.va_diff_results.shape == lf_result.va_diff_results.shape
        converged_shape_match = self.converged.shape == lf_result.converged.shape
        shape_matches = (
            branch_shape_match
            and node_shape_match
            and regulating_element_shape_match
            and va_diff_shape_match
            and converged_shape_match
        )
        if not (shape_matches and simple_checks):
            return False

        # Check indices. One way is enough since the lengths are equal
        node_indizes_match = all(self.node_results.index.isin(lf_result.node_results.index))
        branch_indizes_match = all(self.branch_results.index.isin(lf_result.branch_results.index))
        regulating_element_indizes_match = all(
            self.regulating_element_results.index.isin(lf_result.regulating_element_results.index)
        )
        va_diff_indizes_match = all(self.va_diff_results.index.isin(lf_result.va_diff_results.index))
        converged_indizes_match = all(self.converged.index.isin(lf_result.converged.index))
        indices_match = (
            node_indizes_match
            and branch_indizes_match
            and regulating_element_indizes_match
            and va_diff_indizes_match
            and converged_indizes_match
        )
        if not indices_match:
            return False

        node_columns_match = all(self.node_results.columns.isin(lf_result.node_results.columns))
        branch_columns_match = all(self.branch_results.columns.isin(lf_result.branch_results.columns))
        regulating_element_columns_match = all(
            self.regulating_element_results.columns.isin(lf_result.regulating_element_results.columns)
        )
        va_diff_columns_match = all(self.va_diff_results.columns.isin(lf_result.va_diff_results.columns))
        converged_columns_match = all(self.converged.columns.isin(lf_result.converged.columns))
        columns_match = (
            node_columns_match
            and branch_columns_match
            and regulating_element_columns_match
            and va_diff_columns_match
            and converged_columns_match
        )
        if not columns_match:
            return False

        # Check values
        own_node_results = self.node_results.loc[
            lf_result.node_results.index.drop_duplicates(), lf_result.node_results.columns
        ].round(rounding_accuracy)
        node_values_match = own_node_results.equals(lf_result.node_results.round(rounding_accuracy))
        own_branch_results = self.branch_results.loc[
            lf_result.branch_results.index.drop_duplicates(), lf_result.branch_results.columns
        ].round(rounding_accuracy)
        branch_values_match = own_branch_results.equals(lf_result.branch_results.round(rounding_accuracy))

        own_regulating_element_results = self.regulating_element_results.loc[
            lf_result.regulating_element_results.index.drop_duplicates(), lf_result.regulating_element_results.columns
        ].round(rounding_accuracy)
        regulating_element_values_match = own_regulating_element_results.equals(
            lf_result.regulating_element_results.round(rounding_accuracy)
        )

        own_va_diff_results = self.va_diff_results.loc[
            lf_result.va_diff_results.index.drop_duplicates(), lf_result.va_diff_results.columns
        ].round(rounding_accuracy)
        va_diff_values_match = own_va_diff_results.equals(lf_result.va_diff_results.round(rounding_accuracy))

        own_converged = self.converged.loc[lf_result.converged.index.drop_duplicates(), lf_result.converged.columns].round(
            rounding_accuracy
        )
        converged_values_match = own_converged.equals(lf_result.converged.round(rounding_accuracy))
        values_match = (
            node_values_match
            and branch_values_match
            and regulating_element_values_match
            and va_diff_values_match
            and converged_values_match
        )
        if not values_match:
            return False
        return True
