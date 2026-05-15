# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines performance-improved polars versions of the loadflow results.

The loadflow results here mirror what is defined in loadflow_results.py, but use polars dataframes which are faster.
"""

import pandera.polars as pal
import pandera.typing.polars as patpl
import polars as pl
from beartype.typing import Self, Union
from polars.testing import assert_frame_equal
from pydantic import BaseModel, Field
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    CascadeResultSchema,
    ConnectivityResultSchema,
    ConvergedSchema,
    NodeResultSchema,
    RegulatingElementResultSchema,
    SppsResultsSchema,
    SwitchResultsSchema,
    VADiffResultSchema,
)


class BranchResultSchemaPolars(pal.DataFrameModel, BranchResultSchema):
    """Polars variant of BranchResultSchema."""

    pass


class NodeResultSchemaPolars(pal.DataFrameModel, NodeResultSchema):
    """Polars variant of NodeResultSchema."""

    pass


class ConnectivityResultSchemaPolars(pal.DataFrameModel, ConnectivityResultSchema):
    """Polars variant of ConnectivityResultSchema."""

    pass


class VADiffResultSchemaPolars(pal.DataFrameModel, VADiffResultSchema):
    """Polars variant of VADiffResultSchema."""

    pass


class SwitchResultsSchemaPolars(pal.DataFrameModel, SwitchResultsSchema):
    """Polars variant of SwitchResultsSchema."""

    pass


class RegulatingElementResultSchemaPolars(pal.DataFrameModel, RegulatingElementResultSchema):
    """Polars variant of RegulatingElementResultSchema."""

    pass


class ConvergedSchemaPolars(pal.DataFrameModel, ConvergedSchema):
    """Polars variant of ConvergedSchema."""

    pass


class SppsResultsSchemaPolars(pal.DataFrameModel, SppsResultsSchema):
    """Polars variant of SppsResultsSchema."""

    pass


class CascadeResultSchemaPolars(pal.DataFrameModel, CascadeResultSchema):
    """Polars variant of CascadeResultSchema."""

    pass


LoadflowResultTablePolars = Union[
    patpl.LazyFrame[NodeResultSchemaPolars],
    patpl.LazyFrame[BranchResultSchemaPolars],
    patpl.LazyFrame[VADiffResultSchemaPolars],
    patpl.LazyFrame[ConnectivityResultSchemaPolars],
    patpl.LazyFrame[SwitchResultsSchemaPolars],
    patpl.LazyFrame[RegulatingElementResultSchemaPolars],
    patpl.LazyFrame[ConvergedSchemaPolars],
    patpl.LazyFrame[SppsResultsSchemaPolars],
    patpl.LazyFrame[CascadeResultSchemaPolars],
]


class LoadflowResultsPolars(BaseModel):
    """A container for the loadflow results for a computation job."""

    job_id: str
    """The id of the computation job that created these loadflows"""

    branch_results: Union[patpl.LazyFrame[BranchResultSchemaPolars], pl.LazyFrame] = None
    """The results for the branches. If no branches are monitored, this is the empty DataFrame.
    Non converging contingencys/timesteps are to be omitted"""

    node_results: Union[patpl.LazyFrame[NodeResultSchemaPolars], pl.LazyFrame] = None
    """The results for the nodes. If no nodes are monitored, this is the empty DataFrame."""

    regulating_element_results: Union[patpl.LazyFrame[RegulatingElementResultSchemaPolars], pl.LazyFrame] = None
    """The results for the regulating elements. If no regulating elements are monitored, this is the empty DataFrame."""

    converged: Union[patpl.LazyFrame[ConvergedSchemaPolars], pl.LazyFrame] = None
    """The convergence information for each timestep and contingency.
    If there were non-converging loadflows for some timesteps/contingencys, these results should be omitted from the other
    tables.
    """

    va_diff_results: Union[patpl.LazyFrame[VADiffResultSchemaPolars], pl.LazyFrame] = None
    """The voltage angle difference results for each timestep and contingency.
    Considers the ends of the outaged branch, aswell as all open switches in monitored elements.
    """

    switch_results: Union[patpl.LazyFrame[SwitchResultsSchemaPolars], pl.LazyFrame, None] = None
    """The results for the switches.

    Contains aggregated power flow and injection results per switch for each
    timestep and contingency.

    Switch results are computed by aggregating contributions from all elements
    (branches and buses) electrically connected to one side of the switch.
    This represents the power flowing through the switch.

    If no switches are monitored, this is the empty DataFrame.
    For non-converging contingencies/timesteps, result values are present but set to NaN.
    """

    connectivity_result: Union[patpl.LazyFrame[ConnectivityResultSchemaPolars], pl.LazyFrame, None] = None
    """Connectivity mapping between contingencies and affected grid elements.
        This DataFrame defines which elements become unavailable for each contingency,
        based on outage group logic. Each row represents a (contingency, element) pair,
        indicating that the element is part of the outage group triggered by the
        contingency.
    """

    warnings: list[str] = Field(default_factory=list)
    """Global warnings that occured during the computation (e.g. monitored elements/contingencies that were not found)"""

    spps_results: Union[patpl.LazyFrame[SppsResultsSchemaPolars], pl.LazyFrame] = None
    """SpPS run summaries, concatenated in single-outage order. Empty when no
    SpPS was recorded (default)."""

    cascade_results: Union[patpl.LazyFrame[CascadeResultSchemaPolars], pl.LazyFrame, None] = None
    """Cascade simulation events. Empty when cascade simulation is disabled or has no events."""

    class Config:
        """Pydantic configuration for the LoadflowResultsPolars model."""

        arbitrary_types_allowed = True
        """Allow arbitrary types in the model."""

    def __eq__(self, lf_result: Self) -> bool:
        """Compare two LoadflowResults objects for equality.

        Rounds floats to 6 decimal places for comparison.
        This is necessary because floating point arithmetic can lead to small differences in the results.

        Ignores the order of the DataFrames, but checks that the indices are equal.

        Note: This functions is not very efficient and can take up to half a minute for >10Mio rows.

        Parameters
        ----------
        lf_result : LoadflowResults
            The LoadflowResults object to compare with.

        Returns
        -------
        bool
            True if the two LoadflowResults objects are equal, False otherwise.
        """
        rounding_accuracy = 1e-6

        if not isinstance(lf_result, LoadflowResultsPolars):
            return False

        job_match = self.job_id == lf_result.job_id
        warnings_match = self.warnings == lf_result.warnings
        simple_checks = job_match and warnings_match
        if not simple_checks:
            return False

        kw_args_testing = {
            "check_row_order": False,
            "check_column_order": False,
            "check_dtypes": True,
            "check_exact": False,
            "abs_tol": rounding_accuracy,
        }

        def assert_optional_frame_equal(left: pl.LazyFrame | None, right: pl.LazyFrame | None) -> None:
            """Assert that optional polars frames are equal.

            Parameters
            ----------
            left : pl.LazyFrame | None
                First optional frame.
            right : pl.LazyFrame | None
                Second optional frame.

            Raises
            ------
            AssertionError
                If one frame is None and the other is not, or if frame contents differ.
            """
            if left is None or right is None:
                if left is not right:
                    raise AssertionError("One frame is None and the other is not.")
                return
            assert_frame_equal(left, right, **kw_args_testing)

        try:
            assert_optional_frame_equal(self.branch_results, lf_result.branch_results)
            assert_optional_frame_equal(self.node_results, lf_result.node_results)
            assert_optional_frame_equal(self.regulating_element_results, lf_result.regulating_element_results)
            assert_optional_frame_equal(self.va_diff_results, lf_result.va_diff_results)
            assert_optional_frame_equal(self.converged, lf_result.converged)
            assert_optional_frame_equal(self.spps_results, lf_result.spps_results)
            assert_optional_frame_equal(self.cascade_results, lf_result.cascade_results)
        except AssertionError:
            return False

        return True
