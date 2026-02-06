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
from beartype.typing import Any, Self, Union
from polars.testing import assert_frame_equal
from pydantic import BaseModel, Field
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    ConvergedSchema,
    NodeResultSchema,
    RegulatingElementResultSchema,
    VADiffResultSchema,
)


class BranchResultSchemaPolars(pal.DataFrameModel, BranchResultSchema):
    """Polars variant of BranchResultSchema."""

    pass


class NodeResultSchemaPolars(pal.DataFrameModel, NodeResultSchema):
    """Polars variant of NodeResultSchema."""

    pass


class VADiffResultSchemaPolars(pal.DataFrameModel, VADiffResultSchema):
    """Polars variant of VADiffResultSchema."""

    pass


class RegulatingElementResultSchemaPolars(pal.DataFrameModel, RegulatingElementResultSchema):
    """Polars variant of RegulatingElementResultSchema."""

    pass


class ConvergedSchemaPolars(pal.DataFrameModel, ConvergedSchema):
    """Polars variant of ConvergedSchema."""

    pass


LoadflowResultTablePolars = Union[
    patpl.LazyFrame[NodeResultSchemaPolars],
    patpl.LazyFrame[BranchResultSchemaPolars],
    patpl.LazyFrame[VADiffResultSchemaPolars],
    patpl.LazyFrame[RegulatingElementResultSchemaPolars],
    patpl.LazyFrame[ConvergedSchemaPolars],
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

    warnings: list[str] = Field(default_factory=list)
    """Global warnings that occured during the computation (e.g. monitored elements/contingencies that were not found)"""

    additional_information: list[Any] = Field(default_factory=list)
    """Additional information that the loadflow solver wants to convey to the user. There is no limitation what can
    be put in here except that it needs to be json serializable."""

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
        additional_info_match = self.additional_information == lf_result.additional_information
        simple_checks = job_match and warnings_match and additional_info_match
        if not simple_checks:
            return False

        kw_args_testing = {
            "check_row_order": False,
            "check_column_order": False,
            "check_dtypes": True,
            "check_exact": False,
            "abs_tol": rounding_accuracy,
        }
        try:
            assert_frame_equal(self.branch_results, lf_result.branch_results, **kw_args_testing)
            assert_frame_equal(self.node_results, lf_result.node_results, **kw_args_testing)
            assert_frame_equal(self.regulating_element_results, lf_result.regulating_element_results, **kw_args_testing)
            assert_frame_equal(self.va_diff_results, lf_result.va_diff_results, **kw_args_testing)
            assert_frame_equal(self.converged, lf_result.converged, **kw_args_testing)
        except AssertionError:
            return False

        return True
