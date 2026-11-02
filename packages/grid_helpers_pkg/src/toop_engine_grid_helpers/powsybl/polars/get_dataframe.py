# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""PyPowsybl Polars functions

The Pypowsybl functions can be slow, once you have millions of rows.
This package aims to improve the performance of these functions by
leveraging the Polars library for DataFrame operations.

"""

from beartype.typing import Union
from polars import DataFrame, LazyFrame
from pypowsybl import _pypowsybl
from pypowsybl._pypowsybl import JavaHandle
from pypowsybl.security import SecurityAnalysisResult

#: A finished security analysis: either the pypowsybl result wrapper or the bare native handle it wraps.
#:
#: The wrapper builds a pandas multi-index violation table in its constructor, so callers that only need polars frames
#: can keep the native handle of ``_pypowsybl.run_security_analysis`` instead and pass it here.
SecurityAnalysisHandle = Union[SecurityAnalysisResult, JavaHandle]


def _to_handle(security_analysis: SecurityAnalysisHandle) -> JavaHandle:
    """Return the native handle of a finished security analysis.

    Parameters
    ----------
    security_analysis : SecurityAnalysisHandle
        The security analysis result wrapper or its native handle.

    Returns
    -------
    JavaHandle
        The native handle the result frames are read from.
    """
    if isinstance(security_analysis, SecurityAnalysisResult):
        return security_analysis._handle
    return security_analysis


def get_df_from_series_array(series_array: _pypowsybl.SeriesArray, lazy: bool = True) -> Union[DataFrame, LazyFrame]:
    """Convert a PyPowSyBl SeriesArray to a Polars DataFrame.

    Use the java handle to get the _pypowsybl.SeriesArray, e.g.

    Parameters
    ----------
    series_array : _pypowsybl.SeriesArray
        The SeriesArray to convert.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the data from the SeriesArray.

    Example
    -------
    analysis = pypowsybl.security.create_analysis()
    security_analysis_result = analysis.run_ac(net)
    branch_results_series = _pypowsybl.get_branch_results(security_analysis_result._handle)
    bus_results_series = _pypowsybl.get_bus_results(security_analysis_result._handle)

    """
    data = {series.name: series.data for series in series_array}

    if not lazy:
        df = DataFrame(data)
    else:
        df = LazyFrame(data)
    return df


def get_ca_branch_results(
    security_analysis_result: SecurityAnalysisHandle, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Get the branch results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisHandle
        The contingency analysis result to get the branch results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the branch results.
    """
    return get_df_from_series_array(_pypowsybl.get_branch_results(_to_handle(security_analysis_result)), lazy=lazy)


def get_ca_bus_results(security_analysis_result: SecurityAnalysisHandle, lazy: bool = True) -> Union[DataFrame, LazyFrame]:
    """Get the bus results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisHandle
        The contingency analysis result to get the bus results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the bus results.
    """
    return get_df_from_series_array(_pypowsybl.get_bus_results(_to_handle(security_analysis_result)), lazy=lazy)


def get_ca_three_windings_transformer_results(
    security_analysis_result: SecurityAnalysisHandle, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Get the three windings transformer results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisHandle
        The contingency analysis result to get the three windings transformer results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the three windings transformer results.
    """
    return get_df_from_series_array(
        _pypowsybl.get_three_windings_transformer_results(_to_handle(security_analysis_result)), lazy=lazy
    )


def get_ca_limit_violations(
    security_analysis_result: SecurityAnalysisHandle, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Get the limit violations from the contingency analysis result.

    One row per violated limit, pre-contingency rows carry an empty contingency id. Unlike the ``limit_violations``
    property of the pypowsybl result, this never builds a pandas multi-index.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisHandle
        The contingency analysis result to get the limit violations from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the limit violations with the columns contingency_id, subject_id, subject_name,
        limit_type, limit_name, limit, acceptable_duration, limit_reduction, value and side.
    """
    return get_df_from_series_array(_pypowsybl.get_limit_violations(_to_handle(security_analysis_result)), lazy=lazy)
