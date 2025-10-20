"""PyPowsybl Polars functions

The Pypowsybl functions can be slow, once you have millions of rows.
This package aims to improve the performance of these functions by
leveraging the Polars library for DataFrame operations.

"""

from beartype.typing import Union
from polars import DataFrame, LazyFrame
from pypowsybl import _pypowsybl
from pypowsybl.security import SecurityAnalysisResult


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
    security_analysis_result: SecurityAnalysisResult, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Get the branch results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisResult
        The contingency analysis result to get the branch results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the branch results.
    """
    return get_df_from_series_array(_pypowsybl.get_branch_results(security_analysis_result._handle), lazy=lazy)


def get_ca_bus_results(security_analysis_result: SecurityAnalysisResult, lazy: bool = True) -> Union[DataFrame, LazyFrame]:
    """Get the bus results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisResult
        The contingency analysis result to get the bus results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the bus results.
    """
    return get_df_from_series_array(_pypowsybl.get_bus_results(security_analysis_result._handle), lazy=lazy)


def get_ca_three_windings_transformer_results(
    security_analysis_result: SecurityAnalysisResult, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Get the three windings transformer results from the contingency analysis result.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisResult
        The contingency analysis result to get the three windings transformer results from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame. Default is True.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A Polars DataFrame containing the three windings transformer results.
    """
    return get_df_from_series_array(
        _pypowsybl.get_three_windings_transformer_results(security_analysis_result._handle), lazy=lazy
    )
