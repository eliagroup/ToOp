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

import json
import tempfile
import uuid
from pathlib import Path

import polars as pl
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


def get_ca_connectivity_results(
    security_analysis_result: SecurityAnalysisResult, lazy: bool = True
) -> Union[DataFrame, LazyFrame]:
    """Build contingency-to-disconnected-element mapping from powsybl security analysis results.

    Parameters
    ----------
    security_analysis_result : SecurityAnalysisResult
        The contingency analysis result to extract propagated disconnected elements from.
    lazy : bool
        Whether to return a LazyFrame instead of a DataFrame.

    Returns
    -------
    Union[DataFrame, LazyFrame]
        A dataframe with columns contingency, element and outage_group_id.
    """
    records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = Path(temp_dir) / "security_analysis_result.json"
        security_analysis_result.export_to_json(str(export_path))
        with export_path.open("r", encoding="utf-8") as file:
            exported_result = json.load(file)

    post_contingency_results = exported_result.get("postContingencyResults", [])
    for post_result in post_contingency_results:
        connectivity_result = post_result.get("connectivityResult", {})
        disconnected_elements = connectivity_result.get("disconnectedElements", [])
        if not disconnected_elements:
            continue
        outage_group_id = str(uuid.uuid4())
        contingency = post_result.get("contingency", {})
        contingency_id = contingency.get("id")
        if contingency_id is None:
            continue
        for element_id in disconnected_elements:
            records.append(
                {
                    "contingency": contingency_id,
                    "element": element_id,
                    "outage_group_id": outage_group_id,
                }
            )

    df = pl.DataFrame(records, schema={"contingency": pl.String, "element": pl.String, "outage_group_id": pl.String})
    return df.lazy() if lazy else df
