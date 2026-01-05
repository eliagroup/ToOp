# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides functions to compute the metrics directly from the results dataframes.

This is similar to jax.aggregate_results.py but straight on the results dataframes.
"""

import pandas as pd
import pandera as pa
import pandera.typing.polars as patpl
import polars as pl
from beartype.typing import Literal, Optional
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    LoadflowResultsPolars,
    VADiffResultSchemaPolars,
)
from toop_engine_interfaces.types import MetricType


def compute_overload_column(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars], field: Literal["p", "i"] = "i"
) -> pl.LazyFrame:
    """Compute the overload column for further aggregation.

    This is just a max operation

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    field : Literal["p", "i"], optional
        The field to use for the overload calculation, either "p" for power or "i" for current, by default "i".

    branch_results_with_overload : patpl.LazyFrame
    -------
        The branch results dataframe with an additional "overload" column.
    """
    branch_results_with_overload = branch_results.with_columns(
        _val_max=(pl.col(field) / pl.col("loading")).abs(),
    ).with_columns(
        overload=(pl.col(field).abs() - pl.col("_val_max")),
    )
    return branch_results_with_overload


@pa.check_types
def compute_max_load(branch_results: patpl.LazyFrame[BranchResultSchemaPolars]) -> float:
    """Compute the highest loading of the branches in the results.

    This is just a max operation

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.

    Returns
    -------
    float
        The maximum loading in factor of maximum rated current (percent / 100) of any branch in the results.
    """
    max_loading = branch_results.select(pl.col("loading").max()).collect().item()
    return max_loading


@pa.check_types
def compute_overload_energy(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars], field: Literal["p", "i"] = "i"
) -> float:
    """Compute the maximum overload current of the branches in the results.

    This is just a max operation

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    field : Literal["p", "i"], optional
        The field to use for the overload calculation, either "p" for power or "i" for current, by default "i".

    Returns
    -------
    float
        The maximum overload total current or power
    """
    branch_results_with_overload = compute_overload_column(branch_results, field=field)
    overload = (
        branch_results_with_overload.select("timestep", "element", "overload")
        .drop_nulls()
        .filter(pl.col("overload") > 0)
        .group_by(["timestep", "element"])
        .agg(pl.max("overload").alias("overload"))
        .drop_nans("overload")
        .select(pl.col("overload").sum())
        .collect()
        .item()
    )

    return overload


@pa.check_types
def count_critical_branches(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars], critical_threshold: float = 1.0
) -> int:
    """Count how many branches are above 100% in any side/contingency

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    critical_threshold : float, optional
        The loading threshold to consider a branch as critical, by default 1.0 (100%)

    Returns
    -------
    int
        The number of branches that are overloaded in any side/contingency.
    """
    # Do an any-aggregation across branch sides and contingencies (group by timestep/element)
    # This will return a boolean series with True for each branch that is overloaded in any contingency/side
    # Summing this will give the count of critical branches
    return int(
        branch_results.filter(pl.col("loading").fill_nan(-1.0) > critical_threshold)
        .select("timestep", "element")
        .unique()
        .select(pl.len())
        .collect()
        .item()
    )


def compute_max_va_diff(va_diff_results: patpl.LazyFrame[VADiffResultSchemaPolars]) -> float:
    """Compute the maximum voltage angle difference.

    Parameters
    ----------
    va_diff_results : patpl.LazyFrame[VADiffResultSchemaPolars]
        The voltage angle difference results dataframe.

    Returns
    -------
    float
        The maximum voltage angle difference in degrees.
    """
    max_va_diff = va_diff_results.select(pl.col("va_diff").max()).collect().item()
    if max_va_diff is None or pd.isna(max_va_diff):
        return 0.0
    return float(max_va_diff)


def get_worst_k_contingencies_ac(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars],
    k: int = 10,
    field: Literal["p", "i"] = "p",
    base_case_id: str = "BASECASE",
) -> tuple[list[list[str]], list[float]]:
    """Get the worst k contingencies based on overload energy.

    If k is greater than the number of contingencies, all contingencies will be returned.

    Parameters
    ----------
    branch_results : DataFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    k : int, optional
        The number of worst contingencies to return, by default 10.
    field : Literal["p", "i"], optional
        The field to use for the overload calculation, either "p" for power or "i" for current, by default "p".
    base_case_id : str, optional
        The contingency ID for the base case (N-0), by default "BASECASE".

    Returns
    -------
    tuple[list[list[str]], list[float]]
        A tuple containing:
        - A list of lists with the contingency IDs for each timestep. The length of the outer list is
        the number of timesteps while the inner lists contain the top k contingencies for that timestep.
        - A list of total overload energy for each timestep. The length matches the number of timesteps.
    """
    branch_results_with_overload = compute_overload_column(branch_results, field=field).drop_nans("overload")
    overload = branch_results_with_overload.filter(pl.col("overload") > 0)
    overload_n1 = overload.filter(pl.col("contingency") != base_case_id)
    # Compute per (timestep, contingency) max overload using polars lazy API
    overload_per_cont = overload_n1.group_by(["timestep", "contingency"]).agg(pl.max("overload").alias("overload")).collect()

    if overload_per_cont.height == 0:
        return [], []

    contingencies: list[list[str]] = []
    overloads: list[float] = []

    for t in overload_per_cont.get_column("timestep").unique().to_list():
        df_t = overload_per_cont.filter(pl.col("timestep") == t).sort("overload", descending=True).head(k)
        cont_ids = df_t.get_column("contingency").to_list()
        contingencies.append(cont_ids)

        if cont_ids:
            br_results_top_k = branch_results.filter(pl.col("contingency").is_in(cont_ids))
            overload_top_k = compute_overload_energy(br_results_top_k, field=field)
        else:
            overload_top_k = 0.0
        overloads.append(float(overload_top_k))

    return contingencies, overloads


def compute_metrics(
    loadflow_results: LoadflowResultsPolars,
    base_case_id: Optional[str] = None,
) -> dict[MetricType, float]:
    """Compute the metrics from the loadflow results.

    Parameters
    ----------
    loadflow_results : LoadflowResultsPolars
        The loadflow results containing the branch results.
    base_case_id : Optional[str], optional
        The contingency ID for the base case (N-0). If not provided, no n-0 metrics will be computed.

    Returns
    -------
    dict[MetricType, float]
        A dictionary with the computed metrics.
    """
    metrics = {
        "max_flow_n_1": compute_max_load(loadflow_results.branch_results),
        "overload_energy_n_1": compute_overload_energy(loadflow_results.branch_results, field="p"),
        "max_va_diff_n_1": compute_max_va_diff(loadflow_results.va_diff_results),
        "overload_current_n_1": compute_overload_energy(loadflow_results.branch_results, field="i"),
        "critical_branch_count_n_1": count_critical_branches(loadflow_results.branch_results),
    }

    if base_case_id is not None:
        # Base case (N-0) results as Polars LazyFrames
        n_0_branch_res = loadflow_results.branch_results.filter(pl.col("contingency") == base_case_id)

        # Va diff may not contain the base case; filtering will yield an empty frame if absent.
        # compute_max_va_diff already returns 0.0 if empty/None, so no explicit fallback needed.
        n_0_va_diff = loadflow_results.va_diff_results.filter(pl.col("contingency") == base_case_id)
        metrics.update(
            {
                "max_flow_n_0": compute_max_load(n_0_branch_res),
                "overload_energy_n_0": compute_overload_energy(n_0_branch_res, field="p"),
                "max_va_diff_n_0": compute_max_va_diff(n_0_va_diff),
                "overload_current_n_0": compute_overload_energy(n_0_branch_res, field="i"),
                "critical_branch_count_n_0": count_critical_branches(n_0_branch_res),
            }
        )
    return metrics
