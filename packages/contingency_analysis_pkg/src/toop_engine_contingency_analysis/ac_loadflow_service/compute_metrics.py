# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
    NodeResultSchemaPolars,
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
def compute_max_load(branch_results: patpl.LazyFrame[BranchResultSchemaPolars]) -> float | None:
    """Compute the highest loading of the branches in the results.

    This is just a max operation

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.

    Returns
    -------
    float | None
        The maximum loading in factor of maximum rated current (percent / 100) of any branch in the results.
        None if the loading column is missing or if there are no valid loading values (e.g., all NaN).
    """
    max_loading = branch_results.select(pl.col("loading").max()).collect().item()
    return max_loading


@pa.check_types
def compute_overload_energy(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars], field: Literal["p", "i"] = "i"
) -> float | None:
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
    float | None
        The maximum overload total current or power, or None if there are no valid overload values.
    """
    branch_results_with_overload = compute_overload_column(branch_results, field=field)
    overload = (
        branch_results_with_overload.select("timestep", "element", "overload")
        .drop_nans("overload")
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
) -> int | None:
    """Count how many branches are above 100% in any side/contingency

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    critical_threshold : float, optional
        The loading threshold to consider a branch as critical, by default 1.0 (100%)

    Returns
    -------
    int | None
        The number of branches that are overloaded in any side/contingency, or None if there are no valid loading values.
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


def compute_max_va_diff(va_diff_results: patpl.LazyFrame[VADiffResultSchemaPolars]) -> float | None:
    """Compute the maximum absolute voltage angle difference.

    Parameters
    ----------
    va_diff_results : patpl.LazyFrame[VADiffResultSchemaPolars]
        The voltage angle difference results dataframe.

    Returns
    -------
    float | None
        The maximum absolute voltage angle difference in degrees, or None if there are no valid values.
    """
    max_va_diff = va_diff_results.select(pl.col("va_diff").abs().max()).collect().item()
    if max_va_diff is None or pd.isna(max_va_diff):
        return None
    return float(max_va_diff)


def count_critical_va_diff_cases(
    va_diff_results: patpl.LazyFrame[VADiffResultSchemaPolars],
    critical_threshold: float = 0.0,
) -> int | None:
    """Count how many monitored elements have a maximum absolute voltage angle difference above a threshold.

    Parameters
    ----------
    va_diff_results : patpl.LazyFrame[VADiffResultSchemaPolars]
        The voltage angle difference results dataframe.
    critical_threshold : float, optional
        The threshold in degrees above which a `(timestep, element)` maximum is counted, by default 0.0.

    Returns
    -------
    int | None
        The number of `(timestep, element)` groups whose maximum absolute voltage angle difference across contingencies
        is strictly above the threshold, or None if the required columns are not available.
    """
    required_columns = {"timestep", "element", "va_diff"}
    if not required_columns.issubset(va_diff_results.collect_schema().names()):
        return None

    return int(
        va_diff_results.drop_nans("va_diff")
        .group_by(["timestep", "element"])
        .agg(pl.col("va_diff").abs().max().alias("max_va_diff"))
        .filter(pl.col("max_va_diff") > critical_threshold)
        .select(pl.len())
        .collect()
        .item()
    )


def count_voltage_jumps(
    node_results: patpl.LazyFrame[NodeResultSchemaPolars],
    base_case_id: Optional[str],
    jump_threshold_percent: float = 5.0,
) -> int | None:
    """Count nodal voltage jumps above the basecase deviation threshold.

    Parameters
    ----------
    node_results : patpl.LazyFrame[NodeResultSchemaPolars]
        The node results dataframe containing both basecase and N-1 node voltages.
    base_case_id : Optional[str]
        The contingency id of the base case. If None, the voltage jump metric cannot be computed.
    jump_threshold_percent : float, optional
        The minimum relative voltage jump in percent that counts as a voltage jump, by default 5.0.

    Returns
    -------
    int | None
        The number of N-1 node results whose voltage deviation from the basecase is strictly above the threshold.
        None if the required columns are not available or if no base case id is provided.
    """
    if base_case_id is None:
        return None

    required_columns = {"timestep", "contingency", "element", "vm"}
    if not required_columns.issubset(node_results.collect_schema().names()):
        return None

    basecase_vm = (
        node_results.filter(pl.col("contingency") == base_case_id)
        .select("timestep", "element", pl.col("vm").alias("vm_basecase"))
        .unique(subset=["timestep", "element"], keep="first")
    )
    n_1_node_results = node_results.filter(pl.col("contingency") != base_case_id)

    return int(
        n_1_node_results.join(basecase_vm, on=["timestep", "element"], how="left")
        .filter(pl.col("vm").is_not_null() & pl.col("vm_basecase").is_not_null() & (pl.col("vm_basecase") != 0))
        .with_columns(
            voltage_jump_percent=((pl.col("vm") - pl.col("vm_basecase")).abs() / pl.col("vm_basecase").abs()) * 100.0
        )
        .filter(pl.col("voltage_jump_percent") > jump_threshold_percent)
        .select(pl.len())
        .collect()
        .item()
    )


def get_worst_k_contingencies_ac(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars],
    convergence_results: patpl.LazyFrame[NodeResultSchemaPolars],
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
    convergence_results : DataFrame[NodeResultSchemaPolars]
        The convergence results dataframe containing the convergence information.
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
        contingencies_sorted_by_overloads = (
            overload_per_cont.filter(pl.col("timestep") == t).sort("overload", descending=True).head(k)
        )
        contingencies_with_high_overload = contingencies_sorted_by_overloads.get_column("contingency").cast(str).to_list()
        non_converging_contingencies = (
            convergence_results.filter(pl.col("timestep") == t)
            .filter(pl.col("status") != "CONVERGED")
            .collect()
            .get_column("contingency")
            .cast(str)
            .to_list()
        )

        contingencies.append(contingencies_with_high_overload + non_converging_contingencies)

        if contingencies_with_high_overload:
            br_results_top_k = branch_results.filter(pl.col("contingency").is_in(contingencies_with_high_overload))
            overload_top_k = compute_overload_energy(br_results_top_k, field=field)
        else:
            overload_top_k = 0.0
        overloads.append(float(overload_top_k))

    return contingencies, overloads


def compute_metrics(
    loadflow_results: LoadflowResultsPolars,
    base_case_id: Optional[str] = None,
    critical_va_diff_threshold: float = 0.0,
) -> dict[MetricType, float | None]:
    """Compute the metrics from the loadflow results.

    N-1 overload energy will exclude the base case results if base_case_id is provided,
    otherwise it will include all contingencies.
    This method will return None for metrics that cannot be computed due to missing or invalid data. For example,
    if basecase is provided and is the only contingency, then N-1 metrics will be None
    since there are no valid N-1 contingencies to compute on.

    Parameters
    ----------
    loadflow_results : LoadflowResultsPolars
        The loadflow results containing the branch results.
    base_case_id : Optional[str], optional
        The contingency ID for the base case (N-0). If not provided, no n-0 metrics will be computed.
    critical_va_diff_threshold : float, optional
        Threshold in degrees above which a contingency case is counted in the voltage-angle-difference count metrics.

    Returns
    -------
    dict[MetricType, float | None]
        A dictionary with the computed metrics.
    """
    n_1_branch_res = (
        loadflow_results.branch_results.filter(pl.col("contingency") != base_case_id)
        if base_case_id is not None
        else loadflow_results.branch_results
    )
    n_1_va_diff_res = (
        loadflow_results.va_diff_results.filter(pl.col("contingency") != base_case_id)
        if base_case_id is not None
        else loadflow_results.va_diff_results
    )
    metrics = {
        "max_flow_n_1": compute_max_load(n_1_branch_res),
        "overload_energy_n_1": compute_overload_energy(n_1_branch_res, field="p"),
        "max_va_diff_n_1": compute_max_va_diff(n_1_va_diff_res),
        "critical_va_diff_count_n_1": count_critical_va_diff_cases(
            n_1_va_diff_res, critical_threshold=critical_va_diff_threshold
        ),
        "overload_current_n_1": compute_overload_energy(n_1_branch_res, field="i"),
        "critical_branch_count_n_1": count_critical_branches(n_1_branch_res),
        "voltage_jump_count_n_1": count_voltage_jumps(loadflow_results.node_results, base_case_id=base_case_id),
    }

    if base_case_id is not None:
        # Base case (N-0) results as Polars LazyFrames
        n_0_branch_res = loadflow_results.branch_results.filter(pl.col("contingency") == base_case_id)

        # Va diff may not contain the base case; filtering will yield an empty frame if absent.
        # compute_max_va_diff already returns 0.0 if empty/None, so no explicit fallback needed.
        n_0_va_diff = loadflow_results.va_diff_results.filter(pl.col("contingency") == base_case_id)

        new_metrics = {
            "max_flow_n_0": compute_max_load(n_0_branch_res),
            "overload_energy_n_0": compute_overload_energy(n_0_branch_res, field="p"),
            "max_va_diff_n_0": compute_max_va_diff(n_0_va_diff) or 0.0,  # Default to 0.0 if no valid va_diff values for n-0
            "critical_va_diff_count_n_0": count_critical_va_diff_cases(
                n_0_va_diff, critical_threshold=critical_va_diff_threshold
            ),
            "overload_current_n_0": compute_overload_energy(n_0_branch_res, field="i"),
            "critical_branch_count_n_0": count_critical_branches(n_0_branch_res),
        }

        for metric_name, value in new_metrics.items():
            assert value is not None, (
                f"{metric_name} could not be computed, possibly due to missing or invalid base case results."
            )

        metrics.update(new_metrics)
    return metrics
