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
import pandera.pandas as pa
import pandera.typing.polars as patpl
import polars as pl
import structlog
from beartype.typing import Collection, Literal, Optional
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    ConvergedSchemaPolars,
    LoadflowResultsPolars,
    NodeResultSchemaPolars,
    VADiffResultSchemaPolars,
)
from toop_engine_interfaces.types import LimitCase, MetricLimitScope, MetricType, OperationalLimit

logger = structlog.get_logger(__name__)

METRIC_LIMITS: dict[MetricLimitScope, dict[LimitCase, OperationalLimit]] = {
    "permanent": {"n_0": OperationalLimit.PERMANENT, "n_1": OperationalLimit.PERMANENT},
    "artificial_limit": {"n_0": OperationalLimit.ARTIFICIAL_N_0, "n_1": OperationalLimit.ARTIFICIAL_N_1},
}


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
    return branch_results.select(pl.col("loading").max()).collect().item()


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
    return (
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
    critical_voltage_jump_threshold: float = 100.0,
) -> int | None:
    """Count nodal voltage jumps above the basecase deviation threshold.

    Parameters
    ----------
    node_results : patpl.LazyFrame[NodeResultSchemaPolars]
        The node results dataframe containing `vm_basecase_deviation` values.
    critical_voltage_jump_threshold : float, optional
        The minimum relative voltage jump in percent that counts as a voltage jump.

    Returns
    -------
    int | None
        The number of node results whose voltage deviation from the basecase is strictly above the threshold.
        None if the required columns are not available.
    """
    schema_names = set(node_results.collect_schema().names())
    if "vm_basecase_deviation" not in schema_names:
        return None

    return int(
        node_results.filter(pl.col("vm_basecase_deviation").is_not_null() & pl.col("vm_basecase_deviation").is_not_nan())
        .filter(pl.col("vm_basecase_deviation") > critical_voltage_jump_threshold)
        .select(pl.len())
        .collect()
        .item()
    )


def _with_loading_of_limit(
    branch_results: pl.LazyFrame, branch_limits: pl.LazyFrame, limit: OperationalLimit
) -> pl.LazyFrame:
    """Replace the loading by the current relative to one limit, dropping branch sides without that limit."""
    limits = branch_limits.filter(pl.col("limit_name") == limit.value).select("timestep", "element", "side", "limit")
    return branch_results.join(limits, on=["timestep", "element", "side"], how="inner").with_columns(
        loading=pl.col("i") / pl.col("limit")
    )


def get_worst_k_contingencies_ac(
    branch_results: patpl.LazyFrame[BranchResultSchemaPolars],
    convergence_results: patpl.LazyFrame[ConvergedSchemaPolars],
    k: int = 10,
    field: Literal["p", "i"] = "p",
    base_case_id: str = "BASECASE",
    include_non_converging_loadflows: bool = True,
    branch_limits: Optional[pl.LazyFrame] = None,
    limit_scope: MetricLimitScope = "permanent",
) -> tuple[list[list[str]], list[float]]:
    """Get the worst k contingencies based on overload energy.

    If k is greater than the number of contingencies, all contingencies will be returned.

    Parameters
    ----------
    branch_results : patpl.LazyFrame[BranchResultSchemaPolars]
        The branch results dataframe containing the loading information.
    convergence_results : patpl.LazyFrame[ConvergedSchemaPolars]
        The convergence results dataframe containing the convergence information.
    k : int, optional
        The number of worst contingencies to return, by default 10.
    field : Literal["p", "i"], optional
        The field to use for the overload calculation, either "p" for power or "i" for current, by default "p".
    base_case_id : str, optional
        The contingency ID for the base case (N-0), by default "BASECASE".
    include_non_converging_loadflows : bool, optional
        Whether contingencies with non-converged loadflows should be appended to the worst-k list,
        by default True.
    branch_limits : Optional[pl.LazyFrame], optional
        The limits of the branch sides. If given, the overload is measured against the N-1 limit of ``limit_scope``
        instead of the loading of the branch results, which refers to the permanent limit.
    limit_scope : MetricLimitScope, optional
        The limit scope the overload is measured against if ``branch_limits`` is given, by default "permanent".

    Returns
    -------
    tuple[list[list[str]], list[float]]
        A tuple containing:
        - A list of lists with the contingency IDs for each timestep. The length of the outer list is
        the number of timesteps while the inner lists contain the top k contingencies for that timestep.
        - A list of total overload energy for each timestep. The length matches the number of timesteps.
    """
    if branch_limits is not None:
        branch_results = _with_loading_of_limit(branch_results, branch_limits, METRIC_LIMITS[limit_scope]["n_1"])
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
        if include_non_converging_loadflows:
            non_converging_contingencies = (
                convergence_results.filter(pl.col("timestep") == t)
                .filter(pl.col("status") != "CONVERGED")
                .collect()
                .get_column("contingency")
                .cast(str)
                .to_list()
            )
        else:
            non_converging_contingencies = []

        contingencies.append(contingencies_with_high_overload + non_converging_contingencies)

        if contingencies_with_high_overload:
            br_results_top_k = branch_results.filter(pl.col("contingency").is_in(contingencies_with_high_overload))
            overload_top_k = compute_overload_energy(br_results_top_k, field=field)
        else:
            overload_top_k = 0.0
        overloads.append(float(overload_top_k))

    return contingencies, overloads


def _family_metrics(branch_results: pl.LazyFrame) -> dict[str, float | None]:
    """Compute the metrics for a branch-result frame."""
    return {
        "max_flow": compute_max_load(branch_results),
        "overload_energy": compute_overload_energy(branch_results, "p"),
        "overload_current": compute_overload_energy(branch_results, "i"),
        "critical_branch_count": count_critical_branches(branch_results),
    }


def _family_metrics_for_limit(
    branch_results: pl.LazyFrame,
    branch_limits: pl.LazyFrame,
    limit: OperationalLimit,
) -> dict[str, float | None] | None:
    """Compute branch metrics relative to one limit."""
    limit_frame = _with_loading_of_limit(branch_results, branch_limits, limit)
    if limit_frame.select(pl.len()).collect().item() == 0:
        logger.warning("No branch has the limit, the metrics of this limit are None", limit_name=limit.value)
        return None
    return _family_metrics(limit_frame)


def _compute_branch_limit_metrics(
    branch_results: pl.LazyFrame, branch_limits: Optional[pl.LazyFrame], contingency_level: LimitCase
) -> dict[str, float | None]:
    """Compute the branch metrics of one contingency level against the permanent and the artificial limit.

    The limit is the rating that defines 100% loading (``loading = i / limit``). A threshold, such as the critical
    threshold of the critical branch count, is the value this loading is compared against.

    Parameters
    ----------
    branch_results : pl.LazyFrame
        The branch results of the contingency level.
    branch_limits : Optional[pl.LazyFrame]
        The limits of the branch sides, one row per limit name. If None, the loading column of the branch results is
        taken as the loading relative to the permanent rating and there is no artificial limit.
    contingency_level : LimitCase
        Whether the results are the base case (n_0) or the contingencies (n_1).

    Returns
    -------
    dict[str, float | None]
        The metrics named <family>_permanent_<level> and <family>_artificial_limit_<level>. A metric is None if no
        branch has the limit it is measured against.
    """
    if branch_limits is None:
        scope_metrics = {"permanent": _family_metrics(branch_results), "artificial_limit": None}
    else:
        scope_metrics = {
            scope: _family_metrics_for_limit(branch_results, branch_limits, limits[contingency_level])
            for scope, limits in METRIC_LIMITS.items()
        }

    metrics: dict[str, float | None] = {}
    for scope, family_values in scope_metrics.items():
        for family in ("max_flow", "overload_energy", "overload_current", "critical_branch_count"):
            metrics[f"{family}_{scope}_{contingency_level}"] = family_values[family] if family_values is not None else None
    return metrics


def _max_flow_displayed_permanent(
    branch_results: pl.LazyFrame,
    branch_limits: Optional[pl.LazyFrame],
    displayed_branch_ids: Collection[str],
) -> float | None:
    """Compute the highest loading against the permanent limit over the displayed branches only."""
    displayed_results = branch_results.filter(pl.col("element").is_in(list(displayed_branch_ids)))
    if branch_limits is None:
        return compute_max_load(displayed_results)
    family_values = _family_metrics_for_limit(displayed_results, branch_limits, OperationalLimit.PERMANENT)
    return family_values["max_flow"] if family_values is not None else None


def compute_metrics(
    loadflow_results: LoadflowResultsPolars,
    base_case_id: Optional[str] = None,
    critical_va_diff_threshold: float = 360.0,
    critical_voltage_jump_threshold: float = 100.0,
    displayed_branch_ids: Optional[Collection[str]] = None,
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
    critical_va_diff_threshold : float
        Threshold in degrees above which a contingency case is counted in the voltage-angle-difference count metrics.
        Defaults to 360. ie there will never be a critical diff
    critical_voltage_jump_threshold: float
        Threshold in percent on when a voltage jump between n0 and n1 is considered critical.
        voltage-jump count metric. Defaults to 100.
    displayed_branch_ids : Optional[Collection[str]]
        The ids of the branches flagged as displayed in the N-1 definition. If given, the metrics
        max_flow_displayed_permanent_n_0 and max_flow_displayed_permanent_n_1 are added, which are None if
        no displayed branch has a result. If None, these metrics are not computed.

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
        **_compute_branch_limit_metrics(n_1_branch_res, loadflow_results.branch_limits, "n_1"),
        "max_va_diff_n_1": compute_max_va_diff(n_1_va_diff_res),
        "critical_va_diff_count_n_1": count_critical_va_diff_cases(
            n_1_va_diff_res, critical_threshold=critical_va_diff_threshold
        ),
        "voltage_jump_count_n_1": count_voltage_jumps(
            loadflow_results.node_results,
            critical_voltage_jump_threshold=critical_voltage_jump_threshold,
        ),
    }

    if base_case_id is not None:
        # Base case (N-0) results as Polars LazyFrames
        n_0_branch_res = loadflow_results.branch_results.filter(pl.col("contingency") == base_case_id)

        # Va diff may not contain the base case; filtering will yield an empty frame if absent.
        # compute_max_va_diff already returns 0.0 if empty/None, so no explicit fallback needed.
        n_0_va_diff = loadflow_results.va_diff_results.filter(pl.col("contingency") == base_case_id)

        new_metrics = {
            **_compute_branch_limit_metrics(n_0_branch_res, loadflow_results.branch_limits, "n_0"),
            "max_va_diff_n_0": compute_max_va_diff(n_0_va_diff) or 0.0,  # Default to 0.0 if no valid va_diff values for n-0
            "critical_va_diff_count_n_0": count_critical_va_diff_cases(
                n_0_va_diff, critical_threshold=critical_va_diff_threshold
            ),
        }

        # The artificial limit only exists for optimized and non-worsening branches, so its metrics may be None
        for metric_name, value in new_metrics.items():
            assert value is not None or "artificial_limit" in metric_name, (
                f"{metric_name} could not be computed, possibly due to missing or invalid base case results."
            )

        metrics.update(new_metrics)

    if displayed_branch_ids is not None:
        metrics["max_flow_displayed_permanent_n_1"] = _max_flow_displayed_permanent(
            n_1_branch_res, loadflow_results.branch_limits, displayed_branch_ids, "n_1"
        )
        if base_case_id is not None:
            metrics["max_flow_displayed_permanent_n_0"] = _max_flow_displayed_permanent(
                n_0_branch_res, loadflow_results.branch_limits, displayed_branch_ids, "n_0"
            )
    return metrics
