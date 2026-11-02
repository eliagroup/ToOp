# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build the loadflow result tables from the powsybl limit violation table.

All functions work on polars lazy frames. The violation table is read once through the native handle and never
materialized as a pandas multi-index. Pre-contingency rows carry an empty contingency id, exactly as powsybl reports
them; renaming that to the base case id is left to the caller.
"""

from enum import Enum

import pandera.pandas as pa
import pandera.typing.polars as patpl
import polars as pl
from beartype.typing import Iterable, Optional, Protocol, Sequence, runtime_checkable
from toop_engine_contingency_analysis.pypowsybl_limit_table.definition import LimitViolationFilter
from toop_engine_interfaces.loadflow_results import BranchSide, ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    ConnectivityResultSchemaPolars,
    ConvergedSchemaPolars,
    NodeResultSchemaPolars,
    VADiffResultSchemaPolars,
)

#: The columns of the powsybl limit violation table and the dtypes the extraction relies on.
LIMIT_VIOLATION_SCHEMA = {
    "contingency_id": pl.String,
    "subject_id": pl.String,
    "subject_name": pl.String,
    "limit_type": pl.String,
    "limit_name": pl.String,
    "limit": pl.Float64,
    "acceptable_duration": pl.Int64,
    "limit_reduction": pl.Float64,
    "value": pl.Float64,
    "side": pl.String,
}

#: Maps the side names of the violation table to the integer sides of the branch results.
POWSYBL_SIDE_MAP = {"ONE": BranchSide.ONE.value, "TWO": BranchSide.TWO.value, "THREE": BranchSide.THREE.value}

#: The limit name powsybl reports in a violation of the permanent limit.
POWSYBL_PERMANENT_LIMIT_NAME = "permanent"

CURRENT_VIOLATION = "CURRENT"
LOW_VOLTAGE_VIOLATION = "LOW_VOLTAGE"
HIGH_VOLTAGE_VIOLATION = "HIGH_VOLTAGE"
LOW_VOLTAGE_ANGLE_VIOLATION = "LOW_VOLTAGE_ANGLE"
HIGH_VOLTAGE_ANGLE_VIOLATION = "HIGH_VOLTAGE_ANGLE"

#: Maps the status names of the pre and post contingency results to the convergence status of the results.
POWSYBL_STATUS_NAME_MAP = {
    "CONVERGED": ConvergenceStatus.CONVERGED.value,
    "NO_IMPACT": ConvergenceStatus.CONVERGED.value,
    "FAILED": ConvergenceStatus.FAILED.value,
    "SOLVER_FAILED": ConvergenceStatus.FAILED.value,
    "MAX_ITERATION_REACHED": ConvergenceStatus.MAX_ITERATION_REACHED.value,
    "NO_CALCULATION": ConvergenceStatus.NO_CALCULATION.value,
}

NO_IMPACT_WARNING = "NO_IMPACT: the contingency did not change the network, the results equal the base case"

_NAN = float("nan")


@runtime_checkable
class PostContingencyStatus(Protocol):
    """What the extraction reads from a native pypowsybl PostContingencyResult."""

    @property
    def contingency_id(self) -> str:
        """The id of the contingency."""

    @property
    def status(self) -> Enum:
        """The computation status, its name is mapped through POWSYBL_STATUS_NAME_MAP."""

    @property
    def disconnected_elements(self) -> Iterable[str]:
        """The ids of the elements powsybl disconnected for the contingency."""


def normalize_limit_violations(violations: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the raw violation table to stable dtypes and the result column names.

    An empty table arrives with null dtypes, so the cast is what makes the string and arithmetic expressions downstream
    valid for every run.

    Parameters
    ----------
    violations : pl.LazyFrame
        The violation table as read from the native handle.

    Returns
    -------
    pl.LazyFrame
        The table with the dtypes of LIMIT_VIOLATION_SCHEMA, contingency_id renamed to contingency and subject_id to
        element.
    """
    return violations.cast(LIMIT_VIOLATION_SCHEMA).rename({"contingency_id": "contingency", "subject_id": "element"})


def _without_basecase(frame: pl.LazyFrame, drop_basecase: bool) -> pl.LazyFrame:
    """Drop the pre-contingency rows if the base case is taken from full monitoring instead."""
    return frame.filter(pl.col("contingency") != "") if drop_basecase else frame


@pa.check_types
def get_branch_results_from_violations(
    violations: pl.LazyFrame,
    branch_limits: pl.LazyFrame,
    timestep: int,
    drop_basecase: bool,
) -> patpl.LazyFrame[BranchResultSchemaPolars]:
    """Build the branch results from the current violations.

    The loading is always relative to the permanent limit. When a temporary limit is exceeded as well, powsybl reports
    that limit in the row, so the permanent limit is joined from the branch limit table instead of read from the row.
    The violation table carries no active or reactive power, those columns are NaN.

    Parameters
    ----------
    violations : pl.LazyFrame
        The normalized violation table.
    branch_limits : pl.LazyFrame
        The prepared branch limits with the columns element_id, side and value, where value is the permanent limit.
    timestep : int
        The timestep stored in the results.
    drop_basecase : bool
        Whether to drop the pre-contingency rows, because the base case comes from full monitoring.

    Returns
    -------
    patpl.LazyFrame[BranchResultSchemaPolars]
        One row per violated branch side and contingency.
    """
    current = _without_basecase(violations.filter(pl.col("limit_type") == CURRENT_VIOLATION), drop_basecase)
    current = current.with_columns(
        pl.col("side").replace_strict(POWSYBL_SIDE_MAP, default=None, return_dtype=pl.Int64).alias("side")
    )
    permanent_limits = branch_limits.select(
        pl.col("element_id").cast(pl.String).alias("element"),
        pl.col("side").cast(pl.Int64),
        pl.col("value").cast(pl.Float64).alias("permanent_limit"),
    )
    current = current.join(permanent_limits, on=["element", "side"], how="left")  # m:1 join
    current = current.with_columns(
        pl.coalesce(
            pl.col("permanent_limit"),
            pl.when(pl.col("limit_name") == POWSYBL_PERMANENT_LIMIT_NAME).then(pl.col("limit")),
        ).alias("permanent_limit")
    )
    return current.select(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.col("contingency"),
        pl.col("element"),
        pl.col("side"),
        pl.lit(_NAN, dtype=pl.Float64).alias("p"),
        pl.lit(_NAN, dtype=pl.Float64).alias("q"),
        pl.col("value").alias("i"),
        (pl.col("value") / pl.col("permanent_limit")).fill_null(_NAN).alias("loading"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


@pa.check_types
def get_node_results_from_violations(
    violations: pl.LazyFrame,
    voltage_levels: pl.LazyFrame,
    bus_map: pl.LazyFrame,
    basecase_node_results: pl.LazyFrame,
    timestep: int,
    drop_basecase: bool,
) -> patpl.LazyFrame[NodeResultSchemaPolars]:
    """Build the node results from the voltage violations.

    Powsybl keys voltage violations by voltage level and reports neither the bus nor its angle, so the element of
    these rows is the voltage level id, va, p and q are NaN. The loading is the distance from nominal voltage relative
    to the violated bound, negative for undervoltage, like the loading of the monitored node results. The deviation
    from the base case is measured against the base case bus of the same voltage level that sits closest to the
    violated bound.

    Parameters
    ----------
    violations : pl.LazyFrame
        The normalized violation table.
    voltage_levels : pl.LazyFrame
        The voltage levels with the columns id and nominal_v.
    bus_map : pl.LazyFrame
        The bus map with the columns id and voltage_level_id, to find the base case buses of a voltage level.
    basecase_node_results : pl.LazyFrame
        The node results of the fully monitored base case, with an empty contingency id. May be empty.
    timestep : int
        The timestep stored in the results.
    drop_basecase : bool
        Whether to drop the pre-contingency rows, because the base case comes from full monitoring.

    Returns
    -------
    patpl.LazyFrame[NodeResultSchemaPolars]
        One row per violated voltage level and contingency.
    """
    voltage = _without_basecase(
        violations.filter(pl.col("limit_type").is_in([LOW_VOLTAGE_VIOLATION, HIGH_VOLTAGE_VIOLATION])), drop_basecase
    )
    voltage = voltage.join(
        voltage_levels.select(pl.col("id").cast(pl.String).alias("element"), pl.col("nominal_v").cast(pl.Float64)),
        on="element",
        how="left",
    )  # m:1 join
    basecase_extremes = (
        basecase_node_results.filter(pl.col("contingency") == "")
        .join(
            bus_map.select(pl.col("id").cast(pl.String).alias("element"), pl.col("voltage_level_id").cast(pl.String)),
            on="element",
            how="inner",
        )
        .group_by("voltage_level_id")
        .agg(pl.col("vm").min().alias("vm_basecase_min"), pl.col("vm").max().alias("vm_basecase_max"))
    )
    voltage = voltage.join(basecase_extremes, left_on="element", right_on="voltage_level_id", how="left")  # m:1 join
    is_high = pl.col("limit_type") == HIGH_VOLTAGE_VIOLATION
    vm_deviation = pl.col("value") - pl.col("nominal_v")
    vm_basecase = pl.when(is_high).then(pl.col("vm_basecase_max")).otherwise(pl.col("vm_basecase_min"))
    return voltage.select(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.col("contingency"),
        pl.col("element"),
        pl.col("value").alias("vm"),
        pl.lit(_NAN, dtype=pl.Float64).alias("va"),
        pl.when(is_high)
        .then(vm_deviation / (pl.col("limit") - pl.col("nominal_v")))
        .otherwise(vm_deviation / (pl.col("nominal_v") - pl.col("limit")))
        .fill_null(_NAN)
        .alias("vm_loading"),
        pl.lit(_NAN, dtype=pl.Float64).alias("p"),
        pl.lit(_NAN, dtype=pl.Float64).alias("q"),
        pl.when(vm_basecase.is_not_null() & vm_basecase.is_not_nan() & (vm_basecase != 0))
        .then((pl.col("value") - vm_basecase).abs() / vm_basecase.abs() * 100.0)
        .otherwise(pl.lit(_NAN))
        .alias("vm_basecase_deviation"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


@pa.check_types
def get_va_diff_results_from_violations(
    violations: pl.LazyFrame, timestep: int
) -> patpl.LazyFrame[VADiffResultSchemaPolars]:
    """Build the voltage angle difference results from the voltage angle violations.

    Voltage angle limits are defined on the network and not by this library. The element of a row is the id of the
    violated angle limit, the value is the angle difference in degrees.

    Parameters
    ----------
    violations : pl.LazyFrame
        The normalized violation table.
    timestep : int
        The timestep stored in the results.

    Returns
    -------
    patpl.LazyFrame[VADiffResultSchemaPolars]
        One row per violated angle limit and contingency, pre-contingency rows included.
    """
    angle = violations.filter(pl.col("limit_type").is_in([LOW_VOLTAGE_ANGLE_VIOLATION, HIGH_VOLTAGE_ANGLE_VIOLATION]))
    return angle.select(
        pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        pl.col("contingency"),
        pl.col("element"),
        pl.col("value").alias("va_diff"),
        pl.lit("").alias("element_name"),
        pl.lit("").alias("contingency_name"),
    )


@pa.check_types
def get_convergence_results(
    pre_contingency_status: str,
    post_contingency_results: Sequence[PostContingencyStatus],
    outage_ids: list[str],
    timestep: int,
    basecase_id: Optional[str] = None,
) -> patpl.LazyFrame[ConvergedSchemaPolars]:
    """Build the convergence table from the native pre and post contingency results.

    Every requested contingency gets a row, whether or not powsybl computed it. Contingencies without a result are
    reported as NO_CALCULATION, contingencies without impact as CONVERGED with a note in the warnings column.

    Parameters
    ----------
    pre_contingency_status : str
        The status name of the pre-contingency result, e.g. "CONVERGED".
    post_contingency_results : Sequence[PostContingencyStatus]
        The native post contingency results.
    outage_ids : list[str]
        The ids of all contingencies that outage something, in definition order.
    timestep : int
        The timestep stored in the results.
    basecase_id : Optional[str], optional
        The id of the base case contingency, which gets the pre-contingency status. None if there is none.

    Returns
    -------
    patpl.LazyFrame[ConvergedSchemaPolars]
        One row per contingency, with the base case first if present.
    """
    status_by_id = {result.contingency_id: result.status.name for result in post_contingency_results}
    contingencies = [*([basecase_id] if basecase_id is not None else []), *outage_ids]
    status_names = [
        *([pre_contingency_status] if basecase_id is not None else []),
        *[status_by_id.get(outage_id, "NO_CALCULATION") for outage_id in outage_ids],
    ]
    return pl.LazyFrame(
        {
            "timestep": [timestep] * len(contingencies),
            "contingency": contingencies,
            "status": [POWSYBL_STATUS_NAME_MAP[name] for name in status_names],
            "iteration_count": [_NAN] * len(contingencies),
            "warnings": [NO_IMPACT_WARNING if name == "NO_IMPACT" else "" for name in status_names],
            "contingency_name": [""] * len(contingencies),
        },
        schema={
            "timestep": pl.Int64,
            "contingency": pl.String,
            "status": pl.String,
            "iteration_count": pl.Float64,
            "warnings": pl.String,
            "contingency_name": pl.String,
        },
    )


@pa.check_types
def get_connectivity_results(
    post_contingency_results: Sequence[PostContingencyStatus],
) -> patpl.LazyFrame[ConnectivityResultSchemaPolars]:
    """Build the connectivity table from the elements powsybl disconnected for each contingency.

    The outage group id is the contingency id: the N-1 definition already groups the elements that go together.

    Parameters
    ----------
    post_contingency_results : Sequence[PostContingencyStatus]
        The native post contingency results, which carry the disconnected elements.

    Returns
    -------
    patpl.LazyFrame[ConnectivityResultSchemaPolars]
        One row per contingency and disconnected element.
    """
    rows = [
        (result.contingency_id, element)
        for result in post_contingency_results
        for element in sorted(result.disconnected_elements)
    ]
    return pl.LazyFrame(
        {
            "contingency": [contingency for contingency, _ in rows],
            "element": [element for _, element in rows],
            "outage_group_id": [contingency for contingency, _ in rows],
        },
        schema={"contingency": pl.String, "element": pl.String, "outage_group_id": pl.String},
    )


def apply_limit_violation_filter(
    frame: pl.LazyFrame,
    mode: LimitViolationFilter,
    severity: pl.Expr,
    basecase_id: Optional[str] = None,
) -> pl.LazyFrame:
    """Reduce a result table to its worst rows.

    Base case rows are never touched. Among the remaining rows, the one with the highest severity is kept per element
    or per contingency, depending on the mode. Rows whose severity is unknown sort last, so an element or contingency
    with only unknown severities keeps one of its rows.

    Parameters
    ----------
    frame : pl.LazyFrame
        A result table with the columns timestep, contingency and element.
    mode : LimitViolationFilter
        Which rows to keep. ALL returns the table unchanged.
    severity : pl.Expr
        The expression ranking the rows, higher is worse, e.g. the loading of a branch.
    basecase_id : Optional[str], optional
        The id of the base case contingency, whose rows are exempt. None if there is none.

    Returns
    -------
    pl.LazyFrame
        The reduced table, base case rows first.
    """
    if mode == LimitViolationFilter.ALL:
        return frame
    per_element = mode == LimitViolationFilter.WORST_OVERLOAD_PER_ELEMENT
    keys = ["timestep", "element"] if per_element else ["timestep", "contingency"]
    is_basecase = pl.col("contingency") == basecase_id if basecase_id is not None else pl.lit(False)
    worst = (
        frame.filter(~is_basecase)
        .with_columns(severity.cast(pl.Float64).fill_nan(None).alias("_severity"))
        .sort("_severity", descending=True, nulls_last=True, maintain_order=True)
        .unique(subset=keys, keep="first", maintain_order=True)
        .drop("_severity")
    )
    return pl.concat([frame.filter(is_basecase), worst])
