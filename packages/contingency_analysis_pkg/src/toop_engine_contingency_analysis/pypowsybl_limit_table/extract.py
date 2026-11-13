# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build the violation, convergence and connectivity tables from the powsybl security analysis result.

All functions work on polars lazy frames. The violation table is read once through the native handle and never
materialized as a pandas multi-index. Pre-contingency rows carry an empty contingency id, exactly as powsybl reports
them; renaming that to the base case id is left to the caller.
"""

import pandera.pandas as pa
import pandera.typing.polars as patpl
import polars as pl
from beartype.typing import Any, Optional, Sequence
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import (
    ConnectivityResultSchemaPolars,
    ConvergedSchemaPolars,
    LimitViolationSchemaPolars,
    PostContingencyResultPolars,
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

#: The limit types powsybl reports on branch sides.
BRANCH_VIOLATIONS = ["CURRENT", "ACTIVE_POWER", "APPARENT_POWER"]
#: The limit types of the voltage angle limits defined on the network.
VA_DIFF_VIOLATIONS = ["LOW_VOLTAGE_ANGLE", "HIGH_VOLTAGE_ANGLE"]
HIGH_VOLTAGE_VIOLATION = "HIGH_VOLTAGE"
LOW_VOLTAGE_VIOLATION = "LOW_VOLTAGE"


NAN = float("nan")


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


@pa.check_types
def split_limit_violations(
    violations: pl.LazyFrame, voltage_levels: pl.LazyFrame, timestep: int
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Split the violation table into branch, node and voltage angle difference violations and add their loading.

    No row is changed or dropped. The branch violations are the rows with a limit type of BRANCH_VIOLATIONS, the
    voltage angle difference violations those of VA_DIFF_VIOLATIONS, the node violations all other rows, e.g. low and
    high voltage violations. The loading is computed per limit type, see LimitViolationSchemaPolars.loading.

    Parameters
    ----------
    violations : pl.LazyFrame
        The normalized violation table.
    voltage_levels : pl.LazyFrame
        The voltage levels with the columns id and nominal_v, the reference of the voltage loading.
    timestep : int
        The timestep stored in the violations.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]
        The LimitViolationSchemaPolars frames: the branch violations, the node violations and the voltage angle
        difference violations.
    """
    is_branch = pl.col("limit_type").is_in(BRANCH_VIOLATIONS)
    is_va_diff = pl.col("limit_type").is_in(VA_DIFF_VIOLATIONS)
    deviation = pl.col("value") - pl.col("nominal_v")
    loading = (
        pl.when(is_branch)
        .then(pl.col("value").abs() / pl.col("limit"))
        .when(pl.col("limit_type") == HIGH_VOLTAGE_VIOLATION)
        .then(deviation / (pl.col("limit") - pl.col("nominal_v")))
        .when(pl.col("limit_type") == LOW_VOLTAGE_VIOLATION)
        .then(deviation / (pl.col("nominal_v") - pl.col("limit")))
        .when(is_va_diff)
        .then(pl.col("value") / pl.col("limit"))
    )
    nominal_v = voltage_levels.select(pl.col("id").cast(pl.String).alias("element"), pl.col("nominal_v").cast(pl.Float64))
    violations = (
        violations.join(nominal_v, on="element", how="left", maintain_order="left")  # m:1 join
        .with_columns(pl.lit(timestep, dtype=pl.Int64).alias("timestep"), loading.alias("loading"))
        .select(list(LimitViolationSchemaPolars.to_schema().columns))
    )
    return (
        LimitViolationSchemaPolars.validate(violations.filter(is_branch)),
        LimitViolationSchemaPolars.validate(violations.filter(~is_branch & ~is_va_diff)),
        LimitViolationSchemaPolars.validate(violations.filter(is_va_diff)),
    )


def _convergence_frame(contingencies: list[str], status_names: list[str], timestep: int) -> pl.LazyFrame:
    """Build a convergence table with one row per contingency and its status name."""
    return pl.LazyFrame(
        {
            "timestep": [timestep] * len(contingencies),
            "contingency": contingencies,
            "status": [ConvergenceStatus(name).value for name in status_names],
            "iteration_count": [NAN] * len(contingencies),
            "warnings": [""] * len(contingencies),
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
def get_post_contingency_results(
    post_contingency_results: Sequence[Any],
) -> patpl.LazyFrame[PostContingencyResultPolars]:
    """Mirror the native post contingency results as a polars frame.

    Parameters
    ----------
    post_contingency_results : Sequence[Any]
        The native pypowsybl PostContingencyResult objects.

    Returns
    -------
    patpl.LazyFrame[PostContingencyResultPolars]
        One row per post contingency result.
    """
    return pl.LazyFrame(
        {
            "contingency_id": [result.contingency_id for result in post_contingency_results],
            "status": [result.status.name for result in post_contingency_results],
            "disconnected_elements": [list(result.disconnected_elements) for result in post_contingency_results],
        },
        schema={"contingency_id": pl.String, "status": pl.String, "disconnected_elements": pl.List(pl.String)},
    )


@pa.check_types
def get_convergence_results(
    post_contingency_results: patpl.LazyFrame[PostContingencyResultPolars],
    pre_contingency_status: str,
    outage_ids: list[str],
    timestep: int,
    basecase_id: Optional[str] = None,
) -> patpl.LazyFrame[ConvergedSchemaPolars]:
    """Build the convergence table, giving every requested contingency a row whether or not powsybl computed it.

    Parameters
    ----------
    post_contingency_results : patpl.LazyFrame[PostContingencyResultPolars]
        The post contingency results, see get_post_contingency_results.
    pre_contingency_status : str
        The status name of the pre-contingency result, e.g. "CONVERGED".
    outage_ids : list[str]
        The ids of all contingencies that outage something, in definition order.
    timestep : int
        The timestep stored in the results.
    basecase_id : Optional[str], optional
        The id of the base case contingency, which gets the pre-contingency status. None if there is none.

    Returns
    -------
    patpl.LazyFrame[ConvergedSchemaPolars]
        One row per requested contingency, with the base case first if present. Contingencies powsybl did not
        compute are NO_CALCULATION.
    """
    has_basecase = basecase_id is not None
    requested = _convergence_frame(
        [*([basecase_id] if has_basecase else []), *outage_ids],
        [*([pre_contingency_status] if has_basecase else []), *["NO_CALCULATION"] * len(outage_ids)],
        timestep,
    )
    computed = post_contingency_results.select(
        pl.col("contingency_id").alias("contingency"), pl.col("status").alias("computed_status")
    )
    return (
        requested.join(computed, on="contingency", how="left", maintain_order="left")  # m:1 join
        .with_columns(pl.coalesce("computed_status", "status").alias("status"))
        .drop("computed_status")
    )


@pa.check_types
def get_connectivity_results(
    post_contingency_results: patpl.LazyFrame[PostContingencyResultPolars],
) -> patpl.LazyFrame[ConnectivityResultSchemaPolars]:
    """Build the connectivity table from the elements powsybl disconnected for each contingency.

    The outage group id is the contingency id: the N-1 definition already groups the elements that go together.

    Parameters
    ----------
    post_contingency_results : patpl.LazyFrame[PostContingencyResultPolars]
        The post contingency results, see get_post_contingency_results.

    Returns
    -------
    patpl.LazyFrame[ConnectivityResultSchemaPolars]
        One row per contingency and disconnected element.
    """
    return (
        post_contingency_results.select(
            pl.col("contingency_id").alias("contingency"),
            pl.col("disconnected_elements").list.sort().alias("element"),
            pl.col("contingency_id").alias("outage_group_id"),
        )
        .explode("element")
        .drop_nulls("element")
    )
