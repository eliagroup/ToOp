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

from enum import Enum

import pandera.pandas as pa
import pandera.typing.polars as patpl
import polars as pl
from beartype.typing import Iterable, Optional, Protocol, Sequence, runtime_checkable
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import (
    ConnectivityResultSchemaPolars,
    ConvergedSchemaPolars,
    LimitViolationSchemaPolars,
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


_NAN = float("nan")


@runtime_checkable
class PostContingencyStatus(Protocol):
    """What the extraction reads from a native pypowsybl PostContingencyResult."""

    @property
    def contingency_id(self) -> str:
        """The id of the contingency."""

    @property
    def status(self) -> Enum:
        """The computation status, its name is a ConvergenceStatus."""

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


@pa.check_types
def split_limit_violations(violations: pl.LazyFrame, voltage_levels: pl.LazyFrame, timestep: int) -> dict[str, pl.LazyFrame]:
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
    dict[str, pl.LazyFrame]
        The LimitViolationSchemaPolars frames keyed by their field in LoadflowResultsPolars: branch_violations,
        node_violations and va_diff_violations.
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
    return {
        "branch_violations": LimitViolationSchemaPolars.validate(violations.filter(is_branch)),
        "node_violations": LimitViolationSchemaPolars.validate(violations.filter(~is_branch & ~is_va_diff)),
        "va_diff_violations": LimitViolationSchemaPolars.validate(violations.filter(is_va_diff)),
    }


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
    reported as NO_CALCULATION, contingencies without impact as NO_IMPACT with a note in the warnings column.

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
            "status": [ConvergenceStatus(name).value for name in status_names],
            "iteration_count": [_NAN] * len(contingencies),
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
