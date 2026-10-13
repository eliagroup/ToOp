# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Translate a :class:`LoadflowResultFilter` policy into polars predicates and apply it.

The policy itself lives in ``toop_engine_interfaces.loadflow_result_filter``; this module owns the interpretation, which is
backend independent: both the pandapower and the powsybl runner apply the same predicates to the same result schemas.

The predicates are written in *keep* space, so call sites read ``frame.filter(branch_keep_expr(...))``. Internally each one
is the negation of a conjunction: every clause has to agree before a row is dropped, which is what makes "any active filter
can keep a row" and "an unknown value is always kept" fall out of the same expression.
"""

import polars as pl
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars


def _value_is_known_and_below_threshold(column: str, threshold: float | None) -> pl.Expr:
    """Build the drop clause "the absolute value of the column is known and below the threshold".

    Parameters
    ----------
    column : str
        The result column this threshold tests.
    threshold : float | None
        The threshold to test against, compared against the absolute value of *column*. None disables the clause.

    Returns
    -------
    pl.Expr
        True where the row is uninteresting as far as this clause is concerned: a disabled threshold never blocks a drop,
        while an unknown (null or NaN) value always does.
    """
    if threshold is None:
        return pl.lit(True)
    value = pl.col(column)
    # Both null and NaN occur: the converged pandapower path blanks p/q to null where the terminal carries no current,
    # while the non-converged path fills every electrical column with NaN. In polars neither test covers the other.
    return value.is_not_null() & value.is_not_nan() & (value.abs() < threshold)


def _is_droppable_basecase(retain_basecase: bool, basecase_contingency_id: str | None) -> pl.Expr:
    """Build the drop clause "the row does not belong to the basecase".

    Parameters
    ----------
    retain_basecase : bool
        Whether basecase rows are exempt from filtering.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, or None when the run has no basecase and there is nothing to exempt.

    Returns
    -------
    pl.Expr
        True where the row is not protected by the basecase exemption.
    """
    if not retain_basecase or basecase_contingency_id is None:
        return pl.lit(True)
    return pl.col("contingency") != basecase_contingency_id


def keep_branch_rows(
    filters: BranchLoadflowResultFilter, basecase_contingency_id: str | None, loading_column: str = "loading"
) -> pl.Expr:
    """Build the polars predicate selecting the branch result rows worth keeping.

    Intended to be applied directly to a branch result frame, eager or lazy::

        branch_results = branch_results.filter(keep_branch_rows(filters, basecase_contingency_id))

    Rows with a null or NaN loading are kept, because the loading of an element without a rated current or of a
    non-converged contingency is unknown.

    Parameters
    ----------
    filters : BranchLoadflowResultFilter
        The branch filter policy to translate.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, kept while ``filters.retain_basecase`` is set. None if the run has no basecase.
        Take it from :attr:`Nminus1Definition.base_case`.
    loading_column : str
        The column holding the loading the threshold is tested against.

    Returns
    -------
    pl.Expr
        A boolean expression over ``contingency`` and the loading column, True for rows to keep.
        Always True if the filter is not active.
    """
    if not filters.is_active():
        return pl.lit(True)
    drop = _value_is_known_and_below_threshold(loading_column, filters.loading_above) & _is_droppable_basecase(
        filters.retain_basecase, basecase_contingency_id
    )
    return ~drop


def keep_node_rows(filters: NodeLoadflowResultFilter, basecase_contingency_id: str | None) -> pl.Expr:
    """Build the predicate selecting the node result rows to keep.

    A row is kept if ``vm_loading`` is null or NaN, if its absolute ``vm_loading`` reaches ``filters.vm_loading_above``, or
    if its ``vm_basecase_deviation`` reaches ``filters.vm_basecase_deviation_above``. ``vm_loading`` is negative for
    undervoltage, so the absolute value catches over- and undervoltage.

    Parameters
    ----------
    filters : NodeLoadflowResultFilter
        The node filter policy to translate.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, kept while ``filters.retain_basecase`` is set. None if the run has no basecase.

    Returns
    -------
    pl.Expr
        A boolean expression over ``contingency``, ``vm_loading`` and ``vm_basecase_deviation``, True for rows to keep.
        Always True if the filter is not active.
    """
    if not filters.is_active():
        return pl.lit(True)
    drop = (
        _value_is_known_and_below_threshold("vm_loading", filters.vm_loading_above)
        & _value_is_known_and_below_threshold("vm_basecase_deviation", filters.vm_basecase_deviation_above)
        & _is_droppable_basecase(filters.retain_basecase, basecase_contingency_id)
    )
    return ~drop


def _filter_branch_results(
    branch_results: pl.LazyFrame,
    branch_limits: pl.LazyFrame | None,
    filters: BranchLoadflowResultFilter,
    basecase_contingency_id: str | None,
) -> pl.LazyFrame:
    """Filter the branch results, keeping every row that is loaded high against any limit of its branch side.

    The stored loading is relative to a single limit. A row below the threshold for that limit can still exceed it
    for a smaller limit, which the limit-specific metrics use. The threshold is therefore compared with the loading
    relative to the smallest limit of the branch side, the highest loading of that row.

    Parameters
    ----------
    branch_results : pl.LazyFrame
        The branch results to filter.
    branch_limits : pl.LazyFrame | None
        The limits of the branch sides. If None, the stored loading is used.
    filters : BranchLoadflowResultFilter
        The branch filter policy.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, or None if the run has no basecase.

    Returns
    -------
    pl.LazyFrame
        The filtered branch results with their original columns.
    """
    if branch_limits is None or not filters.is_active():
        return branch_results.filter(keep_branch_rows(filters, basecase_contingency_id))
    smallest_limit = branch_limits.group_by("timestep", "element", "side").agg(
        pl.col("limit").min().alias("_smallest_limit")
    )
    columns = branch_results.collect_schema().names()
    return (
        branch_results.join(smallest_limit, on=["timestep", "element", "side"], how="left")
        .with_columns(_highest_loading=pl.max_horizontal(pl.col("loading"), pl.col("i") / pl.col("_smallest_limit")))
        .filter(keep_branch_rows(filters, basecase_contingency_id, loading_column="_highest_loading"))
        .select(columns)
    )


def apply_result_filter(
    loadflow_results: LoadflowResultsPolars,
    result_filter: LoadflowResultFilter,
    basecase_contingency_id: str | None,
) -> LoadflowResultsPolars:
    """Apply a filter policy to a complete result object.

    The backends apply the predicates per outage, which is where the memory and serialization savings are. This function is
    the whole-object equivalent, for callers that already hold a finished result: retrofitting a stored result, or
    filtering in one place when the producing backend did not.

    Parameters
    ----------
    loadflow_results : LoadflowResultsPolars
        The results to filter. Not mutated.
    result_filter : LoadflowResultFilter
        The policy to apply. An inert policy returns the results unchanged.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, or None if the run has no basecase.

    Returns
    -------
    LoadflowResultsPolars
        A copy with the branch and node results filtered. All other tables are passed through untouched.
    """
    if not result_filter.is_active():
        return loadflow_results

    filtered = {}
    if loadflow_results.branch_results is not None:
        filtered["branch_results"] = _filter_branch_results(
            loadflow_results.branch_results,
            loadflow_results.branch_limits,
            result_filter.branch_filters,
            basecase_contingency_id,
        )
    if loadflow_results.node_results is not None:
        filtered["node_results"] = loadflow_results.node_results.filter(
            keep_node_rows(result_filter.node_filters, basecase_contingency_id)
        )
    filtered["result_filter"] = result_filter
    return loadflow_results.model_copy(update=filtered)
