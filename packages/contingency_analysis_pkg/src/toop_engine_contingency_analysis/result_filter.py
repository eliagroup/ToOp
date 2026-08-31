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


def _uninteresting(column: str, threshold: float | None) -> pl.Expr:
    """Build the drop-space clause for one threshold.

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


def _basecase_exemption(retain_basecase: bool, basecase_contingency_id: str | None) -> pl.Expr:
    """Build the drop-space clause protecting the N-0 rows.

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


def branch_keep_expr(filters: BranchLoadflowResultFilter, basecase_contingency_id: str | None) -> pl.Expr:
    """Build the polars predicate selecting the branch result rows worth keeping.

    Intended to be applied directly to a branch result frame, eager or lazy::

        branch_results = branch_results.filter(branch_keep_expr(filters, basecase_contingency_id))

    Rows whose ``loading`` is null or NaN are kept: an element without a rated current cannot be assessed, and neither can
    a contingency that did not converge.

    Parameters
    ----------
    filters : BranchLoadflowResultFilter
        The branch filter policy to translate.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, exempted from filtering while ``filters.retain_basecase`` is set. None if the run
        has no basecase. Resolve it from :attr:`Nminus1Definition.base_case` rather than constructing it by hand.

    Returns
    -------
    pl.Expr
        A boolean expression over ``contingency`` and ``loading``, True for rows to keep. Constant True when the filter is
        inert, so an unconfigured policy costs nothing beyond a no-op predicate.
    """
    if not filters.is_active():
        return pl.lit(True)
    drop = _uninteresting("loading", filters.loading_above) & _basecase_exemption(
        filters.retain_basecase, basecase_contingency_id
    )
    return ~drop


def node_keep_expr(filters: NodeLoadflowResultFilter, basecase_contingency_id: str | None) -> pl.Expr:
    """Build the polars predicate selecting the node result rows worth keeping.

    Rows whose ``vm_loading`` is null or NaN are kept, as are rows whose voltage jump from N-0 exceeds
    ``filters.vm_basecase_deviation_above`` regardless of their own voltage. ``vm_loading`` is signed - negative for
    undervoltage - so the threshold is applied to its absolute value and catches both violation directions.

    Parameters
    ----------
    filters : NodeLoadflowResultFilter
        The node filter policy to translate.
    basecase_contingency_id : str | None
        The id of the N-0 contingency, exempted from filtering while ``filters.retain_basecase`` is set. None if the run
        has no basecase.

    Returns
    -------
    pl.Expr
        A boolean expression over ``contingency``, ``vm_loading`` and ``vm_basecase_deviation``, True for rows to keep.
        Constant True when the filter is inert, so an unconfigured policy costs nothing beyond a no-op predicate.
    """
    if not filters.is_active():
        return pl.lit(True)
    drop = (
        _uninteresting("vm_loading", filters.vm_loading_above)
        & _uninteresting("vm_basecase_deviation", filters.vm_basecase_deviation_above)
        & _basecase_exemption(filters.retain_basecase, basecase_contingency_id)
    )
    return ~drop


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
        filtered["branch_results"] = loadflow_results.branch_results.filter(
            branch_keep_expr(result_filter.branch_filters, basecase_contingency_id)
        )
    if loadflow_results.node_results is not None:
        filtered["node_results"] = loadflow_results.node_results.filter(
            node_keep_expr(result_filter.node_filters, basecase_contingency_id)
        )
    filtered["result_filter"] = result_filter
    return loadflow_results.model_copy(update=filtered)
