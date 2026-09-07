# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for translating a result filter policy into polars predicates.

Each fixture holds one row per case the predicate has to decide, so a single run of the predicate answers every question
about it. The tests are grouped per predicate accordingly, with one assert per property and a message naming the property,
so a failure still points at the rule that broke.
"""

import polars as pl
from toop_engine_contingency_analysis.result_filter import apply_result_filter, branch_keep_expr, node_keep_expr
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars

BASECASE = "BASECASE"
THRESHOLD = 0.7


def _branch_results() -> pl.DataFrame:
    """Branch rows covering every case the branch predicate has to decide.

    Returns
    -------
    pl.DataFrame
        One row per (contingency, element), named after the case it represents.
    """
    return pl.DataFrame(
        {
            "contingency": [BASECASE, BASECASE, "CO_1", "CO_1", "CO_1", "CO_1", "CO_1"],
            "element": ["quiet_n0", "loaded_n0", "quiet", "at_threshold", "loaded", "no_rating", "not_converged"],
            "loading": [0.12, 0.95, 0.12, 0.70, 0.95, None, float("nan")],
        }
    )


def _node_results() -> pl.DataFrame:
    """Node rows covering every case the node predicate has to decide.

    ``vm_loading`` is signed - negative means undervoltage - and ``vm_basecase_deviation`` is a percentage.

    Returns
    -------
    pl.DataFrame
        One row per (contingency, element), named after the case it represents.
    """
    return pl.DataFrame(
        {
            "contingency": [BASECASE, BASECASE, "CO_1", "CO_1", "CO_1", "CO_1", "CO_1", "CO_1"],
            "element": [
                "nominal_n0",
                "overvoltage_n0",
                "nominal",
                "overvoltage",
                "undervoltage",
                "at_threshold",
                "jumper",
                "no_limits",
            ],
            "vm_loading": [0.01, 0.95, 0.01, 0.95, -0.95, 0.70, 0.01, None],
            "vm_basecase_deviation": [0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 12.0, 0.1],
        }
    )


def _kept(
    results: pl.DataFrame,
    keep_expr: pl.Expr,
) -> list[str]:
    """Apply a keep predicate and return the surviving element ids.

    Parameters
    ----------
    results : pl.DataFrame
        The result rows to filter.
    keep_expr : pl.Expr
        The predicate under test.

    Returns
    -------
    list[str]
        The ``element`` values of the rows that survived.
    """
    return results.filter(keep_expr)["element"].to_list()


def test_branch_filter():
    """Every rule the branch predicate implements, against one set of rows that exercises all of them."""
    results = _branch_results()
    all_elements = results["element"].to_list()

    inert = _kept(results, branch_keep_expr(BranchLoadflowResultFilter(), BASECASE))
    assert inert == all_elements, "an unconfigured policy must not change the results at all"

    kept = _kept(results, branch_keep_expr(BranchLoadflowResultFilter(loading_above=THRESHOLD), BASECASE))
    assert "quiet" not in kept, "a lightly loaded CO/CB pair carries no decision value and should go"
    assert "loaded" in kept, "a heavily loaded CO/CB pair is the whole point of keeping any rows"
    assert "at_threshold" in kept, "loading_above=0.7 keeps a branch at exactly 70%, as the field name reads"
    assert "no_rating" in kept, "a branch with no rated current cannot be assessed, so it is never discarded"
    assert "not_converged" in kept, "non-converged rows are all-NaN, and unknown-is-kept covers them too"
    assert "quiet_n0" in kept, "N-0 is the reference the other cases are read against, whatever its loading"

    without_exemption = _kept(
        results,
        branch_keep_expr(BranchLoadflowResultFilter(loading_above=THRESHOLD, retain_basecase=False), BASECASE),
    )
    assert "quiet_n0" not in without_exemption, "retain_basecase=False applies the same threshold to N-0"
    assert "loaded_n0" in without_exemption, "a loaded N-0 row clears the threshold on its own merits"

    no_basecase = _kept(results, branch_keep_expr(BranchLoadflowResultFilter(loading_above=THRESHOLD), None))
    assert "quiet_n0" not in no_basecase, "a run without an N-0 case has nothing to exempt"

    # Both backends apply the same expression, one eagerly and one to a LazyFrame.
    expression = branch_keep_expr(BranchLoadflowResultFilter(loading_above=THRESHOLD), BASECASE)
    assert results.filter(expression).equals(results.lazy().filter(expression).collect()), (
        "the predicate must decide identically eagerly and lazily"
    )


def test_node_filter():
    """Every rule the node predicate implements, against one set of rows that exercises all of them."""
    results = _node_results()
    all_elements = results["element"].to_list()

    inert = _kept(results, node_keep_expr(NodeLoadflowResultFilter(), BASECASE))
    assert inert == all_elements, "an unconfigured policy must not change the results at all"

    kept = _kept(results, node_keep_expr(NodeLoadflowResultFilter(vm_loading_above=THRESHOLD), BASECASE))
    assert "nominal" not in kept, "a bus sitting at nominal voltage carries no decision value"
    assert "overvoltage" in kept, "a bus near the top of its band is worth keeping"
    assert "undervoltage" in kept, (
        "vm_loading is signed, so the threshold must be compared against its absolute value - "
        "testing vm_loading >= threshold would discard every undervoltage violation in the grid"
    )
    assert "at_threshold" in kept, "vm_loading_above=0.7 keeps a bus exactly 70% of the way to its band edge"
    assert "no_limits" in kept, "a bus with no voltage limits cannot be assessed, so it is never discarded"
    assert "nominal_n0" in kept, "N-0 is the reference the other cases are read against, whatever its voltage"

    with_jump = _kept(
        results,
        node_keep_expr(
            NodeLoadflowResultFilter(vm_loading_above=THRESHOLD, vm_basecase_deviation_above=5.0),
            BASECASE,
        ),
    )
    assert "jumper" not in kept, "without the jump filter, a stable in-band bus is dropped"
    assert "jumper" in with_jump, "a bus that moves sharply between N-0 and N-1 is worth keeping either way"
    assert "nominal" not in with_jump, "the jump exemption must not turn into a keep-everything"

    without_exemption = _kept(
        results,
        node_keep_expr(NodeLoadflowResultFilter(vm_loading_above=THRESHOLD, retain_basecase=False), BASECASE),
    )
    assert "nominal_n0" not in without_exemption, "retain_basecase=False applies the same threshold to N-0"
    assert "overvoltage_n0" in without_exemption, "an N-0 row outside the band clears the threshold on its own merits"

    no_basecase = _kept(results, node_keep_expr(NodeLoadflowResultFilter(vm_loading_above=THRESHOLD), None))
    assert "nominal_n0" not in no_basecase, "a run without an N-0 case has nothing to exempt"


def test_apply_result_filter():
    """The whole-object entry point filters the two tables independently and leaves everything else alone."""
    branch_results = _branch_results().lazy()
    node_results = _node_results().lazy()
    results = LoadflowResultsPolars.model_construct(
        job_id="job",
        branch_results=branch_results,
        node_results=node_results,
    )
    branch_policy = BranchLoadflowResultFilter(loading_above=THRESHOLD)
    node_policy = NodeLoadflowResultFilter(vm_loading_above=THRESHOLD)

    assert apply_result_filter(results, LoadflowResultFilter(), BASECASE) is results, (
        "an inert policy returns the very same object rather than rebuilding it"
    )

    both = apply_result_filter(
        results,
        LoadflowResultFilter(branch_filters=branch_policy, node_filters=node_policy),
        BASECASE,
    )
    assert "quiet" not in both.branch_results.collect()["element"].to_list(), "the branch table should be filtered"
    assert "nominal" not in both.node_results.collect()["element"].to_list(), "the node table should be filtered"

    branch_only = apply_result_filter(results, LoadflowResultFilter(branch_filters=branch_policy), BASECASE)
    assert branch_only.node_results.collect().equals(node_results.collect()), (
        "a branch-only policy must leave the node table untouched"
    )

    node_only = apply_result_filter(results, LoadflowResultFilter(node_filters=node_policy), BASECASE)
    assert node_only.branch_results.collect().equals(branch_results.collect()), (
        "a node-only policy must leave the branch table untouched"
    )
