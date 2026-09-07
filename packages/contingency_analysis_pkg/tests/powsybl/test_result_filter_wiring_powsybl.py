# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests that a result filter policy reaches the powsybl result frames it is meant to shrink.

The predicates themselves are covered in ``tests/test_result_filter.py``.
"""

import polars as pl
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_contingency_analysis.result_filter import apply_result_filter
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)

BASECASE = "BASECASE"


def test_filter_reaches_the_powsybl_results(powsybl_bus_breaker_net):
    """The policy arrives through the AC service entry point, shrinks both tables, and leaves N-0 whole."""
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    assert nminus1_definition.base_case is not None, "this test needs an N-0 case to check the exemption"

    unfiltered = get_ac_loadflow_results(powsybl_bus_breaker_net, nminus1_definition, job_id="test_job")
    unfiltered_branches = unfiltered.branch_results.collect()
    unfiltered_nodes = unfiltered.node_results.collect()

    branch_threshold = 0.7
    node_threshold = 0.2
    policy = LoadflowResultFilter(
        branch_filters=BranchLoadflowResultFilter(loading_above=branch_threshold),
        node_filters=NodeLoadflowResultFilter(vm_loading_above=node_threshold),
    )

    filtered = get_ac_loadflow_results(powsybl_bus_breaker_net, nminus1_definition, job_id="test_job", result_filter=policy)
    filtered_branches = filtered.branch_results.collect()
    filtered_nodes = filtered.node_results.collect()

    assert filtered_branches.height < unfiltered_branches.height, "a median threshold must drop some branch rows"
    assert filtered_nodes.height < unfiltered_nodes.height, "a median threshold must drop some node rows"

    # Roughly half this grid's branch rows have no rated current, which is what makes the unknown-is-kept assertions
    # below non-vacuous. Guard it, so a fixture that gained ratings would be reported rather than silently turning those
    # assertions into no-ops.
    unknown = pl.col("loading").is_nan() | pl.col("loading").is_null()
    assert unfiltered_branches.filter(unknown).height > 0, (
        "the fixture must contain branches without a rated current for the unknown-is-kept check below to mean anything"
    )

    dropped_branches = unfiltered_branches.join(
        filtered_branches.select(["contingency", "element", "side"]), on=["contingency", "element", "side"], how="anti"
    )
    assert (dropped_branches["contingency"] != BASECASE).all(), "N-0 branch rows are exempt"
    assert dropped_branches["loading"].fill_nan(None).drop_nulls().lt(branch_threshold).all(), (
        "every dropped branch row must have been below the threshold"
    )
    assert dropped_branches.filter(unknown).height == 0, "a branch whose loading is unknown must never be dropped"

    dropped_nodes = unfiltered_nodes.join(
        filtered_nodes.select(["contingency", "element"]), on=["contingency", "element"], how="anti"
    )
    assert (dropped_nodes["contingency"] != BASECASE).all(), "N-0 node rows are exempt"
    assert dropped_nodes["vm_loading"].fill_nan(None).drop_nulls().abs().lt(node_threshold).all(), (
        "every dropped node row must have been inside the band"
    )
    assert dropped_nodes.filter(pl.col("vm_loading").is_nan() | pl.col("vm_loading").is_null()).height == 0, (
        "a node whose vm_loading is unknown must never be dropped"
    )

    assert filtered_branches.filter(pl.col("contingency") == BASECASE).equals(
        unfiltered_branches.filter(pl.col("contingency") == BASECASE)
    ), "the N-0 branch rows come through whole"
    assert filtered_nodes.filter(pl.col("contingency") == BASECASE).equals(
        unfiltered_nodes.filter(pl.col("contingency") == BASECASE)
    ), "the N-0 node rows come through whole"

    # Filtering inside the lazy pipeline has to mean exactly the same thing as filtering the finished result. This is
    # what pins the application point: run any earlier, the deviation and name columns are not final yet.
    expected = apply_result_filter(unfiltered, policy, nminus1_definition.base_case.id)
    assert filtered_branches.equals(expected.branch_results.collect()), (
        "in-pipeline filtering must match filtering the finished result"
    )
    assert filtered_nodes.equals(expected.node_results.collect()), (
        "in-pipeline filtering must match filtering the finished result"
    )
