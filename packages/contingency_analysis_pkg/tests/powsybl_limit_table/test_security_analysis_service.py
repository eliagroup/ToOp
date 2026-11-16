# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Regression tests: both runners behind get_security_analysis_results give the same base case and worst contingencies."""

import polars as pl
import pypowsybl
import pytest
from pypowsybl.security import IncreasedViolationsParameters
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_security_analysis_results
from toop_engine_contingency_analysis.compute_metrics import get_worst_k_contingencies_ac
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    CGMES_DISTRIBUTED_SLACK,
    get_default_security_analysis_parameters,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

GRIDS = ["powsybl_bus_breaker_net", "powsybl_node_breaker_net", "powsybl_complex_node_breaker_net"]


def _violations_as_branch_results(results: LoadflowResultsPolars) -> pl.LazyFrame:
    """The current violations as branch results, with the loading against the permanent limit.

    Whichever limit powsybl reported, the loading is recomputed against the permanent limit like the monitored branch
    results. Branch sides without a violation are left out, they have no overload. The runner already dropped the
    violations of branches that are not monitored.
    """
    permanent_limits = results.branch_limits.filter(pl.col("limit_name") == "permanent_limit").select(
        "timestep", "element", "side", "limit"
    )
    return (
        results.branch_violations.filter(pl.col("limit_type") == "CURRENT")
        .with_columns(pl.col("side").replace_strict({"ONE": 1, "TWO": 2, "THREE": 3}, return_dtype=pl.Int64))
        .group_by("timestep", "contingency", "element", "side")
        .agg(pl.col("value").abs().max().alias("i"))
        .join(permanent_limits, on=["timestep", "element", "side"], how="inner")
        .select(
            "timestep",
            "contingency",
            "element",
            "side",
            "i",
            pl.lit(float("nan")).alias("p"),
            pl.lit(float("nan")).alias("q"),
            (pl.col("i") / pl.col("limit")).alias("loading"),
            pl.lit("").alias("element_name"),
            pl.lit("").alias("contingency_name"),
        )
    )


def _run_monitored_and_violation_sa(
    net: pypowsybl.network.Network, security_analysis_params: pypowsybl.security.Parameters
) -> tuple[LoadflowResultsPolars, LoadflowResultsPolars, Nminus1Definition]:
    definition = get_full_nminus1_definition_powsybl(net)
    monitored = get_security_analysis_results(net, definition, security_analysis_params=security_analysis_params)
    violations = get_security_analysis_results(
        net, definition, security_analysis_params=security_analysis_params, use_limit_violations=True
    )
    return monitored, violations, definition


def _assert_same_worst_contingencies(
    monitored: LoadflowResultsPolars,
    violations: LoadflowResultsPolars,
    definition: Nminus1Definition,
    k: int,
    compare_overload_energy: bool,
) -> None:
    basecase_id = definition.base_case.id
    expected, expected_energy = get_worst_k_contingencies_ac(
        monitored.branch_results,
        monitored.converged,
        k=k,
        field="i",
        base_case_id=basecase_id,
        include_non_converging_loadflows=False,
    )
    actual, actual_energy = get_worst_k_contingencies_ac(
        _violations_as_branch_results(violations),
        violations.converged,
        k=k,
        field="i",
        base_case_id=basecase_id,
        include_non_converging_loadflows=False,
    )
    assert expected, "the grid needs N-1 overloads for the comparison to mean anything"
    # Compared as sets: the worst k do not break ties, e.g. of parallel lines, so their order is not stable
    assert [set(contingencies) for contingencies in actual] == [set(contingencies) for contingencies in expected]
    if compare_overload_energy:
        assert actual_energy == pytest.approx(expected_energy, rel=1e-6)


@pytest.mark.parametrize("grid", GRIDS)
def test_same_worst_contingencies_with_default_parameters(grid: str, request: pytest.FixtureRequest) -> None:
    """With the default increased violation thresholds, the top of the ranking is the same.

    Violations already present in N-0 that did not grow by the threshold are not reported, so only the tail of the
    ranking and the overload energy of the worst contingencies may differ.
    """
    monitored, violations, definition = _run_monitored_and_violation_sa(
        request.getfixturevalue(grid), get_default_security_analysis_parameters()
    )

    _assert_same_worst_contingencies(monitored, violations, definition, k=10, compare_overload_energy=False)
    assert (
        violations.converged.collect().sort("contingency").select("contingency", "status").rows()
        == monitored.converged.collect().sort("contingency").select("contingency", "status").rows()
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_same_worst_contingencies_without_increased_violation_filter(grid: str, request: pytest.FixtureRequest) -> None:
    """Without the increased violation filter, every overloaded contingency is ranked the same, with the same energy."""
    report_every_violation = IncreasedViolationsParameters(
        flow_proportional_threshold=-1.0,
        low_voltage_proportional_threshold=0.0,
        low_voltage_absolute_threshold=0.0,
        high_voltage_proportional_threshold=0.0,
        high_voltage_absolute_threshold=0.0,
    )
    monitored, violations, definition = _run_monitored_and_violation_sa(
        request.getfixturevalue(grid),
        pypowsybl.security.Parameters(
            load_flow_parameters=CGMES_DISTRIBUTED_SLACK, increased_violations_parameters=report_every_violation
        ),
    )

    _assert_same_worst_contingencies(
        monitored, violations, definition, k=len(monitored.converged.collect()), compare_overload_energy=True
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_same_basecase_results(grid: str, request: pytest.FixtureRequest) -> None:
    """The base case of the violation run monitors exactly what the monitored element run monitors."""
    monitored, violations, definition = _run_monitored_and_violation_sa(
        request.getfixturevalue(grid), get_default_security_analysis_parameters()
    )
    basecase = pl.col("contingency") == definition.base_case.id
    keys = {
        "branch_results": ["element", "side"],
        "node_results": ["element"],
        "regulating_element_results": ["element"],
    }
    for name, key in keys.items():
        expected = getattr(monitored, name).filter(basecase).collect().sort(key)
        actual = getattr(violations, name).filter(basecase).collect().sort(key).select(expected.columns)
        assert not expected.is_empty(), name
        pl.testing.assert_frame_equal(actual, expected, check_exact=False, rel_tol=1e-6, abs_tol=1e-6)
    limit_key = ["element", "side", "limit_name"]
    pl.testing.assert_frame_equal(
        violations.branch_limits.collect().sort(limit_key), monitored.branch_limits.collect().sort(limit_key)
    )
    assert violations.warnings == monitored.warnings
