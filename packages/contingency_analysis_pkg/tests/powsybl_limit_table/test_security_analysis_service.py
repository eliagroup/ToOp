# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Regression tests: both runners behind get_security_analysis_results give the same base case, worst contingencies
and metrics."""

import pandas as pd
import polars as pl
import pypowsybl
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_security_analysis_results
from toop_engine_contingency_analysis.compute_metrics import (
    compute_metrics,
    get_branch_results_for_metrics,
    get_worst_k_contingencies_ac,
    is_unavailable_from_limit_violations,
)
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_grid_helpers.powsybl.loadflow_parameters import get_default_security_analysis_parameters
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.types import OperationalLimit

GRIDS = ["powsybl_bus_breaker_net", "powsybl_node_breaker_net", "powsybl_complex_node_breaker_net"]


def _run_monitored_and_violation_sa(
    net: pypowsybl.network.Network, security_analysis_params: pypowsybl.security.Parameters
) -> tuple[LoadflowResultsPolars, LoadflowResultsPolars, Nminus1Definition]:
    definition = get_full_nminus1_definition_powsybl(net)
    monitored = get_security_analysis_results(net, definition, security_analysis_params=security_analysis_params)
    violations = get_security_analysis_results(
        net, definition, security_analysis_params=security_analysis_params, use_limit_violations=True
    )
    return monitored, violations, definition


@pytest.mark.parametrize("grid", GRIDS)
def test_same_worst_contingencies(grid: str, request: pytest.FixtureRequest) -> None:
    """With the default parameters, every overloaded contingency is ranked the same, with the same overload energy.

    The default flow threshold of the increased violations reports every N-1 flow violation, also those that are not
    higher than in N-0, see VIOLATION_THRESHOLD.
    """
    monitored, violations, definition = _run_monitored_and_violation_sa(
        request.getfixturevalue(grid), get_default_security_analysis_parameters()
    )

    def worst_k(results: LoadflowResultsPolars) -> tuple[list[list[str]], list[float]]:
        return get_worst_k_contingencies_ac(
            get_branch_results_for_metrics(results),
            results.converged,
            k=len(definition.contingencies),
            field="i",
            base_case_id=definition.base_case.id,
            include_non_converging_loadflows=False,
            branch_limits=results.branch_limits,
            limit_scope="permanent",
        )

    expected, expected_energy = worst_k(monitored)
    actual, actual_energy = worst_k(violations)
    assert expected, "the grid needs N-1 overloads for the comparison to mean anything"
    # Compared as sets: the ranking does not break ties, e.g. of parallel lines, so the order is not stable
    assert [set(contingencies) for contingencies in actual] == [set(contingencies) for contingencies in expected]
    assert actual_energy == pytest.approx(expected_energy, rel=1e-6)
    assert (
        violations.converged.collect().sort("contingency").select("contingency", "status").rows()
        == monitored.converged.collect().sort("contingency").select("contingency", "status").rows()
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


def _add_artificial_limits(net: pypowsybl.network.Network) -> None:
    """Add artificial limits below the permanent limit of every line and two winding transformer, like the importer.

    The importer writes them as temporary current limits, see create_artificial_limits of the importer. They are below
    the permanent limit, so the violation table reports branch sides that are only overloaded against them.
    """
    limits = net.get_operational_limits(all_attributes=True).reset_index()
    limits = limits[limits["element_type"].isin(["LINE", "TWO_WINDINGS_TRANSFORMER"])]
    permanent = limits[(limits["type"] == "CURRENT") & (limits["acceptable_duration"] == -1)]
    artificial = [
        permanent.assign(name=limit.value, acceptable_duration=limit.acceptable_duration, value=permanent["value"] * share)
        for limit, share in ((OperationalLimit.ARTIFICIAL_N_0, 0.7), (OperationalLimit.ARTIFICIAL_N_1, 0.9))
    ]
    columns = ["element_id", "element_type", "side", "name", "type", "value", "acceptable_duration", "group_name"]
    net.create_operational_limits(pd.concat([limits, *artificial])[columns].set_index("element_id"))


def test_same_metrics_from_violation_table(powsybl_complex_node_breaker_net: pypowsybl.network.Network) -> None:
    """With the default parameters, the metrics from the violation table are those of the monitored run.

    The metrics the table cannot provide are None, see is_unavailable_from_limit_violations.
    """
    _add_artificial_limits(powsybl_complex_node_breaker_net)
    monitored, violations, definition = _run_monitored_and_violation_sa(
        powsybl_complex_node_breaker_net, get_default_security_analysis_parameters()
    )
    displayed = [element.id for element in definition.monitored_elements if element.kind == "branch"][:5]
    expected = compute_metrics(monitored, base_case_id=definition.base_case.id, displayed_branch_ids=displayed)
    actual = compute_metrics(violations, base_case_id=definition.base_case.id, displayed_branch_ids=displayed)

    unavailable = {name for name in expected if is_unavailable_from_limit_violations(name)}
    assert unavailable == {
        "overload_energy_permanent_n_1",
        "overload_energy_artificial_limit_n_1",
        "overload_energy_displayed_permanent_n_1",
        "max_va_diff_n_1",
        "critical_va_diff_count_n_1",
        "voltage_jump_count_n_1",
    }
    assert {name: actual[name] for name in unavailable} == dict.fromkeys(unavailable)
    available = {name: value for name, value in expected.items() if name not in unavailable}
    assert {name: actual[name] for name in available} == pytest.approx(available, rel=1e-9)
    # The comparison means something: the grid is overloaded against both limits in N-1
    assert available["overload_current_permanent_n_1"] > 0
    assert available["overload_current_artificial_limit_n_1"] > available["overload_current_permanent_n_1"]


def test_artificial_limit_overload_of_a_single_outage() -> None:
    """One outage overloads one line: the metrics of each scope are measured against the limit of that scope.

    NHV1_NHV2_2 is the only monitored branch and carries about 1000 A after the outage of its parallel line. Its
    artificial N-1 limit of 800 A is below its permanent limit of 900 A, so the artificial overload is 100 A larger.
    """
    permanent_limit, artificial_n_0, artificial_n_1 = 900.0, 700.0, 800.0
    line, outage = "NHV1_NHV2_2", "NHV1_NHV2_1"
    net = pypowsybl.network.create_eurostag_tutorial_example1_network()
    limits = {
        OperationalLimit.PERMANENT.value: (permanent_limit, -1),
        OperationalLimit.ARTIFICIAL_N_0.value: (artificial_n_0, OperationalLimit.ARTIFICIAL_N_0.acceptable_duration),
        OperationalLimit.ARTIFICIAL_N_1.value: (artificial_n_1, OperationalLimit.ARTIFICIAL_N_1.acceptable_duration),
    }
    net.create_operational_limits(
        pd.DataFrame.from_records(
            index="element_id",
            data=[
                {
                    "element_id": line,
                    "name": name,
                    "side": side,
                    "type": "CURRENT",
                    "value": value,
                    "acceptable_duration": duration,
                    "fictitious": False,
                }
                for side in ("ONE", "TWO")
                for name, (value, duration) in limits.items()
            ],
        )
    )
    full_definition = get_full_nminus1_definition_powsybl(net)
    definition = full_definition.model_copy(
        update={
            "monitored_elements": [element for element in full_definition.monitored_elements if element.id == line],
            "contingencies": [
                contingency
                for contingency in full_definition.contingencies
                if contingency.is_basecase() or contingency.id == outage
            ],
        }
    )
    basecase_id = definition.base_case.id

    monitored = get_security_analysis_results(
        net, definition, security_analysis_params=get_default_security_analysis_parameters()
    )
    violations = get_security_analysis_results(
        net, definition, security_analysis_params=get_default_security_analysis_parameters(), use_limit_violations=True
    )

    # The reference current is the worst side of the line after the outage, from the monitored element runner
    current = monitored.branch_results.filter(pl.col("contingency") == outage).select(pl.col("i").max()).collect().item()
    assert artificial_n_1 < permanent_limit < current < 1100.0, "the outage overloads the line against both limits"

    reported = violations.branch_violations.filter(pl.col("contingency") == outage).collect()
    assert set(reported["element"]) == {line}
    assert reported["value"].max() == pytest.approx(current, rel=1e-6)

    # The overload artifical vs permanent should be based on the limit and should reflect this in the current metrics
    expected = {
        "overload_current_permanent_n_1": current - permanent_limit,
        "overload_current_artificial_limit_n_1": current - artificial_n_1,
        "max_flow_permanent_n_1": current / permanent_limit,
        "max_flow_artificial_limit_n_1": current / artificial_n_1,
        "critical_branch_count_permanent_n_1": 1,
        "critical_branch_count_artificial_limit_n_1": 1,
    }
    for results in (monitored, violations):
        metrics = compute_metrics(results, base_case_id=basecase_id)
        assert {name: metrics[name] for name in expected} == pytest.approx(expected, rel=1e-6)
