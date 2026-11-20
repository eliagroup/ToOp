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
from beartype.typing import Optional
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_security_analysis_results
from toop_engine_contingency_analysis.compute_metrics import (
    compute_metrics,
    get_branch_results_for_metrics,
    get_worst_k_contingencies_ac,
    is_unavailable_from_limit_violations,
)
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_grid_helpers.powsybl.loadflow_parameters import get_default_security_analysis_parameters
from toop_engine_interfaces.loadflow_result_filter import BranchLoadflowResultFilter, LoadflowResultFilter
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition
from toop_engine_interfaces.types import OperationalLimit

BASECASE_ID = "BASECASE"
GRIDS = ["powsybl_bus_breaker_net", "powsybl_node_breaker_net", "powsybl_complex_node_breaker_net"]


def _run_monitored_and_violation_table_analysis(
    net: pypowsybl.network.Network,
    violation_table_result_filter: Optional[LoadflowResultFilter] = None,
) -> tuple[LoadflowResultsPolars, LoadflowResultsPolars, Nminus1Definition]:
    """Run the full N-1 of the network once monitoring every element and once from the violation table.

    Both use the default security analysis parameters. The result filter is only given to the violation table run, so
    the monitored run stays the unfiltered reference. Returns the monitored results, the violation table results and
    the N-1 definition.
    """
    definition = get_full_nminus1_definition_powsybl(net)
    security_analysis_params = get_default_security_analysis_parameters()
    monitored = get_security_analysis_results(net, definition, security_analysis_params=security_analysis_params)
    violations = get_security_analysis_results(
        net,
        definition,
        security_analysis_params=security_analysis_params,
        result_filter=violation_table_result_filter,
        use_limit_violations=True,
    )
    return monitored, violations, definition


@pytest.mark.parametrize("grid", GRIDS)
def test_violation_table_ranks_worst_contingencies_like_monitored_runner(grid: str, request: pytest.FixtureRequest) -> None:
    """With the default parameters, every overloaded contingency is ranked the same, with the same overload energy.

    The default flow threshold of the increased violations reports every N-1 flow violation, also those that are not
    higher than in N-0, see get_increased_violations_parameters().
    """
    monitored, violations, definition = _run_monitored_and_violation_table_analysis(request.getfixturevalue(grid))

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
def test_violation_table_runner_has_the_basecase_results_of_monitored_runner(
    grid: str, request: pytest.FixtureRequest
) -> None:
    """The base case of the violation run monitors exactly what the monitored element run monitors."""
    monitored, violations, definition = _run_monitored_and_violation_table_analysis(request.getfixturevalue(grid))
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


@pytest.mark.parametrize("loading_above", [None, 0.8], ids=["no_result_filter", "loading_above_0.8"])
def test_violation_table_metrics_match_monitored_runner_or_are_none(
    powsybl_complex_node_breaker_net: pypowsybl.network.Network, loading_above: Optional[float]
) -> None:
    """With the default parameters, the metrics from the violation table are those of the monitored run.

    The metrics the table cannot provide are None, see is_unavailable_from_limit_violations. The deprecated branch
    result filter becomes the current limit reduction: the table then also reports the branch sides loaded above
    loading_above, which leaves the overload metrics unchanged. Powsybl does not reduce the limits of tie lines.
    """
    _add_artificial_limits(powsybl_complex_node_breaker_net)
    result_filter = (
        None
        if loading_above is None
        else LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=loading_above))
    )
    monitored, violations, definition = _run_monitored_and_violation_table_analysis(
        powsybl_complex_node_breaker_net, violation_table_result_filter=result_filter
    )
    is_tie_line = pl.col("element").is_in(powsybl_complex_node_breaker_net.get_tie_lines().index.to_list())
    current_violations = violations.branch_violations.filter(pl.col("limit_type") == "CURRENT").collect()
    limit_reduction = 1.0 if loading_above is None else loading_above
    assert set(current_violations.filter(~is_tie_line)["limit_reduction"]) == {limit_reduction}
    assert set(current_violations.filter(is_tie_line)["limit_reduction"]) == {1.0}, "tie line limits are not reduced"

    displayed = [element.id for element in definition.monitored_elements if element.kind == "branch"][:5]
    expected = compute_metrics(monitored, base_case_id=definition.base_case.id, displayed_branch_ids=displayed)
    actual = compute_metrics(violations, base_case_id=definition.base_case.id, displayed_branch_ids=displayed)

    unavailable = {name for name in expected if is_unavailable_from_limit_violations(name)}
    assert unavailable == {
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


def test_busbar_outages_match_the_monitored_element_runner(
    powsybl_complex_node_breaker_net: pypowsybl.network.Network,
) -> None:
    """Busbar outages overload the same branches and get the same statuses in both runners, powsybl propagates them."""
    net = powsybl_complex_node_breaker_net
    full_definition = get_full_nminus1_definition_powsybl(net)
    busbar_outages = [
        Contingency(id=f"BB_{busbar_id}", elements=[GridElement(id=busbar_id, type="BUSBAR_SECTION", kind="bus")])
        for busbar_id in net.get_busbar_sections(attributes=[]).index
    ]
    definition = full_definition.model_copy(update={"contingencies": [full_definition.base_case, *busbar_outages]})
    security_params = get_default_security_analysis_parameters()
    monitored = get_security_analysis_results(net, definition, security_analysis_params=security_params)
    violations = get_security_analysis_results(
        net, definition, security_analysis_params=security_params, use_limit_violations=True
    )
    assert security_params.provider_parameters == {"contingencyPropagation": "true"}, "the caller's are not modified"

    names = ["overload_current_permanent_n_1", "critical_branch_count_permanent_n_1", "max_flow_permanent_n_1"]
    expected = compute_metrics(monitored, base_case_id=definition.base_case.id)
    actual = compute_metrics(violations, base_case_id=definition.base_case.id)
    assert expected["overload_current_permanent_n_1"] > 0, "the busbar outages overload the grid"
    assert {name: actual[name] for name in names} == pytest.approx({name: expected[name] for name in names}, rel=1e-6)

    statuses = (
        monitored.converged.select("contingency", pl.col("status").alias("monitored"))
        .join(violations.converged.select("contingency", pl.col("status").alias("violations")), on="contingency")
        .filter(pl.col("monitored") != pl.col("violations"))
        .collect()
    )
    assert statuses.is_empty(), statuses


def _connectivity(results: LoadflowResultsPolars) -> dict[str, list[str]]:
    rows = results.connectivity_result.collect()
    assert (rows["outage_group_id"] == rows["contingency"]).all(), "powsybl has no outage groups"
    return {
        contingency: sorted(group["element"].to_list())
        for (contingency,), group in rows.group_by("contingency", maintain_order=True)
    }


def test_both_runners_fill_the_connectivity_table(powsybl_complex_node_breaker_net: pypowsybl.network.Network) -> None:
    """Both runners list what each contingency disconnects, an island cut off by a line outage included."""
    net = powsybl_complex_node_breaker_net
    line = lambda element_id: GridElement(id=element_id, type="LINE", kind="branch")  # noqa: E731
    definition = get_full_nminus1_definition_powsybl(net).model_copy(
        update={
            "contingencies": [
                Contingency(id="BASECASE", elements=[]),
                Contingency(id="island", elements=[line("L_NL_5_1")]),
                Contingency(id="plain", elements=[line("L1")]),
                Contingency(id="multi", elements=[line("L_NL_5_1"), line("L_NL_4_2")]),
            ]
        }
    )
    security_params = get_default_security_analysis_parameters()
    monitored = get_security_analysis_results(net, definition, security_analysis_params=security_params, batch_size=1)
    violations = get_security_analysis_results(
        net, definition, security_analysis_params=security_params, use_limit_violations=True, batch_size=1
    )

    expected_island = ["GEN_GB_deep_island", *(f"L_NL_5_{i}" for i in range(1, 8)), "load_GB_deep_island"]
    for results in (monitored, violations):
        connectivity = _connectivity(results)
        assert BASECASE_ID not in connectivity, "the base case disconnects nothing"
        assert connectivity["island"] == sorted(expected_island)
        assert connectivity["plain"] == ["L1"]
        assert connectivity["multi"] == sorted([*expected_island, "L_NL_4_2", "load_NL_4_2"])
    assert _connectivity(monitored) == _connectivity(violations)


@pytest.mark.parametrize("with_basecase", [True, False], ids=["with_basecase", "without_basecase"])
@pytest.mark.parametrize("use_limit_violations", [False, True], ids=["monitored", "violation_table"])
def test_failed_base_case_fails_every_contingency(with_basecase: bool, use_limit_violations: bool) -> None:
    """Powsybl computes no contingency when the base case fails, so none of them may count as harmlessly skipped."""
    net = pypowsybl.network.create_eurostag_tutorial_example1_network()
    net.update_loads(id="LOAD", p0=50000.0)  # far beyond what the grid can supply
    definition = get_full_nminus1_definition_powsybl(net)
    if not with_basecase:
        definition = definition.model_copy(
            update={
                "contingencies": [contingency for contingency in definition.contingencies if not contingency.is_basecase()]
            }
        )

    results = get_security_analysis_results(
        net,
        definition,
        security_analysis_params=get_default_security_analysis_parameters(),
        use_limit_violations=use_limit_violations,
    )

    statuses = results.converged.collect()["status"].to_list()
    assert len(statuses) == len(definition.contingencies)
    assert set(statuses) == {"MAX_ITERATION_REACHED"}
