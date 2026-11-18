# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Integration tests of the limit table contingency analysis on real powsybl networks."""

from dataclasses import replace

import pandas as pd
import polars as pl
import pypowsybl
import pytest
from beartype.typing import Literal, Optional
from toop_engine_contingency_analysis.pypowsybl import (
    PowsyblBranchLimitCache,
    build_branch_limit_cache,
    get_full_nminus1_definition_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table import (
    create_powsybl_analysis_limit_table,
    run_contingency_analysis_powsybl_violation_table,
    run_powsybl_analysis_limit_table,
    translate_nminus1_for_limit_table,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    CGMES_DISTRIBUTED_SLACK,
    SINGLE_SLACK,
    get_default_security_analysis_parameters,
)
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition

BASECASE = "BASECASE"
PERMANENT_NHV1_NHV2_2_SIDE_2 = 500.0


def _branch(element_id: str) -> GridElement:
    return GridElement(id=element_id, kind="branch", type="LINE")


@pytest.fixture
def eurostag_net() -> pypowsybl.network.Network:
    """The eurostag tutorial network with a voltage angle limit, a tight temporary limit and a tight voltage limit.

    - NHV1_NHV2_2 side TWO gets a 10' limit of 600 A below the current after the outage of NHV1_NHV2_1 (1047 A), so
      powsybl reports that temporary limit instead of the permanent 500 A.
    - VLHV1 gets a low voltage limit of 399 kV, violated after the same outage (398.3 kV) but not in the base case.
    """
    net = pypowsybl.network.create_eurostag_tutorial_example1_network()
    net.create_operational_limits(
        pd.DataFrame.from_records(
            index="element_id",
            data=[
                {
                    "element_id": "NHV1_NHV2_2",
                    "name": "permanent_limit",
                    "side": "TWO",
                    "type": "CURRENT",
                    "value": PERMANENT_NHV1_NHV2_2_SIDE_2,
                    "acceptable_duration": -1,
                    "fictitious": False,
                },
                {
                    "element_id": "NHV1_NHV2_2",
                    "name": "10'",
                    "side": "TWO",
                    "type": "CURRENT",
                    "value": 600.0,
                    "acceptable_duration": 600,
                    "fictitious": False,
                },
            ],
        )
    )
    net.create_voltage_angle_limits(
        id="VAL1",
        from_element_id="NHV1_NHV2_1",
        from_side="ONE",
        to_element_id="NHV1_NHV2_1",
        to_side="TWO",
        low_limit=-1.0,
        high_limit=1.0,
    )
    net.update_voltage_levels(id="VLHV1", low_voltage_limit=399.0)
    return net


def _all_monitored(net: pypowsybl.network.Network) -> list[MonitoredElement]:
    """Every branch, bus and switch of the network, with NHV1_NHV2_2 renamed to check the names."""
    return [
        element.model_copy(update={"name": "Line two"}) if element.id == "NHV1_NHV2_2" else element
        for element in get_full_nminus1_definition_powsybl(net).monitored_elements
    ]


@pytest.fixture
def eurostag_definition(eurostag_net: pypowsybl.network.Network) -> Nminus1Definition:
    """Base case, a line outage, a generator outage and a line outage that makes the AC loadflow fail.

    The whole network is monitored.
    """
    return Nminus1Definition(
        monitored_elements=_all_monitored(eurostag_net),
        contingencies=[
            Contingency(id=BASECASE, name="N-0", elements=[]),
            Contingency(id="c1", name="line one", elements=[_branch("NHV1_NHV2_1")]),
            Contingency(id="cgen", elements=[GridElement(id="GEN", kind="injection", type="GENERATOR")]),
            Contingency(id="cfail", elements=[_branch("NGEN_NHV1")]),
        ],
    )


def _run_violation_table(
    net: pypowsybl.network.Network,
    definition: Nminus1Definition,
    method: Literal["ac", "dc"] = "ac",
    batch_size: Optional[int] = None,
    branch_limit_cache: Optional[PowsyblBranchLimitCache] = None,
) -> LoadflowResultsPolars:
    """Run the violation table analysis with the default parameters of the method and no limit reduction."""
    loadflow_params = CGMES_DISTRIBUTED_SLACK if method == "ac" else SINGLE_SLACK
    return run_contingency_analysis_powsybl_violation_table(
        net,
        definition,
        "job",
        0,
        security_analysis_params=get_default_security_analysis_parameters(loadflow_params),
        current_limit_reduction=1.0,
        method=method,
        polars=True,
        batch_size=batch_size,
        branch_limit_cache=branch_limit_cache,
    )


def test_network_is_not_modified(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    """An AC run with distributed slack leaves the generator targets and the variants of the network alone."""
    generators_before = eurostag_net.get_generators()[["target_p", "target_q", "target_v"]].copy()
    loads_before = eurostag_net.get_loads()[["p0", "q0"]].copy()

    _run_violation_table(eurostag_net, eurostag_definition)

    pd.testing.assert_frame_equal(eurostag_net.get_generators()[["target_p", "target_q", "target_v"]], generators_before)
    pd.testing.assert_frame_equal(eurostag_net.get_loads()[["p0", "q0"]], loads_before)
    assert eurostag_net.get_variant_ids() == ["InitialState"]


def test_basecase_is_fully_monitored(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """N-0 carries every branch side and every bus, with p and q, whatever the limit reduction."""
    results = _run_violation_table(eurostag_net, eurostag_definition)
    branches = results.branch_results.collect().filter(pl.col("contingency") == BASECASE)
    nodes = results.node_results.collect().filter(pl.col("contingency") == BASECASE)

    assert len(branches) == 4 * 2
    assert set(branches["element"]) == set(eurostag_net.get_branches().index)
    assert branches["p"].is_nan().sum() == 0
    line_one = branches.filter((pl.col("element") == "NHV1_NHV2_1") & (pl.col("side") == 1)).row(0, named=True)
    assert line_one["loading"] == pytest.approx(line_one["i"] / 500.0)
    unlimited = branches.filter(pl.col("element") == "NGEN_NHV1")
    assert unlimited["loading"].is_nan().all(), "a branch without limits is still reported in N-0"
    assert set(nodes["element"]) == {"NGEN", "NHV1", "NHV2", "NLOAD"}
    assert nodes["vm_basecase_deviation"].to_list() == pytest.approx([0.0] * 4)
    assert set(branches["contingency_name"]) == {"N-0"}
    assert branches.filter(pl.col("element") == "NHV1_NHV2_2")["element_name"].to_list() == ["Line two"] * 2


def test_contingency_violations_are_split(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """Contingencies only show up in the violation tables, which hold the powsybl rows unchanged."""
    results = _run_violation_table(eurostag_net, eurostag_definition)
    branch_violations = results.branch_violations.collect()

    assert set(results.branch_results.collect()["contingency"]) == {BASECASE}
    assert set(results.node_results.collect()["contingency"]) == {BASECASE}
    assert results.va_diff_results.collect().is_empty()
    assert branch_violations.select("contingency", "element", "side", "limit_type", "limit_name").rows() == [
        ("c1", "NHV1_NHV2_2", "TWO", "CURRENT", "10'")
    ], "the temporary limit is exceeded, so powsybl reports the 10' limit"
    row = branch_violations.row(0, named=True)
    assert row["limit"] == 600.0
    assert row["value"] > 600.0
    assert row["loading"] == pytest.approx(row["value"] / 600.0)
    assert row["timestep"] == 0


def test_failed_contingency_is_only_in_the_converged_table(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    results = _run_violation_table(eurostag_net, eurostag_definition)
    converged = results.converged.collect()

    assert converged["contingency"].to_list() == [BASECASE, "c1", "cgen", "cfail"]
    assert dict(zip(converged["contingency"], converged["status"], strict=True)) == {
        BASECASE: "CONVERGED",
        "c1": "CONVERGED",
        "cgen": "CONVERGED",
        "cfail": "SOLVER_FAILED",
    }
    assert converged["contingency_name"].to_list() == ["N-0", "line one", "", ""]
    assert converged["iteration_count"].is_nan().all()
    for frame in (
        results.branch_results,
        results.node_results,
        results.branch_violations,
        results.node_violations,
        results.va_diff_violations,
    ):
        assert "cfail" not in set(frame.collect()["contingency"])


def test_voltage_and_angle_violations(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """Voltage violations are keyed by voltage level, angle violations by the angle limit, base case included."""
    results = _run_violation_table(eurostag_net, eurostag_definition)

    voltage = results.node_violations.collect()
    assert voltage.select("contingency", "element", "limit_type").rows() == [("c1", "VLHV1", "LOW_VOLTAGE")]
    assert voltage["limit"].to_list() == [399.0]
    assert voltage["value"].to_list() == pytest.approx([398.26], abs=0.01)
    nominal_v = eurostag_net.get_voltage_levels().loc["VLHV1", "nominal_v"]
    assert voltage["loading"].to_list() == pytest.approx([(voltage["value"][0] - nominal_v) / (nominal_v - 399.0)])
    angle = results.va_diff_violations.collect().sort("contingency")
    assert set(angle["limit_type"]) == {"LOW_VOLTAGE_ANGLE"}
    assert angle.select("contingency", "element").rows() == [(BASECASE, "VAL1"), ("c1", "VAL1"), ("cgen", "VAL1")]
    values = dict(zip(angle["contingency"], angle["value"], strict=True))
    assert values[BASECASE] == pytest.approx(-3.51, abs=0.01)
    assert values["c1"] == pytest.approx(-7.50, abs=0.01)
    assert angle["loading"].to_list() == pytest.approx((angle["value"] / angle["limit"]).to_list())


def test_violations_of_unmonitored_elements_are_dropped(eurostag_net: pypowsybl.network.Network) -> None:
    """Only violations of monitored branches and of voltage levels holding a monitored bus are kept, angles always."""
    contingencies = [
        Contingency(id=BASECASE, elements=[]),
        Contingency(id="c1", elements=[_branch("NHV1_NHV2_1")]),
    ]
    unmonitored = _run_violation_table(
        eurostag_net,
        Nminus1Definition(
            monitored_elements=[MonitoredElement(id="NHV1_NHV2_1", kind="branch", type="LINE")],
            contingencies=contingencies,
        ),
    )
    assert unmonitored.branch_violations.collect().is_empty(), "NHV1_NHV2_2 is violated after c1 but not monitored"
    assert unmonitored.node_violations.collect().is_empty(), "no bus of VLHV1 is monitored"
    assert set(unmonitored.va_diff_violations.collect()["contingency"]) == {BASECASE, "c1"}, "angles are not filtered"

    monitored = _run_violation_table(
        eurostag_net,
        Nminus1Definition(
            monitored_elements=[
                MonitoredElement(id="NHV1_NHV2_2", kind="branch", type="LINE"),
                MonitoredElement(id="NHV1", kind="bus", type="BUS"),
            ],
            contingencies=contingencies,
        ),
    )
    assert monitored.branch_violations.collect().select("contingency", "element").rows() == [("c1", "NHV1_NHV2_2")]
    assert monitored.node_violations.collect().select("contingency", "element").rows() == [("c1", "VLHV1")]
    assert monitored.warnings == []


def test_without_basecase_contingency(eurostag_net: pypowsybl.network.Network) -> None:
    """Without a base case contingency, nothing is monitored and pre-contingency violations are dropped."""
    definition = Nminus1Definition(
        monitored_elements=_all_monitored(eurostag_net),
        contingencies=[Contingency(id="c1", elements=[_branch("NHV1_NHV2_1")])],
    )
    results = _run_violation_table(eurostag_net, definition)
    assert set(results.converged.collect()["contingency"]) == {"c1"}
    for frame in (results.branch_violations, results.node_violations, results.va_diff_violations):
        assert set(frame.collect()["contingency"]) == {"c1"}
    for frame in (results.branch_results, results.node_results, results.va_diff_results):
        assert frame.collect().is_empty()


def test_batches_match_a_single_run(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    single = _run_violation_table(eurostag_net, eurostag_definition)
    batched = _run_violation_table(eurostag_net, eurostag_definition, batch_size=1)
    for name in (
        "branch_results",
        "node_results",
        "va_diff_results",
        "converged",
        "connectivity_result",
        "branch_violations",
        "node_violations",
        "va_diff_violations",
    ):
        left = getattr(single, name).collect()
        right = getattr(batched, name).collect().select(left.columns)
        sort_columns = [column for column in ("contingency", "element", "side", "limit_type") if column in left.columns]
        pl.testing.assert_frame_equal(left.sort(sort_columns), right.sort(sort_columns))


def test_pandas_results_are_indexed_like_the_polars_tables(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    results = run_contingency_analysis_powsybl_violation_table(
        eurostag_net, eurostag_definition, "job", 0, get_default_security_analysis_parameters(SINGLE_SLACK), 1.0, method="dc"
    )
    assert isinstance(results, LoadflowResults)
    assert results.branch_results.index.names == ["timestep", "contingency", "element", "side"]
    assert results.connectivity_result.index.names == ["contingency", "element"]
    assert results.converged.loc[(0, "cfail"), "status"] == "CONVERGED", "DC does not fail on this outage"
    assert results.branch_violations.index.names == [
        "timestep", "contingency", "element", "side", "limit_type", "limit_name"
    ]  # fmt: skip
    assert results.branch_violations.index.get_level_values("contingency").unique().tolist() == ["c1"]
    assert set(results.branch_violations.index.get_level_values("side")) <= {"ONE", "TWO", "THREE"}
    assert set(results.va_diff_violations.index.get_level_values("side")) == {""}, "angle limits have no side"


def test_dc_run_reports_nominal_voltages_and_no_voltage_violations(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    results = _run_violation_table(eurostag_net, eurostag_definition, method="dc")
    branches = results.branch_results.collect()
    assert set(branches["contingency"]) == {BASECASE}
    assert branches["q"].is_nan().all()
    assert set(results.converged.collect()["contingency"]) == {BASECASE, "c1", "cgen", "cfail"}
    assert set(results.branch_violations.collect()["contingency"]) == {"c1"}
    assert results.node_violations.collect().is_empty(), "no voltage violations in DC"
    assert "cfail" in set(results.va_diff_violations.collect()["contingency"]), "DC does not fail on this outage"
    nodes = results.node_results.collect()
    assert nodes["vm"].to_list() == pytest.approx([24.0, 380.0, 380.0, 150.0]), "nominal voltages in DC"


def test_missing_and_duplicate_contingencies(eurostag_net: pypowsybl.network.Network) -> None:
    """Unknown elements are reported as warnings, repeated single outages and multi outages run fine."""
    definition = Nminus1Definition(
        monitored_elements=[],
        contingencies=[
            Contingency(id="missing", elements=[_branch("NOPE")]),
            Contingency(id="first", elements=[_branch("NHV1_NHV2_1")]),
            Contingency(id="second", elements=[_branch("NHV1_NHV2_1")]),
            Contingency(id="both", elements=[_branch("NHV1_NHV2_1"), _branch("NHV1_NHV2_2")]),
        ],
    )
    results = _run_violation_table(eurostag_net, definition, method="dc")
    converged = results.converged.collect()
    assert converged["contingency"].to_list() == ["first", "second", "both"]
    assert results.warnings == ["Contingency with id missing contains elements that are not found in the network."]
    assert set(results.connectivity_result.collect().filter(pl.col("contingency") == "both")["element"]) >= {
        "NHV1_NHV2_1",
        "NHV1_NHV2_2",
    }, "both lines go, and with them the islanded load"


def test_limit_reduction_reports_violations_of_permanent_and_temporary_limits(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    definition = translate_nminus1_for_limit_table(eurostag_definition, eurostag_net)
    run = run_powsybl_analysis_limit_table(
        eurostag_net,
        create_powsybl_analysis_limit_table(definition, limit_reduction=0.001),
        definition.basecase_id,
        get_default_security_analysis_parameters(SINGLE_SLACK),
        method="dc",
    )
    assert run.basecase_id == BASECASE
    assert run.pre_contingency_status() == "CONVERGED"
    assert {result.contingency_id for result in run.post_contingency_results()} == {"c1", "cgen", "cfail"}
    violations = run.limit_violations().collect()
    assert set(violations["contingency_id"]) == {"", "c1", "cgen", "cfail"}
    current_rows = violations.filter(pl.col("limit_type") == "CURRENT")
    assert set(current_rows["limit_name"]) > {"permanent"}, "temporary limits are violated as well"
    assert set(current_rows["limit_reduction"]) == {0.001}, "permanent and temporary limits are reduced"
    assert not run.branch_results().collect().is_empty(), "the base case is monitored"


def test_branch_limit_cache_is_used(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    """A valid cache provides the branch limits, a stale one is ignored."""
    monitored_branches = translate_nminus1_for_limit_table(eurostag_definition, eurostag_net).monitored_elements["branches"]
    cache = build_branch_limit_cache(eurostag_net, monitored_branches)
    uncached = _run_violation_table(eurostag_net, eurostag_definition)
    assert _run_violation_table(eurostag_net, eurostag_definition, branch_limit_cache=cache) == uncached

    # A marked cache shows that its limits are the ones that end up in the results
    marked = replace(cache, branch_limits=cache.branch_limits * 2)
    limit_key = ["element", "side", "limit_name"]
    marked_limits = (
        _run_violation_table(eurostag_net, eurostag_definition, branch_limit_cache=marked)
        .branch_limits.collect()
        .sort(limit_key)
    )
    uncached_limits = uncached.branch_limits.collect().sort(limit_key)
    assert marked_limits["limit"].to_list() == pytest.approx((uncached_limits["limit"] * 2).to_list())

    stale = replace(marked, current_limit_fingerprint="stale")
    assert _run_violation_table(eurostag_net, eurostag_definition, branch_limit_cache=stale) == uncached
