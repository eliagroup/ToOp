# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Integration tests of the limit table contingency analysis on real powsybl networks."""

import math
import warnings

import pandas as pd
import polars as pl
import pypowsybl
import pytest
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl, run_contingency_analysis_powsybl
from toop_engine_contingency_analysis.pypowsybl_limit_table import (
    LimitViolationFilter,
    run_contingency_analysis_powsybl_limit_table,
    run_powsybl_analysis_limit_table,
    translate_nminus1_for_limit_table,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK, SINGLE_SLACK
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
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


@pytest.fixture
def eurostag_definition() -> Nminus1Definition:
    """Base case, a line outage, a generator outage and a line outage that makes the AC loadflow fail."""
    return Nminus1Definition(
        monitored_elements=[MonitoredElement(id="NHV1_NHV2_2", name="Line two", kind="branch", type="LINE")],
        contingencies=[
            Contingency(id=BASECASE, name="N-0", elements=[]),
            Contingency(id="c1", name="line one", elements=[_branch("NHV1_NHV2_1")]),
            Contingency(id="cgen", elements=[GridElement(id="GEN", kind="injection", type="GENERATOR")]),
            Contingency(id="cfail", elements=[_branch("NGEN_NHV1")]),
        ],
    )


def _run(net: pypowsybl.network.Network, definition: Nminus1Definition, **kwargs: object) -> LoadflowResultsPolars:
    return run_contingency_analysis_powsybl_limit_table(net, definition, "job", 0, polars=True, **{"method": "ac", **kwargs})


def test_network_is_not_modified(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    """An AC run with distributed slack leaves the generator targets and the variants of the network alone."""
    generators_before = eurostag_net.get_generators()[["target_p", "target_q", "target_v"]].copy()
    loads_before = eurostag_net.get_loads()[["p0", "q0"]].copy()

    _run(eurostag_net, eurostag_definition, lf_params=CGMES_DISTRIBUTED_SLACK)

    pd.testing.assert_frame_equal(eurostag_net.get_generators()[["target_p", "target_q", "target_v"]], generators_before)
    pd.testing.assert_frame_equal(eurostag_net.get_loads()[["p0", "q0"]], loads_before)
    assert eurostag_net.get_variant_ids() == ["InitialState"]


def test_basecase_is_fully_monitored(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """N-0 carries every branch side and every bus, with p and q, whatever the limit reduction."""
    results = _run(eurostag_net, eurostag_definition)
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


def test_contingency_rows_come_from_the_violation_table(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """Every loaded branch side with a limit is reported, loaded against the permanent limit, without p and q."""
    results = _run(eurostag_net, eurostag_definition)
    branches = results.branch_results.collect().filter(pl.col("contingency") != BASECASE)

    assert set(branches["contingency"]) == {"c1", "cgen"}, "the failed contingency has no rows"
    assert branches["p"].is_nan().all() and branches["q"].is_nan().all()
    assert "NGEN_NHV1" not in set(branches["element"]), "a branch without limits never violates one"
    c1 = branches.filter(pl.col("contingency") == "c1")
    assert set(zip(c1["element"], c1["side"], strict=True)) == {("NHV1_NHV2_2", 1), ("NHV1_NHV2_2", 2)}
    side_two = c1.filter(pl.col("side") == 2).row(0, named=True)
    assert side_two["i"] > 600.0, "the temporary limit is exceeded, so powsybl reports the 10' limit"
    assert side_two["loading"] == pytest.approx(side_two["i"] / PERMANENT_NHV1_NHV2_2_SIDE_2)
    cgen = branches.filter(pl.col("contingency") == "cgen")
    assert len(cgen) == 4, "unchanged flows are reported too, the increased violation filter is off"
    assert set(cgen["contingency_name"]) == {""}


def test_failed_contingency_is_only_in_the_converged_table(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    results = _run(eurostag_net, eurostag_definition)
    converged = results.converged.collect()

    assert converged["contingency"].to_list() == [BASECASE, "c1", "cgen", "cfail"]
    assert dict(zip(converged["contingency"], converged["status"], strict=True)) == {
        BASECASE: "CONVERGED",
        "c1": "CONVERGED",
        "cgen": "CONVERGED",
        "cfail": "FAILED",
    }
    assert converged["contingency_name"].to_list() == ["N-0", "line one", "", ""]
    assert converged["iteration_count"].is_nan().all()
    for frame in (results.branch_results, results.node_results, results.va_diff_results):
        assert "cfail" not in set(frame.collect()["contingency"])


def test_connectivity_lists_the_disconnected_elements(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    connectivity = _run(eurostag_net, eurostag_definition).connectivity_result.collect()
    by_contingency = connectivity.group_by("contingency").agg(pl.col("element").sort()).to_dict(as_series=False)
    disconnected = dict(zip(by_contingency["contingency"], by_contingency["element"], strict=True))
    assert disconnected["c1"] == ["NHV1_NHV2_1"]
    assert disconnected["cgen"] == ["GEN"]
    assert "NGEN_NHV1" in disconnected["cfail"]
    assert set(connectivity["outage_group_id"]) == set(connectivity["contingency"])


def test_voltage_violations_become_node_rows(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """A contingency node row is keyed by voltage level, signed and compared to the base case bus."""
    results = _run(eurostag_net, eurostag_definition)
    nodes = results.node_results.collect().filter(pl.col("contingency") != BASECASE)
    basecase_nhv1 = results.node_results.collect().filter(pl.col("element") == "NHV1").row(0, named=True)

    assert nodes.select("contingency", "element").rows() == [("c1", "VLHV1")]
    row = nodes.row(0, named=True)
    nominal_v = eurostag_net.get_voltage_levels().loc["VLHV1", "nominal_v"]
    assert row["vm"] == pytest.approx(398.26, abs=0.01)
    assert row["vm_loading"] == pytest.approx((row["vm"] - nominal_v) / (nominal_v - 399.0))
    assert row["vm_basecase_deviation"] == pytest.approx(abs(row["vm"] - basecase_nhv1["vm"]) / basecase_nhv1["vm"] * 100)
    assert math.isnan(row["va"])


def test_angle_limit_violations_become_va_diff_rows(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    va_diff = _run(eurostag_net, eurostag_definition).va_diff_results.collect().sort("contingency")
    assert va_diff.select("contingency", "element").rows() == [(BASECASE, "VAL1"), ("c1", "VAL1"), ("cgen", "VAL1")]
    values = dict(zip(va_diff["contingency"], va_diff["va_diff"], strict=True))
    assert values[BASECASE] == pytest.approx(-3.51, abs=0.01)
    assert values["c1"] == pytest.approx(-7.50, abs=0.01)


def test_regulating_elements_are_empty(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    assert _run(eurostag_net, eurostag_definition).regulating_element_results.collect().is_empty()


def test_loading_threshold_is_pushed_into_powsybl(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """The loading threshold of the result filter becomes the limit reduction, N-0 stays whole."""
    result_filter = LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=0.5))
    results = _run(eurostag_net, eurostag_definition, result_filter=result_filter)
    branches = results.branch_results.collect()

    assert len(branches.filter(pl.col("contingency") == BASECASE)) == 8
    contingency_rows = branches.filter(pl.col("contingency") != BASECASE)
    assert (contingency_rows["loading"] >= 0.5).all()
    assert len(contingency_rows) == 4, "two sides of c1, and the two sides of cgen above 50%"
    assert results.result_filter == result_filter


def test_node_filter_still_applies_after_the_run(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    result_filter = LoadflowResultFilter(node_filters=NodeLoadflowResultFilter(vm_loading_above=5.0))
    nodes = _run(eurostag_net, eurostag_definition, result_filter=result_filter).node_results.collect()
    assert set(nodes["contingency"]) == {BASECASE}, "the 1.7 kV undervoltage of c1 is below the threshold"


def test_worst_overload_per_element(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    results = _run(eurostag_net, eurostag_definition, violation_filter=LimitViolationFilter.WORST_OVERLOAD_PER_ELEMENT)
    branches = results.branch_results.collect()
    assert len(branches.filter(pl.col("contingency") == BASECASE)) == 8
    worst = branches.filter(pl.col("contingency") != BASECASE)
    assert worst["element"].n_unique() == len(worst)
    line_two = worst.filter(pl.col("element") == "NHV1_NHV2_2").row(0, named=True)
    assert (line_two["contingency"], line_two["side"]) == ("c1", 2)
    assert line_two["loading"] > 2.0


def test_worst_element_per_contingency(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    results = _run(eurostag_net, eurostag_definition, violation_filter=LimitViolationFilter.WORST_ELEMENT_PER_CONTINGENCY)
    branches = results.branch_results.collect().filter(pl.col("contingency") != BASECASE)
    nodes = results.node_results.collect().filter(pl.col("contingency") != BASECASE)
    assert branches["contingency"].to_list() == ["c1", "cgen"]
    assert branches.filter(pl.col("contingency") == "c1").select("element", "side").rows() == [("NHV1_NHV2_2", 2)]
    assert nodes.select("contingency", "element").rows() == [("c1", "VLHV1")]


def test_without_basecase_contingency(eurostag_net: pypowsybl.network.Network) -> None:
    definition = Nminus1Definition(
        monitored_elements=[], contingencies=[Contingency(id="c1", elements=[_branch("NHV1_NHV2_1")])]
    )
    results = _run(eurostag_net, definition)
    for frame in (results.branch_results, results.node_results, results.va_diff_results, results.converged):
        assert set(frame.collect()["contingency"]) == {"c1"}


def test_basecase_from_the_violation_table(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    """Without full monitoring the base case has only the branch sides with limits and no voltage row."""
    results = _run(eurostag_net, eurostag_definition, basecase_full_monitoring=False)
    branches = results.branch_results.collect().filter(pl.col("contingency") == BASECASE)
    nodes = results.node_results.collect().filter(pl.col("contingency") == BASECASE)
    assert set(branches["element"]) == {"NHV1_NHV2_1", "NHV1_NHV2_2"}
    assert len(branches) == 4
    assert branches["p"].is_nan().all()
    assert nodes.is_empty()
    assert results.converged.collect()["contingency"].to_list()[0] == BASECASE


def test_batches_match_a_single_run(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    single = _run(eurostag_net, eurostag_definition)
    batched = _run(eurostag_net, eurostag_definition, batch_size=1)
    for name in ("branch_results", "node_results", "va_diff_results", "converged", "connectivity_result"):
        left = getattr(single, name).collect()
        right = getattr(batched, name).collect().select(left.columns)
        sort_columns = [column for column in ("contingency", "element", "side") if column in left.columns]
        pl.testing.assert_frame_equal(left.sort(sort_columns), right.sort(sort_columns))


def test_pandas_results(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    results = run_contingency_analysis_powsybl_limit_table(eurostag_net, eurostag_definition, "job", 0, method="dc")
    assert isinstance(results, LoadflowResults)
    assert results.branch_results.index.names == ["timestep", "contingency", "element", "side"]
    assert results.connectivity_result.index.names == ["contingency", "element"]
    assert results.converged.loc[(0, "cfail"), "status"] == "CONVERGED", "DC does not fail on this outage"


def test_dc_method(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    results = _run(eurostag_net, eurostag_definition, method="dc")
    branches = results.branch_results.collect()
    assert set(branches["contingency"]) == {BASECASE, "c1", "cgen", "cfail"}
    assert branches.filter(pl.col("contingency") == BASECASE)["q"].is_nan().all()
    nodes = results.node_results.collect()
    assert set(nodes["contingency"]) == {BASECASE}, "no voltage violations in DC"
    assert nodes["vm"].to_list() == pytest.approx([24.0, 380.0, 380.0, 150.0]), "nominal voltages in DC"


@pytest.mark.parametrize("limit_reduction", [0.0, -0.1, 1.5])
def test_invalid_limit_reduction(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition, limit_reduction: float
) -> None:
    with pytest.raises(ValueError, match="limit_reduction"):
        _run(eurostag_net, eurostag_definition, limit_reduction=limit_reduction)


def test_invalid_batch_size(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _run(eurostag_net, eurostag_definition, batch_size=0)


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
    results = _run(eurostag_net, definition, method="dc")
    converged = results.converged.collect()
    assert converged["contingency"].to_list() == ["first", "second", "both"]
    assert results.warnings == ["Contingency with id missing contains elements that are not found in the network."]
    assert set(results.connectivity_result.collect().filter(pl.col("contingency") == "both")["element"]) >= {
        "NHV1_NHV2_1",
        "NHV1_NHV2_2",
    }, "both lines go, and with them the islanded load"


def test_low_level_run_keeps_the_native_handle(
    eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition
) -> None:
    definition = translate_nminus1_for_limit_table(eurostag_definition, eurostag_net)
    run = run_powsybl_analysis_limit_table(eurostag_net, definition, SINGLE_SLACK, method="dc")
    assert run.basecase_id == BASECASE
    assert run.pre_contingency_status() == "CONVERGED"
    assert {result.contingency_id for result in run.post_contingency_results()} == {"c1", "cgen", "cfail"}
    violations = run.limit_violations().collect()
    assert set(violations["contingency_id"]) == {"", "c1", "cgen", "cfail"}
    permanent_rows = violations.filter((pl.col("limit_type") == "CURRENT") & (pl.col("limit_name") == "permanent"))
    assert set(permanent_rows["limit_reduction"]) == {0.001}, "only the permanent limits are reduced"
    assert not run.branch_results().collect().is_empty(), "the base case is monitored"


def test_parity_with_the_monitored_element_runner(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    """Where both runners report a row, the currents agree, and the convergence tables are identical."""
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = run_contingency_analysis_powsybl(
            powsybl_bus_breaker_net, nminus1_definition, "job", 0, method="dc", polars=True, lf_params=SINGLE_SLACK
        )
    new = _run(powsybl_bus_breaker_net, nminus1_definition, method="dc", lf_params=SINGLE_SLACK)

    old_converged = old.converged.collect().sort("contingency")
    new_converged = new.converged.collect().sort("contingency")
    assert old_converged.select("contingency", "status").rows() == new_converged.select("contingency", "status").rows()

    keys = ["contingency", "element", "side"]
    old_branches = old.branch_results.collect().select(*keys, "i", "loading")
    new_branches = new.branch_results.collect().select(*keys, "i", "loading")
    basecase_id = nminus1_definition.base_case.id
    old_basecase = old_branches.filter(pl.col("contingency") == basecase_id)
    new_basecase = new_branches.filter(pl.col("contingency") == basecase_id)
    assert set(old_basecase.select("element", "side").rows()) <= set(new_basecase.select("element", "side").rows())

    joined = old_branches.join(new_branches, on=keys, how="inner", suffix="_new")
    assert len(joined) > len(old_basecase), "contingency rows overlap as well"
    assert joined["i"].to_list() == pytest.approx(joined["i_new"].to_list(), rel=1e-6, abs=1e-6)
    comparable = joined.filter(pl.col("loading").is_not_nan() & pl.col("loading_new").is_not_nan())
    assert comparable["loading"].to_list() == pytest.approx(comparable["loading_new"].to_list(), rel=1e-6)
    limit_order = ["element", "side", "limit_name"]
    old_limits = old.branch_limits.collect().sort(limit_order)
    pl.testing.assert_frame_equal(old_limits, new.branch_limits.collect().sort(limit_order))


def test_contingency_propagation_flag(powsybl_node_breaker_net: pypowsybl.network.Network) -> None:
    """Native propagation runs on a node breaker network and reports at least what the plain run reports."""
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_node_breaker_net)
    plain = _run(powsybl_node_breaker_net, nminus1_definition, method="dc", contingency_propagation=False)
    propagated = _run(powsybl_node_breaker_net, nminus1_definition, method="dc", contingency_propagation=True)
    assert set(plain.converged.collect()["contingency"]) == set(propagated.converged.collect()["contingency"])
    assert len(propagated.connectivity_result.collect()) >= len(plain.connectivity_result.collect())


def test_old_runner_is_deprecated(eurostag_net: pypowsybl.network.Network, eurostag_definition: Nminus1Definition) -> None:
    with pytest.warns(DeprecationWarning, match="run_contingency_analysis_powsybl_limit_table"):
        run_contingency_analysis_powsybl(eurostag_net, eurostag_definition, "job", 0, method="dc", polars=True)
