# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Unit tests of the extraction from the limit violation table, on hand-built frames."""

import math
from dataclasses import dataclass
from enum import Enum

import polars as pl
import pytest
from toop_engine_contingency_analysis.pypowsybl_limit_table import (
    LimitViolationFilter,
    apply_limit_violation_filter,
    get_branch_results_from_violations,
    get_connectivity_results,
    get_convergence_results,
    get_node_results_from_violations,
    get_va_diff_results_from_violations,
    normalize_limit_violations,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table.extract import NO_IMPACT_WARNING

NAN = float("nan")


def _violations(rows: list[dict]) -> pl.LazyFrame:
    """Build a normalized violation table from row dicts, defaults filled in."""
    defaults = {
        "contingency": "",
        "element": "",
        "subject_name": "",
        "limit_type": "CURRENT",
        "limit_name": "permanent",
        "limit": NAN,
        "acceptable_duration": 2147483647,
        "limit_reduction": 1.0,
        "value": NAN,
        "side": "",
    }
    return pl.LazyFrame([{**defaults, **row} for row in rows]).cast(
        {"limit": pl.Float64, "value": pl.Float64, "limit_reduction": pl.Float64, "acceptable_duration": pl.Int64}
    )


class FakeStatus(Enum):
    CONVERGED = 0
    NO_IMPACT = 1
    SOLVER_FAILED = 2
    MAX_ITERATION_REACHED = 3


@dataclass
class FakePostContingencyResult:
    contingency_id: str
    status: FakeStatus
    disconnected_elements: list[str]


def test_normalize_limit_violations_casts_an_empty_table() -> None:
    """An empty native table arrives with null dtypes and must still get the string and float columns."""
    raw = pl.LazyFrame({"contingency_id": [], "subject_id": [], "subject_name": [], "limit_type": [], "limit_name": [],
                        "limit": [], "acceptable_duration": [], "limit_reduction": [], "value": [], "side": []})  # fmt: skip
    normalized = normalize_limit_violations(raw)
    schema = normalized.collect_schema()
    assert schema["contingency"] == pl.String
    assert schema["element"] == pl.String
    assert schema["value"] == pl.Float64
    assert normalized.filter(pl.col("limit_type") == "CURRENT").collect().is_empty()


def test_branch_results_use_the_permanent_limit_from_the_table() -> None:
    """A row reporting a temporary limit still gets its loading against the permanent limit of the limit table."""
    violations = _violations(
        [
            {"contingency": "c1", "element": "L1", "limit_name": "10'", "limit": 600.0, "value": 1000.0, "side": "TWO"},
            {"contingency": "c1", "element": "L2", "limit_name": "permanent", "limit": 400.0, "value": 200.0, "side": "ONE"},
            {"contingency": "c1", "element": "L3", "limit_name": "10'", "limit": 800.0, "value": 900.0, "side": "ONE"},
            {"contingency": "", "element": "L1", "limit_name": "permanent", "limit": 500.0, "value": 100.0, "side": "ONE"},
            {"contingency": "c1", "element": "VL1", "limit_type": "LOW_VOLTAGE", "limit": 380.0, "value": 370.0},
        ]
    )
    branch_limits = pl.LazyFrame({"element_id": ["L1"], "side": [2], "value": [500.0]})

    result = get_branch_results_from_violations(violations, branch_limits, timestep=3, drop_basecase=False).collect()

    assert set(result["element"]) == {"L1", "L2", "L3"}, "voltage rows are not branch rows"
    by_element = {row["element"]: row for row in result.filter(pl.col("contingency") == "c1").to_dicts()}
    assert by_element["L1"]["loading"] == pytest.approx(1000.0 / 500.0), "table permanent limit wins over the row"
    assert by_element["L1"]["side"] == 2
    assert by_element["L2"]["loading"] == pytest.approx(0.5), "a permanent row without table entry uses its own limit"
    assert math.isnan(by_element["L3"]["loading"]), "a temporary row without table entry has no permanent limit"
    assert all(math.isnan(value) for value in result["p"]) and all(math.isnan(value) for value in result["q"])
    assert result["i"].to_list()[0] == pytest.approx(1000.0)
    assert set(result["timestep"]) == {3}
    assert "" in set(result["contingency"]), "pre-contingency rows are kept unless dropped"

    without_basecase = get_branch_results_from_violations(violations, branch_limits, timestep=3, drop_basecase=True)
    assert "" not in set(without_basecase.collect()["contingency"])


def test_node_results_from_voltage_violations() -> None:
    """Voltage rows are keyed by voltage level, signed by direction and compared to the worst base case bus."""
    violations = _violations(
        [
            {"contingency": "c1", "element": "VL1", "limit_type": "LOW_VOLTAGE", "limit": 380.0, "value": 370.0},
            {"contingency": "c1", "element": "VL2", "limit_type": "HIGH_VOLTAGE", "limit": 420.0, "value": 430.0},
            {"contingency": "c2", "element": "VL3", "limit_type": "HIGH_VOLTAGE", "limit": 420.0, "value": 440.0},
            {"contingency": "", "element": "VL1", "limit_type": "LOW_VOLTAGE", "limit": 380.0, "value": 379.0},
            {"contingency": "c1", "element": "L1", "limit_type": "CURRENT", "limit": 500.0, "value": 600.0, "side": "ONE"},
        ]
    )
    voltage_levels = pl.LazyFrame({"id": ["VL1", "VL2", "VL3"], "nominal_v": [400.0, 400.0, 400.0]})
    bus_map = pl.LazyFrame({"id": ["B1a", "B1b", "B2"], "voltage_level_id": ["VL1", "VL1", "VL2"]})
    basecase_nodes = pl.LazyFrame(
        {"contingency": ["", "", ""], "element": ["B1a", "B1b", "B2"], "vm": [395.0, 385.0, 410.0]}
    )

    result = get_node_results_from_violations(
        violations, voltage_levels, bus_map, basecase_nodes, timestep=0, drop_basecase=True
    ).collect()

    assert set(result["contingency"]) == {"c1", "c2"}
    rows = {(row["contingency"], row["element"]): row for row in result.to_dicts()}
    low = rows[("c1", "VL1")]
    assert low["vm"] == pytest.approx(370.0)
    assert low["vm_loading"] == pytest.approx((370.0 - 400.0) / (400.0 - 380.0)), "undervoltage loading is negative"
    assert low["vm_basecase_deviation"] == pytest.approx(abs(370.0 - 385.0) / 385.0 * 100.0), "lowest base case bus"
    high = rows[("c1", "VL2")]
    assert high["vm_loading"] == pytest.approx((430.0 - 400.0) / (420.0 - 400.0))
    assert high["vm_basecase_deviation"] == pytest.approx(abs(430.0 - 410.0) / 410.0 * 100.0), "highest base case bus"
    assert math.isnan(rows[("c2", "VL3")]["vm_basecase_deviation"]), "no base case bus known for VL3"
    assert all(math.isnan(value) for value in result["va"])


def test_va_diff_results_from_angle_violations() -> None:
    """Angle rows keep the pre-contingency rows and carry the angle limit id as element."""
    violations = _violations(
        [
            {"contingency": "", "element": "VAL1", "limit_type": "LOW_VOLTAGE_ANGLE", "limit": -1.0, "value": -3.5},
            {"contingency": "c1", "element": "VAL1", "limit_type": "HIGH_VOLTAGE_ANGLE", "limit": 1.0, "value": 7.5},
            {"contingency": "c1", "element": "L1", "limit_type": "CURRENT", "limit": 500.0, "value": 600.0, "side": "ONE"},
        ]
    )
    result = get_va_diff_results_from_violations(violations, timestep=1).collect()
    assert result.select("contingency", "element", "va_diff").rows() == [("", "VAL1", -3.5), ("c1", "VAL1", 7.5)]


def test_convergence_results_map_every_status() -> None:
    """Each requested contingency gets a row, missing ones as NO_CALCULATION, no impact as CONVERGED with a note."""
    post = [
        FakePostContingencyResult("c1", FakeStatus.CONVERGED, ["L1"]),
        FakePostContingencyResult("c2", FakeStatus.NO_IMPACT, []),
        FakePostContingencyResult("c3", FakeStatus.SOLVER_FAILED, ["L3"]),
        FakePostContingencyResult("c4", FakeStatus.MAX_ITERATION_REACHED, []),
    ]
    converged = get_convergence_results("CONVERGED", post, ["c1", "c2", "c3", "c4", "c5"], timestep=2, basecase_id="N0")
    rows = converged.collect()
    assert rows["contingency"].to_list() == ["N0", "c1", "c2", "c3", "c4", "c5"]
    assert rows["status"].to_list() == [
        "CONVERGED", "CONVERGED", "CONVERGED", "FAILED", "MAX_ITERATION_REACHED", "NO_CALCULATION"
    ]  # fmt: skip
    assert rows["warnings"].to_list()[2] == NO_IMPACT_WARNING
    assert all(math.isnan(value) for value in rows["iteration_count"])
    assert set(rows["timestep"]) == {2}

    without_basecase = get_convergence_results("CONVERGED", post, ["c1"], timestep=2).collect()
    assert without_basecase["contingency"].to_list() == ["c1"]


def test_connectivity_results_use_the_contingency_as_outage_group() -> None:
    post = [
        FakePostContingencyResult("c1", FakeStatus.CONVERGED, ["L1", "G1"]),
        FakePostContingencyResult("c2", FakeStatus.CONVERGED, []),
    ]
    rows = get_connectivity_results(post).collect()
    assert rows.rows() == [("c1", "G1", "c1"), ("c1", "L1", "c1")]


@pytest.fixture
def branch_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "timestep": [0] * 7,
            "contingency": ["N0", "N0", "c1", "c1", "c2", "c2", "c3"],
            "element": ["L1", "L2", "L1", "L2", "L1", "L2", "L3"],
            "side": [1, 1, 1, 2, 1, 1, 1],
            "loading": [0.1, 0.2, 0.9, 1.5, 1.2, 0.3, NAN],
        }
    )


def test_violation_filter_all_is_identity(branch_frame: pl.LazyFrame) -> None:
    result = apply_limit_violation_filter(branch_frame, LimitViolationFilter.ALL, pl.col("loading"), "N0")
    assert result.collect().equals(branch_frame.collect())


def test_violation_filter_worst_overload_per_element(branch_frame: pl.LazyFrame) -> None:
    result = apply_limit_violation_filter(
        branch_frame, LimitViolationFilter.WORST_OVERLOAD_PER_ELEMENT, pl.col("loading"), "N0"
    ).collect()
    basecase = result.filter(pl.col("contingency") == "N0")
    assert len(basecase) == 2, "base case rows are exempt"
    worst = {row["element"]: row for row in result.filter(pl.col("contingency") != "N0").to_dicts()}
    assert set(worst) == {"L1", "L2", "L3"}
    assert (worst["L1"]["contingency"], worst["L1"]["loading"]) == ("c2", 1.2)
    assert (worst["L2"]["contingency"], worst["L2"]["side"]) == ("c1", 2)
    assert worst["L3"]["contingency"] == "c3", "an element with only unknown severity keeps a row"


def test_violation_filter_worst_element_per_contingency(branch_frame: pl.LazyFrame) -> None:
    result = apply_limit_violation_filter(
        branch_frame, LimitViolationFilter.WORST_ELEMENT_PER_CONTINGENCY, pl.col("loading"), None
    ).collect()
    worst = {row["contingency"]: row for row in result.to_dicts()}
    assert set(worst) == {"N0", "c1", "c2", "c3"}, "without a base case id every contingency is reduced"
    assert worst["N0"]["element"] == "L2"
    assert worst["c1"]["element"] == "L2"
    assert worst["c2"]["element"] == "L1"
    assert worst["c3"]["element"] == "L3"
