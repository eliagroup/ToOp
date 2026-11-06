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
    get_connectivity_results,
    get_convergence_results,
    normalize_limit_violations,
    split_limit_violations,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table.extract import NO_IMPACT_WARNING
from toop_engine_interfaces.loadflow_results_polars import LimitViolationSchemaPolars

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


def test_split_limit_violations_keeps_every_row() -> None:
    """Branch and angle limit types go to their own violations, everything else to the node violations, unchanged."""
    violations = _violations(
        [
            {"contingency": "c1", "element": "L1", "limit_name": "10'", "limit": 600.0, "value": 1000.0, "side": "TWO"},
            {
                "contingency": "c1",
                "element": "L1",
                "limit_type": "ACTIVE_POWER",
                "limit": 200.0,
                "value": 300.0,
                "side": "TWO",
            },
            {
                "contingency": "",
                "element": "L2",
                "limit_type": "APPARENT_POWER",
                "limit": 400.0,
                "value": 500.0,
                "side": "ONE",
            },
            {"contingency": "c1", "element": "VL1", "limit_type": "LOW_VOLTAGE", "limit": 380.0, "value": 370.0},
            {"contingency": "c1", "element": "VL1", "limit_type": "HIGH_VOLTAGE", "limit": 420.0, "value": 440.0},
            {"contingency": "c1", "element": "VL2", "limit_type": "LOW_VOLTAGE", "limit": 380.0, "value": 370.0},
            {"contingency": "c2", "element": "VAL1", "limit_type": "HIGH_VOLTAGE_ANGLE", "limit": 30.0, "value": 35.0},
            {"contingency": "c2", "element": "VAL1", "limit_type": "LOW_VOLTAGE_ANGLE", "limit": -10.0, "value": -15.0},
        ]
    )
    voltage_levels = pl.LazyFrame({"id": ["VL1"], "nominal_v": [400.0]})

    split = {name: frame.collect() for name, frame in split_limit_violations(violations, voltage_levels, timestep=3).items()}
    branch, node, va_diff = split["branch_violations"], split["node_violations"], split["va_diff_violations"]

    assert branch["limit_type"].to_list() == ["CURRENT", "ACTIVE_POWER", "APPARENT_POWER"]
    assert node["limit_type"].to_list() == ["LOW_VOLTAGE", "HIGH_VOLTAGE", "LOW_VOLTAGE"]
    assert va_diff["limit_type"].to_list() == ["HIGH_VOLTAGE_ANGLE", "LOW_VOLTAGE_ANGLE"]
    assert branch["loading"].to_list() == pytest.approx([1000.0 / 600.0, 1.5, 1.25]), "relative to the reported limit"
    assert node["loading"].to_list()[:2] == pytest.approx([-30.0 / 20.0, 40.0 / 20.0]), "like the vm_loading"
    assert node["loading"].to_list()[2] is None, "without a nominal voltage there is no loading"
    assert va_diff["loading"].to_list() == pytest.approx([35.0 / 30.0, 1.5])
    assert branch["value"].to_list() == [1000.0, 300.0, 500.0], "values are not converted"
    assert branch["limit"].to_list() == [600.0, 200.0, 400.0], "the reported limit is kept, also a temporary one"
    assert branch["side"].to_list() == ["TWO", "TWO", "ONE"]
    assert "" in set(branch["contingency"]), "pre-contingency rows are kept"
    assert set(branch["timestep"]) == set(node["timestep"]) == set(va_diff["timestep"]) == {3}
    assert all(frame.columns == list(LimitViolationSchemaPolars.to_schema().columns) for frame in split.values())


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
