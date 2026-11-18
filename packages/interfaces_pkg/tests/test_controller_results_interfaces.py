# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for ``controller_results`` on the loadflow result models and helpers."""

import pandera
import polars as pl
import pytest
from test_loadflow_results_new import get_loadflow_results_example
from toop_engine_interfaces.loadflow_result_helpers import (
    convert_pandas_loadflow_results_to_polars,
    convert_polars_loadflow_results_to_pandas,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import concatenate_loadflow_results_polars
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.loadflow_results_polars import ControllerResultSchemaPolars, LoadflowResultsPolars


def _controller_row(contingency: str, timestep: int = 0, **overrides: object) -> dict:
    """A valid step row with every ControllerResultSchema column.

    Returns
    -------
    dict
        One controller row, with ``overrides`` applied.
    """
    row = {
        "timestep": timestep,
        "contingency": contingency,
        "row_type": "step",
        "element": "trafo",
        "tap_control_method": "direct",
        "tap_step_method": "tap_step_percent",
        "tap_direction_method": "pandapower",
        "controlled_bus": 0,
        "vn_kv": 110.0,
        "vm_set_pu": 1.0,
        "vm_lower_pu": 0.99,
        "vm_upper_pu": 1.01,
        "tap_min": -9.0,
        "tap_max": 9.0,
        "control_step": 1,
        "vm_start_pu": 0.97,
        "vm_pu": 0.97,
        "in_band": False,
        "voltage_violation": 0.02,
        "voltage_region": -1.0,
        "tap_start": 0.0,
        "tap_pos_before": 0.0,
        "tap_pos_after": 1.0,
        "proposed_increment": 1.0,
        "actual_increment": 1.0,
        "controller_output": 1.0,
        "tap_limit": None,
        "tap_limit_reached": False,
        "total_tap_changes": 1.0,
        "hunting_transition_detected": False,
        "hunting_counter": 0,
        "hunting_limit_reached": False,
        "hunting_detected": False,
    }
    row.update(overrides)
    return row


def _controller_frame(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows)).with_columns(pl.col("timestep").cast(pl.Int64))


def _results_polars(
    timestep: int, contingencies: list[str], controller_results: pl.DataFrame | None
) -> LoadflowResultsPolars:
    results = convert_pandas_loadflow_results_to_polars(
        get_loadflow_results_example(job_id="job", timestep=timestep, size=3, contingencies=contingencies)
    )
    lazy = controller_results.lazy() if controller_results is not None else None
    return results.model_copy(update={"controller_results": lazy})


def test_schema_accepts_step_and_final_rows() -> None:
    frame = _controller_frame(
        _controller_row("c1"),
        _controller_row(
            "c1",
            row_type="final",
            proposed_increment=None,
            actual_increment=None,
            controller_output=None,
            tap_limit="max",
            tap_limit_reached=True,
        ),
    )

    ControllerResultSchemaPolars.validate(frame)


@pytest.mark.parametrize(
    "overrides",
    [
        {"row_type": "intermediate"},
        {"voltage_region": 0.5},
        {"voltage_violation": -0.01},
        {"control_step": -1},
        {"tap_limit": "middle"},
        {"hunting_counter": -1},
        {"total_tap_changes": -1.0},
    ],
)
def test_schema_rejects_invalid_values(overrides: dict) -> None:
    frame = _controller_frame(_controller_row("c1", **overrides))

    with pytest.raises(pandera.errors.SchemaError):
        ControllerResultSchemaPolars.validate(frame)


def test_concatenate_keeps_controller_rows_of_every_result() -> None:
    res_1 = _results_polars(0, ["BASECASE"], _controller_frame(_controller_row("BASECASE", timestep=0)))
    res_2 = _results_polars(
        1, ["c1"], _controller_frame(_controller_row("c1", timestep=1), _controller_row("c1", timestep=1))
    )

    rows = concatenate_loadflow_results_polars([res_1, res_2]).controller_results.collect()

    assert rows.height == 3
    assert rows.group_by("timestep").len().sort("timestep")["len"].to_list() == [1, 2]


def test_concatenate_relaxes_controller_column_types_between_outages() -> None:
    """Untyped controller rows can differ per outage: int vs float, or an all-null column."""
    ints = _controller_frame(_controller_row("c1", tap_min=-9, tap_max=9, controller_output=None)).with_columns(
        pl.col("tap_min").cast(pl.Int64), pl.col("tap_max").cast(pl.Int64), pl.col("controller_output").cast(pl.Null)
    )
    floats = _controller_frame(_controller_row("c2", tap_min=-9.5, controller_output=2.5))

    rows = concatenate_loadflow_results_polars(
        [_results_polars(0, ["c1"], ints), _results_polars(0, ["c2"], floats)]
    ).controller_results.collect()

    assert rows.schema["tap_min"] == pl.Float64
    assert rows.schema["controller_output"] == pl.Float64
    assert sorted(rows["tap_min"].to_list()) == [-9.5, -9.0]
    assert rows.filter(pl.col("contingency") == "c1")["controller_output"].to_list() == [None]


def test_concatenate_skips_results_without_controller_rows() -> None:
    with_rows = _results_polars(0, ["c1"], _controller_frame(_controller_row("c1")))
    without_rows = _results_polars(1, ["c2"], None)

    rows = concatenate_loadflow_results_polars([without_rows, with_rows]).controller_results.collect()

    assert rows["contingency"].to_list() == ["c1"]


def test_concatenate_without_any_controller_rows_returns_none() -> None:
    results = concatenate_loadflow_results_polars([_results_polars(0, ["c1"], None), _results_polars(1, ["c2"], None)])

    assert results.controller_results is None


def test_polars_equality_compares_controller_results() -> None:
    frame = _controller_frame(_controller_row("c1"))
    base = _results_polars(0, ["c1"], frame)

    assert base == base.model_copy(update={"controller_results": frame.lazy()})
    assert base != base.model_copy(update={"controller_results": frame.with_columns(pl.lit(1.02).alias("vm_pu")).lazy()})
    assert base != base.model_copy(update={"controller_results": None})


def test_pandas_conversion_indexes_controller_results_and_compares_them() -> None:
    frame = _controller_frame(_controller_row("c1"), _controller_row("c1", control_step=2))
    pandas_results = convert_polars_loadflow_results_to_pandas(_results_polars(0, ["c1"], frame))

    assert isinstance(pandas_results, LoadflowResults)
    assert pandas_results.controller_results.index.names == ["timestep", "contingency"]
    assert len(pandas_results.controller_results) == 2

    changed = pandas_results.controller_results.copy()
    changed["control_step"] = [5, 6]
    assert pandas_results != pandas_results.model_copy(update={"controller_results": changed})


def test_pandas_conversion_leaves_missing_controller_results_unset() -> None:
    pandas_results = convert_polars_loadflow_results_to_pandas(_results_polars(0, ["c1"], None))

    assert pandas_results.controller_results is None
