# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests that ``controller_results`` and ``der_controller_results`` survive filtering, conversion and storage."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from polars.testing import assert_frame_equal
from test_controller_results_interfaces import _controller_frame, _controller_row
from test_loadflow_results_new import get_loadflow_results_example
from toop_engine_interfaces.loadflow_result_helpers import (
    convert_pandas_loadflow_results_to_polars,
    convert_polars_loadflow_results_to_pandas,
    load_loadflow_results,
    save_loadflow_results,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    concatenate_loadflow_results_polars,
    load_loadflow_results_polars,
    save_loadflow_results_polars,
    select_timestep_polars,
    subset_contingencies_polars,
)
from toop_engine_interfaces.loadflow_results_polars import DerControllerResultSchemaPolars, LoadflowResultsPolars

CONTINGENCIES = ["c1", "c2"]
TABLES = ("controller_results", "der_controller_results")


def _der_row(contingency: str, timestep: int = 0, **overrides: object) -> dict:
    """A valid step row with every DerControllerResultSchema column.

    Returns
    -------
    dict
        One Q(U) controller row, with ``overrides`` applied.
    """
    row = {
        "timestep": timestep,
        "contingency": contingency,
        "row_type": "step",
        "controller_index": 4,
        "control_step": 1,
        "element": "sgen",
        "element_index": 0,
        "bus": 2,
        "measurement_bus": 1,
        "vm_pu": 1.07,
        "in_service": True,
        "bus_supplied": True,
        "measurement_supplied": True,
        "active": True,
        "held_reason": "none",
        "share_factor": 1.0,
        "q_request_mvar": -50.0,
        "q_limited_mvar": -50.0,
        "capability_limited": False,
        "p_mw": 80.0,
        "q_mvar": -50.0,
    }
    row.update(overrides)
    return row


def _der_frame(*rows: dict) -> pl.DataFrame:
    frame = pl.DataFrame(list(rows)).with_columns(
        pl.col("timestep").cast(pl.Int64),
        # An active element has no held reason
        pl.lit(None, dtype=pl.String).alias("held_reason"),
    )
    return frame


def _results(timesteps: tuple[int, ...] = (0, 1)) -> LoadflowResultsPolars:
    """Results for ``timesteps`` and two contingencies, with one tap and one Q(U) row per contingency and timestep.

    Returns
    -------
    LoadflowResultsPolars
        The results with both controller tables set.
    """
    per_timestep = []
    for timestep in timesteps:
        results = convert_pandas_loadflow_results_to_polars(
            get_loadflow_results_example(job_id="job", timestep=timestep, size=3, contingencies=CONTINGENCIES)
        )
        # The engine stores an all-empty nullable string column as String, not Null
        tap_rows = _controller_frame(*(_controller_row(c, timestep=timestep) for c in CONTINGENCIES)).with_columns(
            pl.col("tap_limit").cast(pl.String)
        )
        der_rows = _der_frame(*(_der_row(c, timestep=timestep) for c in CONTINGENCIES))
        update = {"controller_results": tap_rows.lazy(), "der_controller_results": der_rows.lazy()}
        per_timestep.append(results.model_copy(update=update))
    combined = concatenate_loadflow_results_polars(per_timestep)
    return combined


def _sorted(frame: pl.LazyFrame) -> pl.DataFrame:
    collected = frame.collect().sort(["timestep", "contingency"])
    return collected


def test_der_schema_accepts_held_and_active_elements() -> None:
    frame = pl.concat(
        [
            _der_frame(_der_row("c1")),
            _der_frame(_der_row("c1", active=False, in_service=False, q_request_mvar=None)).with_columns(
                pl.lit("out_of_service").alias("held_reason")
            ),
        ]
    )

    DerControllerResultSchemaPolars.validate(frame)


@pytest.mark.parametrize("table", TABLES)
def test_select_timestep_keeps_the_controller_rows_of_that_timestep(table: str) -> None:
    results = _results()

    selected = select_timestep_polars(results, 1)

    rows = getattr(selected, table).collect()
    assert set(rows["timestep"].to_list()) == {1}
    assert set(rows["contingency"].to_list()) == set(CONTINGENCIES)


@pytest.mark.parametrize("table", TABLES)
def test_subset_contingencies_keeps_the_controller_rows_of_those_contingencies(table: str) -> None:
    results = _results()

    subset = subset_contingencies_polars(results, ["c2"])

    rows = getattr(subset, table).collect()
    assert set(rows["contingency"].to_list()) == {"c2"}
    assert set(rows["timestep"].to_list()) == {0, 1}


def test_filters_leave_missing_controller_tables_unset() -> None:
    results = _results().model_copy(update={"controller_results": None, "der_controller_results": None})

    selected = select_timestep_polars(results, 0)
    subset = subset_contingencies_polars(results, ["c1"])

    assert (selected.controller_results, selected.der_controller_results) == (None, None)
    assert (subset.controller_results, subset.der_controller_results) == (None, None)


@pytest.mark.parametrize("table", TABLES)
def test_pandas_to_polars_conversion_keeps_the_controller_rows(table: str) -> None:
    results = _results()
    pandas_results = convert_polars_loadflow_results_to_pandas(results)

    back = convert_pandas_loadflow_results_to_polars(pandas_results)

    assert_frame_equal(_sorted(getattr(back, table)), _sorted(getattr(results, table)), check_column_order=False)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("table", TABLES)
def test_polars_save_and_load_keep_the_controller_rows(tmp_path: Path, table: str, validate: bool) -> None:
    results = _results()
    fs = DirFileSystem(tmp_path)

    reference = save_loadflow_results_polars(fs, "results", results)
    loaded = load_loadflow_results_polars(fs, reference, validate=validate)

    assert_frame_equal(_sorted(getattr(loaded, table)), _sorted(getattr(results, table)))


def test_polars_save_without_controller_rows_removes_the_files_of_an_earlier_save(tmp_path: Path) -> None:
    """The loader only checks whether a file exists, so a leftover would be read as the new result's rows."""
    results = _results()
    fs = DirFileSystem(tmp_path)
    save_loadflow_results_polars(fs, "results", results)
    without = results.model_copy(update={"controller_results": None, "der_controller_results": None})

    reference = save_loadflow_results_polars(fs, "results", without)
    loaded = load_loadflow_results_polars(fs, reference)

    assert (loaded.controller_results, loaded.der_controller_results) == (None, None)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("table", TABLES)
def test_pandas_save_and_load_keep_the_controller_rows(tmp_path: Path, table: str, validate: bool) -> None:
    pandas_results = convert_polars_loadflow_results_to_pandas(_results())
    fs = DirFileSystem(tmp_path)

    reference = save_loadflow_results(fs, "results", pandas_results)
    loaded = load_loadflow_results(fs, reference, validate=validate)

    pd.testing.assert_frame_equal(getattr(loaded, table), getattr(pandas_results, table))


def test_pandas_save_without_controller_rows_loads_them_as_unset(tmp_path: Path) -> None:
    pandas_results = convert_polars_loadflow_results_to_pandas(_results())
    fs = DirFileSystem(tmp_path)
    save_loadflow_results(fs, "results", pandas_results)
    without = pandas_results.model_copy(update={"controller_results": None, "der_controller_results": None})

    reference = save_loadflow_results(fs, "results", without)
    loaded = load_loadflow_results(fs, reference)

    assert (loaded.controller_results, loaded.der_controller_results) == (None, None)
