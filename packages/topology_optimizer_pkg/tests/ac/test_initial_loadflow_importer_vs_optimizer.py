# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The initial loadflow of the importer and the one the AC optimizer computes for the unsplit grid are the same.

The optimizer can take the initial loadflow from the importer (ACOptimizerParameters.initial_loadflow) or compute it
with its own runner. Both have to describe the same N-0 and N-1 state, otherwise the unsplit reference of the
acceptance depends on where the loadflow came from.
"""

from pathlib import Path

import polars as pl
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess import load_grid
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.loadflow_parameters import get_cgmes_distributed_slack_parameters
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_topology_optimizer.ac.optimizer import initialize_optimization
from toop_engine_topology_optimizer.ac.storage import create_session
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile

GRID_FOLDER = "complex_node_breaker"
SOLVER_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def imported_complex_node_breaker_grid(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LoadflowResultsPolars]:
    """Import the small complex node breaker grid and keep the initial loadflow of the importer.

    Returns the folder holding the processed grid folder GRID_FOLDER and the initial loadflow of the importer.
    """
    root = tmp_path_factory.mktemp("initial_loadflow")
    folder = root / GRID_FOLDER
    net = create_complex_grid_battery_hvdc_svc_3w_trafo(connect_line_out_of_service=True)
    assert set(net.get_voltage_levels(attributes=["topology_kind"])["topology_kind"]) == {"NODE_BREAKER"}
    pypowsybl.loadflow.run_dc(net, get_cgmes_distributed_slack_parameters())
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    imported = preprocessing.import_grid_with_security_analysis(
        importer_parameters=CgmesImporterParameters(
            grid_model_file=grid_file_path,
            data_folder=folder,
            area_settings=AreaSettings(
                cutoff_voltage=1.0,
                control_area=["BE", "NL"],
                view_area=["BE", "NL"],
                nminus1_area=["BE", "NL"],
                dso_trafo_factors=None,
                dso_trafo_weight=1.0,
                border_line_factors=None,
                border_line_weight=1.0,
            ),
        )
    )
    # The DC preprocessing writes the action set the optimizer loads. The loadflow parameters the importer found and
    # saved stay as they are: the optimizer has to run with the same ones to reproduce the initial loadflow.
    load_grid(data_folder_dirfs=DirFileSystem(str(folder)), pandapower=False)
    return root, imported.initial_loadflow


def _assert_tables_equal(importer: pl.LazyFrame, optimizer: pl.LazyFrame, keys: list[str], name: str) -> None:
    """Assert that a result table of the importer and the optimizer hold the same rows, in any order."""
    importer_frame = importer.collect().sort(keys)
    optimizer_frame = optimizer.select(importer_frame.columns).collect().sort(keys)
    joined = importer_frame.join(optimizer_frame, on=keys, how="full", suffix="_optimizer", nulls_equal=True)
    differs = pl.lit(False)
    for column in (column for column in importer_frame.columns if column not in keys):
        importer_value, optimizer_value = pl.col(column), pl.col(f"{column}_optimizer")
        if importer_frame.schema[column].is_numeric():
            same = (importer_value - optimizer_value).abs() <= SOLVER_TOLERANCE * (1 + importer_value.abs())
            same = same | (importer_value.is_nan() & optimizer_value.is_nan())
        else:
            same = importer_value == optimizer_value
        differs = differs | ~same.fill_null(False)
    mismatches = joined.filter(differs)
    with pl.Config(tbl_rows=20, tbl_cols=-1, fmt_str_lengths=60):
        assert mismatches.is_empty(), f"{name}: {mismatches.height} of {joined.height} rows differ\n{mismatches}"


def _initialize(root: Path, loadflow_result_folder: Path, params: ACOptimizerParameters):  # noqa: ANN202
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_file=GridFile(framework=Framework.PYPOWSYBL, grid_folder=GRID_FOLDER),
        loadflow_result_fs=DirFileSystem(str(loadflow_result_folder)),
        processed_gridfile_fs=DirFileSystem(str(root)),
    )
    return optimizer_data, strategy.timesteps[0]


def test_initial_loadflow_of_importer_and_optimizer_match(
    imported_complex_node_breaker_grid: tuple[Path, LoadflowResultsPolars], tmp_path: Path
) -> None:
    root, importer_loadflow = imported_complex_node_breaker_grid
    ga_config = ACGAParameters(runtime_seconds=10, seed=42)

    # The optimizer computes the initial loadflow itself
    optimizer_data, computed = _initialize(root, tmp_path, ACOptimizerParameters(ga_config=ga_config))
    optimizer_loadflow = optimizer_data.load_loadflow_fn(computed.loadflow_results)

    assert not importer_loadflow.uses_limit_violations
    assert not optimizer_loadflow.uses_limit_violations
    keys = ["timestep", "contingency", "element"]
    _assert_tables_equal(
        importer_loadflow.converged.select("timestep", "contingency", "status"),
        optimizer_loadflow.converged,
        ["timestep", "contingency"],
        "converged",
    )
    _assert_tables_equal(
        importer_loadflow.branch_results.select(*keys, "side", "p", "q", "i", "loading"),
        optimizer_loadflow.branch_results,
        [*keys, "side"],
        "branch_results",
    )
    _assert_tables_equal(
        importer_loadflow.node_results.select(*keys, "vm", "va", "vm_loading"),
        optimizer_loadflow.node_results,
        keys,
        "node_results",
    )
    assert importer_loadflow.branch_limits is not None
    assert optimizer_loadflow.branch_limits is not None
    # The importer writes the artificial DC limits into the grid after the initial loadflow, so only the optimizer
    # reads them from the saved grid. They are the limits of the DC solver, no AC metric uses them.
    is_dc_limit = pl.col("limit_name").str.starts_with("artificial_dc_limit")
    assert importer_loadflow.branch_limits.filter(is_dc_limit).collect().is_empty()
    assert not optimizer_loadflow.branch_limits.filter(is_dc_limit).collect().is_empty()
    _assert_tables_equal(
        importer_loadflow.branch_limits,
        optimizer_loadflow.branch_limits.filter(~is_dc_limit),
        ["timestep", "element", "side", "limit_name"],
        "branch_limits",
    )
    assert computed.metrics.extra_scores[f"overload_current_{ga_config.limit_scope}_n_1"] > 0, (
        "the grid is overloaded, so the comparison of the overloads means something"
    )

    # The optimizer takes the initial loadflow of the importer, the unsplit metrics are the same
    reference = save_loadflow_results_polars(DirFileSystem(str(tmp_path)), "importer_initial_loadflow", importer_loadflow)
    _, from_importer = _initialize(root, tmp_path, ACOptimizerParameters(ga_config=ga_config, initial_loadflow=reference))
    assert from_importer.metrics.fitness == pytest.approx(computed.metrics.fitness, rel=SOLVER_TOLERANCE)
    assert from_importer.metrics.extra_scores == pytest.approx(computed.metrics.extra_scores, rel=SOLVER_TOLERANCE)
    assert sorted(from_importer.metrics.worst_k_contingency_cases) == sorted(computed.metrics.worst_k_contingency_cases)
