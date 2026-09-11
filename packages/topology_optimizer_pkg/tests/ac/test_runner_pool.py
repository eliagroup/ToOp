# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for process-local AC loadflow runners."""

from pathlib import Path

from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import load_action_set_fs
from toop_engine_topology_optimizer.ac.runner_pool import (
    RunnerSpec,
    create_runner_process_pool,
    get_worker_runner,
    store_worker_loadflow_results,
    warm_runner_process_pool,
)
from toop_engine_topology_optimizer.ac.scoring_functions import compute_loadflow
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile


def _compute_and_store_worker_loadflow_results() -> StoredLoadflowReference:
    """Run a loadflow and store it entirely in the initialized process worker."""
    loadflow_results, _ = compute_loadflow(actions=[], disconnections=[], pst_setpoints=[], runner=get_worker_runner())
    return store_worker_loadflow_results(loadflow_results)


def test_runner_process_pool_warmup_and_loadflow_round_trip(grid_folder: Path, loadflow_result_folder: Path) -> None:
    """Spawned workers initialize a runner and round-trip stored loadflow results."""
    grid_file = GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    action_set = load_action_set_fs(
        filesystem=processed_gridfile_fs,
        json_file_path=grid_file.action_set_file,
        diff_file_path=grid_file.action_set_diff_file,
    )
    nminus1_definition = load_pydantic_model_fs(
        filesystem=processed_gridfile_fs,
        file_path=grid_file.nminus1_definition_file,
        model_class=Nminus1Definition,
    )
    lf_params = load_lf_params_from_fs(filesystem=processed_gridfile_fs, file_path=grid_file.loadflow_parameters_file)
    runner_spec = RunnerSpec(
        action_set=action_set,
        nminus1_definition=nminus1_definition,
        grid_file=grid_file,
        contingency_processes=1,
        processed_gridfile_fs_json=processed_gridfile_fs.to_json(),
        loadflow_result_fs_json=loadflow_result_fs.to_json(),
        loadflow_result_prefix="runner-pool-test",
        lf_params=lf_params,
        result_filter=LoadflowResultFilter(),
    )

    with create_runner_process_pool(runner_spec, runner_processes=1) as process_pool:
        worker_pids = warm_runner_process_pool(process_pool, runner_processes=1)
        reference = process_pool.submit(_compute_and_store_worker_loadflow_results).result()
    loaded_results = load_loadflow_results_polars(loadflow_result_fs, reference)

    assert len(worker_pids) == 1
    assert reference.relative_path.startswith("runner-pool-test-")
    assert loaded_results.branch_results.collect().height > 0
