# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import shutil
import time
from functools import partial
from typing import Optional, get_args

import logbook
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_importer.worker import preprocessor
from toop_engine_importer.worker.preprocessor import run_initial_loadflow
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    PreprocessParameters,
    StartPreprocessingCommand,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    PreprocessingSuccessResult,
    UcteImportResult,
)

# Set up logging for the test
logger = logbook.Logger("test_preprocessor")


def test_run_initial_loadflow(imported_ucte_file_data_folder, ucte_importer_parameters: UcteImporterParameters, tmp_path):
    temp_file = imported_ucte_file_data_folder
    ucte_importer_parameters.data_folder = temp_file
    # def parameters for function

    logged_messages = []

    def heartbeat_working(
        stage: PreprocessStage,
        message: Optional[str],
        preprocess_id: str,
        start_time: float,
    ):
        logged_messages.append(
            f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: {message}"
        )

    start_time = time.time()
    heartbeat_fn = partial(
        heartbeat_working,
        preprocess_id="test_id",
        start_time=start_time,
    )

    # Create dummy import result
    import_result = UcteImportResult(
        data_folder=ucte_importer_parameters.data_folder,
    )

    start_command = StartPreprocessingCommand(
        importer_parameters=ucte_importer_parameters,
        preprocess_parameters=PreprocessParameters(),
        preprocess_id="test_ID",
    )
    filesystem_dir = DirFileSystem(str(import_result.data_folder))
    info, _, _ = load_grid(
        data_folder_dirfs=filesystem_dir,
        pandapower=False,
        parameters=PreprocessParameters(),
        status_update_fn=heartbeat_fn,
    )

    logged_messages = []
    loadflow_result_dirfs = DirFileSystem(str(tmp_path))
    initial_loadflow, metrics = run_initial_loadflow(
        start_command=start_command,
        processed_gridfile_dirfs=filesystem_dir,
        status_update_fn=heartbeat_fn,
        loadflow_result_fs=loadflow_result_dirfs,
    )

    assert initial_loadflow is not None

    lf_res = load_loadflow_results_polars(loadflow_result_dirfs, reference=initial_loadflow)
    assert lf_res is not None
    assert len(lf_res.branch_results.collect())
    assert "max_flow_n_1" in metrics
    assert "overload_energy_n_1" in metrics


def test_import_ucte(ucte_importer_parameters: UcteImporterParameters):
    # def parameters for function
    ucte_importer_parameters.area_settings.dso_trafo_factors = None
    ucte_importer_parameters.area_settings.border_line_factors = None

    temp_dir = ucte_importer_parameters.data_folder

    logged_messages = []

    def heartbeat_working(
        stage: PreprocessStage,
        message: Optional[str],
        preprocess_id: str,
        start_time: float,
    ):
        logged_messages.append(
            f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: {message}"
        )

    start_time = time.time()
    heartbeat_fn = partial(
        heartbeat_working,
        preprocess_id="test_id",
        start_time=start_time,
    )

    start_command = StartPreprocessingCommand(
        importer_parameters=ucte_importer_parameters,
        preprocess_parameters=PreprocessParameters(),
        preprocess_id="test_ID",
    )

    with logbook.handlers.TestHandler() as caplog:
        import_result = preprocessor.import_grid_model(
            start_command=start_command,
            status_update_fn=heartbeat_fn,
            unprocessed_gridfile_fs=LocalFileSystem(),
            processed_gridfile_fs=LocalFileSystem(),
        )
        importer_auxiliary_file = temp_dir / PREPROCESSING_PATHS["importer_auxiliary_file_path"]
        grid_file_path = temp_dir / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        mask_dir = temp_dir / PREPROCESSING_PATHS["masks_path"]
        asset_topology_file = temp_dir / PREPROCESSING_PATHS["asset_topology_file_path"]
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        assert asset_topology_file.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        # Remove all files and folders in output_path
        shutil.rmtree(temp_dir)
        assert isinstance(import_result, UcteImportResult)

        # Filter and assert logs
        logs = [record for record in caplog.formatted_records]
        assert len(logs) > 0, "No logs found"
        # Check if all stages are logged
        for record in logged_messages:
            stage_substring = record.split("Preprocessing stage ")[1].split(" for job")[0]
            assert stage_substring in list(get_args(PreprocessStage)), (
                f"Log message does not contain valid stage name: {record}"
            )
            assert "test_id" in record, f"Log message does not contain correct preprocess_id: {record}"


def test_preprocess(imported_ucte_file_data_folder, ucte_importer_parameters: UcteImporterParameters, tmp_path):
    temp_file = imported_ucte_file_data_folder
    ucte_importer_parameters.data_folder = temp_file
    # def parameters for function

    logged_messages = []

    def heartbeat_working(
        stage: PreprocessStage,
        message: Optional[str],
        preprocess_id: str,
        start_time: float,
    ):
        logged_messages.append(
            f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: {message}"
        )

    start_time = time.time()
    heartbeat_fn = partial(
        heartbeat_working,
        preprocess_id="test_id",
        start_time=start_time,
    )

    # Create dummy import result
    import_result = UcteImportResult(
        data_folder=ucte_importer_parameters.data_folder,
    )

    start_command = StartPreprocessingCommand(
        importer_parameters=ucte_importer_parameters,
        preprocess_parameters=PreprocessParameters(),
        preprocess_id="test_ID",
    )

    processed_gridfile_fs = LocalFileSystem()
    loadflow_result_fs = LocalFileSystem()
    with logbook.handlers.TestHandler() as caplog:
        preprocess_result = preprocessor.preprocess(
            start_command=start_command,
            import_results=import_result,
            status_update_fn=heartbeat_fn,
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )

        static_info_path = ucte_importer_parameters.data_folder / PREPROCESSING_PATHS["static_information_file_path"]
        network_data_path = ucte_importer_parameters.data_folder / PREPROCESSING_PATHS["network_data_file_path"]

        assert static_info_path.exists()
        assert network_data_path.exists()
        assert isinstance(preprocess_result, PreprocessingSuccessResult)
        assert preprocess_result.data_folder == ucte_importer_parameters.data_folder
        assert preprocess_result.importer_results == import_result

        # Filter and assert logs
        logs = [record for record in caplog.formatted_records]
        assert len(logs) > 0, "No logs found"
        # Check if all stages are logged
        for record in logged_messages:
            stage_substring = record.split("Preprocessing stage ")[1].split(" for job")[0]
            assert stage_substring in list(get_args(PreprocessStage)), (
                f"Log message does not contain stage name {stage_substring}: {record}"
            )
            assert "test_id" in record, f"Log message does not contain correct preprocess_id: {record}"
