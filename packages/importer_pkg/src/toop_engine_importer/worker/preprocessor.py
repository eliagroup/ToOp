# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions holds preprocessor commands for kafka communication in the importer repo.

File: preprocessor.py
Author:  Nico Westerbeck
Created: 2024
"""

from pathlib import Path
from typing import Optional

import structlog
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_importer.pypowsybl_import.preprocessing import ImportedGridModel
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    StartPreprocessingCommand,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    ImportResult,
    NoOptimizationRequiredResult,
    PreprocessingSuccessResult,
)
from toop_engine_interfaces.status_update import StatusUpdateFn
from toop_engine_interfaces.types import MetricType

logger = structlog.get_logger(__name__)


def run_preprocessing(
    start_command: StartPreprocessingCommand,
    unprocessed_gridfile_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
    loadflow_result_fs: AbstractFileSystem,
    status_update_fn: StatusUpdateFn,
) -> PreprocessingSuccessResult | NoOptimizationRequiredResult:
    """Run the whole preprocessing of a start command: import, initial loadflow and DC preprocessing.

    The AC N-1 security analysis runs once, during the import. It sets the artificial limits and is stored as the
    initial loadflow, before the DC preprocessing reads the grid with these limits.

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command to start the preprocessing run with
    unprocessed_gridfile_fs: AbstractFileSystem
        A filesystem where the unprocessed gridfiles are stored.
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
    loadflow_result_fs: AbstractFileSystem
        A filesystem where the loadflow results are stored - this should be a NFS share together with the backend and
        optimizer. The initial loadflow is stored here.
    status_update_fn: StatusUpdateFn
        A function to call to signal progress in the preprocessing pipeline. Takes a stage, an
        optional message as parameters and network size stats.

    Returns
    -------
    PreprocessingSuccessResult | NoOptimizationRequiredResult
        A result dataclass for the entire preprocessing, including paths to the ready
        static_information and network_data dataclasses.

    Raises
    ------
    Exception
        Any exception raised will be caught by the worker and sent back
    """
    logger.info("Starting preprocessing", preprocess_id=start_command.preprocess_id)
    imported_grid_model = import_grid_model(
        start_command=start_command,
        unprocessed_gridfile_fs=unprocessed_gridfile_fs,
        processed_gridfile_fs=processed_gridfile_fs,
        status_update_fn=status_update_fn,
    )
    import_result = imported_grid_model.import_result.model_copy(
        update={
            "optimization_required": _determine_optimization_required(
                imported_grid_model.import_result,
                start_command.preprocess_parameters.minimum_initial_ac_overload_energy,
            ),
        }
    )
    initial_metrics = imported_grid_model.initial_metrics
    initial_loadflow = store_initial_loadflow(
        start_command=start_command,
        initial_loadflow=imported_grid_model.initial_loadflow,
        loadflow_result_fs=loadflow_result_fs,
    )
    # Free the AC results before the memory peak of the DC preprocessing
    del imported_grid_model

    preprocessing_results: PreprocessingSuccessResult | NoOptimizationRequiredResult
    if import_result.optimization_required:
        preprocessing_results = preprocess(
            start_command=start_command,
            import_results=import_result,
            status_update_fn=status_update_fn,
            processed_gridfile_fs=processed_gridfile_fs,
            initial_loadflow=initial_loadflow,
            initial_metrics=initial_metrics,
        )
    else:
        preprocessing_results = NoOptimizationRequiredResult(
            data_folder=import_result.data_folder,
            importer_results=import_result,
        )
        logger.warning("No optimization required for this preprocessing run", preprocess_id=start_command.preprocess_id)

    logger.info("Finished preprocessing", preprocess_id=start_command.preprocess_id)
    return preprocessing_results


def _determine_optimization_required(
    import_result: ImportResult,
    minimum_initial_ac_overload_energy: Optional[float],
) -> bool:
    """Return whether the importer AC overload metrics require DC preprocessing.

    Missing overload metrics require preprocessing because they cannot establish that
    the grid is below the threshold.

    Parameters
    ----------
    import_result : ImportResult
        Import result containing initial AC overload energies.
    minimum_initial_ac_overload_energy : float | None
        Threshold for skipping optimization, or None to disable the check.

    Returns
    -------
    bool
        Whether DC preprocessing and optimization are required.
    """
    overload_n_0 = import_result.overload_energy_artificial_limit_n_0
    overload_n_1 = import_result.overload_energy_artificial_limit_n_1
    return (
        minimum_initial_ac_overload_energy is None
        or overload_n_0 is None
        or overload_n_1 is None
        or overload_n_0 > minimum_initial_ac_overload_energy
        or overload_n_1 > minimum_initial_ac_overload_energy
    )


def import_grid_model(
    start_command: StartPreprocessingCommand,
    unprocessed_gridfile_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
    status_update_fn: StatusUpdateFn,
) -> ImportedGridModel:
    """Run the import procedure, including the AC security analysis that sets the artificial limits.

    This only performs the import until there's a grid model, the preprocessing in the loadflow
    solver is run by preprocess

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command to start the preprocessing run with
    unprocessed_gridfile_fs: AbstractFileSystem
        A filesystem where the unprocessed gridfiles are stored. The concrete folder to use is determined by the start
        command, which contains an import location relative to the root of the unprocessed_gridfile_fs.
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder is created.
    status_update_fn: StatusUpdateFn
        A function to call to signal progress in the preprocessing pipeline. Takes a stage, an
        optional message as parameters and network size stats.

    Returns
    -------
    ImportedGridModel
        The import statistics, mainly including the grid folder, and the initial loadflow with its metrics.

    Raises
    ------
    Exception
        Any exception raised will be caught by the worker and sent back
    """
    imported_grid_model = preprocessing.import_grid_with_security_analysis(
        importer_parameters=start_command.importer_parameters,
        status_update_fn=status_update_fn,
        unprocessed_gridfile_fs=unprocessed_gridfile_fs,
        processed_gridfile_fs=processed_gridfile_fs,
        job_id=start_command.preprocess_id,
    )
    return imported_grid_model


def store_initial_loadflow(
    start_command: StartPreprocessingCommand,
    initial_loadflow: LoadflowResultsPolars,
    loadflow_result_fs: AbstractFileSystem,
) -> StoredLoadflowReference:
    """Store the AC N-1 results of the import as the initial loadflow.

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command that was sent to the worker
    initial_loadflow: LoadflowResultsPolars
        The AC N-1 results of the import, with the permanent and the artificial branch limits.
    loadflow_result_fs: AbstractFileSystem
        A filesystem where the loadflow results are stored - this should be a NFS share together with the backend and
        optimizer.

    Returns
    -------
    StoredLoadflowReference
        A reference to the stored loadflow results
    """
    reference = save_loadflow_results_polars(
        loadflow_result_fs, f"initial_loadflow_{start_command.preprocess_id}", initial_loadflow
    )
    logger.info(
        "Stored the initial loadflow",
        preprocess_id=start_command.preprocess_id,
        relative_path=reference.relative_path,
    )
    return reference


def preprocess(
    start_command: StartPreprocessingCommand,
    import_results: ImportResult,
    status_update_fn: StatusUpdateFn,
    processed_gridfile_fs: AbstractFileSystem,
    initial_loadflow: StoredLoadflowReference,
    initial_metrics: dict[MetricType, float],
) -> PreprocessingSuccessResult:
    """Run the preprocessing pipeline that is independent of the data source.

    This only performs the preprocessing in the loadflow solver

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command to start the preprocessing run with
    import_results: ImportResult
        Results from the import procedure
    status_update_fn: StatusUpdateFn
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.
    initial_loadflow: StoredLoadflowReference
        The reference to the stored initial loadflow of the import.
    initial_metrics: dict[MetricType, float]
        The metrics of the initial loadflow.

    Returns
    -------
    PreprocessingSuccessResult
        The result of completed DC preprocessing.

    Raises
    ------
    Exception
        Any exception raised will be caught by the worker and sent back
    """
    preprocess_parameters = start_command.preprocess_parameters
    pandapower = False
    if import_results.grid_type == "power_factory":
        pandapower = True

    # Create a dirfs that points to the data folder, so we can pass around the dirfs instead of the path + fs
    output_dirfs = DirFileSystem(path=str(import_results.data_folder), fs=processed_gridfile_fs)

    lf_params = load_lf_params_from_fs(output_dirfs, Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    info, _, _ = load_grid(
        data_folder_dirfs=output_dirfs,
        pandapower=pandapower,
        parameters=preprocess_parameters,
        status_update_fn=status_update_fn,
        lf_params=lf_params,
    )

    return PreprocessingSuccessResult(
        data_folder=import_results.data_folder,
        static_information_stats=info,
        importer_results=import_results,
        initial_loadflow=initial_loadflow,
        initial_metrics=initial_metrics,
    )
