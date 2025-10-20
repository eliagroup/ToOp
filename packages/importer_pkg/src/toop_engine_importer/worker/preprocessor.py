"""Module contains functions holds preprocessor commands for kafka communication in the importer repo.

File: preprocessor.py
Author:  Nico Westerbeck
Created: 2024
"""

from pathlib import Path
from typing import Callable, Optional, Union

from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics
from toop_engine_contingency_analysis.ac_loadflow_service.lf_worker import load_base_grid
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import save_loadflow_results_polars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    StartPreprocessingCommand,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    PowerFactoryImportResult,
    PreprocessingSuccessResult,
    UcteImportResult,
)
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.types import MetricType


def import_grid_model(
    start_command: StartPreprocessingCommand,
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None],
) -> UcteImportResult:
    """Run the import procedure.

    This only performs the import until there's a grid model, the preprocessing in the loadflow
    solver is run by preprocess

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command to start the preprocessing run with
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None]
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters

    Returns
    -------
    UcteImportResult
        A result dataclass from the importer, mainly including the grid folder and some stats

    Raises
    ------
    Exception
        Any exception raised will be caught by the worker and sent back
    """
    importer_parameters = start_command.importer_parameters
    import_result = preprocessing.convert_file(
        importer_parameters=importer_parameters,
        status_update_fn=status_update_fn,
    )
    return import_result


def run_initial_loadflow(
    start_command: StartPreprocessingCommand,
    data_folder: Path,
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None],
    loadflow_result_folder: Path,
) -> tuple[StoredLoadflowReference, dict[MetricType, float]]:
    """Run the initial AC contingency analysis

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command that was sent to the worker
    data_folder: Path
        The folder containing the grid model with already preprocessed N-1 definition and grid model file.
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None]
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters
    loadflow_result_folder: Path
        A folder where the loadflow results are stored - this should be a NFS share together with the backend and
        optimizer. The importer needs this to store the initial loadflows

    Returns
    -------
    StoredLoadflowReference
        A reference to the stored loadflow results
    dict[MetricType, float]
        A dictionary containing the computed metrics
    """
    status_update_fn("prepare_contingency_analysis", "Preparing initial loadflow contingency analysis")
    n_minus_1_definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    net = load_base_grid(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"], "powsybl")
    status_update_fn("run_contingency_analysis", "Running initial loadflow contingency analysis")
    timestep_result_polars = get_ac_loadflow_results(
        net=net,
        n_minus_1_definition=n_minus_1_definition,
        timestep=0,
        job_id=start_command.preprocess_id,
        n_processes=start_command.preprocess_parameters.initial_loadflow_processes,
    )
    dirfs = DirFileSystem(str(loadflow_result_folder))
    ref_polars = save_loadflow_results_polars(
        dirfs, f"initial_loadflow_{start_command.preprocess_id}", timestep_result_polars
    )
    metrics = compute_metrics(
        timestep_result_polars,
        base_case_id=n_minus_1_definition.base_case.id if n_minus_1_definition.base_case is not None else None,
    )
    return ref_polars, metrics


def preprocess(
    start_command: StartPreprocessingCommand,
    import_results: Union[UcteImportResult, PowerFactoryImportResult],
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None],
    loadflow_result_folder: Path,
) -> PreprocessingSuccessResult:
    """Run the preprocessing pipeline that is independent of the data source.

    This only performs the preprocessing in the loadflow solver

    Parameters
    ----------
    start_command: StartPreprocessingCommand
        The command to start the preprocessing run with
    import_results: Union[UcteImportResult, PowerFactoryImportResult]
        Results from the import procedure
    status_update_fn: Callable[[PreprocessStage, Optional[str]], None]
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters
    loadflow_result_folder: Path
        A folder where the loadflow results are stored - this should be a NFS share together with the backend and
        optimizer. The importer needs this to store the initial loadflows

    Returns
    -------
    PreprocessingSuccessResult
        A result dataclass for the entire preprocessing, including paths to the ready
        static_information and network_data dataclasses.

    Raises
    ------
    Exception
        Any exception raised will be caught by the worker and sent back
    """
    preprocess_parameters = start_command.preprocess_parameters
    pandapower = False
    if isinstance(import_results, PowerFactoryImportResult):
        pandapower = True

    info, _, _ = load_grid(
        data_folder=import_results.data_folder,
        pandapower=pandapower,
        parameters=preprocess_parameters,
        status_update_fn=status_update_fn,
    )

    initial_loadflow, lf_metrics = run_initial_loadflow(
        start_command=start_command,
        data_folder=import_results.data_folder,
        status_update_fn=status_update_fn,
        loadflow_result_folder=loadflow_result_folder,
    )

    preprocessing_results = PreprocessingSuccessResult(
        data_folder=import_results.data_folder,
        static_information_stats=info,
        importer_results=import_results,
        initial_loadflow=initial_loadflow,
        initial_metrics=lf_metrics,
    )

    return preprocessing_results
