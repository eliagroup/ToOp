# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Loading of the preprocessing inputs and construction of the AC loadflow runners."""

from pathlib import Path

import pypowsybl
import structlog
from beartype.typing import Optional
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set_fs
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile

logger = structlog.get_logger(__name__)


def load_preprocessing_inputs(
    grid_file: GridFile, processed_gridfile_fs: AbstractFileSystem
) -> tuple[ActionSet, Nminus1Definition, pypowsybl.loadflow.Parameters | dict | None]:
    """Load the action set, the N-1 definition and the loadflow parameters of a processed grid.

    Parameters
    ----------
    grid_file : GridFile
        The grid file to load the inputs for
    processed_gridfile_fs : AbstractFileSystem
        The filesystem containing the processed grid files

    Returns
    -------
    ActionSet
        The action set
    Nminus1Definition
        The N-1 definition
    pypowsybl.loadflow.Parameters | dict | None
        The loadflow parameters of the preprocessing
    """
    action_set = load_action_set_fs(
        filesystem=processed_gridfile_fs,
        json_file_path=grid_file.action_set_file,
        diff_file_path=grid_file.action_set_diff_file,
    )
    nminus1_definition = load_pydantic_model_fs(
        filesystem=processed_gridfile_fs, file_path=grid_file.nminus1_definition_file, model_class=Nminus1Definition
    )
    lf_params = load_lf_params_from_fs(filesystem=processed_gridfile_fs, file_path=grid_file.loadflow_parameters_file)
    return action_set, nminus1_definition, lf_params


def make_runner(
    action_set: ActionSet,
    nminus1_definition: Nminus1Definition,
    grid_file: GridFile,
    n_processes: int,
    batch_size: Optional[int],
    processed_gridfile_fs: AbstractFileSystem,
    lf_params: pypowsybl.loadflow.Parameters | dict | None = None,
    result_filter: Optional[LoadflowResultFilter] = None,
) -> AbstractLoadflowRunner:
    """Initialize a runner for a gridfile, action set and n-1 def

    Parameters
    ----------
    action_set : ActionSet
        The action set to use
    nminus1_definition : Nminus1Definition
        The N-1 definition to use
    grid_file : GridFile
        The grid file to use
    n_processes : int
        The number of processes to use, from the ACGAParameters
    batch_size : Optional[int]
        The batch size to use, if any, from the ACGAParameters
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.
    lf_params: Optional[pypowsybl.loadflow.Parameters | dict]
        The loadflow parameters to use for the runner, if any. This is passed in the preprocessing results
        and can be used to run the loadflows with the same parameters
        as the initial loadflow in the preprocessing step.
        If None, the runner will use default parameters.
    result_filter: Optional[LoadflowResultFilter]
        Policy for dropping result rows that carry no decision value, applied to every full N-1 loadflow the runner runs.
        If None, every row is kept.

    Returns
    -------
    AbstractLoadflowRunner
        The initialized loadflow runner, either Pandapower or Powsybl
    """
    logger.debug(
        "Initializing loadflow runner "
        f"framework={grid_file.framework}, grid_folder={grid_file.grid_folder}, "
        f"n_processes={n_processes}, batch_size={batch_size}"
    )

    if grid_file.framework == Framework.PANDAPOWER:
        runner = PandapowerRunner(
            n_processes=n_processes,
            batch_size=batch_size,
            lf_params=lf_params if isinstance(lf_params, dict) else None,
            result_filter=result_filter,
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    elif grid_file.framework == Framework.PYPOWSYBL:
        runner = PowsyblRunner(
            n_processes=n_processes,
            batch_size=batch_size,
            lf_params=lf_params if isinstance(lf_params, pypowsybl.loadflow.Parameters) else None,
            result_filter=result_filter,
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    else:
        raise ValueError(f"Unknown framework {grid_file.framework}")
    runner.load_base_grid_fs(filesystem=processed_gridfile_fs, grid_path=grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    logger.debug(
        f"Runner prepared for {grid_file_path}: n_actions={len(action_set.local_actions)}, "
        f"n_contingencies={len(nminus1_definition.contingencies)}"
    )
    return runner
