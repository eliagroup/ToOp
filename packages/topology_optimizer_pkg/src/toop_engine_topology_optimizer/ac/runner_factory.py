# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Factory for AC loadflow runners."""

from pathlib import Path

import pypowsybl
from beartype.typing import Optional
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile


def make_runner(
    action_set: ActionSet,
    nminus1_definition: Nminus1Definition,
    grid_file: GridFile,
    n_processes: int,
    batch_size: Optional[int],
    processed_gridfile_fs: AbstractFileSystem,
    lf_params: pypowsybl.loadflow.Parameters | dict | None = None,
) -> AbstractLoadflowRunner:
    """Initialize a loadflow runner from preprocessed grid inputs.

    Parameters
    ----------
    action_set : ActionSet
        The action set to apply to the grid.
    nminus1_definition : Nminus1Definition
        The N-1 definition to evaluate.
    grid_file : GridFile
        The grid metadata and backend selection.
    n_processes : int
        The backend-specific contingency parallelism.
    batch_size : Optional[int]
        The optional backend batch size.
    processed_gridfile_fs : AbstractFileSystem
        Filesystem containing the preprocessed grid artifacts.
    lf_params : pypowsybl.loadflow.Parameters | dict | None, optional
        Backend loadflow parameters.

    Returns
    -------
    AbstractLoadflowRunner
        A loaded runner configured with the action set and N-1 definition.
    """
    if grid_file.framework == Framework.PANDAPOWER:
        runner = PandapowerRunner(
            n_processes=n_processes, batch_size=batch_size, lf_params=lf_params if isinstance(lf_params, dict) else None
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    elif grid_file.framework == Framework.PYPOWSYBL:
        runner = PowsyblRunner(
            n_processes=n_processes,
            batch_size=batch_size,
            lf_params=lf_params if isinstance(lf_params, pypowsybl.loadflow.Parameters) else None,
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    else:
        raise ValueError(f"Unknown framework {grid_file.framework}")
    runner.load_base_grid_fs(filesystem=processed_gridfile_fs, grid_path=grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    return runner
