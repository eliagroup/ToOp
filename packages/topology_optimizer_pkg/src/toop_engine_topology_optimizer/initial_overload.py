# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Determine whether an optimization should start from preprocessed DC baseline metrics."""

from collections.abc import Sequence
from pathlib import Path

from fsspec import AbstractFileSystem
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_results import DynamicInformationStats
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile


def initial_overload_is_at_or_below_threshold(
    grid_files: Sequence[GridFile],
    threshold: float,
    processed_gridfile_fs: AbstractFileSystem,
) -> bool:
    """Return whether all N-0 and N-1 baseline overload energies are below the threshold."""
    return all(
        (stats.overload_energy_artificial_limit_n_0 <= threshold and stats.overload_energy_artificial_limit_n_1 <= threshold)
        for grid_file in grid_files
        for stats in (
            load_pydantic_model_fs(
                filesystem=processed_gridfile_fs,
                file_path=Path(grid_file.grid_folder) / PREPROCESSING_PATHS["static_information_stats_file_path"],
                model_class=DynamicInformationStats,
            ),
        )
    )
