# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for the initial-overload eligibility check."""

from fsspec.implementations.memory import MemoryFileSystem
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_results import DynamicInformationStats
from toop_engine_topology_optimizer.initial_overload import initial_overload_is_at_or_below_threshold
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile


def test_zero_initial_overload_skips_at_zero_threshold() -> None:
    """Zero overload is at a zero threshold."""
    filesystem = MemoryFileSystem()
    grid_file = GridFile(framework=Framework.PYPOWSYBL, grid_folder="grid")
    save_pydantic_model_fs(
        filesystem=filesystem,
        file_path=f"{grid_file.grid_folder}/{PREPROCESSING_PATHS['static_information_stats_file_path']}",
        pydantic_model=DynamicInformationStats(),
    )

    assert initial_overload_is_at_or_below_threshold(
        grid_files=[grid_file],
        threshold=0.0,
        processed_gridfile_fs=filesystem,
    )
