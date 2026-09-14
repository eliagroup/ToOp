# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to extract and write the N-1 definition and the action set from a filled network_data"""

from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, extract_nminus1_definition
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import save_action_set_fs


def write_aux_data(
    data_folder: Path,
    network_data: NetworkData,
) -> None:
    """Write the N-1 definition and the action set to disk

    Parameters
    ----------
    data_folder : Path
        The root folder of the processed timestep, the N-1 definition will be stored in
        data_folder/PREPROCESSING_PATHS["dc_nminus1_definition_file_path"] and the action set in
        data_folder/PREPROCESSING_PATHS["action_set_file_path"]
    network_data : NetworkData
        The filled network data from where to extract the N-1 definition and action set
    """
    filesystem_dir = DirFileSystem(str(data_folder))
    write_aux_data_fs(network_data=network_data, filesystem=filesystem_dir)


def write_aux_data_fs(
    network_data: NetworkData,
    filesystem: AbstractFileSystem,
) -> None:
    """Write the N-1 definition and the action set to disk

    Parameters
    ----------
    network_data : NetworkData
        The filled network data from where to extract the N-1 definition and action set
    filesystem : AbstractFileSystem
        Filesystem where the auxiliary data is persisted using PREPROCESSING_PATHS
    """
    action_set = extract_action_set(network_data)
    save_action_set_fs(
        filesystem=filesystem,
        json_file_path=PREPROCESSING_PATHS["action_set_file_path"],
        diff_file_path=PREPROCESSING_PATHS["action_set_diff_path"],
        action_set=action_set,
        revalidate_action_set=False,
    )

    dc_definition_path = PREPROCESSING_PATHS["dc_nminus1_definition_file_path"]
    canonical_definition_path = PREPROCESSING_PATHS["nminus1_definition_file_path"]
    missing_paths = [path for path in (dc_definition_path, canonical_definition_path) if not filesystem.exists(path)]
    if missing_paths:
        # A folder built without the importer has no canonical definition, yet AC contingency
        # analysis reads it as its contract. Write whichever artifact is missing, and never
        # overwrite an existing one: the canonical definition belongs to the importer.
        nminus1_definition = extract_nminus1_definition(network_data)
        for definition_path in missing_paths:
            save_pydantic_model_fs(
                filesystem=filesystem,
                file_path=definition_path,
                pydantic_model=nminus1_definition,
            )
