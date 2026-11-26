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
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition_fs
from toop_engine_interfaces.stored_action_set import save_action_set_fs


def write_aux_data(
    data_folder: Path,
    network_data: NetworkData,
) -> None:
    """Write the DC N-1 definition and the action set to disk

    Parameters
    ----------
    data_folder : Path
        The root folder of the processed timestep, see :func:`write_aux_data_fs` for the files written to it.
    network_data : NetworkData
        The filled network data from where to extract the N-1 definition and action set
    """
    filesystem_dir = DirFileSystem(str(data_folder))
    write_aux_data_fs(network_data=network_data, filesystem=filesystem_dir)


def write_aux_data_fs(
    network_data: NetworkData,
    filesystem: AbstractFileSystem,
) -> None:
    """Write the DC N-1 definition and the action set to disk

    The DC N-1 definition goes to PREPROCESSING_PATHS["dc_nminus1_definition_file_path"]. The importer's N-1
    definition at PREPROCESSING_PATHS["nminus1_definition_file_path"] is never overwritten, only written if missing.

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

    dc_nminus1_definition = extract_nminus1_definition(network_data)
    nminus1_definition_path = PREPROCESSING_PATHS["nminus1_definition_file_path"]
    if filesystem.exists(nminus1_definition_path):
        # The id type decides how contingency analysis resolves the element ids
        id_type = load_nminus1_definition_fs(filesystem, nminus1_definition_path).id_type
        dc_nminus1_definition = dc_nminus1_definition.model_copy(update={"id_type": id_type})
    else:
        save_pydantic_model_fs(filesystem, nminus1_definition_path, dc_nminus1_definition)
    save_pydantic_model_fs(filesystem, PREPROCESSING_PATHS["dc_nminus1_definition_file_path"], dc_nminus1_definition)
