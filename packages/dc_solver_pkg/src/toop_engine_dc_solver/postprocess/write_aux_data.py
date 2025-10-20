"""Functions to extract and write the N-1 definition and the action set from a filled network_data"""

from pathlib import Path

from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, extract_nminus1_definition
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import (
    save_nminus1_definition,
)
from toop_engine_interfaces.stored_action_set import save_action_set


def write_aux_data(
    data_folder: Path,
    network_data: NetworkData,
) -> None:
    """Write the N-1 definition and the action set to disk

    Parameters
    ----------
    data_folder : Path
        The root folder of the processed timestep, the N-1 definition will be stored in
        data_folder/PREPROCESSING_PATHS["nminus1_definition_file_path"] and the action set in
        data_folder/PREPROCESSING_PATHS["action_set_file_path"]
    network_data : NetworkData
        The filled network data from where to extract the N-1 definition and action set
    """
    action_set = extract_action_set(network_data)
    save_action_set(
        data_folder / PREPROCESSING_PATHS["action_set_file_path"],
        action_set,
    )

    nminus1_definition = extract_nminus1_definition(network_data)
    save_nminus1_definition(
        data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"],
        nminus1_definition,
    )
