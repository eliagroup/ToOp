"""Defines constants for the folder structure.

File: folder_structure.py
Author:  Benjamin Petrick
Created: 2024-09-12
"""

from beartype.typing import Final

PREPROCESSING_PATHS: Final[dict[str, str]] = {
    "grid_file_path_powsybl": "grid.xiidm",
    "grid_file_path_pandapower": "grid.json",
    "network_data_file_path": "network_data.pkl",
    "masks_path": "masks",
    "static_information_file_path": "static_information.hdf5",
    "importer_auxiliary_file_path": "importer_auxiliary_data.json",
    "initial_topology_path": "initial_topology",
    "LF_CA_path": "initial_topology/LF_CA",
    "single_line_diagram_path": "initial_topology/single_line_diagram",
    "asset_topology_file_path": "initial_topology/asset_topology.json",
    "original_gridfile_path": "initial_topology/original_gridfile",
    "logs_path": "logs",
    "start_datetime_info_file_path": "logs/start_datetime.info",
    "chronics_path": "chronics",
    "action_set_file_path": "action_set.json",
    "nminus1_definition_file_path": "nminus1_definition.json",
    "ignore_file_path": "ignore_elements.csv",
    "contingency_list_file_path": "contingency_list.csv",
    "static_information_stats_file_path": "static_information_stats.json",
}

# Postprocessing paths that are relative to a snapshot directory. There can be multiple of these
# postprocessing paths, where one corresponds to one snapshot of the optimizer.
POSTPROCESSING_PATHS: Final[dict[str, str]] = {
    "optimizer_snapshots_path": "optimizer_snapshots",
    "dc_optimizer_snapshots_path": "optimizer_snapshots/dc",
    "ac_optimizer_snapshots_path": "optimizer_snapshots/ac",
    "dc_plus_optimizer_snapshots_path": "optimizer_snapshots/dc_plus",
    "LF_CA_ac_path": "optimizer_snapshots/ac/LF_CA",
    "single_line_diagram_ac_path": "optimizer_snapshots/ac/single_line_diagram",
    "logs_path": "logs",
}


NETWORK_MASK_NAMES: Final[dict[str, str]] = {
    "relevant_subs": "relevant_subs.npy",
    "line_for_nminus1": "line_for_nminus1.npy",
    "line_for_reward": "line_for_reward.npy",
    "line_overload_weight": "line_overload_weight.npy",
    "line_disconnectable": "line_disconnectable.npy",
    "line_tso_border": "line_tso_border.npy",
    "line_blacklisted": "line_blacklisted.npy",
    "trafo_for_nminus1": "trafo_for_nminus1.npy",
    "trafo_for_reward": "trafo_for_reward.npy",
    "trafo_overload_weight": "trafo_overload_weight.npy",
    "trafo_disconnectable": "trafo_disconnectable.npy",
    "trafo_dso_border": "trafo_dso_border.npy",
    "trafo_n0_n1_max_diff_factor": "trafo_n0_n1_max_diff_factor.npy",
    "trafo_blacklisted": "trafo_blacklisted.npy",
    "trafo_pst_controllable": "trafo_pst_controllable.npy",
    "trafo3w_for_nminus1": "trafo3w_for_nminus1.npy",
    "trafo3w_for_reward": "trafo3w_for_reward.npy",
    "trafo3w_overload_weight": "trafo3w_overload_weight.npy",
    "trafo3w_disconnectable": "trafo3w_disconnectable.npy",
    "trafo3w_n0_n1_max_diff_factor": "trafo3w_n0_n1_max_diff_factor.npy",
    "tie_line_for_reward": "tie_line_for_reward.npy",
    "tie_line_for_nminus1": "tie_line_for_nminus1.npy",
    "tie_line_overload_weight": "tie_line_overload_weight.npy",
    "tie_line_disconnectable": "tie_line_disconnectable.npy",
    "tie_line_tso_border": "tie_line_tso_border.npy",
    "dangling_line_for_nminus1": "dangling_line_for_nminus1.npy",
    "generator_for_nminus1": "generator_for_nminus1.npy",
    "load_for_nminus1": "load_for_nminus1.npy",
    "switch_for_nminus1": "switch_for_nminus1.npy",
    "switch_for_reward": "switch_for_reward.npy",
    "cross_coupler_limits": "cross_coupler_limits.npy",
    "sgen_for_nminus1": "sgen_for_nminus1.npy",
    "busbar_for_nminus1": "busbar_for_nminus1.npy",
}

OUTPUT_FILE_NAMES: Final[dict[str, str]] = {
    "multiple_topologies": "repertoire.json",
    "realized_asset_topology": "asset_topology.json",
    "postprocessed_topology": "topology.json",
    "loadflows_ac": "loadflows_ac.hdf5",
    "loadflows_dc": "loadflows_dc.hdf5",
    "loadflows_ac_cross_coupler": "loadflows_ac_cross_coupler.hdf5",
    "loadflows_dc_cross_coupler": "loadflows_dc_cross_coupler.hdf5",
}

CHRONICS_FILE_NAMES: Final[dict[str, str]] = {
    "load_p": "load_p.npy",
    "gen_p": "gen_p.npy",
    "sgen_p": "sgen_p.npy",
    "dcline_p": "dcline_p.npy",
}
