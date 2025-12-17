# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Import data from PyPowSyBl networks to the Topology Optimizer."""

from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    get_list_of_stations,
    get_topology,
)

from .dacf_whitelists import (
    apply_white_list_to_operational_limits,
    assign_element_id_to_cb_df,
)
from .data_classes import (
    PowsyblSecurityAnalysisParam,
    PreProcessingStatistics,
)
from .network_analysis import (
    apply_cb_lists,
    convert_low_impedance_lines,
    get_branches_df_with_element_name,
    remove_branches_across_switch,
)
from .powsybl_masks import (
    NetworkMasks,
    create_default_network_masks,
    make_masks,
    save_masks_to_files,
    validate_network_masks,
)
from .preprocessing import (
    apply_preprocessing_changes_to_network,
    convert_file,
    load_preprocessing_statistics_filesystem,
    save_preprocessing_statistics_filesystem,
)

__all__ = [
    "NetworkMasks",
    "PowsyblSecurityAnalysisParam",
    "PreProcessingStatistics",
    "apply_cb_lists",
    "apply_preprocessing_changes_to_network",
    "apply_white_list_to_operational_limits",
    "assign_element_id_to_cb_df",
    "convert_file",
    "convert_low_impedance_lines",
    "create_default_network_masks",
    "get_branches_df_with_element_name",
    "get_list_of_stations",
    "get_topology",
    "load_preprocessing_statistics_filesystem",
    "make_masks",
    "remove_branches_across_switch",
    "save_masks_to_files",
    "save_preprocessing_statistics_filesystem",
    "validate_network_masks",
]
