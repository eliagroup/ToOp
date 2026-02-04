# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Backend/preprocessing/postprocessing functionalities for the DC solver, refactored version."""

from beartype.claw import beartype_this_package

# Make sure beartype_this_package is the only imported module
beartype_only_non_dunder_import = all(d.startswith("_") or d == "beartype_this_package" for d in dir())
if beartype_only_non_dunder_import:
    beartype_this_package()  # Leave this at the top. Otherwise the modules imported before wont be beartyped
else:
    raise ImportError(
        "Please make sure that beartype_this_package is the only imported module before calling beartype_this_package"
        "Please check the import statements."
    )

from toop_engine_interfaces.backend import BackendInterface

from .action_set import (
    enumerate_branch_actions,
    pad_out_action_set,
)
from .convert_to_jax import convert_to_jax, load_grid
from .network_data import (
    NetworkData,
    assert_network_data,
    extract_network_data_from_interface,
    load_network_data,
    save_network_data,
)
from .pandapower.pandapower_backend import PandaPowerBackend
from .powsybl.powsybl_backend import PowsyblBackend
from .preprocess import (
    add_bus_b_columns_to_ptdf,
    add_nodal_injections_to_network_data,
    combine_phaseshift_and_injection,
    compute_branch_topology_info,
    compute_bridging_branches,
    compute_injection_topology_info,
    compute_psdf_if_not_given,
    compute_ptdf_if_not_given,
    convert_multi_outages,
    exclude_bridges_from_outage_masks,
    filter_relevant_nodes_branch_count,
    preprocess,
    reduce_branch_dimension,
)

__all__ = [
    "BackendInterface",
    "NetworkData",
    "PandaPowerBackend",
    "PowsyblBackend",
    "add_bus_b_columns_to_ptdf",
    "add_nodal_injections_to_network_data",
    "assert_network_data",
    "combine_phaseshift_and_injection",
    "compute_branch_topology_info",
    "compute_bridging_branches",
    "compute_injection_topology_info",
    "compute_psdf_if_not_given",
    "compute_ptdf_if_not_given",
    "convert_multi_outages",
    "convert_to_jax",
    "enumerate_branch_actions",
    "exclude_bridges_from_outage_masks",
    "extract_network_data_from_interface",
    "filter_relevant_nodes_branch_count",
    "load_grid",
    "load_network_data",
    "pad_out_action_set",
    "preprocess",
    "reduce_branch_dimension",
    "save_network_data",
]
