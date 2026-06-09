# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Backend/preprocessing/postprocessing functionalities for the DC solver, refactored version."""

from importlib import import_module
from importlib.util import find_spec

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
)
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


def _is_missing_pandapower_dependency(exc: ModuleNotFoundError) -> bool:
    """Return whether the import failed because pandapower is not installed."""
    return (exc.name == "pandapower" or (exc.name is not None and exc.name.startswith("pandapower."))) or (
        "pandapower" in str(exc)
    )


def _has_pandapower_dependency() -> bool:
    """Return whether pandapower can be resolved without importing optional exports."""
    try:
        return find_spec("pandapower") is not None
    except ModuleNotFoundError as exc:
        if not _is_missing_pandapower_dependency(exc):
            raise
        return False


__all__ = [
    "BackendInterface",
    "NetworkData",
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
    "pad_out_action_set",
    "preprocess",
    "reduce_branch_dimension",
]

if _has_pandapower_dependency():
    PandaPowerBackend = import_module(".pandapower.pandapower_backend", package=__name__).PandaPowerBackend
    __all__.append("PandaPowerBackend")
