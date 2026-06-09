# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to get an AssetTopology from a Node-Breaker grid model."""

from importlib import import_module
from importlib.util import find_spec

from .data_classes import (
    BRANCH_TYPES,
    BRANCH_TYPES_PANDAPOWER,
    BRANCH_TYPES_POWSYBL,
    NODE_TYPES,
    SWITCH_TYPES,
    AssetSchema,
    BranchSchema,
    BusbarConnectionInfo,
    EdgeConnectionInfo,
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SubstationInformation,
    SwitchSchema,
    WeightValues,
    get_empty_dataframe_from_df_model,
)
from .default_filter_strategy import (
    run_default_filter_strategy,
    set_all_busbar_coupling_switches,
    set_asset_bay_edge_attr,
    set_bay_weights,
    set_connectable_busbars,
    set_empty_bay_weights,
    set_switch_busbar_connection_info,
    set_zero_impedance_connected,
    shortest_paths_to_target_ids,
)
from .filter_weights import (
    set_all_weights,
)
from .graph_to_asset_topo import (
    get_asset_bay,
    get_busbar_df,
    get_coupler_df,
    get_station_connection_tables,
    get_switchable_asset,
)
from .network_graph import (
    generate_graph,
)
from .network_graph_data import (
    add_graph_specific_data,
    add_node_tuple_column,
    remove_helper_branches,
)
from .powsybl_station_to_graph import (
    get_node_breaker_topology_graph,
    node_breaker_topology_to_graph_data,
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
    "BRANCH_TYPES",
    "BRANCH_TYPES_PANDAPOWER",
    "BRANCH_TYPES_POWSYBL",
    "NODE_TYPES",
    "SWITCH_TYPES",
    "AssetSchema",
    "BranchSchema",
    "BusbarConnectionInfo",
    "EdgeConnectionInfo",
    "HelperBranchSchema",
    "NetworkGraphData",
    "NodeAssetSchema",
    "NodeSchema",
    "SubstationInformation",
    "SwitchSchema",
    "WeightValues",
    "add_graph_specific_data",
    "add_node_tuple_column",
    "generate_graph",
    "get_asset_bay",
    "get_busbar_df",
    "get_coupler_df",
    "get_empty_dataframe_from_df_model",
    "get_node_breaker_topology_graph",
    "get_station_connection_tables",
    "get_switchable_asset",
    "node_breaker_topology_to_graph_data",
    "remove_helper_branches",
    "run_default_filter_strategy",
    "set_all_busbar_coupling_switches",
    "set_all_weights",
    "set_asset_bay_edge_attr",
    "set_bay_weights",
    "set_connectable_busbars",
    "set_empty_bay_weights",
    "set_switch_busbar_connection_info",
    "set_zero_impedance_connected",
    "shortest_paths_to_target_ids",
]

if _has_pandapower_dependency():
    pandapower_network_to_graph = import_module(".pandapower_network_to_graph", package=__name__)
    get_network_graph = pandapower_network_to_graph.get_network_graph
    get_network_graph_data = pandapower_network_to_graph.get_network_graph_data
    __all__.extend(["get_network_graph", "get_network_graph_data"])
