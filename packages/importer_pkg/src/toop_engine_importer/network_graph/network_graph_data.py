# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to initialize and modify NetworkGraphData."""

import numpy as np
from toop_engine_importer.network_graph.data_classes import (
    BranchSchema,
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SwitchSchema,
)
from toop_engine_importer.network_graph.filter_weights import set_all_weights
from toop_engine_importer.network_graph.network_graph_helper_functions import add_suffix_to_duplicated_grid_model_id


def add_graph_specific_data(network_graph_data: NetworkGraphData) -> None:
    """Add graph specific data to the NetworkGraphData model.

    This functions prepares the NetworkGraphData model for the network graph by setting general data.
    The network graph uses the node_tuple to identify nodes for switches and branches.
    The weight strategy is set for the branches and switches.
    The network graph data is validated.

    Parameters
    ----------
    network_graph_data : NetworkGraphData
        The NetworkGraphData model to prepare for the network graph
        Note: The NetworkGraphData is modified in place.

    """
    add_node_tuple_column(network_graph_data)
    set_all_weights(network_graph_data.branches, network_graph_data.switches, network_graph_data.helper_branches)
    add_suffix_to_duplicated_grid_model_id(df=network_graph_data.node_assets)
    network_graph_data.validate_network_graph_data()


def add_node_tuple_column(network_graph_data: NetworkGraphData) -> None:
    """Add node tuple to the edges of the NetworkGraphData model.

    This function adds the node tuple to the NetworkGraphData model.
    The node tuple is used to identify the nodes for the switches and branches.

    Parameters
    ----------
    network_graph_data : NetworkGraphData
        The NetworkGraphData model to add the node tuple.
        Note: The NetworkGraphData is modified in place.
    """
    network_graph_data.switches["node_tuple"] = list(
        map(
            lambda x: tuple(map(int, sorted(set(x)))),
            np.column_stack((network_graph_data.switches["from_node"], network_graph_data.switches["to_node"])),
        )
    )
    if not network_graph_data.branches.empty:
        network_graph_data.branches["node_tuple"] = list(
            map(
                lambda x: tuple(map(int, sorted(set(x)))),
                np.column_stack((network_graph_data.branches["from_node"], network_graph_data.branches["to_node"])),
            )
        )
    if not network_graph_data.helper_branches.empty:
        network_graph_data.helper_branches["node_tuple"] = list(
            map(
                lambda x: tuple(map(int, sorted(set(x)))),
                np.column_stack(
                    (network_graph_data.helper_branches["from_node"], network_graph_data.helper_branches["to_node"])
                ),
            )
        )


def remove_helper_branches(
    nodes_df: NodeSchema,
    helper_branches_df: HelperBranchSchema,
    node_assets_df: NodeAssetSchema,
    switches_df: SwitchSchema,
    branches_df: BranchSchema,
) -> None:
    """Remove helper branches from the network.

    Helper branches are used to connect nodes in the network, but are not part of the grid model.
    They are an artifact of the network model creation and it is optional to remove them.
    This function removes the helper branches and the helper nodes from the network.
    It also removes some nodes that are not specified as helper nodes, if they hold not additional information.

    This function is mainly used to test the functionality with the helper branches vs without helper branches.

    Parameters
    ----------
    nodes_df : NodeSchema
        The DataFrame containing the nodes. See the NodeSchema for more information.
        Note: The nodes_df is modified in place.
    helper_branches_df : HelperBranchSchema
        The DataFrame containing the helper branches. See the HelperBranchSchema for more information.
        Note: this DataFrame is obsolete after the function is called.
    node_assets_df : NodeAssetSchema
        The DataFrame containing the node assets. See the NodeAssetsSchema for more information.
    switches_df : SwitchSchema
        The DataFrame containing the switches. See the SwitchSchema for more information.
        Note: The switches_df is modified in place.
    branches_df : BranchSchema
        The DataFrame containing the branches. See the BranchSchema for more information.
        Note: The branches_df is modified in place.
    """
    helper_bus = nodes_df[nodes_df["helper_node"]]
    delete_node = []
    for helper_node in helper_bus.index.to_list():
        nodes_list = helper_branches_df[helper_branches_df["to_node"] == helper_node]["from_node"].to_list()
        nodes_list.extend(helper_branches_df[helper_branches_df["from_node"] == helper_node]["to_node"].to_list())
        grid_model_id = nodes_df.loc[nodes_list]
        grid_model_id_entry = grid_model_id[grid_model_id["grid_model_id"] != ""]
        connected_assets = node_assets_df[node_assets_df["node"].isin(nodes_df.loc[nodes_list].index)]
        if (len(connected_assets) == 1) and (len(grid_model_id_entry) == 0):
            final_node = connected_assets["node"].values[0]
            delete_node.append(helper_node)
        elif (len(connected_assets) == 0) and (len(grid_model_id_entry) == 1):
            final_node = grid_model_id_entry.index[0]
            delete_node.append(helper_node)
        elif (len(connected_assets) == 0) and (len(grid_model_id_entry) == 0):
            final_node = grid_model_id.index[0]
            delete_node.append(helper_node)
        else:
            raise ValueError(f"The helper node {helper_node} is not connected to only one node or asset.")
        switches_df.loc[switches_df["from_node"].isin(nodes_list), "from_node"] = final_node
        switches_df.loc[switches_df["to_node"].isin(nodes_list), "to_node"] = final_node
        branches_df.loc[branches_df["from_node"].isin(nodes_list), "from_node"] = final_node
        branches_df.loc[branches_df["to_node"].isin(nodes_list), "to_node"] = final_node
    nodes_df.drop(delete_node, inplace=True)
    assert len(nodes_df[nodes_df["helper_node"]]) == 0, "There are still helper nodes in the network"
