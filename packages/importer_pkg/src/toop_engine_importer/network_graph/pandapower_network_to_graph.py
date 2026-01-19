# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Convert a pandapower network to a network graph."""

import logbook
import networkx as nx
import pandas as pd
from pandapower.auxiliary import pandapowerNet
from toop_engine_importer.network_graph.data_classes import (
    BranchSchema,
    NetworkGraphData,
    NodeSchema,
    SwitchSchema,
    get_empty_dataframe_from_df_model,
)
from toop_engine_importer.network_graph.default_filter_strategy import run_default_filter_strategy
from toop_engine_importer.network_graph.network_graph import generate_graph, set_substation_id
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data

logger = logbook.Logger(__name__)


def get_network_graph_data(net: pandapowerNet, only_relevant_col: bool = True) -> NetworkGraphData:
    """Get the network graph from the pandapower network.

    Parameters
    ----------
    net : pandapower
        The pandapower network.
    only_relevant_col : bool, optional
        Whether to return only the relevant columns, by default True
        relevant is determined by the default BranchSchema columns

    Returns
    -------
    net_graph : NetworkGraphData
        The network graph  in the format of the NetworkGraph class.
        Contains the full network with all substations.
    """
    branches_df = get_branch_df(net, only_relevant_col=only_relevant_col)
    nodes_df = get_nodes(net, only_relevant_col=only_relevant_col)
    switches_df = get_switches_df(net, only_relevant_col=only_relevant_col)
    logger.warning("generators are missing in the network graph - they are not implemented yet")
    network_graph_data = NetworkGraphData(nodes=nodes_df, switches=switches_df, branches=branches_df)
    add_graph_specific_data(network_graph_data)
    return network_graph_data


def get_network_graph(network_graph_data: NetworkGraphData) -> nx.Graph:
    """Get the network graph from the NetworkGraphData and run default filter.

    Parameters
    ----------
    network_graph_data : NetworkGraphData
        The NetworkGraphData containing the nodes, switches, branches and node_assets.

    Returns
    -------
    nx.Graph
        The network graph.
    """
    graph = generate_graph(network_graph_data)
    set_substation_id(graph=graph, network_graph_data=network_graph_data)
    run_default_filter_strategy(graph=graph)
    return graph


# TODO: Rework this function. It does to many things at once.
# TODO: make 3wtrafo workflow more clear
def get_edges_data(dataframe: pd.DataFrame, asset_type: str, only_relevant_col: bool = True) -> pd.DataFrame:
    """Get the edges data from the dataframe and return a df ready to use for the BranchSchema.

    Can be used for switches, trafos and other branches.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the edges.
    asset_type : str
        The type of the asset.
    only_relevant_col : bool, optional
        Whether to return only the relevant columns, by default True
        relevant is determined by the default BranchSchema columns

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the edges in the format of the


    """
    needed_col = list(get_empty_dataframe_from_df_model(BranchSchema).columns)
    if asset_type == "switch":
        dataframe.rename(
            columns={
                "bus": "from_node",
                "element": "to_node",
            },
            inplace=True,
        )
        needed_col.append("open")
    elif asset_type in ["trafo", "trafo3w"]:
        dataframe["type"] = asset_type
        dataframe.rename(
            columns={
                "hv_bus": "from_node",
                "lv_bus": "to_node",
            },
            inplace=True,
        )
    else:
        dataframe["type"] = asset_type
        dataframe.rename(
            columns={
                "from_bus": "from_node",
                "to_bus": "to_node",
            },
            inplace=True,
        )
    if "equipment" in dataframe.columns:  # a parameter from the Frauenhofer script
        dataframe.rename(columns={"equipment": "foreign_id"}, inplace=True)
    else:
        dataframe["foreign_id"] = dataframe["name"]
    dataframe.rename(
        columns={
            "name": "grid_model_id",
            "type": "asset_type",
        },
        inplace=True,
    )
    dataframe.fillna({"foreign_id": ""}, inplace=True)
    cond = dataframe["foreign_id"] == ""
    dataframe.loc[cond, "foreign_id"] = dataframe.loc[cond, "grid_model_id"]
    dataframe["from_node"] = dataframe["from_node"].astype(int)
    dataframe["to_node"] = dataframe["to_node"].astype(int)
    if only_relevant_col:
        dataframe = dataframe[needed_col]
    return dataframe


def get_nodes(net: pandapowerNet, only_relevant_col: bool = True) -> NodeSchema:
    """Get the nodes data from the pandapower network and return a df ready to use for the NodeSchema.

    Parameters
    ----------
    net : pandapower
        The pandapower network.
    only_relevant_col : bool, optional
        Whether to return only the relevant columns, by default True
        relevant is determined by the default NodeSchema columns

    Returns
    -------
    nodes_df : NodeSchema
        The DataFrame containing the nodes in the format of the NodeSchema.
    """
    nodes_df = net.bus.copy()
    if "equipment" in nodes_df.columns:
        nodes_df.rename(columns={"equipment": "foreign_id"}, inplace=True)
    else:
        nodes_df["foreign_id"] = nodes_df["name"]
    nodes_df.rename(
        columns={"name": "grid_model_id", "type": "node_type", "vn_kv": "voltage_level", "zone": "system_operator"},
        inplace=True,
    )
    nodes_df["node_type"] = "node"
    logger.warning(
        "There is a bug in Pandapower where the 'Busbar_id' is not extracted from the CGMES data. See:"
        "https://github.com/e2nIEE/pandapower/issues/2517"
        "The next line will fail if the bug is not fixed."
        "add the following hotfix to pandapower at:"
        "pandapower/converter/cim/cim2pp/converter_classes/connectivitynodes/connectivityNodesCim16.py"
        "line 24:connectivity_nodes, eqssh_terminals = self._prepare_connectivity_nodes_cim16()"
        "new line:connectivity_nodes = connectivity_nodes.rename(columns={'busbar_id': 'Busbar_id', 'busbar_name': 'Busbar_name'})"  # noqa: E501
    )
    nodes_df.loc[~(nodes_df.Busbar_id.isna() | (nodes_df.Busbar_id == "")), "node_type"] = "busbar"

    nodes_df.loc[nodes_df["system_operator"].isna(), "system_operator"] = ""

    if "Substation_id" in nodes_df.columns:
        nodes_df.rename(columns={"Substation_id": "substation_id"}, inplace=True)
    else:
        nodes_df["substation_id"] = ""
    nodes_df.fillna({"foreign_id": ""}, inplace=True)
    nodes_df.fillna({"voltage_level": 0}, inplace=True)
    nodes_df["helper_node"] = False
    nodes_df["voltage_level"] = nodes_df["voltage_level"].astype(int)
    nodes_df["bus_id"] = nodes_df.index.astype(str)
    if only_relevant_col:
        needed_col = list(NodeSchema.to_schema().columns.keys())
        nodes_df = nodes_df[needed_col]

    return nodes_df


def get_switches_df(net: pandapowerNet, only_relevant_col: bool = True) -> SwitchSchema:
    """Get the switches data from the pandapower network and return a df ready to use for the SwitchSchema.

    Parameters
    ----------
    net : pandapower
        The pandapower network.
    only_relevant_col : bool, optional
        Whether to return only the relevant columns, by default True
        relevant is determined by the default SwitchSchema columns

    Returns
    -------
    switches_df : SwitchSchema
        The DataFrame containing the switches in the format of the SwitchSchema.
    """
    switches_df = net.switch.copy()
    # get only switches that interconnect buses
    switches_df = switches_df[switches_df["et"] == "b"]
    # reverse the "closed" column to "open"
    switches_df["closed"] = ~switches_df["closed"]
    switches_df.rename(columns={"closed": "open"}, inplace=True)
    if "in_service" not in switches_df.columns:
        switches_df["in_service"] = True
    switches_df = get_edges_data(switches_df, asset_type="switch", only_relevant_col=only_relevant_col)
    return switches_df


def get_branch_df(net: pandapowerNet, only_relevant_col: bool = True) -> BranchSchema:
    """Get the branches data from the pandapower network and return a df ready to use for the BranchSchema.

    Note: A star equivalent transformation for three winding trafos is not needed before calling this module.
    The the graph module only needs the connection and does no calculation.

    Parameters
    ----------
    net : pandapower
        The pandapower network.
    only_relevant_col : bool, optional
        Whether to return only the relevant columns, by default True
        relevant is determined by the default BranchSchema columns

    Returns
    -------
    branch_df : BranchSchema
        The DataFrame containing the branches in the format of the BranchSchema.
    """
    line_df = net.line.copy()
    line_df = get_edges_data(line_df, asset_type="line", only_relevant_col=only_relevant_col)

    impedances_df = net.impedance.copy()
    impedances_df = get_edges_data(impedances_df, asset_type="impedance", only_relevant_col=only_relevant_col)
    tcsc_df = net.tcsc.copy()
    tcsc_df = get_edges_data(tcsc_df, asset_type="tcsc", only_relevant_col=only_relevant_col)

    dclines = net.dcline.copy()
    dclines = get_edges_data(dclines, asset_type="dcline", only_relevant_col=only_relevant_col)

    transformers = net.trafo.copy()
    transformers = get_edges_data(transformers, asset_type="trafo", only_relevant_col=only_relevant_col)

    trafos3w1 = net.trafo3w.copy()
    trafos3w2 = net.trafo3w.copy()
    trafos3w2["lv_bus"] = trafos3w2["mv_bus"]
    trafos3w = pd.concat([trafos3w1, trafos3w2])
    trafos3w = get_edges_data(trafos3w, asset_type="trafo3w", only_relevant_col=only_relevant_col)

    branches_df = pd.concat([line_df, impedances_df, tcsc_df, dclines, transformers, trafos3w])
    branches_df.reset_index(drop=True, inplace=True)
    return branches_df
