# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
from toop_engine_importer.network_graph.data_classes import BusbarConnectionInfo, NetworkGraphData
from toop_engine_importer.network_graph.network_graph import generate_graph
from toop_engine_importer.network_graph.pandapower_network_to_graph import (
    get_branch_df,
    get_network_graph,
    get_network_graph_data,
    get_nodes,
    get_switches_df,
    set_substation_id,
)


def test_dataframe_creation(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler

    only_relevant_col = True
    branches_df = get_branch_df(net, only_relevant_col=only_relevant_col)
    nodes_df = get_nodes(net, only_relevant_col=only_relevant_col)
    switches_df = get_switches_df(net, only_relevant_col=only_relevant_col)
    assert (branches_df["asset_type"].unique() == ["line", "impedance", "trafo", "trafo3w"]).all()
    assert (
        len(branches_df)
        == len(net.line) + len(net.impedance) + len(net.tcsc) + len(net.dcline) + len(net.trafo) + len(net.trafo3w) * 2
    )

    assert len(nodes_df) == len(net.bus)
    assert len(switches_df) == len(net.switch[net.switch["et"] == "b"])
    assert nodes_df["helper_node"].unique() == [False]


def test_get_network_graph(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    graph_data = get_network_graph_data(net)
    assert isinstance(graph_data, NetworkGraphData)
    graph = get_network_graph(graph_data)
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == len(net.bus)
    assert list(dict(graph.nodes(data="node_type")).values()).count("busbar") == 6
    assert len(graph.edges) == len(net.line) + len(net.impedance) + len(net.tcsc) + len(net.dcline) + len(net.trafo) + len(
        net.trafo3w
    ) * 2 + len(net.switch[net.switch["et"] == "b"])
    # assert len(graph.node_assets) == 0 # not implemented yet
    assert list(dict(graph.nodes(data="helper_node")).values()).count(True) == 0  # not used in pandapower


def test_set_substation_id(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    assert "substation_id" not in net.bus.columns
    graph_data = get_network_graph_data(net)
    graph = generate_graph(graph_data)
    set_substation_id(graph=graph, network_graph_data=graph_data)
    nodes_df = get_nodes(net, only_relevant_col=True)

    substation_ids_busbars_expected = [
        "Double Busbar 1",
        "Double Busbar 1",
        "Double Busbar Coupler 1",
        "Double Busbar Coupler 1",
        "Double Busbar Coupler 1",
        "Double Busbar Coupler 1",
    ]
    busbar_ids = nodes_df[(nodes_df["node_type"] == "busbar")].index
    substation_ids = dict(graph.nodes(data="substation_id"))
    substation_id_list = [substation_ids[busbar_id] for busbar_id in busbar_ids]
    # assert graph.nodes[(graph.nodes["node_type"] == "busbar")]["substation_id"].to_list() == substation_ids_busbars_expected
    assert substation_id_list == substation_ids_busbars_expected
    substation_ids_all = [
        "Double Busbar 1",
        "Double Busbar Coupler 1",
        "Bus HV1",
        "Bus HV2",
        "Bus HV3",
        "Bus HV4",
        "Bus MV0 20kV",
        "Bus MV0",
        "Bus MV1",
        "Bus MV2",
        "Bus MV3",
        "Bus MV4",
        "Bus MV5",
        "Bus MV6",
        "Bus MV7",
        "Bus LV0",
        "Bus LV1.1",
        "Bus LV1.2",
        "Bus LV1.3",
        "Bus LV1.4",
        "Bus LV1.5",
        "Bus LV2.1",
        "Bus LV2.2",
        "Bus LV2.3",
        "Bus LV2.4",
        "Bus LV2.2.1",
        "Bus LV2.2.2",
    ]

    substation_id_list = list(dict.fromkeys(dict(graph.nodes(data="substation_id")).values()))
    assert substation_id_list == substation_ids_all


def test_busbar_connection_info(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    graph_data = get_network_graph_data(net)
    graph = get_network_graph(graph_data)

    # set_substation_id(graph=graph, network_graph_data=graph_data)
    # for node_id in graph.nodes:
    #     assert graph.nodes[node_id]["busbar_connection_info"] == BusbarConnectionInfo()
    # run_default_filter_strategy(graph=graph)
    # Note: EHV-HV-Trafo is connected to two substations -> is found both stations -> connectable_assets_busbar[0:4]
    # There is also an error in the grid where EHV-HV-Trafo is connected to two busbars in one substation -> connectable_assets_busbar[2] and connectable_assets_busbar[3]
    connectable_assets_busbar = [
        BusbarConnectionInfo(
            connectable_assets_node_ids=[13],
            connectable_assets=["EHV-HV-Trafo"],
            connectable_busbars_node_ids=[1],
            connectable_busbars=["Double Busbar 2"],
            zero_impedance_connected_assets=[],
            zero_impedance_connected_assets_node_ids=[],
            zero_impedance_connected_busbars=["Double Busbar 2"],
            zero_impedance_connected_busbars_node_ids=[1],
        ),
        BusbarConnectionInfo(
            connectable_assets_node_ids=[13],
            connectable_assets=["EHV-HV-Trafo"],
            connectable_busbars_node_ids=[0],
            connectable_busbars=["Double Busbar 1"],
            zero_impedance_connected_assets=["EHV-HV-Trafo"],
            zero_impedance_connected_assets_node_ids=[13],
            zero_impedance_connected_busbars=["Double Busbar 1"],
            zero_impedance_connected_busbars_node_ids=[0],
        ),
        BusbarConnectionInfo(
            connectable_assets_node_ids=[17, 18],
            connectable_assets=["EHV-HV-Trafo", "HV Line1"],
            connectable_busbars_node_ids=[58, 57],
            connectable_busbars=["Double Busbar Coupler 3", "Double Busbar Coupler 2"],
            zero_impedance_connected_assets=["EHV-HV-Trafo", "HV Line1"],
            zero_impedance_connected_assets_node_ids=[17, 18],
            zero_impedance_connected_busbars=["Double Busbar Coupler 2", "Double Busbar Coupler 3"],
            zero_impedance_connected_busbars_node_ids=[57, 58],
        ),
        BusbarConnectionInfo(
            connectable_assets_node_ids=[17, 18],
            connectable_assets=["EHV-HV-Trafo", "HV Line1"],
            connectable_busbars_node_ids=[59, 16],
            connectable_busbars=["Double Busbar Coupler 4", "Double Busbar Coupler 1"],
            zero_impedance_connected_assets=["EHV-HV-Trafo"],
            zero_impedance_connected_assets_node_ids=[17],
            zero_impedance_connected_busbars=["Double Busbar Coupler 1", "Double Busbar Coupler 4"],
            zero_impedance_connected_busbars_node_ids=[16, 59],
        ),
        BusbarConnectionInfo(
            connectable_assets_node_ids=[19],
            connectable_assets=["HV Line6"],
            connectable_busbars_node_ids=[16, 59],
            connectable_busbars=["Double Busbar Coupler 1", "Double Busbar Coupler 4"],
            zero_impedance_connected_assets=["HV Line6"],
            zero_impedance_connected_assets_node_ids=[19],
            zero_impedance_connected_busbars=["Double Busbar Coupler 1", "Double Busbar Coupler 4"],
            zero_impedance_connected_busbars_node_ids=[16, 59],
        ),
        BusbarConnectionInfo(
            connectable_assets_node_ids=[19],
            connectable_assets=["HV Line6"],
            connectable_busbars_node_ids=[57, 58],
            connectable_busbars=["Double Busbar Coupler 2", "Double Busbar Coupler 3"],
            zero_impedance_connected_assets=[],
            zero_impedance_connected_assets_node_ids=[],
            zero_impedance_connected_busbars=["Double Busbar Coupler 2", "Double Busbar Coupler 3"],
            zero_impedance_connected_busbars_node_ids=[57, 58],
        ),
    ]
    busbar_ids = graph_data.nodes[(graph_data.nodes["node_type"] == "busbar")].index
    res = [graph.nodes[busbar_id]["busbar_connection_info"] for busbar_id in busbar_ids]
    assert res[0] == connectable_assets_busbar[0]
    assert res[1] == connectable_assets_busbar[1]
    assert res[2] == connectable_assets_busbar[2]
    assert res[3] == connectable_assets_busbar[3]
    assert res[4] == connectable_assets_busbar[4]
    assert res[5] == connectable_assets_busbar[5]
