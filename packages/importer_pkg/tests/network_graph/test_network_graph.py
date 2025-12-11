import copy

import networkx as nx
import pandas as pd
import pytest
from toop_engine_importer.network_graph.data_classes import (
    BRANCH_TYPES,
    SWITCH_TYPES,
    BusbarConnectionInfo,
    EdgeConnectionInfo,
    NetworkGraphData,
    WeightValues,
)
from toop_engine_importer.network_graph.network_graph import (
    append_connection_info,
    flatten_list_of_mixed_entries,
    generate_graph,
    get_all_node_paths_of_a_station_from_a_node,
    get_branch_ids_by_type_list,
    get_busbar_connection_info,
    get_busbar_connection_info_attribute,
    get_busbar_true_nodes,
    get_edge_connection_info,
    get_edge_list_by_attribute,
    get_helper_node_ids,
    get_node_list_by_attribute,
    get_nodes_ids_with_a_connected_asset,
    graph_creation_edge_helper,
    graph_creation_node_assets_helper,
    graph_creation_nodes_helper,
    multi_weight_function,
    shortest_paths_to_target_ids,
    update_busbar_connection_info,
    update_edge_connection_info,
    validate_update_dict_for_connection_info,
)
from toop_engine_importer.network_graph.network_graph_data import add_graph_specific_data
from toop_engine_importer.network_graph.powsybl_station_to_graph import get_node_breaker_topology_graph


def test_get_busbar_connection_info(get_graph_input_dicts):
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    switches_df = pd.DataFrame(switches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True

    network_graph_data = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df)

    assert network_graph_data.nodes[nodes_df.columns].equals(nodes_df)
    assert network_graph_data.switches[switches_df.columns].equals(switches_df)
    assert network_graph_data.node_assets[nodes_asstets_df.columns].equals(nodes_asstets_df)
    assert network_graph_data.helper_branches.empty
    assert network_graph_data.branches.empty

    add_graph_specific_data(network_graph_data)
    graph = get_node_breaker_topology_graph(network_graph_data)

    # this tests the endresult -> if this fails other functions failed to work
    busbar_info_1 = BusbarConnectionInfo(
        connectable_assets=["L3", "L6", "L7", "L9", "load2"],
        connectable_assets_node_ids=[2, 4, 6, 8, 12],
        connectable_busbars=["BBS3_2"],
        connectable_busbars_node_ids=[1],
        zero_impedance_connected_assets=["L3", "L6", "L9"],
        zero_impedance_connected_assets_node_ids=[2, 4, 8],
        zero_impedance_connected_busbars=["BBS3_2"],
        zero_impedance_connected_busbars_node_ids=[1],
        node_assets=[],
        node_assets_ids=[],
    )
    busbar_info_2 = BusbarConnectionInfo(
        connectable_assets=["L3", "L6", "L7", "L9", "load2"],
        connectable_assets_node_ids=[2, 4, 6, 8, 12],
        connectable_busbars=["BBS3_1"],
        connectable_busbars_node_ids=[0],
        zero_impedance_connected_assets=["L7", "load2"],
        zero_impedance_connected_assets_node_ids=[6, 12],
        zero_impedance_connected_busbars=["BBS3_1"],
        zero_impedance_connected_busbars_node_ids=[0],
        node_assets=[],
        node_assets_ids=[],
    )
    expected_busbar_info = {"BBS3_1": busbar_info_1, "BBS3_2": busbar_info_2}

    bus_info = get_busbar_connection_info(graph=graph)
    assert expected_busbar_info == bus_info
    bus_info = get_busbar_connection_info(graph=graph, busbar_grid_model_id="BBS3_1")
    assert bus_info["BBS3_1"] == busbar_info_1


def test_get_edge_connection_info(get_graph_input_dicts):
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    switches_df = pd.DataFrame(switches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True

    network_graph_data = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df)

    assert network_graph_data.nodes[nodes_df.columns].equals(nodes_df)
    assert network_graph_data.switches[switches_df.columns].equals(switches_df)
    assert network_graph_data.node_assets[nodes_asstets_df.columns].equals(nodes_asstets_df)
    assert network_graph_data.helper_branches.empty
    assert network_graph_data.branches.empty

    add_graph_specific_data(network_graph_data)
    graph = get_node_breaker_topology_graph(network_graph_data)

    expcted_edge_info = {
        "L32_DISCONNECTOR_3_0": EdgeConnectionInfo(
            direct_busbar_grid_model_id="BBS3_1", bay_id="L3", coupler_type="", coupler_grid_model_id_list=[]
        )
    }
    edge_info = get_edge_connection_info(graph=graph, edge_grid_model_ids=["L32_DISCONNECTOR_3_0"])
    assert edge_info == expcted_edge_info
    edge_info = get_edge_connection_info(graph=graph)
    for edge in edge_info.values():
        assert isinstance(edge, EdgeConnectionInfo)


def test_generate_graph_missing_dataframes(get_graph_input_dicts_helper_branches):
    nodes_dict, switches_dict, node_assets_dict, helper_branches_dict = get_graph_input_dicts_helper_branches
    switches_df = pd.DataFrame(switches_dict)
    nodes_df = pd.DataFrame(nodes_dict)
    helper_branches_df = pd.DataFrame(helper_branches_dict)
    nodes_asstets_df = pd.DataFrame(node_assets_dict)
    nodes_df["in_service"] = True
    switches_df["in_service"] = True
    nodes_asstets_df["in_service"] = True
    helper_branches_df["in_service"] = True
    helper_branches_df["grid_model_id"] = ""

    # should not fail as this is the intended use case
    network_graph_data = NetworkGraphData(
        nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df, helper_branches=helper_branches_df
    )
    network_graph = generate_graph(network_graph_data)
    assert isinstance(network_graph, nx.Graph)

    # missing helper_branches -> should not fail
    network_graph = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=nodes_asstets_df)
    network_graph = generate_graph(network_graph_data)
    assert isinstance(network_graph, nx.Graph)


def test_graph_creation_nodes_helper():
    graph = nx.Graph()
    node_dict = {
        0: {
            "connectable_id": "",
            "connectable_type": "",
            "foreign_id": "",
            "grid_model_id": "",
            "node_type": "node",
            "substation_id": "Test_station1",
            "system_operator": "TSO",
            "voltage_level": 150,
            "helper_node": True,
        }
    }
    nodes_df = pd.DataFrame(node_dict).T
    graph_creation_nodes_helper(nodes_df=nodes_df, graph=graph)
    assert graph.nodes[0] == {
        "connectable_id": "",
        "connectable_type": "",
        "foreign_id": "",
        "grid_model_id": "",
        "node_type": "node",
        "substation_id": "Test_station1",
        "system_operator": "TSO",
        "voltage_level": 150,
        "helper_node": True,
        "busbar_connection_info": BusbarConnectionInfo(),
    }


def test_graph_creation_edge_helper():
    graph = nx.Graph()
    edge_dict = {
        0: {
            "from_node": 1,
            "to_node": 2,
            "grid_model_id": "Test_edge",
            "foreign_id": "Test_edge",
            "asset_type": "Test_edge",
            "open": True,
        }
    }
    edges_df = pd.DataFrame(edge_dict).T
    graph_creation_edge_helper(edge_df=edges_df, graph=graph)
    assert graph[1][2] == {
        "grid_model_id": "Test_edge",
        "foreign_id": "Test_edge",
        "asset_type": "Test_edge",
        "open": True,
        "edge_connection_info": EdgeConnectionInfo(),
    }


def test_graph_creation_node_assets_helper():
    node_assets_data = {"node": [1, 1, 2], "grid_model_id": ["asset_1", "asset_2", "asset_3"]}
    nodes_data = {"node": [1, 2]}
    node_assets_df = pd.DataFrame(node_assets_data)
    nodes_df = pd.DataFrame(nodes_data).set_index("node")
    # create graph
    graph = nx.Graph()
    graph.add_node(1, busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(2, busbar_connection_info=BusbarConnectionInfo())
    graph_creation_node_assets_helper(node_assets_df=node_assets_df, graph=graph)

    assert graph.nodes[1]["busbar_connection_info"].node_assets == ["asset_1", "asset_2"]
    assert graph.nodes[2]["busbar_connection_info"].node_assets == ["asset_3"]
    assert graph.nodes[1]["busbar_connection_info"].node_assets_ids == [0, 1]
    assert graph.nodes[2]["busbar_connection_info"].node_assets_ids == [2]


def test_shortest_paths_to_target_ids():
    graph = nx.Graph()
    edges = [
        (1, 2, {"station_weight": 1}),
        (2, 3, {"station_weight": 2}),
        (3, 4, {"station_weight": 1}),
        (4, 5, {"station_weight": 3}),
        (1, 6, {"station_weight": 4}),
        (6, 5, {"station_weight": 2}),
    ]
    graph.add_edges_from(edges)

    target_node_ids = [4, 5]
    node_id = 1

    result = shortest_paths_to_target_ids(graph, target_node_ids, node_id)

    assert result == {4: [1, 2, 3, 4], 5: [1, 6, 5]}

    result_with_cutoff = shortest_paths_to_target_ids(graph, target_node_ids, node_id, cutoff=4)
    assert result_with_cutoff == {4: [1, 2, 3, 4]}

    result_with_cutoff = shortest_paths_to_target_ids(graph, target_node_ids, node_id, cutoff=5)
    assert result_with_cutoff == {4: [1, 2, 3, 4]}

    result_with_weight = shortest_paths_to_target_ids(graph, target_node_ids, node_id, weight="station_weight")
    assert result_with_weight == {4: [1, 2, 3, 4], 5: [1, 6, 5]}

    result_with_nonexistent_target = shortest_paths_to_target_ids(graph, [7], node_id)
    assert result_with_nonexistent_target == {}


def test_get_list_of_int_from_list_int_tuple():
    input_data = [1, (2, 3), 4, (5, 6)]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output

    input_data = [(1, 2), (3, 4), (5, 6)]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output

    input_data = [1, 2, 3, 4, 5, 6]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output

    input_data = []
    expected_output = []
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output

    input_data = [1, (2, 3), (4, 5), 6]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output

    input_data = [(1, 2)]
    expected_output = [1, 2]
    assert list(flatten_list_of_mixed_entries(input_data)) == expected_output


def test_get_busbar_true_nodes_with_helper_nodes(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    network_graph = generate_graph(network_graph_data)
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=network_graph)
    assert busbars == [28, 29]
    assert busbars_helper_nodes == [10, 5]


def test_get_busbar_true_nodes_no_helper_nodes(network_graph_data_test1):
    network_graph_data = network_graph_data_test1
    network_graph = generate_graph(network_graph_data)
    assert get_helper_node_ids(network_graph) == []
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=network_graph)
    assert busbars == busbars_helper_nodes


def test_helper_nodes_vs_removed_helper_nodes_logic(
    network_graph_data_test2_helper_branches, network_graph_data_test2_helper_branches_removed
):
    network_graph_data_hb = network_graph_data_test2_helper_branches
    graph_hb = generate_graph(network_graph_data_hb)
    network_graph_data_no_hb = network_graph_data_test2_helper_branches_removed
    graph_no_hb = generate_graph(network_graph_data_no_hb)
    busbars, busbars_helper_nodes = get_busbar_true_nodes(graph=graph_hb)
    busbars_no_hb, busbars_helper_nodes_no_hb = get_busbar_true_nodes(graph=graph_no_hb)
    assert network_graph_data_hb.nodes["helper_node"].any()
    assert not network_graph_data_no_hb.nodes["helper_node"].any()
    assert busbars == busbars_no_hb
    assert busbars == busbars_helper_nodes_no_hb

    # all inclusive integration test
    bus_info_no_hb = get_busbar_connection_info(graph=graph_no_hb)
    bus_info_hb = get_busbar_connection_info(graph=graph_hb)
    assert bus_info_hb == bus_info_no_hb


def test_multi_weight_function():
    graph = nx.Graph()
    weight1 = 1.0
    weight2 = 2.0
    graph.add_edge(1, 2, weight1=weight1, weight2=weight2)

    result = nx.shortest_path_length(graph, source=1, target=2, weight="weight1", method="dijkstra")
    assert result == 1
    weight_list = ["weight1"]
    weight_func = multi_weight_function(weight_list)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1

    result = nx.shortest_path_length(graph, source=1, target=2, weight="weight2", method="dijkstra")
    assert result == weight2
    weight_list = ["weight2"]
    weight_func = multi_weight_function(weight_list)
    assert result == weight2

    weight_list = ["weight1", "weight2"]
    weight_func = multi_weight_function(weight_list)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1 + weight2

    # test weight_multiplier
    weight_list = ["weight1", "weight2"]
    weight_multiplier = {"weight1": 2.0, "weight2": 3.0}
    weight_func = multi_weight_function(weight_list, weight_multiplier=weight_multiplier)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1 * weight_multiplier["weight1"] + weight2 * weight_multiplier["weight2"]

    weight_list = ["weight1", "weight2"]
    weight_multiplier = {"weight1": 2.0, "weight2": 3.0, "weight3": 4.0}
    weight_func = multi_weight_function(weight_list, weight_multiplier=weight_multiplier)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1 * weight_multiplier["weight1"] + weight2 * weight_multiplier["weight2"]

    weight_list = ["weight1", "weight2"]
    weight_multiplier = {"weight1": 2.0}
    weight_func = multi_weight_function(weight_list, weight_multiplier=weight_multiplier)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1 * weight_multiplier["weight1"] + weight2 * 1.0

    weight_list = ["weight1", "weight2"]
    weight_multiplier = {}
    weight_func = multi_weight_function(weight_list, weight_multiplier=weight_multiplier)
    result = nx.shortest_path_length(graph, source=1, target=2, weight=weight_func, method="dijkstra")
    assert result == weight1 * 1.0 + weight2 * 1.0


def test_get_node_list_by_attribute():
    graph = nx.Graph()
    graph.add_node(1, attribute1="value1", attribute2="value2")
    graph.add_node(2, attribute1="value1", attribute2="value3")
    graph.add_node(3, attribute1="value2", attribute2="value2")
    graph.add_node(4, attribute1="value3", attribute2="value1")

    # Test for attribute1 with value "value1"
    result = get_node_list_by_attribute(graph, "attribute1", ["value1"])
    assert result == [1, 2]

    # Test for attribute2 with value "value2"
    result = get_node_list_by_attribute(graph, "attribute2", ["value2"])
    assert result == [1, 3]

    # Test for attribute1 with value "value3"
    result = get_node_list_by_attribute(graph, "attribute1", ["value3"])
    assert result == [4]

    # Test for attribute2 with value "value1"
    result = get_node_list_by_attribute(graph, "attribute2", ["value1"])
    assert result == [4]

    # Test for non-existent attribute
    result = get_node_list_by_attribute(graph, "attribute3", ["value1"])
    assert result == []

    # Test for non-existent value
    result = get_node_list_by_attribute(graph, "attribute1", ["value4"])
    assert result == []


def test_get_helper_node_ids():
    graph = nx.Graph()
    graph.add_node(1, helper_node=True)
    graph.add_node(2, helper_node=False)
    graph.add_node(3, helper_node=True)
    graph.add_node(4, helper_node=False)
    graph.add_node(5, helper_node=True)

    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5]

    graph.add_node(6, helper_node=False)
    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5]

    graph.add_node(7, helper_node=True)
    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5, 7]

    graph.add_node(8)
    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5, 7]

    graph.add_node(9, helper_node=False)
    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5, 7]

    graph.add_node(10, helper_node=True)
    result = get_helper_node_ids(graph)
    assert result == [1, 3, 5, 7, 10]


def test_get_asset_nodes():
    graph = nx.Graph()
    graph.add_node(1, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[1, 2]))
    graph.add_node(2, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[]))
    graph.add_node(3, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[3]))
    graph.add_node(4, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[]))
    graph.add_node(5, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[4, 5, 6]))

    result = get_nodes_ids_with_a_connected_asset(graph)
    assert result == [1, 3, 5]

    graph.add_node(6, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[]))
    result = get_nodes_ids_with_a_connected_asset(graph)
    assert result == [1, 3, 5]

    graph.add_node(7, busbar_connection_info=BusbarConnectionInfo(node_assets_ids=[7]))
    result = get_nodes_ids_with_a_connected_asset(graph)
    assert result == [1, 3, 5, 7]


def test_get_busbar_connection_info_attribute():
    graph = nx.Graph()
    graph.add_node(1, node_type="busbar", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[2, 3]))
    graph.add_node(2, node_type="busbar", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[1]))
    graph.add_node(3, node_type="busbar", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[1]))
    graph.add_node(4, node_type="node", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[]))
    graph.add_node(5, node_type="node", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[6]))
    graph.add_node(6, node_type="node", busbar_connection_info=BusbarConnectionInfo(connectable_busbars_node_ids=[5]))

    # Test for busbar node_type
    result = get_busbar_connection_info_attribute(graph, "connectable_busbars_node_ids", node_type="busbar")
    expected_result = {1: [2, 3], 2: [1], 3: [1]}
    assert result == expected_result

    # Test for node node_type
    result = get_busbar_connection_info_attribute(graph, "connectable_busbars_node_ids", node_type="node")
    expected_result = {4: [], 5: [6], 6: [5]}
    assert result == expected_result

    # Test for all nodes
    result = get_busbar_connection_info_attribute(graph, "connectable_busbars_node_ids")
    expected_result = {1: [2, 3], 2: [1], 3: [1], 4: [], 5: [6], 6: [5]}
    assert result == expected_result

    # Test for non-existent attribute
    with pytest.raises(KeyError):
        get_busbar_connection_info_attribute(graph, "non_existent_attribute")

    # Test for empty graph
    graph.add_node(8)
    with pytest.raises(AttributeError):
        get_busbar_connection_info_attribute(graph, "connectable_busbars_node_ids")


def test_get_branch_ids_by_type_list():
    graph = nx.Graph()
    edges = [
        (1, 2, {"asset_type": BRANCH_TYPES.__args__[0].__args__[0], "node_tuple": set([1, 2])}),
        (2, 3, {"asset_type": BRANCH_TYPES.__args__[0].__args__[1], "node_tuple": set([2, 3])}),
        (3, 4, {"asset_type": BRANCH_TYPES.__args__[1].__args__[0], "node_tuple": set([3, 4])}),
        (4, 5, {"asset_type": BRANCH_TYPES.__args__[1].__args__[1], "node_tuple": set([4, 5])}),
        (5, 6, {"asset_type": "not_a_branch_type", "node_tuple": set([5, 6])}),
        (6, 7, {"asset_type": SWITCH_TYPES.__args__[0], "node_tuple": set([6, 7])}),
    ]
    graph.add_edges_from(edges)

    result = get_branch_ids_by_type_list(graph)
    expected_result = [{1, 2}, {2, 3}, {3, 4}, {4, 5}]
    assert result == expected_result

    branch_types = [BRANCH_TYPES.__args__[0].__args__[0], BRANCH_TYPES.__args__[1].__args__[0]]
    result = get_branch_ids_by_type_list(graph, branch_types)
    expected_result = [{1, 2}, {3, 4}]
    assert result == expected_result

    branch_types = ["not_a_branch_type"]
    result = get_branch_ids_by_type_list(graph, branch_types)
    expected_result = [{5, 6}]
    assert result == expected_result


def test_get_edge_list_by_attribute():
    graph = nx.Graph()
    edges = [
        (1, 2, {"attribute1": "value1", "node_tuple": {1, 2}}),
        (2, 3, {"attribute1": "value2", "node_tuple": {2, 3}}),
        (3, 4, {"attribute1": "value1", "node_tuple": {3, 4}}),
        (4, 5, {"attribute1": "value3", "node_tuple": {4, 5}}),
        (5, 6, {"attribute2": "value1", "node_tuple": {5, 6}}),
        (6, 7, {"attribute2": "value2", "node_tuple": {6, 7}}),
    ]
    graph.add_edges_from(edges)

    # Test for attribute1 with value "value1"
    result = get_edge_list_by_attribute(graph, "attribute1", ["value1"])
    assert result == [{1, 2}, {3, 4}]

    # Test for attribute1 with value "value2"
    result = get_edge_list_by_attribute(graph, "attribute1", ["value2"])
    assert result == [{2, 3}]

    # Test for attribute2 with value "value1"
    result = get_edge_list_by_attribute(graph, "attribute2", ["value1"])
    assert result == [{5, 6}]

    # Test for attribute2 with value "value2"
    result = get_edge_list_by_attribute(graph, "attribute2", ["value2"])
    assert result == [{6, 7}]

    # Test for non-existent attribute
    result = get_edge_list_by_attribute(graph, "attribute3", ["value1"])
    assert result == []

    # Test for non-existent value
    result = get_edge_list_by_attribute(graph, "attribute1", ["value4"])
    assert result == []

    result = get_edge_list_by_attribute(graph, "attribute1", ["value1", "value2", "value3"])
    assert result == [{1, 2}, {2, 3}, {3, 4}, {4, 5}]


def test_update_busbar_connection_info():
    graph = nx.Graph()
    graph.add_node(1, busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(2, busbar_connection_info=BusbarConnectionInfo())
    graph.add_node(3, busbar_connection_info=BusbarConnectionInfo())

    update_busbar_connection_info(graph=graph, update_node_dict={1: {"connectable_assets": []}})
    for node in graph.nodes:
        assert graph.nodes[node]["busbar_connection_info"] == BusbarConnectionInfo()

    connectable_assets = ["test", "test"]
    update_dict = {2: {"connectable_assets": connectable_assets}}
    update_busbar_connection_info(graph=graph, update_node_dict=update_dict)
    assert graph.nodes[2]["busbar_connection_info"] == BusbarConnectionInfo(connectable_assets=connectable_assets)

    connectable_busbars_node_ids = [1, 2]
    update_dict = {
        3: {"connectable_assets": connectable_assets},
        2: {"connectable_busbars_node_ids": connectable_busbars_node_ids},
    }
    update_busbar_connection_info(graph=graph, update_node_dict=update_dict)
    assert graph.nodes[2]["busbar_connection_info"] == BusbarConnectionInfo(
        connectable_assets=connectable_assets, connectable_busbars_node_ids=connectable_busbars_node_ids
    )
    assert graph.nodes[3]["busbar_connection_info"] == BusbarConnectionInfo(connectable_assets=connectable_assets)

    connectable_busbars_node_ids = [1]
    update_dict = {2: {"connectable_busbars_node_ids": connectable_busbars_node_ids}}
    update_busbar_connection_info(graph=graph, update_node_dict=update_dict)
    assert graph.nodes[2]["busbar_connection_info"] == BusbarConnectionInfo(
        connectable_assets=connectable_assets, connectable_busbars_node_ids=connectable_busbars_node_ids
    )

    not_existing_arg = {1: {"NOT_EXISTING_ARG": ""}}
    with pytest.raises(ValueError, match="Update dictionary contains unknown keys for"):
        update_busbar_connection_info(graph=graph, update_node_dict=not_existing_arg)

    connectable_assets = ["test_set", "test_set2"]
    connectable_busbars_node_ids = [1, 2]
    update_dict = {
        2: {"connectable_assets": connectable_assets, "connectable_busbars_node_ids": connectable_busbars_node_ids}
    }
    update_busbar_connection_info(graph=graph, update_node_dict=update_dict)
    res = copy.deepcopy(graph.nodes[2]["busbar_connection_info"])
    update_busbar_connection_info(graph=graph, update_node_dict=update_dict, method="set")
    assert graph.nodes[2]["busbar_connection_info"] == res

    update_busbar_connection_info(graph=graph, update_node_dict=update_dict, method="append")
    assert graph.nodes[2]["busbar_connection_info"].connectable_assets == ["test_set", "test_set2", "test_set", "test_set2"]
    assert graph.nodes[2]["busbar_connection_info"].connectable_busbars_node_ids == [1, 2, 1, 2]


def test_update_edge_connection_info():
    graph = nx.Graph()
    graph.add_edge(1, 2, edge_connection_info=EdgeConnectionInfo())
    graph.add_edge(2, 3, edge_connection_info=EdgeConnectionInfo())
    graph.add_edge(3, 4, edge_connection_info=EdgeConnectionInfo())

    update_edge_connection_info(graph=graph, update_edge_dict={})
    for edge in graph.edges:
        assert graph.edges[edge]["edge_connection_info"] == EdgeConnectionInfo()

    bay_id = "test"
    update_dict = {(1, 2): {"bay_id": bay_id}}
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict)
    assert graph.edges[(1, 2)]["edge_connection_info"] == EdgeConnectionInfo(bay_id=bay_id)

    bay_id2 = "test2"
    update_dict = {(2, 3): {"bay_id": bay_id2}, (1, 2): {"bay_id": bay_id}}
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict)
    assert graph.edges[(1, 2)]["edge_connection_info"] == EdgeConnectionInfo(bay_id=bay_id)
    assert graph.edges[(2, 3)]["edge_connection_info"] == EdgeConnectionInfo(bay_id=bay_id2)

    update_dict = {(2, 3): {"bay_id": bay_id}, (1, 2): {"bay_id": bay_id2}}
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict)
    assert graph.edges[(1, 2)]["edge_connection_info"] == EdgeConnectionInfo(bay_id=bay_id2)
    assert graph.edges[(2, 3)]["edge_connection_info"] == EdgeConnectionInfo(bay_id=bay_id)

    not_existing_arg = {(1, 2): {"NOT_EXISTING_ARG": ""}}
    with pytest.raises(ValueError, match="Update dictionary contains unknown keys for"):
        update_edge_connection_info(graph=graph, update_edge_dict=not_existing_arg)

    bay_id = "test_set"
    bay_id2 = "test_set2"
    update_dict = {(2, 3): {"bay_id": bay_id2}, (1, 2): {"bay_id": bay_id}}
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict)
    res1 = copy.deepcopy(graph.edges[(1, 2)]["edge_connection_info"])
    res2 = copy.deepcopy(graph.edges[(2, 3)]["edge_connection_info"])
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict, method="set")
    assert graph.edges[(1, 2)]["edge_connection_info"] == res1
    assert graph.edges[(2, 3)]["edge_connection_info"] == res2

    update_dict = {(2, 3): {"coupler_grid_model_id_list": [(1, 2)]}}
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict, method="set")
    assert graph.edges[(2, 3)]["edge_connection_info"].coupler_grid_model_id_list == [(1, 2)]
    update_edge_connection_info(graph=graph, update_edge_dict=update_dict, method="append")
    assert graph.edges[(2, 3)]["edge_connection_info"].coupler_grid_model_id_list == [(1, 2), (1, 2)]


def test_validate_update_dict_for_connection_info():
    busbar_connection_info = BusbarConnectionInfo()
    edge_connection_info = EdgeConnectionInfo()

    # Test with valid update_dict for BusbarConnectionInfo
    update_dict = {"connectable_assets": ["asset1", "asset2"]}
    assert validate_update_dict_for_connection_info(busbar_connection_info, update_dict)

    # Test with invalid update_dict for BusbarConnectionInfo
    update_dict = {"invalid_key": "value"}
    with pytest.raises(ValueError, match="Update dictionary contains unknown keys for"):
        validate_update_dict_for_connection_info(busbar_connection_info, update_dict)

    # Test with valid update_dict for EdgeConnectionInfo
    update_dict = {"bay_id": "bay1"}
    assert validate_update_dict_for_connection_info(edge_connection_info, update_dict)

    # Test with invalid update_dict for EdgeConnectionInfo
    update_dict = {"invalid_key": "value"}
    with pytest.raises(ValueError, match="Update dictionary contains unknown keys for"):
        validate_update_dict_for_connection_info(edge_connection_info, update_dict)

    # Test with empty update_dict
    update_dict = {}
    assert validate_update_dict_for_connection_info(busbar_connection_info, update_dict)
    assert validate_update_dict_for_connection_info(edge_connection_info, update_dict)


def test_append_connection_info():
    # Test with valid update_dict for BusbarConnectionInfo
    busbar_connection_info = BusbarConnectionInfo(connectable_assets=["asset1"], connectable_busbars=["busbar1"])
    update_dict = {"connectable_assets": ["asset2"], "connectable_busbars": ["busbar2"]}
    expected_result = BusbarConnectionInfo(
        connectable_assets=["asset1", "asset2"], connectable_busbars=["busbar1", "busbar2"]
    )
    result = append_connection_info(busbar_connection_info, update_dict)
    assert result == expected_result

    # Test with valid update_dict for EdgeConnectionInfo
    edge_connection_info = EdgeConnectionInfo(coupler_grid_model_id_list=[("coupler1", "coupler2")])
    update_dict = {"coupler_grid_model_id_list": [("coupler2", "coupler3")]}
    expected_result = EdgeConnectionInfo(coupler_grid_model_id_list=[("coupler1", "coupler2"), ("coupler2", "coupler3")])
    result = append_connection_info(edge_connection_info, update_dict)
    assert result == expected_result

    # Test with invalid update_dict (non-list value) for BusbarConnectionInfo
    busbar_connection_info = BusbarConnectionInfo(connectable_assets=["asset1"])
    update_dict = {"connectable_assets": "asset2"}
    with pytest.raises(ValueError, match="The value for key: connectable_assets is not a list."):
        append_connection_info(busbar_connection_info, update_dict)

    # Test with invalid update_dict (non-list value) for EdgeConnectionInfo
    edge_connection_info = EdgeConnectionInfo(coupler_grid_model_id_list=[("coupler1", "coupler2")])
    update_dict = {"coupler_grid_model_id_list": "coupler2"}
    with pytest.raises(ValueError, match="The value for key: coupler_grid_model_id_list is not a list."):
        append_connection_info(edge_connection_info, update_dict)

    # Test with invalid update_dict (non-list existing value) for EdgeConnectionInfo
    edge_connection_info = EdgeConnectionInfo(direct_busbar_grid_model_id="coupler1")
    update_dict = {"direct_busbar_grid_model_id": ["coupler2"]}
    with pytest.raises(ValueError, match="The value for key: direct_busbar_grid_model_id is not a list."):
        append_connection_info(edge_connection_info, update_dict)

    update_dict = {"direct_busbar_grid_model_id": "coupler2"}
    with pytest.raises(ValueError, match="The value for key: direct_busbar_grid_model_id is not a list."):
        append_connection_info(edge_connection_info, update_dict)


def test_get_all_node_paths_of_a_station_from_a_node():
    graph = nx.Graph()
    edges = [
        (1, 2, {"station_weight": 1}),
        (2, 3, {"station_weight": 2}),
        (3, 4, {"station_weight": 1}),
        (4, 5, {"station_weight": 3}),
        (5, 6, {"station_weight": WeightValues.high.value}),
        (6, 7, {"station_weight": 4}),
        (7, 8, {"station_weight": 2}),
        (8, 9, {"station_weight": WeightValues.high.value}),
        (9, 10, {"station_weight": 2}),
    ]
    graph.add_edges_from(edges)

    # Test station1
    result1 = get_all_node_paths_of_a_station_from_a_node(graph, 1)
    result2 = get_all_node_paths_of_a_station_from_a_node(graph, 2)
    result3 = get_all_node_paths_of_a_station_from_a_node(graph, 3)
    result4 = get_all_node_paths_of_a_station_from_a_node(graph, 4)
    result5 = get_all_node_paths_of_a_station_from_a_node(graph, 5)
    expected_result = {1, 2, 3, 4, 5}
    assert set(result1.keys()) == expected_result
    assert set(result2.keys()) == expected_result
    assert set(result3.keys()) == expected_result
    assert set(result4.keys()) == expected_result
    assert set(result5.keys()) == expected_result

    # Test station 2
    result6 = get_all_node_paths_of_a_station_from_a_node(graph, 6)
    result7 = get_all_node_paths_of_a_station_from_a_node(graph, 7)
    result8 = get_all_node_paths_of_a_station_from_a_node(graph, 8)
    expected_result = {6, 7, 8}
    assert set(result6.keys()) == expected_result
    assert set(result7.keys()) == expected_result
    assert set(result8.keys()) == expected_result

    # Test station 3
    result9 = get_all_node_paths_of_a_station_from_a_node(graph, 9)
    result10 = get_all_node_paths_of_a_station_from_a_node(graph, 10)
    expected_result = {9, 10}
    assert set(result9.keys()) == expected_result
    assert set(result10.keys()) == expected_result
