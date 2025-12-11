import networkx as nx
from toop_engine_importer.network_graph.data_classes import BusbarConnectionInfo, EdgeConnectionInfo, WeightValues
from toop_engine_importer.network_graph.default_filter_strategy import set_switch_busbar_connection_info
from toop_engine_importer.network_graph.filter_strategy.helper_functions import (
    calculate_asset_bay_for_node_assets,
    get_edge_attr_for_dict_key,
    get_edge_attr_for_dict_list,
)


def test_calculate_asset_bay_for_node_assets():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_edge(1, 2, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value)
    graph.add_edge(2, 3, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)
    graph.add_edge(3, 4, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)
    graph.add_edge(4, 5, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)
    # node 6 = asset
    graph.add_edge(5, 6, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)

    # busbar coupler to busbar2
    graph.add_edge(1, 7, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value)
    graph.add_edge(7, 8, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)
    # node 9 = busbar2
    graph.add_edge(8, 9, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.max_step.value)

    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=6, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [6, 5, 4, 3, 2, 1]}
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=5, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [5, 4, 3, 2, 1]}
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=4, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [4, 3, 2, 1]}
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=3, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [3, 2, 1]}
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=2, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [2, 1]}
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=1, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [1]}

    graph.add_edge(7, 2, bay_weight=WeightValues.low.value, busbar_weight=WeightValues.low.value)
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=6, busbars_helper_nodes=[1, 9]
    )
    # busbar2 now directly connected to asset with the now added path
    assert shortest_path_to_busbar_dict == {1: [6, 5, 4, 3, 2, 1], 9: [6, 5, 4, 3, 2, 7, 8, 9]}

    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=2, busbars_helper_nodes=[1, 9]
    )
    # busbar2 now directly connected to asset with the now added path
    assert shortest_path_to_busbar_dict == {1: [2, 1], 9: [2, 7, 8, 9]}


def test_set_switch_busbar_connection_info_and_get_asset_bay_for_node_assets():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        4,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 6 = asset
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        7,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    # node 9 = busbar2
    graph.add_node(9, node_type="busbar", grid_model_id="bb2", busbar_connection_info=BusbarConnectionInfo())
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    set_switch_busbar_connection_info(graph=graph)
    bay_weight_res = [(1, 2, 0.0), (1, 7, 0.0), (2, 3, 0.0), (3, 4, 0.0), (4, 5, 0.0), (5, 6, 0.0), (7, 8, 0.0), (8, 9, 0.0)]
    busbar_weight_res = [
        (1, 2, 10.0),
        (1, 7, 10.0),
        (2, 3, 0.0),
        (3, 4, 0.0),
        (4, 5, 0.0),
        (5, 6, 0.0),
        (7, 8, 0.0),
        (8, 9, 10.0),
    ]
    assert list(graph.edges(data="busbar_weight")) == busbar_weight_res
    assert list(graph.edges(data="bay_weight")) == bay_weight_res

    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=6, busbars_helper_nodes=[1, 9]
    )
    # do not hop over busbar1 to reach busbar2
    assert shortest_path_to_busbar_dict == {1: [6, 5, 4, 3, 2, 1]}

    graph.add_edge(
        7,
        2,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    set_switch_busbar_connection_info(graph=graph)
    bay_weight_res = [
        (1, 2, 0.0),
        (1, 7, 0.0),
        (2, 3, 0.0),
        (2, 7, 0.0),
        (3, 4, 0.0),
        (4, 5, 0.0),
        (5, 6, 0.0),
        (7, 8, 0.0),
        (8, 9, 0.0),
    ]
    busbar_weight_res = [
        (1, 2, 10.0),
        (1, 7, 10.0),
        (2, 3, 0.0),
        (2, 7, 0.0),
        (3, 4, 0.0),
        (4, 5, 0.0),
        (5, 6, 0.0),
        (7, 8, 0.0),
        (8, 9, 10.0),
    ]
    assert list(graph.edges(data="busbar_weight")) == busbar_weight_res
    assert list(graph.edges(data="bay_weight")) == bay_weight_res
    shortest_path_to_busbar_dict = calculate_asset_bay_for_node_assets(
        graph=graph, asset_node=6, busbars_helper_nodes=[1, 9]
    )
    # busbar2 now directly connected to asset with the now added path
    assert shortest_path_to_busbar_dict == {1: [6, 5, 4, 3, 2, 1], 9: [6, 5, 4, 3, 2, 7, 8, 9]}


def test_get_edge_attr_for_dict_list():
    graph = nx.Graph()
    # Add edges with attributes
    graph.add_edge(1, 2, grid_model_id="edge_1_2", weight=1.0)
    graph.add_edge(2, 3, grid_model_id="edge_2_3", weight=2.0)
    graph.add_edge(3, 4, grid_model_id="edge_3_4", weight=3.0)
    graph.add_edge(4, 5, grid_model_id="edge_4_5", weight=4.0)

    # Input dictionary
    input_dict = {
        "group_1": [(1, 2), (2, 3)],
        "group_2": [(3, 4), (4, 5)],
    }

    # Call the function
    result = get_edge_attr_for_dict_list(graph, input_dict, attribute="grid_model_id")

    # Expected output
    expected_result = {
        "group_1": ["edge_1_2", "edge_2_3"],
        "group_2": ["edge_3_4", "edge_4_5"],
    }

    # Assert the result matches the expected output
    assert result == expected_result

    # Test with a different attribute
    result = get_edge_attr_for_dict_list(graph, input_dict, attribute="weight")
    expected_result = {
        "group_1": [1.0, 2.0],
        "group_2": [3.0, 4.0],
    }

    # Assert the result matches the expected output
    assert result == expected_result


def test_get_edge_attr_for_dict_key():
    graph = nx.Graph()
    # Add edges with attributes
    graph.add_edge(1, 2, grid_model_id="edge_1_2", weight=1.0)
    graph.add_edge(2, 3, grid_model_id="edge_2_3", weight=2.0)
    graph.add_edge(3, 4, grid_model_id="edge_3_4", weight=3.0)
    graph.add_edge(4, 5, grid_model_id="edge_4_5", weight=4.0)

    # Input dictionary
    input_dict = {
        (1, 2): "value_1",
        (2, 3): "value_2",
        (3, 4): "value_3",
        (4, 5): "value_4",
    }

    # Call the function
    result = get_edge_attr_for_dict_key(graph, input_dict, attribute="grid_model_id")

    # Expected output
    expected_result = {
        "edge_1_2": "value_1",
        "edge_2_3": "value_2",
        "edge_3_4": "value_3",
        "edge_4_5": "value_4",
    }

    # Assert the result matches the expected output
    assert result == expected_result

    # Test with a different attribute
    result = get_edge_attr_for_dict_key(graph, input_dict, attribute="weight")
    expected_result = {
        1.0: "value_1",
        2.0: "value_2",
        3.0: "value_3",
        4.0: "value_4",
    }

    # Assert the result matches the expected output
    assert result == expected_result
