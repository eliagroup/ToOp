import networkx as nx
from toop_engine_importer.network_graph.data_classes import EdgeConnectionInfo, WeightValues
from toop_engine_importer.network_graph.filter_strategy.empty_bay import set_empty_bay_weights


def test_empty_bay():
    graph = nx.Graph()
    # node 1 = busbar1
    graph.add_node(1, node_type="busbar", grid_model_id="bb1")
    graph.add_node(10, node_type="busbar", grid_model_id="bb2")

    # bay 1 has been set
    graph.add_edge(
        1,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        2,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        2,
        3,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        3,
        4,
        bay_weight=WeightValues.max_step.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # bay 2 has been set
    graph.add_edge(
        1,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        10,
        5,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        5,
        6,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        6,
        7,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    # busbar coupler to busbar2
    graph.add_edge(
        1,
        8,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        8,
        9,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.low.value,
        edge_connection_info=EdgeConnectionInfo(),
    )
    graph.add_edge(
        9,
        10,
        bay_weight=WeightValues.low.value,
        busbar_weight=WeightValues.max_step.value,
        edge_connection_info=EdgeConnectionInfo(),
    )

    set_empty_bay_weights(graph=graph)
    # empty bay should be set
    assert graph.edges[(1, 5)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(10, 5)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(5, 6)]["bay_weight"] == WeightValues.max_step.value
    assert graph.edges[(6, 7)]["bay_weight"] == WeightValues.max_step.value

    # coupler should not be set
    assert graph.edges[(1, 8)]["bay_weight"] == WeightValues.low.value
    assert graph.edges[(8, 9)]["bay_weight"] == WeightValues.low.value
    assert graph.edges[(9, 10)]["bay_weight"] == WeightValues.low.value
