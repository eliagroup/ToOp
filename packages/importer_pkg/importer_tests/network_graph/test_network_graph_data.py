from unittest import mock

import pandas as pd
from toop_engine_importer.network_graph.network_graph_data import add_node_tuple_column


def test_add_node_tuple():
    network_graph_data = mock.Mock()
    network_graph_data.switches = pd.DataFrame({"from_node": [1, 4], "to_node": [3, 2]})
    network_graph_data.branches = pd.DataFrame({"from_node": [5, 8], "to_node": [7, 6]})
    network_graph_data.helper_branches = pd.DataFrame({"from_node": [9, 12], "to_node": [11, 10]})
    add_node_tuple_column(network_graph_data)

    assert "node_tuple" in network_graph_data.switches.columns
    assert "node_tuple" in network_graph_data.branches.columns
    assert "node_tuple" in network_graph_data.helper_branches.columns

    assert network_graph_data.switches["node_tuple"].iloc[0] == (1, 3)
    # important: the order does matter, the set will be (2, 4) not (4, 2)
    assert network_graph_data.switches["node_tuple"].iloc[1] == (2, 4)

    assert network_graph_data.branches["node_tuple"].iloc[0] == (5, 7)
    assert network_graph_data.branches["node_tuple"].iloc[1] == (6, 8)

    assert network_graph_data.helper_branches["node_tuple"].iloc[0] == (9, 11)
    assert network_graph_data.helper_branches["node_tuple"].iloc[1] == (10, 12)
