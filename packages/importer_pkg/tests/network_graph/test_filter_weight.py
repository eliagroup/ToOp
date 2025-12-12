import pandas as pd
import pytest
from toop_engine_importer.network_graph.data_classes import (
    BranchSchema,
    HelperBranchSchema,
    SwitchSchema,
)
from toop_engine_importer.network_graph.filter_weights import (
    set_all_weights,
    set_bay_weight,
    set_busbar_weight,
    set_coupler_weight,
    set_station_weight,
    set_switch_open_weight,
    set_trafo_weight,
    set_weights_for_edges,
)


def test_set_all_weights_empty():
    switches = pd.DataFrame(columns=list(SwitchSchema.to_schema().columns.keys())).astype(
        {
            column_name: column_type.type
            for column_name, column_type in SwitchSchema.to_schema().dtypes.items()
            if SwitchSchema.to_schema().columns[column_name].description != "optional"
        }
    )
    branches = pd.DataFrame(columns=list(BranchSchema.to_schema().columns.keys())).astype(
        {
            column_name: column_type.type
            for column_name, column_type in BranchSchema.to_schema().dtypes.items()
            if SwitchSchema.to_schema().columns[column_name].description != "optional"
        }
    )
    helper_branches = pd.DataFrame(columns=list(HelperBranchSchema.to_schema().columns.keys())).astype(
        {
            column_name: column_type.type
            for column_name, column_type in HelperBranchSchema.to_schema().dtypes.items()
            if SwitchSchema.to_schema().columns[column_name].description != "optional"
        }
    )
    edges_col = ["station_weight", "bay_weight", "trafo_weight", "coupler_weight", "busbar_weight", "switch_open_weight"]
    assert not switches.columns.isin(edges_col).all()
    assert not branches.columns.isin(edges_col).all()
    assert not helper_branches.columns.isin(edges_col).all()
    set_all_weights(branches_df=branches, switches_df=switches, helper_branches_df=helper_branches)
    assert all([col in switches.columns for col in edges_col])
    assert all([col in branches.columns for col in edges_col])
    assert all([col in helper_branches.columns for col in edges_col])


def test_set_weights_for_edges(get_graph_input_dicts):
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    switches_df = pd.DataFrame(switches_dict)
    branches_df = pd.DataFrame(switches_dict)
    helper_branches_df = pd.DataFrame(switches_dict)

    weight_name = "test_weight"
    weights = [1.0, 2.0, 3.0]
    assert weight_name not in switches_df.columns
    assert weight_name not in branches_df.columns
    assert weight_name not in helper_branches_df.columns
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )
    assert weight_name in switches_df.columns
    assert weight_name in branches_df.columns
    assert weight_name in helper_branches_df.columns
    assert branches_df[weight_name].unique()[0] == weights[0]
    assert switches_df[weight_name].unique()[0] == weights[1]
    assert helper_branches_df[weight_name].unique()[0] == weights[2]

    with pytest.raises(ValueError):
        weights = [1.0, 2.0, 3.0, 4.0]
        set_weights_for_edges(
            branches_df=branches_df,
            switches_df=switches_df,
            helper_branches_df=helper_branches_df,
            weights=weights,
            weight_name=weight_name,
        )
    with pytest.raises(ValueError):
        weights = [1.0, 2.0]
        set_weights_for_edges(
            branches_df=branches_df,
            switches_df=switches_df,
            helper_branches_df=helper_branches_df,
            weights=weights,
            weight_name=weight_name,
        )


def test_set_bay_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_bay_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "bay_weight" in network_graph_data.branches.columns
    assert "bay_weight" in network_graph_data.switches.columns
    assert "bay_weight" in network_graph_data.helper_branches.columns


def test_set_busbar_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_busbar_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "busbar_weight" in network_graph_data.branches.columns
    assert "busbar_weight" in network_graph_data.switches.columns
    assert "busbar_weight" in network_graph_data.helper_branches.columns


def test_set_coupler_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_coupler_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "coupler_weight" in network_graph_data.branches.columns
    assert "coupler_weight" in network_graph_data.switches.columns
    assert "coupler_weight" in network_graph_data.helper_branches.columns


def test_set_switch_open_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_switch_open_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "switch_open_weight" in network_graph_data.branches.columns
    assert "switch_open_weight" in network_graph_data.switches.columns
    assert "switch_open_weight" in network_graph_data.helper_branches.columns


def test_set_trafo_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_trafo_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "trafo_weight" in network_graph_data.branches.columns
    assert "trafo_weight" in network_graph_data.switches.columns
    assert "trafo_weight" in network_graph_data.helper_branches.columns


def test_set_station_weight(network_graph_data_test2_helper_branches):
    network_graph_data = network_graph_data_test2_helper_branches
    set_station_weight(
        branches_df=network_graph_data.branches,
        switches_df=network_graph_data.switches,
        helper_branches_df=network_graph_data.helper_branches,
    )
    assert "station_weight" in network_graph_data.branches.columns
    assert "station_weight" in network_graph_data.switches.columns
    assert "station_weight" in network_graph_data.helper_branches.columns
