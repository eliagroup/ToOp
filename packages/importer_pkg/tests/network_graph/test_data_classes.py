# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
import pytest
from pandera.errors import SchemaError
from toop_engine_importer.network_graph.data_classes import (
    BranchSchema,
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SwitchSchema,
    WeightValues,
)


def test_weight_values_enum():
    assert WeightValues.high.value == 100.0
    assert WeightValues.low.value == 0.0
    assert WeightValues.step.value == 1.0
    assert WeightValues.max_step.value == 10.0
    assert WeightValues.over_step.value == 11.0
    assert WeightValues.max_coupler.value == 5.0


def test_node_schema():
    data = {
        "int_id": [1],
        "grid_model_id": ["node_1"],
        "bus_id": ["bus_1"],
        "foreign_id": ["foreign_1"],
        "node_type": ["busbar"],
        "voltage_level": [110],
        "system_operator": ["operator_1"],
        "substation_id": ["substation_1"],
        "helper_node": [False],
        "in_service": [True],
    }
    df = pd.DataFrame(data)
    try:
        NodeSchema.validate(df)
    except SchemaError:
        pytest.fail("NodeSchema validation failed")


def test_branch_schema():
    data = {
        "int_id": [1],
        "grid_model_id": ["branch_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["LINE"],
        "from_node": [1],
        "to_node": [2],
        "in_service": [True],
        "node_tuple": [(1, 2)],
    }
    df = BranchSchema(pd.DataFrame(data))

    BranchSchema.validate(df)
    df.drop(columns=["node_tuple"], inplace=True)
    BranchSchema.validate(df)


def test_switch_schema():
    data = {
        "int_id": [1],
        "grid_model_id": ["switch_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["DISCONNECTOR"],
        "from_node": [1],
        "to_node": [2],
        "open": [True],
        "in_service": [True],
        "node_tuple": [(1, 2)],
    }
    df = pd.DataFrame(data)
    SwitchSchema.validate(df)
    df.drop(columns=["node_tuple"], inplace=True)
    SwitchSchema.validate(df)


def test_node_asset_schema():
    data = {
        "int_id": [1],
        "grid_model_id": ["asset_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["type_1"],
        "node": [1],
        "in_service": [True],
    }
    df = pd.DataFrame(data)

    NodeAssetSchema.validate(df)


def test_helper_branch_schema():
    data = {"from_node": [1], "to_node": [2], "grid_model_id": [""]}
    df = pd.DataFrame(data)
    try:
        HelperBranchSchema.validate(df)
    except SchemaError:
        pytest.fail("HelperBranchSchema validation failed")


def test_network_graph_initialization_no_data(get_graph_input_dicts):
    nodes_dict, switches_dict, _ = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    nodes_df["in_service"] = True
    switches_df = pd.DataFrame(switches_dict)
    switches_df["in_service"] = True
    with pytest.raises(ValueError):
        NetworkGraphData(nodes=nodes_df)
    with pytest.raises(ValueError, match="Branches or node_assets must be provided."):
        NetworkGraphData(nodes=nodes_df, switches=switches_df)


def test_network_graph_initialization_with_all_data(get_graph_input_dicts):
    nodes_dict, switches_dict, node_assets_dict = get_graph_input_dicts
    nodes_df = pd.DataFrame(nodes_dict)
    nodes_df["in_service"] = True
    switches_df = pd.DataFrame(switches_dict)
    switches_df["in_service"] = True
    node_assets_df = pd.DataFrame(node_assets_dict)
    node_assets_df["in_service"] = True

    network_graph = NetworkGraphData(nodes=nodes_df, switches=switches_df, node_assets=node_assets_df)

    assert network_graph.nodes.equals(nodes_df)
    assert network_graph.switches.equals(switches_df)
    assert network_graph.node_assets.equals(node_assets_df)


def test_branch_schema():
    data = {
        "int_id": [1],
        "grid_model_id": ["branch_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["LINE"],
        "from_node": [1],
        "to_node": [2],
        "in_service": [True],
        "node_tuple": [(1, 2)],
    }
    df = BranchSchema(pd.DataFrame(data))

    BranchSchema.validate(df)
    df.drop(columns=["node_tuple"], inplace=True)
    BranchSchema.validate(df)


def test_branch_schema_invalid_node_tuple():
    data = {
        "int_id": [1],
        "grid_model_id": ["branch_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["LINE"],
        "from_node": [1],
        "to_node": [2],
        "in_service": [True],
        "node_tuple": [(1, "2")],  # Invalid node_tuple
    }
    df = pd.DataFrame(data)
    with pytest.raises(SchemaError):
        BranchSchema.validate(df)


def test_branch_schema_missing_columns():
    data = {
        "int_id": [1],
        "grid_model_id": ["branch_1"],
        "foreign_id": ["foreign_1"],
        "asset_type": ["LINE"],
        "from_node": [1],
        "to_node": [2],
        "in_service": [True],
    }
    df = pd.DataFrame(data)
    BranchSchema.validate(df)
