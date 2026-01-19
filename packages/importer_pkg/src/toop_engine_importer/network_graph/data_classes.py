# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The data_classes module contains the data classes for the network graph model."""

from enum import Enum
from typing import List, Literal, Optional, Tuple, TypeAlias, Union

import pandas as pd
import pandera as pa
import pandera.typing as pat
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

NODE_TYPES: TypeAlias = Literal["busbar", "node"]
SWITCH_TYPES: TypeAlias = Literal["DISCONNECTOR", "BREAKER", "LOAD_BREAK_SWITCH", "CB", "DS", "LBS"]

# Note: a PHASE_SHIFTER is a special type of TWO_WINDING_TRANSFORMER with the same voltage level on both sides
BRANCH_TYPES_POWSYBL: TypeAlias = Literal[
    "LINE",
    "TWO_WINDING_TRANSFORMER",
    "PHASE_SHIFTER",
    "TWO_WINDING_TRANSFORMER_WITH_TAP_CHANGER",
    "THREE_WINDINGS_TRANSFORMER",
]
BRANCH_TYPES_PANDAPOWER: TypeAlias = Literal["line", "trafo", "trafo3w", "dcline", "tcsc", "impedance"]

BRANCH_TYPES: TypeAlias = Union[BRANCH_TYPES_POWSYBL, BRANCH_TYPES_PANDAPOWER]

EDGE_ID: TypeAlias = tuple[int, int]

DUPLICATED_EDGE_SUFFIX: TypeAlias = Literal["_FROM", "_TO"]


class SubstationInformation(BaseModel):
    """SubstationInformation contains information about a powsybl substation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """The name of the substation.
    from net.get_substations()["name"]"""

    region: str
    """The region of the substation.
    from net.get_substations(attributes=["country"])"""

    voltage_level_id: str
    """The voltage level of the substation.
    from net.get_voltage_levels()"""

    nominal_v: int
    """The nominal voltage of the substation.
    from net.get_voltage_levels()"""


class BusbarConnectionInfo(BaseModel):
    """BusbarConnectionInfo is a data class that contains information about the connections of a busbar.

    The given connection information is always related to a specific busbar. A connection that is only
    possible when including another busbar in it's path is never included.
    """

    model_config = ConfigDict(extra="forbid")
    """The configuration for the model.
    The extra parameter is set to forbid, to raise an error if an unexpected field is passed to the model."""

    connectable_assets_node_ids: List[Union[int, tuple[int, int]]] = Field(default_factory=list)
    """The node_ids of the connectable assets.
    Connectable is referring to the assets that can be connected and disconnected
    with a circuit breaker switch.
    A connectable asset can be a single node or a tuple of two nodes, if the asset is a branch.
    """

    connectable_assets: List[str] = Field(default_factory=list)
    """The connectable assets grid_model_id.
    Connectable is referring to the assets that can be connected and disconnected"""

    connectable_busbars_node_ids: List[int] = Field(default_factory=list)
    """The node_ids of the connectable busbars.
    Interconnectable is referring to the busbars that can be connected and disconnected
    with a circuit breaker switch."""

    connectable_busbars: List[str] = Field(default_factory=list)
    """The connectable busbars grid_model_id.
    Interconnectable is referring to the busbars that can be connected and disconnected
    with a circuit breaker switch."""

    zero_impedance_connected_assets: List[str] = Field(default_factory=list)
    """The asset grid_model_ids, connected by zero impedance.
    Zero impedance refers to assets that are currently connected to the busbar
    with a path where all switches are closed.
    Only the asset bays are considered."""

    zero_impedance_connected_assets_node_ids: List[Union[int, tuple[int, int]]] = Field(default_factory=list)
    """The asset node_ids, connected by zero impedance.
    Zero impedance refers to assets that are currently connected to the busbar
    with a path where all switches are closed.
    Only the asset bays are considered."""

    zero_impedance_connected_busbars: List[str] = Field(default_factory=list)
    """The busbars grid_model_ids, connected by zero impedance.
    Zero impedance refers to busbars that are currently connected to the busbar
    with a path where all switches are closed.
    Only busbar/cross coupler are considered that directly connect the two busbars."""

    zero_impedance_connected_busbars_node_ids: List[int] = Field(default_factory=list)
    """The busbars node_ids, connected by zero impedance
    Zero impedance refers to busbars that are currently connected to the busbar
    with a path where all switches are closed.
    Only busbar/cross coupler are considered that directly connect the two busbars."""

    node_assets: List[str] = Field(default_factory=list)
    """The node assets grid_model_id.
    Node assets are assets that are connected to a node in the network graph.
    For instance, a transformer or line at the border of the network graph or a generator or load."""

    node_assets_ids: List[Union[int, tuple[int, int]]] = Field(default_factory=list)
    """The node assets node_id.
    Node assets are assets that are connected to a node in the network graph.
    For instance, a transformer or line at the border of the network graph or a generator or load.
    The node asset is for instance the index of the node_assets DataFrame or the node_tuple of the branch."""


class EdgeConnectionInfo(BaseModel):
    """EdgeConnectionInfo is a data class that contains information about an edge in relation to busbars or bays.

    Use this class to store information about the connection of an edge.
    """

    model_config = ConfigDict(extra="forbid")
    """The configuration for the model.
    The extra parameter is set to forbid, to raise an error if an unexpected field is passed to the model."""

    direct_busbar_grid_model_id: str = ""
    """The direct_busbar_grid_model_id is set only if the edge is directly connected to a busbar.
    Leave blank if there is any edge in between the busbar and the edge.
    Fill with the grid_model_id of the busbar if the edge is directly connected to the busbar."""

    bay_id: str = ""
    """The bay_id is set only if the edge is part of an asset/coupler bay.
    Leave blank if the edge is not part of an asset/coupler bay.
    Fill with the grid_model_id of the asset/coupler if the edge is part of an asset/coupler bay."""

    coupler_type: str = ""
    """The coupler_type is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the type of the busbar coupler if the edge is part of a busbar coupler."""

    # Note: this is deprecated and should be removed in the future
    # only used in test cases
    coupler_grid_model_id_list: List[tuple[str, str]] = Field(default_factory=list)
    """The coupler_grid_model_id_list is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the grid_model_id of the busbar coupler if the edge is part of a busbar coupler.
    Reason for tuples: one coupler can connect only two busbars at once.
    Reason for list: one coupler can have multiple from and to busbars.
    Example: [("busbar1", "busbar2"), ("busbar1", "busbar3"), ("busbar2", "busbar3")]
    """

    from_busbar_grid_model_ids: List[str] = Field(default_factory=list)
    """The from_busbar_grid_model_ids is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the grid_model_id of the busbar if the edge is part of a busbar coupler.
    Reason for list: one coupler can have multiple from busbars.
    Example: ["busbar1", "busbar2", "busbar3"]
    """

    to_busbar_grid_model_ids: List[str] = Field(default_factory=list)
    """The to_busbar_grid_model_ids is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the grid_model_id of the busbar if the edge is part of a busbar coupler.
    Reason for list: one coupler can have multiple to busbars.
    Example: ["busbar1", "busbar2", "busbar3"]
    """

    from_coupler_ids: List[str] = Field(default_factory=list)
    """The from_coupler_ids is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the grid_model_id all edges that are part of the busbar coupler.
    Enables the correct identification an coupler bay.
    """

    to_coupler_ids: List[str] = Field(default_factory=list)
    """The to_coupler_ids is set only if the edge is part of a busbar coupler.
    Leave blank if the edge is not part of a busbar coupler.
    Fill with the grid_model_id all edges that are part of the busbar coupler.
    Enables the correct identification an coupler bay.
    """


class WeightValues(Enum):
    """WeightValues is an Enum that contains the weight values for the edges in the NetworkGraphData model.

    The weight values are used to map assets to categories in the network graph.
    high: A values used to cut off shortest paths.
    half: A value confidently ignore the steps, but still use a weight,
        where step or a few max_step are set.
    low: A value to ignore the edge in the shortest path.
    step: A value to count the steps in the shortest path.
    max_step: A value to set the cutoff in the shortest path.
    over_step: A value to set the cutoff in the shortest path.
        Used if this edge should not be part in the max_step cutoff, but may be part of a longer path
    max_coupler: A max value, counting switches in a busbar coupler path.
    """

    high = 100.0
    half = 50.0
    low = 0.0
    step = 1.0
    max_step = 10.0
    over_step = 11.0
    max_coupler = 5.0


class NodeSchema(pa.DataFrameModel):
    """A NodeSchema is a DataFrameModel that represents a node in a network graph."""

    int_id: pat.Index[int] = pa.Field(check_name=False, description="Index of Dataframe")
    """The int_id of the node, used to connect Assets by their node_id.
    This needs to be a unique int_id for the nodes DataFrame."""

    grid_model_id: pat.Series[str]
    """The unique ID of the node in the grid model."""

    foreign_id: Optional[pat.Series[str]] = pa.Field(coerce=False)
    """The unique ID of the node in the foreign model.
    This id is optional is only dragged along. Can for instance used for the DGS model."""

    node_type: pat.Series[str] = pa.Field(isin=NODE_TYPES.__args__)
    """The type of the node NODE_TYPES."""

    voltage_level: pat.Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 800})
    """The voltage level of the node."""

    bus_id: pat.Series[str] = pa.Field(nullable=True)
    """The bus_id of the node.
    The bus_id is the refers to the id in the bus-branch topology."""

    system_operator: pat.Series[str]
    """The system operator of the node.
    Can be used to categorize the node. For instance to identify border lines."""

    substation_id: pat.Series[str] = pa.Field(nullable=True)
    """The unique ID of the substation.
    This id is optional and can be used to identify the substation the node is part of.
    If set as an empty string, the node is not part of a relevant substation.
    If set as None, the node is part of an unknown substation."""

    helper_node: pat.Series[bool] = pa.Field(default=False)
    """A helper node is a node that is used to connect other nodes in the network graph.
    Helper nodes are not part of the network and do not contain any information."""

    in_service: pat.Series[bool] = pa.Field(default=True)
    """The state of the node.
    True: The node is in service. Normally expected to be True or not included in the network graph."""


class AssetSchema(pa.DataFrameModel):
    """An AssetSchema is a DataFrameModel that represents an asset in a network graph.

    This is the parent class for SwitchSchema and BranchSchema and should not be used directly.
    """

    grid_model_id: pat.Series[str]
    """The unique ID of the node in the grid model."""

    foreign_id: Optional[pat.Series[str]] = pa.Field(coerce=False)
    """The unique ID of the node in the foreign model.
    This id is optional is only dragged along. Can for instance used for the DGS model."""

    asset_type: pat.Series[str]
    """The type of the asset."""

    int_id: pat.Index[int] = pa.Field(check_name=False, description="Index of Dataframe")
    """The int_id of the asset.
    This is the index of the dataframe and is expected to be unique for the asset DataFrame
    and of type int."""

    in_service: pat.Series[bool] = pa.Field(default=True)
    """The state of the asset.
    True: The asset is in service. Normally expected to be True or not included in the network graph."""


class BranchSchema(AssetSchema):
    """A BranchSchema is an AssetSchema that represents a branch in a network graph.

    A branch is a connection between two nodes in a network graph.
    It can be for instance a LINE, TWO_WINDING_TRANSFORMER, but not a THREE_WINDING_TRANSFORMER.
    A SwitchSchema is a special type of BranchSchema that represents a switch in a network graph.
    """

    from_node: pat.Series[int]
    """The nodes int_id of the node where the branch starts."""

    to_node: pat.Series[int]
    """The nodes int_id of the node where the branch ends."""

    asset_type: pat.Series[str] = pa.Field(
        isin=[branch_type for branch_type_model in BRANCH_TYPES.__args__ for branch_type in branch_type_model.__args__]
    )
    """The type of the branch."""

    node_tuple: Optional[pat.Series[Tuple[int, int]]] = pa.Field(default=None, nullable=True, description="optional")
    """The node tuple of the branch.
    The node tuple is a tuple of two nodes int_id that are connected by the branch."""


class SwitchSchema(AssetSchema):
    """A SwitchSchema is a BranchSchema that represents a switch in a network graph."""

    from_node: pat.Series[int]
    """The nodes int_id of the node where the branch starts."""

    to_node: pat.Series[int]
    """The nodes int_id of the node where the branch ends."""

    asset_type: pat.Series[str] = pa.Field(isin=SWITCH_TYPES.__args__)
    """The type of the switch SWITCH_TYPES."""

    open: pat.Series[bool]
    """The state of the switch.
    True: The switch is open.
    False: The switch is closed."""

    node_tuple: Optional[pat.Series[Tuple[int, int]]] = pa.Field(default=None, nullable=True, description="optional")
    """The node tuple of the branch.
    The node tuple is a tuple of two nodes int_id that are connected by the branch."""


class NodeAssetSchema(AssetSchema):
    """A NodeAssetSchema is an AssetSchema that represents an asset in a network graph.

    A NodeAssetSchema is an asset that is located at a node in the network graph.
    It can be for instance a transformer or line at the border of the network graph or a generator or load.
    """

    node: pat.Series[int]
    """The nodes_index of the node where the asset is located."""


class HelperBranchSchema(pa.DataFrameModel):
    """A HelperBranchSchema is a BranchSchema that represents a helper branch in a network graph.

    Helper branches no real branches, but are used to connect nodes in the network graph.
    These branches can occur for instance when there is an other abstraction level
    in the network graph e.g. a node for plotting svg.
    Note: The HelperBranch may contain all branches and switches in addition to the helper branches.
    """

    from_node: pat.Series[int]
    """The nodes int_id of the node where the branch starts."""

    to_node: pat.Series[int]
    """The nodes int_id of the node where the branch ends."""

    grid_model_id: pat.Series[str] = pa.Field(isin=[""])
    """A helper branch does not have a grid_model_id.
    It is set to an empty string, creating all edges with a grid_model_id."""


class SwitchableAssetSchema(pa.DataFrameModel):
    """A SwitchableAssetSchema to collect assets for the AssetTopology model."""

    grid_model_id: pat.Series[str]
    """The unique ID of the asset in the grid model."""

    name: pat.Series[str]
    """The name of the asset."""

    type: pat.Series[str]
    """The type of the asset, e.g. LINE, TWO_WINDING_TRANSFORMER, etc."""

    in_service: pat.Series[bool]
    """The in_service information of the asset."""


def get_empty_dataframe_from_df_model(df_model: pa.DataFrameModel) -> pd.DataFrame:
    """Get an empty DataFrame from a DataFrameModel.

    This functions creates an empty DataFrame with the columns and correct dtype of the DataFrameModel.
    It does not initialize Optional columns.

    Parameters
    ----------
    df_model : pa.DataFrameModel
        The DataFrameModel to get an empty DataFrame from.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the columns and correct dtype of the DataFrameModel.
    """
    schema = df_model.to_schema()
    columns_dtypes = {
        column_name: column_type.type
        for column_name, column_type in schema.dtypes.items()
        if schema.columns[column_name].description != "optional"
    }
    return pd.DataFrame(columns=columns_dtypes.keys()).astype(columns_dtypes)


class NetworkGraphData(BaseModel):
    """A NetworkGraphData contains all data to create a nx.Graph from a grid model (e.g. from CGMES).

    It contains nodes, switches, branches and node_assets. This network is used to
    find substations and categorize the elements of the substation into known categories.
    It can be used to create an AssetTopology model and an action_set for the substation.

    """

    nodes: pat.DataFrame[NodeSchema]
    """ A DataFrame containing the nodes."""

    switches: pat.DataFrame[SwitchSchema]
    """ A DataFrame containing the switches."""

    branches: pat.DataFrame[BranchSchema] = Field(
        default_factory=lambda: get_empty_dataframe_from_df_model(df_model=BranchSchema)
    )
    """ A DataFrame containing the branches."""

    node_assets: pat.DataFrame[NodeAssetSchema] = Field(
        default_factory=lambda: get_empty_dataframe_from_df_model(df_model=NodeAssetSchema)
    )
    """A DataFrame containing the node assets"""

    helper_branches: pat.DataFrame[HelperBranchSchema] = Field(
        default_factory=lambda: get_empty_dataframe_from_df_model(df_model=HelperBranchSchema)
    )
    """A DataFrame containing the helper branches."""

    @model_validator(mode="after")
    def validate_network_graph_data(self) -> Self:
        """Validate the NetworkGraphData model."""
        if self.branches.empty and self.node_assets.empty:
            raise ValueError("Branches or node_assets must be provided.")
        # validate the input data
        self.nodes = NodeSchema.validate(self.nodes)
        self.switches = SwitchSchema.validate(self.switches)
        self.branches = BranchSchema.validate(self.branches)
        self.node_assets = NodeAssetSchema.validate(self.node_assets)
        self.helper_branches = HelperBranchSchema.validate(self.helper_branches)
        return self
