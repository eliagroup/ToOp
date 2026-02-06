# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Set all weights for the edges DataFrame in the NetworkGraphData model.

Weights are used to map assets to categories in the network graph.
The weights are used to cut off the shortest path.
Multiple weights can be used in combination to categorize the assets in the network graph.

This module sets all weights for the edges in the Schema DataFrames of the NetworkGraphData model.
The weights are used in the default filter strategy.
"""

from .data_classes import (
    BranchSchema,
    HelperBranchSchema,
    SwitchSchema,
    WeightValues,
)


def set_all_weights(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set all weights for the edges DataFrame in the NetworkGraphData model.

    All weights are set in place in the DataFrames.
    This function sets the following weights for the edges:
    - station_weight
    - bay_weight
    - trafo_weight
    - coupler_weight
    - busbar_weight
    - switch_open_weight

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    set_station_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)
    set_bay_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)
    set_trafo_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)
    set_coupler_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)
    set_busbar_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)
    set_switch_open_weight(branches_df=branches_df, switches_df=switches_df, helper_branches_df=helper_branches_df)


def set_station_weight(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set the station weight for the edges DataFrame in the NetworkGraphData model.

    station_weight:
        - set borders of a substation to a high value
        -> find shortest paths within a substation using the cutoff
        - edge cases like PHASE_SHIFTER are part of the substation

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.high.value, WeightValues.step.value, WeightValues.low.value)
    weight_name = "station_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )
    cond = branches_df["asset_type"] == "PHASE_SHIFTER"  # TODO: Phase shifter detection is not yet implemented
    branches_df.loc[cond, weight_name] = WeightValues.step.value


def set_bay_weight(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set the bay weight for the edges DataFrame in the NetworkGraphData model.

    bay_weight:
        - placeholder, to be set in the default filter strategy in the graph
        - a bay is a collection of equipments in a substation connected to a single or multiple busbars

    Important:
    Only set bay_weight to low for switches. Branches are never part of a bay -> set to high.

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.high.value, WeightValues.low.value, WeightValues.low.value)
    weight_name = "bay_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )


def set_trafo_weight(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set the trafo weight for the edges DataFrame in the NetworkGraphData model.

    trafo_weight:
        - set the trafo weight to a high value
        - used to find the shortest path within a voltage level

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.low.value, WeightValues.low.value, WeightValues.low.value)
    weight_name = "trafo_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )
    cond = branches_df["asset_type"] == "TWO_WINDING_TRANSFORMER"
    branches_df.loc[cond, weight_name] = WeightValues.high.value
    cond = branches_df["asset_type"] == "TWO_WINDING_TRANSFORMER_WITH_TAP_CHANGER"
    branches_df.loc[cond, weight_name] = WeightValues.high.value


def set_coupler_weight(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set the coupler weight for the edges DataFrame in the NetworkGraphData model.

    coupler_weight:
        - all but switches to a high value to get a substation
        - a coupler can be either a DISCONNECTOR or a BREAKER
        - is used in combination with bay_weight by default filter strategy to determine connectable busbars

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.high.value, WeightValues.step.value, WeightValues.low.value)
    weight_name = "coupler_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )


def set_busbar_weight(branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema) -> None:
    """Set the busbar weight for the edges DataFrame in the NetworkGraphData model.

    busbar_weight:
        - set busbars to a high value to find all connection paths for branches
        - used to cut off a shortest path at a busbar

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.low.value, WeightValues.low.value, WeightValues.low.value)
    weight_name = "busbar_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )


def set_switch_open_weight(
    branches_df: BranchSchema, switches_df: SwitchSchema, helper_branches_df: HelperBranchSchema
) -> None:
    """Set the open switch weight for the edges DataFrame in the NetworkGraphData model.

    switch_open_weight:
        - set switches to a high value if they are open
        - used to cut off a shortest path at an open switch

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    """
    # weights =  [self.branches, self.switches, self.helper_branches]
    weights = (WeightValues.low.value, WeightValues.low.value, WeightValues.low.value)
    weight_name = "switch_open_weight"
    set_weights_for_edges(
        branches_df=branches_df,
        switches_df=switches_df,
        helper_branches_df=helper_branches_df,
        weights=weights,
        weight_name=weight_name,
    )
    cond = switches_df["open"]
    switches_df.loc[cond, weight_name] = WeightValues.high.value


def set_weights_for_edges(
    branches_df: BranchSchema,
    switches_df: SwitchSchema,
    helper_branches_df: HelperBranchSchema,
    weights: tuple[float, float, float],
    weight_name: str,
) -> None:
    """Set the weights for the edges DataFrame in the NetworkGraphData model.

    Weights can be used to map assets to categories in the network graph.
    For instance set switches and PSTs to 1; lines and transformers to 100
    -> and search in an area <100 to get a substation.
    Use WeightValues to set the weights.

    Parameters
    ----------
    branches_df : BranchSchema
        The BranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    switches_df : SwitchSchema
        The SwitchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    helper_branches_df : HelperBranchSchema
        The HelperBranchSchema DataFrame from the NetworkGraphData model.
        Note: The DataFrame is modified in place.
    weights : tuple[float, float, float]
        Contains the weights for the edges defined in the model
        [self.branches, self.switches, self.helper_branches]
    weight_name : str
        The name of the weight that will be set.
    """
    branch_weight, switch_weight, helper_weight = weights

    branches_df[weight_name] = branch_weight
    switches_df[weight_name] = switch_weight
    helper_branches_df[weight_name] = helper_weight
