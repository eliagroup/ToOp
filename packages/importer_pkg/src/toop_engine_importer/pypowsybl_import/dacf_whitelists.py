# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to process DACF white and black lists.

File: dacf_whitelists.py
Author: Benjamin Petrick
Created: 2024
"""

import pandas as pd
from pypowsybl.network.impl.network import Network


def assign_element_id_to_cb_df(branches_with_elementname: pd.DataFrame, cb_df: pd.DataFrame) -> None:
    """Get the element_id for the elements in the cb_df based on the power network model.

    Parameters
    ----------
    branches_with_elementname : pd.DataFrame
        powsybl branches DataFrame with the columns "elementName", "bus_breaker_bus1_id",
        "bus_breaker_bus2_id", "voltage_level1_id", "voltage_level2_id", "pairing_key"
    cb_df : pd.DataFrame
        DataFrame with the columns "Elementname", "Anfangsknoten", "Endknoten"
        Note: The element_id column is added to the DataFrame in place

    Returns
    -------
    None

    """
    eight_letter_nodes = 8
    seven_letter_nodes = 7
    cb_df["element_id"] = None
    for index, row in cb_df.iterrows():
        # determine the column names for the bus ids, based on the length of the bus id given in the cb_df
        if len(row["Anfangsknoten"]) == eight_letter_nodes:
            column_start_node = "bus_breaker_bus"
        elif len(row["Anfangsknoten"]) == seven_letter_nodes:
            column_start_node = "voltage_level"
        else:
            # should this trigger an error? -> would be an error in the black/white list
            pass

        if len(row["Endknoten"]) == eight_letter_nodes:
            column_end_node = "bus_breaker_bus"
        elif len(row["Endknoten"]) == seven_letter_nodes:
            column_end_node = "voltage_level"
        else:
            # should this trigger an error? -> would be an error in the black/white list
            pass

        # search for the element in the power network model
        # search for the element name in the branches_with_elementname
        # search for "Anfangsknoten"/start node and "Endknoten"/end node in the bus ids left and right + pairing key
        condition_name = branches_with_elementname["elementName"].str.contains(row["Elementname"])
        condition_start_node_left = (branches_with_elementname[f"{column_start_node}1_id"] == row["Anfangsknoten"]) | (
            branches_with_elementname["pairing_key"] == row["Anfangsknoten"]
        )
        condition_start_node_right = (branches_with_elementname[f"{column_start_node}2_id"] == row["Anfangsknoten"]) | (
            branches_with_elementname["pairing_key"] == row["Anfangsknoten"]
        )
        condition_end_node_left = (branches_with_elementname[f"{column_end_node}1_id"] == row["Endknoten"]) | (
            branches_with_elementname["pairing_key"] == row["Endknoten"]
        )
        condition_end_node_right = (branches_with_elementname[f"{column_end_node}2_id"] == row["Endknoten"]) | (
            branches_with_elementname["pairing_key"] == row["Endknoten"]
        )
        condition_key_order1 = condition_start_node_left & condition_end_node_right
        condition_key_order2 = condition_start_node_right & condition_end_node_left

        # apply the conditions to the branches_with_elementname DataFrame
        found_list = branches_with_elementname[
            condition_name & (condition_key_order1 | condition_key_order2)
        ].index.to_list()
        # check if only one element was found -> add to the cb_df
        if len(found_list) == 1:
            cb_df.at[index, "element_id"] = found_list[0]
        # if more than one/no element was found -> check if the id is in the "bus_breaker_bus" column
        # this often happens for TWO_WINDINGS_TRANSFORMER elements, where powsybl creates it's own id for the voltage level
        # this could be done immediately, but it would be a lot slower
        else:
            column_start_node = "bus_breaker_bus"
            column_end_node = "bus_breaker_bus"
            condition_name = branches_with_elementname["elementName"].str.contains(row["Elementname"])
            condition_start_node_left = (
                branches_with_elementname[f"{column_start_node}1_id"].str.contains(row["Anfangsknoten"])
            ) | (branches_with_elementname["pairing_key"] == row["Anfangsknoten"])
            condition_start_node_right = (
                branches_with_elementname[f"{column_start_node}2_id"].str.contains(row["Anfangsknoten"])
            ) | (branches_with_elementname["pairing_key"] == row["Anfangsknoten"])
            condition_end_node_left = (
                branches_with_elementname[f"{column_end_node}1_id"].str.contains(row["Endknoten"])
            ) | (branches_with_elementname["pairing_key"] == row["Endknoten"])
            condition_end_node_right = (
                branches_with_elementname[f"{column_end_node}2_id"].str.contains(row["Endknoten"])
            ) | (branches_with_elementname["pairing_key"] == row["Endknoten"])
            condition_key_order1 = condition_start_node_left & condition_end_node_right
            condition_key_order2 = condition_start_node_right & condition_end_node_left
            found_list = branches_with_elementname[
                condition_name & (condition_key_order1 | condition_key_order2)
            ].index.to_list()
            if len(found_list) == 1:
                cb_df.at[index, "element_id"] = found_list[0]


def apply_white_list_to_operational_limits(network: Network, white_list_df: pd.DataFrame) -> None:
    """Apply the white list to the operational limits of the network.

    Parameters
    ----------
    network : Network
        The network to modify. Note: The network is modified in place.
    white_list_df : pd.DataFrame
        DataFrame with the columns "element_id", "Anfangsknoten", "Endknoten",
        "Auslastungsgrenze_n_0", "Auslastungsgrenze_n_1"

    """
    white_list_df["Auslastungsgrenze_n_0"] = white_list_df["Auslastungsgrenze_n_0"] / 100
    white_list_df["Auslastungsgrenze_n_1"] = white_list_df["Auslastungsgrenze_n_1"] / 100
    # get the current operational limits
    op_lim = network.get_operational_limits().reset_index()
    # filter the operational limits to the elements in the white list
    op_lim = op_lim[op_lim["element_id"].isin(white_list_df["element_id"].to_list())]
    # merge the white list with the operational limits -> add the "Auslastungsgrenze_n_0" and "Auslastungsgrenze_n_1" columns
    op_lim = op_lim.merge(white_list_df, how="left", left_on="element_id", right_on="element_id")
    op_lim.set_index("element_id", inplace=True)
    # remove tie lines, as they can't be set
    op_lim = op_lim[op_lim["element_type"] != "TIE_LINE"]
    # copy the operational limits for N-1 limits
    n1_limits = op_lim[op_lim["Auslastungsgrenze_n_0"] != op_lim["Auslastungsgrenze_n_1"]].copy()
    n1_limits["acceptable_duration"] = 3600
    n1_limits["name"] = "N-1"
    # apply the limits to the operational limits
    n1_limits["value"] = n1_limits["value"] * n1_limits["Auslastungsgrenze_n_1"]
    op_lim["value"] = op_lim["value"] * op_lim["Auslastungsgrenze_n_0"]
    # merge the operational limits with the N-1 limits
    op_lim = pd.concat([op_lim, n1_limits]).sort_index()
    # drop white list columns, to be able to create new operational limits
    op_lim = op_lim.set_index(["side", "type", "group_name"], append=True)[["acceptable_duration", "name", "value"]]
    # drop element_type column -> deprecated
    # create the new operational limits
    network.create_operational_limits(op_lim)
