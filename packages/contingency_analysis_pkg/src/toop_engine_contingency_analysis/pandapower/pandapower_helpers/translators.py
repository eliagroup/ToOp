# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Translation utilities for converting N-1 definitions into pandapower-ready monitored elements and contingencies."""

from typing import Any, get_args

import numpy as np
import pandas as pd
import pandera.typing as pat
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.extractors import (
    extract_contingencies_with_cgmes_id,
    extract_contingencies_with_unique_pandapower_id,
    extract_monitored_elements_with_cgmes_id,
    extract_monitored_elements_with_unique_pandapower_id,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
    PandapowerNMinus1Definition,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.va_diff_info import get_va_diff_info
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id_from_index,
)
from toop_engine_interfaces.nminus1_definition import (
    ELEMENT_ID_TYPES,
    PANDAPOWER_SUPPORTED_ID_TYPES,
    Contingency,
    GridElement,
    Nminus1Definition,
)


def match_node_to_next_switch_type(
    node_ids: np.ndarray[tuple[Any, ...], np.dtype[np.integer]],
    switches_df: pd.DataFrame,
    actual_buses: np.ndarray[tuple[Any, ...], np.dtype[np.integer]],
    switch_type: str,
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES,
    max_jumps: int = 4,
) -> pd.DataFrame:
    """Find the next switch of a given type for each node.

    Stops at nodes that are actual busbars.
    Finds switches that are connected by other swiches of another type.
    Only considers switches that are closed at the point of calling.

    Parameters
    ----------
    node_ids : np.ndarray
        The node ids to find the next switch for.
    switches_df : pd.DataFrame
        The switches dataframe from pandapower.
    actual_buses : np.ndarray
        The node ids of the actual busbars in the network.
        All others are helper nodes for modelling purposes.
    switch_type : str
        The type of switch to find. E.g. "CB" for circuit breaker.
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.
    max_jumps : int
        The maximum number of jumps to make to find the next switch.

    Returns
    -------
    pd.DataFrame
        A dataframe with the original node id, the switch id, switch name and unique id.
    """
    switches_to_check = switches_df.reset_index(names="original_index").query("closed")
    if id_type == "unique_pandapower":
        switches_to_check["unique_id"] = get_globally_unique_id_from_index(switches_to_check.original_index, "switch")
    elif id_type == "cgmes":
        switches_to_check["unique_id"] = switches_to_check["origin_id"]
    else:
        raise ValueError(f"Unsupported ID Type: {id_type}")
    bidirectional_switches = np.concatenate(
        [
            switches_to_check[["original_index", "bus", "element", "type", "name", "unique_id"]].values,
            switches_to_check[["original_index", "element", "bus", "type", "name", "unique_id"]].values,
        ],
        axis=0,
    )
    all_switches_df = pd.DataFrame(
        bidirectional_switches, columns=["switch_idx", "bus", "element", "type", "name", "unique_id"]
    ).drop_duplicates()
    selected_switches = all_switches_df[all_switches_df.type == switch_type]
    other_switches = all_switches_df[all_switches_df.type != switch_type]

    node_df = pd.DataFrame.from_dict({"original_node": node_ids, "merge_node": node_ids})
    switch_found = []
    nodes_to_match = node_df[node_df.original_node.isin(all_switches_df.bus)]

    all_switches_df = all_switches_df[~all_switches_df.bus.isin(actual_buses)]
    for _ in range(max_jumps):
        merged_cbs = nodes_to_match.merge(selected_switches, left_on="merge_node", how="left", right_on="bus")
        switch_found.append(merged_cbs.dropna())
        merged_non_cbs = nodes_to_match.merge(other_switches, left_on="merge_node", how="left", right_on="bus")
        other_switches = other_switches[~other_switches.switch_idx.isin(merged_non_cbs.switch_idx)]
        nodes_to_match = merged_non_cbs.dropna()[["original_node", "element"]].rename(columns={"element": "merge_node"})
        nodes_to_match = nodes_to_match[~nodes_to_match.merge_node.isin(actual_buses)]
        if nodes_to_match.empty:
            break
    matched = pd.concat(switch_found)[["original_node", "switch_idx", "name", "unique_id"]]
    return matched


def get_node_to_switch_map(net: pandapowerNet, id_type: PANDAPOWER_SUPPORTED_ID_TYPES) -> dict[int, dict[str, str]]:
    """Get a mapping from nodes at branches and their closest Circuit breaker switches.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    id_type: PANDAPOWER_SUPPORTED_ID_TYPES
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.

    Returns
    -------
    node_to_switch_map: dict[int, list[int]]
        A mapping from nodes at branches and their closest Circuit breaker switches.
    """
    considered_nodes = np.concatenate(
        [
            net.line.from_bus.values,
            net.line.to_bus.values,
            net.trafo.hv_bus.values,
            net.trafo.lv_bus.values,
            net.trafo3w.hv_bus.values,
            net.trafo3w.mv_bus.values,
            net.trafo3w.lv_bus.values,
        ]
    )
    actual_busbars = net.bus[net.bus.type == "b"].index.values
    matched = match_node_to_next_switch_type(
        considered_nodes, net.switch, actual_busbars, switch_type="CB", id_type=id_type, max_jumps=4
    )
    grouped_by_bus = matched.groupby("original_node").agg(list)[["unique_id", "name"]].to_dict(orient="index")
    node_to_switch_map = {
        outage: dict(zip(info["unique_id"], info["name"], strict=True)) for outage, info in grouped_by_bus.items()
    }
    return node_to_switch_map


def translate_contingencies(
    net: pandapowerNet, contingencies: list[Contingency], id_type: ELEMENT_ID_TYPES = "unique_pandapower"
) -> tuple[list[PandapowerContingency], list[Contingency], list[str]]:
    """Translate the contingencies to a format that can be used in Pandapower.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    contingencies : list[Contingency]
        The list of contingencies to translate.
    id_type: ELEMENT_ID_TYPES = "unique_pandapower"
        The type of ids to use for the contingencies. Currently only "unique_pandapower" and "cgmes" is supported.

    Returns
    -------
    pp_contingencies: list[PandapowerContingency]
        A list translated Contingency to be used in Pandapower.
    missing_contingencies: list[Contingency]
        A list of contingencies that were not found in the network.
    duplicated_ids: list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    if id_type == "unique_pandapower":
        pp_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_unique_pandapower_id(
            net, contingencies
        )
    elif id_type == "cgmes":
        pp_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_cgmes_id(net, contingencies)
    else:
        raise ValueError(f"Unsupported id_type: {id_type}. Supported id_types are: ['unique_pandapower', 'cgmes']")
    node_to_switch_map = get_node_to_switch_map(net, id_type=id_type)
    for contingency in pp_contingencies:
        contingency.va_diff_info = get_va_diff_info(contingency, net, node_to_switch_map)
    return pp_contingencies, missing_contingencies, duplicated_ids


def translate_monitored_elements(
    net: pandapowerNet, monitored_elements: list[GridElement], id_type: PANDAPOWER_SUPPORTED_ID_TYPES = "unique_pandapower"
) -> tuple[pat.DataFrame[PandapowerMonitoredElementSchema], list[GridElement], list[str]]:
    """Translate the monitored elements to a format that can be used in Pandapower.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    monitored_elements : list[GridElement]
        The list of monitored elements to translate.
    id_type: ELEMENT_ID_TYPES = "unique_pandapower"
        The type of ids to use for the monitored elements. Currently only "unique_pandapower" and "cgmes" is supported.
        TODO: Add support for other id types.

    Returns
    -------
    pat.DataFrame[PandapowerMonitoredElementSchema]
        A pandas DataFrame containing the monitored elements with their globally unique ids, table, table_id, kind and name.
    list[GridElement]
        A list of monitored elements that were not found in the network.
    list[str]
        A list of ids that were not unique in the grid. This is only relevant for cgmes ids.
    """
    if id_type == "unique_pandapower":
        pandapower_monitored_elements, missing_elements, duplicated_ids = (
            extract_monitored_elements_with_unique_pandapower_id(net, monitored_elements)
        )
    elif id_type == "cgmes":
        pandapower_monitored_elements, missing_elements, duplicated_ids = extract_monitored_elements_with_cgmes_id(
            net, monitored_elements
        )
    else:
        raise ValueError(f"Unsupported id_type: {id_type}")
    return pandapower_monitored_elements, missing_elements, duplicated_ids


def translate_nminus1_for_pandapower(
    n_minus_1_definition: Nminus1Definition, net: pandapowerNet
) -> PandapowerNMinus1Definition:
    """Translate the N-1 definition to a format that can be used in Powsybl.

    Parameters
    ----------
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to translate.
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc

    Returns
    -------
    PowsyblNMinus1Definition
        The translated N-1 definition that can be used in Powsybl.
    """
    id_type = n_minus_1_definition.id_type or "unique_pandapower"
    # If no id_type is specified, we assume pandapower's unique ids
    if id_type not in (supported_id_types := get_args(PANDAPOWER_SUPPORTED_ID_TYPES)):
        # If the id_type is not supported, we raise an error
        raise ValueError(f"Unsupported id_type: {id_type}. Supported id_types are: {supported_id_types}")
    pandapower_monitored_elements, missing_elements, duplicated_monitored_ids = translate_monitored_elements(
        net, n_minus_1_definition.monitored_elements, id_type=id_type
    )
    contingencies, missing_contingencies, duplicated_outaged_element_ids = translate_contingencies(
        net, n_minus_1_definition.contingencies, id_type=id_type
    )

    return PandapowerNMinus1Definition(
        monitored_elements=pandapower_monitored_elements,
        missing_elements=missing_elements,
        contingencies=contingencies,
        missing_contingencies=missing_contingencies,
        duplicated_grid_elements=duplicated_monitored_ids + duplicated_outaged_element_ids,
    )
