# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module containing functions to translate a RealizedTopology json file to a UCTE file.

DeprecationWarning:
This module is deprecated and will be removed in the future,
due to deprecation of RealizedTopology. Use Topology (AssetTopology) instead.

File: ucte_exporter.py
Author:  Benjamin Petrick
Created: 2024

Note: this module ignores generator and load reassignments.
"""

import json
import re
from pathlib import Path
from typing import Any

import logbook
import pandas as pd
from toop_engine_importer.ucte_toolset.ucte_io import make_ucte, parse_ucte

logger = logbook.Logger(__name__)


def process_file(
    input_uct: Path,
    input_json: Path,
    output_uct: Path,
    topo_id: int = 0,
    reassign_branches: bool = True,
    reassign_injections: bool = False,
) -> dict:
    """Process a UCTE file and a preprocessed json file to include split substations.

    Parameters
    ----------
    input_uct : Path
        The path to the input UCTE file, the original UCTE
    input_json : Path
        The preprocessed json holding the split substations and information, use the loadflowsolver's
        preprocessing notebook to generate this
    output_uct : Path
        The path to the output UCTE file, will be overwritten
    topo_id : int
        The id of the topology to use in the json file
    reassign_branches : bool
        If True, reassign branches to the new busbars
    reassign_injections : bool
        If True, reassign injections to the new busbars
        Note: not implemented yet

    Returns
    -------
    list[str]
        The codes of the fake busbars that were inserted
    """
    if reassign_injections:
        raise NotImplementedError("Reassigning injections is not implemented yet.")

    with open(input_uct, "r") as f:
        ucte_contents = f.read()
    with open(input_json, "r") as f:
        json_contents = json.load(f)
    topo = json_contents[topo_id]["topology"]["substation_info"]
    split_subs = [s for s in topo if is_split(s)]

    preamble, nodes, lines, trafos, trafo_reg, postamble = parse_ucte(ucte_contents)

    statistics = {"changed_ids": {}}  # type: dict

    for topo_element in split_subs:
        statistics["changed_ids"][topo_element["id"]] = {}
        statistics["changed_ids"][topo_element["id"]]["branches"] = {}
        statistics["changed_ids"][topo_element["id"]]["injections"] = {}

        code = topo_element["id"][0:7]
        switches = find_switches(lines, code)
        switches_dict = group_switches(switches)
        if len(switches_dict) == 0:
            raise ValueError(f"No switches found for substation {code}")
        switch_group_id = get_switch_group_number(switches_dict)
        bus_a, bus_b = get_bus_a_b(switches_dict[switch_group_id])

        if (topo_element["branch_assignments"] is not None) and reassign_branches:
            statistics["changed_ids"][topo_element["id"]]["branches"] = apply_branch_assignment(
                topo_element,
                lines,
                trafos,
                trafo_reg,
                bus_a,
                bus_b,
                statistics["changed_ids"],
            )

        statistics["changed_ids"][topo_element["id"]]["switches"] = open_switches(lines, switches_dict[switch_group_id])

    new_ucte = make_ucte(preamble, nodes, lines, trafos, trafo_reg, postamble)

    with open(output_uct, "w") as f:
        f.write(new_ucte)

    validate_ucte_changes(ucte_contents, new_ucte)

    return statistics


def validate_ucte_changes(ucte_contents: str, ucte_contents_out: str) -> None:
    """Validate the changes made to the UCTE file.

    Parameters
    ----------
    ucte_contents : str
        The original UCTE file
    ucte_contents_out : str
        The modified UCTE file

    Raises
    ------
    RuntimeError
        If the changes are not as expected

    """
    if len(ucte_contents) != len(ucte_contents_out):
        raise RuntimeError(
            f"File sizes are different -> error in applying topology. "
            f"Length of original UCTE: {len(ucte_contents)}, Length of modified UCTE: {len(ucte_contents_out)}. "
            + "Length should not change, due to renaming of branches and opening switches."
        )


def is_split(sub_info: dict) -> bool:
    """Check if the substation was split."""
    return any(b["on_bus_b"] for b in sub_info["branch_assignments"])


def get_switch_group_number(grouped_switches: dict) -> str:
    """Decide which switch group to open. Selects the group with the fewest unique busbars.

    Parameters
    ----------
    grouped_switches : dict
        The grouped switches data-frame from group_switches(). Each key contains all switches necessary to isolate a busbar

    Returns
    -------
    reassignment_key : str
        The dict key of the switch group to open

    """
    reassignment_key = ""
    ideal_candidate = False
    n_unique_busbars = 2
    n_switchtes = 1

    for sw, sw_values in grouped_switches.items():
        unique_busbars = get_unique_busbars(sw_values)
        if len(unique_busbars) == n_unique_busbars and len(sw_values) == n_switchtes:
            # ideal candidate for branch assignment
            reassignment_key = sw
            ideal_candidate = True
            break
        if len(unique_busbars) == n_unique_busbars:
            # still only one busbar to reassign, but with multiple switches -> continue searching
            reassignment_key = sw
            ideal_candidate = True
        elif len(unique_busbars) > n_unique_busbars and reassignment_key == "":
            # not ideal candidate for branch assignment -> continue searching
            reassignment_key = sw
            # candidates with only one unique busbar are left out -> doesn't make sense

    if reassignment_key == "":
        raise ValueError(f"No switch group found. Using the first switch group: {grouped_switches}")
    if not ideal_candidate:
        logger.warning(
            f"Switch group {reassignment_key} has more than 2 busbars. "
            + f"Validate if behavior is as expected: {get_unique_busbars(grouped_switches[reassignment_key])}"
        )
    return reassignment_key


def find_switches(lines: pd.DataFrame, node_id: list[str]) -> pd.DataFrame:
    """Find the all switches on the input node.

    Parameters
    ----------
    lines : pd.DataFrame
        The lines data-frame from UCTE file
    node_id : list[str]
        The node id to search for switches. Note: expects switches to be closed -> status = 2 (closed)
        Note: this is the first 7 characters of the id (Node), not the full id of busbar that has one additional character

    Returns
    -------
    switches : pd.DataFrame
        All switches found on the input node, with status = 2 (closed)


    """
    switches = lines[
        ((lines["from"].str.startswith(node_id)) & (lines["to"].str.startswith(node_id))) & (lines["status"] == "2")
    ]
    return switches


def get_unique_busbars(switches: pd.DataFrame) -> list[str]:
    """Get the unique busbars from the switches.

    Parameters
    ----------
    switches : pd.DataFrame
        The switches data-frame from find_switches()

    Returns
    -------
    unique_busbars : list[str]
        A list of unique busbars found in the switches

    """
    unique_busbars = []
    for _, row in switches.iterrows():
        if row["from"] not in unique_busbars:
            unique_busbars.append(row["from"])
        if row["to"] not in unique_busbars:
            unique_busbars.append(row["to"])

    return unique_busbars


def group_switches(switches: pd.DataFrame) -> dict:
    """Group the switches by the busbar id.

    There can be multiple switches between the same busbars.
    This function groups them together. One list element is one group of switches to isolate the bus completely.

    Parameters
    ----------
    switches : pd.DataFrame
        All switches data-frame from UCTE file from a specific node

    Returns
    -------
    switches_sort : dict
        A dict of switches data-frames sorted by the busbar id
    """
    unique_switch_ids = get_unique_busbars(switches)

    # sort switches by busbar id of switch
    switches_sort = {}
    for switch_id in unique_switch_ids:
        switches_sort[switch_id] = switches[(switches["from"] == switch_id) | (switches["to"] == switch_id)]
    return switches_sort


def get_bus_a_b(switches: pd.DataFrame) -> tuple[str, str]:
    """Get the bus A and B from the switches.

    Parameters
    ----------
    switches : pd.DataFrame
        The switches data-frame from find_switches()

    Returns
    -------
    bus_a : str
        The bus A of the substation
    bus_b : str
        The bus B of the substation

    Raises
    ------
    ValueError
        If the switches contain switches from multiple nodes.

    """
    from_values = switches["from"].values
    to_values = switches["to"].values

    if all(x == from_values[0] for x in from_values) and all(x == to_values[0] for x in to_values):
        bus_a = from_values[0]
        bus_b = to_values[0]
    else:
        raise ValueError(
            f"Switches DataFrame contains switches from multiple nodes. Node 'from' {from_values}, Node 'to' {to_values}"
        )

    return bus_a, bus_b


def open_switches(lines: pd.DataFrame, switches: pd.DataFrame) -> dict:
    """Open switches in the UCTE data by changing the status code value.

    Parameters
    ----------
    lines : pd.DataFrame
        The lines data-frame from UCTE file
        Note: modifies the data-frames in place
    switches : pd.DataFrame
        The switches data-frame from find_switches()
        Note: modifies the data-frames in place


    Returns
    -------
    stats : dict
        A dictionary containing the bus A and B of the substation,
        the number of switches and the from and to busbar of the switches


    """
    switch_idx = switches.index
    bus_a, bus_b = get_bus_a_b(switches)
    lines.loc[switch_idx, "status"] = "7"  # 7 -> open switch
    stats = {
        "bus_a": bus_a,
        "bus_b": bus_b,
        "from": switches.iloc[0]["from"],
        "to": switches.iloc[0]["to"],
        "order_of_switches": list(switches["order"].values),
    }
    return stats


def handle_order_of_branch(branch_df: pd.DataFrame, replacement_id: str) -> str:
    """Get a unique order number of the branch for new id.

    Parameters
    ----------
    branch_df : pd.DataFrame
        The branch data-frame from UCTE file. e.g. lines or trafos or trafo_reg
    replacement_id : str
        The replacement ID of the branch

    Returns
    -------
    order : str
        The order of the branch

    """
    from_node, to_node, order = replacement_id.split(" ")
    order_int = int(order)
    loop_count = 0
    max_loop = 100
    while len(find_branch_index(branch_df, f"{from_node} {to_node} {order_int}")) > 0:
        order_int += 1

        loop_count += 1
        if loop_count > max_loop:
            raise ValueError("handle_order_of_branch() Loop count exceeded 100")

    updated_replacement_id = f"{from_node} {to_node} {order_int}"
    return updated_replacement_id


def find_branch_index(branch_df: pd.DataFrame, id: str) -> pd.DataFrame:
    """Find the index of the branch in the UCTE data.

    Parameters
    ----------
    branch_df : pd.DataFrame
        The branch data-frame from UCTE file. e.g. lines or trafos or trafo_reg
    id : str
        The ID of the branch

    Returns
    -------
    branch_df_idx : pd.DataFrame
        The index of the branch in the data-frame

    """
    from_node = id[0:8]
    to_node = id[9:17]
    order = id[18:19]
    branch_df_idx = branch_df[
        (branch_df["from"] == from_node) & (branch_df["to"] == to_node) & (branch_df["order"] == order)
    ].index
    return branch_df_idx


def execute_branch_assignment(
    branch_df: pd.DataFrame,
    id: str,
    replacement_id: str,
    statistics_all_stations: dict[str, dict[str, dict[str, list[str]]]],
) -> bool:
    """Apply branch assignment to the UCTE data.

    Parameters
    ----------
    branch_df : pd.DataFrame
        The branch data-frame from UCTE file. e.g. lines or trafos or trafo_reg
        Note: modifies the data-frames in place
    id : str
        The original ID of the branch
    replacement_id : str
        The replacement ID of the branch
    statistics_all_stations : dict
        The statistics dictionary from process_file()
        expects as input the statistics["changed_ids"] dictionary

    Returns
    -------
    replaced : bool
        True if the branch was replaced, False if the branch was not found in the data-frame

    """
    from_node_replacement = replacement_id[0:8]
    to_node_replacement = replacement_id[9:17]
    order_replacement = replacement_id[18:19]
    branch_df_idx = find_branch_index(branch_df, id)
    if len(branch_df_idx) == 0:
        # check if the ID has been replaced in the statistics
        id = update_id_if_has_been_replaced(id, statistics_all_stations)
        branch_df_idx = find_branch_index(branch_df, id)

    if len(branch_df_idx) > 0:
        replaced = True
        branch_df.loc[branch_df_idx, "from"] = from_node_replacement
        branch_df.loc[branch_df_idx, "to"] = to_node_replacement
        branch_df.loc[branch_df_idx, "order"] = order_replacement
    else:
        replaced = False
    return replaced


def get_replacement_id(element: dict, code: str, bus_a: str, bus_b: str) -> str:
    """Get the replacement ID of the branch.

    Decides if the element is on bus A or B and replaces the ID accordingly.
    Bus A is a logical bus and can be electrically connected to other buses e.g. bus C.
    Bus B is the new Bus that will be split off.

    Parameters
    ----------
    element : dict
        The element to replace imported from the json
    code : str
        The code of the substation
    bus_a : str
        The bus A of the substation
    bus_b : str
        The bus B of the substation

    Returns
    -------
    replacement_id : str
        The replacement ID of the branch
    """
    id = element["id"]

    if element["on_bus_b"]:
        # element on bus B -> replace with bus B
        replacement_id = re.sub(rf"{re.escape(code)}\d?", bus_b, id)
    elif bus_b in id:
        # element on bus A -> replace with bus A
        replacement_id = re.sub(rf"{re.escape(code)}\d?", bus_a, id)
    else:
        # element on the logical bus A but is not on the new electrically isolated bus B
        # -> keep ID, as it is not affected by the split
        # e.g. lines/switches between bus A and C should still be closed and therefore A and C should be logically connected
        replacement_id = id
    return replacement_id


def update_id_if_has_been_replaced(id: int, statistics: dict[str, Any]) -> int:
    """Update the ID if it has been replaced in the statistics.

    Each branch has a "from" and "to" bus.
    It can oocur that e.g. the "from" bus has been already replaced, but the "to" bus not.
    This function checks if the ID has been replaced and returns the new ID if it has been replaced.

    Parameters
    ----------
    id : str
        The ID of the branch, which might have been replaced
    statistics : dict
        The statistics dictionary from process_file()
        expects as input the statistics["changed_ids"] dictionary

    """
    for station in statistics.values():
        for element in station["branches"]:
            if element["original_id"] == id:
                if element["replacement_id"] != "":
                    id = element["replacement_id"]
                return id
    return id


def apply_branch_assignment(  # noqa: PLR0912, C901
    topology_optimizer_results: dict,
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    trafo_reg: pd.DataFrame,
    bus_a: str,
    bus_b: str,
    statistics_all_stations: dict[str, dict[str, dict[str, list[str]]]],
) -> list:
    """Apply branch assignment to the UCTE data.

    Uses the method to split a busbar into two busbars A and B

    Parameters
    ----------
    topology_optimizer_results : dict
        The substation split to apply from the postprocessed json
    lines : pd.DataFrame
        The lines data-frame from UCTE file
        Note: modifies the data-frames in place
    trafos : pd.DataFrame
        The transformers data-frame from UCTE file
        Note: modifies the data-frames in place
    trafo_reg : pd.DataFrame
        The transformer regulation data-frame from UCTE file
        Note: modifies the data-frames in place
    bus_a : str
        The bus A of the substation
    bus_b : str
        The bus B of the substation
    statistics_all_stations : dict
        The statistics dictionary from process_file()

    Returns
    -------
    statistics : list
        A list of dictionaries containing the original and replacement IDs of the branches that were modified

    Raises
    ------
    ValueError
        - If the branch type is not recognized
        - If the branch is not found in the data-frame

    """
    statistics = []  # type: list
    code = topology_optimizer_results["id"][0:7]

    # replace busbar ID in lines, trafos and trafo_reg df
    for element in topology_optimizer_results["branch_assignments"]:
        replacement_id = get_replacement_id(element, code, bus_a, bus_b)

        id = element["id"]
        if id != replacement_id:
            # replace only if ID is different
            if element["type"] == "LINE":
                replacement_id = handle_order_of_branch(lines, replacement_id)
                replaced = execute_branch_assignment(lines, id, replacement_id, statistics_all_stations)
                if replaced:
                    statistics.append(
                        {
                            "original_id": id,
                            "replacement_id": replacement_id,
                            "type": element["type"],
                        }
                    )
                else:
                    raise ValueError(
                        f"Line not found: bus_a: {bus_a}, bus_b: {bus_b}, id:{id}, replacement_id: {replacement_id}"
                    )

            elif element["type"] == "TWO_WINDINGS_TRANSFORMER":
                replacement_id = handle_order_of_branch(trafos, replacement_id)
                replaced = execute_branch_assignment(trafos, id, replacement_id, statistics_all_stations)
                replaced2 = execute_branch_assignment(trafo_reg, id, replacement_id, statistics_all_stations)
                if replaced or replaced2:
                    statistics.append(
                        {
                            "original_id": id,
                            "replacement_id": replacement_id,
                            "type": element["type"],
                        }
                    )
                else:
                    raise ValueError(
                        f"Transformer not found: bus_a: {bus_a}, bus_b: {bus_b}, id:{id}, replacement_id: {replacement_id}"
                    )
            elif element["type"] == "TIE_LINE":
                # TIE_LINE is a special case, as it is consists of two lines
                # UCTE has only lines -> split TIE_LINE into two lines -> search for both lines and replace
                id1 = id.split(" + ")[0]
                id2 = id.split(" + ")[1]
                replacement_id1 = replacement_id.split(" + ")[0]
                replacement_id2 = replacement_id.split(" + ")[1]
                if bus_a in id1 and bus_b in replacement_id1:
                    replacement_id1 = handle_order_of_branch(lines, replacement_id1)
                    replaced = execute_branch_assignment(lines, id1, replacement_id1, statistics_all_stations)
                elif bus_a in id2 and bus_b in replacement_id2:
                    replacement_id2 = handle_order_of_branch(lines, replacement_id2)
                    replaced = execute_branch_assignment(lines, id2, replacement_id2, statistics_all_stations)
                else:
                    raise ValueError(
                        f"TIE_LINE not found: bus_a: {bus_a}, bus_b: {bus_b}, id:{id}, replacement_id: {replacement_id}"
                    )
                if replaced:
                    statistics.append(
                        {
                            "original_id": id,
                            "replacement_id": replacement_id,
                            "type": element["type"],
                        }
                    )
                else:
                    raise ValueError(
                        f"TIE_LINE not found: bus_a: {bus_a}, bus_b: {bus_b}, id:{id}, replacement_id: {replacement_id}"
                    )

            else:
                raise ValueError(f"Unknown branch type: {element['type']}")
        else:
            statistics.append(
                {
                    "original_id": id,
                    "replacement_id": "",
                    "type": element["type"],
                }
            )

    return statistics
