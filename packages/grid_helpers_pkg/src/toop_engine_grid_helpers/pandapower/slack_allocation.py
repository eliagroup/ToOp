# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Utility functions for selecting and assigning slack generators."""

from typing import Optional, Union

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.create import create_gen
from pandapower.toolbox.grid_modification import (
    _adapt_profiles_in_replace_functions,
    _adapt_result_tables_in_replace_functions,
    _replace_group_member_element_type,
)
from toop_engine_grid_helpers.pandapower.network_topology_utils import collect_element_edges


def _slack_allocation_tie_break(df_min: pd.DataFrame) -> tuple[int, str]:
    """
    Return the index of the selected (s)gen & the respective element type after applying tie break rules.

    Parameters
    ----------
    df_min : pd.DataFrame
        A filtered DataFrame containing candidate generators/sgen entries

    Returns
    -------
    tuple[int, str]
        A tuple containing:
        - The index of the selected element.
        - The element type: either `"gen"` or `"sgen"`.
    """
    # fast path: no tie
    if len(df_min) == 1:
        row = df_min.iloc[0]
        return int(df_min.index[0]), str(row["etype"])

    tied_df = df_min
    # there is a tie and we have at least one sn_mva value populated
    if df_min["sn_mva"].notna().any():
        max_sn = df_min["sn_mva"].max()
        tied_df = df_min[df_min["sn_mva"] == max_sn]

    # still a tie --> just take the first one (treats corner case of duplicate gen/sgen indices too)
    # works even if the sn_mva step may have given us only one row already
    return int(tied_df.index[0]), str(tied_df["etype"].iloc[0])


def _get_vm_pu_for_bus(net: pp.pandapowerNet, bus: np.int64) -> float:
    """
    Determine the vm_pu setpoint for a given bus.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    bus : int
        The bus index whose vm_pu value should be retrieved.

    Returns
    -------
    float
        The vm_pu setpoint. Taken from res_bus if present, otherwise from
        gen or ext_grid entries on the same bus. Falls back to 1.0 pu.
    """
    # res_bus has priority
    if bus in net.res_bus.index:
        return float(net.res_bus.at[bus, "vm_pu"])

    # generator-defined setpoint
    if bus in net.gen.bus.values:
        return float(net.gen.vm_pu.loc[net.gen.bus == bus].values[0])

    # external grid setpoint
    if bus in net.ext_grid.bus.values:
        return float(net.ext_grid.vm_pu.loc[net.ext_grid.bus == bus].values[0])

    # fallback
    return 1.0


def _handle_replaced_sgen(net: pp.pandapowerNet, sgen: int, retain_sgen_elm: bool) -> None:
    """
    Handle removal or deactivation of an sgen after replacement.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network.
    sgen : int
        Index of the sgen element being replaced.
    retain_sgen_elm : bool
        Whether to keep the sgen element but mark it out of service.
    """
    if retain_sgen_elm:
        net.sgen.loc[sgen, "in_service"] = False

        # create column only if missing
        if "replaced_by_gen" not in net.sgen.columns:
            net.sgen["replaced_by_gen"] = False

        net.sgen.at[sgen, "replaced_by_gen"] = True
    else:
        net.sgen = net.sgen.drop(sgen)


def replace_sgen_by_gen(
    net: pp.pandapowerNet,
    sgen: int,
    cols_to_keep: Optional[list[str]] = None,
    retain_sgen_elm: bool = True,
) -> int:
    """
    Replace an sgen with a gen element.

    A new column "replaced_sgen" is created in net.gen which is
    set to True only for the new gen element. net.gen.vm_pu is set based on power flow results, if
    available, otherwise the default value of 1.0 pu is used.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    sgen : int
        sgen index to be replaced by gen
    cols_to_keep : Optional[list[str]]
        List of column names which should be kept while replacing sgen. If None, these columns
        are kept if values exist: "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar". However,
        these columns are always set: "bus", "vm_pu", "p_mw", "name", "in_service", "controllable".
    retain_sgen_elm : bool
        Flag to indicate whether to retain or drop the sgen element in net.sgen. If True, this element
        is set as out of service (in_service=False) and a new column "replaced_by_gen" is added which
        is set to True only for this element.

    Returns
    -------
    new_idx : int
        The new gen index.
    """
    # --- error handling
    if not isinstance(sgen, (int, np.integer)):
        raise ValueError("sgen must be a positive integer")
    if sgen not in net.sgen.index:
        raise ValueError(f"sgen index {sgen} not found in net.sgen")

    # --- determine which columns should be kept while replacing
    if not cols_to_keep:
        cols_to_keep = [
            "min_q_mvar",
            "max_q_mvar",
            "min_p_mw",
            "max_p_mw",
            "sn_mva",
            "id_q_capability_characteristic",
            "reactive_capability_curve",
            "curve_style",
            "scaling",
            "origin_id",
            "origin_class",
            "terminal",
            "description",
            "RegulatingControl.mode",
            "vn_kv",
            "rdss_ohm",
            "xdss_pu",
            "RegulatingControl.targetValue",
            "type",
            "referencePriority",
            "RegulatingControl.enabled",
        ]
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service", "controllable"})

    existing_cols_to_keep = net.sgen.loc[[sgen]].dropna(axis=1).columns.intersection(cols_to_keep)

    # add columns which should be kept from sgen but miss in gen to net.gen
    missing_cols_to_keep = existing_cols_to_keep.difference(net.gen.columns)
    for col in missing_cols_to_keep:
        net.gen[col] = pd.Series(data=None, dtype=net.sgen[col].dtype, name=col)

    # --- create gen
    sgen_row = net.sgen.loc[sgen]
    bus = sgen_row.bus

    vm_pu = _get_vm_pu_for_bus(net, bus)

    controllable = False if "controllable" not in net.sgen.columns else sgen_row.controllable

    new_idx = create_gen(
        net,
        bus,
        vm_pu=vm_pu,
        p_mw=sgen_row.p_mw,
        name=sgen_row["name"],
        # here sgen_row.name returns the sgen index
        in_service=sgen_row.in_service,
        controllable=controllable,
    )

    # copy selected existing columns from sgen to gen
    net.gen.loc[new_idx, existing_cols_to_keep] = net.sgen.loc[sgen, existing_cols_to_keep].values

    # populate slack_weight column
    if "referencePriority" in net.gen.columns:
        net.gen.at[new_idx, "slack_weight"] = net.gen.at[new_idx, "referencePriority"]

    # populate replaced_sgen gen column
    if "replaced_sgen" not in net.gen.columns:
        net.gen["replaced_sgen"] = False
    net.gen.at[new_idx, "replaced_sgen"] = True

    # --- update group info
    _replace_group_member_element_type(net, [sgen], "sgen", [new_idx], "gen")

    # --- adapt result data
    _adapt_result_tables_in_replace_functions(net, "sgen", [sgen], "gen", [new_idx])

    # --- adapt profiles
    _adapt_profiles_in_replace_functions(net, "sgen", [sgen], "gen", [new_idx])

    # --- drop replaced sgens
    _handle_replaced_sgen(net, sgen, retain_sgen_elm)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "sgen") & (net[table].element == sgen)]
            if len(to_change):
                net[table].loc[to_change, "et"] = "gen"
                net[table].loc[to_change, "element"] = new_idx

    return int(new_idx)


def get_generating_units_with_load(net: pp.pandapowerNet) -> set[int]:
    """
    Return all bus indices that host generation *and/or* load elements.

    Includes:
        - gen
        - sgen
        - ext_grid
        - ward / xward
        - load

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.

    Returns
    -------
    set[int]
        Unique bus indices that have either generation or load.
    """
    result = []
    for df_name in ["gen", "sgen", "ext_grid", "ward", "xward", "load"]:
        if df_name in net:
            result += list(getattr(net, df_name).bus)
    return set(result)


def get_buses_with_reference_sources(net: pp.pandapowerNet) -> set[int]:
    """
    Return all bus indices that have generators or static generators with a positive referencePriority value.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.

    Returns
    -------
    set[int]
        Set of bus indices with reference-capable generators or static generators.
    """
    gen_buses = set(net.gen.loc[net.gen["referencePriority"].fillna(0) > 0, "bus"].astype(int))
    sgen_buses = set(net.sgen.loc[net.sgen["referencePriority"].fillna(0) > 0, "bus"].astype(int))
    return gen_buses | sgen_buses


def assign_slack_gen_by_weight(net: pp.pandapowerNet, bus_idx_set: set[np.int64]) -> tuple[int, str]:
    """
    Select the (s)gen index to be assigned as slack based on referencePriority rules.

    Logic:
      1. Filter (s)gens by bus∈bus_idx_set and non-NaN/positive referencePriority.
         If the network is to be reduced to 50Hz area, (s)gens located on the Danish area
         (their bus zone containing "EnDK" string) are excluded to ensure correct reduction.
      2. Find minimum referencePriority among those.
      3. If exactly one gen has that min weight, choose it.
         Otherwise, among the tied:
           - If any sn_mva is non-NaN, pick those with the max sn_mva.
           - Else (all NaN), tie-break by random choice.
      4. If an sgen element is selected, further processing is required to convert sgen to gen
         and set slack=True.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    bus_idx_set : set[int]
        Set of bus indices to consider (e.g., a connected component).

    Returns
    -------
    tuple[int, str]
        A tuple containing:
        - The index of the chosen element (in `net.gen` or `net.sgen`).
        - The element type: either `"gen"` or `"sgen"`.
    """
    # Filter gens/sgens by bus set and positive referencePriority
    mask_gen = net.gen["bus"].isin(bus_idx_set) & (net.gen["referencePriority"].fillna(0) > 0)
    mask_sgen = net.sgen["bus"].isin(bus_idx_set) & (net.sgen["referencePriority"].fillna(0) > 0)

    candidates_gen = net.gen.loc[mask_gen, ["referencePriority", "sn_mva"]].copy()
    candidates_sgen = net.sgen.loc[mask_sgen, ["referencePriority", "sn_mva"]].copy()

    # Create single candidate table
    candidates_gen["etype"] = "gen"
    candidates_sgen["etype"] = "sgen"

    candidates = pd.concat(
        [candidates_gen.assign(idx=candidates_gen.index), candidates_sgen.assign(idx=candidates_sgen.index)],
        ignore_index=False,
    )

    # Find minimum referencePriority
    min_w = candidates["referencePriority"].min()
    df_min = candidates[candidates["referencePriority"] == min_w]

    # Select slack gen
    chosen_idx, element_type = _slack_allocation_tie_break(df_min)

    return chosen_idx, element_type


def assign_slack_per_island(
    net: pp.pandapowerNet,
    net_graph: Union[nx.Graph, nx.MultiGraph],
    bus_lookup: list[int],
    elements_ids: list[str],
    min_island_size: int,
) -> list[tuple[int, int]]:
    """
    Assign one slack generator per valid island in the network after isolating specific elements.

    This function deactivates all existing slack generators, removes selected element edges
    to simulate outages or disconnections, and identifies independent network islands.
    For each valid island (meeting size and generation criteria), it assigns a new slack generator
    based on generator weighting rules.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object containing buses, lines, generators, etc.
    net_graph : Union[nx.Graph, nx.MultiGraph]
        The network graph representation of the power system.
    bus_lookup : list[int]
        A list mapping graph node indices to pandapower bus indices.
    elements_ids : list[str]
        A list of element IDs (e.g., line or transformer IDs) to deactivate or remove.
    min_island_size : int
        Minimum number of nodes required for an island to be considered valid.

    Returns
    -------
    list of tuple[int, int]
        List of edges (tuples of node indices) that were removed from the graph.
    """
    if (not net.sgen.empty and "referencePriority" not in net.sgen.columns) or (
        not net.gen.empty and "referencePriority" not in net.gen.columns
    ):
        # This function requires 'referencePriority' columns in both sgen and gen tables.
        # Networks without these columns are not supported.
        return []
    # Deactivate all pre-allocated slacks
    net.gen["slack"] = False

    edges = collect_element_edges(net, elements_ids)
    net_graph.remove_edges_from(edges)

    components = list(nx.connected_components(net_graph))
    candidate_buses = get_buses_with_reference_sources(net)
    generating_units_with_load = get_generating_units_with_load(net)

    # Filter components based on criteria
    valid_components = [
        cc
        for cc in components
        if len(set(bus_lookup[i] for i in cc)) > min_island_size
        and not candidate_buses.isdisjoint(cc)
        and len(generating_units_with_load.intersection(cc)) >= 2
    ]

    for cc in valid_components:
        chosen_idx, element_type = assign_slack_gen_by_weight(net, cc)
        if element_type == "sgen":
            chosen_idx = replace_sgen_by_gen(net, sgen=chosen_idx, retain_sgen_elm=True)

        net.gen.at[chosen_idx, "slack"] = True

    return edges
