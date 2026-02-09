# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Outage group computation for pandapower networks."""

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
from beartype.typing import Iterable, List, Optional, Tuple


def elem_node_id(kind: str, idx: int, etype: Optional[str] = None) -> str:
    """
    Stable node id format:

      - buses:  "b_{bus}"
      - elems:  "e_{etype}_{idx}"
    """
    if kind == "bus":
        return f"b_{int(idx)}"
    if kind == "elem":
        if not etype:
            raise ValueError("etype required for element node ids")
        return f"e_{etype!s}_{int(idx)}"
    raise ValueError(f"Unknown kind={kind}")


def preprocess_bus_bus_switches(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Return a normalized switch dataframe containing only bus-bus switches (et == 'b'),

    with normalized columns: bus(int), element(int), type(str upper), closed(bool).
    """
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return pd.DataFrame(columns=["bus", "element", "type", "closed"])

    sw = net.switch.copy()

    # Keep only bus-bus switches; your original summary says bus-to-bus switches.
    if "et" in sw.columns:
        sw = sw.loc[sw["et"] == "b"]

    if sw.empty:
        return pd.DataFrame(columns=["bus", "element", "type", "closed"])

    sw["bus"] = sw["bus"].astype(int)
    sw["element"] = sw["element"].astype(int)
    sw["type"] = sw["type"].astype(str)
    sw["closed"] = sw["closed"].astype(bool)

    return sw[["bus", "element", "type", "closed"]]


def aggregate_switch_pairs(sw: pd.DataFrame) -> pd.DataFrame:
    """
    Build unordered pairs (u=min(bus, element), v=max(...)) and aggregate booleans per pair.

    Produces columns: u, v, closed_non_cb, closed_cb, total_switches.
    """
    if sw is None or sw.empty:
        return pd.DataFrame(columns=["u", "v", "closed_non_cb"])

    u_arr = np.minimum(sw["bus"].to_numpy(), sw["element"].to_numpy()).astype(int)
    v_arr = np.maximum(sw["bus"].to_numpy(), sw["element"].to_numpy()).astype(int)

    closed_arr = sw["closed"].to_numpy()
    is_cb_arr = sw["type"].to_numpy() == "CB"

    pair_df = pd.DataFrame(
        {
            "u": u_arr,
            "v": v_arr,
            "closed_non_cb": closed_arr & (~is_cb_arr),
        }
    )
    # For each unordered (bus, element) pair,
    # check whether at least one non-circuit-breaker switch is closed between them
    agg = (
        pair_df.groupby(["u", "v"], sort=False)
        .agg(
            {
                "closed_non_cb": "any",
            }
        )
        .reset_index()
    )

    return agg


def get_traversable_bus_bus_pairs(agg: pd.DataFrame) -> List[Tuple[int, int]]:
    """Pairs that are traversable for connectivity: closed non-CB."""
    if agg is None or agg.empty:
        return []
    return [tuple(map(int, uv)) for uv in agg.loc[agg["closed_non_cb"], ["u", "v"]].to_numpy()]


def element_tables_to_scan_default() -> List[Tuple[str, str]]:
    """Return element tables to scan for connectivity."""
    return [
        ("line", "line"),
        ("impedance", "impedance"),
        ("trafo", "trafo"),
        ("trafo3w", "trafo3w"),
        ("shunt", "shunt"),
        ("sgen", "sgen"),
        ("gen", "gen"),
        ("ward", "ward"),
        ("xward", "xward"),
        ("ext_grid", "ext_grid"),
    ]


def _add_bus_node(graph: nx.Graph, bus: int) -> str:
    """Add a bus node with the given index."""
    nid = elem_node_id("bus", int(bus))
    graph.add_node(nid, kind="bus")
    return nid


def _add_element_node(graph: nx.Graph, etype: str, idx: int) -> str:
    """Add an element node with the given index and type."""
    nid = elem_node_id("elem", int(idx), etype=etype)
    graph.add_node(nid, kind="elem", etype=etype, idx=int(idx))
    return nid


def _add_element_bus_edge(graph: nx.Graph, elem_nid: str, bus: int) -> None:
    """Add an edge from the element node to the bus node."""
    bus_nid = _add_bus_node(graph, int(bus))
    graph.add_edge(elem_nid, bus_nid)


def _add_line_edges(graph: nx.Graph, tbl: pd.DataFrame) -> None:
    """Add element nodes and element→bus edges for tables with two ``bus`` columns."""
    for row in tbl.itertuples(index=True):
        idx = int(row.Index)
        try:
            fb = int(row.from_bus)
            tb = int(row.to_bus)
        except Exception as e:
            raise RuntimeError(f"Malformed line row idx={idx}") from e

        eid = _add_element_node(graph, "line", idx)
        _add_element_bus_edge(graph, eid, fb)
        _add_element_bus_edge(graph, eid, tb)


def _add_impedance_edges(graph: nx.Graph, tbl: pd.DataFrame) -> None:
    """Add element nodes and element→bus edges for tables with two ``bus`` columns."""
    for row in tbl.itertuples(index=True):
        idx = int(row.Index)
        try:
            fb = int(row.from_bus)
            tb = int(row.to_bus)
        except Exception as e:
            raise RuntimeError(f"Malformed line row idx={idx}") from e

        eid = _add_element_node(graph, "impedance", idx)
        _add_element_bus_edge(graph, eid, fb)
        _add_element_bus_edge(graph, eid, tb)


def _add_trafo_edges(graph: nx.Graph, tbl: pd.DataFrame) -> None:
    """Add element nodes and element→bus edges for tables with two ``bus`` columns."""
    for row in tbl.itertuples(index=True):
        idx = int(row.Index)
        try:
            hv = int(row.hv_bus)
            lv = int(row.lv_bus)
        except Exception as e:
            raise RuntimeError(f"Malformed trafo row idx={idx}") from e

        eid = _add_element_node(graph, "trafo", idx)
        _add_element_bus_edge(graph, eid, hv)
        _add_element_bus_edge(graph, eid, lv)


def _add_trafo3w_edges(graph: nx.Graph, tbl: pd.DataFrame) -> None:
    """Add element nodes and element→bus edges for tables with three ``bus`` columns."""
    for row in tbl.itertuples(index=True):
        idx = int(row.Index)
        try:
            hv = int(row.hv_bus)
            mv = int(row.mv_bus)
            lv = int(row.lv_bus)
        except Exception as e:
            raise RuntimeError(f"Malformed trafo3w row idx={idx}") from e

        eid = _add_element_node(graph, "trafo3w", idx)
        for b in (hv, mv, lv):
            _add_element_bus_edge(graph, eid, b)


def _add_single_bus_element_edges(graph: nx.Graph, tbl: pd.DataFrame, etype: str) -> None:
    """Add element nodes and element→bus edges for tables with a single ``bus`` column."""
    # common assumption in your original code
    if "bus" not in tbl.columns:
        return

    for row in tbl.itertuples(index=True):
        idx = int(row.Index)
        try:
            b = int(row.bus)
        except Exception as e:
            raise RuntimeError(f"Malformed {etype} row idx={idx}") from e

        eid = _add_element_node(graph, etype, idx)
        _add_element_bus_edge(graph, eid, b)


def add_elements_bipartite(net: pp.pandapowerNet, graph: nx.Graph, tables: List[Tuple[str, str]]) -> None:
    """Add element nodes + element->bus edges for the provided element tables."""
    for etype, table_name in tables:
        if not hasattr(net, table_name):
            continue
        tbl = getattr(net, table_name)
        if tbl is None or tbl.empty:
            continue

        if etype == "line":
            _add_line_edges(graph, tbl)
        if etype == "impedance":
            _add_impedance_edges(graph, tbl)
        elif etype == "trafo":
            _add_trafo_edges(graph, tbl)
        elif etype == "trafo3w":
            _add_trafo3w_edges(graph, tbl)
        else:
            _add_single_bus_element_edges(graph, tbl, etype)


def add_traversable_bus_bus_edges(graph: nx.Graph, pairs: Iterable[Tuple[int, int]]) -> None:
    """Add bus-bus edges for each (u, v) traversable pair."""
    for u, v in pairs:
        u_n = _add_bus_node(graph, int(u))
        v_n = _add_bus_node(graph, int(v))
        graph.add_edge(u_n, v_n)


def build_connectivity_graph_for_contingency(
    net: pp.pandapowerNet,
    element_tables: Optional[List[Tuple[str, str]]] = None,
) -> nx.Graph:
    """
    Full graph assembly:

      - element/bus bipartite edges
      - plus traversable bus-bus edges from closed non-CB switches
    """
    graph = nx.Graph()

    sw = preprocess_bus_bus_switches(net)
    agg = aggregate_switch_pairs(sw)
    traversable_pairs = get_traversable_bus_bus_pairs(agg)

    tables = element_tables if element_tables is not None else element_tables_to_scan_default()
    add_elements_bipartite(net, graph, tables)
    add_traversable_bus_bus_edges(graph, traversable_pairs)

    return graph


def build_connected_components_for_contingency_analysis(net: pp.pandapowerNet) -> list:
    """
    Build connected components for contingency analysis.

    Given a pandapower network `net`, this function returns the connected
    components based on closed non-circuit-breaker (non-CB) bus <-> bus
    connectivity.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing buses, switches, and element tables.
        The network is assumed to be internally consistent and indexed
        according to pandapower conventions.

    Returns
    -------
    list[set[str]]
        A list of connected components. Each component is represented as a
        set of node identifiers (strings), where:
        - bus nodes are labeled as `"b_<bus_index>"`
        - element nodes are labeled as `"e_<element_type>_<element_index>"`

        Each set represents a maximal group of mutually reachable nodes
        under the contingency connectivity rules.

    Behavior / Algorithm (summary)
    ------------------------------
    1. Preprocess switches: select only bus-to-bus switches (`et == 'b`),
    2. Build `pair_df` with normalized unordered pairs
       (`u = min(bus, element)`, `v = max(...)`) and per-switch booleans
       `closed_non_cb` and `closed_cb`.
    3. Aggregate per pair using `.groupby(...).any()` → `agg`.
    4. Build a graph with:
       - bus nodes: `"b_<bus_index>"`
       - element nodes: `"e_<element_type>_<idx>"`
       - element-to-bus edges for all scanned element tables
       - traversable bus-bus edges for pairs with `closed_non_cb == True`
    5. Compute connected components of the graph.

    Notes
    -----
    - Switches are not treated as contingency elements themselves.
    - To restrict which element types participate in connectivity,
      edit `element_tables_to_scan` in the graph builder.
    """
    graph = build_connectivity_graph_for_contingency(net)
    return list(nx.connected_components(graph))
