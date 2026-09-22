# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Outage group computation for pandapower networks."""

import hashlib

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
from beartype.typing import Callable, Iterable, List, Optional, Tuple

OUTAGE_GROUP_SEPARATOR = "&&"


def is_node_of_element_type(node_id: str, element_type: str) -> bool:
    """
    Check whether a node_id corresponds to a specific element type.

    The function assumes that element nodes follow the naming pattern:
    'e_<element_type>_' as part of the node_id string.

    Args:
        node_id: Full identifier of the node.
        element_type: Element type to check (e.g., 'line', 'switch', 'trafo').

    Returns
    -------
        True if the node_id contains the element type marker, False otherwise.

    Example:
        >>> is_node_of_element_type("e&&switch&&123", "switch")
        True
        >>> is_node_of_element_type("e&&line&&45", "switch")
        False
    """
    return f"e{OUTAGE_GROUP_SEPARATOR}{element_type}{OUTAGE_GROUP_SEPARATOR}" in node_id


def get_node_table_id(node_id: str) -> int:
    """
    Extract the numeric table ID from a node identifier.

    The function assumes that the node_id follows a convention where
    the last underscore-separated part is an integer ID.

    Args:
        node_id: Full node identifier (e.g., 'e_switch_123').

    Returns
    -------
        Integer table ID extracted from the node_id.

    Raises
    ------
        ValueError: If the last part of node_id is not a valid integer.

    Example:
        >>> get_node_table_id("e&&switch&&123")
        123
    """
    return int(node_id.rsplit(OUTAGE_GROUP_SEPARATOR, maxsplit=1)[-1])


def elem_node_id(kind: str, idx: int, etype: Optional[str] = None) -> str:
    """
    Stable node id format:

      - buses:  "b&&{bus}"
      - elems:  "e&&{etype}&&{idx}"
    """
    if kind == "bus":
        return f"b{OUTAGE_GROUP_SEPARATOR}{int(idx)}"
    if kind == "elem":
        if not etype:
            raise ValueError("etype required for element node ids")
        return f"e{OUTAGE_GROUP_SEPARATOR}{etype!s}{OUTAGE_GROUP_SEPARATOR}{int(idx)}"
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
        ("trafo", "trafo"),
        ("trafo3w", "trafo3w"),
        ("shunt", "shunt"),
        ("sgen", "sgen"),
        ("gen", "gen"),
        ("ward", "ward"),
        ("xward", "xward"),
        ("ext_grid", "ext_grid"),
    ]


#: Bus columns per element type; anything else is a single-``bus`` element.
_ELEMENT_BUS_COLUMNS: dict[str, tuple[str, ...]] = {
    "line": ("from_bus", "to_bus"),
    "impedance": ("from_bus", "to_bus"),
    "trafo": ("hv_bus", "lv_bus"),
    "trafo3w": ("hv_bus", "mv_bus", "lv_bus"),
}


def _bus_node_ids(buses: Iterable[int]) -> list[str]:
    prefix = f"b{OUTAGE_GROUP_SEPARATOR}"
    return [f"{prefix}{int(bus)}" for bus in buses]


def _add_element_table(graph: nx.Graph, tbl: pd.DataFrame, etype: str, bus_columns: Tuple[str, ...]) -> None:
    """Add one element node per row of *tbl* and an edge to each of its buses.

    Whole columns go through ``add_nodes_from`` / ``add_edges_from``: on a transmission grid the
    element tables hold tens of thousands of rows, and inserting them one ``itertuples`` row at a
    time dominated :func:`build_connectivity_graph_for_contingency`.
    """
    indices = tbl.index.to_numpy()
    bus_arrays = []
    for column in bus_columns:
        buses = pd.to_numeric(tbl[column], errors="coerce").to_numpy(dtype=float)
        # NaN (missing or non-numeric) and fractional values are both malformed bus ids.
        bad = np.flatnonzero(~np.isfinite(buses) | (buses != np.floor(buses)))
        if len(bad):
            raise RuntimeError(f"Malformed {etype} row idx={int(indices[bad[0]])}")
        bus_arrays.append(buses.astype(np.int64))

    prefix = f"e{OUTAGE_GROUP_SEPARATOR}{etype}{OUTAGE_GROUP_SEPARATOR}"
    element_ids = [f"{prefix}{int(idx)}" for idx in indices.tolist()]
    graph.add_nodes_from(
        (nid, {"kind": "elem", "etype": etype, "idx": int(idx)})
        for nid, idx in zip(element_ids, indices.tolist(), strict=True)
    )
    for buses in bus_arrays:
        bus_ids = _bus_node_ids(buses.tolist())
        graph.add_nodes_from(bus_ids, kind="bus")
        graph.add_edges_from(zip(element_ids, bus_ids, strict=True))


def add_elements_bipartite(net: pp.pandapowerNet, graph: nx.Graph, tables: List[Tuple[str, str]]) -> None:
    """Add element nodes + element->bus edges for the provided element tables."""
    for etype, table_name in tables:
        if not hasattr(net, table_name):
            continue
        tbl = getattr(net, table_name)
        if tbl is None or tbl.empty:
            continue

        bus_columns = _ELEMENT_BUS_COLUMNS.get(etype, ("bus",))
        if any(column not in tbl.columns for column in bus_columns):
            continue
        _add_element_table(graph, tbl, etype, bus_columns)


def add_traversable_bus_bus_edges(graph: nx.Graph, pairs: Iterable[Tuple[int, int]]) -> None:
    """Add bus-bus edges for each (u, v) traversable pair."""
    pairs = list(pairs)
    if not pairs:
        return
    u_ids = _bus_node_ids(u for u, _ in pairs)
    v_ids = _bus_node_ids(v for _, v in pairs)
    graph.add_nodes_from(u_ids, kind="bus")
    graph.add_nodes_from(v_ids, kind="bus")
    graph.add_edges_from(zip(u_ids, v_ids, strict=True))


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


class ConnectivityGraphCache:
    """Reuse the connectivity graph and its components across contingencies.

    :func:`build_connectivity_graph_for_contingency` is expensive (tens of thousands of
    networkx node/edge insertions) but its inputs are element-bus incidence and the closed
    state of *non-CB* bus-bus switches. A contingency opens CB switches and takes elements
    out of service, so it normally leaves the graph completely unchanged.

    SpPS actions and cascade steps can toggle non-CB switches, though, so the cached graph
    is keyed on a fingerprint of the closed non-CB bus-bus switches and rebuilt only when
    those actually change. CB switches are not part of the key at all: they never become
    graph edges, so opening or closing one must not invalidate the graph.

    Hold one instance per run. It must not live on the network itself, since the network is
    deep-copied per outage and copying the graph would cost as much as rebuilding it.
    """

    def __init__(self) -> None:
        self._fingerprint: Optional[bytes] = None
        self._graph: Optional[nx.Graph] = None
        self._components: Optional[list] = None
        self._derived: dict = {}

    @staticmethod
    def _fingerprint_of(net: pp.pandapowerNet) -> bytes:
        """Cheap digest of everything the graph is built from that can change per outage."""
        switches = getattr(net, "switch", None)
        if switches is None or switches.empty:
            return b"no-switches"

        is_bus_bus = switches["et"].to_numpy() == "b"
        is_cb = switches["type"].to_numpy().astype(str) == "CB"
        closed = switches["closed"].to_numpy(dtype=bool)
        traversable = np.ascontiguousarray(closed & is_bus_bus & ~is_cb)
        digest = hashlib.blake2b(traversable.tobytes(), digest_size=16)
        # Element tables are static within a run, but their sizes are a cheap guard against
        # being handed a structurally different network.
        for table, _ in element_tables_to_scan_default():
            digest.update(str(len(net[table])).encode())
        return digest.digest()

    def get(self, net: pp.pandapowerNet) -> tuple[nx.Graph, list]:
        """Return ``(graph, connected_components)`` for *net*, rebuilding only when needed."""
        fingerprint = self._fingerprint_of(net)
        if self._graph is None or fingerprint != self._fingerprint:
            self._graph = build_connectivity_graph_for_contingency(net)
            self._components = list(nx.connected_components(self._graph))
            self._fingerprint = fingerprint
            self._derived.clear()
        return self._graph, self._components

    def derive(self, net: pp.pandapowerNet, key: str, factory: "Callable[[], object]") -> object:
        """Memoize a value derived from the cached graph under *key*.

        *factory* is called only when the graph is (re)built, so anything computed purely
        from the graph or its components is shared across contingencies. Do not use this
        for values that depend on the outage itself.
        """
        self.get(net)
        if key not in self._derived:
            self._derived[key] = factory()
        return self._derived[key]


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
