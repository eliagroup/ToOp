# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""
Switch mapping and result aggregation for contingency analysis.

This module builds mappings between switches and electrically connected elements
and uses them to aggregate branch flows and node injections per switch.
Results are computed based on the topology defined by closed bus-bus switches,
considering one side of each switch.
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pandera as pa
import pandera.typing as pat
import polars as pl
from beartype.typing import Literal
from pandera.typing import Series
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    SEPARATOR,
    get_globally_unique_id,
    get_globally_unique_id_from_index,
)
from toop_engine_interfaces.loadflow_results import BranchSide, SwitchResultsSchema
from toop_engine_interfaces.nminus1_definition import SwitchMonitoringScope

# ``sqrt(3)`` as a plain Python float so polars expressions reuse the exact value the
# previous numpy-based implementation used, keeping current results bit-for-bit comparable.
_SQRT3 = float(np.sqrt(3))


class SwitchElementMappingSchema(pa.DataFrameModel):
    """Schema for mapping switches to electrically connected elements.

    This table defines which elements belong to the electrically connected
    component of each switch side. The component is built by traversing
    zero-impedance connections, i.e. buses/busbars and closed switches
    (CBs and disconnectors).

    The traversal stops when an element with non-zero impedance is reached,
    such as a line, transformer (2W/3W), or impedance element. These elements
    form the boundary of the connected component and are included in the mapping.

    The mapping may therefore include:
    - buses connected through zero-impedance paths
    - branch-like boundary elements (lines, trafos, 3W trafos, impedances, etc.)

    It is used to aggregate branch flows and node injections when computing
    switch-level results.

    If no switches are mapped, this is an empty DataFrame.
    """

    switch_id: Series[int] = pa.Field(nullable=False)
    """The pandapower index of the switch.

    This identifies the switch for which connected elements are collected and
    used in result aggregation.
    """

    element: Series[str] = pa.Field(nullable=False)
    """The globally unique identifier of the connected element.

    This can represent either:
    - a branch-like element
    - a bus
    """

    side: Series[float] = pa.Field(nullable=True)
    """The side of the branch element.

    - For branch-like elements:
        Indicates the terminal of the element connected to the bus
        (e.g. BranchSide.ONE, BranchSide.TWO, BranchSide.THREE).
    - For bus entries:
        This value is NaN, since buses do not have sides.
    """


def create_closed_bb_switches_graph(net: pp.pandapowerNet) -> nx.Graph:
    """Create a bus-level connectivity graph based on closed bus-bus switches.

    This function builds an undirected graph representing electrical connectivity
    between buses through closed bus-bus switches only. Each node corresponds to
    a bus, and an edge is added between two buses if they are directly connected
    by a closed switch with ``et == "b"``.

    The resulting graph is used for topological analysis, such as identifying
    electrically connected components or determining the set of buses reachable
    from one side of a switch when computing switch results.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing buses and switches. The function uses:
        - ``net.bus.index`` as graph nodes
        - ``net.switch`` to determine connectivity

    Returns
    -------
    nx.Graph
        Undirected graph where:
        - nodes represent bus indices
        - edges represent closed bus-bus switches
    """
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)

    closed_busbus = net.switch.loc[
        (net.switch.et == "b") & (net.switch.closed),
        ["bus", "element"],
    ]

    graph.add_edges_from(closed_busbus.to_numpy())

    return graph


def _build_bus_to_branch_map(
    net: pp.pandapowerNet,
) -> dict[int, list[tuple[str, int]]]:
    """Precompute mapping from buses to connected branch-like elements.

    This function builds a lookup structure that maps each bus to all connected
    branch-like elements such as impedances, lines, transformers, and
    3-winding transformers.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing branch elements in the tables
        ``impedance``, ``line``, ``trafo``, and ``trafo3w``.

    Returns
    -------
    dict[int, list[tuple[str, int]]]
        Mapping of bus index to a list of tuples:

        ``bus -> [(global_id, side), ...]``

        Where:
        - ``global_id`` (str):
            Globally unique identifier of the connected element
            (for example ``"12__line"``).
        - ``side`` (int):
            Branch side identifier
            (for example ``BranchSide.ONE``, ``BranchSide.TWO``,
            or ``BranchSide.THREE``).

    Notes
    -----
    - Each branch-like element is added once per connected terminal bus.
    - The order follows the insertion order of the pandapower tables and the
      sequence in which element types are processed in this function.
    """
    bus_to_branches: dict[int, list[tuple[str, int]]] = defaultdict(list)

    def add_side(df: pd.DataFrame, bus_col: str, element_type: str, side: int) -> None:
        if df.empty:
            return
        for row in df.itertuples():
            bus = int(getattr(row, bus_col))
            uid = get_globally_unique_id(int(row.Index), element_type)
            bus_to_branches[bus].append((uid, side))

    add_side(net.impedance, "from_bus", "impedance", BranchSide.ONE.value)
    add_side(net.impedance, "to_bus", "impedance", BranchSide.TWO.value)

    add_side(net.line, "from_bus", "line", BranchSide.ONE.value)
    add_side(net.line, "to_bus", "line", BranchSide.TWO.value)

    add_side(net.trafo, "hv_bus", "trafo", BranchSide.ONE.value)
    add_side(net.trafo, "lv_bus", "trafo", BranchSide.TWO.value)

    add_side(net.trafo3w, "hv_bus", "trafo3w", BranchSide.ONE.value)
    add_side(net.trafo3w, "mv_bus", "trafo3w", BranchSide.TWO.value)
    add_side(net.trafo3w, "lv_bus", "trafo3w", BranchSide.THREE.value)

    return bus_to_branches


def _connected_component_without_edge(
    graph: nx.Graph,
    source: int,
    blocked_edge: tuple[int, int] | None = None,
) -> set[int | np.int64]:
    """Return the connected component of a node while optionally ignoring one edge.

    This function computes all nodes reachable from a given ``source`` node in
    an undirected graph. If ``blocked_edge`` is provided, that edge is treated
    as removed during traversal, without modifying the original graph.

    In the context of switch result computation, this is used to isolate the
    electrical component on one side of a switch. By temporarily ignoring the
    edge representing the switch connection, the traversal returns only the
    buses (and therefore elements) located on that specific side. This allows
    correct aggregation of power flows and injections associated with the switch.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph representing connectivity (e.g. buses connected via
        closed bus-bus switches).
    source : int
        Starting node (bus) from which the connected component is explored.
    blocked_edge : tuple[int, int] | None, optional
        Edge to ignore during traversal, given as ``(u, v)``. The edge is treated
        as absent in both directions. If ``None``, the full connected component
        is returned.

    Returns
    -------
    set[int]
        Set of node indices reachable from ``source`` under the given conditions.

    Raises
    ------
    nx.NetworkXError
        If the ``source`` node is not present in the graph.
    """
    if source not in graph:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")

    if blocked_edge is None:
        return nx.node_connected_component(graph, source)

    u_blocked, v_blocked = blocked_edge
    seen = {source}
    stack = [source]
    adj = graph.adj

    while stack:
        node = stack.pop()
        for nbr in adj[node]:
            if (node == u_blocked and nbr == v_blocked) or (node == v_blocked and nbr == u_blocked):
                continue
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)

    return seen


def _get_elements_for_buses(
    switch_id: int | np.int64,
    sw_buses: set[int | np.int64],
    bus_to_branch_map: dict[int, list[tuple[str, int]]],
) -> list[tuple[int | np.int64, str, int | np.int64]]:
    """Collect branch-like elements connected to a set of buses for a switch.

    This function retrieves all branch-like elements (lines, impedances,
    transformers, etc.) connected to the provided set of buses and formats them
    for switch result aggregation.

    Parameters
    ----------
    switch_id : int
        Pandapower index of the switch for which the elements are collected.
    sw_buses : set[int]
        Set of bus indices representing the electrically connected component
        on one side of the switch.
    bus_to_branch_map : dict[int, list[tuple[str, int]]]
        Precomputed mapping from bus to connected branch elements as produced by
        ``_build_bus_to_branch_map``.

        Each entry has the form:
        ``bus -> [(element_uid, side), ...]``

    Returns
    -------
    list[tuple[int, str, int]]
        List of tuples:

        ``[(switch_id, element_uid, side), ...]``

        Where:
        - ``switch_id`` (int): pandapower switch index
        - ``element_uid`` (str): globally unique identifier of the element
        - ``side`` (int): branch side identifier
    """
    matches: list[tuple[str, int]] = []

    for bus in sw_buses:
        matches.extend(bus_to_branch_map.get(int(bus), ()))

    return [(switch_id, element_uid, side) for element_uid, side in matches]


# Branch tables and their terminal columns, in the same order _build_bus_to_branch_map uses.
_BRANCH_TERMINALS: tuple[tuple[str, str, int], ...] = (
    ("impedance", "from_bus", BranchSide.ONE.value),
    ("impedance", "to_bus", BranchSide.TWO.value),
    ("line", "from_bus", BranchSide.ONE.value),
    ("line", "to_bus", BranchSide.TWO.value),
    ("trafo", "hv_bus", BranchSide.ONE.value),
    ("trafo", "lv_bus", BranchSide.TWO.value),
    ("trafo3w", "hv_bus", BranchSide.ONE.value),
    ("trafo3w", "mv_bus", BranchSide.TWO.value),
    ("trafo3w", "lv_bus", BranchSide.THREE.value),
)


def _build_branch_terminal_frame(net: pp.pandapowerNet) -> pd.DataFrame:
    """Vectorized ``(bus, element, side)`` table of every branch terminal.

    Same content as :func:`_build_bus_to_branch_map`, built with array ops instead of
    ``itertuples`` so it stays cheap on nets with tens of thousands of branches.
    """
    frames = []
    for table, bus_col, side in _BRANCH_TERMINALS:
        df = net[table]
        if df.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "bus": df[bus_col].to_numpy(dtype=np.int64),
                    "element": get_globally_unique_id_from_index(df.index, table).to_numpy(dtype=object),
                    "side": np.full(len(df), side, dtype=np.int64),
                }
            )
        )

    if not frames:
        return pd.DataFrame({"bus": np.array([], dtype=np.int64), "element": [], "side": np.array([], dtype=np.int64)})

    return pd.concat(frames, ignore_index=True)


class _BusBranchIndex:
    """CSR-style lookup from bus position to the branch terminals at that bus.

    ``element[indptr[p]:indptr[p + 1]]`` are the branch uids at bus position ``p``, with
    matching sides in ``side``. Positions index into ``bus_ids`` (``net.bus.index``), which
    is the space the traversal below works in, so no per-bus dict lookups are needed.
    """

    def __init__(self, bus_ids: np.ndarray, terminals: pd.DataFrame) -> None:
        n_buses = len(bus_ids)
        bus_index = pd.Index(bus_ids)

        # Terminals on buses that are not in net.bus are dropped, as before.
        positions = bus_index.get_indexer(terminals["bus"].to_numpy(dtype=np.int64))
        keep = positions >= 0
        positions = positions[keep]

        # A stable sort keeps the element-type / side order within each bus.
        order = np.argsort(positions, kind="stable")
        positions = positions[order]

        self.element = terminals["element"].to_numpy(dtype=object)[keep][order]
        self.side = terminals["side"].to_numpy(dtype=np.int64)[keep][order]
        self.indptr = np.zeros(n_buses + 1, dtype=np.int64)
        np.cumsum(np.bincount(positions, minlength=n_buses), out=self.indptr[1:])

    def gather(self, bus_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Branch terminals on *bus_positions*.

        Returns ``(elements, sides, counts)``, where ``counts[i]`` is how many terminals
        ``bus_positions[i]`` contributed, so callers can expand their own per-bus columns
        onto the terminal rows.
        """
        starts = self.indptr[bus_positions]
        counts = self.indptr[bus_positions + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return self.element[:0], self.side[:0], counts

        # Ragged gather: expand each [start, start + count) range without a Python loop.
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        flat = offsets + np.arange(total, dtype=np.int64)
        return self.element[flat], self.side[flat], counts


def _build_bus_branch_index(net: pp.pandapowerNet, bus_ids: np.ndarray) -> _BusBranchIndex:
    """Build the CSR bus -> branch-terminal index used by the switch mapping."""
    return _BusBranchIndex(bus_ids, _build_branch_terminal_frame(net))


def _build_bus_adjacency(net: pp.pandapowerNet, bus_ids: np.ndarray) -> list[list[int]]:
    """Adjacency list of the closed bus-bus switch graph, in bus-position space.

    Same graph as :func:`create_closed_bb_switches_graph`, but as plain Python lists of
    positions instead of an ``nx.Graph``: the traversal below walks it per switch, and
    networkx's adjacency views dominate the runtime at that call count.
    """
    bus_index = pd.Index(bus_ids)
    closed_busbus = net.switch.loc[(net.switch.et == "b") & (net.switch.closed), ["bus", "element"]]

    adjacency: list[list[int]] = [[] for _ in range(len(bus_ids))]
    if closed_busbus.empty:
        return adjacency

    from_pos = bus_index.get_indexer(closed_busbus["bus"].to_numpy(dtype=np.int64))
    to_pos = bus_index.get_indexer(closed_busbus["element"].to_numpy(dtype=np.int64))

    for u, v in zip(from_pos.tolist(), to_pos.tolist(), strict=True):
        if u < 0 or v < 0:  # switch on a bus outside net.bus: no edge, as in the nx graph
            continue
        adjacency[u].append(v)
        adjacency[v].append(u)

    return adjacency


def _reachable_positions(
    adjacency: list[list[int]],
    source: int,
    blocked_edge: tuple[int, int] | None,
) -> list[int]:
    """Positions reachable from *source*, ignoring *blocked_edge* in both directions.

    Array-based twin of :func:`_connected_component_without_edge`; the traversal stays
    inside *source*'s connected component, so cost scales with that component, not the net.
    """
    seen = {source}
    stack = [source]
    u_blocked, v_blocked = blocked_edge if blocked_edge is not None else (-1, -1)

    while stack:
        node = stack.pop()
        for nbr in adjacency[node]:
            if (node == u_blocked and nbr == v_blocked) or (node == v_blocked and nbr == u_blocked):
                continue
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)

    return list(seen)


def _get_switch_mapped_elements_by_origin_ids(
    net: pp.pandapowerNet, switches_ids: list[int], side: Literal["bus", "element"]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute switch-to-element mappings for given switch origin IDs.

    For each closed switch identified by its ``origin_id``, this function:
    - selects one side of the switch (``bus`` or ``element``)
    - computes the electrically connected buses on that side
      (treating the switch itself as open)
    - collects all branch-like elements connected to those buses
    - collects all reachable buses

    Same result as walking :func:`create_closed_bb_switches_graph` with networkx per
    switch, but the graph is built as flat arrays and the per-switch traversal only
    touches its own connected component - the graph rebuild and repeated adjacency-view
    lookups otherwise dominate the runtime on large nets.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing switches, buses, and branch elements.
    switches_ids : list[int]
        List of switch IDs identifying which switches to process.
        Only switches with matching ``id`` and ``closed == True`` are used.
    side : Literal["bus", "element"]
        Defines from which side of each switch the topology traversal starts:

        - ``"bus"``: start from ``sw.bus``
        - ``"element"``: start from ``sw.element``

        This determines which connected component is explored.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames:

        1. branch_map_df
            Columns:
            - ``switch_id`` (int): pandapower switch index
            - ``element`` (str): globally unique ID of the branch-like element
            - ``side`` (int): branch side (e.g. from/to/hv/lv)

        2. bus_map_df
            Columns:
            - ``switch_id`` (int): pandapower switch index
            - ``element`` (str): globally unique ID of the bus
    """
    # Open switches are mapped too, not just closed ones: SpPS may close a switch mid-run
    # and then re-run cascading, so its connectivity must already be available.
    switch_mask = net.switch.index.isin(switches_ids)
    switches = net.switch.loc[switch_mask]
    if switches.empty:
        return (
            pd.DataFrame(columns=["switch_id", "element", "side"]),
            pd.DataFrame(columns=["switch_id", "element"]),
        )

    bus_ids = net.bus.index.to_numpy(dtype=np.int64)
    bus_index = pd.Index(bus_ids)
    bus_uids = get_globally_unique_id_from_index(net.bus.index, "bus").to_numpy(dtype=object)

    adjacency = _build_bus_adjacency(net, bus_ids)
    bus_branch_index = _build_bus_branch_index(net, bus_ids)

    switch_ids = switches.index.to_numpy(dtype=np.int64)
    bus_pos = bus_index.get_indexer(switches["bus"].to_numpy(dtype=np.int64))
    element_pos = bus_index.get_indexer(switches["element"].to_numpy(dtype=np.int64))
    source_pos = bus_pos if side == "bus" else element_pos

    # The loop does the graph traversal only: it collects, per switch, the buses reachable
    # on the chosen side. Turning those into rows is a single vectorized pass afterwards,
    # because per-switch numpy calls cost more in overhead than the work they do.
    reached_switch_ids: list[int] = []
    reached_bus_counts: list[int] = []
    reached_bus_positions: list[int] = []
    # A switch whose (bus, element) pair is not an edge of the graph blocks nothing and so
    # reaches its whole component; switches sharing such a source bus reach the same buses.
    full_component_cache: dict[int, list[int]] = {}

    for switch_id, source, u, v in zip(
        switch_ids.tolist(), source_pos.tolist(), bus_pos.tolist(), element_pos.tolist(), strict=True
    ):
        if source < 0:  # switch side is not a bus in net.bus -> nothing reachable
            continue

        has_edge = u >= 0 and v >= 0 and v in adjacency[u]
        if has_edge:
            reachable = _reachable_positions(adjacency, source, (u, v))
        else:
            reachable = full_component_cache.get(source)
            if reachable is None:
                reachable = _reachable_positions(adjacency, source, None)
                full_component_cache[source] = reachable

        if not reachable:
            continue

        reached_switch_ids.append(switch_id)
        reached_bus_counts.append(len(reachable))
        reached_bus_positions.extend(reachable)

    if not reached_switch_ids:
        return (
            pd.DataFrame(columns=["switch_id", "element", "side"]),
            pd.DataFrame(columns=["switch_id", "element"]),
        )

    bus_positions = np.fromiter(reached_bus_positions, dtype=np.int64, count=len(reached_bus_positions))
    # One row per (switch, reachable bus).
    switch_per_bus = np.repeat(
        np.fromiter(reached_switch_ids, dtype=np.int64, count=len(reached_switch_ids)),
        np.fromiter(reached_bus_counts, dtype=np.int64, count=len(reached_bus_counts)),
    )

    bus_map_df = pd.DataFrame({"switch_id": switch_per_bus, "element": bus_uids[bus_positions]})

    # One row per (switch, branch terminal on a reachable bus).
    elements, sides, branches_per_bus = bus_branch_index.gather(bus_positions)
    branch_map_df = pd.DataFrame(
        {
            "switch_id": np.repeat(switch_per_bus, branches_per_bus),
            "element": elements,
            "side": sides,
        }
    )

    return branch_map_df, bus_map_df


def compute_current_a(
    p: np.ndarray,
    q: np.ndarray,
    vm_kv: np.ndarray,
) -> np.ndarray:
    """Compute the three-phase current magnitude from apparent power and voltage.

    Derives the current at a measurement point given active power, reactive power,
    and the local voltage magnitude. The apparent power is first reconstructed from
    ``p`` and ``q``, then divided by the three-phase voltage base to produce a
    current in amperes.

    The formula applied is::

        i [A] = sqrt(p**2 + q**2) / (sqrt(3) * vm_kv) * 1000

    Parameters
    ----------
    p : np.ndarray
        Active power in MW.
    q : np.ndarray
        Reactive power in Mvar.
    vm_kv : np.ndarray
        Voltage magnitude in kV at the measurement point.

    Returns
    -------
    np.ndarray
        Current magnitude in A.
    """
    denom = np.sqrt(3) * np.where(vm_kv == 0, np.nan, vm_kv)
    return np.sqrt(p**2 + q**2) / denom * 1000


def _compute_switch_flow_and_injection_results(
    branch_results: pl.DataFrame,
    node_results: pl.DataFrame,
    switch_element_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Compute aggregated switch results from branch flows and node injections.

    This function maps branch- and node-level load-flow results to switches using
    the provided switch-to-element mapping and aggregates them per switch.

    The computation is performed in three steps:

    1. Branch flow aggregation
       Branch results are matched to switch mappings by ``element`` and ``side``.
       Active and reactive power contributions are summed per switch.

    2. Node injection aggregation
       Node results are matched to switch mappings by ``element``.
       Active and reactive power contributions are summed per switch, while
       voltage magnitude ``vm`` is taken as the last value within each switch group.

    3. Final switch result computation
       Branch-flow and node-injection contributions are added together per switch.
       The apparent power ``s`` and current ``i`` are then computed as:

       - ``s = sqrt(p**2 + q**2)``
       - ``i`` is computed via :func:`compute_current_a` (result in A)

       Rows with ``vm == 0`` are removed before current calculation.

    Parameters
    ----------
    branch_results : pat.DataFrame[BranchResultSchema]
        Branch-level load-flow results. Expected to contain at least:
        - ``element``: globally unique branch element identifier
        - ``side``: branch side identifier
        - ``p``: active power contribution
        - ``q``: reactive power contribution

    node_results : pat.DataFrame[NodeResultSchema]
        Node-level load-flow results. Expected to contain at least:
        - ``element``: globally unique bus element identifier
        - ``p``: active power contribution
        - ``q``: reactive power contribution
        - ``vm``: voltage magnitude

    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements used for switch result
        aggregation. Contains both:
        - branch mappings with ``element`` and ``side``
        - bus mappings with ``element``

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per switch containing aggregated switch results.

        The result contains at least:
        - ``switch_id``: pandapower switch index
        - ``p``: aggregated active power
        - ``q``: aggregated reactive power
        - ``vm``: voltage magnitude used for current calculation
        - ``s``: apparent power
        - ``i``: current in A
    """
    # Branch- and mapping-side share a join key ``side`` which is integer on the branch
    # results but nullable float on the mapping (buses carry NaN); cast to a common float
    # so the polars join matches, mirroring pandas' int/float key up-casting.
    mapping = switch_element_mapping.with_columns(pl.col("side").cast(pl.Float64))

    # Branch flow aggregation: match branch results to the mapping on (element, side)
    # and sum active/reactive power per switch.
    if branch_results.height == 0:
        switch_flow = pl.DataFrame(schema={"switch_id": pl.Int64, "p": pl.Float64, "q": pl.Float64})
    else:
        switch_flow = (
            branch_results.select("element", pl.col("side").cast(pl.Float64), "p", "q")
            .join(mapping.select("switch_id", "element", "side"), on=["element", "side"], how="inner")
            .group_by("switch_id")
            .agg(pl.col("p").sum(), pl.col("q").sum())
        )

    # Node injection aggregation: match node results to the mapping on ``element`` only
    # (bus mapping rows), sum p/q per switch and take the last voltage magnitude in group
    # order. ``__row`` preserves the input node order so ``last`` matches the pandas result.
    switch_inj = (
        node_results.select("element", "p", "q", "vm")
        .with_row_index("__row")
        .join(mapping.select("switch_id", "element"), on="element", how="inner")
        .sort("__row")
        .group_by("switch_id", maintain_order=True)
        .agg(
            pl.col("p").sum(),
            pl.col("q").sum(),
            # pandas' groupby "last" skips NaN, so drop nulls before taking the last value.
            pl.col("vm").drop_nulls().last(),
        )
    )

    # Combine branch-flow and node-injection contributions per switch. A full join keeps
    # switches present in either source; missing contributions count as zero, matching the
    # pandas ``add(fill_value=0)`` step. ``vm`` only comes from injections, so a switch with
    # branch flow but no injection ends up with vm == 0 and is dropped below.
    combined = switch_inj.join(switch_flow, on="switch_id", how="full", suffix="_flow", coalesce=True)
    combined = combined.with_columns(
        (pl.col("p").fill_null(0.0) + pl.col("p_flow").fill_null(0.0)).alias("p"),
        (pl.col("q").fill_null(0.0) + pl.col("q_flow").fill_null(0.0)).alias("q"),
        pl.col("vm").fill_null(0.0).alias("vm"),
    ).select("switch_id", "p", "q", "vm")

    # ``i`` in amperes: s [MVA] / (sqrt(3) * vm [kV]) gives kA, the trailing * 1000 converts
    # to A (see ``compute_current_a``). vm == 0 means the switch has no slack connection.
    switch_results = (
        combined.with_columns((pl.col("p").pow(2) + pl.col("q").pow(2)).sqrt().alias("s"))
        .filter(pl.col("vm") != 0)
        .with_columns((pl.col("s") / (_SQRT3 * pl.col("vm")) * 1000).alias("i"))
    )

    return switch_results


def _orient_switch_results_to_relay_side(net: pp.pandapowerNet, switch_results: pl.DataFrame) -> pl.DataFrame:
    """Make switch power values use the relay point of view.

    Some relays measure from the bus side and some from the element side. For
    element-side relays, this flips active and reactive power signs so all rows
    can be compared in the same direction.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing switch metadata and relay characteristics.
    switch_results : pat.DataFrame[SwitchResultsSchema]
        Switch result table with ``switch_id``, ``p`` and ``q`` columns.

    Returns
    -------
    pat.DataFrame[SwitchResultsSchema]
        Switch results with active and reactive power flipped for relays that
        measure from the element side.
    """
    if "sw_characteristics" not in net or net.sw_characteristics.empty or "origin_id" not in net.switch.columns:
        return switch_results

    required_columns = {"breaker_uuid", "relay_side"}
    if not required_columns.issubset(net.sw_characteristics.columns):
        return switch_results

    # Resolve, from the (pandas) network metadata, which switch ids belong to relays that
    # measure from the element side. Their active/reactive power is flipped so every row is
    # expressed in the same (bus-side) direction.
    relay_side_by_origin = net.sw_characteristics.drop_duplicates(subset="breaker_uuid").set_index("breaker_uuid")[
        "relay_side"
    ]
    relay_sides = net.switch["origin_id"].map(relay_side_by_origin)
    element_side_switch_ids = relay_sides.index[relay_sides.eq("element")].to_list()

    return switch_results.with_columns(
        pl.when(pl.col("switch_id").is_in(element_side_switch_ids)).then(-pl.col("p")).otherwise(pl.col("p")).alias("p"),
        pl.when(pl.col("switch_id").is_in(element_side_switch_ids)).then(-pl.col("q")).otherwise(pl.col("q")).alias("q"),
    )


def get_switch_mapped_elements(
    net: pp.pandapowerNet,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    side: Literal["bus", "element"] = "bus",
) -> pat.DataFrame[SwitchElementMappingSchema]:
    """Build mapping between switches and electrically connected elements.

    This function determines, for each monitored closed switch, which buses
    and branch-like elements are electrically connected to one selected side
    of the switch.

    The mapping is used for switch result computation, where power flows and
    injections are aggregated over all elements connected to the switch.

    The connectivity is computed using a graph of closed bus-bus switches.
    For each switch, its own connection is temporarily ignored so that the
    traversal represents one side of the switch.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing buses, switches, and branch elements.
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        Table of monitored elements. Only switches whose ``monitoring_scope`` includes
        :attr:`SwitchMonitoringScope.FLOW` are included.
    side : Literal["bus", "element"]
        Defines from which side of the switch the traversal starts:

        - "bus": use ``sw.bus``
        - "element": use ``sw.element``

        This determines which connected component is used for aggregation
        and therefore directly affects computed switch results.

    Returns
    -------
    pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements.

        Each row represents either:
        - a branch-like element with a defined ``side``
        - a bus (with ``side = NaN``)
    """
    monitored_switches = monitored_elements[
        monitored_elements["monitoring_scope"].apply(lambda s: s is not None and SwitchMonitoringScope.FLOW in s)
    ]["table_id"].to_list()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(net, monitored_switches, side)

    result_df = pd.concat([branch_map_df, bus_map_df], ignore_index=True)

    # Ensure schema-compatible empty output
    if result_df.empty:
        return pd.DataFrame(
            {
                "switch_id": pd.Series(dtype="int64"),
                "element": pd.Series(dtype="object"),
                "side": pd.Series(dtype="float64"),
            }
        )

    # Ensure schema-compatible column set and dtypes also for non-empty output
    result_df = result_df.reindex(columns=["switch_id", "element", "side"])
    result_df = result_df.astype(
        {
            "switch_id": "int64",
            "element": "object",
            "side": "float64",
        }
    )

    return result_df


_SWITCH_RESULT_SCHEMA = {
    "timestep": pl.Int64,
    "contingency": pl.Utf8,
    "element": pl.Utf8,
    "switch_id": pl.Int64,
    "p": pl.Float64,
    "q": pl.Float64,
    "vm": pl.Float64,
    "s": pl.Float64,
    "i": pl.Float64,
    "element_name": pl.Utf8,
    "contingency_name": pl.Utf8,
    "side": pl.Utf8,
}


def _empty_switch_results_polars() -> pl.DataFrame:
    """Empty switch-result frame with the full output schema (used when nothing is mapped)."""
    return pl.DataFrame(schema=_SWITCH_RESULT_SCHEMA)


def _direct_switch_result_rows(net: pp.pandapowerNet, res_switch: pd.DataFrame, direct_ids: pd.Index) -> pl.DataFrame:
    """Build per-terminal (``side`` "from"/"to") result rows for impedance switches.

    These switches were solved directly by pandapower, so their from/to power and current
    are read from ``net.res_switch`` instead of aggregated. Two rows per switch.
    """
    rs = res_switch.loc[direct_ids]
    switch_ids = direct_ids.to_numpy()
    from_bus = net.switch.loc[direct_ids, "bus"].to_numpy()
    to_bus = net.switch.loc[direct_ids, "element"].to_numpy()
    vm_from = net.res_bus.loc[from_bus, "vm_pu"].to_numpy() * net.bus.loc[from_bus, "vn_kv"].to_numpy()
    vm_to = net.res_bus.loc[to_bus, "vm_pu"].to_numpy() * net.bus.loc[to_bus, "vn_kv"].to_numpy()

    def terminal_rows(p: np.ndarray, q: np.ndarray, vm: np.ndarray, side: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "switch_id": switch_ids,
                "p": p,
                "q": q,
                "vm": vm,
                "s": np.sqrt(p**2 + q**2),
                "i": compute_current_a(p, q, vm),
                "side": np.full(len(switch_ids), side),
            }
        )

    return pl.concat(
        [
            terminal_rows(rs["p_from_mw"].to_numpy(), rs["q_from_mvar"].to_numpy(), vm_from, "from"),
            terminal_rows(rs["p_to_mw"].to_numpy(), rs["q_to_mvar"].to_numpy(), vm_to, "to"),
        ],
        how="vertical",
    )


def get_switch_results(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    branch_results: pl.DataFrame,
    node_results: pl.DataFrame,
    switch_element_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Compute final switch-level results for a given contingency and timestep.

    This function aggregates branch flows and node injections per switch and
    enriches the results with metadata required for downstream processing.

    The computation consists of:
    - aggregating power flows and injections per switch using the provided
      switch-to-element mapping
    - computing apparent power and current (performed in the helper function)
    - attaching identifiers, names, and indexing information

    The whole computation runs on polars. Inputs and output are flat polars
    ``DataFrame`` objects (no index): ``branch_results`` and ``node_results`` carry the
    ``timestep``/``contingency``/``element`` (and ``side`` for branches) key values as
    ordinary columns, and callers holding pandas frames should convert with
    ``pl.from_pandas(df.reset_index())`` before calling.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing switch definitions and metadata.
        Used to map ``switch_id`` to human-readable switch names.
    contingency : PandapowerContingency
        Contingency for which the results are computed. Provides:
        - ``unique_id``: used as ``contingency`` column
        - ``name``: stored in ``contingency_name`` column
    timestep : int
        Timestep associated with the results.
    branch_results : pl.DataFrame
        Branch-level load-flow results (columns follow ``BranchResultSchema``,
        including ``element``, ``side``, ``p`` and ``q``).
    node_results : pl.DataFrame
        Node-level load-flow results (columns follow ``NodeResultSchema``,
        including ``element``, ``p``, ``q`` and ``vm``).
    switch_element_mapping : pl.DataFrame
         Mapping between switches and connected elements, used to compute
         switch-level results during each outage.

    Returns
    -------
    pl.DataFrame
        Switch-level results as a flat polars frame with the key columns

        - ``timestep``
        - ``contingency``
        - ``element`` (globally unique switch identifier)

        and the value columns:
        - aggregated active/reactive power (``p``, ``q``)
        - voltage magnitude (``vm``)
        - apparent power (``s``) and current (``i``)
        - the pandapower ``switch_id``
        - metadata columns (``element_name``, ``contingency_name``)
    """
    # Only closed switches carry a meaningful result; the mapping also contains open ones
    # (see _get_switch_mapped_elements_by_origin_ids) so they can be closed by SpPS later.
    closed_ids = net.switch.index[net.switch["closed"]]
    mapping_switch_ids = pd.Index(switch_element_mapping["switch_id"].unique().to_list())
    all_switch_ids = mapping_switch_ids[mapping_switch_ids.isin(closed_ids)]

    # Switches pandapower solved directly (modelled with impedance) have per-terminal
    # results in net.res_switch; everything else is derived from branch/node aggregation.
    res_switch = (
        net.res_switch[net.res_switch["p_from_mw"].notna()]
        if hasattr(net, "res_switch") and not net.res_switch.empty
        else pd.DataFrame()
    )
    direct_ids = (
        all_switch_ids[all_switch_ids.isin(res_switch.index)] if not res_switch.empty else pd.Index([], dtype="int64")
    )
    calc_ids = all_switch_ids[~all_switch_ids.isin(res_switch.index)]

    parts: list[pl.DataFrame] = []

    if len(direct_ids):
        # Direct switches get one row per terminal, tagged side "from"/"to".
        parts.append(_direct_switch_result_rows(net, res_switch, direct_ids))

    if len(calc_ids):
        calc_mapping = switch_element_mapping.filter(pl.col("switch_id").is_in(calc_ids.to_list()))
        calc_results = _compute_switch_flow_and_injection_results(
            branch_results=branch_results,
            node_results=node_results,
            switch_element_mapping=calc_mapping,
        )
        # Zero-impedance switches have identical conditions on both terminals: one row, no side.
        parts.append(calc_results.with_columns(pl.lit(None, dtype=pl.Utf8).alias("side")))

    if not parts:
        return _empty_switch_results_polars()

    switch_results = pl.concat(parts, how="vertical")
    switch_results = _orient_switch_results_to_relay_side(net, switch_results)

    # Look up switch display names from the (pandas) network metadata as a small polars
    # frame we can join on ``switch_id``.
    name_map = pl.from_pandas(
        pd.DataFrame(
            {
                "switch_id": net.switch.index.to_numpy(),
                "element_name": net.switch["name"].to_numpy(),
            }
        )
    )

    switch_results = (
        switch_results.with_columns(
            (pl.col("switch_id").cast(pl.Utf8) + pl.lit(f"{SEPARATOR}switch")).alias("element"),
            pl.lit(contingency.unique_id).alias("contingency"),
            pl.lit(contingency.name).alias("contingency_name"),
            pl.lit(timestep, dtype=pl.Int64).alias("timestep"),
        )
        .join(name_map, on="switch_id", how="left")
        .select(
            "timestep",
            "contingency",
            "element",
            "switch_id",
            "p",
            "q",
            "vm",
            "s",
            "i",
            "element_name",
            "contingency_name",
            "side",
        )
    )

    return switch_results


@pa.check_types
def get_failed_switch_results(
    timestep: int,
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    contingency: PandapowerContingency,
) -> pat.DataFrame[SwitchResultsSchema]:
    """Return switch results filled with NaN when the load flow fails.

    A result row is created for every monitored switch for the given timestep and
    contingency. Electrical result columns are filled with ``NaN`` and name columns
    are filled with empty strings.

    Parameters
    ----------
    timestep : int
        Timestep of the results.
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements, used to compute
        switch-level results during each outage.
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        Monitored elements table. Only rows with ``kind == "switch"`` are used.
    contingency : PandapowerContingency
        Contingency for which the failed results are created.

    Returns
    -------
    pat.DataFrame[SwitchResultsSchema]
        A dataframe indexed by ``timestep``, ``contingency``, and ``element``,
        containing one row per monitored switch. Result values are set to ``NaN``
        because no valid load-flow result is available.
    """
    monitored_closed_switches = get_globally_unique_id_from_index(
        switch_element_mapping["switch_id"].drop_duplicates(),
        "switch",
    )

    failed_switch_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [[timestep], [contingency.unique_id], monitored_closed_switches],
            names=["timestep", "contingency", "element"],
        ),
        data={
            "p": np.nan,
            "q": np.nan,
            "vm": np.nan,
            "i": np.nan,
            "element_name": "",
            "contingency_name": "",
            "side": None,
        },
    )

    return failed_switch_results
