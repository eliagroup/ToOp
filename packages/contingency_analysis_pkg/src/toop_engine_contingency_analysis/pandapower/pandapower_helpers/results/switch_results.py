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
from beartype.typing import Literal
from pandera.typing import Series
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id,
    get_globally_unique_id_from_index,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import BranchResultSchema, BranchSide, NodeResultSchema, SwitchResultsSchema


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
    sw_els_map: list[tuple[int, str, int]] = []
    buses_map: list[tuple[int, str]] = []
    graph = create_closed_bb_switches_graph(net)
    # TODO: we need to calculate for open too, because in spps we can close  switches and then run cascading
    switch_mask = net.switch.index.isin(switches_ids)
    closed_mask = net.switch.closed

    switches = net.switch.loc[switch_mask & closed_mask]
    if switches.empty:
        return (
            pd.DataFrame(columns=["switch_id", "element", "side"]),
            pd.DataFrame(columns=["switch_id", "element"]),
        )

    bus_to_branch_map = _build_bus_to_branch_map(net)
    bus_uid_map = {int(bus): get_globally_unique_id(int(bus), "bus") for bus in net.bus.index}

    for sw in switches.itertuples():
        candidate_bus = sw.bus if side == "bus" else sw.element

        blocked_edge = (sw.bus, sw.element) if graph.has_edge(sw.bus, sw.element) else None
        reachable_buses = _connected_component_without_edge(graph, candidate_bus, blocked_edge)

        sw_els_map.extend(
            _get_elements_for_buses(
                switch_id=sw.Index,
                sw_buses=reachable_buses,
                bus_to_branch_map=bus_to_branch_map,
            )
        )
        buses_map.extend((sw.Index, bus_uid_map[int(bus)]) for bus in reachable_buses)

    branch_map_df = pd.DataFrame(sw_els_map, columns=["switch_id", "element", "side"])
    bus_map_df = pd.DataFrame(buses_map, columns=["switch_id", "element"])
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
    branch_results: pat.DataFrame[BranchResultSchema],
    node_results: pat.DataFrame[NodeResultSchema],
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
) -> pd.DataFrame:
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
    # Branch contribution
    if branch_results.empty:
        switch_flow = pd.DataFrame(
            {
                "switch_id": pd.Series(dtype="int64"),
                "p": pd.Series(dtype="float64"),
                "q": pd.Series(dtype="float64"),
            }
        )
    else:
        merged = branch_results.merge(
            switch_element_mapping,
            on=["element", "side"],
            how="inner",
        )
        switch_flow = merged.groupby("switch_id", as_index=False)[["p", "q"]].sum()
        switch_flow = switch_flow[["switch_id", "p", "q"]]

    merged = node_results.merge(switch_element_mapping, on="element", how="inner")
    merged = merged[["switch_id", "p", "q", "vm"]]
    switch_inj = merged.groupby(["switch_id"], as_index=False).agg({"p": "sum", "q": "sum", "vm": "last"})
    switch_inj = switch_inj[["switch_id", "p", "q", "vm"]]

    switch_inj = switch_inj.set_index(["switch_id"]).fillna(0)
    switch_flow = switch_flow.set_index(["switch_id"]).fillna(0)
    res = switch_inj.add(switch_flow, fill_value=0)
    switch_results = res.reset_index()

    switch_results["s"] = np.sqrt(switch_results["p"] ** 2 + switch_results["q"] ** 2)
    switch_results = switch_results[switch_results["vm"] != 0]
    switch_results["i"] = compute_current_a(
        switch_results["p"].values, switch_results["q"].values, switch_results["vm"].values
    )

    return switch_results


def _orient_switch_results_to_relay_side(
    net: pp.pandapowerNet, switch_results: pat.DataFrame[SwitchResultsSchema]
) -> pat.DataFrame[SwitchResultsSchema]:
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

    origin_ids = switch_results["switch_id"].map(net.switch["origin_id"])
    relay_side_by_origin = net.sw_characteristics.drop_duplicates(subset="breaker_uuid").set_index("breaker_uuid")[
        "relay_side"
    ]
    relay_sides = origin_ids.map(relay_side_by_origin)

    mask = relay_sides.eq("element")
    switch_results.loc[mask, ["p", "q"]] *= -1
    return switch_results


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
        Table of monitored elements. Only rows with ``kind == "switch"`` or ``kind == "switch_relay"`` are used.
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
    monitored_switches = monitored_elements.query("kind in ('switch', 'switch_relay')")["table_id"].to_list()

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


def get_switch_results(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    branch_results: pat.DataFrame[BranchResultSchema],
    node_results: pat.DataFrame[NodeResultSchema],
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
) -> pat.DataFrame[SwitchResultsSchema]:
    """Compute final switch-level results for a given contingency and timestep.

    This function aggregates branch flows and node injections per switch and
    enriches the results with metadata required for downstream processing.

    The computation consists of:
    - aggregating power flows and injections per switch using the provided
      switch-to-element mapping
    - computing apparent power and current (performed in the helper function)
    - attaching identifiers, names, and indexing information

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing switch definitions and metadata.
        Used to map ``switch_id`` to human-readable switch names.
    contingency : PandapowerContingency
        Contingency for which the results are computed. Provides:
        - ``unique_id``: used as index level
        - ``name``: stored in ``contingency_name`` column
    timestep : int
        Timestep associated with the results.
    branch_results : pat.DataFrame[BranchResultSchema]
        Branch-level load-flow results.
    node_results : pat.DataFrame[NodeResultSchema]
        Node-level load-flow results.
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
         Mapping between switches and connected elements, used to compute
         switch-level results during each outage.

    Returns
    -------
    pat.DataFrame[SwitchResultsSchema]
        Switch-level results indexed by:

        - ``timestep``
        - ``contingency``
        - ``element`` (globally unique switch identifier)

        The DataFrame includes:
        - aggregated active/reactive power (``p``, ``q``)
        - voltage magnitude (``vm``)
        - current (``i``)
        - apparent power (``s``)
        - metadata columns (``element_name``, ``contingency_name``)
    """
    closed_ids = net.switch.index[net.switch["closed"]]
    all_switch_ids = switch_element_mapping["switch_id"].drop_duplicates()
    all_switch_ids = all_switch_ids[all_switch_ids.isin(closed_ids)]
    res_switch = (
        net.res_switch[net.res_switch["p_from_mw"].notna()]
        if hasattr(net, "res_switch") and not net.res_switch.empty
        else pd.DataFrame()
    )

    parts: list[pd.DataFrame] = []

    direct_ids = (
        all_switch_ids[all_switch_ids.isin(res_switch.index)] if not res_switch.empty else pd.Series([], dtype="int64")
    )
    calc_ids = all_switch_ids[~all_switch_ids.isin(res_switch.index)]

    if not direct_ids.empty:
        rs = res_switch.loc[direct_ids]
        from_bus_ids = net.switch.loc[direct_ids, "bus"]
        to_bus_ids = net.switch.loc[direct_ids, "element"]
        vm_from_kv = net.res_bus.loc[from_bus_ids.values, "vm_pu"].values * net.bus.loc[from_bus_ids.values, "vn_kv"].values
        vm_to_kv = net.res_bus.loc[to_bus_ids.values, "vm_pu"].values * net.bus.loc[to_bus_ids.values, "vn_kv"].values
        p_from = rs["p_from_mw"].values
        q_from = rs["q_from_mvar"].values
        p_to = rs["p_to_mw"].values
        q_to = rs["q_to_mvar"].values
        i_from_a = compute_current_a(p_from, q_from, vm_from_kv)
        i_to_a = compute_current_a(p_to, q_to, vm_to_kv)
        from_rows = pd.DataFrame(
            {
                "switch_id": direct_ids.values,
                "p": p_from,
                "q": q_from,
                "vm": vm_from_kv,
                "i": i_from_a,
                "side": "from",
            }
        )
        to_rows = pd.DataFrame(
            {
                "switch_id": direct_ids.values,
                "p": p_to,
                "q": q_to,
                "vm": vm_to_kv,
                "i": i_to_a,
                "side": "to",
            }
        )
        parts.extend([from_rows, to_rows])

    if not calc_ids.empty:
        calc_mapping = switch_element_mapping[switch_element_mapping["switch_id"].isin(calc_ids)]
        calc_results = _compute_switch_flow_and_injection_results(
            branch_results=branch_results,
            node_results=node_results,
            switch_element_mapping=calc_mapping,
        )
        calc_results["side"] = None
        parts.append(calc_results)

    if not parts:
        return get_empty_dataframe_from_model(SwitchResultsSchema)

    switch_results = pd.concat(parts, ignore_index=True)

    switch_results = _orient_switch_results_to_relay_side(net, switch_results)
    switch_results["element"] = get_globally_unique_id_from_index(switch_results["switch_id"], element_type="switch")
    switch_results["element_name"] = switch_results["switch_id"].map(net.switch["name"])
    switch_results["contingency"] = contingency.unique_id
    switch_results["contingency_name"] = contingency.name
    switch_results["timestep"] = timestep
    switch_results.set_index(["timestep", "contingency", "element"], inplace=True)

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
