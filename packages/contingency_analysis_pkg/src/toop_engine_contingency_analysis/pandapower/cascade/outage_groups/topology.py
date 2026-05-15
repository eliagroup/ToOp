# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Topology helpers for converting cascade triggers into outage groups."""

import logging

import networkx as nx
import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import PandapowerElements
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.contingency_outage_group import (
    _elements_in_component,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    build_connected_components_for_contingency_analysis,
    elem_node_id,
)
from toop_engine_grid_helpers.pandapower.pandapower_helpers import get_element_table
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id

_logger = logging.getLogger(__name__)


def apply_outages_in_service_flags(net: pp.pandapowerNet, outages: list[tuple[int, str]]) -> None:
    """Mark outaged elements as unavailable in the network.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to update.
    outages : list[tuple[int, str]]
        List of element ids and element table names, such as line or trafo.

    Returns
    -------
    None
        None. The network is modified in place.
    """
    for element_id, element_type in outages:
        try:
            table = net[element_type]
        except KeyError:
            _logger.debug("Skipping unknown pandapower element table %s", element_type)
            continue
        if "in_service" in table.columns:
            table.loc[element_id, "in_service"] = False


def create_closed_bb_switches_graph(net: pp.pandapowerNet) -> nx.Graph:
    """Build a simple graph of buses connected by closed bus-bus switches.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with buses and switches.

    Returns
    -------
    nx.Graph
        NetworkX graph where nodes are buses and edges are closed bus-bus switches.
    """
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)

    closed_busbus = net.switch[(net.switch.et == "b") & net.switch.closed]
    for _, sw in closed_busbus.iterrows():
        graph.add_edge(sw.bus, sw.element)

    return graph


def compute_affected_nodes(
    net: pp.pandapowerNet,
    el_list: pd.DataFrame,
) -> dict[int, list[int]]:
    """Find which buses are affected if each relay switch opens.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    el_list : pd.DataFrame
        Switch rows that include switch_id and relay_side columns.

    Returns
    -------
    dict[int, list[int]]
        Mapping from switch id to the list of buses on the affected side.
    """
    affected_nodes: dict[int, list[int]] = {}
    graph = create_closed_bb_switches_graph(net)
    for sw_row in el_list.itertuples():
        sw_idx = sw_row.switch_id
        relay_side = sw_row.relay_side

        sw = net.switch.loc[sw_idx]
        from_node = sw[relay_side]

        nodes: list[int] = []
        if graph.has_edge(sw.bus, sw.element):
            graph.remove_edge(sw.bus, sw.element)
        try:
            nodes = [int(n) for n in nx.node_connected_component(graph, from_node)]
        except nx.NetworkXError as e:
            _logger.error("compute_affected_nodes: %s, %s. Message: %s", sw.bus, sw.element, e)

        affected_nodes[sw_idx] = nodes

    return affected_nodes


def get_elements(net: pp.pandapowerNet, buses: list[int]) -> list[tuple[int, str]]:
    """Find in-service network elements connected to a set of buses.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    buses : list[int]
        Bus ids that define the affected area.

    Returns
    -------
    list
        List of element id and element type pairs connected to those buses.
    """
    if not buses:
        return []

    lines = net.line[net.line.from_bus.isin(buses) | net.line.to_bus.isin(buses)]
    lines = lines[lines["in_service"]]
    line_els = [(int(element_id), "line") for element_id in lines.index]

    trafos = net.trafo[net.trafo.lv_bus.isin(buses) | net.trafo.hv_bus.isin(buses)]
    trafos = trafos[trafos["in_service"]]
    trafo_els = [(int(element_id), "trafo") for element_id in trafos.index]

    trafo3w = net.trafo3w[net.trafo3w.lv_bus.isin(buses) | net.trafo3w.mv_bus.isin(buses) | net.trafo3w.hv_bus.isin(buses)]
    trafo3w = trafo3w[trafo3w["in_service"]]
    trafo3w_els = [(int(element_id), "trafo3w") for element_id in trafo3w.index]

    impedance = net.impedance[net.impedance.from_bus.isin(buses) | net.impedance.to_bus.isin(buses)]
    impedance = impedance[impedance["in_service"]]
    impedance_els = [(int(element_id), "impedance") for element_id in impedance.index]

    return list(set(line_els + trafo_els + trafo3w_els + impedance_els))


def get_busbars_couplers(net: pp.pandapowerNet, switch_origin_ids: list[str]) -> list[str]:
    """Find circuit breakers that connect two busbar areas.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    switch_origin_ids : list[str]
        External switch ids to consider.

    Returns
    -------
    list[str]
        External ids of switches that act as busbar couplers.
    """
    graph = create_closed_bb_switches_graph(net)
    switches = net.switch[(net.switch.origin_id.isin(switch_origin_ids)) & (net.switch.type == "CB")]
    res_busbars_couplers = []
    for sw in switches.itertuples():
        if sw.bus not in graph or sw.element not in graph:
            continue

        removed = False
        try:
            if graph.has_edge(sw.bus, sw.element):
                graph.remove_edge(sw.bus, sw.element)
                removed = True

            from_nodes = list(nx.node_connected_component(graph, sw.bus))
            to_nodes = list(nx.node_connected_component(graph, sw.element))

            from_busbar = net.bus.loc[from_nodes, "Busbar_id"]
            to_busbar = net.bus.loc[to_nodes, "Busbar_id"]

            from_has = from_busbar.replace("", pd.NA).notna().any()
            to_has = to_busbar.replace("", pd.NA).notna().any()

            if from_has and to_has:
                res_busbars_couplers.append(sw.origin_id)

        finally:
            if removed:
                graph.add_edge(sw.bus, sw.element)
    return res_busbars_couplers


def get_outage_group_for_elements(
    net: pp.pandapowerNet,
    contingency_elements: dict,
) -> dict:
    """Expand directly affected elements into full outage groups.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network used to compute connected components.
    contingency_elements : dict
        Mapping from trigger id to directly affected elements.

    Returns
    -------
    dict
        Mapping from trigger id to all elements in the same outage group.
    """
    connected_components = build_connected_components_for_contingency_analysis(net)
    node_to_component = {node: comp_idx for comp_idx, component in enumerate(connected_components) for node in component}

    res_elements = {}
    for sw_idx, elements in contingency_elements.items():
        sw_elements = []
        for element in elements:
            idx = int(element[0])
            etype = element[1]
            node_id = elem_node_id("elem", idx, etype)
            comp_idx = node_to_component[node_id]
            sw_elements.extend(_elements_in_component(connected_components, comp_idx))
        res_elements[sw_idx] = list(set(sw_elements))

    return res_elements


def pandapower_grid_element_from_network_outage(
    net: pp.pandapowerNet,
    table_id: int,
    pp_element_type: str,
) -> PandapowerElements | None:
    """Convert an outage tuple into the project's pandapower element model.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing the element.
    table_id : int
        Element id inside its pandapower table.
    pp_element_type : str
        Element type, such as bus, line, trafo, or trafo3w.

    Returns
    -------
    PandapowerElements | None
        PandapowerElements object, or None when the element type or id is not supported.
    """
    if pp_element_type not in ("bus", "line", "trafo", "trafo3w"):
        return None
    table = get_element_table(pp_element_type, res_table=False)
    tid = int(table_id)
    element_table = net[table]
    if tid not in element_table.index.values:
        return None

    name = ""
    if "name" in element_table.columns:
        nm = element_table.loc[tid, "name"]
        if pd.notna(nm):
            name = str(nm) or ""

    return PandapowerElements(
        unique_id=get_globally_unique_id(tid, pp_element_type),
        table=table,
        table_id=tid,
        name=name,
    )
