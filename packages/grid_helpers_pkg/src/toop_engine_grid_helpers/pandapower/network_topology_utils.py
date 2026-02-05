# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for executing N-1 contingency analysis on multiple network islands."""

from itertools import chain

import numpy as np
import pandapower as pp
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR


def _get_line_edges(net: pp.pandapowerNet, el_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    For a line element, return its edge as (from_bus, to_bus).

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    el_id : int
        ID of the line element.

    Returns
    -------
    list[tuple[int, int]]
        A single edge [(from_bus, to_bus)] for the specified line.
    """
    row = net.line.loc[el_id]
    return [(np.int64(row.from_bus), np.int64(row.to_bus))]


def _get_switch_edges(net: pp.pandapowerNet, el_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    For a switch element, return its edge as (from_bus, to_bus).

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    el_id : int
        ID of the switch element.

    Returns
    -------
    list[tuple[int, int]]
        A single edge [(from_bus, to_bus)] for the specified switch.
    """
    row = net.switch.loc[el_id]
    return [(np.int64(row.bus), np.int64(row.element))]


def _get_trafo_edges(net: pp.pandapowerNet, el_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    For a 2-winding transformer, return its edge as (hv_bus, lv_bus).

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object.
    el_id : int
        ID of the transformer element.

    Returns
    -------
    list[tuple[int, int]]
        A single edge [(hv_bus, lv_bus)] for the specified transformer.
    """
    row = net.trafo.loc[el_id]
    return [(np.int64(row.hv_bus), np.int64(row.lv_bus))]


def _get_trafo3w_edges(net: pp.pandapowerNet, el_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    For a 3-winding transformer, return edges between all three windings.

    Connections:
        hv <-> lv
        mv <-> lv
        hv <-> mv

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    el_id : int
        ID of the 3-winding transformer element.

    Returns
    -------
    list[tuple[int, int]]
        Edges connecting all transformer windings: [(hv, lv), (mv, lv), (hv, mv)].
    """
    row = net.trafo3w.loc[el_id]
    hv, mv, lv = np.int64(row.hv_bus), np.int64(row.mv_bus), np.int64(row.lv_bus)

    return [
        (hv, lv),
        (mv, lv),
        (hv, mv),
    ]


def _get_bus_edges(net: pp.pandapowerNet, bus_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    Get all edges connected to a given bus via closed switches.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network.
    bus_id : int
        ID of the target bus.

    Returns
    -------
    list[tuple[int, int]]
        Edges (from_bus, to_bus) connected to the given bus.
    """
    closed_switches = net.switch[net.switch.closed]
    switches = closed_switches[(closed_switches.element == bus_id) | (closed_switches.bus == bus_id)]
    switches_edges = list(chain.from_iterable((_get_switch_edges(net, el_id) for el_id in switches.index)))
    return switches_edges


def _edges_for_branch_element(net: pp.pandapowerNet, el_type: str, el_id: int) -> list[tuple[np.int64, np.int64]]:
    """
    Dispatch helper: given an element type and its ID, return its corresponding edges.

    Unknown element types raise a ValueError.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    el_type : str
        Type of the branch element ("line", "trafo", or "trafo3w").
    el_id : int
        Numeric ID of the element.

    Returns
    -------
    list[tuple[int, int]]
        List of (from_bus, to_bus) edges for the given element.
    """
    if el_type == "line":
        res = _get_line_edges(net, el_id)
    elif el_type == "trafo":
        res = _get_trafo_edges(net, el_id)
    elif el_type == "trafo3w":
        res = _get_trafo3w_edges(net, el_id)
    else:
        raise ValueError(f"Unknown element type: {el_type}")

    return res


def collect_element_edges(net: pp.pandapowerNet, elements_ids: list[str]) -> list[tuple[np.int64, np.int64]]:
    """
    Build a list of bus-to-bus edges touched by the given elements.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    elements_ids : list[str]
        List of element identifiers in the form "<id><SEPARATOR><type>"

    Returns
    -------
    list[tuple[int, int]]
        List of (from_bus, to_bus) edges corresponding to all given elements.
    """
    branch_edges = list()
    bus_edges = set()
    for element_id in elements_ids:
        el_id_str, el_type = element_id.split(SEPARATOR, 1)
        el_id = int(el_id_str)
        if el_type == "bus":
            bus_edges.update(_get_bus_edges(net, el_id))
        else:
            branch_edges += _edges_for_branch_element(net, el_type, el_id)

    return list(bus_edges) + branch_edges
