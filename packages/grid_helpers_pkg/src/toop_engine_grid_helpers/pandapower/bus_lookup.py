# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build a bus lookup that merges buses connected through closed bus-bus switches."""

import numpy as np
import pandapower as pp
from scipy import sparse
from scipy.sparse.csgraph import connected_components as _scipy_connected_components


def _bus_components_from_switches(net: pp.pandapowerNet, bb_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Label the connected components of pp-bus indices joined by the (filtered) bus-bus switches.

    Labels the components with scipy rather than a Python union-find: on a net with tens of
    thousands of bus-bus switches the recursive ``find``/``union`` calls dominated
    :func:`assign_slack_per_island`, which runs this once per outage. Same partition, far
    faster. Matches the approach ``slack_allocation._fast_connected_components`` already uses.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    bb_mask : np.ndarray[bool]
        Boolean mask over net["switch"] rows selecting the closed, zero-impedance bus-bus switches.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(buses, component_ids)`` - two parallel arrays holding every bus that appears in one
        of the masked switches, and the id of the component it belongs to. Component ids are
        dense (``0 .. n_components - 1``). Returned as arrays rather than grouped containers so
        the caller can pick representatives without a Python loop.
    """
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if not np.any(bb_mask):
        return empty

    # Get the pp-bus indices connected by those switches
    fbus = net["switch"]["bus"].values[bb_mask].astype(np.int64)
    tbus = net["switch"]["element"].values[bb_mask].astype(np.int64)

    # Only buses touched by a switch can be fused, so label components over that compacted
    # set instead of the full bus range: `positions` indexes back into `touched`.
    touched, positions = np.unique(np.concatenate([fbus, tbus]), return_inverse=True)
    u_pos, v_pos = np.split(positions.ravel(), 2)
    adjacency = sparse.coo_matrix(
        (np.ones(len(u_pos)), (u_pos, v_pos)),
        shape=(len(touched), len(touched)),
    )
    _, component_ids = _scipy_connected_components(adjacency, directed=False)
    return touched, component_ids


def create_bus_lookup_simple(net: pp.pandapowerNet) -> tuple[list[int], list[bool]]:
    """
     Build a bus lookup that merges buses connected through closed bus-bus switches.

     - No PV/active-bus handling.
     - Merges transitively (chains of switches).

    Parameters
    ----------
    net : pp.pandapowerNet
         The pandapower network object.

    Returns
    -------
     tuple[list[int], list[bool]]
         - bus_lookup : list[int]
             Mapping from pandapower bus index to merged bus index.
         - merged_bus : list[bool]
             True for buses that were merged into another bus.
    """
    # Start from the "no-fuse" mapping you already use
    bus_index = list(net.bus.index)
    closed_bb_switch_mask = (
        net["switch"]["closed"].values
        & (net["switch"]["et"].values == "b")
        & np.isin(net["switch"]["bus"].values, bus_index)
        & np.isin(net["switch"]["element"].values, bus_index)
    )

    if len(bus_index) == 0:
        return [], []
    consec_buses = np.arange(len(bus_index), dtype=np.int64)
    bus_lookup = -np.ones(max(bus_index) + 1, dtype=np.int64)
    bus_lookup[bus_index] = consec_buses
    merged_bus = np.zeros(len(bus_lookup), dtype=bool)

    # Only consider closed bus-bus switches with zero impedance (same as your original mask)
    bb_mask = closed_bb_switch_mask & (net["switch"]["z_ohm"].values <= 0)

    # Early out if nothing to fuse
    if not np.any(bb_mask):
        return bus_lookup.tolist(), merged_bus.tolist()

    buses, component_ids = _bus_components_from_switches(net, bb_mask)

    # Order by component, and within a component by the bus's current lookup value, so the
    # first bus of every component is the representative the rest are merged into. The lookup
    # values are distinct at this point, so the representative is unambiguous.
    order = np.lexsort((bus_lookup[buses], component_ids))
    buses = buses[order]
    counts = np.bincount(component_ids)
    representatives = buses[np.concatenate(([0], np.cumsum(counts)[:-1]))]

    # Every bus takes its representative's lookup value. For the representative itself that is
    # a no-op, and it is excluded from merged_bus again below - which also covers the
    # single-bus components (a switch whose two ends are the same bus), left untouched as before.
    bus_lookup[buses] = np.repeat(bus_lookup[representatives], counts)
    merged_bus[buses] = True
    merged_bus[representatives] = False

    return bus_lookup.tolist(), merged_bus.tolist()
