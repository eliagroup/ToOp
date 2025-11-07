"""Build a bus lookup that merges buses connected through closed bus-bus switches."""
import numpy as np
import pandapower as pp


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
    closed_bb_switch_mask = (net["switch"]["closed"].values &
                             (net["switch"]["et"].values == "b") &
                             np.isin(net["switch"]["bus"].values, bus_index) &
                             np.isin(net["switch"]["element"].values, bus_index))

    if len(bus_index) == 0:
        return [],[]
    consec_buses = np.arange(len(bus_index), dtype=np.int64)
    bus_lookup = -np.ones(max(bus_index) + 1, dtype=np.int64)
    bus_lookup[bus_index] = consec_buses

    # Only consider closed bus-bus switches with zero impedance (same as your original mask)
    bb_mask = closed_bb_switch_mask & (net["switch"]["z_ohm"].values <= 0)

    # Early out if nothing to fuse
    if not np.any(bb_mask):
        return bus_lookup.tolist(), np.zeros(len(bus_lookup), dtype=bool).tolist()

    # Get the pp-bus indices connected by those switches
    fbus = net["switch"]["bus"].values[bb_mask]
    tbus = net["switch"]["element"].values[bb_mask]

    # --- Union-Find (Disjoint Set) over the involved buses ---
    parent = {}

    def find(x):
        # path compression
        px = parent.setdefault(x, x)
        if px != x:
            parent[x] = find(px)
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in zip(fbus, tbus):
        union(a, b)

    # Collect components (only among buses that appeared in switches)
    comps = {}
    for b in parent:
        r = find(b)
        comps.setdefault(r, set()).add(b)

    # For each fused component, choose a representative and map all others to it
    merged_bus = np.zeros(len(bus_lookup), dtype=bool)

    for nodes in comps.values():
        if len(nodes) <= 1:
            continue

        # Choose a stable representative — the bus whose *current* bus_lookup is minimal
        rep_bus = min(nodes, key=lambda b: bus_lookup[b])
        target_ppc_bus = bus_lookup[rep_bus]

        for b in nodes:
            if b != rep_bus:
                bus_lookup[b] = target_ppc_bus
                merged_bus[b] = True

    return bus_lookup.tolist(), merged_bus.tolist()
