# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Utility functions for selecting and assigning slack generators."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import pandas as pd
from beartype.typing import Iterable, Optional
from pandapower.create import create_gen
from pandapower.toolbox.grid_modification import (
    _adapt_profiles_in_replace_functions,
    _adapt_result_tables_in_replace_functions,
    _replace_group_member_element_type,
)
from scipy import sparse
from scipy.sparse.csgraph import connected_components as _scipy_connected_components
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple


def _get_vm_pu_for_bus(net: pp.pandapowerNet, bus: np.int64, bus_lookup: list[int]) -> float:
    """
    Determine the vm_pu setpoint for a given bus.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    bus : int
        The bus index whose vm_pu value should be retrieved.
    bus_lookup : list[int]
        A list mapping graph node indices to pandapower bus indices.

    Returns
    -------
    float
        The vm_pu setpoint. Taken from measurement table if populated with load flow results
        from CGMES SV profile, otherwise from res_bus table if present, otherwise from
        gen or ext_grid entries on the same bus. Falls back to 1.0 pu.
    """
    # load flow results from CGMES SV profile
    if hasattr(net, "measurement") and not net.measurement.empty:
        required_cols = {"measurement_type", "element_type", "source", "element", "value"}
        if required_cols.issubset(net.measurement.columns):
            mask = (
                (net.measurement.measurement_type == "v")
                & (net.measurement.element_type == "bus")
                & (net.measurement.source == "SV")
                & (net.measurement.element == bus)
            )
            vals = net.measurement.loc[mask, "value"].unique()
            if len(vals) == 1:
                return float(vals[0])

    # load flow result from res_bus
    if bus in net.res_bus.index and net.converged:
        vm = net.res_bus.at[bus, "vm_pu"]
        if not np.isnan(vm):
            return float(vm)

    ppci_id = bus_lookup[bus]
    all_buses = [b_id for b_id, pp_id in enumerate(bus_lookup) if pp_id == ppci_id]

    # generator-defined setpoint
    if bus in net.gen.bus.values:
        vm = net.gen.loc[net.gen.bus.isin(all_buses), "vm_pu"].dropna()
        if not vm.empty:
            return float(vm.iloc[0])

    # external grid setpoint
    if bus in net.ext_grid.bus.values:
        vm = net.ext_grid.loc[net.ext_grid.bus.isin(all_buses), "vm_pu"].dropna()
        if not vm.empty:
            return float(vm.iloc[0])

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
    bus_lookup: list[int],
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
    bus_lookup : list[int]
        A list mapping graph node indices to pandapower bus indices.

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

    vm_pu = _get_vm_pu_for_bus(net, bus, bus_lookup)

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
    buses: set[int] = set()
    for table in (net.gen, net.sgen):
        if "referencePriority" in table.columns:
            buses |= set(table.loc[table["referencePriority"].fillna(0) > 0, "bus"].astype(int))
    return buses


def _labels_of_buses(label_of_bus: np.ndarray, buses: Iterable[int]) -> np.ndarray:
    """Component labels of *buses*, dropping buses that are out of service or unknown."""
    bus_ids = np.fromiter(buses, dtype=np.int64)
    bus_ids = bus_ids[(bus_ids >= 0) & (bus_ids < len(label_of_bus))]
    labels = label_of_bus[bus_ids]
    return labels[labels >= 0]


def select_slack_per_component(
    net: pp.pandapowerNet,
    label_of_bus: np.ndarray,
    valid_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank the (s)gens of every valid component and pick one slack candidate each.

    Ranking per component:

    1. Only (s)gens with a positive ``referencePriority`` are candidates.
    2. Lowest ``referencePriority`` wins.
    3. Among ties, the largest ``sn_mva`` wins; rows without ``sn_mva`` rank last.
    4. Still tied: the first row in table order, gens before sgens.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network object.
    label_of_bus : np.ndarray
        Dense ``bus id -> component label`` array, ``-1`` for buses without a component.
    valid_labels : np.ndarray
        Boolean mask over component labels; only these components receive a candidate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(labels, element_index, is_sgen)`` - one entry per component that had at least one
        candidate. ``element_index`` refers to ``net.gen`` or ``net.sgen`` according to ``is_sgen``.
    """
    labels, priorities, ratings, indices, sgen_flags = [], [], [], [], []
    for is_sgen, table in ((False, net.gen), (True, net.sgen)):
        if table.empty:
            continue
        priority = np.nan_to_num(table["referencePriority"].to_numpy(dtype=float), nan=0.0)
        bus = table["bus"].to_numpy(dtype=np.int64)
        in_range = (bus >= 0) & (bus < len(label_of_bus))
        label = np.full(len(bus), -1, dtype=np.int64)
        label[in_range] = label_of_bus[bus[in_range]]
        keep = (priority > 0) & (label >= 0)
        keep[keep] &= valid_labels[label[keep]]
        if not keep.any():
            continue
        rating = table["sn_mva"].to_numpy(dtype=float) if "sn_mva" in table.columns else np.full(len(bus), np.nan)

        labels.append(label[keep])
        priorities.append(priority[keep])
        ratings.append(rating[keep])
        indices.append(table.index.to_numpy()[keep])
        sgen_flags.append(np.full(int(keep.sum()), is_sgen))

    if not labels:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=bool)

    label = np.concatenate(labels)
    priority = np.concatenate(priorities)
    rating = np.concatenate(ratings)
    index = np.concatenate(indices)
    is_sgen = np.concatenate(sgen_flags)

    # Missing ratings sort after every real one; table position breaks the remaining ties.
    rating_key = np.where(np.isnan(rating), np.inf, -rating)
    order = np.lexsort((np.arange(len(label)), rating_key, priority, label))
    label, index, is_sgen = label[order], index[order], is_sgen[order]
    first = np.r_[True, label[1:] != label[:-1]]
    return label[first], index[first], is_sgen[first]


def _branch_in_service(
    table: pd.DataFrame,
    switch_et: str,
    sw_et: np.ndarray,
    sw_elem: np.ndarray,
    open_sw: np.ndarray,
) -> np.ndarray:
    """In-service mask for a branch table, minus branches interrupted by an open switch."""
    in_service = table.in_service.values.copy()
    interrupted = (sw_et == switch_et) & open_sw
    if interrupted.any():
        in_service &= ~np.isin(table.index.to_numpy(), sw_elem[interrupted])
    return in_service


def _trafo3w_edge_masks(
    net: pp.pandapowerNet,
    open_sw: np.ndarray,
    sw_et: np.ndarray,
    sw_elem: np.ndarray,
    sw_bus: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(u, v, in_service)`` for the three trafo3w side pairs.

    An open ``et == "t3"`` switch disables only the pairs that touch its bus, mirroring
    ``create_nxgraph``'s per-side handling.
    """
    trafo3w = net.trafo3w
    idx = trafo3w.index.to_numpy()
    t3_open_mask = (sw_et == "t3") & open_sw
    # Same (index + bus*1j) encoding create_nxgraph uses to key open t3 switches per side.
    open_t3 = (sw_elem[t3_open_mask] + sw_bus[t3_open_mask] * 1j) if t3_open_mask.any() else None

    edges = []
    for from_side, to_side in (("hv", "mv"), ("hv", "lv"), ("mv", "lv")):
        u = trafo3w[f"{from_side}_bus"].values
        v = trafo3w[f"{to_side}_bus"].values
        in_service = trafo3w.in_service.values.copy()
        if open_t3 is not None:
            in_service &= ~np.isin(idx + u * 1j, open_t3)
            in_service &= ~np.isin(idx + v * 1j, open_t3)
        edges.append((u, v, in_service))
    return edges


def _collect_topology_edges(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Return the bus-pair edge arrays of the graph ``create_nxgraph(net)`` would build.

    Covers the ``create_nxgraph`` defaults: in-service lines (minus those interrupted by
    an open ``et == "l"`` switch), impedances, dclines, trafos (minus open ``et == "t"``
    switches), all three trafo3w side pairs (minus open ``et == "t3"`` switches at either
    end), and closed bus-bus switches. Out-of-service *buses* are handled by the caller.
    """
    open_sw = ~net.switch.closed.values.astype(bool)
    sw_et = net.switch.et.values
    sw_elem = net.switch.element.values
    sw_bus = net.switch.bus.values

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if len(net.line):
        in_service = _branch_in_service(net.line, "l", sw_et, sw_elem, open_sw)
        candidates.append((net.line.from_bus.values, net.line.to_bus.values, in_service))

    if len(net.impedance):
        candidates.append(
            (net.impedance.from_bus.values, net.impedance.to_bus.values, net.impedance.in_service.values.astype(bool))
        )

    if "dcline" in net and len(net.dcline):
        candidates.append((net.dcline.from_bus.values, net.dcline.to_bus.values, net.dcline.in_service.values.astype(bool)))

    if len(net.trafo):
        in_service = _branch_in_service(net.trafo, "t", sw_et, sw_elem, open_sw)
        candidates.append((net.trafo.hv_bus.values, net.trafo.lv_bus.values, in_service))

    if len(net.trafo3w):
        candidates.extend(_trafo3w_edge_masks(net, open_sw, sw_et, sw_elem, sw_bus))

    if len(net.switch):
        candidates.append((sw_bus, sw_elem, (sw_et == "b") & ~open_sw))

    kept = [(u[mask], v[mask]) for u, v, mask in candidates if mask.any()]
    if not kept:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    return np.concatenate([u for u, _ in kept]), np.concatenate([v for _, v in kept])


@dataclass(frozen=True)
class BusComponents:
    """Connected-component label of every in-service bus.

    The partition is the one ``pandapower.topology.create_nxgraph(net)`` followed by
    ``nx.connected_components`` yields on the post-outage network.
    """

    bus_ids: np.ndarray
    """In-service pandapower bus indices."""
    labels: np.ndarray
    """Component label per entry of :attr:`bus_ids`, ``0 .. n_components - 1``."""

    @property
    def n_components(self) -> int:
        """Number of components."""
        return int(self.labels.max()) + 1 if len(self.labels) else 0

    def label_of_bus(self) -> np.ndarray:
        """Dense ``bus id -> label`` array, ``-1`` for buses that are not part of any component."""
        size = int(self.bus_ids.max()) + 1 if len(self.bus_ids) else 0
        label_of_bus = np.full(size, -1, dtype=np.int64)
        label_of_bus[self.bus_ids] = self.labels
        return label_of_bus

    @classmethod
    def from_sets(cls, components: Iterable[set[int]]) -> "BusComponents":
        """Build from the ``nx.connected_components`` representation."""
        chunks = [np.fromiter(component, dtype=np.int64) for component in components]
        if not chunks:
            empty = np.array([], dtype=np.int64)
            return cls(bus_ids=empty, labels=empty)
        labels = np.repeat(np.arange(len(chunks)), [len(chunk) for chunk in chunks])
        return cls(bus_ids=np.concatenate(chunks), labels=labels)


@dataclass(frozen=True)
class SlackAllocation:
    """Outcome of :func:`assign_slack_per_island`."""

    components: BusComponents
    """The bus components the slacks were allocated on."""
    slack_gen_by_label: dict[int, int]
    """``net.gen`` index of the slack per component label; components without a slack are absent."""


def _fast_connected_components(net: pp.pandapowerNet) -> Optional[BusComponents]:
    """Label the bus components exactly like ``create_nxgraph`` + ``nx.connected_components``.

    ``top.create_nxgraph`` inserts nodes and edges one by one through networkx's Python
    API, which dominates the cost of :func:`assign_slack_per_island` on large nets. This
    builds the same graph as flat edge arrays and labels components with scipy instead
    (verified to produce identical partitions, ~10-20x faster).

    Returns ``None`` when the net contains element types this fast path does not model
    (TCSC, VSC, DC lines); callers should then fall back to the networkx implementation.
    """
    for exotic_table in ("tcsc", "vsc", "line_dc"):
        if exotic_table in net and len(net[exotic_table]):
            return None

    bus_ids = net.bus.index.to_numpy()
    live_bus = pd.Index(bus_ids[net.bus.in_service.values.astype(bool)])

    edges_u, edges_v = _collect_topology_edges(net)
    u_pos = live_bus.get_indexer(edges_u)
    v_pos = live_bus.get_indexer(edges_v)
    # Edges touching an out-of-service bus vanish with the node, as in create_nxgraph.
    keep = (u_pos >= 0) & (v_pos >= 0)
    u_pos, v_pos = u_pos[keep], v_pos[keep]

    n_buses = len(live_bus)
    adjacency = sparse.coo_matrix((np.ones(len(u_pos)), (u_pos, v_pos)), shape=(n_buses, n_buses))
    _, labels = _scipy_connected_components(adjacency, directed=False)
    return BusComponents(bus_ids=live_bus.to_numpy(dtype=np.int64), labels=labels.astype(np.int64))


def bus_components(net: pp.pandapowerNet) -> BusComponents:
    """Label the connected bus components of the post-outage network, see :class:`BusComponents`."""
    components = _fast_connected_components(net)
    if components is None:
        # Net contains element types the fast path does not model; use networkx.
        components = BusComponents.from_sets(nx.connected_components(top.create_nxgraph(net)))
    return components


def assign_slack_per_island(
    net: pp.pandapowerNet,
    min_island_size: int,
    components: Optional[BusComponents] = None,
) -> Optional[SlackAllocation]:
    """
    Assign one slack generator per valid electrical island in the network.

    Deactivates all existing slack generators, then labels the bus components of *net*.
    Because outages are applied to *net* before this function is called (elements flagged
    ``in_service=False``, circuit breakers opened), the components already reflect the
    post-outage topology - no explicit edge removal is needed.

    For each valid island (more than *min_island_size* electrical buses after fusing
    closed bus-bus switches, at least one reference-capable generator, and at least two
    generating/load buses) a slack generator is selected via
    :func:`select_slack_per_component`. If the chosen candidate is an ``sgen``, it is
    promoted to a ``gen`` via :func:`replace_sgen_by_gen` before being marked as slack.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object.  Must be in its post-outage state so that the
        components reflect the correct topology.
    min_island_size : int
        Minimum number of buses required for an island to receive a slack bus.
    components : BusComponents, optional
        Precomputed components of *net*. Must be the partition :func:`bus_components`
        would return for the current state of *net*; pass it when the caller already has
        it so the connected-components pass is not repeated.

    Returns
    -------
    SlackAllocation or None
        The components and the chosen slack per component label. ``None`` when the network
        carries no ``referencePriority`` columns and is therefore left untouched.
    """
    if (not net.sgen.empty and "referencePriority" not in net.sgen.columns) or (
        not net.gen.empty and "referencePriority" not in net.gen.columns
    ):
        # This function requires 'referencePriority' columns in both sgen and gen tables.
        # Networks without these columns are not supported.
        return None
    bus_lookup = create_bus_lookup_simple(net)[0]
    # Deactivate all pre-allocated slacks
    net.gen["slack"] = False

    if components is None:
        components = bus_components(net)
    label_of_bus = components.label_of_bus()
    n_components = components.n_components

    # Island size counts electrical buses, i.e. after fusing closed bus-bus switches.
    fused = np.asarray(bus_lookup, dtype=np.int64)[components.bus_ids]
    fused_labels = np.unique(np.stack([components.labels, fused]), axis=1)[0]
    large_enough = np.bincount(fused_labels, minlength=n_components) > min_island_size

    reference_labels = _labels_of_buses(label_of_bus, get_buses_with_reference_sources(net))
    has_reference = np.bincount(reference_labels, minlength=n_components) > 0

    unit_labels = _labels_of_buses(label_of_bus, get_generating_units_with_load(net))
    has_two_units = np.bincount(unit_labels, minlength=n_components) >= 2

    labels, indices, is_sgen = select_slack_per_component(net, label_of_bus, large_enough & has_reference & has_two_units)

    slack_gen_by_label: dict[int, int] = {}
    for label, index, promote in zip(labels.tolist(), indices.tolist(), is_sgen.tolist(), strict=True):
        gen_index = replace_sgen_by_gen(net, sgen=index, bus_lookup=bus_lookup, retain_sgen_elm=True) if promote else index
        slack_gen_by_label[label] = int(gen_index)

    if slack_gen_by_label:
        net.gen.loc[list(slack_gen_by_label.values()), "slack"] = True
    return SlackAllocation(components=components, slack_gen_by_label=slack_gen_by_label)
