# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import pandas as pd
import pytest
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple
from toop_engine_grid_helpers.pandapower.example_grids import pandapower_extended_oberrhein
from toop_engine_grid_helpers.pandapower.slack_allocation import (
    BusComponents,
    SlackAllocation,
    assign_slack_per_island,
    bus_components,
    get_buses_with_reference_sources,
    get_generating_units_with_load,
    select_slack_per_component,
)

IMPORT_PATH = "toop_engine_grid_helpers.pandapower.slack_allocation"


def _net_with_buses(n=8):
    net = pp.create_empty_network()
    for _ in range(n):
        pp.create_bus(net, vn_kv=110.0)
    return net


def test_empty_network_returns_empty_set():
    net = _net_with_buses(3)
    result = get_generating_units_with_load(net)
    assert result == set()


def test_only_generators_included_across_types():
    net = _net_with_buses(6)
    pp.create_gen(net, bus=0, p_mw=1.0, vm_pu=1.0)
    pp.create_sgen(net, bus=1, p_mw=0.7, q_mvar=0.0)
    pp.create_ext_grid(net, bus=2, vm_pu=1.0)
    pp.create_ward(net, bus=3, ps_mw=0.0, qs_mvar=0.0, pz_mw=0.0, qz_mvar=0.0)
    pp.create_xward(net, bus=4, ps_mw=0.0, qs_mvar=0.0, vm_pu=1.0, pz_mw=0.0, qz_mvar=0.0, r_ohm=0.0, x_ohm=0.0)
    result = get_generating_units_with_load(net)
    assert result == {0, 1, 2, 3, 4}


def test_generators_and_loads_union_and_dedup():
    net = _net_with_buses(5)
    pp.create_gen(net, bus=1, p_mw=2.0, vm_pu=1.0)
    pp.create_load(net, bus=1, p_mw=1.0, q_mvar=0.0)
    pp.create_sgen(net, bus=2, p_mw=0.5, q_mvar=0.0)
    pp.create_load(net, bus=3, p_mw=0.4, q_mvar=0.0)

    result = get_generating_units_with_load(net)
    assert result == {1, 2, 3}


def test_multiple_elements_same_bus_only_once():
    net = _net_with_buses(4)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    pp.create_sgen(net, bus=0, p_mw=0.3, q_mvar=0.0)
    pp.create_load(net, bus=0, p_mw=0.2, q_mvar=0.0)
    result = get_generating_units_with_load(net)
    assert result == {0}


def test_resilient_when_some_tables_missing():
    net = _net_with_buses(4)
    pp.create_gen(net, bus=0, p_mw=1.0, vm_pu=1.0)
    pp.create_load(net, bus=1, p_mw=0.5, q_mvar=0.0)

    for key in ["ward", "xward", "sgen"]:
        if key in net:
            del net[key]

    result = get_generating_units_with_load(net)
    assert result == {0, 1}


def _single_component(net: pp.pandapowerNet) -> np.ndarray:
    """Every bus of *net* in component 0."""
    return np.zeros(int(net.bus.index.max()) + 1, dtype=np.int64)


def _pick(net: pp.pandapowerNet) -> tuple[int, str]:
    """The slack candidate of the single component, as ``(index, "gen" | "sgen")``."""
    labels, indices, is_sgen = select_slack_per_component(net, _single_component(net), np.array([True]))
    assert labels.tolist() == [0]
    return int(indices[0]), "sgen" if bool(is_sgen[0]) else "gen"


def _candidate_net(rows: list[tuple[str, float, float]]) -> pp.pandapowerNet:
    """One bus with one (s)gen per ``(etype, referencePriority, sn_mva)`` row, in order."""
    net = _net_with_buses(1)
    for etype, priority, sn_mva in rows:
        if etype == "gen":
            idx = pp.create_gen(net, bus=0, p_mw=1.0, vm_pu=1.0)
        else:
            idx = pp.create_sgen(net, bus=0, p_mw=0.5)
        net[etype].at[idx, "referencePriority"] = priority
        net[etype].at[idx, "sn_mva"] = sn_mva
    return net


def test_single_candidate_returns_index_and_etype() -> None:
    net = _candidate_net([("gen", 1.0, 50.0)])
    assert _pick(net) == (0, "gen")


def test_tie_prefers_highest_sn_mva() -> None:
    net = _candidate_net([("gen", 1.0, 10.0), ("sgen", 1.0, 40.0), ("gen", 1.0, 30.0)])
    assert _pick(net) == (0, "sgen")


def test_tie_with_partial_nan_sn_mva_uses_max_of_available() -> None:
    net = _candidate_net([("gen", 1.0, np.nan), ("sgen", 1.0, 25.0), ("gen", 1.0, 25.0)])
    # gens rank before sgens among the remaining ties
    assert _pick(net) == (1, "gen")


def test_all_sn_mva_nan_falls_back_to_first_row() -> None:
    net = _candidate_net([("sgen", 1.0, np.nan), ("gen", 1.0, np.nan)])
    # gens rank before sgens when nothing else separates them
    assert _pick(net) == (0, "gen")


def test_same_sn_mva_pick_first_by_order() -> None:
    net = _candidate_net([("gen", 1.0, 30.0), ("sgen", 1.0, 30.0), ("gen", 1.0, 30.0)])
    assert _pick(net) == (0, "gen")


def test_narrowing_to_single_row_after_max_sn_mva() -> None:
    net = _candidate_net([("gen", 1.0, 5.0), ("sgen", 1.0, 10.0), ("sgen", 1.0, 3.0)])
    assert _pick(net) == (0, "sgen")


def test_no_sn_mva_column_is_tolerated() -> None:
    net = _candidate_net([("gen", 2.0, np.nan), ("gen", 1.0, np.nan)])
    net.gen = net.gen.drop(columns="sn_mva")
    assert _pick(net) == (1, "gen")


@pytest.fixture()
def net_with_sgen():
    net = _net_with_buses(2)
    b1 = net.bus.index[1]
    sgen_idx = pp.create_sgen(
        net,
        bus=b1,
        p_mw=5.0,
        q_mvar=0.0,
        name="SGen A",
        in_service=True,
    )

    add_cols = {
        "min_q_mvar": -3.0,
        "max_q_mvar": 3.0,
        "min_p_mw": 0.0,
        "max_p_mw": 6.0,
        "sn_mva": 6.3,
        "referencePriority": 2.0,
        "controllable": True,
        "description": "demo",
    }
    for c, v in add_cols.items():
        if c not in net.sgen.columns:
            net.sgen[c] = pd.Series(dtype=type(v))
        net.sgen.at[sgen_idx, c] = v

    for table in ("poly_cost", "pwl_cost"):
        if table not in net or not isinstance(net[table], pd.DataFrame):
            net[table] = pd.DataFrame(columns=["et", "element"])
    net.poly_cost.loc[len(net.poly_cost)] = {"et": "sgen", "element": sgen_idx}

    return net, sgen_idx


def test_errors_on_invalid_index(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    bus_lookup, _ = create_bus_lookup_simple(net)
    with pytest.raises(ValueError):
        slack_allocation_module.replace_sgen_by_gen(net, -5, bus_lookup=bus_lookup)
    with pytest.raises(ValueError):
        slack_allocation_module.replace_sgen_by_gen(net, 999999, bus_lookup=bus_lookup)


def test_vm_pu_from_res_bus(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    net.res_bus = pd.DataFrame(index=net.bus.index, data={"vm_pu": 1.023})
    net.converged = True
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)

    assert new_idx in net.gen.index
    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-5) == 1.023
    assert bool(net.gen.at[new_idx, "replaced_sgen"]) is True
    assert bool(net.sgen.at[sgen_idx, "in_service"]) is False
    assert bool(net.sgen.at[sgen_idx, "replaced_by_gen"]) is True


def test_vm_pu_from_existing_gen_same_bus():
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net = _net_with_buses(2)
    b1 = net.bus.index[1]
    pp.create_gen(net, bus=b1, p_mw=0.0, vm_pu=1.034, name="Ref Gen")

    sgen_idx = pp.create_sgen(net, bus=b1, p_mw=2.0, name="SGen B", in_service=True)
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)

    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-5) == 1.034


def test_vm_pu_from_ext_grid():
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net = _net_with_buses(2)
    b1 = net.bus.index[1]
    sgen_idx = pp.create_sgen(net, bus=b1, p_mw=1.0, name="SGen C", in_service=True)
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)
    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-9) == 1.0


def test_vm_pu_default_when_no_sources():
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net = _net_with_buses(2)
    net.ext_grid.drop(index=net.ext_grid.index, inplace=True)
    b1 = net.bus.index[1]
    sgen_idx = pp.create_sgen(net, bus=b1, p_mw=1.0, name="SGen D", in_service=True)
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)
    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-9) == 1.0


def test_cols_copied_and_slack_weight(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)

    for c in ["min_q_mvar", "max_q_mvar", "min_p_mw", "max_p_mw", "sn_mva", "referencePriority", "description"]:
        assert c in net.gen.columns, f"Column {c} should be added to net.gen"
        assert pd.notna(net.gen.at[new_idx, c])

    assert "slack_weight" in net.gen.columns
    assert pytest.approx(net.gen.at[new_idx, "slack_weight"], rel=1e-9) == net.gen.at[new_idx, "referencePriority"]


def test_retain_false_removes_original_sgen(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), retain_sgen_elm=False, bus_lookup=bus_lookup)
    assert sgen_idx not in net.sgen.index
    assert new_idx in net.gen.index


def test_cost_tables_rewired(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    net.pwl_cost.loc[len(net.pwl_cost)] = {"et": "sgen", "element": sgen_idx}
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)
    for table in ("poly_cost", "pwl_cost"):
        # If the table has rows, ensure none still point to sgen/old index
        df = net[table]
        assert not ((df["et"] == "sgen") & (df["element"] == sgen_idx)).any()
        # And at least one rewired to gen/new_idx
        assert ((df["et"] == "gen") & (df["element"] == new_idx)).any()


def test_controllable_and_name_and_in_service_preserved(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    bus_lookup, _ = create_bus_lookup_simple(net)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), bus_lookup=bus_lookup)
    assert bool(net.gen.at[new_idx, "controllable"]) is True
    assert net.gen.at[new_idx, "name"] == net.sgen.at[sgen_idx, "name"]
    assert bool(net.sgen.at[sgen_idx, "in_service"]) is False


@pytest.fixture()
def net_with_refcol():
    net = pp.create_empty_network()
    for _ in range(5):
        pp.create_bus(net, vn_kv=20)

    # Ensure the column exists even when tables are empty
    if "referencePriority" not in net.gen.columns:
        net.gen["referencePriority"] = pd.Series(dtype=float)
    if "referencePriority" not in net.sgen.columns:
        net.sgen["referencePriority"] = pd.Series(dtype=float)

    return net


def test_empty_network_returns_empty_set_reference_sources(net_with_refcol):
    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == set()


def test_positive_priority_gen_included(net_with_refcol):
    b0 = net_with_refcol.bus.index[0]
    g = pp.create_gen(net_with_refcol, bus=b0, p_mw=1.0, vm_pu=1.0)
    net_with_refcol.gen.at[g, "referencePriority"] = 1.0
    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == {b0}


def test_positive_priority_sgen_included(net_with_refcol):
    b1 = net_with_refcol.bus.index[1]
    s = pp.create_sgen(net_with_refcol, bus=b1, p_mw=2.0)
    net_with_refcol.sgen.at[s, "referencePriority"] = 3.5
    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == {b1}


def test_zero_or_negative_priority_excluded(net_with_refcol):
    b0, b1, b2 = net_with_refcol.bus.index[:3]
    g0 = pp.create_gen(net_with_refcol, bus=b0, p_mw=1.0, vm_pu=1.0)
    net_with_refcol.gen.at[g0, "referencePriority"] = 0.0
    s0 = pp.create_sgen(net_with_refcol, bus=b1, p_mw=1.5)
    net_with_refcol.sgen.at[s0, "referencePriority"] = -2.0
    g1 = pp.create_gen(net_with_refcol, bus=b2, p_mw=0.5, vm_pu=1.0)
    net_with_refcol.gen.at[g1, "referencePriority"] = np.nan
    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == set()


def test_union_and_uniqueness_when_multiple_on_same_bus(net_with_refcol):
    b3 = net_with_refcol.bus.index[3]
    g0 = pp.create_gen(net_with_refcol, bus=b3, p_mw=1.0, vm_pu=1.0)
    net_with_refcol.gen.at[g0, "referencePriority"] = 2.0
    g1 = pp.create_gen(net_with_refcol, bus=b3, p_mw=0.5, vm_pu=1.0)
    net_with_refcol.gen.at[g1, "referencePriority"] = 5.0
    s0 = pp.create_sgen(net_with_refcol, bus=b3, p_mw=0.2)
    net_with_refcol.sgen.at[s0, "referencePriority"] = 1.0

    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == {b3}
    assert len(result) == 1


def test_mixed_gens_and_sgens_across_buses(net_with_refcol):
    b0, b1, b2, b4 = (
        net_with_refcol.bus.index[0],
        net_with_refcol.bus.index[1],
        net_with_refcol.bus.index[2],
        net_with_refcol.bus.index[4],
    )
    g_pos = pp.create_gen(net_with_refcol, bus=b0, p_mw=1.0, vm_pu=1.0)
    net_with_refcol.gen.at[g_pos, "referencePriority"] = 0.1
    g_zero = pp.create_gen(net_with_refcol, bus=b1, p_mw=1.0, vm_pu=1.0)
    net_with_refcol.gen.at[g_zero, "referencePriority"] = 0.0
    s_pos = pp.create_sgen(net_with_refcol, bus=b2, p_mw=2.2)
    net_with_refcol.sgen.at[s_pos, "referencePriority"] = 10.0
    s_nan = pp.create_sgen(net_with_refcol, bus=b4, p_mw=0.3)
    net_with_refcol.sgen.at[s_nan, "referencePriority"] = np.nan
    result = get_buses_with_reference_sources(net_with_refcol)
    assert result == {b0, b2}


def _add_gen(net, bus, refp=None, sn_mva=None, **kwargs):
    idx = pp.create_gen(net, bus=bus, p_mw=1.0, vm_pu=1.0, **kwargs)
    if refp is not None:
        net.gen.at[idx, "referencePriority"] = refp
    if sn_mva is not None:
        net.gen.at[idx, "sn_mva"] = sn_mva
    return idx


def _add_sgen(net, bus, refp=None, sn_mva=None, **kwargs):
    idx = pp.create_sgen(net, bus=bus, p_mw=0.5, **kwargs)
    if refp is not None:
        net.sgen.at[idx, "referencePriority"] = refp
    if sn_mva is not None:
        net.sgen.at[idx, "sn_mva"] = sn_mva
    return idx


def _labels(net: pp.pandapowerNet, *components: set[int]) -> np.ndarray:
    label_of_bus = np.full(int(net.bus.index.max()) + 1, -1, dtype=np.int64)
    for label, buses in enumerate(components):
        label_of_bus[list(buses)] = label
    return label_of_bus


def test_unique_minimum_priority_picks_that_element() -> None:
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    g_min = _add_gen(net4, b1, refp=1.0, sn_mva=5.0)
    _add_sgen(net4, b2, refp=3.0, sn_mva=50.0)
    _add_gen(net4, b3, refp=4.0, sn_mva=100.0)

    labels, indices, is_sgen = select_slack_per_component(net4, _labels(net4, {b0, b1, b2, b3}), np.array([True]))
    assert (labels.tolist(), indices.tolist(), is_sgen.tolist()) == ([0], [g_min], [False])


def test_tie_between_gen_and_sgen_uses_sn_mva() -> None:
    net4 = _net_with_buses(4)
    b0, b1, b2, _ = net4.bus.index
    _add_gen(net4, b1, refp=2.0, sn_mva=10.0)
    s = _add_sgen(net4, b2, refp=2.0, sn_mva=20.0)

    _, indices, is_sgen = select_slack_per_component(net4, _labels(net4, {b0, b1, b2}), np.array([True]))
    assert (indices.tolist(), is_sgen.tolist()) == ([s], [True])


def test_candidates_are_ranked_per_component() -> None:
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    g_other = _add_gen(net4, b3, refp=1.0, sn_mva=10.0)
    g_in = _add_gen(net4, b1, refp=1.5, sn_mva=20.0)
    _add_sgen(net4, b2, refp=3.0, sn_mva=30.0)

    labels, indices, _ = select_slack_per_component(net4, _labels(net4, {b0, b1, b2}, {b3}), np.array([True, True]))
    assert labels.tolist() == [0, 1]
    assert indices.tolist() == [g_in, g_other]


def test_only_valid_components_and_labelled_buses_get_a_candidate() -> None:
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    _add_gen(net4, b0, refp=1.0, sn_mva=10.0)
    g1 = _add_gen(net4, b1, refp=1.0, sn_mva=10.0)
    _add_gen(net4, b3, refp=1.0, sn_mva=10.0)  # b3 has no component label

    labels, indices, _ = select_slack_per_component(net4, _labels(net4, {b0}, {b1}, {b2}), np.array([False, True, True]))
    assert labels.tolist() == [1]
    assert indices.tolist() == [g1]


def test_non_positive_and_nan_priorities_are_excluded() -> None:
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index

    _add_gen(net4, b0, refp=0.0, sn_mva=10.0)
    _add_sgen(net4, b1, refp=-1.0, sn_mva=20.0)
    _add_gen(net4, b2, refp=None, sn_mva=30.0)
    g_ok = _add_gen(net4, b3, refp=4.0, sn_mva=40.0)

    _, indices, is_sgen = select_slack_per_component(net4, _labels(net4, {b0, b1, b2, b3}), np.array([True]))
    assert (indices.tolist(), is_sgen.tolist()) == ([g_ok], [False])


def test_no_candidates_returns_empty_arrays() -> None:
    net4 = _net_with_buses(4)
    labels, indices, is_sgen = select_slack_per_component(net4, _labels(net4, set(net4.bus.index)), np.array([True]))
    assert (len(labels), len(indices), len(is_sgen)) == (0, 0, 0)


def _add_chain_lines(net: pp.pandapowerNet) -> None:
    """Connect buses sequentially as a chain using line-from-parameters."""
    buses = list(net.bus.index)
    for from_bus, to_bus in zip(buses, buses[1:]):
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0,
            max_i_ka=1.0,
        )


def test_clears_existing_slacks_and_assigns_new_one() -> None:
    net4 = _net_with_buses(4)
    _add_chain_lines(net4)
    b0, b1, b2, _ = net4.bus.index
    g_old = _add_gen(net4, b1, refp=2.0)
    net4.gen.at[g_old, "slack"] = True
    g_new = _add_gen(net4, b2, refp=1.0)
    pp.create_load(net4, bus=b0, p_mw=1.0)

    allocation = assign_slack_per_island(net4, min_island_size=1)

    assert net4.gen["slack"].tolist() == [False, True]
    assert isinstance(allocation, SlackAllocation)
    assert allocation.slack_gen_by_label == {0: g_new}
    assert allocation.components.n_components == 1


def test_converts_sgen_then_sets_slack() -> None:
    net4 = _net_with_buses(4)
    _add_chain_lines(net4)
    _, b1, b2, _ = net4.bus.index
    s0 = _add_sgen(net4, b2, refp=1.0)
    pp.create_load(net4, bus=b1, p_mw=1.0)

    allocation = assign_slack_per_island(net4, min_island_size=1)

    (gen_idx,) = allocation.slack_gen_by_label.values()
    assert bool(net4.gen.at[gen_idx, "slack"]) is True
    assert net4.gen.at[gen_idx, "bus"] == b2
    assert bool(net4.sgen.at[s0, "in_service"]) is False


def test_filters_islands_by_min_size_and_candidates() -> None:
    net4 = _net_with_buses(4)
    _add_chain_lines(net4)
    b0, b1, b2, b3 = net4.bus.index
    g0 = _add_gen(net4, b0, refp=1.0)
    pp.create_load(net4, bus=b1, p_mw=1.0)
    # {b2, b3}: reference-capable but only one generating/load bus
    g2 = _add_gen(net4, b2, refp=1.0)

    # Disconnect b1-b2 so the chain splits into {b0, b1} and {b2, b3}.
    net4.line.at[net4.line.index[1], "in_service"] = False

    allocation = assign_slack_per_island(net4, min_island_size=1)

    assert list(allocation.slack_gen_by_label.values()) == [g0]
    assert bool(net4.gen.at[g0, "slack"]) is True
    assert bool(net4.gen.at[g2, "slack"]) is False

    label_of_bus = allocation.components.label_of_bus()
    assert label_of_bus[b0] == label_of_bus[b1] != label_of_bus[b2] == label_of_bus[b3]

    # Too small once fused buses are counted: min_island_size is exclusive.
    net4.gen["slack"] = True
    allocation = assign_slack_per_island(net4, min_island_size=2)
    assert allocation.slack_gen_by_label == {}
    assert not net4.gen["slack"].any()


def test_island_size_counts_fused_buses() -> None:
    net4 = _net_with_buses(4)
    _add_chain_lines(net4)
    b0, b1, b2, b3 = net4.bus.index
    g0 = _add_gen(net4, b0, refp=1.0)
    pp.create_load(net4, bus=b3, p_mw=1.0)
    # Fuse b0-b1 and b2-b3; the four-bus chain is two electrical buses.
    pp.create_switch(net4, bus=b0, element=b1, et="b", closed=True)
    pp.create_switch(net4, bus=b2, element=b3, et="b", closed=True)

    assert assign_slack_per_island(net4, min_island_size=1).slack_gen_by_label == {0: g0}
    assert assign_slack_per_island(net4, min_island_size=2).slack_gen_by_label == {}


def test_precomputed_components_are_used_as_given() -> None:
    net4 = _net_with_buses(4)
    _add_chain_lines(net4)
    b0, b1, b2, b3 = net4.bus.index
    g0 = _add_gen(net4, b0, refp=1.0)
    g2 = _add_gen(net4, b2, refp=1.0)
    pp.create_load(net4, bus=b1, p_mw=1.0)
    pp.create_load(net4, bus=b3, p_mw=1.0)

    components = BusComponents.from_sets([{b0, b1}, {b2, b3}])
    allocation = assign_slack_per_island(net4, min_island_size=1, components=components)

    assert allocation.components is components
    assert allocation.slack_gen_by_label == {0: g0, 1: g2}


def test_bus_components_matches_networkx() -> None:
    net = pandapower_extended_oberrhein()
    net.line.loc[net.line.index[::7], "in_service"] = False
    net.switch.loc[net.switch.index[::5], "closed"] = False
    net.bus.loc[net.bus.index[::11], "in_service"] = False

    components = bus_components(net)
    expected = list(nx.connected_components(top.create_nxgraph(net)))

    got = {frozenset(chunk.tolist()) for chunk in _split_by_label(components)}
    assert got == {frozenset(component) for component in expected}


def _split_by_label(components: BusComponents) -> list[np.ndarray]:
    order = np.argsort(components.labels, kind="stable")
    counts = np.bincount(components.labels, minlength=components.n_components)
    return np.split(components.bus_ids[order], np.cumsum(counts)[:-1])


def test_bus_components_from_sets_roundtrip() -> None:
    components = BusComponents.from_sets([{3, 1}, {7}, {5, 6}])
    assert components.n_components == 3
    label_of_bus = components.label_of_bus()
    assert label_of_bus.tolist() == [-1, 0, -1, 0, -1, 2, 2, 1]

    empty = BusComponents.from_sets([])
    assert empty.n_components == 0
    assert empty.label_of_bus().tolist() == []


def test_skips_allocation_when_reference_priority_columns_missing():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20)
    g0 = pp.create_gen(net, bus=b0, p_mw=1.0, vm_pu=1.0)
    pp.create_sgen(net, bus=b0, p_mw=0.5)
    net.gen.at[g0, "slack"] = True  # pre-existing slack should remain untouched

    mod = __import__(IMPORT_PATH, fromlist=["*"])
    mod.assign_slack_per_island(net, min_island_size=1)

    # Early return — gen.slack is not cleared
    assert bool(net.gen.at[g0, "slack"]) is True
