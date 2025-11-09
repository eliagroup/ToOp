import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pytest

from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_gen_by_weight
from toop_engine_grid_helpers.pandapower.slack_allocation import (
    get_generating_units_with_load,
    _slack_allocation_tie_break,
    get_buses_with_reference_sources,
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


def test_single_row_returns_index_and_etype():
    df = pd.DataFrame(
        {"etype": ["gen"], "sn_mva": [50.0]},
        index=[7],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert idx == 7
    assert etype == "gen"


def test_tie_prefers_highest_sn_mva():
    df = pd.DataFrame(
        {"etype": ["gen", "sgen", "gen"], "sn_mva": [10.0, 40.0, 30.0]},
        index=[1, 2, 3],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (2, "sgen")


def test_tie_with_partial_nan_sn_mva_uses_max_of_available():
    df = pd.DataFrame(
        {"etype": ["gen", "sgen", "gen"], "sn_mva": [float("nan"), 25.0, 25.0]},
        index=[10, 11, 12],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (11, "sgen")


def test_all_sn_mva_nan_falls_back_to_first_row():
    df = pd.DataFrame(
        {"etype": ["sgen", "gen"], "sn_mva": [float("nan"), float("nan")]},
        index=[5, 6],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (5, "sgen")


def test_same_sn_mva_pick_first_by_order():
    df = pd.DataFrame(
        {"etype": ["gen", "sgen", "gen"], "sn_mva": [30.0, 30.0, 30.0]},
        index=[101, 102, 103],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (101, "gen")


def test_duplicate_indices_allowed_pick_first_occurrence():
    df = pd.DataFrame(
        {"etype": ["gen", "sgen"], "sn_mva": [20.0, 20.0]},
        index=[9, 9],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (9, "gen")


def test_narrowing_to_single_row_after_max_sn_mva():
    df = pd.DataFrame(
        {"etype": ["gen", "sgen", "sgen"], "sn_mva": [5.0, 10.0, 3.0]},
        index=[21, 22, 23],
    )
    idx, etype = _slack_allocation_tie_break(df)
    assert (idx, etype) == (22, "sgen")


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
    with pytest.raises(ValueError):
        slack_allocation_module.replace_sgen_by_gen(net, -5)
    with pytest.raises(ValueError):
        slack_allocation_module.replace_sgen_by_gen(net, 999999)


def test_vm_pu_from_res_bus(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    net.res_bus = pd.DataFrame(index=net.bus.index, data={"vm_pu": 1.023})
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))

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
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))

    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-5) == 1.034


def test_vm_pu_from_ext_grid():
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net = _net_with_buses(2)
    b1 = net.bus.index[1]
    sgen_idx = pp.create_sgen(net, bus=b1, p_mw=1.0, name="SGen C", in_service=True)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))
    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-9) == 1.0


def test_vm_pu_default_when_no_sources():
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net = _net_with_buses(2)
    net.ext_grid.drop(index=net.ext_grid.index, inplace=True)
    b1 = net.bus.index[1]
    sgen_idx = pp.create_sgen(net, bus=b1, p_mw=1.0, name="SGen D", in_service=True)
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))
    assert pytest.approx(net.gen.at[new_idx, "vm_pu"], rel=1e-9) == 1.0


def test_cols_copied_and_slack_weight(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))

    for c in ["min_q_mvar", "max_q_mvar", "min_p_mw", "max_p_mw", "sn_mva", "referencePriority", "description"]:
        assert c in net.gen.columns, f"Column {c} should be added to net.gen"
        assert pd.notna(net.gen.at[new_idx, c])

    assert "slack_weight" in net.gen.columns
    assert pytest.approx(net.gen.at[new_idx, "slack_weight"], rel=1e-9) == net.gen.at[new_idx, "referencePriority"]


def test_retain_false_removes_original_sgen(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx), retain_sgen_elm=False)
    assert sgen_idx not in net.sgen.index
    assert new_idx in net.gen.index


def test_cost_tables_rewired(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    net.pwl_cost.loc[len(net.pwl_cost)] = {"et": "sgen", "element": sgen_idx}
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))
    for table in ("poly_cost", "pwl_cost"):
        # If the table has rows, ensure none still point to sgen/old index
        df = net[table]
        assert not ((df["et"] == "sgen") & (df["element"] == sgen_idx)).any()
        # And at least one rewired to gen/new_idx
        assert ((df["et"] == "gen") & (df["element"] == new_idx)).any()


def test_controllable_and_name_and_in_service_preserved(net_with_sgen):
    slack_allocation_module = __import__(IMPORT_PATH, fromlist=["*"])
    net, sgen_idx = net_with_sgen
    new_idx = slack_allocation_module.replace_sgen_by_gen(net, int(sgen_idx))
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


def test_unique_minimum_priority_picks_that_element(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    g_min = _add_gen(net4, b1, refp=1.0, sn_mva=5.0)
    _add_sgen(net4, b2, refp=3.0, sn_mva=50.0)
    _add_gen(net4, b3, refp=4.0, sn_mva=100.0)
    chosen_idx, etype = assign_slack_gen_by_weight(net4, set(map(np.int64, net4.bus.index)))
    assert chosen_idx == g_min
    assert etype == "gen"


def test_tie_between_gen_and_sgen_delegates_to_tiebreaker(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, _ = net4.bus.index
    _add_gen(net4, b1, refp=2.0, sn_mva=10.0)
    s = _add_sgen(net4, b2, refp=2.0, sn_mva=20.0)
    chosen_idx, etype = assign_slack_gen_by_weight(net4, {np.int64(b) for b in [b0, b1, b2]})
    assert chosen_idx == s
    assert etype == "sgen"


def test_filters_by_bus_set(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    _add_gen(net4, b3, refp=1.0, sn_mva=10.0)
    g_in = _add_gen(net4, b1, refp=1.5, sn_mva=20.0)
    _add_sgen(net4, b2, refp=3.0, sn_mva=30.0)
    chosen_idx, etype = assign_slack_gen_by_weight(net4, {np.int64(b) for b in [b0, b1, b2]})
    assert chosen_idx == g_in
    assert etype == "gen"


def test_non_positive_and_nan_priorities_are_excluded(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index

    _add_gen(net4, b0, refp=0.0, sn_mva=10.0)
    _add_sgen(net4, b1, refp=-1.0, sn_mva=20.0)
    _add_gen(net4, b2, refp=None, sn_mva=30.0)
    g_ok = _add_gen(net4, b3, refp=4.0, sn_mva=40.0)

    chosen_idx, etype = assign_slack_gen_by_weight(net4, {np.int64(b) for b in [b0, b1, b2, b3]})
    assert chosen_idx == g_ok
    assert etype == "gen"


def make_graph(n=4):
    """Create a simple chain graph 0-1-2-3."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return G


def test_clears_existing_slacks_and_assigns_new_one(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    g0 = pp.create_gen(net4, bus=b1, p_mw=1.0, vm_pu=1.0)
    net4.gen.at[g0, "slack"] = True
    net4.gen.at[g0, "referencePriority"] = 1.0

    s0 = pp.create_sgen(net4, bus=b2, p_mw=0.5)
    net4.sgen.at[s0, "referencePriority"] = 2.0

    G = make_graph()
    bus_lookup = list(net4.bus.index)  # [0,1,2,3]

    def fake_collect_edges(net, element_ids):
        return [(1, 2)]

    def fake_get_ref_buses(net):
        return {b1, b2}

    def fake_get_genload_buses(net):
        return {b0, b1, b2}

    def fake_assign_by_weight(net, cc):
        assert cc in ({0, 1}, {2, 3})
        if 1 in cc:
            return g0, "gen"
        return g0, "gen"

    def fake_replace_sgen(net, sgen, retain_sgen_elm=True):
        raise AssertionError("Should not be called in this test")

    mod = __import__(IMPORT_PATH, fromlist=["*"])
    monkeypatch.setattr(mod, "collect_element_edges", fake_collect_edges)
    monkeypatch.setattr(mod, "get_buses_with_reference_sources", lambda net: fake_get_ref_buses(net))
    monkeypatch.setattr(mod, "get_generating_units_with_load", lambda net: fake_get_genload_buses(net))
    monkeypatch.setattr(mod, "assign_slack_gen_by_weight", fake_assign_by_weight)
    monkeypatch.setattr(mod, "replace_sgen_by_gen", fake_replace_sgen)

    removed = mod.assign_slack_per_island(net4, G, bus_lookup, elements_ids=["dummy"], min_island_size=1)

    assert removed == [(1, 2)]
    assert not G.has_edge(1, 2)

    assert net4.gen["slack"].fillna(False).sum() == 1
    assert bool(net4.gen.at[g0, "slack"]) is True


def test_converts_sgen_then_sets_slack(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    s0 = pp.create_sgen(net4, bus=b2, p_mw=0.5)
    net4.sgen.at[s0, "referencePriority"] = 1.0

    G = make_graph()
    bus_lookup = list(net4.bus.index)

    def fake_collect_edges(net, element_ids):
        return []

    mod = __import__(IMPORT_PATH, fromlist=["*"])
    monkeypatch.setattr(mod, "collect_element_edges", fake_collect_edges)
    monkeypatch.setattr(mod, "get_buses_with_reference_sources", lambda net: {b2})
    monkeypatch.setattr(mod, "get_generating_units_with_load", lambda net: {b1, b2})

    monkeypatch.setattr(mod, "assign_slack_gen_by_weight", lambda net, cc: (s0, "sgen"))

    new_gen_idx = pp.create_gen(net4, bus=b2, p_mw=0.0, vm_pu=1.0)
    net4.gen.drop(new_gen_idx, inplace=True)

    def fake_replace_sgen(net, sgen, retain_sgen_elm=True):
        return new_gen_idx

    monkeypatch.setattr(mod, "replace_sgen_by_gen", fake_replace_sgen)

    removed = mod.assign_slack_per_island(net4, G, bus_lookup, elements_ids=[], min_island_size=1)
    assert removed == []

    assert new_gen_idx in net4.gen.index
    assert bool(net4.gen.at[new_gen_idx, "slack"]) is True


def test_filters_islands_by_min_size_and_candidates(monkeypatch):
    net4 = _net_with_buses(4)
    b0, b1, b2, b3 = net4.bus.index
    g0 = pp.create_gen(net4, bus=b0, p_mw=1.0, vm_pu=1.0)
    net4.gen.at[g0, "referencePriority"] = 1.0

    G = make_graph()
    bus_lookup = list(net4.bus.index)
    mod = __import__(IMPORT_PATH, fromlist=["*"])
    monkeypatch.setattr(mod, "collect_element_edges", lambda net, ids: [(1, 2)])
    monkeypatch.setattr(mod, "get_buses_with_reference_sources", lambda net: {b0})
    monkeypatch.setattr(mod, "get_generating_units_with_load", lambda net: {b0, b1})

    called = {"count": 0}

    def fake_assign(net, cc):
        called["count"] += 1
        assert cc == {0, 1}
        return g0, "gen"

    monkeypatch.setattr(mod, "assign_slack_gen_by_weight", fake_assign)
    monkeypatch.setattr(mod, "replace_sgen_by_gen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    mod.assign_slack_per_island(net4, G, bus_lookup, elements_ids=["x"], min_island_size=1)
    assert called["count"] == 1
    assert bool(net4.gen.at[g0, "slack"]) is True


@pytest.mark.xfail(reason="Function compares node indices to bus indices without mapping via bus_lookup")
def test_bus_vs_node_index_mismatch_exposes_filtering_bug(monkeypatch):
    net4 = _net_with_buses(4)
    net4.bus.drop(net4.bus.index, inplace=True)
    net = pp.create_empty_network()
    [pp.create_bus(net, vn_kv=20) for _ in range(4)]
    bus_lookup = [100, 101, 102, 103]

    G = make_graph()

    def fake_collect_edges(n, ids):
        return [(0, 1)]

    mod = __import__(IMPORT_PATH, fromlist=["*"])
    monkeypatch.setattr(mod, "collect_element_edges", fake_collect_edges)
    monkeypatch.setattr(mod, "get_buses_with_reference_sources", lambda n: {101})
    monkeypatch.setattr(mod, "get_generating_units_with_load", lambda n: {101, 102})

    called = {"hit": False}

    def fake_assign(*args, **kwargs):
        called["hit"] = True
        return 0, "gen"

    monkeypatch.setattr(mod, "assign_slack_gen_by_weight", fake_assign)
    monkeypatch.setattr(mod, "replace_sgen_by_gen", lambda *a, **k: 0)

    edges = mod.assign_slack_per_island(net, G, bus_lookup, elements_ids=["irrelevant"], min_island_size=1)
    assert edges == [(0, 1)]
    assert called["hit"] is False


def test_returns_empty_when_reference_priority_columns_missing(monkeypatch):
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20)
    pp.create_gen(net, bus=b0, p_mw=1.0, vm_pu=1.0)
    pp.create_sgen(net, bus=b0, p_mw=0.5)

    G = nx.Graph()
    G.add_nodes_from([0])
    bus_lookup = [int(b0)]
    mod = __import__(IMPORT_PATH, fromlist=["*"])
    edges = mod.assign_slack_per_island(net, G, bus_lookup, elements_ids=["x"], min_island_size=1)
    assert edges == []
