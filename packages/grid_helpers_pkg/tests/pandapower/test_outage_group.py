# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


def _df(rows):
    """Helper to build a switch dataframe with required columns."""
    return pd.DataFrame(rows, columns=["bus", "element", "closed", "type"])


def test_empty_df_returns_empty_schema():
    sw = pd.DataFrame(columns=["bus", "element", "closed", "type"])
    out = aggregate_switch_pairs(sw)
    assert list(out.columns) == ["u", "v", "closed_non_cb"]
    assert out.empty


def test_builds_unordered_pairs_min_max():
    sw = _df(
        [
            (10, 3, True, "DS"),  # unordered -> (3,10)
            (3, 10, False, "DS"),  # same pair -> (3,10)
        ]
    )
    out = aggregate_switch_pairs(sw)

    assert len(out) == 1
    assert out.loc[0, "u"] == 3
    assert out.loc[0, "v"] == 10


def test_aggregates_any_for_closed_non_cb_true_if_any_non_cb_closed():
    sw = _df(
        [
            (1, 3, False, "DS"),
            (3, 1, True, "DS"),  # same pair, closed non-CB exists
            (1, 3, False, "DS"),
        ]
    )
    out = aggregate_switch_pairs(sw)

    assert len(out) == 1
    assert out.loc[0, "closed_non_cb"] is True or bool(out.loc[0, "closed_non_cb"]) is True


def test_cb_switches_do_not_set_closed_non_cb():
    sw = _df(
        [
            (1, 2, True, "CB"),  # closed but CB => should NOT count
            (2, 1, False, "CB"),
        ]
    )
    out = aggregate_switch_pairs(sw)

    assert len(out) == 1
    assert out.loc[0, "u"] == 1
    assert out.loc[0, "v"] == 2
    assert bool(out.loc[0, "closed_non_cb"]) is False


def test_mixed_cb_and_non_cb_counts_only_non_cb_for_closed_non_cb():
    sw = _df(
        [
            (5, 9, True, "CB"),  # closed CB doesn't matter for closed_non_cb
            (9, 5, False, "DS"),  # non-CB open
        ]
    )
    out = aggregate_switch_pairs(sw)

    assert len(out) == 1
    assert bool(out.loc[0, "closed_non_cb"]) is False

    sw2 = _df(
        [
            (5, 9, True, "CB"),
            (9, 5, True, "DS"),  # non-CB closed => now True
        ]
    )
    out2 = aggregate_switch_pairs(sw2)
    assert bool(out2.loc[0, "closed_non_cb"]) is True


def test_multiple_pairs_and_no_sort_preserves_first_seen_pair_order():
    # With sort=False, group order should follow first appearance in data (typical pandas behavior).
    sw = _df(
        [
            (10, 11, False, "DS"),  # pair (10,11) appears first
            (1, 9, True, "DS"),  # pair (1,9) appears second
            (11, 10, True, "DS"),  # still (10,11)
            (9, 1, False, "DS"),  # still (1,9)
        ]
    )
    out = aggregate_switch_pairs(sw)

    assert len(out) == 2

    # Check order (10,11) then (1,9)
    assert (out.loc[0, "u"], out.loc[0, "v"]) == (10, 11)
    assert (out.loc[1, "u"], out.loc[1, "v"]) == (1, 9)

    # Check aggregated values
    # (10,11): any(False, True) => True
    assert bool(out.loc[0, "closed_non_cb"]) is True
    # (1,9): any(True, False) => True
    assert bool(out.loc[1, "closed_non_cb"]) is True


def test_output_dtypes_u_v_are_int():
    sw = _df(
        [
            (1.0, 2.0, True, "DS"),
        ]
    )
    out = aggregate_switch_pairs(sw)
    assert np.issubdtype(out["u"].dtype, np.integer)
    assert np.issubdtype(out["v"].dtype, np.integer)


def make_net_with_switches(switch_df: pd.DataFrame) -> pp.pandapowerNet:
    """Create a real pandapowerNet with a predefined switch table."""
    net = pp.create_empty_network()
    net.switch = switch_df.copy()
    return net


def test_returns_empty_schema_when_switch_is_none():
    net = pp.create_empty_network()
    net.switch = None
    out = preprocess_bus_bus_switches(net)
    assert list(out.columns) == ["bus", "element", "type", "closed"]
    assert out.empty


def test_returns_empty_schema_when_switch_is_empty_df():
    net = pp.create_empty_network()
    net.switch = pd.DataFrame()
    out = preprocess_bus_bus_switches(net)
    assert list(out.columns) == ["bus", "element", "type", "closed"]
    assert out.empty


def test_filters_only_bus_bus_when_et_present():
    sw = pd.DataFrame(
        {
            "bus": [1, 2, 3],
            "element": [10, 20, 30],
            "type": ["CB", "DS", "CB"],
            "closed": [True, False, True],
            "et": ["b", "l", "b"],  # keep only 'b'
        }
    )
    net = make_net_with_switches(sw)
    out = preprocess_bus_bus_switches(net)

    assert len(out) == 2
    assert set(out["bus"].tolist()) == {1, 3}
    assert list(out.columns) == ["bus", "element", "type", "closed"]


def test_when_et_present_but_no_b_rows_returns_empty_schema():
    sw = pd.DataFrame(
        {
            "bus": [1, 2],
            "element": [10, 20],
            "type": ["CB", "DS"],
            "closed": [True, False],
            "et": ["l", "t"],  # none 'b'
        }
    )
    net = make_net_with_switches(sw)
    out = preprocess_bus_bus_switches(net)

    assert list(out.columns) == ["bus", "element", "type", "closed"]
    assert out.empty


def test_does_not_filter_if_et_missing():
    sw = pd.DataFrame(
        {
            "bus": [1, 2],
            "element": [10, 20],
            "type": ["CB", "DS"],
            "closed": [True, False],
            # no 'et' column
        }
    )
    net = make_net_with_switches(sw)
    out = preprocess_bus_bus_switches(net)

    assert len(out) == 2
    assert out["bus"].tolist() == [1, 2]
    assert out["element"].tolist() == [10, 20]


def test_returns_only_normalized_columns_in_order():
    sw = pd.DataFrame(
        {
            "bus": [1],
            "element": [2],
            "type": ["CB"],
            "closed": [True],
            "et": ["b"],
            "foo": ["bar"],  # extra column should be dropped
        }
    )
    net = make_net_with_switches(sw)
    out = preprocess_bus_bus_switches(net)

    assert list(out.columns) == ["bus", "element", "type", "closed"]
    assert "foo" not in out.columns


def test_missing_closed_raises_keyerror_current_behavior():
    """
    Docstring says: Missing 'closed' -> False.
    Current implementation does NOT do that; it raises KeyError.
    This test locks in current behavior so changes are explicit.
    """
    sw = pd.DataFrame(
        {
            "bus": [1],
            "element": [2],
            "type": ["CB"],
            "et": ["b"],
        }
    )
    net = make_net_with_switches(sw)

    with pytest.raises(KeyError):
        preprocess_bus_bus_switches(net)


def test_empty_df_returns_empty_list():
    agg = pd.DataFrame(columns=["u", "v", "closed_non_cb"])
    assert get_traversable_bus_bus_pairs(agg) == []


def test_returns_only_rows_where_closed_non_cb_true():
    agg = pd.DataFrame(
        {
            "u": [1, 2, 3],
            "v": [10, 20, 30],
            "closed_non_cb": [True, False, True],
        }
    )
    out = get_traversable_bus_bus_pairs(agg)
    assert out == [(1, 10), (3, 30)]


def test_casts_to_int_even_if_input_is_float_or_numpy_types():
    agg = pd.DataFrame(
        {
            "u": [1.0, 2.0],
            "v": [10.0, 20.0],
            "closed_non_cb": [True, True],
        }
    )
    out = get_traversable_bus_bus_pairs(agg)
    assert out == [(1, 10), (2, 20)]
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in out)


def test_preserves_row_order_of_filtered_result():
    agg = pd.DataFrame(
        {
            "u": [5, 1, 9],
            "v": [6, 2, 10],
            "closed_non_cb": [False, True, True],
        }
    )
    out = get_traversable_bus_bus_pairs(agg)
    assert out == [(1, 2), (9, 10)]


def test_all_false_returns_empty_list():
    agg = pd.DataFrame({"u": [1, 2], "v": [3, 4], "closed_non_cb": [False, False]})
    assert get_traversable_bus_bus_pairs(agg) == []


def test_missing_column_closed_non_cb_raises_keyerror():
    agg = pd.DataFrame({"u": [1], "v": [2]})
    with pytest.raises(KeyError):
        get_traversable_bus_bus_pairs(agg)


def _element_nodes(graph: nx.Graph, bus_nodes: set[str]) -> set:
    """
    Heuristic: bus nodes are strs equal to pandapower bus indices.
    Element nodes are everything else (tuples/strings/etc.).
    """
    return {n for n in graph.nodes if n not in bus_nodes}


def test_add_elements_bipartite_line():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1, std_type="NAYY 4x50 SE")

    g = nx.Graph()
    add_elements_bipartite(net, g, tables=[("line", "line")])

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert bus_nodes == {"b_0", "b_1"}
    assert len(_element_nodes(g, bus_nodes)) == len(net.line)


def test_add_elements_bipartite_impedance():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=20.0)

    pp.create_impedance(net, from_bus=b0, to_bus=b1, rft_pu=0.01, xft_pu=0.05, sn_mva=10.0)

    g = nx.Graph()
    add_elements_bipartite(net, g, tables=[("impedance", "impedance")])

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert len(_element_nodes(g, bus_nodes)) == len(net.impedance)


def test_add_elements_bipartite_trafo():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.0)
    lv = pp.create_bus(net, vn_kv=20.0)

    pp.create_transformer_from_parameters(
        net,
        hv_bus=hv,
        lv_bus=lv,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=20.0,
        vk_percent=10.0,
        vkr_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
    )

    g = nx.Graph()
    add_elements_bipartite(net, g, tables=[("trafo", "trafo")])

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert len(_element_nodes(g, bus_nodes)) == len(net.trafo)


def test_add_elements_bipartite_trafo3w():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.0)
    mv = pp.create_bus(net, vn_kv=20.0)
    lv = pp.create_bus(net, vn_kv=10.0)

    pp.create_transformer3w_from_parameters(
        net,
        hv_bus=hv,
        mv_bus=mv,
        lv_bus=lv,
        sn_hv_mva=25.0,
        sn_mv_mva=15.0,
        sn_lv_mva=10.0,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
    )

    g = nx.Graph()
    add_elements_bipartite(net, g, tables=[("trafo3w", "trafo3w")])

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert len(_element_nodes(g, bus_nodes)) == len(net.trafo3w)


def test_add_elements_bipartite_single_bus_element_bus_column():
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=10.0)
    pp.create_load(net, bus=b, p_mw=5.0, q_mvar=1.0)

    g = nx.Graph()
    add_elements_bipartite(net, g, tables=[("load", "load")])

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert len(_element_nodes(g, bus_nodes)) == len(net.load)


def _create_net_all_types() -> pp.pandapowerNet:
    """
    Create a real pandapowerNet containing:
      - line
      - impedance
      - trafo (2-winding)
      - trafo3w (3-winding)
      - a single-bus element (load)
    """
    net = pp.create_empty_network()

    # Buses
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=20.0)
    b3 = pp.create_bus(net, vn_kv=10.0)

    # line: from_bus/to_bus
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.2,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )

    # impedance: from_bus/to_bus
    # (pandapower impedance is per unit; pick simple valid params)
    pp.create_impedance(
        net,
        from_bus=b1,
        to_bus=b2,
        rft_pu=0.01,
        xft_pu=0.05,
        sn_mva=10.0,
    )

    # trafo: hv_bus/lv_bus
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=20.0,
        vk_percent=10.0,
        vkr_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
    )

    # trafo3w: hv_bus/mv_bus/lv_bus
    # Use from_parameters to avoid std types dependency.
    pp.create_transformer3w_from_parameters(
        net,
        hv_bus=b0,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=25.0,
        sn_mv_mva=15.0,
        sn_lv_mva=10.0,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
    )

    # single-bus element (bus column)
    pp.create_load(net, bus=b3, p_mw=5.0, q_mvar=1.0)

    return net


def test_add_elements_bipartite_all_types_in_one_call():
    net = _create_net_all_types()
    g = nx.Graph()

    add_elements_bipartite(
        net,
        g,
        tables=[
            ("line", "line"),
            ("impedance", "impedance"),
            ("trafo", "trafo"),
            ("trafo3w", "trafo3w"),
            ("load", "load"),
        ],
    )

    bus_nodes = {f"b_{i}" for i in net.bus.index.to_list()}
    assert bus_nodes.issubset(set(g.nodes))

    # Total element nodes should equal total rows across included tables
    expected_elements = len(net.line) + len(net.impedance) + len(net.trafo) + len(net.trafo3w) + len(net.load)
    assert len(_element_nodes(g, bus_nodes)) == expected_elements


def test_adds_edge_for_each_pair():
    g = nx.Graph()

    pairs = [(1, 2), (2, 3)]
    add_traversable_bus_bus_edges(g, pairs)

    # Nodes should exist
    assert "b_1" in g.nodes
    assert "b_2" in g.nodes
    assert "b_3" in g.nodes

    # Edges should exist (undirected)
    assert g.has_edge("b_1", "b_2")
    assert g.has_edge("b_2", "b_3")
    assert g.number_of_edges() == 2


def test_duplicate_pairs_do_not_create_multiple_edges():
    g = nx.Graph()

    pairs = [(1, 2), (2, 1), (1, 2)]
    add_traversable_bus_bus_edges(g, pairs)

    # NetworkX Graph is simple: only one undirected edge between 1 and 2
    assert g.has_edge("b_1", "b_2")
    assert g.number_of_edges() == 1


def test_self_loop_pair_creates_self_loop_edge():
    g = nx.Graph()

    pairs = [(5, 5)]
    add_traversable_bus_bus_edges(g, pairs)

    assert "b_5" in g.nodes
    assert g.has_edge("b_5", "b_5")
    assert g.number_of_edges() == 1


def test_empty_pairs_adds_nothing():
    g = nx.Graph()
    add_traversable_bus_bus_edges(g, [])

    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0
