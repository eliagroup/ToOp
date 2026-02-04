# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
import numpy as np
import pandapower as pp
import pytest
from toop_engine_grid_helpers.pandapower.network_topology_utils import (
    SEPARATOR,
    _edges_for_branch_element,
    _get_bus_edges,
    _get_line_edges,
    _get_switch_edges,
    _get_trafo3w_edges,
    _get_trafo_edges,
    collect_element_edges,
)


def _net_with_n_buses(n=4):
    net = pp.create_empty_network()
    for _ in range(n):
        pp.create_bus(net, vn_kv=110.0)
    return net


def test_get_line_edges_returns_from_to_bus_pair():
    net = _net_with_n_buses(3)
    line_id = pp.create_line_from_parameters(
        net, from_bus=0, to_bus=2, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    edges = _get_line_edges(net, int(line_id))
    assert isinstance(edges, list) and len(edges) == 1
    assert edges[0] == (0, 2)
    assert all(isinstance(x, np.int64) for x in edges[0])


def test_get_switch_edges_bus_bus_returns_bus_pair():
    net = _net_with_n_buses(3)
    sw_id = pp.create_switch(net, bus=1, element=2, et="b", closed=True)
    edges = _get_switch_edges(net, int(sw_id))
    assert edges == [(1, 2)]
    assert all(isinstance(x, np.int64) for x in edges[0])


def test_get_switch_edges_line_switch_returns_bus_and_line_id():
    net = _net_with_n_buses(3)
    line_id = pp.create_line_from_parameters(
        net, from_bus=0, to_bus=2, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    sw_id = pp.create_switch(net, bus=0, element=int(line_id), et="l", closed=True)
    edges = _get_switch_edges(net, int(sw_id))
    assert edges == [(0, line_id)]
    assert all(isinstance(x, np.int64) for x in edges[0])


def test_get_trafo_edges_maps_hv_to_lv():
    net = _net_with_n_buses(3)
    trafo_id = pp.create_transformer_from_parameters(
        net,
        hv_bus=0,
        lv_bus=2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=10,
        vkr_percent=0.5,
        vk_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
    )
    edges = _get_trafo_edges(net, int(trafo_id))
    assert edges == [(0, 2)]
    assert all(isinstance(x, np.int64) for x in edges[0])


def test_get_trafo3w_edges_connects_all_windings():
    net = _net_with_n_buses(5)
    t3_id = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=0,
        mv_bus=1,
        lv_bus=3,
        sn_hv_mva=60,
        sn_mv_mva=30,
        sn_lv_mva=30,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vkr_hv_percent=0.5,
        vkr_mv_percent=0.6,
        vkr_lv_percent=0.7,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_mv_degree=0.0,
        shift_lv_degree=0.0,
    )
    edges = _get_trafo3w_edges(net, int(t3_id))
    assert set(edges) == {(0, 3), (1, 3), (0, 1)}
    for e in edges:
        assert all(isinstance(x, np.int64) for x in e)


def test_get_bus_edges_collects_all_closed_switch_edges_for_bus():
    net = _net_with_n_buses(5)

    pp.create_switch(net, bus=1, element=2, et="b", closed=True)
    pp.create_switch(net, bus=1, element=3, et="b", closed=False)

    line_id = pp.create_line_from_parameters(
        net, from_bus=1, to_bus=4, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    pp.create_switch(net, bus=1, element=line_id, et="l", closed=True)

    pp.create_switch(net, bus=0, element=1, et="b", closed=True)

    edges = _get_bus_edges(net, 1)

    expected = {(1, 2), (1, line_id), (0, 1)}
    assert set(edges) == expected

    assert len(edges) == len(set(edges))

    for e in edges:
        assert isinstance(e, tuple) and len(e) == 2
        assert isinstance(e[0], np.int64) and isinstance(e[1], np.int64)


def test_get_bus_edges_empty_when_no_closed_switches_touch_bus():
    net = _net_with_n_buses(3)
    pp.create_switch(net, bus=2, element=1, et="b", closed=False)
    edges = _get_bus_edges(net, 2)
    assert edges == []


def test_edges_for_branch_element_line():
    net = _net_with_n_buses(3)
    line_id = pp.create_line_from_parameters(
        net, from_bus=0, to_bus=2, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    edges = _edges_for_branch_element(net, "line", int(line_id))
    assert edges == [(0, 2)]
    assert all(isinstance(x, np.int64) for x in edges[0])


def test_edges_for_branch_element_trafo():
    net = _net_with_n_buses(3)
    trafo_id = pp.create_transformer_from_parameters(
        net,
        hv_bus=0,
        lv_bus=1,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=10,
        vkr_percent=0.5,
        vk_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
    )
    edges = _edges_for_branch_element(net, "trafo", int(trafo_id))
    assert edges == [(0, 1)]


def test_edges_for_branch_element_trafo3w():
    net = _net_with_n_buses(5)
    t3_id = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=0,
        mv_bus=2,
        lv_bus=4,
        sn_hv_mva=60,
        sn_mv_mva=30,
        sn_lv_mva=30,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vkr_hv_percent=0.5,
        vkr_mv_percent=0.6,
        vkr_lv_percent=0.7,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_mv_degree=0.0,
        shift_lv_degree=0.0,
    )
    edges = _edges_for_branch_element(net, "trafo3w", int(t3_id))
    assert set(edges) == {(0, 4), (2, 4), (0, 2)}  # (hv, lv), (mv, lv), (hv, mv)


def test_edges_for_branch_element_unknown_type_raises():
    net = _net_with_n_buses(2)
    with pytest.raises(ValueError) as exc:
        _edges_for_branch_element(net, "unknown", 0)
    assert "Unknown element type" in str(exc.value)


def test_collect_element_edges_mixed_types_and_closed_bus_switches_only():
    """
    Mix of:
      - bus id with closed/open switches -> only closed considered
      - line element
      - 2-winding transformer
      - 3-winding transformer
    """
    net = _net_with_n_buses(8)

    # Bus 1 has two bus-bus switches: one closed, one open (open should be ignored)
    pp.create_switch(net, bus=1, element=3, et="b", closed=True)  # -> (1, 3)
    pp.create_switch(net, bus=1, element=4, et="b", closed=False)  # ignored

    # Also a line from bus 1 to 5 with a closed line switch at bus 1 (will appear as (1, line_id) in _get_bus_edges)
    line_id_for_switch = pp.create_line_from_parameters(
        net, from_bus=1, to_bus=5, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    pp.create_switch(net, bus=1, element=line_id_for_switch, et="l", closed=True)  # -> (1, line_id_for_switch)

    # Branch elements to collect via dispatcher
    line_id = pp.create_line_from_parameters(
        net, from_bus=0, to_bus=2, length_km=2.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    trafo_id = pp.create_transformer_from_parameters(
        net,
        hv_bus=6,
        lv_bus=7,
        sn_mva=25,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.5,
        vk_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
    )
    t3_id = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=0,
        mv_bus=6,
        lv_bus=4,
        sn_hv_mva=60,
        sn_mv_mva=30,
        sn_lv_mva=30,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vkr_hv_percent=0.5,
        vkr_mv_percent=0.6,
        vkr_lv_percent=0.7,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_mv_degree=0.0,
        shift_lv_degree=0.0,
    )

    elements = [
        f"1{SEPARATOR}bus",  # bus edges via closed switches touching bus 1
        f"{line_id}{SEPARATOR}line",  # line edges
        f"{trafo_id}{SEPARATOR}trafo",
        f"{t3_id}{SEPARATOR}trafo3w",
    ]

    edges = collect_element_edges(net, elements)

    # Expected BUS edges (set semantics in implementation)
    expected_bus_edges = {(1, 3), (1, line_id_for_switch)}
    # Expected BRANCH edges (dispatcher)
    expected_branch_edges = {(0, 2), (6, 7), (0, 4), (6, 4), (0, 6)}

    # Check presence irrespective of ordering of bus set or 3W trafo edges
    assert expected_bus_edges.issubset(set(edges))
    assert expected_branch_edges.issubset(set(edges))

    # No open-switch edge included
    assert (1, 4) not in edges

    # Type sanity: all entries are pairs of ints
    for e in edges:
        assert isinstance(e, tuple) and len(e) == 2
        assert all(isinstance(x, np.int64) for x in e)


def test_collect_element_edges_deduplicates_bus_edges():
    """
    If multiple closed switches produce the same (from,to) edge for a bus,
    the bus-edge portion should be de-duplicated via the set() inside collect_element_edges.
    """
    net = _net_with_n_buses(4)

    # Two closed bus-bus switches giving the same edge (1, 2)
    pp.create_switch(net, bus=1, element=2, et="b", closed=True)
    pp.create_switch(net, bus=1, element=2, et="b", closed=True)

    # Include only a bus element so we test the set de-dup behavior directly
    elements = [f"1{SEPARATOR}bus"]
    edges = collect_element_edges(net, elements)

    # Should contain exactly one (1,2)
    assert edges.count((1, 2)) == 1
    assert set(edges) == {(1, 2)}
