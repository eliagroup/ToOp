# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    create_closed_bb_switches_graph,
    get_switch_results,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import (
    _build_bus_to_branch_map,
    _connected_component_without_edge,
    _get_elements_for_buses,
    _get_switch_mapped_elements_by_origin_ids,
    get_failed_switch_results,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_results import BranchSide


def create_test_net_for_bb_graph():
    net = pp.create_empty_network()

    # Create buses
    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")
    b4 = pp.create_bus(net, vn_kv=110, name="bus_4")
    b5 = pp.create_bus(net, vn_kv=110, name="bus_5")

    # Closed bus-bus switches (should appear in graph)
    pp.create_switch(net, bus=b1, element=b2, et="b", closed=True, type="CB", name="sw_1")
    pp.create_switch(net, bus=b2, element=b3, et="b", closed=True, type="CB", name="sw_2")

    # Open switch (should NOT appear)
    pp.create_switch(net, bus=b3, element=b4, et="b", closed=False, type="CB", name="sw_3")

    # Non bus-bus switch (should NOT appear)
    pp.create_line(net, from_bus=b4, to_bus=b5, length_km=1, std_type="NAYY 4x50 SE")
    pp.create_switch(net, bus=b4, element=0, et="l", closed=True, type="CB", name="sw_4")

    return net


def test_create_closed_bb_switches_graph_basic():
    net = create_test_net_for_bb_graph()

    graph = create_closed_bb_switches_graph(net)

    # --- Type check ---
    assert isinstance(graph, nx.Graph)

    # --- All buses must be nodes ---
    assert set(graph.nodes) == set(net.bus.index)

    # --- Only closed bus-bus switches should create edges ---
    expected_edges = {
        (0, 1),
        (1, 2),
    }

    # Normalize edge representation (undirected)
    graph_edges = {tuple(sorted(edge)) for edge in graph.edges}
    expected_edges = {tuple(sorted(edge)) for edge in expected_edges}

    assert graph_edges == expected_edges


def test_create_closed_bb_switches_graph_connectivity():
    net = create_test_net_for_bb_graph()

    graph = create_closed_bb_switches_graph(net)

    # Buses 0-1-2 should be connected
    component = nx.node_connected_component(graph, 0)
    assert component == {0, 1, 2}

    # Bus 3 should be isolated (open switch)
    component_3 = nx.node_connected_component(graph, 3)
    assert component_3 == {3}

    # Bus 4 should be isolated (only line switch, ignored)
    component_4 = nx.node_connected_component(graph, 4)
    assert component_4 == {4}


def test_create_closed_bb_switches_graph_no_closed_switches():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)

    # Only open switch
    pp.create_switch(net, bus=b1, element=b2, et="b", closed=False)

    graph = create_closed_bb_switches_graph(net)

    # Nodes exist
    assert set(graph.nodes) == {b1, b2}

    # No edges
    assert len(graph.edges) == 0


def test_create_closed_bb_switches_graph_multiple_components():
    net = pp.create_empty_network()

    # Component 1
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_switch(net, bus=b1, element=b2, et="b", closed=True)

    # Component 2
    b3 = pp.create_bus(net, vn_kv=110)
    b4 = pp.create_bus(net, vn_kv=110)
    pp.create_switch(net, bus=b3, element=b4, et="b", closed=True)

    graph = create_closed_bb_switches_graph(net)

    components = list(nx.connected_components(graph))

    assert {b1, b2} in components
    assert {b3, b4} in components
    assert len(components) == 2


def test_create_closed_bb_switches_graph_empty_net():
    net = pp.create_empty_network()

    graph = create_closed_bb_switches_graph(net)

    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def create_test_net_for_bus_to_branch_map():
    net = pp.create_empty_network()

    # --- Transmission backbone ---
    b_hv_1 = pp.create_bus(net, vn_kv=380, name="HV_BUS_1")
    b_hv_2 = pp.create_bus(net, vn_kv=380, name="HV_BUS_2")
    b_hv_3 = pp.create_bus(net, vn_kv=220, name="HV_BUS_3")
    b_hv_4 = pp.create_bus(net, vn_kv=220, name="HV_BUS_4")

    # --- Subtransmission / distribution buses ---
    b_mv_1 = pp.create_bus(net, vn_kv=110, name="MV_BUS_1")
    b_mv_2 = pp.create_bus(net, vn_kv=30, name="MV_BUS_2")
    b_lv_1 = pp.create_bus(net, vn_kv=10, name="LV_BUS_1")

    # --- Branch-like elements ---

    # impedance between 380 kV buses
    imp_0 = pp.create_impedance(
        net,
        from_bus=b_hv_1,
        to_bus=b_hv_2,
        rft_pu=0.001,
        xft_pu=0.01,
        sn_mva=1000,
        name="IMP_HV_1_HV_2",
    )

    # transmission line
    line_0 = pp.create_line(
        net,
        from_bus=b_hv_2,
        to_bus=b_hv_3,
        length_km=15,
        std_type="NAYY 4x50 SE",
        name="LINE_HV_2_HV_3",
    )

    # second line to check multiple connected branches on one bus
    line_1 = pp.create_line(
        net,
        from_bus=b_hv_3,
        to_bus=b_hv_4,
        length_km=8,
        std_type="NAYY 4x50 SE",
        name="LINE_HV_3_HV_4",
    )

    # 2-winding transformer
    trafo_0 = pp.create_transformer(
        net,
        hv_bus=b_hv_3,
        lv_bus=b_mv_1,
        std_type="100 MVA 220/110 kV",
        name="TRAFO_HV_3_MV_1",
    )

    # 3-winding transformer
    trafo3w_0 = pp.create_transformer3w(
        net,
        hv_bus=b_hv_4,
        mv_bus=b_mv_2,
        lv_bus=b_lv_1,
        std_type="63/25/38 MVA 110/20/10 kV",
        name="TRAFO3W_HV_4_MV_2_LV_1",
    )
    return net, {
        "b_hv_1": b_hv_1,
        "b_hv_2": b_hv_2,
        "b_hv_3": b_hv_3,
        "b_hv_4": b_hv_4,
        "b_mv_1": b_mv_1,
        "b_mv_2": b_mv_2,
        "b_lv_1": b_lv_1,
        "imp_0": imp_0,
        "line_0": line_0,
        "line_1": line_1,
        "trafo_0": trafo_0,
        "trafo3w_0": trafo3w_0,
    }


def test_build_bus_to_branch_map_realistic_net():
    net, ids = create_test_net_for_bus_to_branch_map()

    bus_to_branch_map = _build_bus_to_branch_map(net)

    assert isinstance(bus_to_branch_map, dict)

    # --- Check exact mapping per bus, including side semantics, but ignoring order ---

    assert set(bus_to_branch_map[ids["b_hv_1"]]) == {
        (get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
    }

    assert set(bus_to_branch_map[ids["b_hv_2"]]) == {
        (get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.TWO.value),
        (get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
    }

    assert set(bus_to_branch_map[ids["b_hv_3"]]) == {
        (get_globally_unique_id(ids["line_0"], "line"), BranchSide.TWO.value),
        (get_globally_unique_id(ids["line_1"], "line"), BranchSide.ONE.value),
        (get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.ONE.value),
    }

    assert set(bus_to_branch_map[ids["b_hv_4"]]) == {
        (get_globally_unique_id(ids["line_1"], "line"), BranchSide.TWO.value),
        (get_globally_unique_id(ids["trafo3w_0"], "trafo3w"), BranchSide.ONE.value),
    }

    assert set(bus_to_branch_map[ids["b_mv_1"]]) == {
        (get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.TWO.value),
    }

    assert set(bus_to_branch_map[ids["b_mv_2"]]) == {
        (get_globally_unique_id(ids["trafo3w_0"], "trafo3w"), BranchSide.TWO.value),
    }

    assert set(bus_to_branch_map[ids["b_lv_1"]]) == {
        (get_globally_unique_id(ids["trafo3w_0"], "trafo3w"), BranchSide.THREE.value),
    }


def test_build_bus_to_branch_map_each_branch_appears_once_per_terminal():
    net, ids = create_test_net_for_bus_to_branch_map()

    bus_to_branch_map = _build_bus_to_branch_map(net)

    imp_uid = get_globally_unique_id(ids["imp_0"], "impedance")
    line0_uid = get_globally_unique_id(ids["line_0"], "line")
    line1_uid = get_globally_unique_id(ids["line_1"], "line")
    trafo_uid = get_globally_unique_id(ids["trafo_0"], "trafo")
    trafo3w_uid = get_globally_unique_id(ids["trafo3w_0"], "trafo3w")

    flattened = [item for branch_list in bus_to_branch_map.values() for item in branch_list]

    assert flattened.count((imp_uid, BranchSide.ONE.value)) == 1
    assert flattened.count((imp_uid, BranchSide.TWO.value)) == 1

    assert flattened.count((line0_uid, BranchSide.ONE.value)) == 1
    assert flattened.count((line0_uid, BranchSide.TWO.value)) == 1

    assert flattened.count((line1_uid, BranchSide.ONE.value)) == 1
    assert flattened.count((line1_uid, BranchSide.TWO.value)) == 1

    assert flattened.count((trafo_uid, BranchSide.ONE.value)) == 1
    assert flattened.count((trafo_uid, BranchSide.TWO.value)) == 1

    assert flattened.count((trafo3w_uid, BranchSide.ONE.value)) == 1
    assert flattened.count((trafo3w_uid, BranchSide.TWO.value)) == 1
    assert flattened.count((trafo3w_uid, BranchSide.THREE.value)) == 1


def test_build_bus_to_branch_map_respects_element_type_processing_order():
    net, ids = create_test_net_for_bus_to_branch_map()

    bus_to_branch_map = _build_bus_to_branch_map(net)

    # b_hv_3 is connected to:
    # - line_0 as TO side
    # - line_1 as FROM side
    # - trafo_0 as HV side
    #
    # Expected order follows implementation:
    # impedance -> line -> trafo -> trafo3w
    # and within each type, table insertion order.
    assert set(bus_to_branch_map[ids["b_hv_3"]]) == {
        (get_globally_unique_id(ids["line_0"], "line"), BranchSide.TWO.value),
        (get_globally_unique_id(ids["line_1"], "line"), BranchSide.ONE.value),
        (get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.ONE.value),
    }


def test_build_bus_to_branch_map_empty_net():
    net = pp.create_empty_network()

    bus_to_branch_map = _build_bus_to_branch_map(net)

    assert isinstance(bus_to_branch_map, dict)
    assert len(bus_to_branch_map) == 0


def test_build_bus_to_branch_map_net_with_buses_but_no_branch_elements():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=110, name="BUS_1")
    b2 = pp.create_bus(net, vn_kv=110, name="BUS_2")
    b3 = pp.create_bus(net, vn_kv=20, name="BUS_3")

    bus_to_branch_map = _build_bus_to_branch_map(net)

    assert isinstance(bus_to_branch_map, dict)
    assert len(bus_to_branch_map) == 0

    # Accessing an unconnected bus should yield no precomputed entries
    assert b1 not in bus_to_branch_map
    assert b2 not in bus_to_branch_map
    assert b3 not in bus_to_branch_map


def test_build_bus_to_branch_map_does_not_include_non_branch_elements():
    net, ids = create_test_net_for_bus_to_branch_map()

    # Add elements that must be ignored by this helper
    pp.create_switch(net, bus=ids["b_hv_1"], element=ids["b_hv_2"], et="b", closed=True, name="BUS_COUPLER")
    pp.create_load(net, bus=ids["b_mv_1"], p_mw=10.0, q_mvar=2.0, name="LOAD_1")
    pp.create_sgen(net, bus=ids["b_lv_1"], p_mw=3.0, q_mvar=0.5, name="SGEN_1")
    pp.create_gen(net, bus=ids["b_hv_1"], p_mw=100.0, vm_pu=1.01, slack=True, name="GEN_1")

    bus_to_branch_map = _build_bus_to_branch_map(net)

    # Same expected branch-only mapping as before
    assert bus_to_branch_map[ids["b_hv_1"]] == [
        (get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
    ]

    assert bus_to_branch_map[ids["b_mv_1"]] == [
        (get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.TWO.value),
    ]

    assert bus_to_branch_map[ids["b_lv_1"]] == [
        (get_globally_unique_id(ids["trafo3w_0"], "trafo3w"), BranchSide.THREE.value),
    ]


def create_test_graph_for_connected_component():
    graph = nx.Graph()

    # Main chain: 0 - 1 - 2 - 3
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    # Side branch from 1
    graph.add_edge(1, 4)

    # Separate island
    graph.add_edge(10, 11)

    # Isolated node
    graph.add_node(20)

    return graph


def test_connected_component_without_edge_no_blocked_edge():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=0)

    assert result == {0, 1, 2, 3, 4}


def test_connected_component_without_edge_blocks_middle_edge_from_left_side():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=0, blocked_edge=(1, 2))

    assert result == {0, 1, 4}


def test_connected_component_without_edge_blocks_middle_edge_from_right_side():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=3, blocked_edge=(1, 2))

    assert result == {2, 3}


def test_connected_component_without_edge_blocked_edge_is_undirected():
    graph = create_test_graph_for_connected_component()

    result_1 = _connected_component_without_edge(graph, source=0, blocked_edge=(1, 2))
    result_2 = _connected_component_without_edge(graph, source=0, blocked_edge=(2, 1))

    assert result_1 == result_2 == {0, 1, 4}


def test_connected_component_without_edge_blocking_nonexistent_edge_has_no_effect():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=0, blocked_edge=(0, 99))

    assert result == {0, 1, 2, 3, 4}


def test_connected_component_without_edge_source_in_other_component():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=10)

    assert result == {10, 11}


def test_connected_component_without_edge_isolated_node():
    graph = create_test_graph_for_connected_component()

    result = _connected_component_without_edge(graph, source=20)

    assert result == {20}


def test_connected_component_without_edge_source_not_in_graph():
    graph = create_test_graph_for_connected_component()

    with pytest.raises(nx.NetworkXError, match="The node 999 is not in the graph."):
        _connected_component_without_edge(graph, source=999)


def test_connected_component_without_edge_does_not_modify_original_graph():
    graph = create_test_graph_for_connected_component()
    original_edges = set(graph.edges())
    original_nodes = set(graph.nodes())

    _connected_component_without_edge(graph, source=0, blocked_edge=(1, 2))

    assert set(graph.nodes()) == original_nodes
    assert set(graph.edges()) == original_edges
    assert graph.has_edge(1, 2)


def test_connected_component_without_edge_with_cycle():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # cycle
            (2, 4),
        ]
    )

    # Blocking one edge in the cycle should not disconnect 0 from 2 or 3
    result = _connected_component_without_edge(graph, source=0, blocked_edge=(1, 2))

    assert result == {0, 1, 2, 3, 4}


def test_connected_component_without_edge_bridge_split():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),  # bridge between left and right
            (3, 4),
            (4, 5),
        ]
    )

    left_result = _connected_component_without_edge(graph, source=0, blocked_edge=(2, 3))
    right_result = _connected_component_without_edge(graph, source=5, blocked_edge=(2, 3))

    assert left_result == {0, 1, 2}
    assert right_result == {3, 4, 5}


def create_test_net_for_get_elements():
    net = pp.create_empty_network()

    # --- Buses ---
    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")
    b4 = pp.create_bus(net, vn_kv=20, name="bus_4")

    # --- Branch elements ---
    line_0 = pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1, std_type="NAYY 4x50 SE")
    imp_0 = pp.create_impedance(net, from_bus=b2, to_bus=b3, rft_pu=0.01, xft_pu=0.05, sn_mva=100)
    trafo_0 = pp.create_transformer(net, hv_bus=b3, lv_bus=b4, std_type="25 MVA 110/20 kV")

    return net, {
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "line_0": line_0,
        "imp_0": imp_0,
        "trafo_0": trafo_0,
    }


def test_get_elements_for_buses_single_bus():
    net, ids = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    switch_id = 10
    sw_buses = {ids["b1"]}

    result = _get_elements_for_buses(switch_id, sw_buses, bus_map)

    expected = {
        (switch_id, get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
    }

    assert set(result) == expected


def test_get_elements_for_buses_multiple_buses():
    net, ids = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    switch_id = 5
    sw_buses = {ids["b2"], ids["b3"]}

    result = _get_elements_for_buses(switch_id, sw_buses, bus_map)

    expected = {
        # b2
        (switch_id, get_globally_unique_id(ids["line_0"], "line"), BranchSide.TWO.value),
        (switch_id, get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
        # b3
        (switch_id, get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.TWO.value),
        (switch_id, get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.ONE.value),
    }

    assert set(result) == expected


def test_get_elements_for_buses_empty_bus_set():
    net, _ = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    result = _get_elements_for_buses(1, set(), bus_map)

    assert result == []


def test_get_elements_for_buses_bus_not_in_map():
    net, _ = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    # bus id that does not exist in map
    result = _get_elements_for_buses(1, {999}, bus_map)

    assert result == []


def test_get_elements_for_buses_mixed_valid_and_invalid_buses():
    net, ids = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    switch_id = 7
    sw_buses = {ids["b1"], 999}

    result = _get_elements_for_buses(switch_id, sw_buses, bus_map)

    expected = {
        (switch_id, get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
    }

    assert set(result) == expected


def test_get_elements_for_buses_duplicate_elements_possible():
    """
    If two buses from the same branch are included,
    both sides should appear. All other branch elements connected
    to those buses should also be included.
    """
    net, ids = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    switch_id = 3
    sw_buses = {ids["b1"], ids["b2"]}  # both ends of line_0, plus b2 is also connected to imp_0

    result = _get_elements_for_buses(switch_id, sw_buses, bus_map)

    expected = {
        (switch_id, get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
        (switch_id, get_globally_unique_id(ids["line_0"], "line"), BranchSide.TWO.value),
        (switch_id, get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
    }

    assert set(result) == expected


def test_get_elements_for_buses_switch_id_propagation():
    net, ids = create_test_net_for_get_elements()
    bus_map = _build_bus_to_branch_map(net)

    switch_id = 12345
    sw_buses = {ids["b3"]}

    result = _get_elements_for_buses(switch_id, sw_buses, bus_map)

    # all tuples must contain the same switch_id
    assert all(r[0] == switch_id for r in result)


def create_test_net_for_switch_mapped_elements():
    net = pp.create_empty_network()

    # --- Buses ---
    b0 = pp.create_bus(net, vn_kv=110, name="bus_0")
    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")
    b4 = pp.create_bus(net, vn_kv=20, name="bus_4")
    b5 = pp.create_bus(net, vn_kv=10, name="bus_5")

    # --- Closed bus-bus switches topology ---
    # chain: b0 -- b1 -- b2
    sw0 = pp.create_switch(net, bus=b0, element=b1, et="b", closed=True, type="CB", name="sw_0")
    sw1 = pp.create_switch(net, bus=b1, element=b2, et="b", closed=True, type="CB", name="sw_1")

    # isolated closed bus-bus switch
    sw2 = pp.create_switch(net, bus=b3, element=b4, et="b", closed=True, type="CB", name="sw_2")

    # open switch with matching origin_id candidate, should be ignored
    sw3 = pp.create_switch(net, bus=b4, element=b5, et="b", closed=False, type="CB", name="sw_3")

    net.switch.loc[sw0, "origin_id"] = "ORIGIN_SW_0"
    net.switch.loc[sw1, "origin_id"] = "ORIGIN_SW_1"
    net.switch.loc[sw2, "origin_id"] = "ORIGIN_SW_2"
    net.switch.loc[sw3, "origin_id"] = "ORIGIN_SW_3"

    # --- Branch-like elements ---
    line_0 = pp.create_line(net, from_bus=b0, to_bus=b5, length_km=1, std_type="NAYY 4x50 SE", name="line_0")
    imp_0 = pp.create_impedance(net, from_bus=b1, to_bus=b3, rft_pu=0.01, xft_pu=0.05, sn_mva=100, name="imp_0")
    trafo_0 = pp.create_transformer(net, hv_bus=b2, lv_bus=b4, std_type="25 MVA 110/10 kV", name="trafo_0")

    return net, {
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "b5": b5,
        "sw0": sw0,
        "sw1": sw1,
        "sw2": sw2,
        "sw3": sw3,
        "line_0": line_0,
        "imp_0": imp_0,
        "trafo_0": trafo_0,
    }


def test_get_switch_mapped_elements_by_origin_ids_empty_when_no_matching_switches():
    net, _ = create_test_net_for_switch_mapped_elements()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[-1],
        side="bus",
    )

    assert isinstance(branch_map_df, pd.DataFrame)
    assert isinstance(bus_map_df, pd.DataFrame)

    assert list(branch_map_df.columns) == ["switch_id", "element", "side"]
    assert list(bus_map_df.columns) == ["switch_id", "element"]

    assert branch_map_df.empty
    assert bus_map_df.empty


def test_get_switch_mapped_elements_by_origin_ids_ignores_open_switches():
    net, _ = create_test_net_for_switch_mapped_elements()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[3],
        side="bus",
    )

    assert branch_map_df.empty
    assert bus_map_df.empty


def test_get_switch_mapped_elements_by_origin_ids_bus_side_single_switch():
    net, ids = create_test_net_for_switch_mapped_elements()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[1],
        side="bus",
    )

    expected_buses = {
        (ids["sw1"], get_globally_unique_id(ids["b0"], "bus")),
        (ids["sw1"], get_globally_unique_id(ids["b1"], "bus")),
    }

    expected_branch_elements = {
        (ids["sw1"], get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
        (ids["sw1"], get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
    }

    assert set(map(tuple, bus_map_df[["switch_id", "element"]].to_records(index=False))) == expected_buses
    assert (
        set(map(tuple, branch_map_df[["switch_id", "element", "side"]].to_records(index=False))) == expected_branch_elements
    )


def test_get_switch_mapped_elements_by_origin_ids_element_side_single_switch():
    net, ids = create_test_net_for_switch_mapped_elements()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[1],
        side="element",
    )

    expected_buses = {
        (ids["sw1"], get_globally_unique_id(ids["b2"], "bus")),
    }

    expected_branch_elements = {
        (ids["sw1"], get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.ONE.value),
    }

    assert set(map(tuple, bus_map_df[["switch_id", "element"]].to_records(index=False))) == expected_buses
    assert (
        set(map(tuple, branch_map_df[["switch_id", "element", "side"]].to_records(index=False))) == expected_branch_elements
    )


def test_get_switch_mapped_elements_by_origin_ids_multiple_switches():
    net, ids = create_test_net_for_switch_mapped_elements()

    branch_map_df, bus_map_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[1, 2],
        side="bus",
    )

    expected_buses = {
        # sw1 -> from bus side => component {b0, b1}
        (ids["sw1"], get_globally_unique_id(ids["b0"], "bus")),
        (ids["sw1"], get_globally_unique_id(ids["b1"], "bus")),
        # sw2 -> from bus side, edge blocked so only b3 remains
        (ids["sw2"], get_globally_unique_id(ids["b3"], "bus")),
    }

    expected_branch_elements = {
        # sw1 side
        (ids["sw1"], get_globally_unique_id(ids["line_0"], "line"), BranchSide.ONE.value),
        (ids["sw1"], get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.ONE.value),
        # sw2 side
        (ids["sw2"], get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.TWO.value),
    }

    assert set(map(tuple, bus_map_df[["switch_id", "element"]].to_records(index=False))) == expected_buses
    assert (
        set(map(tuple, branch_map_df[["switch_id", "element", "side"]].to_records(index=False))) == expected_branch_elements
    )


def test_get_switch_mapped_elements_by_origin_ids_bus_and_element_side_differ():
    net, ids = create_test_net_for_switch_mapped_elements()

    branch_bus_df, bus_bus_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[2],
        side="bus",
    )
    branch_element_df, bus_element_df = _get_switch_mapped_elements_by_origin_ids(
        net=net,
        switches_ids=[2],
        side="element",
    )

    expected_bus_side_buses = {
        (ids["sw2"], get_globally_unique_id(ids["b3"], "bus")),
    }
    expected_element_side_buses = {
        (ids["sw2"], get_globally_unique_id(ids["b4"], "bus")),
    }

    expected_bus_side_elements = {
        (ids["sw2"], get_globally_unique_id(ids["imp_0"], "impedance"), BranchSide.TWO.value),
    }
    expected_element_side_elements = {
        (ids["sw2"], get_globally_unique_id(ids["trafo_0"], "trafo"), BranchSide.TWO.value),
    }

    assert set(map(tuple, bus_bus_df[["switch_id", "element"]].to_records(index=False))) == expected_bus_side_buses
    assert set(map(tuple, bus_element_df[["switch_id", "element"]].to_records(index=False))) == expected_element_side_buses

    assert (
        set(map(tuple, branch_bus_df[["switch_id", "element", "side"]].to_records(index=False)))
        == expected_bus_side_elements
    )
    assert (
        set(map(tuple, branch_element_df[["switch_id", "element", "side"]].to_records(index=False)))
        == expected_element_side_elements
    )


def create_test_net_for_switch_results():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")

    sw1 = pp.create_switch(net, bus=b1, element=b2, et="b", closed=True, type="CB", name="switch_1")
    sw2 = pp.create_switch(net, bus=b2, element=b3, et="b", closed=True, type="CB", name="switch_2")

    return net, {
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "sw1": sw1,
        "sw2": sw2,
    }


def test_get_switch_results_basic_aggregation():
    net, ids = create_test_net_for_switch_results()

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 7

    line_uid = get_globally_unique_id(10, "line")
    trafo_uid = get_globally_unique_id(3, "trafo")
    bus_uid = get_globally_unique_id(ids["b1"], "bus")

    branch_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": line_uid,
                "side": BranchSide.ONE.value,
                "i": 100.0,
                "p": 10.0,
                "q": 4.0,
                "loading": 80.0,
                "element_name": "line_10",
                "contingency_name": contingency.name,
            },
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": trafo_uid,
                "side": BranchSide.TWO.value,
                "i": 200.0,
                "p": 5.0,
                "q": 6.0,
                "loading": 70.0,
                "element_name": "trafo_3",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element", "side"])

    node_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus_uid,
                "vm": 110.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 3.0,
                "q": 2.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_1",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element"])

    switch_element_mapping = pd.DataFrame(
        [
            {"switch_id": ids["sw1"], "element": line_uid, "side": BranchSide.ONE.value},
            {"switch_id": ids["sw1"], "element": trafo_uid, "side": BranchSide.TWO.value},
            {"switch_id": ids["sw1"], "element": bus_uid, "side": np.nan},
        ]
    )

    result = get_switch_results(
        net=net,
        contingency=contingency,
        timestep=timestep,
        branch_results=branch_results,
        node_results=node_results,
        switch_element_mapping=switch_element_mapping,
    )

    expected_element = get_globally_unique_id(ids["sw1"], "switch")

    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["timestep", "contingency", "element"]
    assert result.index.tolist() == [(timestep, contingency.unique_id, expected_element)]

    row = result.loc[(timestep, contingency.unique_id, expected_element)]

    expected_p = 10.0 + 5.0 + 3.0
    expected_q = 4.0 + 6.0 + 2.0
    expected_vm = 110.0
    expected_s = np.sqrt(expected_p**2 + expected_q**2)
    expected_i = expected_s / (np.sqrt(3) * expected_vm)

    assert row["switch_id"] == ids["sw1"]
    assert row["p"] == expected_p
    assert row["q"] == expected_q
    assert row["vm"] == expected_vm
    assert np.isclose(row["s"], expected_s)
    assert np.isclose(row["i"], expected_i)
    assert row["element_name"] == "switch_1"
    assert row["contingency_name"] == contingency.name


def test_get_switch_results_multiple_switches():
    net, ids = create_test_net_for_switch_results()

    contingency = PandapowerContingency(
        unique_id="contingency_2",
        name="contingency_2_name",
        elements=[],
    )
    timestep = 3

    line_uid = get_globally_unique_id(1, "line")
    imp_uid = get_globally_unique_id(2, "impedance")
    bus1_uid = get_globally_unique_id(ids["b1"], "bus")
    bus2_uid = get_globally_unique_id(ids["b2"], "bus")

    branch_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": line_uid,
                "side": BranchSide.ONE.value,
                "i": 0.0,
                "p": 8.0,
                "q": 6.0,
                "loading": 0.0,
                "element_name": "line_1",
                "contingency_name": contingency.name,
            },
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": imp_uid,
                "side": BranchSide.TWO.value,
                "i": 0.0,
                "p": 1.0,
                "q": 2.0,
                "loading": 0.0,
                "element_name": "imp_2",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element", "side"])

    node_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus1_uid,
                "vm": 110.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 2.0,
                "q": 1.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_1",
                "contingency_name": contingency.name,
            },
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus2_uid,
                "vm": 111.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 4.0,
                "q": 3.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_2",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element"])

    switch_element_mapping = pd.DataFrame(
        [
            {"switch_id": ids["sw1"], "element": line_uid, "side": BranchSide.ONE.value},
            {"switch_id": ids["sw1"], "element": bus1_uid, "side": np.nan},
            {"switch_id": ids["sw2"], "element": imp_uid, "side": BranchSide.TWO.value},
            {"switch_id": ids["sw2"], "element": bus2_uid, "side": np.nan},
        ]
    )

    result = get_switch_results(
        net=net,
        contingency=contingency,
        timestep=timestep,
        branch_results=branch_results,
        node_results=node_results,
        switch_element_mapping=switch_element_mapping,
    )

    sw1_element = get_globally_unique_id(ids["sw1"], "switch")
    sw2_element = get_globally_unique_id(ids["sw2"], "switch")

    assert set(result.index.tolist()) == {
        (timestep, contingency.unique_id, sw1_element),
        (timestep, contingency.unique_id, sw2_element),
    }

    row1 = result.loc[(timestep, contingency.unique_id, sw1_element)]
    assert row1["switch_id"] == ids["sw1"]
    assert row1["p"] == 10.0
    assert row1["q"] == 7.0
    assert row1["vm"] == 110.0
    assert row1["element_name"] == "switch_1"
    assert row1["contingency_name"] == contingency.name

    row2 = result.loc[(timestep, contingency.unique_id, sw2_element)]
    assert row2["switch_id"] == ids["sw2"]
    assert row2["p"] == 5.0
    assert row2["q"] == 5.0
    assert row2["vm"] == 111.0
    assert row2["element_name"] == "switch_2"
    assert row2["contingency_name"] == contingency.name


def test_get_switch_results_vm_zero_row_removed():
    net, ids = create_test_net_for_switch_results()

    contingency = PandapowerContingency(
        unique_id="contingency_vm0",
        name="contingency_vm0_name",
        elements=[],
    )
    timestep = 1

    line_uid = get_globally_unique_id(5, "line")
    bus_uid = get_globally_unique_id(ids["b1"], "bus")

    branch_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": line_uid,
                "side": BranchSide.ONE.value,
                "i": 0.0,
                "p": 7.0,
                "q": 1.0,
                "loading": 0.0,
                "element_name": "line_5",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element", "side"])

    node_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus_uid,
                "vm": 0.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 2.0,
                "q": 3.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_1",
                "contingency_name": contingency.name,
            },
        ]
    ).set_index(["timestep", "contingency", "element"])

    switch_element_mapping = pd.DataFrame(
        [
            {"switch_id": ids["sw1"], "element": line_uid, "side": BranchSide.ONE.value},
            {"switch_id": ids["sw1"], "element": bus_uid, "side": np.nan},
        ]
    )

    result = get_switch_results(
        net=net,
        contingency=contingency,
        timestep=timestep,
        branch_results=branch_results,
        node_results=node_results,
        switch_element_mapping=switch_element_mapping,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_switch_results_empty_mapping_returns_empty_result():
    net, _ = create_test_net_for_switch_results()

    contingency = PandapowerContingency(
        unique_id="contingency_empty",
        name="contingency_empty_name",
        elements=[],
    )
    timestep = 0

    branch_results = pd.DataFrame(
        columns=["timestep", "contingency", "element", "side", "i", "p", "q", "loading", "element_name", "contingency_name"]
    ).set_index(["timestep", "contingency", "element", "side"])

    node_results = pd.DataFrame(
        columns=[
            "timestep",
            "contingency",
            "element",
            "vm",
            "vm_loading",
            "va",
            "p",
            "q",
            "vm_basecase_deviation",
            "element_name",
            "contingency_name",
        ]
    ).set_index(["timestep", "contingency", "element"])

    switch_element_mapping = pd.DataFrame(columns=["switch_id", "element", "side"])

    result = get_switch_results(
        net=net,
        contingency=contingency,
        timestep=timestep,
        branch_results=branch_results,
        node_results=node_results,
        switch_element_mapping=switch_element_mapping,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_switch_results_uses_last_vm_from_node_group():
    net, ids = create_test_net_for_switch_results()

    contingency = PandapowerContingency(
        unique_id="contingency_last_vm",
        name="contingency_last_vm_name",
        elements=[],
    )
    timestep = 2

    bus1_uid = get_globally_unique_id(ids["b1"], "bus")
    bus2_uid = get_globally_unique_id(ids["b2"], "bus")

    branch_results = pd.DataFrame(
        columns=["timestep", "contingency", "element", "side", "i", "p", "q", "loading", "element_name", "contingency_name"]
    )

    node_results = pd.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus1_uid,
                "vm": 109.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 1.0,
                "q": 2.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_1",
                "contingency_name": contingency.name,
            },
            {
                "timestep": timestep,
                "contingency": contingency.unique_id,
                "element": bus2_uid,
                "vm": 111.0,
                "vm_loading": 0.0,
                "va": 0.0,
                "p": 3.0,
                "q": 4.0,
                "vm_basecase_deviation": 0.0,
                "element_name": "bus_2",
                "contingency_name": contingency.name,
            },
        ]
    )

    switch_element_mapping = pd.DataFrame(
        [
            {"switch_id": ids["sw1"], "element": bus1_uid, "side": np.nan},
            {"switch_id": ids["sw1"], "element": bus2_uid, "side": np.nan},
        ]
    )

    result = get_switch_results(
        net=net,
        contingency=contingency,
        timestep=timestep,
        branch_results=branch_results,
        node_results=node_results,
        switch_element_mapping=switch_element_mapping,
    )

    switch_element = get_globally_unique_id(ids["sw1"], "switch")
    row = result.loc[(timestep, contingency.unique_id, switch_element)]

    assert row["switch_id"] == ids["sw1"]
    assert row["p"] == 4.0
    assert row["q"] == 6.0
    assert row["vm"] == 111.0

    expected_s = np.sqrt(4.0**2 + 6.0**2)
    expected_i = expected_s / (np.sqrt(3) * 111.0)

    assert np.isclose(row["s"], expected_s)
    assert np.isclose(row["i"], expected_i)


def test_get_failed_switch_results_basic():
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 4

    switch_element_mapping = pd.DataFrame(
        [
            {"switch_id": 0, "element": get_globally_unique_id(10, "line"), "side": 1.0},
            {"switch_id": 1, "element": get_globally_unique_id(20, "line"), "side": 2.0},
        ]
    )

    result = get_failed_switch_results(
        timestep=timestep,
        switch_element_mapping=switch_element_mapping,
        contingency=contingency,
    )

    expected_index = pd.MultiIndex.from_tuples(
        [
            (timestep, contingency.unique_id, get_globally_unique_id(0, "switch")),
            (timestep, contingency.unique_id, get_globally_unique_id(1, "switch")),
        ],
        names=["timestep", "contingency", "element"],
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["timestep", "contingency", "element"]
    assert set(result.index.tolist()) == set(expected_index.tolist())

    assert result["p"].isna().all()
    assert result["q"].isna().all()
    assert result["vm"].isna().all()
    assert result["i"].isna().all()

    assert (result["element_name"] == "").all()
    assert (result["contingency_name"] == "").all()
