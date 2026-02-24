import pandapower as pp
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerElements,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.contingency_outage_group import (
    get_grid_element,
    _elements_in_component,
    get_outage_group_for_contingency,
)
from toop_engine_grid_helpers.pandapower.outage_group import (
    OUTAGE_GROUP_SEPARATOR,
)


def build_simple_net():
    """
    Topology:

        bus0 --(line0)-- bus1 ==(switch0, DS)== bus2 --(line1 idx=1)-- bus3 ==(switch1, CB)== bus4 --(line1 idx=2)-- bus5

    Notes:
      - you create 3 lines total: line index 0, 1, 2
      - the last line also has name="line1" (duplicate name), but different index
    """
    net = pp.create_empty_network()

    b0 = pp.create_bus(net, vn_kv=110, name="bus0")
    b1 = pp.create_bus(net, vn_kv=110, name="bus1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus3")

    pp.create_line_from_parameters(
        net, b0, b1,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line0",
    )
    pp.create_switch(net, b1, b2, et="b", closed=True, type="DS", name="switch0")

    pp.create_line_from_parameters(
        net, b2, b3,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line1",
    )

    b4 = pp.create_bus(net, vn_kv=110, name="bus4")
    pp.create_switch(net, b3, b4, et="b", closed=True, type="CB", name="switch1")

    b5 = pp.create_bus(net, vn_kv=110, name="bus5")
    pp.create_line_from_parameters(
        net, b4, b5,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line2",
    )

    return net


def test_get_grid_element_returns_valid_wrapper():
    net = build_simple_net()
    element = get_grid_element(net, 0, "line")
    assert isinstance(element, PandapowerElements)
    assert element.table == "line"
    assert element.table_id == 0
    assert element.name == "line0"


def test_elements_in_component_parses_nodes_correctly():
    sep = OUTAGE_GROUP_SEPARATOR

    components = [
        {
            f"e{sep}line{sep}5",
            f"e{sep}trafo{sep}3",
            f"b{sep}7",
        }
    ]

    result = _elements_in_component(components, 0)

    assert result == {(5, "line"), (3, "trafo"), (7, "bus")}


def test_grouping_single_line_returns_connected_component():
    """
    With the current net, line0 is between bus0 and bus1.
    Since switch0 is closed between bus1 and bus2, the connected component
    for line0 *may* include further downstream elements, depending on how
    build_connected_components_for_contingency_analysis models bus-bus switches.

    We assert the minimal must-haves (line0 + its endpoint buses).
    """
    net = build_simple_net()

    initial_element = get_grid_element(net, 0, "line")

    contingency = PandapowerContingency(
        unique_id="c1",
        name="line_outage",
        elements=[initial_element],
    )

    result = get_outage_group_for_contingency(net, [contingency])

    assert len(result) == 1
    grouped = result[0]
    assert grouped.unique_id == "c1"

    ids = {(e.table, e.table_id) for e in grouped.elements}

    # must contain original element
    assert ("line", 0) in ids

    # must contain endpoint buses of line0 (bus0 and bus1)
    assert ("bus", 0) in ids
    assert ("bus", 1) in ids

    # optional (implementation-dependent): switch0 might be included
    # if switches are represented as elements in the component graph.
    # assert ("switch", 0) in ids  # <-- only enable if you're sure


def test_grouping_multiple_elements_unions_components():
    """
    Use two different lines in the chain: line0 (idx=0) and the last line (idx=2).
    In this topology, if all bus-bus switches are considered closed connections,
    everything is in one component, so union still returns that component.
    """
    net = build_simple_net()

    e1 = get_grid_element(net, 0, "line")  # bus0-bus1
    e2 = get_grid_element(net, 2, "line")  # bus4-bus5 (name also "line1")

    contingency = PandapowerContingency(
        unique_id="c2",
        name="double",
        elements=[e1, e2],
    )

    result = get_outage_group_for_contingency(net, [contingency])
    grouped = result[0]

    ids = {(e.table, e.table_id) for e in grouped.elements}

    assert ("line", 0) in ids
    assert ("line", 2) in ids

    # sanity: should include at least some buses
    assert any(t == "bus" for (t, _) in ids)


def test_unmapped_element_creates_component():
    """
    Covers branch where element node isn't found in connected components.
    """
    net = build_simple_net()

    # isolated bus not connected to anything
    isolated_bus = pp.create_bus(net, vn_kv=110, name="isolated")

    bus_element = get_grid_element(net, int(isolated_bus), "bus")

    contingency = PandapowerContingency(
        unique_id="c3",
        name="isolated_bus",
        elements=[bus_element],
    )

    result = get_outage_group_for_contingency(net, [contingency])
    grouped = result[0]

    ids = {(e.table, e.table_id) for e in grouped.elements}

    assert ("bus", isolated_bus) in ids