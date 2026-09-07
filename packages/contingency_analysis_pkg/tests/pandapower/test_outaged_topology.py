# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower as pp
from toop_engine_contingency_analysis.pandapower.outaged_topology import (
    restore_elements_to_service,
    set_outaged_elements_out_of_service,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PandapowerElements


def create_net_for_outage_tests():
    """Double-busbar substation connected to a remote substation via a line.

    Topology::

        [ext_grid]
             |
          bb1 (110 kV, main busbar)
             |
          [CB bus coupler]   ← switch under test (et="b")
             |
          bb2 (110 kV, reserve busbar)

          bb1 ──── line (110 kV, 50 km) ────── remote (110 kV)
                        ↑ line under test
    """
    net = pp.create_empty_network()

    bb1 = pp.create_bus(net, vn_kv=110, name="bb1_main")
    bb2 = pp.create_bus(net, vn_kv=110, name="bb2_reserve")
    remote = pp.create_bus(net, vn_kv=110, name="remote")

    pp.create_ext_grid(net, bus=bb1, vm_pu=1.0, name="slack")

    # Bus coupler CB connecting the two busbars of the same substation
    sw = pp.create_switch(net, bus=bb1, element=bb2, et="b", closed=True, type="CB", name="bus_coupler")

    # Transmission line from the substation to a remote bus
    line = pp.create_line(net, from_bus=bb1, to_bus=remote, length_km=50, std_type="NAYY 4x50 SE", name="line_bb1_remote")

    return net, {"bb1": int(bb1), "bb2": int(bb2), "remote": int(remote), "sw": int(sw), "line": int(line)}


# ---------------------------------------------------------------------------
# set_outaged_elements_out_of_service
# ---------------------------------------------------------------------------


def test_set_outaged_switch_closed_to_open():
    net, ids = create_net_for_outage_tests()

    element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])

    were_in_service = set_outaged_elements_out_of_service(net, [element])

    assert were_in_service == [True]
    assert net.switch.loc[ids["sw"], "closed"] == False


def test_set_outaged_switch_already_open():
    net, ids = create_net_for_outage_tests()
    net.switch.loc[ids["sw"], "closed"] = False

    element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])

    were_in_service = set_outaged_elements_out_of_service(net, [element])

    assert were_in_service == [False]
    assert net.switch.loc[ids["sw"], "closed"] == False


def test_set_outaged_non_switch_element():
    net, ids = create_net_for_outage_tests()

    element = PandapowerElements(unique_id=f"{ids['line']}%%line", table="line", table_id=ids["line"])

    were_in_service = set_outaged_elements_out_of_service(net, [element])

    assert were_in_service == [True]
    assert net.line.loc[ids["line"], "in_service"] == False
    # switch is unaffected
    assert net.switch.loc[ids["sw"], "closed"] == True


def test_set_outaged_mixed_switch_and_line():
    net, ids = create_net_for_outage_tests()

    sw_element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])
    line_element = PandapowerElements(unique_id=f"{ids['line']}%%line", table="line", table_id=ids["line"])

    were_in_service = set_outaged_elements_out_of_service(net, [sw_element, line_element])

    assert were_in_service == [True, True]
    assert net.switch.loc[ids["sw"], "closed"] == False
    assert net.line.loc[ids["line"], "in_service"] == False


def test_set_outaged_empty_list():
    net, _ = create_net_for_outage_tests()

    were_in_service = set_outaged_elements_out_of_service(net, [])

    # Base-case: dummy True so downstream restore logic has something to iterate
    assert were_in_service == [True]


# ---------------------------------------------------------------------------
# restore_elements_to_service
# ---------------------------------------------------------------------------


def test_restore_switch_was_closed():
    net, ids = create_net_for_outage_tests()
    net.switch.loc[ids["sw"], "closed"] = False  # simulate outage

    element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])

    restore_elements_to_service(net, [element], were_in_service=[True])

    assert net.switch.loc[ids["sw"], "closed"] == True


def test_restore_switch_was_open():
    net, ids = create_net_for_outage_tests()
    net.switch.loc[ids["sw"], "closed"] = False

    element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])

    restore_elements_to_service(net, [element], were_in_service=[False])

    # Switch was originally open, so restore must NOT close it
    assert net.switch.loc[ids["sw"], "closed"] == False


def test_restore_non_switch_element():
    net, ids = create_net_for_outage_tests()
    net.line.loc[ids["line"], "in_service"] = False

    element = PandapowerElements(unique_id=f"{ids['line']}%%line", table="line", table_id=ids["line"])

    restore_elements_to_service(net, [element], were_in_service=[True])

    assert net.line.loc[ids["line"], "in_service"] == True


def test_restore_mixed():
    net, ids = create_net_for_outage_tests()
    net.switch.loc[ids["sw"], "closed"] = False
    net.line.loc[ids["line"], "in_service"] = False

    sw_element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])
    line_element = PandapowerElements(unique_id=f"{ids['line']}%%line", table="line", table_id=ids["line"])

    restore_elements_to_service(net, [sw_element, line_element], were_in_service=[True, True])

    assert net.switch.loc[ids["sw"], "closed"] == True
    assert net.line.loc[ids["line"], "in_service"] == True


# ---------------------------------------------------------------------------
# Round-trip: outage + restore leaves network unchanged
# ---------------------------------------------------------------------------


def test_outage_and_restore_switch_roundtrip():
    net, ids = create_net_for_outage_tests()

    element = PandapowerElements(unique_id=f"{ids['sw']}%%switch", table="switch", table_id=ids["sw"])

    were_in_service = set_outaged_elements_out_of_service(net, [element])

    assert net.switch.loc[ids["sw"], "closed"] == False

    restore_elements_to_service(net, [element], were_in_service)

    assert net.switch.loc[ids["sw"], "closed"] == True


def test_outage_and_restore_line_roundtrip():
    net, ids = create_net_for_outage_tests()

    element = PandapowerElements(unique_id=f"{ids['line']}%%line", table="line", table_id=ids["line"])

    were_in_service = set_outaged_elements_out_of_service(net, [element])

    assert net.line.loc[ids["line"], "in_service"] == False

    restore_elements_to_service(net, [element], were_in_service)

    assert net.line.loc[ids["line"], "in_service"] == True
