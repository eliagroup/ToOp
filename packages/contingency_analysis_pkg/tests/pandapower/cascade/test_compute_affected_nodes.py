# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Unit tests for compute_affected_nodes in topology.py."""

import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import compute_affected_nodes


@pytest.fixture()
def linear_net() -> pp.pandapowerNet:
    """Three buses connected in a line by two closed bus-bus switches."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0, name="bus0")
    b1 = pp.create_bus(net, vn_kv=110.0, name="bus1")
    b2 = pp.create_bus(net, vn_kv=110.0, name="bus2")
    pp.create_switch(net, bus=b0, element=b1, et="b", closed=True, type="CB", name="sw0")
    pp.create_switch(net, bus=b1, element=b2, et="b", closed=True, type="CB", name="sw1")
    return net


def _make_el_list(switch_id: int, protection_side: str, relay_side: str) -> pd.DataFrame:
    return pd.DataFrame({"switch_id": [switch_id], "protection_side": [protection_side], "relay_side": [relay_side]})


class TestComputeAffectedNodes:
    def test_protection_side_bus_returns_bus_side_nodes(self, linear_net):
        """protection_side='bus' → only bus0 is isolated when sw0 opens."""
        el_list = _make_el_list(switch_id=0, protection_side="bus", relay_side="bus")
        result = compute_affected_nodes(linear_net, el_list)
        assert result[0] == [0]  # only bus0

    def test_protection_side_element_returns_element_side_nodes(self, linear_net):
        """protection_side='element' → bus1 and bus2 are the affected side when sw0 opens."""
        el_list = _make_el_list(switch_id=0, protection_side="element", relay_side="element")
        result = compute_affected_nodes(linear_net, el_list)
        assert sorted(result[0]) == [1, 2]  # bus1 + bus2

    def test_protection_side_differs_from_relay_side_uses_protection_side(self, linear_net):
        """When protection_side != relay_side the function must follow protection_side.

        relay_side='bus'  → would give {bus0}
        protection_side='element' → must give {bus1, bus2}
        """
        el_list = _make_el_list(switch_id=0, protection_side="element", relay_side="bus")
        result = compute_affected_nodes(linear_net, el_list)

        # protection_side wins: element side of sw0 = bus1, connected to bus2 via sw1
        assert sorted(result[0]) == [1, 2]

    def test_protection_side_bus_differs_from_relay_side_element(self, linear_net):
        """Mirror case: protection_side='bus', relay_side='element' → {bus0}."""
        el_list = _make_el_list(switch_id=0, protection_side="bus", relay_side="element")
        result = compute_affected_nodes(linear_net, el_list)

        # protection_side wins: bus side of sw0 = bus0, isolated after sw0 opens
        assert result[0] == [0]
