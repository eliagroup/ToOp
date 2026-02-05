# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower as pp
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple


def _add_z_ohm_column_if_missing(net):
    if "z_ohm" not in net.switch.columns:
        net.switch["z_ohm"] = np.nan


def _mk_net(n_buses: int) -> pp.pandapowerNet:
    net = pp.create_empty_network()
    for _ in range(n_buses):
        pp.create_bus(net, vn_kv=110.0)
    _add_z_ohm_column_if_missing(net)
    return net


def _bus_switch(net, a, b, *, closed=True, z_ohm=0.0):
    """Create a bus-bus switch between bus a and bus b."""
    idx = pp.create_switch(net, bus=a, element=b, et="b", closed=closed)
    _add_z_ohm_column_if_missing(net)
    net.switch.loc[idx, "z_ohm"] = z_ohm
    return idx


def test_empty_network():
    net = pp.create_empty_network()
    # ensure required columns exist even if empty
    _add_z_ohm_column_if_missing(net)
    bus_lookup, merged = create_bus_lookup_simple(net)
    assert bus_lookup == []
    assert merged == []


def test_no_switches_no_merge():
    net = _mk_net(3)
    bus_lookup, merged = create_bus_lookup_simple(net)
    # bus_lookup should map [0,1,2] -> [0,1,2] (consecutive indexing)
    assert bus_lookup == [0, 1, 2]
    assert merged == [False, False, False]


def test_open_switch_is_ignored():
    net = _mk_net(2)
    _bus_switch(net, 0, 1, closed=False, z_ohm=0.0)  # open -> ignored
    bus_lookup, merged = create_bus_lookup_simple(net)
    assert bus_lookup == [0, 1]
    assert merged == [False, False]


def test_nonzero_impedance_switch_is_ignored():
    net = _mk_net(2)
    _bus_switch(net, 0, 1, closed=True, z_ohm=0.1)  # non-zero -> ignored
    bus_lookup, merged = create_bus_lookup_simple(net)
    assert bus_lookup == [0, 1]
    assert merged == [False, False]


def test_simple_closed_bus_bus_merge():
    net = _mk_net(2)
    _bus_switch(net, 0, 1, closed=True, z_ohm=0.0)
    bus_lookup, merged = create_bus_lookup_simple(net)
    # Representative is the bus with minimal current lookup (0),
    # so both buses map to 0; bus 1 is marked merged.
    assert bus_lookup == [0, 0]
    assert merged == [False, True]


def test_transitive_chain_merges():
    net = _mk_net(3)
    _bus_switch(net, 0, 1, closed=True, z_ohm=0.0)
    _bus_switch(net, 1, 2, closed=True, z_ohm=0.0)
    bus_lookup, merged = create_bus_lookup_simple(net)
    # All three should collapse to representative 0
    assert bus_lookup == [0, 0, 0]
    assert merged == [False, True, True]


def test_multiple_disjoint_components_and_isolated_bus():
    net = _mk_net(6)  # buses: 0..5
    # Component A: 0-1 (merge -> rep 0)
    _bus_switch(net, 0, 1, closed=True, z_ohm=0.0)
    # Component B (chain): 3-4-5 (merge -> rep 3)
    _bus_switch(net, 3, 4, closed=True, z_ohm=0.0)
    _bus_switch(net, 4, 5, closed=True, z_ohm=0.0)
    # Bus 2 stays isolated
    bus_lookup, merged = create_bus_lookup_simple(net)

    # Expected mapping:
    # 0,1 -> 0; 2 -> 2; 3,4,5 -> 3
    assert bus_lookup == [0, 0, 2, 3, 3, 3]
    assert merged == [False, True, False, False, True, True]


def test_ignores_line_switches_and_element_types_other_than_bus():
    net = _mk_net(3)
    # Make a line so we can create a line switch (ignored by et != 'b')
    line = pp.create_line_from_parameters(
        net, from_bus=0, to_bus=1, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10.0, max_i_ka=0.2
    )
    # Line switch (should be ignored due to et='l')
    idx_l = pp.create_switch(net, bus=0, element=line, et="l", closed=True)
    _add_z_ohm_column_if_missing(net)
    net.switch.loc[idx_l, "z_ohm"] = 0.0

    # Add also a valid bus-bus zero-ohm switch to check it still works
    _bus_switch(net, 1, 2, closed=True, z_ohm=0.0)

    bus_lookup, merged = create_bus_lookup_simple(net)
    # Only 1-2 should merge; 0 should be unaffected.
    assert bus_lookup == [0, 1, 1]
    assert merged == [False, False, True]
