# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for the per-outage net copy.

The copy shares memory with the source net, so the property that matters is isolation: an
outage may write any column listed in ``MUTABLE_COLUMNS_BY_TABLE`` and the source net must not
notice. A missed column would corrupt every later outage silently, which is what these tests
exist to prevent.
"""

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.outage_net_copy import (
    MUTABLE_COLUMNS_BY_TABLE,
    copy_net_for_outage,
)

#: Impedance the characteristic table hands back at tap position 0. Different from the std type's
#: 12.0 / 0.41 so the write is visible, but still solvable.
TAP_VK_PERCENT = 13.5
TAP_VKR_PERCENT = 0.55


def build_net() -> pp.pandapowerNet:
    """Small net carrying one row in every table the copy treats specially."""
    net = pp.create_empty_network()
    hv_a = pp.create_bus(net, vn_kv=110.0, name="hv_a")
    hv_b = pp.create_bus(net, vn_kv=110.0, name="hv_b")
    lv = pp.create_bus(net, vn_kv=20.0, name="lv")

    pp.create_ext_grid(net, bus=hv_a, vm_pu=1.0)
    pp.create_line(net, from_bus=hv_a, to_bus=hv_b, length_km=10.0, std_type="NAYY 4x50 SE")
    pp.create_transformer(net, hv_bus=hv_b, lv_bus=lv, std_type="25 MVA 110/20 kV")
    pp.create_switch(net, bus=hv_a, element=hv_b, et="b", closed=True)
    pp.create_impedance(net, from_bus=hv_a, to_bus=hv_b, rft_pu=0.1, xft_pu=0.1, sn_mva=100.0)

    pp.create_load(net, bus=lv, p_mw=1.0, q_mvar=0.5)
    pp.create_shunt(net, bus=lv, q_mvar=-1.0, p_mw=0.1)
    pp.create_gen(net, bus=hv_b, p_mw=5.0, vm_pu=1.0)
    pp.create_sgen(net, bus=lv, p_mw=2.0, q_mvar=0.1)
    pp.create_ward(net, bus=lv, ps_mw=1.0, qs_mvar=1.0, pz_mw=1.0, qz_mvar=1.0)
    pp.create_xward(net, bus=lv, ps_mw=1.0, qs_mvar=1.0, pz_mw=1.0, qz_mvar=1.0, r_ohm=0.1, x_ohm=0.1, vm_pu=1.0)
    pp.create_measurement(net, meas_type="v", element_type="bus", value=1.0, std_dev=0.01, element=hv_a)
    return net


def distinct_value(current: object) -> object:
    """A value of the same kind as *current* but different from it."""
    if isinstance(current, (bool, np.bool_)):
        return not bool(current)
    return float(current) + 1.0 if not np.isnan(float(current)) else 1.0


def mutable_column_cases() -> list[tuple[str, str]]:
    """Every (table, column) pair in the map that exists in :func:`build_net`."""
    net = build_net()
    return [
        (table, column)
        for table, columns in MUTABLE_COLUMNS_BY_TABLE.items()
        for column in columns
        if table in net and len(net[table]) and column in net[table].columns
    ]


@pytest.fixture()
def net() -> pp.pandapowerNet:
    return build_net()


def test_copy_holds_the_same_values(net: pp.pandapowerNet) -> None:
    copied = copy_net_for_outage(net)

    assert set(copied) == set(net)
    for key, value in net.items():
        if hasattr(value, "columns"):
            assert copied[key].equals(value), f"{key} differs from the source net"


@pytest.mark.parametrize(("table", "column"), mutable_column_cases())
def test_writing_a_mutable_column_does_not_reach_the_original(net: pp.pandapowerNet, table: str, column: str) -> None:
    """The whole point of the map: these columns are writable without side effects."""
    copied = copy_net_for_outage(net)
    row = net[table].index[0]
    before = net[table].at[row, column]

    copied[table].loc[row, column] = distinct_value(before)

    assert copied[table].at[row, column] != before
    assert net[table].at[row, column] == before


@pytest.mark.parametrize("table", ["gen", "sgen"])
def test_unlisted_tables_are_fully_independent(net: pp.pandapowerNet, table: str) -> None:
    """Tables outside the map are deep-copied, so even row changes stay local.

    ``replace_sgen_by_gen`` adds and drops rows during slack allocation, which is why these
    cannot be shared column-wise.
    """
    copied = copy_net_for_outage(net)
    row = net[table].index[0]

    copied[table].at[row, "p_mw"] = 999.0
    copied[table] = copied[table].drop(row)

    assert len(net[table]) == 1
    assert net[table].at[row, "p_mw"] != 999.0


def test_shared_columns_really_share_memory(net: pp.pandapowerNet) -> None:
    """Guards the optimisation itself: if this fails the copy silently became a deepcopy."""
    copied = copy_net_for_outage(net)

    assert np.shares_memory(copied.bus["name"].to_numpy(), net.bus["name"].to_numpy())
    assert np.shares_memory(copied.measurement["value"].to_numpy(), net.measurement["value"].to_numpy())
    # ... while the writable ones were detached.
    assert not np.shares_memory(copied.bus["in_service"].to_numpy(), net.bus["in_service"].to_numpy())
    assert not np.shares_memory(copied.switch["closed"].to_numpy(), net.switch["closed"].to_numpy())


def test_adding_or_replacing_a_column_stays_local(net: pp.pandapowerNet) -> None:
    """A shared table still gets its own block manager, so structural edits do not leak."""
    copied = copy_net_for_outage(net)

    copied.bus["extra_flag"] = True
    copied.measurement["value"] = 42.0

    assert "extra_flag" not in net.bus.columns
    assert net.measurement["value"].iloc[0] != 42.0


def test_map_lists_only_columns_that_exist(net: pp.pandapowerNet) -> None:
    """A column named here but absent from the table is silently skipped - catch the typo."""
    unknown = [
        f"{table}.{column}"
        for table, columns in MUTABLE_COLUMNS_BY_TABLE.items()
        if table in net
        for column in columns
        if column not in net[table].columns
    ]
    assert not unknown, f"columns in MUTABLE_COLUMNS_BY_TABLE that no such table has: {unknown}"


def with_tap_dependent_impedance(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Make the trafo take its impedance from the characteristic table at its tap position.

    This is the branch in ``build_branch._get_vk_values_from_table`` that writes the looked-up
    values back into the trafo columns; without it the write never happens.
    """
    net.trafo["tap_dependency_table"] = True
    net.trafo["id_characteristic_table"] = 0
    net.trafo["tap_pos"] = 0
    net.trafo_characteristic_table = pd.DataFrame(
        {
            "id_characteristic": [0],
            "step": [0],
            "voltage_ratio": [1.0],
            "angle_deg": [0.0],
            "vk_percent": [TAP_VK_PERCENT],
            "vkr_percent": [TAP_VKR_PERCENT],
            "vkr_hv_percent": [np.nan],
            "vkr_mv_percent": [np.nan],
            "vkr_lv_percent": [np.nan],
            "vk_hv_percent": [np.nan],
            "vk_mv_percent": [np.nan],
            "vk_lv_percent": [np.nan],
        }
    )
    return net


def test_tap_dependent_impedance_write_stays_on_the_copy(net: pp.pandapowerNet) -> None:
    """pandapower writes tap-dependent impedance into net.trafo in place during every solve.

    That is why ``vk_percent``/``vkr_percent`` are in the map. Without them the trafo columns
    would be shared and the first outage would rewrite the source net's impedances.
    """
    with_tap_dependent_impedance(net)
    before = net.trafo["vk_percent"].copy()

    copied = copy_net_for_outage(net)
    pp.runpp(copied)

    # The solve really did take the write path, otherwise this test proves nothing.
    assert copied.trafo["vk_percent"].iloc[0] == TAP_VK_PERCENT
    assert not before.equals(copied.trafo["vk_percent"])
    # ... and the source net kept its own values.
    assert net.trafo["vk_percent"].equals(before)


def test_power_flow_on_the_copy_leaves_the_original_untouched(net: pp.pandapowerNet) -> None:
    """End-to-end property, and the reason the trafo impedance columns are in the map.

    pandapower rewrites tap-dependent impedance into the trafo columns during ``_pd2ppc``, so a
    solve on the copy must not be visible on the source net.
    """
    pp.runpp(net)
    before = {key: value.copy(deep=True) for key, value in net.items() if hasattr(value, "columns")}

    copied = copy_net_for_outage(net)
    copied.line.loc[net.line.index[0], "in_service"] = False
    pp.runpp(copied)

    changed = [key for key, frame in before.items() if not net[key].equals(frame)]
    assert not changed, f"the source net was modified by a solve on its copy: {changed}"
