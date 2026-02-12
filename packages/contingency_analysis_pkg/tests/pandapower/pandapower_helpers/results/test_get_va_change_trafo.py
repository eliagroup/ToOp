# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.va_diff_results import (
    get_va_change_for_tr3,
    get_va_change_for_trafo,
    get_va_change_tr3_hv_lv,
    get_va_change_tr3_hv_mv,
    get_va_change_tr3_mv_lv,
)


def create_test_net_with_trafos() -> pp.pandapowerNet:
    net = pp.create_empty_network()

    # --- buses ---
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=20)
    b3 = pp.create_bus(net, vn_kv=10)

    # --- 2W trafo ---
    t1 = pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )

    # --- 3W trafo ---
    t3 = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=40,
        sn_mv_mva=20,
        sn_lv_mva=20,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10,
        vk_mv_percent=10,
        vk_lv_percent=10,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_mv_degree=0,
        shift_lv_degree=0,
    )

    # --- characteristic table (tabular PST info) ---
    net.trafo_characteristic_table = pd.DataFrame(
        {
            "id_characteristic": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "angle_deg": [0.0, 15.0, 0.0, 0.0],  # only characteristic 1 has non-zero angle
        }
    )

    # --- assign tabular tap changers ---
    net.trafo.loc[t1, "tap_changer_type"] = "Tabular"
    net.trafo.loc[t1, "id_characteristic_table"] = 1

    net.trafo3w.loc[t3, "tap_changer_type"] = "Tabular"
    net.trafo3w.loc[t3, "id_characteristic_table"] = 1

    return net


@pytest.fixture
def net_pp():
    return create_test_net_with_trafos()


def test_hv_to_lv_tap_on_hv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(hv), int(lv))
    assert res == 15.0  # angle + shift_lv


def test_hv_to_lv_tap_on_mv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(hv), int(lv))
    assert res == 5.0  # only shift_lv


def test_hv_to_lv_tap_on_lv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(hv), int(lv))
    assert res == -5.0  # -angle + shift_lv


def test_lv_to_hv_tap_on_hv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(lv), int(hv))
    assert res == -15.0  # -angle - shift_lv


def test_lv_to_hv_tap_on_mv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(lv), int(hv))
    assert res == -5.0  # -shift_lv


def test_lv_to_hv_tap_on_lv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(lv), int(hv))
    assert res == 5.0  # angle - shift_lv


def test_returns_none_when_not_hv_lv_pair(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    res = get_va_change_tr3_hv_lv(net_pp, t3, int(hv), int(mv))
    assert res is None


def test_hv_to_mv_tap_on_hv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(hv), int(mv))
    assert res == 13.0  # angle + shift_mv


def test_hv_to_mv_tap_on_mv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(hv), int(mv))
    assert res == -7.0  # -angle + shift_mv


def test_hv_to_mv_tap_on_lv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(hv), int(mv))
    assert res == 3.0  # only shift_mv


def test_mv_to_hv_tap_on_hv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(mv), int(hv))
    assert res == -13.0  # -angle - shift_mv


def test_mv_to_hv_tap_on_mv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(mv), int(hv))
    assert res == 7.0  # angle - shift_mv


def test_mv_to_hv_tap_on_lv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(mv), int(hv))
    assert res == -3.0  # -shift_mv


def test_returns_none_when_not_hv_mv_pair(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0

    res = get_va_change_tr3_hv_mv(net_pp, t3, int(hv), int(lv))
    assert res is None


def test_mv_to_lv_tap_on_hv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(mv), int(lv))
    assert res == 4.0  # shift_lv - shift_mv


def test_mv_to_lv_tap_on_mv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(mv), int(lv))
    assert res == 14.0  # angle + shift_lv - shift_mv


def test_mv_to_lv_tap_on_lv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(mv), int(lv))
    assert res == -6.0  # -angle + shift_lv - shift_mv


def test_lv_to_mv_tap_on_hv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(lv), int(mv))
    assert res == -4.0  # -shift_lv + shift_mv


def test_lv_to_mv_tap_on_mv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(lv), int(mv))
    assert res == -14.0  # -angle - shift_lv + shift_mv


def test_lv_to_mv_tap_on_lv(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(lv), int(mv))
    assert res == 6.0  # angle - shift_lv + shift_mv


def test_returns_none_when_not_mv_lv_pair(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_tr3_mv_lv(net_pp, t3, int(mv), int(hv))
    assert res is None


def test_hv_lv_path_uses_hv_lv_function(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0  # irrelevant for HV↔LV

    res = get_va_change_for_tr3(net_pp, t3, int(hv), int(lv))
    assert res == 15.0  # angle + shift_lv


def test_hv_mv_path_uses_hv_mv_function(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0  # irrelevant for HV↔MV

    res = get_va_change_for_tr3(net_pp, t3, int(hv), int(mv))
    assert res == -7.0  # -angle + shift_mv


def test_mv_lv_path_uses_mv_lv_function(net_pp):
    t3 = 0
    mv = net_pp.trafo3w.loc[t3, "mv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "lv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_for_tr3(net_pp, t3, int(mv), int(lv))
    assert res == -6.0  # -angle + shift_lv - shift_mv


def test_reverse_direction_negates_value_for_hv_lv(net_pp):
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]
    lv = net_pp.trafo3w.loc[t3, "lv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 5.0

    forward = get_va_change_for_tr3(net_pp, t3, int(hv), int(lv))
    backward = get_va_change_for_tr3(net_pp, t3, int(lv), int(hv))

    assert forward == 15.0
    assert backward == -15.0
    assert backward == -forward


def test_returns_zero_for_non_matching_bus_pair(net_pp):
    # e.g. HV->HV: none of the pairwise functions match, so it should return 0.0
    t3 = 0
    hv = net_pp.trafo3w.loc[t3, "hv_bus"]

    net_pp.trafo3w.loc[t3, "tap_side"] = "hv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_for_tr3(net_pp, t3, int(hv), int(hv))
    assert res == 0.0


def test_returns_zero_for_wrong_transformer_id_pair(net_pp):
    # still a valid net, but passing buses that don't correspond to the trafo's sides
    t3 = 0
    other_bus = pp.create_bus(net_pp, vn_kv=20)

    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "angle_deg"] = 10.0
    net_pp.trafo3w.loc[t3, "shift_mv_degree"] = 3.0
    net_pp.trafo3w.loc[t3, "shift_lv_degree"] = 7.0

    res = get_va_change_for_tr3(net_pp, t3, int(other_bus), int(other_bus))
    assert res == 0.0


def test_from_hv_tap_on_hv(net_pp):
    t = 0
    hv = net_pp.trafo.loc[t, "hv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    res = get_va_change_for_trafo(net_pp, t, int(hv))
    assert res == 15.0  # angle + shift


def test_from_hv_tap_on_lv(net_pp):
    t = 0
    hv = net_pp.trafo.loc[t, "hv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "lv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    res = get_va_change_for_trafo(net_pp, t, int(hv))
    assert res == -5.0  # -angle + shift


def test_from_lv_tap_on_hv(net_pp):
    t = 0
    lv = net_pp.trafo.loc[t, "lv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    res = get_va_change_for_trafo(net_pp, t, int(lv))
    assert res == -15.0  # -angle - shift


def test_from_lv_tap_on_lv(net_pp):
    t = 0
    lv = net_pp.trafo.loc[t, "lv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "lv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    res = get_va_change_for_trafo(net_pp, t, int(lv))
    assert res == 5.0  # angle - shift


def test_from_other_bus_returns_zero(net_pp):
    t = 0
    other = pp.create_bus(net_pp, vn_kv=20)

    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    res = get_va_change_for_trafo(net_pp, t, int(other))
    assert res == 0.0


def test_reverse_traversal_negates_when_tap_on_hv(net_pp):
    t = 0
    hv = net_pp.trafo.loc[t, "hv_bus"]
    lv = net_pp.trafo.loc[t, "lv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    forward = get_va_change_for_trafo(net_pp, t, int(hv))  # HV -> LV
    backward = get_va_change_for_trafo(net_pp, t, int(lv))  # LV -> HV

    assert forward == 15.0
    assert backward == -15.0
    assert backward == -forward


def test_reverse_traversal_negates_when_tap_on_lv(net_pp):
    t = 0
    hv = net_pp.trafo.loc[t, "hv_bus"]
    lv = net_pp.trafo.loc[t, "lv_bus"]

    net_pp.trafo.loc[t, "tap_side"] = "lv"
    net_pp.trafo.loc[t, "angle_deg"] = 10.0
    net_pp.trafo.loc[t, "shift_degree"] = 5.0

    forward = get_va_change_for_trafo(net_pp, t, int(hv))  # HV -> LV
    backward = get_va_change_for_trafo(net_pp, t, int(lv))  # LV -> HV

    assert forward == -5.0
    assert backward == 5.0
    assert backward == -forward
