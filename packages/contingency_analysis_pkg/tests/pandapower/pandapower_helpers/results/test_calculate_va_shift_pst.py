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
    _apply_ideal_angle,
    _apply_ratio_angle,
    _apply_symmetrical_angle,
    _apply_tabular_angle,
    _select_vn_kv,
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


import numpy as np
import pandapower as pp


def test_applies_angle_for_tabular_trafo_matching_characteristic_and_step(net_pp):
    # existing trafo 0 is Tabular and points to characteristic 1
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[t, "id_characteristic_table"] = 1
    net_pp.trafo.loc[t, "tap_pos"] = 1  # step=1 in characteristic 1 -> 15 deg
    net_pp.trafo.loc[t, "angle_deg"] = 999.0  # sentinel

    _apply_tabular_angle(net_pp, net_pp.trafo)

    assert net_pp.trafo.loc[t, "angle_deg"] == 15.0


def test_applies_angle_for_step_zero(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[t, "id_characteristic_table"] = 1
    net_pp.trafo.loc[t, "tap_pos"] = 0  # step=0 in characteristic 1 -> 0 deg
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_tabular_angle(net_pp, net_pp.trafo)

    assert net_pp.trafo.loc[t, "angle_deg"] == 0.0


def test_does_not_modify_non_tabular_rows(net_pp):
    # Make trafo 0 tabular and matching so we can see changes only there
    net_pp.trafo.loc[0, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[0, "id_characteristic_table"] = 1
    net_pp.trafo.loc[0, "tap_pos"] = 1
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Add another trafo (non-tabular) with a known angle that must remain unchanged
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t2, "angle_deg"] = 111.0
    net_pp.trafo.loc[t2, "tap_pos"] = 1
    net_pp.trafo.loc[t2, "id_characteristic_table"] = 1

    _apply_tabular_angle(net_pp, net_pp.trafo)

    assert net_pp.trafo.loc[0, "angle_deg"] == 15.0
    assert net_pp.trafo.loc[t2, "angle_deg"] == 111.0  # unchanged


def test_sets_angle_to_nan_when_no_characteristic_step_match(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[t, "id_characteristic_table"] = 1
    net_pp.trafo.loc[t, "tap_pos"] = 999  # no such step
    net_pp.trafo.loc[t, "angle_deg"] = 123.0  # will be overwritten to NaN

    _apply_tabular_angle(net_pp, net_pp.trafo)

    assert np.isnan(net_pp.trafo.loc[t, "angle_deg"])


def test_noop_when_no_tabular_rows(net_pp):
    # Force all to non-tabular
    net_pp.trafo["tap_changer_type"] = "Ratio"
    net_pp.trafo["angle_deg"] = 42.0

    before = net_pp.trafo["angle_deg"].copy()
    _apply_tabular_angle(net_pp, net_pp.trafo)
    after = net_pp.trafo["angle_deg"]

    assert before.equals(after)


def test_updates_multiple_tabular_rows_independently(net_pp):
    # Ensure characteristic 2 exists and has angle 0 at step 1 in your fixture
    # characteristic 2: steps 0,1 -> angles 0,0
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )

    # Trafo 0 -> characteristic 1, step 1 -> 15
    net_pp.trafo.loc[0, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[0, "id_characteristic_table"] = 1
    net_pp.trafo.loc[0, "tap_pos"] = 1
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Trafo t2 -> characteristic 2, step 1 -> 0
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[t2, "id_characteristic_table"] = 2
    net_pp.trafo.loc[t2, "tap_pos"] = 1
    net_pp.trafo.loc[t2, "angle_deg"] = 888.0

    _apply_tabular_angle(net_pp, net_pp.trafo)

    assert net_pp.trafo.loc[0, "angle_deg"] == 15.0
    assert net_pp.trafo.loc[t2, "angle_deg"] == 0.0


def test_applies_ideal_angle_from_tap_pos_minus_neutral_times_step_degree(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_degree"] = 2.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0  # sentinel

    _apply_ideal_angle(net_pp.trafo)

    # (5 - 3) * 2 = 4
    assert net_pp.trafo.loc[t, "angle_deg"] == 4.0


def test_negative_angle_when_tap_pos_below_neutral(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t, "tap_pos"] = 1
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_degree"] = 2.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_ideal_angle(net_pp.trafo)

    # (1 - 3) * 2 = -4
    assert net_pp.trafo.loc[t, "angle_deg"] == -4.0


def test_zero_angle_when_tap_pos_equals_neutral(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t, "tap_pos"] = 3
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_degree"] = 2.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_ideal_angle(net_pp.trafo)

    assert net_pp.trafo.loc[t, "angle_deg"] == 0.0


def test_does_not_modify_non_ideal_rows(net_pp):
    # Make trafo 0 ideal so it changes
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Add another trafo that is non-ideal and must not be touched
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t2, "angle_deg"] = 111.0
    net_pp.trafo.loc[t2, "tap_pos"] = 5
    net_pp.trafo.loc[t2, "tap_neutral"] = 3
    net_pp.trafo.loc[t2, "tap_step_degree"] = 2.0

    _apply_ideal_angle(net_pp.trafo)

    assert net_pp.trafo.loc[0, "angle_deg"] == 4.0
    assert net_pp.trafo.loc[t2, "angle_deg"] == 111.0  # unchanged


def test_noop_when_no_ideal_rows(net_pp):
    net_pp.trafo["tap_changer_type"] = "Ratio"
    net_pp.trafo["angle_deg"] = 42.0

    before = net_pp.trafo["angle_deg"].copy()
    _apply_ideal_angle(net_pp.trafo)
    after = net_pp.trafo["angle_deg"]

    assert before.equals(after)


def test_updates_multiple_ideal_rows_independently(net_pp):
    # Add a second trafo and make both ideal with different tap settings
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )

    # Trafo 0: (5-3)*2 = 4
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Trafo t2: (2-4)*1.5 = -3
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t2, "tap_pos"] = 2
    net_pp.trafo.loc[t2, "tap_neutral"] = 4
    net_pp.trafo.loc[t2, "tap_step_degree"] = 1.5
    net_pp.trafo.loc[t2, "angle_deg"] = 888.0

    _apply_ideal_angle(net_pp.trafo)

    assert net_pp.trafo.loc[0, "angle_deg"] == 4.0
    assert net_pp.trafo.loc[t2, "angle_deg"] == -3.0


def _expected_ratio_angle_deg(
    vn_kv: float, tap_pos: float, tap_neutral: float, tap_step_percent: float, tap_step_degree: float
) -> float:
    tap_diff = tap_pos - tap_neutral
    du = vn_kv * tap_diff * (tap_step_percent / 100.0)

    step_rad = np.deg2rad(tap_step_degree)
    num = du * np.sin(step_rad)
    den = vn_kv + du * np.cos(step_rad)

    return float(np.rad2deg(np.arctan2(num, den)))


def test_select_vn_kv_2w_hv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    vn = _select_vn_kv(net_pp.trafo.loc[[t]])
    assert vn[0] == 110.0


def test_select_vn_kv_2w_lv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_side"] = "lv"
    vn = _select_vn_kv(net_pp.trafo.loc[[t]])
    assert vn[0] == 20.0


def test_select_vn_kv_2w_mv_is_nan(net_pp):
    # 2W trafo has no vn_mv_kv -> mv selection should be NaN
    t = 0
    net_pp.trafo.loc[t, "tap_side"] = "mv"
    vn = _select_vn_kv(net_pp.trafo.loc[[t]])
    assert np.isnan(vn[0])


def test_select_vn_kv_3w_mv(net_pp):
    t3 = 0
    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    vn = _select_vn_kv(net_pp.trafo3w.loc[[t3]])
    assert vn[0] == 20.0


def test_apply_ratio_angle_2w_uses_vn_hv_kv_when_tap_side_hv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_ratio_angle(net_pp.trafo)

    expected = _expected_ratio_angle_deg(110.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_ratio_angle_2w_uses_vn_lv_kv_when_tap_side_lv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t, "tap_side"] = "lv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_ratio_angle(net_pp.trafo)

    expected = _expected_ratio_angle_deg(20.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_ratio_angle_3w_uses_vn_mv_kv_when_tap_side_mv(net_pp):
    t3 = 0
    net_pp.trafo3w.loc[t3, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "tap_pos"] = 5
    net_pp.trafo3w.loc[t3, "tap_neutral"] = 3
    net_pp.trafo3w.loc[t3, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[t3, "tap_step_degree"] = 30.0
    net_pp.trafo3w.loc[t3, "angle_deg"] = 999.0

    _apply_ratio_angle(net_pp.trafo3w)

    expected = _expected_ratio_angle_deg(20.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo3w.loc[t3, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_ratio_angle_2w_tap_side_mv_results_in_nan_angle(net_pp):
    # because vn_kv becomes NaN for 2W when tap_side == "mv"
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t, "tap_side"] = "mv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 123.0

    _apply_ratio_angle(net_pp.trafo)

    assert np.isnan(net_pp.trafo.loc[t, "angle_deg"])


def test_apply_ratio_angle_zero_step_degree_is_zero_angle_when_vn_defined(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 0.0

    _apply_ratio_angle(net_pp.trafo)

    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(0.0, abs=1e-15)


def test_apply_ratio_angle_negative_tap_diff_changes_sign(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 1
    net_pp.trafo.loc[t, "tap_neutral"] = 3  # tap_diff=-2
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0

    _apply_ratio_angle(net_pp.trafo)

    expected = _expected_ratio_angle_deg(110.0, 1, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t, "angle_deg"] < 0


def test_noop_when_no_ratio_rows(net_pp):
    net_pp.trafo["tap_changer_type"] = "Tabular"
    net_pp.trafo["angle_deg"] = 42.0

    before = net_pp.trafo["angle_deg"].copy()
    _apply_ratio_angle(net_pp.trafo)
    after = net_pp.trafo["angle_deg"]

    assert before.equals(after)


def test_does_not_modify_non_ratio_rows(net_pp):
    # make trafo 0 ratio so it changes
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_side"] = "hv"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Add a second trafo that is non-ratio and must remain unchanged
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t2, "angle_deg"] = 111.0

    _apply_ratio_angle(net_pp.trafo)

    expected0 = _expected_ratio_angle_deg(110.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[0, "angle_deg"] == pytest.approx(expected0, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t2, "angle_deg"] == 111.0  # unchanged


def test_updates_multiple_ratio_rows_independently(net_pp):
    # Add second trafo and make both ratio with different params (both tap_side hv for determinism)
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )

    # trafo 0
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_side"] = "hv"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # trafo t2
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[t2, "tap_side"] = "hv"
    net_pp.trafo.loc[t2, "tap_pos"] = 2
    net_pp.trafo.loc[t2, "tap_neutral"] = 4  # tap_diff=-2
    net_pp.trafo.loc[t2, "tap_step_percent"] = 2.0
    net_pp.trafo.loc[t2, "tap_step_degree"] = 15.0
    net_pp.trafo.loc[t2, "angle_deg"] = 888.0

    _apply_ratio_angle(net_pp.trafo)

    expected0 = _expected_ratio_angle_deg(110.0, 5, 3, 1.0, 30.0)
    expected2 = _expected_ratio_angle_deg(110.0, 2, 4, 2.0, 15.0)

    assert net_pp.trafo.loc[0, "angle_deg"] == pytest.approx(expected0, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t2, "angle_deg"] == pytest.approx(expected2, rel=1e-12, abs=1e-12)


def _expected_sym_angle_deg(
    vn_kv: float, tap_pos: float, tap_neutral: float, tap_step_percent: float, tap_step_degree: float
) -> float:
    tap_diff = tap_pos - tap_neutral
    du = vn_kv * tap_diff * (tap_step_percent / 100.0)

    step_rad = np.deg2rad(tap_step_degree)
    num = du * np.sin(step_rad)
    den = vn_kv + du * np.cos(step_rad)

    return float(np.rad2deg(np.arctan2(num, den)))


def test_apply_symmetrical_angle_2w_uses_vn_hv_kv_when_tap_side_hv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0  # sentinel

    _apply_symmetrical_angle(net_pp.trafo)

    expected = _expected_sym_angle_deg(110.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_symmetrical_angle_2w_uses_vn_lv_kv_when_tap_side_lv(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t, "tap_side"] = "lv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 999.0

    _apply_symmetrical_angle(net_pp.trafo)

    expected = _expected_sym_angle_deg(20.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_symmetrical_angle_3w_uses_vn_mv_kv_when_tap_side_mv(net_pp):
    t3 = 0
    net_pp.trafo3w.loc[t3, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[t3, "tap_side"] = "mv"
    net_pp.trafo3w.loc[t3, "tap_pos"] = 5
    net_pp.trafo3w.loc[t3, "tap_neutral"] = 3
    net_pp.trafo3w.loc[t3, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[t3, "tap_step_degree"] = 30.0
    net_pp.trafo3w.loc[t3, "angle_deg"] = 999.0

    _apply_symmetrical_angle(net_pp.trafo3w)

    expected = _expected_sym_angle_deg(20.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo3w.loc[t3, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_apply_symmetrical_angle_2w_tap_side_mv_results_in_nan_angle(net_pp):
    # because vn_kv becomes NaN for 2W when tap_side == "mv"
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t, "tap_side"] = "mv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[t, "angle_deg"] = 123.0

    _apply_symmetrical_angle(net_pp.trafo)

    assert np.isnan(net_pp.trafo.loc[t, "angle_deg"])


def test_apply_symmetrical_angle_zero_step_degree_is_zero_angle_when_vn_defined(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 5
    net_pp.trafo.loc[t, "tap_neutral"] = 3
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 0.0

    _apply_symmetrical_angle(net_pp.trafo)

    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(0.0, abs=1e-15)


def test_apply_symmetrical_angle_negative_tap_diff_changes_sign(net_pp):
    t = 0
    net_pp.trafo.loc[t, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t, "tap_side"] = "hv"
    net_pp.trafo.loc[t, "tap_pos"] = 1
    net_pp.trafo.loc[t, "tap_neutral"] = 3  # tap_diff=-2
    net_pp.trafo.loc[t, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[t, "tap_step_degree"] = 30.0

    _apply_symmetrical_angle(net_pp.trafo)

    expected = _expected_sym_angle_deg(110.0, 1, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[t, "angle_deg"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t, "angle_deg"] < 0


def test_noop_when_no_symmetrical_rows(net_pp):
    net_pp.trafo["tap_changer_type"] = "Tabular"
    net_pp.trafo["angle_deg"] = 42.0

    before = net_pp.trafo["angle_deg"].copy()
    _apply_symmetrical_angle(net_pp.trafo)
    after = net_pp.trafo["angle_deg"]

    assert before.equals(after)


def test_does_not_modify_non_symmetrical_rows(net_pp):
    # make trafo 0 symmetrical so it changes
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_side"] = "hv"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # Add a second trafo that is non-symmetrical and must remain unchanged
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[t2, "angle_deg"] = 111.0

    _apply_symmetrical_angle(net_pp.trafo)

    expected0 = _expected_sym_angle_deg(110.0, 5, 3, 1.0, 30.0)
    assert net_pp.trafo.loc[0, "angle_deg"] == pytest.approx(expected0, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t2, "angle_deg"] == 111.0  # unchanged


def test_updates_multiple_symmetrical_rows_independently(net_pp):
    # Add second trafo and make both symmetrical with different params (both tap_side hv for determinism)
    b_hv = net_pp.bus.index[0]
    b_lv = net_pp.bus.index[1]
    t2 = pp.create_transformer_from_parameters(
        net_pp,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vk_percent=10,
        vkr_percent=0.3,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )

    # trafo 0
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_side"] = "hv"
    net_pp.trafo.loc[0, "tap_pos"] = 5
    net_pp.trafo.loc[0, "tap_neutral"] = 3
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 30.0
    net_pp.trafo.loc[0, "angle_deg"] = 999.0

    # trafo t2
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t2, "tap_side"] = "hv"
    net_pp.trafo.loc[t2, "tap_pos"] = 2
    net_pp.trafo.loc[t2, "tap_neutral"] = 4  # tap_diff=-2
    net_pp.trafo.loc[t2, "tap_step_percent"] = 2.0
    net_pp.trafo.loc[t2, "tap_step_degree"] = 15.0
    net_pp.trafo.loc[t2, "angle_deg"] = 888.0

    _apply_symmetrical_angle(net_pp.trafo)

    expected0 = _expected_sym_angle_deg(110.0, 5, 3, 1.0, 30.0)
    expected2 = _expected_sym_angle_deg(110.0, 2, 4, 2.0, 15.0)

    assert net_pp.trafo.loc[0, "angle_deg"] == pytest.approx(expected0, rel=1e-12, abs=1e-12)
    assert net_pp.trafo.loc[t2, "angle_deg"] == pytest.approx(expected2, rel=1e-12, abs=1e-12)
