# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.va_diff_results import (
    get_pst_elements_ideal,
    get_pst_elements_ratio,
    get_pst_elements_symmetrical,
    get_pst_elements_tabular,
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


def test_returns_trafo_and_trafo3w_when_tabular_and_characteristic_has_nonzero_angle(net_pp):
    res = get_pst_elements_tabular(net_pp)
    # created net has exactly one of each with characteristic 1 (non-zero angle)
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_excludes_characteristic_with_zero_angle_tabular(net_pp):
    # Add another 2W trafo, but point to characteristic 2 which has only 0 angles
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
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[t2, "id_characteristic_table"] = 2  # zero-angle characteristic

    res = get_pst_elements_tabular(net_pp)
    assert "trafo:1" not in res
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_excludes_non_tabular_even_if_characteristic_has_nonzero_angle_tabular(net_pp):
    # Flip the existing trafo to Ratio -> must be excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"

    res = get_pst_elements_tabular(net_pp)
    assert "e&&trafo&&0" not in res
    assert res == ["e&&trafo3w&&0"]


def test_empty_when_no_characteristic_has_nonzero_angle_tabular(net_pp):
    # Force all angles to zero
    net_pp.trafo_characteristic_table["angle_deg"] = 0.0

    res = get_pst_elements_tabular(net_pp)
    assert res == []


def test_returns_trafo_and_trafo3w_when_ratio_and_steps_nonzero_tabular(net_pp):
    # make both existing trafos "Ratio" and with non-zero step percent/degree
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_ratio(net_pp)
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_excludes_non_ratio_even_if_steps_nonzero_tabular(net_pp):
    # steps valid but wrong type -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Tabular"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Tabular"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_excludes_when_tap_step_percent_is_zero_tabular(net_pp):
    # Ratio, but percent is zero -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 0.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 0.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_excludes_when_tap_step_degree_is_zero_tabular(net_pp):
    # Ratio, but degree is zero -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 0.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 0.0

    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_not_includes_when_tap_step_percent_is_nan_tabular(net_pp):
    # Ratio, percent NaN is allowed (per docstring)
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = np.nan
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = np.nan
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_not_includes_when_tap_step_degree_is_nan_tabular(net_pp):
    # Ratio, degree NaN is allowed (per docstring)
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = np.nan

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = np.nan

    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_empty_when_no_ratio_pst_present_tabular(net_pp):
    # default fixture uses Tabular -> should be empty
    res = get_pst_elements_ratio(net_pp)
    assert res == []


def test_excludes_added_ratio_trafo_when_steps_both_zero_or_invalid_tabular(net_pp):
    # Add a 2W trafo that is Ratio, but invalid (both zero) -> must not appear
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
    net_pp.trafo.loc[t2, "tap_step_percent"] = 0.0
    net_pp.trafo.loc[t2, "tap_step_degree"] = 0.0

    # Make the original ones valid so we can assert t2 is excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_ratio(net_pp)
    assert "e&&trafo&&1" not in res
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_returns_trafo_and_trafo3w_when_symmetrical_and_steps_nonzero_symmetrical(net_pp):
    # make both existing trafos "Symmetrical" and with non-zero step percent/degree
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)

    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_excludes_non_symmetrical_even_if_steps_nonzero_symmetrical(net_pp):
    # steps valid but wrong type -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_excludes_when_tap_step_percent_is_zero_symmetrical(net_pp):
    # Symmetrical, but percent is zero -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 0.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 0.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_excludes_when_tap_step_degree_is_zero_symmetrical(net_pp):
    # Symmetrical, but degree is zero -> excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 0.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 0.0

    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_not_includes_when_tap_step_percent_is_nan_symmetrical(net_pp):
    # Symmetrical, percent NaN is allowed (per docstring)
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = np.nan
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = np.nan
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_not_includes_when_tap_step_degree_is_nan_symmetrical(net_pp):
    # Symmetrical, degree NaN is allowed (per docstring)
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = np.nan

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = np.nan

    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_empty_when_no_symmetrical_pst_present_symmetrical(net_pp):
    # default fixture uses Tabular -> should be empty
    res = get_pst_elements_symmetrical(net_pp)
    assert res == []


def test_order_is_trafo_then_trafo3w_symmetrical(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)
    assert res[:1] == ["e&&trafo&&0"]
    assert res[1:] == ["e&&trafo3w&&0"]


def test_excludes_added_symmetrical_trafo_when_steps_both_zero_or_invalid_symmetrical(net_pp):
    # Add a 2W trafo that is Symmetrical, but invalid (both zero) -> must not appear
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
    net_pp.trafo.loc[t2, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[t2, "tap_step_percent"] = 0.0
    net_pp.trafo.loc[t2, "tap_step_degree"] = 0.0

    # Make the original ones valid so we can assert t2 is excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo.loc[0, "tap_step_degree"] = 2.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_percent"] = 1.0
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 2.0

    res = get_pst_elements_symmetrical(net_pp)
    assert "e&&trafo&&1" not in res
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_returns_trafo_and_trafo3w_when_ideal_and_tap_step_degree_nonzero_ideal(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_step_degree"] = 5.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 5.0

    res = get_pst_elements_ideal(net_pp)

    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]


def test_excludes_non_ideal_even_if_tap_step_degree_nonzero_ideal(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ratio"
    net_pp.trafo.loc[0, "tap_step_degree"] = 5.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Symmetrical"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 5.0

    res = get_pst_elements_ideal(net_pp)
    assert res == []


def test_excludes_when_tap_step_degree_is_zero_ideal(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_step_degree"] = 0.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 0.0

    res = get_pst_elements_ideal(net_pp)
    assert res == []


def test_not_includes_when_tap_step_degree_is_nan_ideal(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_step_degree"] = np.nan

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = np.nan

    res = get_pst_elements_ideal(net_pp)
    assert res == []


def test_empty_when_no_ideal_pst_present_ideal(net_pp):
    # default fixture uses Tabular -> should be empty
    res = get_pst_elements_ideal(net_pp)
    assert res == []


def test_order_is_trafo_then_trafo3w_ideal(net_pp):
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_step_degree"] = 5.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 5.0

    res = get_pst_elements_ideal(net_pp)
    assert res[:1] == ["e&&trafo&&0"]
    assert res[1:] == ["e&&trafo3w&&0"]


def test_excludes_added_ideal_trafo_when_tap_step_degree_zero_ideal(net_pp):
    # Add a 2W trafo that is Ideal, but invalid (degree zero) -> must not appear
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
    net_pp.trafo.loc[t2, "tap_step_degree"] = 0.0

    # Make the originals valid so we can assert the added one is excluded
    net_pp.trafo.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo.loc[0, "tap_step_degree"] = 5.0

    net_pp.trafo3w.loc[0, "tap_changer_type"] = "Ideal"
    net_pp.trafo3w.loc[0, "tap_step_degree"] = 5.0

    res = get_pst_elements_ideal(net_pp)
    assert "e&&trafo&&1" not in res
    assert res == ["e&&trafo&&0", "e&&trafo3w&&0"]
