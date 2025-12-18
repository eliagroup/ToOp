# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower
import pytest
from pypowsybl.loadflow import run_ac, run_dc
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
    create_complex_substation_layout_grid,
    powsybl_case30_with_psts,
    powsybl_case9241,
    powsybl_extended_case57,
)
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_pandapower_net_for_powsybl


def test_powsybl_case_30_with_psts_converges():
    net = powsybl_case30_with_psts()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_powsybl_extended_case57_converges():
    net = powsybl_extended_case57()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_powsybl_case9241_converges():
    net = powsybl_case9241()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_powsybl_case9241_fails_with_negative_trafos():
    pandapower_net = pandapower.networks.case9241pegase()
    with pytest.raises(ValueError) as e_info:
        load_pandapower_net_for_powsybl(pandapower_net, check_trafo_resistance=True)


def test_basic_node_breaker_network_powsybl_converges():
    net = basic_node_breaker_network_powsybl()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_create_complex_grid_battery_hvdc_svc_3w_trafo_converges():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_create_complex_substation_layout_grid_converges():
    net = create_complex_substation_layout_grid()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"
