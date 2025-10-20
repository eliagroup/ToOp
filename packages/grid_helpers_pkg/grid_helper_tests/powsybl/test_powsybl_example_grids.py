from pypowsybl.loadflow import run_ac, run_dc
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    powsybl_case30_with_psts,
    powsybl_extended_case57,
    powsybl_texas,
)


def test_powsybl_case_30_with_psts_converges():
    net = powsybl_case30_with_psts()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"


def test_powsybl_texas_converges():
    net = powsybl_texas()
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


def test_basic_node_breaker_network_powsybl_converges():
    net = basic_node_breaker_network_powsybl()
    result_dc = run_dc(net)
    assert result_dc[0].status_text == "Converged"
    result_ac = run_ac(net)
    assert result_ac[0].status_text == "Converged"
