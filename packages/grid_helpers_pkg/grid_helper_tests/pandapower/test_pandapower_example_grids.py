import pandapower
import pytest
from toop_engine_grid_helpers.pandapower.example_grids import (
    example_multivoltage_cross_coupler,
    pandapower_case30_with_psts,
    pandapower_case30_with_psts_and_weak_branches,
    pandapower_extended_case57,
    pandapower_extended_oberrhein,
    pandapower_non_converging_case57,
    pandapower_texas,
)


def test_pandapower_case30_with_psts_converges():
    net = pandapower_case30_with_psts()
    pandapower.rundcpp(net)
    pandapower.runpp(net)


def test_pandapower_case30_with_psts_and_weak_branches():
    net = pandapower_case30_with_psts_and_weak_branches()
    pandapower.rundcpp(net)
    pandapower.runpp(net)


def test_pandapower_texas():
    net = pandapower_texas()
    pandapower.rundcpp(net)
    pandapower.runpp(net)


def test_pandapower_extended_oberrhein():
    net = pandapower_extended_oberrhein()
    pandapower.rundcpp(net)
    pandapower.runpp(net)


def test_pandapower_non_converging_case57():
    net = pandapower_non_converging_case57()
    pandapower.rundcpp(net)
    with pytest.raises(Exception):
        pandapower.runpp(net)


def test_pandapower_extended_case57():
    net = pandapower_extended_case57()
    pandapower.rundcpp(net)
    pandapower.runpp(net)


def test_example_multivoltage_cross_coupler():
    net = example_multivoltage_cross_coupler()
    pandapower.rundcpp(net)
    pandapower.runpp(net)
