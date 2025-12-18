# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

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
