# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import pandapower as pp
import pytest
from toop_engine_grid_helpers.pandapower.station_extraction import add_substation_column_to_bus


@pytest.fixture(scope="session")
def _pp_network_w_switches() -> pp.pandapowerNet:
    net = pp.networks.example_multivoltage()
    net.trafo["tap_dependent_impedance"] = False
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=True)
    return net


@pytest.fixture
def pp_network_w_switches(_pp_network_w_switches: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_pp_network_w_switches)


@pytest.fixture(scope="session")
def _pp_network_w_switches_open_coupler() -> pp.pandapowerNet:
    net = pp.networks.example_multivoltage()
    add_substation_column_to_bus(net, substation_col="substat", get_name_col="name", only_closed_switches=True)
    net.switch.loc[14, "closed"] = False
    return net


@pytest.fixture
def pp_network_w_switches_open_coupler(_pp_network_w_switches_open_coupler: pp.pandapowerNet) -> pp.pandapowerNet:
    return deepcopy(_pp_network_w_switches_open_coupler)
