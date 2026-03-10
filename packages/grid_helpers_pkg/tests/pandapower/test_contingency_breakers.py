# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower as pp
import pytest
from toop_engine_grid_helpers.pandapower.contingency_breakers import get_observed_circuit_breakers


@pytest.fixture
def network_with_transformer3w_and_breakers():
    grid = pp.create_empty_network(name="three_winding_transformer_with_breakers")

    # Buses
    hv_bus_main = pp.create_bus(grid, vn_kv=110, name="HV Main")
    mv_bus_main = pp.create_bus(grid, vn_kv=20, name="MV Main")
    lv_bus_main = pp.create_bus(grid, vn_kv=10, name="LV Main")

    hv_bus_aux = pp.create_bus(grid, vn_kv=110, name="HV Aux")
    mv_bus_aux = pp.create_bus(grid, vn_kv=20, name="MV Aux")
    mv_bus_branch = pp.create_bus(grid, vn_kv=20, name="MV Branch")
    lv_bus_aux = pp.create_bus(grid, vn_kv=10, name="LV Aux")
    lv_bus_branch = pp.create_bus(grid, vn_kv=10, name="LV Branch")
    lv_bus_terminal = pp.create_bus(grid, vn_kv=10, name="LV Terminal")

    # Three-winding transformer
    pp.create_transformer3w(
        grid,
        hv_bus=hv_bus_main,
        mv_bus=mv_bus_main,
        lv_bus=lv_bus_main,
        std_type="63/25/38 MVA 110/20/10 kV",
        name="T1_3W",
    )

    # Circuit breakers
    pp.create_switch(grid, bus=hv_bus_main, element=hv_bus_aux, et="b", closed=True, type="CB", name="CB_HV")
    pp.create_switch(grid, bus=mv_bus_main, element=mv_bus_aux, et="b", closed=True, type="CB", name="CB_MV")
    pp.create_switch(grid, bus=lv_bus_main, element=lv_bus_aux, et="b", closed=True, type="CB", name="CB_LV")

    # Additional LV topology
    pp.create_line_from_parameters(
        grid,
        lv_bus_aux,
        lv_bus_branch,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line_lv",
    )
    pp.create_switch(grid, bus=lv_bus_branch, element=lv_bus_terminal, et="b", closed=True, type="CB", name="CB_LV_BRANCH")

    # Non-circuit-breaker switch
    pp.create_switch(grid, bus=mv_bus_aux, element=mv_bus_branch, et="b", closed=True, type="DS", name="DS_MV_BRANCH")

    # Loads
    pp.create_load(grid, bus=mv_bus_aux, p_mw=8.0, q_mvar=2.0, name="MV Load")
    pp.create_load(grid, bus=lv_bus_aux, p_mw=3.0, q_mvar=0.8, name="LV Load")

    return grid


def test_returns_all_observed_circuit_breakers_without_filters(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers

    observed_breakers = get_observed_circuit_breakers(grid)

    assert observed_breakers == {0, 1, 2, 3}


def test_excludes_breakers_below_voltage_threshold_when_threshold_is_100(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers
    transformer_hv_bus = grid.trafo3w.hv_bus.iloc[0]
    grid.bus.loc[transformer_hv_bus, "vn_kv"] = 220

    observed_breakers = get_observed_circuit_breakers(grid, exclude_lower_vn_kv=100.0)

    assert observed_breakers == {0, 1, 2}


def test_excludes_all_breakers_when_lower_voltage_threshold_is_110(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers

    observed_breakers = get_observed_circuit_breakers(grid, exclude_lower_vn_kv=110.0)

    assert observed_breakers == set()


def test_higher_voltage_filter_does_not_remove_expected_breakers(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers

    observed_breakers = get_observed_circuit_breakers(grid, exclude_higher_vn_kv=100.0)

    assert observed_breakers == {0, 1, 2, 3}


def test_returns_breakers_for_matching_region(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers
    grid.bus["GeographicalRegion_name"] = "North"

    observed_breakers = get_observed_circuit_breakers(grid, region="North")

    assert observed_breakers == {0, 1, 2, 3}


def test_returns_no_breakers_for_non_matching_region(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers
    grid.bus["GeographicalRegion_name"] = "South"

    observed_breakers = get_observed_circuit_breakers(grid, region="North")

    assert observed_breakers == set()


def test_applies_voltage_and_region_filters_together(network_with_transformer3w_and_breakers):
    grid = network_with_transformer3w_and_breakers
    grid.bus["GeographicalRegion_name"] = "North"

    observed_breakers = get_observed_circuit_breakers(
        grid,
        exclude_lower_vn_kv=100.0,
        exclude_higher_vn_kv=200.0,
        region="North",
    )

    assert observed_breakers == {0, 1, 2}
