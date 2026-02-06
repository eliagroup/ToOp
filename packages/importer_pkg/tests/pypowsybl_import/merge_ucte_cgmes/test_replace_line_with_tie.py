# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import.merge_ucte_cgmes.replace_line_with_tie import (
    DanglingGeneratorSchema,
    DanglingLineCreationSchema,
    check_dangling_node,
    get_dangling_creation_schema,
    get_dangling_generator_creation_schema,
    get_dangling_lines_creation_schema,
    get_dangling_voltage_levels,
    replace_voltage_level_with_tie_line,
    set_dangling_generator_ids,
)


def test_replace_voltage_level_with_tie_line(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    lf = pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    assert lf[0].status_text == "Converged", "Test grid did not converge"
    br = network.get_branches()
    p1_test_grid_ac = br[~br["p1"].isna()]["p1"]
    p2_test_grid_ac = br[~br["p2"].isna()]["p2"]
    # get p values from test grid for DC
    pypowsybl.loadflow.run_dc(network, DISTRIBUTED_SLACK)
    br = network.get_branches()
    p1_test_grid_dc = br[~br["p1"].isna()]["p1"]
    p2_test_grid_dc = br[~br["p2"].isna()]["p2"]

    # convert German dangling node to tie line
    dangling_voltage_level = "DXSU1_1"
    replace_voltage_level_with_tie_line(network=network, voltage_level_id=dangling_voltage_level)
    lf = pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    assert lf[0].status_text == "Converged", "Tie line modificated grid did not converge"
    br = network.get_branches()
    p1_tie_grid_ac = br[~br["p1"].isna()]["p1"]
    p2_tie_grid_ac = br[~br["p2"].isna()]["p2"]
    dc_lf = pypowsybl.loadflow.run_dc(network, DISTRIBUTED_SLACK)
    assert dc_lf[0].status_text == "Converged", "Test grid did not converge"

    br = network.get_branches()
    p1_tie_grid_dc = br[~br["p1"].isna()]["p1"]
    p2_tie_grid_dc = br[~br["p2"].isna()]["p2"]

    tol = 1e-9
    # test in DC -> network should be the same
    for index, element in p1_test_grid_dc.items():
        if index in p1_tie_grid_dc.index:
            assert abs(element - p1_tie_grid_dc[index]) < tol, (
                f"Mismatch in power flow for index {index}: {element} != {p1_tie_grid_dc[index]}"
            )
    # replacement names
    tie_name = "D8SU1_12 DXSU1_12 2 + DXSU1_12 D7SU2_11 1"
    dangling1 = "D8SU1_12 DXSU1_12 2"
    dangling2 = "DXSU1_12 D7SU2_11 1"

    assert tie_name in p1_tie_grid_dc.index
    assert dangling1 in p1_test_grid_dc.index
    assert dangling2 in p1_test_grid_dc.index
    assert abs(p1_test_grid_dc[dangling1] - p1_tie_grid_dc[tie_name]) < tol, (
        f"Mismatch in power flow for index {dangling1}: {p1_test_grid_dc[dangling1]} != {p1_tie_grid_dc[tie_name]}, abs: {abs(p1_test_grid_dc[dangling1] - p1_tie_grid_dc[tie_name])}"
    )
    assert abs(p2_test_grid_dc[dangling2] - p2_tie_grid_dc[tie_name]) < tol, (
        f"Mismatch in power flow for index {dangling2}: {p1_test_grid_dc[dangling2]} != {p2_tie_grid_dc[tie_name]}, abs: {abs(p2_test_grid_dc[dangling2] - p2_tie_grid_dc[tie_name])}"
    )

    # test in AC -> network should be the same
    tol = 1e-6
    parallel_line_name = "D8SU1_12 D7SU2_11 1"
    for index, element in p1_test_grid_ac.items():
        if index in p1_tie_grid_ac.index:
            if parallel_line_name == index:
                # this is the parallel line, this has a different value in AC powsybl
                continue
            assert abs(element - p1_tie_grid_ac[index]) < tol, (
                f"Mismatch in power flow for index {index}: {element} != {p1_tie_grid_ac[index]}"
            )

    tol = 1e-5
    p1_parallel_lines_before = p1_test_grid_ac[parallel_line_name] + p1_test_grid_ac[dangling1]
    p2_parallel_lines_before = p2_test_grid_ac[parallel_line_name] + p2_test_grid_ac[dangling2]

    p1_parallel_lines_after = p1_tie_grid_ac[parallel_line_name] + p1_tie_grid_ac[tie_name]
    p2_parallel_lines_after = p2_tie_grid_ac[parallel_line_name] + p2_tie_grid_ac[tie_name]
    assert abs(p1_parallel_lines_before - p1_parallel_lines_after) < tol, (
        f"Mismatch in power flow for index {parallel_line_name}: {p1_parallel_lines_before} != {p1_parallel_lines_after}, abs: {abs(p1_parallel_lines_before - p1_parallel_lines_after)}"
    )
    assert abs(p2_parallel_lines_before - p2_parallel_lines_after) < tol, (
        f"Mismatch in power flow for index {parallel_line_name}: {p2_parallel_lines_before} != {p2_parallel_lines_after}, abs: {abs(p2_parallel_lines_before - p2_parallel_lines_after)}"
    )


def test_check_dangling_node_with_valid_topology(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    dangling_voltage_level = "DXSU1_1"
    bus_breaker_topo = network.get_bus_breaker_topology(dangling_voltage_level)
    check_dangling_node(bus_breaker_topo)

    station_voltage_level = "D8SU1_1"
    bus_breaker_topo = network.get_bus_breaker_topology(station_voltage_level)
    with pytest.raises(ValueError, match="The dangling Node contains switches, wrong voltage_level?"):
        check_dangling_node(bus_breaker_topo)

    station_voltage_level = "D8SU1_2"
    # contains TWO_WINDINGS_TRANSFORMER and DANGLING_LINE
    bus_breaker_topo = network.get_bus_breaker_topology(station_voltage_level)
    with pytest.raises(ValueError, match="TWO_WINDINGS_TRANSFORMER"):
        check_dangling_node(bus_breaker_topo)

    dangling_voltage_level = "DXSU1_1"
    network.create_lines(
        id="one_line_to_much",
        voltage_level1_id="DXSU1_1",
        bus1_id="DXSU1_12",
        voltage_level2_id="D8SU1_1",
        bus2_id="D8SU1_11",
        b1=0,
        b2=0,
        g1=0,
        g2=0,
        r=0.5,
        x=10,
    )
    bus_breaker_topo = network.get_bus_breaker_topology(dangling_voltage_level)
    with pytest.raises(ValueError, match="The dangling Node contains more than 2 lines connected to one bus"):
        check_dangling_node(bus_breaker_topo)


def test_get_dangling_creation_schema(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    dangling_voltage_level = "DXSU1_1"
    dangling_line_creation_df, dangling_generator_df = get_dangling_creation_schema(
        network=network, dangling_voltage_level=dangling_voltage_level
    )
    assert isinstance(dangling_line_creation_df, pd.DataFrame)
    assert isinstance(dangling_generator_df, pd.DataFrame)
    DanglingGeneratorSchema.validate(dangling_generator_df)
    DanglingLineCreationSchema.validate(dangling_line_creation_df)
    assert ["D8SU1_12 DXSU1_12 2", "DXSU1_12 D7SU2_11 1"] == dangling_line_creation_df.index.tolist()
    assert dangling_generator_df.empty

    # Note: not a true Dangling node, but has a generator to test
    dangling_voltage_level = "D7SU2_1"
    dangling_line_creation_df, dangling_generator_df = get_dangling_creation_schema(
        network=network, dangling_voltage_level=dangling_voltage_level
    )
    # generator ids must match dangling line creation ids
    assert ["D8SU1_12 D7SU2_11 1", "DXSU1_12 D7SU2_11 1"] == dangling_generator_df.index.tolist()
    assert ["D8SU1_12 D7SU2_11 1", "DXSU1_12 D7SU2_11 1"] == dangling_line_creation_df.index.tolist()


def test_set_dangling_generation_ids(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    # Note: not a true Dangling node, but has a generator to test
    dangling_voltage_level = "D7SU2_1"
    bus_breaker_topo = network.get_bus_breaker_topology(dangling_voltage_level)
    elements = bus_breaker_topo.elements
    lines = elements[elements["type"] == "LINE"]
    generators = elements[elements["type"] == "GENERATOR"]
    name_col = "elementName"
    dangling_line_creation_df = get_dangling_lines_creation_schema(
        network=network, bus_breaker_topo_lines=lines, dangling_voltage_level=dangling_voltage_level, name_col=name_col
    )
    dangling_generator_df = get_dangling_generator_creation_schema(network=network, generators=generators)
    assert dangling_generator_df.index.to_list() == ["D7SU2_11_generator"]
    set_dangling_generator_ids(
        dangling_line_creation_df=dangling_line_creation_df, dangling_generator_df=dangling_generator_df
    )
    assert ["D8SU1_12 D7SU2_11 1", "DXSU1_12 D7SU2_11 1"] == dangling_generator_df.index.tolist()

    # now the ids are set and will throw an error if tried to set again
    with pytest.raises(ValueError):
        set_dangling_generator_ids(
            dangling_line_creation_df=dangling_line_creation_df, dangling_generator_df=dangling_generator_df
        )

    # test with empty dangling_generator_df
    generators = elements[elements["type"] == "NO_GENERATOR"]
    assert generators.empty
    name_col = "elementName"
    dangling_line_creation_df = get_dangling_lines_creation_schema(
        network=network, bus_breaker_topo_lines=lines, dangling_voltage_level=dangling_voltage_level, name_col=name_col
    )
    dangling_generator_df = get_dangling_generator_creation_schema(network=network, generators=generators)
    assert dangling_generator_df.empty


def test_get_dangling_voltage_levels(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    area_codes = ["D8"]
    lines = network.get_lines()
    external_border_mask = lines["voltage_level1_id"].str.startswith("D8") & lines["voltage_level2_id"].str.startswith("DX")
    dangling_voltage_level = get_dangling_voltage_levels(
        network=network, area_codes=area_codes, external_border_mask=external_border_mask
    )
    assert ["DXSU1_1"] == dangling_voltage_level
    # test with empty area_codes
    area_codes = []
    dangling_voltage_level = get_dangling_voltage_levels(
        network=network, area_codes=area_codes, external_border_mask=external_border_mask
    )
    assert dangling_voltage_level == []

    # test with empty external_border_mask
    area_codes = ["D8"]
    external_border_mask = np.array([False] * len(lines))
    dangling_voltage_level = get_dangling_voltage_levels(
        network=network, area_codes=area_codes, external_border_mask=external_border_mask
    )
    assert dangling_voltage_level == []

    # fail with line if other region
    area_codes = ["D8"]
    external_border_mask = lines["voltage_level1_id"].str.startswith("DX") | lines["voltage_level2_id"].str.startswith("DX")
    with pytest.raises(ValueError):
        dangling_voltage_level = get_dangling_voltage_levels(
            network=network, area_codes=area_codes, external_border_mask=external_border_mask
        )
