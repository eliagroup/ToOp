# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import math
from pathlib import Path

import pandas as pd
import pypowsybl
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import (
    get_lines,
    get_network_as_pu,
    get_p_max,
    get_tie_lines,
    get_trafos,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_powsybl_helpers(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    assert net is not None

    assert len(get_lines(net)) == len(net.get_lines())
    assert len(get_tie_lines(net)) == len(net.get_tie_lines())
    assert len(get_trafos(net)) == len(net.get_2_windings_transformers())
    assert len(get_p_max(net)) == len(net.get_branches())


def get_op_lims_for_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    op_lims = lines_df.copy()
    op_lims.index.name = "element_id"
    op_lims["side"] = "TWO"
    op_lims["name"] = "permanent_limit"
    op_lims["value"] = 200
    op_lims["acceptable_duration"] = -1
    op_lims["type"] = "CURRENT"
    # Add second side
    op_lims_permanent = pd.concat([op_lims, op_lims.assign(side="ONE")])
    # Add N-1
    op_lims_n1 = pd.concat([op_lims, op_lims.assign(value=100, name="N-1", acceptable_duration=10)])
    # Add loadflow_based_n0 and loadflow_based_n1
    op_lims_lf_based = pd.concat(
        [
            op_lims,
            op_lims.assign(value=50, name="loadflow_based_n0", acceptable_duration=100),
            op_lims.assign(value=25, name="loadflow_based_n1", acceptable_duration=200),
        ]
    )

    all_op_lims = pd.concat([op_lims_permanent, op_lims_n1, op_lims_lf_based])
    return all_op_lims[["side", "name", "value", "acceptable_duration", "type"]]


def test_get_p_max(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    lines = net.get_lines(attributes=["voltage_level1_id"])
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])
    lines_with_voltage_level = pd.merge(lines, voltage_levels, left_on="voltage_level1_id", right_index=True)
    new_operational_limits = get_op_lims_for_lines(lines)

    # Add only permanent_limit
    net.create_operational_limits(new_operational_limits[new_operational_limits["name"] == "permanent_limit"])
    p_max_lines = get_p_max(net).loc[lines.index]
    expected_p_max = 200 * lines_with_voltage_level["nominal_v"] * 1e-3 * math.sqrt(3)
    assert all(p_max_lines["p_max_mw"] == expected_p_max)
    assert all(p_max_lines["p_max_mw_n_1"] == expected_p_max)

    # Add permanent_limit and N-1
    net.create_operational_limits(
        new_operational_limits[
            (new_operational_limits["name"] == "permanent_limit") | (new_operational_limits["name"] == "N-1")
        ]
    )
    p_max_lines = get_p_max(net).loc[lines.index]
    expected_p_max = 200 * lines_with_voltage_level["nominal_v"] * 1e-3 * math.sqrt(3)
    expected_p_max_n_1 = 100 * lines_with_voltage_level["nominal_v"] * 1e-3 * math.sqrt(3)
    assert all(p_max_lines["p_max_mw"] == expected_p_max)
    assert all(p_max_lines["p_max_mw_n_1"] == expected_p_max_n_1)

    # Add permanent_limit and N-1 and loadflow based limits
    net.create_operational_limits(new_operational_limits)
    p_max_lines = get_p_max(net).loc[lines.index]
    expected_p_max = 50 * lines_with_voltage_level["nominal_v"] * 1e-3 * math.sqrt(3)
    expected_p_max_n_1 = 25 * lines_with_voltage_level["nominal_v"] * 1e-3 * math.sqrt(3)
    assert all(p_max_lines["p_max_mw"] == expected_p_max)
    assert all(p_max_lines["p_max_mw_n_1"] == expected_p_max_n_1)


def test_get_tie_lines():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_tie_lines_and_areas()
    net_pu = get_network_as_pu(net)
    orig_tie_lines = net.get_tie_lines(all_attributes=True)
    dangling_lines = net.get_dangling_lines(all_attributes=True)

    tie_lines = get_tie_lines(net)
    dangling_pu = net_pu.get_dangling_lines()

    assert len(tie_lines) == len(orig_tie_lines)

    for id in orig_tie_lines.index:
        assert id in tie_lines.index, f"Expected tie line {id} to be in dangling lines"
        connected_dangling_pu = dangling_pu[dangling_pu["tie_line_id"] == id]
        dangling_1 = orig_tie_lines.loc[id, "dangling_line1_id"]
        dangling_2 = orig_tie_lines.loc[id, "dangling_line2_id"]
        assert tie_lines.loc[id, "x"] == connected_dangling_pu["x"].sum(), (
            f"Expected x for tie line {id} to match sum of per unit dangling lines"
        )
        assert tie_lines.loc[id, "r"] == connected_dangling_pu["r"].sum(), (
            f"Expected r for tie line {id} to match sum of per unit dangling lines"
        )
        bus_breaker_d1 = dangling_lines.loc[dangling_1, "bus_breaker_bus_id"]
        bus_breaker_d2 = dangling_lines.loc[dangling_2, "bus_breaker_bus_id"]
        pairing = orig_tie_lines.loc[id, "pairing_key"]
        element_name_d1 = (
            dangling_lines.loc[dangling_1, "elementName"]
            if "elementName" in dangling_lines.columns
            else dangling_lines.loc[dangling_1, "name"]
        )
        element_name_d2 = (
            dangling_lines.loc[dangling_2, "elementName"]
            if "elementName" in dangling_lines.columns
            else dangling_lines.loc[dangling_2, "name"]
        )
        expected_name = " ## ".join(
            [
                bus_breaker_d1,
                pairing,
                bus_breaker_d2,
                element_name_d1,
                element_name_d2,
            ]
        )
        assert tie_lines.loc[id, "name"] == expected_name


def test_get_tie_lines_empty():
    net = pypowsybl.network.create_ieee57()
    tie_lines = get_tie_lines(net)
    assert tie_lines.empty, "Expected no tie lines in IEEE 57 network"
