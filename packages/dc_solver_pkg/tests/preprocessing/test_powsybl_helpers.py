# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import (
    _build_pst_group_labels,
    _identify_pst_buckets,
    _is_linear_pst_step_table,
    add_missing_branch_model_columns,
    get_linear_pst,
    get_lines,
    get_network_as_pu,
    get_p_max,
    get_tie_lines,
    get_trafos,
)
from toop_engine_grid_helpers.powsybl.example_grids import grouped_pst_grid_example, parallel_pst_example
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_powsybl_helpers(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    assert net is not None

    assert len(get_lines(net)) == len(net.get_lines())
    assert len(get_tie_lines(net)) == len(net.get_tie_lines())
    assert len(get_trafos(net)) == len(net.get_2_windings_transformers())
    assert len(get_p_max(net)) == len(net.get_branches())


def test_add_missing_branch_model_columns() -> None:
    branches = pd.DataFrame(
        {"x": [0.1], "r": [0.2], "name": ["branch"]},
        index=pd.Index(["branch_id"], name="id"),
    )

    normalized_branches = add_missing_branch_model_columns(branches)

    assert normalized_branches.loc["branch_id", "name"] == "branch"
    assert pd.isna(normalized_branches.loc["branch_id", "rho"])
    assert pd.isna(normalized_branches.loc["branch_id", "alpha"])
    assert bool(normalized_branches.loc["branch_id", "has_pst_tap"]) is False
    assert bool(normalized_branches.loc["branch_id", "for_reward"]) is False
    assert bool(normalized_branches.loc["branch_id", "for_nminus1"]) is False
    assert normalized_branches.loc["branch_id", "overload_weight"] == 1.0
    assert pd.isna(normalized_branches.loc["branch_id", "p_max_mw"])
    assert pd.isna(normalized_branches.loc["branch_id", "p_max_mw_n_1"])
    assert bool(normalized_branches.loc["branch_id", "disconnectable"]) is False
    assert bool(normalized_branches.loc["branch_id", "pst_linear"]) is False
    # pst_group must be a recognized BranchModel column so it survives normalization (incl. the
    # empty-trafo path) and reaches _get_branches; non-PST branches default to -1.
    assert "pst_group" in normalized_branches.columns
    assert normalized_branches.loc["branch_id", "pst_group"] == -1
    assert normalized_branches.loc["branch_id", "n0_n1_max_diff_factor"] == -1.0


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


def test_get_p_max_uses_operational_limit_side_voltage(
    complex_grid_battery_hvdc_svc_3w_trafo_linear_1_1_data_folder: Path,
) -> None:
    net = pypowsybl.network.load(
        complex_grid_battery_hvdc_svc_3w_trafo_linear_1_1_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    )

    branches = net.get_branches(attributes=["voltage_level1_id", "voltage_level2_id"])
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])
    operational_limits = net.get_operational_limits().reset_index()[["element_id", "side", "name", "value"]]
    transformer_limits = operational_limits[
        operational_limits["element_id"].isin(["2W_MV_HV_1", "2W_MV_HV_2", "2W_MV_LV"])
        & (operational_limits["name"] == "permanent_limit")
    ].copy()

    transformer_limits["expected_voltage"] = np.where(
        transformer_limits["side"] == "ONE",
        voltage_levels.loc[branches.loc[transformer_limits["element_id"], "voltage_level1_id"].values, "nominal_v"].values,
        voltage_levels.loc[branches.loc[transformer_limits["element_id"], "voltage_level2_id"].values, "nominal_v"].values,
    )
    transformer_limits["expected_p_max"] = (
        transformer_limits["value"] * transformer_limits["expected_voltage"] * 1e-3 * math.sqrt(3)
    )

    p_max = get_p_max(net)

    for row in transformer_limits.itertuples(index=False):
        assert p_max.loc[row.element_id, "p_max_mw"] == pytest.approx(row.expected_p_max)
        assert p_max.loc[row.element_id, "p_max_mw_n_1"] == pytest.approx(row.expected_p_max)


def test_get_tie_lines():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_tie_lines_and_areas()
    net_pu = get_network_as_pu(net)
    orig_tie_lines = net.get_tie_lines(all_attributes=True)
    dangling_lines = net.get_boundary_lines(all_attributes=True)

    tie_lines = get_tie_lines(net)
    dangling_pu = net_pu.get_boundary_lines()

    assert len(tie_lines) == len(orig_tie_lines)

    for id in orig_tie_lines.index:
        assert id in tie_lines.index, f"Expected tie line {id} to be in dangling lines"
        connected_dangling_pu = dangling_pu[dangling_pu["tie_line_id"] == id]
        dangling_1 = orig_tie_lines.loc[id, "boundary_line1_id"]
        dangling_2 = orig_tie_lines.loc[id, "boundary_line2_id"]
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


def test_get_tie_lines_hybrid(ucte_file: Path) -> None:
    net = pypowsybl.network.load(ucte_file)
    net._source_format = "hybrid"
    tie_lines_orig = net.get_tie_lines(all_attributes=True)
    with pytest.raises(ValueError, match="No CGMES sub-network"):
        get_tie_lines(net)

    with mock.patch("toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids", return_value=[]):
        tie_lines = get_tie_lines(net)

    assert np.array_equal(tie_lines["name"].values, tie_lines_orig.index.values)

    dangline_lines = net.get_boundary_lines(all_attributes=True)
    with mock.patch(
        "toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids",
        return_value=dangline_lines.index.values.tolist(),
    ):
        tie_lines = get_tie_lines(net)

    assert not np.array_equal(tie_lines["name"].values, tie_lines_orig.index.values)


def test_get_trafos_hybrid(ucte_file: Path) -> None:
    net = pypowsybl.network.load(ucte_file)
    net._source_format = "hybrid"
    trafos_orig = net.get_2_windings_transformers(all_attributes=True)
    with pytest.raises(ValueError, match="No CGMES sub-network"):
        get_trafos(net)

    with mock.patch("toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids", return_value=[]):
        trafos = get_trafos(net)

    trafos_orig["ucte_name"] = trafos_orig.index.astype(str) + ": " + trafos_orig["elementName"]
    assert np.array_equal(trafos["name"].values, trafos_orig["ucte_name"].values)

    with mock.patch(
        "toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids",
        return_value=trafos_orig.index.values.tolist(),
    ):
        trafos = get_trafos(net)

    trafos_orig["cgmes_name"] = trafos_orig["name"]
    assert np.array_equal(trafos["name"].values, trafos_orig["cgmes_name"].values)


def test_get_trafos_groups_parallel_psts() -> None:
    net = parallel_pst_example()

    trafos = get_trafos(net)
    label_by_id = dict(zip(trafos.index, trafos["pst_group"].to_numpy(dtype=int), strict=True))

    assert label_by_id["PST1"] == label_by_id["PST2"]
    assert label_by_id["PST3"] >= 0
    assert label_by_id["PST3"] != label_by_id["PST1"]


@pytest.mark.parametrize(
    ("linear_pst", "split_pst_station", "expected_group_count", "expected_grouped_pairs"),
    [
        (
            [True, True, True, True],
            False,
            1,
            [("PST_1_group_1", "PST_2_group_1"), ("PST_3_group_2", "PST_4_group_2")],
        ),
        (
            [True, True, True, True],
            True,
            2,
            [("PST_1_group_1", "PST_3_group_2"), ("PST_2_group_1", "PST_4_group_2")],
        ),
        (
            [True, False, True, False],
            False,
            2,
            [("PST_1_group_1", "PST_3_group_2"), ("PST_2_group_1", "PST_4_group_2")],
        ),
    ],
)
def test_get_trafos_grouped_pst_grid_assigns_expected_pst_groups(
    linear_pst: list[bool],
    split_pst_station: bool,
    expected_group_count: int,
    expected_grouped_pairs: list[tuple[str, str]],
) -> None:
    net = grouped_pst_grid_example(linear_pst=linear_pst)
    if split_pst_station:
        net.open_switch("VL2_BREAKER#0")

    trafos = get_trafos(net)
    label_by_id = dict(zip(trafos.index, trafos["pst_group"].to_numpy(dtype=int), strict=True))

    unique_labels = set(label_by_id.values())
    assert len(unique_labels) == expected_group_count
    if expected_group_count == 2:
        assert unique_labels == {0, 1}
    else:
        assert unique_labels == {0}
    for first_pst_id, second_pst_id in expected_grouped_pairs:
        assert label_by_id[first_pst_id] == label_by_id[second_pst_id]


def _get_raw_trafos_with_pst_metadata(net: pypowsybl.network.Network) -> pd.DataFrame:
    """Prepare the raw transformer dataframe expected by the PST grouping helpers."""
    trafos = net.get_2_windings_transformers(all_attributes=True).copy()
    linear_psts = get_linear_pst(net, mode="dc")
    trafos["pst_linear"] = False
    trafos["has_pst_tap"] = False
    trafos.loc[linear_psts.index, "pst_linear"] = linear_psts.values
    trafos.loc[linear_psts.index, "has_pst_tap"] = True
    return trafos


def test_identify_pst_buckets_groups_parallel_candidates() -> None:
    net = parallel_pst_example()
    trafos = _get_raw_trafos_with_pst_metadata(net)

    step_tables, buckets = _identify_pst_buckets(trafos=trafos, net=net)

    assert set(step_tables) == {"PST1", "PST2", "PST3"}
    bucket_members = {frozenset(trafos.index[positions]) for positions in buckets.values()}

    assert frozenset({"PST1", "PST2"}) in bucket_members
    assert frozenset({"PST3"}) in bucket_members


def test_build_pst_group_labels_marks_non_psts_and_groups_parallel_psts() -> None:
    net = parallel_pst_example()
    trafos = _get_raw_trafos_with_pst_metadata(net)
    trafos.loc["NON_PST"] = trafos.loc["PST1"]
    trafos.loc["NON_PST", "has_pst_tap"] = False
    trafos.loc["NON_PST", "pst_linear"] = False

    group_labels = _build_pst_group_labels(trafos=trafos, net=net)
    label_by_id = dict(zip(trafos.index, group_labels, strict=True))

    assert label_by_id["PST1"] == label_by_id["PST2"]
    assert label_by_id["PST3"] != label_by_id["PST1"]
    assert label_by_id["PST3"] >= 0
    assert label_by_id["NON_PST"] == -1


def test_is_linear_pst_step_table_detects_variable_impedance() -> None:
    linear_step_table = pd.DataFrame(
        {
            "rho": [1.0, 1.0, 1.0],
            "x": [1.0, 1.0, 1.0],
            "r": [0.1, 0.1, 0.1],
            "g": [0.0, 0.0, 0.0],
            "b": [0.0, 0.0, 0.0],
            "alpha": [-0.1, 0.0, 0.1],
        }
    )
    non_linear_step_table = linear_step_table.copy()
    non_linear_step_table.loc[2, "x"] = 1.2

    assert _is_linear_pst_step_table(linear_step_table) is True
    assert _is_linear_pst_step_table(non_linear_step_table) is False


def test_get_lines_hybrid(ucte_file: Path) -> None:
    net = pypowsybl.network.load(ucte_file)
    net._source_format = "hybrid"
    lines_orig = net.get_lines(all_attributes=True)
    with pytest.raises(ValueError, match="No CGMES sub-network"):
        get_lines(net)

    with mock.patch("toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids", return_value=[]):
        lines = get_lines(net)

    lines_orig["ucte_name"] = lines_orig.index.astype(str) + ": " + lines_orig["elementName"]
    assert np.array_equal(lines["name"].values, lines_orig["ucte_name"].values)

    with mock.patch(
        "toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers.get_cgmes_ids",
        return_value=lines_orig.index.values.tolist(),
    ):
        lines = get_lines(net)

    lines_orig["cgmes_name"] = lines_orig["name"]
    assert np.array_equal(lines["name"].values, lines_orig["cgmes_name"].values)


def test_get_cgmes_ids_hybrid(ucte_file: Path) -> None:
    net = pypowsybl.network.load(ucte_file)
    net._source_format = "hybrid"

    from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import get_cgmes_ids

    with pytest.raises(ValueError, match="No CGMES sub-network"):
        get_cgmes_ids(net)

    cgmes_net = pypowsybl.network.load(ucte_file)
    cgmes_net._source_format = "CGMES"

    net.get_sub_networks = mock.MagicMock(return_value=pd.Series(index=["some_sub_network_id"]))
    net.get_sub_network = mock.MagicMock(return_value=cgmes_net)
    # Mock the get_cgmes_ids function to return the line IDs
    cgmes_ids = get_cgmes_ids(net)

    assert np.array_equal(cgmes_ids, cgmes_net.get_identifiables().index.values)
