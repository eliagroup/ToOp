from dataclasses import replace

import numpy as np
import pandas as pd
import pypowsybl
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_importer.pypowsybl_import.loadflow_based_current_limits import (
    create_current_limits_df,
    create_new_border_limits,
    get_all_border_line_limits,
    get_all_dso_trafo_limits,
    get_branches_including_limits_and_dangling_lines,
    get_loadflow_based_line_limits,
    get_loadflow_based_tie_line_limits,
    get_loadflow_based_trafo_limits,
    get_new_limits_for_branch,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    LimitAdjustmentParameters,
)


def test_create_current_limits_df():
    new_limit_series = pd.Series(data=[100, 200, 300], index=["line_1", "line_2", "line_3"])
    group_names = pd.Series(data=["group_1", "group_2", "group_3"], index=["line_1", "line_2", "line_3"])
    new_limit_df = create_current_limits_df(
        new_limit_series,
        element_type="LINE",
        side="ONE",
        limit_name="new_limit",
        acceptable_duration=100,
        group_names=group_names,
    )
    assert len(new_limit_df) == len(new_limit_series)

    assert np.array_equal(new_limit_df.index.get_level_values("element_id"), new_limit_series.index)
    assert np.all(new_limit_df.element_type.values == "LINE")
    assert np.all(new_limit_df.index.get_level_values("side").values == "ONE")
    assert np.all(new_limit_df.name.values == "new_limit")
    assert np.all(new_limit_df.index.get_level_values("acceptable_duration").values == 100)
    assert np.array_equal(new_limit_series.values, new_limit_df.value.values)
    assert np.array_equal(group_names.values, new_limit_df.index.get_level_values("group_name").values)
    assert list(new_limit_df.index.names) == ["element_id", "side", "type", "acceptable_duration", "group_name"]


def test_get_branches_including_limits_and_dangling_lines():
    shared_index = ["line1", "tie_line1"]
    branch_df = pd.DataFrame(
        index=shared_index,
        data={"type": ["LINE", "TIE_LINE"], "i1": [1.0, 2.0], "i2": [3.0, 3.0]},
    )
    operational_limits_df = pd.DataFrame(
        index=shared_index + shared_index,
        data={
            "name": ["permanent_limit", "permanent_limit", "N-1", "N-1"],
            "side": ["ONE", "TWO", "ONE", "TWO"],
            "value": [1.0, 3.0, 4.0, 4.0],
            "group_name": ["group1", "group2", "group3", "group4"],
        },
    )
    operational_limits_df.index.name = "element_id"
    operational_limits_df.set_index(["side", "group_name"], append=True, inplace=True)

    tie_line_df = pd.DataFrame(
        index=shared_index[1:],
        data={"dangling_line1_id": ["d1"], "dangling_line2_id": ["d2"]},
    )
    updated_branch_df = get_branches_including_limits_and_dangling_lines(branch_df, operational_limits_df, tie_line_df)

    assert np.array_equal(updated_branch_df["i1"], [1.0, 2.0])
    assert np.array_equal(updated_branch_df["i2"], [3.0, 3.0])

    assert np.array_equal(updated_branch_df["n0_i1_max"], [1.0, np.nan], equal_nan=True)
    assert np.array_equal(updated_branch_df["n0_i2_max"], [np.nan, 3.0], equal_nan=True)

    assert np.array_equal(updated_branch_df["n1_i1_max"], [4.0, np.nan], equal_nan=True)
    assert np.array_equal(updated_branch_df["n1_i2_max"], [np.nan, 4.0], equal_nan=True)

    assert updated_branch_df.loc["line1", "dangling_line1_id"] is np.nan
    assert updated_branch_df.loc["tie_line1", "dangling_line1_id"] == "d1"

    assert updated_branch_df.loc["line1", "dangling_line2_id"] is np.nan
    assert updated_branch_df.loc["tie_line1", "dangling_line2_id"] == "d2"


def test_get_new_limit_for_branch():
    current_100 = np.array([100.0, 100.0, 100.0])
    current_50 = np.array([50.0, 50.0, 50.0])
    branch_df = pd.DataFrame(
        index=["line1", "line2", "line3"],
        data={
            "update_i": current_50,
            "old_limit_n0": current_100,
            "old_limit_n1": current_100,
        },
    )
    limit_parameters = LimitAdjustmentParameters(n_0_factor=1.5, n_1_factor=2, n_0_min_increase=0.1, n_1_min_increase=0.1)
    factor, min_increase = limit_parameters.get_parameters_for_case("n0")
    new_limit = get_new_limits_for_branch(branch_df["update_i"], branch_df["old_limit_n0"], factor, min_increase)
    assert np.array_equal(new_limit.values, current_50 * 1.5)
    factor, min_increase = limit_parameters.get_parameters_for_case("n1")
    new_limit = get_new_limits_for_branch(branch_df["update_i"], branch_df["old_limit_n1"], factor, min_increase)
    assert np.array_equal(new_limit.values, current_50 * 2)

    assert np.array_equal(new_limit.index, branch_df.index)
    # Test new limit > old limit
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=2.1,
    )
    factor, min_increase = limit_parameters.get_parameters_for_case("n0")

    new_limit = get_new_limits_for_branch(branch_df["update_i"], branch_df["old_limit_n0"], factor, min_increase)
    assert np.array_equal(new_limit.values, current_100)

    # Test new_limit increase really small
    small_currents_branch_df = branch_df.assign(update_i=[1.0, 1.0, 1.0])
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=1.1,
        n_0_min_increase=0.1,  # At least 10% of 100 increase -> 11.
    )
    factor, min_increase = limit_parameters.get_parameters_for_case("n0")

    new_limit = get_new_limits_for_branch(
        small_currents_branch_df["update_i"], small_currents_branch_df["old_limit_n0"], factor, min_increase
    )
    assert np.array_equal(new_limit.values, [11.0, 11.0, 11.0])

    high_currents_branch_df = branch_df.assign(update_i=[95.0, 95.0, 1.0])
    # Test min increase over old limit
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=1.1,
        n_0_min_increase=0.1,  # At least 10% of 100 increase -> 11. or 105.< too high
    )
    factor, min_increase = limit_parameters.get_parameters_for_case("n0")
    new_limit = get_new_limits_for_branch(
        high_currents_branch_df["update_i"], high_currents_branch_df["old_limit_n0"], factor, min_increase
    )
    assert np.array_equal(new_limit.values, [100.0, 100.0, 11.0])


def test_get_loadflow_based_line_limits():
    lines_df = pd.DataFrame(
        index=["line"],
        data={
            "i1": [10.0],
            "i2": [10.0],
            "n0_i1_max": [100.0],
            "n0_i2_max": [100.0],
            "n1_i1_max": [100.0],
            "n1_i2_max": [100.0],
            "n0_group_name_1": ["group_1"],
            "n0_group_name_2": ["group_2"],
            "n1_group_name_1": ["group_3"],
            "n1_group_name_2": ["group_4"],
        },
    )
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=1.1,
        n_0_min_increase=0.1,  # New limit should be min(10*1.1, 10+0.1*100) = 20
    )
    line_limits = get_loadflow_based_line_limits(lines_df, limit_parameters, "n0")

    assert type(line_limits) == list
    assert type(line_limits[0]) == pd.DataFrame
    assert len(line_limits) == 2
    assert line_limits[0].index.get_level_values("side").values == np.array(["ONE"])
    assert line_limits[0].element_type.values == np.array(["LINE"])
    assert line_limits[0]["name"].values == np.array(["loadflow_based_n0"])
    assert line_limits[0].index.get_level_values("group_name").values == np.array(["group_1"])
    assert line_limits[1].index.get_level_values("side").values == np.array(["TWO"])
    assert line_limits[1].element_type.values == np.array(["LINE"])
    assert line_limits[1]["name"].values == np.array(["loadflow_based_n0"])
    assert line_limits[1].index.get_level_values("group_name").values == np.array(["group_2"])

    line_limits = get_loadflow_based_line_limits(lines_df, limit_parameters, "n1")

    assert type(line_limits) == list
    assert type(line_limits[0]) == pd.DataFrame
    assert len(line_limits) == 2
    assert line_limits[0].index.get_level_values("side").values == np.array(["ONE"])

    assert line_limits[0]["name"].values == np.array(["loadflow_based_n1"])
    assert line_limits[1].index.get_level_values("side").values == np.array(["TWO"])
    assert line_limits[1]["name"].values == np.array(["loadflow_based_n1"])
    assert line_limits[0].index.get_level_values("group_name").values == np.array(["group_3"])
    assert line_limits[1].index.get_level_values("group_name").values == np.array(["group_4"])

    empty_limits = get_loadflow_based_line_limits(lines_df[:0], limit_parameters, "n0")
    assert empty_limits == []


def test_get_loadflow_based_tie_line_limits():
    tie_lines_df = pd.DataFrame(
        index=["tie_line"],
        data={
            "i1": [10.0],
            "i2": [10.0],
            "n0_i1_max": [100.0],
            "n0_i2_max": [100.0],
            "n1_i1_max": [100.0],
            "n1_i2_max": [100.0],
            "dangling_line1_id": ["d1"],
            "dangling_line2_id": ["d2"],
            "n0_group_name_1": ["group_1"],
            "n0_group_name_2": ["group_2"],
            "n1_group_name_1": ["group_3"],
            "n1_group_name_2": ["group_4"],
        },
    )
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=1.1,
        n_0_min_increase=0.1,  # New limit should be min(10*1.1, 10+0.1*100) = 20
    )
    tie_line_limits = get_loadflow_based_tie_line_limits(tie_lines_df, limit_parameters, "n0")

    assert type(tie_line_limits) == list
    assert type(tie_line_limits[0]) == pd.DataFrame
    assert len(tie_line_limits) == 2
    assert tie_line_limits[0].element_type.values == np.array(["DANGLING_LINE"])
    assert tie_line_limits[0].index.get_level_values("side").values == np.array(["NONE"])
    assert tie_line_limits[0]["name"].values == np.array(["loadflow_based_n0"])
    assert tie_line_limits[1].index.get_level_values("side").values == np.array(["NONE"])
    assert tie_line_limits[1].element_type.values == np.array(["DANGLING_LINE"])
    assert tie_line_limits[1]["name"].values == np.array(["loadflow_based_n0"])
    assert tie_line_limits[0].index.get_level_values("group_name").values == np.array(["group_1"])
    assert tie_line_limits[1].index.get_level_values("group_name").values == np.array(["group_2"])

    tie_line_limits = get_loadflow_based_tie_line_limits(tie_lines_df, limit_parameters, "n1")

    assert type(tie_line_limits) == list
    assert type(tie_line_limits[0]) == pd.DataFrame
    assert len(tie_line_limits) == 2
    assert tie_line_limits[0]["name"].values == np.array(["loadflow_based_n1"])
    assert tie_line_limits[1]["name"].values == np.array(["loadflow_based_n1"])
    assert tie_line_limits[0].index.get_level_values("group_name").values == np.array(["group_3"])
    assert tie_line_limits[1].index.get_level_values("group_name").values == np.array(["group_4"])

    empty_limits = get_loadflow_based_tie_line_limits(tie_lines_df[:0], limit_parameters, "n0")
    assert empty_limits == []


def test_get_loadflow_based_trafo_limits():
    trafos_df = pd.DataFrame(
        index=["trafo"],
        data={
            "i1": [10.0],
            "i2": [10.0],
            "n0_i1_max": [100.0],
            "n0_i2_max": [100.0],
            "n1_i1_max": [100.0],
            "n1_i2_max": [100.0],
            "n0_group_name_1": ["group_1"],
            "n0_group_name_2": ["group_2"],
            "n1_group_name_1": ["group_3"],
            "n1_group_name_2": ["group_4"],
        },
    )
    limit_parameters = LimitAdjustmentParameters(
        n_0_factor=1.1,
        n_0_min_increase=0.1,  # New limit should be min(10*1.1, 10+0.1*100) = 20
    )
    trafo_limits = get_loadflow_based_trafo_limits(trafos_df, limit_parameters, "n0")

    assert type(trafo_limits) == list
    assert type(trafo_limits[0]) == pd.DataFrame
    assert len(trafo_limits) == 2, "For Trafos we create a new limit for each existing side limit"
    assert np.array_equal(trafo_limits[0].index.get_level_values("side").values, np.array(["ONE"]))
    assert np.array_equal(trafo_limits[0]["element_type"].values, np.array(["TWO_WINDINGS_TRANSFORMER"]))
    assert np.array_equal(trafo_limits[0]["name"].values, np.array(["loadflow_based_n0"]))
    assert np.array_equal(trafo_limits[0].index.get_level_values("group_name").values, np.array(["group_1"]))

    assert np.array_equal(trafo_limits[1].index.get_level_values("side").values, np.array(["TWO"]))
    assert np.array_equal(trafo_limits[1]["element_type"].values, np.array(["TWO_WINDINGS_TRANSFORMER"]))
    assert np.array_equal(trafo_limits[1]["name"].values, np.array(["loadflow_based_n0"]))
    assert np.array_equal(trafo_limits[1].index.get_level_values("group_name").values, np.array(["group_2"]))

    trafo_limits = get_loadflow_based_trafo_limits(trafos_df, limit_parameters, "n1")

    assert type(trafo_limits) == list
    assert type(trafo_limits[0]) == pd.DataFrame
    assert len(trafo_limits) == 2
    assert np.array_equal(trafo_limits[0].index.get_level_values("side").values, np.array(["ONE"]))
    assert np.array_equal(trafo_limits[0]["element_type"].values, np.array(["TWO_WINDINGS_TRANSFORMER"]))
    assert np.array_equal(trafo_limits[0]["name"].values, np.array(["loadflow_based_n1"]))

    assert np.array_equal(trafo_limits[1].index.get_level_values("side").values, np.array(["TWO"]))
    assert np.array_equal(trafo_limits[1]["element_type"].values, np.array(["TWO_WINDINGS_TRANSFORMER"]))
    assert np.array_equal(trafo_limits[1]["name"].values, np.array(["loadflow_based_n1"]))
    assert np.array_equal(trafo_limits[0].index.get_level_values("group_name").values, np.array(["group_3"]))
    assert np.array_equal(trafo_limits[1].index.get_level_values("group_name").values, np.array(["group_4"]))
    empty_limits = get_loadflow_based_trafo_limits(trafos_df[:0], limit_parameters, "n0")
    assert empty_limits == []


def test_get_all_border_line_limits(limit_update_input):
    branch_df, limit_parameters, network_masks = limit_update_input
    new_limits = get_all_border_line_limits(
        branch_df, limit_parameters, network_masks.line_tso_border, network_masks.tie_line_tso_border
    )

    assert len(new_limits) == 8  # (2 for each dangling Tie line, 2 for each side of the line) for each case (2)

    limit_df = pd.concat(new_limits)
    assert np.array_equal(limit_df["element_type"].unique(), ["LINE", "DANGLING_LINE"])
    assert np.array_equal(limit_df["name"].unique(), ["loadflow_based_n0", "loadflow_based_n1"])
    assert np.array_equal(limit_df["value"].unique(), [20.0])
    assert np.array_equal(
        limit_df[limit_df.element_type == "DANGLING_LINE"].index.get_level_values("side").unique(), ["NONE"]
    )
    assert np.array_equal(
        limit_df[limit_df.element_type == "DANGLING_LINE"].index.get_level_values("group_name").unique(),
        ["group_1", "group_2", "group_3", "group_4"],
    )

    assert np.array_equal(limit_df[limit_df.element_type == "LINE"].index.get_level_values("side").unique(), ["ONE", "TWO"])
    assert np.array_equal(
        limit_df[limit_df.element_type == "LINE"].index.get_level_values("group_name").unique(),
        ["group_1", "group_2", "group_3", "group_4"],
    )


def test_get_all_border_line_limits_only_tie_lines(limit_update_input):
    branch_df, limit_parameters, network_masks = limit_update_input

    new_limits = get_all_border_line_limits(
        branch_df, limit_parameters, np.array([False]), network_masks.tie_line_tso_border
    )

    assert len(new_limits) == 4  # ( 2 for each side of the line) for each case (2)

    limit_df = pd.concat(new_limits)
    assert np.array_equal(limit_df["element_type"].unique(), ["DANGLING_LINE"])
    assert np.array_equal(limit_df["name"].unique(), ["loadflow_based_n0", "loadflow_based_n1"])
    assert np.array_equal(limit_df["value"].unique(), [20.0])
    assert np.array_equal(limit_df.index.get_level_values("side").unique(), ["NONE"])
    assert np.array_equal(
        limit_df.index.get_level_values("group_name").unique(), ["group_1", "group_2", "group_3", "group_4"]
    )


def test_get_all_border_line_limits_only_lines(limit_update_input):
    branch_df, limit_parameters, network_masks = limit_update_input
    only_line_border_masks = replace(network_masks, tie_line_tso_border=np.array([False]))
    new_limits = get_all_border_line_limits(branch_df, limit_parameters, network_masks.line_tso_border, np.array([False]))

    assert len(new_limits) == 4  # ( 2 for each side of the line) for each case (2)

    limit_df = pd.concat(new_limits)
    assert np.array_equal(limit_df["element_type"].unique(), ["LINE"])
    assert np.array_equal(limit_df["name"].unique(), ["loadflow_based_n0", "loadflow_based_n1"])
    assert np.array_equal(limit_df["value"].unique(), [20.0])
    assert np.array_equal(limit_df.index.get_level_values("side").unique(), ["ONE", "TWO"])
    assert np.array_equal(
        limit_df.index.get_level_values("group_name").unique(), ["group_1", "group_2", "group_3", "group_4"]
    )


def test_get_all_border_line_limits_no_borders(limit_update_input):
    branch_df, limit_parameters, _ = limit_update_input
    new_limits = get_all_border_line_limits(branch_df, limit_parameters, np.array([False]), np.array([False]))

    assert len(new_limits) == 0


def test_get_all_dso_trafo_limits(limit_update_input):
    branch_df, limit_parameters, network_masks = limit_update_input
    new_limits = get_all_dso_trafo_limits(branch_df, limit_parameters, network_masks.trafo_dso_border)
    assert len(new_limits) == 4  # For each case (n0, n1) and each side (ONE, TWO) one

    limit_df = pd.concat(new_limits)
    assert np.array_equal(limit_df["element_type"].unique(), ["TWO_WINDINGS_TRANSFORMER"])
    assert np.array_equal(limit_df["name"].unique(), ["loadflow_based_n0", "loadflow_based_n1"])
    assert np.array_equal(limit_df["value"].unique(), [20.0])
    assert np.array_equal(limit_df.index.get_level_values("side").unique(), ["ONE", "TWO"])
    assert np.array_equal(
        limit_df.index.get_level_values("group_name").unique(), ["group_1", "group_2", "group_3", "group_4"]
    )


def test_get_all_dso_trafo_limits_no_border(limit_update_input):
    branch_df, limit_parameters, network_masks = limit_update_input
    no_border_mask = np.array([False])
    new_limits = get_all_dso_trafo_limits(branch_df, limit_parameters, trafo_dso_border=no_border_mask)
    assert len(new_limits) == 0


def test_create_new_border_limits_no_limits_set(ucte_file_with_border, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    limits_before = network.get_operational_limits().copy()

    ucte_importer_parameters.area_settings.border_line_factors = None
    ucte_importer_parameters.area_settings.dso_trafo_factors = None
    ucte_importer_parameters.area_settings.nminus1_area = ["D8", "0"]
    network_masks = powsybl_masks.make_masks(network=network, importer_parameters=ucte_importer_parameters)
    n_border_lines = network_masks.line_tso_border.sum()
    n_border_tie_lines = network_masks.tie_line_tso_border.sum()
    n_border_trafos = network_masks.trafo_dso_border.sum()

    assert n_border_lines == 2
    assert n_border_tie_lines == 2
    assert n_border_trafos == 1
    create_new_border_limits(network, network_masks, ucte_importer_parameters)
    limits_after = network.get_operational_limits()
    assert np.array_equal(limits_before, limits_after)


def test_create_new_border_limits(ucte_file_with_border, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    pypowsybl.loadflow.run_ac(network)
    limits_before = network.get_operational_limits().copy()
    ucte_importer_parameters.area_settings.border_line_factors = LimitAdjustmentParameters()
    ucte_importer_parameters.area_settings.dso_trafo_factors = LimitAdjustmentParameters()
    ucte_importer_parameters.area_settings.nminus1_area = ["D8", "0"]
    network_masks = powsybl_masks.make_masks(network=network, importer_parameters=ucte_importer_parameters)
    n_cases = 2
    pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    branches = network.get_branches()
    trafos = branches[branches.type == "TWO_WINDINGS_TRANSFORMER"]
    lines = branches[branches.type == "LINE"]
    tie_lines = branches[branches.type == "TIE_LINE"]
    border_trafos_with_lf = trafos[network_masks.trafo_dso_border & ~trafos["i2"].isna()]
    border_lines_with_lf = lines[network_masks.line_tso_border & ~lines["i2"].isna()]
    border_tie_lines_with_lf = tie_lines[network_masks.tie_line_tso_border & ~tie_lines["i2"].isna()]
    n_new_limits = n_cases * (
        len(border_lines_with_lf) * 2
        + len(border_tie_lines_with_lf) * 4  # For each dangling line, there are 2 tie line limits. so 4
        + len(border_trafos_with_lf)
    )
    create_new_border_limits(network, network_masks, ucte_importer_parameters)
    limits_after = network.get_operational_limits()

    assert len(limits_after) == len(limits_before) + n_new_limits


def test_create_new_border_limits_3wtrf(test_pypowsybl_cgmes_with_3w_trafo, cgmes_importer_parameters):
    network = pypowsybl.network.load(test_pypowsybl_cgmes_with_3w_trafo)
    pypowsybl.loadflow.run_ac(network)
    limits_before = network.get_operational_limits().copy()
    cgmes_importer_parameters.area_settings.border_line_factors = LimitAdjustmentParameters()
    cgmes_importer_parameters.area_settings.dso_trafo_factors = LimitAdjustmentParameters()
    cgmes_importer_parameters.area_settings.nminus1_area = ["BE"]
    network_masks = powsybl_masks.make_masks(network=network, importer_parameters=cgmes_importer_parameters)
    n_cases = 2
    pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    branches = network.get_branches()
    trafos = branches[branches.type == "TWO_WINDINGS_TRANSFORMER"]
    lines = branches[branches.type == "LINE"]
    tie_lines = branches[branches.type == "TIE_LINE"]
    border_trafos_with_lf = trafos[network_masks.trafo_dso_border & ~trafos["i2"].isna()]
    border_lines_with_lf = lines[network_masks.line_tso_border & ~lines["i2"].isna()]
    border_tie_lines_with_lf = tie_lines[network_masks.tie_line_tso_border & ~tie_lines["i2"].isna()]
    n_new_limits = n_cases * (
        len(border_lines_with_lf) * 2
        + len(border_tie_lines_with_lf) * 4  # For each dangling line, there are 2 tie line limits. so 4
        + len(border_trafos_with_lf)
    )
    create_new_border_limits(network, network_masks, cgmes_importer_parameters)
    limits_after = network.get_operational_limits()

    assert len(limits_after) == len(limits_before) + n_new_limits


def test_create_new_border_limits_3wtrf_conversion(test_pypowsybl_cgmes_with_3w_trafo, cgmes_importer_parameters):
    network = pypowsybl.network.load(test_pypowsybl_cgmes_with_3w_trafo)
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(network)
    if pypowsybl.__version__ <= "1.12.0":
        # Fix the bug, where the operational limits of the 2winding transformers are not set correctly
        op_lim = network.get_operational_limits(all_attributes=True, show_inactive_sets=True)
        trafo3w_lims = op_lim[op_lim.index.str.contains("-Leg")][["group_name"]].rename(
            columns={"group_name": "selected_limits_group_1"}
        )
        trafo3w_lims.index.name = "id"
        network.update_2_windings_transformers(trafo3w_lims)
    pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    limits_before = network.get_operational_limits().copy()
    cgmes_importer_parameters.area_settings.border_line_factors = LimitAdjustmentParameters()
    cgmes_importer_parameters.area_settings.dso_trafo_factors = LimitAdjustmentParameters()
    cgmes_importer_parameters.area_settings.nminus1_area = [""]
    network_masks = powsybl_masks.make_masks(network=network, importer_parameters=cgmes_importer_parameters)
    n_cases = 2
    pypowsybl.loadflow.run_ac(network, DISTRIBUTED_SLACK)
    branches = network.get_branches()
    trafos = branches[branches.type == "TWO_WINDINGS_TRANSFORMER"]
    lines = branches[branches.type == "LINE"]
    tie_lines = branches[branches.type == "TIE_LINE"]
    border_trafos_with_lf = trafos[network_masks.trafo_dso_border]
    border_lines_with_lf = lines[network_masks.line_tso_border & ~lines["i2"].isna()]
    border_tie_lines_with_lf = tie_lines[network_masks.tie_line_tso_border & ~tie_lines["i2"].isna()]

    side_2_limits = limits_before[limits_before.index.get_level_values("side") == "TWO"]
    side_1_limits = limits_before[limits_before == "ONE"]
    n_new_limits = n_cases * (
        len(border_lines_with_lf) * 2
        + len(border_tie_lines_with_lf) * 4  # For each dangling line, there are 2 tie line limits. so 4
        + border_trafos_with_lf.index.isin(side_2_limits.index.get_level_values("element_id")).sum()
        + border_trafos_with_lf.index.isin(side_1_limits.index.get_level_values("element_id")).sum()
    )
    create_new_border_limits(network, network_masks, cgmes_importer_parameters)
    limits_after = network.get_operational_limits()

    assert len(limits_after) == len(limits_before) + n_new_limits
