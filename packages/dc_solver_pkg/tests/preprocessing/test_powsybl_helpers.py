import math
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import (
    get_lines,
    get_network_as_pu,
    get_p_max,
    get_tie_lines,
    get_trafos,
)
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl
from toop_engine_grid_helpers.powsybl.powsybl_helpers import (
    change_dangling_to_tie,
    extract_single_branch_loadflow_result,
    extract_single_injection_loadflow_result,
    get_branches_with_i,
    get_branches_with_i_max,
    get_injections_with_i,
    get_voltage_level_with_region,
    load_powsybl_from_fs,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_powsybl_helpers(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    assert net is not None

    assert len(get_lines(net)) == len(net.get_lines())
    assert len(get_tie_lines(net)) == len(net.get_tie_lines())
    assert len(get_trafos(net)) == len(net.get_2_windings_transformers())
    assert len(get_p_max(net)) == len(net.get_branches())


def test_extract_single_loadflow_result(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    pypowsybl.loadflow.run_dc(net)
    # Pick a bus, check if everything on that bus sums to zero
    bus = net.get_buses().index[4]
    all_branches = net.get_branches()
    all_injections = net.get_injections()
    branches_from = all_branches[all_branches["bus1_id"] == bus]
    branches_to = all_branches[all_branches["bus2_id"] == bus]
    injections = all_injections[all_injections["bus_id"] == bus]

    p_sum = 0
    for elem_id, branch in branches_from.iterrows():
        p, _ = extract_single_branch_loadflow_result(all_branches, elem_id, from_side=True)
        p_sum += p

    for elem_id, branch in branches_to.iterrows():
        p, _ = extract_single_branch_loadflow_result(all_branches, elem_id, from_side=False)
        p_sum += p

    for elem_id, inj in injections.iterrows():
        p, _ = extract_single_injection_loadflow_result(all_injections, elem_id)
        p_sum += p

    assert np.isclose(p_sum, 0)


def test_get_branches_with_i(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    pypowsybl.loadflow.run_dc(net)
    branches_with_i = get_branches_with_i(net.get_branches(), net)

    pypowsybl.loadflow.run_ac(net)
    branches_with_i_ac = get_branches_with_i(net.get_branches(), net)

    assert len(branches_with_i) == len(branches_with_i_ac)
    assert branches_with_i["i1"].isna().sum() == branches_with_i["p1"].isna().sum()
    assert branches_with_i["i2"].isna().sum() == branches_with_i["p2"].isna().sum()


def test_get_injections_with_i(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    pypowsybl.loadflow.run_dc(net)
    injections_with_i = get_injections_with_i(net.get_injections(), net)

    pypowsybl.loadflow.run_ac(net)
    injections_with_i_ac = get_injections_with_i(net.get_injections(), net)

    assert len(injections_with_i) == len(injections_with_i_ac)
    assert injections_with_i["i"].isna().sum() == injections_with_i["p"].isna().sum()


def test_get_branches_with_imax(powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    pypowsybl.loadflow.run_dc(net)
    branches_with_imax = get_branches_with_i_max(net.get_branches(), net)

    assert branches_with_imax["i1_max"].notna().sum() > 0
    assert branches_with_imax["i2_max"].notna().sum() > 0


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


def test_get_voltage_level_with_region():
    net = basic_node_breaker_network_powsybl()
    res = get_voltage_level_with_region(net).columns
    assert len(res) == 6
    for col in ["name", "substation_id", "nominal_v", "high_voltage_limit", "low_voltage_limit", "region"]:
        assert col in res

    res = get_voltage_level_with_region(net, all_attributes=True).columns
    assert len(res) >= 8  # in case of new attributes added in pypowsybl
    for col in [
        "name",
        "substation_id",
        "nominal_v",
        "high_voltage_limit",
        "low_voltage_limit",
        "fictitious",
        "topology_kind",
        "region",
    ]:
        assert col in res

    attributes = ["name", "substation_id"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 3
    for col in attributes + ["region"]:
        assert col in res

    attributes = ["name", "substation_id", "region"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 3
    for col in attributes:
        assert col in res

    attributes = ["region"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 1
    for col in attributes:
        assert col in res

    with pytest.raises(ValueError):
        get_voltage_level_with_region(net, attributes=attributes, all_attributes=True)


def test_change_dangling_to_tie_no_tie():
    station_elements = pd.DataFrame(
        index=["line1", "line2"],
        data={
            "type": ["LINE", "LINE"],
            "name": ["line_name", "line_name"],
            "in_service": [True, True],
        },
    )
    dangling_lines = pd.DataFrame(
        index=["dangling1", "dangling2"],
        data={"tie_line_id": ["tie_line1", "tie_line2"]},
    )
    result_new = change_dangling_to_tie(dangling_lines, station_elements)
    assert np.all(result_new == station_elements)


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


def test_load_powsybl_from_fs_success(powsybl_data_folder: Path) -> None:
    """Test successful loading of a Powsybl network from filesystem."""
    # Create a mock filesystem
    mock_fs = MagicMock(spec=AbstractFileSystem)

    # Get an actual network file for testing
    grid_file = powsybl_data_folder / "test_grid.xiidm"
    file_path = Path("remote/path/test_grid.xiidm")

    # Mock the download method to copy the actual file
    def mock_download(remote_path, local_path):
        shutil.copy2(grid_file, local_path)

    mock_fs.download.side_effect = mock_download

    # Test the function
    network = load_powsybl_from_fs(mock_fs, file_path)

    # Verify the filesystem download was called correctly
    mock_fs.download.assert_called_once()
    args = mock_fs.download.call_args[0]
    assert args[0] == str(file_path)
    assert Path(args[1]).name == file_path.name

    # Verify network was loaded successfully
    assert network is not None
    assert hasattr(network, "get_buses")
