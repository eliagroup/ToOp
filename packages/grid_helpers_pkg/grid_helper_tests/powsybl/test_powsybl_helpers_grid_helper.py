import fsspec
import numpy as np
import pandas as pd
import pypowsybl
import pytest
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


def test_extract_single_loadflow_result() -> None:
    net = basic_node_breaker_network_powsybl()
    pypowsybl.loadflow.run_dc(net)
    # Pick a bus, check if everything on that bus sums to zero
    bus = net.get_buses().index[2]
    all_branches = net.get_branches()
    all_injections = net.get_injections()
    all_injections = all_injections[all_injections["p"].notna()]
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


def test_get_branches_with_i() -> None:
    net = basic_node_breaker_network_powsybl()
    pypowsybl.loadflow.run_dc(net)
    branches_with_i = get_branches_with_i(net.get_branches(), net)

    pypowsybl.loadflow.run_ac(net)
    branches_with_i_ac = get_branches_with_i(net.get_branches(), net)

    assert len(branches_with_i) == len(branches_with_i_ac)
    assert branches_with_i["i1"].isna().sum() == branches_with_i["p1"].isna().sum()
    assert branches_with_i["i2"].isna().sum() == branches_with_i["p2"].isna().sum()


def test_get_injections_with_i() -> None:
    net = basic_node_breaker_network_powsybl()
    pypowsybl.loadflow.run_dc(net)
    injections_with_i = get_injections_with_i(net.get_injections(), net)

    pypowsybl.loadflow.run_ac(net)
    injections_with_i_ac = get_injections_with_i(net.get_injections(), net)

    assert len(injections_with_i) == len(injections_with_i_ac)
    assert injections_with_i["i"].isna().sum() == injections_with_i["i"].isna().sum()


def test_get_branches_with_imax() -> None:
    net = basic_node_breaker_network_powsybl()
    pypowsybl.loadflow.run_dc(net)
    branches_with_imax = get_branches_with_i_max(net.get_branches(), net)

    assert branches_with_imax["i1_max"].notna().sum() > 0
    assert branches_with_imax["i2_max"].notna().sum() > 0


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


def test_load_powsybl_from_fs_mat(ieee14_mat):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, ieee14_mat)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_uct(ucte_file):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, ucte_file)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_cgmes(eurostag_tutorial_example1_cgmes):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, eurostag_tutorial_example1_cgmes)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_xiidm(basic_node_breaker_grid_xiidm):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, basic_node_breaker_grid_xiidm)
    assert isinstance(pp_net, pypowsybl.network.Network)
