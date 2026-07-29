# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower
import pandas as pd
import pypowsybl
import pytest
from fsspec.implementations.local import LocalFileSystem
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
)
from toop_engine_grid_helpers.powsybl.powsybl_helpers import (
    change_dangling_to_tie,
    check_powsybl_import,
    extract_single_branch_loadflow_result,
    extract_single_injection_loadflow_result,
    get_branches_with_i,
    get_branches_with_i_max,
    get_injections_with_i,
    get_voltage_level_with_region,
    load_pandapower_net_for_powsybl,
    load_pandapower_net_via_grid2opt_for_powsybl,
    load_powsybl_from_fs,
    save_powsybl_to_fs,
    select_a_generator_as_slack_and_run_loadflow,
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
    file_system = LocalFileSystem()

    pp_net = load_powsybl_from_fs(file_system, ieee14_mat)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_uct(ucte_file):
    file_system = LocalFileSystem()

    pp_net = load_powsybl_from_fs(file_system, ucte_file)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_cgmes(eurostag_tutorial_example1_cgmes):
    file_system = LocalFileSystem()

    pp_net = load_powsybl_from_fs(file_system, eurostag_tutorial_example1_cgmes)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_xiidm(basic_node_breaker_grid_xiidm):
    file_system = LocalFileSystem()

    pp_net = load_powsybl_from_fs(file_system, basic_node_breaker_grid_xiidm)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_save_powsybl_to_fs_xiidm_mat(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("powsybl_save_load")
    net_original = basic_node_breaker_network_powsybl()
    file_system = LocalFileSystem()

    save_powsybl_to_fs(net=net_original, filesystem=file_system, file_path=tmp_path / "grid.xiidm")
    net_loaded = load_powsybl_from_fs(file_system, tmp_path / "grid.xiidm")

    assert net_original.get_buses().equals(net_loaded.get_buses())
    assert net_original.get_branches().equals(net_loaded.get_branches())
    assert net_original.get_injections().equals(net_loaded.get_injections())

    save_powsybl_to_fs(net=net_original, filesystem=file_system, file_path=tmp_path / "grid.xiidm", format="XIIDM")
    net_loaded = load_powsybl_from_fs(file_system, tmp_path / "grid.xiidm")

    assert net_original.get_buses().equals(net_loaded.get_buses())
    assert net_original.get_branches().equals(net_loaded.get_branches())
    assert net_original.get_injections().equals(net_loaded.get_injections())

    save_powsybl_to_fs(net=net_original, filesystem=file_system, file_path=tmp_path / "grid.mat", format="MATPOWER")
    net_loaded = load_powsybl_from_fs(file_system, tmp_path / "grid.mat")

    assert len(net_original.get_buses()) == len(net_loaded.get_buses())
    assert len(net_original.get_branches()) == len(net_loaded.get_branches())
    injection = net_original.get_injections()
    injection = injection[injection["type"] != "BUSBAR_SECTION"]
    assert len(injection) == len(net_loaded.get_injections())


def test_save_powsybl_to_fs_ucte(tmp_path_factory: pytest.TempPathFactory, ucte_file) -> None:
    tmp_path = tmp_path_factory.mktemp("powsybl_save_load")
    net_original = pypowsybl.network.load(ucte_file)
    file_system = LocalFileSystem()
    save_powsybl_to_fs(net=net_original, filesystem=file_system, file_path=tmp_path / "grid.uct", format="UCTE")
    net_loaded = load_powsybl_from_fs(file_system, tmp_path / "grid.uct")

    assert len(net_original.get_buses()) == len(net_loaded.get_buses())
    assert len(net_original.get_branches()) == len(net_loaded.get_branches())
    assert len(net_original.get_injections()) == len(net_loaded.get_injections())


def test_save_powsybl_to_fs_cgmes(tmp_path_factory: pytest.TempPathFactory, eurostag_tutorial_example1_cgmes) -> None:
    tmp_path = tmp_path_factory.mktemp("powsybl_save_load")
    net_original = pypowsybl.network.load(eurostag_tutorial_example1_cgmes)
    file_system = LocalFileSystem()
    save_powsybl_to_fs(net=net_original, filesystem=file_system, file_path=tmp_path / "grid.zip", format="CGMES")
    net_loaded = load_powsybl_from_fs(file_system, tmp_path / "grid.zip")

    assert net_original.get_buses().equals(net_loaded.get_buses())
    assert net_original.get_branches().equals(net_loaded.get_branches())
    assert net_original.get_injections().equals(net_loaded.get_injections())


def test_save_and_load_complex_example_grid_as_cgmes(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("powsybl_save_load")
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    cgmes_file = tmp_path / "cgmes.zip"

    # Export the CGMES files. Note that no load flow is run here on purpose: the fixture already leaves a
    # converged state on the network, and a default-parameter run would leave a slack bus residual large
    # enough for the SV export to emit an SvInjection, which cannot be imported into a node/breaker grid.
    net.save(
        cgmes_file,
        format="CGMES",
    )
    loaded_net = pypowsybl.network.load(cgmes_file)

    assert isinstance(loaded_net, pypowsybl.network.Network)

    n_boundary_lines = len(net.get_boundary_lines())
    n_tie_lines = len(net.get_tie_lines())
    n_batteries = len(net.get_batteries())
    # CGMES models a boundary line as a full AC line ending in a fictitious substation/voltage level/bus.
    # Two boundary lines coupled by a tie line share a single boundary point.
    n_boundary_points = n_boundary_lines - n_tie_lines

    # Element counts that CGMES round trips, ids included, either one to one or up to the fictitious
    # elements added per boundary point.
    for getter, extra in [
        ("get_loads", 0),
        ("get_2_windings_transformers", 0),
        ("get_3_windings_transformers", 0),
        ("get_shunt_compensators", 0),
        ("get_static_var_compensators", 0),
        ("get_hvdc_lines", 0),
        ("get_vsc_converter_stations", 0),
        ("get_lcc_converter_stations", 0),
        ("get_busbar_sections", 0),
        ("get_substations", n_boundary_points),
        ("get_voltage_levels", n_boundary_points),
        ("get_buses", n_boundary_points),
        ("get_lines", n_boundary_lines),
    ]:
        before = getattr(net, getter)()
        after = getattr(loaded_net, getter)()
        assert len(after) == len(before) + extra, f"unexpected {getter} count after the CGMES round trip"
        assert set(before.index) <= set(after.index), f"{getter} lost ids during the CGMES round trip"

    assert len(loaded_net.get_boundary_lines()) == 0
    assert len(loaded_net.get_tie_lines()) == 0

    # CGMES has no battery class, so batteries come back as generators, and every boundary line adds an
    # equivalent injection generator.
    assert len(loaded_net.get_batteries()) == 0
    assert len(loaded_net.get_generators()) == len(net.get_generators()) + n_batteries + n_boundary_lines
    assert set(net.get_generators().index) | set(net.get_batteries().index) <= set(loaded_net.get_generators().index)

    # The injection count is preserved, but the boundary line ids are renamed to their equivalent injections.
    assert len(loaded_net.get_injections()) == len(net.get_injections())

    # Fictitious switches are added for the open terminals of LINE_out_of_service.
    assert set(net.get_switches().index) <= set(loaded_net.get_switches().index)

    result_columns = {
        "get_lines": ["p1", "q1", "p2", "q2", "i1", "i2"],
        "get_2_windings_transformers": ["p1", "q1", "i1"],
        "get_3_windings_transformers": ["p1", "p2", "p3"],
        "get_loads": ["p", "q"],
        "get_generators": ["p", "q"],
        "get_shunt_compensators": ["q"],
        "get_static_var_compensators": ["q"],
    }

    def max_abs_diff(getter: str, column: str) -> float:
        before = getattr(net, getter)()
        after = getattr(loaded_net, getter)()
        common = sorted(set(before.index) & set(after.index))
        # initial=0.0 keeps element types the grid does not have (empty frames) a no-op.
        return float(
            np.nanmax(
                np.abs(before.loc[common, column].to_numpy(float) - after.loc[common, column].to_numpy(float)),
                initial=0.0,
            )
        )

    # Tap changers keep their position and their full step table. Note that target_deadband is not part of
    # the comparison, CGMES does not carry it.
    tap_columns = {
        "get_phase_tap_changers": ["tap", "low_tap", "high_tap", "step_count"],
        "get_ratio_tap_changers": ["tap", "low_tap", "high_tap", "step_count"],
        "get_phase_tap_changer_steps": ["rho", "alpha", "r", "x", "g", "b"],
        "get_ratio_tap_changer_steps": ["rho", "r", "x", "g", "b"],
    }
    for getter, columns in tap_columns.items():
        before = getattr(net, getter)()
        assert before.index.equals(getattr(loaded_net, getter)().index), f"{getter} changed during the round trip"
        for column in columns:
            assert max_abs_diff(getter, column) < 1e-6, f"{getter}.{column} changed during the CGMES round trip"

    # The state variables of the exported load flow are restored on import. The first assertion guards
    # against an all NaN comparison, which max_abs_diff would report as no difference.
    assert net.get_lines()["p1"].notna().any(), "the fixture is expected to leave a converged load flow"
    for getter, columns in result_columns.items():
        for column in columns:
            assert max_abs_diff(getter, column) < 1e-6, f"{getter}.{column} changed during the CGMES round trip"
    for column in ["v_mag", "v_angle"]:
        assert max_abs_diff("get_buses", column) < 1e-6, f"bus {column} changed during the CGMES round trip"

    # The imported grid still solves and reproduces the same operating point.
    lf_result = pypowsybl.loadflow.run_ac(loaded_net)
    assert lf_result[0].status == pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED
    for getter, columns in result_columns.items():
        for column in columns:
            assert max_abs_diff(getter, column) < 0.5, f"{getter}.{column} differs after solving the imported grid"
    assert max_abs_diff("get_buses", "v_mag") < 1e-2

    # CGMES does not carry the slackTerminal extension, so the rerun picks another reference bus and the bus
    # angles only match up to a global offset.
    common_buses = sorted(set(net.get_buses().index) & set(loaded_net.get_buses().index))
    angles_before = net.get_buses().loc[common_buses, "v_angle"].to_numpy(float)
    angles_after = loaded_net.get_buses().loc[common_buses, "v_angle"].to_numpy(float)
    # Disconnected buses have no angle at all, they are skipped by comparing with nanmax.
    assert np.nanmax(np.abs((angles_before - angles_before[0]) - (angles_after - angles_after[0]))) < 1e-2


def test_load_pandapower_net_for_powsybl_with_convert_from_pandapower():
    net = pandapower.networks.case14()
    pypowsybl_network = load_pandapower_net_for_powsybl(net)
    assert isinstance(pypowsybl_network, pypowsybl.network.Network)
    assert len(pypowsybl_network.get_buses()) == len(net.bus)
    assert len(pypowsybl_network.get_branches()) >= len(net.line) + len(net.trafo)
    assert len(pypowsybl_network.get_injections()) >= len(net.load) + len(net.sgen) + len(net.gen)
    # Run load flow to verify conversion
    lf_result = pypowsybl.loadflow.run_ac(pypowsybl_network)
    assert lf_result[0].status == pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED


def test_load_pandapower_net_via_grid2opt_for_powsybl():
    net = pandapower.networks.case9()
    pypowsybl_network = load_pandapower_net_via_grid2opt_for_powsybl(net)
    assert isinstance(pypowsybl_network, pypowsybl.network.Network)
    assert len(pypowsybl_network.get_buses()) == len(net.bus)
    assert len(pypowsybl_network.get_branches()) >= len(net.line) + len(net.trafo)
    assert len(pypowsybl_network.get_injections()) >= len(net.load) + len(net.sgen) + len(net.gen)
    # Run load flow to verify conversion
    lf_result = pypowsybl.loadflow.run_ac(pypowsybl_network)
    assert lf_result[0].status == pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED


def test_check_powsybl_import():
    net = pypowsybl.network.create_ieee57()
    check_powsybl_import(net)


def test_select_a_generator_as_slack_and_run_loadflow():
    net = pypowsybl.network.create_ieee57()
    select_a_generator_as_slack_and_run_loadflow(net)
