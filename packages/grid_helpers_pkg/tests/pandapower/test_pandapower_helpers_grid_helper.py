import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from jaxtyping import Bool
from toop_engine_grid_helpers.pandapower.example_grids import pandapower_case30_with_psts
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    check_for_splits,
    get_bus_key,
    get_bus_key_injection,
    get_dc_bus_voltage,
    get_element_table,
    get_pandapower_branch_loadflow_results_sequence,
    get_pandapower_bus_loadflow_results_sequence,
    get_pandapower_loadflow_results_in_ppc,
    get_pandapower_loadflow_results_injection,
    get_phaseshift_key,
    get_phaseshift_mask,
    get_power_key,
    get_remotely_connected_buses,
    get_shunt_real_power,
    load_pandapower_from_fs,
    save_pandapower_to_fs,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import parse_globally_unique_id_series


def test_get_phaseshift_key() -> None:
    assert get_phaseshift_key("trafo") == "shift_degree", "Wrong shift column returned"
    assert get_phaseshift_key("trafo3w_lv") == "shift_lv_degree", "Wrong shift column returned"
    assert get_phaseshift_key("trafo3w_mv") == "shift_mv_degree", "Wrong shift column returned"

    with pytest.raises(AssertionError):
        get_phaseshift_key("trafo3w_hv")


def test_get_phaseshift_mask() -> None:
    net = pandapower_case30_with_psts()

    mask, controllable, shift_taps = get_phaseshift_mask(net)
    assert mask.shape == (len(net.line) + len(net.trafo),)
    assert controllable.shape == mask.shape
    assert not any(controllable & ~mask)
    assert len(shift_taps) == controllable.sum()

    controllable_trafos = controllable[len(net.line) :]
    tap_min = (net.trafo.tap_min * net.trafo.tap_step_degree)[controllable_trafos]
    tap_max = (net.trafo.tap_max * net.trafo.tap_step_degree)[controllable_trafos]
    for taps, t_min, t_max in zip(shift_taps, tap_min, tap_max):
        assert len(taps) > 0
        assert t_min == taps[0] == taps.min()
        assert t_max == taps[-1] == taps.max()

    assert sum(controllable) <= len(net.trafo)
    assert not any(mask[: len(net.line)])

    assert isinstance(controllable, Bool[np.ndarray, " n_branches"])
    assert isinstance(mask, Bool[np.ndarray, " n_branches"])


def test_get_remotely_connected_buses() -> None:
    net = pp.networks.case14()

    # It should find all 14 buses
    connected_buses = get_remotely_connected_buses(net, buses=[0])
    assert len(connected_buses) == 14

    # However it needs more than 2 iterations
    with pytest.raises(RuntimeError):
        get_remotely_connected_buses(net, buses=[0], max_num_iterations=2)


def test_parse_globally_unique_id_series():
    series = pd.Series(["1%%bus", "2%%line", "3%%trafo", "4%%ext_grid"])
    results = parse_globally_unique_id_series(series)
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (4, 2)
    assert results["id"].tolist() == [1, 2, 3, 4]
    assert results["type"].tolist() == ["bus", "line", "trafo", "ext_grid"]


def test_get_bus_key():
    assert get_bus_key("line", True) == "from_bus"
    assert get_bus_key("line", False) == "to_bus"

    assert get_bus_key("trafo", True) == "hv_bus"
    assert get_bus_key("trafo", False) == "lv_bus"

    assert get_bus_key("trafo3w_lv", True) == "lv_bus"
    assert get_bus_key("trafo3w_lv", False) == "lv_bus"

    assert get_bus_key("trafo3w_mv", True) == "mv_bus"
    assert get_bus_key("trafo3w_mv", False) == "mv_bus"

    assert get_bus_key("trafo3w_hv", True) == "hv_bus"
    assert get_bus_key("trafo3w_hv", False) == "hv_bus"

    with pytest.raises(AssertionError):
        get_bus_key("non_existing_branch", False)


def test_get_bus_key_injection():
    assert get_bus_key_injection("gen") == "bus"
    assert get_bus_key_injection("load") == "bus"
    assert get_bus_key_injection("sgen") == "bus"
    assert get_bus_key_injection("shunt") == "bus"
    assert get_bus_key_injection("dcline_from") == "from_bus"
    assert get_bus_key_injection("dcline_to") == "to_bus"

    with pytest.raises(AssertionError):
        get_bus_key_injection("non_existing_injection")


def test_get_element_table():
    assert get_element_table("line") == "line"
    assert get_element_table("trafo") == "trafo"
    assert get_element_table("bus") == "bus"
    assert get_element_table("load") == "load"
    assert get_element_table("gen") == "gen"
    assert get_element_table("sgen") == "sgen"
    assert get_element_table("shunt") == "shunt"
    assert get_element_table("dcline_from") == "dcline"
    assert get_element_table("dcline_to") == "dcline"
    assert get_element_table("trafo3w") == "trafo3w"
    assert get_element_table("trafo3w_hv") == "trafo3w"
    assert get_element_table("trafo3w_mv") == "trafo3w"
    assert get_element_table("trafo3w_lv") == "trafo3w"
    assert get_element_table("ward_load") == "ward"
    assert get_element_table("ward_shunt") == "ward"
    assert get_element_table("xward_load") == "xward"
    assert get_element_table("xward_shunt") == "xward"
    assert get_element_table("xward_gen") == "xward"

    with pytest.raises(AssertionError):
        get_element_table("non_existing_element")

    assert get_element_table("line", res_table=True).startswith("res_")


@pytest.mark.parametrize(
    "element_type,res_table,reactive,expected",
    [
        ("gen", False, False, "p_mw"),
        ("sgen", False, False, "p_mw"),
        ("load", False, False, "p_mw"),
        ("shunt", False, False, "p_mw"),
        ("dcline_from", False, False, "p_mw"),
        ("ward_load", False, False, "ps_mw"),
        ("ward_shunt", False, False, "pz_mw"),
        ("gen", False, True, None),  # Should raise KeyError
        ("sgen", False, True, "q_mvar"),
        ("load", False, True, "q_mvar"),
        ("shunt", False, True, "q_mvar"),
        ("ward_load", False, True, "qs_mvar"),
        ("ward_shunt", False, True, "qz_mvar"),
        ("gen", True, False, "p_mw"),
        ("sgen", True, False, "p_mw"),
        ("load", True, False, "p_mw"),
        ("shunt", True, False, "p_mw"),
        ("dcline_from", True, False, "p_from_mw"),
        ("dcline_to", True, False, "p_to_mw"),
        ("ward_load", True, False, "p_mw"),
        ("gen", True, True, "q_mvar"),
        ("sgen", True, True, "q_mvar"),
        ("load", True, True, "q_mvar"),
        ("shunt", True, True, "q_mvar"),
        ("dcline_from", True, True, "q_from_mvar"),
        ("dcline_to", True, True, "q_to_mvar"),
        ("ward_load", True, True, "q_mvar"),
    ],
)
def test_get_power_key(element_type, res_table, reactive, expected):
    if expected is None:
        with pytest.raises(KeyError):
            get_power_key(element_type, res_table=res_table, reactive=reactive)
    else:
        assert get_power_key(element_type, res_table=res_table, reactive=reactive) == expected


def test_get_pandapower_loadflow_results_in_ppc_basic():
    # Create a minimal pandapower network with all branch types
    net = pp.create_empty_network()
    # Add buses
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    b4 = pp.create_bus(net, vn_kv=110)
    b5 = pp.create_bus(net, vn_kv=110)
    # Add line
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # Add trafo
    t1 = pp.create_transformer_from_parameters(
        net,
        hv_bus=b2,
        lv_bus=b3,
        sn_mva=10,
        vn_hv_kv=110,
        vn_lv_kv=110,
        vkr_percent=0.1,
        vk_percent=10,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    # Add trafo3w
    t3w = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=b3,
        mv_bus=b4,
        lv_bus=b5,
        sn_hv_mva=10,
        sn_mv_mva=5,
        sn_lv_mva=5,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10,
        vk_mv_percent=10,
        vk_lv_percent=10,
        vkr_hv_percent=0.1,
        vkr_mv_percent=0.1,
        vkr_lv_percent=0.1,
        pfe_kw=0,
        i0_percent=0,
    )
    # Add impedance
    z1 = pp.create_impedance(net, from_bus=b1, to_bus=b3, rft_pu=0.01, xft_pu=0.01, sn_mva=10, name="imp")
    # Add xward (aux branch, result is always 0 in this function)
    xw1 = pp.create_xward(
        net,
        bus=b4,
        ps_mw=1,
        qs_mvar=1,
        pz_mw=1,
        qz_mvar=1,
        vm_pu=1.0,
        rx=0.01,
        xz_ohm=0.01,
        rz_ohm=0.01,
        r_ohm=0.01,
        x_ohm=0.01,
    )

    # Add ext_grid and runpp to get results
    pp.create_ext_grid(net, bus=b1)
    pp.rundcpp(net)

    # Call the function
    res = get_pandapower_loadflow_results_in_ppc(net)

    # Check length: should be sum of all branch elements (lines, trafos, trafo3w: 3 branches, impedance, xward)
    expected_length = (
        len(net.res_line)
        + len(net.res_trafo)
        + len(net.res_trafo3w) * 3  # trafo3w: hv, mv, lv
        + len(net.res_impedance)
        + len(net.xward)
    )
    assert res.shape == (expected_length,)
    # Check that the last len(net.xward) elements are zeros
    if len(net.xward):
        assert np.all(res[-len(net.xward) :] == 0)


def test_get_pandapower_loadflow_results_in_ppc_empty():
    # Test with an empty network (no branches)
    net = pp.create_empty_network()
    # Add buses only
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.rundcpp(net)
    res = get_pandapower_loadflow_results_in_ppc(net)
    # Should be empty array
    assert isinstance(res, np.ndarray)
    assert res.size == 0


def test_get_pandapower_loadflow_results_in_ppc_with_nan():
    # Test with a network where some results are NaN (simulate split)
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Manually set a result to NaN
    net.res_line.at[l1, "p_from_mw"] = np.nan
    res = get_pandapower_loadflow_results_in_ppc(net)
    assert np.isnan(res[0])


def test_get_pandapower_loadflow_results_injection_active_and_reactive():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)

    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    l2 = pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # Add elements
    g1 = pp.create_gen(net, bus=b2, p_mw=10, vm_pu=1.01)
    s1 = pp.create_sgen(net, bus=b2, p_mw=5, q_mvar=2)
    l1 = pp.create_load(net, bus=b3, p_mw=7, q_mvar=3)
    sh1 = pp.create_shunt(net, bus=b3, q_mvar=-4)
    # Add ext_grid and runpp
    pp.create_ext_grid(net, bus=b1)
    pp.rundcpp(net)
    # Test active power
    types = ["gen", "sgen", "load", "shunt"]
    ids = [g1, s1, l1, sh1]
    res = get_pandapower_loadflow_results_injection(net, types, ids, reactive=False)
    # Should match res_gen.p_mw, res_sgen.p_mw, res_load.p_mw, res_shunt.p_mw
    assert np.allclose(res[0], -net.res_gen.loc[g1, "p_mw"])
    assert np.allclose(res[1], -net.res_sgen.loc[s1, "p_mw"])
    assert np.allclose(res[2], net.res_load.loc[l1, "p_mw"])
    assert np.allclose(res[3], -net.res_shunt.loc[sh1, "p_mw"])
    # Test reactive power
    res_q = get_pandapower_loadflow_results_injection(net, types, ids, reactive=True)
    # DC LF so all should be nan
    assert np.isnan(res_q[0])
    assert np.allclose(res_q[1], np.nan, equal_nan=True)
    assert np.allclose(res_q[2], np.nan, equal_nan=True)
    assert np.allclose(res_q[3], np.nan, equal_nan=True)


def test_get_pandapower_loadflow_results_injection_empty_input():
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=0)
    pp.rundcpp(net)
    res = get_pandapower_loadflow_results_injection(net, [], [], reactive=False)
    assert isinstance(res, np.ndarray)
    assert res.size == 0


def test_get_pandapower_loadflow_results_injection_invalid_type_returns_zero():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Use a type not in power_key_lookup_table
    with pytest.raises(AssertionError):
        get_pandapower_loadflow_results_injection(net, ["nonexistent"], [0], reactive=False)


def test_get_pandapower_branch_loadflow_results_sequence_active_and_reactive():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    # Add line
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # Add trafo
    t1 = pp.create_transformer_from_parameters(
        net,
        hv_bus=b2,
        lv_bus=b3,
        sn_mva=10,
        vn_hv_kv=110,
        vn_lv_kv=110,
        vkr_percent=0.1,
        vk_percent=10,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    # Add impedance
    z1 = pp.create_impedance(net, from_bus=b1, to_bus=b3, rft_pu=0.01, xft_pu=0.01, sn_mva=10)
    # Add ext_grid and runpp
    pp.create_ext_grid(net, bus=b1)
    pp.rundcpp(net)

    # Test active power from end
    types = ["line", "trafo", "impedance"]
    ids = [l1, t1, z1]
    res_active = get_pandapower_branch_loadflow_results_sequence(net, types, ids, measurement="active", from_end=True)
    assert isinstance(res_active, np.ndarray)
    assert res_active.shape == (3,)
    # Should match the sign convention and values in res tables
    assert np.allclose(res_active[0], -net.res_line.loc[l1, "p_from_mw"])
    assert np.allclose(res_active[1], -net.res_trafo.loc[t1, "p_hv_mw"])
    assert np.allclose(res_active[2], -net.res_impedance.loc[z1, "p_from_mw"])

    # Test active power to end
    res_active_to = get_pandapower_branch_loadflow_results_sequence(net, types, ids, measurement="active", from_end=False)
    assert np.allclose(res_active_to[0], -net.res_line.loc[l1, "p_to_mw"])
    assert np.allclose(res_active_to[1], -net.res_trafo.loc[t1, "p_lv_mw"])
    assert np.allclose(res_active_to[2], -net.res_impedance.loc[z1, "p_to_mw"])

    # Test reactive power from end
    res_reactive = get_pandapower_branch_loadflow_results_sequence(net, types, ids, measurement="reactive", from_end=True)
    assert res_reactive.shape == (3,)
    assert np.allclose(res_reactive[0], -net.res_line.loc[l1, "q_from_mvar"])
    assert np.allclose(res_reactive[1], -net.res_trafo.loc[t1, "q_hv_mvar"])
    assert np.allclose(res_reactive[2], -net.res_impedance.loc[z1, "q_from_mvar"])

    # Test current from end (should be in A, not kA)
    res_current = get_pandapower_branch_loadflow_results_sequence(net, types, ids, measurement="current", from_end=True)
    assert res_current.shape == (3,)
    assert np.allclose(res_current[0], -net.res_line.loc[l1, "i_from_ka"] * 1000)
    assert np.allclose(res_current[1], -net.res_trafo.loc[t1, "i_hv_ka"] * 1000)
    assert np.allclose(res_current[2], -net.res_impedance.loc[z1, "i_from_ka"] * 1000)


def test_get_pandapower_branch_loadflow_results_sequence_adjust_signs_false():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.rundcpp(net)
    res = get_pandapower_branch_loadflow_results_sequence(net, ["line"], [l1], measurement="active", adjust_signs=False)
    assert np.allclose(res[0], net.res_line.loc[l1, "p_from_mw"])


def test_get_pandapower_branch_loadflow_results_sequence_fill_na():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Set result to NaN
    net.res_line.at[l1, "p_from_mw"] = np.nan
    res = get_pandapower_branch_loadflow_results_sequence(net, ["line"], [l1], measurement="active", fill_na=42.0)
    assert np.allclose(res[0], 42.0)


def test_get_pandapower_branch_loadflow_results_sequence_empty_input():
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=0)
    pp.rundcpp(net)
    res = get_pandapower_branch_loadflow_results_sequence(net, [], [], measurement="active")
    assert isinstance(res, np.ndarray)
    assert res.size == 0


def test_get_pandapower_branch_loadflow_results_sequence_invalid_type_raises():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    with pytest.raises(KeyError):
        get_pandapower_branch_loadflow_results_sequence(net, ["nonexistent"], [0], measurement="active")


def test_get_pandapower_bus_loadflow_results_sequence_angle_and_magnitude():
    # Create a simple network
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.runpp(net)

    # Test voltage angle
    bus_ids = [b1, b2, b3]
    res_angle = get_pandapower_bus_loadflow_results_sequence(net, bus_ids, voltage_magnitude=False)
    expected_angle = net.res_bus.va_degree.loc[bus_ids].values
    assert isinstance(res_angle, np.ndarray)
    assert np.allclose(res_angle, expected_angle)

    # Test voltage magnitude
    res_vm = get_pandapower_bus_loadflow_results_sequence(net, bus_ids, voltage_magnitude=True)
    expected_vm = net.res_bus.vm_pu.loc[bus_ids].values
    assert isinstance(res_vm, np.ndarray)
    assert np.allclose(res_vm, expected_vm)


def test_get_pandapower_bus_loadflow_results_sequence_empty_input():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    res = get_pandapower_bus_loadflow_results_sequence(net, [], voltage_magnitude=True)
    assert isinstance(res, np.ndarray)
    assert res.size == 0


def test_get_pandapower_bus_loadflow_results_sequence_nan_for_invalid_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Pass a bus id that does not exist
    with pytest.raises(KeyError):
        get_pandapower_bus_loadflow_results_sequence(net, [999], voltage_magnitude=True)


def test_get_dc_bus_voltage_basic():
    # Test with buses only, no generators
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=220)
    b3 = pp.create_bus(net, vn_kv=380)
    # No generators, so should return vn_kv for each bus
    result = get_dc_bus_voltage(net)
    assert isinstance(result, pd.Series)
    assert result.loc[b1] == 110
    assert result.loc[b2] == 220
    assert result.loc[b3] == 380
    assert set(result.index) == set(net.bus.index)


def test_get_dc_bus_voltage_with_generator_vm_pu():
    # Test with generator connected, vm_pu specified
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=220)
    g1 = pp.create_gen(net, bus=b1, p_mw=10, vm_pu=1.05)
    # Should use generator vm_pu for b1, default for b2
    result = get_dc_bus_voltage(net)
    assert np.isclose(result.loc[b1], 110 * 1.05)
    assert np.isclose(result.loc[b2], 220)


def test_get_dc_bus_voltage_with_generator_vm_pu_nan():
    # Test with generator but vm_pu is NaN, should fallback to 1.0
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    g1 = pp.create_gen(net, bus=b1, p_mw=10, vm_pu=np.nan)
    result = get_dc_bus_voltage(net)
    assert np.isclose(result.loc[b1], 110)


def test_get_dc_bus_voltage_multiple_buses_and_gens():
    # Multiple buses, some with generators, some without
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=220)
    b3 = pp.create_bus(net, vn_kv=380)
    g1 = pp.create_gen(net, bus=b1, p_mw=10, vm_pu=1.02)
    g2 = pp.create_gen(net, bus=b3, p_mw=20, vm_pu=0.98)
    result = get_dc_bus_voltage(net)
    assert np.isclose(result.loc[b1], 110 * 1.02)
    assert np.isclose(result.loc[b2], 220)
    assert np.isclose(result.loc[b3], 380 * 0.98)


def test_get_dc_bus_voltage_empty_network():
    # No buses at all
    net = pp.create_empty_network()
    result = get_dc_bus_voltage(net)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_check_for_splits_no_split():
    # Create a simple network with a line, no split
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Should not detect a split
    assert check_for_splits(net, ["line"], [l1]) is False


def test_check_for_splits_with_nan():
    # Create a network and manually set a result to NaN to simulate a split
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Set current result to NaN
    net.res_line.at[l1, "i_from_ka"] = np.nan
    assert check_for_splits(net, ["line"], [l1]) is True


def test_check_for_splits_multiple_branches_some_nan():
    # Multiple branches, only one has NaN
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    l2 = pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Set one result to NaN
    net.res_line.at[l2, "i_from_ka"] = np.nan
    assert check_for_splits(net, ["line", "line"], [l1, l2]) is True


def test_check_for_splits_multiple_branches_none_nan():
    # Multiple branches, none have NaN
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    l2 = pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    assert check_for_splits(net, ["line", "line"], [l1, l2]) is False


def test_check_for_splits_raises_on_length_mismatch():
    net = pp.create_empty_network()
    # Length mismatch should raise AssertionError
    with pytest.raises(AssertionError):
        check_for_splits(net, ["line"], [1, 999])


def test_check_for_splits_trafo_nan():
    # Test with a transformer branch
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    t1 = pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=10,
        vn_hv_kv=110,
        vn_lv_kv=110,
        vkr_percent=0.1,
        vk_percent=10,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    # Set transformer current to NaN
    net.res_trafo.at[t1, "i_hv_ka"] = np.nan
    assert check_for_splits(net, ["trafo"], [t1]) is True


def test_check_for_splits_trafo_no_nan():
    # Transformer branch, no NaN
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    t1 = pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=10,
        vn_hv_kv=110,
        vn_lv_kv=110,
        vkr_percent=0.1,
        vk_percent=10,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0,
    )
    pp.create_ext_grid(net, bus=b1)
    pp.runpp(net)
    assert check_for_splits(net, ["trafo"], [t1]) is False


def test_get_shunt_real_power_basic():
    # Simple case: bus_voltage == shunt_voltage, shunt_step default
    bus_voltage = np.array([110.0, 220.0])
    shunt_power = np.array([10.0, 20.0])
    result = get_shunt_real_power(bus_voltage, shunt_power)
    # shunt_vratio = 1, shunt_step = 1, so result == shunt_power
    np.testing.assert_allclose(result, shunt_power)


def test_get_shunt_real_power_with_shunt_voltage():
    # bus_voltage != shunt_voltage
    bus_voltage = np.array([110.0, 220.0])
    shunt_voltage = np.array([100.0, 200.0])
    shunt_power = np.array([10.0, 20.0])
    expected = shunt_power * ((bus_voltage / shunt_voltage) ** 2)
    result = get_shunt_real_power(bus_voltage, shunt_power, shunt_voltage=shunt_voltage)
    np.testing.assert_allclose(result, expected)


def test_get_shunt_real_power_with_shunt_step():
    # shunt_step provided
    bus_voltage = np.array([110.0, 220.0])
    shunt_power = np.array([10.0, 20.0])
    shunt_step = np.array([2, 3])
    expected = shunt_power * shunt_step
    result = get_shunt_real_power(bus_voltage, shunt_power, shunt_step=shunt_step)
    np.testing.assert_allclose(result, expected)


def test_get_shunt_real_power_with_all_args():
    # All arguments provided
    bus_voltage = np.array([110.0, 220.0])
    shunt_voltage = np.array([100.0, 200.0])
    shunt_power = np.array([10.0, 20.0])
    shunt_step = np.array([2, 3])
    expected = shunt_power * shunt_step * ((bus_voltage / shunt_voltage) ** 2)
    result = get_shunt_real_power(bus_voltage, shunt_power, shunt_voltage=shunt_voltage, shunt_step=shunt_step)
    np.testing.assert_allclose(result, expected)


def test_get_shunt_real_power_empty():
    # Empty input arrays
    bus_voltage = np.array([])
    shunt_power = np.array([])
    result = get_shunt_real_power(bus_voltage, shunt_power)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_get_shunt_real_power_single_element():
    # Single element input
    bus_voltage = np.array([110.0])
    shunt_power = np.array([10.0])
    shunt_voltage = np.array([100.0])
    shunt_step = np.array([2])
    expected = shunt_power * shunt_step * ((bus_voltage / shunt_voltage) ** 2)
    result = get_shunt_real_power(bus_voltage, shunt_power, shunt_voltage=shunt_voltage, shunt_step=shunt_step)
    np.testing.assert_allclose(result, expected)


def test_get_remotely_connected_buses_all_connected():
    # Create a simple network with 3 buses connected in a line
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # All buses should be connected starting from b1
    result = get_remotely_connected_buses(net, buses=[b1])
    assert set(result) == {b1, b2, b3}


def test_get_remotely_connected_buses_islanded_bus():
    # Create two islands
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    b4 = pp.create_bus(net, vn_kv=110)
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # b3 and b4 are not connected to anything
    result = get_remotely_connected_buses(net, buses=[b3])
    assert set(result) == {b3}
    # b1 should only find b1 and b2
    result2 = get_remotely_connected_buses(net, buses=[b1])
    assert set(result2) == {b1, b2}


def test_get_remotely_connected_buses_multiple_start_buses():
    # Create a network with two islands
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    b4 = pp.create_bus(net, vn_kv=110)
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_line_from_parameters(
        net, from_bus=b3, to_bus=b4, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    # Start from both islands
    result = get_remotely_connected_buses(net, buses=[b1, b3])
    assert set(result) == {b1, b2, b3, b4}


def test_get_remotely_connected_buses_respects_max_iterations():
    # Create a network with a loop to test max_num_iterations
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=110) for _ in range(5)]
    for i in range(5):
        pp.create_line_from_parameters(
            net,
            from_bus=buses[i],
            to_bus=buses[(i + 1) % 5],
            length_km=1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0,
            max_i_ka=1,
        )
    # Should raise if max_num_iterations is too low
    with pytest.raises(RuntimeError):
        get_remotely_connected_buses(net, buses=[buses[0]], max_num_iterations=1)


def test_get_remotely_connected_buses_empty_input():
    net = pp.create_empty_network()
    # No buses in network
    result = get_remotely_connected_buses(net, buses=[])
    assert result == set()


def test_get_remotely_connected_buses_nonexistent_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    # Pass a bus id that does not exist
    result = get_remotely_connected_buses(net, buses=[999])
    assert result == set()


def test_get_remotely_connected_buses_respects_switches():
    # Create a network with a switch that is open
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    l1 = pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1
    )
    sw = pp.create_switch(net, bus=b1, element=l1, et="l", closed=False)
    # Should not hop over open switch
    result = get_remotely_connected_buses(net, buses=[b1], respect_switches=True)
    assert set(result) == {b1}
    # Should hop over if respect_switches is False
    result2 = get_remotely_connected_buses(net, buses=[b1], respect_switches=False)
    assert set(result2) == {b1, b2}


def test_load_pp_from_fs_json(ieee14_json):
    file_system = LocalFileSystem()

    pp_net = load_pandapower_from_fs(file_system, ieee14_json)
    assert isinstance(pp_net, pp.pandapowerNet)


def test_load_pp_from_fs_mat(ieee14_mat):
    file_system = LocalFileSystem()

    pp_net = load_pandapower_from_fs(file_system, ieee14_mat)
    assert isinstance(pp_net, pp.pandapowerNet)


def test_load_pp_from_fs_uct(ucte_file):
    file_system = LocalFileSystem()

    pp_net = load_pandapower_from_fs(file_system, ucte_file)
    assert isinstance(pp_net, pp.pandapowerNet)


def test_load_pp_from_fs_cgmes(eurostag_tutorial_example1_cgmes):
    file_system = LocalFileSystem()

    pp_net = load_pandapower_from_fs(file_system, eurostag_tutorial_example1_cgmes)
    assert isinstance(pp_net, pp.pandapowerNet)


def test_save_load_pandapower_to_from_fs(tmp_path):
    file_system = LocalFileSystem()
    file_path = tmp_path / "test_net.json"

    net = pp.networks.case14()
    pp.runpp(net)
    # Save the network to the filesystem
    save_pandapower_to_fs(net=net, filesystem=file_system, file_path=file_path)

    # Load the network back from the filesystem
    loaded_net = load_pandapower_from_fs(file_system, file_path)

    assert isinstance(loaded_net, pp.pandapowerNet)
    assert loaded_net.bus.equals(net.bus)
    assert loaded_net.line.equals(net.line)
    assert loaded_net.gen.equals(net.gen)

    save_pandapower_to_fs(net=net, filesystem=file_system, file_path=file_path, format="JSON")

    # Load the network back from the filesystem
    loaded_net = load_pandapower_from_fs(file_system, file_path)

    assert isinstance(loaded_net, pp.pandapowerNet)
    assert loaded_net.bus.equals(net.bus)
    assert loaded_net.line.equals(net.line)
    assert loaded_net.gen.equals(net.gen)

    file_path = tmp_path / "test_net.mat"
    save_pandapower_to_fs(net=net, filesystem=file_system, file_path=file_path, format="MATPOWER")

    # Load the network back from the filesystem
    loaded_net = load_pandapower_from_fs(file_system, file_path)

    assert isinstance(loaded_net, pp.pandapowerNet)
    assert len(loaded_net.bus) == len(net.bus)
    assert len(loaded_net.line) == len(net.line)
    assert len(loaded_net.gen) == len(net.gen)

    # test value error, if beartype is active it raises Exception instead of ValueError
    with pytest.raises((ValueError, Exception)):
        save_pandapower_to_fs(net=net, filesystem=file_system, file_path=file_path, format="WRONG_FORMAT")
