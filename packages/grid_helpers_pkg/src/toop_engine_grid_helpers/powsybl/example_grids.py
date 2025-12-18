# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import datetime
import os
from pathlib import Path

import numpy as np
import pandapower
import pandas as pd
import pypowsybl
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import get_stations_bus_breaker
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_pandapower_net_for_powsybl
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.asset_topology_helpers import save_asset_topology
from toop_engine_interfaces.folder_structure import NETWORK_MASK_NAMES, PREPROCESSING_PATHS


def add_phaseshift_transformer_to_line_powsybl(
    net: pypowsybl.network.Network,
    line_idx: str,
) -> None:
    """Add a phaseshift transformer to the from side of a line

    Inserts a helper bus and a transformer with a tap changer to the given line.

    Parameters
    ----------
    net : Network
        The powsybl network, will be modified in place.
    line_idx : str
        The index of the line in net.line on which to insert the phase-shifting transformer.
    """
    line = net.get_lines(all_attributes=True).loc[line_idx]
    vl = line["voltage_level1_id"]
    nominal_v = net.get_voltage_levels().loc[vl, "nominal_v"]
    helper_bus = f"pst_bus_{line_idx}"
    net.create_buses(
        id=helper_bus,
        voltage_level_id=vl,
    )
    original_bus = line["bus_breaker_bus1_id"]
    net.update_branches(
        id=line_idx,
        bus_breaker_bus1_id=helper_bus,
    )

    pst = f"pst_{line_idx}"
    net.create_2_windings_transformers(
        id=pst,
        bus1_id=original_bus,
        voltage_level1_id=vl,
        bus2_id=helper_bus,
        voltage_level2_id=vl,
        rated_u1=nominal_v,
        rated_u2=nominal_v,
        rated_s=100,
        b=0.0,
        g=0.0,
        r=0.1,
        x=1.0,
    )

    ptc_df = pd.DataFrame.from_records(
        index="id",
        columns=[
            "id",
            "target_deadband",
            "regulation_mode",
            "regulating",
            "low_tap",
            "tap",
        ],
        data=[(pst, 2, "CURRENT_LIMITER", False, -30, 0)],
    )
    steps_df = pd.DataFrame.from_records(
        index="id",
        columns=["id", "b", "g", "r", "x", "rho", "alpha"],
        data=[(pst, 0, 0, 0.1, 1, 1, 2 * tap) for tap in range(-30, 31)],
    )
    net.create_phase_tap_changers(ptc_df, steps_df)


def powsybl_case30_with_psts() -> pypowsybl.network.Network:
    """Create a Powsybl IEEE 30 bus grid with phase-shifting transformers.

    Returns
    -------
    pypowsybl.network.Network
        The Powsybl IEEE 30 bus network with phase-shifting transformers.
    """
    net = pypowsybl.network.create_ieee30()
    add_phaseshift_transformer_to_line_powsybl(net, "L8-28-1")
    add_phaseshift_transformer_to_line_powsybl(net, "L6-28-1")
    return net


def powsybl_extended_case57() -> pypowsybl.network.Network:
    """Create an extended version of the Powsybl IEEE 57 bus grid with additional elements.


    Returns
    -------
    pypowsybl.network.Network
        The extended Powsybl IEEE 57 bus network.
    """
    net = pypowsybl.network.create_ieee57()

    # Update max power of the generators to be able to have a distributed slack (it does not work if the max power is
    # deemed inplausible). This could also be achieved by setting the provider parameter "plausibleActivePowerLimit":"20000",
    # but since we do not want to add this limit to the function updating the values is cleaner.
    gens = net.get_generators()
    gens["max_p"] = 4999
    # Set some gens to voltage regulator False, so that changes in target_q actually do something
    gens.loc[gens.index[2:4], "voltage_regulator_on"] = False
    net.update_generators(gens[["max_p", "voltage_regulator_on"]])

    # Add fake PST
    net.create_buses(id="PSTBus", voltage_level_id="VL6")
    net.create_lines(
        id="PSTLine",
        voltage_level1_id="VL6",
        bus1_id="PSTBus",
        voltage_level2_id="VL17",
        bus2_id="B17",
        r=0.1 / 100,
        x=0.1 / 100,
        g1=0,
        b1=0,
        g2=0,
        b2=0,
    )
    net.create_2_windings_transformers(
        id="PST",
        voltage_level1_id="VL6",
        bus1_id="PSTBus",
        voltage_level2_id="VL6",
        bus2_id="B6",
        rated_u1=1,
        rated_u2=1,
        rated_s=np.nan,
        r=0,
        x=0.1 / 100,
        g=0,
        b=0,
    )
    ptc_df = pd.DataFrame.from_records(
        index="id",
        columns=["id", "regulation_mode", "regulating", "low_tap", "tap"],
        data=[("PST", "CURRENT_LIMITER", False, 0, 1)],
    )
    steps_df = pd.DataFrame.from_records(
        index="id",
        columns=["id", "b", "g", "r", "x", "rho", "alpha"],
        data=[
            ("PST", 0, 0, 0, 0, 1, 0),
            ("PST", 0, 0, 0, 0, 1, 8),
            ("PST", 0, 0, 0, 0, 1, 16),
        ],
    )
    net.create_phase_tap_changers(ptc_df=ptc_df, steps_df=steps_df)
    pypowsybl.loadflow.run_ac(net)

    # Add artificial operational limits
    lines = net.get_lines()
    limits = pd.DataFrame(
        data={
            "element_id": lines.index,
            "side": ["ONE"] * len(lines),
            "name": ["permanent_limit"] * len(lines),
            "type": ["CURRENT"] * len(lines),
            "value": lines["i1"].values,
            "acceptable_duration": [-1] * len(lines),
        }
    )
    limits.index = limits["element_id"]

    limits2 = pd.DataFrame(
        data={
            "element_id": lines.index,
            "side": ["ONE"] * len(lines),
            "name": ["N-1"] * len(lines),
            "type": ["CURRENT"] * len(lines),
            "value": lines["i1"].values * 2,
            "acceptable_duration": [2] * len(lines),
        }
    )
    limits2.index = limits2["element_id"]
    limits = pd.concat([limits, limits2])
    net.create_operational_limits(limits)

    limits = pd.DataFrame(
        data={
            "element_id": net.get_2_windings_transformers().index,
            "side": ["ONE"] * len(net.get_2_windings_transformers()),
            "name": ["permanent_limit"] * len(net.get_2_windings_transformers()),
            "type": ["CURRENT"] * len(net.get_2_windings_transformers()),
            "value": net.get_2_windings_transformers()["i1"].values,
            "acceptable_duration": [-1] * len(net.get_2_windings_transformers()),
        }
    )
    limits.index = limits["element_id"]
    trafo2w = net.get_2_windings_transformers()
    limits2 = pd.DataFrame(
        data={
            "element_id": trafo2w.index,
            "side": ["ONE"] * len(trafo2w),
            "name": ["N-1"] * len(trafo2w),
            "type": ["CURRENT"] * len(trafo2w),
            "value": trafo2w["i1"].values * 2,
            "acceptable_duration": [2] * len(trafo2w),
        }
    )
    limits2.index = limits2["element_id"]
    limits = pd.concat([limits, limits2])
    net.create_operational_limits(limits)
    return net


def basic_node_breaker_network_powsybl() -> Network:
    """Create a basic node breaker network with 5 substations, 5 voltage levels, and 10 buses.

    Returns
    -------
    pypowsybl.network.Network
        The created Powsybl network.
    """
    net = pypowsybl.network.create_empty()

    n_subs = 5
    n_vls = 5
    # substation_id : number of buses
    n_buses = {1: 3, 2: 3, 3: 2, 4: 2, 5: 1}

    stations = pd.DataFrame.from_records(
        index="id", data=[{"id": f"S{i + 1}", "country": "BE", "name": f"Station{i + 1}"} for i in range(n_subs)]
    )
    voltage_levels = pd.DataFrame.from_records(
        index="id",
        data=[
            {
                "substation_id": f"S{i + 1}",
                "id": f"VL{i + 1}",
                "topology_kind": "NODE_BREAKER",
                "nominal_v": 225,
                "name": f"VLevel{i + 1}",
            }
            for i in range(n_vls)
        ],
    )
    busbars = pd.DataFrame.from_records(
        index="id",
        data=[
            {"voltage_level_id": f"VL{sub_id}", "id": f"BBS{sub_id}_{bus_id}", "node": bus_id - 1, "name": f"bus{bus_id}"}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )
    busbar_section_position = pd.DataFrame.from_records(
        index="id",
        data=[
            {"id": f"BBS{sub_id}_{bus_id}", "section_index": 1, "busbar_index": bus_id}
            for sub_id, num_buses in n_buses.items()
            for bus_id in range(1, num_buses + 1)
        ],
    )

    net.create_substations(stations)
    net.create_voltage_levels(voltage_levels)
    net.create_busbar_sections(busbars)
    net.create_extensions("busbarSectionPosition", busbar_section_position)

    lines = pd.DataFrame.from_records(
        data=[
            {"bus_or_busbar_section_id_1": "BBS1_1", "bus_or_busbar_section_id_2": "BBS2_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS2_2"},
            {"bus_or_busbar_section_id_1": "BBS1_3", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS1_3", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS1_2", "bus_or_busbar_section_id_2": "BBS4_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS3_1"},
            {"bus_or_busbar_section_id_1": "BBS2_2", "bus_or_busbar_section_id_2": "BBS3_2"},
            {"bus_or_busbar_section_id_1": "BBS2_1", "bus_or_busbar_section_id_2": "BBS4_1"},
            {"bus_or_busbar_section_id_1": "BBS3_1", "bus_or_busbar_section_id_2": "BBS5_1"},
        ]
    )
    lines["r"] = 0.1
    lines["x"] = 10
    lines["g1"] = 0
    lines["b1"] = 0
    lines["g2"] = 0
    lines["b2"] = 0
    lines["position_order_1"] = 1
    lines["position_order_2"] = 1
    for i, _ in lines.iterrows():
        lines.loc[i, "id"] = f"L{i + 1}"
    lines = lines.set_index("id")
    pypowsybl.network.create_line_bays(net, lines)
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS1_1", "BBS1_2"], bus_or_busbar_section_id_2=["BBS1_2", "BBS1_3"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_1"], bus_or_busbar_section_id_2=["BBS2_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS2_2"], bus_or_busbar_section_id_2=["BBS2_3"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS3_1"], bus_or_busbar_section_id_2=["BBS3_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["BBS4_1"], bus_or_busbar_section_id_2=["BBS4_2"]
    )
    pypowsybl.network.create_load_bay(net, id="load1", bus_or_busbar_section_id="BBS2_1", p0=100, q0=10, position_order=1)
    pypowsybl.network.create_load_bay(net, id="load2", bus_or_busbar_section_id="BBS3_2", p0=100, q0=10, position_order=2)
    pypowsybl.network.create_generator_bay(
        net,
        id="generator1",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=50,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS1_1",
        position_order=1,
    )
    pypowsybl.network.create_generator_bay(
        net,
        id="generator2",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=50,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS1_2",
        position_order=1,
    )
    pypowsybl.network.create_generator_bay(
        net,
        id="generator3",
        max_p=1000,
        min_p=0,
        voltage_regulator_on=True,
        target_p=100,
        target_q=10,
        target_v=225,
        bus_or_busbar_section_id="BBS5_1",
        position_order=2,
    )
    limits = pd.DataFrame.from_records(
        data=[
            {
                "element_id": "L1",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L2",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L3",
                "value": 90,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
        ],
        index="element_id",
    )
    net.create_operational_limits(limits)
    return net


def powsybl_case9241() -> pypowsybl.network.Network:
    """Load the Powsybl case9241 grid.

    Returns
    -------
    pypowsybl.network.Network
        The loaded Powsybl pegase case9241 network.
    """
    pandapower_net = pandapower.networks.case9241pegase()
    net = load_pandapower_net_for_powsybl(pandapower_net, check_trafo_resistance=False)

    return net


def create_busbar_b_in_ieee(net: pypowsybl.network.Network) -> None:
    """Create busbar B in the IEEE grid

    This is needed to create a busbar B for each bus in the IEEE grid. The busbar A is already there, so we just need to
    create the busbar B and connect it to the busbar A with a coupler.

    The bus B will have an id similar to bus A but with '_b' suffixed.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to create the busbar Bs in, will be modified in place
    """
    for index, bus in net.get_bus_breaker_view_buses().iterrows():
        net.create_buses(
            id=index + "_b",
            voltage_level_id=bus.voltage_level_id,
            name=bus.name + "_b",
        )
        net.create_switches(
            id="SWITCH-" + index,
            bus1_id=index,
            bus2_id=index + "_b",
            voltage_level_id=bus.voltage_level_id,
            kind="BREAKER",
            open=False,
            retained=True,
        )


def extract_station_info_powsybl(net: Network, base_folder: Path) -> None:
    stations = get_stations_bus_breaker(net)
    target = base_folder / PREPROCESSING_PATHS["asset_topology_file_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    save_asset_topology(
        target,
        Topology(
            stations=stations,
            topology_id="extracted_topology",
            timestamp=datetime.datetime.now(),
        ),
    )


def case14_matching_asset_topo_powsybl(folder: Path) -> None:
    net = pypowsybl.network.create_ieee14()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)

    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_path)

    # create asset topology
    extract_station_info_powsybl(net, folder)

    # create masks
    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)
    rel_sub_mask = np.ones(len(net.get_buses()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)
    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)
    trafo_mask = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)
    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)


# ruff: noqa: PLR0915
# sonar: noqa: S3776
def create_complex_grid_battery_hvdc_svc_3w_trafo() -> Network:
    """Create a complex grid with batteries, HVDC, SVC, and 3-winding transformers using Powsybl.

    This grid includes various components to test different functionalities. It is not aimed to be a realistic
    representation of an actual power grid but rather a comprehensive test case. The Basecase should converge
    in about 10 iterations with a tolerance of 1e-6.

    TODO: add sensable operational limits, maybe some ratio/phase tap changers, etc.
    Ideally it should have some overloads that can be solved by ToOp

    Returns
    -------
    Network
        The created complex grid network.
    """
    n = pypowsybl.network.create_empty("TESTGRID_NODE_BREAKER_HVDC_BAT_SVC_3W_TRAFO")

    # ---------------------------------------------------------------------
    # 1) Substations
    # ---------------------------------------------------------------------
    substations_df = pd.DataFrame(
        [
            {"id": "S_3W", "name": "S_3W", "tso": "TSO", "country": "BE"},
            {"id": "S_2W_MV_LV", "name": "S_2W_MV_LV", "tso": "TSO", "country": "BE"},
            {"id": "S_LV_load", "name": "S_LV_load", "tso": "TSO", "country": "BE"},
            {"id": "S_MV_load", "name": "S_MV_load", "tso": "TSO", "country": "BE"},
            {"id": "S_MV_svc", "name": "S_MV_svc", "tso": "TSO", "country": "BE"},
            {"id": "S_MV", "name": "S_MV", "tso": "TSO", "country": "BE"},
            {"id": "S_2W_MV_HV", "name": "S_2W_MV_HV", "tso": "TSO", "country": "BE"},
            {"id": "S_HV_gen", "name": "S_2W_HV_gen", "tso": "TSO", "country": "BE"},
            {"id": "S_HV_vsc", "name": "S_HV_vsc", "tso": "TSO", "country": "BE"},
            {"id": "S_DE_1", "name": "S_DE_1", "tso": "TSO", "country": "DE"},
            {"id": "S_DE_2", "name": "S_DE_2", "tso": "TSO", "country": "DE"},
        ]
    ).set_index("id")
    n.create_substations(df=substations_df)

    # ---------------------------------------------------------------------
    # 2) Voltage levels
    # ---------------------------------------------------------------------
    vls_df = pd.DataFrame(
        [
            {
                "id": "VL_3W_HV",
                "name": "VL_3W_HV",
                "substation_id": "S_3W",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_3W_MV",
                "name": "VL_3W_MV",
                "substation_id": "S_3W",
                "nominal_v": 110.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_3W_LV",
                "name": "VL_3W_LV",
                "substation_id": "S_3W",
                "nominal_v": 63.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_2W_MV_LV_MV",
                "name": "VL_2W_MV_LV_MV",
                "substation_id": "S_2W_MV_LV",
                "nominal_v": 110.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_2W_MV_LV_LV",
                "name": "VL_2W_MV_LV_LV",
                "substation_id": "S_2W_MV_LV",
                "nominal_v": 63.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_LV_load",
                "name": "VL_LV_load",
                "substation_id": "S_LV_load",
                "nominal_v": 63.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_MV_load",
                "name": "VL_MV_load",
                "substation_id": "S_MV_load",
                "nominal_v": 110.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_MV_svc",
                "name": "VL_MV_svc",
                "substation_id": "S_MV_svc",
                "nominal_v": 110.0,
                "topology_kind": "NODE_BREAKER",
            },
            {"id": "VL_MV", "name": "VL_MV", "substation_id": "S_MV", "nominal_v": 110.0, "topology_kind": "NODE_BREAKER"},
            {
                "id": "VL_2W_MV_HV_MV",
                "name": "VL_2W_MV_HV_MV",
                "substation_id": "S_2W_MV_HV",
                "nominal_v": 110.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_2W_MV_HV_HV",
                "name": "VL_2W_MV_HV_HV",
                "substation_id": "S_2W_MV_HV",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_HV_gen",
                "name": "VL_HV_gen",
                "substation_id": "S_HV_gen",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_HV_vsc",
                "name": "VL_HV_vsc",
                "substation_id": "S_HV_vsc",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_DE_1",
                "name": "VL_DE_1",
                "substation_id": "S_DE_1",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
            {
                "id": "VL_DE_2",
                "name": "VL_DE_2",
                "substation_id": "S_DE_2",
                "nominal_v": 380.0,
                "topology_kind": "NODE_BREAKER",
            },
        ]
    ).set_index("id")
    n.create_voltage_levels(df=vls_df)

    # ---------------------------------------------------------------------
    #  Busbar layouts
    # ---------------------------------------------------------------------
    kwargs_no_layout = {"aligned_buses_or_busbar_count": 1, "section_count": 1, "switch_kinds": ""}
    kwargs_basic_layout = {"aligned_buses_or_busbar_count": 1, "section_count": 2, "switch_kinds": "BREAKER"}
    kwargs_two_busbar_layout = {"aligned_buses_or_busbar_count": 2, "section_count": 1, "switch_kinds": ""}
    kwargs_four_busbar_layout = {"aligned_buses_or_busbar_count": 2, "section_count": 2, "switch_kinds": "BREAKER"}
    kwargs_four_busbar_disconnector_layout = {
        "aligned_buses_or_busbar_count": 2,
        "section_count": 2,
        "switch_kinds": "DISCONNECTOR",
    }

    no_layout_list = ["VL_LV_load", "VL_DE_1", "VL_DE_2"]
    basic_layout_list = ["VL_2W_MV_LV_LV", "VL_3W_LV"]
    two_busbar_layout_list = ["VL_3W_MV", "VL_2W_MV_LV_MV", "VL_MV_load", "VL_MV_svc", "VL_2W_MV_HV_MV", "VL_HV_gen"]
    four_busbar_layout_list = ["VL_3W_HV", "VL_2W_MV_HV_HV", "VL_HV_vsc"]
    four_busbar_disconnector_layout_list = ["VL_MV"]

    def _create_busbars(voltage_list: list, kwargs: dict) -> None:
        for vl in voltage_list:
            pypowsybl.network.create_voltage_level_topology(network=n, id=vl, **kwargs)
            if kwargs["aligned_buses_or_busbar_count"] == 2:
                pypowsybl.network.create_coupling_device(
                    n,
                    bus_or_busbar_section_id_1=[f"{vl}_1_1"],
                    bus_or_busbar_section_id_2=[f"{vl}_2_1"],
                )

    _create_busbars(no_layout_list, kwargs_no_layout)
    _create_busbars(basic_layout_list, kwargs_basic_layout)
    _create_busbars(two_busbar_layout_list, kwargs_two_busbar_layout)
    _create_busbars(four_busbar_layout_list, kwargs_four_busbar_layout)
    _create_busbars(four_busbar_disconnector_layout_list, kwargs_four_busbar_disconnector_layout)

    # refine busbar layouts for specific voltage levels
    pypowsybl.network.create_coupling_device(
        n,
        bus_or_busbar_section_id_1=["VL_2W_MV_HV_HV_1_2"],
        bus_or_busbar_section_id_2=["VL_2W_MV_HV_HV_2_2"],
    )
    # FIX ME: currently not working due to an importing issue in the simplyfied station function
    # pypowsybl.network.create_coupling_device(
    #     n,
    #     bus_or_busbar_section_id_1=["VL_MV_1_2"],
    #     bus_or_busbar_section_id_2=["VL_MV_2_2"],
    # )
    # pypowsybl.network.create_coupling_device(
    #     n,
    #     bus_or_busbar_section_id_1=["VL_MV_1_1"],
    #     bus_or_busbar_section_id_2=["VL_MV_1_2"],
    # )
    # n.open_switch("VL_MV_DISCONNECTOR_0_2")

    # ---------------------------------------------------------------------
    # 3) AC lines
    # ---------------------------------------------------------------------
    # LV (63 kV)
    lv_short = {"r": 3.5, "x": 9.0, "g1": 0.0, "b1": 7.5586e-06, "g2": 0.0, "b2": 7.5586e-06}
    lv_long = {"r": 5.0, "x": 15.0, "g1": 0.0, "b1": 2.5195e-05, "g2": 0.0, "b2": 2.5195e-05}

    # MV (110 kV)
    mv_short = {"r": 1.8, "x": 5.1, "g1": 0.0, "b1": 3.3058e-06, "g2": 0.0, "b2": 3.3058e-06}
    mv_long = {"r": 4.8, "x": 20.5, "g1": 0.0, "b1": 9.9174e-06, "g2": 0.0, "b2": 9.9174e-06}

    # HV (380 kV)
    hv_short = {"r": 0.8, "x": 8.8, "g1": 0.0, "b1": 3.4626e-07, "g2": 0.0, "b2": 3.4626e-07}
    hv_long = {"r": 1.0, "x": 15.0, "g1": 0.0, "b1": 1.1080e-06, "g2": 0.0, "b2": 1.1080e-06}

    # LV lines
    lv_lines = pd.DataFrame(
        [
            {"bus_or_busbar_section_id_1": "VL_3W_LV_1_1", "bus_or_busbar_section_id_2": "VL_LV_load_1_1", **lv_short},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_LV_LV_1_1", "bus_or_busbar_section_id_2": "VL_LV_load_1_1", **lv_long},
        ]
    )
    lv_lines["position_order_1"] = 1
    lv_lines["position_order_2"] = 1

    # MV lines (first 5 short, rest long based on your list)
    mv_lines = pd.DataFrame(
        [
            {"bus_or_busbar_section_id_1": "VL_MV_svc_1_1", "bus_or_busbar_section_id_2": "VL_3W_MV_1_1", **mv_short},
            {"bus_or_busbar_section_id_1": "VL_MV_svc_1_1", "bus_or_busbar_section_id_2": "VL_2W_MV_HV_MV_1_1", **mv_short},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_LV_MV_1_1", "bus_or_busbar_section_id_2": "VL_3W_MV_1_1", **mv_short},
            {"bus_or_busbar_section_id_1": "VL_MV_load_1_1", "bus_or_busbar_section_id_2": "VL_MV_2_2", **mv_short},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_HV_MV_1_1", "bus_or_busbar_section_id_2": "VL_MV_2_1", **mv_short},
            {"bus_or_busbar_section_id_1": "VL_MV_load_1_1", "bus_or_busbar_section_id_2": "VL_2W_MV_LV_MV_1_1", **mv_long},
            {"bus_or_busbar_section_id_1": "VL_MV_1_1", "bus_or_busbar_section_id_2": "VL_3W_MV_1_1", **mv_long},
            {"bus_or_busbar_section_id_1": "VL_MV_1_2", "bus_or_busbar_section_id_2": "VL_3W_MV_1_1", **mv_long},
            {"bus_or_busbar_section_id_1": "VL_MV_svc_1_1", "bus_or_busbar_section_id_2": "VL_2W_MV_LV_MV_1_1", **mv_long},
            {"bus_or_busbar_section_id_1": "VL_MV_svc_1_1", "bus_or_busbar_section_id_2": "VL_MV_2_1", **mv_long},
            {"bus_or_busbar_section_id_1": "VL_MV_load_1_1", "bus_or_busbar_section_id_2": "VL_2W_MV_HV_MV_1_1", **mv_long},
        ]
    )
    mv_lines["position_order_1"] = 1
    mv_lines["position_order_2"] = 1

    # HV lines (first 4 short, last 4 long)
    hv_lines = pd.DataFrame(
        [
            {"bus_or_busbar_section_id_1": "VL_3W_HV_1_1", "bus_or_busbar_section_id_2": "VL_HV_vsc_1_1", **hv_short},
            {"bus_or_busbar_section_id_1": "VL_3W_HV_2_1", "bus_or_busbar_section_id_2": "VL_HV_vsc_2_1", **hv_short},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_HV_HV_1_2", "bus_or_busbar_section_id_2": "VL_HV_gen_1_1", **hv_short},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_HV_HV_2_2", "bus_or_busbar_section_id_2": "VL_HV_gen_2_1", **hv_short},
            {"bus_or_busbar_section_id_1": "VL_3W_HV_1_1", "bus_or_busbar_section_id_2": "VL_HV_gen_1_1", **hv_long},
            {"bus_or_busbar_section_id_1": "VL_3W_HV_2_1", "bus_or_busbar_section_id_2": "VL_HV_gen_2_1", **hv_long},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_HV_HV_1_1", "bus_or_busbar_section_id_2": "VL_HV_vsc_1_1", **hv_long},
            {"bus_or_busbar_section_id_1": "VL_2W_MV_HV_HV_2_1", "bus_or_busbar_section_id_2": "VL_HV_vsc_2_1", **hv_long},
        ]
    )
    hv_lines["position_order_1"] = 1
    hv_lines["position_order_2"] = 1

    lines = pd.concat([lv_lines, mv_lines, hv_lines], ignore_index=True)
    lines["id"] = [f"L{i + 1}" for i in range(len(lines))]
    lines = lines.set_index("id")
    pypowsybl.network.create_line_bays(n, df=lines)

    # ---------------------------------------------------------------------
    # 4) Transformers
    # ---------------------------------------------------------------------
    # 2W: 110/63 kV
    pypowsybl.network.create_2_windings_transformer_bays(
        n,
        id="2W_MV_LV",
        b=0.0,
        g=0.0,
        r=0.005,
        x=0.15,
        rated_u1=110.0,
        rated_u2=63.0,
        bus_or_busbar_section_id_1="VL_2W_MV_LV_MV_1_1",
        position_order_1=35,
        direction_1="BOTTOM",
        bus_or_busbar_section_id_2="VL_2W_MV_LV_LV_1_1",
        position_order_2=5,
        direction_2="TOP",
    )

    # 2W: 380/110 kV (two parallel transformers in S_2W_MV_HV)
    pypowsybl.network.create_2_windings_transformer_bays(
        n,
        id="2W_MV_HV_1",
        b=0.0,
        g=0.0,
        r=0.004,
        x=0.12,
        rated_u1=380.0,
        rated_u2=110.0,
        bus_or_busbar_section_id_1="VL_2W_MV_HV_HV_1_2",
        position_order_1=35,
        direction_1="BOTTOM",
        bus_or_busbar_section_id_2="VL_2W_MV_HV_MV_1_1",
        position_order_2=5,
        direction_2="TOP",
    )
    pypowsybl.network.create_2_windings_transformer_bays(
        n,
        id="2W_MV_HV_2",
        b=0.0,
        g=0.0,
        r=0.004,
        x=0.12,
        rated_u1=380.0,
        rated_u2=110.0,
        bus_or_busbar_section_id_1="VL_2W_MV_HV_HV_2_1",
        position_order_1=35,
        direction_1="BOTTOM",
        bus_or_busbar_section_id_2="VL_2W_MV_HV_MV_2_1",
        position_order_2=5,
        direction_2="TOP",
    )

    # 2W inside S_3W to help split flows: 380/110
    pypowsybl.network.create_2_windings_transformer_bays(
        n,
        id="2W_3W_MV_HV",
        b=0.0,
        g=0.0,
        r=0.004,
        x=0.12,
        rated_u1=380.0,
        rated_u2=110.0,
        bus_or_busbar_section_id_1="VL_3W_HV_1_1",
        position_order_1=35,
        direction_1="BOTTOM",
        bus_or_busbar_section_id_2="VL_3W_MV_1_1",
        position_order_2=5,
        direction_2="TOP",
    )

    # 3W: 380/110/63 kV - direct node connections + bay switches around it
    three_w_df = pd.DataFrame(
        [
            {
                "id": "3W",
                "name": "3W 380/110/63",
                "voltage_level1_id": "VL_3W_HV",
                "voltage_level2_id": "VL_3W_MV",
                "voltage_level3_id": "VL_3W_LV",
                "node1": 30,
                "node2": 30,
                "node3": 30,
                "rated_u1": 380.0,
                "rated_u2": 110.0,
                "rated_u3": 63.0,
                "r1": 0.005,
                "x1": 0.15,
                "g1": 0.0,
                "b1": 0.0,
                "r2": 0.005,
                "x2": 0.15,
                "g2": 0.0,
                "b2": 0.0,
                "r3": 0.006,
                "x3": 0.18,
                "g3": 0.0,
                "b3": 0.0,
            }
        ]
    ).set_index("id")
    n.create_3_windings_transformers(three_w_df)

    # Bay switches
    n.create_switches(id="BREAKER_3W_HV", voltage_level_id="VL_3W_HV", node1=30, node2=31, kind="BREAKER", open=False)
    n.create_switches(
        id="DISCONNECTOR_3W_HV_1", voltage_level_id="VL_3W_HV", node1=0, node2=31, kind="DISCONNECTOR", open=False
    )
    n.create_switches(
        id="DISCONNECTOR_3W_HV_2", voltage_level_id="VL_3W_HV", node1=1, node2=31, kind="DISCONNECTOR", open=False
    )
    n.create_switches(id="BREAKER_3W_MV", voltage_level_id="VL_3W_MV", node1=30, node2=31, kind="BREAKER", open=False)
    n.create_switches(
        id="DISCONNECTOR_3W_MV_1", voltage_level_id="VL_3W_MV", node1=0, node2=31, kind="DISCONNECTOR", open=False
    )
    n.create_switches(
        id="DISCONNECTOR_3W_MV_2", voltage_level_id="VL_3W_MV", node1=1, node2=31, kind="DISCONNECTOR", open=False
    )
    n.create_switches(id="BREAKER_3W_LV", voltage_level_id="VL_3W_LV", node1=30, node2=31, kind="BREAKER", open=False)
    n.create_switches(
        id="DISCONNECTOR_3W_LV_1", voltage_level_id="VL_3W_LV", node1=0, node2=31, kind="DISCONNECTOR", open=False
    )

    # ---------------------------------------------------------------------
    # 5) HVDC
    # ---------------------------------------------------------------------
    # LCC converter stations
    lcc_df = pd.DataFrame(
        [
            {
                "id": "LCC1",
                "name": "LCC station A",
                "power_factor": 0.98,
                "loss_factor": 1.0,
                "bus_or_busbar_section_id": "VL_3W_HV_1_1",
                "position_order": 45,
            },
            {
                "id": "LCC2",
                "name": "LCC station B",
                "power_factor": 0.98,
                "loss_factor": 1.0,
                "bus_or_busbar_section_id": "VL_2W_MV_HV_HV_1_2",
                "position_order": 45,
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_lcc_converter_station_bay(n, df=lcc_df)

    n.create_hvdc_lines(
        id="HVDC_LCC",
        converter_station1_id="LCC1",
        converter_station2_id="LCC2",
        r=1.0,
        nominal_v=380.0,
        converters_mode="SIDE_1_RECTIFIER_SIDE_2_INVERTER",
        max_p=300.0,
        target_p=75.0,
    )

    # VSC converter stations (one on S_3W HV, one on HV_gen)
    vsc_df = pd.DataFrame(
        [
            {
                "id": "VSC_A",
                "name": "VSC A (3W-HV)",
                "loss_factor": 1.0,
                "voltage_regulator_on": True,
                "target_v": 380.0,
                "target_q": 0.0,
                "bus_or_busbar_section_id": "VL_HV_vsc_1_1",
                "position_order": 60,
                "direction": "TOP",
            },
            {
                "id": "VSC_B",
                "name": "VSC B (HV-gen)",
                "loss_factor": 1.0,
                "voltage_regulator_on": True,
                "target_v": 380.0,
                "target_q": 0.0,
                "bus_or_busbar_section_id": "VL_HV_gen_2_1",
                "position_order": 60,
                "direction": "TOP",
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_vsc_converter_station_bay(n, df=vsc_df)

    n.create_hvdc_lines(
        id="HVDC_VSC",
        converter_station1_id="VSC_A",
        converter_station2_id="VSC_B",
        r=1.0,
        nominal_v=380.0,
        converters_mode="SIDE_1_RECTIFIER_SIDE_2_INVERTER",
        max_p=300.0,
        target_p=50.0,  # small transfer to avoid stressing balance
    )

    # ---------------------------------------------------------------------
    # 6) assets: battery, SVC, shunts/reactor, gens, loads, dangling line
    # ---------------------------------------------------------------------
    # Generators (set one big HV gen as main source)
    gens_df = pd.DataFrame(
        [
            {
                "id": "GEN_HV",
                "name": "HV main generator",
                "energy_source": "THERMAL",
                "min_p": 0.0,
                "max_p": 1200.0,
                "target_p": 700.0,
                "voltage_regulator_on": True,
                "target_v": 380.0,
                "bus_or_busbar_section_id": "VL_HV_gen_1_1",
                "position_order": 10,
                "direction": "BOTTOM",
            },
            {
                "id": "GEN_MV",
                "name": "MV local generator",
                "energy_source": "THERMAL",
                "min_p": 0.0,
                "max_p": 80.0,
                "target_p": 30.0,
                "voltage_regulator_on": True,
                "target_v": 110.0,
                "bus_or_busbar_section_id": "VL_MV_svc_1_1",
                "position_order": 10,
                "direction": "BOTTOM",
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_generator_bay(n, df=gens_df)

    # Loads (balanced across VLs; your style)
    loads_df = pd.DataFrame(
        [
            {
                "id": "load_HV_gen",
                "name": "HV interconnection load",
                "p0": 120.0,
                "q0": 40.0,
                "bus_or_busbar_section_id": "VL_HV_gen_2_1",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_HV_vsc",
                "name": "HV local load",
                "p0": 350.0,
                "q0": 120.0,
                "bus_or_busbar_section_id": "VL_HV_vsc_1_1",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_MV",
                "name": "MV interconnection load",
                "p0": 80.0,
                "q0": 20.0,
                "bus_or_busbar_section_id": "VL_MV_1_2",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_MV_load",
                "name": "MV local load",
                "p0": 150.0,
                "q0": 60.0,
                "bus_or_busbar_section_id": "VL_MV_load_1_1",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_VL_MV_svc",
                "name": "LV local load 2",
                "p0": 200.0,
                "q0": 200.0,
                "bus_or_busbar_section_id": "VL_MV_svc_1_1",
                "position_order": 30,
                "direction": "BOTTOM",
            },
            {
                "id": "load_2W_MV_LV_LV",
                "name": "LV interconnection load",
                "p0": 30.0,
                "q0": 10.0,
                "bus_or_busbar_section_id": "VL_2W_MV_LV_LV_1_1",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_LV_load",
                "name": "LV local load 1",
                "p0": 90.0,
                "q0": 30.0,
                "bus_or_busbar_section_id": "VL_LV_load_1_1",
                "position_order": 20,
                "direction": "BOTTOM",
            },
            {
                "id": "load_3W_LV",
                "name": "LV local load 2",
                "p0": 25.0,
                "q0": 8.0,
                "bus_or_busbar_section_id": "VL_3W_LV_1_1",
                "position_order": 30,
                "direction": "BOTTOM",
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_load_bay(n, df=loads_df)

    # Batteries (MV + LV)
    bat_df = pd.DataFrame(
        [
            {
                "id": "BAT_MV",
                "name": "MV battery",
                "min_p": -60.0,
                "max_p": 60.0,
                "bus_or_busbar_section_id": "VL_MV_2_1",
                "position_order": 30,
                "direction": "TOP",
                "target_p": -20.0,
                "target_q": 0.0,
            },
            {
                "id": "BAT_LV",
                "name": "LV battery",
                "min_p": -60.0,
                "max_p": 60.0,
                "bus_or_busbar_section_id": "VL_LV_load_1_1",
                "position_order": 30,
                "direction": "TOP",
                "target_p": 30.0,
                "target_q": 0.0,
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_battery_bay(n, df=bat_df)

    # Shunt capacitor and reactor (inductor) - keep one of each on HV side
    # --- SHUNT BAY DEFINITIONS ---
    shunt_df = pd.DataFrame(
        [
            {
                "id": "SHUNT_HV_CAP",
                "model_type": "LINEAR",
                "section_count": 1,
                "target_v": 380.0,
                "target_deadband": 2.0,
                "bus_or_busbar_section_id": "VL_HV_gen_1_1",
                "position_order": 40,
            },
            {
                "id": "SHUNT_MV_svc",
                "model_type": "LINEAR",
                "section_count": 1,
                "target_v": 110.0,
                "target_deadband": 2.0,
                "bus_or_busbar_section_id": "VL_MV_svc_1_1",
                "position_order": 50,
            },
            {
                "id": "SHUNT_MV",
                "model_type": "LINEAR",
                "section_count": 1,
                "target_v": 110.0,
                "target_deadband": 2.0,
                "bus_or_busbar_section_id": "VL_MV_1_1",
                "position_order": 40,
            },
        ]
    ).set_index("id")

    # --- LINEAR MODEL DEFINITIONS ---
    linear_model_df = pd.DataFrame(
        [
            {"id": "SHUNT_HV_CAP", "g_per_section": 0.0, "b_per_section": 0.0020, "max_section_count": 1},
            {"id": "SHUNT_MV_svc", "g_per_section": 0.0, "b_per_section": 0.0020, "max_section_count": 1},
            {"id": "SHUNT_MV", "g_per_section": 0.0, "b_per_section": 0.0012, "max_section_count": 1},
        ]
    ).set_index("id")

    # --- CREATE SHUNTS ---
    pypowsybl.network.create_shunt_compensator_bay(
        n,
        shunt_df=shunt_df,
        linear_model_df=linear_model_df,
    )

    # STATCOM (SVC) - IMPORTANT: regulating=True
    svc_df = pd.DataFrame(
        [
            {
                "id": "STATCOM_HV",
                "name": "HV STATCOM",
                "b_min": -0.01,
                "b_max": 0.01,
                "regulation_mode": "VOLTAGE",
                "target_v": 110.0,
                "regulating": True,
                "bus_or_busbar_section_id": "VL_MV_svc_1_1",
                "position_order": 55,
                "direction": "TOP",
            },
        ]
    ).set_index("id")
    pypowsybl.network.create_static_var_compensator_bay(n, df=svc_df)

    dangling_df = pd.DataFrame(
        [
            {
                "id": "Dangling_inbound",
                "name": "Dangling inbound",
                "p0": -300,
                "q0": -100,
                "r": hv_long["r"],
                "x": hv_long["x"],
                "g": hv_long["g1"],
                "b": hv_long["b1"],
                "bus_or_busbar_section_id": "VL_2W_MV_HV_HV_1_1",
                "position_order": 60,
                "direction": "BOTTOM",
            },
            {
                "id": "Dangling_outbound",
                "name": "Dangling outbound",
                "p0": 300,
                "q0": 100,
                "r": hv_long["r"],
                "x": hv_long["x"],
                "g": hv_long["g1"],
                "b": hv_long["b1"],
                "bus_or_busbar_section_id": "VL_3W_HV_1_1",
                "position_order": 60,
                "direction": "TOP",
            },
        ]
    ).set_index("id")

    pypowsybl.network.create_dangling_line_bay(network=n, df=dangling_df)

    # line limits
    limits = pd.DataFrame.from_records(
        data=[
            {
                "element_id": "L14",
                "value": 200,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L15",
                "value": 200,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L16",
                "value": 400,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
            {
                "element_id": "L7",
                "value": 400,
                "side": "ONE",
                "name": "permanent",
                "type": "CURRENT",
                "acceptable_duration": -1,
            },
        ],
        index="element_id",
    )

    # Set Slack bus
    slack_voltage_id = "VL_HV_gen"
    slack_bus_id = "VL_HV_gen_0"
    dict_slack = {"voltage_level_id": slack_voltage_id, "bus_id": slack_bus_id}
    pypowsybl.network.Network.create_extensions(n, extension_name="slackTerminal", **dict_slack)

    pypowsybl.loadflow.run_ac(n)
    i1 = abs(n.get_lines()["i1"])
    i1_arr = np.asarray(i1, dtype=float)
    rounded_i1 = (np.ceil(i1_arr / 100) * 100).astype(int)
    limits = pd.Series(rounded_i1, index=i1.index, name="value").reset_index()
    limits.rename(columns={"id": "element_id"}, inplace=True)
    limits.set_index("element_id", inplace=True)
    limits["side"] = "ONE"
    limits["name"] = "permanent"
    limits["type"] = "CURRENT"
    limits["acceptable_duration"] = -1
    n.create_operational_limits(limits)

    # transformer limits
    i1 = abs(n.get_2_windings_transformers()["i1"])
    i1_arr = np.asarray(i1, dtype=float)
    rounded_i1 = (np.ceil(i1_arr / 100) * 100).astype(int)
    limits_tr = pd.Series(rounded_i1, index=n.get_2_windings_transformers().index, name="value").reset_index()
    limits_tr.rename(columns={"id": "element_id"}, inplace=True)
    limits_tr.set_index("element_id", inplace=True)
    limits_tr["side"] = "ONE"
    limits_tr["name"] = "permanent"
    limits_tr["type"] = "CURRENT"
    limits_tr["acceptable_duration"] = -1
    n.create_operational_limits(limits_tr)

    return n


def create_complex_substation_layout_grid() -> Network:
    """Create a simplified complex substation-layout network using bay helpers.

    This version uses `create_voltage_level_topology`, `create_line_bays`,
    `create_2_windings_transformer_bays` and `create_coupling_device` to build
    the same conceptual layout (two aligned busbars with three sections and
    three couplers) in a lot fewer lines of code.
    """
    net = pypowsybl.network.create_empty("TEST_COMPLEX_SUBSTATION_LAYOUT_V2")

    # Substations
    net.create_substations(
        id=["S1", "S2", "S3"], name=["Substation 1", "Substation 2", "Substation 3"], country=["DE", "DE", "DE"]
    )

    # Voltage levels
    # VL1: complex node-breaker with 2 aligned busbars and 3 sections each (replicates B1..B6)
    net.create_voltage_levels(
        id=["VL1", "VL2", "VL3", "VL4", "VL5"],
        substation_id=["S1", "S2", "S1", "S3", "S2"],
        nominal_v=[380.0, 380.0, 220.0, 380.0, 220.0],
        topology_kind=["NODE_BREAKER"] * 5,
        name=["VL1", "VL2", "VL3", "VL4", "VL5"],
    )

    # Create topology layouts for the voltage levels. For VL1 we want two aligned
    # busbars with three sections each (3 sections x 2 aligned -> 6 busbar sections)
    pypowsybl.network.create_voltage_level_topology(
        network=net, id="VL1", aligned_buses_or_busbar_count=3, section_count=2, switch_kinds="DISCONNECTOR"
    )
    pypowsybl.network.create_voltage_level_topology(
        network=net, id="VL2", aligned_buses_or_busbar_count=2, section_count=1, switch_kinds=""
    )

    # For the other voltage levels use simple single-busbar layouts
    for vl in ["VL3", "VL4", "VL5"]:
        pypowsybl.network.create_voltage_level_topology(
            network=net, id=vl, aligned_buses_or_busbar_count=1, section_count=1, switch_kinds=""
        )

    # Create three coupling devices (couplers) between corresponding sections of the
    # two aligned busbars in VL1. This models the three couplers in the original function.
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_1"], bus_or_busbar_section_id_2=["VL1_2_1"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_2"], bus_or_busbar_section_id_2=["VL1_3_2"]
    )
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL1_1_2"], bus_or_busbar_section_id_2=["VL1_2_1"]
    )
    # VL2
    pypowsybl.network.create_coupling_device(
        net, bus_or_busbar_section_id_1=["VL2_1_1"], bus_or_busbar_section_id_2=["VL2_2_1"]
    )

    # Lines: create a small set of line bays connecting VL1 sections to other VLs.
    lines = pd.DataFrame(
        [
            {"bus_or_busbar_section_id_1": "VL1_2_1", "bus_or_busbar_section_id_2": "VL2_1_1", "r": 0.1, "x": 0.2},
            {"bus_or_busbar_section_id_1": "VL1_1_2", "bus_or_busbar_section_id_2": "VL2_1_1", "r": 0.15, "x": 0.25},
            {"bus_or_busbar_section_id_1": "VL1_3_1", "bus_or_busbar_section_id_2": "VL4_1_1", "r": 0.2, "x": 0.3},
            {"bus_or_busbar_section_id_1": "VL1_3_2", "bus_or_busbar_section_id_2": "VL4_1_1", "r": 0.25, "x": 0.35},
        ]
    )
    lines["g1"] = 0.0
    lines["b1"] = 0.0
    lines["g2"] = 0.0
    lines["b2"] = 0.0
    lines["position_order_1"] = 1
    lines["position_order_2"] = 1
    lines["direction_1"] = "TOP"
    lines["direction_2"] = "TOP"
    lines["id"] = [f"L{i + 1}" for i in range(len(lines))]
    lines = lines.set_index("id")
    pypowsybl.network.create_line_bays(net, df=lines)

    # Transformers (use bay helper to create simpler transformer representation)
    pypowsybl.network.create_2_windings_transformer_bays(
        net,
        id="T1",
        b=1e-6,
        g=1e-6,
        r=0.5,
        x=10.0,
        rated_u1=380.0,
        rated_u2=220.0,
        bus_or_busbar_section_id_1="VL1_2_2",
        position_order_1=10,
        direction_1="TOP",
        bus_or_busbar_section_id_2="VL3_1_1",
        position_order_2=5,
        direction_2="TOP",
    )
    pypowsybl.network.create_2_windings_transformer_bays(
        net,
        id="T2",
        b=1e-6,
        g=1e-6,
        r=0.6,
        x=12.0,
        rated_u1=380.0,
        rated_u2=220.0,
        bus_or_busbar_section_id_1="VL1_1_1",
        position_order_1=10,
        direction_1="TOP",
        bus_or_busbar_section_id_2="VL3_1_1",
        position_order_2=5,
        direction_2="TOP",
    )
    pypowsybl.network.create_2_windings_transformer_bays(
        net,
        id="T3",
        b=1e-6,
        g=1e-6,
        r=0.5,
        x=10.0,
        rated_u1=380.0,
        rated_u2=220.0,
        bus_or_busbar_section_id_1="VL2_1_1",
        position_order_1=10,
        direction_1="TOP",
        bus_or_busbar_section_id_2="VL5_1_1",
        position_order_2=5,
        direction_2="TOP",
    )
    pypowsybl.network.create_2_windings_transformer_bays(
        net,
        id="T4",
        b=1e-6,
        g=1e-6,
        r=0.6,
        x=12.0,
        rated_u1=380.0,
        rated_u2=220.0,
        bus_or_busbar_section_id_1="VL2_1_1",
        position_order_1=10,
        direction_1="TOP",
        bus_or_busbar_section_id_2="VL5_1_1",
        position_order_2=5,
        direction_2="TOP",
    )

    # # A couple of generators/loads to make the network usable in tests
    pypowsybl.network.create_generator_bay(
        net,
        id="Gen1",
        max_p=1000.0,
        min_p=0.0,
        target_p=500.0,
        voltage_regulator_on=True,
        target_v=230.0,
        bus_or_busbar_section_id="VL3_1_1",
        position_order=1,
        direction="TOP",
    )
    pypowsybl.network.create_load_bay(
        net, id="Load1", bus_or_busbar_section_id="VL5_1_1", p0=200.0, q0=30.0, position_order=1, direction="TOP"
    )
    pypowsybl.network.create_load_bay(
        net, id="Load2", bus_or_busbar_section_id="VL4_1_1", p0=250.0, q0=20.0, position_order=1, direction="TOP"
    )

    return net
