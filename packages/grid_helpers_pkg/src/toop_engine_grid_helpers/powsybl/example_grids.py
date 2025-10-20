import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import get_stations_bus_breaker
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


def powsybl_texas() -> pypowsybl.network.Network:
    """Load the powsybl Texas grid.

    Returns
    -------
    pypowsybl.network.Network
        The Powsybl Texas network.
    """
    texas_grid_file = Path(__file__).parent.parent / "data" / "texas" / "ACTIVSg2000.mat"
    net = pypowsybl.network.load(str(texas_grid_file))
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
    # Load the Powsybl case9241 grid from the MAT file
    pegase_grid_file = Path(__file__).parent.parent / "data" / "pegase" / "case9241pegase.mat"
    net = pypowsybl.network.load(pegase_grid_file)
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
