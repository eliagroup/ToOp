# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pypowsybl as pp
import os
import pypowsybl.network as pyn

def build_substation_with_2x2_busbars() -> pyn.Network:
    """Create a substation with a 2x2 busbar matrix for demonstration purposes.

    Returns
    -------
    pypowsybl.network.Network
        The Demo network
    """
    n = pyn.create_empty()

    n.create_substations(id='S1', name='Substation-2x2')
    n.create_voltage_levels(
        id='VL1', substation_id='S1', topology_kind='NODE_BREAKER', nominal_v=225.0
    )
    n.create_voltage_levels(
        id='VL2', substation_id='S1', topology_kind='NODE_BREAKER', nominal_v=63.0
    )

    pp.network.create_voltage_level_topology(
        n,
        id='VL1',
        aligned_buses_or_busbar_count=2,
        section_count=2,
        switch_kinds='BREAKER',
    )
    pp.network.create_voltage_level_topology(
        n,
        id='VL2',
        aligned_buses_or_busbar_count=1,
        section_count=2,
        switch_kinds='BREAKER',
    )  
    
    pp.network.create_line_bays(
        n,
        id='Lline1',
        r=0.1, x=10.0, g1=0.0, b1=0.0, g2=0.0, b2=0.0,
        bus_or_busbar_section_id_1='VL1_1_1', position_order_1=25, direction_1='TOP',   # BB1
        bus_or_busbar_section_id_2='VL2_1_1', position_order_2=15, direction_2='TOP',   # BB3
    )
    n.remove_elements('Lline11_BREAKER')
    n.create_switches(id='sl_switch', voltage_level_id='VL1', node1=8, node2=10,
                        kind='DISCONNECTOR', open=False)
    
    n.create_switches(id='Lline11_BREAKER', voltage_level_id='VL1', node1=10, node2=9,
                        kind='BREAKER', open=False)

    pp.network.create_2_windings_transformer_bays(
        n,
        id='T1',
        b=1e-6, g=1e-6, r=0.5, x=10.0, rated_u1=225.0, rated_u2=63.0,
        bus_or_busbar_section_id_1='VL1_1_1', position_order_1=35, direction_1='TOP',  # BB1
        bus_or_busbar_section_id_2='VL2_1_1', position_order_2=5,  direction_2='TOP',
    )
    n.remove_elements('T11_BREAKER')
    n.create_switches(id='T11_BREAKER1', voltage_level_id='VL1', node1=11, node2=17,
                        kind='BREAKER', open=False)
    n.create_switches(id='T11_BREAKER2', voltage_level_id='VL1', node1=17, node2=12,
                        kind='BREAKER', open=False)

    pp.network.create_load_bay(
        n,
        id='Load1',
        bus_or_busbar_section_id='VL1_1_2',  # BB2
        p0=50.0, q0=10.0,
        position_order=25, direction='TOP',
    )

    pp.network.create_coupling_device(
        n,
        bus_or_busbar_section_id_1=['VL1_1_1', 'VL1_1_2'],
        bus_or_busbar_section_id_2=['VL1_2_1', 'VL1_2_2'],
    )
    return n


if __name__ == "__main__":
    network = build_substation_with_2x2_busbars()
    sld_param = pp.network.SldParameters(
        use_name=False, component_library="Convergence", nodes_infos=False, display_current_feeder_info=False
    )
    
    folder = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(folder, "asset_bay_raw.svg")
    network.write_single_line_diagram_svg("VL1", svg_file=svg_path, parameters=sld_param)
