# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from copy import deepcopy

import numpy as np
import pandapower as pp
from beartype.typing import Union


def add_phaseshift_transformer_to_line(
    net: pp.pandapowerNet,
    line_idx: int,
    at_from_bus: bool = True,
    tap_min: int = -30,
    tap_max: int = 30,
    tap_step_degree: float = 2.0,
) -> tuple[np.integer, np.integer]:
    """
    Inserts a phase-shifting transformer into the pandapower network on the given line.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    line_idx : int
        Index of the line in net.line on which to insert the phase-shifting transformer.
    at_from_bus : bool, optional
        If True, the phase-shifting transformer is inserted at the from-bus of the line.
        If False, it is inserted at the to-bus (default is True
    tap_min : int, optional
        Minimum tap position of the phase-shifting transformer (default is -30).
    tap_max : int, optional
        Maximum tap position of the phase-shifting transformer (default is 30).
    tap_step_degree : float, optional
        Step size in degrees for the tap position of the phase-shifting transformer (default is 2.0).

    Returns
    -------
    trafo_idx : int
        The index of the newly created transformer in net.trafo.
    helper_bus : int
        The index of the newly created helper bus.
    """
    # 1) Get the from-bus (and base voltage) of the given line
    from_bus = net.line.at[line_idx, "from_bus"]
    to_bus = net.line.at[line_idx, "to_bus"]
    if not at_from_bus:
        from_bus, to_bus = to_bus, from_bus

    base_kv = net.bus.at[from_bus, "vn_kv"]

    # 2) Create a new helper bus with the same base voltage as from_bus, and put it in the middle
    # of the buses, slightly towards the from_bus

    from_x = json.loads(net.bus.at[from_bus, "geo"])["coordinates"][0]
    from_y = json.loads(net.bus.at[from_bus, "geo"])["coordinates"][1]
    to_x = json.loads(net.bus.at[to_bus, "geo"])["coordinates"][0]
    to_y = json.loads(net.bus.at[to_bus, "geo"])["coordinates"][1]
    helper_x = (from_x + to_x) / 2 + (from_x - to_x) * 0.1
    helper_y = (from_y + to_y) / 2 + (from_y - to_y) * 0.1

    helper_bus = pp.create_bus(net, vn_kv=base_kv, name=f"Helper Bus for Line {line_idx}", geodata=(helper_x, helper_y))

    # 3) Insert a transformer with the specified shift angle
    #    (For simplicity, we pick some typical parameters. Adjust as needed.)
    trafo_idx = pp.create_transformer_from_parameters(
        net,
        hv_bus=from_bus,
        lv_bus=helper_bus,
        sn_mva=max(net.sn_mva, 10),  # or a relevant S rated MVA
        vn_hv_kv=base_kv,
        vn_lv_kv=base_kv,
        vk_percent=10.0,  # Example short-circuit voltage
        vkr_percent=0.1,  # Example short-circuit real part
        pfe_kw=0,  # Example iron losses
        i0_percent=0,  # Example open-circuit currents
        name=f"Phase Shift Trafo on Line {line_idx}",
        tap_side="hv",
        tap_neutral=0,  # position = 0 is the neutral
        tap_min=tap_min,  # min tap position
        tap_max=tap_max,  # max tap position
        tap_step_degree=tap_step_degree,
        tap_pos=0,  # start tap position
        tap_changer_type=True,
    )

    # 4) “Move” the from-bus connection of the line to the helper bus
    net.line.at[line_idx, ("from_bus" if at_from_bus else "to_bus")] = helper_bus

    return trafo_idx, helper_bus


def pandapower_case30_with_psts() -> pp.pandapowerNet:
    """Create a pandapower IEEE 30 bus grid with phase-shifting transformers.

    Returns
    -------
    pp.pandapowerNet
        The pandapower IEEE 30 bus network with phase-shifting transformers.
    """
    net = pp.networks.case30()
    # Add two phase shifters in the middle of the grid and one on the edge to almost separate
    # the grid into two parts, only connected by line 34 if you removed all psts
    add_phaseshift_transformer_to_line(net, 13, at_from_bus=False, tap_min=-20, tap_max=20, tap_step_degree=1.0)
    add_phaseshift_transformer_to_line(net, 11, at_from_bus=False)
    add_phaseshift_transformer_to_line(net, 14, tap_max=40, tap_step_degree=10.0)
    return net


def pandapower_case30_with_psts_and_weak_branches() -> pp.pandapowerNet:
    """Create a pandapower IEEE 30 bus grid with phase-shifting transformers and overloaded branches.

    Returns
    -------
    pp.pandapowerNet
        The pandapower IEEE 30 bus network with phase-shifting transformers and weak branches.
    """
    net = pandapower_case30_with_psts()
    # Also create a weird trafo to check for bugs
    lv_bus = pp.create_bus(net, vn_kv=20, name="LV Bus", geodata=(5, -5))
    pp.create_transformer_from_parameters(
        net,
        hv_bus=25,
        vn_hv_kv=net.bus.at[25, "vn_kv"],
        lv_bus=lv_bus,
        vn_lv_kv=20,
        sn_mva=100,
        vk_percent=10,
        vkr_percent=0.1,
        pfe_kw=30,
        i0_percent=0.1,
        shift_degree=120,  # This trafo is wired weirdly
    )

    # Weaken the line 34 to make it a bottleneck
    net.line.loc[34, "max_i_ka"] /= 2
    return net


def replace_bus_index(net: pp.pandapowerNet, new_index: list[Union[int, np.integer]]) -> None:
    """Replaces the bus index in a pandapower network

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to modify.
    new_index : list[Union[int, np.integer]]
        The new bus index
    """
    bus_idx_map = {new: old for old, new in zip(net.bus.index, new_index, strict=True)}
    for table, key in pp.element_bus_tuples():
        if table in net and key in net[table] and len(net[table]) > 0:
            net[table][key] = net[table][key].map(bus_idx_map)
    net.bus = net.bus.reindex(new_index).reset_index(drop=True)


def pandapower_extended_oberrhein() -> pp.pandapowerNet:
    """Creates a pandapower extended version of the oberrhein grid.

    Returns
    -------
    pp.pandapowerNet
        A pandapower grid with additional elements.
    """
    net = pp.networks.mv_oberrhein()

    # Add branches to create splittable substations
    std_type = "NA2XS2Y 1x95 RM/25 12/20 kV"
    pp.create_line(net, 4, 290, length_km=1, std_type=std_type, name="imaginary_line_1")
    pp.create_line(net, 3, 290, length_km=1, std_type=std_type, name="imaginary_line_2")
    pp.create_line(net, 273, 304, length_km=1, std_type=std_type, name="imaginary_line_3")
    pp.create_line(net, 273, 305, length_km=1, std_type=std_type, name="imaginary_line_4")
    pp.create_line(
        net,
        from_bus=116,
        to_bus=237,
        length_km=1,
        std_type=std_type,
        name="imaginary_line_5",
    )
    # net.line.geo.loc[line_id, "coords"] = [
    #     (net.bus.geo.loc[116, "x"], net.bus.geo.loc[116, "y"]),
    #     (net.bus.geo.loc[237, "x"], net.bus.geo.loc[237, "y"]),
    # ]

    pp.create_line(
        net,
        from_bus=116,
        to_bus=298,
        length_km=10,
        std_type=std_type,
        name="n_2_safe_line",  # This is a line that should be safe in N-2
    )
    # net.line.geo.loc[line_id, "coords"] = [
    #     (net.bus.geo.loc[116, "x"], net.bus.geo.loc[116, "y"]),
    #     (net.bus.geo.loc[298, "x"], net.bus.geo.loc[298, "y"]),
    # ]
    # Create a "splittable" substation but with a lot of stub lines going into it
    # This should be recognized as a relevant substation first but in the action selection
    # The actions should be filtered out
    stub_bus_1 = pp.create_bus(net, vn_kv=20, name="stub_bus_1")
    stub_bus_2 = pp.create_bus(net, vn_kv=20, name="stub_bus_2")
    stub_bus_3 = pp.create_bus(net, vn_kv=20, name="stub_bus_3")

    pp.create_line(net, 42, stub_bus_1, length_km=1, std_type=std_type, name="stub_line_1")
    pp.create_line(net, 42, stub_bus_2, length_km=1, std_type=std_type, name="stub_line_2")
    pp.create_line(net, 42, stub_bus_3, length_km=1, std_type=std_type, name="stub_line_3")

    # create a 10kv "grid" to add two 3w trafos
    lowv_bus = pp.create_bus(net, vn_kv=10, name="lowv_bus_1")
    midv_bus = pp.create_bus(net, vn_kv=20, name="midv_bus_1")
    pp.create_transformer3w(
        net,
        hv_bus=net.trafo.hv_bus.values[0],
        mv_bus=midv_bus,
        lv_bus=lowv_bus,
        name="3w_trafo_1",
        std_type="63/25/38 MVA 110/20/10 kV",
    )
    pp.create_transformer3w(
        net,
        hv_bus=net.trafo.hv_bus.values[1],
        mv_bus=midv_bus,
        lv_bus=lowv_bus,
        name="3w_trafo_2",
        std_type="63/25/38 MVA 110/20/10 kV",
    )

    # Add a PST
    pp.create_transformer_from_parameters(
        net,
        hv_bus=5,
        lv_bus=100,
        sn_mva=25,
        vn_hv_kv=20,
        vn_lv_kv=20,
        vkr_percent=0.1,
        pfe_kw=1,
        vk_percent=10,
        i0_percent=0.1,
        tap_step_degree=1,
        tap_side="lv",
        tap_pos=5,
        tap_neutral=0,
        tap_max=30,
        tap_min=-30,
        tap_changer_type=True,
    )

    # Add out of service injecions
    pp.create_gen(
        net,
        net.bus.index[1],
        p_mw=1.0,
        q_mvar=1.0,
        in_service=False,
        name="outofservice_gen",
    )
    pp.create_sgen(
        net,
        net.bus.index[2],
        p_mw=1.0,
        q_mvar=1.0,
        in_service=False,
        name="outofservice_sgen",
    )
    pp.create_load(
        net,
        net.bus.index[3],
        p_mw=1.0,
        q_mvar=1.0,
        in_service=False,
        name="outofservice_load",
    )

    # Add shunts, dclines, wards, xwards elements to test their influence
    pp.create_shunts(
        net,
        net.bus.index[1:4],
        q_mvar=[1.0, 2.0, 3.0],
        p_mw=[2.0, 1.0, 4.0],
        in_service=[True, True, False],
        name=["test_shunt_1", "test_shunt_2", "outofservice_shunt"],
        id_characteristic_table=[0, 1, 2],  # due to a pandapower bug, this is not optional
        # references the index of the characteristic from the lookup table net.shunt_characteristic_table
    )
    pp.create_dcline(
        net,
        from_bus=116,
        to_bus=237,
        p_mw=10.0,
        loss_percent=1.0,
        loss_mw=1.0,
        vm_from_pu=1.0,
        vm_to_pu=1.0,
        name="test_dcline",
    )
    pp.create_dcline(
        net,
        from_bus=116,
        to_bus=237,
        p_mw=10.0,
        loss_percent=1.0,
        loss_mw=1.0,
        vm_from_pu=1.0,
        vm_to_pu=1.0,
        in_service=False,
        name="outofservice_test_dcline",
    )
    pp.create_ward(net, net.bus.index[4], 1.0, 2.0, 3.0, 4.0, name="test_ward")
    pp.create_ward(
        net,
        net.bus.index[4],
        1.0,
        2.0,
        3.0,
        4.0,
        in_service=False,
        name="outofservice_test_ward",
    )

    pp.create_xward(
        net,
        bus=net.bus.index[3],
        ps_mw=1.0,
        qs_mvar=2.0,
        pz_mw=3.0,
        qz_mvar=4.0,
        r_ohm=1.0,
        x_ohm=1.0,
        vm_pu=1,
        name="test_xward",
    )

    # Set scaling of sgens so they are recognized
    net.sgen.scaling = 1.0

    np.random.seed(0)
    pp.rundcpp(net)

    # Convert one of the slack busses to a generator
    pp.create_gen(
        net,
        bus=net.ext_grid.bus[1],
        name=net.ext_grid.name[1],
        p_mw=net.res_ext_grid.p_mw[1],
        q_mvar=net.res_ext_grid.q_mvar[1],
        vm_pu=1,
    )
    net.ext_grid.drop(1, inplace=True)
    net.switch.closed = True

    # Reindex
    pp.toolbox.create_continuous_bus_index(net)
    pp.toolbox.create_continuous_elements_index(net)

    # Remove the scaling factor from the loads
    net.load.p_mw *= net.load.scaling
    net.load.q_mvar *= net.load.scaling
    net.load.scaling = 1
    return net


def pandapower_non_converging_case57() -> pp.pandapowerNet:
    """Creates a ac-non-converging pandapower case57 grid.

    Still converges in DC.

    Returns
    -------
    pp.pandapowerNet
        A pandapower grid that does not converge in AC load flow.
    """

    net = pp.networks.case57()
    # Change the 115kv to a 50kV bus to prevent convergence in AC but keep it converging in DC
    net.bus.loc[net.bus.vn_kv == 115, "vn_kv"] = 50
    net.trafo.loc[net.trafo.vn_lv_kv == 115, "vn_lv_kv"] = 50
    return net


def pandapower_extended_case57() -> pp.pandapowerNet:
    """Creates a pandapower case57 grid with additional elements.

    Returns
    -------
    pp.pandapowerNet
        A pandapower grid with additional elements"""
    net = pp.networks.case57()
    net.line["name"] = net.line.index.astype(str)
    net.trafo["name"] = net.trafo.index.astype(str)
    net.bus["name"] = net.bus.index.astype(str)
    pst_bus = pp.create_bus(net, vn_kv=115, name="PSTBus")

    pp.create_line_from_parameters(
        net,
        from_bus=pst_bus,
        to_bus=16,
        length_km=1,
        r_ohm_per_km=0.1 * (np.square(115) / net.sn_mva),
        x_ohm_per_km=0.1 * (np.square(115) / net.sn_mva),
        c_nf_per_km=0,
        max_i_ka=9999,
        name="PSTLine",
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=5,  # hv and lv will be switched later
        lv_bus=pst_bus,
        sn_mva=1,
        vn_hv_kv=115,
        vn_lv_kv=115,
        vk_percent=0.1,
        vkr_percent=0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=-8,
        name="PST",
    )

    # The bus needs to appear in the sixth place to match powsybl
    new_idx = net.bus.index[:6].tolist() + [pst_bus] + net.bus.index[6:-1].tolist()
    replace_bus_index(net, new_idx)

    # We have to change from- and to buses on all trafos (except for the one that we added manually)
    # So we can just change all of them, the order on the create_transformer call was switched
    trafo = net.trafo
    trafo_copy = deepcopy(net.trafo)
    trafo_copy["lv_bus"] = trafo["hv_bus"]
    trafo_copy["hv_bus"] = trafo["lv_bus"]
    trafo_copy["vn_hv_kv"] = trafo["vn_lv_kv"]
    trafo_copy["vn_lv_kv"] = trafo["vn_hv_kv"]
    trafo_copy["tap_side"] = "lv"
    trafo_copy.sort_values(["hv_bus", "lv_bus"], inplace=True)
    trafo_copy.reset_index(drop=True, inplace=True)
    net.trafo = trafo_copy

    # We have to re-order the busbars again, this time I didn't bother to reverse-engineer the
    # powsybl order, so I just sort them through Kuhn-Munkres
    # Code to reproduce this order:
    # a = pp_backend.net.res_bus.va_degree.values
    # b = powsybl_backend.net.get_buses()["v_angle"].values
    # cost_matrix = (a[None, :] - b[:, None]) ** 2
    # _, indices = scipy.optimize.linear_sum_assignment(cost_matrix)

    indices = [
        0,
        1,
        2,
        3,
        18,
        4,
        5,
        6,
        7,
        29,
        8,
        9,
        55,
        10,
        51,
        11,
        41,
        43,
        12,
        13,
        49,
        14,
        46,
        15,
        45,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        30,
        31,
        32,
        34,
        33,
        35,
        36,
        37,
        38,
        39,
        57,
        40,
        56,
        42,
        44,
        47,
        48,
        50,
        52,
        53,
        54,
    ]
    replace_bus_index(net, indices)
    return net


def example_multivoltage_cross_coupler() -> pp.pandapowerNet:
    """
    Expands the single busbar coupler to a double busbar with a busbar coupler and a cross coupler
    See also: https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb

    DS- Disconnector (type CB)
    BB- Busbar (type b)
    BC- Busbar coupler (two nodes and three switches)
    CC- Cross coupler (two nodes and three switches)
    PS- Power switch / Branch switch
                            Branch 1                         Branch 2
                                |                                    |
                                / DS                                 / DS
                                |                                    |
                                |                                    |
                                / CB                                 / CB
                                |                                    |
                            ____|                                ____|
                            |   |                               |   |
                            |   / DS                            |   / DS
                            |   |              CC 1/3           |   |
        BB 1----------------|----------- _______/______ --------|-------------- BB 3
                |           |                                   |          |
            DS /       DS /                                 DS /       DS /
                |           |             CC 2/4                |          |
        BB 2----|-----------/--------- _______/________ -------------------|--------BB 4
            |   |                                                          |   |
        DS /   |                                                          |   / DS
            |   |                                                          |   |
            |_/_|                                                          |_/_|
            BC 1/2                                                         BC 3/4

    """
    net = pp.networks.example_multivoltage()

    # convert the Single Busbar Coupler to Double Busbar Coupler
    net.bus.loc[16, "name"] = "Double Busbar Coupler 1"
    # create the Double Busbar Coupler
    pp.create_bus(net, name="Double Busbar Coupler 2", vn_kv=110, type="b")  # id = 57
    pp.create_bus(net, name="Double Busbar Coupler 3", vn_kv=110, type="b")  # id = 58
    pp.create_bus(net, name="Double Busbar Coupler 4", vn_kv=110, type="b")  # id = 59

    # create the Cross Coupler for bus 1 and 3
    # two buses and three switches
    pp.create_bus(net, name="Cross Coupler 1/3", vn_kv=110, type="n")  # id = 60
    pp.create_bus(net, name="Cross Coupler 3/1", vn_kv=110, type="n")  # id = 61
    pp.create_switch(net, 16, 60, et="b", closed=True, type="DS", name="Cross Coupler 1/3 DS")  # id = 88
    pp.create_switch(net, 60, 61, et="b", closed=True, type="CB", name="Cross Coupler 1-3 CB")  # id = 89
    pp.create_switch(net, 61, 58, et="b", closed=True, type="DS", name="Cross Coupler 1-3 DS")  # id = 90

    # create the Cross Coupler for bus 2 and 4
    # two buses and three switches
    pp.create_bus(net, name="Cross Coupler 2/4", vn_kv=110, type="n")  # id = 62
    pp.create_bus(net, name="Cross Coupler 4/2", vn_kv=110, type="n")  # id = 63
    pp.create_switch(net, 57, 62, et="b", closed=True, type="DS", name="Cross Coupler 2/4 DS")  # id = 91
    pp.create_switch(net, 62, 63, et="b", closed=True, type="CB", name="Cross Coupler 2-4 CB")  # id = 92
    pp.create_switch(net, 63, 59, et="b", closed=True, type="DS", name="Cross Coupler 2-4 DS")  # id = 93

    # create the Busbar Coupler for bus 1 and 2
    # two buses and three switches
    pp.create_bus(net, name="Busbar Coupler 1/2", vn_kv=110, type="n")  # id = 64
    pp.create_bus(net, name="Busbar Coupler 2/1", vn_kv=110, type="n")  # id = 65
    pp.create_switch(net, 16, 64, et="b", closed=True, type="DS", name="Busbar Coupler 1/2 DS")  # id = 94
    pp.create_switch(net, 64, 65, et="b", closed=True, type="CB", name="Busbar Coupler 1-2 CB")  # id = 95
    pp.create_switch(net, 65, 57, et="b", closed=True, type="DS", name="Busbar Coupler 1-2 DS")  # id = 96

    # create the Busbar Coupler for bus 3 and 4
    # two buses and three switches
    pp.create_bus(net, name="Busbar Coupler 3/4", vn_kv=110, type="n")  # id = 66
    pp.create_bus(net, name="Busbar Coupler 4/3", vn_kv=110, type="n")  # id = 67
    pp.create_switch(net, 58, 66, et="b", closed=True, type="DS", name="Busbar Coupler 3/4 DS")  # id = 97
    pp.create_switch(net, 66, 67, et="b", closed=True, type="CB", name="Busbar Coupler 3-4 CB")  # id = 98
    pp.create_switch(net, 67, 59, et="b", closed=True, type="DS", name="Busbar Coupler 3-4 DS")  # id = 99

    # there are 5 assets linked to the original busbar
    # create new switches for the parallel busbars
    # move some of the assets to the new busbars
    # current assets linked to the busbar
    # {'sgen': [0], 'trafo': [0], 'load': [0], 'line': [0, 5]}
    # new assignment of the assets
    # BB1 -> line 0 + load 0
    # BB2 -> trafo 0
    # BB3 -> line 5
    # BB4 -> sgen 0

    # create new switches for the first two busbars
    pp.create_switch(net, 23, 57, et="b", closed=True, type="DS", name="Bus SB T1.2.2")  # id = 100 -> trafo 0
    pp.create_switch(net, 25, 57, et="b", closed=False, type="DS", name="Bus SB T2.2.2")  # id = 101 -> line 0
    pp.create_switch(net, 29, 57, et="b", closed=False, type="DS", name="Bus SB T4.2.2")  # id = 102 -> load 0
    # create new switches for the last two busbars
    pp.create_switch(net, 27, 59, et="b", closed=False, type="DS", name="Bus SB T3.2.2")  # id = 103 -> line 5
    pp.create_switch(net, 31, 59, et="b", closed=True, type="DS", name="Bus SB T5.2.2")  # id = 104 -> sgen 0

    # move bus from existing busbar to new busbar
    net.switch.loc[24, "bus"] = 58
    net.switch.loc[28, "bus"] = 58
    net.switch.loc[28, "closed"] = False  # BB4 -> sgen 0 therefore switch open to bus 58

    return net
