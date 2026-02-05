# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""
See details of the used testing grid:
https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb

"""

from copy import deepcopy

import logbook
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_importer.pandapower_import import pandapower_toolset_node_breaker

logger = logbook.Logger(__name__)


def test_get_type_b_busbars(pp_network_w_switches):
    net = pp_network_w_switches
    res = pandapower_toolset_node_breaker.get_type_b_nodes(net, net.bus.index)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    res = pandapower_toolset_node_breaker.get_type_b_nodes(net)
    assert len(res) == 3


def test_get_indirect_connected_switch(pp_network_w_switches):
    net = pp_network_w_switches
    # test intended use case
    # a series of 3 switches are between bus 0 and 3 -> return the busbarcoupler
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1)
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 1
    assert "switch" in busbarcoupler
    assert isinstance(busbarcoupler["switch"], list)
    assert busbarcoupler["switch"] == [14]
    assert net.switch.loc[14]["type"] == "CB"
    # test unintended use case
    # a series of 2 switches are between bus 0 and 3 -> return both
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 3)
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 0


def test_get_indirect_connected_switch_three_buses(pp_network_w_switches):
    net = pp_network_w_switches
    # add an additional switch between bus 0 and 3
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 68
    pp.create_switch(net, 57, 2, et="b", closed=True, type="CB", name="CB1")
    net.switch.loc[0, "element"] = 57
    # a series of 3 switches are between bus 0 and 3 -> return the busbarcoupler
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1, consider_three_buses=True)
    assert busbarcoupler == {"switch": [88, 14]}
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1, consider_three_buses=False)
    assert busbarcoupler == {}


def test_get_indirect_connected_switch_empty(pp_network_w_switches_open_coupler):
    net = pp_network_w_switches_open_coupler
    # test for del "switch" in busbarcoupler
    # dict should be empty if switch is not closed, but exists
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1)
    assert busbarcoupler == {}


def test_get_indirect_connected_switch_parallel_coupler(
    pp_network_w_switches_parallel_coupler,
):
    net = pp_network_w_switches_parallel_coupler
    # test two parallel switches
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1)
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 1
    assert isinstance(busbarcoupler["switch"], list)
    assert len(busbarcoupler["switch"]) == 2
    assert 90 in busbarcoupler["switch"]
    assert 14 in busbarcoupler["switch"]
    assert net.switch.loc[90]["type"] == "CB"
    # test open switch
    net.switch.loc[14, "closed"] = False
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1)
    assert busbarcoupler["switch"] == [90]
    # test ignore open switch
    busbarcoupler = pandapower_toolset_node_breaker.get_indirect_connected_switch(net, 0, 1, only_closed_switches=False)
    assert 90 in busbarcoupler["switch"]
    assert 14 in busbarcoupler["switch"]


def test_get_indirect_connected_switch_parallel_line(pp_network_w_switches):
    net = pp_network_w_switches
    pp.create_line(net, 2, 3, std_type="184-AL1/30-ST1A 110.0", length_km=1)
    bus_1 = 0
    bus_2 = 1
    connection_value = {"line": {25}, "switch": [14]}
    error_value = [f"{key!s}:{value!s}" for key, values in connection_value.items() for value in values]
    with pytest.raises(
        ValueError,
        match=f"Indirect connection between bus {bus_1} and {bus_2} must contain only switches {error_value}",
    ):
        pandapower_toolset_node_breaker.get_indirect_connected_switch(net, bus_1, bus_2)

    pp.toolbox.drop_elements(net, "switch", 14)
    connection_value = {"line": {25}}
    error_value = [f"{key!s}:{value!s}" for key, values in connection_value.items() for value in values]
    with pytest.raises(
        ValueError,
        match=f"Indirect connection between bus {bus_1} and {bus_2} must contain only switches {error_value}",
    ):
        pandapower_toolset_node_breaker.get_indirect_connected_switch(net, bus_1, bus_2)


def test_get_indirect_connected_switches_three_buses(pp_network_w_switches):
    net = pp_network_w_switches
    # add an additional switch between bus 0 and 3
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 68
    pp.create_switch(net, 57, 2, et="b", closed=True, type="CB", name="CB1")
    net.switch.loc[0, "element"] = 57
    bus_1 = 0
    bus_2 = 1
    bus_1_connected = [3]
    bus_2_connected = [57, 5, 7, 9, 11]
    only_closed_switches = True
    exclude_buses = [bus_1, bus_2]
    expected = [88, 14]
    # a series of 3 switches are between bus 0 and 3 -> return the busbarcoupler
    res = pandapower_toolset_node_breaker.get_indirect_connected_switches_three_buses(
        net=net,
        bus_1=bus_1,
        bus_2=bus_2,
        bus_1_connected=bus_1_connected,
        bus_2_connected=bus_2_connected,
        only_closed_switches=only_closed_switches,
    )
    assert res["switch"] == expected
    res = pandapower_toolset_node_breaker.get_indirect_connected_switches_three_buses(
        net=net,
        bus_1=bus_1,
        bus_2=bus_2,
        bus_1_connected=bus_1_connected,
        bus_2_connected=bus_2_connected,
        only_closed_switches=only_closed_switches,
        exclude_buses=exclude_buses,
    )
    assert res["switch"] == expected
    exclude_buses.append(57)
    res = pandapower_toolset_node_breaker.get_indirect_connected_switches_three_buses(
        net=net,
        bus_1=bus_1,
        bus_2=bus_2,
        bus_1_connected=bus_1_connected,
        bus_2_connected=bus_2_connected,
        only_closed_switches=only_closed_switches,
        exclude_buses=exclude_buses,
    )
    assert res["switch"] == []


def test_get_all_switches_from_bus_ids(pp_network_w_switches):
    net = pp_network_w_switches
    bus_switches = pandapower_toolset_node_breaker.get_all_switches_from_bus_ids(net, [2, 3])
    assert all(bus_switches.index == [0, 1, 14])
    net = pp.networks.example_simple()
    bus_switches = pandapower_toolset_node_breaker.get_all_switches_from_bus_ids(net, [0])
    assert len(bus_switches) == 0


def test_get_closed_switch(pp_network_w_switches):
    net = pp_network_w_switches
    col_name = "name"
    # "DB DS3" is open
    col_value = ["DB DS0", "DB DS3", "DB DS4"]
    res = pandapower_toolset_node_breaker.get_closed_switch(net.switch, col_name, col_value)
    assert res["closed"].all()
    assert res["name"].isin(["DB DS0", "DB DS4"]).all()
    assert "DB DS3" not in res["name"].values


def fuse_closed_switches_by_bus_ids_helper(net: pp.pandapowerNet, save_switches: pd.DataFrame, len_bus: int) -> None:
    # check if correct switches are removed
    # the following switches are not part of the substation and should not be changed
    assert save_switches.equals(net.switch.loc[19:]), "the switches 19: should not be changed"
    # see diagram which switches are removed:
    # https://github.com/e2nIEE/pandapower/blob/develop/tutorials/create_advanced.ipynb
    assert 0 not in net.switch.index, "the switch 0 DS0 should be removed"
    assert 1 not in net.switch.index, "the switch 1 DS1 should be removed"
    # CB1 in the diagram -> wrong label in the diagram
    assert 14 not in net.switch.index, "the switch 14 CB1 should be removed"
    # DS2-DS9 should now be connected to bus 0
    assert all(net.switch.loc[2:9, "bus"] == 0), "the switches 2:10 should be connected to bus 0"

    # check if correct buses are removed
    assert len(net.bus) == len_bus - 3, "the buses 1, 2, 3 should be removed"
    assert all(~net.bus.index.isin([1, 3, 2])), "the buses 1, 2, 3 should be removed"
    assert net.bus.loc[0, "type"] == "b", "bus 0 was of type b and should remain"


def test_fuse_closed_switches_by_bus_ids(pp_network_w_switches):
    net = deepcopy(pp_network_w_switches)
    assert len(net.switch) == 88
    len_bus = len(net.bus)
    assert len_bus == 57
    save_switches = net.switch.loc[19:].copy()
    expected = [
        0,
        0,
        0,
        0,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
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
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
    ]
    # intended order
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [0, 1, 3, 2])
    fuse_closed_switches_by_bus_ids_helper(net, save_switches, len_bus)
    assert all(res == expected)
    # random order
    net = deepcopy(pp_network_w_switches)
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [1, 3, 0, 2])
    assert all(res == expected)
    fuse_closed_switches_by_bus_ids_helper(net, save_switches, len_bus)
    # test if busbar 2 comes before 0
    net = deepcopy(pp_network_w_switches)
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [1, 3, 2, 0])
    assert all(res == expected)
    fuse_closed_switches_by_bus_ids_helper(net, save_switches, len_bus)


def test_fuse_closed_switches_by_bus_ids_two_busbars(pp_network_w_switches):
    net = pp_network_w_switches
    save_switches = net.switch.loc[10:].copy()
    save_switches2 = net.switch.loc[:9].copy()
    # unintended use case -> fuse two busbars directly without the path of the busbarcoupler
    # the busbar coupler is not removed and now connected to bus 0
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [0, 1])
    assert all(
        res
        == [
            0,
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
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
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
        ]
    )
    assert len(net.switch) == 88
    assert len(net.bus) == 57 - 1
    assert save_switches.equals(net.switch.loc[10:])
    save_switches2["bus"] = 0
    assert save_switches2.equals(net.switch.loc[:9])


def test_fuse_closed_switches_by_bus_ids_two_buses(pp_network_w_switches):
    net = pp_network_w_switches
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [0, 2])
    # this doesn't make sense electrically, but is possible to do
    # both buses are not directly connected by a switch
    assert all(
        res
        == [
            0,
            1,
            0,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
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
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
        ]
    )
    net = pp_network_w_switches
    # this could make sense, as both nodes are interconnected by a switch
    res = pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [0, 1])
    assert all(
        res
        == [
            0,
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
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
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
        ]
    )


def test_fuse_closed_switches_by_bus_ids_value_error(pp_network_w_switches):
    net = pp_network_w_switches
    with pytest.raises(ValueError):
        pandapower_toolset_node_breaker.fuse_closed_switches_by_bus_ids(net, [2, 3])


def test_get_vertical_connected_busbars(pp_network_w_switches):
    net = pp_network_w_switches
    busbarcoupler = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, [0, 1])
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 2
    assert busbarcoupler == {0: [1], 1: [0]}
    net.bus.loc[30, "type"] = "b"
    busbarcoupler = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, [0, 1, 30])
    assert busbarcoupler == {0: [1], 1: [0], 30: []}

    net = pp_network_w_switches
    net.switch.loc[14, "bus"] = 0
    net.switch.loc[14, "element"] = 1
    net.switch.drop([0, 1], inplace=True)
    busbarcoupler = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, [1, 0])
    assert busbarcoupler == {0: [1], 1: [0]}


def test_get_vertical_connected_busbars_three_busbars(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    type_b_bus_name = "Double Busbar Coupler 2.3"
    # third busbar
    pp.create_bus(net, name=type_b_bus_name, vn_kv=110, type="b")  # id = 68
    # nodes for the switches
    pp.create_bus(net, name="Busbar Coupler 1/2.3", vn_kv=110, type="n")  # id = 69
    pp.create_bus(net, name="Busbar Coupler 2.3/1", vn_kv=110, type="n")  # id = 70
    new_bus_index = net.bus[net.bus.name == type_b_bus_name].index[0]
    # create the Busbar Coupler for bus 1 and 2.3
    pp.create_switch(
        net,
        16,
        new_bus_index + 1,
        et="b",
        closed=True,
        type="DS",
        name="Busbar Coupler 3/4 DS",
    )
    pp.create_switch(
        net,
        new_bus_index + 1,
        new_bus_index + 2,
        et="b",
        closed=True,
        type="CB",
        name="Busbar Coupler 3-4 CB",
    )
    pp.create_switch(
        net,
        new_bus_index + 2,
        new_bus_index,
        et="b",
        closed=True,
        type="DS",
        name="Busbar Coupler 3-4 DS",
    )

    # the new busbar has a busbar coupler, but no asset connected -> no vertical connection
    id_list = list(pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 16))
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert res == {68: [], 16: [57], 57: [16], 58: [59], 59: [58]}

    # create new switches for the new busbar 2.3
    # this is the vertical connection that is checked
    pp.create_switch(net, 23, new_bus_index, et="b", closed=True, type="DS", name="Bus SB T1.2.3")  # id = 100 -> trafo 0
    id_list = list(pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 16))
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert res == {68: [16, 57], 16: [68, 57], 57: [68, 16], 58: [59], 59: [58]}


def test_get_vertical_connected_busbars_cross_coupler(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    id_list = list(pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 16))
    expected = {16: [57], 57: [16], 58: [59], 59: [58]}
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert res == expected
    id_list = [16, 57, 58, 59]
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert res == expected
    # 13 not part of the substation and not of type b
    # 57 and 58 are of type be, but not connected vertically
    id_list = [13, 57, 58]
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert {57: [], 58: []}
    id_list = [16, 57, 58]
    res = pandapower_toolset_node_breaker.get_vertical_connected_busbars(net, id_list)
    assert {16: [57], 57: [16], 58: []}


def test_get_coupler_types_of_substation_empty_return(pp_network_w_switches):
    net = pp_network_w_switches
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0])
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 4
    dict_keys = [
        "busbar_coupler_bus_ids",
        "cross_coupler_bus_ids",
        "busbar_coupler_switch_ids",
        "cross_coupler_switch_ids",
    ]
    assert all([key in busbarcoupler.keys() for key in dict_keys])
    assert all([isinstance(busbarcoupler[key], list) for key in dict_keys])
    assert all([len(value) == 0 for value in busbarcoupler.values()])
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 2])
    assert all([len(value) == 0 for value in busbarcoupler.values()])
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 3])
    assert all([len(value) == 0 for value in busbarcoupler.values()])


def test_get_coupler_types_of_substation_test_one_busbar_coupler(pp_network_w_switches):
    net = pp_network_w_switches
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 1])
    assert isinstance(busbarcoupler, dict)
    assert len(busbarcoupler) == 4
    dict_keys = [
        "busbar_coupler_bus_ids",
        "cross_coupler_bus_ids",
        "busbar_coupler_switch_ids",
        "cross_coupler_switch_ids",
    ]
    assert all([key in busbarcoupler.keys() for key in dict_keys])
    assert all([isinstance(busbarcoupler[key], list) for key in dict_keys])
    assert busbarcoupler["busbar_coupler_bus_ids"] == [[0, 1, 3, 2]]
    assert busbarcoupler["busbar_coupler_switch_ids"] == [[14, 1, 0]]
    assert busbarcoupler["cross_coupler_bus_ids"] == []
    assert busbarcoupler["cross_coupler_switch_ids"] == []


def test_get_coupler_types_of_substation_test_two_busbar_coupler(
    pp_network_w_switches_parallel_coupler,
):
    net = pp_network_w_switches_parallel_coupler
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 1])
    assert busbarcoupler["busbar_coupler_bus_ids"] == [[0, 1, 58, 57], [0, 1, 3, 2]]
    assert busbarcoupler["busbar_coupler_switch_ids"] == [[90, 89, 88], [14, 1, 0]]
    assert busbarcoupler["cross_coupler_bus_ids"] == []
    assert busbarcoupler["cross_coupler_switch_ids"] == []


@pytest.mark.skip(reason="Not implemented anymore")
def test_get_coupler_types_of_substation_test_three_bus_parallel(caplog):
    net = pp.networks.example_multivoltage()
    # create a parallel three bus coupler
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 57
    pp.create_switch(net, 57, 2, et="b", closed=True, type="CB", name="CB1")
    net.switch.loc[0, "element"] = 57
    id_max = net.switch.index.max()
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 58
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 59
    pp.create_bus(net, name="test_node", vn_kv=110, type="b")  # id = 60
    second_busbar_coupler = net.switch.loc[[0, 1, 14, 88]]
    second_busbar_coupler["index"] = [id_max + 1, id_max + 2, id_max + 3, id_max + 4]
    second_busbar_coupler.set_index("index", inplace=True, drop=True)
    net.switch = pd.concat([net.switch, second_busbar_coupler])
    net.switch.loc[id_max + 1, "element"] = 58
    net.switch.loc[id_max + 2, "element"] = 59
    net.switch.loc[id_max + 3, "element"] = 59
    net.switch.loc[id_max + 4, "element"] = 60
    net.switch.loc[id_max + 3, "bus"] = 60
    net.switch.loc[id_max + 4, "bus"] = 58
    with logbook.handlers.TestHandler() as caplog:
        pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 1])
    assert "Unknown Switch configuration" in "".join(caplog.formatted_records)


def test_get_coupler_types_of_substation_test_cross_coupler(
    net_multivoltage_cross_coupler,
):
    net = net_multivoltage_cross_coupler
    expected = {
        "busbar_coupler_bus_ids": [[16, 57, 65, 64], [58, 59, 67, 66]],
        "cross_coupler_bus_ids": [[16, 58, 61, 60], [57, 59, 63, 62]],
        "busbar_coupler_switch_ids": [[95, 94, 96], [98, 97, 99]],
        "cross_coupler_switch_ids": [[89, 88, 90], [92, 91, 93]],
    }

    # test full substation
    id_list = [16, 57, 58, 59]
    busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, id_list, only_closed_switches=True)
    assert busbarcoupler1 == expected
    busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        net, id_list, only_closed_switches=False
    )
    assert busbarcoupler2 == expected

    # test all combinations
    # test no connections
    id_lists = [[], [16], [16, 59], [57, 58]]
    expected_empty = {
        "busbar_coupler_bus_ids": [],
        "cross_coupler_bus_ids": [],
        "busbar_coupler_switch_ids": [],
        "cross_coupler_switch_ids": [],
    }
    for id_list in id_lists:
        busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
            net, id_list, only_closed_switches=True
        )
        assert busbarcoupler1 == expected_empty
        busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
            net, id_list, only_closed_switches=False
        )
        assert busbarcoupler2 == expected_empty

    # test one connection
    id_list = [16, 58]
    expected_cross = {
        "busbar_coupler_bus_ids": [],
        "cross_coupler_bus_ids": [[16, 58, 61, 60]],
        "busbar_coupler_switch_ids": [],
        "cross_coupler_switch_ids": [[89, 88, 90]],
    }

    busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, id_list, only_closed_switches=True)
    assert busbarcoupler1 == expected_cross
    busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        net, id_list, only_closed_switches=False
    )
    assert busbarcoupler2 == expected_cross

    id_list = [16, 57]
    expected_busbar = {
        "busbar_coupler_bus_ids": [[16, 57, 65, 64]],
        "cross_coupler_bus_ids": [],
        "busbar_coupler_switch_ids": [[95, 94, 96]],
        "cross_coupler_switch_ids": [],
    }
    busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, id_list, only_closed_switches=True)
    assert busbarcoupler1 == expected_busbar
    busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        net, id_list, only_closed_switches=False
    )
    assert busbarcoupler2 == expected_busbar

    # test only_closed_switches=True/False
    net.switch.loc[95, "closed"] = False
    busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, id_list, only_closed_switches=True)
    assert busbarcoupler1 == expected_empty
    busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        net, id_list, only_closed_switches=False
    )
    assert busbarcoupler2 == expected_busbar

    id_list = [16, 58]
    net.switch.loc[89, "closed"] = False
    busbarcoupler1 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, id_list, only_closed_switches=True)
    assert busbarcoupler1 == expected_empty
    busbarcoupler2 = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        net, id_list, only_closed_switches=False
    )
    assert busbarcoupler2 == expected_cross


@pytest.mark.skip(reason="feature not implemented anymore")
def test_get_coupler_types_of_substation_test_three_bus(
    net_multivoltage_cross_coupler,
):
    net = net_multivoltage_cross_coupler
    pp.create_bus(net, name="test_node", vn_kv=380, type="b")  # id = 57
    id_max_bus = net.bus.index.max()
    pp.create_switch(net, id_max_bus, 2, et="b", closed=True, type="CB", name="CB1")
    id_max_switch = net.switch.index.max()
    net.switch.loc[0, "element"] = id_max_bus
    # test intended use case
    # a series of 3 switches are between bus 0 and 3 -> return the busbarcoupler
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [0, 1])
    assert busbarcoupler["busbar_coupler_bus_ids"][0] == [0, 1, 2, 3, id_max_bus]
    assert busbarcoupler["busbar_coupler_switch_ids"][0] == [id_max_switch, 1, 0, 14]
    assert busbarcoupler["cross_coupler_bus_ids"] == []
    assert busbarcoupler["cross_coupler_switch_ids"] == []


def test_get_coupler_types_of_substation_cross_connector(
    net_multivoltage_cross_coupler,
):
    net = net_multivoltage_cross_coupler
    # create a cross connector by replacing one busbar coupler
    net.switch.loc[91, "element"] = 59
    net.switch.drop(92, inplace=True)
    net.switch.drop(93, inplace=True)

    # test intended use case
    # a series of 3 switches are between bus 0 and 3 -> return the busbarcoupler
    busbarcoupler = pandapower_toolset_node_breaker.get_coupler_types_of_substation(net, [57, 59])
    assert busbarcoupler["busbar_coupler_bus_ids"] == []
    assert busbarcoupler["busbar_coupler_switch_ids"] == []
    assert busbarcoupler["cross_coupler_bus_ids"][0] == [57, 59, 59, 57]
    assert busbarcoupler["cross_coupler_switch_ids"][0] == [91, 91, 91]


def test_get_substation_buses_from_bus_id(pp_network_w_switches):
    net = pp_network_w_switches
    nodes = pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 0)
    assert isinstance(nodes, set)
    assert nodes == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    nodes = pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 32)
    assert nodes == {32}


def test_get_substation_buses_from_bus_id_open_coupler(
    pp_network_w_switches_open_coupler,
):
    net = pp_network_w_switches_open_coupler
    nodes = pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 0, only_closed_switches=False)
    assert nodes == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    nodes = pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 0, only_closed_switches=True)
    assert nodes == {0, 3}


def test_get_substation_buses_from_bus_id_inf_loop():
    net = pp.networks.example_multivoltage()
    start_bus = 56
    end_bus = 57
    pp.create_bus(net, name=f"test_bus{start_bus}", vn_kv=380, type="n")
    pp.create_switch(
        net,
        0,
        end_bus,
        et="b",
        closed=True,
        type="DS",
        name=f"test_bus_switch{start_bus}",
    )

    for i in range(1, 30):
        start_bus += 1
        end_bus += 1
        pp.create_bus(net, name=f"test_bus{start_bus}", vn_kv=380, type="n")
        pp.create_switch(
            net,
            start_bus,
            end_bus,
            et="b",
            closed=True,
            type="DS",
            name=f"test_bus_switch{start_bus}",
        )
    with pytest.raises(RuntimeError):
        pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, 0, only_closed_switches=False)


def test_add_substation_column_to_bus(pp_network_w_switches, pp_network_w_switches_open_coupler):
    net = pp_network_w_switches
    net_open = pp_network_w_switches_open_coupler
    pandapower_toolset_node_breaker.add_substation_column_to_bus(net)
    res_col = ["Double Busbar 1", "Single Busbar", ""]
    substat_cols = ["substat", "substation"]
    assert all(net.bus[substat_cols[0]].unique() == res_col)
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[0]]) == 16
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[1]]) == 16
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[2]]) == 25

    # test that open coupler are ignored in the naming of the substation
    pandapower_toolset_node_breaker.add_substation_column_to_bus(net_open)
    assert all(net_open.bus[substat_cols[0]] == net.bus[substat_cols[0]])

    pandapower_toolset_node_breaker.add_substation_column_to_bus(net, substation_col="substation")
    assert all(net.bus[substat_cols[1]].unique() == res_col)
    assert len(net.bus[net.bus[substat_cols[1]] == res_col[0]]) == 16
    assert len(net.bus[net.bus[substat_cols[1]] == res_col[1]]) == 16
    assert len(net.bus[net.bus[substat_cols[1]] == res_col[2]]) == 25

    pandapower_toolset_node_breaker.add_substation_column_to_bus(net, get_name_col="type")
    res_col = ["b", "b_0", ""]
    assert all(net.bus[substat_cols[0]].unique() == res_col)
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[0]]) == 16
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[1]]) == 16
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[2]]) == 25


def test_add_substation_column_to_bus_open_switch(pp_network_w_switches_open_coupler):
    net_open = pp_network_w_switches_open_coupler
    substat_cols = ["substat", "substation"]
    pandapower_toolset_node_breaker.add_substation_column_to_bus(
        net_open, only_closed_switches=False, substation_col=substat_cols[0]
    )
    pandapower_toolset_node_breaker.add_substation_column_to_bus(
        net_open,
        only_closed_switches=True,
        get_name_col="substat",
        substation_col=substat_cols[1],
    )
    res_col1 = ["Double Busbar 1", "Single Busbar", ""]
    res_col2 = ["Double Busbar 1", "Double Busbar 1_0", "Single Busbar", ""]
    assert all(net_open.bus[substat_cols[0]].unique() == res_col1)
    assert all(net_open.bus[substat_cols[1]].unique() == res_col2)


def test_add_substation_column_to_bus_two_subs(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    pandapower_toolset_node_breaker.add_substation_column_to_bus(net)
    res_col = ["Double Busbar 1", "Double Busbar Coupler 1", ""]
    substat_cols = ["substat", "substation"]
    assert all(net.bus[substat_cols[0]].unique() == res_col)
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[0]]) == 16
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[1]]) == 27
    assert len(net.bus[net.bus[substat_cols[0]] == res_col[2]]) == 25


def test_get_station_id_list(pp_network_w_switches):
    net = pp_network_w_switches
    res = pandapower_toolset_node_breaker.get_station_id_list(net.bus, substation_col="name")
    assert len(res) == len(net.bus)
    res = pandapower_toolset_node_breaker.get_station_id_list(net.bus, substation_col="substat")
    assert len(res) == 3
    assert len(res[0]) == 16
    assert len(res[1]) == 16
    assert len(res[2]) == 25
