# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import pypowsybl
import pytest
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_importer.network_graph.data_classes import (
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SubstationInformation,
    SwitchSchema,
)
from toop_engine_importer.network_graph.network_graph_helper_functions import add_suffix_to_duplicated_grid_model_id
from toop_engine_importer.network_graph.powsybl_station_to_graph import (
    get_helper_branches,
    get_node_assets,
    get_node_breaker_topology_graph,
    get_nodes,
    get_relevant_voltage_levels,
    get_station,
    get_station_list,
    get_switches,
    get_topology,
    node_breaker_topology_to_graph_data,
)
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import get_busbar_sections_with_in_service
from toop_engine_interfaces.asset_topology import Busbar, BusbarCoupler, Station, Topology
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    RelevantStationRules,
)


def test_node_breaker_topology_to_graph(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    graph_data = node_breaker_topology_to_graph_data(net, substation_information)
    assert isinstance(graph_data, NetworkGraphData)
    graph = get_node_breaker_topology_graph(graph_data)
    assert isinstance(graph, nx.Graph)
    nbt = net.get_node_breaker_topology("VL1")
    assert len(graph.nodes) == len(nbt.nodes)
    assert len(graph.edges) == len(nbt.switches)


def test_get_switches(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    nbt = net.get_node_breaker_topology("VL1")
    switches_df = get_switches(switches_df=nbt.switches)
    switches_df["in_service"] = True
    SwitchSchema.validate(switches_df)


def test_get_nodes(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    nbt = net.get_node_breaker_topology("VL1")
    switches_df = get_switches(switches_df=nbt.switches)
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service", "bus_id"])
    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        switches_df=switches_df,
        substation_info=substation_information,
    )
    NodeSchema.validate(nodes_df)


def test_get_helper_branches(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    nbt = net.get_node_breaker_topology("VL1")
    helper_branches = get_helper_branches(internal_connections_df=nbt.internal_connections)
    HelperBranchSchema.validate(helper_branches)


def test_get_node_assets(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    nbt = net.get_node_breaker_topology("VL1")
    names_dict = {
        "L1": "",
        "L2": "",
        "L3": "",
        "L4": "",
        "L5": "",
        "L6": "",
        "L7": "",
        "L8": "",
        "L9": "",
        "L10": "",
        "generator1": "",
        "generator2": "",
        "generator3": "",
        "load6": "",
        "load1": "",
        "load2": "",
    }
    all_names_df = pd.DataFrame.from_dict(names_dict, orient="index", columns=["name"])
    switches_df = get_switches(switches_df=nbt.switches)
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service", "bus_id"])
    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        switches_df=switches_df,
        substation_info=substation_information,
    )
    node_assets_df = get_node_assets(nodes_df=nodes_df, all_names_df=all_names_df)
    node_assets_df["in_service"] = True
    NodeAssetSchema.validate(node_assets_df)


def test_get_station(basic_node_breaker_network_powsybl_network_graph):
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL3"}
    station_info = SubstationInformation(**station_info)
    res = get_station(basic_node_breaker_network_powsybl_network_graph, "VL3_0", station_info)
    assert isinstance(res, Station)
    assert res.name == "Station_ID"
    assert res.grid_model_id == "VL3_0"
    assert res.region == "BE"
    assert res.voltage_level == 380

    busbars = res.busbars
    assert len(busbars) == 2
    assert busbars[0].grid_model_id == "BBS3_1"
    assert busbars[0].type == "busbar"
    assert busbars[0].name == "bus1"
    assert busbars[0].int_id == 0
    assert busbars[0].in_service is True
    assert busbars[1].grid_model_id == "BBS3_2"
    assert busbars[1].type == "busbar"
    assert busbars[1].name == "bus2"
    assert busbars[1].int_id == 1
    assert busbars[1].in_service is True

    couplers = res.couplers
    assert len(couplers) == 1
    assert couplers[0].grid_model_id == "VL3_BREAKER"
    assert couplers[0].type == "BREAKER"
    assert couplers[0].name == "VL3_BREAKER"
    assert couplers[0].busbar_from_id == 0
    assert couplers[0].busbar_to_id == 1
    assert not couplers[0].open
    assert couplers[0].in_service

    assets = res.assets
    assert len(assets) == 5
    assert assets[0].grid_model_id == "L3"
    assert assets[0].type == "LINE"
    assert assets[0].name == ""
    assert assets[0].in_service
    assert assets[0].branch_end is None
    assert assets[0].asset_bay.sl_switch_grid_model_id is None
    assert assets[0].asset_bay.dv_switch_grid_model_id == "L32_BREAKER"
    assert assets[0].asset_bay.sr_switch_grid_model_id == {
        "BBS3_1": "L32_DISCONNECTOR_3_0",
        "BBS3_2": "L32_DISCONNECTOR_3_1",
    }

    assert assets[1].grid_model_id == "L6"
    assert assets[1].type == "LINE"
    assert assets[1].name == ""
    assert assets[1].in_service
    assert assets[1].branch_end is None
    assert assets[1].asset_bay.sl_switch_grid_model_id is None
    assert assets[1].asset_bay.dv_switch_grid_model_id == "L62_BREAKER"
    assert assets[1].asset_bay.sr_switch_grid_model_id == {
        "BBS3_1": "L62_DISCONNECTOR_5_0",
        "BBS3_2": "L62_DISCONNECTOR_5_1",
    }

    assert assets[2].grid_model_id == "L7"
    assert assets[2].type == "LINE"
    assert assets[2].name == ""
    assert assets[2].in_service
    assert assets[2].branch_end is None
    assert assets[2].asset_bay.sl_switch_grid_model_id is None
    assert assets[2].asset_bay.dv_switch_grid_model_id == "L72_BREAKER"
    assert assets[2].asset_bay.sr_switch_grid_model_id == {
        "BBS3_1": "L72_DISCONNECTOR_7_0",
        "BBS3_2": "L72_DISCONNECTOR_7_1",
    }

    assert assets[3].grid_model_id == "L9"
    assert assets[3].type == "LINE"
    assert assets[3].name == ""
    assert assets[3].in_service
    assert assets[3].branch_end is None
    assert assets[3].asset_bay.sl_switch_grid_model_id is None
    assert assets[3].asset_bay.dv_switch_grid_model_id == "L91_BREAKER"
    assert assets[3].asset_bay.sr_switch_grid_model_id == {
        "BBS3_1": "L91_DISCONNECTOR_9_0",
        "BBS3_2": "L91_DISCONNECTOR_9_1",
    }

    assert assets[4].grid_model_id == "load2"
    assert assets[4].type == "LOAD"
    assert assets[4].name == ""
    assert assets[4].in_service
    assert assets[4].branch_end is None
    assert assets[4].asset_bay.sl_switch_grid_model_id is None
    assert assets[4].asset_bay.dv_switch_grid_model_id == "load2_BREAKER"
    assert assets[4].asset_bay.sr_switch_grid_model_id == {
        "BBS3_1": "load2_DISCONNECTOR_13_0",
        "BBS3_2": "load2_DISCONNECTOR_13_1",
    }

    assert len(res.asset_switching_table) == 2
    assert list(res.asset_switching_table[0]) == [True, True, False, True, False]
    assert list(res.asset_switching_table[1]) == [False, False, True, False, True]

    assert len(res.asset_connectivity) == 2
    assert list(res.asset_connectivity[0]) == [True, True, True, True, True]
    assert list(res.asset_connectivity[1]) == [True, True, True, True, True]


@pytest.mark.skip(reason="Known limitation in the current implementation")
def test_get_station_edge_cases_one_bay_two_assets(asset_topo_edge_cases_node_breaker_grid):
    net = asset_topo_edge_cases_node_breaker_grid
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL1_1", station_info)
    load_assets = [asset for asset in res.assets if "load" in asset.grid_model_id]
    assert len(load_assets) == 2, "Expected two loads"
    assert load_assets[0].asset_bay == load_assets[1].asset_bay, "Both loads should be in the same asset bay"


def test_get_station_edge_cases(asset_topo_edge_cases_node_breaker_grid):
    net = asset_topo_edge_cases_node_breaker_grid
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL1_1", station_info)
    # make sure the int ids match for the following tests
    expected_busbars = [
        Busbar(
            grid_model_id="VL1_1_1", type="busbar", name="VL1_1_1", int_id=0, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
        Busbar(
            grid_model_id="VL1_1_2", type="busbar", name="VL1_1_2", int_id=1, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
        Busbar(
            grid_model_id="VL1_1_3", type="busbar", name="VL1_1_3", int_id=2, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
        Busbar(grid_model_id="VL1_2_1", type="busbar", name="VL1_2_1", int_id=3, in_service=True, bus_branch_bus_id="VL1_1"),
        Busbar(grid_model_id="VL1_2_2", type="busbar", name="VL1_2_2", int_id=4, in_service=True, bus_branch_bus_id="VL1_1"),
        Busbar(grid_model_id="VL1_2_3", type="busbar", name="VL1_2_3", int_id=5, in_service=True, bus_branch_bus_id="VL1_1"),
        Busbar(grid_model_id="VL1_3_1", type="busbar", name="VL1_3_1", int_id=6, in_service=True, bus_branch_bus_id="VL1_1"),
        Busbar(grid_model_id="VL1_3_2", type="busbar", name="VL1_3_2", int_id=7, in_service=True, bus_branch_bus_id="VL1_1"),
        Busbar(grid_model_id="VL1_3_3", type="busbar", name="VL1_3_3", int_id=8, in_service=True, bus_branch_bus_id="VL1_1"),
    ]
    assert res.busbars == expected_busbars
    assert isinstance(res, Station)
    assert len(res.couplers) == 9
    assert len([coupler for coupler in res.couplers if coupler.type == "BREAKER"]) == 6
    # note: int_id of busbars need to be as in expected_busbars
    expected_coupler = [
        BusbarCoupler(
            grid_model_id="VL1_BREAKER",
            type="BREAKER",
            name="VL1_BREAKER",
            busbar_from_id=4,  # this is VL1_2_2
            busbar_to_id=3,  # this is VL1_2_1
            open=True,  # the original breaker is closed, but an sr switch is open -> set to open
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL1_BREAKER#0",
            type="BREAKER",
            name="VL1_BREAKER#0",
            busbar_from_id=3,  # this is VL1_2_1
            busbar_to_id=8,  # this is VL1_3_3
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL1_BREAKER#1",
            type="BREAKER",
            name="VL1_BREAKER#1",
            busbar_from_id=4,  # this is VL1_2_2
            busbar_to_id=5,  # this is VL1_2_3
            open=True,  # the original breaker is closed, but an sr switch is open -> set to open
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler
            grid_model_id="VL1_BREAKER_1_2",
            type="BREAKER",
            name="VL1_BREAKER_1_2",
            busbar_from_id=1,
            busbar_to_id=2,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler
            grid_model_id="VL1_BREAKER_2_2",
            type="BREAKER",
            name="VL1_BREAKER_2_2",
            busbar_from_id=4,
            busbar_to_id=5,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler
            grid_model_id="VL1_BREAKER_3_2",
            type="BREAKER",
            name="VL1_BREAKER_3_2",
            busbar_from_id=7,
            busbar_to_id=8,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler DISCONNECTOR
            grid_model_id="VL1_DISCONNECTOR_0_3",
            type="DISCONNECTOR",
            name="VL1_DISCONNECTOR_0_3",
            busbar_from_id=0,
            busbar_to_id=1,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler DISCONNECTOR
            grid_model_id="VL1_DISCONNECTOR_1_4",
            type="DISCONNECTOR",
            name="VL1_DISCONNECTOR_1_4",
            busbar_from_id=3,
            busbar_to_id=4,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # cross coupler DISCONNECTOR
            grid_model_id="VL1_DISCONNECTOR_2_5",
            type="DISCONNECTOR",
            name="VL1_DISCONNECTOR_2_5",
            busbar_from_id=6,
            busbar_to_id=7,
            open=False,
            in_service=True,
            asset_bay=None,
        ),
    ]
    for coupler in res.couplers:
        assert coupler in expected_coupler

    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL2"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL2_0", station_info)

    expected_busbars = [
        Busbar(grid_model_id="VL2_1_1", type="busbar", name="VL2_1_1", int_id=0, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(
            grid_model_id="VL2_1_2", type="busbar", name="VL2_1_2", int_id=1, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
        Busbar(
            grid_model_id="VL2_1_3", type="busbar", name="VL2_1_3", int_id=2, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
        Busbar(grid_model_id="VL2_1_4", type="busbar", name="VL2_1_4", int_id=3, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(grid_model_id="VL2_1_5", type="busbar", name="VL2_1_5", int_id=4, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(grid_model_id="VL2_1_6", type="busbar", name="VL2_1_6", int_id=5, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(grid_model_id="VL2_1_7", type="busbar", name="VL2_1_7", int_id=6, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(
            grid_model_id="VL2_1_8", type="busbar", name="VL2_1_8", int_id=7, in_service=True, bus_branch_bus_id="VL2_14"
        ),
        Busbar(grid_model_id="VL2_2_1", type="busbar", name="VL2_2_1", int_id=8, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(grid_model_id="VL2_2_2", type="busbar", name="VL2_2_2", int_id=9, in_service=True, bus_branch_bus_id="VL2_0"),
        Busbar(
            grid_model_id="VL2_2_3", type="busbar", name="VL2_2_3", int_id=10, in_service=True, bus_branch_bus_id="VL2_0"
        ),
        Busbar(
            grid_model_id="VL2_2_4", type="busbar", name="VL2_2_4", int_id=11, in_service=True, bus_branch_bus_id="VL2_0"
        ),
        Busbar(
            grid_model_id="VL2_2_5", type="busbar", name="VL2_2_5", int_id=12, in_service=True, bus_branch_bus_id="VL2_0"
        ),
        Busbar(
            grid_model_id="VL2_2_6", type="busbar", name="VL2_2_6", int_id=13, in_service=True, bus_branch_bus_id="VL2_0"
        ),
        Busbar(
            grid_model_id="VL2_2_7", type="busbar", name="VL2_2_7", int_id=14, in_service=True, bus_branch_bus_id="VL2_0"
        ),
        Busbar(
            grid_model_id="VL2_2_8", type="busbar", name="VL2_2_8", int_id=15, in_service=False, bus_branch_bus_id=""
        ),  # out of service busbar -> no bus_id
    ]
    assert res.busbars == expected_busbars

    expected_coupler = [
        BusbarCoupler(
            grid_model_id="BBS1_1-BBS1_4",
            type="DISCONNECTOR",
            name="BBS1_1-BBS1_4",
            busbar_from_id=0,  # this is VL2_1_1
            busbar_to_id=10,  # this is VL2_2_3
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="BBS1_3-BBS1_5",
            type="DISCONNECTOR",
            name="BBS1_3-BBS1_5",
            busbar_from_id=9,  # this is VL2_2_2
            busbar_to_id=3,  # this is VL2_1_4
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_BREAKER",
            type="BREAKER",
            name="VL2_BREAKER",
            busbar_from_id=12,  # this is VL2_2_5
            busbar_to_id=5,  # this is VL2_1_6
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_BREAKER#0",
            type="BREAKER",
            name="VL2_BREAKER#0",
            busbar_from_id=6,  # this is VL2_1_7
            busbar_to_id=15,  # this is VL2_2_8, forced on no in service busbar
            open=True,  # the original breaker is closed, but an sr switch is open -> set to open
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_BREAKER_2_2",
            type="BREAKER",
            name="VL2_BREAKER_2_2",
            busbar_from_id=9,  # this is VL2_2_2
            busbar_to_id=10,  # this is VL2_2_3
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_10_12",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_10_12",
            busbar_from_id=5,  # this is VL2_1_6
            busbar_to_id=6,  # this is VL2_1_7
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_11_13",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_11_13",
            busbar_from_id=13,  # this is VL2_2_6
            busbar_to_id=14,  # this is VL2_2_7
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_1_3",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_1_3",
            busbar_from_id=8,  # this is VL2_2_1
            busbar_to_id=9,  # this is VL2_2_2
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_5_7",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_5_7",
            busbar_from_id=10,  # this is VL2_2_4
            busbar_to_id=11,  # this is VL2_2_5
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_6_8",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_6_8",
            busbar_from_id=3,  # this is VL2_1_4
            busbar_to_id=4,  # this is VL2_1_5
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_7_9",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_7_9",
            busbar_from_id=11,  # this is VL2_2_4
            busbar_to_id=12,  # this is VL2_2_5
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_8_10",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_8_10",
            busbar_from_id=4,  # this is VL2_1_5
            busbar_to_id=5,  # this is VL2_1_6
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(
            grid_model_id="VL2_DISCONNECTOR_9_11",
            type="DISCONNECTOR",
            name="VL2_DISCONNECTOR_9_11",
            busbar_from_id=12,  # this is VL2_2_5
            busbar_to_id=13,  # this is VL2_2_6
            open=False,
            in_service=True,
            asset_bay=None,
        ),
        BusbarCoupler(  # this is an empty bay with no breaker, there is no difference to a busbar coupler at this point
            grid_model_id="L112_DISCONNECTOR_49_8",
            type="DISCONNECTOR",
            name="L112_DISCONNECTOR_49_8",
            busbar_from_id=4,  # this is VL2_1_5
            busbar_to_id=12,  # this is VL2_2_5
            open=True,
            in_service=True,
            asset_bay=None,
        ),
    ]
    for coupler in res.couplers:
        assert coupler in expected_coupler, f"Coupler {coupler} not in expected couplers"

    assert len(res.assets) == 12
    L9_assets = [asset for asset in res.assets if asset.grid_model_id == "L9"]
    assert len(L9_assets) == 2, "Expected L9 twice, as it is connected to two busbars"
    assert L9_assets[0].asset_bay != L9_assets[1].asset_bay, "Expected different asset bays for L9 assets"

    expected_switching_table = [
        [True, False, False, False, False, False, False, False, False, False, False, True],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, True, True, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, True, True, False, False, False],
        [False, True, False, False, False, False, False, False, False, False, True, False],
        [False, False, True, False, False, False, True, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, True, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
    ]

    assert (res.asset_switching_table == expected_switching_table).all(), (
        "Asset switching table does not match expected values"
    )


def test_get_topo_integration(basic_node_breaker_network_powsybl_network_graph):
    net = basic_node_breaker_network_powsybl_network_graph
    importer_parameters = CgmesImporterParameters(
        grid_model_file=Path("cgmes_file.zip"),
        data_folder="data_folder",
        area_settings=AreaSettings(cutoff_voltage=220, control_area=["BE"], view_area=["BE"], nminus1_area=["BE"]),
    )
    lf_result, *_ = pypowsybl.loadflow.run_dc(net)
    network_masks = powsybl_masks.make_masks(
        network=net,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    relevant_voltage_level_with_region = get_relevant_voltage_levels(network=net, network_masks=network_masks)
    expected = ["VL1", "VL2", "VL3", "VL4", "VL5"]
    assert all(net.get_voltage_levels().index == expected)
    # VL4 has only 3 branches, VL5 has only 1 busbar
    assert all(relevant_voltage_level_with_region["voltage_level_id"] == expected[1:3])
    assert all(relevant_voltage_level_with_region.index == ["VL2_0", "VL3_0"])

    res = get_station_list(network=net, relevant_voltage_level_with_region=relevant_voltage_level_with_region)
    assert len(res) == 2
    assert all([isinstance(station, Station) for station in res])

    timestamp = datetime.datetime.now()
    res = get_topology(network=net, network_masks=network_masks, importer_parameters=importer_parameters)
    assert isinstance(res, Topology)
    assert len(res.stations) == 2
    assert res.topology_id == "cgmes_file.zip"
    assert res.grid_model_file == "cgmes_file.zip"
    assert res.timestamp - timestamp < datetime.timedelta(seconds=3)


def make_node_assets_df(rows):
    df = pd.DataFrame(rows)
    df = NodeAssetSchema.validate(df)
    return df


def test_add_suffix_to_duplicated_grid_model_id():
    rows = [
        {"grid_model_id": "A", "foreign_id": "A", "node": 1, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "B", "foreign_id": "B", "node": 2, "asset_type": "LINE", "in_service": True},
    ]
    df = make_node_assets_df(rows)
    add_suffix_to_duplicated_grid_model_id(df)
    assert set(df["grid_model_id"]) == {"A", "B"}

    rows = [
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 1, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 2, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L2", "foreign_id": "L2", "node": 3, "asset_type": "LINE", "in_service": True},
    ]
    df = make_node_assets_df(rows)
    add_suffix_to_duplicated_grid_model_id(df)
    l1_ids = sorted(df[df["foreign_id"] == "L1"]["grid_model_id"])
    assert l1_ids == ["L1_FROM", "L1_TO"]
    assert "L2" in df["grid_model_id"].values
    NodeAssetSchema.validate(df)

    rows = [
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 1, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 2, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L2", "foreign_id": "L2", "node": 3, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L2", "foreign_id": "L2", "node": 4, "asset_type": "LINE", "in_service": True},
    ]
    df = make_node_assets_df(rows)
    add_suffix_to_duplicated_grid_model_id(df)
    l1_ids = sorted(df[df["foreign_id"] == "L1"]["grid_model_id"])
    l2_ids = sorted(df[df["foreign_id"] == "L2"]["grid_model_id"])
    assert l1_ids == ["L1_FROM", "L1_TO"]
    assert l2_ids == ["L2_FROM", "L2_TO"]

    rows = [
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 1, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 2, "asset_type": "LINE", "in_service": True},
        {"grid_model_id": "L1", "foreign_id": "L1", "node": 3, "asset_type": "LINE", "in_service": True},
    ]
    df = make_node_assets_df(rows)
    with pytest.raises(AssertionError):
        add_suffix_to_duplicated_grid_model_id(df)


def test_create_complex_grid_battery_hvdc_svc_3w_trafo_asset_topo():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

    importer_parameters = CgmesImporterParameters(
        grid_model_file=Path("cgmes_file.zip"),
        data_folder="data_folder",
        area_settings=AreaSettings(cutoff_voltage=1, control_area=[""], view_area=[""], nminus1_area=[""]),
        relevant_station_rules=RelevantStationRules(
            min_busbars=2,
            min_connected_branches=4,
            min_connected_elements=4,
        ),
    )

    lf_result, *_ = pypowsybl.loadflow.run_dc(net)
    network_masks = powsybl_masks.make_masks(
        network=net,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    relevant_voltage_level_with_region = get_relevant_voltage_levels(network=net, network_masks=network_masks)
    expected = [
        "VL_3W_HV",
        "VL_3W_MV",
        "VL_2W_MV_LV_MV",
        "VL_MV_svc",
        "VL_MV",
        "VL_2W_MV_HV_MV",
        "VL_2W_MV_HV_HV",
        "VL_HV_vsc",
        "VL_MV_load",
    ]
    # 'VL_HV_gen' not included as it is the slack

    for vl in expected:
        assert vl in relevant_voltage_level_with_region["voltage_level_id"].values, f"Expected voltage level {vl} not found"

    res = get_station_list(network=net, relevant_voltage_level_with_region=relevant_voltage_level_with_region)
    assert len(res) >= len(expected)
    assert all([isinstance(station, Station) for station in res])

    res = get_topology(network=net, network_masks=network_masks, importer_parameters=importer_parameters)
    assert isinstance(res, Topology)
    assert len(res.stations) == len(expected)
