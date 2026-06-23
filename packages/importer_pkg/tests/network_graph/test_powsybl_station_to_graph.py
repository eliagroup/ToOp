# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pypowsybl
import pytest
from pypowsybl.network import Network
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
from toop_engine_interfaces.asset_topology.asset_topology import Topology
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    RelevantStationRules,
)


def all_station_connections(station: MaterializedStation):
    return [*station.branch_connections, *station.injection_connections]


def all_station_switching_table(station: MaterializedStation):
    return np.concatenate([station.branch_switching_table, station.injection_switching_table], axis=1)


def all_station_connectivity(station: MaterializedStation):
    return np.concatenate([station.branch_connectivity, station.injection_connectivity], axis=1)


def test_node_breaker_topology_to_graph(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    graph_data = node_breaker_topology_to_graph_data(net, substation_information)
    assert isinstance(graph_data, NetworkGraphData)
    graph = get_node_breaker_topology_graph(graph_data)
    assert isinstance(graph, nx.Graph)
    nbt = net.get_node_breaker_topology("VL1")
    assert len(graph.nodes) == len(nbt.nodes)
    assert len(graph.edges) == len(nbt.switches)


def test_get_switches(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    nbt = net.get_node_breaker_topology("VL1")
    switches_df = get_switches(switches_df=nbt.switches)
    switches_df["in_service"] = True
    SwitchSchema.validate(switches_df)


def test_get_nodes(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
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


def test_get_helper_branches(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    nbt = net.get_node_breaker_topology("VL1")
    helper_branches = get_helper_branches(internal_connections_df=nbt.internal_connections)
    HelperBranchSchema.validate(helper_branches)


def test_get_node_assets(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
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
    all_names_df = pd.DataFrame.from_dict(names_dict, orient="index", columns=["name"])["name"]
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


def test_get_station(basic_node_breaker_network_powsybl_grid: Network):
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL3"}
    station_info = SubstationInformation(**station_info)
    res = get_station(basic_node_breaker_network_powsybl_grid, "VL3_0", station_info)
    assert isinstance(res, MaterializedStation)
    assert res.name == "Station_ID"
    assert res.grid_model_id == "VL3_0"
    assert res.region == "BE"
    assert res.voltage_level == 380

    busbars = res.busbars
    assert len(busbars) == 2
    assert busbars[0].grid_model_id == "BBS3_1"
    assert busbars[0].busbar_type == "busbar"
    assert busbars[0].name == "bus1"
    assert busbars[0].int_id == 0
    assert busbars[0].in_service is True
    assert busbars[1].grid_model_id == "BBS3_2"
    assert busbars[1].busbar_type == "busbar"
    assert busbars[1].name == "bus2"
    assert busbars[1].int_id == 1
    assert busbars[1].in_service is True

    couplers = res.couplers
    assert len(couplers) == 1
    assert couplers[0].grid_model_id == "VL3_BREAKER"
    assert couplers[0].coupler_type == "BREAKER"
    assert couplers[0].name == "VL3_BREAKER"
    assert couplers[0].busbar_from_id == 0
    assert couplers[0].busbar_to_id == 1
    assert not couplers[0].open
    assert couplers[0].in_service

    assets = [asset_connection.asset for asset_connection in all_station_connections(res)]
    asset_bays = [asset_connection.asset_bay for asset_connection in all_station_connections(res)]
    asset_terminals = [asset_connection.branch_end for asset_connection in all_station_connections(res)]
    assert len(assets) == 5
    assert asset_terminals == [None] * len(assets)
    assert assets[0].grid_model_id == "L3"
    assert assets[0].asset_type == "LINE"
    assert assets[0].name == ""
    assert assets[0].in_service
    assert asset_bays[0].sl_switch_grid_model_id is None
    assert asset_bays[0].dv_switch_grid_model_id == "L32_BREAKER"
    assert asset_bays[0].sr_switch_grid_model_id == {
        "BBS3_1": "L32_DISCONNECTOR_3_0",
        "BBS3_2": "L32_DISCONNECTOR_3_1",
    }

    assert assets[1].grid_model_id == "L6"
    assert assets[1].asset_type == "LINE"
    assert assets[1].name == ""
    assert assets[1].in_service
    assert asset_bays[1].sl_switch_grid_model_id is None
    assert asset_bays[1].dv_switch_grid_model_id == "L62_BREAKER"
    assert asset_bays[1].sr_switch_grid_model_id == {
        "BBS3_1": "L62_DISCONNECTOR_5_0",
        "BBS3_2": "L62_DISCONNECTOR_5_1",
    }

    assert assets[2].grid_model_id == "L7"
    assert assets[2].asset_type == "LINE"
    assert assets[2].name == ""
    assert assets[2].in_service
    assert asset_bays[2].sl_switch_grid_model_id is None
    assert asset_bays[2].dv_switch_grid_model_id == "L72_BREAKER"
    assert asset_bays[2].sr_switch_grid_model_id == {
        "BBS3_1": "L72_DISCONNECTOR_7_0",
        "BBS3_2": "L72_DISCONNECTOR_7_1",
    }

    assert assets[3].grid_model_id == "L9"
    assert assets[3].asset_type == "LINE"
    assert assets[3].name == ""
    assert assets[3].in_service
    assert asset_bays[3].sl_switch_grid_model_id is None
    assert asset_bays[3].dv_switch_grid_model_id == "L91_BREAKER"
    assert asset_bays[3].sr_switch_grid_model_id == {
        "BBS3_1": "L91_DISCONNECTOR_9_0",
        "BBS3_2": "L91_DISCONNECTOR_9_1",
    }

    assert assets[4].grid_model_id == "load2"
    assert assets[4].asset_type == "LOAD"
    assert assets[4].name == ""
    assert assets[4].in_service
    assert asset_bays[4].sl_switch_grid_model_id is None
    assert asset_bays[4].dv_switch_grid_model_id == "load2_BREAKER"
    assert asset_bays[4].sr_switch_grid_model_id == {
        "BBS3_1": "load2_DISCONNECTOR_13_0",
        "BBS3_2": "load2_DISCONNECTOR_13_1",
    }

    switching_table = all_station_switching_table(res)
    assert len(switching_table) == 2
    assert list(switching_table[0]) == [True, True, False, True, False]
    assert list(switching_table[1]) == [False, False, True, False, True]

    connectivity = all_station_connectivity(res)
    assert len(connectivity) == 2
    assert list(connectivity[0]) == [True, True, True, True, True]
    assert list(connectivity[1]) == [True, True, True, True, True]


@pytest.mark.skip(reason="Known limitation in the current implementation")
def test_get_station_edge_cases_one_bay_two_assets(asset_topo_edge_cases_node_breaker_grid):
    net = asset_topo_edge_cases_node_breaker_grid
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL1_1", station_info)
    load_assets = [
        asset_connection.asset
        for asset_connection in all_station_connections(res)
        if "load" in asset_connection.asset.grid_model_id
    ]
    load_asset_bays = [
        asset_connection.asset_bay
        for asset_connection in all_station_connections(res)
        if "load" in asset_connection.asset.grid_model_id
    ]
    assert len(load_assets) == 2, "Expected two loads"
    assert load_asset_bays[0] == load_asset_bays[1], "Both loads should be in the same asset bay"


def test_get_station_edge_cases(asset_topo_edge_cases_node_breaker_grid):
    net = asset_topo_edge_cases_node_breaker_grid
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL1_1", station_info)
    assert [busbar.grid_model_id for busbar in res.busbars] == [
        "VL1_1_1",
        "VL1_1_2",
        "VL1_1_3",
        "VL1_2_1",
        "VL1_2_2",
        "VL1_2_3",
        "VL1_3_1",
        "VL1_3_2",
        "VL1_3_3",
    ]
    assert all(busbar.bus_branch_bus_id in {"", "VL1_1"} for busbar in res.busbars)
    assert [busbar.int_id for busbar in res.busbars] == list(range(len(res.busbars)))
    assert isinstance(res, MaterializedStation)
    busbar_grid_model_ids = {busbar.grid_model_id for busbar in res.busbars}
    busbar_int_ids = {busbar.int_id for busbar in res.busbars}
    assert all(
        coupler.busbar_from_id in busbar_int_ids and coupler.busbar_to_id in busbar_int_ids for coupler in res.couplers
    )
    assert len([coupler for coupler in res.couplers if coupler.coupler_type == "BREAKER"]) > 0
    assert all_station_switching_table(res).shape == (len(res.busbars), len(all_station_connections(res)))
    assert all_station_connectivity(res).shape == (len(res.busbars), len(all_station_connections(res)))

    for asset_bay in [asset_connection.asset_bay for asset_connection in all_station_connections(res)]:
        if asset_bay is None:
            continue
        assert set(asset_bay.sr_switch_grid_model_id) <= busbar_grid_model_ids

    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL2"}
    station_info = SubstationInformation(**station_info)
    res = get_station(net, "VL2_0", station_info)

    expected_busbar_grid_model_ids = [
        "VL2_1_1",
        "VL2_1_4",
        "VL2_1_5",
        "VL2_1_6",
        "VL2_1_7",
        "VL2_2_1",
        "VL2_2_2",
        "VL2_2_3",
        "VL2_2_4",
        "VL2_2_5",
        "VL2_2_6",
        "VL2_2_7",
        "VL2_2_8",
    ]
    assert [busbar.grid_model_id for busbar in res.busbars] == expected_busbar_grid_model_ids
    assert all(busbar.bus_branch_bus_id in {"", "VL2_0"} for busbar in res.busbars)
    assert [busbar.int_id for busbar in res.busbars] == list(range(len(res.busbars)))

    busbar_grid_model_ids = {busbar.grid_model_id for busbar in res.busbars}
    busbar_int_ids = {busbar.int_id for busbar in res.busbars}
    assert all(
        coupler.busbar_from_id in busbar_int_ids and coupler.busbar_to_id in busbar_int_ids for coupler in res.couplers
    )

    assert all_station_switching_table(res).shape == (len(res.busbars), len(all_station_connections(res)))
    assert all_station_connectivity(res).shape == (len(res.busbars), len(all_station_connections(res)))
    assert all_station_connectivity(res).any(axis=0).all()

    for asset_bay in [asset_connection.asset_bay for asset_connection in all_station_connections(res)]:
        if asset_bay is None:
            continue
        assert set(asset_bay.sr_switch_grid_model_id) <= busbar_grid_model_ids


def test_get_topo_integration(basic_node_breaker_network_powsybl_grid: Network):
    net = basic_node_breaker_network_powsybl_grid
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
    assert all([isinstance(station, MaterializedStation) for station in res])

    timestamp = datetime.datetime.now()
    res = get_topology(network=net, network_masks=network_masks, importer_parameters=importer_parameters)
    assert isinstance(res, Topology)
    assert len(res.materialize_stations()) == 2
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
    assert all([isinstance(station, MaterializedStation) for station in res])

    res = get_topology(network=net, network_masks=network_masks, importer_parameters=importer_parameters)
    assert isinstance(res, Topology)
    assert len(res.materialize_stations()) == len(expected)
