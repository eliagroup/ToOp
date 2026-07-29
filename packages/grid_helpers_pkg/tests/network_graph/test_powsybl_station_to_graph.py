# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
import pypowsybl
import pytest
from pypowsybl.network import Network
from toop_engine_grid_helpers.network_graph.data_classes import (
    BusbarConnectionInfo,
    HelperBranchSchema,
    NetworkGraphData,
    NodeAssetSchema,
    NodeSchema,
    SubstationInformation,
    SwitchSchema,
)
from toop_engine_grid_helpers.network_graph.graph_to_asset_topo import (
    get_station_connection_tables,
    remove_double_connections,
)
from toop_engine_grid_helpers.network_graph.network_graph_helper_functions import add_suffix_to_duplicated_grid_model_id
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import materialize_stations_from_network_state
from toop_engine_grid_helpers.powsybl.powsybl_station_to_graph import (
    _build_master_station_from_busbar_group,
    _build_station_connectivity_by_asset_type,
    _get_station_asset_connections,
    _get_station_busbar_view,
    _get_station_topology_frames,
    _get_structural_busbar_groups,
    get_helper_branches,
    get_node_assets,
    get_node_breaker_topology_graph,
    get_node_breaker_topology_master_data,
    get_nodes,
    get_relevant_voltage_levels,
    get_switches,
    node_breaker_topology_to_graph_data,
)
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import get_busbar_sections_with_in_service
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    RelevantStationRules,
)


def all_station_connections(station: MaterializedStation) -> list[MaterializedAssetConnection]:
    return [*station.branch_connections, *station.injection_connections]


def all_station_switching_table(station: MaterializedStation) -> np.ndarray:
    return np.concatenate([station.branch_switching_table, station.injection_switching_table], axis=1)


def all_station_connectivity(station: MaterializedStation) -> np.ndarray:
    return np.concatenate([station.branch_connectivity, station.injection_connectivity], axis=1)


def build_station_from_bus_id(network: Network, bus_id: str, station_info: SubstationInformation) -> MaterializedStation:
    """Build one materialized station view from a selected bus id for graph-conversion tests."""
    graph_data = node_breaker_topology_to_graph_data(network, substation_info=station_info)
    graph = get_node_breaker_topology_graph(graph_data)
    _busbar_df, selected_busbar_ids, _busbar_connection_info = _get_station_busbar_view(
        graph=graph,
        graph_data=graph_data,
        bus_id=bus_id,
        substation_id=station_info.name,
    )
    station, branch_assets, injection_assets, asset_bays = _build_master_station_from_busbar_group(
        network=network,
        station_info=station_info,
        selected_busbar_ids=selected_busbar_ids,
        station_grid_model_id=bus_id,
    )
    busbar_df, _coupler_df, busbar_connection_info, switchable_assets_df, asset_bays_by_asset_id, station_logs = (
        _get_station_topology_frames(
            network=network,
            station_info=station_info,
            selected_busbar_ids=selected_busbar_ids,
            station_grid_model_id=bus_id,
        )
    )
    _branch_assets, _injection_assets, branch_connections, injection_connections, branch_mask, _asset_bays = (
        _get_station_asset_connections(
            network=network,
            station_info=station_info,
            busbar_df=busbar_df,
            switchable_assets_df=switchable_assets_df,
            asset_bays_by_asset_id=asset_bays_by_asset_id,
        )
    )
    branch_connectivity, injection_connectivity = _build_station_connectivity_by_asset_type(
        busbar_connection_info=busbar_connection_info,
        busbar_df=busbar_df,
        switchable_assets_df=switchable_assets_df,
        branch_mask=branch_mask,
    )
    _asset_connectivity, asset_switching_table, _busbar_connectivity, _busbar_switching_table = (
        get_station_connection_tables(
            busbar_connection_info,
            busbar_df=busbar_df,
            switchable_assets_df=switchable_assets_df,
        )
    )
    asset_switching_table = remove_double_connections(asset_switching_table, substation_id=station_info.name)
    branch_switching_table = asset_switching_table[:, branch_mask]
    injection_switching_table = asset_switching_table[:, [not is_branch for is_branch in branch_mask]]
    asset_bay_by_id = {asset_bay.asset_bay_id: asset_bay for asset_bay in asset_bays if asset_bay.asset_bay_id is not None}
    branch_asset_by_id = {asset.grid_model_id: asset for asset in branch_assets}
    injection_asset_by_id = {asset.grid_model_id: asset for asset in injection_assets}
    return MaterializedStation(
        bus_group_id=station.bus_group_id,
        voltage_level_id=station.voltage_level_id,
        name=station.name,
        station_type=station.station_type,
        region=station.region,
        voltage_level=station.voltage_level,
        busbars=[busbar.model_copy(deep=True) for busbar in station.busbars],
        couplers=[coupler.model_copy(deep=True) for coupler in station.couplers],
        branch_connections=[
            MaterializedAssetConnection(
                asset=branch_asset_by_id[connection.asset_id].model_copy(deep=True),
                branch_end=connection.branch_end,
                asset_bay=(
                    asset_bay_by_id[connection.asset_bay_id].model_copy(deep=True)
                    if connection.asset_bay_id is not None
                    else None
                ),
            )
            for connection in branch_connections
        ],
        injection_connections=[
            MaterializedAssetConnection(
                asset=injection_asset_by_id[connection.asset_id].model_copy(deep=True),
                branch_end=connection.branch_end,
                asset_bay=(
                    asset_bay_by_id[connection.asset_bay_id].model_copy(deep=True)
                    if connection.asset_bay_id is not None
                    else None
                ),
            )
            for connection in injection_connections
        ],
        branch_switching_table=branch_switching_table,
        injection_switching_table=injection_switching_table,
        branch_connectivity=branch_connectivity,
        injection_connectivity=injection_connectivity,
        model_log=station_logs,
    )


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
    bbt = net.get_bus_breaker_topology("VL1")
    bus_breaker_view_buses_df = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    switches_df = get_switches(switches_df=nbt.switches)
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])
    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        bus_breaker_elements_df=bbt.elements,
        switches_df=switches_df,
        bus_breaker_view_buses_df=bus_breaker_view_buses_df,
        substation_info=substation_information,
    )
    NodeSchema.validate(nodes_df)


def test_node_schema_validate_rejects_wrong_dtype(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    nbt = net.get_node_breaker_topology("VL1")
    bbt = net.get_bus_breaker_topology("VL1")
    bus_breaker_view_buses_df = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    switches_df = get_switches(switches_df=nbt.switches)
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])

    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        bus_breaker_elements_df=bbt.elements,
        bus_breaker_view_buses_df=bus_breaker_view_buses_df,
        switches_df=switches_df,
        substation_info=substation_information,
    )
    nodes_df["in_service"] = nodes_df["in_service"].astype(object)

    with pa.config.config_context(validation_enabled=True):
        with pytest.raises(pa.errors.SchemaError):
            NodeSchema.validate(nodes_df)


def test_get_helper_branches(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    nbt = net.get_node_breaker_topology("VL1")
    helper_branches = get_helper_branches(internal_connections_df=nbt.internal_connections)
    HelperBranchSchema.validate(helper_branches)


def test_get_structural_busbar_groups_is_independent_of_runtime_switch_state():
    """Verify that structural busbar grouping ignores runtime switch states."""
    busbar_connection_info = {
        "BBS1": BusbarConnectionInfo(connectable_busbars=["BBS1", "BBS2"]),
        "BBS2": BusbarConnectionInfo(connectable_busbars=["BBS1", "BBS2"]),
        "BBS3": BusbarConnectionInfo(connectable_busbars=["BBS3"]),
        "BBS4": BusbarConnectionInfo(connectable_busbars=["BBS4", "BBS5"]),
        "BBS5": BusbarConnectionInfo(connectable_busbars=["BBS4", "BBS5"]),
    }

    result = _get_structural_busbar_groups(
        full_busbar_connection_info=busbar_connection_info,
        allowed_busbar_ids={"BBS5", "BBS3", "BBS2", "BBS4", "BBS1"},
    )

    assert result == [
        {"BBS1", "BBS2"},
        {"BBS3"},
        {"BBS4", "BBS5"},
    ]


def test_get_node_assets(basic_node_breaker_network_powsybl_grid):
    net = basic_node_breaker_network_powsybl_grid
    nbt = net.get_node_breaker_topology("VL1")
    bbt = net.get_bus_breaker_topology("VL1")
    branches_df = net.get_branches(attributes=["connected1", "connected2"])
    boundary_line_tie_ids = net.get_boundary_lines(attributes=["tie_line_id"])["tie_line_id"]
    injections_df = net.get_injections(attributes=["connected"])

    asset_in_service = pd.concat(
        [
            (branches_df["connected1"].fillna(False) & branches_df["connected2"].fillna(False)).rename("in_service"),
            injections_df["connected"].fillna(False).rename("in_service"),
        ]
    )
    asset_in_service.loc["L1"] = False
    asset_in_service.loc["generator1"] = False
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
    bus_breaker_view_buses_df = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    substation_dict = {"name": "Station1", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL1"}
    substation_information = SubstationInformation(**substation_dict)
    busbar_sections_names_df = get_busbar_sections_with_in_service(network=net, attributes=["name", "in_service"])
    nodes_df = get_nodes(
        busbar_sections_names_df=busbar_sections_names_df,
        nodes_df=nbt.nodes,
        bus_breaker_elements_df=bbt.elements,
        bus_breaker_view_buses_df=bus_breaker_view_buses_df,
        switches_df=switches_df,
        substation_info=substation_information,
    )
    node_assets_df = get_node_assets(
        nodes_df=nodes_df,
        all_names_df=all_names_df,
        asset_in_service=asset_in_service,
        boundary_line_tie_ids=boundary_line_tie_ids,
    )

    assert not node_assets_df.loc[node_assets_df["grid_model_id"] == "L1", "in_service"].item()
    assert not node_assets_df.loc[node_assets_df["grid_model_id"] == "generator1", "in_service"].item()
    assert node_assets_df.loc[node_assets_df["grid_model_id"] == "L2", "in_service"].item()
    NodeAssetSchema.validate(node_assets_df)


def test_get_station(basic_node_breaker_network_powsybl_grid: Network):
    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL3"}
    station_info = SubstationInformation(**station_info)
    res = build_station_from_bus_id(basic_node_breaker_network_powsybl_grid, "VL3_0", station_info)
    assert isinstance(res, MaterializedStation)
    assert res.name == "Station_ID"
    assert res.bus_group_id == "VL3_0"
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
    assert asset_terminals == ["to", "to", "to", "from", None]
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
    res = build_station_from_bus_id(net, "VL1_1", station_info)
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
    res = build_station_from_bus_id(net, "VL1_1", station_info)
    # make sure the int ids match for the following tests
    expected_busbars = [
        ("VL1_1_1", "busbar", 0, True, "", "VL1_0"),
        ("VL1_1_2", "busbar", 1, True, "", "VL1_0"),
        ("VL1_1_3", "busbar", 2, True, "", "VL1_6"),
        ("VL1_2_1", "busbar", 3, True, "VL1_1", "VL1_1"),
        ("VL1_2_2", "busbar", 4, True, "VL1_1", "VL1_1"),
        ("VL1_2_3", "busbar", 5, True, "VL1_1", "VL1_7"),
        ("VL1_3_1", "busbar", 6, True, "VL1_1", "VL1_2"),
        ("VL1_3_2", "busbar", 7, True, "VL1_1", "VL1_2"),
        ("VL1_3_3", "busbar", 8, True, "VL1_1", "VL1_8"),
    ]
    assert [
        (
            busbar.grid_model_id,
            busbar.busbar_type,
            busbar.int_id,
            busbar.in_service,
            busbar.bus_branch_bus_id,
            busbar.bus_breaker_bus_id,
        )
        for busbar in res.busbars
    ] == expected_busbars
    assert isinstance(res, MaterializedStation)
    assert len(res.couplers) == 9
    assert len([coupler for coupler in res.couplers if coupler.coupler_type == "BREAKER"]) == 6
    # note: int_id of busbars need to be as in expected_busbars
    expected_coupler = [
        ("VL1_BREAKER", "BREAKER", 4, 3, False, True),
        ("VL1_BREAKER#0", "BREAKER", 3, 8, False, True),
        ("VL1_BREAKER#1", "BREAKER", 4, 5, False, True),
        ("VL1_BREAKER_1_2", "BREAKER", 1, 2, False, True),
        ("VL1_BREAKER_2_2", "BREAKER", 4, 5, False, True),
        ("VL1_BREAKER_3_2", "BREAKER", 7, 8, False, True),
        ("VL1_DISCONNECTOR_0_3", "DISCONNECTOR", 0, 1, False, True),
        ("VL1_DISCONNECTOR_1_4", "DISCONNECTOR", 3, 4, False, True),
        ("VL1_DISCONNECTOR_2_5", "DISCONNECTOR", 6, 7, False, True),
    ]
    assert [
        (
            coupler.grid_model_id,
            coupler.coupler_type,
            coupler.busbar_from_id,
            coupler.busbar_to_id,
            coupler.open,
            coupler.in_service,
        )
        for coupler in res.couplers
    ] == expected_coupler

    station_info = {"name": "Station_ID", "region": "BE", "nominal_v": 380, "voltage_level_id": "VL2"}
    station_info = SubstationInformation(**station_info)
    res = build_station_from_bus_id(net, "VL2_0", station_info)

    expected_busbars = [
        ("VL2_1_1", "busbar", 0, True, "VL2_0", "VL2_0"),
        ("VL2_1_4", "busbar", 1, True, "VL2_0", "VL2_1"),
        ("VL2_1_5", "busbar", 2, True, "VL2_0", "VL2_1"),
        ("VL2_1_6", "busbar", 3, True, "VL2_0", "VL2_1"),
        ("VL2_1_7", "busbar", 4, True, "VL2_0", "VL2_1"),
        ("VL2_2_1", "busbar", 5, True, "VL2_0", "VL2_1"),
        ("VL2_2_2", "busbar", 6, True, "VL2_0", "VL2_1"),
        ("VL2_2_3", "busbar", 7, True, "VL2_0", "VL2_0"),
        ("VL2_2_4", "busbar", 8, True, "VL2_0", "VL2_0"),
        ("VL2_2_5", "busbar", 9, True, "VL2_0", "VL2_0"),
        ("VL2_2_6", "busbar", 10, True, "VL2_0", "VL2_0"),
        ("VL2_2_7", "busbar", 11, True, "VL2_0", "VL2_0"),
        ("VL2_2_8", "busbar", 12, True, "", "VL2_15"),
    ]
    assert [
        (
            busbar.grid_model_id,
            busbar.busbar_type,
            busbar.int_id,
            busbar.in_service,
            busbar.bus_branch_bus_id,
            busbar.bus_breaker_bus_id,
        )
        for busbar in res.busbars
    ] == expected_busbars

    expected_coupler = [
        ("BBS1_1-BBS1_4", "DISCONNECTOR", 0, 7, False, True),
        ("BBS1_3-BBS1_5", "DISCONNECTOR", 6, 1, False, True),
        ("L112_DISCONNECTOR_49_8", "DISCONNECTOR", 2, 9, False, True),
        ("VL2_BREAKER", "BREAKER", 9, 3, False, True),
        ("VL2_BREAKER#0", "BREAKER", 4, 12, False, True),
        ("VL2_BREAKER_2_2", "BREAKER", 6, 7, False, True),
        ("VL2_DISCONNECTOR_10_12", "DISCONNECTOR", 3, 4, False, True),
        ("VL2_DISCONNECTOR_11_13", "DISCONNECTOR", 10, 11, False, True),
        ("VL2_DISCONNECTOR_1_3", "DISCONNECTOR", 5, 6, False, True),
        ("VL2_DISCONNECTOR_5_7", "DISCONNECTOR", 7, 8, False, True),
        ("VL2_DISCONNECTOR_6_8", "DISCONNECTOR", 1, 2, False, True),
        ("VL2_DISCONNECTOR_7_9", "DISCONNECTOR", 8, 9, False, True),
        ("VL2_DISCONNECTOR_8_10", "DISCONNECTOR", 2, 3, False, True),
        ("VL2_DISCONNECTOR_9_11", "DISCONNECTOR", 9, 10, False, True),
    ]
    assert [
        (
            coupler.grid_model_id,
            coupler.coupler_type,
            coupler.busbar_from_id,
            coupler.busbar_to_id,
            coupler.open,
            coupler.in_service,
        )
        for coupler in res.couplers
    ] == expected_coupler

    assert len(res.assets) == 12
    l9_connections = [connection for connection in res.branch_connections if connection.asset.grid_model_id == "L9"]
    assert len(l9_connections) == 2, "Expected L9 twice, as it is connected to two busbars"
    assert l9_connections[0].asset_bay != l9_connections[1].asset_bay, "Expected different asset bays for L9 assets"

    expected_switching_table = [
        [True, False, False, False, False, False, False, False, False, False, False, True],
        [False, False, False, True, True, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, True, True, False, False, False],
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
    # net.get_buses().index
    # [
    #    VL1_0, slack -> not relevant
    #    VL2_0, relevant
    #    VL3_0, relevant
    #    VL4_0, RelevantStationRules() require per default 4 braches -> only 3 branches, but on busbaroutage list > relevant
    #    VL5_0, 1 branch, 1 injection, 1 busbar -> but on busbar outage list -> relevant
    # ]
    expected_relevant_voltage_level_with_region = ["VL2", "VL3", "VL4", "VL5"]
    assert all(net.get_voltage_levels().index == ["VL1", "VL2", "VL3", "VL4", "VL5"])
    assert all(relevant_voltage_level_with_region["voltage_level_id"] == expected_relevant_voltage_level_with_region)
    assert all(relevant_voltage_level_with_region.index == ["VL2_0", "VL3_0", "VL4_0", "VL5_0"])

    master_data = get_node_breaker_topology_master_data(
        network=net, network_masks=network_masks, importer_parameters=importer_parameters
    )
    materialized_stations = materialize_stations_from_network_state(network=net, master_data=master_data)
    assert [station.bus_group_id for station in materialized_stations] == ["VL2_a", "VL3_a", "VL4_a", "VL5_a"]
    assert [station.voltage_level_id for station in materialized_stations] == expected_relevant_voltage_level_with_region
    assert master_data.topology_id == "cgmes_file.zip"
    assert master_data.grid_model_file == "cgmes_file.zip"


def test_materialized_stations_from_master_data_are_stable(
    basic_node_breaker_network_powsybl_grid: Network,
):
    """Verify that re-materializing node-breaker stations from master data is stable."""
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

    master_data = get_node_breaker_topology_master_data(
        network=net, network_masks=network_masks, importer_parameters=importer_parameters
    )
    new_stations = materialize_stations_from_network_state(network=net, master_data=master_data)
    repeated_stations = materialize_stations_from_network_state(network=net, master_data=master_data)

    assert len(new_stations) == len(repeated_stations)
    for new_station, repeated_station in zip(new_stations, repeated_stations, strict=True):
        assert new_station == repeated_station
        assert new_station.name == repeated_station.name
        assert new_station.station_type == repeated_station.station_type
        assert new_station.voltage_level == repeated_station.voltage_level
        assert new_station.model_log == repeated_station.model_log


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

    lf_result, *_ = pypowsybl.loadflow.run_ac(net)
    network_masks = powsybl_masks.make_masks(
        network=net,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    relevant_voltage_level_with_region = get_relevant_voltage_levels(network=net, network_masks=network_masks)
    # net.get_buses().index =
    # [
    #     'VL_3W_HV_0', large station
    #     'VL_3W_MV_0', large station
    #     'VL_3W_LV_0', 2 branches, 1 injections, but busbar relevant
    #     'VL_2W_MV_LV_MV_0', 4 branches -> relevant
    #     'VL_2W_MV_LV_LV_0', 2 branches, 1 injections, but busbar relevant
    #     'VL_LV_load_0', 2 branches, 2 injections, 1 busbar, but busbar relevant
    #     'VL_MV_load_0', 3 branches + pst -> 4 branches, 1 injection -> relevant
    #     'VL_MV_load_3', other side of pst -> not relevant
    #     'VL_MV_svc_0', large station
    #     'VL_MV_0', large station
    #     'VL_2W_MV_HV_MV_0', large station + pst
    #     'VL_2W_MV_HV_MV_2', other side of pst -> not relevant
    #     'VL_2W_MV_HV_HV_0', large station
    #     'VL_2W_MV_HV_MV_INT_0', VL where two trafos are connected to two lines -> not relevant
    #     'VL_HV_gen_0',  large station, slack bus -> not relevant
    #     'VL_HV_vsc_0', 4 branches, 1 injection, 1 HVDC
    #     'VL_CH_1_0', connected to the BE border via a tie line -> relevant
    #     'VL_DE_1_0', 2 busbar sections and an internal PST -> relevant
    #     'VL_DE_2_0', connected to BE and DE_1 -> relevant
    #     'VL_FR_1_0', 2 busbar sections and an internal PST in the isolated FR island -> relevant
    #     'VL_FR_2_0', connected to FR_1 in the isolated FR island -> relevant
    #     '3W-Star-VL_0' not relevant
    #    ]
    expected = [
        "VL_3W_HV",
        "VL_3W_MV",
        "VL_3W_LV",
        "VL_2W_MV_LV_MV",
        "VL_2W_MV_LV_LV",
        "VL_LV_load",
        "VL_MV_load",
        "VL_MV_svc",
        "VL_MV",
        "VL_2W_MV_HV_MV",
        "VL_2W_MV_HV_HV",
        "VL_2W_MV_HV_MV_INT",
        "VL_HV_vsc",
        "VL_CH_1",
        "VL_DE_1",
        "VL_DE_2",
        "VL_FR_1",
        "VL_FR_2",
    ]
    # 'VL_HV_gen_0' not included as it is the slack

    for vl in expected:
        assert vl in relevant_voltage_level_with_region["voltage_level_id"].values, f"Expected voltage level {vl} not found"

    master_data = get_node_breaker_topology_master_data(
        network=net, network_masks=network_masks, importer_parameters=importer_parameters
    )
    materialized_stations = materialize_stations_from_network_state(network=net, master_data=master_data)
    assert [station.voltage_level_id for station in materialized_stations] == [
        "VL_3W_HV",
        "VL_3W_MV",
        "VL_3W_LV",
        "VL_2W_MV_LV_MV",
        "VL_2W_MV_LV_LV",
        "VL_LV_load",
        "VL_MV_load",
        "VL_MV_svc",
        "VL_MV",
        "VL_2W_MV_HV_MV",
        "VL_2W_MV_HV_MV_INT",
        "VL_2W_MV_HV_MV_INT",
        "VL_2W_MV_HV_HV",
        "VL_HV_vsc",
        "VL_DE_1",
        "VL_DE_2",
        "VL_FR_1",
        "VL_FR_2",
        "VL_CH_1",
    ]
    assert [station.bus_group_id for station in materialized_stations] == [
        "VL_3W_HV_a",
        "VL_3W_MV_a",
        "VL_3W_LV_a",
        "VL_2W_MV_LV_MV_a",
        "VL_2W_MV_LV_LV_a",
        "VL_LV_load_a",
        "VL_MV_load_a",
        "VL_MV_svc_a",
        "VL_MV_a",
        "VL_2W_MV_HV_MV_a",
        "VL_2W_MV_HV_MV_INT_a",
        "VL_2W_MV_HV_MV_INT_b",
        "VL_2W_MV_HV_HV_a",
        "VL_HV_vsc_a",
        "VL_DE_1_a",
        "VL_DE_2_a",
        "VL_FR_1_a",
        "VL_FR_2_a",
        "VL_CH_1_a",
    ]
