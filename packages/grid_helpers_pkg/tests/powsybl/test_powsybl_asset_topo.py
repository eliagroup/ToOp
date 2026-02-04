# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl, create_busbar_b_in_ieee
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    assert_station_in_network,
    get_all_element_names,
    get_asset_info_from_topology,
    get_asset_switching_table,
    get_bus_info_from_topology,
    get_coupler_info_from_topology,
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
    get_list_of_switchable_assets_from_df,
    get_name_of_station_elements,
    get_relevant_network_data,
    get_relevant_stations,
    get_stations_bus_breaker,
    get_topology,
)
from toop_engine_interfaces.asset_topology import Busbar, BusbarCoupler, Station, SwitchableAsset, Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_get_name_for_branches():
    station_elements = pd.DataFrame(index=["line1", "trafo2"])
    element_names = pd.Series(index=["line1", "trafo2", "gen3"], data=["line_name", "", "gen_name"])
    station_elements = get_name_of_station_elements(station_elements, element_names)

    assert all(station_elements["name"] == ["line_name", ""]), "Wrong names for branches"


def test_get_asset_switching_table():
    station_busses = pd.DataFrame(data={"int_id": [1, 2, 3]})
    station_elements = pd.DataFrame(data={"bus_int_id": [1, 1, 2]})

    excepted_switching_table = np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    switching_table = get_asset_switching_table(station_busses, station_elements)

    assert np.all(excepted_switching_table == switching_table), "Wrong switching table"


def test_get_asset_switching_table_disconnected():
    station_busses = pd.DataFrame(data={"int_id": [1, 2, 3]})
    station_elements = pd.DataFrame(data={"bus_int_id": [1, 1, -1]})

    excepted_switching_table = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    switching_table = get_asset_switching_table(station_busses, station_elements)

    assert np.all(excepted_switching_table == switching_table), "Wrong switching table"


def test_get_list_of_coupler_from_df():
    coupler_elements = pd.DataFrame(
        data={
            "grid_model_id": ["coupler1", "coupler2"],
            "bus_int_id": [1, 2],
            "type": ["type1", "type2"],
            "name": ["name1", "name2"],
            "busbar_from_id": [1, 2],
            "busbar_to_id": [2, 3],
            "open": [True, False],
            "in_service": [True, True],
        }
    )
    expected_coupler_list = [
        BusbarCoupler(
            grid_model_id="coupler1",
            type="type1",
            name="name1",
            busbar_from_id=1,
            busbar_to_id=2,
            open=True,
            in_service=True,
        ),
        BusbarCoupler(
            grid_model_id="coupler2",
            type="type2",
            name="name2",
            busbar_from_id=2,
            busbar_to_id=3,
            open=False,
            in_service=True,
        ),
    ]
    coupler_list = get_list_of_coupler_from_df(coupler_elements)
    assert coupler_list == expected_coupler_list


def test_get_list_of_switchable_assets_from_df():
    asset_elements = pd.DataFrame(
        data={
            "grid_model_id": ["asset1", "asset2"],
            "bus_int_id": [1, 2],
            "type": ["TIE_LINE", "LINE"],
            "name": ["name1", "name2"],
            "in_service": [True, True],
        }
    )
    expected_asset_list = [
        SwitchableAsset(
            grid_model_id="asset1",
            type="TIE_LINE",
            name="name1",
            bus_int_id=1,
            in_service=True,
        ),
        SwitchableAsset(
            grid_model_id="asset2",
            type="LINE",
            name="name2",
            bus_int_id=2,
            in_service=True,
        ),
    ]
    asset_list = get_list_of_switchable_assets_from_df(asset_elements)
    assert asset_list == expected_asset_list


def test_get_list_of_busbars_from_df():
    busbar_elements = pd.DataFrame(
        data={
            "grid_model_id": ["busbar1", "busbar2"],
            "int_id": [1, 2],
            "type": ["type1", "type2"],
            "name": ["name1", "name2"],
            "in_service": [True, True],
        }
    )
    expected_busbar_list = [
        Busbar(
            grid_model_id="busbar1",
            int_id=1,
            type="type1",
            name="name1",
            in_service=True,
        ),
        Busbar(
            grid_model_id="busbar2",
            int_id=2,
            type="type2",
            name="name2",
            in_service=True,
        ),
    ]
    busbar_list = get_list_of_busbars_from_df(busbar_elements)
    assert busbar_list == expected_busbar_list


def test_get_bus_info_from_topology():
    busses_df = pd.DataFrame(
        index=["busbar1", "busbar2", "busbar3"],
        data={
            "bus_id": ["node1", "node1", "node2"],
            "name": ["name1", "name2", "name3"],
        },
    )
    busses_df.index.name = "id"
    bus_id = "node1"
    bus_info = get_bus_info_from_topology(busses_df, bus_id)

    expected_bus_info = pd.DataFrame(
        data={
            "grid_model_id": ["busbar1", "busbar2"],
            "name": ["name1", "name2"],
            "int_id": [0, 1],
            "in_service": [True, True],
        }
    )
    assert np.all(expected_bus_info == bus_info)


def test_get_asset_info_from_topology():
    busses_df = pd.DataFrame(
        index=["busbar1", "busbar2", "busbar3"],
        data={
            "bus_id": ["node1", "node1", "node2"],
            "name": ["name1", "name2", "name3"],
        },
    )
    elements_df = pd.DataFrame(
        index=["line1", "trafo2", "dangling1"],
        data={
            "bus_id": ["busbar1", "busbar3", "busbar1"],
            "in_service": [True, True, True],
            "type": ["LINE", "TWO_WINDINGS_TRANSFORMER", "DANGLING_LINE"],
        },
    )
    busses_df.index.name = "id"
    elements_df.index.name = "id"
    bus_id = "node1"
    station_busses = get_bus_info_from_topology(busses_df, bus_id)

    dangling_lines = pd.DataFrame(
        index=["dangling1"],
        data={"tie_line_id": ["tie_line1"]},
    )
    element_names = pd.Series(index=["line1", "trafo2", "dangling1", "tie_line1"], data=["line_name", "", "gen_name", "tie"])
    station_elements, _ = get_asset_info_from_topology(elements_df, station_busses, dangling_lines, element_names)
    print(station_elements)
    expected_station_elements = pd.DataFrame(
        index=[0, 1],
        data={
            "grid_model_id": ["line1", "tie_line1"],
            "type": ["LINE", "TIE_LINE"],
            "name": ["line_name", "tie"],
            "in_service": [True, True],
        },
    )
    assert np.all(expected_station_elements == station_elements)


def test_get_coupler_info_from_topology():
    busses_df = pd.DataFrame(
        index=["busbar1", "busbar2", "busbar3"],
        data={
            "bus_id": ["node1", "node1", "node2"],
            "name": ["name1", "name2", "name3"],
        },
    )
    switches_df = pd.DataFrame(
        index=["switch1", "switch2"],
        data={
            "bus1_id": ["busbar1", "busbar1"],
            "bus2_id": ["busbar2", "busbar2"],
            "open": [True, True],
            "kind": ["BREAKER", "DISCONNECTOR"],
        },
    )
    busses_df.index.name = "id"
    switches_df.index.name = "id"
    bus_id = "node1"
    station_busses = get_bus_info_from_topology(busses_df, bus_id)
    all_switches = pd.DataFrame(
        index=["switch1", "switch2"],
        data={"name": ["break_1", "disco_1"]},
    )
    station_couplers = get_coupler_info_from_topology(switches_df, all_switches, station_busses)
    print(station_couplers)
    expected_station_couplers = pd.DataFrame(
        index=[0, 1],
        data={
            "grid_model_id": ["switch1", "switch2"],
            "busbar_from_id": [0, 0],
            "busbar_to_id": [1, 1],
            "open": [True, True],
            "type": ["BREAKER", "DISCONNECTOR"],
            "name": ["break_1", "disco_1"],
            "in_service": [True, True],
        },
    )
    assert np.all(expected_station_couplers == station_couplers)


def test_get_all_element_names(ucte_file: Path):
    network = pypowsybl.network.load(ucte_file)
    element_names = get_all_element_names(network)

    expected_element_names = [
        "Test C. Line",
        "Test Line",
        "Test Line 2",
        "Test Line 3",
        "Test Line 4",
        "",
        "Test 2WT 2",
        "Test 2WT 1",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "FR-BE Xnode1",
        "FR-BE Xnode1",
        "FR-BE Xnode2",
        "FR-BE Xnode2",
        "France Xnode",
        "",
        "",
    ]

    assert np.all(list(element_names.values) == expected_element_names)


def test_get_relevant_network_data(ucte_file: Path):
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    buses_with_substation_and_voltage, switches, dangling_lines, element_names = get_relevant_network_data(
        network, relevant_subs
    )

    assert "substation_id" in buses_with_substation_and_voltage.columns, (
        "substation_id not in busses_with_substation_and_voltage"
    )
    assert "nominal_v" in buses_with_substation_and_voltage.columns, "voltage not in busses_with_substation_and_voltage"
    assert "name" in switches.columns, "name not in switches"
    assert "tie_line_id" in dangling_lines.columns, "tie_line_id not in dangling_lines"
    assert len(element_names) == len(network.get_branches()) + len(network.get_injections())


def test_get_relevant_stations(ucte_file: Path):
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    stations = get_relevant_stations(network, relevant_subs)

    assert len(stations) == sum(relevant_subs), "Wrong number of stations"
    assert isinstance(stations[0], Station), "Wrong type of station"


def test_get_topology_ucte(ucte_file: Path):
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    topology = get_topology(network, relevant_subs, grid_model_file="booga", topology_id="wooga")

    assert isinstance(topology, Topology), "Wrong type of topology"
    assert len(topology.stations) == sum(relevant_subs), "Wrong number of stations"
    assert topology.grid_model_file == "booga"
    assert topology.topology_id == "wooga"


def test_get_relevant_network_data_node_breaker():
    net = basic_node_breaker_network_powsybl()
    relevant_subs = np.ones(len(net.get_buses()), dtype=bool)
    buses_with_substation_and_voltage, switches, dangling_lines, element_names = get_relevant_network_data(
        net, relevant_subs
    )
    assert isinstance(buses_with_substation_and_voltage, pd.DataFrame)
    assert isinstance(switches, pd.DataFrame)
    assert isinstance(dangling_lines, pd.DataFrame)
    assert isinstance(element_names, pd.Series)
    assert len(buses_with_substation_and_voltage) == 5
    assert list(element_names.index) == [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
        "L9",
        "generator1",
        "generator2",
        "generator3",
        "load1",
        "load2",
    ]


def test_assert_station_in_network(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    for station in topology.stations:
        assert_station_in_network(net, station)

    # Change the station ID
    station = topology.stations[0].model_copy(update={"grid_model_id": "hugawuga"})
    with pytest.raises(ValueError, match="Station hugawuga not found in the network"):
        assert_station_in_network(net, station)


def test_assert_station_in_network_coupler(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Add a coupler to the station that is not in the grid
    station = topology.stations[0].model_copy(
        update={
            "couplers": [
                topology.stations[0].couplers[0],
                topology.stations[0].couplers[0].model_copy(update={"grid_model_id": "hugawuga"}),
            ],
        }
    )

    with pytest.raises(ValueError, match="Coupler hugawuga not found in the station switches"):
        assert_station_in_network(net, station)

    # Remove a coupler from the station
    station = topology.stations[0].model_copy(
        update={
            "couplers": [],
        }
    )
    # Should pass without strict
    assert_station_in_network(net, station, couplers_strict=False)
    with pytest.raises(ValueError, match="Coupler count mismatch"):
        assert_station_in_network(net, station, couplers_strict=True)


def test_assert_station_in_network_busbar(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Add a busbar to the station that is not in the grid
    station = topology.stations[0].model_copy(
        update={
            "busbars": [
                topology.stations[0].busbars[0],
                topology.stations[0].busbars[1],
                topology.stations[0].busbars[0].model_copy(update={"grid_model_id": "hugawuga", "int_id": 3}),
            ],
            "asset_switching_table": np.concatenate(
                [topology.stations[0].asset_switching_table, topology.stations[0].asset_switching_table[0:1]], axis=0
            ),
        }
    )
    with pytest.raises(ValueError, match="Busbar hugawuga not found in the station buses"):
        assert_station_in_network(net, station)

    # Remove a busbar from the station
    station = topology.stations[0].model_copy(
        update={
            "busbars": [
                topology.stations[0].busbars[0],
            ],
            "asset_switching_table": topology.stations[0].asset_switching_table[0:1],
        }
    )
    # Should pass without strict
    assert_station_in_network(net, station, busbars_strict=False)
    with pytest.raises(ValueError, match="Busbar count mismatch"):
        assert_station_in_network(net, station, busbars_strict=True)


def test_assert_station_in_network_asset(case14_data_with_asset_topo: tuple[Path, Topology]) -> None:
    grid_path, topology = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Add a switchable asset to the station that is not in the grid
    station = topology.stations[0].model_copy(
        update={
            "assets": topology.stations[0].assets
            + [
                topology.stations[0].assets[0].model_copy(update={"grid_model_id": "hugawuga"}),
            ],
            "asset_switching_table": np.concatenate(
                [topology.stations[0].asset_switching_table, topology.stations[0].asset_switching_table[:, 0:1]], axis=1
            ),
        }
    )
    with pytest.raises(ValueError, match="Asset hugawuga not found in the station elements"):
        assert_station_in_network(net, station)

    # Remove a switchable asset from the station
    station = topology.stations[0].model_copy(
        update={
            "assets": topology.stations[0].assets[:-1],
            "asset_switching_table": topology.stations[0].asset_switching_table[:, :-1],
        }
    )
    # Should pass without strict
    assert_station_in_network(net, station, assets_strict=False)
    with pytest.raises(ValueError, match="Asset count mismatch"):
        assert_station_in_network(net, station, assets_strict=True)


def test_convert_bus_breaker_stations_to_asset_topo() -> None:
    net = pypowsybl.network.create_ieee30()
    create_busbar_b_in_ieee(net)

    stations = get_stations_bus_breaker(net)
    assert len(stations) == 30

    for station in stations:
        assert len(station.busbars) == 2
        assert len(station.couplers) == 1
        for asset in station.assets:
            if asset.is_branch():
                assert asset.grid_model_id in net.get_branches().index
            else:
                assert asset.grid_model_id in net.get_injections().index
