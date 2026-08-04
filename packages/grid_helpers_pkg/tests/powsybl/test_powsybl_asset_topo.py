# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
from toop_engine_grid_helpers.powsybl import powsybl_station_to_graph
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_busbar_b_in_ieee,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    _get_branch_station_assets_from_df,
    _get_bus_breaker_structural_bus_groups,
    _get_injection_station_assets_from_df,
    assert_station_in_network,
    get_all_element_names,
    get_asset_info_from_topology,
    get_asset_switching_table,
    get_bus_breaker_master_asset_topology,
    get_bus_info_from_topology,
    get_coupler_info_from_topology,
    get_list_of_busbars_from_df,
    get_list_of_coupler_from_df,
    get_name_of_station_elements,
    get_relevant_network_data,
    get_stations_and_assets_bus_breaker,
    materialize_runtime_bus_groups_from_network_state,
)
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_interfaces.asset_topology.assets import BranchAsset, Busbar, BusbarCoupler, CouplerBay, InjectionAsset
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeBusGroup
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters


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
            "coupler_type": ["type1", "type2"],
            "name": ["name1", "name2"],
            "open": [True, False],
            "in_service": [True, True],
            "coupler_bay": [
                {
                    "dv_switch_grid_model_id": "coupler1",
                    "from_busbar_disconnector_grid_model_id": {"busbar1": "sw1"},
                    "to_busbar_disconnector_grid_model_id": {"busbar2": "sw2"},
                },
                {
                    "dv_switch_grid_model_id": "coupler2",
                    "from_busbar_disconnector_grid_model_id": {"busbar2": "sw3"},
                    "to_busbar_disconnector_grid_model_id": {"busbar3": "sw4"},
                },
            ],
        }
    )
    expected_coupler_list = [
        BusbarCoupler(
            grid_model_id="coupler1",
            coupler_type="type1",
            name="name1",
            coupler_bay=CouplerBay(
                dv_switch_grid_model_id="coupler1",
                from_busbar_disconnector_grid_model_id={"busbar1": "sw1"},
                to_busbar_disconnector_grid_model_id={"busbar2": "sw2"},
            ),
        ),
        BusbarCoupler(
            grid_model_id="coupler2",
            coupler_type="type2",
            name="name2",
            coupler_bay=CouplerBay(
                dv_switch_grid_model_id="coupler2",
                from_busbar_disconnector_grid_model_id={"busbar2": "sw3"},
                to_busbar_disconnector_grid_model_id={"busbar3": "sw4"},
            ),
        ),
    ]
    coupler_list = get_list_of_coupler_from_df(coupler_elements)
    assert coupler_list == expected_coupler_list


def test_get_branch_and_injection_station_assets_from_df(monkeypatch: pytest.MonkeyPatch):
    asset_elements = pd.DataFrame(
        data={
            "grid_model_id": ["asset1", "asset2", "asset3"],
            "asset_type": ["TIE_LINE", "LINE", "LOAD"],
            "name": ["name1", "name2", "name3"],
            "in_service": [True, True, True],
        }
    )
    asset_terminals = ["from", "to", None]
    switching_matrix = np.array(
        [
            [True, False, False],
            [False, True, True],
        ],
        dtype=bool,
    )
    asset_connectivity = np.ones_like(switching_matrix, dtype=bool)

    expected_branch_assets = [
        BranchAsset(
            grid_model_id="asset1",
            asset_type="TIE_LINE",
            name="name1",
        ),
        BranchAsset(
            grid_model_id="asset2",
            asset_type="LINE",
            name="name2",
        ),
    ]
    expected_injection_assets = [
        InjectionAsset(
            grid_model_id="asset3",
            asset_type="LOAD",
            name="name3",
        )
    ]

    branch_assets, branch_terminals, branch_switching_table, branch_connectivity = _get_branch_station_assets_from_df(
        asset_elements,
        asset_terminals,
        switching_matrix,
        asset_connectivity,
    )
    injection_assets, injection_terminals, injection_switching_table, injection_connectivity = (
        _get_injection_station_assets_from_df(
            asset_elements,
            asset_terminals,
            switching_matrix,
            asset_connectivity,
        )
    )

    assert branch_assets == expected_branch_assets
    assert injection_assets == expected_injection_assets
    assert branch_terminals == ["from", "to"]
    assert injection_terminals == [None]
    assert np.array_equal(branch_switching_table, switching_matrix[:, :2])
    assert np.array_equal(injection_switching_table, switching_matrix[:, 2:])
    assert np.array_equal(branch_connectivity, asset_connectivity[:, :2])
    assert np.array_equal(injection_connectivity, asset_connectivity[:, 2:])


def test_get_list_of_busbars_from_df():
    busbar_elements = pd.DataFrame(
        data={
            "grid_model_id": ["busbar1", "busbar2"],
            "int_id": [1, 2],
            "busbar_type": ["type1", "type2"],
            "name": ["name1", "name2"],
            "in_service": [True, True],
        }
    )
    expected_busbar_list = [
        Busbar(
            grid_model_id="busbar1",
            int_id=1,
            busbar_type="type1",
            name="name1",
        ),
        Busbar(
            grid_model_id="busbar2",
            int_id=2,
            busbar_type="type2",
            name="name2",
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
            "bus_breaker_bus_id": ["busbar1", "busbar2"],
            "bus_branch_bus_id": ["node1", "node1"],
        }
    )
    pd.testing.assert_frame_equal(expected_bus_info, bus_info)


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
            "type": ["LINE", "TWO_WINDINGS_TRANSFORMER", "BOUNDARY_LINE"],
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
            "asset_type": ["LINE", "TIE_LINE"],
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
        index=["switch1", "switch2", "switch3"],
        data={
            "bus1_id": ["busbar1", "busbar1", "busbar2"],
            "bus2_id": ["busbar2", "busbar2", "busbar3"],
            "open": [True, True, False],
            "kind": ["BREAKER", "DISCONNECTOR", "DISCONNECTOR"],
            "retained": [True, True, False],
        },
    )
    busses_df.index.name = "id"
    switches_df.index.name = "id"
    bus_id = "node1"
    station_busses = get_bus_info_from_topology(busses_df, bus_id)
    all_switches = pd.DataFrame(index=["switch1", "switch2", "switch3"], data={"name": ["break_1", "disco_1", "disco_2"]})
    station_couplers = get_coupler_info_from_topology(switches_df, all_switches, station_busses)
    print(station_couplers)
    expected_station_couplers = pd.DataFrame(
        index=[0, 1],
        data={
            "grid_model_id": ["switch1", "switch2"],
            "busbar_from_id": [0, 0],
            "busbar_to_id": [1, 1],
            "open": [True, True],
            "coupler_type": ["BREAKER", "DISCONNECTOR"],
            "name": ["break_1", "disco_1"],
            "in_service": [True, True],
        },
    )
    assert np.all(expected_station_couplers == station_couplers[expected_station_couplers.columns])
    assert station_couplers["coupler_bay"].to_list() == [
        {
            "dv_switch_grid_model_id": "switch1",
            "from_busbar_grid_model_ids": ["busbar1"],
            "to_busbar_grid_model_ids": ["busbar2"],
            "from_busbar_disconnector_grid_model_id": {},
            "to_busbar_disconnector_grid_model_id": {},
        },
        {
            "dv_switch_grid_model_id": "switch2",
            "from_busbar_grid_model_ids": ["busbar1"],
            "to_busbar_grid_model_ids": ["busbar2"],
            "from_busbar_disconnector_grid_model_id": {},
            "to_busbar_disconnector_grid_model_id": {},
        },
    ]


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
    master_data = get_bus_breaker_master_asset_topology(network, relevant_subs, topology_id="relevant_stations")
    stations = materialize_runtime_bus_groups_from_network_state(network=network, master_data=master_data)

    assert len(master_data.stations) <= sum(relevant_subs), "Bus groups should not outnumber relevant electrical buses"
    assert len(stations) == len(master_data.stations), "Wrong number of stations"
    assert isinstance(stations[0], RuntimeBusGroup), "Wrong type of station"


def test_get_master_asset_topology_and_stations_ucte(ucte_file: Path):
    """Verify canonical master data and runtime stations for a UCTE bus-breaker grid."""
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    master_data = get_bus_breaker_master_asset_topology(
        network=network,
        relevant_stations=relevant_subs,
        grid_model_file="booga",
        topology_id="wooga",
    )
    stations = materialize_runtime_bus_groups_from_network_state(network=network, master_data=master_data)

    assert master_data.topology_id == "wooga"
    assert master_data.grid_model_file == "booga"
    assert len(stations) == len(master_data.stations)
    assert all(isinstance(station, RuntimeBusGroup) for station in stations)


def test_get_topology_ucte(ucte_file: Path):
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    master_data = get_bus_breaker_master_asset_topology(
        network=network,
        relevant_stations=relevant_subs,
        grid_model_file="booga",
        topology_id="wooga",
    )
    stations = materialize_runtime_bus_groups_from_network_state(network=network, master_data=master_data)
    assert [station.model_dump(mode="json") for station in stations] == [
        station.model_dump(mode="json") for station in stations
    ]
    assert len(stations) == len(master_data.stations), "Wrong number of runtime stations"
    assert master_data.grid_model_file == "booga"
    assert master_data.topology_id == "wooga"


def test_materialize_stations_from_network_state(ucte_file: Path) -> None:
    """Verify that runtime station materialization succeeds for a UCTE grid."""
    network = pypowsybl.network.load(ucte_file)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    master_data = get_bus_breaker_master_asset_topology(network, relevant_subs, grid_model_file="booga", topology_id="wooga")
    materialized_stations = materialize_runtime_bus_groups_from_network_state(network, master_data)

    assert len(materialized_stations) <= len(master_data.stations)
    assert all(isinstance(asset, BranchAsset) for asset in master_data.branch_assets)
    assert all(isinstance(station, RuntimeBusGroup) for station in materialized_stations)


def test_materialize_stations_from_network_state_preserves_bus_branch_bus_ids_node_breaker() -> None:
    """Verify that node-breaker materialization preserves bus-branch bus ids."""
    network = basic_node_breaker_network_powsybl()
    network_masks = powsybl_masks.create_default_network_masks(network)
    relevant_subs = network.get_buses(attributes=[]).index.isin(["VL2_0", "VL3_0"])
    network_masks = powsybl_masks.NetworkMasks(**{**network_masks.__dict__, "relevant_subs": relevant_subs})
    importer_parameters = CgmesImporterParameters(
        grid_model_file=Path("node_breaker_network.xiidm"),
        data_folder=Path("."),
        white_list_file=None,
        black_list_file=None,
        area_settings=AreaSettings(
            cutoff_voltage=110,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
        ),
    )

    master_data = powsybl_station_to_graph.get_node_breaker_master_asset_topology(
        network=network,
        network_masks=network_masks,
        importer_parameters=importer_parameters,
    )
    materialized_stations = materialize_runtime_bus_groups_from_network_state(network, master_data)

    station_bus_ids = {
        station.voltage_level_id: {busbar.bus_branch_bus_id for busbar in station.busbars}
        for station in materialized_stations
    }

    assert station_bus_ids["VL2"] == {"VL2_0"}
    assert station_bus_ids["VL3"] == {"VL3_0"}


def test_materialize_stations_from_network_state_marks_disconnected_transformer_out_of_service() -> None:
    """Verify that a transformer disconnected on both sides is materialized out of service."""
    network = create_complex_grid_battery_hvdc_svc_3w_trafo()
    transformer_id = network.get_2_windings_transformers().index[0]
    network.update_2_windings_transformers(id=transformer_id, connected1=False, connected2=False)

    relevant_subs = np.ones(len(network.get_buses()), dtype=bool)
    master_data = get_bus_breaker_master_asset_topology(
        network,
        relevant_subs,
        grid_model_file="booga",
        topology_id="wooga",
    )
    materialized_stations = materialize_runtime_bus_groups_from_network_state(network, master_data)

    materialized_assets = [
        asset_connection.asset
        for station in materialized_stations
        for asset_connection in station.branch_connections
        if asset_connection.asset.grid_model_id == transformer_id
    ]

    assert materialized_assets, "Expected the disconnected transformer to appear in the materialized station view"
    assert all(not asset.in_service for asset in materialized_assets)


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


def test_assert_station_in_network(
    case14_data_with_asset_topo: tuple[Path, object, list[RuntimeBusGroup]],
) -> None:
    """Verify strict station presence checks against a powsybl network."""
    grid_path, _master_data, stations = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    for station in stations:
        assert_station_in_network(net, station)

    # Change the station ID
    station = stations[0].model_copy(update={"bus_group_id": "hugawuga", "voltage_level_id": None})
    with pytest.raises(ValueError, match="Station hugawuga is missing voltage_level_id"):
        assert_station_in_network(net, station)


def test_assert_station_in_network_uses_voltage_level_id_for_synthetic_station_id(
    case14_data_with_asset_topo: tuple[Path, object, list[RuntimeBusGroup]],
) -> None:
    """Verify that synthetic station ids resolve via voltage_level_id during station checks."""
    grid_path, _master_data, stations = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    station = stations[0].model_copy(
        update={
            "bus_group_id": "synthetic_station_id",
            "voltage_level_id": "VL1",
        }
    )

    assert_station_in_network(net, station)


def test_assert_station_in_network_coupler(
    case14_data_with_asset_topo: tuple[Path, object, list[RuntimeBusGroup]],
) -> None:
    """Verify coupler subset and strict-count checks in station validation."""
    grid_path, _master_data, stations = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Add a coupler to the station that is not in the grid
    base_station = stations[0]
    station = base_station.model_copy(
        update={
            "couplers": [
                base_station.couplers[0],
                base_station.couplers[0].model_copy(update={"grid_model_id": "hugawuga"}),
            ],
        }
    )

    with pytest.raises(ValueError, match="Coupler hugawuga not found in the station switches"):
        assert_station_in_network(net, station)

    # Remove a coupler from the station
    station = stations[0].model_copy(
        update={
            "couplers": [],
        }
    )
    # Should pass without strict
    assert_station_in_network(net, station, couplers_strict=False)
    with pytest.raises(ValueError, match="Coupler count mismatch"):
        assert_station_in_network(net, station, couplers_strict=True)


def test_assert_station_in_network_busbar(
    case14_data_with_asset_topo: tuple[Path, object, list[RuntimeBusGroup]],
) -> None:
    """Verify busbar subset and strict-count checks in station validation."""
    grid_path, _master_data, stations = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    base_station = stations[0]

    # Add a busbar to the station that is not in the grid
    station = base_station.model_copy(
        update={
            "busbars": [
                base_station.busbars[0],
                base_station.busbars[1],
                base_station.busbars[0].model_copy(update={"grid_model_id": "hugawuga", "int_id": 3}),
            ],
            "branch_switching_table": np.concatenate(
                [
                    base_station.branch_switching_table,
                    base_station.branch_switching_table[0:1],
                ],
                axis=0,
            ),
            "injection_switching_table": np.concatenate(
                [
                    base_station.injection_switching_table,
                    base_station.injection_switching_table[0:1],
                ],
                axis=0,
            ),
            "branch_connectivity": np.concatenate(
                [
                    base_station.branch_connectivity,
                    base_station.branch_connectivity[0:1],
                ],
                axis=0,
            ),
            "injection_connectivity": np.concatenate(
                [
                    base_station.injection_connectivity,
                    base_station.injection_connectivity[0:1],
                ],
                axis=0,
            ),
        }
    )
    with pytest.raises(ValueError, match="Busbar hugawuga not found in the station buses"):
        assert_station_in_network(net, station)

    # Remove a busbar from the station
    station = type(base_station).model_construct(
        grid_model_id=base_station.bus_group_id,
        voltage_level_id=base_station.voltage_level_id,
        name=base_station.name,
        station_type=base_station.station_type,
        region=base_station.region,
        voltage_level=base_station.voltage_level,
        busbars=[base_station.busbars[0]],
        couplers=base_station.couplers,
        branch_connections=base_station.branch_connections,
        injection_connections=base_station.injection_connections,
        branch_switching_table=base_station.branch_switching_table[0:1],
        injection_switching_table=base_station.injection_switching_table[0:1],
        branch_connectivity=None,
        injection_connectivity=None,
        model_log=base_station.model_log,
    )
    # Should pass without strict
    assert_station_in_network(net, station, busbars_strict=False)
    with pytest.raises(ValueError, match="Busbar count mismatch"):
        assert_station_in_network(net, station, busbars_strict=True)


def test_assert_station_in_network_asset(
    case14_data_with_asset_topo: tuple[Path, object, list[RuntimeBusGroup]],
) -> None:
    """Verify asset subset and strict-count checks in station validation."""
    grid_path, _master_data, stations = case14_data_with_asset_topo
    net = pypowsybl.network.load(grid_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    # Add a switchable asset to the station that is not in the grid
    base_station = stations[0]
    if base_station.branch_connections:
        station = base_station.model_copy(
            update={
                "branch_connections": base_station.branch_connections
                + [
                    base_station.branch_connections[0].model_copy(
                        update={
                            "asset": base_station.branch_connections[0].asset.model_copy(
                                update={"grid_model_id": "hugawuga"}
                            )
                        }
                    ),
                ],
                "branch_switching_table": np.concatenate(
                    [
                        base_station.branch_switching_table,
                        base_station.branch_switching_table[:, 0:1],
                    ],
                    axis=1,
                ),
                "branch_connectivity": np.concatenate(
                    [
                        base_station.branch_connectivity,
                        base_station.branch_connectivity[:, 0:1],
                    ],
                    axis=1,
                ),
            }
        )
    else:
        station = base_station.model_copy(
            update={
                "injection_connections": base_station.injection_connections
                + [
                    base_station.injection_connections[0].model_copy(
                        update={
                            "asset": base_station.injection_connections[0].asset.model_copy(
                                update={"grid_model_id": "hugawuga"}
                            )
                        }
                    ),
                ],
                "injection_switching_table": np.concatenate(
                    [
                        base_station.injection_switching_table,
                        base_station.injection_switching_table[:, 0:1],
                    ],
                    axis=1,
                ),
                "injection_connectivity": np.concatenate(
                    [
                        base_station.injection_connectivity,
                        base_station.injection_connectivity[:, 0:1],
                    ],
                    axis=1,
                ),
            }
        )
    with pytest.raises(ValueError, match="Asset hugawuga not found in the station elements"):
        assert_station_in_network(net, station)

    # Remove a switchable asset from the station
    if base_station.branch_connections:
        station = base_station.model_copy(
            update={
                "branch_connections": base_station.branch_connections[:-1],
                "branch_switching_table": base_station.branch_switching_table[:, :-1],
                "branch_connectivity": base_station.branch_connectivity[:, :-1],
            }
        )
    else:
        station = base_station.model_copy(
            update={
                "injection_connections": base_station.injection_connections[:-1],
                "injection_switching_table": base_station.injection_switching_table[:, :-1],
                "injection_connectivity": base_station.injection_connectivity[:, :-1],
            }
        )
    # Should pass without strict
    assert_station_in_network(net, station, assets_strict=False)
    with pytest.raises(ValueError, match="Asset count mismatch"):
        assert_station_in_network(net, station, assets_strict=True)


def test_convert_bus_breaker_stations_to_asset_topo() -> None:
    net = pypowsybl.network.create_ieee30()
    create_busbar_b_in_ieee(net)

    stations, branch_assets, injection_assets = get_stations_and_assets_bus_breaker(net)
    assets = [*branch_assets, *injection_assets]
    assert len(stations) == 30

    for station in stations:
        assert len(station.busbars) == 2
        assert len(station.couplers) == 1
        for asset_id in [
            asset_connection.asset.grid_model_id
            for asset_connection in [*station.branch_connections, *station.injection_connections]
        ]:
            assert any(asset.grid_model_id == asset_id for asset in assets)

    for asset in branch_assets:
        assert asset.grid_model_id in net.get_branches().index

    for asset in injection_assets:
        assert asset.grid_model_id in net.get_injections().index


def test_get_bus_breaker_master_asset_topology_groups_connected_buses_per_voltage_level() -> None:
    net = pypowsybl.network.create_ieee30()
    create_busbar_b_in_ieee(net)

    relevant_subs = np.ones(len(net.get_buses()), dtype=bool)
    master_data = get_bus_breaker_master_asset_topology(net, relevant_subs, topology_id="ieee30")

    assert len(master_data.stations) == 30
    assert all(station.bus_group_id.endswith("_a") for station in master_data.stations)
    assert {station.bus_group_id for station in master_data.stations} == {
        f"VL{voltage_level_index}_a" for voltage_level_index in range(1, 31)
    }
    assert all(len(station.busbars) == 2 for station in master_data.stations)


def test_get_bus_breaker_structural_bus_groups_ignores_retained_flag() -> None:
    station_topology_buses = pd.DataFrame(index=["bus_a", "bus_b", "bus_c"])
    station_topology_switches = pd.DataFrame(
        {
            "bus1_id": ["bus_a", "bus_b"],
            "bus2_id": ["bus_b", "bus_c"],
            "open": [True, False],
            "retained": [False, False],
        }
    )

    structural_groups = _get_bus_breaker_structural_bus_groups(
        station_topology_buses=station_topology_buses,
        station_topology_switches=station_topology_switches,
    )

    assert structural_groups == [{"bus_a", "bus_b", "bus_c"}]
