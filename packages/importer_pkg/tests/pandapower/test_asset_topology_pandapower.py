# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime

import logbook
import numpy as np
import pandapower
import pytest
from toop_engine_importer.pandapower_import import asset_topology
from toop_engine_interfaces.asset_topology import (
    AssetBay,
    Station,
    Topology,
)


def test_get_busses_from_station(pp_network_w_switches):
    net = pp_network_w_switches
    # test 1
    expected = {
        "grid_model_id": {0: "0%%bus"},
        "type": {0: "b"},
        "name": {0: "Double Busbar 1"},
        "int_id": {0: 0},
        "in_service": {0: True},
    }
    result = asset_topology.get_busses_from_station(network=net, station_bus_index=0, station_col="name").to_dict()
    assert result == expected
    result = asset_topology.get_busses_from_station(network=net, station_bus_index=[0], station_col="name").to_dict()
    assert result == expected
    result = asset_topology.get_busses_from_station(
        network=net, station_name="Double Busbar 1", station_col="name"
    ).to_dict()
    assert result == expected
    result = asset_topology.get_busses_from_station(
        network=net,
        station_name="Double Busbar 1",
        station_col="name",
        foreign_key="type",
    ).to_dict()
    expected["name"] = {0: "b"}
    assert result == expected

    # test 2
    expected = {
        "grid_model_id": {0: "0%%bus", 2: "2%%bus"},
        "type": {0: "b", 2: "n"},
        "name": {0: "Double Busbar 1", 2: "Bus DB T0"},
        "int_id": {0: 0, 2: 2},
        "in_service": {0: True, 2: True},
    }
    result = asset_topology.get_busses_from_station(network=net, station_bus_index=[0, 2], station_col="name").to_dict()
    assert result == expected


def test_get_coupler_from_station(pp_network_w_switches):
    net = pp_network_w_switches
    # test 1
    # network has coupler_elements["et"] == "b" and coupler_elements["type"] == "CB"
    expected = {
        "grid_model_id": {14: "14%%switch"},
        "type": {14: "CB"},
        "name": {14: "CB"},
        "busbar_from_id": {14: 1},
        "busbar_to_id": {14: 0},
        "open": {14: False},
        "in_service": {14: True},
    }
    station_buses = asset_topology.get_busses_from_station(
        network=net, station_name="Double Busbar 1", station_col="substat"
    )
    result = asset_topology.get_coupler_from_station(network=net, station_buses=station_buses, foreign_key="type").to_dict()
    assert result == expected


def test_get_branches_from_station(pp_network_w_switches):
    net = pp_network_w_switches
    # test 1 - test three winding transformer
    expected_station_branches = [
        {
            "grid_model_id": "0%%line",
            "type": "line",
            "name": "HV Line1",
            "branch_end": "from",
            "in_service": True,
        },
        {
            "grid_model_id": "5%%line",
            "type": "line",
            "name": "HV Line6",
            "branch_end": "from",
            "in_service": True,
        },
        {
            "grid_model_id": "0%%trafo",
            "type": "trafo",
            "name": "EHV-HV-Trafo",
            "branch_end": "lv",
            "in_service": True,
        },
        {
            "grid_model_id": "0%%load",
            "type": "load",
            "name": "MV Net 0",
            "branch_end": None,
            "in_service": True,
        },
        {
            "grid_model_id": "0%%sgen",
            "type": "sgen",
            "name": "Wind Park",
            "branch_end": None,
            "in_service": True,
        },
    ]
    expected_switching_matrix = np.array([[True, True, True, True, True]])
    expected_asset_connection = [
        AssetBay(
            sl_switch_grid_model_id="SB DS2.1",
            dv_switch_grid_model_id="SB CB2",
            sr_switch_grid_model_id={"16%%bus": "SB DS2.2"},
        ),
        AssetBay(
            sl_switch_grid_model_id="SB DS3.1",
            dv_switch_grid_model_id="SB CB3",
            sr_switch_grid_model_id={"16%%bus": "SB DS3.2"},
        ),
        AssetBay(
            sl_switch_grid_model_id="SB DS1.1",
            dv_switch_grid_model_id="SB CB1",
            sr_switch_grid_model_id={"16%%bus": "SB DS1.2"},
        ),
        AssetBay(
            sl_switch_grid_model_id="SB DS4.1",
            dv_switch_grid_model_id="SB CB4",
            sr_switch_grid_model_id={"16%%bus": "SB DS4.2"},
        ),
        AssetBay(
            sl_switch_grid_model_id="SB DS5.1",
            dv_switch_grid_model_id="SB CB5",
            sr_switch_grid_model_id={"16%%bus": "SB DS5.2"},
        ),
    ]
    station_buses = asset_topology.get_busses_from_station(network=net, station_name="Single Busbar", station_col="substat")
    (
        station_branches,
        switching_matrix,
        asset_connection,
    ) = asset_topology.get_branches_from_station(network=net, station_buses=station_buses, foreign_key="name")
    assert station_branches.to_dict(orient="records") == expected_station_branches
    assert np.array_equal(switching_matrix, expected_switching_matrix)
    assert asset_connection == expected_asset_connection

    # test 4 - test wrong branch
    branch_types = ["line"]
    bus_types = [("bus", None, "")]
    branches_busses = [1]
    station_busses = asset_topology.get_busses_from_station(network=net, station_bus_index=branches_busses)
    with pytest.raises(ValueError):
        asset_topology.get_branches_from_station(
            network=net,
            station_buses=station_busses,
            branch_types=branch_types,
            bus_types=bus_types,
        )

    # test 5 - line mapping with only one type
    branch_types = ["line"]
    bus_types = [("from_bus", "from", "")]
    station_buses = asset_topology.get_busses_from_station(network=net, station_name="Single Busbar", station_col="substat")

    (
        station_branches,
        switching_matrix,
        asset_connection,
    ) = asset_topology.get_branches_from_station(
        network=net,
        station_buses=station_buses,
        foreign_key="name",
        branch_types=branch_types,
        bus_types=bus_types,
    )
    assert expected_station_branches[0:2] == station_branches.to_dict(orient="records")
    assert np.array_equal(switching_matrix, np.array([[True, True]]))
    assert asset_connection == expected_asset_connection[0:2]

    bus_types = [("to_bus", "to", "")]
    (
        station_branches,
        switching_matrix,
        asset_connection,
    ) = asset_topology.get_branches_from_station(
        network=net,
        station_buses=station_busses,
        branch_types=branch_types,
        bus_types=bus_types,
    )
    assert len(station_branches) == 0
    assert switching_matrix.shape == (1, 0)
    assert len(asset_connection) == 0

    # test 6 - branch_types unknown
    branch_types = ["unknown"]
    station_busses = asset_topology.get_busses_from_station(network=net, station_bus_index=branches_busses)
    with pytest.raises(ValueError):
        asset_topology.get_branches_from_station(
            network=net,
            station_buses=station_busses,
            branch_types=branch_types,
        )


def test_get_branches_from_station_edge_cases(pp_network_w_switches):
    with logbook.handlers.TestHandler() as caplog:
        net = pp_network_w_switches
        net.switch.drop(23, inplace=True)
        net.switch.loc[31, "element"] = 19
        net.switch.loc[33, "closed"] = False
        expected_path = AssetBay(
            sl_switch_grid_model_id=None,
            dv_switch_grid_model_id="SB CB3",
            sr_switch_grid_model_id={"16%%bus": "SB DS3.2"},
        )
        station_buses = asset_topology.get_busses_from_station(
            network=net, station_name="Single Busbar", station_col="substat"
        )
        (
            station_branches,
            switching_matrix,
            asset_connection,
        ) = asset_topology.get_branches_from_station(network=net, station_buses=station_buses, foreign_key="name")
        assert (
            "No closed switch found (Element is disconnected and will be dropped) for element_type:sgen element"
            in "".join(caplog.formatted_records)
        )
        assert asset_connection[1] == expected_path

        net.switch.loc[5, "closed"] = True
        expected_path = [
            AssetBay(
                sl_switch_grid_model_id="DB DS11",
                dv_switch_grid_model_id="DB CB2",
                sr_switch_grid_model_id={"1%%bus": "DB DS4", "0%%bus": "DB DS5"},
            )
        ]
        station_buses = asset_topology.get_busses_from_station(
            network=net, station_name="Double Busbar 1", station_col="substat"
        )
        (
            station_branches,
            switching_matrix,
            asset_connection,
        ) = asset_topology.get_branches_from_station(network=net, station_buses=station_buses, foreign_key="name")
        assert "Expected one closed switch for element_type" in "".join(caplog.formatted_records)
        assert "Using the first one." in "".join(caplog.formatted_records)
        assert asset_connection == expected_path


def test_get_parameter_from_station():
    net = pandapower.networks.mv_oberrhein()
    # test 1 - vn_kv has two values
    with pytest.raises(ValueError):
        asset_topology.get_parameter_from_station(network=net, station_bus_index=[57, 58], parameter="vn_kv")
    # test 2 - NOT_A_PARAMETER
    with pytest.raises(ValueError):
        asset_topology.get_parameter_from_station(network=net, station_bus_index=57, parameter="NOT_A_PARAMETER")
    # test 3 - vn_kv
    expected = 20.0
    result = asset_topology.get_parameter_from_station(network=net, station_bus_index=[57], parameter="vn_kv")
    assert result == expected
    result = asset_topology.get_parameter_from_station(network=net, station_bus_index=[56, 57], parameter="vn_kv")
    assert result == expected
    # test 4 - station_name
    expected = 20.0
    result = asset_topology.get_parameter_from_station(
        network=net, station_name=20.0, station_col="vn_kv", parameter="vn_kv"
    )
    assert result == expected


def test_get_station_from_id(pp_network_w_switches):
    net = pp_network_w_switches
    station_id_list = [el for el in range(0, 15)]
    result = asset_topology.get_station_from_id(network=net, station_id_list=station_id_list, foreign_key="name")
    assert isinstance(result, Station)
    assert result.grid_model_id == r"0%%bus"
    assert result.name == "Double Busbar 1"
    assert result.type is None
    assert result.voltage_level == 380.0
    assert len(result.busbars) == 2


def test_get_list_of_stations_ids(pp_network_w_switches):
    net = pp_network_w_switches
    station_id_list = [[el for el in range(0, 15)], [el for el in range(16, 32)]]
    result = asset_topology.get_list_of_stations_ids(network=net, station_list=station_id_list, foreign_key="name")
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], Station)
    assert isinstance(result[1], Station)
    assert result[0].grid_model_id == r"0%%bus"
    assert result[0].name == "Double Busbar 1"
    assert result[1].grid_model_id == r"16%%bus"
    assert result[1].name == "Single Busbar"


def test_get_asset_topology_from_network(pp_network_w_switches):
    net = pp_network_w_switches
    station_id_list = [[el for el in range(0, 15)], [el for el in range(16, 32)]]
    result = asset_topology.get_asset_topology_from_network(
        network=net,
        station_id_list=station_id_list,
        topology_id="1",
        grid_model_file="test",
        foreign_key="name",
    )
    assert isinstance(result, Topology)
    assert result.topology_id == "1"
    assert result.grid_model_file == "test"
    assert isinstance(result.stations, list)
    assert isinstance(result.timestamp, datetime)
    current_time = datetime.now()
    time_difference = current_time - result.timestamp
    assert time_difference.total_seconds() < 60, "Timestamp is not recent"


def test_get_station_bus_df(pp_network_w_switches):
    net = pp_network_w_switches
    # test 1 - station_name and station_col
    result = asset_topology.get_station_bus_df(network=net, station_name="Double Busbar 1", station_col="substat")
    assert result.equals(net.bus[net.bus.substat == "Double Busbar 1"])
    result = asset_topology.get_station_bus_df(network=net, station_name="b", station_col="type")
    assert result.equals(net.bus[net.bus.type == "b"])

    # test 2 - station_bus_index
    result = asset_topology.get_station_bus_df(network=net, station_bus_index=0)
    assert result.equals(net.bus.loc[[0]])
    result = asset_topology.get_station_bus_df(network=net, station_bus_index=[0])
    assert result.equals(net.bus.loc[[0]])
    result = asset_topology.get_station_bus_df(network=net, station_bus_index=[0, 1])
    assert result.equals(net.bus.loc[[0, 1]])
    # test 3 - missing station_name or station_bus_index
    with pytest.raises(ValueError):
        asset_topology.get_station_bus_df(network=net)
    with pytest.raises(ValueError):
        asset_topology.get_station_bus_df(network=net, station_col="substat")
    with pytest.raises(ValueError):
        asset_topology.get_station_bus_df(
            network=net,
            station_col="substat",
            station_bus_index=0,
            station_name="Double Busbar 1",
        )
