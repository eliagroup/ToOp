import math
from copy import deepcopy
from unittest.mock import MagicMock

import logbook
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_importer.pandapower_import import (
    asset_topology,
    pandapower_toolset_node_breaker,
    preprocessing,
)
from toop_engine_interfaces.asset_topology import Topology

logger = logbook.Logger(__name__)


def test_handle_switches(pp_network_w_switches):
    net = deepcopy(pp_network_w_switches)
    preprocessing.handle_switches(net)
    assert len(net.switch) == 0
    net = deepcopy(pp_network_w_switches)
    net.switch.loc[84, "et"] = "t3"
    net.switch.loc[84, "closed"] = False
    with pytest.raises(NotImplementedError) as e_info:
        preprocessing.handle_switches(net)
    assert "No logic implemented for open switches of type t3" in e_info.value.args[0]


def assert_load_flow_after_preprocessing(
    net_org: pp.pandapowerNet,
    net_after: pp.pandapowerNet,
    abs_tol: float = 1e-8,
) -> None:
    res_line_org = net_org.res_line.copy()
    res_line_after = net_after.res_line
    for i in res_line_after.index:
        for col in res_line_after.columns:
            if col != "loading_percent":
                assert math.isclose(
                    res_line_after.loc[i, col],
                    res_line_org.loc[i, col],
                    abs_tol=abs_tol,
                )


def test_preprocess_net_step1(pp_network_w_switches: pp.pandapowerNet) -> None:
    net_org = pp_network_w_switches
    pp.runpp(net_org)
    net_after = deepcopy(net_org)
    assert net_after.converged
    net_after["controler"] = pd.DataFrame()
    net_after = preprocessing.preprocess_net_step1(net_after)
    assert "controler" not in net_after
    pp.runpp(net_after)
    assert net_after.converged
    assert_load_flow_after_preprocessing(net_org, net_after)
    assert len(net_org.res_line) == len(net_after.res_line)


def preprocess_net_step2_network_helper(net: pp.pandapowerNet, station_ids) -> list:
    connected = []
    for station_id in station_ids:
        station = pandapower_toolset_node_breaker.get_substation_buses_from_bus_id(net, station_id)
        connected_elements = pp.toolbox.get_connected_elements_dict(
            net,
            station,
            respect_switches=False,
            respect_in_service=False,
            include_empty_lists=False,
        )
        if "switch" in connected_elements:
            del connected_elements["switch"]
        if "bus" in connected_elements:
            del connected_elements["bus"]
        connected.append(connected_elements)
    return connected


def run_preprocess_net_step2_network(net: pp.pandapowerNet) -> pp.pandapowerNet:
    assert net.converged
    net = preprocessing.preprocess_net_step1(net)
    mock_topology = MagicMock(spec=Topology)
    mock_topology.stations = []
    net["bus_geodata"] = pd.DataFrame()
    preprocessing.preprocess_net_step2(net, mock_topology)
    assert "bus_geodata" not in net
    pp.runpp(net)
    assert net.converged
    return net


def test_preprocess_net_step2_network_elements_still_connected(
    pp_network_w_switches: pp.pandapowerNet,
) -> None:
    net_org = pp_network_w_switches
    pp.runpp(net_org)
    net = deepcopy(net_org)
    net = run_preprocess_net_step2_network(net)
    len_switch_open_line = len(net_org.switch[(net_org.switch.et == "l") & (~net_org.switch.closed)])
    assert len(net_org.res_line) == len(net.res_line) + len_switch_open_line

    station_ids_org = [0, 16] + [i for i in range(32, 57)]
    station_ids_after = net.bus.index
    connected_org = preprocess_net_step2_network_helper(net_org, station_ids_org)
    connected_after = preprocess_net_step2_network_helper(net, station_ids_after)
    len_xward = len(net_org.xward)

    # special case: station 11 & 12 is a special case: line 10 has been removed
    connected_org[11]["line"].remove(10)
    connected_org[12]["line"].remove(10)

    for i in range(len(connected_org)):
        if "xward" not in connected_org[i]:
            assert connected_org[i] == connected_after[i], f"Connected elements are not the same at bus {i}"
        else:
            key_list = ["load", "shunt", "impedance"]
            len_dict = {key: len(connected_org[i][key]) + 1 if key in connected_org[i] else 1 for key in key_list}
            for key in key_list:
                assert len(connected_after[i][key]) == len_dict[key], f"Length of {key} is not the same"
            for key, values in connected_org[i].items():
                if key != "xward":
                    for value in values:
                        assert value in connected_after[i][key], (
                            "The element id is missing in the new network at the same bus"
                        )
    # xward gets converted to a bus -> + len_xward
    assert len(connected_org) + len_xward == len(connected_after)
    # check if xward is converted to bus
    assert all(net_org.xward.name.values == net.bus.loc[len(net.bus) - len_xward :].name.values)


def test_preprocess_net_step2_network_load_flows(
    pp_network_w_switches: pp.pandapowerNet,
) -> None:
    net_org = pp_network_w_switches

    net_after = deepcopy(net_org)
    net_after = run_preprocess_net_step2_network(net_after)
    # ac load flow changes due to line removal with one open switch
    # -> acts as capacitor -> reactive power changes
    pp.rundcpp(net_after)
    pp.rundcpp(net_org)
    assert_load_flow_after_preprocessing(net_org, net_after)


def test_preprocess_net_step2(pp_network_w_switches):
    net = pp_network_w_switches
    # 2. get asset topology
    # fuse cross coupler for only relevant substations to save computation time
    station_id_list = [[el for el in range(16, 32)]]
    # preprocessing.fuse_cross_coupler(network=net, station_id_list=station_id_list)

    # get relevant substations for asset topology
    topology_model = asset_topology.get_asset_topology_from_network(
        network=net,
        station_id_list=station_id_list,
        topology_id="1",
        grid_model_file="test",
        foreign_key="name",
    )
    # 3. preprocess_net_step2: after creation of the asset topology
    topology_model = preprocessing.preprocess_net_step2(net, topology_model)
    assert isinstance(topology_model, Topology)


def test_fuse_cross_coupler(net_multivoltage_cross_coupler):
    net = net_multivoltage_cross_coupler
    busbars = pandapower_toolset_node_breaker.get_coupler_types_of_substation(network=net, substation_bus_list=net.bus.index)
    assert len(busbars["cross_coupler_bus_ids"]) > 0
    assert len(busbars["cross_coupler_switch_ids"]) > 0
    pandapower_toolset_node_breaker.add_substation_column_to_bus(net, substation_col="substation")
    preprocessing.fuse_cross_coupler(net, substation_column="substation")
    busbars_fused = pandapower_toolset_node_breaker.get_coupler_types_of_substation(
        network=net, substation_bus_list=net.bus.index
    )
    assert len(busbars_fused["cross_coupler_bus_ids"]) == 0
    assert len(busbars_fused["cross_coupler_switch_ids"]) == 0

    # one busbar remains -> all other are fused into this one
    for bus_ids in busbars["cross_coupler_bus_ids"]:
        counter = 0
        for bus_id in bus_ids:
            if bus_id in net.bus.index:
                counter += 1
        assert counter == 1
    # all switches are removed
    for switch_ids in busbars["cross_coupler_switch_ids"]:
        for switch_id in switch_ids:
            assert switch_id not in net.switch.index


def test_validate(pp_network_w_switches):
    net = deepcopy(pp_network_w_switches)
    # 2. get asset topology
    # fuse cross coupler for only relevant substations to save computation time
    station_id_list = [[el for el in range(16, 32)]]
    # preprocessing.fuse_cross_coupler(network=net, station_id_list=station_id_list)

    # get relevant substations for asset topology
    topology_model = asset_topology.get_asset_topology_from_network(
        network=net,
        station_id_list=station_id_list,
        topology_id="1",
        grid_model_file="test",
        foreign_key="name",
    )
    # 3. preprocess_net_step2: after creation of the asset topology
    topology_model = preprocessing.preprocess_net_step2(net, topology_model)
    assert isinstance(topology_model, Topology)
    # conversion to bus-branch model completed
    # validate network model
    preprocessing.validate_asset_topology(net, topology_model)

    net = deepcopy(pp_network_w_switches)
    station_id_list = [[el for el in range(0, 15)], [el for el in range(16, 32)]]
    # get relevant substations for asset topology
    topology_model = asset_topology.get_asset_topology_from_network(
        network=net,
        station_id_list=station_id_list,
        topology_id="1",
        grid_model_file="test",
        foreign_key="name",
    )
    # 3. preprocess_net_step2: after creation of the asset topology
    topology_model = preprocessing.preprocess_net_step2(net, topology_model)
    # conversion to bus-branch model completed
    # validate network model
    with logbook.handlers.TestHandler() as caplog:
        with pytest.raises(ValueError):
            preprocessing.validate_asset_topology(net, topology_model)
        # ext_grid is there and not supported as a connection
        assert "Station 0%%bus has 1 assets but only 2 connections in the network" in "".join(caplog.formatted_records)


def test_validate_trafo_model(pp_network_w_switches):
    net = pp_network_w_switches
    net.trafo.loc[0, "tap_dependent_impedance"] = True
    assert net.trafo.loc[0, "tap_side"] is None, "setup of test failed"
    assert net.trafo.loc[0, "tap_dependent_impedance"], "setup of test failed"
    with logbook.handlers.TestHandler() as caplog:
        preprocessing.validate_trafo_model(net)
        assert (
            r"Error in trafo model: ['EHV-HV-Trafo']: tap_side = None and tap_dependent_impedance = True. Changing to tap_dependent_impedance = False"
            in "".join(caplog.formatted_records)
        ), "Error message not found in log"
    assert not net.trafo.loc[0, "tap_dependent_impedance"], "tap_dependent_impedance not changed"
