# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions for the preprocessing process in pandapower.

File: preprocessing.py
Author:  Benjamin Petrick
Created: 2024-11-07

The goal is to create a static information for the optimizer.
For this the bus-break model is converted to a bus-branch model.

The preprocessing is done in several steps:
1. preprocess_net_step1: General preprocessing
    - select connected subnet
    - Remove zero branches
    - remove out of service elements
    - handle_constan_z_load
    - drop elements connected to one bus
    - replace xward by internal elements
    - replace ward by internal elements
    - drop controler
2. get asset topology
3. preprocess_net_step2: after creation of the asset topology
    - fuse buses with closed switches
    - remove switches
    - delete bus_geodata as it is not up to date after fusing buses
    - create continuous bus index (asset topology index and bus index do not match)
    - update station ids in asset topology
    -> conversion to bus-brach model completed
4. get masks for optimization
5. create static information for optimizer

"""

import logbook
import numpy as np
import pandapower as pp
from beartype.typing import Optional
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_grid_helpers.pandapower.pandapower_import_helpers import (
    drop_elements_connected_to_one_bus,
    fuse_closed_switches_fast,
    remove_out_of_service,
    replace_zero_branches,
    select_connected_subnet,
)
from toop_engine_importer.pandapower_import.pandapower_toolset_node_breaker import (
    fuse_closed_switches_by_bus_ids,
    get_coupler_types_of_substation,
)
from toop_engine_interfaces.asset_topology import Topology

logger = logbook.Logger(__name__)


def modify_constan_z_load(net: pp.pandapowerNet, value: float = 0.0) -> None:
    """Modify constant z load with a value of 100.0.

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place
    value : float
        value to set the constant z load to

    Returns
    -------
    None

    """
    const_z_percent_value = 100.0
    constan_z_load = net.load[np.isclose(net.load["const_z_percent"], const_z_percent_value)].index
    net.load.loc[constan_z_load, "const_z_percent"] = value


def handle_switches(network: pp.pandapowerNet) -> None:
    """Handle closed, open and end of element switches in the network.

    This function makes to transition from bus-breaker model to bus-branch model by removing switches:
        - fuse closed switches of type b
        - drop closed element switches
        - remove elements with open switch
        - not implemented: open switches of type t3

    Parameters
    ----------
    network : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place

    Raises
    ------
    NotImplementedError
        if open switches of type t3 are found
    """
    # fuse closed switches of type b
    _, _ = fuse_closed_switches_fast(network)
    # drop open switches of type b
    cond = ~network.switch.closed & (network.switch.et == "b")
    network.switch.drop(network.switch[cond].index, inplace=True)
    assert "b" not in network.switch["et"].unique(), (
        "fuse_closed_switches_fast should hve removed all closed switches of type b"
    )
    # drop closed element switches of type l, t and t3 (as type "b" is already removed)
    cond = network.switch.closed
    network.switch.drop(network.switch[cond].index, inplace=True)
    # remove elements with open switch
    cond = network.switch.et == "l"
    line_index = network.switch[cond].element.values
    # drop lines with open switch, including the switch
    pp.toolbox.drop_elements(net=network, element_type="line", element_index=line_index)
    # remove elements with open switch
    cond = network.switch.et == "t"
    trafo_index = network.switch[cond].element.values
    # drop lines with open switch, including the switch
    pp.toolbox.drop_elements(net=network, element_type="trafo", element_index=trafo_index)

    if "t3" in network.switch["et"].unique():
        raise NotImplementedError("No logic implemented for open switches of type t3 - a three winding transformer")
    # network.switch = network.switch[0:0]
    assert len(network.switch) == 0


def preprocess_net_step1(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """General preprocessing - e.g. a PowerFactory network may converge in AC -> change elements.

    Step 1: General preprocessing
        - select connected subnet
        - Remove zero branches
        - remove out of service elements
        - handle_constant_z_load
        - drop elements connected to one bus
        - replace xward by internal elements
        - replace ward by internal elements
        - drop controler

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place

    Returns
    -------
    net : pp.pandapowerNet
        modified pandapower network


    """
    # preprocessing: remove zero branches, fuse closed switches, remove out of service elements
    net = select_connected_subnet(net)
    replace_zero_branches(net)
    remove_out_of_service(net)
    # sometimes if the load is 100% constan z it will not converge -> investigate, cosinder setting to 99%
    modify_constan_z_load(net)
    drop_elements_connected_to_one_bus(net)
    pp.replace_xward_by_internal_elements(net)
    pp.replace_ward_by_internal_elements(net)
    validate_trafo_model(net)
    if "controler" in net:
        del net["controler"]

    return net


def preprocess_net_step2(network: pp.pandapowerNet, topology_model: Topology) -> Topology:
    """Step 2: after creation of the asset topology.

        - fuse buses with closed switches
        - remove switches
        - delete bus_geodata as it is not up to date after fusing buses
        - create continuous bus index (asset topology index and bus index do not match)
        - update station ids in asset topology
        -> conversion to bus-brach model completed

    Parameters
    ----------
    network : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place
    topology_model : Topology
        asset topology model

    Returns
    -------
    topology_model : Topology
        modified asset topology model

    """
    handle_switches(network)
    drop_elements_connected_to_one_bus(network)
    if "bus_geodata" in network:
        del network["bus_geodata"]
    old_index = pp.toolbox.create_continuous_bus_index(network, start=0, store_old_index=True)
    for station in topology_model.stations:
        station_id = int(station.grid_model_id.split(SEPARATOR)[0])
        new_id = old_index[station_id]
        station.grid_model_id = f"{new_id}{SEPARATOR}{station.grid_model_id.split(SEPARATOR)[1]}"
    return topology_model


def fuse_cross_coupler(
    network: pp.pandapowerNet,
    station_id_list: Optional[list[list[int]]] = None,
    substation_column: str = "substat",
) -> None:
    """Fuse cross coupler.

    Parameters
    ----------
    network : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place
    station_id_list : Optional[list[list[int]]]
        list of station ids
    substation_column : str
        column name of the substation, default is "substat"

    Returns
    -------
    None

    """
    if station_id_list is None:
        substation_names = network.bus[substation_column].unique()
        # remove empty strings and None
        substation_names = substation_names[(substation_names != "") & (substation_names is not None)]
        # get all substation ids
        station_id_list = [
            network.bus[network.bus[substation_column] == substation_name].index for substation_name in substation_names
        ]

    fuse_list = []
    for station_ids in station_id_list:
        coupler = get_coupler_types_of_substation(network=network, substation_bus_list=station_ids)
        cross_coupler_ids = [item for sublist in coupler["cross_coupler_switch_ids"] for item in sublist]
        fuse_list.extend(cross_coupler_ids)
        if len(coupler["cross_coupler_bus_ids"]) > 0:
            # relabel bus ids after fusing
            bus_labels = np.arange(np.max(network.bus.index) + 1)
            for bus_ids in coupler["cross_coupler_bus_ids"]:
                bus_ids_list = list(bus_labels[bus_ids])
                bus_labels = fuse_closed_switches_by_bus_ids(network, bus_ids_list)


def validate_asset_topology(net: pp.pandapowerNet, topology_model: Topology) -> None:
    """Validate the asset topology with the network.

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
    topology_model : Topology
        asset topology model

    Returns
    -------
    None

    Raises
    ------
    ValueError
        if the number of connections in the network does not match the number of assets in the station
    """
    for station in topology_model.stations:
        s_id = int(station.grid_model_id.split(r"%%")[0])
        connection_dict = pp.toolbox.get_connected_elements_dict(net, [s_id])
        del connection_dict["bus"]
        len_connection = len([element for key in connection_dict for element in connection_dict[key]])
        if len_connection != len(station.assets):
            logger.warning(
                f"Station {station.grid_model_id} has {len(station.assets)} assets but only "
                + f"{len_connection} connections in the network"
            )
            logger.warning(connection_dict)
            for asset in station.assets:
                logger.warning(asset)
            raise ValueError(
                f"Station {station.grid_model_id} has {len(station.assets)} assets but only "
                + f"{len_connection} connections in the network"
            )


def validate_trafo_model(net: pp.pandapowerNet) -> None:
    """Validate the transformer model.

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place

    """
    if "tap_dependent_impedance" in net.trafo.columns:
        model_error = net.trafo[(net.trafo["tap_side"].astype(str) == "None") & net.trafo["tap_dependent_impedance"]]

        if len(model_error) > 0:
            logger.warning(
                f"Error in trafo model: {model_error['name'].to_list()}: tap_side = None and tap_dependent_impedance = True."
                + " Changing to tap_dependent_impedance = False"
            )
            net.trafo.loc[model_error.index, "tap_dependent_impedance"] = False
