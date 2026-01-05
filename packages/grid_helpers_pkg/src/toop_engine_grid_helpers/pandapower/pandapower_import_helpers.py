# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains additional functions to process pandapower network.

File: pandapower_toolset.py
Author:
Created:
"""

import logbook
import numpy as np
import pandapower as pp
import pandas as pd
from beartype.typing import Optional

logger = logbook.Logger(__name__)


def fuse_closed_switches_fast(
    net: pp.pandapowerNet,
    switch_ids: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuse closed switches in the network by merging busbars.

    This routine uses an algorithm to number each busbar and then find the lowest connected busbar
    iteratively. If a busbar is connected to a lower-numbered busbar, it will be re-labeled to the
    lower-numbered busbar. This algorithm needs as many iterations as the maximum number of hops
    between the lowest and highest busbar in any of the substations.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to fuse closed switches in, will be modified in-place.
    switch_ids: list[int]
        The switch ids to fuse. If None, all closed switches are fused.

    Returns
    -------
    pd.DataFrame
        The closed switches that were fused.
    pd.DataFrame
        The buses that were dropped because they were relabeled to a lower-numbered busbar.
    """
    # Label the busbars, find the lowest index that every busbar is coupled to
    labels = np.arange(np.max(net.bus.index) + 1)
    closed_switches = net.switch[net.switch.closed & (net.switch.et == "b") & (net.switch.bus != net.switch.element)]
    if switch_ids is not None:
        closed_switches = closed_switches[closed_switches.index.isin(switch_ids)]
    while not np.array_equal(labels[closed_switches.bus.values], labels[closed_switches.element.values]):
        bus_smaller = labels[closed_switches.bus.values] < labels[closed_switches.element.values]
        element_smaller = labels[closed_switches.bus.values] > labels[closed_switches.element.values]

        # Where the element is smaller, set the bus labels to the element labels
        _, change_idx = np.unique(closed_switches.bus.values[element_smaller], return_index=True)
        labels[closed_switches.bus.values[element_smaller][change_idx]] = labels[
            closed_switches.element.values[element_smaller][change_idx]
        ]

        # Where the bus is smaller (and where the element was not already touched), set the element labels to the bus labels
        was_touched = np.isin(
            closed_switches.element.values,
            closed_switches.bus.values[element_smaller][change_idx],
        )
        cond = bus_smaller & ~was_touched
        _, change_idx = np.unique(closed_switches.element.values[cond], return_index=True)
        labels[closed_switches.element.values[cond][change_idx]] = labels[closed_switches.bus.values[cond][change_idx]]

    # Move all elements over to the lowest index busbar
    move_elements_based_on_labels(net, labels)
    # Drop all busbars that were re-labeled because they were connected to a lower-labeled bus
    buses_to_drop = net.bus[~np.isin(net.bus.index, labels)]
    switch_cond = (net.switch.et == "b") & (net.switch.bus == net.switch.element)
    switch_to_drop = net.switch[switch_cond]
    pp.toolbox.drop_elements(net, "switch", switch_to_drop.index)
    pp.drop_buses(net, buses_to_drop.index)
    return closed_switches, buses_to_drop


def move_elements_based_on_labels(
    net: pp.pandapowerNet,
    labels: np.ndarray,
) -> None:
    """Move all elements in the network to the lowest labeled busbar.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to move elements in, will be modified in-place.
    labels: np.ndarray
        The labels of the busbars to move the elements to.
    """
    for element, column in pp.element_bus_tuples():
        if element == "switch":
            net[element][column] = labels[net[element][column]]
            switch_cond = net[element].et == "b"
            net[element].loc[switch_cond, "element"] = labels[net[element].loc[switch_cond, "element"]]
            net[element].loc[net[element].index, "bus"] = labels[net[element].loc[net[element].index, "bus"]]
        else:
            net[element][column] = labels[net[element][column]]


def select_connected_subnet(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Select the connected subnet of the grid that has a slack and return it.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to select the connected subnet from.

    Returns
    -------
    pp.pandapowerNet
        The connected subnet of the grid that has a slack.
    """
    name = net.name
    mg = pp.topology.create_nxgraph(net, respect_switches=True)

    slack_bus = net.ext_grid[net.ext_grid.in_service].bus
    if len(slack_bus) == 0:
        slack_bus = net.gen[net.gen.slack & net.gen.in_service].bus
        if len(slack_bus) == 0:
            raise ValueError("No slack bus found in the network.")
    slack_bus = slack_bus.iloc[0]

    cc = pp.topology.connected_component(mg, slack_bus)

    next_grid_buses = set(cc)
    net_new = pp.select_subnet(
        net,
        next_grid_buses,
        include_switch_buses=True,
        include_results=False,
        keep_everything_else=True,
    )
    net_new.name = name
    return net_new


def replace_zero_branches(net: pp.pandapowerNet) -> None:
    """Replace zero-impedance branches with switches in the network.

    Some leftover lines and xwards will be bumped to a higher impedance to avoid numerical issues.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to replace zero branches in, will be modified in-place.

    Returns
    -------
    pp.pandapowerNet
        The network with zero branches replaced.
    """
    pp.toolbox.replace_zero_branches_with_switches(
        net,
        min_length_km=0.0,
        min_r_ohm_per_km=0.002,
        min_x_ohm_per_km=0.002,
        min_c_nf_per_km=0,
        min_rft_pu=0,
        min_xft_pu=0,
    )
    threshold_x_ohm = 0.001
    # net.xward.x_ohm[net.xward.x_ohm == 1e-6] = 1e-2
    net.xward.loc[net.xward.x_ohm < threshold_x_ohm, "x_ohm"] = 0.01
    zero_lines = (net.line.x_ohm_per_km * net.line.length_km) < threshold_x_ohm
    net.line.loc[zero_lines, "x_ohm_per_km"] = 0.01
    net.line.loc[zero_lines, "length_km"] = 1.0


def drop_unsupplied_buses(net: pp.pandapowerNet) -> None:
    """Drop all unsupplied buses from the network.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to drop unsupplied buses from, will be modified in-place.
    """
    pp.drop_buses(net, pp.topology.unsupplied_buses(net))
    assert len(pp.topology.unsupplied_buses(net)) == 0


def create_virtual_slack(net: pp.pandapowerNet) -> None:
    """Create a virtual slack bus for all ext_grids in the network.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to create a virtual slack for, will be modified in-place.
        Note: network is modified in-place.

    Returns
    -------
    pp.pandapowerNet
        The network with a virtual slack.
    """
    if net.gen.slack.sum() <= 1:
        return
    # Create a virtual slack where all ext_grids are connected to
    virtual_slack_bus = pp.create_bus(net, vn_kv=380, in_service=True, name="virtual_slack")

    for generator in net.gen[net.gen.slack].index:
        cur_bus = net.gen.loc[generator].bus
        # Connect each gen through a trafo to the virtual slack
        pp.create_transformer_from_parameters(
            net,
            hv_bus=virtual_slack_bus,
            lv_bus=cur_bus,
            name="con_" + str(net.gen.loc[generator].name),
            sn_mva=9999,
            vn_hv_kv=net.bus.vn_kv[cur_bus],
            vn_lv_kv=net.bus.vn_kv[cur_bus],
            # shift_degree=net.ext_grid.loc[generator].va_degree,
            shift_degree=0,
            pfe_kw=1,
            i0_percent=0.1,
            vk_percent=1,
            vkr_percent=0.1,
            xn_ohm=10,
        )

    net.gen.drop(net.gen[net.gen.slack].index, inplace=True)

    pp.create_ext_grid(
        net,
        virtual_slack_bus,
        vm_pu=1,
        va_degree=0,
        in_service=True,
        name="virtual_slack",
    )


def remove_out_of_service(net: pp.pandapowerNet) -> None:
    """Remove all out-of-service elements from the network.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network to remove out-of-service elements from, will be modified in-place.
    """
    for element in pp.pp_elements():
        if "bus" == element and "in_service" in net[element]:
            pp.drop_buses(net, net[element][~net[element]["in_service"]].index)
        elif "in_service" in net[element]:
            net[element] = net[element][net[element]["in_service"]]


def drop_elements_connected_to_one_bus(net: pp.pandapowerNet, branch_types: Optional[list[str]] = None) -> None:
    """Drop elements connected to one bus.

    - impedance -> Capacitor will end up on the same bus
    - trafo3w -> edgecase: trafo3w that goes from one hv to the same level but two
                 different busbars will end up on the same bus

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place
    branch_types : list[str]
        list of branch types to drop elements connected to one bus

    Returns
    -------
    None

    """
    if branch_types is None:
        branch_types = ["line", "trafo", "trafo3w", "impedance", "switch"]

    for branch_type in branch_types:
        handle_elements_connected_to_one_bus(net, branch_type)


def handle_elements_connected_to_one_bus(net: pp.pandapowerNet, branch_type: str) -> None:
    """Drop elements of a specific branch type connected to one bus.

    Parameters
    ----------
    net : pp.pandapowerNet
        pandapower network
        Note: the network is modified in place
    branch_type : str
        branch type to drop elements connected to one bus

    Raises
    ------
    ValueError
        If the branch type is not recognized.
    AssertionError
        If a two-winding transformer with same hv and lv bus is found.
        If a three-winding transformer with same hv == lv or hv == mv bus is found

    Returns
    -------
    None
    """
    branch_df = getattr(net, branch_type)
    if branch_type == "switch":
        branch_index = branch_df[(branch_df["bus"] == branch_df["element"]) & (branch_df["et"] == "b")].index
        pp.drop_elements(net, element_type=branch_type, element_index=branch_index)
    elif branch_type in ["line", "impedance"]:
        branch_index = branch_df[branch_df["from_bus"] == branch_df["to_bus"]].index
        pp.drop_elements(net, element_type=branch_type, element_index=branch_index)
    elif branch_type == "trafo":
        branch_index = branch_df[branch_df["hv_bus"] == branch_df["lv_bus"]].index
        assert len(branch_index) == 0, (
            "Two winding transformer with same hv and lv bus found in " + f"{branch_df.loc[branch_index].to_dict()}"
        )
    elif branch_type == "trafo3w":
        hv_cond = (branch_df["hv_bus"] == branch_df["lv_bus"]) | (branch_df["hv_bus"] == branch_df["mv_bus"])
        branch_index = branch_df[hv_cond].index
        assert len(branch_index) == 0, (
            "Three winding transformer with same hv == lv or hv == mv bus found in "
            + f"{branch_df.loc[branch_index].to_dict()}"
        )
        lv_cond = branch_df["lv_bus"] == branch_df["mv_bus"]
        branch_index = branch_df[lv_cond].index
        if len(branch_index) > 0:
            logger.warning(
                "Three winding transformer with same mv and lv bus found in " + f"{branch_df.loc[branch_index].to_dict()}"
            )
    else:
        raise ValueError(f"Branch type {branch_type} not recognized for dropping elements connected to one bus.")
