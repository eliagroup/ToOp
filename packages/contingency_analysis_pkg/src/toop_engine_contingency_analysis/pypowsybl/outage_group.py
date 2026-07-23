"""Outage group identification module."""

import pandas as pd
import pypowsybl


def update_switches_for_outage_group_identification(net: pypowsybl.network.Network, keep_fictitious: bool = False) -> None:
    """Update switches for outage group identification.

    This function relies on the bus_breaker topology in powsybl. It updates all switches,
    that all elements are not separated by a breaker remain connected.
    Note: open disconnectors are not modified and can alter the outage group identification,
    depending on the network topology.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to update.
    keep_fictitious : bool, optional
        Whether to keep fictitious switches, by default False.

    Returns
    -------
        None
    """
    switches = net.get_switches(attributes=["retained", "kind", "fictitious"])
    switches["retained"] = True
    switches.loc[switches["kind"] == "DISCONNECTOR", "retained"] = False
    if not keep_fictitious:
        switches.loc[switches["fictitious"], "retained"] = False
    net.update_switches(switches[["retained"]])

    # open all breaker -> only isolated components remain -> outage grouping
    if not keep_fictitious:
        switches = switches[~switches["fictitious"]]

    for sw_id in switches[(switches["kind"] == "BREAKER")].index:
        net.open_switch(sw_id)


def get_connected_components_branches(net: pypowsybl.network.Network) -> pd.DataFrame:
    """Get connected components for branches in the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing branches and their connected components.
    """
    group_branches = net.get_branches(
        attributes=["bus1_id", "bus2_id", "bus_breaker_bus1_id", "bus_breaker_bus2_id"]
    ).reset_index()
    group_branches = group_branches.merge(
        net.get_buses()[["connected_component"]], left_on="bus1_id", right_index=True, suffixes=("", "_bus1"), how="left"
    )
    group_branches = group_branches.merge(
        net.get_buses()[["connected_component"]], left_on="bus2_id", right_index=True, suffixes=("", "_bus2"), how="left"
    )
    cond = group_branches["connected_component"].isna()
    group_branches.loc[cond, "connected_component"] = group_branches.loc[cond, "connected_component_bus2"]
    group_branches["connected_component"] = group_branches["connected_component"].fillna(-1).astype(int)

    return group_branches


def identify_outage_groups(net: pypowsybl.network.Network, keep_fictitious: bool = False) -> dict:
    """Identify outage groups in the network.

    Example layout:
    x = DISCONNECTOR

        | BREAKER1             | BREAKER2
        |                      |
        |                      |
        | Line1                | Line2
        |                      |
        |                      |
        x---------x-----------x
                  |
                  |
                  | Line3
                  |
                  |
                  | BREAKER3

    Line1, Line2, and Line3 are connected only at the end are BREAKER1, BREAKER2, and BREAKER3.
    Therefore all lines 1,2,3 are in the same outage group if one has a fault.
    Example output:
    outage_groups = {
        0: {
        "ids": ["Line1", "Line2", "Line3"],
        "switches": ["BREAKER1", "BREAKER2", "BREAKER3"]}
        }


    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    keep_fictitious : bool, optional
        Whether to keep fictitious switches, by default False.

    Returns
    -------
    dict
        A dictionary containing outage groups with their corresponding branch IDs and switch IDs.
        key: connected component ID
        value: {"ids": [branch IDs], "switches": [switch IDs]}

    """
    # create a variant
    net_variant = net.clone_variant("outage_group_identification")
    # make sure we don't relay on wrong data
    update_switches_for_outage_group_identification(net_variant, keep_fictitious=keep_fictitious)

    # now after all BREAKER are open, powsybl will identify the connected components (islanded grids) of the network,
    # which are the outage groups

    # get all branches and their connected components
    group_branches = get_connected_components_branches(net_variant)

    # prepare switches for merging with branches
    switches = net_variant.get_switches(
        attributes=["retained", "kind", "bus_breaker_bus1_id", "bus_breaker_bus2_id", "fictitious"]
    )
    switches["switch_id"] = switches.index
    switches = switches[switches["kind"] == "BREAKER"]
    switches = switches.rename(
        columns={"bus_breaker_bus1_id": "switch_bus_breaker_bus1_id", "bus_breaker_bus2_id": "switch_bus_breaker_bus2_id"}
    )
    switches_bus1 = switches[["switch_bus_breaker_bus1_id", "switch_id"]]
    switches_bus2 = switches[["switch_bus_breaker_bus2_id", "switch_id"]]
    # merge all switch bus ids to the branch ends
    group_branches = group_branches.merge(
        switches_bus1, left_on="bus_breaker_bus1_id", right_on="switch_bus_breaker_bus1_id", how="left"
    )
    group_branches = group_branches.merge(
        switches_bus2,
        left_on="bus_breaker_bus2_id",
        right_on="switch_bus_breaker_bus2_id",
        how="left",
        suffixes=("_bus11", "_bus22"),
    )
    group_branches = group_branches.merge(
        switches_bus1, left_on="bus_breaker_bus2_id", right_on="switch_bus_breaker_bus1_id", how="left"
    )
    group_branches = group_branches.merge(
        switches_bus2,
        left_on="bus_breaker_bus1_id",
        right_on="switch_bus_breaker_bus2_id",
        how="left",
        suffixes=("_bus12", "_bus21"),
    )
    # fill unmerged branch switch data
    group_branches.loc[group_branches["switch_id_bus11"].isna(), "switch_id_bus11"] = ""
    group_branches.loc[group_branches["switch_id_bus22"].isna(), "switch_id_bus22"] = ""
    group_branches.loc[group_branches["switch_id_bus12"].isna(), "switch_id_bus12"] = ""
    group_branches.loc[group_branches["switch_id_bus21"].isna(), "switch_id_bus21"] = ""
    connected_components = group_branches["connected_component"].unique()
    # filter out nan
    # this happens for isolated branches, e.g. if both DISCONNECTORRs are open and there is no BREAKER in between
    connected_components = connected_components[connected_components != -1]

    # create outage groups
    outage_groups = {}
    for connected_component in connected_components:
        for connected_component_group in group_branches[
            group_branches["connected_component"] == connected_component
        ].itertuples():
            if connected_component not in outage_groups:
                outage_groups[connected_component] = {"ids": [], "switches": []}
            outage_groups[connected_component]["ids"].append(connected_component_group.id)
            if connected_component_group.switch_id_bus11 != "":
                outage_groups[connected_component]["switches"].append(connected_component_group.switch_id_bus11)
            if connected_component_group.switch_id_bus22 != "":
                outage_groups[connected_component]["switches"].append(connected_component_group.switch_id_bus22)
            if connected_component_group.switch_id_bus12 != "":
                outage_groups[connected_component]["switches"].append(connected_component_group.switch_id_bus12)
            if connected_component_group.switch_id_bus21 != "":
                outage_groups[connected_component]["switches"].append(connected_component_group.switch_id_bus21)

    return outage_groups
