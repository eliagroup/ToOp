# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Outage group identification module."""

import pandera as pa
import pandera.typing as pat
import pypowsybl
import rustworkx as rx
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.types import (
    BranchElectricalCircuitGroupSchema,
    BusBreakerIdSchema,
    EdgeSchema,
    ElectricalCircuitGroup,
    ElectricalCircuitGroupIdentification,
    ElectricalCircuitGroupMap,
    InjectionElectricalCircuitGroupSchema,
    SwitchElectricalCircuitGroupSchema,
)


def _update_switches_for_outage_group_identification(net: pypowsybl.network.Network, keep_fictitious: bool = False) -> None:
    """Update switches for outage group identification.

    This function relies on the bus_breaker topology in powsybl. It updates all switches,
    that all elements are not separated by a breaker remain connected.
    Note: DISCONNECTORs are not set to retained == False, as this is a BREAKER only search
    Note: Network updated in place.

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


@pa.check_types
def _get_bus_breaker_int_ids(net: pypowsybl.network.Network) -> pat.DataFrame[BusBreakerIdSchema]:
    """Get bus breaker int IDs for the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.

    Returns
    -------
    DataFrame[BusBreakerIdSchema]
        A DataFrame containing bus breaker int IDs.
    """
    bus_breaker_int_id = net.get_bus_breaker_view_buses(attributes=[]).reset_index()
    bus_breaker_int_id["bus_breaker_id_int"] = bus_breaker_int_id.index
    bus_breaker_int_id.set_index("id", inplace=True)
    return bus_breaker_int_id


@pa.check_types
def _get_graph_edges(
    net: pypowsybl.network.Network, bus_breaker_int_id: pat.DataFrame[BusBreakerIdSchema]
) -> pat.DataFrame[EdgeSchema]:
    """Get graph edges for the network.

    Note: it is expected that fictitious switches are at this stage handled,
    e.g. by setting them to retained == False, so that bus_breaker id is handeled correctly.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    bus_breaker_int_id : pat.DataFrame[BusBreakerIdSchema]
        Mapping from bus-breaker bus identifiers to their integer graph node ids.

    Returns
    -------
    DataFrame[EdgeSchema]
        A DataFrame containing graph edges.
    """
    branches = net.get_branches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"])
    branches.reset_index(inplace=True, drop=False)
    branches.rename(columns={"id": "id_str"}, inplace=True)
    branches = branches.merge(
        bus_breaker_int_id[["bus_breaker_id_int"]], left_on="bus_breaker_bus1_id", right_index=True, how="left"
    )
    branches = branches.merge(
        bus_breaker_int_id[["bus_breaker_id_int"]],
        left_on="bus_breaker_bus2_id",
        right_index=True,
        suffixes=("_bus1", "_bus2"),
        how="left",
    )
    return branches


@pa.check_types
def _get_electrical_circuit_group(
    bus_breaker_int_id: pat.DataFrame[BusBreakerIdSchema], edges: pat.DataFrame[EdgeSchema]
) -> pat.DataFrame[BusBreakerIdSchema]:
    """Get electrical circuit groups for the network.

    Creates a graph based on all edge connections.
    It respects the current switching state of the network, e.g. if a DISCONNECTOR is open
    The electrical circuit groups are based on the bus_breaker topology and stops at BREAKERs.
    Fictional switches must be considered in _update_switches_for_outage_group_identification()
    before calling this function.

    Parameters
    ----------
    bus_breaker_int_id : pat.DataFrame[BusBreakerIdSchema]
        The bus breaker IDs.
    edges : pat.DataFrame[EdgeSchema]
        The graph edges.

    Returns
    -------
    DataFrame[BusBreakerIdSchema]
        The bus breaker IDs with electrical circuit groups.
    """
    # create a graph based on all edges we want to consider for the connected components
    edge_tuples = []
    for idx, row in edges.iterrows():
        edge_tuples.append((row["bus_breaker_id_int_bus1"], row["bus_breaker_id_int_bus2"], idx))

    graph = rx.PyGraph(multigraph=True)
    # add ALL breaker ids, no matter if they are connected by a branch or not
    # as later on we also want to map injections that might be connected to an isolated bus_breaker_id
    graph.add_nodes_from(bus_breaker_int_id["bus_breaker_id_int"])
    graph.add_edges_from(edge_tuples)
    connected_nodes = rx.connected_components(graph)
    # map nodes back to bus_breaker_id
    connected_components = {}
    for i, component in enumerate(connected_nodes):
        for node in component:
            connected_components[node] = i
    # map connected components back to bus_breaker_id
    bus_breaker_int_id["electrical_circuit_group"] = bus_breaker_int_id["bus_breaker_id_int"].map(connected_components)
    return bus_breaker_int_id


@pa.check_types
def _get_electrical_circuit_group_branches(
    net: pypowsybl.network.Network, bus_breaker_int_id: pat.DataFrame[BusBreakerIdSchema]
) -> pat.DataFrame[BranchElectricalCircuitGroupSchema]:
    """Get electrical circuit groups for branches in the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    bus_breaker_int_id : pat.DataFrame[BusBreakerIdSchema]
        DataFrame containing bus breaker IDs and their electrical circuit groups.

    Returns
    -------
    DataFrame[BranchElectricalCircuitGroupSchema]
        A DataFrame containing branches and their electrical circuit groups.
    """
    branches = net.get_branches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"])
    # per definition has one branch the same connected component id on both sides -> sufficient to merge on one side only
    branches = branches.merge(
        bus_breaker_int_id[["electrical_circuit_group"]], left_on="bus_breaker_bus1_id", right_index=True, how="left"
    )
    return branches


@pa.check_types
def _get_electrical_circuit_group_switches(
    net: pypowsybl.network.Network, bus_breaker_int_id: pat.DataFrame[BusBreakerIdSchema], keep_fictitious: bool = False
) -> pat.DataFrame[SwitchElectricalCircuitGroupSchema]:
    """Get electrical circuit groups for switches in the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    bus_breaker_int_id : pat.DataFrame[BusBreakerIdSchema]
        DataFrame containing bus breaker IDs and their electrical circuit groups.
    keep_fictitious : bool, optional
        Whether to keep fictitious switches, by default False.

    Returns
    -------
    DataFrame[SwitchElectricalCircuitGroupSchema]
        A DataFrame containing switches and their electrical circuit groups.
    """
    switches = net.get_switches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "kind", "fictitious"])
    switches = switches[
        ((switches["bus_breaker_bus1_id"] != "") | (switches["bus_breaker_bus2_id"] != "")) & (switches["kind"] == "BREAKER")
    ]
    if not keep_fictitious:
        switches = switches[~switches["fictitious"]]
    # per definition has one BREAKER switch has two electrical circuit group ids
    switches = switches.merge(
        bus_breaker_int_id[["electrical_circuit_group"]], left_on="bus_breaker_bus1_id", right_index=True, how="left"
    )
    switches = switches.merge(
        bus_breaker_int_id[["electrical_circuit_group"]],
        left_on="bus_breaker_bus2_id",
        right_index=True,
        how="left",
        suffixes=("_bus1", "_bus2"),
    )
    switches = switches[
        [
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "kind",
            "electrical_circuit_group_bus1",
            "electrical_circuit_group_bus2",
        ]
    ]
    return switches


@pa.check_types
def _get_electrical_circuit_group_injections(
    net: pypowsybl.network.Network, bus_breaker_int_id: pat.DataFrame[BusBreakerIdSchema]
) -> pat.DataFrame[InjectionElectricalCircuitGroupSchema]:
    """Get electrical circuit groups for injections in the network.

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    bus_breaker_int_id : pat.DataFrame[BusBreakerIdSchema]
        DataFrame containing bus breaker IDs and their electrical circuit groups for injections.

    Returns
    -------
    DataFrame[InjectionElectricalCircuitGroupSchema]
        A DataFrame containing injections and their electrical circuit groups.
    """
    injection = net.get_injections(attributes=["bus_breaker_bus_id", "type"])
    injection = injection.merge(
        bus_breaker_int_id[["electrical_circuit_group"]], left_on="bus_breaker_bus_id", right_index=True, how="left"
    )
    injection = injection[injection["electrical_circuit_group"].notnull()]
    return injection


def identify_circuit_groups(
    net: pypowsybl.network.Network, keep_fictitious: bool = False
) -> ElectricalCircuitGroupIdentification:
    """Identify electrical circuit groups in the network.

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
        ElectricalCircuitGroupIdentification(
            circuit_group_map={
                0: ElectricalCircuitGroup(
                    branches=["Line1", "Line2", "Line3"],
                    switches=["BREAKER1", "BREAKER2", "BREAKER3"],
                    injections=[],
                    busbar_section=[],
                )
            },
            branches=<validated branch DataFrame>,
            switches=<validated switch DataFrame>,
            injections=<validated injection DataFrame>,
        )

    Note:

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to analyze.
    keep_fictitious : bool, optional
        Whether to keep fictitious switches, by default False.

    Returns
    -------
    ElectricalCircuitGroupIdentification
        The typed electrical circuit group mapping together with the validated branch, switch,
        and injection tables used to build it.

    Raises
    ------
    NotImplementedError
        If a three winding transformer is present in the network,
        as this is not supported by the current implementation.

    """
    if len(net.get_3_windings_transformers()) > 0:
        raise NotImplementedError(
            "Three winding transformers are not supported by the current implementation. "
            "Please replace them with equivalent two winding transformers before calling this function."
        )
    # create a variant
    variant_name = "outage_group_identification"
    original_variant = net.get_working_variant_id()
    net.clone_variant(src=original_variant, target=variant_name, may_overwrite=True)
    net.set_working_variant(variant_name)
    net_variant = net

    _update_switches_for_outage_group_identification(net=net_variant, keep_fictitious=keep_fictitious)
    bus_breaker_int_id = _get_bus_breaker_int_ids(net=net_variant)
    edges = _get_graph_edges(net=net_variant, bus_breaker_int_id=bus_breaker_int_id)
    bus_breaker_int_id = _get_electrical_circuit_group(bus_breaker_int_id=bus_breaker_int_id, edges=edges)
    branches = _get_electrical_circuit_group_branches(net=net_variant, bus_breaker_int_id=bus_breaker_int_id)
    switches = _get_electrical_circuit_group_switches(
        net=net_variant, bus_breaker_int_id=bus_breaker_int_id, keep_fictitious=keep_fictitious
    )
    injection = _get_electrical_circuit_group_injections(net=net_variant, bus_breaker_int_id=bus_breaker_int_id)

    outage_groups: ElectricalCircuitGroupMap = {
        int(component_id): ElectricalCircuitGroup()
        for component_id in bus_breaker_int_id["electrical_circuit_group"].dropna().unique()
    }
    for idx, row in branches.iterrows():
        component_id = int(row["electrical_circuit_group"])
        outage_groups[component_id].branches.append(idx)
    for idx, row in switches.iterrows():
        component_id1 = int(row["electrical_circuit_group_bus1"])
        component_id2 = int(row["electrical_circuit_group_bus2"])

        outage_groups[component_id1].switches.append(idx)
        outage_groups[component_id2].switches.append(idx)
    for idx, row in injection.iterrows():
        component_id = int(row["electrical_circuit_group"])
        if row["type"] == "BUSBAR_SECTION":
            outage_groups[component_id].busbar_section.append(idx)
        else:
            outage_groups[component_id].injections.append(idx)

    net.set_working_variant(original_variant)
    return ElectricalCircuitGroupIdentification(
        circuit_group_map=outage_groups,
        branches=branches,
        switches=switches,
        injections=injection,
    )
