# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions to identify bridges inside a network."""

import networkx as nx
import numpy as np
from beartype.typing import Optional
from jaxtyping import Bool, Int


def _get_graph_with_branch_keys(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_nodes: int,
) -> nx.MultiGraph:
    """Build a multigraph keyed by branch index for efficient edge toggling.

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_nodes: int
        How many nodes are in the system

    Returns
    -------
    nx.MultiGraph
        A multigraph keyed by branch index for efficient edge toggling.
    """
    graph_nx = nx.MultiGraph()
    graph_nx.add_nodes_from(range(number_of_nodes))
    graph_nx.add_edges_from(
        (int(from_bus), int(to_bus), int(branch_id))
        for branch_id, (from_bus, to_bus) in enumerate(zip(from_node, to_node, strict=True))
    )
    return graph_nx


def _count_bridges_after_outage_on_static_graph(
    cases_to_check: Int[np.ndarray, " n_cases"],
    outage_edges: set[tuple[int, int]],
    graph: nx.MultiGraph,
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
) -> Int[np.ndarray, " n_cases"]:
    """Count outage-relevant bridges by mutating one keyed graph in place.

    Parameters
    ----------
    cases_to_check: Int[np.ndarray, " n_cases"]
        A list of cases that should be checked.
    outage_edges: set[tuple[int, int]]
        A set of edges that are considered outages. Only bridges in the outage_edges are counted.
    graph: nx.MultiGraph
        A multigraph keyed by branch index for efficient edge toggling.
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-b
        us of a branch can be set to the second bus of a substation.

    Returns
    -------
    Int[np.ndarray, " n_cases"]
        Integer Array of length n_cases with the count of bridges after outaging the cases in cases_to_check
    """
    n_bridges = np.zeros(len(cases_to_check), dtype=int)
    for index, branch in enumerate(cases_to_check):
        from_bus = int(from_node[branch])
        to_bus = int(to_node[branch])
        graph.remove_edge(from_bus, to_bus, key=int(branch))
        n_bridges[index] = len(set(nx.bridges(graph)) & outage_edges)
        graph.add_edge(from_bus, to_bus, key=int(branch))
    return n_bridges


def find_bridges(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
) -> Bool[np.ndarray, " n_branch"]:
    """
    Identify branches whose outages would lead to islanding of the network (like bridges to islands)

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_branches: int
        How many branches are in the system
    number_of_nodes: int
        How many busbars are in the system

    Returns
    -------
    Bool[np.ndarray, " n_branch"]
        Boolean Array of length branch that is true for all bridges
    """
    graph_nx = _get_graph_with_branch_keys(from_node, to_node, number_of_nodes)
    # Get bridges using networkx function
    bridges = list(nx.bridges(graph_nx))
    bridges = np.array(bridges, dtype=int)
    if not bridges.any():
        return np.zeros(from_node.size, dtype=bool)
    bridges = np.r_[bridges, bridges[:, [1, 0]]]

    # Networkx gives back the from-to node pairs for each bridge. Therefore we use
    from_to_node = np.c_[from_node, to_node]
    # Faster than previous solution https://stackoverflow.com/a/8317403
    ncols = from_to_node.shape[1]
    dtype = {"names": ["f{}".format(i) for i in range(ncols)], "formats": ncols * [from_to_node.dtype]}
    _, bridge_idx, _ = np.intersect1d(from_to_node.view(dtype), bridges.view(dtype), return_indices=True)
    branch_is_bridge = np.zeros(number_of_branches, dtype=bool)
    branch_is_bridge[bridge_idx] = True
    return branch_is_bridge


def find_islanding_branch_groups(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_nodes: int,
    branch_group_mask: Bool[np.ndarray, " n_groups n_branch"],
) -> Bool[np.ndarray, " n_groups"]:
    """Identify branch groups whose simultaneous outage would island the network.

    A single branch islands the network exactly if it is a bridge, but a group of non-bridges can
    still form a cut set. The DC solver cannot represent an islanded grid: the MODF denominator of
    such a group is singular for every topology, so the group has to be excluded the same way
    :func:`find_bridges` excludes islanding single outages.

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector.
    number_of_nodes: int
        How many nodes are in the system
    branch_group_mask: Bool[np.ndarray, " n_groups n_branch"]
        One row per group, true for every branch that the group outages.

    Returns
    -------
    Bool[np.ndarray, " n_groups"]
        Boolean array of length n_groups, true for every group that islands the network
    """
    if branch_group_mask.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    graph = _get_graph_with_branch_keys(from_node, to_node, number_of_nodes)
    n_components = nx.number_connected_components(graph)

    islands = np.zeros(branch_group_mask.shape[0], dtype=bool)
    for group_index, group in enumerate(branch_group_mask):
        edges = [(int(from_node[branch]), int(to_node[branch]), int(branch)) for branch in np.flatnonzero(group)]
        graph.remove_edges_from(edges)
        islands[group_index] = nx.number_connected_components(graph) > n_components
        graph.add_edges_from(edges)
    return islands


def get_bridge_mainland_node_indices(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
    branch_is_bridge: Bool[np.ndarray, " n_branch"],
    monitored_branch_mask: Bool[np.ndarray, " n_branch"],
    slack: int,
) -> Int[np.ndarray, " n_branch"]:
    """Return the mainland-side endpoint node index for each bridging branch."""
    mainland_node_indices = -np.ones(number_of_branches, dtype=int)
    if not np.any(branch_is_bridge):
        return mainland_node_indices

    graph = nx.Graph()
    graph.add_nodes_from(range(number_of_nodes))
    graph.add_edges_from(zip(from_node.tolist(), to_node.tolist(), strict=True))

    def _component_priority(component_nodes: set[int]) -> tuple[int, int]:
        component_mask = np.zeros(number_of_nodes, dtype=bool)
        component_mask[list(component_nodes)] = True
        monitored_count = int(np.sum(monitored_branch_mask & component_mask[from_node] & component_mask[to_node]))
        return monitored_count, len(component_nodes)

    for bridge_index in np.flatnonzero(branch_is_bridge):
        from_node_index = int(from_node[bridge_index])
        to_node_index = int(to_node[bridge_index])
        if not graph.has_edge(from_node_index, to_node_index):
            continue

        graph.remove_edge(from_node_index, to_node_index)
        components = list(nx.connected_components(graph))
        component_priorities = [_component_priority(component) for component in components]
        max_priority = max(component_priorities, default=(0, 0))
        mainland_candidate_indices = [
            index for index, priority in enumerate(component_priorities) if priority == max_priority
        ]
        if len(mainland_candidate_indices) == 1:
            mainland_component = components[mainland_candidate_indices[0]]
        else:
            slack_candidate_indices = [index for index in mainland_candidate_indices if slack in components[index]]
            mainland_component = (
                components[slack_candidate_indices[0]]
                if len(slack_candidate_indices) == 1
                else components[mainland_candidate_indices[0]]
            )

        mainland_node_indices[bridge_index] = from_node_index if from_node_index in mainland_component else to_node_index
        graph.add_edge(from_node_index, to_node_index)

    return mainland_node_indices


def find_n_minus_2_safe_branches(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
    cases_to_check: Optional[Int[np.ndarray, " n_cases"]] = None,
    outage_cases: Optional[Int[np.ndarray, " n_outage_cases"]] = None,
) -> Bool[np.ndarray, " n_cases"]:
    """Return a boolean array of length branch that is true for all branches that are n-2 safe.

    N-2 safe means that when disconnecting the cases to check, no islanding is happening when the outage_cases are outaged.
    Other branches may still be bridges, but if they are not considered for outaging, this does not matter.
    removed. This way, branches that are not N-1 safe can be ignored.
    This method works by removing each branch and checking if the number of bridges stays the same.
    Hence, it might be slow for large networks.


    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_branches: int
        How many branches are in the system
    number_of_nodes: int
        How many busbars are in the system
    cases_to_check: Optional[Int[np.ndarray, " n_cases"]]
        A list of cases that should be checked. If None, all branches are checked
    outage_cases: Optional[Int[np.ndarray, " n_outage_cases"]]
        A list of outage cases that should be checked. If None, all branches are checked

    Returns
    -------
    Bool[np.ndarray, " n_cases"]
        Boolean Array of length branch that is true for all branches that are n-2 safe
    """
    if cases_to_check is None:
        cases_to_check = np.arange(number_of_branches)
    if outage_cases is None:
        outage_cases = np.arange(number_of_branches)

    outage_edges = set((int(from_node[outage_case]), int(to_node[outage_case])) for outage_case in outage_cases)
    outage_edges |= set((int(to_node[outage_case]), int(from_node[outage_case])) for outage_case in outage_cases)

    base_case = _get_graph_with_branch_keys(from_node, to_node, number_of_nodes)
    n_bridges = len(set(nx.bridges(base_case)) & outage_edges)
    n_bridge_per_outage_case = get_number_of_bridges_after_outage(
        cases_to_check=cases_to_check,
        outage_edges=outage_edges,
        from_node=from_node,
        to_node=to_node,
        number_of_nodes=number_of_nodes,
    )
    n_minus_2_safe = n_bridge_per_outage_case == n_bridges
    return n_minus_2_safe


def get_number_of_bridges_after_outage(
    cases_to_check: Int[np.ndarray, " n_cases"],
    outage_edges: set[tuple[int, int]],
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_nodes: int,
) -> Int[np.ndarray, " n_cases"]:
    """Get the number of bridges in the network after outaging the cases in cases_to_check.

    Parameters
    ----------
    cases_to_check: Int[np.ndarray, " n_cases"]
        A list of cases that should be checked.
    outage_edges: set[tuple[int, int]]
        A set of edges that are considered outages. Only bridges in the outage_edges are counted.
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_nodes: int
        How many busbars are in the system

    Returns
    -------
    Int[np.ndarray, " n_cases"]
        Integer Array of length n_cases with the count of bridges after outaging the cases in cases_to_check
    """
    graph = _get_graph_with_branch_keys(from_node, to_node, number_of_nodes)
    return _count_bridges_after_outage_on_static_graph(cases_to_check, outage_edges, graph, from_node, to_node)
