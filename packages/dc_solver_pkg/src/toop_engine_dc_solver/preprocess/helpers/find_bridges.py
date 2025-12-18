# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions to identify bridges inside a network."""

import math

import networkx as nx
import numpy as np
import ray
from beartype.typing import Optional
from jaxtyping import Bool, Int
from toop_engine_dc_solver.preprocess.helpers.ptdf import get_connectivity_matrix


def get_graph(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
) -> nx.MultiGraph:
    """Get a graph representation of the network.

    Parameters
    ----------
    from_node : Int[np.ndarray, " n_branch"]
        The from-nodes vector. Changes if the topology changes, e.g. the
        from-bus of a branch can be set to the second bus of a substation.
    to_node : Int[np.ndarray, " n_branch"]
        The to-nodes vector. Changes if the topology changes, e.g. the to-bus
        of a branch can be set to the second bus of a substation.
    number_of_branches: int
        The number of branches in the grid
    number_of_nodes: int
        The number of busses in the grid

    Returns
    -------
    nx.MultiGraph
        A graph representation of the network
    """
    connectivity_matrix = get_connectivity_matrix(from_node, to_node, number_of_branches, number_of_nodes, directed=False)
    # The restricted from-node and to-node vectors form a graph.
    # Build the graph as a sparse matrix (graph scipy style)
    graph = connectivity_matrix.T @ connectivity_matrix
    graph_nx = nx.from_scipy_sparse_array(graph, parallel_edges=True, create_using=nx.MultiGraph)
    return graph_nx


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
    graph_nx = get_graph(from_node, to_node, number_of_branches, number_of_nodes)
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


def find_n_minus_2_safe_branches(
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
    cases_to_check: Optional[Int[np.ndarray, " n_cases"]] = None,
    n_processes: Optional[int] = 1,
) -> Bool[np.ndarray, " n_cases"]:
    """Return a boolean array of length branch that is true for all branches that are n-2 safe.

    N-2 safe means that the number of bridges in the network does not increase when the branch is
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
    n_processes: Optional[int]
        Number of processes to use for parallelization. Uses ray if n_processes > 1. Default is 1.

    Returns
    -------
    Bool[np.ndarray, " n_cases"]
        Boolean Array of length branch that is true for all branches that are n-2 safe
    """
    if cases_to_check is None:
        cases_to_check = np.arange(number_of_branches)

    base_case = get_graph(from_node, to_node, number_of_branches, number_of_nodes)
    n_bridges = len(list(nx.bridges(base_case)))
    if n_processes == 1:
        n_bridge_per_outage_case = get_number_of_bridges_after_outage(
            cases_to_check, from_node, to_node, number_of_branches, number_of_nodes
        )
    else:
        n_bridge_per_outage_case = get_number_of_bridges_after_outage_parallel(
            cases_to_check, from_node, to_node, number_of_branches, number_of_nodes, n_processes
        )
    n_minus_2_safe = n_bridge_per_outage_case == n_bridges
    return n_minus_2_safe


def get_number_of_bridges_after_outage(
    cases_to_check: Int[np.ndarray, " n_cases"],
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
) -> Int[np.ndarray, " n_cases"]:
    """Get the number of bridges in the network after outaging the cases in cases_to_check.

    Parameters
    ----------
    cases_to_check: Int[np.ndarray, " n_cases"]
        A list of cases that should be checked.
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
    Int[np.ndarray, " n_cases"]
        Integer Array of length n_cases with the count of bridges after outaging the cases in cases_to_check
    """
    n_bridges = np.zeros(len(cases_to_check), dtype=int)
    for index, branch in enumerate(cases_to_check):
        from_node_temp = np.delete(from_node, branch)
        to_node_temp = np.delete(to_node, branch)
        temp_graph = get_graph(from_node_temp, to_node_temp, number_of_branches - 1, number_of_nodes)
        n_bridges[index] = len(set(nx.bridges(temp_graph)))
    return n_bridges


def get_number_of_bridges_after_outage_parallel(
    cases_to_check: Int[np.ndarray, " n_cases"],
    from_node: Int[np.ndarray, " n_branch"],
    to_node: Int[np.ndarray, " n_branch"],
    number_of_branches: int,
    number_of_nodes: int,
    n_processes: int,
) -> Int[np.ndarray, " n_cases"]:
    """Get the number of bridges in the network after outaging the cases in cases_to_check.

    Runs in parallel using Ray.

    Parameters
    ----------
    cases_to_check: Int[np.ndarray, " n_cases"]
        A list of cases that should be checked.
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
    n_processes: int
        Number of processes to use for parallelization.

    Returns
    -------
    Int[np.ndarray, " n_cases"]
        Integer Array of length n_cases with the count of bridges after outaging the cases in cases_to_check
    """
    batch_size = math.ceil(len(cases_to_check) / n_processes)
    work = [cases_to_check[i : i + batch_size] for i in range(0, len(cases_to_check), batch_size)]
    handles = []
    run_n_2_count_bridges_parallel_worker = ray.remote(get_number_of_bridges_after_outage)
    for batch in work:
        handles.append(
            run_n_2_count_bridges_parallel_worker.remote(batch, from_node, to_node, number_of_branches, number_of_nodes)
        )
    results = ray.get(handles)
    n_bridges_after_outage = np.concatenate(results)
    return n_bridges_after_outage
