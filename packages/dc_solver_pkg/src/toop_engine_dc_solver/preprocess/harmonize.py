# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Harmonize network data classes.

The aim of this module is to harmonize a set of network data classes so they can be used in a
joint optimization for multiple timesteps. However with the recent changes, this module has gone quite out of date.
Actually it has been wrong ever since because we want to approach a similar-ptdf setting for the multi-timestamp
optimization. This module is commented out until we decide to take up the multi-timestep optimization again.
"""

# from dataclasses import replace
# from functools import reduce
# from typing import Callable, Union

# import numpy as np
# from jaxtyping import Bool, Int

# from dc_solver.preprocess.network_data import NetworkData

# HashT = Int


# def check_substation_harmony(
#     relevant_node_mask_1: Bool[np.ndarray, " n_relevant_subs"],
#     node_hashes_1: HashT[np.ndarray, " n_nodes"],
#     relevant_node_mask_2: Bool[np.ndarray, " n_relevant_subs"],
#     node_hashes_2: HashT[np.ndarray, " n_nodes"],
# ) -> tuple[bool, str]:
#     """Check whether two sets of relevant substations are the same.

#     Uses the hash arrays to check for equality, as indices might differ.

#     Parameters
#     ----------
#     relevant_node_mask_1 : Bool[np.ndarray, "n_relevant_subs"]
#         The relevant substations for the first network data object.
#     node_hashes_1 : HashT[np.ndarray, "n_nodes"]
#         The hash array for the nodes in the first network data object.
#     relevant_node_mask_2 : Bool[np.ndarray, "n_relevant_subs"]
#         The relevant substations for the second network data object.
#     node_hashes_2 : HashT[np.ndarray, "n_nodes"]
#         The hash array for the nodes in the second network data object.

#     Returns
#     -------
#     bool
#         True if the relevant substations are the same, False otherwise.
#     str
#         A message that describes the difference if the relevant substations are not the same.
#     """
#     rel_hashes_1 = node_hashes_1[relevant_node_mask_1]
#     rel_hashes_2 = node_hashes_2[relevant_node_mask_2]

#     if not np.array_equal(rel_hashes_1, rel_hashes_2):
#         if np.all(np.isin(rel_hashes_1, rel_hashes_2)):
#             return False, "Relevant substations are not in the same order."
#         diff = rel_hashes_1[~np.isin(rel_hashes_1, rel_hashes_2)]
#         return False, f"Relevant substations are not the same, diff: {diff}."

#     return True, ""


# def check_branch_harmony(
#     branches_at_nodes_1: list[Int[np.ndarray, " n_branches_at_nodes"]],
#     branch_hashes_1: HashT[np.ndarray, " n_branches"],
#     branches_at_nodes_2: list[Int[np.ndarray, " n_branches_at_nodes"]],
#     branch_hashes_2: HashT[np.ndarray, " n_branches"],
# ) -> tuple[bool, str]:
#     """Check whether two branches_at_nodes have the same branch assignments.

#     Uses the hash arrays to check for equality, as indices might differ.

#     Parameters
#     ----------
#     branches_at_nodes_1 : list[Int[np.ndarray, " n_branches_at_nodes"]]
#         The branch assignments at each node for the first network data object.
#     branch_hashes_1 : NDArray[Shape[" N_branches"], HashT]
#         The hash array for the branches in the first network data object.
#     branches_at_nodes_2 : list[Int[np.ndarray, " n_branches_at_nodes"]]
#         The branch assignments at each node for the second network data object.
#     branch_hashes_2 : NDArray[Shape[" N_branches"], HashT]
#         The hash array for the branches in the second network data object.

#     Returns
#     -------
#     bool
#         True if the branch assignments are the same, False otherwise.
#     str
#         A message that describes the difference if the branch assignments are not the same.
#     """
#     if len(branches_at_nodes_1) != len(branches_at_nodes_2):
#         return (
#             False,
#             f"Number of substations is different: {len(branches_at_nodes_1)} vs {len(branches_at_nodes_2)}",
#         )

#     for sub_id, (branches_1, branches_2) in enumerate(zip(branches_at_nodes_1, branches_at_nodes_2)):
#         if not np.array_equal(branch_hashes_1[branches_1], branch_hashes_2[branches_2]):
#             if np.all(np.isin(branch_hashes_1[branches_1], branch_hashes_2[branches_2])):
#                 return (
#                     False,
#                     f"Branches at substation {sub_id} are not in the same order.",
#                 )
#             diff = branch_hashes_1[branches_1][~np.isin(branch_hashes_1[branches_1], branch_hashes_2[branches_2])]
#             return (
#                 False,
#                 f"Branches at substation {sub_id} are not the same, diff: {diff}.",
#             )

#     return True, ""


# def get_common_set(
#     indices: list[Int[np.ndarray, " n_relevant_elems"]],
#     hashes: list[HashT[np.ndarray, " n_elems"]],
# ) -> tuple[
#     HashT[np.ndarray, " n_common_elems"],
#     list[HashT[np.ndarray, " n_diff_elems"]],
# ]:
#     """Find a common set that is present in every network in the list

#     A network need only be represented by a set of hashes that are globally unique and the indices
#     into the hash array that are relevant.

#     It returns the order as in the first network.

#     Parameters
#     ----------
#     indices : list[Int[np.ndarray, " n_relevant_elems"]]
#         The indices of the relevant elements in the hash array for each network, the length of the
#         list is the number of networks
#     hashes : list[HashT[np.ndarray, " n_elems"]]
#         The hash array for each network, the length of the list is the number of networks

#     Returns
#     -------
#     HashT[np.ndarray, " N_common_elems"]
#         The common set of relevant elements, represented by their hash
#     list[HashT[np.ndarray, " n_diff_elems"]]
#         The set of relevant elements that are not in the common set for each network
#     """
#     if len(indices) != len(hashes):
#         raise ValueError("Length of indices and hashes must be equal.")
#     # We only need the hashes
#     networks = [h[i] for i, h in zip(indices, hashes)]

#     # Find out which elements from the first entry are present in all other entries
#     common_set = reduce(
#         np.logical_and,
#         (np.isin(networks[0], network) for network in networks[1:]),
#     )

#     common_hashes = networks[0][common_set]

#     # Find out which elements are not in the common set for each network
#     diff_sets = [network[~np.isin(network, common_hashes)] for network in networks]

#     return common_hashes, diff_sets


# def apply_relevant_substation_set(
#     network_data: NetworkData,
#     common_relevant_subs: HashT[np.ndarray, " n_common_elems"],
#     node_hashes: HashT[np.ndarray, " n_nodes"],
# ) -> NetworkData:
#     """Apply a common set of relevant substations to a network data object.

#     Parameters
#     ----------
#     network_data : NetworkData
#         The network data object.
#     common_relevant_subs : HashT[np.ndarray, " n_common_elems"],
#         The common set of relevant substations represented by their hash, can be obtained by
#         get_common_set.
#     node_hashes : HashT[np.ndarray, " n_nodes"],
#         The hash array for the nodes in the network data object.

#     Returns
#     -------
#     NetworkData
#         The network data object with the common set of relevant substations.
#     """
#     assert len(node_hashes) == len(network_data.relevant_node_mask)

#     # First check if the node_hashes are in the correct order
#     # We currently don't have a mechanism to deal with different orders in the bus data
#     mask = np.isin(node_hashes, common_relevant_subs)
#     if not np.array_equal(node_hashes[mask], common_relevant_subs):
#         raise ValueError(
#             "The relevant substations are not in the correct order or the common subs "
#             + "are actually not in common, can't harmonize. This could also be caused by "
#             + "a hash collision."
#         )

#     current_relevant_subs = node_hashes[network_data.relevant_node_mask]
#     still_relevant = np.flatnonzero(np.isin(current_relevant_subs, common_relevant_subs))

#     return replace(
#         network_data,
#         relevant_node_mask=mask,
#         num_branches_per_node=network_data.num_branches_per_node[still_relevant],
#         branches_at_nodes=[network_data.branches_at_nodes[i] for i in still_relevant],
#         branch_direction=[network_data.branch_direction[i] for i in still_relevant],
#         injection_idx_at_nodes=[network_data.injection_idx_at_nodes[i] for i in still_relevant],
#         injection_combi=[network_data.injection_combi[i] for i in still_relevant],
#         injection_combi_bool=[network_data.injection_combi_bool[i] for i in still_relevant],
#         num_injections_per_node=network_data.num_injections_per_node[still_relevant],
#         active_injections=[network_data.active_injections[i] for i in still_relevant],
#     )


# def find_translating_indices(new: Int[np.ndarray, " n_new"], old: Int[np.ndarray, " n_old"]) -> Int[np.ndarray, " n_new"]:
#     """Find the indices that translate the old indices to the new indices.

#     It is assumed that new is a subset and reordering of old and that values in new and old are unique.

#     Parameters
#     ----------
#     new : Int[np.ndarray, " n_new"]
#         The new indices.
#     old : Int[np.ndarray, " n_old"]
#         The old indices.

#     Returns
#     -------
#     Int[np.ndarray, " n_new"]
#         The indices that translate the old indices to the new indices, i.e. old[translation] == new
#     """
#     comp_matrix = new[:, None] == old[None, :]
#     row_sum = np.sum(comp_matrix, axis=1).min()
#     if row_sum != 1:
#         raise ValueError("The new indices are not a subset of the old indices.")
#     return comp_matrix @ np.arange(len(old))


# def apply_branch_sets(
#     network_data: NetworkData,
#     common_branches: list[HashT[np.ndarray, " n_common_elems"]],
#     branch_hashes: HashT[np.ndarray, " n_branches"],
# ) -> NetworkData:
#     """Apply a common set of branches at each relevant substation to a network data object.

#     Parameters
#     ----------
#     network_data : NetworkData
#         The network data object.
#     common_branches : list[HashT[np.ndarray, " n_common_elems"]],
#         The common set of branches at each relevant substation represented by a hash array.
#         The outer list has length N_relevant_subs and each inner list denotes the set and order of
#         common branches in all network data objects, as obtained by get_common_set.
#     branch_hashes : HashT[np.ndarray, " n_branches"],
#         The hash array for the branches in the network data object.

#     Returns
#     -------
#     NetworkData
#         The network data object with the common set of branches at each relevant substation.
#     """
#     new_branches_at_nodes = []
#     new_branch_direction = []

#     for branches_local, branch_direction, branches_common in zip(
#         network_data.branches_at_nodes, network_data.branch_direction, common_branches
#     ):
#         branch_hashes_local = branch_hashes[branches_local]
#         translation = find_translating_indices(new=branches_common, old=branch_hashes_local)

#         new_branches_at_nodes.append(branches_local[translation])
#         new_branch_direction.append(branch_direction[translation])

#     return replace(
#         network_data,
#         branches_at_nodes=new_branches_at_nodes,
#         branch_direction=new_branch_direction,
#         num_branches_per_node=np.array([len(x) for x in new_branches_at_nodes], dtype=int),
#     )


# def substation_harmonize(
#     network_datas: list[NetworkData],
#     hash_func: Callable[[tuple[Union[str, int], str, str]], int],
# ) -> tuple[
#     list[NetworkData],
#     Int[np.ndarray, " n_network_datas"],
# ]:
#     """Harmonize the relevant substations of a list of network datas.

#     This computes the common set of relevant substations and then reduces every network data object
#     to only have this common set as relevant substations. Hence, the number of relevant substations
#     can be reduced.

#     Parameters
#     ----------
#     network_datas : list[NetworkData]
#         The list of network data objects to harmonize
#     hash_func : Callable[[tuple[Union[str, int], str, str]], int]
#         The hash function that returns a globally unique hash for a node, given its
#         id, type and name.

#     Returns
#     -------
#     list[NetworkData]
#         The harmonized list of network data objects
#     Int[np.ndarray, " n_network_datas"],
#         The delta in the number of substations for each network data object, i.e. how many relevant
#         substations were removed

#     """
#     if len(network_datas) < 2:
#         raise ValueError("At least two network data objects are required for harmonization.")

#     node_hashes = [
#         np.array(
#             [
#                 hash_func(data)
#                 for data in zip(
#                     network_data.node_ids,
#                     network_data.node_types,
#                     network_data.node_names,
#                 )
#             ]
#         )
#         for network_data in network_datas
#     ]

#     # First harmonize relevant substations
#     common_relevant_subs, _diff_sets = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in network_datas],
#         node_hashes,
#     )

#     delta_sub = np.array([len(network_data.branches_at_nodes) -
#                           len(common_relevant_subs) for network_data in network_datas])
#     network_datas = [
#         apply_relevant_substation_set(network_data, common_relevant_subs, node_hash)
#         for (node_hash, network_data) in zip(node_hashes, network_datas)
#     ]

#     for node_hash, network_data in zip(node_hashes[1:], network_datas[1:]):
#         match, err = check_substation_harmony(
#             network_datas[0].relevant_node_mask,
#             node_hashes[0],
#             network_data.relevant_node_mask,
#             node_hash,
#         )
#         assert match, err

#     return network_datas, delta_sub


# def branch_harmonize(
#     network_datas: list[NetworkData],
#     hash_func: Callable[[tuple[Union[str, int], str, str]], int],
# ) -> tuple[
#     list[NetworkData],
#     np.ndarray,  # NDArray[Shape[" N_network_datas, N_relevant_subs"], Int],
# ]:
#     """Harmonize the branches of a list of network datas.

#     This computes the common set of branches at each relevant substation and then reduces every
#     network data object to only have this common set as branches at each relevant substation. Hence,
#     looking at only relevant substation, the same branches are present in every network data object.
#     No branches are added, only removed.

#     Parameters
#     ----------
#     network_datas : list[NetworkData]
#         The list of network data objects to harmonize
#     hash_func : Callable[[tuple[Union[str, int], str, str]], int]
#         The hash function that returns a globally unique hash for a branch, given its
#         id, type and name.

#     Returns
#     -------
#     list[NetworkData]
#         The harmonized list of network data objects
#     NDArray[Shape[" N_network_datas, N_relevant_subs"], Int]
#         The delta in the number of branches for each relevant substation in each network data object,
#         i.e. how many branches were removed for each relevant substation
#     """
#     if len(network_datas) < 2:
#         raise ValueError("At least two network data objects are required for harmonization.")

#     branch_hashes = [
#         np.array(
#             [
#                 hash_func(data)
#                 for data in zip(
#                     network_data.branch_ids,
#                     network_data.branch_types,
#                     network_data.branch_names,
#                 )
#             ]
#         )
#         for network_data in network_datas
#     ]

#     n_relevant_subs = len(network_datas[0].branches_at_nodes)
#     common_branches = [
#         get_common_set(
#             [network_data.branches_at_nodes[i] for network_data in network_datas],
#             branch_hashes,
#         )
#         for i in range(n_relevant_subs)
#     ]
#     common_branch_hashes = [common for common, _delta in common_branches]
#     delta_branch = np.array(
#         [
#             [len(network_data.branches_at_nodes[i]) -
#              len(common_branch_hashes[i]) for i in range(len(common_branch_hashes))]
#             for network_data in network_datas
#         ]
#     )
#     network_datas = [
#         apply_branch_sets(network_data, common_branch_hashes, branch_hash)
#         for branch_hash, network_data in zip(branch_hashes, network_datas)
#     ]

#     for branch_hash, network_data in zip(branch_hashes[1:], network_datas[1:]):
#         match, err = check_branch_harmony(
#             network_datas[0].branches_at_nodes,
#             branch_hashes[0],
#             network_data.branches_at_nodes,
#             branch_hash,
#         )
#         if not match:
#             raise ValueError(err)

#     return network_datas, delta_branch


# def harmonize(
#     network_datas: list[NetworkData],
#     hash_func: Callable[[tuple[Union[str, int], str, str]], int] = hash,
# ) -> tuple[
#     list[NetworkData],
#     Int[np.ndarray, " n_network_datas"],
#     Int[np.ndarray, " n_network_datas n_relevant_subs"],
# ]:
#     """Harmonize a list of network data objects.

#     This performs two things, at first it harmonizes substations with substation_harmonize and then
#     harmonizes branches with branch_harmonize. See the documentation of these functions for more
#     information.

#     Parameters
#     ----------
#     network_datas : list[NetworkData]
#         The list of network data objects to harmonize
#     hash_func : Callable[[tuple[Union[str, int], str, str]], HashT]
#         The hash function that returns a globally unique hash for a branch/node, given its
#         id, type and name. Defaults to the built-in hash function.

#     Returns
#     -------
#     list[NetworkData]
#         The harmonized list of network data objects
#     Int[np.ndarray, " n_network_datas"],
#         The delta in the number of substations for each network data object, i.e. how many relevant
#         substations were removed to match the substations to the largest common set
#     Int[np.ndarray, " n_network_datas n_relevant_subs"],
#         The delta in the number of branches for each relevant substation in each network data object,
#         i.e. how many branches were removed for each relevant substation
#     """
#     if len(network_datas) < 2:
#         raise ValueError("At least two network data objects are required for harmonization.")

#     network_datas, delta_sub = substation_harmonize(network_datas, hash_func)
#     network_datas, delta_branch = branch_harmonize(network_datas, hash_func)
#     # TODO: Implement injection harmonization

#     return network_datas, delta_sub, delta_branch
