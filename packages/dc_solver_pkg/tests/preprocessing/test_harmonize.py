# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""This is commented out until we've revisited the harmonization routines, at the moment they don't make sense."""

# from copy import deepcopy
# from dataclasses import replace
# from pathlib import Path

# import numpy as np
# import pytest

# from dc_solver.preprocess.harmonize import (
#     apply_branch_sets,
#     apply_relevant_substation_set,
#     branch_harmonize,
#     check_branch_harmony,
#     find_translating_indices,
#     get_common_set,
#     harmonize,
#     substation_harmonize,
# )
# from dc_solver.preprocess.network_data import NetworkData
# from dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
# from dc_solver.preprocess.preprocessport preprocess

# def test_check_branch_harmony(data_folder_with_more_branches: Path, network_data_preprocessed: NetworkData) -> None:
#     network_data_1 = network_data_preprocessed
#     branch_hashes_1 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_1.branch_ids,
#                 network_data_1.branch_types,
#                 network_data_1.branch_names,
#             )
#         ]
#     )

#     # Test if the function returns True when the branch assignments are the same
#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#         )[0]
#         is True
#     )

#     backend = PandaPowerBackend(data_folder_with_more_branches)
#     network_data_2 = preprocess(backend)
#     branch_hashes_2 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_2.branch_ids,
#                 network_data_2.branch_types,
#                 network_data_2.branch_names,
#             )
#         ]
#     )

#     assert (
#         check_branch_harmony(
#             network_data_2.branches_at_nodes,
#             branch_hashes_2,
#             network_data_2.branches_at_nodes,
#             branch_hashes_2,
#         )[0]
#         is True
#     )
#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#             network_data_2.branches_at_nodes,
#             branch_hashes_2,
#         )[0]
#         is False
#     )


# def test_harmonize_relevant_substation_set(
#     network_data_preprocessed: NetworkData,
# ) -> None:
#     network_data_1 = network_data_preprocessed
#     network_data_2 = deepcopy(network_data_1)
#     network_datas = [network_data_1, network_data_2]
#     node_hashes = [
#         np.array(
#             [
#                 hash(data)
#                 for data in zip(
#                     network_data.node_ids,
#                     network_data.node_types,
#                     network_data.node_names,
#                 )
#             ]
#         )
#         for network_data in network_datas
#     ]
#     branch_hashes = [
#         np.array(
#             [
#                 hash(data)
#                 for data in zip(
#                     network_data.branch_ids,
#                     network_data.branch_types,
#                     network_data.branch_names,
#                 )
#             ]
#         )
#         for network_data in network_datas
#     ]

#     # The two network datas have equal substations, so the function should return the same set
#     reference = node_hashes[0][network_data_1.relevant_node_mask]

#     res, _ = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in network_datas],
#         node_hashes,
#     )
#     assert np.array_equal(res, reference)
#     old_len = len(res)

#     # Remove one relevant substation
#     zero_idx = np.flatnonzero(network_data_1.relevant_node_mask)[0]
#     new_mask = np.copy(network_data_1.relevant_node_mask)
#     new_mask[zero_idx] = False
#     network_data_1 = replace(
#         network_data_1,
#         relevant_node_mask=new_mask,
#         num_branches_per_node=network_data_1.num_branches_per_node[1:],
#         branches_at_nodes=network_data_1.branches_at_nodes[1:],
#         branch_direction=network_data_1.branch_direction[1:],
#     )
#     network_datas = [network_data_1, network_data_2]

#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes[0],
#             network_data_2.branches_at_nodes,
#             branch_hashes[1],
#         )[0]
#         is False
#     )

#     reference = node_hashes[0][network_data_1.relevant_node_mask]

#     res, _ = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in network_datas],
#         node_hashes,
#     )
#     # Also works the other way around
#     res_2, _ = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in reversed(network_datas)],
#         list(reversed(node_hashes)),
#     )

#     assert np.array_equal(res, reference)
#     assert np.array_equal(res_2, reference)
#     assert len(res) == old_len - 1

#     network_data_3 = apply_relevant_substation_set(network_data_2, res, node_hashes[1])

#     assert len(network_data_3.branches_at_nodes) == len(res)
#     assert len(network_data_3.branch_direction) == len(res)
#     assert len(network_data_3.num_branches_per_node) == len(res)
#     assert len(network_data_3.injection_idx_at_nodes) == len(res)
#     assert len(network_data_3.active_injections) == len(res)
#     assert len(network_data_3.num_injections_per_node) == len(res)
#     assert np.array_equal(network_data_3.relevant_node_mask, network_data_1.relevant_node_mask)
#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes[0],
#             network_data_3.branches_at_nodes,
#             branch_hashes[1],
#         )[0]
#         is True
#     )

#     network_datas_harmonized, delta = substation_harmonize([network_data_1, network_data_2], hash)
#     assert np.array_equal(
#         network_datas_harmonized[0].relevant_node_mask,
#         network_data_1.relevant_node_mask,
#     )
#     assert np.array_equal(
#         network_datas_harmonized[1].relevant_node_mask,
#         network_data_3.relevant_node_mask,
#     )
#     assert np.array_equal(delta, np.array([0, 1]))

#     # Test if the function raises an error when the relevant substations are not in the correct order
#     network_data_5 = replace(
#         network_data_2,
#         relevant_node_mask=network_data_1.relevant_node_mask[::-1],
#         node_ids=list(reversed(network_data_2.node_ids)),
#         node_types=list(reversed(network_data_2.node_types)),
#         node_names=list(reversed(network_data_2.node_names)),
#         branches_at_nodes=list(reversed(network_data_2.branches_at_nodes)),
#         branch_direction=list(reversed(network_data_2.branch_direction)),
#         num_branches_per_node=network_data_2.num_branches_per_node[::-1],
#     )
#     node_hashes_5 = node_hashes[1][::-1]

#     res_3, _ = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in [network_data_1, network_data_5]],
#         [node_hashes[0], node_hashes_5],
#     )
#     res_4, _ = get_common_set(
#         [np.flatnonzero(network_data.relevant_node_mask) for network_data in [network_data_5, network_data_1]],
#         [node_hashes_5, node_hashes[0]],
#     )

#     assert np.array_equal(res_3, res)
#     assert not np.array_equal(res_4, res)
#     with pytest.raises(ValueError):
#         apply_relevant_substation_set(network_data_5, res, node_hashes_5)


# def test_find_translating_indices() -> None:
#     # Test if the function returns the correct indices
#     a = np.array([1, 2, 3, 4, 5])
#     b = np.array([5, 4, 3])
#     res = find_translating_indices(new=b, old=a)
#     assert np.array_equal(res, np.array([4, 3, 2]))

#     res = find_translating_indices(new=a, old=a)
#     assert np.array_equal(res, np.array([0, 1, 2, 3, 4]))

#     with pytest.raises(ValueError):
#         find_translating_indices(new=a, old=b)


# def test_harmonize_branches(data_folder_with_more_branches: Path, network_data_preprocessed: NetworkData) -> None:
#     network_data_1 = network_data_preprocessed
#     branch_hashes_1 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_1.branch_ids,
#                 network_data_1.branch_types,
#                 network_data_1.branch_names,
#             )
#         ]
#     )
#     backend = PandaPowerBackend(data_folder_with_more_branches)
#     network_data_2 = preprocess(backend)
#     branch_hashes_2 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_2.branch_ids,
#                 network_data_2.branch_types,
#                 network_data_2.branch_names,
#             )
#         ]
#     )

#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#             network_data_2.branches_at_nodes,
#             branch_hashes_2,
#         )[0]
#         is False
#     )

#     reference = [branch_hashes_1[branches_at_this_node] for branches_at_this_node in network_data_1.branches_at_nodes]
#     n_relevant_subs = sum(network_data_1.relevant_node_mask)
#     assert sum(network_data_2.relevant_node_mask) == n_relevant_subs

#     # The common set with itself should be just all branches at the nodes
#     res = [
#         get_common_set(
#             [network_data_1.branches_at_nodes[i], network_data_1.branches_at_nodes[i]],
#             [branch_hashes_1, branch_hashes_1],
#         )[0]
#         for i in range(n_relevant_subs)
#     ]

#     assert len(res) == len(reference)
#     assert all([np.array_equal(res[i], reference[i]) for i in range(n_relevant_subs)])

#     # The second network data has more branches, so the common set should be exactly the branches
#     # in network_data 1
#     reference2 = [branch_hashes_2[branches_at_this_node] for branches_at_this_node in network_data_2.branches_at_nodes]
#     res = [
#         get_common_set(
#             [network_data_1.branches_at_nodes[i], network_data_2.branches_at_nodes[i]],
#             [branch_hashes_1, branch_hashes_2],
#         )[0]
#         for i in range(n_relevant_subs)
#     ]
#     assert not all([np.array_equal(res[i], reference2[i]) for i in range(n_relevant_subs)])
#     assert all([np.array_equal(res[i], reference[i]) for i in range(n_relevant_subs)])
#     assert len(res) == len(reference)

#     # Network data 2 needs to be shrunk down
#     network_data_3 = apply_branch_sets(network_data_2, res, branch_hashes_2)
#     branch_hashes_3 = branch_hashes_2
#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#             network_data_3.branches_at_nodes,
#             branch_hashes_3,
#         )[0]
#         is True
#     )

#     assert np.sum(network_data_1.num_branches_per_node) == np.sum(network_data_3.num_branches_per_node)

#     # This should be the same
#     network_datas_harmonized, delta = branch_harmonize([network_data_1, network_data_2], hash)
#     assert np.array_equal(
#         network_datas_harmonized[0].num_branches_per_node,
#         network_data_3.num_branches_per_node,
#     )
#     assert np.any(delta[1] == 1)
#     assert np.all(delta[0] == 0)


# def test_harmonize_network_datas(data_folder_with_more_branches: Path, network_data_preprocessed: NetworkData) -> None:
#     network_data_1 = network_data_preprocessed
#     branch_hashes_1 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_1.branch_ids,
#                 network_data_1.branch_types,
#                 network_data_1.branch_names,
#             )
#         ]
#     )
#     backend = PandaPowerBackend(data_folder_with_more_branches)
#     network_data_2 = preprocess(backend)
#     branch_hashes_2 = np.array(
#         [
#             hash(data)
#             for data in zip(
#                 network_data_2.branch_ids,
#                 network_data_2.branch_types,
#                 network_data_2.branch_names,
#             )
#         ]
#     )

#     assert (
#         check_branch_harmony(
#             network_data_1.branches_at_nodes,
#             branch_hashes_1,
#             network_data_2.branches_at_nodes,
#             branch_hashes_2,
#         )[0]
#         is False
#     )

#     [network_data_3, network_data_4], delta_sub, delta_branch = harmonize([network_data_1, network_data_2])

#     assert (
#         check_branch_harmony(
#             network_data_3.branches_at_nodes,
#             branch_hashes_1,
#             network_data_4.branches_at_nodes,
#             branch_hashes_2,
#         )[0]
#         is True
#     )
#     assert np.array_equal(delta_sub, np.array([0, 0]))
#     assert np.sum(delta_branch[1]) >= 1
#     assert np.sum(delta_branch[0]) == 0
#     assert delta_branch.shape == (2, len(network_data_1.branches_at_nodes))
