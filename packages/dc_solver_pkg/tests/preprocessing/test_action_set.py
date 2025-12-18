# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.injections import get_injection_vector
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    StaticInformation,
)
from toop_engine_dc_solver.preprocess.action_set import (
    enumerate_branch_actions,
    enumerate_branch_actions_for_sub,
    filter_splits_by_bridge_lookup,
    filter_splits_by_bsdf,
    make_action_repo,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess import (
    add_bus_b_columns_to_ptdf,
    add_missing_asset_topo_info,
    add_nodal_injections_to_network_data,
    assert_network_data,
    combine_phaseshift_and_injection,
    compute_branch_topology_info,
    compute_bridging_branches,
    compute_injection_actions,
    compute_injection_topology_info,
    compute_psdf_if_not_given,
    compute_ptdf_if_not_given,
    convert_multi_outages,
    exclude_bridges_from_outage_masks,
    filter_disconnectable_branches_nminus2,
    filter_inactive_injections,
    filter_relevant_nodes_branch_count,
    process_injection_outages,
    reduce_branch_dimension,
    remove_relevant_subs_without_actions,
)
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import (
    enumerate_spreaded_nodal_injections_for_rel_subs,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_make_action_repo() -> None:
    repo = make_action_repo(4)

    assert repo.shape[1] == 4

    # All unique
    assert len(set([tuple(x) for x in repo])) == repo.shape[0]

    # Unique against inversion
    assert set([tuple(x) for x in repo]).intersection(set([tuple(~x) for x in repo])) == set()

    # Exclude isolations
    assert jnp.all((jnp.sum(repo, axis=1) >= 2) | (jnp.sum(repo, axis=1) == 0))

    # First one unsplit
    assert not np.any(repo[0])

    # Including isolations increases repo size
    repo2 = make_action_repo(4, exclude_isolations=False)
    assert repo2.shape[0] > repo.shape[0]
    assert jnp.any(jnp.sum(repo2, axis=1) == 1)

    # Including inverses increases repo size
    repo3 = make_action_repo(4, exclude_inverse=False)
    assert repo3.shape[0] == 2 * repo.shape[0]

    assert len(set([tuple(x) for x in repo3])) == repo3.shape[0]
    assert set([tuple(x) for x in repo3]).intersection(set([tuple(~x) for x in repo3])) != set()

    # With both options set to false, we get all 2**4 binary combinations
    repo4 = make_action_repo(4, exclude_isolations=False, exclude_inverse=False)
    assert repo4.shape[0] == 2**4
    assert len(set([tuple(x) for x in repo4])) == repo4.shape[0]

    # When passing in fixed assignments, these elements are always set to the fixed value
    repo5 = make_action_repo(7, fixed_assignments=((0, True), (3, False)))
    assert np.all(repo5[1:, 0] == 1)
    assert np.all(repo5[1:, 3] == 0)
    assert repo5.shape[0] == make_action_repo(5).shape[0]

    repo6 = make_action_repo(
        7,
        fixed_assignments=((0, True), (3, False)),
        exclude_isolations=False,
        exclude_inverse=False,
    )
    assert np.all(repo6[1:, 0] == 1)
    assert np.all(repo6[1:, 3] == 0)

    # When passing a randomly_select threshold, at most this many actions are selected
    repo = make_action_repo(7, randomly_select=20)

    # All unique
    assert len(set([tuple(x) for x in repo])) == repo.shape[0]

    # Unique against inversion
    assert set([tuple(x) for x in repo]).intersection(set([tuple(~x) for x in repo])) == set()


@pytest.mark.timeout(5)
def test_make_action_repo_large() -> None:
    # Work sensible even for rather large amount of branches
    make_action_repo(20)


def test_filter_splits(network_data: NetworkData) -> None:
    # Manually preprocess but without computing branch actions
    network_data = compute_bridging_branches(network_data)
    # network_data = filter_relevant_nodes(network_data)
    assert_network_data(network_data)
    network_data = compute_ptdf_if_not_given(network_data)
    network_data = compute_psdf_if_not_given(network_data)
    network_data = add_nodal_injections_to_network_data(network_data)
    network_data = combine_phaseshift_and_injection(network_data)

    network_data = exclude_bridges_from_outage_masks(network_data)
    network_data = reduce_branch_dimension(network_data)
    network_data = filter_disconnectable_branches_nminus2(network_data)

    network_data = compute_branch_topology_info(network_data)

    network_data = convert_multi_outages(network_data)
    network_data = filter_inactive_injections(network_data)
    network_data = compute_injection_topology_info(network_data)
    network_data = process_injection_outages(network_data)
    # network_data = add_bus_b_columns_to_ptdf(network_data)
    # network_data = compute_branch_actions(network_data)

    # Bus 22 has a lot of stubs, check if the actions on it are filtered properly
    node_idx = network_data.node_names.index("Bus 22")
    # Count how many relevant subs are before this one, to get the right index
    relevant_sub_idx = int(sum(network_data.relevant_node_mask[:node_idx]))
    node_degree = len(network_data.branches_at_nodes[relevant_sub_idx])

    repo = make_action_repo(node_degree)

    filtered_repo = filter_splits_by_bridge_lookup(
        relevant_sub_idx,
        repo,
        network_data,
    )

    assert filtered_repo.shape[0] <= repo.shape[0]
    assert filtered_repo.shape[1] == repo.shape[1]

    # In this case we know there are no actions possible except for the unsplit action as all
    # the lines ending in the station are stub lines
    assert filtered_repo.shape == (1, node_degree)
    assert not np.any(filtered_repo)

    filtered_repo_2 = filter_splits_by_bsdf(
        relevant_sub_idx,
        repo,
        network_data,
    )
    assert filtered_repo_2.shape == (1, node_degree)


def test_enumerate_branch_actions(network_data: NetworkData) -> None:
    # Manually preprocess but without computing branch actions
    network_data = compute_bridging_branches(network_data)
    network_data = filter_relevant_nodes_branch_count(network_data)
    assert_network_data(network_data)
    network_data = compute_ptdf_if_not_given(network_data)
    network_data = compute_psdf_if_not_given(network_data)
    network_data = add_nodal_injections_to_network_data(network_data)
    network_data = combine_phaseshift_and_injection(network_data)

    network_data = exclude_bridges_from_outage_masks(network_data)
    network_data = reduce_branch_dimension(network_data)
    network_data = filter_disconnectable_branches_nminus2(network_data)

    network_data = compute_branch_topology_info(network_data)
    # network_data = compute_branch_actions(network_data)
    sub_id = 0
    valid_actions = enumerate_branch_actions_for_sub(sub_id, network_data)

    degree = len(network_data.branches_at_nodes[sub_id])

    assert valid_actions.shape[1] == (degree)
    assert valid_actions.shape[0] <= 2**degree
    assert valid_actions.shape[0] > 0
    assert not np.any(valid_actions[0])

    all_valid_actions = enumerate_branch_actions(network_data)
    assert np.array_equal(valid_actions, all_valid_actions[sub_id])
    total_actions_ref = sum([len(x) for x in all_valid_actions])

    # Should return more when not excluding actions
    all_valid_actions = enumerate_branch_actions(
        network_data,
        exclude_isolations=False,
        exclude_inverse=False,
        exclude_bridge_lookup_splits=False,
    )

    assert sum([len(x) for x in all_valid_actions]) >= total_actions_ref


def test_enumerate_action_set_converges(preprocessed_data_folder: Path) -> None:
    static_information = load_static_information(
        preprocessed_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    # Try every action in the action set
    actions = ActionIndexComputations(
        action=jnp.arange(static_information.n_actions)[:, None],
        pad_mask=jnp.ones((static_information.n_actions,), dtype=bool),
    )

    _, success = run_solver_symmetric(
        topologies=actions,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda _x: None,
    )
    assert jnp.all(success)


def test_enumerate_action_set_converges_powsybl(preprocessed_powsybl_data_folder: Path) -> None:
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    # Try every action in the action set
    actions = ActionIndexComputations(
        action=jnp.arange(static_information.n_actions)[:, None],
        pad_mask=jnp.ones((static_information.n_actions,), dtype=bool),
    )

    _, success = run_solver_symmetric(
        topologies=actions,
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda _x: None,
    )
    assert jnp.all(success)

    # This is just the same check again, might be useful for debugging purposes
    # topo_vect = convert_action_set_index_to_topo(actions, static_information.dynamic_information.action_set)
    # for i in range(len(topo_vect)):
    #     topo = topo_vect[i]

    #     bsdf_success, _, lodf_success = inspect_topology(
    #         topology=topo.topologies,
    #         sub_ids=topo.sub_ids,
    #         disconnections=None,
    #         static_information=static_information,
    #     )
    #     assert jnp.all(bsdf_success)
    #     assert jnp.all(lodf_success)


def test_remove_relevant_subs_without_actions(
    network_data: NetworkData,
) -> None:
    # Manually preprocess but without computing branch actions
    network_data = compute_bridging_branches(network_data)
    # We want to avoid filtering bridges here, so we set the mask to all False
    network_data = replace(network_data, bridging_branch_mask=np.zeros_like(network_data.bridging_branch_mask))
    network_data = filter_relevant_nodes_branch_count(network_data)
    network_data = compute_bridging_branches(network_data)
    assert_network_data(network_data)
    network_data = compute_ptdf_if_not_given(network_data)
    network_data = compute_psdf_if_not_given(network_data)
    network_data = add_nodal_injections_to_network_data(network_data)
    network_data = combine_phaseshift_and_injection(network_data)

    network_data = exclude_bridges_from_outage_masks(network_data)
    network_data = reduce_branch_dimension(network_data)
    network_data = filter_disconnectable_branches_nminus2(network_data)

    network_data = compute_branch_topology_info(network_data)
    # network_data = compute_branch_actions(network_data)

    actions = enumerate_branch_actions(network_data)
    # We know that bus 22 has only stubs and no sensible actions
    # Hence, this relevant sub isn't actually a sensible relevant sub
    assert len(actions[0]) == 1
    has_actions = [len(x) > 1 and np.any(x) for x in actions]
    n_rel_subs = sum(has_actions)

    network_data = replace(network_data, branch_action_set=actions)

    network_data = remove_relevant_subs_without_actions(network_data)
    network_data = convert_multi_outages(network_data)
    network_data = filter_inactive_injections(network_data)
    network_data = compute_injection_topology_info(network_data)
    network_data = process_injection_outages(network_data)
    network_data = add_missing_asset_topo_info(network_data)
    network_data = add_bus_b_columns_to_ptdf(network_data)

    assert np.sum(network_data.relevant_node_mask) == n_rel_subs
    assert len(network_data.branches_at_nodes) == n_rel_subs
    assert len(network_data.branch_direction) == n_rel_subs
    assert len(network_data.num_branches_per_node) == n_rel_subs
    assert len(network_data.injection_idx_at_nodes) == n_rel_subs
    assert len(network_data.num_injections_per_node) == n_rel_subs
    assert len(network_data.active_injections) == n_rel_subs


def test_enumerate_nodal_injections(
    network_data_preprocessed: NetworkData,
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation],
):
    network_data = compute_injection_actions(network_data_preprocessed)
    topo_indices, static_information = jax_inputs_oberrhein
    topo_indices = topo_indices[0]

    di = static_information.dynamic_information
    solver_config = static_information.solver_config
    action_set = di.action_set
    injection_action_all_relevant_subs = action_set[topo_indices.action].inj_actions
    sub_ids = action_set[topo_indices.action].substation_correspondence

    updated_nodal_injections = get_injection_vector(
        injection_assignment=injection_action_all_relevant_subs,
        sub_ids=sub_ids,
        relevant_injections=di.relevant_injections,
        nodal_injections=di.nodal_injections,
        n_stat=jnp.array(solver_config.n_stat),
        rel_stat_map=jnp.array(network_data.relevant_nodes),
    )

    rel_nodes = network_data.relevant_nodes
    rel_nodal_injections = enumerate_spreaded_nodal_injections_for_rel_subs(network_data)

    for action_index, rel_sub_index in zip(topo_indices.action, sub_ids):
        if action_index > len(action_set):
            local_action_index = 0
            # This means that the sub is unsplit -> this correposnds to the action vector of all False
        else:
            local_action_index = np.argmax(
                np.all(
                    network_data.branch_action_set[rel_sub_index] == (action_set.branch_actions[action_index]),
                    axis=1,
                )
            )
        rel_injections = rel_nodal_injections[rel_sub_index][local_action_index]
        assert_bool_vec = updated_nodal_injections[
            :, [rel_nodes[rel_sub_index], -1 * (len(rel_nodes) - rel_sub_index)]
        ] == rel_injections.reshape(di.n_timesteps, 2)
        assert np.all(assert_bool_vec), "Nodal injections mismatch"
