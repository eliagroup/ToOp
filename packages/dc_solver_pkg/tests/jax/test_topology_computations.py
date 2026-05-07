# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import jax
import jax.numpy as jnp
import numpy as np
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    convert_branch_topo_vect,
    convert_topo_sel_sorted,
    convert_topo_to_action_set_index,
    convert_topo_to_action_set_index_jittable,
    default_topology,
    extract_sub_ids,
    is_in_action_set,
    pad_action_with_unsplit_action_indices,
    product_action_set,
    random_topology,
    sample_from_branch_actions,
)
from toop_engine_dc_solver.jax.types import (
    ActionSet,
    HashableArrayWrapper,
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
)


def test_convert_topo_to_action_set_index() -> None:
    action_set = ActionSet(
        branch_actions=jnp.array([[1, 0, 0], [0, 1, 1]], dtype=bool),
        substation_correspondence=jnp.array([0, 1]),
        n_actions_per_sub=jnp.array([1, 1]),
        action_start_indices=jnp.array([0, 1]),
        unsplit_action_mask=jnp.array([False, False]),
        reassignment_distance=jnp.array([3, 4]),
        inj_actions=jnp.zeros((2, 5), dtype=bool),
    )

    topo = TopoVectBranchComputations(
        topologies=jnp.array([[[1, 0, 0], [0, 1, 1]], [[0, 1, 1], [1, 0, 0]], [[0, 0, 0], [1, 1, 1]]], dtype=bool),
        sub_ids=jnp.array([[0, 1], [0, 1], [1, 0]]),
        pad_mask=jnp.array([True, True, True]),
    )

    topo_new, action_set_new = convert_topo_to_action_set_index(
        topologies=topo, branch_actions=action_set, extend_action_set=False
    )

    assert jnp.array_equal(topo_new.action, jnp.array([[0, 1], [int_max(), int_max()], [int_max(), int_max()]]))
    assert action_set_new == action_set

    topo_new_2 = convert_topo_to_action_set_index_jittable(
        topologies=topo,
        branch_actions=action_set,
    )
    assert jnp.array_equal(topo_new.action, topo_new_2.action)

    # We can only reverse the first topology, as the other ones are not in the action set
    reversed = convert_action_set_index_to_topo(topo_new, action_set_new)
    assert jnp.array_equal(reversed.topologies[0], topo.topologies[0])
    assert jnp.array_equal(reversed.topologies[1:], jnp.zeros_like(reversed.topologies[1:]))
    assert jnp.array_equal(reversed.sub_ids[0], topo.sub_ids[0])
    assert jnp.array_equal(reversed.sub_ids[1:], jnp.full_like(reversed.sub_ids[1:], int_max()))
    assert jnp.array_equal(reversed.pad_mask, topo.pad_mask)
    sub_ids = extract_sub_ids(topo_new.action, action_set_new)
    assert jnp.array_equal(reversed.sub_ids, sub_ids)

    topo_new, action_set_new = convert_topo_to_action_set_index(
        topologies=topo, branch_actions=action_set, extend_action_set=True
    )
    has_splits = jnp.any(topo.topologies, axis=-1)
    assert jnp.array_equal(topo_new.action != int_max(), has_splits)
    assert action_set_new != action_set
    assert jnp.array_equal(action_set_new.inj_actions, jnp.zeros((len(action_set_new), 5), dtype=bool))

    # Re-converting yields the same indices
    topo_new_2, _ = convert_topo_to_action_set_index(topologies=topo, branch_actions=action_set_new, extend_action_set=False)
    assert jnp.array_equal(topo_new.action, topo_new_2.action)

    topo_new_3 = convert_topo_to_action_set_index_jittable(
        topologies=topo,
        branch_actions=action_set_new,
    )
    assert jnp.array_equal(topo_new.action, topo_new_3.action)

    # We can reverse the topology for all topologies
    reversed = convert_action_set_index_to_topo(topo_new, action_set_new)
    assert jnp.array_equal(reversed.topologies, topo.topologies)
    assert jnp.array_equal(reversed.sub_ids[has_splits], topo.sub_ids[has_splits])
    assert jnp.array_equal(reversed.pad_mask, topo.pad_mask)
    sub_ids = extract_sub_ids(topo_new.action, action_set_new)
    assert jnp.array_equal(reversed.sub_ids, sub_ids)


def test_convert_topo_sel_sorted() -> None:
    branches_per_sub = HashableArrayWrapper(np.array([4, 5, 4], dtype=int))

    topo_sel_sorted = np.random.randn(30, 4 + 5 + 4) > 0

    topologies = convert_topo_sel_sorted(topo_sel_sorted, branches_per_sub)

    assert topologies.topologies.shape == (30, 3, 5)

    topo_sel_reconstructed = convert_branch_topo_vect(
        topologies=topologies.topologies,
        sub_ids=topologies.sub_ids,
        branches_per_sub=branches_per_sub,
    )

    assert jnp.array_equal(topo_sel_reconstructed, topo_sel_sorted)


def test_default_topology(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=16),
    )

    topo = default_topology(static_information.solver_config, topo_vect_format=True)
    assert jnp.array_equal(topo.pad_mask, jnp.ones(16, dtype=bool))
    assert jnp.array_equal(topo.sub_ids, jnp.repeat(jnp.arange(5, dtype=int)[None, :], 16, axis=0))
    assert jnp.array_equal(topo.topologies, jnp.zeros((16, 5, 5), dtype=bool))

    topo = default_topology(static_information.solver_config, topo_vect_format=False)
    assert jnp.array_equal(topo.action, jnp.full((16, 1), int_max(), dtype=int))
    assert jnp.array_equal(topo.pad_mask, jnp.ones(16, dtype=bool))


def test_random_topology(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=16),
    )

    topo = random_topology(
        rng_key=jax.random.PRNGKey(0),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=None,
        unsplit_prob=0.0,
        batch_size=16,
        topo_vect_format=True,
    )
    default_topo = default_topology(static_information.solver_config, batch_size=16, topo_vect_format=True)
    assert topo.topologies.shape == default_topo.topologies.shape
    assert topo.sub_ids.shape == default_topo.sub_ids.shape
    assert topo.pad_mask.shape == default_topo.pad_mask.shape

    assert jnp.array_equal(
        topo.sub_ids[0],
        jnp.arange(static_information.solver_config.rel_stat_map.shape[0]),
    )
    assert jnp.any(topo.topologies)
    assert jnp.all(jnp.any(topo.topologies, axis=2))

    topo = random_topology(
        rng_key=jax.random.PRNGKey(0),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=16,
        unsplit_prob=0.0,
        topo_vect_format=True,
    )
    # The sub ids should not be equal
    unique_sub_ids = jnp.unique(topo.sub_ids, axis=0)
    assert len(unique_sub_ids) > 1

    has_splits = jnp.any(topo.topologies, axis=-1)
    assert jnp.all(has_splits)

    assert topo.topologies.shape[1] == 2
    assert topo.topologies.shape[0] == static_information.solver_config.batch_size_bsdf

    # Also works for the topo_vect_format=False
    topo = random_topology(
        rng_key=jax.random.PRNGKey(0),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=128,
        topo_vect_format=False,
    )
    assert topo.action.shape == (128, 2)


def test_sample_from_branch_actions(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, _, static_information = jax_inputs
    branches_per_sub = static_information.solver_config.branches_per_sub
    for i in range(50):
        rng_key = jax.random.PRNGKey(i)
        sub_id = np.random.randint(low=0, high=branches_per_sub.shape[0])
        topo = sample_from_branch_actions(
            rng_key,
            jnp.array(sub_id),
            static_information.dynamic_information.action_set,
        )

        assert topo.shape == (np.max(branches_per_sub.val),)
        assert topo.dtype == bool
        assert not jnp.any(topo[branches_per_sub.val[sub_id] :])

        if branches_per_sub.val[sub_id] < 4:
            assert not jnp.any(topo)
        else:
            assert jnp.sum(topo) >= 2

    # Invalid sub-id returns all zeros
    topo = sample_from_branch_actions(
        rng_key,
        jnp.array(len(branches_per_sub) + 5),
        static_information.dynamic_information.action_set,
    )
    assert not jnp.any(topo)


def test_product_action_set() -> None:
    action_set = ActionSet(
        branch_actions=jnp.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        n_actions_per_sub=jnp.array([3, 4, 3], dtype=int),
        action_start_indices=jnp.array([0, 3, 7], dtype=int),
        substation_correspondence=jnp.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=int),
        unsplit_action_mask=jnp.array([True, False, False, True, False, False, False, True, False, False], dtype=bool),
        reassignment_distance=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
        inj_actions=jnp.zeros((10, 5), dtype=bool),
    )

    combinations = product_action_set([0, 1, 2], action_set)

    assert len(combinations.topologies) == 3 * 4 * 3
    assert len(set(tuple(t.flatten().tolist()) for t in combinations.topologies)) == 3 * 4 * 3
    assert np.all(combinations.sub_ids == np.array([0, 1, 2]))
    assert is_in_action_set(combinations, action_set).all()

    combinations_limited = product_action_set([0, 1, 2], action_set, limit_n_subs=2)

    for topo in combinations_limited.topologies:
        assert np.sum(np.any(topo, axis=-1)) <= 2
    assert is_in_action_set(combinations_limited, action_set).all()


def test_is_in_action_set() -> None:
    branch_actions = ActionSet(
        branch_actions=jnp.array([[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=bool),
        substation_correspondence=jnp.array([0, 0, 1, 1]),
        n_actions_per_sub=jnp.array([2, 2]),
        action_start_indices=jnp.array([0, 2]),
        unsplit_action_mask=jnp.array([False, False, True, False]),
        reassignment_distance=jnp.array([0, 1, 2, 3]),
        inj_actions=jnp.zeros((4, 5), dtype=bool),
    )

    topologies = TopoVectBranchComputations(
        topologies=jnp.array([[[1, 0, 0], [0, 1, 1]], [[0, 1, 1], [1, 0, 0]], [[0, 0, 0], [1, 1, 0]]], dtype=bool),
        sub_ids=jnp.array([[0, 1], [0, 1], [1, 0]]),
        pad_mask=jnp.array([True, True, True]),
    )

    result = is_in_action_set(topologies, branch_actions)
    expected = jnp.array([[True, False], [True, False], [True, False]], dtype=bool)
    assert jnp.array_equal(result, expected)

    # Test with unsplit topologies
    topologies_unsplit = TopoVectBranchComputations(
        topologies=jnp.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=bool),
        sub_ids=jnp.array([[0, 1], [1, 0]]),
        pad_mask=jnp.array([True, True]),
    )

    result_unsplit = is_in_action_set(topologies_unsplit, branch_actions)
    expected_unsplit = jnp.array([[True, True], [True, True]], dtype=bool)
    assert jnp.array_equal(result_unsplit, expected_unsplit)

    # Test with invalid substation ids
    topologies_invalid = TopoVectBranchComputations(
        topologies=jnp.array([[[1, 0, 0], [0, 1, 1]], [[0, 1, 1], [1, 0, 0]]], dtype=bool),
        sub_ids=jnp.array([[0, 2], [3, -1]]),
        pad_mask=jnp.array([True, True]),
    )

    result_invalid = is_in_action_set(topologies_invalid, branch_actions)
    expected_invalid = jnp.array([[True, True], [True, True]], dtype=bool)
    assert jnp.array_equal(result_invalid, expected_invalid)


def test_extract_sub_ids() -> None:
    branch_actions = ActionSet(
        branch_actions=jnp.array([[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=bool),
        substation_correspondence=jnp.array([0, 0, 1, 1]),
        n_actions_per_sub=jnp.array([2, 2]),
        action_start_indices=jnp.array([0, 2]),
        unsplit_action_mask=jnp.array([False, False, True, False]),
        reassignment_distance=jnp.array([0, 1, 2, 3]),
        inj_actions=jnp.zeros((4, 5), dtype=bool),
    )

    # Test with batched actions
    actions_batched = jnp.array([[0, 2], [1, 3]], dtype=int)
    sub_ids_batched = extract_sub_ids(actions_batched, branch_actions)
    expected_sub_ids_batched = jnp.array([[0, 1], [0, 1]], dtype=int)
    assert jnp.array_equal(sub_ids_batched, expected_sub_ids_batched)

    # Test with unbatched actions
    actions_unbatched = jnp.array([0, 2], dtype=int)
    sub_ids_unbatched = extract_sub_ids(actions_unbatched, branch_actions)
    expected_sub_ids_unbatched = jnp.array([0, 1], dtype=int)
    assert jnp.array_equal(sub_ids_unbatched, expected_sub_ids_unbatched)

    # Test with unsplit actions
    actions_unsplit = jnp.array([[2, 2], [2, 2]], dtype=int)
    sub_ids_unsplit = extract_sub_ids(actions_unsplit, branch_actions)
    expected_sub_ids_unsplit = jnp.array([[1, 1], [1, 1]], dtype=int)
    assert jnp.array_equal(sub_ids_unsplit, expected_sub_ids_unsplit)

    # Test with invalid actions
    actions_invalid = jnp.array([[12345, 3], [12345, 2]], dtype=int)
    sub_ids_invalid = extract_sub_ids(actions_invalid, branch_actions)
    expected_sub_ids_invalid = jnp.array([[int_max(), 1], [int_max(), 1]], dtype=int)
    assert jnp.array_equal(sub_ids_invalid, expected_sub_ids_invalid)


def test_pad_action_with_unsplit_action_indices() -> None:
    action_set = ActionSet(
        branch_actions=jnp.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        n_actions_per_sub=jnp.array([3, 4, 3], dtype=int),
        action_start_indices=jnp.array([0, 3, 7], dtype=int),
        substation_correspondence=jnp.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=int),
        unsplit_action_mask=jnp.array([True, False, False, True, False, False, False, True, False, False], dtype=bool),
        reassignment_distance=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
        inj_actions=jnp.zeros((10, 5), dtype=bool),
        rel_bb_outage_data=None,
    )

    # Test case 1: Valid action indices with unsplit action
    action_indices = jnp.array([5, int_max()])
    result = pad_action_with_unsplit_action_indices(action_set, action_indices)
    expected = jnp.array([0, 5, 7])
    assert jnp.array_equal(result, expected)

    # Test case 2: All action indices are unsplit actions
    action_indices = jnp.array([int_max(), int_max()])
    result = pad_action_with_unsplit_action_indices(action_set, action_indices)
    expected = jnp.array([0, 3, 7])  # Corresponding unsplit action indices
    assert jnp.array_equal(result, expected)

    # Test case 3: No unsplit actions, all valid indices
    action_indices = jnp.array([1, 4])
    result = pad_action_with_unsplit_action_indices(action_set, action_indices)
    expected = jnp.array([1, 4, 7])
    assert jnp.array_equal(result, expected)

    # Test case 4: Invalid action indices
    action_indices = jnp.array([12345, 4])
    result = pad_action_with_unsplit_action_indices(action_set, action_indices)
    expected = jnp.array([0, 4, 7])  # Invalid indices default to unsplit actions
    assert jnp.array_equal(result, expected)
