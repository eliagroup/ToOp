import jax.numpy as jnp
from tests.jax.test_busbar_outage import create_dummy_rel_bb_outage_data
from toop_engine_dc_solver.jax.branch_action_set import empty_branch_action_set, merge_branch_action_sets, merge_topologies
from toop_engine_dc_solver.jax.types import ActionSet


def test_merge_branch_action_sets():
    # Create two BranchActionSet instances
    dummy_a_outage_data = create_dummy_rel_bb_outage_data(
        n_br_combis=2, n_max_bb_to_outage_per_sub=3, max_branches_per_sub=4, n_timesteps=1, seed=0
    )
    a = ActionSet(
        branch_actions=jnp.array([[1, 0], [0, 1]]),
        substation_correspondence=jnp.array([0, 1]),
        n_actions_per_sub=jnp.array([1, 1]),
        unsplit_action_mask=jnp.array([False, False]),
        reassignment_distance=jnp.array([3, 4]),
        inj_actions=jnp.array([[1, 0], [0, 1]]),
        rel_bb_outage_data=dummy_a_outage_data,
    )

    dummy_b_outage_data = create_dummy_rel_bb_outage_data(
        n_br_combis=2, n_max_bb_to_outage_per_sub=3, max_branches_per_sub=4, n_timesteps=1, seed=1
    )
    b = ActionSet(
        branch_actions=jnp.array([[1, 1], [0, 0]]),
        substation_correspondence=jnp.array([1, 0]),
        n_actions_per_sub=jnp.array([1, 1]),
        unsplit_action_mask=jnp.array([False, True]),
        reassignment_distance=jnp.array([5, 6]),
        inj_actions=jnp.array([[1, 1], [0, 0]]),
        rel_bb_outage_data=dummy_b_outage_data,
    )

    # Merge the branch action sets
    merged = merge_branch_action_sets(a, b)

    # Expected results
    expected_actions = [[0, 0], [1, 0], [0, 1], [1, 1]]
    expected_substation_correspondence = jnp.array([0, 0, 1, 1])
    expected_n_actions_per_sub = jnp.array([2, 2])
    expected_reassignment_distance = jnp.array([6, 3, 4, 5])
    expected_inj_actions = [[0, 0], [1, 0], [0, 1], [1, 1]]

    # Assertions
    for action in merged.branch_actions:
        assert list(action) in expected_actions
    assert jnp.array_equal(merged.substation_correspondence, expected_substation_correspondence)
    assert jnp.array_equal(merged.n_actions_per_sub, expected_n_actions_per_sub)
    assert jnp.array_equal(merged.reassignment_distance, expected_reassignment_distance)
    assert merged.unsplit_action_mask.sum() == 1
    for action in merged.inj_actions:
        assert list(action) in expected_inj_actions
    assert jnp.array_equal(merged.inj_actions, merged.branch_actions)
    assert merged.rel_bb_outage_data is not None
    assert merged.rel_bb_outage_data.branch_outage_set.shape == (4, 3, 4)
    assert merged.rel_bb_outage_data.deltap_set.shape == (4, 3, 1)
    assert merged.rel_bb_outage_data.nodal_indices.shape == (4, 3)
    assert merged.rel_bb_outage_data.articulation_node_mask.shape == (4, 3)

    merged = merge_branch_action_sets(a, empty_branch_action_set(2, 2, 2))
    assert jnp.array_equal(merged.branch_actions, a.branch_actions)
    assert jnp.array_equal(merged.substation_correspondence, a.substation_correspondence)
    assert jnp.array_equal(merged.n_actions_per_sub, a.n_actions_per_sub)
    assert jnp.array_equal(merged.unsplit_action_mask, a.unsplit_action_mask)
    assert merged.rel_bb_outage_data is None


def test_merge_topologies():
    # Create a BranchActionSet instance
    action_set = ActionSet(
        branch_actions=jnp.array([[1, 0], [0, 1]]),
        substation_correspondence=jnp.array([0, 1]),
        n_actions_per_sub=jnp.array([1, 1]),
        unsplit_action_mask=jnp.array([False, False]),
        reassignment_distance=jnp.array([3, 4]),
        inj_actions=jnp.array([[1, 0], [0, 1]]),
    )

    # Create topologies and substation ids
    topologies = jnp.array([[[1, 1], [0, 0]], [[0, 1], [1, 0]]], dtype=bool)
    sub_ids = jnp.array([[0, 1], [1, 0]], dtype=int)

    # Merge the topologies into the branch action set
    merged = merge_topologies(action_set, topologies, sub_ids)

    assert merged.branch_actions.shape == (4, 2)
    assert jnp.array_equal(merged.n_actions_per_sub, jnp.array([2, 2]))
    assert jnp.array_equal(merged.substation_correspondence, jnp.array([0, 0, 1, 1]))
    assert merged.unsplit_action_mask.sum() == 1
    assert jnp.array_equal(jnp.unique(merged.reassignment_distance), jnp.array([0, 3, 4]))


def test_index_branch_action_set():
    # Create a BranchActionSet instance
    action_set = ActionSet(
        branch_actions=jnp.array([[1, 0], [0, 1], [1, 1]]),
        substation_correspondence=jnp.array([0, 1, 0]),
        n_actions_per_sub=jnp.array([2, 1]),
        unsplit_action_mask=jnp.array([False, False, True]),
        reassignment_distance=jnp.array([3, 4, 5]),
        inj_actions=jnp.array([[1, 0], [0, 1], [1, 1]]),
    )

    # Index with integer array
    index = jnp.array([0, 2])
    indexed = action_set[index]

    assert jnp.array_equal(indexed.branch_actions, jnp.array([[1, 0], [1, 1]]))
    assert jnp.array_equal(indexed.substation_correspondence, jnp.array([0, 0]))
    assert jnp.array_equal(indexed.n_actions_per_sub, jnp.array([2, 0]))
    assert jnp.array_equal(indexed.unsplit_action_mask, jnp.array([False, True]))
    assert jnp.array_equal(indexed.reassignment_distance, jnp.array([3, 5]))
    assert jnp.array_equal(indexed.inj_actions, jnp.array([[1, 0], [1, 1]]))

    # Index with boolean array
    index = jnp.array([True, False, True])
    indexed = action_set[index]

    assert jnp.array_equal(indexed.branch_actions, jnp.array([[1, 0], [1, 1]]))
    assert jnp.array_equal(indexed.substation_correspondence, jnp.array([0, 0]))
    assert jnp.array_equal(indexed.n_actions_per_sub, jnp.array([2, 0]))
    assert jnp.array_equal(indexed.unsplit_action_mask, jnp.array([False, True]))
    assert jnp.array_equal(indexed.reassignment_distance, jnp.array([3, 5]))
    assert jnp.array_equal(indexed.inj_actions, jnp.array([[1, 0], [1, 1]]))

    # Index with single integer
    index = jnp.array([1])
    indexed = action_set[index]

    assert jnp.array_equal(indexed.branch_actions, jnp.array([[0, 1]]))
    assert jnp.array_equal(indexed.substation_correspondence, jnp.array([1]))
    assert jnp.array_equal(indexed.n_actions_per_sub, jnp.array([0, 1]))
    assert jnp.array_equal(indexed.unsplit_action_mask, jnp.array([False]))
    assert jnp.array_equal(indexed.reassignment_distance, jnp.array([4]))
    assert jnp.array_equal(indexed.inj_actions, jnp.array([[0, 1]]))
