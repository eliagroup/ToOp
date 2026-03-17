# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import DisconnectionMutationConfig
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_disconnections import (
    change_disconnected_branch,
    disconnect_additional_branch,
    mutate_disconnections,
    reconnect_disconnected_branch,
)


@pytest.fixture
def random_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(42)


@pytest.mark.parametrize(
    "disconnections,n_disconnectable_branches",
    [
        (jnp.array([1, 2, int_max(), int_max()], dtype=int), 5),
        (jnp.array([int_max(), int_max(), int_max(), int_max()], dtype=int), 4),
        (jnp.array([0, int_max(), int_max(), int_max()], dtype=int), 3),
        (jnp.array([2, 3, 4, int_max()], dtype=int), 6),
    ],
)
def test_change_disconnected_branch_valid(random_key, disconnections, n_disconnectable_branches):
    disc_idx, new_disc_id = change_disconnected_branch(random_key, disconnections, n_disconnectable_branches)
    int_max_value = int_max()
    # disc_idx should be a valid index or int_max_value if nothing to change
    assert isinstance(disc_idx, jnp.ndarray)
    assert disc_idx.shape == ()
    # new_disc_id should be int_max_value if no change possible, else a valid branch index
    assert isinstance(new_disc_id, jnp.ndarray)
    assert new_disc_id.shape == ()
    if jnp.all(disconnections == int_max_value):
        assert disc_idx == int_max_value
    # If there are disconnected branches, new_disc_id should be in [0, n_disconnectable_branches)
    elif new_disc_id != int_max_value:
        assert 0 <= int(new_disc_id) < n_disconnectable_branches


def test_change_disconnected_branch_no_disconnected(random_key):
    disconnections = jnp.array([int_max(), int_max(), int_max()], dtype=int)
    n_disconnectable_branches = 3
    disc_idx, new_disc_id = change_disconnected_branch(random_key, disconnections, n_disconnectable_branches)
    assert disc_idx == int_max()


def test_change_disconnected_branch_all_branches_disconnected(random_key):
    disconnections = jnp.array([0, 1, 2], dtype=int)
    n_disconnectable_branches = 3
    disc_idx, new_disc_id = change_disconnected_branch(random_key, disconnections, n_disconnectable_branches)
    # Since only 3 branches are possible and all are disconnected, new_disc_id  should be
    # the same as before
    assert new_disc_id == disconnections[disc_idx]


def test_change_disconnected_branch_repeatability():
    random_key = jax.random.PRNGKey(123)
    disconnections = jnp.array([1, int_max(), int_max()], dtype=int)
    n_disconnectable_branches = 4
    disc_idx1, new_disc_id1 = change_disconnected_branch(random_key, disconnections, n_disconnectable_branches)
    disc_idx2, new_disc_id2 = change_disconnected_branch(random_key, disconnections, n_disconnectable_branches)
    # Should be deterministic for same key and input
    assert disc_idx1 == disc_idx2
    assert new_disc_id1 == new_disc_id2


@pytest.mark.parametrize(
    "disconnections",
    [
        jnp.array([1, 2, int_max(), int_max()], dtype=int),
        jnp.array([int_max(), int_max(), int_max(), int_max()], dtype=int),
        jnp.array([0, int_max(), int_max(), int_max()], dtype=int),
        jnp.array([2, 3, 4, int_max()], dtype=int),
    ],
)
def test_reconnect_disconnected_branch_valid(random_key, disconnections):
    disc_idx, new_disc_id = reconnect_disconnected_branch(random_key, disconnections)
    int_max_value = int_max()
    assert isinstance(disc_idx, jnp.ndarray)
    assert disc_idx.shape == ()
    assert isinstance(new_disc_id, jnp.ndarray)
    assert new_disc_id.shape == ()
    assert new_disc_id == int_max_value


def test_reconnect_disconnected_branch_no_disconnected(random_key):
    disconnections = jnp.array([int_max(), int_max(), int_max()], dtype=int)
    disc_idx, new_disc_id = reconnect_disconnected_branch(random_key, disconnections)
    assert new_disc_id == int_max()


def test_reconnect_disconnected_branch_repeatability():
    random_key = jax.random.PRNGKey(123)
    disconnections = jnp.array([1, int_max(), int_max()], dtype=int)
    disc_idx1, new_disc_id1 = reconnect_disconnected_branch(random_key, disconnections)
    disc_idx2, new_disc_id2 = reconnect_disconnected_branch(random_key, disconnections)
    assert disc_idx1 == disc_idx2
    assert new_disc_id1 == new_disc_id2


@pytest.mark.parametrize(
    "disconnections,n_disconnectable_branches",
    [
        (jnp.array([int_max(), int_max(), int_max()], dtype=int), 4),
        (jnp.array([1, int_max(), int_max()], dtype=int), 3),
        (jnp.array([0, 1, int_max()], dtype=int), 5),
        (jnp.array([2, 3, 4, int_max()], dtype=int), 6),
    ],
)
def test_disconnect_additional_branch_valid(random_key, disconnections, n_disconnectable_branches):
    disc_idx, new_disc_id = disconnect_additional_branch(random_key, disconnections, n_disconnectable_branches)
    int_max_value = int_max()
    assert isinstance(disc_idx, jnp.ndarray)
    assert disc_idx.shape == ()
    assert isinstance(new_disc_id, jnp.ndarray)
    assert new_disc_id.shape == ()
    # If all branches are already disconnected, new_disc_id should be int_max
    already_disconnected = jnp.zeros(n_disconnectable_branches, dtype=bool).at[disconnections].set(True, mode="drop")
    if jnp.all(already_disconnected):
        assert new_disc_id == int_max_value
    elif new_disc_id != int_max_value:
        assert 0 <= int(new_disc_id) < n_disconnectable_branches
        # Should not select already disconnected branches
        assert not already_disconnected[int(new_disc_id)]


def test_disconnect_additional_branch_no_available(random_key):
    disconnections = jnp.array([0, 1, 2], dtype=int)
    n_disconnectable_branches = 3
    disc_idx, new_disc_id = disconnect_additional_branch(random_key, disconnections, n_disconnectable_branches)
    assert new_disc_id == int_max()


def test_disconnect_additional_branch_repeatability():
    random_key = jax.random.PRNGKey(123)
    disconnections = jnp.array([int_max(), int_max(), int_max()], dtype=int)
    n_disconnectable_branches = 4
    disc_idx1, new_disc_id1 = disconnect_additional_branch(random_key, disconnections, n_disconnectable_branches)
    disc_idx2, new_disc_id2 = disconnect_additional_branch(random_key, disconnections, n_disconnectable_branches)
    assert disc_idx1 == disc_idx2
    assert new_disc_id1 == new_disc_id2


@pytest.mark.parametrize(
    "sub_ids,disconnections,config_kwargs",
    [
        # Only add allowed, no splits
        (
            jnp.array([int_max(), int_max()], dtype=int),
            jnp.array([int_max(), int_max()], dtype=int),
            dict(
                add_disconnection_prob=1.0,
                change_disconnection_prob=0.0,
                remove_disconnection_prob=0.0,
                n_disconnectable_branches=3,
            ),
        ),
        # Only change allowed
        (
            jnp.array([0, int_max()], dtype=int),
            jnp.array([1, int_max()], dtype=int),
            dict(
                add_disconnection_prob=0.0,
                change_disconnection_prob=1.0,
                remove_disconnection_prob=0.0,
                n_disconnectable_branches=3,
            ),
        ),
        # Only remove allowed
        (
            jnp.array([0, int_max()], dtype=int),
            jnp.array([1, 2], dtype=int),
            dict(
                add_disconnection_prob=0.0,
                change_disconnection_prob=0.0,
                remove_disconnection_prob=1.0,
                n_disconnectable_branches=3,
            ),
        ),
        # Only remain allowed
        (
            jnp.array([int_max(), int_max()], dtype=int),
            jnp.array([int_max(), int_max()], dtype=int),
            dict(
                add_disconnection_prob=0.0,
                change_disconnection_prob=0.0,
                remove_disconnection_prob=0.0,
                n_disconnectable_branches=2,
            ),
        ),
        # All allowed, probabilities sum < 1
        (
            jnp.array([0, int_max()], dtype=int),
            jnp.array([1, int_max()], dtype=int),
            dict(
                add_disconnection_prob=0.3,
                change_disconnection_prob=0.3,
                remove_disconnection_prob=0.3,
                n_disconnectable_branches=4,
            ),
        ),
    ],
)
def test_mutate_disconnections_valid(random_key, sub_ids, disconnections, config_kwargs):
    config = DisconnectionMutationConfig(**config_kwargs)
    mutated = mutate_disconnections(random_key, sub_ids, disconnections, config)
    assert isinstance(mutated, jnp.ndarray)
    assert mutated.shape == disconnections.shape
    # All mutated values should be int or in [0, n_disconnectable_branches)
    for val in mutated:
        assert (val == int_max()) or (0 <= int(val) < config.n_disconnectable_branches)


def test_mutate_disconnections_repeatability():
    random_key = jax.random.PRNGKey(42)
    sub_ids = jnp.array([0, int_max()], dtype=int)
    disconnections = jnp.array([1, int_max()], dtype=int)
    config = DisconnectionMutationConfig(
        add_disconnection_prob=0.3,
        change_disconnection_prob=0.3,
        remove_disconnection_prob=0.3,
        n_disconnectable_branches=4,
    )
    mutated1 = mutate_disconnections(random_key, sub_ids, disconnections, config)
    mutated2 = mutate_disconnections(random_key, sub_ids, disconnections, config)
    assert jnp.all(mutated1 == mutated2)


@pytest.mark.parametrize(
    "sub_ids,disconnections,config_kwargs",
    [
        # No splits, only add allowed
        (
            jnp.array([int_max(), int_max()], dtype=int),
            jnp.array([int_max(), int_max()], dtype=int),
            dict(
                add_disconnection_prob=1.0,
                change_disconnection_prob=0.0,
                remove_disconnection_prob=0.0,
                n_disconnectable_branches=2,
            ),
        ),
        # Splits present, only remove allowed
        (
            jnp.array([0, int_max()], dtype=int),
            jnp.array([1, 2], dtype=int),
            dict(
                add_disconnection_prob=0.0,
                change_disconnection_prob=0.0,
                remove_disconnection_prob=1.0,
                n_disconnectable_branches=3,
            ),
        ),
    ],
)
def test_mutate_disconnections_operation_selection(random_key, sub_ids, disconnections, config_kwargs):
    config = DisconnectionMutationConfig(**config_kwargs)
    mutated = mutate_disconnections(random_key, sub_ids, disconnections, config)
    # For add, at least one slot should be filled with a valid branch id
    if config_kwargs["add_disconnection_prob"] > 0:
        assert jnp.any(mutated != int_max())
    # For remove, at least one should be reconnected
    if config_kwargs["remove_disconnection_prob"] > 0:
        assert jnp.any(mutated == int_max())


def test_mutate_disconnections_remain(random_key):
    sub_ids = jnp.array([int_max(), int_max()], dtype=int)
    disconnections = jnp.array([int_max(), int_max()], dtype=int)
    config = DisconnectionMutationConfig(
        add_disconnection_prob=0.0,
        change_disconnection_prob=0.0,
        remove_disconnection_prob=0.0,
        n_disconnectable_branches=2,
    )
    mutated = mutate_disconnections(random_key, sub_ids, disconnections, config)
    assert jnp.all(mutated == disconnections)
