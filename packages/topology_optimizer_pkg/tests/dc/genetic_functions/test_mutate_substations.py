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
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import SubstationMutationConfig
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_substations import (
    change_split_substation,
    mutate_sub_splits,
    split_additional_sub,
    unsplit_substation,
)


@pytest.fixture
def random_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(42)


@pytest.fixture
def int_max_value() -> int:
    return 999


@pytest.mark.parametrize(
    "sub_ids,n_subs_rel",
    [
        (jnp.array([1, 2, 999]), 5),  # Some split, some not
        (jnp.array([999, 999, 999]), 5),  # None split
        (jnp.array([0, 1, 2]), 3),  # All split
    ],
)
def test_change_split_substation_shapes(random_key, sub_ids, n_subs_rel, int_max_value):
    split_idx, new_substation_idx = change_split_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert isinstance(split_idx, int) or split_idx.shape == ()
    assert isinstance(new_substation_idx, jnp.ndarray)
    assert new_substation_idx.shape == ()


def test_change_split_substation_resplits(random_key, int_max_value):
    # All substations already split, should allow resplit
    sub_ids = jnp.array([0, 1, 2])
    n_subs_rel = 3
    split_idx, new_substation_idx = change_split_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx >= 0 and split_idx < sub_ids.shape[0]
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


def test_change_split_substation_no_split(random_key, int_max_value):
    # No substations split, should give split_idx int_max_value
    sub_ids = jnp.array([int_max_value, int_max_value, int_max_value])
    n_subs_rel = 5
    split_idx, new_substation_idx = change_split_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx == int_max_value
    assert new_substation_idx < n_subs_rel


def test_change_split_substation_partial_split(random_key, int_max_value):
    # Some substations split, some not
    sub_ids = jnp.array([int_max_value, 2, int_max_value])
    n_subs_rel = 4
    split_idx, new_substation_idx = change_split_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx >= 0 and split_idx < sub_ids.shape[0]
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


@pytest.mark.parametrize(
    "sub_ids,int_max_value",
    [
        (jnp.array([1, 2, 999]), 999),  # Some split, some not
        (jnp.array([999, 999, 999]), 999),  # None split
        (jnp.array([0, 1, 2]), 999),  # All split
        (jnp.array([5, 999, 999]), 999),  # One split, rest not
    ],
)
def test_unsplit_substation_shapes(random_key, sub_ids, int_max_value):
    split_idx, new_sub_id = unsplit_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        int_max_value=int_max_value,
    )
    assert isinstance(split_idx, int) or split_idx.shape == ()
    assert isinstance(new_sub_id, jnp.ndarray)
    assert new_sub_id.shape == ()
    assert new_sub_id == int_max_value


def test_unsplit_substation_split_idx_valid(random_key):
    sub_ids = jnp.array([1, 2, 999])
    int_max_value = 999
    split_idx, new_sub_id = unsplit_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        int_max_value=int_max_value,
    )
    # Only indices 0 and 1 are split, so split_idx should be 0 or 1
    assert split_idx in [0, 1]
    assert new_sub_id == int_max_value


def test_unsplit_substation_no_split(random_key):
    sub_ids = jnp.array([999, 999, 999])
    int_max_value = 999
    split_idx, new_sub_id = unsplit_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        int_max_value=int_max_value,
    )
    # No substations split, so split_idx should be int_max_value
    assert split_idx == int_max_value
    assert new_sub_id == int_max_value


def test_unsplit_substation_all_split(random_key):
    sub_ids = jnp.array([0, 1, 2])
    int_max_value = 999
    split_idx, new_sub_id = unsplit_substation(
        random_key=random_key,
        sub_ids=sub_ids,
        int_max_value=int_max_value,
    )
    # Any index is valid, since all are split
    assert split_idx >= 0 and split_idx < sub_ids.shape[0]
    assert new_sub_id == int_max_value


@pytest.mark.parametrize(
    "sub_ids,n_subs_rel,int_max_value",
    [
        (jnp.array([999, 999, 999]), 5, 999),  # None split
        (jnp.array([1, 2, 999]), 5, 999),  # Some split, some not
        (jnp.array([0, 1, 2]), 3, 999),  # All split
        (jnp.array([5, 999, 999]), 6, 999),  # One split, rest not
    ],
)
def test_split_additional_sub_shapes(random_key, sub_ids, n_subs_rel, int_max_value):
    split_idx, new_substation_idx = split_additional_sub(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert isinstance(split_idx, int) or split_idx.shape == ()
    assert isinstance(new_substation_idx, jnp.ndarray)
    assert new_substation_idx.shape == ()


def test_split_additional_sub_all_split(random_key):
    # All substations already split, should set split_idx to int_max_value
    sub_ids = jnp.array([0, 1, 2])
    n_subs_rel = 3
    int_max_value = 999
    split_idx, new_substation_idx = split_additional_sub(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx == int_max_value
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


def test_split_additional_sub_no_split(random_key):
    # No substations split, should select one to split
    sub_ids = jnp.array([999, 999, 999])
    n_subs_rel = 5
    int_max_value = 999
    split_idx, new_substation_idx = split_additional_sub(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx >= 0 and split_idx < sub_ids.shape[0]
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


def test_split_additional_sub_partial_split(random_key):
    # Some substations split, some not
    int_max_value = 999

    sub_ids = jnp.array([int_max_value, 2, int_max_value])
    n_subs_rel = 4
    split_idx, new_substation_idx = split_additional_sub(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert split_idx >= 0 and split_idx < len(sub_ids)
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


def test_split_additional_sub_one_split(random_key):
    # One split, rest not
    sub_ids = jnp.array([5, 999, 999])
    n_subs_rel = 6
    int_max_value = 999
    split_idx, new_substation_idx = split_additional_sub(
        random_key=random_key,
        sub_ids=sub_ids,
        n_subs_rel=n_subs_rel,
        int_max_value=int_max_value,
    )
    assert (split_idx == int_max_value) or (split_idx >= 0 and split_idx < sub_ids.shape[0])
    assert new_substation_idx >= 0 and new_substation_idx < n_subs_rel


def dummy_sample_action_index_from_branch_actions(rng_key, sub_id, branch_action_set):
    # Always return 1 for simplicity
    return jnp.array(1)


@pytest.fixture
def sub_mutate_config():
    return SubstationMutationConfig(
        add_split_prob=0.3,
        remove_split_prob=0.3,
        change_split_prob=0.3,
        n_subs_mutated_lambda=1.0,
        n_rel_subs=30,
    )


@pytest.mark.parametrize(
    "sub_ids,action",
    [
        (jnp.array([999, 999, 999]), jnp.array([0, 0, 0])),  # None split
        (jnp.array([1, 2, 999]), jnp.array([1, 2, 0])),  # Some split, some not
        (jnp.array([0, 1, 2]), jnp.array([2, 1, 0])),  # All split
    ],
)
def test_mutate_sub_splits_shapes(random_key, sub_mutate_config, synthetic_action_set, sub_ids, action, monkeypatch):
    # Patch sample_action_index_from_branch_actions to deterministic dummy
    monkeypatch.setattr(
        "toop_engine_dc_solver.jax.topology_computations.sample_action_index_from_branch_actions",
        dummy_sample_action_index_from_branch_actions,
    )
    mutated_sub_ids, mutated_action, new_key = mutate_sub_splits(
        sub_ids=sub_ids,
        action=action,
        random_key=random_key,
        sub_mutate_config=sub_mutate_config,
        action_set=synthetic_action_set,
    )
    assert mutated_sub_ids.shape == sub_ids.shape
    assert mutated_action.shape == action.shape
    assert isinstance(new_key, jnp.ndarray)
    assert new_key.shape == (2,)


def test_mutate_sub_splits_probabilities(random_key, sub_mutate_config, synthetic_action_set, int_max_value, monkeypatch):
    monkeypatch.setattr(
        "toop_engine_dc_solver.jax.topology_computations.sample_action_index_from_branch_actions",
        dummy_sample_action_index_from_branch_actions,
    )
    # All split, should only allow remove/change/remain
    sub_ids = jnp.array([0, 1, 2])
    action = jnp.array([0, 1, 2])
    mutated_sub_ids, mutated_action, _ = mutate_sub_splits(
        sub_ids=sub_ids,
        action=action,
        random_key=random_key,
        sub_mutate_config=sub_mutate_config,
        action_set=synthetic_action_set,
    )
    # At least one substation should be mutated or remain unchanged
    assert mutated_sub_ids.shape == sub_ids.shape
    assert mutated_action.shape == action.shape


def test_mutate_sub_splits_add_split(random_key, sub_mutate_config, synthetic_action_set, int_max_value, monkeypatch):
    monkeypatch.setattr(
        "toop_engine_dc_solver.jax.topology_computations.sample_action_index_from_branch_actions",
        dummy_sample_action_index_from_branch_actions,
    )
    # None split, should allow add split
    sub_ids = jnp.array([int_max_value, int_max_value, int_max_value])
    action = jnp.array([0, 0, 0])
    mutated_sub_ids, mutated_action, _ = mutate_sub_splits(
        sub_ids=sub_ids,
        action=action,
        random_key=random_key,
        sub_mutate_config=sub_mutate_config,
        action_set=synthetic_action_set,
    )
    # At least one substation should now be split (not int_max_value)
    assert jnp.any(mutated_sub_ids != int_max_value)


def test_mutate_sub_splits_remove_split(random_key, sub_mutate_config, synthetic_action_set, int_max_value, monkeypatch):
    monkeypatch.setattr(
        "toop_engine_dc_solver.jax.topology_computations.sample_action_index_from_branch_actions",
        dummy_sample_action_index_from_branch_actions,
    )
    # One split, rest not
    sub_ids = jnp.array([5, int_max_value, int_max_value])
    action = jnp.array([1, 0, 0])
    mutated_sub_ids, mutated_action, _ = mutate_sub_splits(
        sub_ids=sub_ids,
        action=action,
        random_key=random_key,
        sub_mutate_config=sub_mutate_config,
        action_set=synthetic_action_set,
    )
    # Should allow removal of split
    assert mutated_sub_ids.shape == sub_ids.shape
    assert mutated_action.shape == action.shape


def test_mutate_sub_splits_remain(random_key, sub_mutate_config, synthetic_action_set, int_max_value, monkeypatch):
    monkeypatch.setattr(
        "toop_engine_dc_solver.jax.topology_computations.sample_action_index_from_branch_actions",
        dummy_sample_action_index_from_branch_actions,
    )
    # Probabilities sum to 1, remain is possible
    sub_ids = jnp.array([int_max_value, int_max_value, int_max_value])
    action = jnp.array([0, 0, 0])
    mutated_sub_ids, mutated_action, _ = mutate_sub_splits(
        sub_ids=sub_ids,
        action=action,
        random_key=random_key,
        sub_mutate_config=sub_mutate_config,
        action_set=synthetic_action_set,
    )
    # Should be valid, possibly unchanged
    assert mutated_sub_ids.shape == sub_ids.shape
    assert mutated_action.shape == action.shape
