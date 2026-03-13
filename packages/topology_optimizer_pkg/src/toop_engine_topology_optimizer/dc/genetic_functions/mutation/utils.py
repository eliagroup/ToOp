# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation utility functions for the genetic algorithm."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.types import int_max


def sample_new_id(
    random_key: PRNGKeyArray,
    already_used_ids: Int[Array, " n_splits_or_disconnections"],
    n_available_ids: int,
    ignored_idx: Int[ArrayLike, " "],
) -> Int[Array, " "]:
    """Sample a relevant id that is not already used in the active split slots.

    Parameters
    ----------
    random_key : PRNGKeyArray
        Random key used for sampling.
    already_used_ids : Int[Array, " max_num_splits"]
        Current ids that should not be sampled again.
    n_available_ids : int
        Number of available ids that may be selected.
    ignored_idx : Int[Array, " "]
        Index in ``already_used_ids`` to ignore while checking duplicates. This is used for branch replacement,
        where the currently replaced split must not block its own id.

    Returns
    -------
    Int[Array, " "]
        A valid new substation id, or ``int_max()`` if none are available.
    """
    int_max_value = int_max()
    available_mask = jnp.ones((n_available_ids,), dtype=bool).at[already_used_ids].set(False, mode="drop")
    available_mask = available_mask.at[already_used_ids.at[ignored_idx].get(mode="fill", fill_value=int_max_value)].set(
        True, mode="drop"
    )
    return get_random_true_idx(random_key, available_mask, int_max_value)


def sample_unique_indices_small_k(
    random_key: PRNGKeyArray,
    n_choices: int,
    n_samples: int,
) -> Int[Array, " n_samples"]:
    """Sample unique indices from ``range(n_choices)`` efficiently for small ``n_samples``.

    This avoids full permutations when only a handful of unique indices are needed.

    Parameters
    ----------
    random_key : PRNGKeyArray
        Random key used for sampling.
    n_choices : int
        Number of available choices in the range ``[0, n_choices)``.
    n_samples : int
        Number of unique indices to sample. Must be less than or equal to ``n_choices``.

    Returns
    -------
    Int[Array, " n_samples"]
        Unique sampled indices.
    """
    sampled = jnp.full((n_samples,), -1, dtype=int)
    sampled_positions = jnp.arange(n_samples)

    def _sample_one(
        sample_idx: int,
        state: tuple[Int[Array, " n_samples"], PRNGKeyArray],
    ) -> tuple[Int[Array, " n_samples"], PRNGKeyArray]:
        sampled_indices, key = state
        draw_key, key = jax.random.split(key)
        candidate = jax.random.randint(draw_key, shape=(), minval=0, maxval=n_choices)

        def _needs_resample(loop_state: tuple[Int[Array, " "], PRNGKeyArray]) -> Bool[Array, ""]:
            candidate_idx, _loop_key = loop_state
            duplicate = (sampled_indices == candidate_idx) & (sampled_positions < sample_idx)
            return jnp.any(duplicate)

        def _resample(loop_state: tuple[Int[Array, " "], PRNGKeyArray]) -> tuple[Int[Array, " "], PRNGKeyArray]:
            _candidate_idx, loop_key = loop_state
            draw_key_inner, loop_key = jax.random.split(loop_key)
            candidate_idx = jax.random.randint(draw_key_inner, shape=(), minval=0, maxval=n_choices)
            return candidate_idx, loop_key

        candidate, key = jax.lax.while_loop(_needs_resample, _resample, (candidate, key))
        sampled_indices = sampled_indices.at[sample_idx].set(candidate)
        return sampled_indices, key

    sampled, _ = jax.lax.fori_loop(0, n_samples, _sample_one, (sampled, random_key))
    return sampled


def get_random_true_idx(
    random_key: PRNGKeyArray, boolean_array: Bool[Array, " n_possibilities"], int_max_value: int
) -> Int[Array, " "]:
    """Return random index of a True entry, or int_max_value if none are True.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the sampling
    boolean_array : Bool[Array, " n_possibilities"]
        The boolean array to sample from. Should have shape (n_possibilities,).
        Is true where the index is a valid choice to sample, and false where it is not.
        If all entries are false, int_max_value is returned.
    int_max_value : int
        The value to return if all entries in boolean_array are False. This should be the maximum integer value used
        to indicate an empty slot in the genotype.

    Returns
    -------
    Int[Array, " "]
        A random index of a True entry in boolean_array, or int_max_value if all entries
        are False.
    """
    n_possibilities = boolean_array.shape[0]
    candidate_indices = jnp.nonzero(boolean_array, size=n_possibilities, fill_value=int_max_value)[0]
    n_candidates = jnp.sum(boolean_array)
    safe_n_candidates = jnp.maximum(n_candidates, 1)
    sampled_position = jax.random.randint(random_key, shape=(), minval=0, maxval=safe_n_candidates)
    sampled_index = candidate_indices[sampled_position]
    return jnp.where(n_candidates > 0, sampled_index, jnp.array(int_max_value))


def do_nothing(int_max_value: int) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Return a no-op mutation index and value.

    Parameters
    ----------
    int_max_value : int
        The value to use for the no-op mutation, which should be the maximum integer value used
        to indicate an empty slot in the genotype.

    Returns
    -------
    Int[Array, " "]
        The index that is changed. This is set to int_max_value to indicate that no index is changed.
    Int[Array, " "]
        The new value that is added. This is set to int_max_value to indicate that no value is added.
    """
    return jnp.array(int_max_value), jnp.array(int_max_value)
