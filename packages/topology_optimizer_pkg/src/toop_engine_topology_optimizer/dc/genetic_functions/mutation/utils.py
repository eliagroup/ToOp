# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation utility functions for the genetic algorithm."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray


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
    # Pad the array with a False value at the end, so that we can sample from it even if its empty before
    boolean_array = jnp.pad(boolean_array, (0, 1), constant_values=False)

    logits = jnp.where(boolean_array, 0.0, -jnp.inf)
    any_match = jnp.any(boolean_array)
    idx = jax.random.categorical(random_key, logits=logits)
    return jnp.where(any_match, idx, jnp.array(int_max_value))


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
