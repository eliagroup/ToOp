# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation functions for the disconnections in the genetic algorithm."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import DisconnectionMutationConfig
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.utils import (
    do_nothing,
    get_random_true_idx,
    sample_new_id,
)


def change_disconnected_branch(
    random_key: PRNGKeyArray, disconnections: Int[Array, " max_num_disconnections"], n_disconnectable_branches: int
) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Change a disconnected branch in the topology to a different one.

    This assumes, that one branch is already disconnected, otherwise there would be nothing to change.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before mutation of one individual
    n_disconnectable_branches: int
        The number of disconnectable branches in the action set. It is not necessary to know the
        contents of the action set here because we only sample an index into the action set.

    Returns
    -------
    Int[Array, " "]
        The index that is changed
    Int[Array, " "]
        The new disconnection that is added. If no disconnection was changed, this is set to int_max
    """
    int_max_value = int_max()
    random_index_key, random_disc_key = jax.random.split(random_key)
    is_disconnected = disconnections != int_max_value
    disc_idx = get_random_true_idx(random_index_key, is_disconnected, int_max_value)

    new_disc_id = sample_new_id(random_disc_key, disconnections, n_disconnectable_branches, disc_idx)
    return disc_idx, new_disc_id


def reconnect_disconnected_branch(
    random_key: PRNGKeyArray, disconnections: Int[Array, " max_num_disconnections"]
) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Reconnect a disconnected branch in the topology.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before mutation of one individual

    Returns
    -------
    Int[Array, " "]
        The index that is changed
    Int[Array, " "]
        The new disconnection that is added. This is always int_max to indicate that the slot is now empty
    """
    int_max_value = int_max()
    random_index_key = random_key
    is_disconnected = disconnections != int_max_value
    disc_idx = get_random_true_idx(random_index_key, is_disconnected, int_max_value)
    return disc_idx, jnp.array(int_max_value, dtype=int)


def disconnect_additional_branch(
    random_key: PRNGKeyArray, disconnections: Int[Array, " max_num_disconnections"], n_disconnectable_branches: int
) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Add a new disconnection to the topology.

    This assumes, that there is still room to add a disconnection, otherwise there would be no valid branch to add.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before mutation of one individual
    n_disconnectable_branches: int
        The number of disconnectable branches in the action set. It is not necessary to know the
        contents of the action set here because we only sample an index into the action set.

    Returns
    -------
    Int[Array, " "]
        The index that is changed
    Int[Array, " "]
        The new disconnection that is added. If no disconnection was added, this is set to int_max
    """
    int_max_value = int_max()
    random_index_key, random_disc_key = jax.random.split(random_key)
    # List available disconnections
    is_disconnectable = disconnections == int_max_value
    disc_idx = get_random_true_idx(random_index_key, is_disconnectable, int_max_value)

    new_disc_id = sample_new_id(random_disc_key, disconnections, n_disconnectable_branches, int_max_value)
    return disc_idx, new_disc_id


def mutate_disconnections(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    disconnections: Int[Array, " max_num_disconnections"],
    disconnection_mutation_config: DisconnectionMutationConfig,
) -> Int[Array, " max_num_disconnections"]:
    """Mutate the disconnections of a single topology.

    Impossible mutations (e.g. adding a disconnection when there is no room to add one,
    or removing a disconnection when there are none) are handled by setting the probabilities
    for these mutations to zero, and adding them to the remain probability.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids of the topology,
        which are needed to determine whether certain disconnection mutations are allowed
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before mutation of one individual
    disconnection_mutation_config : DisconnectionMutationConfig
        The configuration for the disconnection mutation,
        containing the probabilities for the different mutation types and the
        number of disconnectable branches in the grid,
        which is needed to determine the valid range of branch ids for mutation.

    Returns
    -------
    Int[Array, " max_num_disconnections"]
        The mutated disconnections
    """
    int_max_value = int_max()
    operation_key, disc_key, random_key = jax.random.split(random_key, 3)

    # Gather probabilities and config values from the mutation config
    add_disconnection_prob = disconnection_mutation_config.add_disconnection_prob
    change_disconnection_prob = disconnection_mutation_config.change_disconnection_prob
    remove_disconnection_prob = disconnection_mutation_config.remove_disconnection_prob

    n_disconnectable_branches = disconnection_mutation_config.n_disconnectable_branches

    # Gather info about the current topology
    has_splits = jnp.any(sub_ids != int_max_value)
    n_disconnections = jnp.sum(disconnections != int_max_value)
    max_num_disconnections = disconnections.shape[0]

    # Check which actions are allowed.
    # We only allow to add a disconnection if there are less disconnections than the maximum number of disconnections.
    allow_add = (n_disconnections < max_num_disconnections) & (add_disconnection_prob > 0.0)
    # We only allow to remove a disconnection if there is at least one disconnection,
    # and we don't want to end up with zero disconnections if there are no splits in the topology,
    # because then we would always end up with the same unsplit topology after mutation.
    allow_remove = (has_splits & (n_disconnections == 1)) | (n_disconnections > 1)

    # We only allow to change a disconnection if there is at least one disconnection,
    # otherwise there is nothing to change
    allow_replace = n_disconnections > 0

    allow_remain = True  # We can always choose to remain unchanged
    # We temporarily set the remain probability to 0, because we will add the probabilities
    # of the illegal actions to the remain probability later, after we set the illegal action probabilities to 0.
    temp_remain_prob = 0.0
    # Create an array of the probabilities for the different operations,
    # and set the probabilities to 0 for the operations that are not allowed.
    probs = jnp.array(
        [add_disconnection_prob, remove_disconnection_prob, change_disconnection_prob, temp_remain_prob], dtype=float
    )
    allowed = jnp.array([allow_add, allow_remove, allow_replace, allow_remain], dtype=bool)
    probs = jnp.where(allowed, probs, 0.0)

    # Replace all "illegal" operations with "remain unchanged".
    prob_sum = jnp.sum(probs)
    # If there are no splits, always add a disconnection
    # Otherwise, normalise the allowed probabilities to sum to 1
    # If probs are negative, only the remain option is considered
    probs = jnp.where(
        (~has_splits) & allow_add & (n_disconnections == 0),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        probs.at[3].set(1.0 - prob_sum),
    )

    # Randomly choose which operation to perform based on the probabilities
    # At least one of the operations is executed
    chosen_op = jax.random.choice(operation_key, jnp.arange(4), p=probs)
    disc_idx, new_branch_id = jax.lax.switch(
        chosen_op,
        [
            lambda _disconnections: disconnect_additional_branch(disc_key, _disconnections, n_disconnectable_branches),
            lambda _disconnections: reconnect_disconnected_branch(disc_key, _disconnections),
            lambda _disconnections: change_disconnected_branch(disc_key, _disconnections, n_disconnectable_branches),
            lambda _disconnections: do_nothing(int_max_value),  # remain unchanged
        ],
        disconnections,
    )
    # Update the disconnections with the new branch id at the chosen index
    disconnections = disconnections.at[disc_idx].set(new_branch_id, mode="drop")
    return disconnections
