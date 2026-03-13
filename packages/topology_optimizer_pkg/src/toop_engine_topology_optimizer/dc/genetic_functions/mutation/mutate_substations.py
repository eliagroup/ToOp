# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation functions for substations in the genetic algorithm."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.topology_computations import sample_action_index_from_branch_actions
from toop_engine_dc_solver.jax.types import ActionSet, int_max
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import SubstationMutationConfig
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.utils import (
    do_nothing,
    get_random_true_idx,
    sample_new_id,
)


def change_split_substation(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    n_rel_subs: int,
    int_max_value: int,
) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Change a split to a different one.

    Either in the same substation or in a different substation.
    This assumes, that there is a split already.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before mutation
    n_rel_subs : int
        Number of relevant substations that may be selected.
    int_max_value : int
        The value that indicates an unsplit substation, used to determine which substations
        are currently split and to sample a new substation id from the valid range

    Returns
    -------
    Int[Array, " max_num_splits"]
        The mutated substation ids, where one split substation is changed to a different one.
        If there are no split substations, the input sub_ids are returned unchanged
    Int[Array, " "]
        The index of the substation that was mutated. If no substation was mutated, this is set to int_max
    """
    split_key, sub_key = jax.random.split(random_key, 2)
    is_split = sub_ids != int_max_value
    split_idx = get_random_true_idx(split_key, is_split, int_max_value)
    new_substation_idx = sample_new_id(sub_key, sub_ids, n_rel_subs, split_idx)
    return split_idx, new_substation_idx


def unsplit_substation(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    int_max_value: int,
) -> tuple[
    Int[Array, " "],
    Int[Array, " "],
]:
    """Reset a split substation to the unsplit state, with a certain probability.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the reset
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before the reset
    int_max_value : int
        The value that indicates an unsplit substation, used to determine which substations are currently
        split and to sample a new substation id from the valid range

    Returns
    -------
    Int[Array, " max_num_splits"]
        The substation ids after the reset. If a reset occurred, the substation id at split_idx
        will be set to int_max
    Int[Array, " max_num_splits"]
        The topology action after the reset. If a reset occurred, the topology at split_idx
        will be set to int_max
    """
    already_split = sub_ids != int_max_value
    split_idx = get_random_true_idx(random_key, already_split, int_max_value)
    return split_idx, jnp.array(int_max_value)


def split_additional_sub(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    n_rel_subs: int,
    int_max_value: int,
) -> tuple[Int[Array, " "], Int[Array, " "]]:
    """Mutate the substation ids of a single topology.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before mutation
    n_rel_subs : int
        Number of relevant substations that may be selected.
    int_max_value : int
        The value to use for the mutation, which should be the maximum integer value used
        to indicate an empty slot in the genotype.

    Returns
    -------
    Int[Array, " max_num_splits"]
        The mutated substation ids
    Int[Array, " "]
        The index of the substation that was mutated. If no substation was mutated, this is set to
        a random substation that was already split, so branch and injection mutations can still
        happen despite no new split. If no substation was split yet and no split happened, this is
        set to int_max
    """
    split_key, sub_key = jax.random.split(random_key, 2)
    non_split = sub_ids == int_max_value
    split_idx = get_random_true_idx(split_key, non_split, int_max_value)
    new_substation_idx = sample_new_id(sub_key, sub_ids, n_rel_subs, int_max_value)
    return split_idx, new_substation_idx


def mutate_sub_splits(
    sub_ids: Int[Array, " max_num_splits"],
    action: Int[Array, " max_num_splits"],
    random_key: PRNGKeyArray,
    sub_mutate_config: SubstationMutationConfig,
    action_set: ActionSet,
) -> tuple[
    Int[Array, " max_num_splits"],
    Int[Array, " max_num_splits"],
    PRNGKeyArray,
]:
    """Mutate a single substation, changing the sub_ids, branch and inj topos.

    The sub-ids are implicit to the branch topo action index, however we pass them explicitely
    to aid substation mutation.

    Parameters
    ----------
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before mutation
    action : Int[Array, " max_num_splits"]
        The branch topology before mutation
    random_key : PRNGKeyArray
        The random key to use for the mutation
    sub_mutate_config : SubstationMutationConfig
        The configuration for the substation mutation,
        containing the probabilities for the different mutation types and the
        number of relevant substations in the grid, which is needed to determine
        the valid range of substation ids for mutation.
    action_set : ActionSet
        The actions for every substation. If not provided, will sample from all possible
        actions per substation

    Returns
    -------
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids after mutation
    action : Int[Array, " max_num_splits"]
        The branch topology after mutation
    random_key : PRNGKeyArray
        The random key used for the mutation
    """
    int_max_value = int_max()
    operation_key, sub_key, action_key, random_key = jax.random.split(random_key, 4)

    # Gather probabilities for the different mutation operations
    add_split_prob = sub_mutate_config.add_split_prob
    change_split_prob = sub_mutate_config.change_split_prob
    remove_split_prob = sub_mutate_config.remove_split_prob
    prob_remain = 1 - add_split_prob - change_split_prob - remove_split_prob

    # Gather config values for the mutation operations
    n_rel_subs = sub_mutate_config.n_rel_subs
    n_max_splits = sub_ids.shape[0]
    is_split = sub_ids != int_max_value
    n_splits = jnp.sum(is_split)

    # Determine which mutation operations are allowed based on the current topology, and adjust probabilities accordingly
    allow_add = n_splits < n_max_splits
    allow_remove = n_splits > 0
    allow_replace = n_splits > 0
    allow_remain = True

    probs = jnp.array([add_split_prob, remove_split_prob, change_split_prob, prob_remain], dtype=float)
    # assert jnp.isclose(probs.sum(), 1.0), f"Probabilities must sum to 1, but got {probs}"
    allowed = jnp.array([allow_add, allow_remove, allow_replace, allow_remain], dtype=bool)
    probs = jnp.where(allowed, probs, 0.0)
    probs = jnp.where(jnp.sum(probs) > 0, probs / jnp.sum(probs), jnp.array([0.0, 0.0, 0.0, 1.0]))

    # Pick one of the mutations at random
    chosen_op = jax.random.choice(a=probs.shape[0], shape=(), p=probs, key=operation_key)

    changed_indices, new_sub_ids = jax.lax.switch(
        chosen_op,
        [
            lambda _sub_ids: split_additional_sub(sub_key, _sub_ids, n_rel_subs, int_max_value),
            lambda _sub_ids: unsplit_substation(sub_key, _sub_ids, int_max_value),
            lambda _sub_ids: change_split_substation(sub_key, _sub_ids, n_rel_subs, int_max_value),
            lambda _sub_ids: do_nothing(int_max_value),  # remain unchanged
        ],
        sub_ids,
    )
    sub_ids = sub_ids.at[changed_indices].set(new_sub_ids, mode="drop")

    # Update the action for the changed substation
    new_action = sample_action_index_from_branch_actions(
        rng_key=action_key,
        sub_id=new_sub_ids,
        branch_action_set=action_set,
    )

    action = action.at[changed_indices].set(new_action, mode="drop")
    return sub_ids, action, random_key
