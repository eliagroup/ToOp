# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation functions for the genetic algorithm."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids, sample_action_index_from_branch_actions
from toop_engine_dc_solver.jax.types import ActionSet, int_max
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype, deduplicate_genotypes, fix_dtypes
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    MutationConfig,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_disconnections import mutate_disconnections
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_nodal_inj import mutate_nodal_injections
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_substations import mutate_sub_splits
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.utils import sample_unique_indices_small_k


def mutate_topology(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    disconnections_topo: Int[Array, " max_num_disconnections"],
    action: Int[Array, " max_num_splits"],
    mutate_config: MutationConfig,
    action_set: ActionSet,
) -> tuple[
    Int[Array, " max_num_splits"], Int[Array, " max_num_splits"], Int[Array, " max_num_disconnections"], PRNGKeyArray
]:
    """Mutate the topology by changing the substation splits and disconnections according to the mutation configuration.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before the mutation
    disconnections_topo : Int[Array, " max_num_disconnections"]
        The disconnections before the mutation
    action : Int[Array, " max_num_splits"]
        The actions before the mutation
    mutate_config : MutationConfig
        The mutation configuration
    action_set : ActionSet
        The set of possible actions

    Returns
    -------
    Int[Array, " max_num_splits"],
        The splits substation ids after the mutation

    Int[Array, " max_num_splits"],
        The action ids after the mutation

    Int[Array, " max_num_disconnections"],
        The disconnections after the mutation
    PRNGKeyArray
        The mutated substation ids, actions, disconnections, and the updated random key
    """
    random_key, substation_key, disconnection_key = jax.random.split(random_key, 3)
    max_num_splits = sub_ids.shape[0]
    n_subs_mutated = jax.random.poisson(
        substation_key, lam=mutate_config.substation_mutation_config.n_subs_mutated_lambda, shape=()
    )
    n_subs_mutated = jnp.clip(n_subs_mutated, 1, max_num_splits)

    # Mutate sub_ids, action and injection_topo n_subs_mutated times
    sub_ids, action, random_key = jax.lax.fori_loop(
        0,
        n_subs_mutated,
        lambda _i, args: mutate_sub_splits(
            sub_ids=args[0],
            action=args[1],
            random_key=args[2],
            sub_mutate_config=mutate_config.substation_mutation_config,
            action_set=action_set,
        ),
        (
            sub_ids,
            action,
            substation_key,
        ),
    )
    if (disconnections_topo.size > 0) and mutate_config.disconnection_mutation_config.n_disconnectable_branches > 0:
        disconnections_topo = mutate_disconnections(
            random_key=disconnection_key,
            sub_ids=sub_ids,
            disconnections=disconnections_topo,
            disconnection_mutation_config=mutate_config.disconnection_mutation_config,
        )
    return sub_ids, action, disconnections_topo, random_key


def create_random_topology(
    random_key: PRNGKeyArray,
    sub_ids: Int[Array, " max_num_splits"],
    disconnections: Int[Array, " max_num_disconnections"],
    action_set: ActionSet,
    n_rel_subs: int,
    n_disconnectable_branches: int,
) -> tuple[
    Int[Array, " max_num_splits"], Int[Array, " max_num_splits"], Int[Array, " max_num_disconnections"], PRNGKeyArray
]:
    """Create a random topology by sampling random substation splits, branch topologies and disconnections.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before the mutation, used to determine the number of splits to sample
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before the mutation, used to determine the number of disconnections to sample
    action_set : ActionSet
        The set of possible actions, used to sample valid actions for the new topology
    n_rel_subs : int
        The number of relevant substations in the grid, used to determine the valid range of substation ids for mutation.
    n_disconnectable_branches: int
        The number of disconnectable branches in the grid, used to determine the valid range of branch ids for mutation.

    Returns
    -------
    Int[Array, " max_num_splits"],
        The splits substation ids after the random mutation
    Int[Array, " max_num_splits"],
        The action ids after the random mutation
    Int[Array, " max_num_disconnections"],
        The disconnections after the random mutation
    PRNGKeyArray
        The mutated substation ids, actions, disconnections, and the updated random key
    """
    unsplit_key, subs_key, actions_key, not_disc_key, disc_key, random_key = jax.random.split(random_key, 6)

    max_num_splits = sub_ids.shape[0]
    max_disconnections = disconnections.shape[0]

    unsplit_mask = jax.random.bernoulli(unsplit_key, p=0.5, shape=(max_num_splits,))
    random_subs = sample_unique_indices_small_k(subs_key, n_rel_subs, max_num_splits)

    sub_ids = jnp.where(unsplit_mask, int_max(), random_subs)
    action = jax.vmap(lambda sub_id, key: sample_action_index_from_branch_actions(key, sub_id, action_set))(
        sub_ids,
        jax.random.split(actions_key, max_num_splits),
    )

    not_disconnected_mask = jax.random.bernoulli(not_disc_key, p=0.5, shape=(max_disconnections,))
    random_disconnections = sample_unique_indices_small_k(disc_key, n_disconnectable_branches, max_disconnections)
    disconnections = jnp.where(not_disconnected_mask, int_max(), random_disconnections)

    return sub_ids, action, disconnections, random_key


def mutate(
    topologies: Genotype,
    random_key: PRNGKeyArray,
    mutation_config: MutationConfig,
    action_set: ActionSet,
) -> tuple[Genotype, PRNGKeyArray]:
    """Mutate the topologies by splitting substations, disconnecting branches and mutating nodal injections (PSTs).

    Makes sure that at all times, a substation is split at most once and that all branch actions
    are in range of the available actions for the substation. If a substation is
    not split, this is indicated by the value int_max in the substation, branch.

    We mutate mutation_repetition copies of the initial repertoire to increase the chance of getting
    unique mutations.

    Parameters
    ----------
    topologies : Genotype
        The topologies to mutate
    random_key : PRNGKeyArray
        The random key to use for the mutation
    mutation_config : MutationConfig
        The mutation configuration containing the probabilities for the different mutation operations and their parameters
    action_set : ActionSet
        The action set containing available actions on a per-substation basis.

    Returns
    -------
    Genotype
        The mutated topologies
    PRNGKeyArray
        The new random key
    """
    topologies = fix_dtypes(topologies)
    batch_size = len(topologies.action_index)

    # Repeat the topologies to increase the chance of getting unique mutations
    repeated_topologies = repeat_topologies(topologies, batch_size, mutation_config.mutation_repetition)
    n_mutations = batch_size * mutation_config.mutation_repetition
    mutation_key, replacement_key, pst_key, random_key = jax.random.split(random_key, 4)
    sub_ids = extract_sub_ids(
        repeated_topologies.action_index,
        action_set,
    )
    n_random_topologies = round(mutation_config.random_topo_prob * n_mutations)
    n_random_topologies = max(0, min(n_mutations, n_random_topologies))

    mutate_topologies_batch = jax.vmap(
        lambda sub_id, action_single, disconnection_single, key: mutate_topology(
            random_key=key,
            sub_ids=sub_id,
            disconnections_topo=disconnection_single,
            action=action_single,
            mutate_config=mutation_config,
            action_set=action_set,
        )
    )

    create_random_topologies_batch = jax.vmap(
        lambda sub_id, disconnection_single, key: create_random_topology(
            random_key=key,
            sub_ids=sub_id,
            disconnections=disconnection_single,
            action_set=action_set,
            n_rel_subs=mutation_config.substation_mutation_config.n_rel_subs,
            n_disconnectable_branches=mutation_config.disconnection_mutation_config.n_disconnectable_branches,
        )
    )

    if n_random_topologies == 0:
        sub_ids, action, disconnections_topo, _ = mutate_topologies_batch(
            sub_ids,
            repeated_topologies.action_index,
            repeated_topologies.disconnections,
            jax.random.split(mutation_key, n_mutations),
        )
    elif n_random_topologies == n_mutations:
        sub_ids, action, disconnections_topo, _ = create_random_topologies_batch(
            sub_ids,
            repeated_topologies.disconnections,
            jax.random.split(replacement_key, n_mutations),
        )
    else:
        sub_ids, action, disconnections_topo, _ = mutate_topologies_batch(
            sub_ids,
            repeated_topologies.action_index,
            repeated_topologies.disconnections,
            jax.random.split(mutation_key, n_mutations),
        )

        replacement_idx_key, replacement_topology_key = jax.random.split(replacement_key)
        replacement_indices = jax.random.choice(
            replacement_idx_key,
            n_mutations,
            shape=(n_random_topologies,),
            replace=False,
        )

        random_sub_ids, random_action, random_disconnections, _ = create_random_topologies_batch(
            sub_ids[replacement_indices],
            disconnections_topo[replacement_indices],
            jax.random.split(replacement_topology_key, n_random_topologies),
        )

        sub_ids = sub_ids.at[replacement_indices].set(random_sub_ids)
        action = action.at[replacement_indices].set(random_action)
        disconnections_topo = disconnections_topo.at[replacement_indices].set(random_disconnections)

    nodal_injections_optimized = mutate_nodal_injections(
        random_key=pst_key,
        nodal_inj_info=repeated_topologies.nodal_injections_optimized,
        nodal_mutation_config=mutation_config.nodal_injection_mutation_config,
    )

    # Sort all action_idx and disconnections, so that the order does not matter for the uniqueness check.
    # We can sort the action_idx and disconnections because the order of the splits and disconnections does not matter
    topologies_mutated = Genotype(
        action_index=jnp.sort(action, axis=1),
        disconnections=jnp.sort(disconnections_topo, axis=1),
        nodal_injections_optimized=nodal_injections_optimized,
    )
    if mutation_config.mutation_repetition > 1:
        topologies_mutated, _ = deduplicate_genotypes(
            topologies_mutated,
            desired_size=batch_size,
        )

    return topologies_mutated, random_key


def repeat_topologies(topologies: Genotype, batch_size: int, mutation_repetition: int) -> Genotype:
    """Repeat the topologies mutation_repetition times to increase the chance of getting unique mutations.

    Parameters
    ----------
    topologies : Genotype
        The topologies to repeat
    batch_size : int
        The batch size of the original topologies, used to determine the total size after repetition
    mutation_repetition : int
        The number of times to repeat the topologies

    Returns
    -------
    Genotype
        The repeated topologies
    """
    if mutation_repetition == 1:
        repeated_topologies = topologies
    else:
        repeated_topologies: Genotype = jax.tree.map(
            lambda x: jnp.repeat(
                x, repeats=mutation_repetition, axis=0, total_repeat_length=batch_size * mutation_repetition
            ),
            topologies,
        )

    return repeated_topologies
