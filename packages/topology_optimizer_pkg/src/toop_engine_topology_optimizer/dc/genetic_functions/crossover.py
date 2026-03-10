# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the genetic operations for the topologies.

This module contains the functions to perform the genetic operations of mutation and crossover on
the topologies of the grid. The topologies are represented as Genotype dataclasses, which contain
the substation ids, the branch topology, the injection topology and the disconnections.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids
from toop_engine_dc_solver.jax.types import ActionSet, int_max
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype


def sample_unique_from_array(
    random_key: PRNGKeyArray,
    sample_pool: Int[Array, " n"],
    sample_probs: Float[Array, " n"],
    n_samples: int,
) -> Int[Array, " n_samples"]:
    """Sample n unique elements from an array (returning indices), counting int_max as always unique.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the sampling
    sample_pool : Int[Array, " n"]
        The array to sample from. Only unique elements are sampled, however int_max entries are
        not checked for uniqueness
    sample_probs : Float[Array, " n"]
        The probabilities to sample each element
    n_samples : int
        The number of samples to take, should be less than the number of non-int_max entries in
        array.

    Returns
    -------
    Int[Array, " n_samples"]
        The sampled indices into sample_pool
    """
    subkeys = jax.random.split(random_key, n_samples)

    def _body_fn(
        i: Int[ArrayLike, " "],
        entries_sampled: tuple[Int[Array, " max_num_splits"], Bool[Array, " n_subs_rel"]],
    ) -> tuple[Int[Array, " max_num_splits"], Bool[Array, " n_subs_rel"]]:
        indices_sampled, choice_mask = entries_sampled

        probs = sample_probs * choice_mask
        probs = probs / jnp.sum(probs)

        current_index = jax.random.choice(subkeys[i], jnp.arange(sample_pool.shape[0]), shape=(1,), p=probs)[0]

        # Update the choice mask
        # If a substation has been sampled, we can't sample this substation again
        # If a no-split (int_max) has been sampled, we only mask out the sampled index
        choice_mask = jnp.where(
            sample_pool[current_index] == int_max(),
            choice_mask.at[current_index].set(False, mode="promise_in_bounds"),
            jnp.where(
                sample_pool == sample_pool[current_index],
                False,
                choice_mask,
            ),
        )

        # Update the indices sampled
        indices_sampled = indices_sampled.at[i].set(current_index, mode="promise_in_bounds")

        return (indices_sampled, choice_mask)

    indices_sampled, _choice_mask = jax.lax.fori_loop(
        lower=0,
        upper=n_samples,
        body_fun=_body_fn,
        init_val=(
            jnp.full((n_samples,), int_max(), dtype=int),
            jnp.ones(sample_pool.shape, dtype=bool),
        ),
        unroll=True,
    )

    return indices_sampled


def crossover_unbatched(
    topologies_a: Genotype,
    topologies_b: Genotype,
    random_key: PRNGKeyArray,
    action_set: ActionSet,
    prob_take_a: float,
) -> tuple[Genotype, PRNGKeyArray]:
    """Crossover two topologies while making sure that no substation is present twice.

    This version is unbatched, i.e. it only works on a single topology. Use crossover for batched
    inputs.

    Parameters
    ----------
    topologies_a : Genotype
        The first topology
    topologies_b : Genotype
        The second topology
    random_key : PRNGKeyArray
        The random key to use for the crossover
    action_set : ActionSet
        The branch action set containing available actions on a per-substation basis.
        This is needed to resolve the sub_ids for the branch actions
    prob_take_a : float
        The probability to take the value from topology_a, otherwise the value from topology_b is
        taken

    Returns
    -------
    Genotype
        The new topology
    PRNGKeyArray
        The new random key
    """
    # The tricky part in the crossover is that both topologies could have the same sub-id on
    # different indices. We need to make sure that we don't end up with the same substation twice
    # in the new topology

    max_num_splits = topologies_a.action_index.shape[0]
    max_num_disconnections = topologies_a.disconnections.shape[0]
    sample_bus_key, sample_disc_key, random_key = jax.random.split(random_key, 3)

    # The probability to take the value from a is prob_take_a
    # The probability to take the value from b is 1 - prob_take_a
    base_sample_probs = jnp.ones(2 * max_num_splits)
    base_sample_probs = base_sample_probs.at[:max_num_splits].set(prob_take_a)
    base_sample_probs = base_sample_probs.at[max_num_splits:].set(1 - prob_take_a)

    topologies_concatenated: Genotype = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        topologies_a,
        topologies_b,
    )
    sub_ids_concatenated = extract_sub_ids(topologies_concatenated.action_index, action_set)

    indices_sampled = sample_unique_from_array(
        random_key=sample_bus_key,
        sample_pool=sub_ids_concatenated,
        sample_probs=base_sample_probs,
        n_samples=max_num_splits,
    )

    actions = topologies_concatenated.action_index.at[indices_sampled].get(mode="fill", fill_value=int_max())

    # Sample disconnection choices
    if max_num_disconnections != 0:
        base_sample_probs = jnp.ones(2 * max_num_disconnections)
        base_sample_probs = base_sample_probs.at[:max_num_disconnections].set(prob_take_a)
        base_sample_probs = base_sample_probs.at[max_num_disconnections:].set(1 - prob_take_a)

        disconnections_concatenated: Int[Array, " 2*max_num_disconnections"] = topologies_concatenated.disconnections
        indices_sampled = sample_unique_from_array(
            random_key=sample_disc_key,
            sample_pool=disconnections_concatenated,
            sample_probs=base_sample_probs,
            n_samples=max_num_disconnections,
        )

        disconnections = disconnections_concatenated.at[indices_sampled].get(mode="fill", fill_value=int_max())
    else:
        disconnections = jnp.array([], dtype=int)

    return Genotype(
        action_index=actions,
        disconnections=disconnections,
        nodal_injections_optimized=topologies_a.nodal_injections_optimized,
    ), random_key


def crossover(
    topologies_a: Genotype,
    topologies_b: Genotype,
    random_key: PRNGKeyArray,
    action_set: ActionSet,
    prob_take_a: float,
) -> tuple[Genotype, PRNGKeyArray]:
    """Crossover two topologies while making sure that no substation is present twice.

    Parameters
    ----------
    topologies_a : Genotype
        The first topology
    topologies_b : Genotype
        The second topology
    random_key : PRNGKeyArray
        The random key to use for the crossover
    action_set : ActionSet
        The branch action set containing available actions on a per-substation basis.
    prob_take_a : float
        The probability to take the value from topology_a, otherwise the value from topology_b is
        taken

    Returns
    -------
    Genotype
        The new topology
    PRNGKeyArray
        The new random key
    """
    batch_size = topologies_a.action_index.shape[0]
    crossover_keys = jax.random.split(random_key, batch_size)
    topo, random_keys = jax.vmap(crossover_unbatched, in_axes=(0, 0, 0, None, None))(
        topologies_a, topologies_b, crossover_keys, action_set, prob_take_a
    )
    return topo, random_keys[-1]
