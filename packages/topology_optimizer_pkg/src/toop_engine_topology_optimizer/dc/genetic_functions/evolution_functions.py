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

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids, sample_action_index_from_branch_actions
from toop_engine_dc_solver.jax.types import ActionSet, NodalInjOptimResults, int_max


@pytree_dataclass
class Genotype:
    """A single genome in the repertoire representing a topology."""

    action_index: Int[Array, " *batch_size max_num_splits"]
    """An action index into the action set"""

    disconnections: Int[Array, " *batch_size max_num_disconnections"]
    """The disconnections to apply, padded with int_max for disconnection slots that are unused.
    These are indices into the disconnectable branches set."""

    nodal_injections_optimized: Optional[NodalInjOptimResults]
    """The results of the nodal injection optimization, if any was performed."""


def deduplicate_genotypes(
    genotypes: Genotype,
    desired_size: Optional[int] = None,
) -> tuple[Genotype, Int[Array, " n_unique"]]:
    """Deduplicate the genotypes in the repertoire.

    This version is jittable because we set the size

    Parameters
    ----------
    genotypes : Genotype
        The genotypes to deduplicate
    desired_size : Optional[int]
        How many unique values you are expecting. If not given, this is not jittable

    Returns
    -------
    Genotype
        The deduplicated genotypes
    Int[Array, " n_unique"]
        The indices of the unique genotypes
    """
    # Purposefully not taking into account nodal_injections_optimized, as these are not part of the topology
    genotype_flat = jnp.concatenate(
        [
            genotypes.action_index,
            genotypes.disconnections,
        ],
        axis=1,
    )

    _, indices = jnp.unique(
        genotype_flat,
        axis=0,
        return_index=True,
        size=desired_size,
        # fill_value takes the minimum flattened topology by default
        # it also corresponds to the first index in the list
    )
    unique_genotypes = jax.tree_util.tree_map(lambda x: x[indices], genotypes)
    return unique_genotypes, indices


def fix_dtypes(genotypes: Genotype) -> Genotype:
    """Fix the dtypes of the genotypes to their native type.

    For some reason, qdax aggressively converts everything to float

    Parameters
    ----------
    genotypes : Genotype
        The genotypes to fix

    Returns
    -------
    Genotype
        The genotypes with fixed dtypes
    """
    return Genotype(
        action_index=genotypes.action_index.astype(int),
        disconnections=genotypes.disconnections.astype(int),
        nodal_injections_optimized=genotypes.nodal_injections_optimized,
    )


def empty_repertoire(
    batch_size: int,
    max_num_splits: int,
    max_num_disconnections: int,
    num_psts: int,
) -> Genotype:
    """Create an initial genotype repertoire with all zeros for all entries and int_max for all subs.

    Parameters
    ----------
    batch_size : int
        The batch size
    max_num_splits : int
        The maximum number of splits per topology
    max_num_disconnections : int
        The maximum number of diconncections as topological measures per topology
    num_psts : int
        The number of controllable PSTs in the grid

    Returns
    -------
    Genotype
        The initial genotype
    """
    return Genotype(
        action_index=jnp.full((batch_size, max_num_splits), int_max(), dtype=int),
        disconnections=jnp.full((batch_size, max_num_disconnections), int_max(), dtype=int),
        # TODO: Why don't we use the n_timesteps here?
        nodal_injections_optimized=NodalInjOptimResults(pst_taps=jnp.zeros((batch_size, num_psts), dtype=float))
        if num_psts > 0
        else None,
    )


def mutate(
    topologies: Genotype,
    random_key: jax.random.PRNGKey,
    substation_split_prob: float,
    substation_unsplit_prob: float,
    action_set: ActionSet,
    n_disconnectable_branches: int,
    n_subs_mutated_lambda: float,
    disconnect_prob: float,
    reconnect_prob: float,
    mutation_repetition: int = 1,
) -> tuple[Genotype, jax.random.PRNGKey]:
    """Mutate the topologies by splitting substations and changing the branch and injection topos.

    Makes sure that at all times, a substation is split at most once and that all branch and
    injection actions are in range of the available actions for the substation. If a substation is
    not split, this is indicated by the value int_max in the substation, branch and injection.

    We mutate mutation_repetition copies of the initial repertoire to increase the chance of getting
    unique mutations.

    Parameters
    ----------
    topologies : Genotype
        The topologies to mutate
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    substation_split_prob : float
        The probability to split a substation. In case all substations are already split, this
        probability is ignored
    substation_unsplit_prob : float
        The probability to reset a substation to the unsplit state
    action_set : ActionSet
        The action set containing available actions on a per-substation basis.
    n_disconnectable_branches: int
        The number of disconnectable branches in the action set.
    n_subs_mutated_lambda : float
        The lambda for the poisson distribution to determine the number of substations to mutate
    disconnect_prob : float
        The probability to disconnect a new branch
    reconnect_prob : float
        The probability to reconnect a disconnected branch, will overwrite a possible disconnect
    mutation_repetition : int
        More chance to get unique mutations by mutating mutation_repetition copies of the repertoire

    Returns
    -------
    Genotype
        The mutated topologies
    jax.random.PRNGKey
        The new random key
    """
    max_num_splits = topologies.action_index.shape[1]

    def _mutate_single_topo(
        sub_ids: Int[Array, " max_num_splits"],
        action: Int[Array, " max_num_splits"],
        disconnections_topo: Int[Array, " max_num_disconnections"],
        random_key: jax.random.PRNGKey,
    ) -> tuple[
        Int[Array, " max_num_splits"],
        Int[Array, " max_num_splits"],
        Int[Array, " max_num_disconnections"],
        jax.random.PRNGKey,
    ]:
        """Mutates a single topology n_subs_mutated times and adds disconnections."""
        # Sample number of subs mutated from a poisson
        random_key, subkey = jax.random.split(random_key)
        n_subs_mutated = jax.random.poisson(subkey, lam=n_subs_mutated_lambda, shape=())
        n_subs_mutated = jnp.clip(n_subs_mutated, 1, max_num_splits)

        # Mutate sub_ids, action and injection_topo n_subs_mutated times
        sub_ids, action, random_key = jax.lax.fori_loop(
            0,
            n_subs_mutated,
            lambda _i, args: mutate_sub(
                sub_ids=args[0],
                action=args[1],
                random_key=args[2],
                substation_split_prob=substation_split_prob,
                substation_unsplit_prob=substation_unsplit_prob,
                action_set=action_set,
            ),
            (
                sub_ids,
                action,
                random_key,
            ),
        )

        # Mutate the disconnections
        random_key, subkey = jax.random.split(random_key)
        disconnections_topo = mutate_disconnections(
            random_key=subkey,
            disconnections=disconnections_topo,
            n_disconnectable_branches=n_disconnectable_branches,
            disconnect_prob=disconnect_prob,
            reconnect_prob=reconnect_prob,
        )

        return sub_ids, action, disconnections_topo, random_key

    topologies = fix_dtypes(topologies)
    batch_size = len(topologies.action_index)

    # Repeat the topologies to increase the chance of getting unique mutations
    if mutation_repetition == 1:
        repeated_topologies = topologies
    else:
        repeated_topologies: Genotype = jax.tree.map(
            lambda x: jnp.repeat(
                x, repeats=mutation_repetition, axis=0, total_repeat_length=batch_size * mutation_repetition
            ),
            topologies,
        )

    random_keys = jax.random.split(random_key, batch_size * mutation_repetition)
    sub_ids = extract_sub_ids(
        repeated_topologies.action_index,
        action_set,
    )

    (
        sub_ids,
        action,
        disconnections_topo,
        random_keys,
    ) = jax.vmap(_mutate_single_topo)(
        sub_ids,
        repeated_topologies.action_index,
        repeated_topologies.disconnections,
        random_keys,
    )
    random_key = random_keys[0]

    topologies_mutated = Genotype(
        action_index=action,
        disconnections=disconnections_topo,
        nodal_injections_optimized=repeated_topologies.nodal_injections_optimized,
    )

    if mutation_repetition > 1:
        topologies_mutated, _ = deduplicate_genotypes(
            topologies_mutated,
            desired_size=batch_size,
        )

    return topologies_mutated, random_key


def mutate_sub_id(
    random_key: jax.random.PRNGKey,
    sub_ids: Int[Array, " max_num_splits"],
    n_subs_rel: int,
    substation_split_prob: float,
) -> tuple[Int[Array, " max_num_splits"], Int[Array, " "]]:
    """Mutate the substation ids of a single topology.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before mutation
    n_subs_rel : int
        The number of relevant substations in the grid
    substation_split_prob : float
        The probability to split a substation

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
    int_max = jnp.iinfo(jnp.int32).max
    randkeys = jax.random.split(random_key, 3)
    max_num_splits = sub_ids.shape[0]

    # Write a boolean array which substations have been split already, we don't want to split these
    # again
    substations_split = jnp.zeros(n_subs_rel, dtype=bool).at[sub_ids].set(True, mode="drop")
    all_split = jnp.all(substations_split)

    # Sample a new substation from the substations which have not been split yet for every topology
    # in the batch
    new_substation_idx: Int[Array, " "] = jax.random.categorical(randkeys[0], jnp.log(1 - substations_split.astype(float)))

    # Now, decide on which index to update the substation. We only have max_num_splits substations
    # that can be split, so we need to choose one of the spots.
    # Additionally, we want a certain probability to leave the substation unchanged, we implement
    # This by sampling an index that could be out of bounds, and dropping the update if it is
    split_idx: Int[Array, " "] = jax.random.randint(
        randkeys[1],
        shape=(1,),
        minval=0,
        maxval=int(max_num_splits * (1 / substation_split_prob)),
    )[0]
    # If all substations have been split, do not update
    split_idx = jnp.where(all_split, int_max, split_idx)
    sub_ids = sub_ids.at[split_idx].set(new_substation_idx, mode="drop")

    # Furthermore, we want to mutate the branch topology and the injection topology. We want to
    # mutate exactly one substation per topology every time
    # If the substation id has changed, we're forced to also change the branch topology and the
    # injection topology of that substation. Otherwise (split_idx >= max_num_splits), we'll resample
    # During resampling, we want to make sure we're sampling a spot that has a split (if any)
    resampled_split = jax.random.categorical(randkeys[2], jnp.log((sub_ids != int_max).astype(float)))
    split_idx = jnp.where(split_idx >= max_num_splits, resampled_split, split_idx)
    # Edge case: No substations have been split yet
    split_idx = jnp.where(sub_ids[split_idx] == int_max, int_max, split_idx)

    return sub_ids, split_idx


def mutate_disconnections(
    random_key: jax.random.PRNGKey,
    disconnections: Int[Array, " max_num_disconnections"],
    n_disconnectable_branches: int,
    disconnect_prob: float,
    reconnect_prob: float,
) -> Int[Array, " max_num_disconnections"]:
    """Mutate the disconnections of a single topology.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    disconnections : Int[Array, " max_num_disconnections"]
        The disconnections before mutation of one individual
    n_disconnectable_branches: int,
        The number of disconnectable branches in the action set. It is not necessary to know the
        contents of the action set here because we only sample an index into the action set.
    disconnect_prob : float
        The probability to disconnect a new branch
    reconnect_prob : float
        The probability to reconnect a disconnected branch, will overwrite a possible disconnect

    Returns
    -------
    Int[Array, " max_num_disconnections"]
        The mutated disconnections
    """
    key1, key2, key3 = jax.random.split(random_key, 3)
    if disconnect_prob > 0 and disconnections.size > 0 and n_disconnectable_branches > 0:
        # List available disconnections
        available_disconnections = ~jnp.isin(jnp.arange(n_disconnectable_branches), disconnections, assume_unique=True)

        # Choose a slot so that the probability of choosing out-of-bounds is equal to 1 - disconnect_prob
        disconnect_idx = jax.random.randint(
            key1,
            shape=(1,),
            minval=0,
            maxval=int(disconnections.shape[0] * (1 / disconnect_prob)),
        )[0]

        # Sample a new disconnection from the available disconnections
        new_disconnection = jax.random.categorical(key2, jnp.log(available_disconnections.astype(float)))
        # If no disconnection is available, set disconnect_idx to int_max
        disconnect_idx = jnp.where(available_disconnections.sum() == 0, int_max(), disconnect_idx)
        disconnections = disconnections.at[disconnect_idx].set(new_disconnection, mode="drop")

    # Choose a slot to overwrite with a reconnect
    if reconnect_prob > 0 and disconnections.size > 0 and n_disconnectable_branches > 0:
        reconnect_idx = jax.random.randint(
            key3,
            shape=(1,),
            minval=0,
            maxval=int(disconnections.shape[0] * (1 / reconnect_prob)),
        )[0]
        disconnections = disconnections.at[reconnect_idx].set(int_max(), mode="drop")

    return disconnections


def mutate_sub(
    sub_ids: Int[Array, " max_num_splits"],
    action: Int[Array, " max_num_splits"],
    random_key: jax.random.PRNGKey,
    substation_split_prob: float,
    substation_unsplit_prob: float,
    action_set: ActionSet,
) -> tuple[
    Int[Array, " max_num_splits"],
    Int[Array, " max_num_splits"],
    jax.random.PRNGKey,
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
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    substation_split_prob : float
        The probability to split a substation
    substation_unsplit_prob : float
        The probability to reset a split substation to the unsplit state
    action_set : ActionSet
        The actions for every substation. If not provided, will sample from all possible
        actions per substation

    Returns
    -------
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids after mutation
    action : Int[Array, " max_num_splits"]
        The branch topology after mutation
    random_key : jax.random.PRNGKey
        The random key used for the mutation
    """
    randkeys = jax.random.split(random_key, 7)

    n_subs_rel = len(action_set.n_actions_per_sub)
    assert substation_split_prob > 0 and substation_split_prob <= 1
    assert substation_unsplit_prob >= 0 and substation_unsplit_prob <= 1

    sub_ids, split_idx = mutate_sub_id(
        randkeys[0],
        sub_ids,
        n_subs_rel,
        substation_split_prob,
    )
    selected_sub = sub_ids.at[split_idx].get(mode="fill", fill_value=int_max())

    # The branch action set stores the available actions
    new_action = sample_action_index_from_branch_actions(
        rng_key=randkeys[3],
        sub_id=selected_sub,
        branch_action_set=action_set,
    )

    action = action.at[split_idx].set(new_action, mode="drop")

    sub_ids, action = unsplit_substation(
        random_key=randkeys[5],
        sub_ids=sub_ids,
        action=action,
        split_idx=split_idx,
        substation_unsplit_prob=substation_unsplit_prob,
    )

    # Order by sub_ids
    indices = jnp.argsort(sub_ids)
    sub_ids = sub_ids[indices]
    action = action[indices]

    return sub_ids, action, randkeys[6]


def mutate_nodal_injections(
    random_key: jax.random.PRNGKey,
    nodal_inj_info: Optional[NodalInjOptimResults],
    pst_mutation_sigma: float,
    pst_n_taps: Int[Array, " num_psts"],
) -> Optional[NodalInjOptimResults]:
    """Mutate the nodal injection optimization results, currently only the PST taps.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    nodal_inj_info : Optional[NodalInjOptimResults]
        The nodal injection optimization results before mutation. If None, no mutation is performed and None is returned.
    pst_mutation_sigma : float
        The sigma to use for the normal distribution to sample the mutation from. The mutation will be sampled as an
        integer from a normal distribution with mean 0 and sigma pst_mutation_sigma.
    pst_n_taps : Int[Array, " num_psts"]
        The number of taps for each PST. If a PST has N taps in this array, then it is assumed that all taps from
        0 to N-1 are valid tap positions. Output taps will be clipped to this range.

    Returns
    -------
    Optional[NodalInjOptimResults]
        The mutated nodal injection optimization results. If nodal_inj_info was None, returns None.
    """
    if nodal_inj_info is None:
        return None

    if pst_mutation_sigma <= 0:
        return nodal_inj_info

    batch_size = nodal_inj_info.pst_taps.shape[0]
    n_timesteps = nodal_inj_info.pst_taps.shape[1]
    random_key = jax.random.split(random_key, (batch_size, n_timesteps))

    # vmap to mutate the PST taps for each timestep + batch independently
    new_pst_taps = jax.vmap(jax.vmap(partial(mutate_psts, pst_n_taps=pst_n_taps, pst_mutation_sigma=pst_mutation_sigma)))(
        random_key=random_key,
        pst_taps=nodal_inj_info.pst_taps.astype(int),
    )

    return NodalInjOptimResults(pst_taps=new_pst_taps)


def mutate_psts(
    random_key: jax.random.PRNGKey,
    pst_taps: Int[Array, " num_psts"],
    pst_n_taps: Int[Array, " num_psts"],
    pst_mutation_sigma: float,
) -> Int[Array, " num_psts"]:
    """Mutate the PST taps of a single topology.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the mutation
    pst_taps : Int[Array, " num_psts"]
        The PST tap positions before mutation
    pst_n_taps : Int[Array, " num_psts"]
        The number of taps for each PST. If a PST has N taps in this array, then it is assumed that all taps from
        0 to N-1 are valid tap positions. Output taps will be clipped to this range.
    pst_mutation_sigma : float
        The sigma to use for the normal distribution to sample the mutation from. The mutation will be sampled as an
        integer from a normal distribution with mean 0 and sigma pst_mutation_sigma.

    Returns
    -------
    Int[Array, " num_psts"]
        The mutated PST tap positions, clipped to the valid range of taps for each PST.
    """
    mutation = jax.random.normal(random_key, shape=pst_taps.shape) * pst_mutation_sigma
    mutation = jnp.round(mutation).astype(int)
    new_pst_taps = pst_taps + mutation
    new_pst_taps = jnp.clip(new_pst_taps, a_min=0, a_max=pst_n_taps - 1)
    return new_pst_taps


def sample_unique_from_array(
    random_key: jax.random.PRNGKey,
    sample_pool: Int[Array, " n"],
    sample_probs: Float[Array, " n"],
    n_samples: int,
) -> Int[Array, " n_samples"]:
    """Sample n unique elements from an array (returning indices), counting int_max as always unique.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
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
        i: Int,
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
    random_key: jax.random.PRNGKey,
    action_set: ActionSet,
    prob_take_a: float,
) -> tuple[Genotype, jax.random.PRNGKey]:
    """Crossover two topologies while making sure that no substation is present twice.

    This version is unbatched, i.e. it only works on a single topology. Use crossover for batched
    inputs.

    Parameters
    ----------
    topologies_a : Genotype
        The first topology
    topologies_b : Genotype
        The second topology
    random_key : jax.random.PRNGKey
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
    jax.random.PRNGKey
        The new random key
    """
    # The tricky part in the crossover is that both topologies could have the same sub-id on
    # different indices. We need to make sure that we don't end up with the same substation twice
    # in the new topology

    max_num_splits = topologies_a.action_index.shape[0]
    max_num_disconnections = topologies_a.disconnections.shape[0]
    subkeys = jax.random.split(random_key, 3)

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
        random_key=subkeys[0],
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
            random_key=subkeys[1],
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
    ), subkeys[-1]


def crossover(
    topologies_a: Genotype,
    topologies_b: Genotype,
    random_key: jax.random.PRNGKey,
    action_set: ActionSet,
    prob_take_a: float,
) -> tuple[Genotype, jax.random.PRNGKey]:
    """Crossover two topologies while making sure that no substation is present twice.

    Parameters
    ----------
    topologies_a : Genotype
        The first topology
    topologies_b : Genotype
        The second topology
    random_key : jax.random.PRNGKey
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
    jax.random.PRNGKey
        The new random key
    """
    batch_size = topologies_a.action_index.shape[0]
    random_key = jax.random.split(random_key, batch_size)
    topo, random_keys = jax.vmap(crossover_unbatched, in_axes=(0, 0, 0, None, None))(
        topologies_a, topologies_b, random_key, action_set, prob_take_a
    )
    return topo, random_keys[0]


def unsplit_substation(
    random_key: jax.random.PRNGKey,
    sub_ids: Int[Array, " max_num_splits"],
    action: Int[Array, " max_num_splits"],
    split_idx: Int[Array, " "],
    substation_unsplit_prob: float,
) -> tuple[
    Int[Array, " max_num_splits"],
    Int[Array, " max_num_splits"],
]:
    """Reset a split substation to the unsplit state, with a certain probability.

    Parameters
    ----------
    random_key : jax.random.PRNGKey
        The random key to use for the reset
    sub_ids : Int[Array, " max_num_splits"]
        The substation ids before the reset
    action : Int[Array, " max_num_splits"]
        The branch/inj topology action before the reset
    split_idx : Int[Array, " "]
        The index of the substation to reset
    substation_unsplit_prob : float
        The probability to reset the substation to the unsplit state

    Returns
    -------
    Int[Array, " max_num_splits"]
        The substation ids after the reset. If a reset occurred, the substation id at split_idx
        will be set to int_max
    Int[Array, " max_num_splits"]
        The topology action after the reset. If a reset occurred, the topology at split_idx
        will be set to int_max
    """
    # There is a certain probability that we reset the substation to the unsplit state
    reset_mask = jax.random.bernoulli(random_key, p=substation_unsplit_prob, shape=(1,))[0]

    sub_ids = sub_ids.at[split_idx].set(
        jnp.where(reset_mask, int_max(), sub_ids[split_idx]),
        mode="drop",
    )
    action = action.at[split_idx].set(
        jnp.where(reset_mask, int_max(), action[split_idx]),
        mode="drop",
    )

    return sub_ids, action
