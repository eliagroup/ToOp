# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation functions for the nodal injections in the genetic algorithm."""

from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.types import NodalInjOptimResults
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import NodalInjectionMutationConfig


def mutate_psts(
    random_key: PRNGKeyArray,
    pst_taps: Int[Array, " num_psts"],
    pst_n_taps: Int[Array, " num_psts"],
    pst_mutation_sigma: float | int,
    pst_mutation_probability: float = 0.2,
) -> Int[Array, " num_psts"]:
    """Mutate the PST taps of a single topology.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    pst_taps : Int[Array, " num_psts"]
        The PST tap positions before mutation
    pst_n_taps : Int[Array, " num_psts"]
        The number of taps for each PST. If a PST has N taps in this array, then it is assumed that all taps from
        0 to N-1 are valid tap positions. Output taps will be clipped to this range.
    pst_mutation_sigma : float | int
        The sigma to use for the normal distribution to sample the mutation from. The mutation will be sampled as an
        integer from a normal distribution with mean 0 and sigma pst_mutation_sigma.
    pst_mutation_probability: float
        The probability for an individual PST to be selected for mutation. 1.0 indicates that all PSTs will be mutated,
        0.0 indicates that no PSTs will be mutated. Default 0.2

    Returns
    -------
    Int[Array, " num_psts"]
        The mutated PST tap positions, clipped to the valid range of taps for each PST.
    """
    # Sample number of PSTs to adjust from a num_psts-dimensional uniform distribution
    key, split_key = jax.random.split(random_key, 2)
    # barrier = jax.random.uniform(key, shape=pst_taps.shape)
    pst_indices_to_mutate = jax.random.bernoulli(key=key, p=pst_mutation_probability, shape=pst_taps.shape)
    # psts_to_change =  barrier <= pst_mutation_probability
    num_psts_to_change = jnp.sum(pst_indices_to_mutate)

    mutation = jnp.zeros_like(pst_taps)
    # Sample mutations for PSTs that were selected to change independently from a normal distribution.
    mutation_samples = jax.random.normal(split_key, shape=(num_psts_to_change,)) * pst_mutation_sigma
    mutation = mutation.at[pst_indices_to_mutate].set(mutation_samples)
    mutation = jnp.round(mutation).astype(int)

    new_pst_taps = pst_taps + mutation
    new_pst_taps = jnp.clip(new_pst_taps, a_min=0, a_max=pst_n_taps - 1)
    return new_pst_taps


def mutate_nodal_injections(
    random_key: PRNGKeyArray,
    nodal_inj_info: Optional[NodalInjOptimResults],
    nodal_mutation_config: Optional[NodalInjectionMutationConfig],
) -> Optional[NodalInjOptimResults]:
    """Mutate the nodal injection optimization results, currently only the PST taps.

    Parameters
    ----------
    random_key : PRNGKeyArray
        The random key to use for the mutation
    nodal_inj_info : Optional[NodalInjOptimResults]
        The nodal injection optimization results before mutation. If None, no mutation is performed and None is returned.
    nodal_mutation_config : Optional[NodalInjectionMutationConfig]
        The configuration for the nodal injection mutation. If None, no mutation is performed

    Returns
    -------
    Optional[NodalInjOptimResults]
        The mutated nodal injection optimization results. If nodal_inj_info was None, returns None.
    """
    if nodal_inj_info is None:
        return None

    if nodal_mutation_config is None or nodal_mutation_config.pst_mutation_sigma <= 0:
        return nodal_inj_info

    batch_size = nodal_inj_info.pst_tap_idx.shape[0]
    n_timesteps = nodal_inj_info.pst_tap_idx.shape[1]
    random_key = jax.random.split(random_key, (batch_size, n_timesteps))

    # vmap to mutate the PST taps for each timestep + batch independently
    new_pst_taps = jax.vmap(
        jax.vmap(
            partial(
                mutate_psts,
                pst_n_taps=nodal_mutation_config.pst_n_taps,
                pst_mutation_sigma=nodal_mutation_config.pst_mutation_sigma,
                pst_mutation_probability=nodal_mutation_config.pst_mutation_probability,
            )
        )
    )(
        random_key=random_key,
        pst_taps=nodal_inj_info.pst_tap_idx.astype(int),
    )

    return NodalInjOptimResults(pst_tap_idx=new_pst_taps)
