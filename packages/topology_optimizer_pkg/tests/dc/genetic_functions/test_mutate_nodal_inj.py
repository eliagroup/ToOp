# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_nodal_inj import mutate_psts


def test_mutate_psts() -> None:
    random_key = jax.random.PRNGKey(5345345)
    sigma = 5.0
    bernoulli_mean = 0.3
    # Set to 0 for now
    reset_probability = 0.0
    # Used across tests
    pst_starting_taps = jnp.array([15, 4, 3, 8, 5])

    pst_taps = jnp.array([2, 3, 8, 1, 8])
    pst_n_taps = jnp.array([5, 10, 10, 3, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.any(difference > 0)
    assert jnp.any(difference == 0)

    # Higher mean
    random_key = jax.random.PRNGKey(34536)
    bernoulli_mean = 0.7
    pst_taps = jnp.array([2, 3, 8, 1, 8])
    pst_n_taps = jnp.array([5, 10, 10, 3, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.any(difference > 0)

    # No change
    random_key = jax.random.PRNGKey(3534)
    sigma = 5.0
    bernoulli_mean = 0.0
    pst_taps = jnp.array([3, 3, 3, 1, 15])
    pst_n_taps = jnp.array([5, 10, 10, 3, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.all(difference == 0)

    # Out of range and should be clipped, no mutation
    random_key = jax.random.PRNGKey(345)
    sigma = 1.0
    bernoulli_mean = 0.0
    pst_taps = jnp.array([-5, 12, -8, 5, 28])
    pst_n_taps = jnp.array([5, 10, 10, 3, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(jnp.array([0, 9, 0, 2, 19]) == mutated_pst_taps)


def test_high_sigma_all_psts_change() -> None:
    # High sigma, different shape
    random_key = jax.random.PRNGKey(5345)
    sigma = 50.0
    bernoulli_mean = 0.2
    reset_probability = 0.0

    pst_taps = jnp.array([-5, 12, 8, 5, 8, 2, 2, 2])
    pst_n_taps = jnp.array([5, 10, 10, 3, 20, 10, 10, 10])
    pst_starting_taps = jnp.array([15, 4, 3, 8, 5, 8, 6, 7])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.any(difference > 0)
    assert jnp.any(difference == 0)

    # All change
    random_key = jax.random.PRNGKey(353)
    sigma = 50.0
    bernoulli_mean = 1.0
    pst_taps = jnp.array([1, -5, 12, 8, 5, 8])
    pst_n_taps = jnp.array([20, 5, 10, 10, 3, 20])
    pst_starting_taps = jnp.array([15, 4, 3, 8, 5, 8])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.all(difference > 0)


def test_resetting_psts() -> None:
    # Test resetting all PSTs
    random_key = jax.random.PRNGKey(435345)
    sigma = 5.0
    bernoulli_mean = 1.0
    # Set to 1.0 to reset all
    reset_probability = 1.0
    pst_taps = jnp.array([1, 1, 1, 1, 1, 1])
    pst_starting_taps = jnp.array([15, 4, 3, 8, 5, 8])
    pst_n_taps = jnp.array([20, 5, 10, 10, 8, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.all(mutated_pst_taps == pst_starting_taps)

    # Test resetting some PSTs
    random_key = jax.random.PRNGKey(6364)
    sigma = 5.0
    bernoulli_mean = 0.5
    # Set to 1.0 to reset all
    reset_probability = 0.3
    pst_taps = jnp.array([1, 1, 1, 1, 1, 1])
    pst_starting_taps = jnp.array([15, 4, 3, 8, 5, 8])
    pst_n_taps = jnp.array([20, 5, 10, 10, 3, 20])

    mutated_pst_taps = mutate_psts(
        random_key=random_key,
        pst_taps=pst_taps,
        pst_n_taps=pst_n_taps,
        pst_starting_taps=pst_starting_taps,
        pst_mutation_sigma=sigma,
        pst_mutation_probability=bernoulli_mean,
        pst_reset_probability=reset_probability,
    )

    # Assert boundaries
    assert jnp.all(0 <= mutated_pst_taps)
    assert jnp.all(mutated_pst_taps < pst_n_taps)
    # Assert some PSTs are adjusted but not all
    difference = jnp.abs(pst_taps - mutated_pst_taps)
    assert jnp.any(difference > 0)
    assert jnp.any(difference == 0)
    assert jnp.any(mutated_pst_taps == pst_starting_taps)
