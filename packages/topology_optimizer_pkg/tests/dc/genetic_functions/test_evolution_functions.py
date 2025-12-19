# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import time
from functools import partial

import jax
from jax import numpy as jnp
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    Genotype,
    crossover,
    crossover_unbatched,
    deduplicate_genotypes,
    empty_repertoire,
    mutate,
    mutate_disconnections,
    mutate_sub,
)

from packages.topology_optimizer_pkg.tests.dc.test_main import assert_topology


def test_mutate_disconnection(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None
    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    n_pst = 3
    batch_size = 16

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_pst)
    assert jnp.all(topologies.disconnections == int_max())

    key = jax.random.PRNGKey(0)
    # Test with
    outages = mutate_disconnections(
        random_key=key,
        disconnections=topologies.disconnections[0],
        n_disconnectable_branches=n_disconnectable_branches,
        disconnect_prob=1.0,
        reconnect_prob=0.0,
    )
    assert sum(outages != int_max()) == 1, "One disconnection should have changed"
    # test with both probalities = 1
    outages = mutate_disconnections(
        random_key=key,
        disconnections=topologies.disconnections[0],
        n_disconnectable_branches=n_disconnectable_branches,
        disconnect_prob=1.0,
        reconnect_prob=1.0,
    )
    assert all(outages == int_max()), "Disconnections changes although dis- and reconnection both had 100 % probability"

    # Test with identical probabilities
    for i in range(100):
        test_key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=test_key,
            disconnections=topologies.disconnections[0],
            n_disconnectable_branches=n_disconnectable_branches,
            disconnect_prob=0.1,
            reconnect_prob=0.1,
        )
        if any(outages != int_max()):
            break
    else:
        assert False, "No disconnection mutation appeared in 100 runs"
    for i in range(100):
        test_key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key,
            disconnections=topologies.disconnections[0],
            n_disconnectable_branches=n_disconnectable_branches,
            disconnect_prob=0.0,
            reconnect_prob=0.0,
        )
        assert all(outages == int_max()), "Mutation changed although the probability was zero"


def test_mutate_disconnection_multi(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None
    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    n_pst = 3
    batch_size = 16

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_pst)
    assert jnp.all(topologies.disconnections == int_max())
    key = jax.random.PRNGKey(0)
    # Test with
    outages = mutate_disconnections(
        random_key=key,
        disconnections=topologies.disconnections[0],
        n_disconnectable_branches=n_disconnectable_branches,
        disconnect_prob=1.0,
        reconnect_prob=0.0,
    )
    assert sum(outages != int_max()) == 1, "One disconnection should have changed"
    for i in range(1, 100):
        key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key,
            disconnections=outages,
            n_disconnectable_branches=n_disconnectable_branches,
            disconnect_prob=1.0,
            reconnect_prob=0.0,
        )
        if all(outages != int_max()):
            break
    else:
        assert False, "In 100 disconnection mutations, it disconnected never more than 1"

    # Start reconnection
    outages = mutate_disconnections(
        random_key=key,
        disconnections=outages,
        n_disconnectable_branches=n_disconnectable_branches,
        disconnect_prob=0.0,
        reconnect_prob=1.0,
    )
    assert sum(outages != int_max()) == 1, "1 of the 2 disconnection should have been reconnected"

    for i in range(1, 100):
        key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key,
            disconnections=outages,
            n_disconnectable_branches=n_disconnectable_branches,
            disconnect_prob=0.0,
            reconnect_prob=1.0,
        )
        if all(outages == int_max()):
            break
    else:
        assert False, "In 100 reconnection mutations, it never reconnected the last one"


def test_mutate(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None
    disconnectable_branches = jnp.array([0, 1, 2, 3, 4])
    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    n_pst = 3
    batch_size = 16

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_pst)
    sub_ids = extract_sub_ids(topologies.action_index, action_set)
    assert jnp.all(sub_ids == int_max())

    key = jax.random.PRNGKey(0)

    sub_id, branch, _ = mutate_sub(
        sub_ids=sub_ids[0],
        action=topologies.action_index[0],
        random_key=key,
        substation_split_prob=1.0,
        substation_unsplit_prob=0.0,
        action_set=action_set,
    )

    assert jnp.sum(sub_id != int_max()) == 1
    assert jnp.sum(branch != int_max()) == 1

    outages = mutate_disconnections(
        random_key=key,
        disconnections=topologies.disconnections[0],
        n_disconnectable_branches=n_disconnectable_branches,
        disconnect_prob=1.0,
        reconnect_prob=0.0,
    )

    assert jnp.sum(outages != int_max) == 1

    # Sample a few runs and check if topologies stay valid
    for i in range(10):
        key = jax.random.PRNGKey(i)

        topologies, _ = mutate(
            topologies=topologies,
            random_key=key,
            substation_split_prob=0.2,
            substation_unsplit_prob=0.00001,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=5.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=1,
        )

        assert_topology(
            topologies,
            action_set,
            disconnectable_branches,
        )

    # Check sub_ids are sorted
    sub_ids = extract_sub_ids(topologies.action_index, action_set)
    assert jnp.all(jnp.diff(sub_ids, axis=1) >= 0)

    # Check that not all topologies are the same
    assert (
        jnp.any(jnp.diff(sub_ids, axis=0))
        or jnp.any(jnp.diff(topologies.action_index, axis=0))
        or jnp.any(jnp.diff(topologies.disconnections, axis=0))
    )


# nor implemented yet : xfail
# @pytest.mark.xfail(reason="Not implemented yet")
def test_mutate_multiple_tries(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None

    # disconnectable_branches = jnp.array([0, 1, 2, 3, 4])
    n_disconnectable_branches = 4

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 128
    n_pst = 3

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_pst)

    def count_duplicates(genotypes: Genotype) -> tuple[Genotype, jnp.ndarray, jnp.ndarray]:
        """Return unique genotypes, along with counts."""
        genotype_flat = jnp.concatenate(
            [
                genotypes.action_index,
                genotypes.disconnections,
            ],
            axis=1,
        )
        unique_flat_genotypes, unique_index, unique_counts = jnp.unique(
            genotype_flat, axis=0, return_index=True, return_counts=True
        )
        unique_genotypes = jax.tree_util.tree_map(lambda x: x[unique_index], genotypes)
        return unique_genotypes, unique_index, unique_counts

    percentage_uniquely_mutated_list = []
    percentage_repertoire_mutated_list = []
    time_taken_list = []
    warmup_time_taken_list = []

    for mutation_repetition in range(1, 4):
        key = jax.random.PRNGKey(0)

        start = time.time()
        warmup, _warmup = mutate(  # compile
            topologies=topologies,
            random_key=key,
            substation_split_prob=0.5,
            substation_unsplit_prob=0.5,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=5.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=mutation_repetition,
        )
        warmup_time_taken = time.time() - start

        start = time.time()
        topologies_mutated, _ = mutate(
            topologies=topologies,
            random_key=key,
            substation_split_prob=0.5,
            substation_unsplit_prob=0.5,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=5.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=mutation_repetition,
        )
        time_taken = time.time() - start

        unique_topologies, unique_index, unique_counts = count_duplicates(topologies_mutated)

        # When we mutate an empty repertoire, some topologies are untouched. We don't want to count them in duplicates
        # The actual duplicate topologies post mutation are the ones that are not the first topologies but are present multiple times in the repertoire
        # ie to get the number of uniquely mutated topologies, sum the counts of the duplicates that are not the first minus one per unique topology
        index_of_count_of_first_topology = jnp.where(unique_index == 0)[0].item()
        # Sum everything but the index_of_count_of_first_topology-th element, minus one per element
        num_duplicates_first_topology = unique_counts[index_of_count_of_first_topology] - 1
        num_duplicates_total = jnp.sum(unique_counts - 1)
        num_duplicates_mutated = num_duplicates_total - num_duplicates_first_topology

        num_total_mutated = len(unique_counts) - 1  # -1 because the first topology is not mutated

        percentage_uniquely_mutated = 1 - num_duplicates_mutated / num_total_mutated
        percentage_repertoire_mutated = 1 - num_duplicates_total / len(topologies.action_index)

        percentage_uniquely_mutated_list.append(percentage_uniquely_mutated.item())
        percentage_repertoire_mutated_list.append(percentage_repertoire_mutated.item())
        time_taken_list.append(time_taken)
        warmup_time_taken_list.append(warmup_time_taken)

    # Make sure we are improving with increasing number of mutation tries
    # ie strictly better or already 1.0
    for i in range(1, len(percentage_uniquely_mutated_list)):
        assert (
            percentage_uniquely_mutated_list[i] > percentage_uniquely_mutated_list[i - 1]
        ) or percentage_uniquely_mutated_list[i] == 1.0
        assert (
            percentage_repertoire_mutated_list[i] > percentage_repertoire_mutated_list[i - 1]
        ) or percentage_repertoire_mutated_list[i] == 1.0

    return time_taken_list, warmup_time_taken_list  # so it doesn't get optimized away


def test_crossover(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None
    disconnectable_branches = jnp.array([0, 1, 2, 3, 4])
    n_disconnectable_branches = 5

    max_num_splits = 3
    n_disconnections = 1
    n_psts = 3
    batch_size = 16

    # Randomly create some topologies
    topologies_a = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_psts,
    )
    topologies_b = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_psts,
    )

    for i in range(10):
        key = jax.random.PRNGKey(i)

        topologies_a, key = mutate(
            topologies=topologies_a,
            random_key=key,
            substation_split_prob=0.2,
            substation_unsplit_prob=0.00001,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=1.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=1,
        )

        topologies_b, key = mutate(
            topologies=topologies_b,
            random_key=key,
            substation_split_prob=0.2,
            substation_unsplit_prob=0.00001,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=1.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=1,
        )

        a_single: Genotype = jax.tree_util.tree_map(lambda x: x[0], topologies_a)
        b_single: Genotype = jax.tree_util.tree_map(lambda x: x[0], topologies_b)
        res, _ = crossover_unbatched(a_single, b_single, key, action_set, 0.0)
        # Except for the order, res and b_single should be the same
        assert jnp.all(jnp.isin(res.action_index, b_single.action_index))
        assert jnp.all(jnp.isin(res.disconnections, b_single.disconnections))
        res, _ = crossover_unbatched(a_single, b_single, key, action_set, 1.0)
        assert jnp.all(jnp.isin(res.action_index, a_single.action_index))
        assert jnp.all(jnp.isin(res.disconnections, a_single.disconnections))

        # Crossover
        topologies_c, _ = crossover(topologies_a, topologies_b, key, action_set, 0.5)

        assert_topology(
            topologies_a,
            action_set,
            disconnectable_branches,
        )

        assert_topology(
            topologies_b,
            action_set,
            disconnectable_branches,
        )

        assert_topology(
            topologies_c,
            action_set,
            disconnectable_branches,
        )


def test_deduplicate_genotypes_jitted(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)
    action_set = static_information.dynamic_information.action_set
    assert action_set is not None
    disconnectable_branches = jnp.array([0, 1, 2])
    n_disconnectable_branches = 3

    max_num_splits = 3
    max_num_disconnections = 2
    n_psts = 3
    batch_size = 16

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        max_num_disconnections,
        n_psts,
    )

    with jax.disable_jit():
        topologies, key = mutate(
            topologies=topologies,
            random_key=jax.random.PRNGKey(0),
            substation_split_prob=0.2,
            substation_unsplit_prob=0.00001,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=1.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=1,
        )

        # Case 1 : desired_size > batch_size
        deduplicated_topologies, indices = deduplicate_genotypes(
            topologies,
            desired_size=batch_size * 2,
        )

        assert deduplicated_topologies.action_index.shape[0] == batch_size * 2

        # Case 2 : desired_size < batch_size
        deduplicated_topologies, indices = deduplicate_genotypes(
            topologies,
            desired_size=batch_size // 2,
        )

        assert deduplicated_topologies.action_index.shape[0] == batch_size // 2

    # also test jittability
    partial_fun = partial(
        deduplicate_genotypes,
        desired_size=batch_size,
    )
    compiled_fun = jax.jit(partial_fun)
    compiled_fun(topologies)
