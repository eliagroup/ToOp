# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import time
from dataclasses import replace
from functools import partial

import jax
from jax import numpy as jnp
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids
from toop_engine_dc_solver.jax.types import ActionSet, NodalInjOptimResults, int_max
from toop_engine_topology_optimizer.dc.genetic_functions.crossover import (
    crossover,
    crossover_unbatched,
)
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype, deduplicate_genotypes, empty_repertoire
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    DisconnectionMutationConfig,
    MutationConfig,
    NodalInjectionMutationConfig,
    SubstationMutationConfig,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import mutate
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_disconnections import mutate_disconnections
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_nodal_inj import mutate_nodal_injections
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate_substations import mutate_sub_splits

from packages.topology_optimizer_pkg.tests.dc.test_main import assert_topology


def test_mutate_disconnection(synthetic_action_set: ActionSet) -> None:
    n_rel_subs = 5

    action_set = synthetic_action_set
    assert action_set is not None
    n_disconnectable_branches = 5
    sub_ids_with_split = jnp.arange(n_rel_subs, dtype=int)

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 16
    n_timesteps = 1

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_timesteps, None)
    assert jnp.all(topologies.disconnections == int_max())

    key = jax.random.PRNGKey(0)
    # Test with

    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=1.0,
        remove_disconnection_prob=0.0,
        change_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )

    outages = mutate_disconnections(
        random_key=key,
        sub_ids=sub_ids_with_split,
        disconnections=topologies.disconnections[0],
        disconnection_mutation_config=disc_config,
    )
    assert sum(outages != int_max()) == 1, "One disconnection should have changed"
    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=0.0,
        remove_disconnection_prob=1.0,
        change_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    outages = mutate_disconnections(
        random_key=key,
        sub_ids=sub_ids_with_split,
        disconnections=topologies.disconnections[0],
        disconnection_mutation_config=disc_config,
    )
    assert all(outages == int_max()), "Disconnection wasnt reconnected, although probability was 100%"

    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=0.3,
        remove_disconnection_prob=0.3,
        change_disconnection_prob=0.3,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    for i in range(10):
        test_key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=test_key,
            sub_ids=sub_ids_with_split,
            disconnections=topologies.disconnections[0],
            disconnection_mutation_config=disc_config,
        )
        if any(outages != int_max()):
            break
    else:
        assert False, "No disconnection mutation appeared in 100 runs"

    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=0.0,
        remove_disconnection_prob=0.0,
        change_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    for i in range(10):
        test_key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key,
            sub_ids=sub_ids_with_split,
            disconnections=topologies.disconnections[0],
            disconnection_mutation_config=disc_config,
        )
        assert all(outages == int_max()), "Mutation changed although the probability was zero"


def test_mutate_disconnection_multi(synthetic_action_set: ActionSet) -> None:
    n_rel_subs = 5

    action_set = synthetic_action_set
    assert action_set is not None

    sub_ids_with_split = jnp.arange(n_rel_subs, dtype=int)

    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 16
    n_timesteps = 1

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_timesteps, None)
    assert jnp.all(topologies.disconnections == int_max())
    key = jax.random.PRNGKey(0)
    # Test with
    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=1.0,
        remove_disconnection_prob=0.0,
        change_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    outages = mutate_disconnections(
        random_key=key,
        sub_ids=sub_ids_with_split,
        disconnections=topologies.disconnections[0],
        disconnection_mutation_config=disc_config,
    )
    assert sum(outages != int_max()) == 1, "One disconnection should have changed"
    for i in range(1, 10):
        key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key, sub_ids=sub_ids_with_split, disconnections=outages, disconnection_mutation_config=disc_config
        )
        if all(outages != int_max()):
            break
    else:
        assert False, "In 100 disconnection mutations, it disconnected never more than 1"

    # Start reconnection
    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=0.0,
        remove_disconnection_prob=1.0,
        change_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    outages = mutate_disconnections(
        random_key=key, sub_ids=sub_ids_with_split, disconnections=outages, disconnection_mutation_config=disc_config
    )
    assert sum(outages == int_max()) == 1, "1 of the 2 disconnection should have been reconnected"

    for i in range(1, max_num_disconnections):
        key = jax.random.PRNGKey(i)
        outages = mutate_disconnections(
            random_key=key, sub_ids=sub_ids_with_split, disconnections=outages, disconnection_mutation_config=disc_config
        )
        if all(outages == int_max()):
            break
    else:
        assert False, "With 100 % possibility of reconnection, it should have reconnected both"


def test_mutate(synthetic_action_set: ActionSet) -> None:
    action_set = synthetic_action_set

    disconnectable_branches = jnp.array([0, 1, 2, 3, 4])
    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 16
    n_timesteps = 1
    substation_mutation_config = SubstationMutationConfig(
        n_rel_subs=action_set.n_actions_per_sub.shape[0],
        add_split_prob=1.0,
        change_split_prob=0.0,
        remove_split_prob=0.0,
        n_subs_mutated_lambda=2.0,
    )
    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_timesteps, None)
    sub_ids = extract_sub_ids(topologies.action_index, action_set)
    assert jnp.all(sub_ids == int_max())

    key = jax.random.PRNGKey(0)

    sub_id, branch, _ = mutate_sub_splits(
        sub_ids=sub_ids[0],
        action=topologies.action_index[0],
        random_key=key,
        sub_mutate_config=substation_mutation_config,
        action_set=action_set,
    )

    assert jnp.sum(sub_id != int_max()) == 1
    assert jnp.sum(branch != int_max()) == 1

    disc_config = DisconnectionMutationConfig(
        add_disconnection_prob=1.0,
        change_disconnection_prob=0.0,
        remove_disconnection_prob=0.0,
        n_disconnectable_branches=n_disconnectable_branches,
    )
    outages = mutate_disconnections(
        random_key=key,
        disconnections=topologies.disconnections[0],
        sub_ids=sub_ids[0],
        disconnection_mutation_config=disc_config,
    )

    assert jnp.sum(outages != int_max()) == 1

    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            add_split_prob=0.3,
            remove_split_prob=0.3,
            change_split_prob=0.3,
            n_rel_subs=action_set.n_actions_per_sub.shape[0],
            n_subs_mutated_lambda=2.0,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.3,
            remove_disconnection_prob=0.3,
            change_disconnection_prob=0.3,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )

    # Sample a few runs and check if topologies stay valid
    for i in range(10):
        key = jax.random.PRNGKey(i)

        topologies, _ = mutate(topologies=topologies, random_key=key, action_set=action_set, mutation_config=mutation_config)

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


def test_mutate_on_same_repository_with_different_keys(synthetic_action_set: ActionSet) -> None:
    # When the repository is not changing much, because no better topologies were found,
    # we want to make sure that the mutation can still create different topologies
    # with different random keys, to ensure exploration of the search space.
    n_rel_subs = 5

    action_set = synthetic_action_set

    n_disconnectable_branches = 5

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 16
    n_timesteps = 1

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_timesteps, None)

    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            add_split_prob=0.4,
            remove_split_prob=0.3,
            change_split_prob=0.3,
            n_rel_subs=action_set.n_actions_per_sub.shape[0],
            n_subs_mutated_lambda=2.0,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.4,
            remove_disconnection_prob=0.3,
            change_disconnection_prob=0.3,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )

    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    topologies1, _ = mutate(topologies=topologies, random_key=key1, action_set=action_set, mutation_config=mutation_config)

    topologies2, _ = mutate(topologies=topologies, random_key=key2, action_set=action_set, mutation_config=mutation_config)

    assert not (
        jnp.array_equal(topologies1.action_index, topologies2.action_index)
        or jnp.array_equal(topologies1.disconnections, topologies2.disconnections)
    ), "With different random keys, we should get different mutations"
    all_actions = [topologies1.action_index, topologies2.action_index]
    all_discos = [topologies1.disconnections, topologies2.disconnections]
    for i in range(2, 100):
        key = jax.random.PRNGKey(i)
        topologies_mutated, _ = mutate(
            topologies=topologies, random_key=key, action_set=action_set, mutation_config=mutation_config
        )
        all_actions.append(topologies_mutated.action_index)
        all_discos.append(topologies_mutated.disconnections)

    all_actions_stacked = jnp.concatenate(all_actions, axis=0)
    all_discos_stacked = jnp.concatenate(all_discos, axis=0)
    flat_genos = jnp.concatenate([all_actions_stacked, all_discos_stacked], axis=1)
    unique_genos = jnp.unique(flat_genos, axis=0)
    assert len(unique_genos) > len(flat_genos) * 0.8, (
        "With different random keys, we should get different mutations, but all mutations were the same"
    )


# nor implemented yet : xfail
# @pytest.mark.xfail(reason="Not implemented yet")
def test_mutate_multiple_tries(synthetic_action_set: ActionSet) -> None:
    n_rel_subs = 20

    action_set = synthetic_action_set

    n_disconnectable_branches = 4

    max_num_splits = 3
    max_num_disconnections = 2
    batch_size = 128
    n_timesteps = 1

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, max_num_disconnections, n_timesteps, None)

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

    time_taken_list = []
    warmup_time_taken_list = []

    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            add_split_prob=0.5,
            remove_split_prob=0.5,
            change_split_prob=0.0,
            n_rel_subs=action_set.n_actions_per_sub.shape[0],
            n_subs_mutated_lambda=5.0,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.5,
            remove_disconnection_prob=0.5,
            change_disconnection_prob=0.0,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    n_unique_indizes = []
    for mutation_repetition in range(1, 10):
        mutation_config = replace(mutation_config, mutation_repetition=mutation_repetition)
        key = jax.random.PRNGKey(0)

        start = time.time()
        warmup, _warmup = mutate(  # compile
            topologies=topologies,
            random_key=key,
            action_set=action_set,
            mutation_config=mutation_config,
        )
        warmup_time_taken = time.time() - start

        start = time.time()
        topologies_mutated, _ = mutate(
            topologies=topologies,
            random_key=key,
            mutation_config=mutation_config,
            action_set=action_set,
        )
        time_taken = time.time() - start

        unique_topologies, unique_index, unique_count = count_duplicates(topologies_mutated)

        # When we mutate an empty repertoire, some topologies are untouched. We don't want to count them in duplicates
        # The actual duplicate topologies post mutation are the ones that are not the first topologies but are present multiple times in the repertoire
        # ie to get the number of uniquely mutated topologies, sum the counts of the duplicates that are not the first minus one per unique topology
        # index_of_count_of_first_topology = jnp.where(unique_index == 0)[0].item()
        # # Sum everything but the index_of_count_of_first_topology-th element, minus one per element
        # num_duplicates_first_topology = unique_counts[index_of_count_of_first_topology] - 1
        # num_duplicates_total = jnp.sum(unique_counts - 1)
        # num_duplicates_mutated = num_duplicates_total - num_duplicates_first_topology

        # num_total_mutated = len(unique_counts) - 1  # -1 because the first topology is not mutated

        # percentage_uniquely_mutated = 1 - num_duplicates_mutated / num_total_mutated
        # # percentage_repertoire_mutated = 1 - num_duplicates_total / len(topologies.action_index)

        # percentage_uniquely_mutated_list.append(percentage_uniquely_mutated.item())
        # percentage_repertoire_mutated_list.append(percentage_repertoire_mutated.item())
        # time_taken_list.append(time_taken)
        # warmup_time_taken_list.append(warmup_time_taken)
        n_unique_indizes.append(len(unique_index))
    for i in range(1, len(n_unique_indizes)):
        assert n_unique_indizes[i] >= n_unique_indizes[i - 1], (
            "With more mutation repetitions, we should have more unique topologies"
        )
    # Make sure we are improving with increasing number of mutation tries
    # ie strictly better or already 1.0
    # for i in range(1, len(percentage_uniquely_mutated_list)):
    #     assert (
    #         percentage_uniquely_mutated_list[i] > percentage_uniquely_mutated_list[i - 1]
    #     ) or percentage_uniquely_mutated_list[i] == 1.0
    #     assert (
    #         percentage_repertoire_mutated_list[i] > percentage_repertoire_mutated_list[i - 1]
    #     ) or percentage_repertoire_mutated_list[i] == 1.0

    return time_taken_list, warmup_time_taken_list  # so it doesn't get optimized away


def test_crossover(synthetic_action_set: ActionSet) -> None:
    n_rel_subs = 5

    action_set = synthetic_action_set
    disconnectable_branches = jnp.array([0, 1, 2, 3, 4])
    n_disconnectable_branches = 5

    max_num_splits = 3
    n_disconnections = 1
    batch_size = 16
    n_timesteps = 1

    # Randomly create some topologies
    topologies_a = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
        None,
    )
    topologies_b = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
        None,
    )
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            add_split_prob=0.5,
            remove_split_prob=0.5,
            change_split_prob=0.0,
            n_rel_subs=action_set.n_actions_per_sub.shape[0],
            n_subs_mutated_lambda=5.0,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.5,
            remove_disconnection_prob=0.5,
            change_disconnection_prob=0.0,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    for i in range(10):
        key = jax.random.PRNGKey(i)

        topologies_a, key = mutate(
            topologies=topologies_a,
            random_key=key,
            mutation_config=mutation_config,
            action_set=action_set,
        )

        topologies_b, key = mutate(
            topologies=topologies_b,
            random_key=key,
            mutation_config=mutation_config,
            action_set=action_set,
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


def test_deduplicate_genotypes_jitted(synthetic_action_set: ActionSet) -> None:
    action_set = synthetic_action_set
    n_disconnectable_branches = 3

    max_num_splits = 3
    max_num_disconnections = 2
    n_psts = 3
    batch_size = 16
    n_timesteps = 1

    # Initialize PST setpoints
    starting_taps = jnp.zeros(n_psts, dtype=int)
    pst_n_taps = jnp.array([35, 35, 20], dtype=int)

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        max_num_disconnections,
        n_timesteps,
        starting_taps,
    )
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            add_split_prob=0.5,
            remove_split_prob=0.5,
            change_split_prob=0.0,
            n_rel_subs=action_set.n_actions_per_sub.shape[0],
            n_subs_mutated_lambda=5.0,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.5,
            remove_disconnection_prob=0.5,
            change_disconnection_prob=0.0,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    with jax.disable_jit():
        topologies, key = mutate(
            topologies=topologies,
            random_key=jax.random.PRNGKey(0),
            action_set=action_set,
            mutation_config=mutation_config,
        )

        # Case 1 : desired_size > batch_size
        deduplicated_topologies, indices = deduplicate_genotypes(
            topologies,
            desired_size=batch_size * 2,
        )

        assert deduplicated_topologies.action_index.shape[0] == batch_size * 2
        assert deduplicated_topologies.nodal_injections_optimized is not None
        assert deduplicated_topologies.nodal_injections_optimized.pst_tap_idx.shape[0] == batch_size * 2

        # Case 2 : desired_size < batch_size
        deduplicated_topologies, indices = deduplicate_genotypes(
            topologies,
            desired_size=batch_size // 2,
        )

        assert deduplicated_topologies.action_index.shape[0] == batch_size // 2
        assert deduplicated_topologies.nodal_injections_optimized is not None
        assert deduplicated_topologies.nodal_injections_optimized.pst_tap_idx.shape[0] == batch_size // 2

    # also test jittability
    partial_fun = partial(
        deduplicate_genotypes,
        desired_size=batch_size,
    )
    compiled_fun = jax.jit(partial_fun)
    compiled_fun(topologies)


def test_mutate_nodal_injections() -> None:
    batch_size = 4
    n_timesteps = 10
    n_taps = jnp.array([35, 35, 35, 20, 20])
    current_tap = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(batch_size, n_timesteps, 5), minval=0, maxval=n_taps
    ).astype(int)
    nodal_inj_info = NodalInjOptimResults(pst_tap_idx=current_tap)
    nodal_mutation_config = NodalInjectionMutationConfig(
        pst_mutation_sigma=5.0, pst_mutation_probability=0.2, pst_n_taps=n_taps
    )
    res = mutate_nodal_injections(
        random_key=jax.random.PRNGKey(0), nodal_inj_info=nodal_inj_info, nodal_mutation_config=nodal_mutation_config
    )

    assert res.pst_tap_idx.shape == (batch_size, n_timesteps, 5)
    assert jnp.all(res.pst_tap_idx >= 0)
    assert jnp.all(res.pst_tap_idx[:, :, :3] < 35)
    assert jnp.all(res.pst_tap_idx[:, :, 3:] < 20)
    assert not jnp.array_equal(res.pst_tap_idx, current_tap), "PST taps should have mutated"

    assert (
        mutate_nodal_injections(
            random_key=jax.random.PRNGKey(0), nodal_inj_info=None, nodal_mutation_config=nodal_mutation_config
        )
        is None
    ), "If nodal_inj_info is None, the result should also be None"

    nodal_mutation_config_zero = replace(nodal_mutation_config, pst_mutation_sigma=0.0)
    res_no_mutation = mutate_nodal_injections(
        random_key=jax.random.PRNGKey(0), nodal_inj_info=nodal_inj_info, nodal_mutation_config=nodal_mutation_config_zero
    )
    assert jnp.array_equal(res_no_mutation.pst_tap_idx, current_tap), "The PST taps should not have mutated"

    res_no_mutation = mutate_nodal_injections(
        random_key=jax.random.PRNGKey(0), nodal_inj_info=nodal_inj_info, nodal_mutation_config=None
    )
    assert jnp.array_equal(res_no_mutation.pst_tap_idx, current_tap), "The PST taps should not have mutated"
