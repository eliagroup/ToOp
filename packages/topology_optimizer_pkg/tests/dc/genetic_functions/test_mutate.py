# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import pytest
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import empty_repertoire
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    DisconnectionMutationConfig,
    MutationConfig,
    NodalInjectionMutationConfig,
    SubstationMutationConfig,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import create_random_topology, mutate


def test_create_random_topology_shapes(synthetic_action_set):
    random_key = jax.random.PRNGKey(42)
    max_num_splits = 4
    max_num_disconnections = 3
    sub_ids = jnp.zeros((max_num_splits,), dtype=int)
    disconnections = jnp.zeros((max_num_disconnections,), dtype=int)
    n_rel_subs = 5
    n_disconnectable_branches = 6

    sub_ids_out, action_out, disconnections_out, random_key_out = create_random_topology(
        random_key,
        sub_ids,
        disconnections,
        synthetic_action_set,
        n_rel_subs,
        n_disconnectable_branches,
    )

    assert sub_ids_out.shape == (max_num_splits,)
    assert action_out.shape == (max_num_splits,)
    assert disconnections_out.shape == (max_num_disconnections,)
    assert isinstance(random_key_out, jax.Array)


def test_create_random_topology_values(synthetic_action_set):
    random_key = jax.random.PRNGKey(123)
    max_num_splits = 2
    max_num_disconnections = 2
    sub_ids = jnp.zeros((max_num_splits,), dtype=int)
    disconnections = jnp.zeros((max_num_disconnections,), dtype=int)
    n_rel_subs = 3
    n_disconnectable_branches = 4

    sub_ids_out, action_out, disconnections_out, _ = create_random_topology(
        random_key,
        sub_ids,
        disconnections,
        synthetic_action_set,
        n_rel_subs,
        n_disconnectable_branches,
    )

    # Check that sub_ids_out contains either int_max or values in [0, n_rel_subs)
    for val in sub_ids_out:
        assert val == int_max() or (0 <= val < n_rel_subs)

    # Check that disconnections_out contains either int_max or values in [0, n_disconnectable_branches)
    for val in disconnections_out:
        assert val == int_max() or (0 <= val < n_disconnectable_branches)


def _mutation_config_without_split_mutations(n_disconnectable_branches: int, n_rel_subs: int, pst_mode: str):
    """Build a config that keeps the substations unsplit, so only disconnections and PSTs are mutated.

    pst_mode is one of "none" (no PSTs in the genome), "mutable" (the taps are actually mutated) and
    "never_selected" (PSTs exist, but no PST can ever be picked for a mutation or a reset).
    """
    return MutationConfig(
        mutation_repetition=1,
        random_topo_prob=0.0,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=n_rel_subs,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.1,
            change_disconnection_prob=0.1,
            remove_disconnection_prob=0.1,
            n_disconnectable_branches=n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.5 if pst_mode == "mutable" else 0.0,
            pst_reset_probability=0.0,
            pst_n_taps=jnp.array([10, 10]),
            pst_start_tap_idx=jnp.array([5, 5]),
        )
        if pst_mode != "none"
        else None,
    )


@pytest.mark.parametrize("pst_mode", ["none", "mutable", "never_selected"])
def test_mutate_empty_topology_only_stays_empty_with_mutable_psts(synthetic_action_set, pst_mode):
    # An empty topology (no splits, no disconnections) is the unchanged base topology unless the PST
    # taps are actually mutated as well. Only in that case the mutation may keep it empty.
    batch_size = 32
    n_timesteps = 1
    n_disconnectable_branches = 5
    n_rel_subs = synthetic_action_set.n_actions_per_sub.shape[0]

    topologies = empty_repertoire(
        batch_size,
        max_num_splits=3,
        max_num_disconnections=2,
        n_timesteps=n_timesteps,
        starting_taps=jnp.array([5, 5]) if pst_mode != "none" else None,
    )
    mutation_config = _mutation_config_without_split_mutations(n_disconnectable_branches, n_rel_subs, pst_mode)

    mutated, _ = mutate(
        topologies=topologies,
        random_key=jax.random.PRNGKey(0),
        mutation_config=mutation_config,
        action_set=synthetic_action_set,
    )

    # The substations are never split with this config, so an empty genome is one without disconnections
    assert jnp.all(mutated.action_index == int_max())
    is_empty = jnp.all(mutated.disconnections == int_max(), axis=1)
    if pst_mode == "mutable":
        assert jnp.any(is_empty), "With mutable PSTs, the mutation should be able to keep the empty topology"
    else:
        assert not jnp.any(is_empty), "Without mutable PSTs, the mutation should always add a disconnection"
