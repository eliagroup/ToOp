# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from toop_engine_dc_solver.jax.types import int_max
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import create_random_topology


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
