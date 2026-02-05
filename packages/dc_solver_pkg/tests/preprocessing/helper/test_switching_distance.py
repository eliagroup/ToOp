# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
from toop_engine_dc_solver.preprocess.helpers.switching_distance import (
    hamming_distance,
    per_station_switching_distance,
)


def test_hamming_distance() -> None:
    asset_switching_table = jnp.array(
        [
            [True, False, True, False, True],
            [False, False, False, True, True],
            [False, False, True, False, True],
            [False, True, False, False, True],
        ]
    )
    target = jnp.array([True, False, True, True, True])
    assert jnp.array_equal(hamming_distance(asset_switching_table, target), jnp.array([1, 2, 2, 4]))
    target = ~target
    assert jnp.array_equal(hamming_distance(asset_switching_table, target), jnp.array([4, 3, 3, 1]))

    coupler_states = jnp.array(
        [
            [True, True, True],
            [False, False, True],
            [True, False, False],
            [True, False, True],
        ]
    )
    current_coupler_state = jnp.array([False, False, False])
    assert jnp.array_equal(hamming_distance(coupler_states, current_coupler_state), jnp.array([3, 1, 1, 2]))


def test_station_switching_distance() -> None:
    asset_switching_table = jnp.array(
        [
            [True, False, True, False, True],
            [False, False, False, True, True],
            [False, False, True, False, True],
            [False, True, False, False, True],
        ]
    )
    asset_switching_table = jnp.stack([asset_switching_table, ~asset_switching_table], axis=0)
    asset_switching_table = jnp.transpose(asset_switching_table, (1, 0, 2))
    coupler_states = jnp.array(
        [
            [True, True, True],
            [False, False, True],
            [True, False, False],
            [True, False, True],
        ]
    )
    current_coupler_state = jnp.array([False, False, False])
    target = jnp.array([True, False, True, True, True])

    best_config, invert, reassignment, coupler = per_station_switching_distance(
        current_coupler_state=current_coupler_state,
        separation_set=asset_switching_table,
        coupler_states=jnp.zeros_like(coupler_states),  # Only look at reconfigurations
        target_configuration=target,
    )

    # The first and the last configuration have the same hamming distance (1), so it shall select
    # the non-inverted one (the last one)
    assert best_config.item() == 3
    assert invert.item() is False
    # One reassignment is needed on every busbar
    assert reassignment.item() == 2
    assert coupler.item() == 0
