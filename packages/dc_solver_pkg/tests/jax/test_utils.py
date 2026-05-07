# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from toop_engine_dc_solver.jax.utils import (
    HashableArrayWrapper,
    action_index_to_binary_form,
    argmax_top_k,
)


def test_action_index_to_binary_form() -> None:
    # Test with action_index 0
    action_index = jnp.array(0, dtype=jnp.int32)
    max_degree = 4
    binary_form = action_index_to_binary_form(action_index, max_degree)
    expected_binary_form = jnp.array([0, 0, 0, 0], dtype=bool)
    assert jnp.array_equal(binary_form, expected_binary_form)
    assert binary_form.dtype == jnp.bool_

    # Test with action_index 1
    action_index = jnp.array(15, dtype=jnp.int32)
    max_degree = 4
    binary_form = action_index_to_binary_form(action_index, max_degree)
    expected_binary_form = jnp.array([1, 1, 1, 1], dtype=bool)
    assert jnp.array_equal(binary_form, expected_binary_form)

    # Test with action_index larger than 8 bit
    action_index = jnp.array(513, dtype=jnp.int32)
    max_degree = 10
    binary_form = action_index_to_binary_form(action_index, max_degree)
    expected_binary_form = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    assert jnp.array_equal(binary_form, expected_binary_form)


def test_pickle_hashable_array_wrapper() -> None:
    data = np.random.normal(size=(10, 10))
    wrapped = HashableArrayWrapper(data)

    loaded = pickle.loads(pickle.dumps(wrapped))
    assert isinstance(loaded, HashableArrayWrapper)
    assert np.array_equal(loaded.val, wrapped.val)
    assert loaded == wrapped


def test_argmax_top_k() -> None:
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (100, 100))
    k = 5

    ref_val, ref_idx = jax.lax.top_k(data, k)
    val, idx = argmax_top_k(data, k)

    assert jnp.array_equal(jnp.sort(ref_val), jnp.sort(val))
    assert jnp.array_equal(jnp.sort(ref_idx), jnp.sort(idx))

    data = jax.random.normal(key, (100,))

    ref_val, ref_idx = jax.lax.top_k(data, k)
    val, idx = argmax_top_k(data, k)

    assert jnp.array_equal(jnp.sort(ref_val), jnp.sort(val))
    assert jnp.array_equal(jnp.sort(ref_idx), jnp.sort(idx))

    data = jax.random.normal(key, (10, 10, 10))

    ref_val, ref_idx = jax.lax.top_k(data, k)
    val, idx = argmax_top_k(data, k)

    assert jnp.array_equal(jnp.sort(ref_val), jnp.sort(val))
    assert jnp.array_equal(jnp.sort(ref_idx), jnp.sort(idx))
