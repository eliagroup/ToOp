import pickle

import jax
import jax.numpy as jnp
import numpy as np
from toop_engine_dc_solver.jax.utils import (
    HashableArrayWrapper,
    action_index_to_binary_form,
    argmax_top_k,
    masked_vector_matrix_dot_product,
    masked_vector_vector_dot_product,
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


def test_masked_vector_vector_dot_product() -> None:
    vec_a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vec_b = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    mask_a = jnp.array([1, 0, 1, 0, 1], dtype=bool)
    mask_b = jnp.array([1, 0, 0, 1, 0, 0, 1], dtype=bool)
    ref = vec_a[mask_a] @ vec_b[mask_b]

    masked_vector_vector_dot_product_jit = jax.jit(masked_vector_vector_dot_product)
    assert jnp.isclose(masked_vector_vector_dot_product(vec_a, mask_a, vec_b, mask_b), ref)
    assert jnp.isclose(masked_vector_vector_dot_product_jit(vec_a, mask_a, vec_b, mask_b), ref)
    assert jnp.isclose(
        masked_vector_vector_dot_product(vec_a, mask_a, vec_b, mask_b, upper_bound_nonzero_count=4),
        ref,
    )


def test_masked_vector_matrix_dot_product() -> None:
    vec = jax.random.normal(jax.random.PRNGKey(0), (7,))
    mat = jax.random.normal(jax.random.PRNGKey(0), (5, 7))

    mask_vec = jnp.array([True, False, True, False, True, False, False])
    mask_mat = jnp.array([True, False, False, True, True])

    ref = vec[mask_vec] @ mat[mask_mat, :]

    assert jnp.allclose(masked_vector_matrix_dot_product(vec, mask_vec, mat, mask_mat), ref)
    assert jnp.allclose(jax.jit(masked_vector_matrix_dot_product)(vec, mask_vec, mat, mask_mat), ref)
    assert jnp.allclose(
        masked_vector_matrix_dot_product(vec, mask_vec, mat, mask_mat, upper_bound_nonzero_count=4),
        ref,
    )
