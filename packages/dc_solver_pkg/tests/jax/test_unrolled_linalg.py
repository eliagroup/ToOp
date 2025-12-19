# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from toop_engine_dc_solver.jax.unrolled_linalg import solve2x2, solve3x3, solve_and_check_det


def test_solve_and_check_det():
    for i in range(1, 10):
        a = jax.random.uniform(jax.random.PRNGKey(i), (i, i), minval=-10.0, maxval=10.0)
        b = jax.random.uniform(jax.random.PRNGKey(i), (i,), minval=-10.0, maxval=10.0)
        x, succ = solve_and_check_det(a, b)
        assert succ.shape == ()

        ref = jnp.linalg.solve(a, b)
        succ_ref = jnp.abs(jnp.linalg.det(a)) > 1e-10

        if succ_ref:
            assert jnp.allclose(x, ref)
            assert succ
        else:
            assert not succ


def test_solve2x2():
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([5.0, 6.0])
    x, succ = solve2x2(a, b)
    assert jnp.allclose(jnp.dot(a, x), b)

    for i in range(100):
        key1, key2 = jax.random.split(jax.random.PRNGKey(i), 2)
        a = jax.random.uniform(key1, (2, 2), minval=-10.0, maxval=10.0)
        b = jax.random.uniform(key2, (2,), minval=-10.0, maxval=10.0)
        x, succ = solve2x2(a, b)

        ref = jnp.linalg.solve(a, b)
        succ_ref = jnp.abs(jnp.linalg.det(a)) > 1e-10

        assert jnp.array_equal(succ, succ_ref)
        assert ref.shape == x.shape
        if succ:
            assert jnp.allclose(x, ref)


def test_solve2x2_multi_rhs():
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]]).T
    x, succ = solve2x2(a, b)
    assert jnp.allclose(jnp.dot(a, x), b)

    for i in range(100):
        key1, key2 = jax.random.split(jax.random.PRNGKey(i), 2)
        a = jax.random.uniform(key1, (2, 2), minval=-10.0, maxval=10.0)
        b = jax.random.uniform(key2, (2, 10), minval=-10.0, maxval=10.0)
        x, succ = solve2x2(a, b)

        ref = jnp.linalg.solve(a, b)
        succ_ref = jnp.abs(jnp.linalg.det(a)) > 1e-10

        assert jnp.array_equal(succ, succ_ref)
        assert ref.shape == x.shape
        if succ:
            assert jnp.allclose(x, ref)


def test_solve3x3():
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    b = jnp.array([5.0, 6.0, 7.0])
    x, succ = solve3x3(a, b)
    assert jnp.allclose(jnp.dot(a, x), b)

    for i in range(100):
        key1, key2 = jax.random.split(jax.random.PRNGKey(i), 2)
        a = jax.random.uniform(key1, (3, 3), minval=-10.0, maxval=10.0)
        b = jax.random.uniform(key2, (3,), minval=-10.0, maxval=10.0)
        x, succ = solve3x3(a, b)

        ref = jnp.linalg.solve(a, b)
        succ_ref = jnp.abs(jnp.linalg.det(a)) > 1e-10

        assert jnp.array_equal(succ, succ_ref)
        assert ref.shape == x.shape
        if succ:
            assert jnp.allclose(x, ref)


def test_solve3x3_multi_rhs():
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    b = jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]).T
    x, succ = solve3x3(a, b)
    assert jnp.allclose(jnp.dot(a, x), b)

    for i in range(100):
        key1, key2 = jax.random.split(jax.random.PRNGKey(i), 2)
        a = jax.random.uniform(key1, (3, 3), minval=-10.0, maxval=10.0)
        b = jax.random.uniform(key2, (3, 10), minval=-10.0, maxval=10.0)
        x, succ = solve3x3(a, b)

        ref = jnp.linalg.solve(a, b)
        succ_ref = jnp.abs(jnp.linalg.det(a)) > 1e-10

        assert jnp.array_equal(succ, succ_ref)
        assert ref.shape == x.shape
        if succ:
            assert jnp.allclose(x, ref)
