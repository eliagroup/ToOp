"""Provides unrolled dense solve routines for jax.

Inspired by
https://github.com/google/jax/issues/4258 and
https://gist.github.com/tbenthompson/faae311ec4e465b0ff47b4aabe0d56b2

This is useful for the MODF computation, where a lot of 2x2 equation systems need to be solved for
outaging trafo3ws.
"""

# pylint: disable=invalid-name

import jax
import jax.numpy as jnp
from beartype.typing import Union
from jaxtyping import Array, Bool, Float


def solve_and_check_det(
    a: Float[Array, " n n"], b: Union[Float[Array, " n"], Float[Array, " n k"]]
) -> tuple[Union[Float[Array, " n"], Float[Array, " n k"]], Bool[Array, " "]]:
    """Solve a system of linear equations and check the determinant.

    Parameters
    ----------
    a : Float[Array, " n n"]
        The matrix
    b : Union[Float[Array, " n"], Float[Array, " n k"]]
        The right hand side (multiple right hand sides are supported)

    Returns
    -------
    Union[Float[Array, " n"], Float[Array, " n k"]]
        The solution to the system (or multiple, if multiple right hand sides were passed
    Bool[Array, " "]
        Whether the system was solvable
    """
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == b.shape[0]

    if a.shape == (1, 1):
        return b / a, jnp.squeeze(a != 0)
    if a.shape == (2, 2):
        return solve2x2(a, b)
    if a.shape == (3, 3):
        return solve3x3(a, b)

    # We use explicit lu factor and lu_solve because this way we can cheaply check the determinant
    # and the success of the solve operation
    lower_upper, pivots = jax.scipy.linalg.lu_factor(a)
    determinant = jnp.prod(jnp.diag(lower_upper))
    success = jnp.all(jnp.abs(determinant) > 1e-10)
    solution = jax.scipy.linalg.lu_solve((lower_upper, pivots), b.astype(lower_upper.dtype))
    success = success & jnp.all(jnp.isfinite(solution))

    return solution, success


def solve2x2(
    a: Float[Array, " 2 2"], b: Union[Float[Array, " 2"], Float[Array, " 2 k"]]
) -> tuple[Union[Float[Array, " 2"], Float[Array, " 2 k"]], Bool[Array, " "]]:
    """Solve a 2x2 system of linear equations.

    Parameters
    ----------
    a : Float[Array, " 2 2"]
        The 2x2 matrix
    b : Union[Float[Array, " 2"], Float[Array, " 2 k"]]
        The right hand side (multiple right hand sides are supported)

    Returns
    -------
    Union[Float[Array, " 2"], Float[Array, " 2 k"]]
        The solution to the system (or multiple, if multiple right hand sides were passed
    Bool[Array, " "]
        Whether the system was solvable
    """
    assert a.shape == (2, 2)
    assert b.shape[0] == 2

    m1, m2 = a[0]
    m3, m4 = a[1]

    det = m1 * m4 - m2 * m3
    inv_det = 1.0 / det

    a_inv = jnp.array([[m4, -m2], [-m3, m1]]) * inv_det
    res = jnp.dot(a_inv, b)

    return res, jnp.abs(det) > 1e-10


def solve3x3(
    a: Float[Array, " 3 3"], b: Union[Float[Array, " 3"], Float[Array, " 3 k"]]
) -> tuple[Union[Float[Array, " 3"], Float[Array, " 3 k"]], Bool[Array, " "]]:
    """Solve a 3x3 system of linear equations.

    This is copied from https://gist.github.com/tbenthompson/faae311ec4e465b0ff47b4aabe0d56b2

    Parameters
    ----------
    a : Float[Array, " 3 3"]
        The 3x3 matrix
    b : Union[Float[Array, " 3"], Float[Array, " 3 k"]]
        The right hand side (multiple right hand sides are supported)

    Returns
    -------
    Union[Float[Array, " 3"], Float[Array, " 3 k"]]
        The solution to the system (or multiple, if multiple right hand sides were passed
    Bool[Array, " "]
        Whether the system was solvable
    """
    assert a.shape == (3, 3)
    assert b.shape[0] == 3

    m1, m2, m3 = a[0]
    m4, m5, m6 = a[1]
    m7, m8, m9 = a[2]
    # Cache 2x2 sub-determinants
    sub1 = m5 * m9 - m6 * m8
    sub2 = m3 * m8 - m2 * m9
    sub3 = m2 * m6 - m3 * m5

    det = m1 * sub1 + m4 * sub2 + m7 * sub3
    inv_det = 1.0 / det

    a_inv = (
        jnp.array(
            [
                [sub1, sub2, sub3],
                [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
            ]
        )
        * inv_det
    )

    res = jnp.dot(a_inv, b)

    return res, jnp.abs(det) > 1e-10
