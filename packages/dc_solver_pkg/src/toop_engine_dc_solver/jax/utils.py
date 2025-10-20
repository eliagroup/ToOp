"""Utility functions and classes for JAX-based computations.

This module provides a collection of utility functions and classes designed to
enhance the usability and performance of JAX-based computations. These utilities
include operations for handling binary representations, top-k computations, and
masked dot products, as well as a wrapper class for making arrays hashable.

As described in https://github.com/google/jax/issues/4572#issuecomment-1235458797 this is a class
to avoid recompilation on unchanging data.
"""

import chex
import jax
from beartype.typing import Any, Generic, Optional, TypeVar
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int, Shaped


def action_index_to_binary_form(
    action_index: Int[Array, " "],
    max_degree: int,
) -> Bool[Array, " max_degree"]:
    """Get the binary representation of the action index

    You can vmap this to work with any batch size

    Parameters
    ----------
    action_index : Int[Array, " "]
        The action index to convert
    max_degree : int
        The maximum degree to expect in action_index, i.e. all action index are in the range of 0 to 2**max_degree

    Returns
    -------
    Bool[Array, " max_degree"]
        The binary representation of the action index with little endian bitorder (i.e. the leftmost bit is the least
        significant)
    """
    assert action_index.ndim == 0, "Only scalar action indices are supported"
    return jnp.unpackbits(action_index[None].view("uint8"), bitorder="little")[:max_degree].astype(bool)


T = TypeVar("T")  # Declare type variable


class HashableArrayWrapper(Generic[T]):
    """A wrapper for numpy/jax arrays that makes them hashable.

    Can be used to pass to jax's jit module as static arguments.
    """

    def __init__(self, val: T) -> None:
        self.val = val

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the wrapped array"""
        return self.val.shape

    def __len__(self) -> int:
        """Get length of the wrapped array"""
        return len(self.val)

    # def __getattribute__(self, prop):
    #     if prop in ["val", "__hash__", "__eq__"]:
    #         return super().__getattribute__(prop)
    #     return getattr(self.val, prop)

    def __hash__(self) -> int:
        """Hash the raw bytes of the array"""
        return hash(self.val.tobytes())

    def __eq__(self, other: object) -> bool:
        """Check equality between two HashableArrayWrapper objects.

        This method overrides the equality operator (`==`) to compare two
        HashableArrayWrapper instances. It ensures that the objects are of
        the same type and compares their hash values to determine equality.

        Parameters
        ----------
        other : object
            The object to compare with the current instance.
            It should be an instance of HashableArrayWrapper.

        Returns
        -------
        bool
            True if the hash values of both objects are equal, False otherwise.

        Raises
        ------
            NotImplemented: If the other object is not an instance of
            HashableArrayWrapper.
        """
        if not isinstance(other, HashableArrayWrapper):
            return NotImplemented
        return self.__hash__() == other.__hash__()


def argmax_top_k(data: Float[Array, " ... acc"], k: int) -> tuple[Float[Array, " ... k"], Int[Array, " ... k"]]:
    """Top k implementation built with argmax.

    Faster for smaller k.

    Parameters
    ----------
    data : Float[Array, " ... acc"]
        The data to find the top k of
    k : int
        The number of top elements to find

    Returns
    -------
    tuple[Float[Array, " ... k"], Int[Array, " ... k"]]
        The values and indices of the top k elements
    """
    rank = len(data.shape)
    if rank == 1:
        val, idx = _argmax_top_k(data[None], k)
        return val[0], idx[0]
    if rank == 2:
        return _argmax_top_k(data, k)
    # Rank > 2
    val, idx = _argmax_top_k(data.reshape((-1, data.shape[-1])), k)
    return val.reshape(data.shape[:-1] + (k,)), idx.reshape(data.shape[:-1] + (k,))


def _argmax_top_k(data: Float[Array, " batch acc"], k: int) -> tuple[Float[Array, " batch k"], Int[Array, " batch k"]]:
    """Top k implementation built with argmax.

    Faster for smaller k. See https://github.com/google/jax/issues/9940
    """

    def top_1(
        data: Float[Array, " batch acc"],
    ) -> tuple[Float[Array, " batch acc"], Float[Array, " batch"], Int[Array, " batch"]]:
        indice = jnp.argmax(data, axis=-1)
        value = jax.vmap(lambda x, y: x[y])(data, indice)
        data = jax.vmap(lambda x, y: x.at[y].set(-jnp.inf))(data, indice)
        return data, value, indice

    def scannable_top_1(
        carry: Float[Array, " batch acc"],
        _unused: Any,  # noqa: ANN401
    ) -> tuple[Float[Array, " batch acc"], tuple[Float[Array, " batch"], Int[Array, " batch"]]]:
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)

    chex.assert_rank(data, 2)

    data, (values, indices) = jax.lax.scan(scannable_top_1, data, (), k)

    return values.T, indices.T


def masked_vector_vector_dot_product(
    vector_a: Shaped[Array, " dim1"],
    mask_a: Bool[Array, " dim1"],
    vector_b: Shaped[Array, " dim2"],
    mask_b: Shaped[Array, " dim2"],
    upper_bound_nonzero_count: Optional[int] = None,
) -> Shaped[Array, " "]:
    """Get equivalent of vector_a[mask_a] @ vector_b[mask_b], but jit compatible.

    Assumes the same number of non-zero elements in mask_a and mask_b - if this is not the case,
    the result is undefined.

    Parameters
    ----------
    vector_a : Shaped[Array, " dim1"]
        The first vector
    mask_a : Bool[Array, " dim1"]
        The mask for the first vector
    vector_b : Shaped[Array, " dim2"]
        The second vector
    mask_b : Shaped[Array, " dim2"]
        The mask for the second vector
    upper_bound_nonzero_count : Optional[int], optional
        If you know the upper bound of non-zero elements in the masks, you can pass it here to
        speed up the computation, by default it assumes the length of the shortest vector to be
        the upper bound

    Returns
    -------
    Shaped[Array, " "]
        The dot product result
    """
    if upper_bound_nonzero_count is not None:
        max_size = upper_bound_nonzero_count
    else:
        max_size = min(len(vector_a), len(vector_b))
    assert max_size < jnp.iinfo(jnp.int32).max

    # We rely on out-of-bound indexing to fill the vector with zeros
    nonzero_mask_a = jnp.nonzero(mask_a, size=max_size, fill_value=jnp.iinfo(jnp.int32).max)[0]
    nonzero_mask_b = jnp.nonzero(mask_b, size=max_size, fill_value=jnp.iinfo(jnp.int32).max)[0]
    return jnp.dot(
        vector_a.at[nonzero_mask_a].get(mode="fill", fill_value=0),
        vector_b.at[nonzero_mask_b].get(mode="fill", fill_value=0),
    )


def masked_vector_matrix_dot_product(
    vector: Shaped[Array, " dim1"],
    mask_vec: Bool[Array, " dim1"],
    matrix: Shaped[Array, " dim2 dim1"],
    mask_mat: Bool[Array, " dim2"],
    upper_bound_nonzero_count: Optional[int] = None,
) -> Shaped[Array, " dim1"]:
    """Get equivalent of vec[mask_vec] @ mat[mask_mat, :], but jit compatible

    Assumes the same number of non-zero elements in mask_a and mask_b - if this is not the case,
    the result is undefined.

    Parameters
    ----------
    vector : Shaped[Array, " dim1"]
        The vector
    mask_vec : Bool[Array, " dim1"]
        The mask for the vector
    matrix : Shaped[Array, " dim2 dim1"]
        The matrix
    mask_mat : Shaped[Array, " dim2"]
        The mask for the matrix
    upper_bound_nonzero_count : Optional[int], optional
        If you know the upper bound of non-zero elements in the masks, you can pass it here to
        speed up the computation, by default it assumes min(dim1, dim2) to be the upper bound

    Returns
    -------
    Shaped[Array, " dim1"]
        The dot product result
    """
    if upper_bound_nonzero_count is not None:
        max_size = upper_bound_nonzero_count
    else:
        max_size = min(len(vector), matrix.shape[0])
    assert max_size < jnp.iinfo(jnp.int32).max

    # We rely on out-of-bound indexing to fill the vector with zeros
    nonzero_mask_vec = jnp.nonzero(mask_vec, size=max_size, fill_value=jnp.iinfo(jnp.int32).max)[0]
    nonzero_mask_mat = jnp.nonzero(mask_mat, size=max_size, fill_value=jnp.iinfo(jnp.int32).max)[0]
    return jnp.dot(
        vector.at[nonzero_mask_vec].get(mode="fill", fill_value=0),
        matrix.at[nonzero_mask_mat, :].get(mode="fill", fill_value=0),
    )
