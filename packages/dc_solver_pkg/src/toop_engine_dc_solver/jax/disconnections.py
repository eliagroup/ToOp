# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines for applying disconnections of branches to the PTDF matrix.

It reuses the LODF formulation from the contingency analysis module, but applies it
to the PTDF matrix instead of computing the flows with it.
"""

import math
from functools import partial

import jax
import networkx as nx
import numpy as np
from beartype.typing import Optional, Union
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.lodf import calc_lodf
from toop_engine_dc_solver.jax.multi_outages import apply_modf_matrix, build_modf_matrix, update_ptdf_with_modf
from toop_engine_dc_solver.jax.types import DisconnectionResults, MODFMatrix, int_max


def update_from_to_nodes_after_disconnections(
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    disconnections: Int[Array, " n_disconnections"],
) -> tuple[Int[Array, " n_branches"], Int[Array, " n_branches"]]:
    """Update the from_nodes and to_nodes after disconnections

    Parameters
    ----------
    from_nodes : Int[Array, " n_branches"]
        The from_nodes for each branch
    to_nodes : Int[Array, " n_branches"]
        The to_nodes for each branch
    disconnections : Int[Array, " n_disconnections"]
        The disconnections to perform, can be a single value or an array of values

    Returns
    -------
    tuple[Int[Array, " n_branches"], Int[Array, " n_branches"]]
        The updated from_nodes and to_nodes
    """
    if isinstance(disconnections, int):
        disconnections = jnp.array([disconnections])
    if disconnections.size == 0:
        return from_nodes, to_nodes
    from_nodes = from_nodes.at[disconnections].set(jnp.iinfo(jnp.int32).max)
    to_nodes = to_nodes.at[disconnections].set(jnp.iinfo(jnp.int32).max)

    return from_nodes, to_nodes


def apply_single_disconnection_lodf(
    disconnection: Int[Array, " "],
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
) -> tuple[Float[Array, " n_branches n_bus"], Bool[Array, " "]]:
    """Apply a single outage using the LODF formulation, updating the PTDF matrix

    Parameters
    ----------
    disconnection : Int[Array, " "]
        The disconnection or disconnection to apply, where each integer corresponds to the index of the branch to be
        removed. If the disconnection is out of bounds, it is ignored.
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after the BSDF computation
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches

    Returns
    -------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after the outage or disconnection
    success: Bool[Array, " "]
        Whether the outage was successful. Unsuccessful outages are ignored. An invalid outage is
        always considered successful.
    """
    disconnection_valid = (disconnection >= 0) & (disconnection < ptdf.shape[0])

    lodf, success = calc_lodf(disconnection, ptdf, from_node, to_node, None)
    ptdf_row = ptdf[disconnection]

    # The update formula is ptdf = ptdf + outer(lodf, ptdf_rows)
    new_ptdf = ptdf + jnp.outer(lodf, ptdf_row)
    new_ptdf = new_ptdf.at[disconnection].set(0.0)
    ptdf = jax.lax.select(disconnection_valid & success, new_ptdf, ptdf)

    return ptdf, success | ~disconnection_valid


def apply_disconnections(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    disconnections: Int[Array, " n_disconnections"],
    guarantee_unique: bool = False,
) -> DisconnectionResults:
    """Apply disconnections or disconnections using the LODF formulation, updating the PTDF matrix.

    While disconnections and disconnections are physically the same, and can be calculated identically
    the former is considered a contingency and the latter a topological measure.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix after the BSDF computation
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    disconnections : Int[Array, " n_disconnections"]
        The disconnections or disconnections to apply, where each integer corresponds to the index of the branch to be
        removed. If the disconnection or disconnection is out of bounds, it is ignored.
    guarantee_unique : bool
        Only relevant for MODF, whether to guarantee that the disconnections are unique except for
        padded values. If set to False (the default value), a jnp.unique operation is performed to
        ensure uniqueness. As padded entries are ignored, they don't need to be unique.


    Returns
    -------
    DisconnectionResults
        The results of the disconnections, containing the updated PTDF matrix, from and to nodes, success flag,
        and the MODF matrix.
    """
    if disconnections.size == 0:
        return DisconnectionResults(
            ptdf=ptdf,
            from_node=from_node,
            to_node=to_node,
            success=jnp.array(True),
            modf=MODFMatrix(
                modf=jnp.zeros((from_node.shape[0], 0)),
                branch_indices=jnp.array([], dtype=int),
            ),
        )

    if not guarantee_unique:
        disconnections = jnp.unique(disconnections, size=disconnections.size, fill_value=-1)

    modf, success = build_modf_matrix(ptdf, from_node, to_node, disconnections)
    ptdf = update_ptdf_with_modf(modf, ptdf)
    from_node = from_node.at[disconnections].set(int_max(), mode="drop")
    to_node = to_node.at[disconnections].set(int_max(), mode="drop")

    return DisconnectionResults(ptdf=ptdf, from_node=from_node, to_node=to_node, success=success, modf=modf)


def update_n0_flows_after_disconnections(
    n_0_flows: Float[Array, " n_timesteps n_branches"], disconnection_modf: Optional[MODFMatrix]
) -> Float[Array, " n_timesteps n_branches"]:
    """Update the N-0 flows after disconnections

    Parameters
    ----------
    n_0_flows : Float[Array, " n_timesteps n_branches"]
        The N-0 flows for each branch
    disconnection_modf : Optional[MODFMatrix]
        The MODF matrix containing the disconnections, if None, no update is performed

    Returns
    -------
    Float[Array, " n_timesteps n_branches"]
        The updated N-0 flows
    """
    if disconnection_modf is None:
        return n_0_flows

    return apply_modf_matrix(modf_matrix=disconnection_modf, n_0_flow=n_0_flows, branches_monitored=None)


def random_disconnection_indices(
    rng_key: jax.random.PRNGKey,
    n_disconnections: int,
    batch_size: int,
    disconnectable_branches: Int[Array, " n_disconnectable_branches"],
    chance_for_empty_disconnection: float = 0.0,
) -> Int[Array, " batch_size n_disconnections"]:
    """Generate a set of random disconnection indices into the disconnectable branches.

    Parameters
    ----------
    rng_key :
        jax.random.PRNGKey
    n_disconnections : int
        How many branches to disconnect per topology
    batch_size : int
        How many disconnections to generate, should be the same as number of topologies
    disconnectable_branches : Int[Array, " n_disconnectable_branches"]
        The disconnectable branches, i.e. the branches which can be disconnected
    chance_for_empty_disconnection : float
        The chance for an empty disconnection, i.e. no branch is disconnected

    Returns
    -------
    disconnection_indices : Int[Array, " batch_size n_disconnections"]
        The disconnection indices which are random samples from disconnectable_branches
    """
    if disconnectable_branches.size == 0:
        # Nothing to sample from; return filled with -1 (or int_max) to indicate invalid
        return jnp.full((batch_size, n_disconnections), int_max())

    key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    disconnection_indices = jax.vmap(
        partial(
            jax.random.choice,
            a=disconnectable_branches.shape[0],
            shape=(n_disconnections,),
            replace=False,
        )
    )(keys)
    if math.isclose(chance_for_empty_disconnection, 0.0):
        return disconnection_indices

    empty_disconnections = jax.random.bernoulli(key, float(chance_for_empty_disconnection), (batch_size, n_disconnections))

    disconnection_indices = jnp.where(empty_disconnections, int_max(), disconnection_indices)
    return disconnection_indices


def random_disconnections(
    rng_key: jax.random.PRNGKey,
    batch_size: int,
    n_disconnections: int,
    disconnectable_branches: Int[Array, " n_disconnectable_branches"],
    chance_for_empty_disconnection: float = 0.0,
) -> Int[Array, " batch_size n_disconnections"]:
    """Generate a set of random disconnections

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random number generator key
    batch_size : int
        How many disconnections to generate, should be the same as number of topologies
    n_disconnections : int
        How many branches to disconnect per topology
    disconnectable_branches : Int[Array, " n_disconnectable_branches"]
        The disconnectable branches, i.e. the branches which can be disconnected
    chance_for_empty_disconnection : float
        The chance for an empty disconnection, i.e. no branch is disconnected

    Returns
    -------
    disconnections : Int[Array, " batch_size n_disconnections"]
        The disconnections which are random samples from disconnectable_branches, except if empty
        disconnections are generated, in which case int_max is returned for the empty slots
    """
    if n_disconnections > disconnectable_branches.size:
        raise ValueError(
            f"Cannot disconnect {n_disconnections} branches when only {disconnectable_branches.size} are available"
        )
    disconnection_indices = random_disconnection_indices(
        rng_key=rng_key,
        n_disconnections=n_disconnections,
        batch_size=batch_size,
        disconnectable_branches=disconnectable_branches,
        chance_for_empty_disconnection=chance_for_empty_disconnection,
    )
    disconnection_batch = disconnectable_branches.at[disconnection_indices].get(mode="fill", fill_value=int_max())
    return disconnection_batch


def enumerate_disconnectable_branches(
    from_node: Union[Int[np.ndarray, " n_branches"], Int[Array, " n_branches"]],
    to_node: Union[Int[np.ndarray, " n_branches"], Int[Array, " n_branches"]],
) -> Int[Array, " n_disconnectable_branches"]:
    """Enumerate the disconnectable branches from the from and to nodes

    Disconnectable branches are all branches that leave the network N-1 safe when disconnected.
    Uses networkx, hence it does not work on GPU. More intended as a helper function when importing
    static_information without disconnectable branches.

    Parameters
    ----------
    from_node : Union[Int[np.ndarray, " n_branches"], Int[Array, " n_branches"]]
        The from nodes of the branches
    to_node : Union[Int[np.ndarray, " n_branches"], Int[Array, " n_branches"]]
        The to nodes of the branches

    Returns
    -------
    disconnectable_branches : Int[Array, " n_disconnectable_branches"]
        The disconnectable branches, i.e. the branches which can be disconnected
    """
    basecase = nx.Graph()
    basecase.add_edges_from(zip(from_node.tolist(), to_node.tolist(), strict=True))
    n_bridges_basecase = len(list(nx.bridges(basecase)))
    n_nodes_basecase = basecase.number_of_nodes()

    disconnectable = []
    for branch in range(from_node.size):
        from_node_local = np.delete(from_node, branch)
        to_node_local = np.delete(to_node, branch)

        disc_graph = nx.Graph()
        disc_graph.add_edges_from(zip(from_node_local.tolist(), to_node_local.tolist(), strict=True))

        if disc_graph.number_of_nodes() == n_nodes_basecase and len(list(nx.bridges(disc_graph))) == n_bridges_basecase:
            disconnectable.append(branch)

    return jnp.array(disconnectable)
