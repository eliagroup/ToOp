# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds functions to identify bridges inside a network."""

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.sparse import csr_matrix


def compute_nodal_injection(
    injection_power: Float[np.ndarray, " n_timestep n_injection"],
    injection_nodes: Int[np.ndarray, " n_injection"],
    number_of_nodes: int,
) -> Float[np.ndarray, " n_timestep n_node"]:
    """Compute the netto injection at each node

    Parameters
    ----------
    injection_power: Float[np.ndarray, " n_timestep n_injection"]
        An array sorted by generator and load id that holds the active power for each timestep
    injection_nodes: Int[np.ndarray, " n_injection"]
        An array sorted by generator and load id that holds the node-id for each generator or load in the system
    number_of_nodes: int
        How many busbars are in the system (including Bus B)

    Returns
    -------
    Float[np.ndarray, " n_timestep n_node"]
        An array of the netto injection at each node in the grid
    """
    # Get nodal-prod matrix
    number_of_injections = injection_power.shape[1]
    node_injection_matrix = csr_matrix(
        (
            np.ones(number_of_injections),
            (injection_nodes, np.arange(number_of_injections)),
        ),
        shape=(number_of_nodes, number_of_injections),
        dtype=int,
    )

    # compute the resulting netto injections for each node
    nodal_injections = node_injection_matrix.dot(injection_power.T)
    return nodal_injections.T


def get_mw_injections_at_nodes(
    injection_idx_at_node: list[Int[np.ndarray, " n_injections_at_node"]],
    mw_injections: Float[np.ndarray, " n_timestep n_injection"],
) -> list[Float[np.ndarray, " n_timestep n_injections_at_node"]]:
    """Get the injection power of each injection for each relevant node.

    Parameters
    ----------
    injection_idx_at_node: list[Int[np.ndarray, " n_injections_at_node"]]
        A list of the length of relevant nodes. Contains arrays with the idx of the injections in the injection arrays
    mw_injections: Float[np.ndarray, " n_timestep n_injection"]
        An array of the power of each injection for all timesteps

    Returns
    -------
    list[Float[np.ndarray, " n_timestep n_injections_at_node"]]
        A list of the length of relevant nodes. Contains an array of the power of each injection at the given node
    """
    return [mw_injections[:, injection_ids] for injection_ids in injection_idx_at_node]


def identify_inactive_injections(
    mw_injections_at_node: list[Float[np.ndarray, " n_timestep n_injections_at_node"]],
) -> list[Bool[np.ndarray, " n_injections_at_node"]]:
    """Get a boolean array identifying all injections that are not active.

    Get a boolean array identifying all injections that are always
    close to zero (in service but not generating/loading in all timesteps)

    Parameters
    ----------
    mw_injections_at_node: list[Float[np.ndarray, " n_timestep n_injections_at_node"]]
        A list of the length of relevant nodes.
        Contains arrays of the power of all injections connected to the given node

    Returns
    -------
    list[Bool[np.ndarray, " n_injections_at_node"]]
        A list of the length of relevant nodes.
        Contains Boolean Arrays depicting if injection is non-zero / active in any timestep
    """
    return [(abs(mw_injections) > 1e-12).any(axis=0) for mw_injections in mw_injections_at_node]
