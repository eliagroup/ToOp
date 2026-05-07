# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Batches the inputs according to the three batch sizes.

Three batch sizes:
batch_size_bsdf, buffer_size_injection, batch_size_injection

The batch sizes are defined in the StaticInformation class.
"""

import math

from jax import numpy as jnp
from jaxtyping import ArrayLike, Int
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    NodalInjOptimResults,
    NodalInjStartOptions,
    TopoVectBranchComputations,
    int_max,
)


def pad_topologies_action_index(topologies: ActionIndexComputations, desired_size: int) -> ActionIndexComputations:
    """Pad the topologies to the desired size by adding zero topologies at the end

    The same as pad_topologies but for ActionIndexBranchComputations

    Parameters
    ----------
    topologies : ActionIndexBranchComputations
        The original topologies, shape (n_topologies, ...) where n_topologies is less or equal
        to desired_size
    desired_size : int
        The desired size of the topologies

    Returns
    -------
    ActionIndexBranchComputations
        The padded topologies, shape (desired_size, ...)
    """
    pad_size = desired_size - len(topologies)
    if pad_size == 0:
        return topologies
    assert pad_size > 0, "Desired size must be larger or equal to the original size"

    return ActionIndexComputations(
        action=jnp.pad(
            topologies.action,
            ((0, pad_size), (0, 0)),
            mode="constant",
            constant_values=int_max(),
        ),
        pad_mask=jnp.pad(
            topologies.pad_mask,
            ((0, pad_size),),
            mode="constant",
            constant_values=False,
        ),
    )


def batch_topologies(all_topologies: TopoVectBranchComputations, batch_size_bsdf: int) -> TopoVectBranchComputations:
    """Batches the topology computations from n_topologies to n_batch*batch_size_bsdf

    n_batch will be ceil(n_topologies/batch_size_bsdf)

    Uses reshape and padding

    Parameters
    ----------
    all_topologies : TopoVectBranchComputations
        All topologies that the solver is supposed to compute, shape (n_topologies, ...)
    batch_size_bsdf : int
        The envisioned batch size for the BSDF computations

    Returns
    -------
    TopoVectBranchComputations
        All topologies, but batched as (n_batch, batch_size_bsdf, ...)
    """
    num_computations = len(all_topologies)
    pad_to_size = math.ceil(num_computations / batch_size_bsdf) * batch_size_bsdf
    return TopoVectBranchComputations(
        topologies=jnp.reshape(
            all_topologies.topologies.at[jnp.arange(pad_to_size)].get(mode="fill", fill_value=False),
            (
                pad_to_size // batch_size_bsdf,
                batch_size_bsdf,
                all_topologies.topologies.shape[1],
                all_topologies.topologies.shape[2],
            ),
        ),
        sub_ids=jnp.reshape(
            all_topologies.sub_ids.at[jnp.arange(pad_to_size)].get(mode="fill", fill_value=int_max()),
            (
                pad_to_size // batch_size_bsdf,
                batch_size_bsdf,
                all_topologies.sub_ids.shape[1],
            ),
        ),
        pad_mask=jnp.reshape(
            all_topologies.pad_mask.at[jnp.arange(pad_to_size)].get(mode="fill", fill_value=False),
            (pad_to_size // batch_size_bsdf, batch_size_bsdf),
        ),
    )


def slice_topologies(
    topologies: TopoVectBranchComputations,
    topo_index: Int[ArrayLike, " "],
    batch_size_bsdf: int,
) -> TopoVectBranchComputations:
    """Get a slice of the topologies by batch_size_bsdf.

    Slices the topologies to contain only the topologies between topo_index*batch_size_bsdf and
    (topo_index+1)*batch_size_bsdf

    Pads with zeros

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The original topologies, shape (n_topologies, ...)
    topo_index : Int[Array, " "]
        The index of the topology batch
    batch_size_bsdf : int
        The size of a topology batch

    Returns
    -------
    TopoVectBranchComputations
        The slice of topology computations, shape (batch_size_bsdf, ...)
    """
    cur_range = jnp.arange(batch_size_bsdf) + topo_index * batch_size_bsdf
    return TopoVectBranchComputations(
        topologies=topologies.topologies.at[cur_range].get(mode="fill", fill_value=False),
        sub_ids=topologies.sub_ids.at[cur_range].get(mode="fill", fill_value=int_max()),
        pad_mask=topologies.pad_mask.at[cur_range].get(mode="fill", fill_value=False),
    )


def slice_topologies_action_index(
    topologies: ActionIndexComputations,
    topo_index: Int[ArrayLike, " "],
    batch_size_bsdf: int,
) -> ActionIndexComputations:
    """Get a slice of the topologies by batch_size_bsdf.

    Slices the topologies to contain only the topologies between topo_index*batch_size_bsdf and
    (topo_index+1)*batch_size_bsdf

    The same as slice_topologies but for ActionIndexBranchComputations

    Parameters
    ----------
    topologies : ActionIndexBranchComputations
        The original topologies, shape (n_topologies, ...)
    topo_index : Int[Array, " "]
        The index of the topology batch
    batch_size_bsdf : int
        The size of a topology batch

    Returns
    -------
    ActionIndexBranchComputations
        The slice of topology computations, shape (batch_size_bsdf, ...)
    """
    cur_range = jnp.arange(batch_size_bsdf) + topo_index * batch_size_bsdf
    return ActionIndexComputations(
        action=topologies.action.at[cur_range].get(mode="fill", fill_value=int_max()),
        pad_mask=topologies.pad_mask.at[cur_range].get(mode="fill", fill_value=False),
    )


def slice_nodal_inj_start_options(
    nodal_inj_start_options: NodalInjStartOptions,
    nodal_inj_index: Int[ArrayLike, ""],
    batch_size_bsdf: int,
) -> NodalInjStartOptions:
    """Get a slice of the topologies by batch_size_bsdf.

    Slices the topologies to contain only the topologies between nodal_inj_index*batch_size_bsdf and
    (nodal_inj_index+1)*batch_size_bsdf

    The same as slice_topologies but for ActionIndexBranchComputations

    Parameters
    ----------
    nodal_inj_start_options : NodalInjStartOptions
        The original nodal injection start options, shape (n_topologies, ...)
    nodal_inj_index : Int[Array, " "]
        The index of the topology batch
    batch_size_bsdf : int
        The size of a topology batch

    Returns
    -------
    NodalInjStartOptions
        The slice of topology computations, shape (batch_size_bsdf, ...)
    """
    cur_range = jnp.arange(batch_size_bsdf) + nodal_inj_index * batch_size_bsdf
    previous_results = NodalInjOptimResults(
        pst_tap_idx=nodal_inj_start_options.previous_results.pst_tap_idx.at[cur_range].get(
            mode="fill", fill_value=int_max()
        ),
    )
    result = NodalInjStartOptions(
        previous_results=previous_results,
        precision_percent=nodal_inj_start_options.precision_percent,
    )
    assert result.precision_percent.ndim == 0, (
        f"precision_percent must remain scalar (0-D), got shape {result.precision_percent.shape}"
    )
    return result
