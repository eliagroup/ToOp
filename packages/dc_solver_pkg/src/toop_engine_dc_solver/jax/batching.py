# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
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
from functools import reduce

import equinox
import jax
from beartype.typing import Optional
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    NodalInjOptimResults,
    NodalInjStartOptions,
    TopoVectBranchComputations,
    int_max,
)


def pad_topologies(topologies: TopoVectBranchComputations, desired_size: int) -> TopoVectBranchComputations:
    """Pad the topologies to the desired size by adding zero topologies at the end

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The original topologies, shape (n_topologies, ...) where n_topologies is less or equal
        to desired_size
    desired_size : int
        The desired size of the topologies

    Returns
    -------
    TopoVectBranchComputations
        The padded topologies, shape (desired_size, ...)
    """
    pad_size = desired_size - len(topologies)
    if pad_size == 0:
        return topologies
    assert pad_size > 0, "Desired size must be larger or equal to the original size"

    return TopoVectBranchComputations(
        topologies=jnp.pad(
            topologies.topologies,
            ((0, pad_size), (0, 0), (0, 0)),
            mode="constant",
            constant_values=False,
        ),
        sub_ids=jnp.pad(
            topologies.sub_ids,
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
    topo_index: Int[Array, " "],
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
    topo_index: Int[Array, " "],
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
    nodal_inj_index: Int[Array, " "],
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
        pst_taps=nodal_inj_start_options.previous_results.pst_taps.at[cur_range].get(mode="fill", fill_value=jnp.nan),
    )
    result = NodalInjStartOptions(
        previous_results=previous_results,
        precision_percent=nodal_inj_start_options.precision_percent,
    )
    assert result.precision_percent.ndim == 0, (
        f"precision_percent must remain scalar (0-D), got shape {result.precision_percent.shape}"
    )
    return result


def batch_injection_selection(
    all_injections: InjectionComputations,
    batch_size_injection: int,
    buffer_size_injection: int,
) -> InjectionComputations:
    """Batches the injection computations from n_injections to buffer_size_injections*batch_size_bsdf.

    Uses reshape, but assumes the input is already padded to the correct size

    Parameters
    ----------
    all_injections : InjectionComputations
        All injections that the solver is supposed to compute, shape (n_injections, ...)
    batch_size_injection : int
        The envisioned batch size for the injection computations
    buffer_size_injection : int
        This is a parameter that should be set large enough to hold all the injection
        computations for a batch of topologies, but not so large to waste too much memory. The upper
        bound is `ceil(max(n_injections_per_topology) * batch_size_bsdf / batch_size_injection)`

    Returns
    -------
    InjectionComputations
        All injections, but batched as (buffer_size_injection, batch_size_injection, ...)
    """
    return InjectionComputations(
        injection_topology=jnp.reshape(
            all_injections.injection_topology,
            (
                buffer_size_injection,
                batch_size_injection,
                all_injections.injection_topology.shape[1],
                all_injections.injection_topology.shape[2],
            ),
        ),
        corresponding_topology=jnp.reshape(
            all_injections.corresponding_topology,
            (buffer_size_injection, batch_size_injection),
        ),
        pad_mask=jnp.reshape(
            all_injections.pad_mask,
            (buffer_size_injection, batch_size_injection),
        ),
    )


def get_injections_for_topo_range(
    all_injections: InjectionComputations,
    topo_index: Int[Array, " "],
    batch_size_bsdf: int,
    batch_size_injection: int,
    buffer_size_injection: int,
    return_relative_index: bool = False,
) -> InjectionComputations:
    """Get the injections for a batch of topologies.

    Gets Topologies between topo_index*batch_size_bsdf and (topo_index+1)*batch_size_bsdf

    Parameters
    ----------
    all_injections : InjectionComputations
        All injections that the solver is supposed to compute, shape (n_injections, ...)
    topo_index : int
        The index of the topology batch
    batch_size_bsdf : int
        The envisioned batch size for the BSDF computations
    batch_size_injection : int
        The envisioned batch size for the injection + contingency computations
    buffer_size_injection : int
        This is a parameter that should be set large enough to hold all the injection
        computations for a batch of topologies, but not so large to waste too much memory. The upper
        bound is `ceil(max(n_injections_per_topology) * batch_size_topology / batch_size_injection)`
    return_relative_index : bool, optional
        If True, corresponding_topology will be relative to zero, otherwise it will be the absolute
        index of the injection as in all_injections, by default False

    Returns
    -------
    InjectionComputations
        (buffer_size_injection, batch_size_injection, ...) with the injections that belong to the
        topologies between topo_index*batch_size_bsdf and (topo_index+1)*batch_size_bsdf
        padded with zeros
    """
    relevant_injections = get_injections_for_topo_range_flat(
        all_injections,
        topo_index,
        batch_size_bsdf,
        buffer_size_injection * batch_size_injection,
        return_relative_index,
    )

    # Reshape to (buffer_size_injection, batch_size_injection, ...)
    relevant_injections = batch_injection_selection(relevant_injections, batch_size_injection, buffer_size_injection)

    return relevant_injections


def get_injections_for_topo_range_flat(
    all_injections: InjectionComputations,
    topo_index: Int[Array, " "],
    batch_size_bsdf: int,
    packet_size_injection: int,
    return_relative_index: bool = False,
) -> InjectionComputations:
    """Get the injections that belong to the topologies.

    Gets topologies between topo_index*batch_size_bsdf and (topo_index+1)*batch_size_bsdf.

    Parameters
    ----------
    all_injections : InjectionComputations
        All injections that the solver is supposed to compute, shape (n_injections, ...)
    topo_index : int
        The index of the topology batch
    batch_size_bsdf : int
        The envisioned batch size for the BSDF computations
    packet_size_injection : int
        The envisioned packet size for the injections, should be
        buffer_size_injection * batch_size_injection
    return_relative_index : bool, optional
        If True, corresponding_topology will be relative to zero, otherwise it will be the absolute
        index of the injection as in all_injections, by default False


    Returns
    -------
    InjectionComputations
        (packet_size_injection, ...) with the injections that belong to the
        topologies between topo_index*batch_size_bsdf and (topo_index+1)*batch_size_bsdf
        padded with zeros
    """
    int_max = jnp.iinfo(all_injections.corresponding_topology.dtype).max
    first_topology = topo_index * batch_size_bsdf
    last_topology = (topo_index + 1) * batch_size_bsdf

    injection_mask = (all_injections.corresponding_topology >= first_topology) & (
        all_injections.corresponding_topology < last_topology
    )

    injection_mask = equinox.error_if(
        injection_mask,
        jnp.sum(injection_mask) > packet_size_injection,
        f"Packet size of {packet_size_injection} is not large enough to hold all injections "
        + f"for topology {topo_index} (found {jnp.sum(injection_mask)} injections)",
    )

    # Get the indices of the injections that belong to this topology
    injection_index = jnp.nonzero(
        injection_mask,
        size=packet_size_injection,
        fill_value=int_max,
    )

    # Also subtract the topology index from the corresponding topology if return_relative_index is
    # True
    corresponding_topology = all_injections.corresponding_topology - (topo_index * batch_size_bsdf * return_relative_index)
    relevant_injections = InjectionComputations(
        injection_topology=all_injections.injection_topology.at[injection_index].get(mode="fill", fill_value=False),
        corresponding_topology=corresponding_topology.at[injection_index].get(mode="fill", fill_value=int_max),
        pad_mask=all_injections.pad_mask.at[injection_index].get(mode="fill", fill_value=False),
    )

    return relevant_injections


def slice_injections(
    injections: InjectionComputations,
    topo_index: Int[Array, " "],
    batch_size: int,
) -> InjectionComputations:
    """Slices the injections.

    Will contain only the injections between topo_index*batch_size and
    (topo_index+1)*batch_size. This is helpful for slicing symmetric batches

    Pads with zeros

    Parameters
    ----------
    injections : InjectionComputations
        The original injections, shape (n_injections, ...)
    topo_index : Int[Array, " "]
        The index of the topology batch
    batch_size : int
        The size of a topology batch

    Returns
    -------
    InjectionComputations
        The slice of injections, shape (batch_size, ...)
    """
    cur_range = jnp.arange(batch_size) + topo_index * batch_size
    return InjectionComputations(
        injection_topology=injections.injection_topology.at[cur_range].get(mode="fill", fill_value=False),
        corresponding_topology=injections.corresponding_topology.at[cur_range].get(mode="fill", fill_value=0),
        pad_mask=injections.pad_mask.at[cur_range].get(mode="fill", fill_value=False),
    )


def batch_injections(
    all_injections: InjectionComputations,
    batched_topologies: TopoVectBranchComputations,
    batch_size_injection: int,
    buffer_size_injection: int,
) -> InjectionComputations:
    """Batches the injection computations.

    from (n_injections, ...) to (n_topo_batches, buffer_size_injection, batch_size_injection, ...)

    Parameters
    ----------
    all_injections : InjectionComputations
        All injections that the solver is supposed to compute, shape (n_injections, ...)
    batched_topologies : TopoVectBranchComputations
        The already batched topology computations, shape (n_topo_batches, batch_size_bsdf, ...)
    batch_size_injection : int
        The desired batch size for the injection + contingency computations
    buffer_size_injection : int
        This is a parameter that should be set large enough to hold all the injection
        computations for a batch of topologies, but not so large to waste too much memory. The upper
        bound is `ceil(max(n_injections_per_topology) * batch_size_topology / batch_size_injection)`

    Returns
    -------
    InjectionComputations
        (n_topo_batches, buffer_size_injection, batch_size_injection, ...)
    """
    batch_size_bsdf = batched_topologies.topologies.shape[1]

    collection: list[InjectionComputations] = []
    # TODO rewrite to jax.map
    for topo_index in range(batched_topologies.topologies.shape[0]):
        # find out which injections belong to this topology
        relevant_injections = get_injections_for_topo_range(
            all_injections,
            topo_index,
            batch_size_bsdf,
            batch_size_injection,
            buffer_size_injection,
        )

        collection.append(relevant_injections)

    return InjectionComputations(
        injection_topology=jnp.stack([x.injection_topology for x in collection]),
        corresponding_topology=jnp.stack([x.corresponding_topology for x in collection]),
        pad_mask=jnp.stack([x.pad_mask for x in collection]),
    )


def get_buffer_utilization(
    n_inj_combis_per_topo: Int[Array, " n_topologies"],
    buffer_size_injection: int,
    batch_size_injection: int,
) -> Float[Array, " n_topologies"]:
    """Get the injection buffer utilization for the given topologies.

    The upper bound of this utilization can be computed by inputs.upper_bound_buffer_size_injection
    You can however set a lower value which is just enough to fit the injection combinations needed
    by the topologies by monitoring the buffer utilization and choosing a buffer size that holds
    the topologies with the highest utilization.

    Parameters
    ----------
    n_inj_combis_per_topo : Int[Array, " n_topologies"]
        The number of injection combinations needed by each topology batch, can be obtained by
        injections.count_injection_combinations
    buffer_size_injection : int
        The buffer size for the injection computations
    batch_size_injection : int
        The batch size for the injection computations

    Returns
    -------
    Float[Array, " n_topologies"]
        The buffer utilization for each topology batch
    """
    return n_inj_combis_per_topo.astype(float) / (buffer_size_injection * batch_size_injection)


def count_injection_combinations_from_corresponding_topology(
    corresponding_topology: Int[Array, " n_injections"],
    batch_size_bsdf: int,
    n_topologies: Optional[int] = None,
) -> Int[Array, " n_batches"]:
    """Count the number of injections per topology batch.

    In contrast to injections.count_injection_combinations this assumes already a pre-computed
    set of injections for which we just want to count how many of them belong to each topology.
    Other than that, the two functions should be interchangeable

    Parameters
    ----------
    corresponding_topology : Int[Array, " n_injections"]
        The corresponding topology index for each injection
    batch_size_bsdf : int
        The batch size for the bsdf computations - determines how the topologies are going to
        end up being split
    n_topologies : int, optional
        The number of topologies, if None, uses the maximum index in corresponding_topology + 1.
        If you want to jit this function, you need to provide the number of topologies

    Returns
    -------
    Int[Array, " n_batches"]
        The number of injections per topology batch
    """
    if n_topologies is None:
        n_topologies = jnp.max(corresponding_topology).item() + 1
    # Make sure it's divisible by batch_size_bsdf
    n_topologies = math.ceil(n_topologies / batch_size_bsdf) * batch_size_bsdf
    injs_per_topo = jnp.bincount(corresponding_topology, length=n_topologies)

    # Group them by batch_size_bsdf
    return jnp.sum(jnp.reshape(injs_per_topo, (-1, batch_size_bsdf)), axis=1)


def greedy_buffer_size_selection(
    n_inj_combis_per_topo_batch: Int[Array, " n_topo_batches"],
    batch_size_injection: int,
) -> int:
    """Greedily selects the buffer size to just fit the injection combinations needed by the topologies.

    Uses the batch size bsdf/injections from the static information dataclass but ignores any
    buffer size set there.

    Parameters
    ----------
    n_inj_combis_per_topo_batch : Int[Array, " n_topo_batches"]
        The number of injection combinations needed by each topology batch, can be obtained by
        injections.count_injection_combinations
    batch_size_injection : int
        The batch size for the injection computations

    Returns
    -------
    int
        The minimum buffer size that would just fit the injection combinations
    """
    return jnp.ceil(jnp.max(n_inj_combis_per_topo_batch).astype(float) / batch_size_injection).astype(int).item()


def upper_bound_buffer_size_injections(
    n_inj_combis: Int[Array, " n_sub_relevant"],
    batch_size_bsdf: int,
    batch_size_injection: int,
    limit_n_subs: Optional[int] = None,
) -> int:
    """Compute a safe upper bound for the buffer_size_injections parameter

    Parameters
    ----------
    n_inj_combis : Int[Array, " n_sub_relevant"]
        The number of injection combinations for each substation
    batch_size_bsdf : int
        The envisioned batch size for the bsdf computation
    batch_size_injection : int
        The envisioned batch size for the injection computation
    limit_n_subs : int
        The maximum number of substation splits to consider. If None, uses the number of substations

    Returns
    -------
    int
        A safe upper bound for the buffer_size_injections parameter
    """
    if limit_n_subs is not None:
        n_inj_combis = jnp.sort(n_inj_combis, descending=True)[:limit_n_subs]
    # Do the product in python to avoid overflows
    max_injections_per_topology = reduce(lambda x, y: x * y, n_inj_combis.tolist())
    return math.ceil(max_injections_per_topology * batch_size_bsdf / batch_size_injection)


def greedy_n_subs_selection(
    topologies: TopoVectBranchComputations,
) -> int:
    """Greedily selects the number of substations to expose to the BSDF

    This is just the maximum number of substation splits in any topology

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The topology computations.

    Returns
    -------
    int
        The minimum number of substations to expose to the BSDF
    """
    has_split = jnp.any(topologies.topologies, axis=-1)
    num_split = jnp.sum(has_split, axis=-1)
    return jnp.max(num_split).item()


def split_injections(
    injections: InjectionComputations,
    n_splits: int,
    packet_size_injection: int,
    n_topos_per_split: int,
) -> InjectionComputations:
    """Split the injections into n_splits parts.

    The parts don't necessarily have the same number of injections, as corresponding_topology is
    used to split the injections

    Parameters
    ----------
    injections : InjectionComputations
        The injection computations
    n_splits : int
        The number of splits
    packet_size_injection : int
        The envisioned packet size for the injections, should be
        buffer_size_injection * batch_size_injection * ceil(n_topos_per_split / batch_size_bsdf)
    n_topos_per_split : int
        The number of topologies in each split, should be n_topologies / n_splits. Note that
        n_topologies should be divisible by n_splits, i.e. the result should be an integer

    Returns
    -------
    InjectionComputations
        The split injections, shape (n_splits, packet_size_injection, ...)
    """
    return jax.vmap(get_injections_for_topo_range_flat, in_axes=(None, 0, None, None, None))(
        injections, jnp.arange(n_splits), n_topos_per_split, packet_size_injection, True
    )
