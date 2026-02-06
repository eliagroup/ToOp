# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provide functions to compute injections for the DC power flow problem.

The injections module is much slimmer than the CPU counterpart, it only contains the function
to translate a combination of generator injections to a nodal injection vector. This is because
the task of enumerating combinations has been outsourced to the inputs module.
"""

from functools import partial

import jax
import numpy as np
from beartype.typing import Literal, Optional
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.types import ActionSet, InjectionComputations, TopoVectBranchComputations, int_max
from toop_engine_dc_solver.jax.utils import action_index_to_binary_form


def get_injection_per_bus(
    injection_assignment: Bool[Array, " max_inj_per_sub"],
    sub_id: Int[Array, ""],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
) -> tuple[Float[Array, " n_timesteps"], Float[Array, " n_timesteps"]]:
    """Get the injection on bus A and B for a single substation.

    Parameters
    ----------
    injection_assignment : Bool[Array, " max_inj_per_sub"]
        The assignment of the injections to the busbars as a topo vect
    sub_id : Int[Array, ""]
        The substation id this injection refers to
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations

    Returns
    -------
    Float[Array, " n_timesteps"]
        The summed injection on bus A
    Float[Array, " n_timesteps"]
        The summed injection on bus B
    """
    local_inj = relevant_injections.at[:, sub_id, :].get(mode="fill", fill_value=0)
    bus_a_inj = jnp.sum(local_inj * ~injection_assignment, axis=1)
    bus_b_inj = jnp.sum(local_inj * injection_assignment, axis=1)

    return bus_a_inj, bus_b_inj


def get_single_injection_vector(
    injection_assignment: Bool[Array, " max_inj_per_sub"],
    sub_id: Int[Array, " "],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    n_stat: Int[Array, " "],
    rel_stat_map: Int[Array, " n_relevant_subs"],
) -> Float[Array, " n_timesteps n_bus"]:
    """Update the nodal injection vector with a single assignment.

    Overwrites the injections at bus A and B

    Parameters
    ----------
    injection_assignment : Bool[Array, " max_inj_per_sub"]
        The assignment of the injections to the busbars as a topo vect
    sub_id : Int[Array, " "]
        The substation id this injection refers to. If an invalid sub id is passed in, the
        output nodal injections vector is the same as the input.
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    nodal_injections : Float[Array, " n_timesteps n_bus"]
        The nodal injection vector to update
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-busbar index

    Returns
    -------
    Float[Array, " n_timesteps n_bus"]
        The updated nodal injection vector where bus A and B have been overwritten
    """
    n_subs_rel = rel_stat_map.shape[0]

    bus_a_inj, bus_b_inj = get_injection_per_bus(injection_assignment, sub_id, relevant_injections)
    bus_a_index = rel_stat_map[sub_id]
    bus_b_index = n_stat + sub_id

    nodal_injections_new = nodal_injections.at[:, bus_a_index].set(bus_a_inj)
    nodal_injections_new = nodal_injections_new.at[:, bus_b_index].set(bus_b_inj)

    # Make sure not to overwrite anything upon invalid sub ids
    nodal_injections_new = jnp.where((sub_id >= 0) & (sub_id < n_subs_rel), nodal_injections_new, nodal_injections)

    return nodal_injections_new


def get_injection_vector(
    injection_assignment: Bool[Array, " n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " n_splits"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    nodal_injections: Float[Array, " n_timesteps n_bus"],
    n_stat: Int[Array, " "],
    rel_stat_map: Int[Array, " n_relevant_subs"],
) -> Float[Array, " n_timesteps n_bus"]:
    """Apply a nodal injection combination to a nodal injection vector.

    Parameters
    ----------
    injection_assignment : Bool[Array, " n_splits max_inj_per_sub"]
        Injection combination to apply in the format of a topo vect.
    sub_ids : Int[Array, " n_splits"]
        The substation ids each injection refers to
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    nodal_injections : Float[Array, " n_timesteps n_bus"]
        The previous nodal injection vector before applying the injection combination
    n_stat : Int[Array, " "]
        Number of buses excluding the second busbars for relevant substations
    rel_stat_map : Int[Array, " n_relevant_subs"]
        Map from relevant substation index to all-busbar index

    Returns
    -------
    Float[Array, " n_timesteps n_bus"]
        The new nodal injection vector
    """
    carry, _ys = jax.lax.scan(
        f=lambda nodal_injections_scan, data: (
            get_single_injection_vector(
                injection_assignment=data[0],
                sub_id=data[1],
                relevant_injections=relevant_injections,
                nodal_injections=nodal_injections_scan,
                n_stat=n_stat,
                rel_stat_map=rel_stat_map,
            ),
            None,
        ),
        init=nodal_injections,
        xs=(injection_assignment, sub_ids),
        unroll=True,
    )
    return carry


def get_reassignment_deltap(
    injection_assignment: Bool[Array, " max_inj_per_sub"],
    sub_id: Int[Array, " "],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    n_stat: Int[Array, " "],
    rel_stat_map: Int[Array, " n_relevant_subs"],
) -> tuple[
    Int[Array, " "],
    Int[Array, " "],
    Float[Array, " n_timesteps "],
    Float[Array, " n_timesteps "],
]:
    """Get a node/deltap pair for a reassignment of an injection.

    Applying this using the PTDF should be equivalent to changing the nodal injection vector directly. The formula should
    be roughly: n_0 + PTDF[:,busa_node] * busa_delta + PTDF[:, busb_node] * busb_delta

    Parameters
    ----------
    injection_assignment : Bool[Array, " max_inj_per_sub"]
        The assignment of the injections to the busbars as a topo vect
    sub_id : Int[Array, " "]
        The substation id this injection refers to
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-busbar index

    Returns
    -------
    busa_node: Int[Array, " "]
        The node index for bus A
    busb_node: Int[Array, " "]
        The node index for bus B
    busa_delta: Float[Array, " n_timesteps "]
        The delta p for bus A
    busb_delta: Float[Array, " n_timesteps "]
        The delta p for bus B
    """
    _busa_newinj, busb_newinj = get_injection_per_bus(
        injection_assignment=injection_assignment,
        sub_id=sub_id,
        relevant_injections=relevant_injections,
    )
    # The delta is the new injection minus the unsplit injection
    # For busa this is equal to -busb_newinj, and for busb this is equal to busb_newinj
    # Bus B has 0 in the unsplit case.
    busa_delta = -busb_newinj
    # busa_delta = busa_newinj - relevant_injections.at[:, sub_id, :].get(mode="fill", fill_value=0).sum(axis=1)
    busb_delta = busb_newinj

    busa_node = rel_stat_map[sub_id]
    busb_node = n_stat + sub_id

    return busa_node, busb_node, busa_delta, busb_delta


def apply_reassignments_deltap(
    injection_assignment: Bool[Array, " n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " n_splits"],
    split_n0_flow: Float[Array, " n_timesteps n_branch"],
    ptdf: Float[Array, " n_branch n_bus"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    n_stat: Int[Array, " "],
    rel_stat_map: Int[Array, " n_relevant_subs"],
) -> Float[Array, " n_timesteps n_branch"]:
    """Apply the reassignments using the PTDF deltap method

    This is equivalent to changing the nodal injection vector, just faster

    Parameters
    ----------
    injection_assignment : Bool[Array, " n_splits max_inj_per_sub"]
        The assignment of the injections to the busbars as a topo vect
    sub_ids : Int[Array, " n_splits"]
        The substation id this injection refers to
    split_n0_flow : Float[Array, " n_timesteps n_branch"]
        The branch flows with branch splits already applied. If cross_coupler_flows is True, you can use the bsdf vectors of
        the splits to compute this.
    ptdf : Float[Array, " n_branch n_bus"]
        The PTDF matrix with branch splits already applied. It is assumed that all stations with valid sub_id are split.
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-busbar index

    Returns
    -------
    Float[Array, " n_timesteps n_branch"]
        The branch flows after the reassignments have been applied
    """
    busa_node, busb_node, busa_delta, busb_delta = jax.vmap(
        jax.tree_util.Partial(
            get_reassignment_deltap, relevant_injections=relevant_injections, n_stat=n_stat, rel_stat_map=rel_stat_map
        )
    )(
        injection_assignment=injection_assignment,
        sub_id=sub_ids,
    )

    delta_flow = jnp.einsum("bn,tn->tb", ptdf[:, busa_node], busa_delta)
    delta_flow += jnp.einsum("bn,tn->tb", ptdf[:, busb_node], busb_delta)
    return split_n0_flow + delta_flow


def get_relevant_injection_outage_deltap(
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"],
    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"],
) -> Float[Array, " n_timesteps n_rel_inj_failures"]:
    """Get the deltap for relevant injection outages.

    Parameters
    ----------
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    relevant_injection_outage_sub : Int[Array, " n_rel_inj_failures"]
        The relevant substation index for the injection outages
    relevant_injection_outage_idx : Int[Array, " n_rel_inj_failures"]
        The relevant injection index for the injection outages

    Returns
    -------
    Float[Array, " n_timesteps n_rel_inj_failures"]
        The deltap for the relevant injection outages

    """
    return -relevant_injections.at[:, relevant_injection_outage_sub, relevant_injection_outage_idx].get(
        mode="fill", fill_value=0
    )


def get_all_injection_outage_deltap(
    injection_outage_deltap: Float[Array, " n_timesteps n_nonrel_inj_failures"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"],
    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"],
) -> Float[Array, " n_timesteps n_inj_failures"]:
    """Get the deltap for all injection outages, first non-relevant then relevant.

    This is the same for all topologies, as the injection p does not change with the topology.

    Parameters
    ----------
    injection_outage_deltap : Float[Array, " n_timesteps n_nonrel_inj_failures"]
        The deltap for non-relevant injection outages
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    relevant_injection_outage_sub : Int[Array, " n_rel_inj_failures"]
        The relevant substation index for the injection outages
    relevant_injection_outage_idx : Int[Array, " n_rel_inj_failures"]
        The relevant injection index for the injection outages

    Returns
    -------
    Float[Array, " n_timesteps n_inj_failures"]
        The deltap for all injection outages, concatenated along the inj_failures dimension
    """
    deltap = jnp.concatenate(
        [
            injection_outage_deltap,
            get_relevant_injection_outage_deltap(
                relevant_injections=relevant_injections,
                relevant_injection_outage_sub=relevant_injection_outage_sub,
                relevant_injection_outage_idx=relevant_injection_outage_idx,
            ),
        ],
        axis=1,
    )
    return deltap


def get_single_outaged_injection_node_after_reassignment(
    injection_assignment: Bool[Array, " n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " n_splits"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"],
    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"],
    nonrel_injection_outage_node: Int[Array, " n_nonrel_inj_failures"],
    rel_stat_map: Int[Array, " n_relevant_subs"],
    n_stat: Int[Array, " "],
) -> Int[Array, " n_rel_inj_failures"]:
    """Get the assigned node of all injection outages post-split.

    Takes into account the injection assignment and potentially moves injection outages to bus B

    Parameters
    ----------
    injection_assignment : Bool[Array, " n_splits max_inj_per_sub"]
        The injection assignment for all injections
    sub_ids : Int[Array, " n_splits"]
        The substation ids for all injections
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    relevant_injection_outage_sub : Int[Array, " n_rel_inj_failures"]
        The relevant substation index for the relevant injection outages
    relevant_injection_outage_idx : Int[Array, " n_rel_inj_failures"]
        The relevant injection index for the relevant injection outages
    nonrel_injection_outage_node : Int[Array, " n_nonrel_inj_failures"]
        The node for all non-relevant injection outages
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-busbar index
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations

    Returns
    -------
    Int[Array, " n_inj_failures"]
        The node for the single injection outages
    """
    rel_nodes = convert_relevant_sub_injection_outages(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
        relevant_injections=relevant_injections,
        relevant_injection_outage_sub=relevant_injection_outage_sub,
        relevant_injection_outage_idx=relevant_injection_outage_idx,
        rel_stat_map=rel_stat_map,
        n_stat=n_stat,
    )
    return jnp.concatenate([nonrel_injection_outage_node, rel_nodes], axis=0)


def get_all_outaged_injection_nodes_after_reassignment(
    injection_assignment: Bool[Array, " batch n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " batch n_splits"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"],
    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"],
    nonrel_injection_outage_node: Int[Array, " n_nonrel_inj_failures"],
    rel_stat_map: Int[Array, " n_relevant_subs"],
    n_stat: Int[Array, " "],
) -> Int[Array, " batch n_inj_failures"]:
    """Get the node for a batch of injection outages. Just vmaps over get_single_outaged_injection_node_after_reassignment.

    Parameters
    ----------
    injection_assignment : Bool[Array, " batch n_splits max_inj_per_sub"]
        The injection assignment for all injections
    sub_ids : Int[Array, " batch n_splits"]
        The substation ids for all injections
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections for all relevant substations
    relevant_injection_outage_sub : Int[Array, " n_rel_inj_failures"]
        The relevant substation index for the relevant injection outages
    relevant_injection_outage_idx : Int[Array, " n_rel_inj_failures"]
        The relevant injection index for the relevant injection outages
    nonrel_injection_outage_node : Int[Array, " n_nonrel_inj_failures"]
        The node for all non-relevant injection outages
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-busbar index
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations

    Returns
    -------
    Int[Array, " batch n_inj_failures"]
        The node index for all injection outages
    """
    return jax.vmap(
        jax.tree_util.Partial(
            get_single_outaged_injection_node_after_reassignment,
            relevant_injections=relevant_injections,
            relevant_injection_outage_sub=relevant_injection_outage_sub,
            relevant_injection_outage_idx=relevant_injection_outage_idx,
            nonrel_injection_outage_node=nonrel_injection_outage_node,
            rel_stat_map=rel_stat_map,
            n_stat=n_stat,
        )
    )(
        injection_assignment=injection_assignment,
        sub_ids=sub_ids,
    )


def convert_relevant_sub_injection_outages(
    injection_assignment: Bool[Array, " n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " n_splits"],
    relevant_injections: Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"],
    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"],
    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"],
    rel_stat_map: Int[Array, " n_relevant_subs"],
    n_stat: Int[Array, " "],
) -> Int[Array, " n_rel_inj_failures"]:
    """Convert the relevant sub injection outage indices.

    If a sub was split, an injection outage might have moved to busbar B, hence we need to resolve for each relevant
    injection outage which node in the PTDF it refers to.

    Parameters
    ----------
    injection_assignment : Bool[Array, " n_splits max_inj_per_sub"]
        The injection assignment for all injections at the station, in numpy format
    sub_ids : Int[Array, " n_splits"]
        The sub-ids that are referenced in each split
    relevant_injections : Float[Array, " n_timesteps n_relevant_subs max_inj_per_sub"]
        The injections in MW for all relevant substations
    relevant_injection_outage_sub : Int[Array, " n_rel_inj_failures"]
        The relevant substation index for the relevant injection outages, indexing into relevant substations (not all nodes)
    relevant_injection_outage_idx : Int[Array, " n_rel_inj_failures"]
        The injection index inside the substation, indexing into the max_inj_per_sub dimension
    rel_stat_map : Int[Array, " n_relevant_subs"]
        The map from relevant substation index to all-node index
    n_stat : Int[Array, " "]
        The number of buses excluding the second busbars for relevant substations, to map to the node index

    Returns
    -------
    Int[Array, " n_rel_inj_failures"]
        The node index for the relevant injection outages, indexing into PTDF nodes
    """
    if relevant_injection_outage_idx.size == 0:
        return jnp.array([], dtype=int)

    max_inj_per_sub = relevant_injections.shape[2]

    # Create a map that holds for every substation and every injection whether it is on busbar A or B
    # First set it all to bus A
    busbar_map: Int[Array, " n_relevant_subs max_inj_per_sub"] = jnp.repeat(rel_stat_map[:, None], max_inj_per_sub, axis=1)

    # We will overwrite those parts of the busbar map where there are splits
    busbar_map_per_split = jnp.where(
        injection_assignment,
        (sub_ids + n_stat)[:, None],  # True -> Bus B
        rel_stat_map.at[sub_ids].get(mode="fill", fill_value=int_max())[:, None],  # False -> Bus A
    )

    # Then set all injections that are on bus B to the bus B index
    busbar_map = busbar_map.at[sub_ids].set(busbar_map_per_split, mode="drop")

    injection_outage_node = busbar_map.at[relevant_injection_outage_sub, relevant_injection_outage_idx].get(
        mode="fill", fill_value=int_max()
    )

    return injection_outage_node


def map_injection_candidates_idx_to_topology(
    idx: Int[Array, " "],
    corresponding_topology: Int[Array, " "],
    n_injection_candidates_per_topology: Int[Array, " n_topologies"],
    cumsum_injection_candidates_per_topology: Optional[Int[Array, " n_topologies"]] = None,
) -> Int[Array, " "]:
    """Map injection candidates to topology.

    Map a global index over all injection candidates to a local index over only injections
    for a single topology.

    The global index over all injection candidates (arange(n_injection_combinations)) will be mapped
    to a lot of arange(n_injection_candidates_per_topology[corresponding_topology[idx]]).

    This assumes that the corresponding_topology is sorted, i.e. all entries for the same topology
    are consecutive and that there are no gaps.

    Parameters
    ----------
    idx : Int[Array, " "]
        The index of the injection candidate to be computed, indexes into all injection candidates
        (not just the ones for a single topology), i.e. should be in range
        (0, n_injection_combinations)
    corresponding_topology : Int[Array, " "]
        An index into TopoVectBranchComputations, telling for which topology this injection combination
        is computed. Hence, all values are between [0, n_topologies]. This is the single value
        corresponding_topology[idx] for the idx-th injection candidate.
    n_injection_candidates_per_topology : Int[Array, " n_topologies"]
        The number of injection candidates for each topology, can be obtained by calling
        compute_number_of_injection_candidates
    cumsum_injection_candidates_per_topology : Optional[Int[Array, " n_topologies"]]
        The cumulative sum of injection candidates for each topology, can be obtained through
        jnp.cumsum(n_injection_candidates_per_topology). If not provided, it will be computed
        here.

    Returns
    -------
    Int[Array, " "]
        An index into the injection candidates for the corresponding topology
        Ranges from 0 to n_injection_combinations_per_topology[corresponding_topology]
    """
    if cumsum_injection_candidates_per_topology is None:
        cumsum_injection_candidates_per_topology = jnp.cumsum(n_injection_candidates_per_topology)

    our_topology = corresponding_topology

    # For the first topology we don't need a correction
    # For all others, we should subtract the cumsum of indices before this topology
    first_run = our_topology == 0

    local_idx = jnp.where(
        first_run,
        idx,
        (idx - cumsum_injection_candidates_per_topology[corresponding_topology - 1]),
    )

    return local_idx


def pad_out_has_splits(
    has_splits: Bool[Array, " n_topologies n_subs_limited"],
    sub_ids: Int[Array, " n_topologies n_subs_limited"],
    n_rel_subs: int,
) -> Bool[Array, " n_topologies n_rel_subs"]:
    """Pad out the has_splits array to full width, i.e. n_rel_subs

    has_splits could be of a lower size than n_sub_relevant if sub limiting is active, so we need
    to map it to full width for generate_injection_candidate

    Parameters
    ----------
    has_splits : Bool[Array, " n_topologies n_subs_limited"]
        Whether the substation has a split or not for the currently computed topology for the
        potentially limited substation selection
    sub_ids : Int[Array, " n_topologies n_subs_limited"]
        The substation ids for the currently computed topology
    n_rel_subs : int
        The number of relevant substations. If n_rel_subs == n_subs_limited, this function is a
        no-op

    Returns
    -------
    Bool[Array, " n_topologies n_rel_subs"]
        Whether the substation has a split or not for the currently computed topology, for all
        relevant substations
    """
    if has_splits.shape[1] == n_rel_subs:
        return has_splits
    if has_splits.shape[1] > n_rel_subs:
        raise ValueError("has_splits is larger than n_rel_subs, this should not happen")

    def update_has_splits_row(
        sub_ids: Int[Array, " n_subs_limited"],
        has_splits_row_local: Bool[Array, " n_subs_limited"],
    ) -> Int[Array, " n_rel_subs"]:
        buf = jnp.zeros(n_rel_subs, dtype=bool)
        # Sub ids will be int_max for some unsplit substations
        return buf.at[sub_ids].set(has_splits_row_local, mode="drop")

    has_splits: Bool[Array, " n_topologies, n_rel_subs"] = jax.vmap(
        update_has_splits_row,
        in_axes=(0, 0),
    )(sub_ids, has_splits)
    return has_splits


def default_injection(
    n_splits: int, max_inj_per_sub: int, batch_size: int, buffer_size: Optional[int] = None
) -> InjectionComputations:
    """Get a batch of injection computations without any assignments to bus B (all bus A)

    The corresponding topologies will be arange(batch_size)

    Parameters
    ----------
    n_splits : int
        The number of splits in the grid
    max_inj_per_sub : int
        The maximum number of injections per substation
    batch_size : int
        The number of topologies to generate
    buffer_size: Optional[int]
        If given, adds a leading buffer size dimension

    Returns
    -------
    InjectionComputations
        All-default injection computation with the unsplit injection candidate selected for every
        corresponding topology
    """
    if buffer_size is None:
        return InjectionComputations(
            corresponding_topology=jnp.arange(batch_size),
            injection_topology=jnp.zeros((batch_size, n_splits, max_inj_per_sub), dtype=bool),
            pad_mask=jnp.ones(batch_size, dtype=bool),
        )
    return InjectionComputations(
        corresponding_topology=jnp.arange(batch_size)[None].repeat(buffer_size, axis=0),
        injection_topology=jnp.zeros((buffer_size, batch_size, n_splits, max_inj_per_sub), dtype=bool),
        pad_mask=jnp.ones((buffer_size, batch_size), dtype=bool),
    )


def random_injection_for_topology(
    rng_key: jax.random.PRNGKey,
    branch_topology: Bool[Array, " batch_size n_splits max_branch_per_sub"],
    sub_ids: Int[Array, " batch_size n_splits"],
    n_inj_combis: Int[Array, " n_sub_relevant"],
    n_inj_per_topology: int = 1,
) -> tuple[Int[Array, " batch_size*n_inj_per_topology n_splits"], Int[Array, " batch_size*n_inj_per_topology"]]:
    """Get a batch of random injections as an index into n_inj_combis

    Makes sure to only create non-zero injections where the branch topology has a split

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random key to use for sampling
    branch_topology : Bool[Array, " batch_size n_splits max_branch_per_sub"]
        The branch topology to sample injections for
    sub_ids : Int[Array, " batch_size n_splits"]
        The substation ids for each topology and split
    n_inj_combis : Int[Array, " n_sub_relevant"]
        The number of injection combinations for each substation
    n_inj_per_topology : int
        The number of injections to sample for each topology, defaults to 1 (symmetric)

    Returns
    -------
    Int[Array, " batch_size*n_inj_per_topology n_splits"]
        The sampled injections as an integer index into all possible actions at that sub (n_inj_combis)
    Int[Array, " batch_size*n_inj_per_topology"]
        The corresponding topology index for each injection
    """
    batch_size = branch_topology.shape[0]
    n_splits = branch_topology.shape[1]
    n_rel_subs = n_inj_combis.shape[0]

    has_splits = jnp.any(branch_topology, axis=2)
    # Overwrite unsplit subs with int_max
    # This way, _take_sampled_injection will return all false for unsplit subs
    sub_ids = jnp.where(has_splits, sub_ids, int_max())

    sampled_injections = jax.random.randint(
        key=rng_key,
        shape=(batch_size, n_inj_per_topology, n_rel_subs),
        minval=0,
        maxval=n_inj_combis,
    )

    # Subselect only the injections where the branch has a split
    def _take_sampled_injection(
        si: Int[Array, " n_inj_per_topology n_rel_subs"], sub_ids: Int[Array, " n_splits"]
    ) -> Int[Array, " n_inj_per_topology n_splits"]:
        return jax.vmap(lambda sub_id: si.at[:, sub_id].get(mode="fill", fill_value=False))(sub_ids)

    sampled_injections = jax.vmap(_take_sampled_injection)(sampled_injections, sub_ids)

    sampled_injections = jnp.reshape(sampled_injections, (batch_size * n_inj_per_topology, n_splits), order="C")
    corresponding_topology = jnp.repeat(jnp.arange(batch_size), n_inj_per_topology)

    return sampled_injections, corresponding_topology


def random_injection(
    rng_key: jax.random.PRNGKey,
    n_generators_per_sub: Int[Array, " n_subs_relevant"],
    n_inj_per_topology: int,
    for_topology: TopoVectBranchComputations,
) -> InjectionComputations:
    """Get a batch of random injections

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random key to use for sampling
    n_generators_per_sub : Int[Array, " n_subs_relevant"]
        The number of generators per substation
    n_inj_per_topology : int
        The number of injections to sample for each topology
    for_topology : TopoVectBranchComputations
        Only create non-zero injections where the branch topology has a split by passing the
        corresponding topology computations

    Returns
    -------
    InjectionComputations
        A batch of random injections
    """
    n_inj_combis = 2**n_generators_per_sub
    max_inj_per_sub = jnp.max(n_generators_per_sub).item()
    sampled_injection, corresponding_topology = random_injection_for_topology(
        rng_key=rng_key,
        branch_topology=for_topology.topologies,
        sub_ids=for_topology.sub_ids,
        n_inj_combis=n_inj_combis,
        n_inj_per_topology=n_inj_per_topology,
    )

    # Convert the integer sampled injections into binary form
    sampled_injections = jax.vmap(jax.vmap(partial(action_index_to_binary_form, max_degree=max_inj_per_sub)))(
        sampled_injection
    )

    return InjectionComputations(
        corresponding_topology=corresponding_topology,
        injection_topology=sampled_injections,
        pad_mask=for_topology.pad_mask[corresponding_topology],
    )


def convert_inj_candidates(
    inj_topologies: Bool[Array, " batch_size n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " batch_size n_splits"],
    n_generators_per_sub: Int[Array, " n_subs_relevant"],
) -> Bool[np.ndarray, " batch_size total_inj"]:
    """Convert the injection candidates from the padded jax topo vect to a dense numpy topo vect

    The dense numpy topo vect holds a boolean for every generator in any sub, ordered as in n_generators_per_sub,
    while the jax format splits and pads the topo vect to have a substation dimension

    Parameters
    ----------
    inj_topologies : Bool[Array, " batch_size n_splits max_inj_per_sub"]
        The padded jax topo vect, to be converted to dense numpy
    sub_ids : Int[Array, " batch_size n_splits"]
        The substation ids for each split
    n_generators_per_sub : Int[Array, " n_subs_relevant"]
        The number of generators per substation in the grid

    Returns
    -------
    Bool[np.ndarray, " batch_size total_inj"]
        The dense numpy topo vect for injections
    """
    assert sub_ids.shape == inj_topologies.shape[:2]
    batch_size = inj_topologies.shape[0]
    n_subs_rel = n_generators_per_sub.shape[0]

    output = np.zeros((batch_size, np.sum(n_generators_per_sub)), dtype=bool)
    gen_cumsum = np.cumsum(n_generators_per_sub)
    gen_cumsum = np.concatenate(([0], gen_cumsum))
    for sub in range(n_subs_rel):
        start_idx = gen_cumsum[sub]
        end_idx = gen_cumsum[sub + 1]
        split_mask = sub_ids == sub
        split_idx = np.argmax(split_mask, axis=1)
        split_found = np.any(split_mask, axis=1)
        output[split_found, start_idx:end_idx] = inj_topologies[
            split_found, split_idx[split_found], : n_generators_per_sub[sub]
        ]
    return output


def convert_action_index_to_numpy(
    action_index: Int[Array, " batch_size n_splits"],
    action_set: ActionSet,
    n_generators_per_sub: Int[Array, " n_subs_relevant"],
) -> Bool[np.ndarray, " batch_size total_inj"]:
    """Convert action-set index actions to numpy topo vects

    Parameters
    ----------
    action_index : Int[Array, " batch_size n_splits"]
        The batch of action indices, indexing into the action set, to convert to injection topologies
    action_set : ActionSet
        The action set to use for lookup
    n_generators_per_sub : Int[Array, " n_subs_relevant"]
        The number of generators per substation in the grid, for transforming to numpy topo vects

    Returns
    -------
    Bool[np.ndarray, " batch_size total_inj"]
        The dense numpy topo vect for injections
    """
    sub_ids = action_set.substation_correspondence.at[action_index].get(mode="fill", fill_value=int_max())
    inj_topos = action_set.inj_actions.at[action_index, :].get(mode="fill", fill_value=False)
    return convert_inj_candidates(inj_topos, sub_ids, n_generators_per_sub)


def convert_inj_topo_vect(
    numpy_topo_vect: Bool[np.ndarray, " batch_size total_inj"],
    sub_ids: Int[Array, " batch_size n_splits"],
    generators_per_sub: Int[Array, " n_subs_relevant"],
    missing_split_behavior: Literal["zero", "raise"] = "zero",
) -> Bool[Array, " batch_size n_splits max_inj_per_sub"]:
    """Convert the dense numpy topo vect to a padded jax topo vect

    The dense numpy topo vect holds a boolean for every generator in any sub, ordered as in n_generators_per_sub,
    while the jax format has a split substation dimension. The substation ids are usually stored in the branch
    topology vector, so for the backwards conversion they have to be passed in. It is possible that a substation is
    split in the injection numpy vector but not split in the branch topology vector, in which case this function
    will either raise or zero out the injection (put everything on bus A).

    This performs a symmetric conversion, meaning it requires exactly as many sub ids as numpy topo vects.

    Parameters
    ----------
    numpy_topo_vect : Bool[np.ndarray, " batch_size total_inj"]
        The dense numpy topo vect to be converted to padded jax
    sub_ids : Int[Array, " batch_size n_splits"]
        The substation ids of the branch topology vector
    generators_per_sub : Int[Array, " n_subs_relevant"]
        The number of generators per substation in the grid
    missing_split_behavior : Literal["zero", "raise"]
        The behavior to apply if a substation is split in the numpy topo vect but not in the branch topology subids vector.

    Returns
    -------
    Bool[Array, " batch_size n_splits max_inj_per_sub"]
        The padded jax topo vect for injections
    """
    assert sub_ids.shape[0] == numpy_topo_vect.shape[0]
    assert numpy_topo_vect.shape[1] == np.sum(generators_per_sub)

    batch_size = numpy_topo_vect.shape[0]
    n_subs_rel = generators_per_sub.shape[0]
    n_splits = sub_ids.shape[1]
    max_inj_per_sub = np.max(generators_per_sub)

    output = np.zeros((batch_size, n_splits, max_inj_per_sub), dtype=bool)
    gen_cumsum = np.cumsum(generators_per_sub)
    gen_cumsum = np.concatenate(([0], gen_cumsum))

    for sub in range(n_subs_rel):
        start_idx = gen_cumsum[sub]
        end_idx = gen_cumsum[sub + 1]
        inj_topo = numpy_topo_vect[:, start_idx:end_idx]

        split_subid_mask = sub_ids == sub
        split_subid_idx = np.argmax(split_subid_mask, axis=1)
        split_subid_found = np.any(split_subid_mask, axis=1)

        # Check for missing splits, i.e. substations where the injections have a split but the subids don't
        inj_split = np.any(inj_topo, axis=1)
        missing_splits = inj_split & ~split_subid_found
        if np.any(missing_splits) and missing_split_behavior == "raise":
            raise ValueError("Missing split in branch topology vector")

        output[split_subid_found, split_subid_idx[split_subid_found], : generators_per_sub[sub]] = inj_topo[
            split_subid_found, :
        ]
    return output
