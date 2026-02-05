# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Branch action set module."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int
from toop_engine_dc_solver.jax.types import ActionSet, RelBBOutageData


def merge_branch_action_sets(  # noqa: PLR0915
    a: ActionSet, b: ActionSet
) -> ActionSet:
    """Merge two branch action sets.

    The result will still be ordered by substation and hold only unique actions. This is not jittable.

    Parameters
    ----------
    a : BranchActionSet
        The first branch action set.
    b : BranchActionSet
        The second branch action set.

    Returns
    -------
    BranchActionSet
        The merged branch action set containing both actions from a and b.
    """
    actions = jnp.concatenate([a.branch_actions, b.branch_actions], axis=0)
    substation_correspondence = jnp.concatenate([a.substation_correspondence, b.substation_correspondence], axis=0)
    reassignment_distance = jnp.concatenate([a.reassignment_distance, b.reassignment_distance], axis=0)
    inj_actions = jnp.concatenate([a.inj_actions, b.inj_actions], axis=0)

    # Sort the actions by substation
    sorting_idx = jnp.argsort(substation_correspondence)
    actions = actions[sorting_idx]
    substation_correspondence = substation_correspondence[sorting_idx]
    reassignment_distance = reassignment_distance[sorting_idx]
    inj_actions = inj_actions[sorting_idx]

    if a.rel_bb_outage_data is not None and b.rel_bb_outage_data is not None:
        branch_outage_set = jnp.concatenate(
            [a.rel_bb_outage_data.branch_outage_set, b.rel_bb_outage_data.branch_outage_set], axis=0
        )
        deltap_set = jnp.concatenate([a.rel_bb_outage_data.deltap_set, b.rel_bb_outage_data.deltap_set], axis=0)
        nodal_indices = jnp.concatenate([a.rel_bb_outage_data.nodal_indices, b.rel_bb_outage_data.nodal_indices], axis=0)
        articulation_node_mask = jnp.concatenate(
            [a.rel_bb_outage_data.articulation_node_mask, b.rel_bb_outage_data.articulation_node_mask], axis=0
        )

        branch_outage_set = branch_outage_set[sorting_idx]
        deltap_set = deltap_set[sorting_idx]
        nodal_indices = nodal_indices[sorting_idx]
        articulation_node_mask = articulation_node_mask[sorting_idx]
    else:
        branch_outage_set = None
        deltap_set = None
        nodal_indices = None
        articulation_node_mask = None

    # Remove duplicates on a per-substation basis
    n_subs = a.n_actions_per_sub.shape[0]
    assert b.n_actions_per_sub.shape[0] == n_subs, "The number of substations must be the same in both branch action sets."
    actions_per_sub = []
    reassignment_per_sub = []
    inj_actions_per_sub = []
    branch_outage_set_per_sub = []
    deltap_set_per_sub = []
    nodal_indices_per_sub = []
    articulation_node_mask_per_sub = []

    for sub in range(n_subs):
        mask = substation_correspondence == sub
        unique_vals, unique_idx = jnp.unique(actions[mask], axis=0, return_index=True)
        actions_per_sub.append(unique_vals)
        reassignment_per_sub.append(reassignment_distance[mask][unique_idx])
        inj_actions_per_sub.append(inj_actions[mask][unique_idx])

        if a.rel_bb_outage_data is not None and b.rel_bb_outage_data is not None:
            branch_outage_set_per_sub.append(branch_outage_set[mask][unique_idx])
            deltap_set_per_sub.append(deltap_set[mask][unique_idx])
            nodal_indices_per_sub.append(nodal_indices[mask][unique_idx])
            articulation_node_mask_per_sub.append(articulation_node_mask[mask][unique_idx])

    # Concatenate it back together
    actions = jnp.concatenate(actions_per_sub, axis=0)
    n_actions_per_sub = jnp.array([len(actions_per_sub[i]) for i in range(n_subs)])
    substation_correspondence = jnp.concatenate(
        [jnp.full((len(actions_per_sub[i]),), i, dtype=int) for i in range(n_subs)], axis=0
    )
    reassignment_distance = jnp.concatenate(reassignment_per_sub, axis=0)
    inj_actions = jnp.concatenate(inj_actions_per_sub, axis=0)
    unsplit_action_mask = ~jnp.any(actions, axis=1)

    rel_bb_outage_data = None
    if a.rel_bb_outage_data is not None and b.rel_bb_outage_data is not None:
        branch_outage_set = jnp.concatenate(branch_outage_set_per_sub, axis=0)
        deltap_set = jnp.concatenate(deltap_set_per_sub, axis=0)
        nodal_indices = jnp.concatenate(nodal_indices_per_sub, axis=0)
        articulation_node_mask = jnp.concatenate(articulation_node_mask_per_sub, axis=0)
        rel_bb_outage_data = RelBBOutageData(
            branch_outage_set=branch_outage_set,
            deltap_set=deltap_set,
            nodal_indices=nodal_indices,
            articulation_node_mask=articulation_node_mask,
        )

    return ActionSet(
        branch_actions=actions,
        substation_correspondence=substation_correspondence,
        n_actions_per_sub=n_actions_per_sub,
        unsplit_action_mask=unsplit_action_mask,
        reassignment_distance=reassignment_distance,
        inj_actions=inj_actions,
        rel_bb_outage_data=rel_bb_outage_data,
    )


def merge_topologies(
    action_set: ActionSet,
    topologies: Bool[Array, " n_topologies n_splits max_branch_per_sub"],
    sub_ids: Int[Array, "n_topologies n_splits"],
    reassignment_distance_fill: int = 0,
    injection_fill: bool = False,
) -> ActionSet:
    """Merge the topologies into the branch action set if not already present

    This will fill reassignment distance and injection actions with default values, in case it is desired these should be
    added later on to represent sensible values.

    Parameters
    ----------
    action_set : BranchActionSet
        The branch action set to merge the topologies into.
    topologies : Bool[Array, "n_topologies n_splits max_branch_per_sub"]
        The topologies in topo-vect form to merge.
    sub_ids : Int[Array, "n_topologies n_splits"]
        The substation ids for each topology and split
    reassignment_distance_fill : int, optional
        The value to fill the reassignment distance with, by default 0
    injection_fill : bool, optional
        What to fill the injection actions with, default all False

    Returns
    -------
    BranchActionSet
        The branch action set with the topologies merged in.
    """
    max_branch_per_sub = action_set.branch_actions.shape[-1]
    max_inj_per_sub = action_set.inj_actions.shape[-1]

    fake_action_set = ActionSet(
        branch_actions=topologies.reshape((-1, max_branch_per_sub)),
        substation_correspondence=sub_ids.reshape(-1),
        n_actions_per_sub=jnp.zeros(action_set.n_actions_per_sub.shape),  # Not actually needed for merge
        unsplit_action_mask=jnp.array([]),  # Not actually needed for merge
        reassignment_distance=jnp.full(topologies.shape[0], reassignment_distance_fill, dtype=int),
        inj_actions=jnp.full((topologies.shape[0], max_inj_per_sub), injection_fill, dtype=bool),
    )

    return merge_branch_action_sets(action_set, fake_action_set)


def empty_branch_action_set(
    max_branch_per_sub: int,
    max_inj_per_sub: int,
    n_sub_relevant: int,
) -> ActionSet:
    """Get the empty branch action set with no actions.

    Parameters
    ----------
    max_branch_per_sub : int
        The maximum number of branches at any substation in the grid
    max_inj_per_sub : int
        The maximum number of injections at any substation in
    n_sub_relevant : int
        The number of substations in the grid

    Returns
    -------
    BranchActionSet
        The empty branch action set.
    """
    return ActionSet(
        branch_actions=jnp.zeros((0, max_branch_per_sub), dtype=bool),
        substation_correspondence=jnp.array([], dtype=int),
        n_actions_per_sub=jnp.zeros(n_sub_relevant, dtype=int),
        unsplit_action_mask=jnp.array([], dtype=bool),
        reassignment_distance=jnp.array([], dtype=int),
        inj_actions=jnp.zeros((0, max_inj_per_sub), dtype=bool),
    )
