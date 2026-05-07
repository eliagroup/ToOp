# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to work with the ActionIndexComputations and TopoVectBranchComputations dataclass.

Transforms between the different representations of branch topologies and
offering helper functions to create and split them.
"""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import List, Optional, Union
from jaxtyping import Array, ArrayLike, Bool, Int, PRNGKeyArray
from toop_engine_dc_solver.jax.branch_action_set import merge_topologies
from toop_engine_dc_solver.jax.injections import convert_action_index_to_numpy
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    ActionSet,
    SolverConfig,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
)
from toop_engine_dc_solver.jax.utils import HashableArrayWrapper


def convert_topo_to_action_set_index(
    topologies: TopoVectBranchComputations,
    branch_actions: ActionSet,
    extend_action_set: bool = True,
    fill_unsplit_with_int_max: bool = True,
) -> tuple[ActionIndexComputations, ActionSet]:
    """Convert topologies from the bitvector (topovect) format to the action set index format

    This function is not jittable. For a jittable version that uses a bit more memory, see
    convert_topo_to_action_set_index_jittable

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The topologies to convert, in padded bit vector format
    branch_actions : ActionSet
        The branch action set to use for the conversion
    extend_action_set : bool
        Whether to extend the action set in case there were missing actions in the topologies
        If set to false, the missing actions will be set to the unsplit action for their substation
        and the returned branch action set is always equal to the input branch action set
    fill_unsplit_with_int_max : bool
        Whether to fill the unsplit action with int_max or with their respective unsplit action from
        the action set

    Returns
    -------
    ActionIndexComputations
        The converted topologies in action set index format. Unsplit topologies will be set to int_max()
    ActionSet
        The branch action set used for the conversion
    """
    if extend_action_set and not is_in_action_set(topologies, branch_actions).all():
        # Treat the topology vector as a branch action set and merge it
        # Merge will automatically remove duplicates
        branch_actions = merge_topologies(
            action_set=branch_actions,
            topologies=topologies.topologies[topologies.pad_mask],
            sub_ids=topologies.sub_ids[topologies.pad_mask],
        )

    n_rel_subs = branch_actions.n_actions_per_sub.shape[0]

    new_actions = jnp.full((topologies.topologies.shape[0], topologies.topologies.shape[1]), int_max(), dtype=int)
    has_splits = jnp.any(topologies.topologies, axis=2)

    # Loop through all substations and try to find the actions belonging to that substation in the branch action set
    for sub in range(n_rel_subs):
        # Look at only the branch actions for this substation
        # This also means including an offset
        available_actions = branch_actions.branch_actions[branch_actions.substation_correspondence == sub]
        action_set_offset = jnp.sum(branch_actions.n_actions_per_sub[:sub])

        matching_action_mask: Bool[Array, " n_topologies n_splits n_branch_actions"] = jnp.all(
            topologies.topologies[:, :, None, :] == available_actions[None, None, :, :], axis=3
        )
        matching_sub_mask: Bool[Array, " n_topologies n_splits"] = topologies.sub_ids == sub
        # If it is both the correct substation and an action from the action set, we will copy it over
        copy_mask = jnp.logical_and(matching_action_mask.any(axis=2), matching_sub_mask)
        # Except it is the unsplit topology, then we will leave the int_max
        if fill_unsplit_with_int_max:
            copy_mask = jnp.logical_and(copy_mask, has_splits)
        # Argmax returns the index of the first true element. If there is no true element, copy_mask will be false anyway
        copy_index = jnp.argmax(matching_action_mask, axis=-1)
        new_actions = jnp.where(copy_mask, copy_index + action_set_offset, new_actions)

    return ActionIndexComputations(
        action=new_actions,
        pad_mask=topologies.pad_mask,
    ), branch_actions


def is_in_action_set(
    topologies: TopoVectBranchComputations,
    branch_actions: ActionSet,
) -> Bool[Array, " n_topologies n_splits"]:
    """Check for a set of bitvector topologies whether they are in the branch action set

    Topologies that are unsplit or have an invalid substation id will always return True.

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The topologies to check, in padded bit vector format
    branch_actions : ActionSet
        The branch action set to use for the check

    Returns
    -------
    Bool[Array, " n_topologies n_splits"]
        Whether the topologies are in the action set
    """
    n_rel_subs = branch_actions.n_actions_per_sub.shape[0]

    # Loop through all substations and try to find the actions belonging to that substation in the branch action set
    def _check_single_sub(
        sub: Int[Array, " "], is_in: Bool[Array, " n_topologies n_splits"]
    ) -> Bool[Array, " n_topologies n_splits"]:
        available_action_mask = branch_actions.substation_correspondence == sub

        matching_action_mask: Bool[Array, " n_topologies n_splits n_actions"] = jnp.all(
            topologies.topologies[:, :, None, :] == branch_actions.branch_actions[None, None, :, :], axis=3
        )
        matching_action_mask = (matching_action_mask & available_action_mask[None, None, :]).any(axis=2)
        sub_out_of_bounds = (topologies.sub_ids < 0) | (topologies.sub_ids >= n_rel_subs)
        is_unsplit = jnp.all(~topologies.topologies, axis=2)
        matching_sub_mask: Bool[Array, " n_topologies n_splits"] = topologies.sub_ids == sub
        # If it is both the correct substation and an action from the action set, we will copy it over
        is_in_local = (matching_action_mask & matching_sub_mask) | sub_out_of_bounds | is_unsplit
        return jnp.where(is_in_local, True, is_in)

    return jax.lax.fori_loop(
        lower=0,
        upper=n_rel_subs,
        body_fun=_check_single_sub,
        init_val=jnp.full((topologies.topologies.shape[0], topologies.topologies.shape[1]), False, dtype=bool),
    )


def convert_topo_to_action_set_index_jittable(
    topologies: TopoVectBranchComputations,
    branch_actions: ActionSet,
    fill_unsplit_with_int_max: bool = True,
) -> ActionIndexComputations:
    """Convert the convert_topo_to_action_set_index to a jittable version.

    This does not support extending the action set and will be slightly more memory intensive, however it works on GPU

    Parameters
    ----------
    topologies : TopoVectBranchComputations
        The topologies to convert, in padded bit vector format
    branch_actions : ActionSet
        The branch action set to use for the conversion
    fill_unsplit_with_int_max : bool
        Whether to fill the unsplit action with int_max or with their respective unsplit action from

    Returns
    -------
    ActionIndexComputations
        The converted topologies in action set index format
    """
    if topologies.sub_ids.ndim == 3:
        return jax.vmap(convert_topo_to_action_set_index_jittable, in_axes=(0, None))(topologies, branch_actions)

    n_rel_subs = branch_actions.n_actions_per_sub.shape[0]

    # Matching_action_mask is quite large
    # TODO check if a reduced-memory version is sensible that reduces the available actions to be equal to the
    # highest number of actions for any substation.
    matching_action_mask: Bool[Array, " n_topologies n_splits n_branch_actions"] = jnp.all(
        topologies.topologies[:, :, None, :] == branch_actions.branch_actions[None, None, :, :], axis=3
    )
    has_splits: Bool[Array, " n_topologies n_splits"] = jnp.any(topologies.topologies, axis=2)

    # Loop through all substations and try to find the actions belonging to that substation in the branch action set
    def _match_single_sub(
        sub: Int[Array, ""], new_actions: Int[Array, " n_topologies n_splits"]
    ) -> Int[Array, " n_topologies n_splits"]:
        # This time we keep the shapes constant, but need to AND the matching_action_mask with the available_actions_mask.
        available_actions_mask: Bool[Array, " n_branch_actions"] = branch_actions.substation_correspondence == sub

        sub_matching_action_mask = jnp.logical_and(matching_action_mask, available_actions_mask[None, None, :])
        matching_sub_mask: Bool[Array, " n_topologies n_splits"] = topologies.sub_ids == sub
        # If it is both the correct substation and an action from the action set, we will copy it over
        copy_mask = jnp.logical_and(sub_matching_action_mask.any(axis=2), matching_sub_mask)
        # Argmax returns the index of the first true element. If there is no true element, copy_mask will be false anyway
        copy_index = jnp.argmax(sub_matching_action_mask, axis=-1)
        action_index = jnp.where(copy_mask, copy_index, new_actions)
        if fill_unsplit_with_int_max:
            action_index = jnp.where(has_splits, action_index, int_max())
        return action_index

    new_actions = jax.lax.fori_loop(
        lower=0,
        upper=n_rel_subs,
        body_fun=_match_single_sub,
        init_val=jnp.full((topologies.topologies.shape[0], topologies.topologies.shape[1]), int_max(), dtype=int),
    )

    return ActionIndexComputations(
        action=new_actions,
        pad_mask=topologies.pad_mask,
    )


def get_bitvec_from_action_set(
    branch_actions: ActionSet, action: Int[Array, " "]
) -> tuple[Bool[Array, " max_branch_per_sub"], Int[Array, ""]]:
    """Take a single action from the branch action set as a bitvector topo vect

    This is intended to be used with jax.vmap

    Parameters
    ----------
    branch_actions : ActionSet
        The branch action set to take the action from
    action : Int[Array, " "]
        The action to take

    Returns
    -------
    Bool[Array, " max_branch_per_sub"]
        The action in bitvector format. If the action is invalid, this will be all zeros
    Int[Array, ""]
        The substation id of the action. If the action is invalid, this will be set to int_max()
    """
    topology = branch_actions.branch_actions.at[action].get(mode="fill", fill_value=False)
    sub_id = branch_actions.substation_correspondence.at[action].get(mode="fill", fill_value=int_max())
    return topology, sub_id


def convert_action_set_index_to_topo(
    topologies: ActionIndexComputations,
    action_set: ActionSet,
) -> TopoVectBranchComputations:
    """Convert a set of topologies in action set index format into the topo-vect bitvector format

    This amounts essentially to an array lookup.
    Actions that are not found in the branch action set will be set to the unsplit action for an invalid substation.
    This function is jittable.

    Parameters
    ----------
    topologies : ActionIndexComputations
        The topologies in action set index format
    action_set : ActionSet
        The branch action set to use for the conversion

    Returns
    -------
    TopoVectBranchComputations
        The topologies in bitvector format
    """
    topology, sub_id = jax.vmap(jax.vmap(jax.tree_util.Partial(get_bitvec_from_action_set, action_set)))(topologies.action)
    return TopoVectBranchComputations(
        topologies=topology,
        sub_ids=sub_id,
        pad_mask=topologies.pad_mask,
    )


def extract_sub_ids(
    action: Int[Array, " *batch_size n_splits"],
    branch_actions: ActionSet,
) -> Int[Array, " *batch_size n_splits"]:
    """Extract just the substation ids from a batch of actions

    Will return int_max for unsplit actions
    This works for both batched and unbatched actions. The output will have the same shape as the input

    Parameters
    ----------
    action : Int[Array, " *batch_size n_splits"]
        The branch actions in action index format to extract the substation ids from
    branch_actions : ActionSet
        The branch action set to use for the extraction

    Returns
    -------
    Int[Array, " *batch_size n_splits"]
        The substation ids of the actions
    """
    if action.ndim == 1:
        _, sub_id = jax.vmap(jax.tree_util.Partial(get_bitvec_from_action_set, branch_actions))(action)
    else:
        _, sub_id = jax.vmap(jax.vmap(jax.tree_util.Partial(get_bitvec_from_action_set, branch_actions)))(action)
    return sub_id


def convert_topo_sel_sorted(
    topo_sel_sorted: Bool[ArrayLike, " n_topologies len_topo_vect"],
    branches_per_sub: HashableArrayWrapper[Int[ArrayLike, " n_sub_relevant"]],
    pad_to_size: Optional[int] = None,
) -> TopoVectBranchComputations:
    """Convert the topo_sel_sorted array holding the computations to TopoVectBranchComputations format.

    This will return a TopoVectBranchComputations dataclass indexing into the branch action set
    if the topo_sel_sorted was part of the branch action set.
    If not, the branch action set will be extended by the new actions.

    Parameters
    ----------
    topo_sel_sorted : Bool[ArrayLike, " n_topologies len_topo_vect"]
        The topology computations to be converted. This holds a boolean for every branch that is at a relevant substation
        anywhere in the grid. The len_topo_vect dimension is hence the total number of branch ends connected to the relevant
        subs. The substations are ordered as in the rel_stat_map.
    branches_per_sub : HashableArrayWrapper[Int[ArrayLike, " n_sub_relevant"]]
        The number of branches per substation. You can use static_information.solver_config.branches_per_sub
    pad_to_size : Optional[int]
        The size to which the computations should be padded. If None, the computations will have
        shape n_topologies. Note that not providing this might cause recompilations further
        downstream

    Returns
    -------
    TopoVectBranchComputations
        A populated TopoVectBranchComputations dataclass
    """
    branches_per_sub = branches_per_sub.val
    assert topo_sel_sorted.shape[1] == np.sum(branches_per_sub)
    topo_sel_sorted = jnp.array(topo_sel_sorted, dtype=bool)

    target_size = pad_to_size if pad_to_size is not None else topo_sel_sorted.shape[0]

    # First span out n_rel_subs topologies
    topologies = jnp.zeros(
        (
            target_size,
            branches_per_sub.shape[0],
            np.max(branches_per_sub),
        ),
        dtype=bool,
    )

    sub_ids = (
        jnp.repeat(
            jnp.arange(branches_per_sub.shape[0])[None, :],
            target_size,
            axis=0,
        )
        .at[topo_sel_sorted.shape[0] :]
        .set(0)
    )

    pad_mask = jnp.zeros(target_size, dtype=bool).at[: topo_sel_sorted.shape[0]].set(True)

    for (sub_id, len_sub), topo_vect_position in zip(
        enumerate(branches_per_sub),
        np.cumsum(branches_per_sub),
        strict=True,
    ):
        topologies = topologies.at[: topo_sel_sorted.shape[0], sub_id, :len_sub].set(
            topo_sel_sorted[:, topo_vect_position - len_sub : topo_vect_position]
        )

    return TopoVectBranchComputations(
        topologies=topologies,
        sub_ids=sub_ids,
        pad_mask=pad_mask,
    )


def convert_single_branch_topo_vect(
    topology: Bool[Array, " n_sub_limited max_branch_per_sub"],
    sub_ids: Int[Array, " n_sub_limited"],
    branches_per_sub: HashableArrayWrapper[Int[Array, " n_sub_relevant"]],
) -> Bool[Array, " len_topo_vect"]:
    """Convert a single branch topo vect from batched format back to the original topo-vect format

    Params
    ------
    topology: Bool[Array, " n_sub_limited max_branch_per_sub"]
        The topology in batched format.
    sub_ids: Int[Array, " n_sub_limited"]
        The substation ids of the batched topology.
    branches_per_sub: HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
        The number of branches per substation. (static)

    Returns
    -------
    Bool[Array, " len_topo_vect"]
        The topology in original format.
    """
    cumsum_branches_per_sub = np.concatenate([np.array([0], dtype=int), np.cumsum(branches_per_sub.val)])

    len_topo_vect = cumsum_branches_per_sub[-1].item()

    cumsum_branches_per_sub = jnp.array(cumsum_branches_per_sub)
    n_sub_limited = topology.shape[0]
    max_branch_per_sub = topology.shape[1]
    retval = jnp.zeros(len_topo_vect, dtype=bool)

    for sub_id_lim in range(n_sub_limited):
        valid_sub_id = (sub_ids[sub_id_lim] >= 0) & (sub_ids[sub_id_lim] < branches_per_sub.val.shape[0])

        offset_start = cumsum_branches_per_sub[sub_ids[sub_id_lim]]
        offset_end = cumsum_branches_per_sub[sub_ids[sub_id_lim] + 1]

        indices = jnp.arange(max_branch_per_sub) + offset_start
        indices = jnp.where(indices < offset_end, indices, int_max())
        indices = jnp.where(valid_sub_id, indices, int_max())

        retval = retval.at[indices].set(topology[sub_id_lim], mode="drop")

    return retval


def convert_branch_topo_vect(
    topologies: Bool[Array, " n_topologies n_sub_limited max_branch_per_sub"],
    sub_ids: Int[Array, " n_topologies n_sub_limited"],
    branches_per_sub: HashableArrayWrapper[Int[Array, " n_sub_relevant"]],
) -> Bool[Array, " n_topologies len_topo_vect"]:
    """Convert an array of topologies and sub_ids to the original topo-vect format

    This is the inverse of convert_topo_sel_sorted

    Params
    ------
    topologies: Bool[Array, " n_topologies n_sub_limited max_branch_per_sub"]
        The topologies in batched format.
    sub_ids: Int[Array, " n_topologies n_sub_limited"]
        The substation ids of the batched topologies.
    branches_per_sub: HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
        The number of branches per substation. (static) This needs to be a hashable numpy array so
        jax doesn't attempt to trace it.

    Returns
    -------
    Bool[Array, " n_topologies len_topo_vect"]
        The topologies in original topo_sel_sorted format.
    """
    return jax.vmap(
        lambda top, subs: convert_single_branch_topo_vect(top, subs, branches_per_sub),
        in_axes=(0, 0),
        out_axes=0,
    )(topologies, sub_ids)


def default_topology(
    solver_config: SolverConfig, batch_size: Optional[int] = None, topo_vect_format: bool = False
) -> Union[ActionIndexComputations, TopoVectBranchComputations]:
    """Get a batch of default, unsplit topology computation

    This is a helper that you can use to quickly determine the base N-1 case.

    Parameters
    ----------
    solver_config: SolverConfig
        The solver config to take computation dimensions from
    batch_size : Optional[int]
        The number of topologies to generate. If None, batch_size_bsdf from static_information is
        used
    topo_vect_format: bool
        If True, the returned topology will be in topo_vect format. Otherwise, it will be in action index format

    Returns
    -------
    ActionIndexBranchComputations or TopoVectBranchComputations
        All-zero topology computation
    """
    batch_size = batch_size if batch_size is not None else solver_config.batch_size_bsdf

    if topo_vect_format:
        return TopoVectBranchComputations(
            topologies=jnp.zeros(
                (
                    batch_size,
                    solver_config.n_sub_relevant,
                    solver_config.max_branch_per_sub,
                ),
                dtype=bool,
            ),
            sub_ids=jnp.repeat(
                jnp.arange(solver_config.branches_per_sub.val.shape[0])[None, :],
                batch_size,
                axis=0,
            ),
            pad_mask=jnp.ones(batch_size, dtype=bool),
        )

    return ActionIndexComputations(
        action=jnp.full(
            (
                batch_size,
                1,
            ),
            int_max(),
            dtype=int,
        ),
        pad_mask=jnp.ones(batch_size, dtype=bool),
    )


def sample_action_index_from_branch_actions(
    rng_key: PRNGKeyArray,
    sub_id: Int[Array, " "],
    branch_action_set: ActionSet,
) -> Int[Array, " "]:
    """Sample a branch action from the branch action set, but return only the action index

    This never samples the unsplit action

    Parameters
    ----------
    rng_key : PRNGKeyArray
        The random key to use for sampling
    sub_id : Int[Array, " "]
        The substation ids to sample the branch actions for. If an invalid sub-id is passed, int_max is returned
    branch_action_set : ActionSet
        The branch action set to sample from

    Returns
    -------
    Int[Array, " "]
        A random branch action index for each substation. If sub_id is an invalid sub id, will return int_max
    """
    valid_sub_id = (sub_id >= 0) & (sub_id < branch_action_set.n_actions_per_sub.shape[0])
    safe_sub_id = jnp.where(valid_sub_id, sub_id, 0)

    action_start_indices = branch_action_set.action_start_indices
    n_available_actions = branch_action_set.n_actions_per_sub[safe_sub_id] - 1
    action_offset = action_start_indices[safe_sub_id] + 1
    random_action = jax.random.randint(
        key=rng_key,
        shape=(),
        minval=0,
        maxval=jnp.maximum(n_available_actions, 1),
    )
    new_branch_action = action_offset + random_action

    return jnp.where(valid_sub_id, new_branch_action, jnp.array(int_max()))


def sample_from_branch_actions(
    rng_key: PRNGKeyArray,
    sub_id: Int[Array, " "],
    branch_action_set: ActionSet,
) -> Bool[Array, " max_branches_per_sub"]:
    """Sample a random branch action from the pre-computed set of branch actions for a single substation

    This never samples the unsplit action with all zeros
    Also, converts the result into a topo vect

    Parameters
    ----------
    rng_key : PRNGKeyArray
        The random key to use for sampling
    sub_id : Int[Array, " "]
        The substation id to sample the branch action for. If an invalid sub-id is passed, all zeros
        are returned
    branch_action_set : ActionSet
        The branch action set to sample from

    Returns
    -------
    Bool[Array, " max_branches_per_sub"]
        A random branch action for the substation
    """
    new_branch_action = sample_action_index_from_branch_actions(rng_key, sub_id, branch_action_set)

    return branch_action_set.branch_actions.at[new_branch_action].get(mode="fill", fill_value=False)


def random_topology(
    rng_key: PRNGKeyArray,
    branch_action_set: ActionSet,
    limit_n_subs: Optional[int],
    batch_size: int,
    unsplit_prob: float = 0.1,
    topo_vect_format: bool = False,
) -> Union[ActionIndexComputations, TopoVectBranchComputations]:
    """Get a random topology from the action set

    if topo_vect_format is True, it will convert it to a TopoVectBranchComputations object,
    otherwise an ActionIndexComputations object will be returned

    Parameters
    ----------
    rng_key : PRNGKeyArray
        The random key to use for sampling
    branch_action_set : ActionSet
        The action set to sample from
    limit_n_subs : Optional[int]
        The number of substations to limit the topologies to. If None, a split is sampled for every relevant substation
        in the grid
    batch_size : int
        The number of topologies to sample
    unsplit_prob : float
        The probability to sample an unsplit substation. If 0, then always all split slots actually have a split.
        If higher than 0, some substations might have int_max sampled, which means no branch split
    topo_vect_format : bool
        Whether to return the topologies in topo_vect format (True) or in action index format (False), defaults to False

    Returns
    -------
    Union[ActionIndexComputations, TopoVectBranchComputations]
        The sampled topologies in the requested format
    """
    rng_key1, rng_key2, rng_key3 = jax.random.split(rng_key, 3)
    n_subs = branch_action_set.n_actions_per_sub.shape[0]

    if limit_n_subs is None:
        substation_choice = jnp.arange(n_subs)
        substation_choice = jnp.repeat(substation_choice[None], batch_size, axis=0)
    else:
        assert limit_n_subs <= n_subs
        rng_key1 = jax.random.split(rng_key1, batch_size)
        substation_choice = jax.vmap(
            lambda rng: jax.random.choice(
                rng,
                n_subs,
                shape=(limit_n_subs,),
                replace=False,
            )
        )(rng_key1)

    if unsplit_prob > 0:
        unsplit_mask = jax.random.bernoulli(rng_key2, unsplit_prob, shape=substation_choice.shape)
        substation_choice = jnp.where(unsplit_mask, int_max(), substation_choice)

    rng_key3 = jax.random.split(rng_key3, substation_choice.shape)

    if topo_vect_format:
        sample_fn = partial(sample_from_branch_actions, branch_action_set=branch_action_set)
        topos = jax.vmap(jax.vmap(sample_fn))(rng_key3, substation_choice)

        return TopoVectBranchComputations(
            topologies=topos,
            sub_ids=substation_choice,
            pad_mask=jnp.ones(batch_size, dtype=bool),
        )
    sample_fn = partial(sample_action_index_from_branch_actions, branch_action_set=branch_action_set)
    actions = jax.vmap(jax.vmap(sample_fn))(rng_key3, substation_choice)

    return ActionIndexComputations(
        action=actions,
        pad_mask=jnp.ones(batch_size, dtype=bool),
    )


def pad_action_with_unsplit_action_indices(
    action_set: ActionSet, action_indices: Int[Array, " n_allowed_splits"]
) -> Int[Array, " n_rel_subs"]:
    """Update the branch action indices by padding with unsplit action indices.

    Updates the branch action indices by padding the action indices with unsplit action indices for the
    substations that are not explicitly provided in the input.
    This function is used to ensure that the action indices are consistent with the number of relevant substations

    Parameters
    ----------
    action_set : ActionSet
        An object containing information about the actions and their correspondence to substations.
        It includes attributes such as `n_actions_per_sub` (number of actions per substation)
        and `substation_correspondence` (mapping of branch indices to substations).
    action_indices : Int[Array, " n_allowed_splits"]
        An array of branch action indices for which actions are explicitly provided.
        The size of this array may be smaller than the total number of relevant substations.
        This array can also contain int_max() which signifies that the action corresponds
        to unsplit action.

    Returns
    -------
    Int[Array, " n_rel_subs"]
        An array of updated branch action indices, where default action indices are assigned
        to substations that were not explicitly provided in the input.

    Example:
    --------
    Imagine there are 3 rel_subs, and n_allowed_splits is set to 2. If the input action_indices is [5, int_max], then
    the output will be [5, 7, 14] where 7 is the action index corresponding to the unsplit_action for the 2nd relevant
    sub. Likewise, 14 is the action index corresponding to the unsplit_action for the 3rd relevant sub. Note that these
    indices index into DynamicInformation.ActionSet.
    """
    assert action_indices.shape[0] > 0

    n_rel_subs = action_set.n_actions_per_sub.shape[0]
    rel_sub_indices = jnp.arange(n_rel_subs)
    cumsum_n_actions = jnp.cumsum(action_set.n_actions_per_sub)

    unsplit_action_indices_rel_subs = jax.vmap(
        lambda rel_sub_index: jax.lax.cond(
            rel_sub_index != 0,
            lambda idx: cumsum_n_actions[idx - 1],
            lambda _: jnp.array(0, dtype=cumsum_n_actions.dtype),
            rel_sub_index,
        )
    )(rel_sub_indices)

    # get the sub_indices for each of these branch_action_indices
    substation_correspondence = action_set.substation_correspondence.at[action_indices].get(
        mode="fill", fill_value=int_max()
    )

    updated_branch_action_indices = jax.vmap(
        lambda sub_index: jnp.where(
            jnp.isin(sub_index, substation_correspondence),
            action_indices.at[jnp.argmax(substation_correspondence == sub_index)].get(),
            unsplit_action_indices_rel_subs.at[sub_index].get(),
        )
    )(rel_sub_indices)

    return updated_branch_action_indices


def get_random_topology_results(static_information: StaticInformation, random_seed: int = 42) -> List[dict]:
    """Get random topology results for testing purposes.

    The amount is dependent on the static_information.solver_config.batch_size_bsdf.

    Parameters
    ----------
    static_information : StaticInformation
        The static information containing the action set and solver configuration.
    random_seed : int, optional
        The random seed to use for reproducibility, by default 42

    Returns
    -------
    List[dict]
        A list of dictionaries containing the best actions, branch topologies, injections, and disconnections
        in a format similar to the optimizer output.
    """
    best_actions = random_topology(
        jax.random.PRNGKey(random_seed),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=static_information.solver_config.limit_n_subs,
        batch_size=static_information.solver_config.batch_size_bsdf,
        unsplit_prob=0.0,
        topo_vect_format=False,
    )
    best_branch = convert_action_set_index_to_topo(
        topologies=best_actions,
        action_set=static_information.dynamic_information.action_set,
    )
    best_branch = convert_branch_topo_vect(
        best_branch.topologies,
        best_branch.sub_ids,
        static_information.solver_config.branches_per_sub,
    )
    best_injections = convert_action_index_to_numpy(
        action_index=best_actions.action,
        action_set=static_information.dynamic_information.action_set,
        n_generators_per_sub=static_information.dynamic_information.generators_per_sub,
    )
    best_disconnections = jax.random.choice(
        jax.random.PRNGKey(0),
        len(static_information.dynamic_information.disconnectable_branches),
        shape=(static_information.solver_config.batch_size_bsdf, 1),
    )

    # Then save in a json file similar to the optimizer output
    best = [
        {
            "branch": b.tolist(),
            "injection": i.tolist(),
            "disconnection": d.tolist(),
            "actions": [int(x) for x in a if x < static_information.n_actions],
            "metrics": {"n_failures": 0},
        }
        for a, b, i, d in zip(best_actions.action, best_branch, best_injections, best_disconnections, strict=True)
    ]

    return best


def product_action_set(
    substations: list[int],
    branch_set: ActionSet,
    limit_n_subs: Optional[int] = None,
) -> TopoVectBranchComputations:
    """Create the product set of all actions for each of the substations

    This will explode pretty quickly, so don't use it for more than 3 substations

    Parameters
    ----------
    substations : list[int]
        The substations to create the product set for. This indexes into relevant substations only, not into all nodes.
    branch_set : ActionSet
        The branch action set to use
    limit_n_subs : Optional[int]
        The number of split substations to limit the topologies to. If None, no limit is applied

    Returns
    -------
    TopoVectBranchComputations
        The product set of all actions for the substations
    """
    n_actions_per_chosen_sub = [branch_set.n_actions_per_sub[i] for i in substations]
    actions_per_chosen_sub = [range(n) for n in n_actions_per_chosen_sub]

    # The local action and substation can be translated into an index into the action set like this:
    def action_index(local_action: int, substation: int) -> int:
        if substation == 0:
            return local_action
        return sum(branch_set.n_actions_per_sub[:substation].tolist()) + local_action

    def is_split(local_action: int, substation: int) -> bool:
        global_action = action_index(local_action, substation)
        return bool(~branch_set.unsplit_action_mask[global_action])

    topologies = []
    for combination in product(*actions_per_chosen_sub):
        if limit_n_subs is not None and sum(map(is_split, combination, substations)) > limit_n_subs:
            continue
        topology = np.zeros((len(substations), branch_set.branch_actions.shape[1]), dtype=bool)
        for sub_idx, action in enumerate(combination):
            topology[substations[sub_idx]] = branch_set.branch_actions[action_index(action, substations[sub_idx])]
        topologies.append(topology)

    return TopoVectBranchComputations(
        topologies=jnp.array(topologies),
        sub_ids=jnp.array([substations] * len(topologies)),
        pad_mask=jnp.ones(len(topologies), dtype=bool),
    )
