# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Action Set Creation and Filtering

This module provides functionality for creating, filtering, and managing action sets
for substations in a power grid. Action sets represent possible configurations of
branch and injection topologies, which are used in grid optimization and simulation.

The module includes methods for:
- Enumerating all possible branch actions for substations.
- Filtering actions to exclude invalid configurations, such as those that split the grid.
- Padding and unpadding action sets for compatibility with optimization routines.
- Determining injection topologies based on initial asset configurations.
- Removing substations without valid actions from the network data.

The action creation routines are flexible and can be customized to include or exclude
specific configurations based on user-defined rules or constraints.
"""

from functools import partial

import jax
import logbook
import numpy as np
from beartype.typing import Optional
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.bsdf import calc_bsdf, update_from_to_node
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix
from toop_engine_dc_solver.jax.multi_outages import build_modf_matrices
from toop_engine_dc_solver.jax.types import ActionSet
from toop_engine_dc_solver.preprocess.helpers.ptdf import (
    get_extended_ptdf,
)
from toop_engine_dc_solver.preprocess.helpers.switching_distance import min_hamming_distance_matrix
from toop_engine_dc_solver.preprocess.network_data import NetworkData, get_relevant_stations
from toop_engine_interfaces.asset_topology import Station
from toop_engine_interfaces.asset_topology_helpers import get_connected_assets
from toop_engine_interfaces.messages.preprocess.preprocess_commands import ReassignmentLimits

logger = logbook.Logger(__name__)


def make_action_repo(
    sub_degree: int,
    separation_set: Bool[np.ndarray, " n_configurations 2 n_assets"],
    exclude_isolations: bool = True,
    randomly_select: Optional[int] = None,
    limit_reassignments: Optional[int] = None,
) -> Bool[np.ndarray, " possible_configurations sub_degree"]:
    """Make a repo of all possible topology configurations that are electrically possible

    Excludes actions that can be represented by their inverse with fewer elements on busbar B.
    A big chunk of these actions will be invalid due to splits.

    Parameters
    ----------
    sub_degree : int
        The number of branches in the substation
    separation_set : Bool[np.ndarray, " n_configurations 2 n_assets"]
        The separation set for the substation.
    exclude_isolations : bool
        Whether to exclude actions that isolate a branch. Defaults to True.
    randomly_select : Optional[int]
        If given, only randomly_select actions will be enumerated, which are randomly drawn. If None,
        all actions will be exhaustively enumerated. Defaults to None.
    limit_reassignments : Optional[int]
        If given, the maximum number of reassignments to perform during the electrical reconfiguration.

    Returns
    -------
    Bool[Array, " possible_configurations sub_degree"]
        The repo of physically possible topology actions
    """
    # -1 because we concatenate the inverse only on demand.
    num_possible_splits = 2 ** (sub_degree - 1)
    if randomly_select is not None and randomly_select < num_possible_splits:
        action_range = np.random.choice(int(num_possible_splits), randomly_select)
        # We can't pass replace=False to np.random.choice because that would implement a massive
        # array internally, hence we just filter out duplicates here
        action_range = np.unique(action_range)
    else:
        action_range = np.arange(num_possible_splits)

    # The repo is just a binary representation of the numbers from 0 to num_possible_splits
    # However, we don't want to include the inverse of the actions, hence we only need to count
    # up to num_possible_splits / 2.
    repo = np.zeros((len(action_range), sub_degree), dtype=bool)
    for i in range(sub_degree):
        repo[:, i] = np.bitwise_and(np.right_shift(action_range, i), 1)

    # We also exclude all actions that isolate a branch if that's desired
    if exclude_isolations:
        num_bus_b = np.sum(repo, axis=1)
        repo = repo[(num_bus_b != 1) & (num_bus_b != sub_degree - 1), :]

    # We only want to keep the inverse that has fewer elements in another setup compared to the base config
    # Hence, we invert where the electrical switching distance is lesser for the inverse configuration
    # Currently we ignore the injections in the separation set for this computation
    starting_configurations = separation_set[:, 1, :sub_degree]
    min_distance = min_hamming_distance_matrix(repo, starting_configurations)
    min_distance_inverse = min_hamming_distance_matrix(~repo, starting_configurations)
    better_inverse = min_distance_inverse < min_distance
    repo[better_inverse, :] = ~repo[better_inverse, :]

    if limit_reassignments is not None and limit_reassignments < sub_degree:
        min_distance = np.min([min_distance, min_distance_inverse], axis=0)
        repo = repo[min_distance <= limit_reassignments, :]

    # Make sure the first combination is the unsplit action
    # The fixed assignments might have changed this
    unsplit_action = np.zeros((1, repo.shape[1]), dtype=bool)
    repo = np.vstack((unsplit_action, repo))

    return repo


def filter_splits_by_bridge_lookup(
    sub_id: int,
    repo: Bool[np.ndarray, " possible_configurations sub_degree"],
    network_data: NetworkData,
) -> Bool[np.ndarray, " filtered_configurations sub_degree"]:
    """Filter out actions that probably split the grid by checking for bridges isolated on a bus

    This method is less reliable than filter_splits_by_bsdf but faster, as it just involves a few
    lookups. The cases in which this method fails are when a grid has a large circle and the substation
    sits in this large circle. If it is split, a N-1 case somewhere else in the circle will split
    the grid, which is not detected by this method. For detecting these cases, use filter_splits_by_bsdf
    instead.

    Parameters
    ----------
    sub_id : int
        The relevant substation id that the actions in the repo are for. This indexes into relevant
        substations
    repo : Bool[np.ndarray, " possible_configurations sub_degree"]
        The repo of possible branch topology actions
    network_data : NetworkData
        The network data of the grid, including the bridge branch information

    Returns
    -------
    Bool[np.ndarrary, " filtered_configurations sub_degree"]
        The filtered repo of possible branch topology actions
    """
    bridges_at_station = np.array(
        [network_data.bridging_branch_mask[branch] for branch in network_data.branches_at_nodes[sub_id]]
    )
    assert len(bridges_at_station) == repo.shape[1]

    # We check for each action how many non-bridges are on every busbar
    non_bridge_on_a = np.sum(~repo & ~bridges_at_station, axis=1)
    non_bridge_on_b = np.sum(repo & ~bridges_at_station, axis=1)
    has_splits = np.any(repo, axis=1)

    # We filter out all actions that have a split but fewer than 2 non-bridges on one busbar
    valid_mask = ~has_splits | ((non_bridge_on_a >= 2) & (non_bridge_on_b >= 2))
    return repo[valid_mask, :]


def is_valid_bsdf_lodf(  # noqa: PLR0913
    substation_topology: Bool[Array, " branches_at_sub"],
    ptdf: Float[Array, " n_branches n_bus"],
    i_stat: Int[Array, ""],
    i_stat_rel: Int[Array, ""],
    tot_stat: Int[Array, " branches_at_sub"],
    from_stat_bool: Bool[Array, " branches_at_sub"],
    to_node: Int[Array, " n_branches"],
    from_node: Int[Array, " n_branches"],
    susceptance: Float[Array, " n_branches"],
    slack: Int[Array, ""],
    n_stat: Int[Array, ""],
    branches_to_outage: Int[Array, " n_branches_to_outage"],
    multi_outage_branches: list[Int[Array, " n_multi_outages n_branches_failed"]],
) -> Bool[Array, ""]:
    """Check if a substation split is valid after both BSDF and LODF application

    Valid means it does not split the grid, and even the N-1 cases do not split the grid.
    Note that this currently does not implement multi-outages, which can also split the grid.

    Parameters
    ----------
    substation_topology : Bool[Array, " branches_at_sub"]
        The topology of the substation. The length of the vector does not need to be padded
        as long as it has the same length as tot_stat and from_stat_bool
    ptdf: Float[Array, " n_branches n_bus"]
        The PTDF matrix of the grid
    i_stat: Int[Array, ""]
        The index of the substation in the PTDF matrix node dimension
    i_stat_rel: Int[Array, ""]
        The index of the substation in the relevant nodes
    tot_stat: Int[Array, " branches_at_sub"]
        The indices of the branches leaving or entering the substation
    from_stat_bool: Bool[Array, " branches_at_sub"]
        The direction of the branches leaving or entering the substation
    to_node: Int[Array, " n_branches"]
        The to node of the branches
    from_node: Int[Array, " n_branches"]
        The from node of the branches
    susceptance: Float[Array, " n_branches"]
        The susceptance of the branches
    slack: Int[Array, ""]
        The index of the slack node
    n_stat: Int[Array, ""]
        The number of substations in the grid
    branches_to_outage: Int[Array, " n_branches_to_outage"]
        The indices of the branches to outage
    multi_outage_branches: list[Int[Array, " n_multi_outages n_branches_failed"]]
        The indices of the branches to outage in the multi-outage case

    Returns
    -------
    Bool[Array, ""]
        True if the split is valid, False otherwise
    """
    bsdf, ptdf_th_sw, success = calc_bsdf(
        substation_topology=substation_topology,
        ptdf=ptdf,
        i_stat_rel=i_stat_rel,
        i_stat=i_stat,
        tot_stat=tot_stat,
        from_stat_bool=from_stat_bool,
        to_node=to_node,
        from_node=from_node,
        susceptance=susceptance,
        slack=slack,
        n_stat=n_stat,
    )

    to_node, from_node = update_from_to_node(
        substation_topology=substation_topology,
        tot_stat=tot_stat,
        from_stat_bool=from_stat_bool,
        i_stat_rel_id=i_stat_rel,
        to_node=to_node,
        from_node=from_node,
        n_stat=n_stat,
    )

    ptdf = ptdf + jnp.outer(bsdf, ptdf_th_sw)

    _, lodf_success = calc_lodf_matrix(
        branches_to_outage=branches_to_outage, ptdf=ptdf, from_node=from_node, to_node=to_node, branches_monitored=None
    )

    _, modf_success = build_modf_matrices(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        multi_outage_branches=multi_outage_branches,
    )

    return success & jnp.all(lodf_success) & jnp.all(modf_success)


def filter_splits_by_bsdf(
    sub_id: int,
    repo: Bool[np.ndarray, " possible_configurations sub_degree"],
    network_data: NetworkData,
    batch_size: int = 8,
) -> Bool[np.ndarray, " filtered_configurations sub_degree"]:
    """Filter splits by applying the BSDF and then the LODF

    This will detect if an assignment renders the grid non-N-1 safe. However, this is a
    relatively costly computation, so it might be desirable to leave out this filtering
    step.

    This assumes the un-extended PTDF. The branch action set enumeration can not be run after the PTDF extension
    because the number of relevant nodes might change after the action set enumeration - stations with no actions are
    dropped. If the PTDF is extended before, then it would need to be de-extended to account for the lost relevant subs.
    It's easier to just extend the PTDF here in-place.

    Parameters
    ----------
    sub_id : int
        The relevant substation id that the actions in the repo are for. This indexes into relevant
        substations
    repo : Bool[np.ndarray, " possible_configurations sub_degree"]
        The repo of possible branch topology actions
    network_data : NetworkData
        The network data of the grid
    batch_size : int
        The batch size for the BSDF and LODF computation

    Returns
    -------
    Bool[np.ndarray, " filtered_configurations sub_degree"]
        The filtered repo of possible branch topology actions
    """
    assert network_data.ptdf_is_extended is False, "This assumes an un-extended ptdf"
    assert network_data.split_multi_outage_branches is not None, "Process multi-outages first"
    ptdf = jnp.array(get_extended_ptdf(network_data.ptdf, network_data.relevant_node_mask))

    # Gather some data needed for the BSDF computation and curry it to is_valid_bsdf_lodf
    tot_stat = network_data.branches_at_nodes[sub_id]
    from_stat_bool = network_data.branch_direction[sub_id]

    assert tot_stat.shape == from_stat_bool.shape == (repo.shape[1],)
    assert network_data.split_multi_outage_branches is not None

    is_valid_fn = partial(
        is_valid_bsdf_lodf,
        ptdf=ptdf,
        i_stat=jnp.array(network_data.relevant_nodes[sub_id]),
        i_stat_rel=jnp.array(sub_id),
        tot_stat=jnp.array(tot_stat),
        from_stat_bool=jnp.array(from_stat_bool),
        to_node=jnp.array(network_data.to_nodes),
        from_node=jnp.array(network_data.from_nodes),
        susceptance=jnp.array(network_data.susceptances),
        slack=jnp.array(network_data.slack),
        n_stat=jnp.array(network_data.n_original_nodes),
        branches_to_outage=jnp.flatnonzero(network_data.outaged_branch_mask),
        multi_outage_branches=[jnp.array(x) for x in network_data.split_multi_outage_branches],
    )

    # Vmap doesn't work due to memory issues.
    valid_mask = jax.lax.map(
        is_valid_fn,
        repo,
        batch_size=batch_size,
    )
    has_splits = np.any(repo, axis=1)
    valid_mask = valid_mask | ~has_splits
    return repo[valid_mask, :]


def enumerate_branch_actions_for_sub(
    sub_id: int,
    network_data: NetworkData,
    exclude_isolations: bool = True,
    exclude_bridge_lookup_splits: bool = True,
    exclude_bsdf_lodf_splits: bool = False,
    bsdf_lodf_batch_size: int = 8,
    clip_to_n_actions: int = 2**20,
    limit_reassignments: Optional[int] = None,
) -> Bool[np.ndarray, " n_configurations sub_degree"]:
    """Enumerate all combinations for one substation, optionally excluding some combinations

    Parameters
    ----------
    sub_id : int
        The relevant substation id that the actions in the repo are for. This indexes into relevant
        substations
    network_data : NetworkData
        The network data of the grid
    separation_set : Optional[Bool[np.ndarray, " n_configurations 2 n_assets"]]
        The separation set for the substation. If None, the default unseparated configuration
    exclude_isolations : bool
        Whether to exclude actions that isolate a branch
    exclude_bridge_lookup_splits : bool
        Whether to exclude actions that split the grid by isolating bridges
    exclude_bsdf_lodf_splits : bool
        Whether to exclude actions that split the grid after applying the BSDF and LODF formulas.
        Note that this is very costly, so it might be desirable to leave this out.
    bsdf_lodf_batch_size : int
        The batch size for the BSDF and LODF computation if enabled.
    clip_to_n_actions : int
        The maximum number of actions to return. If the number of actions is anticipated to be
        larger than this, a random subset will be returned. Defaults to 2**20.
    limit_reassignments : Optional[int]
        If given, the maximum number of reassignments to perform during the electrical reconfiguration.

    Returns
    -------
    Bool[Array, " n_configurations sub_degree"]
        The possible branch actions for the substation
    """
    sub_degree = len(network_data.branches_at_nodes[sub_id])
    assert sub_degree >= 4, "Substation has less than 4 branches, this should have been filtered out"
    # -1 because the inverse configuration is not included in the action set
    effective_degree = sub_degree - 1
    randomly_select = None
    if 2**effective_degree > clip_to_n_actions:
        logger.warning(
            f"Substation {network_data.node_ids[network_data.relevant_nodes[sub_id]]} has "
            f"{sub_degree} branches. Resorting to random action enumeration."
        )
        randomly_select = clip_to_n_actions
    separation_set = network_data.separation_sets_info[sub_id].separation_set
    repo = make_action_repo(
        sub_degree,
        separation_set,
        exclude_isolations,
        randomly_select=randomly_select,
        limit_reassignments=limit_reassignments,
    )
    if exclude_bridge_lookup_splits:
        repo = filter_splits_by_bridge_lookup(sub_id, repo, network_data)
    if exclude_bsdf_lodf_splits:
        repo = filter_splits_by_bsdf(sub_id, repo, network_data, batch_size=bsdf_lodf_batch_size)

    return repo


def enumerate_branch_actions(
    network_data: NetworkData,
    exclude_isolations: bool = True,
    exclude_bridge_lookup_splits: bool = True,
    exclude_bsdf_lodf_splits: bool = False,
    bsdf_lodf_batch_size: int = 8,
    clip_to_n_actions: int = 2**20,
    reassignment_limits: Optional[ReassignmentLimits] = None,
) -> list[Bool[np.ndarray, " _ _"]]:
    """Enumerate all possible branch actions for all relevant substations in the network

    Parameters
    ----------
    network_data : NetworkData
        The network data of the grid
    exclude_isolations : bool
        Whether to exclude actions that isolate a branch
    exclude_bridge_lookup_splits : bool
        Whether to exclude actions that split the grid by isolating non-bridges
    exclude_bsdf_lodf_splits : bool
        Whether to exclude actions that split the grid after applying the BSDF and LODF formulas.
        Note that this is very costly, so it might be desirable to leave this out.
    bsdf_lodf_batch_size : int
        The batch size for the BSDF and LODF computation if enabled.
    clip_to_n_actions : int
        The maximum number of actions to return. If the number of actions is anticipated to be
        larger than this, a random subset will be returned. Defaults to 2**20.
    reassignment_limits : Optional[ReassignmentLimits]
        If given, settings to limit the amount of reassignment during the electrical reconfiguration.

    Returns
    -------
    list[Bool[Array, " _ _"]]
        A list of branch actions for each substation where each branch action set has been generated
        through enumerate_branch_actions_for_sub.
    """
    assert network_data.separation_sets_info is not None, "Separation sets must be computed first"
    # get id of relevant substations
    relevant_ids = np.array(network_data.node_ids)[network_data.relevant_node_mask]
    if reassignment_limits is not None:
        station_specific_reassignment_limits = reassignment_limits.station_specific_limits
        reassignment_limit = reassignment_limits.global_limit
    else:
        station_specific_reassignment_limits = {}
        reassignment_limit = None

    return [
        enumerate_branch_actions_for_sub(
            sub_id=sub_id,
            network_data=network_data,
            exclude_isolations=exclude_isolations,
            exclude_bridge_lookup_splits=exclude_bridge_lookup_splits,
            exclude_bsdf_lodf_splits=exclude_bsdf_lodf_splits,
            bsdf_lodf_batch_size=bsdf_lodf_batch_size,
            clip_to_n_actions=clip_to_n_actions,
            limit_reassignments=station_specific_reassignment_limits.get(grid_model_id, reassignment_limit),
        )
        for sub_id, grid_model_id in zip(range(sum(network_data.relevant_node_mask)), relevant_ids, strict=True)
    ]


def pad_out_action_set(
    branch_actions: list[Bool[np.ndarray, " n_branch_actions_per_sub n_assets_per_sub"]],
    injection_actions: list[Bool[np.ndarray, " n_injection_actions_per_sub n_assets_per_sub"]],
    reassignment_distance: list[Int[np.ndarray, " n_branch_actions_per_sub"]],
) -> ActionSet:
    """Pad out a list of actions to a fixed size and concatenate them into the action set.

    Both branch and injection actions are padded to their maximum degree.
    Note that this method does not add relevant busbar outage data to the action set.

    Parameters
    ----------
    branch_actions : list[Bool[Array, " n_branch_actions_per_sub n_assets_per_sub"]]
        The list of branch actions to pad out, where the outer list is per relevant substation and the inner array
        holds branch actions of different number of assets for each substation. The fist dimension is the size of the
        action set and the last one the number of assets in this station.
    injection_actions : list[Bool[Array, " n_injection_actions_per_sub n_assets_per_sub"]]
        The list of injection actions to pad out, where the outer list is per relevant substation and the inner array
        holds injection actions of different number of assets for each substation. The fist dimension is the size of the
        action set and the last one the number of assets in this station.
    reassignment_distance : list[Int[Array, " n_branch_actions_per_sub"]]
        The reassignment distance for each branch action, corresponding to the branch actions.

    Returns
    -------
    ActionSet
        The padded out action set
    """
    assert len(branch_actions) == len(reassignment_distance) == len(injection_actions)
    n_actions_per_sub = jnp.array([ba.shape[0] for ba in branch_actions])
    max_branches_per_sub = max(ba.shape[1] for ba in branch_actions)
    max_injections_per_sub = max(ia.shape[1] for ia in injection_actions)
    total_actions = sum(n_actions_per_sub)

    padded_branch_actions = np.zeros((total_actions, max_branches_per_sub), dtype=bool)
    padded_injection_actions = np.zeros((total_actions, max_injections_per_sub), dtype=bool)
    padded_reassignments = np.zeros((total_actions,), dtype=int)
    index = 0
    for sub_id in range(len(branch_actions)):
        action = branch_actions[sub_id]
        reassignment = reassignment_distance[sub_id]
        injection_action = injection_actions[sub_id]
        assert action.shape[0] == reassignment.shape[0] == injection_action.shape[0]

        padded_branch_actions[index : index + action.shape[0], : action.shape[1]] = action
        padded_injection_actions[index : index + action.shape[0], : injection_action.shape[1]] = injection_action
        padded_reassignments[index : index + action.shape[0]] = reassignment
        index += action.shape[0]

    substation_correspondence = np.repeat(np.arange(len(branch_actions)), n_actions_per_sub)
    unsplit_action_mask = ~jnp.any(padded_branch_actions, axis=1)

    branch_set = ActionSet(
        branch_actions=jnp.array(padded_branch_actions),
        n_actions_per_sub=n_actions_per_sub,
        substation_correspondence=jnp.array(substation_correspondence),
        unsplit_action_mask=unsplit_action_mask,
        reassignment_distance=jnp.array(padded_reassignments),
        inj_actions=jnp.array(padded_injection_actions),
        rel_bb_outage_data=None,
    )

    return branch_set


def unpad_branch_actions(
    branch_set: ActionSet,
    branches_per_sub: Int[Array, " "],
) -> list[Bool[Array, " _ _"]]:
    """Unpad a branch action set into a list of branch actions

    It's the inverse of pad_out_branch_actions

    Parameters
    ----------
    branch_set : BranchActionSet
        The branch action set to unpad
    branches_per_sub : Int[Array, " "]
        The number of branches per substation in the branch action set

    Returns
    -------
    list[Bool[Array, " _ _"]]
        The unpadded list of branch actions
    """
    branch_actions = []
    index = 0
    for n_actions, n_branches in zip(branch_set.n_actions_per_sub, branches_per_sub, strict=True):
        branch_actions.append(branch_set.branch_actions[index : index + n_actions, :n_branches])
        index += n_actions

    return branch_actions


def determine_injection_topology_sub(
    network_data: NetworkData,
    local_injection_idxs: Int[np.ndarray, " n_injections_at_node"],
    station: Station,
    n_local_branch_actions: int,
    local_busbar_a_mapping: list[list[int]],
    n_injections_at_node: int,
) -> Bool[np.ndarray, " n_local_actions n_injections_at_node"]:
    """Determine the injection topology for a single station based on branch actions and busbar mappings.

    Determines the injection_topology or injection_action that are required to be taken in order to
    get the injections as per the intial asset topolgy for a single station. As the branch_action determines
    the busbar_a_mapping for each station, therefore, an injection action is determined correponding
    to each branch_action. This method is called iteratively by the enumerate_injection_topology method
    to determine the injection topology for each station.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing injection and branch information.
    local_injection_idxs : list[int]
        List of local injection indices corresponding to the station.
    station : Station
        The station object containing information about busbars and connected assets.
    n_local_branch_actions : int
        Number of local branch actions to consider.
    local_busbar_a_mapping : list[list[int]]
        A mapping of busbars for each branch action, where each sublist contains indices of busbars.
    n_injections_at_node : int
        Number of injections at the node.

    Returns
    -------
    Bool[np.ndarray, " n_local_actions n_injections_at_node"]
        A boolean array where each row corresponds to a branch action, and each column corresponds
        to an injection. The value is `False` if the injection is connected to the busbar for the
        given branch action, otherwise `True`.

    """
    local_injection_set = np.ones((n_local_branch_actions, n_injections_at_node), dtype=bool)
    for branch_action_index in range(n_local_branch_actions):
        busbar_a_mapping = local_busbar_a_mapping[branch_action_index]
        # get the indexes of the injections connected busbar_a
        bba_connected_injection_ids = [
            asset.grid_model_id
            for bb_index in busbar_a_mapping
            for asset in get_connected_assets(station, bb_index)
            if not asset.is_branch()
        ]
        bba_connected_injection_idxs = [
            np.argmax(local_injection_idxs == network_data.injection_ids.index(injection_id))
            for injection_id in bba_connected_injection_ids
            if injection_id in network_data.injection_ids
        ]
        local_injection_set[branch_action_index, bba_connected_injection_idxs] = False
    return local_injection_set


def determine_injection_topology(
    network_data: NetworkData,
) -> list[Bool[np.ndarray, " n_local_actions n_injections_at_node"]]:
    """Calculate injection actions to align injections with the initial asset topology.

    Determines the injection actions that are required to be taken in order to
    get the injections as per the initial asset topology. When the optimization starts,
    the optimizer assumes all the injections to be allocated to busbar_a. However,
    the actual asset topology may have some injections connected to busbar_b as well.
    Therefore, to get a more accurate representation of the nodal_injections, we calculate
    the injection actions which when applied would get the injections as per the initial
    asset topology.

    As the branch_action determines the busbar_a_mapping for each station, therefore,
    an injection action is determined corresponding to each branch_action.

    Parameters
    ----------
    network_data : NetworkData
        The network data object containing information about the network's
        topology, branch action sets, busbar mappings, and injection indices.

    Returns
    -------
    list[Bool[np.ndarray, " n_local_actions n_injections_at_node"]]
        A list of boolean arrays, where each array represents the injection
        topology for a specific station. The shape of each array is
        (n_local_actions, n_injections_at_node).
    """
    injection_actions = []
    rel_stations = get_relevant_stations(network_data)
    for sub_idx, station in enumerate(rel_stations):
        n_local_branch_actions = len(network_data.branch_action_set[sub_idx])
        local_busbar_a_mapping = network_data.busbar_a_mappings[sub_idx]
        n_injections_at_node = len(network_data.injection_idx_at_nodes[sub_idx])
        local_injection_set = determine_injection_topology_sub(
            network_data,
            network_data.injection_idx_at_nodes[sub_idx],
            station,
            n_local_branch_actions,
            local_busbar_a_mapping,
            n_injections_at_node,
        )
        injection_actions.append(local_injection_set)

    return injection_actions
