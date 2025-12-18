# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The bsdf module, using jax.

Based on the Paper:
Bus Split Distribution Factors
DOI:10.36227/techrxiv.22298950.v1

Unified algebraic deviation of distribution factors in linear power flow
https://doi.org/10.48550/arXiv.2412.16164
"""

from functools import partial

import jax
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.types import BSDFResults, HashableArrayWrapper


def get_bus_data(
    substation_topology: Bool[Array, " max_branch_per_sub"],
    tot_stat: Int[Array, " max_branch_per_sub"],
    from_stat_bool: Bool[Array, " max_branch_per_sub"],
    return_bus_b: bool,
    n_branches: int,
) -> tuple[Int[Array, " max_branch_per_sub"], Int[Array, " max_branch_per_sub"]]:
    """Get index vectors indicating branches assigned to bus a/b current substation after the split.

    The numpy versions work with variable-shape index arrays, but for GPU compatibility we instead
    return padded arrays with int_max values in places where there is no more data.

    Parameters
    ----------
    substation_topology : Bool[Array, " max_branch_per_sub"]
        The topology vector for the current substation, used to determine on which bus the branches
        will be
    tot_stat : Int[Array, " max_branch_per_sub"]
        The static tot_stat information, containing the index of the from- and to-branches to this
        substation. Padded with int_max if the substation has less than max_branch_per_sub branches
    from_stat_bool : Bool[Array, " max_branch_per_sub"]
        The static from_stat_bool information, containing whether the items in the tot_stat array
        are the from-ends (true) or the to-ends(false). Padded with false if the substation has less
        than max_branch_per_sub branches
    return_bus_b : bool
        A boolean whether you're interested in bus a (false) or bus b (true)
    n_branches : int
        How many branches there are in the network, i.e. how wide the PTDF matrix is.

    Returns
    -------
    Int[Array, " max_branch_per_sub"]
        brh_to_bus, an index vector over all branches that indicates whether a to-end of a branch
        is connected to bus a/b. Padded with int_max if the bus has less than max_branch_per_sub
        branches
    Int[Array, " max_branch_per_sub"]
        brh_from_bus, an index vector over all branches that indicates whether a from-end of a
        branch is connected to bus a/b. Padded with int_max if the bus has less than
        max_branch_per_sub branches
    """
    max_branch_per_sub = tot_stat.shape[0]
    # Currently the substation topology is true on bus b, if we're interested in bus a invert.
    if not return_bus_b:
        substation_topology = ~substation_topology

    # tot_stat contains out of bound indices as it is padded with int_max, we will instruct jax to
    # drop operations on these indices through the mode parameter
    brh_to_bus_mask: Bool[Array, " n_branches"] = jnp.zeros(n_branches, dtype=bool)
    brh_to_bus_mask = brh_to_bus_mask.at[tot_stat].set(~from_stat_bool & substation_topology, mode="drop")
    brh_from_bus_mask: Bool[Array, " n_branches"] = jnp.zeros(n_branches, dtype=bool)
    brh_from_bus_mask = brh_from_bus_mask.at[tot_stat].set(from_stat_bool & substation_topology, mode="drop")

    brh_to_bus = jnp.nonzero(
        brh_to_bus_mask,
        size=max_branch_per_sub,
        fill_value=jnp.iinfo(tot_stat.dtype).max,
    )[0]
    brh_from_bus = jnp.nonzero(
        brh_from_bus_mask,
        size=max_branch_per_sub,
        fill_value=jnp.iinfo(tot_stat.dtype).max,
    )[0]

    return brh_to_bus, brh_from_bus


def get_bus_data_other(
    brh_to_bus: Int[Array, " max_branch_per_sub"],
    brh_from_bus: Int[Array, " max_branch_per_sub"],
    to_node: Int[Array, " n_branches"],
    from_node: Int[Array, " n_branches"],
) -> tuple[Int[Array, " max_branch_per_sub"], Int[Array, " max_branch_per_sub"]]:
    """Get index arrays over buses that indicate which buses are connected to the branches opposite ends.

    The numpy versions work with variable-shape index arrays, but for GPU compatibility we instead
    return padded arrays with int_max values in places where there is no more data.

    Note that the number of padded elements in the return values are the same as in the input.

    Parameters
    ----------
    brh_to_bus : Int[Array, " max_branch_per_sub"]
        An index vector indicating which to-ends of branches are connected to bus a/b. Padded with
        int_max if the bus has less than max_branch_per_sub branches
    brh_from_bus : Int[Array, " max_branch_per_sub"]
        An index vector indicating which from-ends of branches are connected to bus a/b. Padded with
        int_max if the bus has less than max_branch_per_sub branches
    to_node : Int[Array, " n_branches"]
        The to nodes of all branches
    from_node : Int[Array, " n_branches"]
        The from nodes of all branches

    Returns
    -------
    Int[Array, " max_branch_per_sub"]
        brh_to_other, an index vector over busbars that indicate where the from-ends of the branches
        in brh_to_bus are connected to. Padded with int_max if there is padding in brh_to_bus
    Int[Array, " max_branch_per_sub"]
        brh_from_other, an index vector over busbars that indicate where the to-ends of the branches
        in brh_from_bus are connected to. Padded with int_max if there is padding in brh_from_bus
    """
    brh_to_other = (
        from_node.astype(brh_to_bus.dtype).at[brh_to_bus].get(mode="fill", fill_value=jnp.iinfo(brh_to_bus.dtype).max)
    )
    brh_from_other = (
        to_node.astype(brh_to_bus.dtype).at[brh_from_bus].get(mode="fill", fill_value=jnp.iinfo(brh_to_bus.dtype).max)
    )

    return brh_to_other, brh_from_other


def calc_bsdf(  # noqa: PLR0913
    substation_topology: Bool[Array, " max_branch_per_sub"],
    ptdf: Float[Array, " n_branches n_bus"],
    i_stat: Int[Array, ""],
    i_stat_rel: Int[Array, ""],
    tot_stat: Int[Array, " max_branch_per_sub"],
    from_stat_bool: Bool[Array, " max_branch_per_sub"],
    to_node: Int[Array, " n_branches"],
    from_node: Int[Array, " n_branches"],
    # Static parameters
    susceptance: Float[Array, " n_branches"],
    slack: Int[Array, ""],
    n_stat: Int[Array, ""],
) -> tuple[Float[Array, " n_branches"], Float[Array, " n_bus"], Bool[Array, " "]]:
    """Calculate the bsdf vector and the ptdf_th_sw vector for a bus split.

    Parameters
    ----------
    substation_topology : Bool[Array, " max_branch_per_sub"]
        The topology vector for the current substation, used to determine on which bus the branches
        will be
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix, potentially changed by previous bus splits
    i_stat : Int[Array, ""]
        The index of the current substation before extending the ptdf, i.e. in case 14 this is
        between 0 and 13. This also corresponds to the first part of busbar indices in the ptdf.
    i_stat_rel: Int[Array, ""]
        The index of the current substation among the relevant substations, i.e. in case 14 there
        are 5 relevant substations and this is between 0 and 4. This also corresponds to the second
        part of busbar indices in the ptdf.
    tot_stat : Int[Array, " max_branch_per_sub"]
        The static tot_stat information, containing the index of the from- and to-branches to this
        substation. Padded with int_max if the substation has less than max_branch_per_sub branches
    from_stat_bool : Bool[Array, " max_branch_per_sub"]
        The static from_stat_bool information, containing whether the items in the tot_stat array
        are the from-ends (true) or the to-ends(false). Padded with false if the substation has less
        than max_branch_per_sub branches
    to_node : Int[Array, " n_branches"]
        The to nodes of all branches
    from_node : Int[Array, " n_branches"]
        The from nodes of all branches
    susceptance : Float[Array, " n_branches"]
        The susceptance of each branch
    slack : Int[Array, ""]
        The index of the slack bus
    n_stat : Int[Array, ""]
        The number of substations in the network before splitting/extending the ptdf. I.e. in
        case 14 there are n_stat=14 substations but because 5 substations can be split, there need
        to be 19 busbars.

    Returns
    -------
    Float[Array, " n_branches"]
        The bsdf vector
    Float[Array, " n_bus"]
        The ptdf_th_sw vector
    Bool[Array, " "]
        A boolean indicating whether the bsdf computation was successful. This is false if the
        computation failed due to a zero denominator, which can be caused by a split in the network
        that was missed during pre-processing
    """
    brh_to_bus_a, brh_from_bus_a = get_bus_data(
        substation_topology,
        tot_stat,
        from_stat_bool,
        return_bus_b=False,
        n_branches=ptdf.shape[0],
    )
    brh_to_other, brh_from_other = get_bus_data_other(
        brh_to_bus_a,
        brh_from_bus_a,
        to_node,
        from_node,
    )

    # Calculate the ptdf_th_sw - this is a vector of shape n_bus that holds for each bus what
    # enters minus what leaves bus a (bus 0) of the split substation
    ptdf_th_sw: Float[Array, " n_bus"] = jnp.sum(ptdf.at[brh_to_bus_a, :].get(mode="fill", fill_value=0), axis=0) - jnp.sum(
        ptdf.at[brh_from_bus_a, :].get(mode="fill", fill_value=0), axis=0
    )

    is_slack = i_stat == slack
    # If its the slack bus:
    # - Set the slack bus column to 0
    # - Subtract 1
    # If it's not the slack bus
    # - Plus 1 to the current bus if it's not the slack
    ptdf_th_sw -= is_slack
    ptdf_th_sw = ptdf_th_sw.at[i_stat].set(jnp.where(is_slack, 0, ptdf_th_sw[i_stat] + 1))

    # Second busbars (busbar b) are at the later part of the PTDF.
    busbar_b_index = n_stat + i_stat_rel
    # Calculate theoretical PEDF for switch to bus B
    ptdf_th_sl_bus_b: Float[Array, " n_bus"] = ptdf_th_sw - ptdf_th_sw[busbar_b_index]

    # Calculate the denominator of the bsdf
    # The original code used this formulation:
    # g_sw = susceptance[brh_from_bus_a] @ ptdf_th_sl_bus_b[brh_from_other]
    # g_sw += susceptance[brh_to_bus_a] @ ptdf_th_sl_bus_b[brh_to_other]
    # We rewrite it slightly to support out-of-bound indexing
    suscept_from_bus_a = susceptance.at[brh_from_bus_a].get(mode="fill", fill_value=0)
    suscept_to_bus_a = susceptance.at[brh_to_bus_a].get(mode="fill", fill_value=0)

    g_sw: Float[Array, " "] = jnp.dot(
        suscept_from_bus_a,
        ptdf_th_sl_bus_b.at[brh_from_other].get(mode="fill", fill_value=0),
    ) + jnp.dot(
        suscept_to_bus_a,
        ptdf_th_sl_bus_b.at[brh_to_other].get(mode="fill", fill_value=0),
    )

    # The original code used
    # susceptance[brh_from_bus_a].sum() + susceptance[brh_to_bus_a].sum() - g_sw
    denom: Float[Array, " "] = jnp.sum(suscept_from_bus_a) + jnp.sum(suscept_to_bus_a) - g_sw

    # denom = equinox.error_if(
    #     denom, jnp.abs(denom) < 1e-5, "BSDF failed. Please check your input topologies."
    # )
    success = jnp.abs(denom) >= 1e-5

    # PTDF to bus B
    ptdf_bus_b: Float[Array, " n_branches 1"] = ptdf[:, i_stat][:, None]
    pedf: Float[Array, " n_branches n_bus"] = ptdf - ptdf_bus_b

    # Index into intermediate - in the original code this was:
    # np.sum(susceptance[brh_from_bus_a] * pedf[:, brh_from_other], axis=1)
    # np.sum(susceptance[brh_to_bus_a] * pedf[:, brh_to_other], axis=1)
    # which is equivalent to
    # susceptance[brh_from_bus_a] @ pedf.T[brh_from_other, :]
    # susceptance[brh_to_bus_a] @ pedf.T[brh_to_other, :]
    # As the index arrays are padded, we need to use the get method to drop out-of-bound indices

    intermediate: Float[Array, " n_branches"] = jnp.dot(
        suscept_from_bus_a,
        pedf.T.at[brh_from_other, :].get(mode="fill", fill_value=0),
    ) + jnp.dot(
        suscept_to_bus_a,
        pedf.T.at[brh_to_other, :].get(mode="fill", fill_value=0),
    )

    nom = intermediate
    nom = nom.at[brh_from_bus_a].add(suscept_from_bus_a, mode="drop")
    nom = nom.at[brh_to_bus_a].add(-suscept_to_bus_a, mode="drop")

    bsdf = nom / denom

    return bsdf, ptdf_th_sw, success


def update_from_to_node(
    substation_topology: Bool[Array, " max_branch_per_sub"],
    tot_stat: Int[Array, " max_branch_per_sub"],
    from_stat_bool: Bool[Array, " max_branch_per_sub"],
    i_stat_rel_id: Int[Array, ""],
    to_node: Int[Array, " n_branches"],
    from_node: Int[Array, " n_branches"],
    n_stat: Int[Array, ""],
) -> tuple[Int[Array, " n_branches"], Int[Array, " n_branches"]]:
    """Compute updated from and to node vectors for a bus split.

    Parameters
    ----------
    substation_topology : Bool[Array, " max_branch_per_sub"]
        The topology vector for the current substation, used to determine on which bus the branches
        will be
    tot_stat : Int[Array, " max_branch_per_sub"]
        The static tot_stat information, containing the index of the from- and to-branches to this
        substation. Padded with int_max if the substation has less than max_branch_per_sub branches
    from_stat_bool : Bool[Array, " max_branch_per_sub"]
        The static from_stat_bool information, containing whether the items in the tot_stat array
        are the from-ends (true) or the to-ends(false). Padded with false if the substation has less
        than max_branch_per_sub branches
    i_stat_rel_id : Int[Array, ""]
        The index of the current substation among the relevant substations, i.e. in case 14 there
        are 5 relevant substations and this is between 0 and 4. This also corresponds to the second
        part of busbar indices in the ptdf.
    to_node : Int[Array, " n_branches"]
        The to nodes of all branches
    from_node : Int[Array, " n_branches"]
        The from nodes of all branches
    n_stat : Int[Array, ""]
        The number of substations in the network before splitting/extending the ptdf. I.e. in
        case 14 there are n_stat=14 substations but because 5 substations can be split, there need
        to be 19 busbars.

    Returns
    -------
    tuple[Int[Array, " n_branches"], Int[Array, " n_branches"]]
        The updated from and to node vectors
    """
    brh_to_bus_b, brh_from_bus_b = get_bus_data(
        substation_topology=substation_topology,
        tot_stat=tot_stat,
        from_stat_bool=from_stat_bool,
        return_bus_b=True,
        n_branches=to_node.shape[0],
    )

    new_bus = jnp.array(i_stat_rel_id + n_stat, dtype=from_node.dtype)

    from_node_new = from_node.at[brh_from_bus_b].set(new_bus, mode="drop")
    to_node_new = to_node.at[brh_to_bus_b].set(new_bus, mode="drop")

    return to_node_new, from_node_new


def init_bsdf_results(
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    n_splits: int,
) -> BSDFResults:
    """Initialize the BSDF results to an unsplit grid.

    Parameters
    ----------
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    n_splits : int
        The number of bus splits that will be performed

    Returns
    -------
    BSDFResults
        A bsdf results dataclass with the copied data
    """
    n_branches = ptdf.shape[0]
    return BSDFResults(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        success=jnp.array(True),
        bsdf=jnp.zeros((n_splits, n_branches)),
    )


def _apply_bus_split(
    current_results: BSDFResults,
    substation_configuration: Bool[Array, " max_branch_per_sub"],
    substation_id: Int[Array, " "],
    split_idx: Int[Array, " "],
    tot_stat: Int[Array, " n_sub_relevant max_branch_per_sub"],
    from_stat_bool: Bool[Array, " n_sub_relevant max_branch_per_sub"],
    susceptance: Float[Array, " n_branches"],
    rel_stat_map: HashableArrayWrapper[Int[Array, " n_sub_relevant"]],
    slack: int,
    n_stat: int,
) -> BSDFResults:
    """Like apply_bus_split, but assumes the current substation has a bus split."""
    i_stat = jnp.array(rel_stat_map.val)[substation_id]
    bsdf, ptdf_th_sw, success = calc_bsdf(
        substation_topology=substation_configuration,
        ptdf=current_results.ptdf,
        i_stat_rel=substation_id,
        i_stat=i_stat,
        tot_stat=tot_stat[substation_id],
        from_stat_bool=from_stat_bool[substation_id],
        to_node=current_results.to_node,
        from_node=current_results.from_node,
        susceptance=susceptance,
        slack=slack,
        n_stat=n_stat,
    )

    to_node, from_node = update_from_to_node(
        substation_topology=substation_configuration,
        tot_stat=tot_stat[substation_id],
        from_stat_bool=from_stat_bool[substation_id],
        i_stat_rel_id=substation_id,
        to_node=current_results.to_node,
        from_node=current_results.from_node,
        n_stat=n_stat,
    )

    ptdf = current_results.ptdf + jnp.outer(bsdf, ptdf_th_sw)
    success = success & current_results.success
    bsdf_storage = current_results.bsdf.at[split_idx].set(bsdf)

    return BSDFResults(
        ptdf=ptdf,
        from_node=from_node,
        to_node=to_node,
        success=success,
        bsdf=bsdf_storage,
    )


def apply_bus_split(
    current_results: BSDFResults,
    substation_configuration: Bool[Array, " max_branch_per_sub"],
    substation_id: Int[Array, " "],
    split_idx: Int[Array, " "],
    tot_stat: Int[Array, " n_sub_relevant max_branch_per_sub"],
    from_stat_bool: Bool[Array, " n_sub_relevant max_branch_per_sub"],
    susceptance: Float[Array, " n_branches"],
    rel_stat_map: HashableArrayWrapper[Int[Array, " n_sub_relevant"]],
    slack: int,
    n_stat: int,
) -> BSDFResults:
    """Apply a bus split to a single substation.

    This function applies a bus split to a single substation, i.e. it updates the ptdf matrix and
    the from/ to nodes of the branches. If there is no bus split, i.e. substation_configuration
    is all false, it returns the current_results unchanged.

    Parameters
    ----------
    current_results : BSDFResults
        The current bsdf results with the output of the previous iterations
    substation_configuration : Bool[Array, " max_branch_per_sub"]
        The topology vector for the substation that we want to split
    substation_id : Int[Array, " "]
        The id of the substation that we want to split
    split_idx : Int[Array, " "]
        The index of the split, i.e. the i-th split in the current topology. Used for filling the
        bsdf vector storage in the bsdf results.
    tot_stat : Int[Array, " n_sub_relevant max_branch_per_sub"]
        The static tot_stat array
    from_stat_bool : Bool[Array, " n_sub_relevant max_branch_per_sub"]
        The static from_stat_bool array
    susceptance : Float[Array, " n_branches"]
        The susceptance of each branch
    rel_stat_map : HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
        A mapping from the relevant substations to the actual substations
    slack : int
        The index of the slack bus
    n_stat : int
        The number of substations in the network before splitting/extending the ptdf.


    Returns
    -------
    BSDFResults
        The updated ptdf matrix, from/ to nodes of the branches
    """
    is_substation_split = jnp.any(substation_configuration)

    _apply_bus_split_partial = partial(
        _apply_bus_split,
        substation_configuration=substation_configuration,
        substation_id=substation_id,
        split_idx=split_idx,
        tot_stat=tot_stat,
        from_stat_bool=from_stat_bool,
        susceptance=susceptance,
        rel_stat_map=rel_stat_map,
        slack=slack,
        n_stat=n_stat,
    )

    return jax.lax.cond(
        is_substation_split,
        _apply_bus_split_partial,
        lambda *_: current_results,
        current_results,
    )


def compute_bus_splits(  # noqa: PLR0913
    topologies: Bool[Array, " max_n_bus_splits max_branch_per_sub"],
    sub_ids: Int[Array, " max_n_bus_splits"],
    ptdf: Float[Array, " n_branches n_bus"],
    from_node: Int[Array, " n_branches"],
    to_node: Int[Array, " n_branches"],
    tot_stat: Int[Array, " n_sub_relevant max_branch_per_sub"],
    from_stat_bool: Bool[Array, " n_sub_relevant max_branch_per_sub"],
    susceptance: Float[Array, " n_branches"],
    rel_stat_map: HashableArrayWrapper[Int[Array, " n_sub_relevant"]],
    slack: int,
    n_stat: int,
) -> BSDFResults:
    """Compute the bus splits for a single topology vector.

    This invokes the bsdf computation for each substation in the topologies array and aggregates
    them.

    Parameters
    ----------
    topologies : Bool[Array, " max_n_bus_splits max_branch_per_sub"]
        The topology for which we want to compute the bus splits
    sub_ids : Int[Array, " max_n_bus_splits"]
        The ids of the substations that are affected by the topology
    ptdf : Float[Array, " n_branches n_bus"]
        The PTDF matrix
    from_node : Int[Array, " n_branches"]
        The from nodes of the branches
    to_node : Int[Array, " n_branches"]
        The to nodes of the branches
    tot_stat : Int[Array, " n_sub_relevant max_branch_per_sub"]
        The static tot_stat array
    from_stat_bool : Bool[Array, " n_sub_relevant max_branch_per_sub"]
        The static from_stat_bool array
    susceptance : Float[Array, " n_branches"]
        The susceptance of each branch
    rel_stat_map : HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
        A mapping from the relevant substations to the actual substations
    slack : int
        The index of the slack bus
    n_stat : int
        The number of substations in the network before splitting/extending the ptdf.

    Returns
    -------
    BSDFResults
        The updated ptdf matrix, from and to nodes after performing all the bus splits in topologies
    """
    # One idea is to use reduce here instead of for_i as reduce offers more options for
    # optimization on compiler side, but I'm not 100% sure if the BSDF computation is a monoid.
    # It seems associative but I'm not sure if the unsplit configuration is an identity.
    return jax.lax.fori_loop(
        0,
        topologies.shape[0],
        lambda i, current_results: apply_bus_split(
            current_results=current_results,
            substation_configuration=topologies[i],
            substation_id=sub_ids[i],
            split_idx=i,
            tot_stat=tot_stat,
            from_stat_bool=from_stat_bool,
            susceptance=susceptance,
            rel_stat_map=rel_stat_map,
            slack=slack,
            n_stat=n_stat,
        ),
        init_bsdf_results(
            ptdf=ptdf,
            from_node=from_node,
            to_node=to_node,
            n_splits=topologies.shape[0],
        ),
    )
