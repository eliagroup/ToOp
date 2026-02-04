# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides inspection helpers for topologies and grids.

Gives information about what's happening in the solver and why it might be failing.
"""

from functools import partial

import jax
from beartype.typing import Optional
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Int
from toop_engine_dc_solver.jax.bsdf import apply_bus_split, init_bsdf_results
from toop_engine_dc_solver.jax.disconnections import apply_disconnections
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix
from toop_engine_dc_solver.jax.types import BSDFResults, StaticInformation


def inspect_topology(
    topology: Bool[Array, " n_sub_limited max_sub_topologies"],
    sub_ids: Int[Array, " n_sub_limited"],
    disconnections: Optional[Bool[Array, " "]],
    static_information: StaticInformation,
) -> tuple[
    Bool[Array, " n_sub_limited"],
    Optional[Bool[Array, " "]],
    Bool[Array, " n_failures"],
]:
    """Inspect a single topology for faults, i.e. if and where a split occurred

    Note that if a split occurred, all subsequent computations deliver undefined results, so if you
    use this for debugging make sure to debug one split at a time.

    Parameters
    ----------
    topology : Bool[Array, " n_sub_limited max_sub_topologies"]
        The topology to inspect
    sub_ids : Int[Array, " n_sub_limited"]
        The substation ids of the topology
    disconnections : Optional[Bool[Array, " n_disconnections"]]
        The disconnections to consider, if any
    static_information : StaticInformation
        The static information of the grid

    Returns
    -------
    Bool[Array, " n_sub_limited"]
        Whether the BSDF computation was successful. Note that once one BSDF computation was
        unsuccessful, all following ones will be unsuccessful as well and the LODF results will also
        be invalid (potentially not indicating "unsuccessful")
    Optional[Bool[Array, " "]]
        Whether the disconnection was successful, if an disconnection was provided
    Bool[Array, " n_failures"]
        Whether the LODF computation for a failure was successful
    """

    # We have to use scan here because we want to access the individual success booleans.
    def _scan_bsdf(bsdf_results: BSDFResults, topo_and_sub: tuple) -> tuple[BSDFResults, Bool[Array, " "]]:
        topo, sub = topo_and_sub
        new_res = apply_bus_split(
            current_results=bsdf_results,
            substation_configuration=topo,
            substation_id=sub,
            split_idx=jax.numpy.array(0, dtype=int),
            tot_stat=static_information.dynamic_information.tot_stat,
            from_stat_bool=static_information.dynamic_information.from_stat_bool,
            susceptance=static_information.dynamic_information.susceptance,
            rel_stat_map=static_information.solver_config.rel_stat_map,
            slack=static_information.solver_config.slack,
            n_stat=static_information.solver_config.n_stat,
        )
        return new_res, new_res.success

    bsdf_results, bsdf_success = jax.lax.scan(
        _scan_bsdf,
        init_bsdf_results(
            ptdf=static_information.dynamic_information.ptdf,
            from_node=static_information.dynamic_information.from_node,
            to_node=static_information.dynamic_information.to_node,
            n_splits=topology.shape[0],
        ),
        (topology, sub_ids),
    )

    disconnection_success = None
    if disconnections is not None:
        disc_res = apply_disconnections(
            bsdf_results.ptdf,
            bsdf_results.from_node,
            bsdf_results.to_node,
            disconnections,
        )
        bsdf_results = replace(bsdf_results, ptdf=disc_res.ptdf)
        disconnection_success = disc_res.success

    _, lodf_success = calc_lodf_matrix(
        branches_to_outage=static_information.dynamic_information.branches_to_fail,
        ptdf=bsdf_results.ptdf,
        from_node=bsdf_results.from_node,
        to_node=bsdf_results.to_node,
        branches_monitored=static_information.dynamic_information.branches_monitored,
    )

    return bsdf_success, disconnection_success, lodf_success


def is_valid_batch(
    topologies: Bool[Array, " batch_size n_sub_limited max_sub_topologies"],
    sub_ids: Int[Array, " batch_size n_sub_limited"],
    static_information: StaticInformation,
) -> Bool[Array, " batch_size"]:
    """Check whether a batch of topologies is valid

    Parameters
    ----------
    topologies : Bool[Array, " batch_size n_sub_limited max_sub_topologies"]
        The batch of topologies to check
    sub_ids : Int[Array, " batch_size n_sub_limited"]
        The substation ids of the topologies
    static_information : StaticInformation
        The static information of the grid

    Returns
    -------
    Bool[Array, " batch_size"]
        Whether the batch of topologies is valid
    """
    inspect_jit = jax.jit(inspect_topology, static_argnames=("static_information",))
    inspect_jit_partial = partial(inspect_jit, disconnections=None, static_information=static_information)

    bsdf_success, _, lodf_success = jax.vmap(inspect_jit_partial)(topologies, sub_ids)
    return jnp.all(bsdf_success, axis=1) & jnp.all(lodf_success, axis=1)
