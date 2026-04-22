# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides routines to compute a single batch of topologies/injections.

If you want to compute any number of batches, use the topology_looper module instead.
"""

from functools import partial

import equinox as eqx
import jax
from beartype.typing import Optional
from jax import numpy as jnp
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits
from toop_engine_dc_solver.jax.busbar_outage import get_busbar_outage_penalty_batched
from toop_engine_dc_solver.jax.contingency_analysis import (
    BatchedContingencyAnalysisParams,
    UnBatchedContingencyAnalysisParams,
    contingency_analysis_matrix,
)
from toop_engine_dc_solver.jax.cross_coupler_flow import compute_cross_coupler_flows
from toop_engine_dc_solver.jax.disconnections import apply_disconnections, update_n0_flows_after_disconnections
from toop_engine_dc_solver.jax.injections import (
    get_all_injection_outage_deltap,
    get_all_outaged_injection_nodes_after_reassignment,
    get_injection_vector,
)
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix, get_failure_cases_to_zero
from toop_engine_dc_solver.jax.multi_outages import build_modf_matrices
from toop_engine_dc_solver.jax.nminus2_outage import split_n_2_analysis_batched
from toop_engine_dc_solver.jax.nodal_inj_optim import nodal_inj_optimization
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    pad_action_with_unsplit_action_indices,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    DynamicInformation,
    InjectionComputations,
    NodalInjStartOptions,
    SolverConfig,
    SolverLoadflowResults,
    SparseNMinus0,
    SparseNMinus1,
    TopologyResults,
    TopoVectBranchComputations,
    int_max,
)


class InjectionIter(eqx.Module):
    """Holds the data for a injection/contingency computation iteration results and the iterator int.

    The sparse results are of shape (batch_size_bsdf) and hold the current best values.
    """

    n_0_results: SparseNMinus0
    n_1_results: SparseNMinus1
    i: Int[Array, " "]
    best_inj_combi: Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]
    metrics: Float[Array, " batch_size_bsdf"]


class InjectionIterMetrics(eqx.Module):
    """Holds the injection iteration results for the metrics based injection computation.

    We don't yet store the SparseNMinus0/1 results, they have to be computed again at the end
    """

    i: Int[Array, " "]
    best_inj_combi: Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]
    metrics: Float[Array, " batch_size_bsdf"]


def compute_bsdf_lodf_static_flows(
    topology_batch: TopoVectBranchComputations,
    disconnection_batch: Optional[Int[Array, " batch_size_bsdf n_disconnections"]],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> TopologyResults:
    """Compute all topology-related results

    This includes the BSDF computation to adjust the PTDF matrix, building the LODF matrix and
    computing the static flows

    Parameters
    ----------
    topology_batch : TopoVectBranchComputations
        The batch of topology computations to perform the computations for, shape
        (batch_size_bsdf, ...). In bitvector format
    disconnection_batch : Optional[Int[Array, " batch_size_bsdf n_disconnections"]]
        The disconnections to perform as topology measures, shape (batch_size_bsdf, n_disconnections).
        If None or size 0, no disconnections are performed.
    dynamic_information : DynamicInformation
        The dynamic information about the grid
    solver_config : SolverConfig
        The solver configuration

    Returns
    -------
    TopologyResults
        The results for the topology batch, shape (batch_size_bsdf, ...)
    """
    bsdf_res = jax.vmap(
        partial(
            compute_bus_splits,
            ptdf=dynamic_information.ptdf,
            from_node=dynamic_information.from_node,
            to_node=dynamic_information.to_node,
            tot_stat=dynamic_information.tot_stat,
            from_stat_bool=dynamic_information.from_stat_bool,
            susceptance=dynamic_information.susceptance,
            rel_stat_map=solver_config.rel_stat_map,
            slack=solver_config.slack,
            n_stat=solver_config.n_stat,
        )
    )(topology_batch.topologies, topology_batch.sub_ids)

    topo_res = TopologyResults(
        ptdf=bsdf_res.ptdf,
        from_node=bsdf_res.from_node,
        to_node=bsdf_res.to_node,
        lodf=None,
        success=bsdf_res.success,
        contingency_success=None,
        outage_modf=None,
        failure_cases_to_zero=None,
        bsdf=bsdf_res.bsdf,
        disconnection_modf=None,
    )
    del bsdf_res

    # Apply disconnections
    has_disconnections = disconnection_batch is not None and disconnection_batch.size > 0
    if has_disconnections:
        disc_res = jax.vmap(
            partial(
                apply_disconnections,
                guarantee_unique=True,
            ),
            in_axes=(0, 0, 0, 0),
        )(topo_res.ptdf, topo_res.from_node, topo_res.to_node, disconnection_batch)

        branch_cases_to_zero = jax.vmap(get_failure_cases_to_zero, in_axes=(0, None))(
            disconnection_batch, dynamic_information.branches_to_fail
        )
        failure_cases_to_zero = jnp.concatenate(
            [
                branch_cases_to_zero,
                jnp.zeros(
                    (
                        branch_cases_to_zero.shape[0],
                        dynamic_information.n_nminus1_cases - branch_cases_to_zero.shape[1],
                    ),
                    dtype=bool,
                ),
            ],
            axis=1,
        )

        topo_res = replace(
            topo_res,
            ptdf=disc_res.ptdf,
            from_node=disc_res.from_node,
            to_node=disc_res.to_node,
            success=disc_res.success & topo_res.success,
            disconnection_modf=disc_res.modf,
            failure_cases_to_zero=failure_cases_to_zero,
        )
        del disc_res

    # Compute the LODF matrix
    # This again is only necessary once per topology batch
    lodf, lodf_success = jax.vmap(calc_lodf_matrix, in_axes=(None, 0, 0, 0, None))(
        dynamic_information.branches_to_fail,
        topo_res.ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.branches_monitored,
    )

    if topo_res.failure_cases_to_zero is not None:
        single_outage_cases_to_zero = topo_res.failure_cases_to_zero[:, : lodf_success.shape[1]]
        lodf_success = jnp.where(single_outage_cases_to_zero, True, lodf_success)

    # Compute multi-outages
    outage_modf, outage_modf_success = jax.vmap(
        build_modf_matrices,
        in_axes=(
            0,
            0,
            0,
            None,
        ),
    )(
        topo_res.ptdf,
        topo_res.from_node,
        topo_res.to_node,
        dynamic_information.multi_outage_branches,
    )
    injection_outage_success = jnp.broadcast_to(
        topo_res.success[:, None],
        (topo_res.success.shape[0], dynamic_information.n_inj_failures),
    )
    bb_outage_success = jnp.broadcast_to(
        topo_res.success[:, None],
        (
            topo_res.success.shape[0],
            dynamic_information.n_bb_outages if dynamic_information.bb_outage_baseline_analysis is None else 0,
        ),
    )
    contingency_success = jnp.concatenate(
        [
            lodf_success,
            outage_modf_success,
            injection_outage_success,
            bb_outage_success,
        ],
        axis=1,
    )

    return replace(
        topo_res,
        outage_modf=outage_modf,
        lodf=lodf,
        contingency_success=contingency_success,
        success=topo_res.success & jnp.all(contingency_success, axis=1),
    )


def compute_injections(
    injections: Bool[Array, " batch_size n_splits max_inj_per_sub"],
    sub_ids: Int[Array, " batch_size n_splits"],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> Float[Array, " batch_size n_timesteps n_bus"]:
    """Compute the injection vectors for an injection computation batch

    Parameters
    ----------
    injections : Bool[Array, " batch_size n_splits max_inj_per_sub"]
        The injection vectors to compute as a boolean topo vect
    sub_ids : Int[Array, " batch_size n_splits"]
        The substation ids for each injection computation
    dynamic_information : DynamicInformation
        The dynamic information about the grid
    solver_config : SolverConfig
        The solver configuration

    Returns
    -------
    Float[Array, " batch_size n_timesteps n_bus"]
        The injection vectors for each injection computation
    """
    # Compute the nodal injections
    get_injection_vector_partial = jax.tree_util.Partial(
        get_injection_vector,
        relevant_injections=dynamic_information.relevant_injections,
        nodal_injections=dynamic_information.nodal_injections,
        n_stat=jnp.array(solver_config.n_stat),
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
    )
    return jax.vmap(get_injection_vector_partial)(injection_assignment=injections, sub_ids=sub_ids)


def validate_shapes_compute_batch(
    topology_batch: ActionIndexComputations,
    disconnection_batch: Optional[Int[Array, " batch_size_bsdf n_disconnections"]],
    injection_batch: InjectionComputations,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> None:
    """Check the inputs to compute batch for correct shapes/dtypes and raises an error if they don't match"""
    batch_size_bsdf = solver_config.batch_size_bsdf
    buffer_size_injection = solver_config.buffer_size_injection
    batch_size_injection = solver_config.batch_size_injection
    max_inj_per_sub = dynamic_information.max_inj_per_sub
    n_splits = topology_batch.action.shape[1]
    n_disconnections = disconnection_batch.shape[1] if disconnection_batch is not None else 0

    assert n_splits <= dynamic_information.n_sub_relevant
    assert topology_batch.action.shape == (batch_size_bsdf, n_splits)
    assert topology_batch.pad_mask.shape == (batch_size_bsdf,)
    if disconnection_batch is not None:
        assert disconnection_batch.shape == (batch_size_bsdf, n_disconnections)
    assert injection_batch.injection_topology.shape == (
        buffer_size_injection,
        batch_size_injection,
        n_splits,
        max_inj_per_sub,
    )
    assert injection_batch.corresponding_topology.shape == (buffer_size_injection, batch_size_injection)
    assert injection_batch.pad_mask.shape == (buffer_size_injection, batch_size_injection)
    assert dynamic_information.ptdf.ndim == 2


# sonar: noqa: S3776
def compute_symmetric_batch(
    topology_batch: ActionIndexComputations,
    disconnection_batch: Optional[Int[Array, " batch_size_bsdf n_disconnections"]],
    injections: Optional[Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]],
    nodal_inj_start_options: Optional[NodalInjStartOptions],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> tuple[
    SolverLoadflowResults,
    Bool[Array, " batch_size_bsdf"],
]:
    """Compute a batch where we have a single injection combination per topology

    This means that we don't need to enumerate and accumulate injections but have a single
    batch size only. batch_size_injection and buffer_size_injection will be ignored.

    Furthermore, this returns the full N-0/N-1 results for each topology, not just the worst N-0/N-1
    result.

    Parameters
    ----------
    topology_batch : ActionIndexBranchComputations
        The topology computations to perform, shape (batch_size_bsdf, ...)
    disconnection_batch : Optional[Int[Array, " batch_size_bsdf n_disconnections"]]
        The disconnections to perform as topological measures. If None, no disconnections are
        performed.
        This assumes indices into dynamic_information.disconnectable_branches
    injections : Optional[Bool[Array, " batch_size_bsdf n_splits max_inj_per_sub"]]
        The injection vectors to compute, shape (batch_size_bsdf, n_splits, max_inj_per_sub). Note that the
        injections are over the substations that are split in the topology batch. If None, the default injection that
        is associated with the topology batch will be used. If not None, busbar outages might be incorrect
    nodal_inj_start_options: Optional[NodalInjStartOptions]
        The nodal injection optimization is more efficient if it does not start with zero guesses, but for example a previous
        computation's results. Furthermore, additional parameters like iteration count might be set.
    dynamic_information : DynamicInformation
        The dynamic information about the grid
    solver_config : SolverConfig
        The solver configuration

    Returns
    -------
    LoadflowMatrices
        The results object for this batch, with a leading batch_size_bsdf dimension on each field
    Bool[Array, " batch_size_bsdf"]
        Whether the computation was successful for each topology

    Notes
    -----
    Busbar outage as an N-1 analysis is not suported when injection are not None.
    """
    batch_size_bsdf = solver_config.batch_size_bsdf
    has_disconnections = disconnection_batch is not None and disconnection_batch.size > 0
    if has_disconnections:
        # Translate the disconnection batch to the actual indices
        disconnection_batch = dynamic_information.disconnectable_branches.at[disconnection_batch].get(
            mode="fill", fill_value=int_max()
        )

    assert len(topology_batch) == batch_size_bsdf

    if solver_config.enable_bb_outages and injections is not None:
        # If we have busbar outages, we need to compute the injections for each topology
        # separately. This is not supported in this mode.
        raise ValueError(
            "Busbar outages are not supported when injections are provided. Please use the compute_batch function instead."
        )

    if injections is None:
        injections = dynamic_information.action_set.inj_actions.at[topology_batch.action].get(mode="fill", fill_value=False)

    assert injections.shape == (
        batch_size_bsdf,
        topology_batch.action.shape[1],
        dynamic_information.max_inj_per_sub,
    )

    bitvector_topology = convert_action_set_index_to_topo(topology_batch, dynamic_information.action_set)
    sub_ids = jnp.where(
        bitvector_topology.topologies.any(axis=-1),
        bitvector_topology.sub_ids,
        int_max(),
    )
    topo_res = compute_bsdf_lodf_static_flows(bitvector_topology, disconnection_batch, dynamic_information, solver_config)

    unbatched_params = UnBatchedContingencyAnalysisParams(
        branches_to_fail=dynamic_information.branches_to_fail,
        injection_outage_deltap=get_all_injection_outage_deltap(
            injection_outage_deltap=dynamic_information.nonrel_injection_outage_deltap,
            relevant_injections=dynamic_information.relevant_injections,
            relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
            relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
        ),
        branches_monitored=dynamic_information.branches_monitored,
        action_set=dynamic_information.action_set,
        non_rel_bb_outage_data=dynamic_information.non_rel_bb_outage_data,
        enable_bb_outages=solver_config.enable_bb_outages and solver_config.bb_outage_as_nminus1,
    )

    nodal_injections = compute_injections(
        injections=injections,
        sub_ids=sub_ids,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
    )

    n_0, cross_coupler_flows = jax.vmap(
        compute_cross_coupler_flows,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(
        topo_res.bsdf,
        bitvector_topology.topologies,
        sub_ids,
        injections,
        dynamic_information.relevant_injections,
        dynamic_information.unsplit_flow,
        dynamic_information.tot_stat,
        dynamic_information.from_stat_bool,
    )

    n_0 = jax.vmap(update_n0_flows_after_disconnections)(n_0, topo_res.disconnection_modf)

    nodal_injections_optimized = None
    if nodal_inj_start_options is not None:
        # TODO replace N-1 computation below with the results from optimization as soon as the optimization is halfway stable
        # It might be a good debug aid to have the original code below still available.
        n_0, _n_1, nodal_injections_optimized = nodal_inj_optimization(
            n_0=n_0,
            nodal_injections=nodal_injections,
            topo_res=topo_res,
            start_options=nodal_inj_start_options,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
        )

    # Compute the N-1 matrix
    batched_params = BatchedContingencyAnalysisParams(
        lodf=topo_res.lodf,
        ptdf=topo_res.ptdf,
        modf=topo_res.outage_modf,
        nodal_injections=nodal_injections,
        n_0_flow=n_0,
        injection_outage_node=get_all_outaged_injection_nodes_after_reassignment(
            injection_assignment=injections,
            sub_ids=sub_ids,
            relevant_injections=dynamic_information.relevant_injections,
            relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
            relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
            nonrel_injection_outage_node=dynamic_information.nonrel_injection_outage_node,
            rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
            n_stat=jnp.array(solver_config.n_stat),
        ),
        **(
            {
                "action_indices": topology_batch.action,
                "from_nodes": topo_res.from_node,
                "to_nodes": topo_res.to_node,
                "disconnections": disconnection_batch,
            }
            if unbatched_params.enable_bb_outages
            else {}
        ),
    )

    contingency_analysis_matrix_partial = partial(
        jax.jit(contingency_analysis_matrix),
        unbatched_params=unbatched_params,
    )
    n_1 = jax.vmap(contingency_analysis_matrix_partial)(batched_params=batched_params)

    if topo_res.failure_cases_to_zero is not None:
        n_1 = jnp.where(topo_res.failure_cases_to_zero[:, None, :, None], 0, n_1)

    n_2_penalty = None
    if dynamic_information.n2_baseline_analysis is not None:
        n_2_penalty = split_n_2_analysis_batched(
            has_splits=bitvector_topology.topologies.any(axis=-1),
            sub_ids=sub_ids,
            disconnections=disconnection_batch if has_disconnections else None,
            nodal_injections=nodal_injections,
            ptdf=topo_res.ptdf,
            from_node=topo_res.from_node,
            to_node=topo_res.to_node,
            l2_outages=dynamic_information.branches_to_fail,
            baseline=dynamic_information.n2_baseline_analysis,
            branches_monitored=dynamic_information.branches_monitored,
        )

    bb_outage_as_penalty = solver_config.enable_bb_outages and not solver_config.bb_outage_as_nminus1
    bb_outage_penalty = None
    if bb_outage_as_penalty:
        padded_action_indices = jax.vmap(pad_action_with_unsplit_action_indices, in_axes=(None, 0))(
            dynamic_information.action_set, topology_batch.action
        )
        bb_outage_penalty, overload, n_grid_splits = get_busbar_outage_penalty_batched(
            action_indices=padded_action_indices,
            ptdf=topo_res.ptdf,
            nodal_injections=nodal_injections,
            from_nodes=topo_res.from_node,
            to_nodes=topo_res.to_node,
            action_set=dynamic_information.action_set,
            branches_monitored=dynamic_information.branches_monitored,
            unsplit_bb_outage_analysis=dynamic_information.bb_outage_baseline_analysis,
            lower_bound=0.0 if solver_config.clip_bb_outage_penalty else None,
            n_0_flows=n_0,
        )

    return (
        SolverLoadflowResults(
            n_0_matrix=n_0[:, :, dynamic_information.branches_monitored],
            n_1_matrix=n_1,
            contingency_success=topo_res.contingency_success,
            cross_coupler_flows=cross_coupler_flows,
            branch_action_index=topology_batch.action,
            branch_topology=bitvector_topology.topologies,
            sub_ids=sub_ids,
            injection_topology=injections,
            n_2_penalty=n_2_penalty,
            disconnections=disconnection_batch,
            bb_outage_penalty=bb_outage_penalty,
            bb_outage_overload=overload if bb_outage_as_penalty else None,
            bb_outage_splits=n_grid_splits if bb_outage_as_penalty else None,
            nodal_injections_optimized=nodal_injections_optimized,
        ),
        topo_res.success,
    )
