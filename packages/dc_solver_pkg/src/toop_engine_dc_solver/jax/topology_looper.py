# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Calls the bsdf, injection and contingency module to perform their computations.

The original topology looper uses a tree formulation to structure the computation, here we rely
on a batched two step process:
- First, compute a batch of BSDF formulations, save the resulting PTDF matrices
- Then, iterate in batches over all related injection combination and compute the contingency
results for each injection.
- Aggregate the injection results down to one per topology
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from beartype.typing import Optional
from jax_dataclasses import replace
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric
from toop_engine_dc_solver.jax.batching import (
    count_injection_combinations_from_corresponding_topology,
    get_injections_for_topo_range,
    greedy_buffer_size_selection,
    pad_topologies_action_index,
    slice_nodal_inj_start_options,
    slice_topologies_action_index,
    split_injections,
)
from toop_engine_dc_solver.jax.compute_batch import (
    compute_batch,
    compute_symmetric_batch,
)
from toop_engine_dc_solver.jax.result_storage import prepare_result_storage, sparsify_results
from toop_engine_dc_solver.jax.topology_computations import convert_topo_to_action_set_index, default_topology
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    ActionSet,
    AggregateMetricProtocol,
    AggregateOutputProtocol,
    BranchLimits,
    DynamicInformation,
    InjectionComputations,
    NodalInjStartOptions,
    SolverConfig,
    SolverLoadflowResults,
    SparseNMinus0,
    SparseNMinus1,
    SparseSolverOutput,
    TopoVectBranchComputations,
)


def concatenate_results(
    results: list[SparseSolverOutput],
    pad_mask: Optional[Bool[Array, " n_topologies"]] = None,
) -> SparseSolverOutput:
    """Concatenate all results along the topology dimension

    Parameters
    ----------
    results : list[SparseSolverOutput]
        A list of topology results
    pad_mask : Optional[Bool[Array, " n_topologies"]]
        A mask to filter out padded results

    Returns
    -------
    SparseSolverOutput
        The concatenated results
    """
    if pad_mask is None:
        total_length = sum(x.n_0_results.pf_n_0_max.shape[0] for x in results)
        pad_mask = jnp.ones(total_length, dtype=bool)

    return SparseSolverOutput(
        n_0_results=SparseNMinus0(
            pf_n_0_max=jnp.concatenate([x.n_0_results.pf_n_0_max for x in results])[pad_mask],
            hist_mon=jnp.concatenate([x.n_0_results.hist_mon for x in results])[pad_mask],
        ),
        n_1_results=SparseNMinus1(
            pf_n_1_max=jnp.concatenate([x.n_1_results.pf_n_1_max for x in results])[pad_mask],
            hist_mon=jnp.concatenate([x.n_1_results.hist_mon for x in results])[pad_mask],
            hist_out=jnp.concatenate([x.n_1_results.hist_out for x in results])[pad_mask],
        ),
        best_inj_combi=jnp.concatenate([x.best_inj_combi for x in results])[pad_mask],
        success=jnp.concatenate([x.success for x in results])[pad_mask],
    )


class DefaultAggregateMetricsFn(AggregateMetricProtocol):
    """Default aggregation function that sums the maximum overloads"""

    def __init__(
        self,
        branch_limits: BranchLimits,
        reassignment_distance: Int[Array, " n_branch_actions"],
        n_relevant_subs: int,
        metric: str,
        fixed_hash: int,
    ) -> None:
        """Create a new DefaultAggregateMetricsFn

        Arguments are mainly static arguments to aggregate_to_metric and a fixed hash to avoid
        recompilation

        Parameters
        ----------
        branch_limits : BranchLimits
            The branch limits
        reassignment_distance : Int[Array, " n_branch_actions"]
            The reassignment distance
        n_relevant_subs : int
            The number of relevant substations
        metric : str
            The metric to use
        fixed_hash : int
            A fixed hash, for this you can use hash(static_information)
        """
        self.branch_limits = branch_limits
        self.reassignment_distance = reassignment_distance
        self.n_relevant_subs = n_relevant_subs
        self.metric = metric
        self.fixed_hash = fixed_hash

    def __call__(
        self,
        lf_results: SolverLoadflowResults,
        _output: Optional[PyTree],
    ) -> Float[Array, " "]:
        """Just returns aggregate_to_metric"""
        return aggregate_to_metric(
            lf_res=lf_results,
            branch_limits=self.branch_limits,
            reassignment_distance=self.reassignment_distance,
            n_relevant_subs=self.n_relevant_subs,
            metric=self.metric,
        )

    def __hash__(self) -> Int:
        """Get the fixed hash passed during initialization to avoid recompliation"""
        return self.fixed_hash

    def __eq__(self, other: object) -> bool:
        """Compare based on hash to avoid recompilation"""
        if not isinstance(other, DefaultAggregateMetricsFn):
            return False
        return hash(self) == hash(other)


class DefaultAggregateOutputFn(AggregateOutputProtocol):
    """Default aggregation function that returns the maximum overloads"""

    def __init__(
        self,
        branches_to_fail: Int[Array, " n_branches"],
        multi_outage_indices: Int[Array, " n_multi_outages"],
        injection_outage_indices: Int[Array, " n_injection_outages"],
        max_mw_flow: Float[Array, " n_branches"],
        number_most_affected: int,
        number_max_out_in_most_affected: Optional[int],
        number_most_affected_n_0: int,
        fixed_hash: int,
    ) -> None:
        """Create a new DefaultAggregateOutputFn

        Arguments are mainly static arguments to sparsify_results and a fixed hash to avoid
        recompilation

        Parameters
        ----------
        branches_to_fail : Int[Array, " n_branches"]
            The branches to fail, used to translate failure indices to branch indices
        multi_outage_indices: Int[Array, " n_multi_outages"]
            The indices of the multi-outages. Should be distinct from the branches_to_fail
        injection_outage_indices: Int[Array, " n_injection_outages"]
            The indices of the injection outages. Should be distinct from the branches_to_fail +
            multi_outage_indices.
        max_mw_flow : Float[Array, " n_branches"]
            The maximum MW flow for every branch
        number_most_affected : int
            The number of most affected branches to return
        number_max_out_in_most_affected : Optional[int]
            The number of most affected branches to return in the N-1 case
        number_most_affected_n_0 : int
            The number of most affected branches to return in the N-0 case
        fixed_hash : int
            A fixed hash, for this you can use hash(static_information)
        """
        self.branches_to_fail = jnp.concatenate(
            [branches_to_fail, multi_outage_indices, injection_outage_indices],
            axis=0,
        )
        self.max_mw_flow = max_mw_flow
        self.number_most_affected = number_most_affected
        self.number_max_out_in_most_affected = number_max_out_in_most_affected
        self.number_most_affected_n_0 = number_most_affected_n_0
        self.fixed_hash = fixed_hash

    def __call__(self, lf_result: SolverLoadflowResults) -> PyTree:
        """Just calls sparsify results"""
        return sparsify_results(
            n_0_matrix=lf_result.n_0_matrix,
            n_1_matrix=lf_result.n_1_matrix,
            branches_to_fail=self.branches_to_fail,
            max_mw_flow=self.max_mw_flow,
            number_most_affected=self.number_most_affected,
            number_max_out_in_most_affected=self.number_max_out_in_most_affected,
            number_most_affected_n_0=self.number_most_affected_n_0,
        )

    def __hash__(self) -> int:
        """Get the fixed hash passed during initialization to avoid recompliation"""
        return self.fixed_hash

    def __eq__(self, other: object) -> bool:
        """Compare based on hash to avoid recompilation"""
        if not isinstance(other, DefaultAggregateOutputFn):
            return False
        return hash(self) == hash(other)


def run_solver(
    topologies: ActionIndexComputations,
    disconnections: Optional[Int[Array, " n_topologies n_disconnections"]],
    injections: Optional[InjectionComputations],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> SparseSolverOutput:
    """Run the solver for a given set of topologies and injections.

    This will decide automatically whether to run run_solver_symmetric or run_solver_inj_bruteforce
    by matching the corresponding_topologies in injections. Furthermore, it assumes a metric
    function based on solver_config.aggregation_metric and uses the sparsify output function.

    Parameters
    ----------
    topologies : ActionIndexBranchComputations
        The topology computations to perform. Must be in action index format.
    disconnections : Optional[Int[Array, " n_topologies n_disconnections"]]
        The disconnections to perform as topological measures. If None, no disconnections are performed
    injections : Optional[InjectionComputations]
        The injection computations to perform, will overwrite the injection from the action set. If None, the injections are
        taken from the action set. This requires passing topologies in the format of ActionIndexBranchComputations.
    dynamic_information : DynamicInformation
        Dynamic information about the grid, such as the PTDF matrix
    solver_config : SolverConfig
        Configuration for the solver

    Returns
    -------
    SparseSolverOutput
        The results of the solver
    """
    if not isinstance(topologies, ActionIndexComputations):
        raise ValueError("Topologies must be in action index format, use convert_topologies to convert them.")

    if topologies.action.size == 0:
        topologies = default_topology(solver_config=solver_config)

    if disconnections is not None and disconnections.size == 0:
        disconnections = None

    output_fn = DefaultAggregateOutputFn(
        branches_to_fail=dynamic_information.branches_to_fail,
        multi_outage_indices=jnp.arange(dynamic_information.n_multi_outages) + jnp.max(dynamic_information.branches_to_fail),
        injection_outage_indices=jnp.arange(dynamic_information.n_inj_failures)
        + jnp.max(dynamic_information.branches_to_fail)
        + dynamic_information.n_multi_outages,
        max_mw_flow=dynamic_information.branch_limits.max_mw_flow,
        number_most_affected=solver_config.number_most_affected,
        number_max_out_in_most_affected=solver_config.number_max_out_in_most_affected,
        number_most_affected_n_0=solver_config.number_most_affected_n_0,
        fixed_hash=hash(solver_config),
    )

    metric_fn = DefaultAggregateMetricsFn(
        branch_limits=dynamic_information.branch_limits,
        reassignment_distance=dynamic_information.action_set.reassignment_distance,
        n_relevant_subs=dynamic_information.n_sub_relevant,
        metric=solver_config.aggregation_metric,
        fixed_hash=hash(solver_config),
    )

    # We can use symmetric mode if there is exactly one injection per topology or if no injections were passed
    if injections is None or jnp.array_equal(injections.corresponding_topology, jnp.arange(len(topologies))):
        results, success = run_solver_symmetric(
            topologies,
            disconnections,
            injections.injection_topology if injections is not None else None,
            dynamic_information,
            solver_config,
            output_fn,
        )
        return SparseSolverOutput(
            n_0_results=results[0],
            n_1_results=results[1],
            best_inj_combi=injections.injection_topology if injections is not None else None,
            success=success,
        )

    results, best_inj_combi, success = run_solver_inj_bruteforce(
        topologies,
        disconnections,
        injections,
        dynamic_information,
        solver_config,
        metric_fn,
        output_fn,
    )

    return SparseSolverOutput(
        n_0_results=results[0],
        n_1_results=results[1],
        best_inj_combi=best_inj_combi,
        success=success,
    )


def convert_topologies(
    topologies: TopoVectBranchComputations,
    action_set: ActionSet,
) -> tuple[ActionIndexComputations, Optional[ActionSet]]:
    """Convert topologies from topo-vect format to action index format if necessary

    This might mean extending the branch action set if an action is not in the branch action set. If the branch action set
    was extended, this returns the new action set.
    Note that extending the action set can create problems during postprocessing if you rely on the fact that the action set
    during preprocessing has the same size as during postprocessing or you want to use the action indices as a reference.

    Parameters
    ----------
    topologies :TopoVectBranchComputations
        The topologies to convert
    action_set : BranchActionSet
        The branch action set currently in the static information

    Returns
    -------
    ActionIndexBranchComputations
        The converted topologies
    Optional[BranchActionSet]
        The new branch action set if it was extended or None if all actions were already in the branch action set
    """
    if isinstance(topologies, ActionIndexComputations):
        raise ValueError("Topologies are already in action index format")

    topologies, action_set_new = convert_topo_to_action_set_index(
        topologies=topologies, branch_actions=action_set, extend_action_set=True
    )

    if action_set_new == action_set:
        action_set_new = None

    return topologies, action_set_new


def run_solver_inj_bruteforce(
    topologies: ActionIndexComputations,
    disconnections: Optional[Int[Array, " n_topologies n_disconnections"]],
    injections: InjectionComputations,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    aggregate_metric_fn: AggregateMetricProtocol,
    aggregate_output_fn: AggregateOutputProtocol,
) -> tuple[
    PyTree[Shaped, " batch_size_bsdf ..."],
    Int[Array, " batch_size_bsdf n_subs_rel"],
    Bool[Array, " batch_size_bsdf"],
]:
    """Run the solver for a given set of topologies and injections, using injection bruteforcing

    This will run a bruteforce search for all possible injections for every topology in the input
    and choose the best one according to the metric function.

    Parameters
    ----------
    topologies : ActionIndexComputations
        The topology computations to perform in action index format
    disconnections : Optional[Int[Array, " n_topologies n_disconnections"]]
        The disconnections to perform as topological measures. If None, no disconnections are performed
    injections : InjectionComputations
        The injection computations to perform, will overwrite the action index computations
    dynamic_information : DynamicInformation
        Dynamic information about the grid, such as the PTDF matrix
    solver_config : SolverConfig
        Configuration for the solver
    aggregate_metric_fn : AggregateMetricProtocol
        A function that takes the N-0 and N-1 matrices of a single topology (no batch dimension)
        and returns a metric that should be maximized. The metric function must be vmappable. If
        metrics_first_mode is False, the metric function will receive the output of
        aggregate_output_fn as the third argument, otherwise it will be passed None.
    aggregate_output_fn : AggregateOutputProtocol
        A function that takes the N-0 and N-1 matrices of a single topology (no batch dimension)
        and returns any aggregated information the user wishes to compute. The aggregate function
        must be vmappable.


    Returns
    -------
    PyTree[Shaped, " batch_size_bsdf ..."]
        The results object for this batch according to aggregate_output_fn
    Int[Array, " batch_size_bsdf n_subs_rel"]
        The best injection combination for each topology
    Bool[Array, " batch_size_bsdf"]
        The success flag for each topology
    """
    if not topologies.action.size:
        topologies = default_topology(solver_config=solver_config)
    if disconnections is not None and disconnections.size == 0:
        disconnections = None

    # Pad out inputs to match the batch size / devices
    distributed = solver_config.distributed
    n_devices = len(jax.devices()) if distributed else 1
    batch_size = solver_config.batch_size_bsdf * n_devices
    n_topologies_orig = len(topologies)
    n_topologies = math.ceil(n_topologies_orig / batch_size) * batch_size
    pad_width = n_topologies - n_topologies_orig

    topologies = pad_topologies_action_index(topologies, n_topologies)
    disconnections = (
        jnp.pad(
            disconnections,
            [[0, pad_width], [0, 0]],
            mode="constant",
            constant_values=-1,
        )
        if disconnections is not None
        else None
    )

    # Compute buffer size if not given
    if solver_config.buffer_size_injection is None:
        n_injs_per_topo = count_injection_combinations_from_corresponding_topology(
            corresponding_topology=injections.corresponding_topology,
            batch_size_bsdf=solver_config.batch_size_bsdf,
            n_topologies=n_topologies,
        )

        solver_config = replace(
            solver_config,
            buffer_size_injection=greedy_buffer_size_selection(
                n_inj_combis_per_topo_batch=n_injs_per_topo,
                batch_size_injection=solver_config.batch_size_injection,
            ),
        )

    # Prepare storage for all results
    result_storage = prepare_result_storage(
        aggregate_output_fn,
        n_timesteps=dynamic_information.n_timesteps,
        n_branches_monitored=dynamic_information.n_branches_monitored,
        n_failures=dynamic_information.n_nminus1_cases,
        n_splits=topologies.action.shape[1],
        n_disconnections=disconnections.shape[1] if disconnections is not None else None,
        max_branch_per_sub=dynamic_information.max_branch_per_sub,
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        nminus2=dynamic_information.n2_baseline_analysis is not None,
        bb_outage=solver_config.enable_bb_outages and not solver_config.bb_outage_as_nminus1,
        size=n_topologies,
    )

    if distributed:
        devices = jax.devices()
        n_devices = len(devices)

        topologies = jax.tree_util.tree_map(
            lambda x: jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices),
            topologies,
        )
        disconnections = (
            jax.device_put_sharded(jnp.split(disconnections, n_devices, axis=0), devices)
            if disconnections is not None
            else None
        )
        # Splitting injections needs a bit more care as we need to split according to corresponding
        # topology
        n_topos_per_device = n_topologies // n_devices
        injections = split_injections(
            injections=injections,
            n_splits=n_devices,
            packet_size_injection=(
                solver_config.batch_size_injection
                * solver_config.buffer_size_injection
                * math.ceil(n_topos_per_device / solver_config.batch_size_bsdf)
            ),
            n_topos_per_split=n_topos_per_device,
        )
        injections = jax.tree_util.tree_map(
            lambda x: jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices),
            injections,
        )
        result_storage = jax.tree_util.tree_map(
            lambda x: jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices),
            result_storage,
        )

        results = jax.pmap(
            iterate_inj_bruteforce_sequential,
            in_axes=(
                0,
                0 if disconnections is not None else None,
                0,
                0,
                jax.tree_util.tree_map(lambda _: None, dynamic_information),
                jax.tree_util.tree_map(lambda _: None, solver_config),
                None,
                None,
            ),
            static_broadcasted_argnums=(5, 6, 7),
            donate_argnums=(3,),
        )(
            topologies,
            disconnections,
            injections,
            result_storage,
            dynamic_information,
            solver_config,
            aggregate_metric_fn,
            aggregate_output_fn,
        )

        results = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), results)
    else:
        results = iterate_inj_bruteforce_sequential(
            topologies=topologies,
            disconnections=disconnections,
            injections=injections,
            result_storage=result_storage,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
            aggregate_metric_fn=aggregate_metric_fn,
            aggregate_output_fn=aggregate_output_fn,
        )
    results = jax.tree_util.tree_map(lambda x: x[:n_topologies_orig], results)
    return results


@partial(
    jax.jit,
    static_argnames=(
        "solver_config",
        "aggregate_metric_fn",
        "aggregate_output_fn",
    ),
)
def iterate_inj_bruteforce_sequential(
    topologies: ActionIndexComputations,
    disconnections: Optional[Int[Array, " n_topologies n_disconnections"]],
    injections: Optional[InjectionComputations],
    result_storage: PyTree[Shaped, " n_topologies ..."],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    aggregate_metric_fn: AggregateMetricProtocol,
    aggregate_output_fn: AggregateOutputProtocol,
) -> tuple[
    PyTree[Shaped, " n_topologies ..."],
    Int[Array, " n_topologies n_sub_relevant"],
    Bool[Array, " n_topologies"],
]:
    """Iterate over already padded topologies and injections sequentially

    Parameters
    ----------
    topologies : ActionIndexBranchComputations
        The topology computations to perform, padded to match the batch size
    disconnections : Optional[Int[Array, " n_topologies n_disconnections"]]
        The disconnections to perform, padded to match the batch size
    injections: Optional[InjectionComputations]
        The injection computations to perform, padded to match the batch size
    result_storage : PyTree[Shaped, " n_topologies ..."]
        An array with storage reserved for the results
    dynamic_information : DynamicInformation
        Dynamic information about the grid, such as the PTDF matrix
    solver_config : SolverConfig
        Configuration for the solver
    aggregate_metric_fn : AggregateMetricProtocol
        A function that aggregates the results to a metric
    aggregate_output_fn : AggregateOutputProtocol
        A function that aggregates the results to a user-defined output

    Returns
    -------
    PyTree[Shaped, " n_topologies ..."]
        The aggregated results for every topology according to the aggregate_output_fn. The results
        are obtained by applying aggregate_output_fn to the N-0 and N-1 matrices of every topology
        and then stacking the results. They are of the same dimension as the inputs
    Int[Array, " n_topologies n_sub_relevant"]
        The best injection combination for every topology
    Bool[Array, " n_topologies"]
        The success mask for every topology
    """
    n_topologies = len(topologies)
    batch_size = solver_config.batch_size_bsdf
    n_batches = n_topologies // batch_size
    n_splits = topologies.action.shape[1]
    assert n_topologies % batch_size == 0
    assert solver_config.buffer_size_injection is not None

    # Reserve storage for the results
    best_inj_combi = jnp.zeros((n_topologies, n_splits, dynamic_information.max_inj_per_sub), dtype=bool)
    success = jnp.zeros(n_topologies, dtype=bool)

    def _run_single_iter(
        i: Int[Array, " "],
        storage: tuple[
            PyTree[Shaped, " n_topologies ..."],
            Int[Array, " n_topologies n_sub_relevant"],
            Bool[Array, " n_topologies"],
        ],
    ) -> tuple[
        PyTree[Shaped, " n_topologies ..."],
        Int[Array, " n_topologies n_sub_relevant"],
        Bool[Array, " n_topologies"],
    ]:
        topologies_cur = slice_topologies_action_index(topologies, i, batch_size)
        disconnections_cur = (
            jax.lax.dynamic_slice_in_dim(disconnections, i * batch_size, batch_size, axis=0)
            if disconnections is not None
            else None
        )
        injections_cur = get_injections_for_topo_range(
            all_injections=injections,
            topo_index=i,
            batch_size_bsdf=batch_size,
            batch_size_injection=solver_config.batch_size_injection,
            buffer_size_injection=solver_config.buffer_size_injection,
            return_relative_index=True,
        )

        output_new, best_inj_combi_new, success_new = compute_batch(
            topologies_cur,
            disconnections_cur,
            injections_cur,
            dynamic_information,
            solver_config,
            aggregate_metric_fn,
            aggregate_output_fn,
        )

        output, best_inj_combi, success = storage
        output = jax.tree_util.tree_map(
            lambda stored, new: jax.lax.dynamic_update_slice_in_dim(stored, new, i * batch_size, axis=0),
            output,
            output_new,
        )
        best_inj_combi = jax.lax.dynamic_update_slice_in_dim(best_inj_combi, best_inj_combi_new, i * batch_size, axis=0)
        success = jax.lax.dynamic_update_slice_in_dim(success, success_new, i * batch_size, axis=0)

        return (output, best_inj_combi, success)

    results = jax.lax.fori_loop(0, n_batches, _run_single_iter, (result_storage, best_inj_combi, success))
    return results


# sonar: noqa: S3776
def run_solver_symmetric(
    topologies: ActionIndexComputations,
    disconnections: Optional[Int[Array, " n_topologies n_disconnections"]],
    injections: Optional[Bool[Array, " n_topologies n_splits max_inj_per_sub"]],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    aggregate_output_fn: AggregateOutputProtocol,
    nodal_inj_start_options: Optional[NodalInjStartOptions] = None,
) -> tuple[PyTree[Shaped, " n_topologies ..."], Bool[Array, " n_topologies"]]:
    """Run the symmetric solver for a given set of topologies and injections.

    Symmetric means that there is exactly one injection topology per branch topology. You can pass
    any number of topologies, the function will batch them internally to batch_size_bsdf.

    Parameters
    ----------
    topologies :ActionIndexBranchComputations
        The topology computations to perform, in either topo vect or action index format. Will be converted
        to action index format internally
    disconnections : Optional[Int[Array, " n_topologies n_disconntections"]]
        The disconntections to perform as topological measures. If None, no disconntections are performed
        There should be exactly one disconntection vector per topology, if given.
    injections : Optional[Bool[Array, " n_topologies n_splits max_inj_per_sub"]]
        The injection computations to perform. There must be exactly one injection computation per
        branch topology, hence the first dimension should match the batch dimension in topologies. If no value is passed, the
        topologies from the action index are used
    dynamic_information : DynamicInformation
        Dynamic information about the grid, such as the PTDF matrix
    solver_config : SolverConfig
        Configuration for the solver
    aggregate_output_fn : AggregateOutputProtocol
        A function that takes the N-0 and N-1 matrices of a single topology (no batch dimension)
        and returns any aggregated information the user wishes to compute. The aggregate function
        must be vmappable.
    nodal_inj_start_options : Optional[NodalInjStartOptions], optional
        Starting options for nodal injection optimization. If None (default), optimization is disabled.

    Returns
    -------
    PyTree[Shaped, " n_topologies ..."]
        The aggregated results for every topology according to the aggregate_output_fn. The results
        are obtained by applying aggregate_output_fn to the N-0 and N-1 matrices of every topology
        and then stacking the results.
    Bool[Array, " n_topologies"]]
        The success mask for every topology
    """
    if not topologies.action.size:
        topologies = default_topology(solver_config=solver_config)
    if disconnections is not None and disconnections.size == 0:
        disconnections = None

    # Pad out inputs to match the batch size / devices
    distributed = solver_config.distributed
    n_devices = len(jax.devices()) if distributed else 1
    batch_size = solver_config.batch_size_bsdf * n_devices
    n_topologies_orig = len(topologies)
    n_topologies = math.ceil(n_topologies_orig / batch_size) * batch_size
    pad_width = n_topologies - n_topologies_orig

    topologies = pad_topologies_action_index(topologies, n_topologies)
    disconnections = (
        jnp.pad(
            disconnections,
            [[0, pad_width], [0, 0]],
            mode="constant",
            constant_values=-1,
        )
        if disconnections is not None
        else None
    )
    injections = (
        jnp.pad(injections, [[0, pad_width], [0, 0], [0, 0]], mode="constant", constant_values=False)
        if injections is not None
        else None
    )

    # Prepare storage for all results
    result_storage = prepare_result_storage(
        aggregate_output_fn,
        n_timesteps=dynamic_information.n_timesteps,
        n_branches_monitored=dynamic_information.n_branches_monitored,
        n_failures=dynamic_information.n_nminus1_cases,
        n_splits=topologies.action.shape[1],
        n_disconnections=disconnections.shape[1] if disconnections is not None else None,
        max_branch_per_sub=solver_config.max_branch_per_sub,
        max_inj_per_sub=dynamic_information.max_inj_per_sub,
        nminus2=dynamic_information.n2_baseline_analysis is not None,
        size=n_topologies,
        bb_outage=solver_config.enable_bb_outages and not solver_config.bb_outage_as_nminus1,
    )

    if distributed:
        devices = jax.devices()
        n_devices = len(devices)

        topologies = jax.tree_util.tree_map(
            lambda x: jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices),
            topologies,
        )
        disconnections = (
            jax.device_put_sharded(jnp.split(disconnections, n_devices, axis=0), devices)
            if disconnections is not None
            else None
        )
        injections = (
            jax.device_put_sharded(jnp.split(injections, n_devices, axis=0), devices) if injections is not None else None
        )
        result_storage = jax.tree_util.tree_map(
            lambda x: jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices),
            result_storage,
        )

        results, success = jax.pmap(
            iterate_symmetric_sequential,
            in_axes=(
                0,
                0 if disconnections is not None else None,
                0 if injections is not None else None,
                0,
                0,
                jax.tree_util.tree_map(lambda _: None, dynamic_information),
                jax.tree_util.tree_map(lambda _: None, solver_config),
                None,
            ),
            out_axes=(0, 0),
            static_broadcasted_argnums=(6, 7),
            donate_argnums=(3,),
        )(
            topologies,
            disconnections,
            injections,
            result_storage,
            nodal_inj_start_options,
            dynamic_information,
            solver_config,
            aggregate_output_fn,
        )

        results = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), results)
        success = jnp.concatenate(success, axis=0)
    else:
        results, success = iterate_symmetric_sequential(
            topologies=topologies,
            disconnections=disconnections,
            injections=injections,
            result_storage=result_storage,
            nodal_inj_start_options=nodal_inj_start_options,
            dynamic_information=dynamic_information,
            solver_config=solver_config,
            aggregate_output_fn=aggregate_output_fn,
        )

    results = jax.tree_util.tree_map(lambda x: x[:n_topologies_orig], results)
    success = success[:n_topologies_orig]
    return results, success


@partial(jax.jit, static_argnames=("solver_config", "aggregate_output_fn"))
def iterate_symmetric_sequential(
    topologies: ActionIndexComputations,
    disconnections: Optional[Int[Array, " n_topologies n_disconnections"]],
    injections: Optional[Bool[Array, " n_topologies n_splits max_inj_per_sub"]],
    result_storage: PyTree[Shaped, " n_topologies ..."],
    nodal_inj_start_options: Optional[NodalInjStartOptions],
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    aggregate_output_fn: AggregateOutputProtocol,
) -> tuple[PyTree[Shaped, " n_topologies ..."], Bool[Array, " n_topologies"]]:
    """Iterate over already padded topologies and injections sequentially

    Parameters
    ----------
    topologies : ActionIndexBranchComputations
        The topology computations to perform, padded to match the batch size
    disconnections : Optional[Int[Array, " n_topologies n_disconnections"]]
        The disconnections to perform as topological measures, padded to match the batch size
    injections : Optional[Bool[Array, " n_topologies n_splits max_inj_per_sub"]]
        The injection computations to perform, padded to match the batch size
    result_storage : PyTree[Shaped, " n_topologies ..."]
        An array with storage reserved for the results
    nodal_inj_start_options : Optional[NodalInjStartOptions]
        Starting options for nodal injection optimization. If None, optimization is disabled.
    dynamic_information : DynamicInformation
        Dynamic information about the grid, such as the PTDF matrix
    solver_config : SolverConfig
        Configuration for the solver
    aggregate_output_fn : AggregateOutputProtocol
        A function that takes the N-0 and N-1 matrices of a single topology (no batch dimension)
        and returns any aggregated information the user wishes to compute. The aggregate function
        must be vmappable.

    Returns
    -------
    PyTree[Shaped, " n_topologies ..."]
        The aggregated results for every topology according to the aggregate_output_fn. The results
        are obtained by applying aggregate_output_fn to the N-0 and N-1 matrices of every topology
        and then stacking the results. They are of the same dimension as the inputs
    Bool[Array, " n_topologies"]
        The success mask for every topology
    """
    n_topologies = len(topologies)
    batch_size = solver_config.batch_size_bsdf
    n_batches = n_topologies // batch_size
    assert n_topologies % batch_size == 0

    success_storage = jnp.zeros(n_topologies, dtype=bool)
    storage = (result_storage, success_storage)

    def _run_single_iter(
        i: Int[Array, " "],
        storage: tuple[PyTree[Shaped, " n_topologies ..."], Bool[Array, " n_topologies"]],
    ) -> tuple[PyTree[Shaped, " n_topologies ..."], Bool[Array, " n_topologies"]]:
        topology_batch = slice_topologies_action_index(topologies, i, batch_size)
        disconnections_batch = (
            jax.lax.dynamic_slice_in_dim(disconnections, i * batch_size, batch_size, axis=0)
            if disconnections is not None
            else None
        )
        injections_batch = (
            jax.lax.dynamic_slice_in_dim(injections, i * batch_size, batch_size, axis=0) if injections is not None else None
        )
        nodal_inj_start_options_batch = (
            slice_nodal_inj_start_options(nodal_inj_start_options, i, batch_size)
            if nodal_inj_start_options is not None
            else None
        )

        lf_res, new_succ = compute_symmetric_batch(
            topology_batch,
            disconnections_batch,
            injections_batch,
            nodal_inj_start_options_batch,
            dynamic_information,
            solver_config,
        )
        new_res = jax.vmap(aggregate_output_fn)(lf_res)
        stored_res, stored_succ = storage
        stored_res = jax.tree_util.tree_map(
            lambda stored, new: jax.lax.dynamic_update_slice_in_dim(stored, new, i * batch_size, axis=0),
            stored_res,
            new_res,
        )
        stored_succ = jax.lax.dynamic_update_slice_in_dim(stored_succ, new_succ, i * batch_size, axis=0)
        return (stored_res, stored_succ)

    results, success = jax.lax.fori_loop(0, n_batches, _run_single_iter, storage)
    return results, success
