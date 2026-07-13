# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Standalone exhaustive DC bruteforce optimizer runtime."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Iterator, Sequence
from fsspec import AbstractFileSystem
from jax_dataclasses import replace
from jaxtyping import Array, ArrayLike, Float, Int
from toop_engine_dc_solver.jax.inputs import load_static_information_fs
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    DynamicInformation,
    SolverConfig,
    SolverLoadflowResults,
    StaticInformation,
    int_max,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import StaticInformationStats, extract_static_information_stats
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (
    update_max_mw_flows_according_to_double_limits,
    update_static_information,
    verify_static_information,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    get_aggregate_metrics,
)
from toop_engine_topology_optimizer.dc_bruteforce.generator import (
    WorksetEntry,
    count_workset_size,
    iter_workset,
    take_workset_chunk,
)
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, Strategy, Topology, TopologyPushResult


@dataclass
class BruteforceRuntimeState:
    """State kept across bruteforce epochs."""

    dynamic_information: DynamicInformation
    """Dynamic solver inputs for the single supported timestep."""

    solver_config: SolverConfig
    """Static solver configuration for the single supported timestep."""

    workset: Iterator[WorksetEntry]
    """Lazy iterator over the remaining bruteforce candidates."""

    total_workset_size: int
    """Exact number of candidates represented by the bruteforce workset."""

    total_branch_topologies_tried: int = 0
    """Number of bruteforce candidates scored so far."""

    exhausted: bool = False
    """Whether the lazy workset has been fully consumed."""

    best_fitness: float = float("-inf")
    """Best fitness observed so far across all evaluated candidates."""


@dataclass
class OptimizerData:
    """Outer state for the standalone bruteforce worker."""

    start_params: DCOptimizerParameters
    """Optimization parameters used to initialize the bruteforce run."""

    optimization_id: str
    """Identifier of the optimization run, propagated to emitted messages."""

    initial_fitness: float
    """Fitness of the unsplit baseline topology."""

    initial_metrics: dict[MetricType, float]
    """Observed metrics for the unsplit baseline topology."""

    runtime_state: BruteforceRuntimeState
    """Mutable bruteforce state updated after every epoch."""

    start_time: float
    """Wall-clock timestamp when the bruteforce run started."""

    latest_topologies: list[Topology] = field(default_factory=list)
    """Improved topologies produced by the most recent epoch."""


def initialize_optimization(
    params: DCOptimizerParameters,
    optimization_id: str,
    static_information_files: Sequence[str | Path],
    processed_gridfile_fs: AbstractFileSystem,
) -> tuple[OptimizerData, list[StaticInformationStats], Strategy]:
    """Initialize the bruteforce optimization run.

    Parameters
    ----------
    params : DCOptimizerParameters
        Parameters controlling the DC bruteforce optimization.
    optimization_id : str
        Identifier of the optimization run, used in emitted messages.
    static_information_files : Sequence[str | Path]
        Paths to the preprocessed static-information files for all timesteps.
    processed_gridfile_fs : AbstractFileSystem
        Filesystem containing the preprocessed grid data.

    Returns
    -------
    tuple[OptimizerData, list[StaticInformationStats], Strategy]
        The initialized optimizer state, static-information descriptions, and initial unsplit
        strategy.
    """
    topologies_per_epoch = _get_topologies_per_epoch(params)
    solver_batch_size = params.loadflow_solver_config.batch_size

    static_information = _load_and_prepare_static_informations(
        params=params,
        static_information_files=static_information_files,
        processed_gridfile_fs=processed_gridfile_fs,
        batch_size=solver_batch_size,
    )
    _validate_bruteforce_inputs(params, topologies_per_epoch, static_information.dynamic_information)
    runtime_state, initial_fitness, initial_metrics, initial_case_ids = _initialize_runtime_state(
        params=params,
        static_information=static_information,
    )
    initial_strategy = build_initial_strategy(
        fitness=initial_fitness,
        initial_metrics=initial_metrics,
        initial_case_ids=initial_case_ids,
    )

    static_information_descriptions = [
        extract_static_information_stats(
            static_information=static_information,
            overload_n0=initial_metrics.get("overload_energy_n_0", 0.0),
            overload_n1=initial_metrics.get("overload_energy_n_1", 0.0),
            time="",
        )
    ]

    return (
        OptimizerData(
            start_params=params,
            optimization_id=optimization_id,
            initial_fitness=initial_fitness,
            initial_metrics=initial_metrics,
            runtime_state=runtime_state,
            start_time=time.time(),
        ),
        static_information_descriptions,
        initial_strategy,
    )


def build_initial_strategy(
    fitness: float,
    initial_metrics: dict[MetricType, float],
    initial_case_ids: list[str],
) -> Strategy:
    """Build the initial unsplit strategy message.

    Parameters
    ----------
    fitness : float
        Fitness of the unsplit baseline topology.
    initial_metrics : dict[MetricType, float]
        Observed metrics of the unsplit baseline topology.
    initial_case_ids : list[str]
        Worst contingency case identifiers for the baseline topology.

    Returns
    -------
    Strategy
        The initial strategy containing the unsplit topology for every timestep.
    """
    metrics = Metrics(
        fitness=fitness,
        extra_scores=dict(initial_metrics),
        worst_k_contingency_cases=initial_case_ids,
    )
    return Strategy(
        timesteps=[
            Topology(
                actions=[],
                disconnections=[],
                pst_setpoints=None,
                metrics=metrics,
            )
        ]
    )


def run_epoch(optimizer_data: OptimizerData) -> OptimizerData:
    """Run one bruteforce epoch.

    Parameters
    ----------
    optimizer_data : OptimizerData
        Current state of the bruteforce optimization.

    Returns
    -------
    OptimizerData
        Updated optimizer state after evaluating one chunk of the workset.
    """
    runtime_state = optimizer_data.runtime_state
    topologies_per_epoch = _get_topologies_per_epoch(optimizer_data.start_params)
    if runtime_state.exhausted:
        return replace(optimizer_data, latest_topologies=[])

    chunk = take_workset_chunk(runtime_state.workset, topologies_per_epoch)
    if not chunk:
        return replace(
            optimizer_data,
            runtime_state=replace(runtime_state, exhausted=True),
            latest_topologies=[],
        )

    max_num_splits = optimizer_data.start_params.loadflow_solver_config.max_num_splits
    max_num_disconnections = optimizer_data.start_params.loadflow_solver_config.max_num_disconnections
    evaluated_count = len(chunk)
    topology_chunk, disconnection_chunk = _chunk_to_topologies(
        chunk=chunk,
        chunk_size=topologies_per_epoch,
        max_num_splits=max_num_splits,
        max_num_disconnections=max_num_disconnections,
    )

    fitness, metrics = _score_chunk(
        topology_chunk=topology_chunk,
        disconnection_chunk=disconnection_chunk,
        evaluated_count=evaluated_count,
        dynamic_information=runtime_state.dynamic_information,
        solver_config=runtime_state.solver_config,
        target_metrics=optimizer_data.start_params.ga_config.target_metrics,
        observed_metrics=optimizer_data.start_params.ga_config.observed_metrics,
        n_worst_contingencies=optimizer_data.start_params.ga_config.n_worst_contingencies,
    )

    evaluated_fitness = np.asarray(fitness[:evaluated_count])
    improved_indices = np.flatnonzero(np.isfinite(evaluated_fitness) & (evaluated_fitness > optimizer_data.initial_fitness))
    topologies_out = _convert_improved_topologies(
        topology_actions=topology_chunk.action,
        disconnection_chunk=disconnection_chunk,
        fitness=evaluated_fitness,
        metrics={metric_name: np.asarray(metric_values[:evaluated_count]) for metric_name, metric_values in metrics.items()},
        observed_metrics=optimizer_data.start_params.ga_config.observed_metrics,
        contingency_ids=list(runtime_state.solver_config.contingency_ids),
        survivor_indices=improved_indices,
    )

    finite_fitness = evaluated_fitness[np.isfinite(evaluated_fitness)]
    best_fitness = runtime_state.best_fitness
    if finite_fitness.size > 0:
        best_fitness = max(best_fitness, float(finite_fitness.max()))

    exhausted = evaluated_count < topologies_per_epoch

    return replace(
        optimizer_data,
        runtime_state=replace(
            runtime_state,
            total_branch_topologies_tried=runtime_state.total_branch_topologies_tried + evaluated_count,
            exhausted=exhausted,
            best_fitness=best_fitness,
        ),
        latest_topologies=topologies_out,
    )


def _get_topologies_per_epoch(params: DCOptimizerParameters) -> int:
    """Compute how many topologies the bruteforce optimizer evaluates per epoch.

    Parameters
    ----------
    params : DCOptimizerParameters
        Optimization parameters controlling the bruteforce runtime.

    Returns
    -------
    int
        Number of topologies evaluated in a single bruteforce epoch.
    """
    return params.ga_config.iterations_per_epoch * params.loadflow_solver_config.batch_size


def convert_topologies_to_messages(topologies: list[Topology], epoch: int) -> list[TopologyPushResult]:
    """Convert topologies to result messages.

    Parameters
    ----------
    topologies : list[Topology]
        Topologies to emit.
    epoch : int
        Epoch in which the topologies were discovered.

    Returns
    -------
    list[TopologyPushResult]
        Result messages ready to be sent to Kafka.
    """
    return [TopologyPushResult(strategy=Strategy(timesteps=[topology]), epoch=epoch) for topology in topologies]


def get_num_branch_topologies_tried(optimizer_data: OptimizerData) -> int:
    """Return the number of branch topologies evaluated so far.

    Parameters
    ----------
    optimizer_data : OptimizerData
        Current state of the bruteforce optimization.

    Returns
    -------
    int
        Number of scored bruteforce candidates.
    """
    return optimizer_data.runtime_state.total_branch_topologies_tried


def is_exhausted(optimizer_data: OptimizerData) -> bool:
    """Check whether the workset has been fully consumed.

    Parameters
    ----------
    optimizer_data : OptimizerData
        Current state of the bruteforce optimization.

    Returns
    -------
    bool
        ``True`` if no more candidates remain to be evaluated.
    """
    return optimizer_data.runtime_state.exhausted


def _validate_bruteforce_inputs(
    params: DCOptimizerParameters, topologies_per_epoch: int, dynamic_information: DynamicInformation
) -> None:
    """Validate standalone bruteforce constraints.

    Parameters
    ----------
    params : DCOptimizerParameters
        Optimization parameters to validate.
    topologies_per_epoch : int
        Number of bruteforce candidates to evaluate per epoch.
    dynamic_information : DynamicInformation
        Dynamic solver inputs for the first timestep, used to validate available actions.

    Raises
    ------
    ValueError
        If the chunk size is invalid or if unsupported optimizer features are enabled.
    """
    if topologies_per_epoch <= 0:
        raise ValueError(f"topologies_per_epoch must be positive, got {topologies_per_epoch}")
    if params.ga_config.enable_nodal_inj_optim:
        raise ValueError("Bruteforce optimizer keeps PSTs untouched and does not support nodal injection optimization.")
    if params.ga_config.enable_parallel_pst_group_optim:
        raise ValueError("Bruteforce optimizer does not support parallel PST group optimization.")
    if params.loadflow_solver_config.distributed:
        raise ValueError("Bruteforce optimizer does not support distributed execution.")

    action_set = dynamic_information.action_set
    assert action_set is not None, "Bruteforce optimization requires an action set."
    assert params.loadflow_solver_config.max_num_splits <= len(action_set.action_start_indices), (
        "Bruteforce optimizer requires max_num_splits to be smaller than or equal to the number of substations "
        "with available actions."
    )
    assert params.loadflow_solver_config.max_num_disconnections <= dynamic_information.n_disconnectable_branches, (
        "Bruteforce optimizer requires max_num_disconnections to be smaller than or equal to the number of "
        "disconnectable branches."
    )


def _load_and_prepare_static_informations(
    params: DCOptimizerParameters,
    static_information_files: Sequence[str | Path],
    processed_gridfile_fs: AbstractFileSystem,
    batch_size: int,
) -> StaticInformation:
    """Load and preprocess the single static information used by bruteforce.

    Parameters
    ----------
    params : DCOptimizerParameters
        Optimization parameters controlling preprocessing options.
    static_information_files : Sequence[str | Path]
        Paths to the preprocessed static-information files. Bruteforce supports exactly one.
    processed_gridfile_fs : AbstractFileSystem
        Filesystem containing the preprocessed grid data.
    batch_size : int
        Batch size to configure in the updated static information.

    Returns
    -------
    StaticInformation
        Prepared static-information object for the single supported timestep.
    """
    assert len(static_information_files) == 1, "Bruteforce optimizer supports exactly one static information file."
    static_information = load_static_information_fs(
        filesystem=processed_gridfile_fs,
        filename=str(static_information_files[0]),
    )
    assert static_information.dynamic_information.n_timesteps == 1, "Bruteforce optimizer supports exactly one timestep."

    verify_static_information(
        (static_information,),
        params.loadflow_solver_config.max_num_disconnections,
        enable_nodal_inj_optim=False,
        enable_parallel_pst_group_optim=False,
    )
    static_information = update_static_information(
        (static_information,),
        batch_size=batch_size,
        enable_nodal_inj_optim=False,
        enable_parallel_pst_group_optim=False,
        enable_bb_outage=params.ga_config.enable_bb_outage,
        bb_outage_as_nminus1=params.ga_config.bb_outage_as_nminus1,
        clip_bb_outage_penalty=params.ga_config.clip_bb_outage_penalty,
        bb_outage_more_islands_penalty=params.ga_config.bb_outage_more_islands_penalty,
    )[0]

    if params.double_limits is not None:
        dynamic_information = update_max_mw_flows_according_to_double_limits(
            dynamic_informations=(static_information.dynamic_information,),
            solver_configs=(static_information.solver_config,),
            lower_limit=params.double_limits.lower,
            upper_limit=params.double_limits.upper,
        )[0]
        static_information = replace(static_information, dynamic_information=dynamic_information)

    return static_information


def _initialize_runtime_state(
    params: DCOptimizerParameters,
    static_information: StaticInformation,
) -> tuple[BruteforceRuntimeState, float, dict[MetricType, float], list[str]]:
    """Create the JAX scoring state and evaluate the unsplit baseline.

    Parameters
    ----------
    params : DCOptimizerParameters
        Optimization parameters for the bruteforce run.
    static_information : StaticInformation
        Prepared static-information object for the single supported timestep.

    Returns
    -------
    tuple[BruteforceRuntimeState, float, dict[MetricType, float], list[str]]
        Runtime state, baseline fitness, baseline metrics, and baseline worst contingency ids.
    """
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config
    action_set = dynamic_information.action_set
    assert action_set is not None, "Bruteforce optimization requires an action set."

    split_action_groups = tuple(
        tuple(range(int(start), int(start) + int(count)))
        for start, count in zip(action_set.action_start_indices.tolist(), action_set.n_actions_per_sub.tolist(), strict=True)
    )
    n_disconnectable_branches = dynamic_information.n_disconnectable_branches
    max_num_splits = params.loadflow_solver_config.max_num_splits
    max_num_disconnections = params.loadflow_solver_config.max_num_disconnections
    solver_batch_size = params.loadflow_solver_config.batch_size
    # Pad the initial topology to the size of one batch and run it through the scoring function
    # Reusing chunking functions.
    initial_topology_chunk, initial_disconnection_chunk = _chunk_to_topologies(
        chunk=[WorksetEntry(action_indices=(), disconnections=())],
        chunk_size=solver_batch_size,
        max_num_splits=max_num_splits,
        max_num_disconnections=max_num_disconnections,
    )
    fitness, metrics = _score_chunk(
        topology_chunk=initial_topology_chunk,
        disconnection_chunk=initial_disconnection_chunk,
        evaluated_count=1,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        target_metrics=params.ga_config.target_metrics,
        observed_metrics=params.ga_config.observed_metrics,
        n_worst_contingencies=params.ga_config.n_worst_contingencies,
    )
    initial_fitness = float(np.asarray(fitness[0]).item())
    initial_metrics: dict[MetricType, float] = {
        metric_name: float(np.asarray(metrics[metric_name][0]).item()) for metric_name in params.ga_config.observed_metrics
    }
    initial_case_ids = _case_ids_from_metric(metrics["case_indices"][0], solver_config.contingency_ids)

    runtime_state = BruteforceRuntimeState(
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        workset=iter_workset(
            action_start_indices=action_set.action_start_indices.tolist(),
            n_actions_per_sub=action_set.n_actions_per_sub.tolist(),
            max_num_splits=max_num_splits,
            n_disconnectable_branches=n_disconnectable_branches,
            max_num_disconnections=max_num_disconnections,
        ),
        total_workset_size=count_workset_size(
            split_action_groups=split_action_groups,
            max_num_splits=max_num_splits,
            n_disconnectable_branches=n_disconnectable_branches,
            max_num_disconnections=max_num_disconnections,
        ),
        best_fitness=initial_fitness,
    )
    return runtime_state, initial_fitness, initial_metrics, initial_case_ids


class _AggregateMetricsOutputFn:
    """Adapt the shared batch metric helper to the solver's per-topology callback API."""

    def __init__(
        self,
        dynamic_information: DynamicInformation,
        observed_metrics: tuple[MetricType, ...],
        n_worst_contingencies: int,
        fixed_hash: int,
    ) -> None:
        """Create a per-topology aggregation callback for ``run_solver_symmetric``.

        Parameters
        ----------
        dynamic_information : DynamicInformation
            Dynamic grid information required by the metric aggregation.
        observed_metrics : tuple[MetricType, ...]
            Metrics that should be returned for each topology.
        n_worst_contingencies : int
            Number of worst contingency indices to retain.
        fixed_hash : int
            Stable hash used for JAX static-argument caching.
        """
        self.dynamic_information = dynamic_information
        self.observed_metrics = observed_metrics
        self.n_worst_contingencies = n_worst_contingencies
        self.fixed_hash = fixed_hash

    def __call__(self, lf_res: SolverLoadflowResults) -> dict[str, Array]:
        """Aggregate one topology's solver output into the bruteforce metrics payload."""
        lf_res_batch = jax.tree_util.tree_map(
            lambda leaf: jnp.expand_dims(leaf, axis=0) if leaf is not None else None,
            lf_res,
        )
        success = (
            jnp.expand_dims(jnp.all(lf_res.contingency_success), axis=0)
            if lf_res.contingency_success is not None
            else jnp.ones((1,), dtype=bool)
        )
        metrics = get_aggregate_metrics(
            lf_res=lf_res_batch,
            success=success,
            dynamic_information=self.dynamic_information,
            observed_metrics=self.observed_metrics,
            n_worst_contingencies=self.n_worst_contingencies,
        )
        return jax.tree_util.tree_map(lambda leaf: leaf[0], metrics)

    def __hash__(self) -> int:
        """Return a stable hash so JAX can cache the compiled solver loop."""
        return self.fixed_hash

    def __eq__(self, other: object) -> bool:
        """Compare callbacks by hash to avoid unnecessary recompilation."""
        if not isinstance(other, _AggregateMetricsOutputFn):
            return False
        return hash(self) == hash(other)


def _score_chunk(
    topology_chunk: ActionIndexComputations,
    disconnection_chunk: Int[Array, " chunk_size n_disconnections"],
    evaluated_count: int,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
    target_metrics: tuple[tuple[MetricType, float], ...],
    observed_metrics: tuple[MetricType, ...],
    n_worst_contingencies: int,
) -> tuple[Float[Array, " chunk_size"], dict[str, Array]]:
    """Score one padded bruteforce chunk directly in solver input format.

    The chunk can span multiple solver mini-batches; ``run_solver_symmetric`` handles
    the internal mini-batching.
    """
    metrics, _contingency_success = run_solver_symmetric(
        topologies=topology_chunk,
        disconnections=disconnection_chunk,
        injections=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=_AggregateMetricsOutputFn(
            dynamic_information=dynamic_information,
            observed_metrics=observed_metrics,
            n_worst_contingencies=n_worst_contingencies,
            fixed_hash=hash((solver_config, observed_metrics, n_worst_contingencies)),
        ),
    )

    fitness = sum(-metrics[metric_name] * weight for metric_name, weight in target_metrics)
    invalid_rows = jnp.arange(len(topology_chunk)) >= evaluated_count
    fitness = jnp.where(invalid_rows, -jnp.inf, fitness)
    for metric_name in observed_metrics:
        metrics[metric_name] = jnp.where(invalid_rows, jnp.nan, metrics[metric_name])
    metrics["case_indices"] = jnp.where(invalid_rows[:, None], -1, metrics["case_indices"])
    return fitness, metrics


def _chunk_to_topologies(
    chunk: list[WorksetEntry],
    chunk_size: int,
    max_num_splits: int,
    max_num_disconnections: int,
) -> tuple[ActionIndexComputations, Int[Array, " chunk_size n_disconnections"]]:
    """Convert a lazy workset chunk into solver-ready topology arrays.

    Parameters
    ----------
    chunk : list[WorksetEntry]
        Bruteforce candidates to convert.
    chunk_size : int
        Fixed chunk size for one bruteforce epoch. Usually this is equal to ``len(chunk)`` except on the last epoch where
        the chunk might be shorter. We still pad to a fixed shape here so the solver-facing chunk path can reuse the same
        array shapes for every epoch.
    max_num_splits : int
        Maximum number of split actions per candidate.
    max_num_disconnections : int
        Maximum number of disconnections per candidate.

    Returns
    -------
    tuple[ActionIndexComputations, Int[Array, " chunk_size n_disconnections"]]
        Padded action-index topologies and aligned disconnections.
    """
    action_index = np.full((chunk_size, max_num_splits), int_max(), dtype=int)
    disconnections = np.full((chunk_size, max_num_disconnections), int_max(), dtype=int)
    pad_mask = np.zeros(chunk_size, dtype=bool)

    for row_index, entry in enumerate(chunk):
        if entry.action_indices:
            action_index[row_index, : len(entry.action_indices)] = np.asarray(entry.action_indices, dtype=int)
        if entry.disconnections:
            disconnections[row_index, : len(entry.disconnections)] = np.asarray(entry.disconnections, dtype=int)
        pad_mask[row_index] = True

    return ActionIndexComputations(
        action=jnp.asarray(action_index),
        pad_mask=jnp.asarray(pad_mask),
    ), jnp.asarray(disconnections)


def _convert_improved_topologies(
    topology_actions: Int[ArrayLike, " batch_size n_splits"],
    disconnection_chunk: Int[ArrayLike, " chunk_size n_disconnections"],
    fitness: np.ndarray,
    metrics: dict[str, Any],
    observed_metrics: tuple[MetricType, ...],
    contingency_ids: list[str],
    survivor_indices: np.ndarray,
) -> list[Topology]:
    """Convert improved chunk members into result topologies.

    Parameters
    ----------
    topology_actions : Int[ArrayLike, " batch_size n_splits"]
        Scored topology actions for the current bruteforce chunk in action-index format.
    disconnection_chunk : Int[ArrayLike, " chunk_size n_disconnections"]
        Disconnections aligned with ``topology_actions`` for the current chunk.
    fitness : np.ndarray
        Fitness values for the valid rows in the current chunk.
    metrics : dict[str, Any]
        Metrics returned by the shared scoring function.
    observed_metrics : tuple[MetricType, ...]
        Metrics that should be copied into the result payload.
    contingency_ids : list[str]
        Contingency identifiers used to resolve worst-case indices.
    survivor_indices : np.ndarray
        Indices of candidates that improve on the baseline fitness.

    Returns
    -------
    list[Topology]
        Result topologies corresponding to the improved candidates.
    """
    topologies = []
    for row_index in survivor_indices.tolist():
        action_indices = [int(value) for value in np.asarray(topology_actions[row_index]) if int(value) != int_max()]
        disconnection_values = [
            int(value) for value in np.asarray(disconnection_chunk[row_index]) if int(value) != int_max()
        ]
        topologies.append(
            Topology(
                actions=action_indices,
                disconnections=disconnection_values,
                pst_setpoints=None,
                metrics=Metrics(
                    fitness=float(fitness[row_index]),
                    extra_scores={
                        metric_name: float(np.asarray(metrics[metric_name][row_index]).item())
                        for metric_name in observed_metrics
                    },
                    worst_k_contingency_cases=_case_ids_from_metric(metrics["case_indices"][row_index], contingency_ids),
                ),
            )
        )
    return topologies


def _case_ids_from_metric(case_indices: Int[ArrayLike, " n_cases"], contingency_ids: Sequence[str]) -> list[str]:
    """Resolve worst-case contingency indices to contingency ids.

    Parameters
    ----------
    case_indices : Int[ArrayLike, " n_cases"]
        Indices of the worst contingencies as returned by the scoring function.
    contingency_ids : Sequence[str]
        Ordered contingency identifiers corresponding to solver output indices.

    Returns
    -------
    list[str]
        Contingency identifiers referenced by ``case_indices``.
    """
    return np.asarray(contingency_ids)[np.asarray(case_indices).astype(int)].tolist()
