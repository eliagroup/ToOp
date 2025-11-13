"""Module: optimiser_dc_params_factory

This module provides factory functions for creating protobuf messages related to DC optimizer parameters
. It includes utilities for constructing configuration messages
for genetic algorithms, loadflow solvers, and target metrics, ensuring proper validation and default values.

Functions
---------
- create_target_metric(metric: MetricType, weight: float) -> PbTargetMetric
    Create a TargetMetric protobuf message for optimization objectives.

- create_batched_me_parameters(...) -> PbBatchedMEParameters
    Create a BatchedMEParameters protobuf message for configuring the genetic algorithm, including mutation,
    crossover, metrics, descriptors, and runtime options.

- create_loadflow_solver_parameters(...) -> PbLoadflowSolverParameters
    Create a LoadflowSolverParameters protobuf message for configuring loadflow solver options.

- create_dc_optimizer_parameters(...) -> PbDCOptimizerParameters
    Create a DCOptimizerParameters protobuf message, aggregating GA and solver configs, double limits,
    and summary/check frequencies.
"""

from typing import List, Optional, Union

from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import DescriptorDef as PbDescriptorDef
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    DoubleLimitsSetpoint as PbDoubleLimitsSetpoint,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.dc_optimizer_params_pb2 import (
    BatchedMEParameters as PbBatchedMEParameters,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.dc_optimizer_params_pb2 import (
    DCOptimizerParameters as PbDCOptimizerParameters,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.dc_optimizer_params_pb2 import (
    LoadflowSolverParameters as PbLoadflowSolverParameters,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.dc_optimizer_params_pb2 import (
    TargetMetric as PbTargetMetric,
)
from toop_engine_interfaces.types import MetricType


def create_target_metric(metric: MetricType, weight: float) -> PbTargetMetric:
    """
    Create a TargetMetric message.

    Parameters
    ----------
    metric : MetricType
        The metric name to optimize.
    weight : float
        The weight assigned to this metric in the optimization objective.

    Returns
    -------
    PbTargetMetric
        A protobuf `TargetMetric` instance.

    Raises
    ------
    ValueError
        If the metric name is empty or weight is not a finite float.
    """
    if not metric:
        raise ValueError("Metric name must not be empty.")
    if not isinstance(weight, (float, int)):
        raise ValueError("Weight must be a float or int.")

    return PbTargetMetric(metric=metric, weight=float(weight))


def create_batched_me_parameters(  # noqa: PLR0913, C901
    substation_split_prob: float = 0.1,
    substation_unsplit_prob: float = 0.01,
    disconnect_prob: float = 0.1,
    reconnect_prob: float = 0.1,
    n_subs_mutated_lambda: float = 2.0,
    proportion_crossover: float = 0.1,
    crossover_mutation_ratio: float = 0.5,
    target_metrics: Optional[tuple[PbTargetMetric, ...]] = None,
    observed_metrics: Union[tuple[MetricType, ...], List[MetricType]] = (
        "max_flow_n_0",
        "overload_energy_n_0",
        "overload_energy_limited_n_0",
        "max_flow_n_1",
        "overload_energy_n_1",
        "overload_energy_limited_n_1",
        "split_subs",
        "switching_distance",
    ),
    me_descriptors: Optional[Union[tuple[PbDescriptorDef, ...], List[PbDescriptorDef]]] = None,
    runtime_seconds: float = 60,
    iterations_per_epoch: int = 100,
    random_seed: int = 42,
    cell_depth: int = 1,
    plot: bool = False,
    mutation_repetition: int = 1,
    n_worst_contingencies: int = 20,
) -> PbBatchedMEParameters:
    """
    Create a BatchedMEParameters message.

    Parameters
    ----------
    substation_split_prob : float
        Probability to split an unsplit substation.
    substation_unsplit_prob : float
        Probability to reset a split substation to the unsplit state.
    disconnect_prob : float
        Probability to disconnect a branch.
    reconnect_prob : float
        Probability to reconnect a disconnected branch.
    n_subs_mutated_lambda : float
        Expected number of substations to mutate per iteration (Poisson λ).
    proportion_crossover : float
        Proportion of the first topology used during crossover.
    crossover_mutation_ratio : float
        Ratio of crossover to mutation events.
    target_metrics : list of PbTargetMetric
        The metrics to optimize with their weights.
    observed_metrics : list of str
        The metrics observed for logging. Must include target metrics and ME descriptors.
    me_descriptors : list of PbDescriptorDef
        The MAP-Elites descriptors defining the repertoire.
    runtime_seconds : float
        The runtime in seconds for the optimization.
    iterations_per_epoch : int
        The number of iterations per epoch.
    random_seed : int
        Random seed for reproducibility.
    cell_depth : int
        Number of unique topologies per cell.
    plot : bool
        Whether to plot the repertoire.
    mutation_repetition : int
        How many times to repeat mutations for diversity.
    n_worst_contingencies : int
        Number of worst contingencies to consider in scoring.

    Returns
    -------
    PbBatchedMEParameters
        A protobuf `BatchedMEParameters` instance.

    Raises
    ------
    ValueError
        If probability values are out of range [0, 1], or numeric parameters are invalid.
    """
    for name, prob in {
        "substation_split_prob": substation_split_prob,
        "substation_unsplit_prob": substation_unsplit_prob,
        "disconnect_prob": disconnect_prob,
        "reconnect_prob": reconnect_prob,
    }.items():
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"{name} must be in range [0, 1]. Got {prob}.")

    if n_subs_mutated_lambda < 0:
        raise ValueError("n_subs_mutated_lambda must be non-negative.")
    if proportion_crossover < 0 or crossover_mutation_ratio < 0:
        raise ValueError("proportion_crossover and crossover_mutation_ratio must be non-negative.")
    if runtime_seconds <= 0:
        raise ValueError("runtime_seconds must be positive.")
    if iterations_per_epoch <= 0 or cell_depth <= 0:
        raise ValueError("iterations_per_epoch and cell_depth must be positive integers.")
    if mutation_repetition < 0 or n_worst_contingencies < 0:
        raise ValueError("mutation_repetition and n_worst_contingencies must be non-negative integers.")

    if me_descriptors is None:
        me_descriptors = (
            PbDescriptorDef(metric="split_subs", num_cells=5),
            PbDescriptorDef(metric="switching_distance", num_cells=45),
        )
    if target_metrics is None:
        target_metrics = (PbTargetMetric(metric="overload_energy_n_1", weight=1.0),)
    # Ensure observed metrics include all target and descriptor metrics
    required_metrics = {t.metric for t in target_metrics} | {d.metric for d in me_descriptors}
    missing = required_metrics - set(observed_metrics)
    if missing:
        observed_metrics = list(set(observed_metrics) | missing)

    return PbBatchedMEParameters(
        substation_split_prob=substation_split_prob,
        substation_unsplit_prob=substation_unsplit_prob,
        disconnect_prob=disconnect_prob,
        reconnect_prob=reconnect_prob,
        n_subs_mutated_lambda=n_subs_mutated_lambda,
        proportion_crossover=proportion_crossover,
        crossover_mutation_ratio=crossover_mutation_ratio,
        target_metrics=target_metrics,
        observed_metrics=observed_metrics,
        me_descriptors=me_descriptors,
        runtime_seconds=runtime_seconds,
        iterations_per_epoch=iterations_per_epoch,
        random_seed=random_seed,
        cell_depth=cell_depth,
        plot=plot,
        mutation_repetition=mutation_repetition,
        n_worst_contingencies=n_worst_contingencies,
    )


def create_loadflow_solver_parameters(
    max_num_splits: int = 4,
    max_num_disconnections: int = 0,
    batch_size: int = 8,
    distributed: bool = False,
    cross_coupler_flow: bool = False,
) -> PbLoadflowSolverParameters:
    """
    Create a LoadflowSolverParameters message.

    Parameters
    ----------
    max_num_splits : int, optional
        Maximum number of splits per topology. Default is 4.
    max_num_disconnections : int, optional
        Maximum number of disconnections per topology. Default is 0.
    batch_size : int, optional
        The batch size for the genetic algorithm. Default is 8.
    distributed : bool, optional
        Whether to run the GA distributed across devices. Default is False.
    cross_coupler_flow : bool, optional
        Whether to compute cross-coupler flows. Default is False.

    Returns
    -------
    PbLoadflowSolverParameters
        A protobuf `LoadflowSolverParameters` instance.

    Raises
    ------
    ValueError
        If numeric parameters are non-positive.
    """
    if max_num_splits < 0 or max_num_disconnections < 0 or batch_size <= 0:
        raise ValueError("max_num_splits, max_num_disconnections must be >= 0 and batch_size > 0.")

    return PbLoadflowSolverParameters(
        max_num_splits=max_num_splits,
        max_num_disconnections=max_num_disconnections,
        batch_size=batch_size,
        distributed=distributed,
        cross_coupler_flow=cross_coupler_flow,
    )


def create_dc_optimizer_parameters(
    ga_config: PbBatchedMEParameters = None,
    loadflow_solver_config: PbLoadflowSolverParameters = None,
    double_limits: Optional[PbDoubleLimitsSetpoint] = None,
    summary_frequency: int = 10,
    check_command_frequency: int = 10,
) -> PbDCOptimizerParameters:
    """
    Create a DCOptimizerParameters message.

    Parameters
    ----------
    ga_config : PbBatchedMEParameters
        The configuration options for the genetic algorithm.
    loadflow_solver_config : PbLoadflowSolverParameters
        The configuration options for the loadflow solver.
    double_limits : PbDoubleLimitsSetpoint, optional
        The double limits for the optimization, if provided.
    summary_frequency : int, optional
        The frequency (in iterations) to push summary results. Default is 10.
    check_command_frequency : int, optional
        The frequency (in iterations) to check for new commands. Default is 10.
        Must be a multiple of `summary_frequency`.

    Returns
    -------
    PbDCOptimizerParameters
        A protobuf `DCOptimizerParameters` instance.

    Raises
    ------
    ValueError
        If check_command_frequency is not a multiple of summary_frequency.
    """
    if ga_config is None:
        ga_config = create_batched_me_parameters()
    if loadflow_solver_config is None:
        loadflow_solver_config = create_loadflow_solver_parameters()

    if summary_frequency <= 0 or check_command_frequency <= 0:
        raise ValueError("Frequencies must be positive integers.")
    if check_command_frequency % summary_frequency != 0:
        raise ValueError("check_command_frequency must be a multiple of summary_frequency.")

    return PbDCOptimizerParameters(
        ga_config=ga_config,
        loadflow_solver_config=loadflow_solver_config,
        double_limits=double_limits,
        summary_frequency=summary_frequency,
        check_command_frequency=check_command_frequency,
    )
