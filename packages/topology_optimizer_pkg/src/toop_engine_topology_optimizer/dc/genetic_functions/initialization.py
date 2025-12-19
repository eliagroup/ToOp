# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Initialization of the genetic algorithm for branch and injection choice optimization."""

from functools import partial
from typing import Iterable, Optional

import jax
import jax.experimental
import jax.numpy as jnp
import logbook
from fsspec import AbstractFileSystem
from jax_dataclasses import pytree_dataclass, replace
from jaxtyping import Array, Float, Int
from qdax.core.emitters.standard_emitters import EmitterState
from qdax.utils.metrics import default_ga_metrics
from toop_engine_dc_solver.jax.aggregate_results import compute_double_limits
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.inputs import load_static_information_fs
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import (
    ActionSet,
    DynamicInformation,
    MetricType,
    SolverConfig,
    StaticInformation,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import StaticInformationStats, extract_static_information_stats
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    crossover,
    empty_repertoire,
    mutate,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    scoring_function,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
)
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DescriptorDef,
    LoadflowSolverParameters,
)

logger = logbook.Logger(__name__)


@pytree_dataclass
class JaxOptimizerData:
    """The part of the optimizer data that lives on GPU.

    If distributed is enabled, every item will have a leading device dimension.
    """

    repertoire: DiscreteMapElitesRepertoire
    """The repertoire object"""

    emitter_state: EmitterState
    """The emitter state object"""

    dynamic_informations: tuple[DynamicInformation, ...]
    """The list containing the dynamic information objects"""

    random_key: jax.random.PRNGKey
    """The random key"""

    latest_iteration: Int[Array, ""]
    """The iteration that this emitter_state/repertoire belong to"""


def update_max_mw_flows_according_to_double_limits(
    dynamic_informations: tuple[DynamicInformation, ...],
    solver_configs: tuple[SolverConfig, ...],
    lower_limit: float,
    upper_limit: float,
) -> tuple[DynamicInformation, ...]:
    """Update all dynamic informations max mw loads.

    Runs an initial n-1 analysis to determine limits in mw.

    Parameters
    ----------
    dynamic_informations: tuple[DynamicInformation, ...]
        List of static informations to calculate with max_mw_flow limits set at 1.0
    solver_configs: tuple[SolverConfig, ...]
        List of solver configurations to use for the loadflow
    lower_limit: float
        The relative lower limit to set, for branches whose n-1 flows are below the lower limit
    upper_limit: float
        The relative upper_limit determining at what relative load a branch is considered overloaded.
        Branches in the band between lower and upper limit are considered overloaded if more load is added.

    Returns
    -------
    tuple[DynamicInformation, ...]
        The updated dynamic informations with new limits set.

    """
    if lower_limit > upper_limit:
        raise ValueError(f"Lower limit {lower_limit} must be smaller than upper limit {upper_limit}")

    updated_dynamic_informations = []
    for dynamic_information, solver_config in zip(dynamic_informations, solver_configs, strict=True):
        solver_config_local = replace(solver_config, batch_size_bsdf=1)
        lf_res, success = compute_symmetric_batch(
            topology_batch=default_topology(solver_config_local),
            disconnection_batch=None,
            injections=None,
            dynamic_information=dynamic_information,
            solver_config=solver_config_local,
        )
        assert jnp.all(success)
        # We will always have N-1 limits, so we compute the N-0 limits on the N-0 loadflow results
        # However, the N-0 results lack a dimension, so we need to add a virtual "failure" dim
        limited_max_mw_flow = compute_double_limits(
            lf_res.n_0_matrix[0, :, None, :],
            dynamic_information.branch_limits.max_mw_flow,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

        limited_max_mw_flow_n_1 = compute_double_limits(
            lf_res.n_1_matrix[0],
            dynamic_information.branch_limits.max_mw_flow_n_1
            if dynamic_information.branch_limits.max_mw_flow_n_1 is not None
            else dynamic_information.branch_limits.max_mw_flow,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

        updated_dynamic_informations.append(
            replace(
                dynamic_information,
                branch_limits=replace(
                    dynamic_information.branch_limits,
                    max_mw_flow_limited=limited_max_mw_flow,
                    max_mw_flow_n_1_limited=limited_max_mw_flow_n_1,
                ),
            )
        )

    return tuple(updated_dynamic_informations)


def verify_static_information(
    static_informations: Iterable[StaticInformation],
    max_num_disconnections: int,
) -> None:
    """Verify the static information.

    This function will be called after loading the static information. It should be used to verify
    that the static information is correct and can be used for the optimization run.

    Parameters
    ----------
    static_informations : Iterable[StaticInformation]
        The static information to verify
    max_num_disconnections : int
        The maximum number of disconnections that can be made

    Raises
    ------
    AssertionError
        If the static information is not correct

    Returns
    -------
    None
    """
    first_static_information = next(iter(static_informations))

    assert all(
        [
            jnp.array_equal(
                static_information.solver_config.branches_per_sub.val,
                first_static_information.solver_config.branches_per_sub.val,
            )
            for static_information in static_informations
        ]
    )
    first_set = first_static_information.dynamic_information.action_set
    if first_set is not None:
        assert all(
            static_information.dynamic_information.action_set is not None for static_information in static_informations
        )
        assert all(
            [(static_information.dynamic_information.action_set == first_set) for static_information in static_informations]
        ), "All static informations must have the same branch actions"
        assert all(
            [
                jnp.array_equal(
                    static_information.dynamic_information.action_set.n_actions_per_sub,
                    first_set.n_actions_per_sub,
                )
                for static_information in static_informations
            ]
        ), "All static informations must have the same number of branch actions"

    assert first_static_information.dynamic_information.disconnectable_branches.shape[0] >= max_num_disconnections, (
        "Not enough disconnectable branches for the maximum number of disconnections, "
        + f"got {first_static_information.dynamic_information.disconnectable_branches.shape[0]} and {max_num_disconnections}"
    )
    if first_static_information.dynamic_information.disconnectable_branches.shape[0] > 0:
        assert all(
            [
                jnp.array_equal(
                    first_static_information.dynamic_information.disconnectable_branches,
                    static_information.dynamic_information.disconnectable_branches,
                )
                for static_information in static_informations
            ]
        ), "All static informations must have the same disconnectable branches"


def update_static_information(
    static_informations: tuple[StaticInformation, ...],
    batch_size: int,
) -> tuple[StaticInformation, ...]:
    """Perform any necessary preprocessing on the static information.

    This harmonizes the static informations and makes sure some information that is optional in the solver is always there.

    Parameters
    ----------
    static_informations : list[StaticInformation]
        The list of static informations to preprocess
    batch_size : int
        The batch size to use, will replace the batch size in the solver config

    Returns
    -------
    list[StaticInformation]
        The updated static informations
    """
    dynamic_informations = [static_information.dynamic_information for static_information in static_informations]
    # Furthermore, we want to make sure that we always have limited max mw flows to be able to
    # compute those metrics
    dynamic_informations = [
        replace(
            dynamic_information,
            branch_limits=replace(
                dynamic_information.branch_limits,
                max_mw_flow_limited=dynamic_information.branch_limits.max_mw_flow
                if dynamic_information.branch_limits.max_mw_flow_limited is None
                else dynamic_information.branch_limits.max_mw_flow_limited,
                n0_n1_max_diff=jnp.zeros_like(dynamic_information.branch_limits.max_mw_flow)
                if dynamic_information.branch_limits.n0_n1_max_diff is None
                else dynamic_information.branch_limits.n0_n1_max_diff,
            ),
        )
        for dynamic_information in dynamic_informations
    ]

    # Make sure all the solver configs have the correct batch size
    solver_configs = [static_information.solver_config for static_information in static_informations]
    solver_configs = [
        replace(solver_config, batch_size_bsdf=batch_size, batch_size_injection=batch_size)
        for solver_config in solver_configs
    ]

    static_informations = [
        replace(
            static_information,
            solver_config=solver_config,
            dynamic_information=dynamic_information,
        )
        for static_information, solver_config, dynamic_information in zip(
            static_informations, solver_configs, dynamic_informations, strict=True
        )
    ]

    return static_informations


# ruff: noqa: PLR0913
def initialize_genetic_algorithm(
    batch_size: int,
    max_num_splits: int,
    max_num_disconnections: int,
    static_informations: tuple[StaticInformation, ...],
    target_metrics: tuple[tuple[MetricType, float], ...],
    substation_split_prob: float,
    substation_unsplit_prob: float,
    action_set: ActionSet,
    n_subs_mutated_lambda: float,
    disconnect_prob: float,
    reconnect_prob: float,
    proportion_crossover: float,
    crossover_mutation_ratio: float,
    random_seed: int,
    observed_metrics: tuple[MetricType, ...],
    me_descriptors: tuple[DescriptorDef, ...],
    distributed: bool,
    devices: Optional[list[jax.Device]] = None,
    cell_depth: int = 1,
    mutation_repetition: int = 1,
    n_worst_contingencies: int = 10,
) -> tuple[DiscreteMapElites, JaxOptimizerData]:
    """Initialize the mapelites algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size to use
    max_num_splits : int
        The maximum number of substations that can be split
    max_num_disconnections : int
        The maximum number of disconnections that can be made
    static_informations : list[StaticInformation]
        The static information to use for the optimization run
    target_metrics : tuple[tuple[MetricType, float], ...]
        The target metrics to use for the optimization run
    substation_split_prob : float
        The probability to split a substation
    substation_unsplit_prob : float
        The probability to reset a split substation to the unsplit state
    action_set : ActionSet
        The action set to use for mutations
    n_subs_mutated_lambda : float
        The lambda parameter for the Poisson distribution to determine the number of substations to mutate
    disconnect_prob : float
        The probability to disconnect a new branch
    reconnect_prob : float
        The probability to reconnect a disconnected branch, will overwrite a possible disconnect
    proportion_crossover : float
        The proportion of crossover to mutation
    crossover_mutation_ratio : float
        The ratio of crossover to mutation
    random_seed: int
        The random seed to use for reproducibility
    observed_metrics: tuple[MetricType, ...]
        The observed metrics, i.e. which metrics are to be computed for logging purposes.
    me_descriptors: tuple[Descriptor, ...]
        The descriptors to use for map elites
    distributed: bool
        Whether to run the optimization on multiple devices
    devices: Optional[list[jax.Device]]
        The devices to run the optimization on, if distributed
    cell_depth: int
        The cell depth to use if applicable
    mutation_repetition: int
        More chance to get unique mutations by mutating mutation_repetition copies of the repertoire
    n_worst_contingencies: int
        The number of worst contingencies to consider in the scoring function for calculating
        top_k_overloads_n_1.

    Returns
    -------
    DiscreteMapElites
        The genetic algorithm object including scoring, mutate and crossover functions
    JaxOptimizerData
        The initialized jax dataclass
    """
    n_devices = len(jax.devices()) if distributed else 1

    dynamic_informations = tuple([static_information.dynamic_information for static_information in static_informations])
    solver_configs = tuple(
        [replace(static_information.solver_config, batch_size_bsdf=batch_size) for static_information in static_informations]
    )

    initial_topologies = empty_repertoire(
        batch_size=batch_size * n_devices,
        max_num_splits=max_num_splits,
        max_num_disconnections=max_num_disconnections,
        num_psts=dynamic_informations[0].n_controllable_pst,
    )

    scoring_function_partial = partial(
        scoring_function,
        solver_configs=solver_configs,
        target_metrics=target_metrics,
        observed_metrics=observed_metrics,
        descriptor_metrics=tuple([desc.metric for desc in me_descriptors]),
        n_worst_contingencies=n_worst_contingencies,
    )

    mutate_partial = partial(
        mutate,
        substation_split_prob=substation_split_prob,
        substation_unsplit_prob=substation_unsplit_prob,
        action_set=action_set,
        n_disconnectable_branches=len(dynamic_informations[0].disconnectable_branches),
        n_subs_mutated_lambda=n_subs_mutated_lambda,
        disconnect_prob=disconnect_prob,
        reconnect_prob=reconnect_prob,
        mutation_repetition=mutation_repetition,
    )
    crossover_partial = partial(crossover, action_set=action_set, prob_take_a=proportion_crossover)

    emitter = TrackingMixingEmitter(
        mutate_partial,
        crossover_partial,
        crossover_mutation_ratio,
        batch_size,
    )
    algo = DiscreteMapElites(
        scoring_function=scoring_function_partial,
        emitter=emitter,
        metrics_function=default_ga_metrics,
        distributed=distributed,
        n_cells_per_dim=tuple([desc.num_cells for desc in me_descriptors]),
        cell_depth=cell_depth,
    )

    random_key = jax.random.PRNGKey(random_seed)
    latest_iteration = jnp.array(1, dtype=int)

    init_fn = algo.init
    # If we are running on multiple devices, we need to replicate the data so it lives on every
    # device. The only exception is the random key, where we want a different one on every device
    if distributed:
        initial_topologies = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    n_devices,
                    batch_size,
                )
                + x.shape[1:],
            ),
            initial_topologies,
        )
        random_key = jax.random.split(random_key, n_devices)
        dynamic_informations = jax.tree_util.tree_map(
            lambda x: jax.device_put_replicated(x, devices),
            dynamic_informations,
        )
        latest_iteration = jax.device_put_replicated(latest_iteration, devices)

        init_fn = jax.pmap(
            init_fn,
            axis_name="p",
            in_axes=(
                jax.tree_util.tree_map(lambda _x: 0, initial_topologies),
                0,
                jax.tree_util.tree_map(
                    lambda _x: 0,
                    dynamic_informations,
                ),
            ),
        )

    repertoire, emitter_state, random_key = init_fn(initial_topologies, random_key, dynamic_informations)

    jax_data = JaxOptimizerData(
        repertoire=repertoire,
        emitter_state=emitter_state,
        dynamic_informations=dynamic_informations,
        random_key=random_key,
        latest_iteration=latest_iteration,
    )
    return algo, jax_data


def flatten_fitnesses_if_distributed(
    fitnesses: Float[Array, " ... individuals metrics"],
) -> Float[Array, " individuals metrics"]:
    """Flatten the fitnesses if distributed.

    Parameters
    ----------
    fitnesses : Float[Array, " ... individuals metrics"]
        The fitnesses to flatten

    Returns
    -------
    Float[Array, " individuals metrics"]
        The flattened fitnesses
    """
    if len(fitnesses.shape) == 3:
        fitnesses = jnp.reshape(
            fitnesses,
            (
                fitnesses.shape[0] * fitnesses.shape[1],
                fitnesses.shape[2],
            ),
        )
    assert len(fitnesses.shape) == 2
    return fitnesses


def get_repertoire_metrics(
    repertoire: DiscreteMapElitesRepertoire, observed_metrics: tuple[MetricType, ...]
) -> tuple[float, dict[MetricType, float], Float[Array, " ... n_cells_per_dim"]]:
    """Get the metrics of the best individual in the Mapelites repertoire.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        The repertoire

    observed_metrics : tuple[MetricType, ...]
        The metrics to observe (max_flow_n_0, median_flow_n_0 ...)

    Returns
    -------
    float
        The fitness
    dict[MetricType, float]
        The metrics as defined in METRICS
    """
    distributed = len(repertoire.fitnesses.shape) > 1
    repertoire = jax.tree_util.tree_map(lambda x: x[0], repertoire) if distributed else repertoire

    fitnesses = repertoire.fitnesses
    # Get best individual and its metrics
    best_idx = jnp.argsort(fitnesses, descending=True)
    metrics = jax.tree_util.tree_map(lambda x: x[best_idx], repertoire.extra_scores)
    # only keep metrics in observed_metrics
    metrics = {key: metrics[key] for key in observed_metrics}
    fitnesses = fitnesses[best_idx]

    best_individual_fitness = fitnesses[0].item()
    # best_individual_metrics corresponds to the first element of each observed metric
    best_individual_metrics = {key: value[0].item() for key, value in metrics.items()}

    return best_individual_fitness, best_individual_metrics  # , descriptors[0]


def algo_setup(
    ga_args: BatchedMEParameters,
    lf_args: LoadflowSolverParameters,
    double_limits: Optional[tuple[float, float]],
    static_information_files: list[str],
    processed_gridfile_fs: AbstractFileSystem,
) -> tuple[
    DiscreteMapElites,
    JaxOptimizerData,
    tuple[SolverConfig, ...],
    float,
    dict,
    list[StaticInformationStats],
]:
    """Set up the genetic algorithm run.

    Parameters
    ----------
    ga_args : GeneticAlgorithParameters
        The genetic algorithm parameters
    lf_args : LoadflowSolverParameters
        The loadflow solver parameters
    double_limits: Optional[tuple[float, float]]
        The lower and upper limit for the relative max mw flow if double limits are used
    static_information_files : list[str]
        A list of files with static information to load
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.

    Returns
    -------
    DiscreteMapElites
        The initialized genetic algorithm object, can be used to update the optimization run
    JaxOptimizerData
        The jax dataclass of all GPU data including dynamic information and the GA data
    tuple[SolverConfig, ...]
        The solver configurations for every timestep (the dynamic information is part of the jax
        dataclass)
    float
        The initial fitness, for logging purposes
    dict
        The initial metrics, for logging purposes
    list[StaticInformationDescription]
        Some statistics on the static information dataclasses that were loaded
    """
    n_devices = len(jax.devices()) if lf_args.distributed else 1

    static_informations = tuple(
        [load_static_information_fs(filesystem=processed_gridfile_fs, filename=str(f)) for f in static_information_files]
    )

    logger.info(f"Running {n_devices} GPUs with config {ga_args}, {lf_args}")

    verify_static_information(static_informations, lf_args.max_num_disconnections)

    static_informations = update_static_information(static_informations, lf_args.batch_size)

    if double_limits is not None:
        logger.info(f"Updating double limits to {double_limits}")
        dynamic_infos = update_max_mw_flows_according_to_double_limits(
            dynamic_informations=[s.dynamic_information for s in static_informations],
            solver_configs=[s.solver_config for s in static_informations],
            lower_limit=double_limits[0],
            upper_limit=double_limits[1],
        )
        static_informations = [
            replace(static_information, dynamic_information=dynamic_info)
            for static_information, dynamic_info in zip(static_informations, dynamic_infos, strict=True)
        ]

    algo, jax_data = initialize_genetic_algorithm(
        batch_size=lf_args.batch_size,
        max_num_splits=lf_args.max_num_splits,
        max_num_disconnections=lf_args.max_num_disconnections,
        static_informations=static_informations,
        target_metrics=ga_args.target_metrics,
        substation_split_prob=ga_args.substation_split_prob,
        substation_unsplit_prob=ga_args.substation_unsplit_prob,
        action_set=static_informations[0].dynamic_information.action_set,
        n_subs_mutated_lambda=ga_args.n_subs_mutated_lambda,
        disconnect_prob=ga_args.disconnect_prob,
        reconnect_prob=ga_args.reconnect_prob,
        proportion_crossover=ga_args.proportion_crossover,
        crossover_mutation_ratio=ga_args.crossover_mutation_ratio,
        random_seed=ga_args.random_seed,
        observed_metrics=ga_args.observed_metrics,
        distributed=lf_args.distributed,
        devices=jax.devices() if lf_args.distributed else None,
        me_descriptors=ga_args.me_descriptors,
        mutation_repetition=ga_args.mutation_repetition,
        cell_depth=ga_args.cell_depth,
        n_worst_contingencies=ga_args.n_worst_contingencies,
    )

    initial_fitness, initial_metrics = get_repertoire_metrics(
        jax.tree_util.tree_map(lambda x: x[0], jax_data.repertoire) if lf_args.distributed else jax_data.repertoire,
        ga_args.observed_metrics,
    )

    static_information_descriptions = [
        extract_static_information_stats(
            static_information=static_information,
            overload_n0=initial_metrics.get("overload_energy_n_0", 0.0),
            overload_n1=initial_metrics.get("overload_energy_n_1", 0.0),
            time="",
        )
        for static_information in static_informations
    ]

    for desc in static_information_descriptions:
        logger.info(f"Starting optimization with static information: {desc}")

    return (
        algo,
        jax_data,
        [static_information.solver_config for static_information in static_informations],
        initial_fitness,
        initial_metrics,
        static_information_descriptions,
    )
