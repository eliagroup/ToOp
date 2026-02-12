# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Callable functions for the optimizer worker."""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
from fsspec import AbstractFileSystem
from jax import lax
from jax_dataclasses import replace
from jaxtyping import Array, Int
from toop_engine_dc_solver.jax.types import SolverConfig
from toop_engine_dc_solver.preprocess.convert_to_jax import StaticInformationStats
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.dc.ga_helpers import EmitterState
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (
    JaxOptimizerData,
    algo_setup,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    summarize_repo,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
)
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    Strategy,
    Topology,
    TopologyPushResult,
)


@dataclass
class OptimizerData:
    """A wrapper dataclass for all the data stored by the optimizer.

    Because this dataclass holds irrelevant information for the GPU optimization, it is split into two dataclasses.
    The parent (this one) is used to store all the information and the child (JaxOptimizerData) is used to
    store the information that is needed on the GPU.
    """

    start_params: DCOptimizerParameters
    """The initial args for the optimization run"""

    optimization_id: str
    """The id of the optimization run"""

    solver_configs: tuple[SolverConfig, ...]
    """The solver config for every timestep"""

    algo: DiscreteMapElites
    """The genetic algorithm object"""

    initial_fitness: float
    """The initial fitness value"""

    initial_metrics: dict[MetricType, float]
    """The initial metrics"""

    jax_data: JaxOptimizerData
    """Everything that needs to live on GPU. This dataclass is updated per-iteration while
    OptimizerData is only updated per-epoch, hence there are points in the command flow where this
    variable is out of sync. At the end of an epoch, this dataclass is updated to match the
    state in the optimization."""

    start_time: float
    """The time the optimization run started"""


def initialize_optimization(
    params: DCOptimizerParameters,
    optimization_id: str,
    static_information_files: tuple[str, ...],
    processed_gridfile_fs: AbstractFileSystem,
) -> tuple[OptimizerData, list[StaticInformationStats], Strategy]:
    """Initialize the optimization run.

    This function will be called at the start of the optimization run. It should be used to load
    the static information files and set up the optimization run.

    Parameters
    ----------
    params : DCOptimizerParameters
        The parameters for the optimization run
    optimization_id : str
        The id of the optimization run, used to annotate results and heartbeats
    static_information_files : tuple[str, ...]
        The paths to the static information files to load
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.


    Returns
    -------
    OptimizerData
        The data to store for the optimization run
    list[StaticInformationStats]
        The static information descriptions, will be sent via the heartbeats channel
    Strategy
        The initial strategy (unsplit) for the grid, including the initial fitness and metrics

    Raises
    ------
    Exception
        If an error occurs during the initialization. It will be caught by the worker and sent back to
        the results topic
    """
    (
        algo,
        jax_data,
        solver_configs,
        initial_fitness,
        initial_metrics,
        static_information_descriptions,
    ) = algo_setup(
        ga_args=params.ga_config,
        lf_args=params.loadflow_solver_config,
        double_limits=(
            params.double_limits.lower,
            params.double_limits.upper,
        )
        if params.double_limits is not None
        else None,
        static_information_files=static_information_files,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    metrics = convert_metrics(initial_fitness, initial_metrics)

    # For now we send None as initial topology PST setpoints, as the AC solver can
    # not distinguish a topology with taps set to default from a topology without taps.
    initial_topology = Strategy(
        timesteps=[
            Topology(
                actions=[],
                disconnections=[],
                # pst_setpoints=di.nodal_injection_information.starting_tap_idx.tolist()
                # if di.nodal_injection_information is not None
                # else None,
                pst_setpoints=None,
                metrics=metrics,
            )
            for _di in jax_data.dynamic_informations
        ]
    )

    optimization_data = OptimizerData(
        start_params=params,
        optimization_id=optimization_id,
        solver_configs=solver_configs,
        algo=algo,
        jax_data=jax_data,
        initial_fitness=initial_fitness,
        initial_metrics=initial_metrics,
        start_time=time.time(),
    )

    return optimization_data, static_information_descriptions, initial_topology


def convert_metrics(fitness: float, metrics_dict: dict[MetricType, float]) -> Metrics:
    """Convert a metrics dictionary to a Metrics dataclass.

    Parameters
    ----------
    fitness : float
        The fitness value

    metrics_dict : dict[MetricType, float]
        The metrics dictionary

    Returns
    -------
    Metrics
        The metrics dataclass
    """
    case_indices = metrics_dict.pop("case_indices", None)
    metrics = Metrics(
        fitness=fitness,
        extra_scores=metrics_dict,
        worst_k_contingency_cases=case_indices,
    )

    return metrics


@partial(
    jax.jit,
    static_argnames=("update_fn"),
    donate_argnames=("jax_data",),
)
def run_single_iteration(
    _i: Int[Array, ""],
    jax_data: JaxOptimizerData,
    update_fn: Callable[
        [DiscreteMapElitesRepertoire, EmitterState, jax.random.PRNGKey, Any],
        tuple[DiscreteMapElitesRepertoire, EmitterState, Any, jax.random.PRNGKey],
    ],
) -> JaxOptimizerData:
    """Run a single iteration of the optimization.

    This involves updating the genetic algorithm and calling the metrics callback

    Parameters
    ----------
    _i : Int[Array, ""]
        The iteration number, will be ignored. Its only purpose is to make the function signature
        compatible with lax.fori_loop
    jax_data : JaxOptimizerData
        The data stored for the optimization run from the last epoch or from initialize_optimization
    update_fn : Callable[(
                        [GARepertoire, EmitterState, jax.random.PRNGKey, Any],
                        tuple[GARepertoire, EmitterState, Any, jax.random.PRNGKey]
                    )]
        The update function of the genetic algorithm

    Returns
    -------
    JaxOptimizerData
        The updated data for the optimization run
    """
    repertoire, emitter_state, _metrics, random_key = update_fn(
        jax_data.repertoire,
        jax_data.emitter_state,
        jax_data.random_key,
        jax_data.dynamic_informations,
    )

    jax_data = replace(
        jax_data,
        repertoire=repertoire,
        emitter_state=emitter_state,
        random_key=random_key,
        latest_iteration=jax_data.latest_iteration + 1,
    )

    return jax_data


@partial(
    jax.jit,
    static_argnames=("iterations_per_epoch", "update_fn"),
    # donate_argnames=("jax_data",),
)
def run_single_device_epoch(
    jax_data: JaxOptimizerData,
    iterations_per_epoch: int,
    update_fn: Callable[
        [DiscreteMapElitesRepertoire, EmitterState, jax.random.PRNGKey, Any],
        tuple[DiscreteMapElitesRepertoire, EmitterState, Any, jax.random.PRNGKey],
    ],
) -> JaxOptimizerData:
    """Run one epoch of the optimization on a single device.

    Basically this is just a fori loop over the iterations_per_epoch, calling run_single_iteration
    This can be used to pass to pmap

    Parameters
    ----------
    jax_data : JaxOptimizerData
        The data stored for the optimization run from the last epoch or from initialize_optimization
    iterations_per_epoch : int
        The number of iterations to run in this epoch
    update_fn : Callable[(
                        [GARepertoire, EmitterState, jax.random.PRNGKey, Any],
                        tuple[GARepertoire, EmitterState, Any, jax.random.PRNGKey]
                    )]
        The update function of the genetic algorithm

    Returns
    -------
    JaxOptimizerData
        The updated data for the optimization run
    """
    return lax.fori_loop(
        0,
        iterations_per_epoch,
        partial(run_single_iteration, update_fn=update_fn),
        jax_data,
    )
    # To debug the fori_loop, use the line "jax.disable_jit()"


def run_epoch(
    optimizer_data: OptimizerData,
) -> OptimizerData:
    """Run one epoch of the optimization.

    This function will be called repeatedly by the worker. It should include multiple iterations,
    according to the configuration of the optimizer. Furthermore it should call the metrics_callback
    during the epoch, at least at the beginning. This could happen through a jax callback to not
    block the main thread.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The data stored for the optimization run from the last epoch or from initialize_optimization

    Returns
    -------
    OptimizerData
        The updated data for the optimization run

    Raises
    ------
    Exception
        If an error occurs during the epoch. It will be caught by the worker and sent back to the
        results topic
    """
    epoch_fn = run_single_device_epoch
    if optimizer_data.start_params.loadflow_solver_config.distributed:
        epoch_fn = jax.pmap(
            epoch_fn,
            axis_name="p",
            static_broadcasted_argnums=(1, 2),
            donate_argnums=(0,),
        )

    jax_data = epoch_fn(
        optimizer_data.jax_data,
        optimizer_data.start_params.ga_config.iterations_per_epoch,
        optimizer_data.algo.update,
    )

    return replace(
        optimizer_data,
        jax_data=jax_data,
    )


def extract_results(optimizer_data: OptimizerData) -> TopologyPushResult:
    """Pull the results from the optimizer.

    This should give the current best topologies along with metrics for each topology, where the
    number of saved topologies is a configuration parameter in the StartOptimizationCommand.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The data stored for the optimization run

    Returns
    -------
    TopologyPushResult
        The current best topologies extracted and in topo-vect form.
    """
    # Assuming that contingency_ids stay the same for all timesteps
    contingency_ids = optimizer_data.solver_configs[0].contingency_ids

    # Get grid_model_low_tap if nodal injection information is available
    nodal_inj_info = optimizer_data.jax_data.dynamic_informations[0].nodal_injection_information
    grid_model_low_tap = nodal_inj_info.grid_model_low_tap if nodal_inj_info is not None else None

    topologies = summarize_repo(
        optimizer_data.jax_data.repertoire,
        initial_fitness=optimizer_data.initial_fitness,
        contingency_ids=contingency_ids,
        grid_model_low_tap=grid_model_low_tap,
    )

    # Convert it to strategies
    # TODO change once we support multiple timesteps
    formatted_topologies = [Strategy(timesteps=[topo]) for topo in topologies]

    return TopologyPushResult(strategies=formatted_topologies, epoch=optimizer_data.jax_data.latest_iteration.max().item())
