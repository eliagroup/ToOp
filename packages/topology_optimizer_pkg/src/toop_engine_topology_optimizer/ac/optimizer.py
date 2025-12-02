"""Implements initialize and run_epoch functions for the AC optimizer"""

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import logbook
import numpy as np
from fsspec import AbstractFileSystem
from numpy.random import Generator as Rng
from sqlmodel import Session
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import (
    get_worst_k_contingencies_ac,
)
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars, save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set_fs
from toop_engine_topology_optimizer.ac.evolution_functions import evolution
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.scoring_functions import compute_metrics, evaluate_acceptance, scoring_function
from toop_engine_topology_optimizer.ac.storage import (
    ACOptimTopology,
)
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    ResultUnion,
    Strategy,
    Topology,
    TopologyPushResult,
)
from toop_engine_topology_optimizer.interfaces.models.base_storage import convert_db_topo_to_message_topo, hash_topologies

logger = logbook.Logger(__name__)


class AcNotConvergedError(Exception):
    """An exception that is raised when the AC optimization did not converge in the base grid"""

    pass


@dataclass
class OptimizerData:
    """The epoch-to-epoch storage for the AC optimizer"""

    params: ACOptimizerParameters
    """The parameters this optimizer was initialized with"""

    session: Session
    """A in-memory sqlite session for storing the repertoire"""

    evolution_fn: Callable[[], list[ACOptimTopology]]
    """The curried evolution function"""

    scoring_fn: Callable[[list[ACOptimTopology]], tuple[LoadflowResultsPolars, list[Metrics]]]
    """The curried scoring function"""

    store_loadflow_fn: Callable[[LoadflowResultsPolars], StoredLoadflowReference]
    """The function to store loadflow results"""

    load_loadflow_fn: Callable[[StoredLoadflowReference], LoadflowResultsPolars]
    """The function to load loadflow results"""

    acceptance_fn: Callable[
        [
            LoadflowResultsPolars,
            list[Metrics],
        ],
        bool,
    ]
    """The acceptance function to decide whether a topology is accepted or not. Takes the
    loadflow results and the metrics of the split topology and returns True/False."""

    rng: Rng
    """The random number generator for the optimizer"""

    action_sets: list[ActionSet]
    """The action sets for the grid files (one for each timestep)"""

    framework: Framework
    """The framework of the grid files"""

    runners: list[AbstractLoadflowRunner]
    """The initialized loadflow runners, one for each grid file"""


def update_initial_metrics_with_worst_k_contingencies(
    initial_loadflow: LoadflowResultsPolars,
    initial_metrics: list[Metrics],
    worst_k: int,
) -> None:
    """Update the initial metrics with the worst k contingencies.

    This function computes the worst k contingencies for each timestep in the initial loadflow results
    and updates the initial metrics with the case ids and the top k overloads.

    Parameters
    ----------
    initial_loadflow : LoadflowResultsPolars
        The initial loadflow results containing the branch results.
    initial_metrics : list[Metrics]
        The initial metrics for each timestep.
    worst_k : int
        The number of worst contingencies to consider for the initial metrics.
    """
    case_ids, top_k_overloads_n_1 = get_worst_k_contingencies_ac(
        initial_loadflow.branch_results,
        k=worst_k,
    )

    # case_ids is an empty list if the loadflow didn't converge -> the initial_loadflow.branch_results is full of NaNs
    if len(case_ids) == 0:
        logger.warning("No worst case ids found as the loadflow didn't converge")
        top_k_overloads_n_1 = [0] * len(initial_metrics)
        case_ids = [[]] * len(initial_metrics)

    # Update extra_scores of initial_metrics with case_ids and top_k_overloads_n_1
    for timestep, metric in enumerate(initial_metrics):
        metric.extra_scores.update(
            {
                "top_k_overloads_n_1": top_k_overloads_n_1[timestep],
            }
        )
        metric.worst_k_contingency_cases = case_ids[timestep]


def make_runner(
    action_set: ActionSet,
    nminus1_definition: Nminus1Definition,
    grid_file: GridFile,
    n_processes: int,
    batch_size: Optional[int],
    processed_gridfile_fs: AbstractFileSystem,
) -> AbstractLoadflowRunner:
    """Initialize a runner for a gridfile, action set and n-1 def

    Parameters
    ----------
    action_set : ActionSet
        The action set to use
    nminus1_definition : Nminus1Definition
        The N-1 definition to use
    grid_file : GridFile
        The grid file to use
    n_processes : int
        The number of processes to use, from the ACGAParameters
    batch_size : Optional[int]
        The batch size to use, if any, from the ACGAParameters
    processed_gridfile_fs: AbstractFileSystem
        The target filesystem for the preprocessing worker. This contains all processed grid files.
        During the import job,  a new folder import_results.data_folder was created
        which will be completed with the preprocess call to this function.
        Internally, only the data folder is passed around as a dirfs.
        Note that the unprocessed_gridfile_fs is not needed here anymore, as all preprocessing steps that need the
        unprocessed gridfiles were already done.

    Returns
    -------
    AbstractLoadflowRunner
        The initialized loadflow runner, either Pandapower or Powsybl
    """
    if grid_file.framework == Framework.PANDAPOWER:
        runner = PandapowerRunner(n_processes=n_processes, batch_size=batch_size)
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    elif grid_file.framework == Framework.PYPOWSYBL:
        runner = PowsyblRunner(n_processes=n_processes, batch_size=batch_size)
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    else:
        raise ValueError(f"Unknown framework {grid_file.framework}")
    runner.load_base_grid_fs(filesystem=processed_gridfile_fs, grid_path=grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    return runner


def initialize_optimization(
    params: ACOptimizerParameters,
    session: Session,
    optimization_id: str,
    grid_files: list[GridFile],
    loadflow_result_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
) -> tuple[OptimizerData, Strategy]:
    """Initialize an optimization run for the AC optimizer

    Parameters
    ----------
    params : ACOptimizerParameters
        The parameters for the AC optimizer
    session : Session
        The database session to use for storing topologies
    optimization_id : str
        The ID of the optimization run
    grid_files : list[GridFile]
        The grid files to optimize on, must contain at least one file
    loadflow_result_fs: AbstractFileSystem
        A filesystem where the loadflow results are stored. Loadflows will be stored here using the uuid generation process
        and passed as a StoredLoadflowReference which contains the subfolder in this filesystem.
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
        The initial optimizer data
    Strategy
        The initial strategy
    """
    if not len(grid_files):
        raise ValueError("At least one grid file must be provided")

    ga_config = params.ga_config

    # Load the network datas
    action_sets = [
        load_action_set_fs(filesystem=processed_gridfile_fs, file_path=grid_file.action_set_file) for grid_file in grid_files
    ]
    nminus1_definitions = [
        load_pydantic_model_fs(
            filesystem=processed_gridfile_fs, file_path=grid_file.nminus1_definition_file, model_class=Nminus1Definition
        )
        for grid_file in grid_files
    ]
    base_case_ids = [(n1def.base_case.id if n1def.base_case is not None else None) for n1def in nminus1_definitions]

    num_psts = [len(action_set.pst_ranges) for action_set in action_sets]

    # Prepare the loadflow runners
    runners = [
        make_runner(
            action_set,
            nminus1_definition,
            grid_file,
            n_processes=ga_config.runner_processes,
            batch_size=ga_config.runner_batchsize,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        for action_set, nminus1_definition, grid_file in zip(action_sets, nminus1_definitions, grid_files, strict=True)
    ]

    # Prepare the evolution function
    rng = np.random.default_rng(ga_config.seed)
    evolution_fn = partial(
        evolution,
        rng=rng,
        session=session,
        optimization_id=optimization_id,
        close_coupler_prob=ga_config.close_coupler_prob,
        reconnect_prob=ga_config.reconnect_prob,
        pull_prob=ga_config.pull_prob,
        max_retries=10,
        n_minus1_definitions=nminus1_definitions,
        filter_strategy=ga_config.filter_strategy,
    )
    scoring_fn = partial(
        scoring_function,
        runners=runners,
        base_case_ids=base_case_ids,
        n_timestep_processes=ga_config.timestep_processes,
        early_stop_validation=ga_config.early_stop_validation,
        early_stop_non_converging_threshold=ga_config.early_stopping_non_convergence_percentage_threshold,
    )

    # Prepare the initial strategy
    initial_strategy = [
        ACOptimTopology(
            actions=[],
            disconnections=[],
            pst_setpoints=[0] * n_pst,
            timestep=i,
            fitness=0,
            unsplit=True,
            strategy_hash=b"willbeupdated",
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.AC,
        )
        for i, n_pst in enumerate(num_psts)
    ]
    initial_hash = hash_topologies(initial_strategy)
    for topo in initial_strategy:
        topo.strategy_hash = initial_hash

    def store_loadflow(loadflow: LoadflowResultsPolars) -> StoredLoadflowReference:
        return save_loadflow_results_polars(
            loadflow_result_fs, f"{optimization_id}-{loadflow.job_id}-{datetime.now()}", loadflow
        )

    def loadflow_ref(loadflow: StoredLoadflowReference) -> LoadflowResultsPolars:
        return load_loadflow_results_polars(loadflow_result_fs, reference=loadflow)

    # This requires a full loadflow computation if the loadflow results are not passed in
    initial_loadflow_reference = params.initial_loadflow
    if initial_loadflow_reference is None:
        initial_loadflow, initial_metrics = scoring_function(
            strategy=initial_strategy,
            runners=runners,
            base_case_ids=base_case_ids,
            n_timestep_processes=ga_config.timestep_processes,
        )
        initial_loadflow_reference = store_loadflow(initial_loadflow)
    else:
        # If the initial loadflow is passed in, we load it from the database
        initial_loadflow = loadflow_ref(initial_loadflow_reference)
        # Compute the metrics for the initial loadflow
        initial_metrics = compute_metrics(
            strategy=initial_strategy,
            lfs=initial_loadflow,
            base_case_ids=base_case_ids,
            additional_info=[None] * len(initial_strategy),
        )

    # Update the initial metrics with the worst k contingencies
    update_initial_metrics_with_worst_k_contingencies(
        initial_loadflow, initial_metrics, params.ga_config.n_worst_contingencies
    )

    for timestep_metrics, n1_def in zip(initial_metrics, nminus1_definitions, strict=True):
        if timestep_metrics.extra_scores["non_converging_loadflows"] > len(n1_def.contingencies) / 2:
            raise AcNotConvergedError(
                "Too many non-converging loadflows in initial loadflow: "
                f"{timestep_metrics.extra_scores['non_converging_loadflows']} > {len(n1_def.contingencies) / 2}"
            )

    # Store the initial strategy in the database
    for topo, metric in zip(initial_strategy, initial_metrics, strict=True):
        topo.fitness = metric.fitness
        topo.metrics = metric.extra_scores
        topo.worst_k_contingency_cases = metric.worst_k_contingency_cases
        topo.set_loadflow_reference(initial_loadflow_reference)
        session.add(topo)
    session.commit()

    # As we have the initial loadflows, we can now define an acceptance function
    acceptance_fn = partial(
        evaluate_acceptance,
        metrics_unsplit=initial_metrics,
        loadflow_results_unsplit=initial_loadflow,
        reject_convergence_threshold=ga_config.reject_convergence_threshold,
        reject_overload_threshold=ga_config.reject_overload_threshold,
        reject_critical_branch_threshold=ga_config.reject_critical_branch_threshold,
    )

    # Convert the initial strategy to a message strategy
    initial_strategy_message = convert_db_topo_to_message_topo(initial_strategy)
    assert len(initial_strategy_message) == 1

    logger.info(
        f"Initialization completed, metrics: {initial_metrics[0].extra_scores}, fitness: {initial_metrics[0].fitness}, "
        f"worst_k_contingency_cases: {initial_metrics[0].worst_k_contingency_cases}"
    )

    return (
        OptimizerData(
            params=params,
            session=session,
            evolution_fn=evolution_fn,
            scoring_fn=scoring_fn,
            store_loadflow_fn=store_loadflow,
            load_loadflow_fn=loadflow_ref,
            acceptance_fn=acceptance_fn,
            rng=rng,
            framework=grid_files[0].framework,
            runners=runners,
            action_sets=action_sets,
        ),
        initial_strategy_message[0],
    )


def run_epoch(
    optimizer_data: OptimizerData,
    results_consumer: LongRunningKafkaConsumer,
    send_result_fn: Callable[[ResultUnion], None],
    epoch: int,
) -> None:
    """Run a single epoch of the AC optimizer

    This shall send the investigated topology to the result topic upon completion.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data, will be updated in-place
    results_consumer : LongRunningKafkaConsumer
        The consumer where to listen for DC results
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results
    epoch : int
        The current epoch number, used for logging and heartbeat purposes. Also, on epoch 1 the wait time for
        the consumer is longer to allow for dc optim startup.
    """
    poll_results_topic(db=optimizer_data.session, consumer=results_consumer, first_poll=epoch == 1)
    new_strategy = optimizer_data.evolution_fn()

    # It is possible that no new strategy was generated
    if not new_strategy:
        return

    loadflow_results, metrics = optimizer_data.scoring_fn(new_strategy)
    acceptance = optimizer_data.acceptance_fn(loadflow_results, metrics)
    loadflow_result_reference = optimizer_data.store_loadflow_fn(loadflow_results)

    # Update the strategy with the new loadflow results
    message_topos = []
    for topology, metric in zip(new_strategy, metrics, strict=True):
        # TODO: FIXME: remove fitness_dc when "Topology" is refactored and accepts different stages like "dc", "dc+" and "ac"
        # topology should store a dict of metrics instead of a single fitness value
        if "fitness_dc" in topology.metrics:
            metric.extra_scores["fitness_dc"] = topology.metrics["fitness_dc"]
        topology.metrics = metric.extra_scores
        topology.fitness = metric.fitness
        topology.acceptance = acceptance
        topology.set_loadflow_reference(loadflow_result_reference)

        optimizer_data.session.add(topology)

        message_topos.append(
            Topology(
                actions=topology.actions,
                pst_setpoints=topology.pst_setpoints,
                disconnections=topology.disconnections,
                loadflow_results=loadflow_result_reference,
                metrics=metric,
            )
        )

    optimizer_data.session.commit()

    # Send the new strategy to the result topic
    if not optimizer_data.params.ga_config.enable_ac_rejection or acceptance:
        send_result_fn(TopologyPushResult(strategies=[Strategy(timesteps=message_topos)], epoch=epoch))
    logger.info(
        f"Epoch {epoch} completed, accept: {acceptance}, metrics: {metrics[0].extra_scores}, fitness: {metrics[0].fitness}"
    )
