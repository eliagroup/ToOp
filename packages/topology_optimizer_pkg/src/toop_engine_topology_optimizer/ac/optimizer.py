# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Implements initialize and run_epoch functions for the AC optimizer"""

from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pypowsybl
import structlog
from beartype.typing import Callable, Optional
from fsspec import AbstractFileSystem
from pydantic import PositiveInt
from sqlmodel import Session, select
from structlog.typing import BindableLogger
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import (
    get_worst_k_contingencies_ac,
)
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars, save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set_fs
from toop_engine_topology_optimizer.ac.evolution_functions import INF_FITNESS, evolution
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.scoring_functions import (
    ACScoringParameters,
    compute_loadflow_and_metrics,
    compute_metrics,
    score_strategy_worst_k_batch,
    score_topology_batch,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import (
    EarlyStoppingStageResult,
    OptimizerData,
    RunnerGroup,
    TopologyScoringResult,
)
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    ResultUnion,
    Strategy,
    Topology,
    TopologyPushResult,
    TopologyRejectionReason,
    TopologyRejectionResult,
    get_topology_rejection_message,
)
from toop_engine_topology_optimizer.interfaces.models.base_storage import convert_db_topo_to_message_topo, hash_topologies

logger = structlog.get_logger(__name__)


class AcNotConvergedError(Exception):
    """An exception that is raised when the AC optimization did not converge in the base grid"""

    pass


def update_initial_metrics_with_worst_k_contingencies(
    initial_loadflow: LoadflowResultsPolars,
    initial_metrics: Metrics,
    worst_k: int,
) -> None:
    """Update the initial metrics with the worst k contingencies.

    This function computes the worst k contingencies for each timestep in the initial loadflow results
    and updates the initial metrics with the case ids and the top k overloads.
    This way, a baseline for the worst k contingencies to compare to is established, i.e. in the initial loadflow results
    these are the reference, disregarding the worst k for the specific strategy.

    Parameters
    ----------
    initial_loadflow : LoadflowResultsPolars
        The initial loadflow results containing the branch results.
    initial_metrics : Metrics
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
        top_k_overloads_n_1 = [0.0]
        case_ids = [[]]

    initial_metrics.extra_scores.update(
        {
            "top_k_overloads_n_1": top_k_overloads_n_1[0],
        }
    )
    initial_metrics.worst_k_contingency_cases = case_ids[0]


def make_runner(
    action_set: ActionSet,
    nminus1_definition: Nminus1Definition,
    grid_file: GridFile,
    n_processes: int,
    batch_size: Optional[int],
    processed_gridfile_fs: AbstractFileSystem,
    lf_params: pypowsybl.loadflow.Parameters | dict | None = None,
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
    lf_params: Optional[pypowsybl.loadflow.Parameters | dict]
        The loadflow parameters to use for the runner, if any. This is passed in the preprocessing results
        and can be used to run the loadflows with the same parameters
        as the initial loadflow in the preprocessing step.
        If None, the runner will use default parameters.

    Returns
    -------
    AbstractLoadflowRunner
        The initialized loadflow runner, either Pandapower or Powsybl
    """
    logger.debug(
        "Initializing loadflow runner "
        f"framework={grid_file.framework}, grid_folder={grid_file.grid_folder}, "
        f"n_processes={n_processes}, batch_size={batch_size}"
    )

    if grid_file.framework == Framework.PANDAPOWER:
        runner = PandapowerRunner(
            n_processes=n_processes, batch_size=batch_size, lf_params=lf_params if isinstance(lf_params, dict) else None
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    elif grid_file.framework == Framework.PYPOWSYBL:
        runner = PowsyblRunner(
            n_processes=n_processes,
            batch_size=batch_size,
            lf_params=lf_params if isinstance(lf_params, pypowsybl.loadflow.Parameters) else None,
        )
        grid_file_path = Path(grid_file.grid_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    else:
        raise ValueError(f"Unknown framework {grid_file.framework}")
    runner.load_base_grid_fs(filesystem=processed_gridfile_fs, grid_path=grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)
    logger.debug(
        f"Runner prepared for {grid_file_path}: n_actions={len(action_set.local_actions)}, "
        f"n_contingencies={len(nminus1_definition.contingencies)}"
    )
    return runner


def initialize_optimization(
    params: ACOptimizerParameters,
    session: Session,
    optimization_id: str,
    grid_file: GridFile,
    loadflow_result_fs: AbstractFileSystem,
    processed_gridfile_fs: AbstractFileSystem,
    optimization_logger: Optional[BindableLogger] = None,
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
    grid_file : GridFile
        The grid file to optimize on
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
    optimization_logger: Optional[BindableLogger]
        The logger to use for logging during the optimization. If None, the default logger from this module will be used.

    Returns
    -------
    OptimizerData
        The initial optimizer data
    Strategy
        The initial strategy
    """
    optimization_logger = optimization_logger or logger

    ga_config = params.ga_config
    optimization_logger.info(f"Initializing AC optimization: framework={grid_file.framework}, seed={ga_config.seed}")
    # Load the network datas
    action_set = load_action_set_fs(
        filesystem=processed_gridfile_fs,
        json_file_path=grid_file.action_set_file,
        diff_file_path=grid_file.action_set_diff_file,
    )
    nminus1_definition = load_pydantic_model_fs(
        filesystem=processed_gridfile_fs, file_path=grid_file.nminus1_definition_file, model_class=Nminus1Definition
    )

    lf_params = load_lf_params_from_fs(filesystem=processed_gridfile_fs, file_path=grid_file.loadflow_parameters_file)

    optimization_logger.debug("Loaded preprocessing inputs")

    base_case_id = getattr(nminus1_definition.base_case, "id", None)

    # Prepare the loadflow runners
    def build_runner_group(n_topo_processes: int, n_contingency_processes: int) -> RunnerGroup:
        return [
            make_runner(
                action_set,
                nminus1_definition,
                grid_file,
                n_processes=n_contingency_processes,
                batch_size=None,
                processed_gridfile_fs=processed_gridfile_fs,
                lf_params=lf_params,
            )
            for _ in range(n_topo_processes)
        ]

    worst_k_runner_group = build_runner_group(ga_config.worst_k_runner_processes, ga_config.worst_k_contingency_processes)
    optimization_logger.debug(f"Prepared {len(worst_k_runner_group)} runner(s) for Early Stopping AC optimization")

    runner_group = build_runner_group(ga_config.runner_processes, ga_config.contingency_processes)
    optimization_logger.debug(f"Prepared {len(runner_group)} runner(s) for AC optimization")

    # Prepare the evolution function
    rng = np.random.default_rng(ga_config.seed)
    evolution_fn = partial(
        evolution,
        rng=rng,
        session=session,
        optimization_id=optimization_id,
        max_retries=10,
        batch_size=ga_config.worst_k_runner_processes,
        filter_strategy=ga_config.filter_strategy,
    )

    # Prepare the initial strategy
    unsplit_topology = ACOptimTopology(
        actions=[],
        disconnections=[],
        pst_setpoints=None,
        timestep=0,
        fitness=0,
        unsplit=True,
        strategy_hash=b"willbeupdated",
        optimization_id=optimization_id,
        optimizer_type=OptimizerType.AC,
    )
    initial_hash = hash_topologies([unsplit_topology])
    unsplit_topology.strategy_hash = initial_hash

    def store_loadflow(loadflow: LoadflowResultsPolars) -> StoredLoadflowReference:
        return save_loadflow_results_polars(
            loadflow_result_fs, f"{optimization_id}-{loadflow.job_id}-{datetime.now()}", loadflow
        )

    def loadflow_ref(loadflow: StoredLoadflowReference) -> LoadflowResultsPolars:
        return load_loadflow_results_polars(loadflow_result_fs, reference=loadflow)

    # This requires a full loadflow computation if the loadflow results are not passed in
    initial_loadflow_reference = params.initial_loadflow
    if initial_loadflow_reference is None:
        optimization_logger.info("No initial loadflow provided, computing initial AC loadflow")
        initial_loadflow, _, initial_metrics = compute_loadflow_and_metrics(
            topology=unsplit_topology,
            runner=runner_group[0],
            base_case_id=base_case_id,
        )
        initial_loadflow_reference = store_loadflow(initial_loadflow)
        optimization_logger.debug(f"Initial AC loadflow computed and stored under reference={initial_loadflow_reference}")
    else:
        optimization_logger.info(f"Using precomputed initial loadflow reference={initial_loadflow_reference}")
        # If the initial loadflow is passed in, we load it from the database
        initial_loadflow = loadflow_ref(initial_loadflow_reference)
        # Compute the metrics for the initial loadflow
        initial_metrics = compute_metrics(
            topology=unsplit_topology, lfs=initial_loadflow, base_case_id=base_case_id, additional_info=None
        )
        optimization_logger.debug("Computed initial metrics from provided loadflow")

    # Update the initial metrics with the worst k contingencies
    update_initial_metrics_with_worst_k_contingencies(
        initial_loadflow, initial_metrics, params.ga_config.n_worst_contingencies
    )

    optimization_logger.debug(
        "Initial convergence summary: "
        f"non_converging={initial_metrics.extra_scores.get('non_converging_loadflows', 0)}, "
        f"allowed={len(nminus1_definition.contingencies) / 2}"
    )
    if initial_metrics.extra_scores.get("non_converging_loadflows", 0) > len(nminus1_definition.contingencies) / 2:
        raise AcNotConvergedError(
            "Too many non-converging loadflows in initial loadflow: "
            f"{initial_metrics.extra_scores.get('non_converging_loadflows', 0)} > "
            f"{len(nminus1_definition.contingencies) / 2}"
        )

    unsplit_topology.fitness = initial_metrics.fitness
    unsplit_topology.metrics = initial_metrics.extra_scores
    unsplit_topology.worst_k_contingency_cases = initial_metrics.worst_k_contingency_cases
    unsplit_topology.set_loadflow_reference(initial_loadflow_reference)
    session.add(unsplit_topology)
    session.commit()
    optimization_logger.debug("Stored initial AC strategy in DB")

    # As we have the initial loadflows, we can now define a scoring+acceptance function
    scoring_params = ACScoringParameters(
        base_case_id=base_case_id,
        early_stop_validation=ga_config.early_stop_validation,
        reject_convergence_threshold=ga_config.reject_convergence_threshold,
        reject_overload_threshold=ga_config.reject_overload_threshold,
        reject_critical_branch_threshold=ga_config.reject_critical_branch_threshold,
    )
    scoring_fn = partial(
        score_topology_batch,
        runner_group=runner_group,
        metrics_unsplit=initial_metrics,
        scoring_params=scoring_params,
    )
    worst_k_scoring_fn = partial(
        score_strategy_worst_k_batch,
        worst_k_runner_groups=worst_k_runner_group,
        metrics_unsplit=initial_metrics,
        loadflow_results_unsplit=initial_loadflow,
        scoring_params=scoring_params,
    )
    # Convert the initial strategy to a message strategy
    initial_strategy_message = convert_db_topo_to_message_topo([unsplit_topology])
    assert len(initial_strategy_message) == 1

    optimization_logger.info(
        f"Initialization completed, metrics: {initial_metrics.extra_scores}, fitness: {initial_metrics.fitness}, "
        f"worst_k_contingency_cases: {initial_metrics.worst_k_contingency_cases}. Waiting for DC results..."
    )

    return (
        OptimizerData(
            params=params,
            session=session,
            evolution_fn=evolution_fn,
            scoring_fn=scoring_fn,
            worst_k_scoring_fn=worst_k_scoring_fn,
            store_loadflow_fn=store_loadflow,
            load_loadflow_fn=loadflow_ref,
            rng=rng,
            framework=grid_file.framework,
            runners=runner_group,
            worst_k_runner_groups=worst_k_runner_group,
            action_set=action_set,
        ),
        initial_strategy_message[0],
    )


def wait_for_first_dc_results(
    results_consumer: LongRunningKafkaConsumer,
    session: Session,
    max_wait_time: PositiveInt,
    optimization_id: str,
    heartbeat_fn: Callable[[], None],
    optimization_logger: Optional[BindableLogger] = None,
) -> None:
    """Wait an initial period for DC results to arrive before proceeding with the optimization.

    Call this after initialize optimization and before run epoch to ensure that the DC optimizer has started, and avoid the
    AC optimizer idling while waiting for the first DC results to arrive.

    Parameters
    ----------
    results_consumer : LongRunningKafkaConsumer
        The consumer where to listen for DC results
    session : Session
        The database session to use for storing topologies
    max_wait_time : PositiveInt
        The maximum time to wait for DC results, in seconds
    optimization_id : str
        The ID of the optimization run, used to filter the incoming topologies and only proceed when DC results from
        the correct optimization run arrive. Note that other DC runs could be active.
    heartbeat_fn : Callable[[], None]
        A function to send heartbeats while waiting for DC results, as this wait time can be relatively long.
    optimization_logger : Optional[BindableLogger]
        The logger to use for logging during the optimization.
        If None, the default logger from this module will be used.

    Raises
    ------
    TimeoutError
        If no DC results arrive within the maximum wait time
    """
    optimization_logger = optimization_logger or logger
    existing_dc_topology_id = session.exec(
        select(ACOptimTopology.id)
        .where(ACOptimTopology.optimization_id == optimization_id)
        .where(ACOptimTopology.optimizer_type == OptimizerType.DC)
    ).first()
    if isinstance(existing_dc_topology_id, int):
        optimization_logger.info("DC results were already available in the database, proceeding with optimization")
        return

    start_wait = datetime.now()
    poll_iteration = 0
    while datetime.now() - start_wait < timedelta(seconds=max_wait_time):
        poll_iteration += 1
        elapsed_seconds = (datetime.now() - start_wait).total_seconds()
        added_topos, stopped_optimization_ids = poll_results_topic(db=session, consumer=results_consumer, first_poll=True)
        optimization_logger.debug(
            f"DC poll iteration={poll_iteration}, elapsed_seconds={elapsed_seconds:.2f}, "
            f"received_topologies={len(added_topos)}"
        )
        new_topos_for_optimization = added_topos.get(optimization_id, 0)
        if new_topos_for_optimization > 0:
            optimization_logger.info(
                f"Received {new_topos_for_optimization} topologies from DC results, proceeding with optimization"
            )
            return
        if optimization_id in stopped_optimization_ids:
            optimization_logger.warning(
                "Received DC optimization stopped message before receiving any DC results, stop optimization"
            )
            break
        heartbeat_fn()
    raise TimeoutError(f"Did not receive DC results within {max_wait_time} seconds, cannot proceed with optimization")


def persist_and_send_topology(
    topology: ACOptimTopology,
    scoring_result: TopologyScoringResult,
    optimizer_data: OptimizerData,
    epoch: int,
    enable_ac_rejection: bool,
    send_result_fn: Callable[[ResultUnion], None],
    strategy_logger: Optional[BindableLogger] = None,
) -> None:
    """Persist the topologies in the database and send the result to the result topic.

    This function takes care of storing the topologies in the database, including the loadflow results if available,
    and sending the appropriate message to the result topic, either a TopologyPushResult if the
    topologies are accepted, or a TopologyRejectionResult if the topologies are
    rejected based on the scoring result and the enable_ac_rejection flag.

    Parameters
    ----------
    topology : ACOptimTopology
        The topology to persist and send
    scoring_result : TopologyScoringResult
        The scoring result for the topology, containing the metrics, loadflow results and rejection reason if any
    optimizer_data : OptimizerData
        The optimizer data containing the session and the loadflow storage function
    epoch : int
        The current epoch number, used for logging
    enable_ac_rejection : bool
        Whether to enable AC rejection based on the scoring result's rejection reason,
        or to always accept the strategy regardless of the scoring result
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results to the result topic
    strategy_logger : Optional[BindableLogger]
        The logger to use for logging during the strategy persistence and sending.
        If None, the default logger from this module will be used.
    """
    strategy_logger = strategy_logger or logger
    rejection_reason = scoring_result.rejection_reason
    loadflow_result_reference = None
    local_enable_ac_rejection = enable_ac_rejection or (
        rejection_reason is not None and (rejection_reason.criterion == "error" or rejection_reason.early_stopping)
    )
    if scoring_result.loadflow_results is not None:
        try:
            loadflow_result_reference = optimizer_data.store_loadflow_fn(scoring_result.loadflow_results)
        except Exception as exc:
            strategy_logger.error("Error while storing loadflow results", error=str(exc))
            rejection_reason = TopologyRejectionReason(
                criterion="error", description=str(exc), value_after=1.0, value_before=0.0, early_stopping=False
            )
            scoring_result = TopologyScoringResult(
                loadflow_results=None,
                metrics=Metrics(fitness=INF_FITNESS, extra_scores={}),
                rejection_reason=rejection_reason,
            )
            local_enable_ac_rejection = True

    message_topos = []
    if "fitness_dc" in topology.metrics:
        scoring_result.metrics.extra_scores["fitness_dc"] = topology.metrics["fitness_dc"]
    topology.metrics = scoring_result.metrics.extra_scores
    topology.fitness = scoring_result.metrics.fitness
    topology.acceptance = rejection_reason is None
    topology.set_loadflow_reference(loadflow_result_reference)

    optimizer_data.session.add(topology)

    message_topos.append(
        Topology(
            actions=topology.actions,
            pst_setpoints=topology.pst_setpoints,
            disconnections=topology.disconnections,
            loadflow_results=loadflow_result_reference,
            metrics=scoring_result.metrics,
        )
    )

    optimizer_data.session.commit()

    if not local_enable_ac_rejection or rejection_reason is None:
        send_result_fn(TopologyPushResult(strategies=[Strategy(timesteps=message_topos)], epoch=epoch))
        strategy_logger.info(
            "Strategy completed",
            accept=True,
            metrics=scoring_result.metrics.extra_scores,
            fitness=scoring_result.metrics.fitness,
        )
    else:
        send_result_fn(
            TopologyRejectionResult(
                reason=rejection_reason,
                strategy=Strategy(timesteps=message_topos),
                epoch=epoch,
            )
        )
        strategy_logger.info(
            "Strategy completed",
            accept=False,
            reason=get_topology_rejection_message(rejection_reason),
        )


def run_fast_failing_epoch(
    optimizer_data: OptimizerData,
    epoch_logger: Optional[BindableLogger] = None,
) -> tuple[list[ACOptimTopology], list[EarlyStoppingStageResult]]:
    """Run one epoch of fast-failing AC evaluation only.

    This imports new DC topologies, pulls up to ``topology_batch_size`` strategies from the
    evolution function, evaluates them with the worst-k fast-failing scorer, and returns the
    evaluated strategies together with their early-stage scoring results.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data containing the evolution and fast-failing scoring functions.
    epoch : int
        The current epoch number, used for logging and for first-poll behavior.
    epoch_logger : Optional[BindableLogger]
        The logger to use for logging during the epoch.

    Returns
    -------
    list[ACOptimTopology]
        The strategies that were pulled and evaluated in the fast-failing stage.
    list[EarlyStoppingStageResult]
        The corresponding fast-failing scoring results.
    """
    epoch_logger = epoch_logger or logger
    epoch_logger.info("Starting AC fast-failing epoch")

    topologies = optimizer_data.evolution_fn()
    n_topologies = len(topologies)
    if n_topologies == 0:
        epoch_logger.debug("Evolution returned no new strategy")
        return [], []
    epoch_logger.debug("Running early-stopping batch", n_topologies=n_topologies)
    fast_failing_results = optimizer_data.worst_k_scoring_fn(topologies)

    return topologies, fast_failing_results


def process_fast_failing_results(
    optimizer_data: OptimizerData,
    topologies: list[ACOptimTopology],
    fast_failing_results: list[EarlyStoppingStageResult],
    epoch: int,
    send_result_fn: Callable[[ResultUnion], None],
    epoch_logger: Optional[BindableLogger] = None,
) -> tuple[list[ACOptimTopology], list[EarlyStoppingStageResult]]:
    """Process the results of the fast-failing stage and send the surviving strategies to the result topic.

    This function takes care of sending the appropriate message to the result topic
    for each strategy, either a TopologyPushResult if the strategy passed the fast-failing stage,
    or a TopologyRejectionResult if the strategy was rejected based on the fast-failing scoring result.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data containing the session and the loadflow storage function.
    topologies : list[ACOptimTopology]
        The topologies that were evaluated in the fast-failing stage.
    fast_failing_results : list[EarlyStoppingStageResult]
        The corresponding fast-failing scoring results, containing the rejection reason if any.
    epoch : int
        The current epoch number, used for logging and for sending in the result messages.
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results to the result topic.
    epoch_logger : Optional[BindableLogger]
        The logger to use for logging during the processing. If None, the default logger from this module will be used.

    Returns
    -------
    list[ACOptimTopology]
        The topologies that passed the fast-failing stage and can proceed to the remaining-contingency evaluation.
    list[EarlyStoppingStageResult]
        The corresponding fast-failing scoring results for the surviving topologies.
    """
    epoch_logger = epoch_logger or logger
    epoch_logger.debug("Validating early-stopping batch", n_topologies=len(topologies))

    survivor_topologies: list[ACOptimTopology] = []
    survivor_early_results: list[EarlyStoppingStageResult] = []
    for topology, early_stop_result in zip(topologies, fast_failing_results, strict=True):
        if early_stop_result.rejection_reason is None:
            # If the topology is not rejected we want to fully evaluate it later
            survivor_topologies.append(topology)
            survivor_early_results.append(early_stop_result)
            continue

        persist_and_send_topology(
            topology=topology,
            scoring_result=early_stop_result,
            optimizer_data=optimizer_data,
            epoch=epoch,
            enable_ac_rejection=True,
            send_result_fn=send_result_fn,
            strategy_logger=epoch_logger,
        )
    return survivor_topologies, survivor_early_results


def run_remaining_epoch(
    optimizer_data: OptimizerData,
    topologies: list[ACOptimTopology],
    early_stage_results: list[EarlyStoppingStageResult],
    epoch_logger: Optional[BindableLogger] = None,
) -> tuple[list[ACOptimTopology], list[TopologyScoringResult]]:
    """Run one epoch of remaining-contingency AC evaluation only.

    This evaluates the full remaining-loadflow stage for strategies that already passed the
    fast-failing worst-k evaluation and returns the same strategies together with their final
    scoring results.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data containing the remaining-stage scoring function.
    topologies : list[ACOptimTopology]
        The survivor topologies to evaluate in the remaining stage.
    early_stage_results : list[EarlyStoppingStageResult]
        The corresponding fast-failing stage results for each topology.
    epoch : int
        The current epoch number, used for logging.
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results to the result topic, used for logging strategy progress in the remaining
    epoch_logger : Optional[BindableLogger]
        The logger to use for logging during the epoch.

    Returns
    -------
    list[ACOptimTopology]
        The topologies that were evaluated in the remaining stage.
    list[TopologyScoringResult]
        The corresponding final scoring results.
    """
    epoch_logger = epoch_logger or logger
    if not topologies:
        epoch_logger.debug("No topologies provided for remaining contingencies")
        return [], []

    epoch_logger.info("Starting AC remaining epoch")
    epoch_logger.debug("Running remaining contingencies", survivor_count=len(topologies))
    full_results = optimizer_data.scoring_fn(topologies, early_stage_results)
    return topologies, full_results


def process_remaining_results(
    optimizer_data: OptimizerData,
    topologies: list[ACOptimTopology],
    full_results: list[TopologyScoringResult],
    epoch: int,
    send_result_fn: Callable[[ResultUnion], None],
    epoch_logger: Optional[BindableLogger] = None,
) -> tuple[list[ACOptimTopology], list[TopologyScoringResult]]:
    """Process the results of the remaining-contingency stage and send the strategies to the result topic.

    Parameters
    ----------
    optimizer_data : OptimizerData
        The optimizer data containing the session and the loadflow storage function.
    topologies : list[ACOptimTopology]
        The topologies that were evaluated in the remaining stage.
    full_results : list[TopologyScoringResult]
        The corresponding full scoring results, containing the final metrics,
        loadflow results and rejection reason if any.
    epoch : int
        The current epoch number, used for logging and for sending in the result messages.
    send_result_fn : Callable[[ResultUnion], None]
        The function to send results to the result topic.
    epoch_logger : Optional[BindableLogger]
        The logger to use for logging during the processing.
        If None, the default logger from this module will be used.

    Returns
    -------
    list[ACOptimTopology]
        The topologies that were evaluated in the remaining stage,
        with their metrics and fitness updated based
        on the full scoring results.
    list[TopologyScoringResult]
        The corresponding full scoring results for each topology,
        which can be used for further processing in the next epoch, e.g. for the evolution function.
    """
    epoch_logger = epoch_logger or logger
    epoch_logger.info("Storing remaining-contingency batch", n_topologies=len(topologies))
    for topology, full_result in zip(topologies, full_results, strict=True):
        persist_and_send_topology(
            topology=topology,
            scoring_result=full_result,
            optimizer_data=optimizer_data,
            epoch=epoch,
            enable_ac_rejection=optimizer_data.params.ga_config.enable_ac_rejection,
            send_result_fn=send_result_fn,
            strategy_logger=epoch_logger,
        )
    return topologies, full_results


def evaluate_remaining_contingencies(
    send_result_fn: Callable[[ResultUnion, str], None],
    optimizer_data: OptimizerData,
    epoch: int,
    survivor_topologies: list[ACOptimTopology],
    survivor_early_results: list[EarlyStoppingStageResult],
    epoch_logger: Optional[BindableLogger] = None,
) -> None:
    """Evaluate the remaining contingencies for the strategies that passed the fast-failing stage.

    Parameters
    ----------
    survivor_topologies : list[ACOptimTopology]
        The topologies that passed the fast-failing stage and need to be evaluated
        on the remaining contingencies.
    survivor_early_results : list[EarlyStoppingStageResult]
        The early results from the fast-failing stage for the topologies that passed.
    send_result_fn : Callable[[ResultUnion, str], None]
        The function to send results to the result topic, used for sending the final results
        after evaluating the remaining contingencies.
    optimizer_data : OptimizerData
        The optimizer data containing the scoring function for the remaining contingencies and other necessary data for the
        evaluation.
    epoch : int
        The current epoch number, used for logging and for sending in the result messages.
    epoch_logger : Optional[BindableLogger]
        The logger to use for logging during the evaluation. If None, the default logger from this module will be used.
    """
    topologies, full_results = run_remaining_epoch(
        topologies=survivor_topologies,
        early_stage_results=survivor_early_results,
        optimizer_data=optimizer_data,
        epoch_logger=epoch_logger,
    )
    process_remaining_results(
        optimizer_data=optimizer_data,
        topologies=topologies,
        full_results=full_results,
        epoch_logger=epoch_logger,
        send_result_fn=send_result_fn,
        epoch=epoch,
    )
