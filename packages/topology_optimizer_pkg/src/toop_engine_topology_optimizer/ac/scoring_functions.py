# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Scoring functions for the AC optimizer - in this case this runs an N-1 and computes metrics for it"""

from typing import Optional

import logbook
import numpy as np
import pandas as pd
import polars as pl
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics as compute_metrics_lfs
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner, AdditionalActionInfo
from toop_engine_interfaces.asset_topology import RealizedTopology
from toop_engine_interfaces.loadflow_result_helpers_polars import concatenate_loadflow_results_polars, select_timestep_polars
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.evolution_functions import get_contingency_indices_from_ids
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

logger = logbook.Logger(__name__)


def get_threshold_n_minus1_overload(
    strategy: list[ACOptimTopology],
) -> tuple[Optional[list[float]], Optional[list[list[int]]]]:
    """Extract the 'top_k_overloads_n_1' thresholds and corresponding case indices from a list of ACOptimTopology strategies.

    overload_threshold is defined as the maximum allowed overload energy for the worst k AC N-1 contingency analysis
    of the split topologies. This threshold is computed using the worst k AC contingencies of the unsplit grid and the
    worst k DC contingencies of the split grid. Refer to the "pull" method in evolution_functions.py for more details.

    Parameters
    ----------
    strategy : list of ACOptimTopology
        A list of ACOptimTopology objects, each containing a 'metrics' dictionary with overload thresholds and case indices.

    Returns
    -------
    tuple of (Optional[list of float], Optional[list of list of int])
        A tuple containing:
        - A list of overload thresholds for each topology, or None if any required metric is missing.
        - A list of lists of case indices for each topology, or None if any required metric is missing.

    """
    overload_threshold_all_t = []
    case_indices_all_t = []
    for topo in strategy:
        threshold_overload = topo.metrics.get("top_k_overloads_n_1", None)
        threshold_case_indices = topo.worst_k_contingency_cases
        if threshold_overload is None or len(threshold_case_indices) == 0:
            logger.warning(
                f"No overload threshold or case indices found in the strategy"
                f"threshold_overload: {threshold_overload}, threshold_case_indices: {threshold_case_indices}"
            )
            return None, None

        overload_threshold_all_t.append(threshold_overload)
        case_indices_all_t.append(threshold_case_indices)

    return overload_threshold_all_t, case_indices_all_t


def update_runner_nminus1(
    runners: list[AbstractLoadflowRunner], nminus1_defs: list[Nminus1Definition], case_ids_all_t: list[list[str]]
) -> None:
    """Update the N-1 definitions in the runners to only include the worst k contingencies.

    This modifies the N-1 definitions in the runners to only include the contingencies at the given indices.

    Parameters
    ----------
    runners : list[AbstractLoadflowRunner]
        The loadflow runners to update. The length of the list equals the number of timesteps.
    nminus1_defs : list[Nminus1Definition]
        The original N-1 definitions to use as a template. The length of the list equals the number of timesteps.
    case_ids_all_t : list[list[str]]
        A list of contingency ids for each runner, indicating which contingencies to keep in the N-1 definition.
        Each element should be an index to the contingencies in the original N-1 definition.
    """
    case_indices_all_t = get_contingency_indices_from_ids(case_ids_all_t, n_minus1_definitions=nminus1_defs)
    for runner, case_indices, n1_def in zip(runners, case_indices_all_t, nminus1_defs, strict=True):
        contingencies = np.array(n1_def.contingencies)[list(case_indices)]
        n1_def_copy = n1_def.model_copy()
        n1_def_copy.contingencies = contingencies.tolist()
        runner.store_nminus1_definition(n1_def_copy)


def compute_loadflow_and_metrics(
    runners: list[AbstractLoadflowRunner],
    strategy: list[ACOptimTopology],
    base_case_ids: list[Optional[str]],
    n_timestep_processes: int = 1,
) -> tuple[LoadflowResultsPolars, list[Optional[AdditionalActionInfo]], list[Metrics]]:
    """Compute loadflow results and associated metrics for a given set of strategies.

    This function runs loadflow simulations for each provided strategy using the specified runners,
    then computes additional metrics based on the simulation results.

    Parameters
    ----------
    runners : list of AbstractLoadflowRunner
        List of loadflow runner instances to use for simulations.
    strategy : list of ACOptimTopology
        List of topology strategies to evaluate.
    base_case_ids : list of Optional[str]
        List of base case identifiers corresponding to each strategy. Can be None.
    n_timestep_processes : int, optional
        Number of parallel processes to use for timestep simulations (default is 1).

    Returns
    -------
    lfs : LoadflowResultsPolars
        The results of the loadflow simulations.
    additional_info : list of Optional[AdditionalActionInfo]
        Additional information for each action taken in the strategies.
    metrics : list of Metrics
        Computed metrics for each strategy.
    """
    lfs, additional_info = compute_loadflow(
        actions=[topo.actions for topo in strategy],
        disconnections=[topo.disconnections for topo in strategy],
        runners=runners,
        n_timestep_processes=n_timestep_processes,
    )
    metrics = compute_metrics(
        strategy=strategy,
        lfs=lfs,
        additional_info=additional_info,
        base_case_ids=base_case_ids,
    )
    return lfs, additional_info, metrics


def compute_loadflow_and_metrics_with_early_stopping(
    runners: list[AbstractLoadflowRunner],
    strategy: list[ACOptimTopology],
    base_case_ids: list[Optional[str]],
    threshold_overload_all_t: list[float],
    threshold_case_ids_all_t: list[list[str]],
    n_timestep_processes: int = 1,
    early_stop_non_converging_threshold: float = 0.1,
) -> tuple[LoadflowResultsPolars, list[Metrics]]:
    """Run N-1 loadflow analysis with early stopping based on overload thresholds.

    This function first runs loadflow analysis for the worst k contingencies, checking if overload energy
    exceeds the specified thresholds. If so, it stops further analysis and returns the results.
    If overload energy is within thresholds, it continues to run loadflow for non-critical contingencies.

    This optimizes loadflow analysis by focusing on critical contingencies first, defined by the provided thresholds.
    Early stopping avoids unnecessary computations if overload energy exceeds the specified thresholds.

    Parameters
    ----------
    runners : list of AbstractLoadflowRunner
        Loadflow runner instances for each timestep.
    strategy : list of ACOptimTopology
        AC optimization topologies for each timestep.
    base_case_ids : list of Optional[str]
        Base case identifiers for each timestep. If None, N-0 analysis is skipped.
    threshold_overload_all_t : list of float
        Overload energy thresholds for early stopping at each timestep.
    threshold_case_ids_all_t : list of list of str
        Case IDs of critical contingencies for each timestep.
    n_timestep_processes : int, optional
        Number of parallel processes for loadflow computation (default is 1).
    early_stop_non_converging_threshold : float, optional
        The threshold for the early stopping criterion, i.e. if the percentage of non-converging cases is greater than
        this value, the ac validation will be stopped early.

    Returns
    -------
    lfs : LoadflowResultsPolars
        Concatenated loadflow results for critical and non-critical contingencies.
    metrics : list of Metrics
        Computed metrics for the strategy and loadflow results.

    Notes
    -----
    - Early stopping is triggered if overload energy for any timestep exceeds the threshold.
    - Critical contingencies are processed first; if early stopping is triggered, non-critical contingencies are skipped.
    - Runner definitions are updated to focus on critical or non-critical contingencies as needed.

    """
    logger.info("Running N-1 analysis with early stopping.")

    contingency_ids_all_t = []
    original_n_minus1_defs = []
    for runner in runners:
        n_1_def = runner.get_nminus1_definition()
        contingency_ids_all_t.append([contingency.id for contingency in n_1_def.contingencies])
        original_n_minus1_defs.append(n_1_def)

    # Update the N-1 definitions in the runners to only include the critical contingencies
    n_critical_contingencies = len(threshold_case_ids_all_t[0])
    update_runner_nminus1(runners, original_n_minus1_defs, threshold_case_ids_all_t)
    logger.info(f"Running N-1 analysis with {n_critical_contingencies} critical contingencies per timestep.")

    # We pass the base case IDs to None to prevent N-0 analysis in the runners
    # Compute the loadflow and metrics with only critical contingencies included in the N-1 analysis. Critical contingencies
    # are defined by the threshold_case_ids_all_t.
    lfs_critical, _, metrics_critical = compute_loadflow_and_metrics(
        runners, strategy, [None] * len(threshold_case_ids_all_t), n_timestep_processes
    )

    # Early stopping: check if overload_energy_n_1 exceed thresholds
    stop_early = False
    for metric, overload_th in zip(metrics_critical, threshold_overload_all_t, strict=True):
        overload = metric.extra_scores.get("overload_energy_n_1", 0)
        logger.info(f"Checking overload for N-1 analysis: overload = {overload}, (overload_worst_k_unsplit)={overload_th}")

        n_nonconverging_cases = metric.extra_scores.get("non_converging_loadflows", 0)
        logger.info(f"Number of non converging cases = {n_nonconverging_cases}")

        # if overload is greater than overload_th or n_nonconverging_cases is greater than
        # early_stop_non_converging_threshold(10%) of all critical cases, we stop
        if overload > overload_th or n_nonconverging_cases > int(
            early_stop_non_converging_threshold * n_critical_contingencies
        ):
            logger.info(
                f"Early stopping N-1 analysis "
                f" overload: {overload}, threshold_overload: {overload_th}, "
                f"n_nonconverging_cases: {n_nonconverging_cases}, "
                f"threshold_n_non_converging_cases: {int(early_stop_non_converging_threshold * n_critical_contingencies)}"
            )
            stop_early = True
            metric.fitness = -99999999  # Set fitness to a very low value to indicate failure
            metric.extra_scores["overload_energy_n_1"] = 9999999
            break

    if not stop_early:
        # Determine non-critical contingencies by excluding critical ones from all contingencies
        non_critical_contingencies_all_t = [
            set(all_ids) - set(critical_ids)
            for all_ids, critical_ids in zip(contingency_ids_all_t, threshold_case_ids_all_t, strict=True)
        ]

        # Update the N-1 definitions in the runners to now include only the non-critical contingencies.
        logger.info(
            f"Running N-1 analysis with {len(non_critical_contingencies_all_t[0])} non-critical contingencies per timestep."
        )
        update_runner_nminus1(runners, original_n_minus1_defs, non_critical_contingencies_all_t)

        lfs_non_critical, additional_info_non_critical = compute_loadflow(
            actions=[topo.actions for topo in strategy],
            disconnections=[topo.disconnections for topo in strategy],
            runners=runners,
            n_timestep_processes=n_timestep_processes,
        )

        lfs = concatenate_loadflow_results_polars([lfs_critical, lfs_non_critical])

        # We can pass the additional info from either critical or non critical contingencies as they are the same
        metrics = compute_metrics(
            strategy=strategy,
            lfs=lfs,
            additional_info=additional_info_non_critical,
            base_case_ids=base_case_ids,
        )
    else:
        lfs = lfs_critical
        metrics = metrics_critical

    # Restore the original N-1 definitions in the runners
    for runner, original_n1_def in zip(runners, original_n_minus1_defs, strict=True):
        runner.store_nminus1_definition(original_n1_def)

    return lfs, metrics


def scoring_function(
    strategy: list[ACOptimTopology],
    runners: list[AbstractLoadflowRunner],
    base_case_ids: list[Optional[str]],
    n_timestep_processes: int = 1,
    early_stop_validation: bool = True,
    early_stop_non_converging_threshold: float = 0.1,
) -> tuple[LoadflowResultsPolars, list[Metrics]]:
    """Compute loadflows and metrics for a given strategy

    Parameters
    ----------
    strategy : list[ACOptimTopology]
        The strategy to score, length n_timesteps
    runners : list[AbstractLoadflowRunner]
        The loadflow runners to use, length n_timesteps.
    base_case_ids : list[Optional[str]]
        The base case ids for the loadflow runners, length n_timesteps (used to separately compute the N-0 flows)
    n_timestep_processes : int, optional
        The number of processes to use for computing timesteps in parallel, by default 1
    early_stop_validation : bool, optional
        Whether to enable early stopping during the optimization process, by default True
    early_stop_non_converging_threshold : float = 0.1
        The threshold for the early stopping criterion, i.e. if the percentage of non-converging cases is greater than
        this value, the ac validation will be stopped early.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for the strategy
    list[Metrics]
        The metrics for the strategy
    """
    # overload_threshold is defined as the maximum allowed overload energy for the worst k N-1 contingencies
    # of split topologies. This threshold is computed using the worst k contingencies of the unsplit grid.
    # The thresholds are available only when the strategy has been pulled from the repertoire using the
    # pull method. This means that early stopping can be used only for AC validation and not for AC optimization.
    threshold_overload_all_t, threshold_case_ids_all_t = get_threshold_n_minus1_overload(strategy)
    overload_threshold_available = threshold_overload_all_t is not None and threshold_case_ids_all_t is not None

    if overload_threshold_available and early_stop_validation:
        lfs, metrics = compute_loadflow_and_metrics_with_early_stopping(
            runners,
            strategy,
            base_case_ids,
            threshold_overload_all_t=threshold_overload_all_t,
            threshold_case_ids_all_t=threshold_case_ids_all_t,
            n_timestep_processes=n_timestep_processes,
            early_stop_non_converging_threshold=early_stop_non_converging_threshold,
        )
    else:
        logger.info("No overload thresholds available, running full N-1 analysis.")
        lfs, _additional_info, metrics = compute_loadflow_and_metrics(runners, strategy, base_case_ids, n_timestep_processes)
    return lfs, metrics


def compute_metrics(
    strategy: list[ACOptimTopology],
    lfs: LoadflowResultsPolars,
    additional_info: list[Optional[AdditionalActionInfo]],
    base_case_ids: list[Optional[str]],
) -> list[Metrics]:
    """Compute the metrics for a given strategy. Just calls compute_metrics_single_timestep for each timestep

    Parameters
    ----------
    strategy : list[ACOptimTopology]
        The strategy to score, length n_timesteps
    lfs : LoadflowResults
        The loadflow results for the strategy, length n_timesteps
    additional_info : list[Optional[AdditionalActionInfo]]
        Additional information about the actions taken, such as switching distance or other metrics. The length of
        the list is n_timesteps.
    base_case_ids : list[Optional[str]]
        The base case ids for the loadflow runners, length n_timesteps (used to separately
        compute the N-0 flows)

    Returns
    -------
    list[Metrics]
        The metrics for the strategy, length n_timesteps
    """
    return [
        compute_metrics_single_timestep(
            actions=topo.actions,
            disconnections=topo.disconnections,
            loadflow=select_timestep_polars(lfs, timestep=timestep),
            additional_info=info,
            base_case_id=base_case_id,
        )
        for timestep, (topo, info, base_case_id) in enumerate(zip(strategy, additional_info, base_case_ids, strict=True))
    ]


def extract_switching_distance(additional_info: AdditionalActionInfo) -> int:
    """Extract the switching distance from the additional action info

    Parameters
    ----------
    additional_info : AdditionalActionInfo
        The additional action info containing the switching distance

    Returns
    -------
    int
        The switching distance, or 0 if not available
    """
    if isinstance(additional_info, RealizedTopology):
        return len(additional_info.reassignment_diff)
    if isinstance(additional_info, pd.DataFrame):
        return len(additional_info)
    raise ValueError("Unknown format for additional info")


def compute_metrics_single_timestep(
    actions: list[int],
    disconnections: list[int],
    loadflow: LoadflowResultsPolars,
    additional_info: Optional[AdditionalActionInfo],
    base_case_id: Optional[str] = None,
) -> Metrics:
    """Compute the metrics for a single timestep

    Parameters
    ----------
    actions : list[int]
        The reconfiguration assignment for the timestep
    disconnections : list[int]
        The disconnections for the timestep
    loadflow : LoadflowResults
        The loadflow results for the timestep, use select_timestep to get the results for a specific timestep
    additional_info : Optional[AdditionalActionInfo]
        Additional information about the actions taken, such as switching distance or other metrics.
    base_case_id: Optional[str]
        The base case id from the nminus1 definition, to separate N-0 flows from N-1

    Returns
    -------
    Metrics
        The metrics for the timestep
    """
    metrics = compute_metrics_lfs(loadflow_results=loadflow, base_case_id=base_case_id)
    metrics = {key: np.nan_to_num(value, nan=0, posinf=99999999, neginf=-99999999).item() for key, value in metrics.items()}
    non_successful_states = [
        ConvergenceStatus.FAILED.value,
        ConvergenceStatus.MAX_ITERATION_REACHED.value,
        ConvergenceStatus.NO_CALCULATION.value,
    ]
    metrics.update(
        {
            "split_subs": len(actions),
            "disconnected_branches": len(disconnections),
            "non_converging_loadflows": loadflow.converged.filter(pl.col("status").is_in(non_successful_states))
            .select(pl.len())
            .collect()
            .item(),
        }
    )
    if additional_info is not None:
        metrics["switching_distance"] = extract_switching_distance(additional_info)
    worst_k_contingent_cases = metrics.pop("worst_k_contingent_cases", None)
    return Metrics(
        fitness=metrics["overload_energy_n_1"], extra_scores=metrics, worst_k_contingent_cases=worst_k_contingent_cases
    )


def compute_loadflow(
    actions: list[list[int]],
    disconnections: list[list[int]],
    runners: list[AbstractLoadflowRunner],
    n_timestep_processes: int = 1,  # noqa: ARG001
) -> tuple[LoadflowResultsPolars, list[AdditionalActionInfo]]:
    """Compute the loadflow for a given strategy

    Parameters
    ----------
    actions : list[list[int]]
        The reconfiguration actions for each timestep, where the outer list is the timestep dimension and
        the inner list the split substation identified through an index into the action set.
    disconnections : list[list[int]]
        The disconnections for each timestep, where the outer list is the timestep dimension and
        the inner list the disconnection indices
    runners : list[AbstractLoadflowRunner]
        The loadflow runners to use
    n_timestep_processes : int
        The number of processes to use for each timestep

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for all timesteps in the strategy
    list[AdditionalActionInfo]
        Additional information about the actions taken, such as switching distance or other metrics. The length of
        the list is n_timesteps.
    """
    lf_results = []
    additional_information = []
    for action, disconnection, runner in zip(actions, disconnections, runners, strict=True):
        loadflow = runner.run_ac_loadflow(action, disconnection)
        lf_results.append(loadflow)
        additional_information.append(runner.get_last_action_info())

    return concatenate_loadflow_results_polars(lf_results), additional_information


def evaluate_acceptance(
    loadflow_results_split: LoadflowResultsPolars,  # noqa: ARG001
    metrics_split: list[Metrics],
    loadflow_results_unsplit: LoadflowResultsPolars,  # noqa: ARG001
    metrics_unsplit: list[Metrics],
    reject_convergence_threshold: float = 1.0,
    reject_overload_threshold: float = 0.95,
    reject_critical_branch_threshold: float = 1.1,
) -> bool:
    """Evaluate if the split loadflow results are acceptable compared to the unsplit results.

    Compares the unsplit metrics * the thresholds to the split metrics. If all split metrics are better than
    the unsplit metrics * thresholds, the split results are accepted.

    Checked metrics are:
        non_converging_loadflows: the number of non-converging loadflows should be less than or equal to
            reject_convergence_threshold * unsplit[non_converging_loadflows]
        overload_energy_n_1: the overload energy should be less than or equal to
            reject_overload_threshold * unsplit[overload_energy_n_1]
        critical_branch_count_n_1: the number of critical branches should be less than or equal
            to reject_critical_branch_threshold * unsplit[critical_branch_count_n_1]
        TODO: Check Voltage Jumps between N0 and N1

    Parameters
    ----------
    loadflow_results_split : LoadflowResultsPolars
        The loadflow results for the split case.
    metrics_split : list[Metrics]
        The metrics for the split case.
    loadflow_results_unsplit : LoadflowResultsPolars
        The loadflow results for the unsplit case.
    metrics_unsplit : list[Metrics]
        The metrics for the unsplit case.
    reject_convergence_threshold : float, optional
        The threshold for the convergence rate, by default 1.
        (i.e. the split case must have at most the same amount of nonconverging loadflows as the unsplit case.)
    reject_overload_threshold : float, optional
        The threshold for the overload energy improvement, by default 0.95
        (i.e. the split case must have at least 5% lower overload energy than the unsplit case).
    reject_critical_branch_threshold : float, optional
        The threshold for the critical branches increase, by default 1.1
        (i.e. the split case must not have more than 110 % of the critical branches in the unsplit case).

    Returns
    -------
    bool
        True if the split results are acceptable, False if rejected.
    """
    n_non_converged_unsplit = np.array(
        [unsplit.extra_scores.get("non_converging_loadflows", 0) for unsplit in metrics_unsplit]
    )
    n_non_converged_split = np.array(
        [
            split.extra_scores.get("non_converging_loadflows", 0) - split.extra_scores.get("disconnected_branches", 0)
            for split in metrics_split
        ]
    )
    convergence_acceptable = np.all(n_non_converged_split <= n_non_converged_unsplit * reject_convergence_threshold)
    if not convergence_acceptable:
        logger.info(
            "Rejecting topology due to insufficient convergence: "
            f"{n_non_converged_split} vs {n_non_converged_unsplit} before"
        )

    unsplit_overload = np.array([unsplit.extra_scores.get("overload_energy_n_1", 0) for unsplit in metrics_unsplit])
    split_overload = np.array([split.extra_scores.get("overload_energy_n_1", 99999) for split in metrics_split])
    overload_improvement = np.all(split_overload <= unsplit_overload * reject_overload_threshold)
    if not overload_improvement:
        logger.info(
            f"Rejecting topology due to overload energy not improving: {split_overload} vs {unsplit_overload} before"
        )

    unsplit_critical_branches = np.array(
        [unsplit.extra_scores.get("critical_branch_count_n_1", 999) for unsplit in metrics_unsplit], dtype=float
    )
    split_critical_branches = np.array(
        [split.extra_scores.get("critical_branch_count_n_1", 0) for split in metrics_split], dtype=float
    )

    critical_branches_acceptable = np.all(
        split_critical_branches <= unsplit_critical_branches * reject_critical_branch_threshold
    )
    if not critical_branches_acceptable:
        logger.info(
            "Rejecting topology due to critical branches increasing too much: "
            f"{split_critical_branches} vs {unsplit_critical_branches} before"
        )

    return convergence_acceptable and overload_improvement and critical_branches_acceptable
