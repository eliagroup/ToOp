# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Scoring functions for the AC optimizer - in this case this runs an N-1 and computes metrics for it"""

from dataclasses import dataclass

import logbook
import numpy as np
import pandas as pd
import polars as pl
from beartype.typing import Optional
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics as compute_metrics_lfs
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner, AdditionalActionInfo
from toop_engine_interfaces.asset_topology import RealizedTopology
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    concatenate_loadflow_results_polars,
    select_timestep_polars,
    subset_contingencies_polars,
)
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.evolution_functions import get_contingency_indices_from_ids
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, TopologyRejectionReason

logger = logbook.Logger(__name__)


def get_early_stopping_contingency_ids(
    strategy: list[ACOptimTopology],
    add_base_case_ids: Optional[list[str]] = None,
) -> Optional[list[list[str]]]:
    """Extract the contingency ids for early stopping from a list of ACOptimTopology strategies.

    This function looks for the 'top_k_overloads_n_1' metric in the strategy's metrics and extracts the corresponding
    worst k contingency case ids for each timestep. These ids are used to determine which contingencies to
    include in the N-1 analysis for early stopping.

    Parameters
    ----------
    strategy : list of ACOptimTopology
        A list of ACOptimTopology objects, each containing a 'metrics' dictionary with overload thresholds and case ids.
    add_base_case_ids : Optional[list[str]]
        An optional list of base case ids to include in the early stopping subset. If provided, these will be added to the
        list of contingency case ids for each timestep. The list is expected to have the same length as the strategy
        (number of timesteps), and each element is a string id of the base case.

    Returns
    -------
    Optional[list of list of str]
        A list of lists of contingency case IDs for each timestep, or None if any required metric is missing.
    """
    case_ids_all_t = []
    for topo, base_case_id in zip(strategy, add_base_case_ids or [None] * len(strategy), strict=True):
        worst_k_contingency_cases = topo.worst_k_contingency_cases
        if len(worst_k_contingency_cases) == 0:
            logger.warning(
                f"No overload threshold or case ids found in the strategy"
                f"worst_k_contingency_cases: {worst_k_contingency_cases}"
            )
            return None
        if base_case_id is not None:
            worst_k_contingency_cases.append(base_case_id)
        case_ids_all_t.append(worst_k_contingency_cases)
    return case_ids_all_t


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
    cases_subset: Optional[list[list[str]]] = None,
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
    cases_subset : list of list of str, optional
        Subset of contingency cases to use for loadflow computation. If None, all available contingencies are used.

    Returns
    -------
    lfs : LoadflowResultsPolars
        The results of the loadflow simulations.
    additional_info : list of Optional[AdditionalActionInfo]
        Additional information for each action taken in the strategies.
    metrics : list of Metrics
        Computed metrics for each strategy.
    """
    original_n_minus1_defs = [runner.get_nminus1_definition() for runner in runners]
    if cases_subset is not None:
        update_runner_nminus1(runners, original_n_minus1_defs, cases_subset)

    lfs, additional_info = compute_loadflow(
        actions=[topo.actions for topo in strategy],
        disconnections=[topo.disconnections for topo in strategy],
        pst_setpoints=[topo.pst_setpoints for topo in strategy],
        runners=runners,
        n_timestep_processes=n_timestep_processes,
    )
    metrics = compute_metrics(
        strategy=strategy,
        lfs=lfs,
        additional_info=additional_info,
        base_case_ids=base_case_ids,
    )

    if cases_subset is not None:
        # Restore the original N-1 definitions in the runners
        for runner, original_n1_def in zip(runners, original_n_minus1_defs, strict=True):
            runner.store_nminus1_definition(original_n1_def)

    return lfs, additional_info, metrics


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
    pst_setpoints: list[Optional[list[int]]],
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
    pst_setpoints : list[Optional[list[int]]]
        The PST setpoints for each timestep, where the outer list is the timestep dimension and the inner list the PST taps
        if computed.
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
    for action, disconnection, pst_setpoint, runner in zip(actions, disconnections, pst_setpoints, runners, strict=True):
        loadflow = runner.run_ac_loadflow(action, disconnection, pst_setpoint)
        lf_results.append(loadflow)
        additional_information.append(runner.get_last_action_info())

    return concatenate_loadflow_results_polars(lf_results), additional_information


def evaluate_acceptance(
    metrics_split: list[Metrics],
    metrics_unsplit: list[Metrics],
    reject_convergence_threshold: float = 1.0,
    reject_overload_threshold: float = 0.95,
    reject_critical_branch_threshold: float = 1.1,
    early_stopping: bool = False,
) -> Optional[TopologyRejectionReason]:
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
    metrics_split : list[Metrics]
        The metrics for the split case.
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
    early_stopping : bool, optional
        Whether the acceptance is computed as part of an early stopping criterion, will set the early_stopping field in the
        TopologyRejectionReason

    Returns
    -------
    Optional[TopologyRejectionReason]
        A TopologyRejectionReason if the split results are rejected, None if accepted.
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
        return TopologyRejectionReason(
            criterion="convergence",
            value_after=float(n_non_converged_split.sum()),
            value_before=float(n_non_converged_unsplit.sum()),
            threshold=reject_convergence_threshold,
            early_stopping=early_stopping,
        )

    unsplit_overload = np.array([unsplit.extra_scores.get("overload_energy_n_1", 0) for unsplit in metrics_unsplit])
    split_overload = np.array([split.extra_scores.get("overload_energy_n_1", 99999) for split in metrics_split])
    overload_improvement = np.all(split_overload <= unsplit_overload * reject_overload_threshold)
    if not overload_improvement:
        return TopologyRejectionReason(
            criterion="overload-energy",
            value_after=float(split_overload.sum()),
            value_before=float(unsplit_overload.sum()),
            threshold=reject_overload_threshold,
            early_stopping=early_stopping,
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
        return TopologyRejectionReason(
            criterion="critical-branch-count",
            value_after=float(split_critical_branches.sum()),
            value_before=float(unsplit_critical_branches.sum()),
            threshold=reject_critical_branch_threshold,
            early_stopping=early_stopping,
        )

    return None


def compute_remaining_loadflows(
    runners: list[AbstractLoadflowRunner],
    strategy: list[ACOptimTopology],
    base_case_ids: list[Optional[str]],
    loadflows_subset: LoadflowResultsPolars,
    cases_subset: list[list[str]],
    n_timestep_processes: int = 1,
) -> tuple[LoadflowResultsPolars, list[Metrics]]:
    """Compute the loadflows for the remaining contingencies that were not included in the early stopping subset.

    This function is called after the early stopping loadflows have been computed and accepted. It computes the loadflows
    for the remaining contingencies that were not included in the early stopping subset, and then computes the metrics for
    the full set of loadflows.

    Parameters
    ----------
    runners : list[AbstractLoadflowRunner]
        The loadflow runners to use, length n_timesteps.
    strategy : list[ACOptimTopology]
        The strategy to score, length n_timesteps
    base_case_ids : list[Optional[str]]
        The base case ids for the loadflow runners, length n_timesteps (used to separately compute the N-0 flows)
    loadflows_subset : LoadflowResultsPolars
        The loadflow results for the early stopping subset, used to avoid recomputing these loadflows.
    cases_subset : list[list[str]]
        The contingency case ids that were included in the early stopping subset for each timestep. This could be extracted
        from the loadflows_subset but as it is available it is faster to pass it in.
    n_timestep_processes : int
        The number of processes to use for computing timesteps in parallel.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for all contingencies, including those from the early stopping subset.
    list[Metrics]
        The metrics for the full set of loadflows.
    """
    all_cases = []
    original_n_minus1_defs = []
    for runner in runners:
        n_1_def = runner.get_nminus1_definition()
        all_cases.append([contingency.id for contingency in n_1_def.contingencies])
        original_n_minus1_defs.append(n_1_def)

    # Remove the already computed contingencies so we do not re-compute them
    remaining_cases = [
        set(all_ids) - set(computed_ids) for all_ids, computed_ids in zip(all_cases, cases_subset, strict=True)
    ]

    # Update the N-1 definitions in the runners to now include only the non-critical contingencies.
    logger.info(f"Running N-1 analysis with {len(remaining_cases[0])} non-critical contingencies per timestep.")
    update_runner_nminus1(runners, original_n_minus1_defs, remaining_cases)

    lfs_remaining, additional_info_remaining = compute_loadflow(
        actions=[topo.actions for topo in strategy],
        disconnections=[topo.disconnections for topo in strategy],
        pst_setpoints=[topo.pst_setpoints for topo in strategy],
        runners=runners,
        n_timestep_processes=n_timestep_processes,
    )

    lfs = concatenate_loadflow_results_polars([loadflows_subset, lfs_remaining])

    # We can pass the additional info from either critical or non critical contingencies as they are the same
    metrics = compute_metrics(
        strategy=strategy,
        lfs=lfs,
        additional_info=additional_info_remaining,
        base_case_ids=base_case_ids,
    )

    return lfs, metrics


@dataclass
class ACScoringParameters:
    """Parameters for ac scoring

    This is a subset of all ac parameters and grouped to shorten the signature of the
    scoring_and_acceptance function.
    """

    # --- Thresholds for acceptance criteria --- #
    reject_convergence_threshold: float
    reject_overload_threshold: float
    reject_critical_branch_threshold: float

    # --- Parameters for early stopping during N-1 analysis --- #
    base_case_ids: list[Optional[str]]
    n_timestep_processes: int
    early_stop_validation: bool
    early_stop_non_converging_threshold: float


def scoring_and_acceptance(
    strategy: list[ACOptimTopology],
    runners: list[AbstractLoadflowRunner],
    loadflow_results_unsplit: LoadflowResultsPolars,
    metrics_unsplit: list[Metrics],
    scoring_params: ACScoringParameters,
) -> tuple[LoadflowResultsPolars, list[Metrics], Optional[TopologyRejectionReason]]:
    """Compute the scoring and acceptance for a given strategy

    This function computes the loadflow results and metrics for the given strategy, and then evaluates the acceptance
    of the strategy based on the computed metrics and the unsplit metrics.

    Parameters
    ----------
    strategy : list[ACOptimTopology]
        The strategy to score, length n_timesteps
    runners : list[AbstractLoadflowRunner]
        The loadflow runners to use, length n_timesteps.
    loadflow_results_unsplit : LoadflowResultsPolars
        The loadflow results for the unsplit case.
    metrics_unsplit : list[Metrics]
        The metrics for the unsplit case.
    scoring_params : ACScoringParameters
        The parameters for scoring and acceptance evaluation.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for the strategy
    list[Metrics]
        The metrics for the strategy
    Optional[TopologyRejectionReason]
        A TopologyRejectionReason if the strategy is rejected, None if accepted.
    """
    # If early stopping is enabled, we compute and evaluate once on a subset of cases
    if scoring_params.early_stop_validation:
        cases_subset = get_early_stopping_contingency_ids(strategy, add_base_case_ids=scoring_params.base_case_ids)
        assert cases_subset is not None, (
            "Early stopping enabled but no contingency case ids found for early stopping."
            "This might happen when the DC optimizer pushes topologies without worst_k entries."
        )
        lfs_early_stop, _, metrics_early_stop = compute_loadflow_and_metrics(
            runners=runners,
            strategy=strategy,
            base_case_ids=scoring_params.base_case_ids,
            n_timestep_processes=scoring_params.n_timestep_processes,
            cases_subset=cases_subset,
        )
        # Flatten cases_subset for subsetting the unsplit results (which contains all timesteps)
        flat_cases_subset = [case_id for timestep_cases in cases_subset for case_id in timestep_cases]
        lfs_early_stop_unsplit = subset_contingencies_polars(loadflow_results_unsplit, flat_cases_subset)
        metrics_early_stop_unsplit = compute_metrics(
            strategy=strategy,
            lfs=lfs_early_stop_unsplit,
            additional_info=[None] * len(strategy),  # We do not pass any additional info - no switching distance available
            base_case_ids=scoring_params.base_case_ids,
        )
        rejection_reason = evaluate_acceptance(
            metrics_split=metrics_early_stop,
            metrics_unsplit=metrics_early_stop_unsplit,
            reject_convergence_threshold=scoring_params.reject_convergence_threshold,
            reject_overload_threshold=scoring_params.reject_overload_threshold,
            reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
            early_stopping=True,
        )
        if rejection_reason is not None:
            return lfs_early_stop, metrics_early_stop, rejection_reason
        lfs, metrics = compute_remaining_loadflows(
            runners=runners,
            strategy=strategy,
            base_case_ids=scoring_params.base_case_ids,
            loadflows_subset=lfs_early_stop,
            cases_subset=cases_subset,
            n_timestep_processes=scoring_params.n_timestep_processes,
        )
    else:
        lfs, _, metrics = compute_loadflow_and_metrics(
            runners=runners,
            strategy=strategy,
            base_case_ids=scoring_params.base_case_ids,
            n_timestep_processes=scoring_params.n_timestep_processes,
        )

    rejection_reason = evaluate_acceptance(
        metrics_split=metrics,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=scoring_params.reject_convergence_threshold,
        reject_overload_threshold=scoring_params.reject_overload_threshold,
        reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
        early_stopping=False,
    )
    return lfs, metrics, rejection_reason
