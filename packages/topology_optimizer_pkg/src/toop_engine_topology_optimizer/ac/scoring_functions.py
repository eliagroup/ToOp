# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Scoring functions for the AC optimizer - in this case this runs an N-1 and computes metrics for it"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import structlog
from beartype.typing import Collection, Optional
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics as compute_metrics_lfs
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner, AdditionalActionInfo
from toop_engine_interfaces.asset_topology import RealizedTopology
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    concatenate_loadflow_results_polars,
    subset_contingencies_polars,
)
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.evolution_functions import INF_FITNESS, get_contingency_indices_from_ids
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import EarlyStoppingStageResult, RunnerGroup, TopologyScoringResult
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, TopologyRejectionReason

logger = structlog.get_logger(__name__)


def get_early_stopping_contingency_ids(
    topology: ACOptimTopology,
    base_case_id: Optional[str] = None,
) -> Optional[list[str]]:
    """Extract the contingency ids for early stopping from a list of ACOptimTopology strategies.

    This function extracts the worst k contingency case ids from each topology's worst_k_contingency_cases
    attribute for each timestep. These ids are used to determine which contingencies to include in the N-1
    analysis for early stopping.

    Parameters
    ----------
    topology : ACOptimTopology
        An ACOptimTopology object containing a worst_k_contingency_cases attribute with contingency case ids.
    base_case_id : Optional[str]
        An optional base case id to include in the early stopping subset. If provided, this will be added to the
        list of contingency case ids for each timestep.

    Returns
    -------
    Optional[list[str]]
        A list of contingency case IDs, or None if any required metric is missing.
    """
    worst_k_contingency_cases = deepcopy(topology.worst_k_contingency_cases)
    if len(worst_k_contingency_cases) == 0:
        # TODO Does this make sense?
        # Shouldnt we just continue?
        logger.warning(
            f"No overload threshold or case ids found in the topologies worst contingency_cases: {worst_k_contingency_cases}"
        )
        return None
    if base_case_id is not None:
        worst_k_contingency_cases.append(base_case_id)
    return worst_k_contingency_cases


def update_runner_nminus1(
    runner: AbstractLoadflowRunner,
    nminus1_def: Nminus1Definition,
    case_ids_all_t: Collection[str],
) -> None:
    """Update the N-1 definitions in the runners to only include the worst k contingencies.

    This modifies the N-1 definitions in the runners to only include the contingencies at the given indices.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        The loadflow runner to update.
    nminus1_def : Nminus1Definition
        The original N-1 definition to use as a template.
    case_ids_all_t : Sequence[Collection[str]]
        A list of contingency ids for each runner, indicating which contingencies to keep in the N-1 definition.
        Each element should be an index to the contingencies in the original N-1 definition.
    """
    case_indices = get_contingency_indices_from_ids(case_ids_all_t, n_minus1_definition=nminus1_def)
    contingencies = np.array(nminus1_def.contingencies)[list(case_indices)]
    n1_def_copy = nminus1_def.model_copy()
    n1_def_copy.contingencies = contingencies.tolist()
    runner.store_nminus1_definition(n1_def_copy)


def compute_loadflow_and_metrics(
    runner: AbstractLoadflowRunner,
    topology: ACOptimTopology,
    base_case_id: Optional[str],
    cases_subset: Optional[Collection[str]] = None,
) -> tuple[LoadflowResultsPolars, Optional[AdditionalActionInfo], Metrics]:
    """Compute loadflow results and associated metrics for a given set of strategies.

    This function runs loadflow simulations for each provided strategy using the specified runners,
    then computes additional metrics based on the simulation results.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        The loadflow runner to use for simulations.
    topology : ACOptimTopology
        The topology to evaluate.
    base_case_id : Optional[str]
        The base case identifier for the topology. Can be None.
    cases_subset : Optional[Collection[str]]
        Subset of contingency cases to use for loadflow computation. If None, all available contingencies are used.

    Returns
    -------
    lfs : LoadflowResultsPolars
        The results of the loadflow simulations.
    additional_info : Optional[AdditionalActionInfo]
        Additional information for the actions taken in the topology.
    metrics : Metrics
        Computed metrics for the topology.
    """
    original_n_minus1_def = runner.get_nminus1_definition()
    if cases_subset is not None:
        update_runner_nminus1(runner, original_n_minus1_def, cases_subset)

    lfs, additional_info = compute_loadflow(
        actions=topology.actions,
        disconnections=topology.disconnections,
        pst_setpoints=topology.pst_setpoints,
        runner=runner,
    )
    metrics = compute_metrics_single_timestep(
        actions=topology.actions,
        disconnections=topology.disconnections,
        loadflow=lfs,
        additional_info=additional_info,
        base_case_id=base_case_id,
    )

    if cases_subset is not None:
        # Restore the original N-1 definitions in the runners
        runner.store_nminus1_definition(original_n_minus1_def)

    return lfs, additional_info, metrics


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
    metrics = {
        key: (0.0 if value is None else np.nan_to_num(value, nan=0, posinf=INF_FITNESS, neginf=-INF_FITNESS).item())
        for key, value in metrics.items()
    }
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
    actions: list[int],
    disconnections: list[int],
    pst_setpoints: Optional[list[int]],
    runner: AbstractLoadflowRunner,
) -> tuple[LoadflowResultsPolars, Optional[AdditionalActionInfo]]:
    """Compute the loadflow for a given strategy

    Parameters
    ----------
    actions : list[int]
        The reconfiguration actions for the timestep
    disconnections : list[int]
        The disconnections for the timestep
    pst_setpoints : Optional[list[int]]
        The PST setpoints for the topology, or None if PST taps are not part of the topology.
    runner : AbstractLoadflowRunner
        The loadflow runner to use

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for all timesteps in the strategy
    list[Optional[AdditionalActionInfo]]
        Additional information about the actions taken, such as switching distance or other metrics. The length of
        the list is n_timesteps.
    """
    loadflow = runner.run_ac_loadflow(actions, disconnections, pst_setpoints)
    additional_information = runner.get_last_action_info()

    return loadflow, additional_information


def evaluate_acceptance(
    metrics_split: Metrics,
    metrics_unsplit: Metrics,
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
            reject_convergence_threshold * unsplit.extra_scores.get("non_converging_loadflows", 0)
        overload_energy_n_1: the overload energy should be less than or equal to
            reject_overload_threshold * unsplit.extra_scores.get("overload_energy_n_1", 0)
        critical_branch_count_n_1: the number of critical branches should be less than or equal
            to reject_critical_branch_threshold * unsplit.extra_scores.get("critical_branch_count_n_1", 0)
        TODO: Check Voltage Jumps between N0 and N1

    Parameters
    ----------
    metrics_split : Metrics
        The metrics for the split case.
    metrics_unsplit : Metrics
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
        [
            metrics_unsplit.extra_scores.get("non_converging_loadflows", 0)
            - metrics_unsplit.extra_scores.get("disconnected_branches", 0)
        ]
    )
    n_non_converged_split = np.array(
        [
            metrics_split.extra_scores.get("non_converging_loadflows", 0)
            - metrics_split.extra_scores.get("disconnected_branches", 0)
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

    unsplit_overload = np.array([metrics_unsplit.extra_scores.get("overload_energy_n_1", 0)])
    split_overload = np.array([metrics_split.extra_scores.get("overload_energy_n_1", 99999)])
    overload_improvement = np.all(split_overload <= unsplit_overload * reject_overload_threshold)
    if not overload_improvement:
        return TopologyRejectionReason(
            criterion="overload-energy",
            value_after=float(split_overload.sum()),
            value_before=float(unsplit_overload.sum()),
            threshold=reject_overload_threshold,
            early_stopping=early_stopping,
        )

    unsplit_critical_branches = np.array([metrics_unsplit.extra_scores.get("critical_branch_count_n_1", 999)], dtype=float)
    split_critical_branches = np.array([metrics_split.extra_scores.get("critical_branch_count_n_1", 0)], dtype=float)

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
    runner: AbstractLoadflowRunner,
    topology: ACOptimTopology,
    base_case_id: Optional[str],
    loadflows_subset: LoadflowResultsPolars,
    cases_subset: list[str],
) -> tuple[LoadflowResultsPolars, Metrics]:
    """Compute the loadflows for the remaining contingencies that were not included in the early stopping subset.

    This function is called after the early stopping loadflows have been computed and accepted. It computes the loadflows
    for the remaining contingencies that were not included in the early stopping subset, and then computes the metrics for
    the full set of loadflows.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        The loadflow runners to use, length n_timesteps.
    topology : ACOptimTopology
        The topology to score, length n_timesteps
    base_case_id : Optional[str]
        The base case id for the loadflow runners, used to separately compute the N-0 flows.
    loadflows_subset : LoadflowResultsPolars
        The loadflow results for the early stopping subset, used to avoid recomputing these loadflows.
    cases_subset : list[str]
        The contingency case ids that were included in the early stopping subset for each timestep. This could be extracted
        from the loadflows_subset but as it is available it is faster to pass it in.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for all contingencies, including those from the early stopping subset.
    Metrics
        The metrics for the full set of loadflows.
    """
    original_n_minus1_def = runner.get_nminus1_definition()
    all_cases = [contingency.id for contingency in original_n_minus1_def.contingencies]

    # Remove the already computed contingencies so we do not re-compute them
    remaining_cases = set(all_cases) - set(cases_subset)

    # Update the N-1 definitions in the runners to now include only the non-critical contingencies.
    logger.info(f"Running N-1 analysis with {len(remaining_cases)} non-critical contingencies per timestep.")
    update_runner_nminus1(runner, original_n_minus1_def, remaining_cases)

    lfs_remaining, additional_info_remaining = compute_loadflow(
        actions=topology.actions,
        disconnections=topology.disconnections,
        pst_setpoints=topology.pst_setpoints,
        runner=runner,
    )

    lfs = concatenate_loadflow_results_polars([loadflows_subset, lfs_remaining])

    # We can pass the additional info from either critical or non critical contingencies as they are the same
    metrics = compute_metrics_single_timestep(
        actions=topology.actions,
        disconnections=topology.disconnections,
        loadflow=lfs,
        additional_info=additional_info_remaining,
        base_case_id=base_case_id,
    )

    # Restore the original N-1 definitions in the runners
    runner.store_nminus1_definition(original_n_minus1_def)

    return lfs, metrics


@dataclass
class ACScoringParameters:
    """Parameters for ac scoring

    This is a subset of all ac parameters and grouped to shorten the signature of the
    scoring and acceptance functions.
    """

    # --- Thresholds for acceptance criteria --- #
    reject_convergence_threshold: float
    """The rejection threshold for the convergence rate, i.e. the split case must have at most the same amount of
    non converging loadflows as the unsplit case or it will be rejected."""

    reject_overload_threshold: float
    """The rejection threshold for the overload energy improvement, i.e. the split case must have at least 5% lower
    overload energy than the unsplit case or it will be rejected."""

    reject_critical_branch_threshold: float
    """The rejection threshold for the critical branches increase, i.e. the split case must have less than 10% more
    critical branches than the unsplit case or it will be rejected."""

    # --- Parameters for early stopping during N-1 analysis --- #
    base_case_id: Optional[str]
    """The base case id for the loadflow runner (used to separately compute the N-0 flows)."""

    early_stop_validation: bool
    """Whether to enable early stopping during the optimization process."""


def _error_result_for_topology(description: str, early_stopping: bool) -> TopologyScoringResult:
    """Create a rejected result for a failed topology evaluation.

    Parameters
    ----------
    description : str
        A description of the error that occurred.
    early_stopping : bool
        Whether the error occurred during the early stopping stage,
        used to set the early_stopping field

    Returns
    -------
    TopologyScoringResult
        A TopologyScoringResult with a rejection reason based on the exception.
        The exception that was raised during the evaluation of the topology.
    """
    if early_stopping:
        return EarlyStoppingStageResult(
            loadflow_results=None,
            metrics=Metrics(fitness=INF_FITNESS, extra_scores={}),
            rejection_reason=TopologyRejectionReason(
                criterion="error",
                description=description,
                value_after=1.0,
                value_before=0.0,
                early_stopping=early_stopping,
            ),
            cases_subset=[],
        )
    return TopologyScoringResult(
        loadflow_results=None,
        metrics=Metrics(fitness=INF_FITNESS, extra_scores={}),
        rejection_reason=TopologyRejectionReason(
            criterion="error",
            description=description,
            value_after=1.0,
            value_before=0.0,
            early_stopping=False,
        ),
    )


def score_strategy_worst_k(
    topology: ACOptimTopology,
    runner: AbstractLoadflowRunner,
    loadflow_results_unsplit: LoadflowResultsPolars,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
) -> EarlyStoppingStageResult:
    """Evaluate only the worst-k stage for a single strategy.

    Parameters
    ----------
    topology : ACOptimTopology
        The topology to evaluate, length n_timesteps
    runner : AbstractLoadflowRunner
        The loadflow runner to use for the evaluation of the strategy
    loadflow_results_unsplit : LoadflowResultsPolars
        The loadflow results for the unsplit case, used for comparison in the acceptance evaluation.
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.

    Returns
    -------
    EarlyStoppingStageResult
        The result of the worst-k stage evaluation, including loadflow results, metrics
        and rejection reason if rejected. If early stopping is enabled and the strategy
        is rejected based on the worst-k contingencies, the early_stopping flag will be set.
    """
    if scoring_params.early_stop_validation:
        cases_subset = get_early_stopping_contingency_ids(topology, base_case_id=scoring_params.base_case_id)
        assert cases_subset is not None, (
            "Early stopping enabled but no contingency case ids found for early stopping."
            "This might happen when the DC optimizer pushes topologies without worst_k entries."
        )
        lfs_early_stop, additional_info, metrics_early_stop = compute_loadflow_and_metrics(
            runner=runner,
            topology=topology,
            base_case_id=scoring_params.base_case_id,
            cases_subset=cases_subset,
        )
        lfs_early_stop_unsplit = subset_contingencies_polars(loadflow_results_unsplit, cases_subset)
        metrics_early_stop_unsplit = compute_metrics_single_timestep(
            actions=topology.actions,
            disconnections=topology.disconnections,
            loadflow=lfs_early_stop_unsplit,
            additional_info=additional_info,
            base_case_id=scoring_params.base_case_id,
        )
        rejection_reason = evaluate_acceptance(
            metrics_split=metrics_early_stop,
            metrics_unsplit=metrics_early_stop_unsplit,
            reject_convergence_threshold=scoring_params.reject_convergence_threshold,
            reject_overload_threshold=scoring_params.reject_overload_threshold,
            reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
            early_stopping=True,
        )
        return EarlyStoppingStageResult(
            loadflow_results=lfs_early_stop,
            metrics=metrics_early_stop,
            rejection_reason=rejection_reason,
            cases_subset=cases_subset,
        )

    lfs, _, metrics = compute_loadflow_and_metrics(
        runner=runner,
        topology=topology,
        base_case_id=scoring_params.base_case_id,
    )
    rejection_reason = evaluate_acceptance(
        metrics_split=metrics,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=scoring_params.reject_convergence_threshold,
        reject_overload_threshold=scoring_params.reject_overload_threshold,
        reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
        early_stopping=False,
    )
    return EarlyStoppingStageResult(
        loadflow_results=lfs, metrics=metrics, rejection_reason=rejection_reason, cases_subset=None
    )


def score_strategy_worst_k_batch(
    topologies: list[ACOptimTopology],
    worst_k_runner_groups: RunnerGroup,
    loadflow_results_unsplit: LoadflowResultsPolars,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
) -> list[EarlyStoppingStageResult]:
    """Evaluate the worst-k stage for a batch of strategies.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to evaluate, length n_strategies.
    worst_k_runner_groups : RunnerGroup
        The loadflow runner groups to use for the evaluation of the strategies, length n_strategies.
    loadflow_results_unsplit : LoadflowResultsPolars
        The loadflow results for the unsplit case, used for comparison in the acceptance evaluation.
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.

    Returns
    -------
    list[EarlyStoppingStageResult]
        The results of the worst-k stage evaluation for each strategy, including loadflow results, metrics
        and rejection reason if rejected. If early stopping is enabled and a strategy is rejected based
        on the worst-k contingencies, the early_stopping flag will be set in the rejection reason.
    """
    if not topologies:
        return []
    if len(topologies) > len(worst_k_runner_groups):
        raise ValueError("Not enough worst-k runner groups configured for the requested strategy batch")

    worst_stage_results: list[Optional[EarlyStoppingStageResult]] = [
        _error_result_for_topology("Initial error", early_stopping=True)
    ] * len(topologies)
    with ThreadPoolExecutor(max_workers=len(topologies)) as executor:
        future_to_index = {
            executor.submit(
                score_strategy_worst_k,
                topology,
                runner,
                loadflow_results_unsplit,
                metrics_unsplit,
                scoring_params,
            ): index
            for index, (topology, runner) in enumerate(
                zip(topologies, worst_k_runner_groups[: len(topologies)], strict=True)
            )
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                worst_stage_results[index] = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Worst-k stage failed")
                final_result = _error_result_for_topology(str(exc), early_stopping=True)
                worst_stage_results[index] = EarlyStoppingStageResult(
                    loadflow_results=loadflow_results_unsplit,
                    metrics=final_result.metrics,
                    rejection_reason=final_result.rejection_reason,
                    cases_subset=None,
                )

    return [result for result in worst_stage_results if result is not None]


def score_topology_remaining(
    topology: ACOptimTopology,
    runner: AbstractLoadflowRunner,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
    early_stage_result: EarlyStoppingStageResult,
) -> TopologyScoringResult:
    """Evaluate the remaining contingencies for a surviving strategy.

    This function is called for strategies that survived the worst-k stage
    (i.e. were not rejected based on the worst-k contingencies).
    It computes the loadflows for the remaining contingencies that were not included in the worst-k stage,
    and then computes the metrics for the full set of loadflows.
    Finally, it evaluates the acceptance of the full set of loadflows compared to the unsplit case.

    Parameters
    ----------
    topology : ACOptimTopology
        The topology to evaluate, length n_timesteps
    runner : AbstractLoadflowRunner
        The loadflow runner to use for the evaluation of the strategy
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.
    early_stage_result : EarlyStoppingStageResult
        The result from the worst-k stage evaluation, including loadflow results, metrics and cases
        subset used for early stopping. This is used to compute the remaining loadflows and metrics
        without recomputing the early stopping subset.

    Returns
    -------
    TopologyScoringResult
        The result of the full evaluation, including loadflow results, metrics
        and rejection reason if rejected.
    """
    if scoring_params.early_stop_validation:
        assert early_stage_result.cases_subset is not None
        lfs, metrics = compute_remaining_loadflows(
            runner=runner,
            topology=topology,
            base_case_id=scoring_params.base_case_id,
            loadflows_subset=early_stage_result.loadflow_results,
            cases_subset=early_stage_result.cases_subset,
        )
    else:
        lfs = early_stage_result.loadflow_results
        metrics = early_stage_result.metrics

    rejection_reason = evaluate_acceptance(
        metrics_split=metrics,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=scoring_params.reject_convergence_threshold,
        reject_overload_threshold=scoring_params.reject_overload_threshold,
        reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
        early_stopping=False,
    )
    return TopologyScoringResult(loadflow_results=lfs, metrics=metrics, rejection_reason=rejection_reason)


def score_strategy_full(
    topology: ACOptimTopology,
    runner: AbstractLoadflowRunner,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
) -> TopologyScoringResult:
    """Evaluate a strategy on the full set of contingencies in one pass.

    Parameters
    ----------
    topology : ACOptimTopology
        The topology to evaluate, length n_timesteps
    runner : AbstractLoadflowRunner
        The loadflow runner to use for the evaluation of the strategy
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.

    Returns
    -------
    TopologyScoringResult
        The result of the full evaluation, including loadflow results, metrics and rejection reason if rejected.
    """
    lfs, _, metrics = compute_loadflow_and_metrics(
        runner=runner,
        topology=topology,
        base_case_id=scoring_params.base_case_id,
    )
    rejection_reason = evaluate_acceptance(
        metrics_split=metrics,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=scoring_params.reject_convergence_threshold,
        reject_overload_threshold=scoring_params.reject_overload_threshold,
        reject_critical_branch_threshold=scoring_params.reject_critical_branch_threshold,
        early_stopping=False,
    )
    return TopologyScoringResult(loadflow_results=lfs, metrics=metrics, rejection_reason=rejection_reason)


def score_strategy_full_batch(
    topologies: list[ACOptimTopology],
    runner_groups: RunnerGroup,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
) -> list[TopologyScoringResult]:
    """Evaluate a batch of topologies on the full set of contingencies.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to evaluate, length n_strategies.
    runner_groups : RunnerGroup
        The loadflow runner groups to use for the evaluation of the strategies, length n_strategies.
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.

    Returns
    -------
    list[TopologyScoringResult]
        The results of the full evaluation for each strategy, including loadflow results,
        metrics and rejection reason if rejected.
    """
    if not topologies:
        return []
    if len(runner_groups) == 0:
        raise ValueError("At least one runner group is required for full-contingency evaluation")

    results: list[Optional[TopologyScoringResult]] = [
        _error_result_for_topology("Initial error", early_stopping=False)
    ] * len(topologies)
    max_parallel = min(len(topologies), len(runner_groups))

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_assignment: dict = {}
        next_topology_index = 0

        def submit(index: int, runner_index: int) -> None:
            topology_for_scoring = ACOptimTopology(**topologies[index].model_dump())
            future = executor.submit(
                score_strategy_full,
                topology_for_scoring,
                runner_groups[runner_index],
                metrics_unsplit,
                scoring_params,
            )
            future_to_assignment[future] = (index, runner_index)

        for runner_index in range(max_parallel):
            submit(next_topology_index, runner_index)
            next_topology_index += 1

        while future_to_assignment:
            future = next(as_completed(tuple(future_to_assignment)))
            index, runner_index = future_to_assignment.pop(future)
            try:
                results[index] = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Full-contingency stage failed")
                results[index] = _error_result_for_topology(str(exc), early_stopping=False)

            if next_topology_index < len(topologies):
                submit(next_topology_index, runner_index)
                next_topology_index += 1

    return [result for result in results if result is not None]


def score_remaining_contingency_batch(
    topologies: list[ACOptimTopology],
    early_stage_results: list[EarlyStoppingStageResult],
    runner_group: RunnerGroup,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
) -> list[TopologyScoringResult]:
    """Evaluate the remaining contingencies for a batch of surviving topologies.

    This function is called for strategies that survived the worst-k stage
    (i.e. were not rejected based on the worst-k contingencies).
    It computes the loadflows for the remaining contingencies that were not included in the worst-k stage,
    and then computes the metrics for the full set of loadflows.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to evaluate, length n_strategies.
    early_stage_results : list[EarlyStoppingStageResult]
        The results from the worst-k stage evaluation for each topology, including loadflow results, metrics
        and cases subset used for early stopping. This is used to compute the remaining loadflows and metrics
        without recomputing the early stopping subset.
    runner_group : RunnerGroup
        The loadflow runner group to use for the evaluation of the strategies, length n_strategies.
    metrics_unsplit : Metrics
        The metrics for the unsplit case, used for comparison in the acceptance evaluation.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.

    Returns
    -------
    list[TopologyScoringResult]
        The results of the full evaluation for each strategy, including loadflow results,
        metrics and rejection reason if rejected.
    """
    if not topologies:
        return []
    if len(runner_group) == 0:
        raise ValueError("At least one remaining-stage runner group is required")
    if len(topologies) != len(early_stage_results):
        raise ValueError("Topologies and early-stage results must have the same length")

    results: list[Optional[TopologyScoringResult]] = [
        _error_result_for_topology("Initial error", early_stopping=False)
    ] * len(topologies)
    max_parallel = min(len(topologies), len(runner_group))

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_assignment: dict = {}
        next_topology_index = 0

        def submit(index: int, runner_index: int) -> None:
            future = executor.submit(
                score_topology_remaining,
                topologies[index],
                runner_group[runner_index],
                metrics_unsplit,
                scoring_params,
                early_stage_results[index],
            )
            future_to_assignment[future] = (index, runner_index)

        for runner_index in range(max_parallel):
            submit(next_topology_index, runner_index)
            next_topology_index += 1

        while future_to_assignment:
            future = next(as_completed(tuple(future_to_assignment)))
            index, runner_index = future_to_assignment.pop(future)
            try:
                results[index] = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Remaining-contingency stage failed")
                results[index] = _error_result_for_topology(str(exc), early_stopping=False)

            if next_topology_index < len(topologies):
                submit(next_topology_index, runner_index)
                next_topology_index += 1

    return [result for result in results if result is not None]


def score_topology_batch(
    topologies: list[ACOptimTopology],
    runner_group: RunnerGroup,
    metrics_unsplit: Metrics,
    scoring_params: ACScoringParameters,
    early_stage_results: Optional[list[EarlyStoppingStageResult]] = None,
) -> list[TopologyScoringResult]:
    """Score a batch of topologies in two stages.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to be scored.
    runner_group : RunnerGroup
        The group of runners to use for scoring.
    metrics_unsplit : Metrics
        The metrics to be used for scoring.
    scoring_params : ACScoringParameters
        The parameters for scoring, including thresholds for acceptance and early stopping settings.
    early_stage_results : Optional[list[EarlyStoppingStageResult]], optional
        The results from the early stage, by default None.

    Returns
    -------
    list[TopologyScoringResult]
        The results of the full evaluation for each strategy, including loadflow results,
        metrics and rejection reason if rejected.
    """
    if early_stage_results is None:
        return score_strategy_full_batch(
            topologies=topologies,
            runner_groups=runner_group,
            metrics_unsplit=metrics_unsplit,
            scoring_params=scoring_params,
        )
    if len(topologies) != len(early_stage_results):
        raise ValueError("Topologies and early-stage results must have the same length")
    results = score_remaining_contingency_batch(
        topologies=topologies,
        early_stage_results=early_stage_results,
        runner_group=runner_group,
        metrics_unsplit=metrics_unsplit,
        scoring_params=scoring_params,
    )

    return results
