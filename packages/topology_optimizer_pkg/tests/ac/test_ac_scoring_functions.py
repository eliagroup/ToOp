# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.stored_action_set import load_action_set, random_actions
from toop_engine_topology_optimizer.ac.scoring_functions import (
    ACScoringParameters,
    compute_loadflow,
    compute_loadflow_and_metrics,
    compute_metrics_single_timestep,
    compute_remaining_loadflows,
    evaluate_acceptance,
    extract_switching_distance,
    score_remaining_contingency_batch,
    score_strategy_worst_k_batch,
    score_topology_batch,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import EarlyStoppingStageResult, TopologyScoringResult
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics


def test_score_strategy_worst_k_batch_parallelizes(monkeypatch: pytest.MonkeyPatch) -> None:
    topology_a = ACOptimTopology(
        actions=[1],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"a",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )
    topology_b = ACOptimTopology(
        actions=[2],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"b",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )

    runner_ids = []

    def fake_worst_k(topology, runner, loadflow_results_unsplit, metrics_unsplit, scoring_params):
        del topology, loadflow_results_unsplit, metrics_unsplit, scoring_params
        runner_ids.append(id(runner))
        return EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=0.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c1"],
        )

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.scoring_functions.score_strategy_worst_k", fake_worst_k)

    scoring_params = ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        base_case_id=None,
        early_stop_validation=True,
    )
    worst_k_runner_groups = [Mock(spec=AbstractLoadflowRunner), Mock(spec=AbstractLoadflowRunner)]

    results = score_strategy_worst_k_batch(
        topologies=[topology_a, topology_b],
        worst_k_runner_groups=worst_k_runner_groups,
        loadflow_results_unsplit=Mock(spec=LoadflowResultsPolars),
        metrics_unsplit=Metrics(fitness=0.0, extra_scores={}),
        scoring_params=scoring_params,
    )

    assert len(results) == 2
    assert len(set(runner_ids)) == 2


def test_score_strategy_remaining_batch_chunks_survivors(monkeypatch: pytest.MonkeyPatch) -> None:
    topology_a = ACOptimTopology(
        actions=[1],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"a",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )
    topology_b = ACOptimTopology(
        actions=[2],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"b",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )
    topology_c = ACOptimTopology(
        actions=[3],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"c",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )

    call_sequence = []

    def fake_remaining(topology, runner, metrics_unsplit, scoring_params, early_stage_result):
        del metrics_unsplit, scoring_params, early_stage_result
        call_sequence.append(id(runner))
        return TopologyScoringResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=float(topology.actions[0]), extra_scores={}),
            rejection_reason=None,
        )

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.scoring_functions.score_topology_remaining", fake_remaining)

    early_stage_results = [
        EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=0.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c1"],
        )
        for _ in range(3)
    ]
    shared_runner = Mock(spec=AbstractLoadflowRunner)
    scoring_params = ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        base_case_id=None,
        early_stop_validation=True,
    )
    remaining_runner_groups = [shared_runner, Mock(spec=AbstractLoadflowRunner)]

    results = score_remaining_contingency_batch(
        topologies=[topology_a, topology_b, topology_c],
        early_stage_results=early_stage_results,
        runner_group=remaining_runner_groups,
        metrics_unsplit=Metrics(fitness=0.0, extra_scores={}),
        scoring_params=scoring_params,
    )

    assert len(results) == 3
    assert call_sequence == [id(shared_runner), id(remaining_runner_groups[1]), id(shared_runner)]


def test_score_strategy_batch_without_early_results_uses_full_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    topology_a = ACOptimTopology(
        actions=[1],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"a",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )
    topology_b = ACOptimTopology(
        actions=[2],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=b"b",
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )

    full_eval_calls = []

    def fake_full_batch(topologies, runner_groups, metrics_unsplit, scoring_params):
        del runner_groups, metrics_unsplit, scoring_params
        full_eval_calls.append(len(topologies))
        return [
            TopologyScoringResult(
                loadflow_results=Mock(spec=LoadflowResultsPolars),
                metrics=Metrics(fitness=float(topology.actions[0]), extra_scores={}),
                rejection_reason=None,
            )
            for topology in topologies
        ]

    def fail_worst_k_batch(*args, **kwargs):
        raise AssertionError("worst-k batch should not be used when early-stage results are omitted")

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.scoring_functions.score_strategy_full_batch", fake_full_batch)
    monkeypatch.setattr(
        "toop_engine_topology_optimizer.ac.scoring_functions.score_strategy_worst_k_batch", fail_worst_k_batch
    )

    scoring_params = ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        base_case_id=None,
        early_stop_validation=True,
    )

    results = score_topology_batch(
        topologies=[topology_a, topology_b],
        runner_group=[Mock(spec=AbstractLoadflowRunner), Mock(spec=AbstractLoadflowRunner)],
        metrics_unsplit=Metrics(fitness=0.0, extra_scores={}),
        scoring_params=scoring_params,
    )

    assert full_eval_calls == [2]
    assert [result.metrics.fitness for result in results] == [1.0, 2.0]


def test_compute_loadflow(grid_folder: Path) -> None:
    action_set = load_action_set(
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"],
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_diff_path"],
    )
    nminus1_definition = load_nminus1_definition(
        grid_folder / "case14" / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )

    runner = PandapowerRunner()
    runner.load_base_grid(grid_folder / "case14" / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    ref_loadflow = runner.run_ac_loadflow([], [])

    res, info = compute_loadflow(
        actions=[],
        disconnections=[],
        pst_setpoints=None,
        runner=runner,
    )

    assert res == ref_loadflow
    assert np.isclose(res.branch_results.collect()["i"].sum(), ref_loadflow.branch_results.collect()["i"].sum())

    metrics = compute_metrics_single_timestep(
        actions=[],
        disconnections=[],
        loadflow=res,
        additional_info=info,
    )

    assert metrics is not None
    assert metrics.extra_scores["non_converging_loadflows"] == 0
    assert metrics.extra_scores["disconnected_branches"] == 0
    assert metrics.extra_scores["split_subs"] == 0


def test_scoring_functions_split(grid_folder: Path) -> None:
    action_set = load_action_set(
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"],
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_diff_path"],
    )
    nminus1_definition = load_nminus1_definition(
        grid_folder / "case14" / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )

    actions = random_actions(
        action_set=action_set,
        rng=np.random.default_rng(42),
        n_split_subs=2,  # Split two substations
    )

    runner = PandapowerRunner()
    runner.load_base_grid(grid_folder / "case14" / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    ref_loadflow = runner.run_ac_loadflow(actions, [])
    info = runner.get_last_action_info()

    assert info is not None
    assert extract_switching_distance(info) > 0

    metrics = compute_metrics_single_timestep(
        actions=actions,
        disconnections=[],
        loadflow=ref_loadflow,
        additional_info=info,
    )

    assert "switching_distance" in metrics.extra_scores


def test_evaluate_acceptance_identical_metrics():
    metrics_unsplit = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 5,
            "overload_energy_n_1": 50.0,
            "critical_branch_count_n_1": 10,
        },
    )
    # Check identical values
    metrics_split = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 5,
            "overload_energy_n_1": 50.0,
            "critical_branch_count_n_1": 10,
        },
    )
    # Accepted if thresholds == 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert reason is None, "Results rejected although they are the same as before and thresholds is exactly 1."
    # Not accepted if any thresholds < 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert reason is not None, "Results rejected although they are the same as before and thresholds is below 1."
    assert reason.criterion == "convergence"

    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert reason is not None, "Results rejected although they are just as good and convergence thresholds below 1."
    assert reason.criterion == "convergence"
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=1.0,
    )
    assert reason is not None, "Results rejected although they are just as good and overload thresholds below 1."
    assert reason.criterion == "overload-energy"

    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=0.9,
    )
    assert reason is not None, "Results rejected although they are just as good and crit branch thresholds below 1."
    assert reason.criterion == "critical-branch-count"

    # Accepted if thresholds > 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert reason is None, "Results rejected although they are just as good and thresholds above 1."


def test_evaluate_acceptance_improved_metrics():
    metrics_unsplit = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 10,
            "overload_energy_n_1": 100.0,
            "critical_branch_count_n_1": 10,
        },
    )
    # Check identical values
    metrics_split = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 9,
            "overload_energy_n_1": 90,
            "critical_branch_count_n_1": 9,
        },
    )
    # Accepted if thresholds == 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert reason is None, "Results rejected although they are the same as before and thresholds is exactly 1."
    # Accepted if all thresholds=0.9.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert reason is None, "Results not accepted although they improved by exactly 10 percent and thresholds is 0.9."

    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.8,
        reject_overload_threshold=0.8,
        reject_critical_branch_threshold=0.8,
    )
    assert reason is not None, "Results accepted although they only improved by exactly 10 percent and thresholds is 0.8."
    assert reason.criterion == "convergence"

    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.8,
        reject_overload_threshold=0.8,
        reject_critical_branch_threshold=0.8,
    )
    assert reason is not None, "Results accepted although they only improved by exactly 10 percent and thresholds is 0.8."
    assert reason.criterion == "convergence"

    # Accepted if thresholds > 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert reason is None, "Results rejected although they are just as good and thresholds above 1."


def test_evaluate_acceptance_worse_metrics():
    metrics_unsplit = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 10,
            "overload_energy_n_1": 100.0,
            "critical_branch_count_n_1": 10,
        },
    )
    # Check identical values
    metrics_split = Metrics(
        fitness=-1.0,
        extra_scores={
            "non_converging_loadflows": 11,
            "overload_energy_n_1": 110,
            "critical_branch_count_n_1": 11,
        },
    )
    # Dont Accept if thresholds == 1.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert reason is not None, "Results accepted although they are worse as before and thresholds is exactly 1."
    assert reason.criterion == "convergence"

    # Not Accepted if all thresholds=0.9.
    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert reason is not None, "Results accepted although they got worse by exactly 10 percent and thresholds is 0.9."
    assert reason.criterion == "convergence"

    reason = evaluate_acceptance(
        metrics_split=metrics_split,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert reason is None, "Results not accepted although they only got worse by exactly 10 percent and thresholds is 1.1."


def test_compute_remaining_loadflows(grid_folder: Path) -> None:
    """Test the compute_remaining_loadflows function.

    This test verifies:
    1. The function correctly computes only the remaining contingencies not in the subset
    2. The function concatenates subset and remaining loadflows correctly
    3. The metrics are computed on the full combined set of loadflows
    4. All contingencies are present in the final result
    """
    action_set = load_action_set(
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"],
        grid_folder / "case14" / PREPROCESSING_PATHS["action_set_diff_path"],
    )
    nminus1_definition = load_nminus1_definition(
        grid_folder / "case14" / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )

    runner = PandapowerRunner()
    runner.load_base_grid(grid_folder / "case14" / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    # Get all contingency IDs
    n_1_def = runner.get_nminus1_definition()
    all_case_ids = [cont.id for cont in n_1_def.contingencies]

    # Select subset of contingencies for the "early stopping" phase
    rng = np.random.default_rng(42)
    n_subset = min(5, len(all_case_ids))
    subset_case_ids = rng.choice(all_case_ids, size=n_subset, replace=False).tolist()

    # Create a strategy with some splits
    actions = random_actions(
        action_set=action_set,
        rng=rng,
        n_split_subs=2,
    )

    topology = ACOptimTopology(
        actions=actions,
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=bytes.fromhex("abcd1234"),
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={"overload_energy_n_1": 50.0},
        worst_k_contingency_cases=subset_case_ids,
    )

    # First compute the subset (simulating early stopping phase)
    lfs_subset, _, _ = compute_loadflow_and_metrics(
        runner=runner,
        topology=topology,
        base_case_id=None,
        cases_subset=subset_case_ids,
    )

    # Verify that subset only contains the specified contingencies
    subset_contingencies = lfs_subset.branch_results.select("contingency").unique().collect()["contingency"].to_list()
    # Remove None/basecase if present
    subset_contingencies = [c for c in subset_contingencies if c is not None]

    assert len(subset_contingencies) == n_subset, (
        f"Subset should contain exactly {n_subset} contingencies, but has {len(subset_contingencies)}"
    )
    for case_id in subset_case_ids:
        assert case_id in subset_contingencies, f"Subset should contain case {case_id}"

    # Now compute the remaining loadflows
    lfs_complete, metrics_complete = compute_remaining_loadflows(
        runner=runner,
        topology=topology,
        base_case_id=None,
        loadflows_subset=lfs_subset,
        cases_subset=subset_case_ids,
    )

    # Verify that the complete result contains all contingencies
    complete_contingencies = lfs_complete.branch_results.select("contingency").unique().collect()["contingency"].to_list()
    # Remove None/basecase if present
    complete_contingencies = [c for c in complete_contingencies if c is not None]

    # Should have all original contingencies
    assert len(complete_contingencies) >= len(all_case_ids), (
        f"Complete result should contain at least {len(all_case_ids)} contingencies, but has {len(complete_contingencies)}"
    )

    # Verify all original contingencies are present
    for case_id in all_case_ids:
        assert case_id in complete_contingencies, f"Complete result should contain case {case_id}"

    # Verify metrics were computed
    assert metrics_complete.fitness is not None, "Metrics should have fitness"
    assert "overload_energy_n_1" in metrics_complete.extra_scores, "Metrics should contain overload_energy_n_1"

    # Verify that the metrics reflect the full set of contingencies, not just the subset
    # The complete metrics should account for all contingencies
    assert "critical_branch_count_n_1" in metrics_complete.extra_scores
    assert "non_converging_loadflows" in metrics_complete.extra_scores
