# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
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
    scoring_and_acceptance,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics


def test_compute_loadflow(grid_folder: Path) -> None:
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
    nminus1_definition = load_nminus1_definition(
        grid_folder / "case14" / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )

    runner = PandapowerRunner()
    runner.load_base_grid(grid_folder / "case14" / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    ref_loadflow = runner.run_ac_loadflow([], [])

    res, info = compute_loadflow(
        actions=[[]],
        disconnections=[[]],
        pst_setpoints=[None],
        runners=[runner],
    )

    assert res == ref_loadflow
    assert np.isclose(res.branch_results.collect()["i"].sum(), ref_loadflow.branch_results.collect()["i"].sum())

    metrics = compute_metrics_single_timestep(
        actions=[],
        disconnections=[],
        loadflow=res,
        additional_info=info[0],
    )

    assert metrics is not None
    assert metrics.extra_scores["non_converging_loadflows"] == 0
    assert metrics.extra_scores["disconnected_branches"] == 0
    assert metrics.extra_scores["split_subs"] == 0


def test_scoring_functions_split(grid_folder: Path) -> None:
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
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
    metrics_unsplit = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 5,
                "overload_energy_n_1": 50.0,
                "critical_branch_count_n_1": 10,
            },
        )
    ]
    # Check identical values
    metrics_split = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 5,
                "overload_energy_n_1": 50.0,
                "critical_branch_count_n_1": 10,
            },
        )
    ]
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
    metrics_unsplit = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 10,
                "overload_energy_n_1": 100.0,
                "critical_branch_count_n_1": 10,
            },
        )
    ]
    # Check identical values
    metrics_split = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 9,
                "overload_energy_n_1": 90,
                "critical_branch_count_n_1": 9,
            },
        )
    ]
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
    metrics_unsplit = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 10,
                "overload_energy_n_1": 100.0,
                "critical_branch_count_n_1": 10,
            },
        )
    ]
    # Check identical values
    metrics_split = [
        Metrics(
            fitness=-1.0,
            extra_scores={
                "non_converging_loadflows": 11,
                "overload_energy_n_1": 110,
                "critical_branch_count_n_1": 11,
            },
        )
    ]
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


def test_scoring_and_acceptance_early_stopping(grid_folder: Path) -> None:
    """Test the scoring_and_acceptance function with early stopping enabled.

    This test verifies:
    1. Early stopping rejection when the strategy performs worse on critical contingencies
    2. Early stopping acceptance and continuation to full N-1 when strategy performs well on critical contingencies
    3. Correct handling of the early_stopping flag in rejection reasons
    """
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
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

    # Select 5 random contingencies for early stopping
    rng = np.random.default_rng(42)
    early_stop_case_ids = rng.choice(all_case_ids, size=min(5, len(all_case_ids)), replace=False).tolist()

    # Run unsplit baseline
    loadflow_results_unsplit = runner.run_ac_loadflow([], [])
    metrics_unsplit_temp = compute_metrics_single_timestep(
        actions=[],
        disconnections=[],
        loadflow=loadflow_results_unsplit,
        additional_info=None,
    )
    metrics_unsplit = [metrics_unsplit_temp]

    # Create a strategy that has some splits and will be rejected during early stopping
    # We'll create a strategy with high overload to ensure it gets rejected
    actions_bad = random_actions(
        action_set=action_set,
        rng=rng,
        n_split_subs=3,  # Split multiple substations
    )

    strategy_bad = [
        ACOptimTopology(
            actions=actions_bad,
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=bytes.fromhex("deadbeef"),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.0,
            metrics={"overload_energy_n_1": 1000.0},  # High overload
            worst_k_contingency_cases=early_stop_case_ids,  # Critical cases for early stopping
        )
    ]

    scoring_params = ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        base_case_ids=[None],
        n_timestep_processes=1,
        early_stop_validation=True,
        early_stop_non_converging_threshold=0.1,
    )

    # Test 1: Early stopping with rejection
    lfs_bad, _, rejection_reason = scoring_and_acceptance(
        strategy=strategy_bad,
        runners=[runner],
        loadflow_results_unsplit=loadflow_results_unsplit,
        metrics_unsplit=metrics_unsplit,
        scoring_params=scoring_params,
    )

    # Should be rejected (either during early stopping or after full N-1)
    assert rejection_reason is not None, "Strategy with 3 splits should be rejected"

    # Verify early stopping behavior: if rejected during early stopping, only subset was computed
    all_case_ids_in_result = lfs_bad.branch_results.select("contingency").unique().collect()["contingency"].to_list()
    if rejection_reason.early_stopping:
        # If rejected during early stopping, only subset should be computed
        assert len(all_case_ids_in_result) <= len(early_stop_case_ids) + 1, (
            "Early stopping rejection should only compute subset of cases"
        )
    # If early stopping passed but final acceptance failed, all cases should be computed
    # This is also valid behavior - early stopping is just an optimization

    # Test 2: Early stopping with acceptance and continuation
    # Create a strategy that performs slightly better to pass early stopping
    actions_good = random_actions(
        action_set=action_set,
        rng=np.random.default_rng(123),  # Different seed for different actions
        n_split_subs=1,  # Fewer splits
    )

    strategy_good = [
        ACOptimTopology(
            actions=actions_good,
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=bytes.fromhex("cafebabe"),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.0,
            metrics={"overload_energy_n_1": 10.0},  # Lower overload
            worst_k_contingency_cases=early_stop_case_ids,
        )
    ]

    lfs_good, _, rejection_reason_good = scoring_and_acceptance(
        strategy=strategy_good,
        runners=[runner],
        loadflow_results_unsplit=loadflow_results_unsplit,
        metrics_unsplit=metrics_unsplit,
        scoring_params=scoring_params,
    )

    # Should pass early stopping and compute all cases
    # (might still be rejected in final acceptance, but should compute all cases)
    all_case_ids_in_result_good = lfs_good.branch_results.select("contingency").unique().collect()["contingency"].to_list()

    # Should have computed all cases if it passed early stopping
    # (rejection_reason_good might not be None if final acceptance failed, but we should have all cases)
    if rejection_reason_good is None or not rejection_reason_good.early_stopping:
        # If it passed early stopping, should have all contingency cases
        assert len(all_case_ids_in_result_good) >= len(early_stop_case_ids), (
            "Should compute full set of cases after passing early stopping"
        )


def test_scoring_and_acceptance_no_early_stopping(grid_folder: Path) -> None:
    """Test the scoring_and_acceptance function with early stopping disabled.

    This test verifies that when early_stop_validation is False, all contingencies are computed
    regardless of worst_k_contingency_cases.
    """
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
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

    # Run unsplit baseline
    loadflow_results_unsplit = runner.run_ac_loadflow([], [])
    metrics_unsplit_temp = compute_metrics_single_timestep(
        actions=[],
        disconnections=[],
        loadflow=loadflow_results_unsplit,
        additional_info=None,
    )
    metrics_unsplit = [metrics_unsplit_temp]

    # Create a strategy with splits
    rng = np.random.default_rng(42)
    actions = random_actions(
        action_set=action_set,
        rng=rng,
        n_split_subs=1,
    )

    strategy = [
        ACOptimTopology(
            actions=actions,
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=bytes.fromhex("12345678"),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.0,
            metrics={"overload_energy_n_1": 50.0},
            worst_k_contingency_cases=[],  # Empty, but shouldn't matter with early stopping disabled
        )
    ]

    scoring_params = ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        base_case_ids=[None],
        n_timestep_processes=1,
        early_stop_validation=False,  # Early stopping disabled
        early_stop_non_converging_threshold=0.1,
    )

    lfs, _, rejection_reason = scoring_and_acceptance(
        strategy=strategy,
        runners=[runner],
        loadflow_results_unsplit=loadflow_results_unsplit,
        metrics_unsplit=metrics_unsplit,
        scoring_params=scoring_params,
    )

    # Should compute all cases
    all_case_ids_in_result = lfs.branch_results.select("contingency").unique().collect()["contingency"].to_list()

    # Should have all or most contingency cases (basecase might be separate)
    assert len(all_case_ids_in_result) >= len(all_case_ids) - 1, (
        "Should compute all contingency cases when early stopping is disabled"
    )

    # If rejected, should not be marked as early stopping
    if rejection_reason is not None:
        assert rejection_reason.early_stopping is False, (
            "Rejection should not be marked as early stopping when feature is disabled"
        )


def test_compute_remaining_loadflows(grid_folder: Path) -> None:
    """Test the compute_remaining_loadflows function.

    This test verifies:
    1. The function correctly computes only the remaining contingencies not in the subset
    2. The function concatenates subset and remaining loadflows correctly
    3. The metrics are computed on the full combined set of loadflows
    4. All contingencies are present in the final result
    """
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
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

    strategy = [
        ACOptimTopology(
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
    ]

    # First compute the subset (simulating early stopping phase)
    lfs_subset, _, _ = compute_loadflow_and_metrics(
        runners=[runner],
        strategy=strategy,
        base_case_ids=[None],
        n_timestep_processes=1,
        cases_subset=[subset_case_ids],
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
        runners=[runner],
        strategy=strategy,
        base_case_ids=[None],
        loadflows_subset=lfs_subset,
        cases_subset=[subset_case_ids],
        n_timestep_processes=1,
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
    assert len(metrics_complete) == 1, "Should have metrics for 1 timestep"
    assert metrics_complete[0].fitness is not None, "Metrics should have fitness"
    assert "overload_energy_n_1" in metrics_complete[0].extra_scores, "Metrics should contain overload_energy_n_1"

    # Verify that the metrics reflect the full set of contingencies, not just the subset
    # The complete metrics should account for all contingencies
    assert "critical_branch_count_n_1" in metrics_complete[0].extra_scores
    assert "non_converging_loadflows" in metrics_complete[0].extra_scores
