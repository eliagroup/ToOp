from copy import deepcopy
from pathlib import Path

import numpy as np
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_interfaces.stored_action_set import load_action_set, random_actions
from toop_engine_topology_optimizer.ac.scoring_functions import (
    compute_loadflow,
    compute_loadflow_and_metrics_with_early_stopping,
    compute_metrics_single_timestep,
    evaluate_acceptance,
    extract_switching_distance,
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


def test_compute_loadflow_and_metrics_with_early_stopping(grid_folder: Path) -> None:
    action_set = load_action_set(grid_folder / "case14" / PREPROCESSING_PATHS["action_set_file_path"])
    nminus1_definition = load_nminus1_definition(
        grid_folder / "case14" / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )

    runner = PandapowerRunner()
    runner.load_base_grid(grid_folder / "case14" / PREPROCESSING_PATHS["grid_file_path_pandapower"])
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    actions = random_actions(
        action_set=action_set,
        rng=np.random.default_rng(42),
        n_split_subs=2,  # Split two substations
    )

    topo = ACOptimTopology(
        actions=actions,
        disconnections=[],
        pst_setpoints=[0, 0, 0, 0],
        unsplit=False,
        timestep=0,
        strategy_hash=bytes.fromhex("deadbeef"),
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.5,
        metrics={"overload_energy_n_1": 123.4},
    )

    # Case 1: Overload threshold is exceeded
    n_1_def = runner.get_nminus1_definition()
    all_case_ids = [cont.id for cont in n_1_def.contingencies]
    runner_before = deepcopy(runner)
    lfs, metrics = compute_loadflow_and_metrics_with_early_stopping(
        runners=[runner],
        strategy=[topo],
        base_case_ids=[None],
        threshold_overload_all_t=[-1.0],
        threshold_case_ids_all_t=[np.random.choice(all_case_ids, size=5, replace=False).tolist()],
    )

    # Ensure that the runner has not been modified
    assert runner_before.nminus1_definition == runner.nminus1_definition  # type: ignore[comparison-overlap]

    assert metrics[0].fitness == -99999999


def test_evaluate_acceptance_identical_metrics():
    empty_lf_results = LoadflowResults(job_id="test")
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
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert accepted, "Results rejected although they are the same as before and thresholds is exactly 1."
    # Not accepted if any thresholds < 1.
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert not accepted, "Results rejected although they are the same as before and thresholds is below 1."
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert not accepted, "Results rejected although they are just as good and convergence thresholds below 1."
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=1.0,
    )
    assert not accepted, "Results rejected although they are just as good and overload thresholds below 1."

    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=0.9,
    )
    assert not accepted, "Results rejected although they are just as good and crit branch thresholds below 1."

    # Accepted if thresholds > 1.
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert accepted, "Results rejected although they are just as good and thresholds above 1."


def test_evaluate_acceptance_improved_metrics():
    empty_lf_results = LoadflowResults(job_id="test")
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
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert accepted, "Results rejected although they are the same as before and thresholds is exactly 1."
    # Accepted if all thresholds=0.9.
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert accepted, "Results not accepted although they improved by exactly 10 percent and thresholds is 0.9."

    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.8,
        reject_overload_threshold=0.8,
        reject_critical_branch_threshold=0.8,
    )
    assert not accepted, "Results accepted although they only improved by exactly 10 percent and thresholds is 0.8."

    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.8,
        reject_overload_threshold=0.8,
        reject_critical_branch_threshold=0.8,
    )
    assert not accepted, "Results accepted although they only improved by exactly 10 percent and thresholds is 0.8."

    # Accepted if thresholds > 1.
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert accepted, "Results rejected although they are just as good and thresholds above 1."


def test_evaluate_acceptance_worse_metrics():
    empty_lf_results = LoadflowResults(job_id="test")
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
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.0,
        reject_overload_threshold=1.0,
        reject_critical_branch_threshold=1.0,
    )
    assert not accepted, "Results accepted although they are worse as before and thresholds is exactly 1."
    # Not Accepted if all thresholds=0.9.
    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=0.9,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.9,
    )
    assert not accepted, "Results accepted although they got worse by exactly 10 percent and thresholds is 0.9."

    accepted = evaluate_acceptance(
        loadflow_results_split=empty_lf_results,
        metrics_split=metrics_split,
        loadflow_results_unsplit=empty_lf_results,
        metrics_unsplit=metrics_unsplit,
        reject_convergence_threshold=1.1,
        reject_overload_threshold=1.1,
        reject_critical_branch_threshold=1.1,
    )
    assert accepted, "Results not accepted although they only got worse by exactly 10 percent and thresholds is 1.1."
