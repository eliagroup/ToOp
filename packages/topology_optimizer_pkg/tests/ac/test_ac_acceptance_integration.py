# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fsspec.implementations.dirfs import DirFileSystem
from omegaconf import DictConfig
from sqlmodel import Session, select
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.ac.optimizer import (
    initialize_optimization,
    process_fast_failing_results,
    process_remaining_results,
    run_fast_failing_epoch,
    run_remaining_epoch,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, convert_message_topo_to_db_topo, create_session
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    prepare_importer_parameters,
    run_preprocessing,
    run_task_process,
    set_environment_variables,
)
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    Strategy,
    Topology,
    TopologyPushResult,
    TopologyRejectionResult,
)


@pytest.fixture(scope="session")
def acceptance_grid_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base_path = tmp_path_factory.mktemp("ac_acceptance_node_breaker")
    input_grid = base_path / "grid.xiidm"

    net = basic_node_breaker_network_powsybl()
    open_switches = ["load1_DISCONNECTOR_18_0", "L71_DISCONNECTOR_10_1"]
    close_switches = ["load1_DISCONNECTOR_18_1", "L71_DISCONNECTOR_10_0"]
    for switch in close_switches:
        net.close_switch(switch)
    for switch in open_switches:
        net.open_switch(switch)

    net.save(input_grid)

    data_folder = base_path / input_grid.stem
    importer_parameters = prepare_importer_parameters(input_grid, data_folder)
    preprocessing_parameters = PreprocessParameters(action_set_clip=2**10, preprocess_bb_outages=False)
    run_preprocessing(
        importer_parameters=importer_parameters,
        data_folder=data_folder,
        preprocessing_parameters=preprocessing_parameters,
        is_pandapower_net=False,
    )
    return data_folder


@pytest.fixture(scope="session")
def dc_acceptance_result(acceptance_grid_folder: Path) -> dict:
    output_dir = acceptance_grid_folder / "dc_results"
    dc_config = DictConfig(
        {
            "task_name": "test_ac_acceptance_dc",
            "fixed_files": [str(acceptance_grid_folder / "static_information.hdf5")],
            "double_precision": None,
            "tensorboard_dir": str(output_dir / "tensorboard" / "{task_name}"),
            "stats_dir": str(output_dir / "stats" / "{task_name}"),
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "stdout": None,
            "double_limits": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "output_json": str(output_dir / "output.json"),
            "lf_config": {"distributed": False},
            "ga_config": {
                "runtime_seconds": 5,
                "n_worst_contingencies": 10,
                "target_metrics": [["overload_energy_n_1", 1.0]],
                "me_descriptors": [
                    {"metric": "split_subs", "num_cells": 5},
                    {"metric": "switching_distance", "num_cells": 40},
                ],
                "observed_metrics": ["overload_energy_n_1", "split_subs", "critical_branch_count_n_1", "switching_distance"],
                "cell_depth": 10,
                "seed": 42,
            },
        }
    )
    set_environment_variables(dc_config)
    result = run_task_process(dc_config)
    assert result is not None
    assert result["best_topos"], "DC optimization did not produce candidate topologies for AC acceptance testing."
    return result


@pytest.fixture
def ac_optimizer_context(
    acceptance_grid_folder: Path,
    dc_acceptance_result: dict,
    tmp_path: Path,
) -> Iterator[tuple[Session, object, list[dict], list[object]]]:
    optimization_id = "test_ac_acceptance"
    loadflow_result_fs = DirFileSystem(str(tmp_path / "loadflows"))
    processed_gridfile_fs = DirFileSystem(str(acceptance_grid_folder.parent))
    session = create_session()

    sorted_dc_topologies = sorted(
        dc_acceptance_result["best_topos"],
        key=lambda topo: topo["metrics"]["fitness"],
        reverse=True,
    )
    dc_topologies: list[dict] = []
    seen_topology_signatures: set[tuple] = set()
    for topology in [*sorted_dc_topologies[:3], *sorted_dc_topologies[-3:]]:
        signature = (
            tuple(topology.get("actions") or []),
            tuple(topology.get("disconnections") or []),
            tuple(topology.get("pst_setpoints") or []),
        )
        if signature in seen_topology_signatures:
            continue
        seen_topology_signatures.add(signature)
        dc_topologies.append(topology)

    for topology_data in dc_topologies:
        db_topologies = convert_message_topo_to_db_topo(
            message_strategy=Strategy(timesteps=[to_message_topology(topology_data)]),
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.DC,
        )
        for topology in db_topologies:
            session.add(topology)
    session.commit()

    persisted_dc_topologies = session.exec(
        select(ACOptimTopology)
        .where(ACOptimTopology.optimization_id == optimization_id)
        .where(ACOptimTopology.optimizer_type == OptimizerType.DC)
    ).all()
    assert len(persisted_dc_topologies) == len(dc_topologies)
    session.rollback()

    optimizer_data, _initial_strategy = initialize_optimization(
        params=ACOptimizerParameters(
            ga_config=ACGAParameters(
                runtime_seconds=10,
                runner_processes=1,
                worst_k_runner_processes=1,
                contingency_processes=1,
                worst_k_contingency_processes=1,
                pull_prob=1.0,
                reconnect_prob=0.0,
                close_coupler_prob=0.0,
                seed=42,
                enable_ac_rejection=False,
                reject_convergence_threshold=10.0,
                reject_overload_threshold=10.0,
                reject_critical_branch_threshold=10.0,
                reject_voltage_jump_threshold=10.0,
                reject_critical_va_diff_threshold=10.0,
                early_stop_validation=False,
                n_worst_contingencies=10,
            )
        ),
        session=session,
        optimization_id=optimization_id,
        grid_file=GridFile(framework=Framework.PYPOWSYBL, grid_folder=acceptance_grid_folder.name),
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    sent_results: list[object] = []
    yield session, optimizer_data, dc_topologies, sent_results


def to_message_topology(topology_data: dict) -> Topology:
    metrics_payload = topology_data["metrics"]
    return Topology(
        actions=topology_data.get("actions") or [],
        disconnections=topology_data.get("disconnections") or [],
        pst_setpoints=topology_data.get("pst_setpoints"),
        metrics=Metrics(
            fitness=metrics_payload["fitness"],
            extra_scores=metrics_payload.get("extra_scores", {}),
            worst_k_contingency_cases=metrics_payload.get("worst_k_contingency_cases"),
        ),
    )


def _run_ac_epoch(
    optimizer_data: object,
    send_result_fn: Callable[[object], None],
) -> tuple[list[ACOptimTopology], list[ACOptimTopology]]:
    fast_topologies, fast_results = run_fast_failing_epoch(optimizer_data=optimizer_data)
    survivor_topologies, survivor_early_results = process_fast_failing_results(
        optimizer_data=optimizer_data,
        topologies=fast_topologies,
        fast_failing_results=fast_results,
        epoch=1,
        send_result_fn=send_result_fn,
    )
    full_topologies, full_results = run_remaining_epoch(
        optimizer_data=optimizer_data,
        topologies=survivor_topologies,
        early_stage_results=survivor_early_results,
    )
    process_remaining_results(
        optimizer_data=optimizer_data,
        topologies=full_topologies,
        full_results=full_results,
        epoch=1,
        send_result_fn=send_result_fn,
    )
    return fast_topologies, full_topologies


def _make_ac_params(**overrides: float | bool | int) -> ACOptimizerParameters:
    return ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            runner_processes=1,
            worst_k_runner_processes=1,
            contingency_processes=1,
            worst_k_contingency_processes=1,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            enable_ac_rejection=True,
            early_stop_validation=False,
            n_worst_contingencies=10,
            **overrides,
        )
    )


def _criterion_specific_params(criterion: str, threshold: float) -> ACOptimizerParameters:
    overrides: dict[str, float | bool | int] = {
        "reject_convergence_threshold": 10.0,
        "reject_overload_threshold": 10.0,
        "reject_critical_branch_threshold": 10.0,
        "reject_voltage_jump_threshold": 10.0,
        "reject_critical_va_diff_threshold": 10.0,
        "enable_critical_voltage_rejection": criterion in {"voltage-magnitude", "voltage-angle"},
    }

    if criterion == "convergence":
        overrides["reject_convergence_threshold"] = threshold
    elif criterion == "overload-energy":
        overrides["reject_overload_threshold"] = threshold
    elif criterion == "critical-branch-count":
        overrides["reject_critical_branch_threshold"] = threshold
    elif criterion == "voltage-magnitude":
        overrides["reject_voltage_jump_threshold"] = threshold
    elif criterion == "voltage-angle":
        overrides["reject_critical_va_diff_threshold"] = threshold
    else:
        raise ValueError(f"Unsupported criterion {criterion}")

    return _make_ac_params(**overrides)


def _build_action_seed_context(
    acceptance_grid_folder: Path,
    tmp_path: Path,
    params: ACOptimizerParameters,
    optimization_id: str,
) -> tuple[Session, object]:
    loadflow_result_fs = DirFileSystem(str(tmp_path / optimization_id / "loadflows"))
    processed_gridfile_fs = DirFileSystem(str(acceptance_grid_folder.parent))
    session = create_session()

    for action_idx in range(10):
        db_topologies = convert_message_topo_to_db_topo(
            message_strategy=Strategy(
                timesteps=[
                    Topology(
                        actions=[action_idx],
                        disconnections=[],
                        pst_setpoints=None,
                        metrics=Metrics(fitness=float(-(action_idx + 1)), extra_scores={}, worst_k_contingency_cases=[]),
                    )
                ]
            ),
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.DC,
        )
        for topology in db_topologies:
            session.add(topology)
    session.commit()
    session.rollback()

    optimizer_data, _ = initialize_optimization(
        params=params,
        session=session,
        optimization_id=optimization_id,
        grid_file=GridFile(framework=Framework.PYPOWSYBL, grid_folder=acceptance_grid_folder.name),
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )
    return session, optimizer_data


def _run_action_seeded_acceptance_epoch(
    acceptance_grid_folder: Path,
    tmp_path: Path,
    params: ACOptimizerParameters,
    optimization_id: str,
) -> tuple[list[object], list[ACOptimTopology], dict]:
    session, optimizer_data = _build_action_seed_context(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=params,
        optimization_id=optimization_id,
    )

    sent_results: list[object] = []
    while True:
        fast_topologies, full_topologies = _run_ac_epoch(
            optimizer_data=optimizer_data,
            send_result_fn=lambda result: sent_results.append(result),
        )
        if not fast_topologies:
            break
        if not full_topologies and not sent_results:
            break

    ac_topologies = session.exec(
        select(ACOptimTopology)
        .where(ACOptimTopology.optimization_id == optimization_id)
        .where(ACOptimTopology.optimizer_type == OptimizerType.AC)
        .where(ACOptimTopology.unsplit == False)  # noqa: E712
    ).all()
    unsplit_topology = session.exec(
        select(ACOptimTopology)
        .where(ACOptimTopology.optimization_id == optimization_id)
        .where(ACOptimTopology.optimizer_type == OptimizerType.AC)
        .where(ACOptimTopology.unsplit == True)  # noqa: E712
    ).one()
    return sent_results, ac_topologies, unsplit_topology.metrics


@pytest.mark.parametrize(
    ("criterion", "param_name", "strict_threshold", "light_threshold"),
    [
        ("convergence", "reject_convergence_threshold", 0.99, 10.0),
        ("overload-energy", "reject_overload_threshold", 0.01, 10.0),
        ("critical-branch-count", "reject_critical_branch_threshold", 0.49, 10.0),
    ],
)
def test_ac_acceptance_rejection_matrix_for_variable_criteria(
    acceptance_grid_folder: Path,
    tmp_path: Path,
    criterion: str,
    param_name: str,
    strict_threshold: float,
    light_threshold: float,
) -> None:
    strict_results, strict_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_criterion_specific_params(criterion=criterion, threshold=strict_threshold),
        optimization_id=f"strict_{criterion.replace('-', '_')}",
    )
    assert strict_topologies
    assert all(topo.acceptance is False for topo in strict_topologies)
    assert strict_results
    assert all(isinstance(result, TopologyRejectionResult) for result in strict_results)
    assert all(
        result.reason.criterion == criterion for result in strict_results if isinstance(result, TopologyRejectionResult)
    )

    light_results, light_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_criterion_specific_params(criterion=criterion, threshold=light_threshold),
        optimization_id=f"light_{criterion.replace('-', '_')}",
    )
    assert light_topologies
    assert all(topo.acceptance is True for topo in light_topologies)
    assert light_results
    assert all(isinstance(result, TopologyPushResult) for result in light_results)


def test_ac_acceptance_rejection_matrix_for_voltage_angle_with_lowered_cutoff(
    acceptance_grid_folder: Path,
    tmp_path: Path,
) -> None:
    strict_results, strict_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=10.0,
            reject_critical_va_diff_threshold=0.3,
            critical_va_diff_degree=0.1,
        ),
        optimization_id="strict_voltage_angle_lower_cutoff",
    )
    assert strict_topologies
    assert all(topo.acceptance is False for topo in strict_topologies)
    assert strict_results
    assert all(isinstance(result, TopologyRejectionResult) for result in strict_results)
    assert all(
        result.reason.criterion == "voltage-angle"
        for result in strict_results
        if isinstance(result, TopologyRejectionResult)
    )

    light_results, light_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=10.0,
            reject_critical_va_diff_threshold=1.2,
            critical_va_diff_degree=0.1,
        ),
        optimization_id="light_voltage_angle_lower_cutoff",
    )
    assert light_topologies
    assert all(topo.acceptance is True for topo in light_topologies)
    assert light_results
    assert all(isinstance(result, TopologyPushResult) for result in light_results)


@pytest.mark.parametrize(
    ("criterion", "param_name", "sensible_threshold"),
    [
        ("overload-energy", "reject_overload_threshold", 1.0),
        ("critical-branch-count", "reject_critical_branch_threshold", 0.75),
    ],
)
def test_ac_acceptance_sensible_threshold_rejects_some_variable_criteria(
    acceptance_grid_folder: Path,
    tmp_path: Path,
    criterion: str,
    param_name: str,
    sensible_threshold: float,
) -> None:
    sent_results, ac_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_criterion_specific_params(criterion=criterion, threshold=sensible_threshold),
        optimization_id=f"sensible_{criterion.replace('-', '_')}",
    )

    assert ac_topologies
    accepted = [topo for topo in ac_topologies if topo.acceptance is True]
    rejected = [topo for topo in ac_topologies if topo.acceptance is False]
    assert accepted, f"Expected at least one accepted topology for sensible {criterion} threshold."
    assert rejected, f"Expected at least one rejected topology for sensible {criterion} threshold."
    assert any(isinstance(result, TopologyPushResult) for result in sent_results)
    rejection_results = [result for result in sent_results if isinstance(result, TopologyRejectionResult)]
    assert rejection_results
    assert all(result.reason.criterion == criterion for result in rejection_results)


def test_ac_acceptance_sensible_voltage_angle_threshold_rejects_some_candidates(
    acceptance_grid_folder: Path,
    tmp_path: Path,
) -> None:
    sent_results, ac_topologies, unsplit_metrics = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=10.0,
            reject_critical_va_diff_threshold=0.85,
            critical_va_diff_degree=0.1,
        ),
        optimization_id="sensible_voltage_angle_lower_cutoff",
    )

    assert unsplit_metrics.get("critical_va_diff_count_n_1", 0.0) > 0.0
    assert ac_topologies
    accepted = [topo for topo in ac_topologies if topo.acceptance is True]
    rejected = [topo for topo in ac_topologies if topo.acceptance is False]
    assert accepted
    assert rejected
    assert any(isinstance(result, TopologyPushResult) for result in sent_results)
    rejection_results = [result for result in sent_results if isinstance(result, TopologyRejectionResult)]
    assert rejection_results
    assert all(result.reason.criterion == "voltage-angle" for result in rejection_results)


def test_ac_acceptance_rejection_matrix_for_voltage_magnitude_with_lowered_cutoff(
    acceptance_grid_folder: Path,
    tmp_path: Path,
) -> None:
    strict_results, strict_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=0.3,
            reject_critical_va_diff_threshold=10.0,
            critical_voltage_jump_percent=0.01,
        ),
        optimization_id="strict_voltage_magnitude_lower_cutoff",
    )
    assert strict_topologies
    assert all(topo.acceptance is False for topo in strict_topologies)
    assert strict_results
    assert all(isinstance(result, TopologyRejectionResult) for result in strict_results)
    assert all(
        result.reason.criterion == "voltage-magnitude"
        for result in strict_results
        if isinstance(result, TopologyRejectionResult)
    )

    light_results, light_topologies, _ = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=1.2,
            reject_critical_va_diff_threshold=10.0,
            critical_voltage_jump_percent=0.01,
        ),
        optimization_id="light_voltage_magnitude_lower_cutoff",
    )
    assert light_topologies
    assert all(topo.acceptance is True for topo in light_topologies)
    assert light_results
    assert all(isinstance(result, TopologyPushResult) for result in light_results)


def test_ac_acceptance_sensible_voltage_magnitude_threshold_rejects_some_candidates(
    acceptance_grid_folder: Path,
    tmp_path: Path,
) -> None:
    sent_results, ac_topologies, unsplit_metrics = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(
            enable_critical_voltage_rejection=True,
            reject_convergence_threshold=10.0,
            reject_overload_threshold=10.0,
            reject_critical_branch_threshold=10.0,
            reject_voltage_jump_threshold=0.9,
            reject_critical_va_diff_threshold=10.0,
            critical_voltage_jump_percent=0.01,
        ),
        optimization_id="sensible_voltage_magnitude_lower_cutoff",
    )

    assert unsplit_metrics.get("voltage_jump_count_n_1", 0.0) > 0.0
    assert ac_topologies
    accepted = [topo for topo in ac_topologies if topo.acceptance is True]
    rejected = [topo for topo in ac_topologies if topo.acceptance is False]
    assert accepted
    assert rejected
    assert any(isinstance(result, TopologyPushResult) for result in sent_results)
    rejection_results = [result for result in sent_results if isinstance(result, TopologyRejectionResult)]
    assert rejection_results
    assert all(result.reason.criterion == "voltage-magnitude" for result in rejection_results)


def test_ac_acceptance_convergence_is_constant_on_node_breaker_grid(
    acceptance_grid_folder: Path,
    tmp_path: Path,
) -> None:
    sent_results, ac_topologies, unsplit_metrics = _run_action_seeded_acceptance_epoch(
        acceptance_grid_folder=acceptance_grid_folder,
        tmp_path=tmp_path,
        params=_make_ac_params(reject_convergence_threshold=0.99),
        optimization_id="constant_convergence",
    )

    assert ac_topologies
    unsplit_non_converging = unsplit_metrics.get("non_converging_loadflows", 0.0)
    assert all(topo.metrics.get("non_converging_loadflows", 0.0) == unsplit_non_converging for topo in ac_topologies)
    assert sent_results
    assert all(isinstance(result, TopologyRejectionResult) for result in sent_results)
    assert all(result.reason.criterion == "convergence" for result in sent_results)


@pytest.mark.timeout(180)
def test_ac_acceptance_off_evaluates_candidates_with_production_flow(
    ac_optimizer_context: tuple[Session, object, list[dict], list[object]],
) -> None:
    session, optimizer_data, dc_topologies, sent_results = ac_optimizer_context

    def _capture(result: object) -> None:
        sent_results.append(result)

    fast_topologies, full_topologies = _run_ac_epoch(optimizer_data=optimizer_data, send_result_fn=_capture)

    assert len(fast_topologies) > 0
    assert len(full_topologies) > 0
    assert len(full_topologies) == len(fast_topologies)
    assert len(sent_results) == len(full_topologies)

    ac_topologies = session.exec(
        select(ACOptimTopology)
        .where(ACOptimTopology.optimization_id == "test_ac_acceptance")
        .where(ACOptimTopology.optimizer_type == OptimizerType.AC)
        .where(ACOptimTopology.unsplit == False)  # noqa: E712
    ).all()
    assert ac_topologies, "Expected evaluated AC topologies to be persisted by the production AC optimizer flow."

    assert len(ac_topologies) >= len(full_topologies)
    assert all(topo.acceptance is not None for topo in ac_topologies)

    unsplit_topology = session.exec(
        select(ACOptimTopology)
        .where(ACOptimTopology.optimization_id == "test_ac_acceptance")
        .where(ACOptimTopology.optimizer_type == OptimizerType.AC)
        .where(ACOptimTopology.unsplit == True)  # noqa: E712
    ).one()

    unsplit_metrics = unsplit_topology.metrics
    split_metrics = [topo.metrics for topo in ac_topologies]

    assert any(
        metrics.get("overload_energy_n_1", 0.0) > unsplit_metrics.get("overload_energy_n_1", 0.0)
        for metrics in split_metrics
    ), "Expected at least one evaluated topology with worse overload energy than the unsplit reference."
    assert all("non_converging_loadflows" in metrics for metrics in split_metrics), (
        "Expected AC evaluation to compute non-converging loadflow counts for all evaluated topologies."
    )

    observed_metric_names = {
        "critical_branch_count_n_1": "critical branch count",
        "voltage_jump_count_n_1": "voltage jump count",
        "critical_va_diff_count_n_1": "critical voltage-angle-difference count",
    }
    for metric_name, description in observed_metric_names.items():
        assert any(metric_name in metrics for metrics in split_metrics), (
            f"Expected AC evaluation to compute {description} on this production path."
        )

    assert len(dc_topologies) >= len(full_topologies)
