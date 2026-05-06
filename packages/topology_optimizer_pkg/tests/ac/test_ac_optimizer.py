# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import structlog
from fsspec.implementations.dirfs import DirFileSystem
from sqlmodel import Session, select
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.loadflow_result_helpers_polars import save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import load_action_set_fs
from toop_engine_topology_optimizer.ac.optimizer import (
    AcNotConvergedError,
    initialize_optimization,
    make_runner,
    process_remaining_results,
    run_fast_failing_epoch,
    run_remaining_epoch,
    wait_for_first_dc_results,
)
from toop_engine_topology_optimizer.ac.scoring_functions import (
    compute_loadflow,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, create_session
from toop_engine_topology_optimizer.ac.types import EarlyStoppingStageResult, OptimizerData, TopologyScoringResult
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
)


def test_initialize_optimization(grid_folder: Path, loadflow_result_folder: Path) -> None:
    # Create start parameters
    params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=0.9,
            reconnect_prob=0.05,
            close_coupler_prob=0.05,
            seed=42,
        )
    )
    grid_file = GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Run the function
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_file=grid_file,
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )
    assert len(optimizer_data.runners) == 1
    assert len(strategy.timesteps) == 1
    assert strategy.timesteps[0].loadflow_results is not None
    assert strategy.timesteps[0].metrics.extra_scores["max_flow_n_1"] > 0

    loaded_initial_loadflow = optimizer_data.load_loadflow_fn(strategy.timesteps[0].loadflow_results)
    assert loaded_initial_loadflow is not None
    assert isinstance(loaded_initial_loadflow, LoadflowResultsPolars)


def test_initialize_with_initial_loadflow(grid_folder: Path, tmp_path: Path) -> None:
    grid_file = GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Load the network datas
    action_sets = [
        load_action_set_fs(
            filesystem=processed_gridfile_fs,
            json_file_path=grid_file.action_set_file,
            diff_file_path=grid_file.action_set_diff_file,
        )
        for grid_file in [grid_file]
    ]
    nminus1_definitions = [
        load_pydantic_model_fs(
            filesystem=processed_gridfile_fs, file_path=grid_file.nminus1_definition_file, model_class=Nminus1Definition
        )
        for grid_file in [grid_file]
    ]
    lf_params = [
        load_lf_params_from_fs(filesystem=processed_gridfile_fs, file_path=grid_file.loadflow_parameters_file)
        for grid_file in [grid_file]
    ]

    # Prepare the loadflow runners
    runners = [
        make_runner(
            action_set,
            nminus1_definition,
            grid_file,
            n_processes=1,
            batch_size=None,
            processed_gridfile_fs=processed_gridfile_fs,
            lf_params=lf_param,
        )
        for action_set, nminus1_definition, grid_file, lf_param in zip(
            action_sets, nminus1_definitions, [grid_file], lf_params, strict=True
        )
    ]

    lfs, additional_info = compute_loadflow(
        actions=[],
        disconnections=[],
        pst_setpoints=[],
        runner=runners[0],
    )

    dirfs = DirFileSystem(str(tmp_path))
    ref = save_loadflow_results_polars(dirfs, "test_initial_loadflow", lfs)

    loadflow_result_fs = DirFileSystem(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        params = ACOptimizerParameters(
            ga_config=ACGAParameters(
                runtime_seconds=10,
                pull_prob=0.9,
                reconnect_prob=0.05,
                close_coupler_prob=0.05,
                seed=42,
            ),
            initial_loadflow=StoredLoadflowReference(relative_path="non_existent_file"),
        )
        optimizer_data, strategy = initialize_optimization(
            params=params,
            session=create_session(),
            optimization_id="test",
            grid_file=grid_file,
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )

    # Create start parameters
    params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=0.9,
            reconnect_prob=0.05,
            close_coupler_prob=0.05,
            seed=42,
        ),
        initial_loadflow=ref,
    )
    # Run the function
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_file=grid_file,
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )
    assert len(optimizer_data.runners) == 1
    assert len(strategy.timesteps) == 1
    assert strategy.timesteps[0].loadflow_results is not None
    assert strategy.timesteps[0].metrics.extra_scores["max_flow_n_1"] > 0

    loaded_initial_loadflow = optimizer_data.load_loadflow_fn(strategy.timesteps[0].loadflow_results)
    assert loaded_initial_loadflow is not None
    assert isinstance(loaded_initial_loadflow, LoadflowResultsPolars)


def test_initialize_powsybl(grid_folder: Path, loadflow_result_folder: Path) -> None:
    # Create start parameters
    params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=0.9,
            reconnect_prob=0.05,
            close_coupler_prob=0.05,
            seed=42,
        )
    )
    grid_file = GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Run the function
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_file=grid_file,
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )
    assert optimizer_data is not None
    assert strategy is not None
    assert len(optimizer_data.runners) == 1
    assert len(strategy.timesteps) == 1
    assert strategy.timesteps[0].loadflow_results is not None

    loaded_initial_loadflow = optimizer_data.load_loadflow_fn(strategy.timesteps[0].loadflow_results)
    assert loaded_initial_loadflow is not None
    # Query the stored topology using optimizer_data.session
    ac_topos = optimizer_data.session.exec(
        select(ACOptimTopology).where(ACOptimTopology.optimizer_type == OptimizerType.AC)
    ).all()
    assert isinstance(ac_topos, list)
    assert ac_topos[0].metrics["top_k_overloads_n_1"] is not None
    assert len(ac_topos[0].worst_k_contingency_cases) == params.ga_config.n_worst_contingencies


def test_initialize_non_converging(case57_non_converging_path: Path, loadflow_result_folder: Path) -> None:
    params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=0.9,
            reconnect_prob=0.05,
            close_coupler_prob=0.05,
            seed=42,
        )
    )
    grid_file = GridFile(framework=Framework.PANDAPOWER, grid_folder=str(case57_non_converging_path.name))
    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(case57_non_converging_path.parent))
    with pytest.raises(AcNotConvergedError, match="Too many non-converging loadflows in initial loadflow*"):
        optimizer_data, strategy = initialize_optimization(
            params=params,
            session=create_session(),
            optimization_id="test",
            grid_file=grid_file,
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
            optimization_logger=structlog.get_logger("test"),
        )


def test_wait_for_first_dc_results_timeout() -> None:
    with patch("toop_engine_topology_optimizer.ac.optimizer.poll_results_topic") as poll_mock:
        poll_mock.return_value = ({}, [])

        heartbeat_counter = 0

        def _heartbeat_fn():
            nonlocal heartbeat_counter
            heartbeat_counter += 1

        with pytest.raises(TimeoutError):
            wait_for_first_dc_results(
                results_consumer=Mock(spec=LongRunningKafkaConsumer),
                session=Mock(spec=Session),
                max_wait_time=1,
                optimization_id="test",
                heartbeat_fn=_heartbeat_fn,
            )
        assert heartbeat_counter > 0


def test_wait_for_first_dc_results_success_with_topology_counts() -> None:
    with patch("toop_engine_topology_optimizer.ac.optimizer.poll_results_topic") as poll_mock:
        poll_mock.return_value = ({"test": 3}, [])

        heartbeat_counter = 0

        def _heartbeat_fn() -> None:
            nonlocal heartbeat_counter
            heartbeat_counter += 1

        wait_for_first_dc_results(
            results_consumer=Mock(spec=LongRunningKafkaConsumer),
            session=Mock(spec=Session),
            max_wait_time=1,
            optimization_id="test",
            heartbeat_fn=_heartbeat_fn,
        )

        assert poll_mock.called
        assert heartbeat_counter == 0


def test_run_fast_failing_epoch_returns_strategies_and_scores() -> None:
    topology_a = ACOptimTopology(id=1, actions=[1], disconnections=[], timestep=0, metrics={})
    topology_b = ACOptimTopology(id=2, actions=[2], disconnections=[], timestep=0, metrics={})

    expected_results = [
        EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=1.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c1"],
        ),
        EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=2.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c2"],
        ),
    ]

    optimizer_data = Mock(spec=OptimizerData)
    optimizer_data.params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            topology_batch_size=2,
        )
    )
    optimizer_data.session = Mock(spec=Session)
    optimizer_data.evolution_fn = Mock(return_value=[topology_a, topology_b])
    optimizer_data.worst_k_scoring_fn = Mock(return_value=expected_results)
    topologies, scoring_results = run_fast_failing_epoch(
        optimizer_data=optimizer_data,
        epoch_logger=Mock(),
    )
    assert topologies == [topology_a, topology_b]
    assert scoring_results == expected_results
    optimizer_data.worst_k_scoring_fn.assert_called_once_with([topology_a, topology_b])


def test_run_fast_failing_epoch_returns_empty_when_no_strategy_available() -> None:
    optimizer_data = Mock(spec=OptimizerData)
    optimizer_data.params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            topology_batch_size=2,
        )
    )
    optimizer_data.session = Mock(spec=Session)
    optimizer_data.evolution_fn = Mock(return_value=[])
    optimizer_data.worst_k_scoring_fn = Mock()
    topologies, scoring_results = run_fast_failing_epoch(
        optimizer_data=optimizer_data,
        epoch_logger=Mock(),
    )
    assert topologies == []
    assert scoring_results == []
    optimizer_data.worst_k_scoring_fn.assert_not_called()


def test_run_remaining_epoch_returns_strategies_and_scores() -> None:
    topology_a = ACOptimTopology(id=1, actions=[1], disconnections=[], timestep=0, metrics={})
    topology_b = ACOptimTopology(id=2, actions=[2], disconnections=[], timestep=0, metrics={})
    early_stage_results = [
        EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=1.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c1"],
        ),
        EarlyStoppingStageResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=2.0, extra_scores={}),
            rejection_reason=None,
            cases_subset=["c2"],
        ),
    ]
    expected_results = [
        TopologyScoringResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=10.0, extra_scores={}),
            rejection_reason=None,
        ),
        TopologyScoringResult(
            loadflow_results=Mock(spec=LoadflowResultsPolars),
            metrics=Metrics(fitness=20.0, extra_scores={}),
            rejection_reason=None,
        ),
    ]

    optimizer_data = Mock(spec=OptimizerData)
    optimizer_data.params = ACOptimizerParameters(ga_config=ACGAParameters())
    optimizer_data.session = Mock(spec=Session)
    optimizer_data.scoring_fn = Mock(return_value=expected_results)
    strategies, scoring_results = run_remaining_epoch(
        optimizer_data=optimizer_data,
        topologies=[topology_a, topology_b],
        early_stage_results=early_stage_results,
        epoch_logger=Mock(),
    )

    assert strategies == [topology_a, topology_b]
    assert scoring_results == expected_results
    optimizer_data.scoring_fn.assert_called_once_with([topology_a, topology_b], early_stage_results=early_stage_results)


def test_run_remaining_epoch_returns_empty_when_no_strategy_available() -> None:
    optimizer_data = Mock(spec=OptimizerData)
    optimizer_data.scoring_fn = Mock()
    results = []

    def mocked_send_result_fn(result):
        results.append(result)

    strategies, scoring_results = run_remaining_epoch(
        optimizer_data=optimizer_data,
        topologies=[],
        early_stage_results=[],
        epoch_logger=Mock(),
    )

    assert strategies == []
    assert scoring_results == []
    optimizer_data.scoring_fn.assert_not_called()


def test_process_remaining_results_sends_each_topology() -> None:
    results = []

    def mocked_send_result_fn(result):
        results.append(result)

    topology_a = ACOptimTopology(id=1, actions=[1], disconnections=[], timestep=0, metrics={})
    topology_b = ACOptimTopology(id=2, actions=[2], disconnections=[], timestep=0, metrics={})
    topology_c = ACOptimTopology(id=3, actions=[3], disconnections=[], timestep=0, metrics={})

    optimizer_data = Mock(spec=OptimizerData)
    optimizer_data.params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10,
            pull_prob=1.0,
            reconnect_prob=0.0,
            close_coupler_prob=0.0,
            seed=42,
            enable_ac_rejection=False,
            topology_batch_size=3,
            full_analysis_batchsize=2,
            remaining_loadflow_wait_seconds=60.0,
        )
    )
    optimizer_data.session = Mock(spec=Session)
    optimizer_data.store_loadflow_fn = Mock(return_value=StoredLoadflowReference(relative_path="test"))
    returned_topologies, returned_results = process_remaining_results(
        optimizer_data=optimizer_data,
        topologies=[topology_a, topology_b, topology_c],
        full_results=[
            TopologyScoringResult(
                loadflow_results=Mock(spec=LoadflowResultsPolars),
                metrics=Metrics(fitness=1.0, extra_scores={}),
                rejection_reason=None,
            ),
            TopologyScoringResult(
                loadflow_results=Mock(spec=LoadflowResultsPolars),
                metrics=Metrics(fitness=2.0, extra_scores={}),
                rejection_reason=None,
            ),
            TopologyScoringResult(
                loadflow_results=Mock(spec=LoadflowResultsPolars),
                metrics=Metrics(fitness=3.0, extra_scores={}),
                rejection_reason=None,
            ),
        ],
        send_result_fn=mocked_send_result_fn,
        epoch=0,
        epoch_logger=Mock(),
    )

    assert returned_topologies == [topology_a, topology_b, topology_c]
    assert len(returned_results) == 3
    assert optimizer_data.store_loadflow_fn.call_count == 3
    assert len(results) == 3
