# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from confluent_kafka import Consumer
from fsspec.implementations.dirfs import DirFileSystem
from sqlmodel import select
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.loadflow_result_helpers_polars import save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import load_action_set_fs, random_actions
from toop_engine_topology_optimizer.ac.optimizer import (
    AcNotConvergedError,
    initialize_optimization,
    make_runner,
    run_epoch,
    wait_for_first_dc_results,
)
from toop_engine_topology_optimizer.ac.scoring_functions import compute_loadflow
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, create_session
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import hash_topo_data


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
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")]

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Run the function
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_files=grid_files,
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
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")]

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
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

    # Prepare the loadflow runners
    runners = [
        make_runner(
            action_set,
            nminus1_definition,
            grid_file,
            n_processes=1,
            batch_size=None,
            processed_gridfile_fs=processed_gridfile_fs,
        )
        for action_set, nminus1_definition, grid_file in zip(action_sets, nminus1_definitions, grid_files, strict=True)
    ]

    lfs, additional_info = compute_loadflow(
        actions=[[]],
        disconnections=[[]],
        runners=runners,
        n_timestep_processes=1,
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
            grid_files=grid_files,
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
        grid_files=grid_files,
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
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]

    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    # Run the function
    optimizer_data, strategy = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_files=grid_files,
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
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder=str(case57_non_converging_path.name))]
    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(case57_non_converging_path.parent))
    with pytest.raises(AcNotConvergedError, match="Too many non-converging loadflows in initial loadflow*"):
        optimizer_data, strategy = initialize_optimization(
            params=params,
            session=create_session(),
            optimization_id="test",
            grid_files=grid_files,
            loadflow_result_fs=loadflow_result_fs,
            processed_gridfile_fs=processed_gridfile_fs,
        )


def test_wait_for_first_dc_results_timeout() -> None:
    with patch("toop_engine_topology_optimizer.ac.optimizer.poll_results_topic") as poll_mock:
        poll_mock.return_value = []

        with pytest.raises(TimeoutError, match="Did not receive DC results within*"):
            wait_for_first_dc_results(results_consumer=MagicMock(), session=MagicMock(), max_wait_time=1)


def test_run_epoch(grid_folder: Path, loadflow_result_folder: Path) -> None:
    params = ACOptimizerParameters(
        ga_config=ACGAParameters(
            runtime_seconds=10, pull_prob=1.0, reconnect_prob=0.0, close_coupler_prob=0.0, seed=42, enable_ac_rejection=False
        )
    )
    loadflow_result_fs = DirFileSystem(str(loadflow_result_folder))
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    grid_files = [GridFile(framework=Framework.PYPOWSYBL, grid_folder="case57")]
    optimizer_data, _ = initialize_optimization(
        params=params,
        session=create_session(),
        optimization_id="test",
        grid_files=grid_files,
        loadflow_result_fs=loadflow_result_fs,
        processed_gridfile_fs=processed_gridfile_fs,
    )

    action_set = optimizer_data.action_sets[0]
    assert action_set is not None
    assert len(action_set.local_actions)

    # Generate random DC topologies to pull
    for _ in range(10):
        actions = random_actions(action_set, np.random.default_rng(42), 2)

        pst_setpoints = [0, 0, 0, 0]

        topo_hash = hash_topo_data([(actions, [], pst_setpoints)])

        topo = ACOptimTopology(
            actions=actions,
            disconnections=[],
            pst_setpoints=pst_setpoints,
            timestep=0,
            fitness=0,
            unsplit=False,
            strategy_hash=topo_hash,
            optimization_id="test",
            optimizer_type=OptimizerType.DC,
        )
        optimizer_data.session.add(topo)
        try:
            optimizer_data.session.commit()
        except Exception:
            optimizer_data.session.rollback()

    # Run the epoch
    # We create an empty consumer, as we have already added the results to the database
    consumer = Mock(spec=Consumer)
    consumer.consume = Mock(return_value=[])
    send_result_fn = Mock()
    run_epoch(optimizer_data, consumer, send_result_fn, epoch=0)

    assert consumer.consume.called
    assert send_result_fn.called

    # We expect 2 AC results in the database now
    ac_topos = optimizer_data.session.exec(
        select(ACOptimTopology).where(ACOptimTopology.optimizer_type == OptimizerType.AC)
    ).all()
    assert len(ac_topos) == 2
    assert sum(topo.unsplit for topo in ac_topos) == 1
    assert sum(not topo.unsplit for topo in ac_topos) == 1
    assert all(topo.get_loadflow_reference() is not None for topo in ac_topos)
