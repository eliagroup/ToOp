# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import toop_engine_topology_optimizer.dc_bruteforce.optimizer as bruteforce_optimizer
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import scoring_function
from toop_engine_topology_optimizer.dc_bruteforce.optimizer import (
    OptimizerData,
    convert_topologies_to_messages,
    extract_topologies,
    get_num_branch_topologies_tried,
    initialize_optimization,
    is_exhausted,
    run_epoch,
)
from toop_engine_topology_optimizer.interfaces.messages.commands import StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    DescriptorDef,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Strategy, Topology, TopologyPushResult


def _ga_config(runtime_seconds: float = 5, enable_nodal_inj_optim: bool = False) -> BatchedMEParameters:
    return BatchedMEParameters(
        runtime_seconds=runtime_seconds,
        enable_nodal_inj_optim=enable_nodal_inj_optim,
        iterations_per_epoch=8,
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=2),
            DescriptorDef(metric="switching_distance", num_cells=45),
        ),
    )


def test_initialize_and_run_epoch(grid_folder: str) -> None:
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=_ga_config(runtime_seconds=5),
            loadflow_solver_config=LoadflowSolverParameters(max_num_splits=1, max_num_disconnections=0),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")],
        optimization_id="test",
    )

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    topologies_per_epoch = (
        start_opt_command.dc_params.ga_config.iterations_per_epoch
        * start_opt_command.dc_params.loadflow_solver_config.batch_size
    )
    optimizer_data, stats, initial_strategy = initialize_optimization(
        params=start_opt_command.dc_params,
        optimization_id="test123",
        static_information_files=tuple(gf.static_information_file for gf in start_opt_command.grid_files),
        processed_gridfile_fs=processed_gridfile_fs,
    )

    assert isinstance(optimizer_data, OptimizerData)
    assert isinstance(stats[0], StaticInformationStats)
    assert isinstance(initial_strategy, Strategy)
    assert (
        optimizer_data.runtime_state.solver_config.batch_size_bsdf
        == start_opt_command.dc_params.loadflow_solver_config.batch_size
    )

    optimizer_data = run_epoch(optimizer_data)
    assert get_num_branch_topologies_tried(optimizer_data) == min(
        topologies_per_epoch, optimizer_data.runtime_state.total_workset_size
    )

    topologies = extract_topologies(optimizer_data)
    assert isinstance(topologies, list)
    if topologies:
        assert isinstance(topologies[0], Topology)

    messages = convert_topologies_to_messages(topologies, epoch=1)
    if messages:
        assert isinstance(messages[0], TopologyPushResult)
        assert isinstance(messages[0].strategy, Strategy)
        assert len(messages[0].strategy.timesteps) == 1

    assert extract_topologies(optimizer_data) == []
    assert is_exhausted(optimizer_data) is (optimizer_data.runtime_state.total_workset_size <= topologies_per_epoch)


def test_initialize_rejects_pst_optimization(grid_folder: str) -> None:
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    params = DCOptimizerParameters(
        ga_config=_ga_config(enable_nodal_inj_optim=True),
        loadflow_solver_config=LoadflowSolverParameters(max_num_splits=1),
    )

    try:
        initialize_optimization(
            params=params,
            optimization_id="test",
            static_information_files=(
                GridFile(framework=Framework.PANDAPOWER, grid_folder="case14").static_information_file,
            ),
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except ValueError as exc:
        assert "PSTs untouched" in str(exc)
    else:
        raise AssertionError("Expected initialize_optimization to reject nodal injection optimization.")


def test_initialize_rejects_multiple_static_information_files(grid_folder: str) -> None:
    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    static_information_file = GridFile(framework=Framework.PANDAPOWER, grid_folder="case14").static_information_file

    try:
        initialize_optimization(
            params=DCOptimizerParameters(
                ga_config=_ga_config(),
                loadflow_solver_config=LoadflowSolverParameters(max_num_splits=1),
            ),
            optimization_id="test",
            static_information_files=(static_information_file, static_information_file),
            processed_gridfile_fs=processed_gridfile_fs,
        )
    except AssertionError as exc:
        assert "exactly one static information file" in str(exc)
    else:
        raise AssertionError("Expected initialize_optimization to reject multiple static-information files.")


def test_chunk_scoring_matches_genotype_scoring(grid_folder: str) -> None:
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=_ga_config(runtime_seconds=5),
            loadflow_solver_config=LoadflowSolverParameters(max_num_splits=1, max_num_disconnections=0),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")],
        optimization_id="test",
    )

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    optimizer_data, _stats, _initial_strategy = initialize_optimization(
        params=start_opt_command.dc_params,
        optimization_id="test123",
        static_information_files=tuple(gf.static_information_file for gf in start_opt_command.grid_files),
        processed_gridfile_fs=processed_gridfile_fs,
    )

    action_set = optimizer_data.runtime_state.dynamic_information.action_set
    assert action_set is not None
    topology_action = int(action_set.action_start_indices[0])
    topology_chunk, disconnection_chunk = bruteforce_optimizer._chunk_to_topologies(
        chunk=[bruteforce_optimizer.WorksetEntry(action_indices=(topology_action,), disconnections=())],
        chunk_size=optimizer_data.runtime_state.solver_config.batch_size_bsdf,
        max_num_splits=start_opt_command.dc_params.loadflow_solver_config.max_num_splits,
        max_num_disconnections=start_opt_command.dc_params.loadflow_solver_config.max_num_disconnections,
    )

    chunk_fitness, chunk_metrics = bruteforce_optimizer._score_chunk(
        topology_chunk=topology_chunk,
        disconnection_chunk=disconnection_chunk,
        evaluated_count=1,
        dynamic_information=optimizer_data.runtime_state.dynamic_information,
        solver_config=optimizer_data.runtime_state.solver_config,
        target_metrics=start_opt_command.dc_params.ga_config.target_metrics,
        observed_metrics=start_opt_command.dc_params.ga_config.observed_metrics,
        n_worst_contingencies=start_opt_command.dc_params.ga_config.n_worst_contingencies,
    )

    genotype = Genotype(
        action_index=topology_chunk.action,
        disconnections=disconnection_chunk,
        nodal_injections_optimized=None,
    )
    genotype_fitness, _descriptors, genotype_metrics, _emitter_info, _random_key, _genotype = scoring_function(
        genotype,
        bruteforce_optimizer.jax.random.PRNGKey(0),
        dynamic_informations=(optimizer_data.runtime_state.dynamic_information,),
        solver_configs=(optimizer_data.runtime_state.solver_config,),
        target_metrics=start_opt_command.dc_params.ga_config.target_metrics,
        observed_metrics=start_opt_command.dc_params.ga_config.observed_metrics,
        descriptor_metrics=tuple(desc.metric for desc in start_opt_command.dc_params.ga_config.me_descriptors),
        n_worst_contingencies=start_opt_command.dc_params.ga_config.n_worst_contingencies,
    )

    assert np.isclose(float(np.asarray(chunk_fitness[0])), float(np.asarray(genotype_fitness[0])))
    for metric_name in start_opt_command.dc_params.ga_config.observed_metrics:
        assert np.isclose(
            float(np.asarray(chunk_metrics[metric_name][0])),
            float(np.asarray(genotype_metrics[metric_name][0])),
        )
    np.testing.assert_array_equal(
        np.asarray(chunk_metrics["case_indices"][0]), np.asarray(genotype_metrics["case_indices"][0])
    )
