# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_topology_optimizer.dc.worker.optimizer import (
    OptimizerData,
    extract_results,
    initialize_optimization,
    run_epoch,
)
from toop_engine_topology_optimizer.interfaces.messages.commands import StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, DCOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.results import Strategy, TopologyPushResult


def test_extract_results(
    grid_folder: str,
) -> None:
    start_opt_command = StartOptimizationCommand(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=BatchedMEParameters(
                runtime_seconds=5,
            ),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="oberrhein")],
        optimization_id="test",
    )

    processed_gridfile_fs = DirFileSystem(str(grid_folder))
    optimizer_data, stats, initial_strategy = initialize_optimization(
        params=start_opt_command.dc_params,
        optimization_id="test123",
        static_information_files=[gf.static_information_file for gf in start_opt_command.grid_files],
        processed_gridfile_fs=processed_gridfile_fs,
    )

    assert isinstance(optimizer_data, OptimizerData)
    assert isinstance(stats[0], StaticInformationStats)
    assert len(stats) == 1
    assert isinstance(initial_strategy, Strategy)

    optimizer_data = run_epoch(optimizer_data)
    res = extract_results(optimizer_data)
    assert isinstance(res, TopologyPushResult)
