# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_topology_optimizer.dc_bruteforce.worker import optimization_loop
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    DescriptorDef,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import (
    OptimizationStartedHeartbeat,
    OptimizationStatsHeartbeat,
)
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    ResultUnion,
    TopologyPushResult,
)


def _ga_config() -> BatchedMEParameters:
    return BatchedMEParameters(
        runtime_seconds=30,
        iterations_per_epoch=32,
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=2),
            DescriptorDef(metric="switching_distance", num_cells=45),
        ),
    )


def test_optimization_loop_emits_start_and_stop(grid_folder: str) -> None:
    results: list[ResultUnion] = []
    heartbeats = []

    optimization_loop(
        dc_params=DCOptimizerParameters(
            summary_frequency=1,
            check_command_frequency=1,
            ga_config=_ga_config(),
            loadflow_solver_config=LoadflowSolverParameters(max_num_splits=1, max_num_disconnections=0),
        ),
        grid_files=[GridFile(framework=Framework.PANDAPOWER, grid_folder="case14")],
        send_result_fn=results.append,
        flush_result_fn=lambda: None,
        send_heartbeat_fn=heartbeats.append,
        optimization_id="test-bruteforce",
        processed_gridfile_fs=DirFileSystem(str(grid_folder)),
    )

    assert any(isinstance(heartbeat, OptimizationStartedHeartbeat) for heartbeat in heartbeats)
    assert any(isinstance(heartbeat, OptimizationStatsHeartbeat) for heartbeat in heartbeats)
    assert isinstance(results[0], OptimizationStartedResult)
    assert isinstance(results[-1], OptimizationStoppedResult)
    assert results[-1].reason == "converged"
    assert all(
        isinstance(result, (OptimizationStartedResult, OptimizationStoppedResult, TopologyPushResult)) for result in results
    )
