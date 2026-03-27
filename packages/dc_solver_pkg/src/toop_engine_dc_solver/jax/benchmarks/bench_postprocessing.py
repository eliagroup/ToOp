# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Postprocessing benchmarker functions

Provides some helper functions and a command line utility to benchmark DC and
AC power flows on pandapower using simple ray parallelization.
"""

import datetime
import json
import time
from pathlib import Path
from typing import cast

import jax
import tyro
from beartype.typing import Literal, Optional, TypeAlias
from pydantic import BaseModel, PositiveInt
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import random_topology
from toop_engine_dc_solver.jax.types import ActionIndexComputations
from toop_engine_dc_solver.postprocess.abstract_runner import (
    AbstractLoadflowRunner,
)
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    PandapowerRunner,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, load_nminus1_definition
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set

LoadflowType: TypeAlias = Literal["ac", "dc", "ac_cross_coupler", "dc_cross_coupler"]


def compute_loadflows(
    runner: AbstractLoadflowRunner,
    topologies: list[list[int]],
    method: LoadflowType,
) -> tuple[list[int], list[int]]:
    """Compute the loadflows for the given topologies

    This can involve multiple methods, e.g. AC and DC loadflows. If no method is passed,
    this does nothing.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        The runner object
    topologies : list[list[int]]
        The topologies to compute the loadflows for. Includes split busbars but no disconnections.
    method : LoadflowType
        The method to use.

    Returns
    -------
    list[int]:
        The number of converged loadflows for each topology.
    list[int]:
        The number of failed loadflows for each topology.
    """
    converged = []
    failures = []
    nminus1_definition = runner.get_nminus1_definition()
    for action in topologies:
        if method == "dc":
            res = runner.run_dc_loadflow(action, [])
            _, _, success = extract_solver_matrices_polars(res, nminus1_definition, 0)
            converged.append(success.sum().item())
            failures.append((~success).sum().item())
        elif method == "ac":
            res = runner.run_ac_loadflow(action, [])
            _, _, success = extract_solver_matrices_polars(res, nminus1_definition, 0)
            converged.append(success.sum().item())
            failures.append((~success).sum().item())
        elif method == "ac_cross_coupler":
            res = runner.run_ac_cross_coupler_loadflow(action)
            converged.append(res.results[0].converged.sum().item())
            failures.append((~res.results[0].converged).sum().item())
        elif method == "dc_cross_coupler":
            res = runner.run_dc_cross_coupler_loadflow(action)
            converged.append(res.results[0].converged.sum().item())
            failures.append((~res.results[0].converged).sum().item())
    return converged, failures


def setup_benchmark(
    data_folder: Path,
    n_topologies: int,
    n_substations_split: int,
    n_processes_per_topology: int,
    batch_size_per_topology: Optional[int],
    framework: Literal["pandapower", "powsybl"] = "pandapower",
    seed: int = 0,
) -> tuple[AbstractLoadflowRunner, list[list[int]]]:
    """Set up a benchmark with dummy data

    Parameters
    ----------
    data_folder : Path
        Path to the preprocessing folder containing the persisted action set and N-1 definition
    n_topologies : int
        Number of topologies to generate
    n_substations_split : int
        Number of substations to split in each topology
    n_processes_per_topology : int
        Number of processes to run per topology
    batch_size_per_topology : Optional[int]
        The batch size to use for running the N-1 analysis.
    framework : Literal["pandapower", "powsybl"]
        Framework to use
    seed : int
        Seed for the random number generator

    Returns
    -------
    AbstractLoadflowRunner
        A loadflow runner which is ready to run loadflows
    list[list[int]]
        A list of topologies of length n_topologies with actions to try
    """
    grid_path = data_folder / PREPROCESSING_PATHS[f"grid_file_path_{framework}"]
    if framework == "pandapower":
        runner = PandapowerRunner(n_processes=n_processes_per_topology, batch_size=batch_size_per_topology)
    else:
        runner = PowsyblRunner(n_processes=n_processes_per_topology, batch_size=batch_size_per_topology)
    runner.load_base_grid(grid_path)
    action_set, nminus1_definition = load_benchmark_aux_data(data_folder)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_definition)

    topology = sample_topologies(
        data_folder=data_folder,
        n_topologies=n_topologies,
        n_substations_split=n_substations_split,
        seed=seed,
    )
    actions = [
        [int(action) for action in topology_row if action < len(action_set.local_actions)]
        for topology_row in topology.action
    ]

    return runner, actions


def load_benchmark_aux_data(data_folder: Path) -> tuple[ActionSet, Nminus1Definition]:
    """Load the persisted preprocessing artifacts needed for postprocessing benchmarks."""
    action_set = load_action_set(
        data_folder / PREPROCESSING_PATHS["action_set_file_path"],
        data_folder / PREPROCESSING_PATHS["action_set_diff_path"],
    )
    nminus1_definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    return action_set, nminus1_definition


def sample_topologies(
    data_folder: Path,
    n_topologies: int,
    n_substations_split: int,
    seed: int,
) -> ActionIndexComputations:
    """Sample topology actions using the persisted static information action set."""
    static_information = load_static_information(data_folder / PREPROCESSING_PATHS["static_information_file_path"])
    return cast(
        ActionIndexComputations,
        random_topology(
            rng_key=jax.random.PRNGKey(seed),
            branch_action_set=static_information.dynamic_information.action_set,
            limit_n_subs=n_substations_split,
            batch_size=n_topologies,
            topo_vect_format=False,
        ),
    )


def run_benchmark(
    runner: AbstractLoadflowRunner,
    topologies: list[list[int]],
    method: LoadflowType = "dc",
) -> tuple[int, int, float]:
    """Run the benchmark.

    Parameters
    ----------
    runner : AbstractLoadflowRunner
        A loadflow runner which is ready to run loadflows
    topologies : list[list[int]]
        A list of topologies of length n_topologies with actions to try
    method : LoadflowType
        Method to use

    Returns
    -------
    int
        Number of loadflows run
    int
        Number of successful loadflows
    float
        Total time taken in seconds
    """
    start_time = time.time()
    converged, failed = compute_loadflows(
        runner,
        topologies,
        method=method,
    )
    total_time = time.time() - start_time

    n_loadflows = sum(converged) + sum(failed)
    n_successful_loadflows = sum(converged)

    return n_loadflows, n_successful_loadflows, total_time


class Args(BaseModel):
    """Command line arguments to the postprocessing benchmarker.

    This benchmarker runs the postprocessing pipeline on a given grid and network data
    file and measures the time it took to compute.
    """

    data_folder: str
    """Path to the preprocessing folder containing the stored action set and N-1 definition"""

    framework: Literal["pandapower", "powsybl"]
    """Framework to use, which also determines the grid file path inside the preprocessing folder"""

    n_topologies: PositiveInt
    """Number of topologies to generate"""

    n_substations_split: PositiveInt = 3
    """Number of substations to split in each topology"""

    n_processes_per_topology: PositiveInt = 2
    """Number of processes to run per topology, i.e. how many workers to spawn for the N-1 analysis
    of a single topology"""

    batch_size_per_topology: Optional[PositiveInt] = None
    """The batch size to use for running the N-1 analysis. If None, the batch size is set to the
    number of N-1 cases divided by the number of processes"""

    seed: int = 0
    """Seed for the random number generator"""

    method: LoadflowType = "dc"
    """The loadflow method to use"""

    result_file: str = "results.json"
    """Where to save the results"""


def main(args: Args) -> None:
    """Execute a benchmark and saves the results to a json file"""
    setup_start = time.time()
    runner, topologies = setup_benchmark(
        data_folder=Path(args.data_folder),
        n_topologies=args.n_topologies,
        n_substations_split=args.n_substations_split,
        n_processes_per_topology=args.n_processes_per_topology,
        batch_size_per_topology=args.batch_size_per_topology,
        framework=args.framework,
        seed=args.seed,
    )
    setup_time = time.time() - setup_start

    n_loadflows, n_success, runtime = run_benchmark(runner, topologies, args.method)

    result_file = Path(args.result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_loadflows": n_loadflows,
                "n_success": n_success,
                "time": runtime,
                "setup_time": setup_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "args": args.model_dump(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
