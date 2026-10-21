# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
# ruff: noqa: T201

"""Benchmark Powsybl AC contingency analysis for batch sizes and process counts."""

import argparse
import os
import threading
import time
from pathlib import Path

import psutil
import pypowsybl
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_dc_solver.preprocess.network_data import load_lf_params
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    load_nminus1_definition,
)


def create_branch_contingency_definition(net: pypowsybl.network.Network) -> Nminus1Definition:
    """Create an N-1 definition that monitors and outages every branch.

    Parameters
    ----------
    net : pypowsybl.network.Network
        Network whose lines and transformers should be monitored and outaged.

    Returns
    -------
    Nminus1Definition
        An N-1 definition containing one outage contingency per branch and a base case.
    """
    branches = [
        ("LINE", net.get_lines(attributes=["name"])),
        ("TWO_WINDINGS_TRANSFORMER", net.get_2_windings_transformers(attributes=["name"])),
        ("THREE_WINDINGS_TRANSFORMER", net.get_3_windings_transformers(attributes=["name"])),
    ]
    branch_elements = [
        GridElement(id=branch_id, name=row.name or "", type=branch_type, kind="branch")
        for branch_type, dataframe in branches
        for branch_id, row in dataframe.iterrows()
    ]
    monitored_elements = [MonitoredElement(**branch.model_dump()) for branch in branch_elements[:1000]]
    contingencies = [
        Contingency(id="BASECASE", name="BASECASE", elements=[]),
        *[Contingency(id=branch.id, name=branch.name, elements=[branch]) for branch in branch_elements[:4000]],
    ]
    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        id_type="powsybl",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the AC contingency benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grid_input",
        type=Path,
        help="Path to a Powsybl-readable grid file or preprocessed data folder containing grid.xiidm.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 500],
        help="Contingency batch sizes to benchmark; non-positive values run unbatched (default: 1 10 50 100 500).",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        nargs="+",
        default=[4],
        help="Positive process counts to benchmark (default: 4).",
    )
    return parser.parse_args()


def load_benchmark_inputs(
    grid_input: Path,
) -> tuple[pypowsybl.network.Network, Nminus1Definition, pypowsybl.loadflow.Parameters | None]:
    """Load grid, N-1 definition, and loadflow parameters for a benchmark.

    Parameters
    ----------
    grid_input : Path
        A Powsybl-readable grid file or a preprocessed data folder containing the
        configured grid, N-1 definition, and loadflow parameter files.

    Returns
    -------
    tuple[pypowsybl.network.Network, Nminus1Definition, pypowsybl.loadflow.Parameters | None]
        Loaded network, stored or generated N-1 definition, and stored loadflow
        parameters. Inputs without stored parameters use the Powsybl defaults.

    Raises
    ------
    FileNotFoundError
        If the grid file or a required preprocessed artifact is missing.
    """
    if grid_input.is_dir():
        grid_file = grid_input / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        nminus1_definition_file = grid_input / PREPROCESSING_PATHS["nminus1_definition_file_path"]
        loadflow_parameters_file = grid_input / PREPROCESSING_PATHS["loadflow_parameters_file_path"]
        missing_files = [file_path for file_path in (grid_file, nminus1_definition_file) if not file_path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Preprocessed data folder is missing: {', '.join(map(str, missing_files))}")
        nminus1_definition = load_nminus1_definition(nminus1_definition_file)
        lf_params = load_lf_params(loadflow_parameters_file) if loadflow_parameters_file.is_file() else None
    else:
        grid_file = grid_input
        if not grid_file.is_file():
            raise FileNotFoundError(f"Grid file does not exist: {grid_file}")
        nminus1_definition = None
        lf_params = None

    net = pypowsybl.network.load(str(grid_file))
    if nminus1_definition is None:
        nminus1_definition = create_branch_contingency_definition(net)
    return net, nminus1_definition, lf_params


def get_process_tree_rss_bytes(process: psutil.Process) -> int:
    """Return the combined resident memory of a process and all descendants.

    Parameters
    ----------
    process : psutil.Process
        Root process of the tree to measure.

    Returns
    -------
    int
        Combined resident set size in bytes. Memory used by threads is included in
        their owning process and must not be counted separately.
    """
    rss_bytes = 0
    for tree_process in [process, *process.children(recursive=True)]:
        try:
            rss_bytes += tree_process.memory_info().rss
        except psutil.Error:
            continue
    return rss_bytes


class ProcessTreeMemorySampler:
    """Sample the peak resident memory of the current process tree."""

    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        sample_interval_seconds : float, default=0.05
            Delay between process-tree RSS samples.
        """
        self.process = psutil.Process(os.getpid())
        self.sample_interval_seconds = sample_interval_seconds
        self.peak_rss_bytes = get_process_tree_rss_bytes(self.process)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        """Start collecting process-tree memory samples."""
        self.thread.start()

    def stop(self) -> int:
        """Stop sampling and return the observed peak RSS in bytes."""
        self.stop_event.set()
        self.thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, get_process_tree_rss_bytes(self.process))
        return self.peak_rss_bytes

    def _sample(self) -> None:
        """Record the largest process-tree RSS until stopped."""
        while not self.stop_event.wait(self.sample_interval_seconds):
            self.peak_rss_bytes = max(self.peak_rss_bytes, get_process_tree_rss_bytes(self.process))


def main() -> None:
    """Run AC contingency analysis benchmarks for every requested parameter combination."""
    args = parse_args()
    if any(n_processes <= 0 for n_processes in args.n_processes):
        raise ValueError("Process counts must be positive integers.")

    net, nminus1_definition, lf_params = load_benchmark_inputs(args.grid_input)
    print(f"Loaded {args.grid_input} with {len(nminus1_definition.contingencies) - 1} contingencies.")
    print("n_processes,batch_size,elapsed_seconds,peak_process_tree_rss_mib,converged_contingencies,branch_result_rows")
    for n_processes in args.n_processes:
        for requested_batch_size in args.batch_sizes:
            batch_size = requested_batch_size if requested_batch_size > 0 else None
            started_at = time.perf_counter()
            memory_sampler = ProcessTreeMemorySampler()
            memory_sampler.start()
            try:
                results = get_ac_loadflow_results(
                    net,
                    nminus1_definition,
                    job_id=f"n_processes_{n_processes}_batch_size_{batch_size}",
                    batch_size=batch_size,
                    n_processes=n_processes,
                    lf_params=lf_params,
                )
            finally:
                peak_rss_mib = memory_sampler.stop() / 1024**2
            elapsed_seconds = time.perf_counter() - started_at
            converged_count = results.converged.collect().height
            branch_result_count = results.branch_results.collect().height
            print(
                f"{n_processes},{batch_size},{elapsed_seconds:.3f},{peak_rss_mib:.1f},"
                f"{converged_count},{branch_result_count}"
            )


if __name__ == "__main__":
    main()
