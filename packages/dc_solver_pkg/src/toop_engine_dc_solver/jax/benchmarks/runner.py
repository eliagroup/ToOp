# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides a framework for running benchmarks according to a yaml configuration."""

import argparse
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from multiprocessing import Process
from multiprocessing.connection import Connection

import yaml
from toop_engine_dc_solver.jax.benchmarks.benchmarks import run_benchmark
from tqdm import tqdm


def main(args: list[str]) -> None:
    """Load a benchmark config parsed from cli and run the benchmark"""
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--yaml_config", type=str, help="Path to the yaml config file")
    parser.add_argument("--output_json", type=str, help="Path to the output json file")
    args_parsed = parser.parse_args(args)

    with open(args_parsed.yaml_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    os.makedirs(os.path.dirname(args_parsed.output_json), exist_ok=True)

    # Expand grid searches in each benchmark
    benchmarks = []
    for benchmark in config["benchmarks"]:
        if "grid_search" in benchmark:
            for i, grid_search in enumerate(benchmark["grid_search"]):
                new_benchmark = deepcopy(benchmark)
                new_benchmark["hyperparameters"].update(grid_search)
                new_benchmark["grid_search_index"] = i
                benchmarks.append(new_benchmark)
        else:
            benchmarks.append(benchmark)

    results = []
    for benchmark in tqdm(benchmarks):
        os.environ["OMP_NUM_THREADS"] = str(benchmark["omp_num_threads"])
        os.environ["JAX_ENABLE_X64"] = "FALSE" if benchmark["single_precision"] else "TRUE"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in benchmark["cuda_visible_devices"]])
        if benchmark["xla_force_host_platform_device_count"] is not None:
            os.environ["XLA_FLAGS"] = (
                f"--xla_force_host_platform_device_count={benchmark['xla_force_host_platform_device_count']}"
            )

        # Run the benchmark in a clean process to avoid side-effects from previous benchmarks
        ctx = mp.get_context("spawn")
        mp.set_start_method("spawn", force=True)
        parent_conn, child_conn = ctx.Pipe()
        process = Process(target=run_benchmark_process, args=(child_conn, benchmark))
        process.start()
        res = parent_conn.recv()
        process.join()
        results.append(res)
        with open(args_parsed.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def run_benchmark_process(conn: Connection, benchmark: dict) -> None:
    """Run a single benchmark, assuming the environment variables are already set.

    Designed to be run in a separate process, hence it communicates the results through a connection

    Parameters
    ----------
    conn: Connection
        A mp.Connection to send the results through. The other end can expect exactly one dict
        being sent through the connection
    benchmark: dict
        The configuration for this benchmark, loaded from yaml
    """
    try:
        res = run_benchmark(benchmark)
        conn.send(res)
    except Exception as exception:
        conn.send({"error": str(exception)})
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
