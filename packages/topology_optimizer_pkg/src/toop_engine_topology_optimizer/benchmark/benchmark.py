"""Run benchmark.

This module provides functionality to run benchmark tasks based on a provided configuration.
It supports grid search for sweeping through the values of number of CUDA devices or a single parameter
not defined inside `lf_config` and `ga_config`. Multiple parameter sweeps are not supported yet.

Functions
---------
main(cfg: DictConfig) -> None

run_task_process(conn: Connection, cfg: DictConfig) -> None
"""

import json
import multiprocessing as mp
import os
from copy import deepcopy
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Optional

import hydra
import logbook
from hydra import compose
from omegaconf import DictConfig
from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.dc.main import main as opt_main
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    LoadflowSolverParameters,
)
from tqdm import tqdm

logger = logbook.Logger(__name__)


@hydra.main(config_path="configs", config_name="ms_benchmark.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run benchmark tasks based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing benchmark tasks and output settings.

    Returns
    -------
    None

    Notes
    -----
    The function performs the following steps:
    1. Iterates over the benchmark tasks specified in the configuration.
    2. Checks for grid search configurations and generates new configurations for each value in the grid.
    3. Sets environment variables for each benchmark task.
    4. Runs each benchmark task in a separate process to avoid side-effects.
    5. Collects the results and logs them into a JSON file specified in the configuration.

    Grid search currently supports sweeping through the values of number of CUDA devices or
    a single parameter not defined inside `lf_config` and `ga_config`. Multiple parameter sweeps are not supported yet.
    """
    benchmarks = []
    for task_cfg in cfg.benchmark_tasks:
        task_cfg_composed = compose(config_name=task_cfg.task_config)["task"]

        # check for grid searches
        if "grid_search" in task_cfg_composed:
            # Note: the grid search only works for sweeping through the values of
            # number of cuda devices or a single parameter not defined inside lf_config and ga_config.
            # The support to sweep multiple paramters are not there yet.
            key = next(iter(task_cfg_composed["grid_search"].keys()))
            values = next(iter(task_cfg_composed["grid_search"].values()))
            for val in values:
                new_config = deepcopy(task_cfg_composed)
                new_config[key] = val

                #  edit task name
                new_config["task_name"] = new_config["task_name"] + "_" + key + "_" + str(val)
                benchmarks.append(new_config)
        else:
            benchmarks.append(task_cfg_composed)

    results = []
    for benchmark in tqdm(benchmarks):
        os.environ["OMP_NUM_THREADS"] = str(benchmark["omp_num_threads"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(0, benchmark["num_cuda_devices"])])
        if benchmark["xla_force_host_platform_device_count"] is not None:
            os.environ["XLA_FLAGS"] = (
                f"--xla_force_host_platform_device_count={benchmark['xla_force_host_platform_device_count']}"
            )

        # Run the benchmark in a clean process to avoid side-effects from previous benchmarks
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = Process(target=run_task_process, args=(child_conn, benchmark))
        process.start()
        res = parent_conn.recv()
        process.join()
        results.append(res)

    #  Log the results as a json file
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_task_process(cfg: DictConfig, conn: Optional[Connection] = None) -> Optional[dict]:
    """Execute a task process based on the provided configuration.

    This function initializes parameters for a genetic algorithm and a load flow solver
    from the given configuration, constructs command-line arguments, and runs the main
    optimization function. If a connection object is provided, the result is sent through
    the connection; otherwise, the result is returned directly.
    Args:
        cfg (DictConfig): Configuration object containing parameters for the task.
        conn (Connection, optional): Connection object for inter-process communication. Defaults to None.

    Returns
    -------
        None
    """
    logger.info(f"************Experiment name: {cfg['task_name']}****************")

    ga_params = BatchedMEParameters(**{k: v for k, v in cfg.ga_config.items() if v is not None})
    lf_params = LoadflowSolverParameters(**{k: v for k, v in cfg.lf_config.items() if v is not None})

    if "{task_name}" in cfg.tensorboard_dir:
        cfg.tensorboard_dir = cfg.tensorboard_dir.replace("{task_name}", cfg.task_name)

    if "{task_name}" in cfg.stats_dir:
        cfg.stats_dir = cfg.stats_dir.replace("{task_name}", cfg.task_name)

    # Combine all config data into the CLIArgs Pydantic model
    cli_args = {
        "ga_config": ga_params,
        "lf_config": lf_params,
        "fixed_files": tuple(cfg.fixed_files) if cfg.fixed_files is not None else None,
        "double_precision": cfg.double_precision if cfg.double_precision is not None else None,
        "tensorboard_dir": cfg.tensorboard_dir if cfg.tensorboard_dir is not None else None,
        "stats_dir": cfg.stats_dir if cfg.stats_dir is not None else None,
        "summary_frequency": cfg.summary_frequency if cfg.summary_frequency is not None else None,
        "checkpoint_frequency": cfg.checkpoint_frequency if cfg.checkpoint_frequency is not None else None,
        "double_limits": tuple(cfg.double_limits) if cfg.double_limits is not None else None,
    }

    # Remove None values
    cli_args = {key: value for key, value in cli_args.items() if value is not None}
    cli_args = CLIArgs(**cli_args)

    # If the connection is None, run the task without sending the result
    if conn is None:
        res = opt_main(cli_args)
        return res

    try:
        res = opt_main(cli_args)
        conn.send(res)
        del res
    except Exception as exception:  # pylint: disable=broad-exception-caught
        conn.send({"error": str(exception)})
    conn.close()
    return None


if __name__ == "__main__":
    main()
