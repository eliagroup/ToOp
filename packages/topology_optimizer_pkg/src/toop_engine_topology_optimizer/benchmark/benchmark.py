"""Run benchmark.

This module provides functionality to run benchmark tasks based on a provided configuration.
It supports grid search for sweeping through the values of number of CUDA devices or a single parameter
not defined inside `lf_config` and `ga_config`. Multiple parameter sweeps are not supported yet.

Functions
---------
main(cfg: DictConfig) -> None
"""

import json
import multiprocessing as mp
from copy import deepcopy
from multiprocessing import Process

import hydra
import logbook
from hydra import compose
from omegaconf import DictConfig
from toop_engine_topology_optimizer.benchmark.benchmark_utils import run_task_process, set_environment_variables
from tqdm import tqdm

logger = logbook.Logger(__name__)


@hydra.main(config_path="configs", config_name="ms_benchmark.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:  # FIXME: Does not work for ToOp engine
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
        set_environment_variables(benchmark)

        # Run the benchmark in a clean process to avoid side-effects from previous benchmarks
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = Process(target=run_task_process, args=(benchmark, child_conn))
        process.start()
        res = parent_conn.recv()
        process.join()
        results.append(res)

    #  Log the results as a json file
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
