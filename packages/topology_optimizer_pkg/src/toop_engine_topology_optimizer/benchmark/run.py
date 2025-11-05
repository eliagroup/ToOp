"""Set up and run a benchmark task based on a provided configuration file.

It configures environment variables for OpenMP, CUDA, and XLA, and logs the results
to a JSON file.

Functions
---------
main(cfg: DictConfig) -> None
    Main function to set environment variables and run a task based on the provided configuration.

Notes
-----
This module supports the multi-runs feature of Hydra,
allowing you to run multiple configurations in parallel or sequentially.
To use the multi-runs feature, you can specify multiple configurations in the command line or in the configuration file.

Example
-------
To run multiple configurations in parallel, you can use the following command:
    python run.py -m lf_config.batch_size=8,16,32 ga_config.population_size=100,200,300

This command will run 9 configurations sequentially,
with all possible combinations of the specified values for `batch_size` and `population_size`.
"""

import json

import hydra
import logbook
from omegaconf import DictConfig
from toop_engine_topology_optimizer.benchmark.benchmark_utils import run_task_process, set_environment_variables

logger = logbook.Logger(__name__)


@hydra.main(config_path="configs/task", config_name="test.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Set environment variables and run a task based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing the following keys:
        - omp_num_threads (int): Number of OpenMP threads.
        - num_cuda_devices (int): Number of CUDA devices.
        - xla_force_host_platform_device_count (Optional[int]): Number of XLA host platform devices.
        - output_json (str): Path to the output JSON file.

    Returns
    -------
    None
    """
    #  Set the env variables
    set_environment_variables(cfg)

    # Run the task
    res = run_task_process(cfg)

    #  Log the results as a json file
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
