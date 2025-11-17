"""Benchmark ToOp end-to-end for a single grid file (Hydra multirun friendly).

Each Hydra run benchmarks exactly one grid defined by the `grid_file` field
in the selected grid config group. Use Hydra's multirun (-m) to sweep multiple
grid configurations or override parameters like GA runtime.

Example:
  uv run python -m benchmark.benchmark_toop --multirun \
    grid=config_grid_node_breaker ga_config.runtime_seconds=10,20 ga_config.split_subs=2,5

This runs ToOp end-to-end on the specified node breaker grid for 4 combinations of,
parameters.

Outputs:
  benchmark_summary.json stored in ${hydra.run.dir} (current working directory).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

import hydra
import logbook
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    PipelineConfig,
    copy_to_initial_topology,
    get_paths,
    perform_ac_analysis,
    prepare_importer_parameters,
    run_dc_optimization_stage,
    run_preprocessing,
)

logger = logbook.Logger("ToOp Benchmark")


class PhaseTimer:
    """Utility for timing code phases.

    Records multiple durations per phase name and summarizes count, total seconds,
    and average seconds. Use as:
        with timer.time("phase_name"):
            ...
    """

    def __init__(self) -> None:
        self.records = {}

    def time(self, name: str) -> object:
        """Context manager to time a specific phase.

        Parameters
        ----------
        name : str
            Name of the phase to be timed.

        Returns
        -------
        Context manager
            A context manager that records the time taken for the phase.
        """

        class _Ctx:
            def __enter__(_s) -> None:
                _s.start = time.perf_counter()

            def __exit__(_s, *_) -> None:  # noqa: ANN002
                self.records.setdefault(name, []).append(time.perf_counter() - _s.start)

        return _Ctx()

    def summary(self) -> dict:
        """Get a summary of the recorded timings.

        Returns
        -------
        dict
            A dictionary with phase names as keys and timing statistics as values.
        """
        return {k: {"count": len(v), "total_s": sum(v), "avg_s": (sum(v) / len(v))} for k, v in self.records.items()}


def load_res(res_path: Path) -> dict:
    """Load the result JSON file from the specified path.

    Parameters
    ----------
    res_path : Path
        Path to the result JSON file.

    Returns
    -------
    dict
        Parsed JSON content as a dictionary.
    """
    with open(res_path) as f:
        return json.load(f)


def extract_dc_quality(res: dict) -> dict:
    """Extract DC quality metrics from the result dictionary.

    Parameters
    ----------
    res : dict
        Result dictionary containing optimization results.

    Returns
    -------
    dict
        Dictionary containing extracted DC quality metrics.
    """
    best = res.get("best_topos", [])
    fitnesses = [t.get("metrics").get("fitness") for t in best if t.get("metrics")]
    splits = [t.get("metrics").get("extra_scores", {}).get("split_subs", {}) for t in best if t.get("metrics")]
    return {
        "n_best": len(best),
        "fitness_max": max(fitnesses) if fitnesses else None,
        "fitness_mean": mean(fitnesses) if fitnesses else None,
        "split_subs_mean": mean([s for s in splits if s is not None]) if splits else None,
        "descriptor_coverage_percent": res.get("descriptor_coverage_percent"),  # add via optimizer extension
        "iterations": res.get("n_iterations"),
    }


def benchmark_single_grid(
    grid: Path,
    dc_optimization_cfg: dict,
    preprocessing_parameters: PreprocessParameters,
    grid_type: str = "powsybl",
) -> dict:
    """Run benchmark for a single grid file.

    Parameters
    ----------
    grid : Path
        Path to the grid model file.
    dc_optimization_cfg : dict
        Dictionary of configuration values consumed by the DC optimization stage.
    preprocessing_parameters : PreprocessParameters
        Parameters controlling preprocessing behaviour.
    grid_type : str, optional
        Grid type identifier (e.g. 'powsybl', 'pandapower').

    Returns
    -------
    dict
        Benchmark entry containing grid path, timings, and DC quality metrics.
    """
    iteration_name = ""
    pipeline_cfg = PipelineConfig(
        root_path=grid.parent, iteration_name=iteration_name, file_name=grid.name, grid_type=grid_type
    )
    _, file_path, data_folder, optimizer_snapshot_dir = get_paths(pipeline_cfg)
    static_information_file = data_folder / pipeline_cfg.static_info_relpath
    timer = PhaseTimer()

    importer_parameters = prepare_importer_parameters(file_path, data_folder)
    importer_parameters.area_settings.cutoff_voltage = dc_optimization_cfg.get("area_settings", {}).get(
        "cutoff_voltage", 380
    )
    copy_to_initial_topology(importer_parameters.grid_model_file, data_folder, pipeline_cfg.initial_topology_subpath)

    with timer.time("preprocess"):
        run_preprocessing(
            importer_parameters,
            data_folder,
            preprocessing_parameters,
            is_pandapower_net=True if pipeline_cfg.grid_type == "pandapower" else False,
        )

    dc_cfg = dict(dc_optimization_cfg)
    dc_cfg["fixed_files"] = [str(static_information_file)]

    with timer.time("dc_optimization"):
        run_dir = run_dc_optimization_stage(dc_cfg, optimizer_snapshot_dir)

    with timer.time("ac_validation"):
        topology_paths = perform_ac_analysis(
            data_folder,
            run_dir,
            k_best_topos=5,
            pandapower_runner=(pipeline_cfg.grid_type == "pandapower"),
        )

    res_path = run_dir / "res.json"
    res = load_res(res_path)
    dc_quality = extract_dc_quality(res)

    return {
        "grid": str(grid),
        "topology_paths": str(topology_paths[0].parent) if topology_paths else None,
        "timings": timer.summary(),
        "gpu_samples": None,
        "dc_quality": dc_quality,
    }


@hydra.main(config_path="configs", config_name="benchmark.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Hydra entry point benchmarking a single grid.

    Configuration groups expected:
      grid: provides grid_file (+ optional grid_type)
      ga_config: genetic algorithm parameters
      lf_config: lightflow / solver parameters
      area_settings: voltage and area constraints

    The grid config should be provided via group `grid=config_grid_node_breaker` or similar.
    """
    logbook.StreamHandler(sys.stdout, level=cfg.get("logging_level", "INFO")).push_application()

    grid_path = Path(cfg.grid.grid_file)
    hydra_output_dirname = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    preprocess_params = PreprocessParameters(action_set_clip=2**10, enable_bb_outage=True, bb_outage_as_nminus1=False)

    dc_optimization_cfg = {
        "task_name": "benchmark" + time.strftime("_%Y%m%d_%H%M%S"),
        "double_precision": None,
        "tensorboard_dir": hydra_output_dirname + "/{task_name}",
        "stats_dir": hydra_output_dirname + "/{task_name}/stats",
        "summary_frequency": None,
        "checkpoint_frequency": None,
        "stdout": None,
        "double_limits": None,
        "num_cuda_devices": 1,
        "omp_num_threads": 1,
        "xla_force_host_platform_device_count": None,
        "output_json": "output_test.json",
        "lf_config": {
            "max_num_splits": cfg.lf_config.max_num_splits,
            "max_num_disconnections": cfg.lf_config.max_num_disconnections,
            "batch_size": cfg.lf_config.batch_size,
        },
        "ga_config": {
            "runtime_seconds": cfg.ga_config.runtime_seconds,
            "me_descriptors": [
                {"metric": "switching_distance", "num_cells": cfg.ga_config.switching_distance},
                {"metric": "split_subs", "num_cells": cfg.ga_config.split_subs},
            ],
            "n_worst_contingencies": cfg.ga_config.n_worst_contingencies,
            "random_seed": cfg.ga_config.random_seed,
            "target_metrics": [["overload_energy_n_1", 1.0], ["split_subs", 1.0]],
        },
        "area_settings": cfg.area_settings,
    }

    entry = benchmark_single_grid(grid_path, dc_optimization_cfg, preprocess_params, cfg.grid.get("grid_type", "powsybl"))
    summary_filepath = Path(hydra_output_dirname) / dc_optimization_cfg["task_name"] / "benchmark_summary.json"
    with open(summary_filepath, "w") as f:
        json.dump([entry], f, indent=2)
    # Copy res.json to next to benchmark_summary.json
    with open(summary_filepath.parent / "res.json", "w") as f:
        res_filepath = (
            Path(dc_optimization_cfg["stats_dir"].replace("{task_name}", dc_optimization_cfg["task_name"])) / "res.json"
        )
        json.dump(load_res(res_filepath), f, indent=2)
    logger.info(f"Benchmark for {grid_path} completed. Summary in {summary_filepath}")


if __name__ == "__main__":
    main()
