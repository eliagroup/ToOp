"""Benchmark ToOp end-to-end by running it on a directory of grid files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

import hydra
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    PipelineConfig,
    copy_to_initial_topology,
    get_paths,
    logger,
    perform_ac_analysis,
    prepare_importer_parameters,
    remove_unsupported_elements_and_save,
    run_dc_optimization_stage,
    run_preprocessing,
)


class PhaseTimer:
    """Utility class for timing different phases of the benchmarking process."""

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


def benchmark_grids(
    grids: list[Path],
    output_dir: Path,
    dc_optimization_cfg: dict,
    preprocessing_parameters: PreprocessParameters,
    grid_type: str = "powsybl",
) -> None:
    """Benchmark ToOp end-to-end on a list of grid files.

    Parameters
    ----------
    grids : list[Path]
        List of paths to grid files to benchmark.
    output_dir : Path
        Path to the output directory where results will be saved.
    dc_optimization_cfg : dict
        Configuration dictionary for the DC optimization stage.
    preprocessing_parameters : PreprocessParameters
        Parameters for the preprocessing stage.
    grid_type : str, optional
        Type of the grid files, by default `powsybl`.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for grid in grids:
        iteration_name = ""

        pipeline_cfg = PipelineConfig(
            root_path=grid.parent, iteration_name=iteration_name, file_name=grid.name, grid_type=grid_type
        )

        # Set up dc_optimisation_configs
        _, file_path, data_folder, optimizer_snapshot_dir = get_paths(pipeline_cfg)
        static_information_file = data_folder / pipeline_cfg.static_info_relpath

        timer = PhaseTimer()

        with timer.time("pre_modify"):
            modified_file_path = remove_unsupported_elements_and_save(
                file_path=file_path,
                data_folder=data_folder,
                pandapower_net=True if pipeline_cfg.grid_type == "pandapower" else False,
            )

        importer_parameters = prepare_importer_parameters(
            modified_file_path,
            data_folder,
        )
        importer_parameters.area_settings.cutoff_voltage = dc_optimization_cfg.get("area_settings", {}).get(
            "cutoff_voltage", 380
        )  # in kV
        # Ignore on timing
        copy_to_initial_topology(importer_parameters.grid_model_file, data_folder, pipeline_cfg.initial_topology_subpath)

        with timer.time("preprocess"):
            run_preprocessing(
                importer_parameters,
                data_folder,
                preprocessing_parameters,
                is_pandapower_net=True if pipeline_cfg.grid_type == "pandapower" else False,
            )

        dc_optimization_cfg = dict(dc_optimization_cfg)
        dc_optimization_cfg["fixed_files"] = [str(static_information_file)]

        with timer.time("dc_optimization"):
            run_dir = run_dc_optimization_stage(
                dc_optimization_cfg,
                optimizer_snapshot_dir,
            )

        with timer.time("ac_validation"):
            topology_paths = perform_ac_analysis(
                data_folder,
                run_dir,
                k_best_topos=0,
                pandapower_runner=(pipeline_cfg.grid_type == "pandapower"),
            )

        res_json = max(list((modified_file_path.parent / "optimizer_snapshot").glob("run_*/res.json")))
        res = load_res(res_json)
        dc_quality = extract_dc_quality(res)

        bench_entry = {
            "grid": str(grid),
            "topology_paths": str(topology_paths[0].parent) if topology_paths else None,
            "timings": timer.summary(),
            "gpu_samples": None,  # TODO: add gpu usage
            "dc_quality": dc_quality,
        }
        all_results.append(bench_entry)

    logger.info("Benchmark completed.")

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


def _single_file_or_dir(cfg: DictConfig) -> list[Path]:
    """Get a list of grid file paths from a single file or directory.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing either 'grid_file' or 'grid_dir'.

    Returns
    -------
    list[Path]
        List of paths to grid files.
    """
    grids = []
    if cfg.get("grid_file") is not None:
        grids.append(Path(cfg.grid_file))
    elif cfg.get("grid_dir") is not None:
        grid_dir = Path(cfg.grid_dir)
        try:
            file_name = cfg.get("file_name")
        except AttributeError:
            logger.info("No file_name specified in config, defaulting to 'grid.xiidm'")
            file_name = "grid.xiidm"
        # Search one level deep for files matching the specified file_name
        ext = f"**/{file_name}"
        grids.extend(sorted(grid_dir.glob(ext)))
        if not grids:
            raise FileNotFoundError(f"No grid files named '{file_name}' found in directory: {grid_dir}")
    else:
        raise ValueError("Either 'grid_file' or 'grid_dir' must be specified in the configuration.")
    return grids


@hydra.main(config_path="configs", config_name="benchmark_configs.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the ToOp benchmark based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing benchmark settings.
    """
    grids = _single_file_or_dir(cfg)
    output_dir = Path(cfg.output_dir)

    preprocess_params = PreprocessParameters(action_set_clip=2**10, enable_bb_outage=True, bb_outage_as_nminus1=False)

    dc_optmization_cfg = {
        "task_name": "benchmark" + time.strftime("_%Y%m%d_%H%M%S"),
        "double_precision": None,
        "tensorboard_dir": str(output_dir) + "/{task_name}",
        "stats_dir": str(output_dir) + "/{task_name}",
        "summary_frequency": None,
        "checkpoint_frequency": None,
        "stdout": None,
        "double_limits": None,
        "num_cuda_devices": 1,
        "omp_num_threads": 1,
        "xla_force_host_platform_device_count": None,
        "output_json": "output_test.json",
        "lf_config": {
            "distributed": False,
            "max_num_splits": 10,
        },
        "ga_config": {
            "runtime_seconds": 30,
            "me_descriptors": [{"metric": "split_subs", "num_cells": 4}, {"metric": "switching_distance", "num_cells": 20}],
            "observed_metrics": ["overload_energy_n_1", "split_subs"],
            "n_worst_contingencies": 2,
            "random_seed": 6543345,
        },
        "area_settings": {
            "cutoff_voltage": cfg.get("cutoff_voltage", 380),  # in kV
        },
    }
    benchmark_grids(grids, output_dir, dc_optmization_cfg, preprocess_params)


if __name__ == "__main__":
    main()
