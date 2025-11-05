"""Assess algorithm performance across multiple benchmark runs.

Usage (Hydra from repo root):
  uv run python -m benchmark.assess_benchmarks root=/workspaces/ToOp/data \
      save=/workspaces/ToOp/data/aggregate_report.json print=true

Or inside package directory:
  uv run python benchmark/assess_benchmarks.py root=/workspaces/ToOp/data

Hydra config file: benchmark/configs/assess.yaml
"""

from __future__ import annotations

import json
import statistics as stats
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

# Disable beartype claw for this module to avoid decorating Hydra's generated main wrapper
__beartype__ = False

# Hydra imports
import hydra
from omegaconf import DictConfig

# Import shared logger
from toop_engine_topology_optimizer.benchmark.benchmark_utils import logger

TIMING_PHASES = ["pre_modify", "preprocess", "dc_optimization", "ac_validation"]

# Accept ints and floats (beartype strictness) and None values
NumericIterable = Iterable[Real | None]


def _safe_mean(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.mean(v) if v else None


def _safe_median(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.median(v) if v else None


def _safe_stdev(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.pstdev(v) if len(v) > 0 else None


def discover_benchmark_summaries(root: Path) -> list[Path]:
    """Discover all benchmark_summary.json files under the given root directory.

    Parameters
    ----------
    root : Path
        Root directory to search for benchmark summary files.

    Returns
    -------
    list[Path]
        List of paths to benchmark_summary.json files.
    """
    return [p for p in root.rglob("benchmark_summary.json")]


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON data from a file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Loaded JSON data as a dictionary.
    """
    with open(path) as f:
        return json.load(f)


def extract_runs(summary_file: Path) -> list[dict]:
    """Extract benchmark run entries from a summary file.

    Parameters
    ----------
    summary_file : Path
        Path to the benchmark summary JSON file.

    Returns
    -------
    list[dict]
        List of run entry dictionaries.
    """
    data = load_json(summary_file)
    if isinstance(data, list):
        return data
    # If single dict
    return [data]


def find_optimizer_res(run_entry: dict) -> list[dict]:
    """Find optimizer result files (res.json) associated with a benchmark run entry.

    Parameters
    ----------
    run_entry : dict
        Benchmark run entry containing grid path information.

    Returns
    -------
    list[dict]
        List of optimizer result dictionaries loaded from res.json files.
    """
    # Try topology_paths or grid parent
    res_list: list[dict] = []
    grid_path = Path(run_entry.get("grid")) if run_entry.get("grid") else None
    if grid_path and grid_path.exists():
        # optimizer_snapshot is expected under modified grid parent or sibling folder; search upward
        base_dir = grid_path.parent
        for res_json in base_dir.rglob("optimizer_snapshot/run_*/res.json"):
            try:
                res_list.append(load_json(res_json))
            except Exception as e:
                logger.info("Failed to load optimizer result from %s: %s", res_json, e)
    return res_list


def aggregate(run_entries: list[dict]) -> dict:
    """Aggregate benchmark metrics across multiple run entries.

    Parameters
    ----------
    run_entries : list[dict]
        List of benchmark run entry dictionaries.

    Returns
    -------
    dict
        Aggregated metrics including fitness statistics, timings, efficiency, and stability flags.
    """
    fitness_max_all = []
    fitness_mean_all = []
    split_subs_mean_all = []
    n_best_all = []

    timing_totals = {phase: [] for phase in TIMING_PHASES}
    timing_avgs = {phase: [] for phase in TIMING_PHASES}

    iterations_all = []

    for e in run_entries:
        dcq = e.get("dc_quality", {})
        if dcq:
            fitness_max_all.append(dcq.get("fitness_max"))
            fitness_mean_all.append(dcq.get("fitness_mean"))
            split_subs_mean_all.append(dcq.get("split_subs_mean"))
            n_best_all.append(dcq.get("n_best"))
            iterations_all.append(dcq.get("iterations"))
        timings = e.get("timings", {})
        for phase in TIMING_PHASES:
            ph = timings.get(phase)
            if ph:
                timing_totals[phase].append(ph.get("total_s"))
                timing_avgs[phase].append(ph.get("avg_s"))

    def coeff_var(values: list[float]) -> float | None:
        v = [x for x in values if x is not None]
        if len(v) < 2:
            return None
        m = stats.mean(v)
        if m == 0:
            return None
        return stats.pstdev(v) / m

    report = {
        "n_runs": len(run_entries),
        "fitness_max": {
            "mean": _safe_mean(fitness_max_all),
            "median": _safe_median(fitness_max_all),
            "stdev": _safe_stdev(fitness_max_all),
            "cv": coeff_var(fitness_max_all),
            "best": max(fitness_max_all) if fitness_max_all else None,
        },
        "fitness_mean": {
            "mean": _safe_mean(fitness_mean_all),
            "median": _safe_median(fitness_mean_all),
            "stdev": _safe_stdev(fitness_mean_all),
            "cv": coeff_var(fitness_mean_all),
        },
        "split_subs_mean": {
            "mean": _safe_mean(split_subs_mean_all),
            "median": _safe_median(split_subs_mean_all),
            "stdev": _safe_stdev(split_subs_mean_all),
        },
        "n_best": {
            "mean": _safe_mean(n_best_all),
            "median": _safe_median(n_best_all),
            "min": min([x for x in n_best_all if x is not None], default=None),
            "max": max([x for x in n_best_all if x is not None], default=None),
        },
        "iterations": {
            "mean": _safe_mean(iterations_all),
            "median": _safe_median(iterations_all),
            "stdev": _safe_stdev(iterations_all),
        },
        "timings": {
            phase: {
                "total_mean_s": _safe_mean(timing_totals[phase]),
                "total_median_s": _safe_median(timing_totals[phase]),
                "avg_mean_s": _safe_mean(timing_avgs[phase]),
                "cv": coeff_var(timing_totals[phase]),
            }
            for phase in TIMING_PHASES
        },
    }

    return report


def build_detailed_runs(summary_files: list[Path]) -> list[dict]:
    """Build detailed run entries by enriching summary data with optimizer results.

    Parameters
    ----------
    summary_files : list[Path]
        List of paths to benchmark summary files.

    Returns
    -------
    list[dict]
        List of enriched run entry dictionaries with optimizer details.
    """
    runs: list[dict] = []
    for sf in summary_files:
        for entry in extract_runs(sf):
            # attach extra optimizer details (aggregate of res.json files)
            optimizer_results = find_optimizer_res(entry)
            if optimizer_results:
                # Collect deeper metrics if available
                fitness_progress = []
                iterations = []
                for res in optimizer_results:
                    best_topos = res.get("best_topos", [])
                    if best_topos:
                        fitness_values = [t.get("metrics", {}).get("fitness") for t in best_topos if t.get("metrics")]
                        if fitness_values:
                            fitness_progress.append(max(fitness_values))
                    if res.get("n_iterations") is not None:
                        iterations.append(res.get("n_iterations"))
                entry["optimizer_detail"] = {
                    "fitness_progress_min_each_run": fitness_progress,
                    "final_fitness_best": max(fitness_progress) if fitness_progress else None,
                    "mean_iterations": _safe_mean(iterations),
                }
            runs.append(entry)
    return runs


def run_assessment(cfg: DictConfig) -> dict:
    """Run benchmark assessment based on configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing root path, save path, and print options.

    Returns
    -------
    dict
        Assessment report dictionary.
    """
    root = Path(cfg.root)
    summary_files = discover_benchmark_summaries(root)
    if not summary_files:
        logger.info("No benchmark_summary.json files found.")
        return {}

    detailed_runs = build_detailed_runs(summary_files)
    report = aggregate(detailed_runs)
    report["source_files"] = [str(p) for p in summary_files]

    if cfg.get("print", True) or not cfg.get("save"):
        logger.info("Aggregate assessment report:\n%s", json.dumps(report, indent=2))

    save_path = cfg.get("save")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved aggregate report to %s", save_path)
    return report


@hydra.main(config_path="configs", config_name="assess", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run benchmark assessment using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    run_assessment(cfg)


if __name__ == "__main__":
    main()
