"""Assess algorithm performance across multiple benchmark runs.

Supports Hydra multirun output directories by aggregating run overrides,
fitness metrics, topology counts, and stage timings.

Usage (Hydra from repo root):
  uv run python -m toop-engine-benchmark.assess_benchmarks root=/workspaces/ToOp/data/grid_node_breaker/results/multirun \
      save=/workspaces/ToOp/data/grid_node_breaker/results/aggregate_report.json print=true

Hydra config file: toop-engine-benchmark/configs/assess.yaml
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Iterable, cast

# Disable beartype claw for this module to avoid decorating Hydra's generated main wrapper
__beartype__ = False

# Hydra imports
import hydra

# Logging setup
import logbook
from omegaconf import DictConfig, OmegaConf

logger = logbook.Logger("Benchmark Assessment")

# Accept ints and floats (beartype strictness) and None values
Numeric = float | int
NumericIterable = Iterable[Numeric | None]


def _safe_mean(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.mean(v) if v else None


def _safe_median(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.median(v) if v else None


def _safe_stdev(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return stats.pstdev(v) if len(v) > 0 else None


def _safe_min(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return min(v) if v else None


def _safe_max(values: NumericIterable) -> float | None:
    v = [float(x) for x in values if x is not None]
    return max(v) if v else None


def _coeff_var(values: Iterable[float | None]) -> float | None:
    nums = [float(x) for x in values if x is not None]
    if len(nums) < 2:
        return None
    mean_val = stats.mean(nums)
    if mean_val == 0:
        return None
    return stats.pstdev(nums) / mean_val


def _compute_stats(values: NumericIterable) -> dict[str, float | None]:
    """Return common statistics for a sequence of numeric values."""
    return {
        "mean": _safe_mean(values),
        "median": _safe_median(values),
        "stdev": _safe_stdev(values),
        "min": _safe_min(values),
        "max": _safe_max(values),
    }


def _locate_run_root(start: Path) -> Path:
    """Find the nearest ancestor that contains a Hydra .hydra folder."""
    for candidate in [start, *start.parents]:
        if (candidate / ".hydra").is_dir():
            return candidate
    return start


def _load_overrides(run_root: Path) -> tuple[list[str], dict[str, Any]]:
    """Read Hydra overrides for a multirun entry if available."""
    overrides_file = run_root / ".hydra" / "overrides.yaml"
    overrides_list: list[str] = []
    overrides_resolved: dict[str, Any] = {}

    if overrides_file.exists():
        raw_lines = overrides_file.read_text().splitlines()
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            overrides_list.append(stripped)
        if overrides_list:
            try:
                resolved_cfg = OmegaConf.from_cli(overrides_list)
                container = OmegaConf.to_container(resolved_cfg, resolve=True)
                if isinstance(container, dict):
                    overrides_resolved = cast(dict[str, Any], container)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.info(f"Failed to parse overrides for {overrides_file}: {exc}")

    return overrides_list, overrides_resolved


def _locate_res_file(summary_dir: Path) -> Path | None:
    """Return the most relevant res.json file for a run, if present."""
    candidates = [summary_dir / "res.json", summary_dir / "stats" / "res.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def load_summary_entries(summary_file: Path) -> list[dict[str, Any]]:
    """Load benchmark summary entries from the given file.

    Parameters
    ----------
    summary_file : Path
        Path to the benchmark summary JSON file.

    Returns
    -------
    list[dict[str, Any]]
        List of benchmark summary entries.
    """
    data = load_json(summary_file)
    if isinstance(data, list):
        return data
    return [data]


def _collect_run_entries(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect enriched run entries for all Hydra multirun outputs under ``root``.

    Parameters
    ----------
    root : Path
        Root directory to search for benchmark summary files.

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        A tuple containing a list of run records and a sorted list of source files.
    """
    run_records: list[dict[str, Any]] = []
    source_files: set[str] = set()

    for summary_path in root.rglob("benchmark_summary.json"):
        summary_dir = summary_path.parent
        run_root = _locate_run_root(summary_dir)
        overrides_list, overrides_resolved = _load_overrides(run_root)
        res_path = _locate_res_file(summary_dir)
        res_data = load_json(res_path) if res_path else None

        entries = load_summary_entries(summary_path)
        source_files.add(str(summary_path))
        for idx, entry in enumerate(entries):
            try:
                run_id = str(run_root.relative_to(root))
            except ValueError:
                run_id = str(run_root)

            n_best_topologies = None
            if isinstance(res_data, dict):
                best_topos = res_data.get("best_topos") or []
                if isinstance(best_topos, list):
                    n_best_topologies = len(best_topos)

            dc_quality = entry.get("dc_quality")
            fitness_max = dc_quality.get("fitness_max") if isinstance(dc_quality, dict) else None

            run_records.append(
                {
                    "summary_entry": entry,
                    "summary_path": summary_path,
                    "summary_index": idx,
                    "run_dir": summary_dir,
                    "run_root": run_root,
                    "run_id": run_id,
                    "overrides": overrides_list,
                    "overrides_resolved": overrides_resolved,
                    "res_path": res_path,
                    "n_best_topologies": n_best_topologies,
                    "fitness_max": fitness_max,
                }
            )

    return run_records, sorted(source_files)


def _aggregate(run_records: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901, PLR0915
    """Aggregate benchmark metrics across collected run records."""
    if not run_records:
        return {}

    fitness_max_all: list[float | None] = []
    fitness_mean_all: list[float | None] = []
    split_subs_mean_all: list[float | None] = []
    n_best_summary_all: list[float | None] = []
    iterations_all: list[float | None] = []
    runtime_seconds_all: list[float | None] = []
    total_branch_combis_all: list[float | None] = []
    total_inj_combis_all: list[float | None] = []
    branch_combis_per_s_all: list[float | None] = []
    n_best_topology_counts: list[int] = []

    timing_totals: dict[str, list[float]] = {}
    timing_avgs: dict[str, list[float]] = {}
    timing_overheads: dict[str, list[float]] = {}

    best_run: dict[str, Any] | None = None
    best_fitness_value: float | None = None

    for record in run_records:
        entry = record["summary_entry"]
        dc_quality = entry.get("dc_quality") or {}
        fitness_max = dc_quality.get("fitness_max")
        fitness_mean = dc_quality.get("fitness_mean")
        split_subs_mean = dc_quality.get("split_subs_mean")
        n_best_summary = dc_quality.get("n_best")
        n_iterations = dc_quality.get("n_iterations")
        runtime_seconds = dc_quality.get("runtime_seconds")
        total_branch_combis = dc_quality.get("total_branch_combis")
        total_inj_combis = dc_quality.get("total_inj_combis")
        branch_combis_per_s = dc_quality.get("branch_combis_per_s")

        fitness_max_all.append(fitness_max)
        fitness_mean_all.append(fitness_mean)
        split_subs_mean_all.append(split_subs_mean)
        n_best_summary_all.append(n_best_summary)
        iterations_all.append(n_iterations)
        runtime_seconds_all.append(runtime_seconds)
        total_branch_combis_all.append(total_branch_combis)
        total_inj_combis_all.append(total_inj_combis)
        branch_combis_per_s_all.append(branch_combis_per_s)

        n_best_count = record.get("n_best_topologies")
        if isinstance(n_best_count, int):
            n_best_topology_counts.append(n_best_count)

        if isinstance(fitness_max, (int, float)):
            if best_fitness_value is None or float(fitness_max) > best_fitness_value:
                best_fitness_value = float(fitness_max)
                best_run = record

        timings = entry.get("timings") or {}
        assert isinstance(timings, dict), "Timings entry is not a dictionary"
        for phase, phase_stats in timings.items():
            assert isinstance(phase_stats, dict), "Invalid phase stats format"
            total_val = phase_stats.get("total_s")
            avg_val = phase_stats.get("avg_s")
            overhead_val = phase_stats.get("runtime_overhead_s")
            if total_val is not None:
                timing_totals.setdefault(phase, []).append(float(total_val))
            if avg_val is not None:
                timing_avgs.setdefault(phase, []).append(float(avg_val))
            if overhead_val is not None:
                timing_overheads.setdefault(phase, []).append(float(overhead_val))

    phases = sorted(set(timing_totals) | set(timing_avgs) | set(timing_overheads))
    timings_report = {}
    for phase in phases:
        phase_report: dict[str, Any] = {
            "total_s": _compute_stats(timing_totals.get(phase, [])),
            "avg_s": _compute_stats(timing_avgs.get(phase, [])),
            "cv": _coeff_var(timing_totals.get(phase, [])),
        }
        if timing_overheads.get(phase):
            phase_report["runtime_overhead_s"] = _compute_stats(timing_overheads.get(phase, []))
        timings_report[phase] = phase_report

    report: dict[str, Any] = {
        "n_runs": len(run_records),
        "fitness_max": {
            "mean": _safe_mean(fitness_max_all),
            "median": _safe_median(fitness_max_all),
            "stdev": _safe_stdev(fitness_max_all),
            "cv": _coeff_var(fitness_max_all),
            "best": best_fitness_value,
        },
        "fitness_mean": {
            "mean": _safe_mean(fitness_mean_all),
            "median": _safe_median(fitness_mean_all),
            "stdev": _safe_stdev(fitness_mean_all),
            "cv": _coeff_var(fitness_mean_all),
        },
        "split_subs_mean": {
            "mean": _safe_mean(split_subs_mean_all),
            "median": _safe_median(split_subs_mean_all),
            "stdev": _safe_stdev(split_subs_mean_all),
        },
        "n_best": {
            "mean": _safe_mean(n_best_summary_all),
            "median": _safe_median(n_best_summary_all),
            "stdev": _safe_stdev(n_best_summary_all),
            "min": _safe_min(n_best_summary_all),
            "max": _safe_max(n_best_summary_all),
        },
        "n_best_topologies": _compute_stats(n_best_topology_counts),
        "n_iterations": {
            "mean": _safe_mean(iterations_all),
            "median": _safe_median(iterations_all),
            "stdev": _safe_stdev(iterations_all),
        },
        "runtime_seconds": _compute_stats(runtime_seconds_all),
        "total_branch_combis": _compute_stats(total_branch_combis_all),
        "total_inj_combis": _compute_stats(total_inj_combis_all),
        "branch_combis_per_s": _compute_stats(branch_combis_per_s_all),
        "timings": timings_report,
    }

    if best_run:
        best_timings = best_run["summary_entry"].get("timings") or {}
        dc_timing = best_timings.get("dc_optimization") if isinstance(best_timings, dict) else None
        best_dc_quality = best_run["summary_entry"].get("dc_quality") or {}
        report["best_run"] = {
            "run_id": best_run["run_id"],
            "summary_path": str(best_run["summary_path"]),
            "res_path": str(best_run["res_path"]) if best_run.get("res_path") else None,
            "fitness_max": best_fitness_value,
            "overrides": best_run.get("overrides"),
            "overrides_resolved": best_run.get("overrides_resolved"),
            "n_best_topologies": best_run.get("n_best_topologies"),
            "n_iterations": best_dc_quality.get("n_iterations"),
            "runtime_seconds": best_dc_quality.get("runtime_seconds"),
            "runtime_overhead_s": dc_timing.get("runtime_overhead_s") if isinstance(dc_timing, dict) else None,
            "total_branch_combis": best_dc_quality.get("total_branch_combis"),
            "total_inj_combis": best_dc_quality.get("total_inj_combis"),
            "branch_combis_per_s": best_dc_quality.get("branch_combis_per_s"),
            "timings": best_timings,
            "dc_quality": best_dc_quality,
        }

    return report


@hydra.main(config_path="configs", config_name="assess", version_base="1.2")
def main(cfg: DictConfig) -> dict:
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
    logbook.StreamHandler(sys.stdout, level=cfg.get("logging_level", "INFO")).push_application()

    root = Path(cfg.root)
    run_records, source_files = _collect_run_entries(root)
    if not run_records:
        logger.info(f"No benchmark_summary.json files found under {root}.")
        return {}

    report = _aggregate(run_records)
    report["source_files"] = source_files

    if cfg.get("print", True) or not cfg.get("save"):
        logger.info(f"Aggregate assessment report:\n {json.dumps(report, indent=2)}")

    save_path = cfg.get("save")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved aggregate report to {save_path}")
    return report


if __name__ == "__main__":
    main()
