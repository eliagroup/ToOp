# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run reproducible, DC-first parent-selection benchmark studies."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import structlog
from beartype.typing import Any, Literal, Sequence
from fsspec.implementations.local import LocalFileSystem
from qdax.core.emitters.standard_emitters import EmitterState
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    perform_ac_analysis,
    prepare_importer_parameters,
    run_preprocessing,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import summarize
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire
from toop_engine_topology_optimizer.dc.repertoire.parent_selection import ParentSelectionMode
from toop_engine_topology_optimizer.dc.worker.optimizer import OptimizerData, initialize_optimization, run_epoch
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)
from tqdm import tqdm

logger = structlog.get_logger(__name__)

PARENT_SELECTION_LABELS: dict[ParentSelectionMode, str] = {
    "uniform": "UNIi",
    "uniform_cell": "UNIc",
    "ucb": "UCBc",
    "ucb_batched": "UCBb",
    "ucb_snapshot": "UCBs",
    "exploitation": "Ec",
    "exploration": "Xc",
    "greedy": "G",
}
# Short, stable labels used in benchmark tables and progress output.

AUTO_RESOLVABLE_DESCRIPTOR_METRICS = frozenset({"pst_activated", "pst_switching_distance"})


@dataclass(frozen=True)
class GridSpecification:
    """A grid input included in a parent-selection study."""

    identifier: str
    """Stable identifier used in result paths and reports."""

    source_path: Path
    """Path to the source grid file."""

    grid_type: Literal["powsybl", "pandapower"] = "powsybl"
    """Backend used to preprocess and optionally validate the grid."""


@dataclass(frozen=True)
class ACValidationConfiguration:
    """Optional AC validation settings for final DC candidates."""

    enabled: bool = False
    """Whether to validate final DC candidates with AC loadflow."""

    k_best_topos: int = 5
    """Number of final DC candidates to validate per run."""

    n_processes: int = 1
    """Number of AC validation worker processes."""

    critical_voltage_jump_percent: float = 5.0
    """Voltage-jump threshold supplied to AC acceptance metrics."""

    critical_va_diff_degree: float = 20.0
    """Voltage-angle threshold supplied to AC acceptance metrics."""


@dataclass(frozen=True)
class DescriptorResolutionConfiguration:
    """Configure benchmark-specific descriptor resolution after preprocessing."""

    auto_metrics: tuple[str, ...] = ()
    """Descriptor metrics whose number of cells is derived from each prepared grid."""

    max_logical_cells: int | None = None
    """Optional upper bound for the product of resolved descriptor cell counts."""

    def __post_init__(self) -> None:
        """Validate the configured automatic descriptor-resolution policy."""
        if isinstance(self.auto_metrics, str):
            raise ValueError("Descriptor auto_metrics must be a sequence of metric names, not a single string.")

        auto_metrics = tuple(self.auto_metrics)
        object.__setattr__(self, "auto_metrics", auto_metrics)
        duplicate_metrics = sorted({metric for metric in auto_metrics if auto_metrics.count(metric) > 1})
        if duplicate_metrics:
            raise ValueError(f"Duplicate automatic descriptor metrics: {duplicate_metrics}.")

        unsupported_metrics = sorted(set(auto_metrics) - AUTO_RESOLVABLE_DESCRIPTOR_METRICS)
        if unsupported_metrics:
            raise ValueError(f"Unsupported automatic descriptor metrics: {unsupported_metrics}.")
        if self.max_logical_cells is not None and self.max_logical_cells < 1:
            raise ValueError("max_logical_cells must be at least 1 when configured.")


@dataclass(frozen=True)
class RunSpecification:
    """One mode-and-seed execution for a prepared grid."""

    grid: GridSpecification
    """Grid used for this optimization run."""

    parent_selection_mode: ParentSelectionMode
    """Parent-selection policy evaluated by this run."""

    seed: int
    """Random seed supplied to the genetic algorithm."""

    @property
    def mode_label(self) -> str:
        """Return the concise label for this run's selection policy."""
        return PARENT_SELECTION_LABELS[self.parent_selection_mode]

    @property
    def identifier(self) -> str:
        """Return a stable relative identifier for files and reports."""
        return f"{self.grid.identifier}/{self.mode_label}/seed_{self.seed}"


@dataclass(frozen=True)
class PreparedGrid:
    """A staged grid and its reusable preprocessing result."""

    specification: GridSpecification
    """Source-grid metadata."""

    data_directory: Path
    """Directory containing the staged grid and preprocessing artifacts."""

    grid_path: Path
    """Staged XIIDM grid file used by preprocessing and AC validation."""

    static_information_path: Path
    """Static-information HDF5 input reused by every DC run for this grid."""

    source_sha256: str
    """SHA-256 digest of the original grid file."""


@dataclass(frozen=True)
class EvaluationCounters:
    """Cumulative candidate-evaluation counters supplied by the emitter."""

    branch_combinations: int
    """Total evaluated branch combinations."""

    injection_combinations: int
    """Total evaluated injection combinations."""

    split_grids: int
    """Total scored candidates that produced split grids."""


@dataclass(frozen=True)
class ArchiveStatistics:
    """Quality-diversity and occupancy metrics for one MAP-Elites repertoire."""

    n_logical_cells: int
    n_flat_slots: int
    occupied_cells: int
    occupied_candidates: int
    cell_coverage: float
    candidate_coverage: float
    occupied_candidates_per_cell_mean: float | None
    occupied_candidates_per_cell_min: int | None
    occupied_candidates_per_cell_max: int | None
    qd_score_raw: float
    qd_score_raw_per_cell: float
    qd_score_improvement: float
    qd_score_improvement_per_cell: float


@dataclass(frozen=True)
class RunOutcome:
    """Final artifact locations and result state for one benchmark run."""

    specification: RunSpecification
    """Mode-and-seed execution that produced this outcome."""

    output_directory: Path
    """Directory containing the trajectory, manifest, and final result."""

    final_result: dict[str, Any]
    """Serialized final DC optimizer result."""

    phase_seconds: dict[str, float]
    """Measured initialization and DC execution durations."""


def validate_study_inputs(
    grids: Sequence[GridSpecification],
    modes: Sequence[ParentSelectionMode],
    seeds: Sequence[int],
    loadflow_parameters: LoadflowSolverParameters,
) -> None:
    """Validate study inputs before expensive preprocessing or JAX compilation.

    Parameters
    ----------
    grids : Sequence[GridSpecification]
        Grid sources included in the study.
    modes : Sequence[ParentSelectionMode]
        Parent-selection modes to compare.
    seeds : Sequence[int]
        Shared seeds applied to every mode.
    loadflow_parameters : LoadflowSolverParameters
        DC solver configuration shared by every run.

    Raises
    ------
    ValueError
        If the experiment matrix is incomplete or not comparable.
    FileNotFoundError
        If a configured grid source does not exist.
    """
    if not grids:
        raise ValueError("At least one grid is required for a parent-selection study.")
    if not modes:
        raise ValueError("At least one parent-selection mode is required for a study.")
    if not seeds:
        raise ValueError("At least one random seed is required for a study.")
    if loadflow_parameters.distributed:
        raise ValueError("Parent-selection studies require distributed=False so all modes use the same execution path.")

    _validate_grids(grids)
    _validate_modes_and_seeds(modes, seeds)


def expand_run_specifications(
    grids: Sequence[GridSpecification],
    modes: Sequence[ParentSelectionMode],
    seeds: Sequence[int],
) -> list[RunSpecification]:
    """Expand grids, modes, and seeds into a deterministic run matrix.

    Parameters
    ----------
    grids : Sequence[GridSpecification]
        Grids to benchmark.
    modes : Sequence[ParentSelectionMode]
        Parent-selection modes to compare.
    seeds : Sequence[int]
        Shared random seeds for every mode.

    Returns
    -------
    list[RunSpecification]
        Run specifications ordered by grid, mode, then seed.
    """
    return [
        RunSpecification(grid=grid, parent_selection_mode=mode, seed=seed)
        for grid in grids
        for mode in modes
        for seed in seeds
    ]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File whose bytes are hashed.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_grid(
    grid: GridSpecification,
    prepared_grids_directory: Path,
    preprocessing_parameters: PreprocessParameters,
) -> PreparedGrid:
    """Stage and preprocess one grid for reuse by all of its benchmark runs.

    Parameters
    ----------
    grid : GridSpecification
        Source grid to stage and preprocess.
    prepared_grids_directory : Path
        Root directory for reusable, study-owned grid artifacts.
    preprocessing_parameters : PreprocessParameters
        Importer and solver preprocessing options.

    Returns
    -------
    PreparedGrid
        Reusable staged grid paths and source digest.
    """
    data_directory = prepared_grids_directory / grid.identifier
    data_directory.mkdir(parents=True, exist_ok=True)
    grid_path = data_directory / "grid.xiidm"
    source_sha256 = sha256_file(grid.source_path)
    static_information_path = data_directory / PREPROCESSING_PATHS["static_information_file_path"]
    grid_manifest_path = data_directory / "prepared_grid_manifest.json"

    existing_manifest = _load_json_if_exists(grid_manifest_path)
    is_current = (
        existing_manifest is not None
        and existing_manifest.get("source_sha256") == source_sha256
        and static_information_path.is_file()
        and grid_path.is_file()
    )
    if not is_current:
        shutil.copy2(grid.source_path, grid_path)
        importer_parameters = prepare_importer_parameters(grid_path, data_directory)
        run_preprocessing(
            importer_parameters=importer_parameters,
            data_folder=data_directory,
            preprocessing_parameters=preprocessing_parameters,
            is_pandapower_net=grid.grid_type == "pandapower",
        )
        _write_json(
            grid_manifest_path,
            {
                "grid_id": grid.identifier,
                "grid_type": grid.grid_type,
                "source_path": str(grid.source_path),
                "source_sha256": source_sha256,
                "static_information_path": str(static_information_path),
            },
        )

    if not static_information_path.is_file():
        raise FileNotFoundError(f"Preprocessing did not create static information: {static_information_path}")

    return PreparedGrid(
        specification=grid,
        data_directory=data_directory,
        grid_path=grid_path,
        static_information_path=static_information_path,
        source_sha256=source_sha256,
    )


def derive_pst_switching_distance_num_cells(
    pst_n_taps: np.ndarray,
    starting_tap_idx: np.ndarray,
    n_timesteps: int,
) -> int:
    """Return the cells required for an unclipped PST-distance descriptor.

    Parameters
    ----------
    pst_n_taps : np.ndarray
        Number of valid discrete tap positions per controllable PST.
    starting_tap_idx : np.ndarray
        Initial tap index for each controllable PST.
    n_timesteps : int
        Number of timesteps over which PST distance is summed.

    Returns
    -------
    int
        Number of cells covering every integer distance from zero through the
        largest possible cumulative PST switching distance.

    Raises
    ------
    ValueError
        If the PST data is inconsistent or no controllable PST can move away
        from its starting tap.
    """
    n_taps = np.asarray(pst_n_taps, dtype=int).reshape(-1)
    starting_taps = np.asarray(starting_tap_idx, dtype=int).reshape(-1)
    if n_timesteps < 1:
        raise ValueError("PST descriptor resolution requires at least one timestep.")
    if not n_taps.size:
        raise ValueError("PST descriptor resolution requires at least one controllable PST.")
    if n_taps.shape != starting_taps.shape:
        raise ValueError("PST tap counts and starting tap indices must have matching shapes.")
    if np.any(n_taps < 1):
        raise ValueError("Every controllable PST must have at least one tap position.")
    if np.any(starting_taps < 0) or np.any(starting_taps >= n_taps):
        raise ValueError("PST starting tap indices must be valid positions for their PSTs.")

    max_distances = np.maximum(starting_taps, n_taps - 1 - starting_taps)
    maximum_distance = int(max_distances.sum()) * n_timesteps
    if maximum_distance == 0:
        raise ValueError("PST descriptor resolution requires at least one PST that can change tap position.")
    return maximum_distance + 1


def derive_pst_activated_num_cells(
    pst_n_taps: np.ndarray,
    starting_tap_idx: np.ndarray,
    n_timesteps: int,
) -> int:
    """Return the cells required for an unclipped PST-activation descriptor.

    The activation metric counts each movable PST whose tap differs from its
    starting tap at each timestep. Its maximum is therefore the number of
    movable PSTs multiplied by the number of timesteps.

    Parameters
    ----------
    pst_n_taps : np.ndarray
        Number of valid discrete tap positions per controllable PST.
    starting_tap_idx : np.ndarray
        Initial tap index for each controllable PST.
    n_timesteps : int
        Number of timesteps over which changed PST positions are counted.

    Returns
    -------
    int
        Number of cells covering every activation count from zero through the
        largest possible count.

    Raises
    ------
    ValueError
        If the PST data is inconsistent or no controllable PST can move away
        from its starting tap.
    """
    derive_pst_switching_distance_num_cells(pst_n_taps, starting_tap_idx, n_timesteps)
    n_movable_psts = int(np.count_nonzero(np.asarray(pst_n_taps, dtype=int).reshape(-1) > 1))
    return n_movable_psts * n_timesteps + 1


def _resolve_automatic_descriptor(
    metric: str,
    ga_parameters: BatchedMEParameters,
    prepared_grid: PreparedGrid,
) -> DescriptorDef:
    """Resolve one validated automatic descriptor definition for a prepared grid."""
    if metric not in AUTO_RESOLVABLE_DESCRIPTOR_METRICS:
        raise AssertionError(f"Unexpected validated automatic descriptor metric: {metric}.")
    if not ga_parameters.enable_nodal_inj_optim:
        raise ValueError(f"{metric} requires ga.enable_nodal_inj_optim=True.")

    static_information = load_static_information(prepared_grid.static_information_path)
    dynamic_information = static_information.dynamic_information
    nodal_injection_information = dynamic_information.nodal_injection_information
    if nodal_injection_information is None:
        raise ValueError(f"Grid {prepared_grid.specification.identifier!r} has no controllable PSTs for {metric}.")
    try:
        pst_n_taps = np.asarray(jax.device_get(nodal_injection_information.pst_n_taps), dtype=int)
        starting_tap_idx = np.asarray(jax.device_get(nodal_injection_information.starting_tap_idx), dtype=int)
        if metric == "pst_switching_distance":
            num_cells = derive_pst_switching_distance_num_cells(
                pst_n_taps=pst_n_taps,
                starting_tap_idx=starting_tap_idx,
                n_timesteps=dynamic_information.n_timesteps,
            )
        else:
            num_cells = derive_pst_activated_num_cells(
                pst_n_taps=pst_n_taps,
                starting_tap_idx=starting_tap_idx,
                n_timesteps=dynamic_information.n_timesteps,
            )
    except ValueError as error:
        raise ValueError(f"Grid {prepared_grid.specification.identifier!r} cannot resolve {metric}: {error}") from error
    return DescriptorDef(metric=metric, num_cells=num_cells)


def _validate_repertoire_capacity(
    ga_parameters: BatchedMEParameters,
    prepared_grid: PreparedGrid,
    descriptor_resolution: DescriptorResolutionConfiguration,
) -> None:
    """Raise when a resolved grid-specific repertoire exceeds its configured capacity."""
    n_cells_per_dim = tuple(descriptor.num_cells for descriptor in ga_parameters.me_descriptors)
    n_logical_cells = math.prod(n_cells_per_dim)
    max_logical_cells = descriptor_resolution.max_logical_cells
    if max_logical_cells is not None and n_logical_cells > max_logical_cells:
        n_flat_slots = n_logical_cells * ga_parameters.cell_depth
        raise ValueError(
            f"Grid {prepared_grid.specification.identifier!r} resolves to {n_logical_cells} logical repertoire cells "
            f"with descriptor shape {n_cells_per_dim} ({n_flat_slots} slots at cell_depth="
            f"{ga_parameters.cell_depth}), exceeding max_logical_cells={max_logical_cells}."
        )


def resolve_ga_parameters_for_grid(
    ga_parameters: BatchedMEParameters,
    prepared_grid: PreparedGrid,
    descriptor_resolution: DescriptorResolutionConfiguration | None,
) -> BatchedMEParameters:
    """Resolve automatic descriptor dimensions for one prepared grid.

    Parameters
    ----------
    ga_parameters : BatchedMEParameters
        Base benchmark GA configuration.
    prepared_grid : PreparedGrid
        Preprocessed grid whose static PST data determines automatic dimensions.
    descriptor_resolution : DescriptorResolutionConfiguration | None
        Automatic descriptor and capacity policy. ``None`` preserves the given
        GA configuration unchanged.

    Returns
    -------
    BatchedMEParameters
        A validated, grid-specific GA configuration.

    Raises
    ------
    ValueError
        If descriptor names collide, PST optimization is unavailable, or the
        resolved repertoire exceeds the configured capacity.
    """
    if descriptor_resolution is None:
        return ga_parameters

    descriptor_metrics = tuple(descriptor.metric for descriptor in ga_parameters.me_descriptors)
    duplicate_descriptor_metrics = sorted({metric for metric in descriptor_metrics if descriptor_metrics.count(metric) > 1})
    if duplicate_descriptor_metrics:
        raise ValueError(f"Configured descriptor metrics must be unique: {duplicate_descriptor_metrics}.")

    overlapping_auto_metrics = sorted(set(descriptor_metrics) & set(descriptor_resolution.auto_metrics))
    if overlapping_auto_metrics:
        raise ValueError(f"Automatic descriptor metrics must not also be configured manually: {overlapping_auto_metrics}.")

    resolved_descriptors = list(ga_parameters.me_descriptors)
    resolved_descriptors.extend(
        _resolve_automatic_descriptor(metric, ga_parameters, prepared_grid) for metric in descriptor_resolution.auto_metrics
    )

    resolved_ga_parameters = BatchedMEParameters.model_validate(
        {
            **ga_parameters.model_dump(mode="python"),
            "me_descriptors": resolved_descriptors,
        }
    )
    _validate_repertoire_capacity(resolved_ga_parameters, prepared_grid, descriptor_resolution)
    return resolved_ga_parameters


def compute_archive_statistics(
    fitnesses: np.ndarray,
    cell_depth: int,
    initial_fitness: float,
) -> ArchiveStatistics:
    """Compute occupancy and QD metrics from a depth-expanded repertoire.

    Each occupied logical cell contributes its best retained candidate to the raw
    QD score. The improvement QD score sums only positive elite improvements over
    the unsplit initial fitness. Both are also normalized by the number of logical
    MAP-Elites cells.

    Parameters
    ----------
    fitnesses : np.ndarray
        Host-resident flat repertoire fitness array, with cell-depth layers stored
        contiguously.
    cell_depth : int
        Number of candidate slots stored per logical cell.
    initial_fitness : float
        Fitness of the initial, unsplit topology.

    Returns
    -------
    ArchiveStatistics
        Occupancy, coverage, and raw/improvement QD metrics.

    Raises
    ------
    ValueError
        If the repertoire shape cannot be partitioned into the requested depth.
    """
    flat_fitnesses = np.asarray(jax.device_get(fitnesses), dtype=float).reshape(-1)
    if cell_depth <= 0 or flat_fitnesses.size % cell_depth != 0:
        raise ValueError("Repertoire size must be divisible by a positive cell depth.")

    n_flat_slots = int(flat_fitnesses.size)
    n_logical_cells = n_flat_slots // cell_depth
    by_depth_layer = flat_fitnesses.reshape((cell_depth, n_logical_cells))
    occupied_by_depth_layer = np.isfinite(by_depth_layer)
    occupied_candidates = int(occupied_by_depth_layer.sum())
    occupied_per_cell = occupied_by_depth_layer.sum(axis=0)
    occupied_cells_mask = occupied_per_cell > 0
    occupied_cells = int(occupied_cells_mask.sum())
    cell_elite_fitnesses = np.max(
        np.where(occupied_by_depth_layer, by_depth_layer, -np.inf),
        axis=0,
    )
    occupied_cell_elites = cell_elite_fitnesses[occupied_cells_mask]

    qd_score_raw = float(occupied_cell_elites.sum())
    improvements = np.maximum(occupied_cell_elites - float(initial_fitness), 0.0)
    qd_score_improvement = float(improvements.sum())
    occupied_counts = occupied_per_cell[occupied_cells_mask]

    return ArchiveStatistics(
        n_logical_cells=n_logical_cells,
        n_flat_slots=n_flat_slots,
        occupied_cells=occupied_cells,
        occupied_candidates=occupied_candidates,
        cell_coverage=occupied_cells / n_logical_cells if n_logical_cells else 0.0,
        candidate_coverage=occupied_candidates / n_flat_slots if n_flat_slots else 0.0,
        occupied_candidates_per_cell_mean=float(occupied_counts.mean()) if occupied_cells else None,
        occupied_candidates_per_cell_min=int(occupied_counts.min()) if occupied_cells else None,
        occupied_candidates_per_cell_max=int(occupied_counts.max()) if occupied_cells else None,
        qd_score_raw=qd_score_raw,
        qd_score_raw_per_cell=qd_score_raw / n_logical_cells if n_logical_cells else 0.0,
        qd_score_improvement=qd_score_improvement,
        qd_score_improvement_per_cell=qd_score_improvement / n_logical_cells if n_logical_cells else 0.0,
    )


def extract_evaluation_counters(emitter_state: EmitterState) -> EvaluationCounters:
    """Extract cumulative candidate-evaluation counters from an emitter state.

    Parameters
    ----------
    emitter_state : Any
        Emitter state produced by ``TrackingMixingEmitter``.

    Returns
    -------
    EvaluationCounters
        Cumulative branch, injection, and split-grid counters.
    """
    return EvaluationCounters(
        branch_combinations=_as_int(getattr(emitter_state, "total_branch_combis", 0)),
        injection_combinations=_as_int(getattr(emitter_state, "total_inj_combis", 0)),
        split_grids=_as_int(getattr(emitter_state, "total_num_splits", 0)),
    )


def extract_feedback_statistics(emitter_state: EmitterState) -> dict[str, Any] | None:
    """Extract compact, policy-neutral logical-cell selection telemetry.

    Parameters
    ----------
    emitter_state : Any
        Emitter state produced by ``TrackingMixingEmitter``.

    Returns
    -------
    dict[str, Any] | None
        Scalar selection and offspring-survival summaries, or ``None`` when the
        emitter does not expose universal parent-selection telemetry.
    """
    if not hasattr(emitter_state, "parent_selection_counts"):
        return None

    selection_counts = np.asarray(jax.device_get(emitter_state.parent_selection_counts), dtype=int)
    success_counts = np.asarray(jax.device_get(emitter_state.parent_success_counts), dtype=int)
    total_selections = int(selection_counts.sum())
    total_successes = int(success_counts.sum())
    selected_mask = selection_counts > 0
    success_rates = np.divide(
        success_counts[selected_mask],
        selection_counts[selected_mask],
        out=np.zeros(selected_mask.sum(), dtype=float),
        where=selection_counts[selected_mask] > 0,
    )
    selection_share = (
        selection_counts / total_selections if total_selections else np.zeros_like(selection_counts, dtype=float)
    )

    return {
        "n_logical_cells": int(selection_counts.size),
        "total_parent_selections": total_selections,
        "total_attributed_survivals": total_successes,
        "selected_cells": int(selected_mask.sum()),
        "selected_cell_fraction": float(selected_mask.mean()) if selected_mask.size else 0.0,
        "unvisited_cells": int((~selected_mask).sum()),
        "attributed_survival_rate": total_successes / total_selections if total_selections else None,
        "success_rate_mean": float(success_rates.mean()) if success_rates.size else None,
        "success_rate_min": float(success_rates.min()) if success_rates.size else None,
        "success_rate_max": float(success_rates.max()) if success_rates.size else None,
        "selection_concentration": float(np.square(selection_share).sum()),
        "effective_selected_cells": float(1 / np.square(selection_share).sum()) if total_selections else None,
    }


def make_epoch_record(
    repertoire: DiscreteMapElitesRepertoire,
    emitter_state: EmitterState,
    observed_metrics: Sequence[str],
    initial_fitness: float,
    epoch: int,
    jax_iteration: int,
    elapsed_seconds: float,
    epoch_seconds: float,
    previous_counters: EvaluationCounters | None,
) -> tuple[dict[str, Any], EvaluationCounters]:
    """Create one serializable benchmark observation after a completed epoch.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        Current CPU-resident MAP-Elites archive.
    emitter_state : Any
        Current CPU-resident emitter state.
    observed_metrics : Sequence[str]
        Metric names to extract for the current best candidate.
    initial_fitness : float
        Fitness of the initial topology.
    epoch : int
        Completed optimizer epoch, or zero for the initialized archive.
    jax_iteration : int
        Cumulative number of JAX optimization iterations.
    elapsed_seconds : float
        Monotonic DC loop time since initialization completed.
    epoch_seconds : float
        Duration of this epoch, zero for the initial archive.
    previous_counters : EvaluationCounters | None
        Counters from the preceding record, if any.

    Returns
    -------
    tuple[dict[str, Any], EvaluationCounters]
        JSON-ready epoch record and its cumulative counters for the next record.
    """
    fitnesses = np.asarray(jax.device_get(repertoire.fitnesses), dtype=float).reshape(-1)
    occupied_mask = np.isfinite(fitnesses)
    if not occupied_mask.any():
        best_fitness = float(initial_fitness)
        best_index = None
    else:
        best_index = int(np.nanargmax(np.where(occupied_mask, fitnesses, -np.inf)))
        best_fitness = float(fitnesses[best_index])

    best_metrics = _extract_best_metrics(repertoire, observed_metrics, best_index)
    archive = compute_archive_statistics(
        fitnesses=fitnesses,
        cell_depth=repertoire.cell_depth,
        initial_fitness=initial_fitness,
    )
    counters = extract_evaluation_counters(emitter_state)
    counter_delta = _counter_delta(counters, previous_counters)
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    epoch_seconds = max(float(epoch_seconds), 0.0)
    fitness_improvement = best_fitness - float(initial_fitness)
    relative_improvement = fitness_improvement / abs(float(initial_fitness)) if initial_fitness else None

    record = {
        "epoch": int(epoch),
        "jax_iteration": int(jax_iteration),
        "elapsed_seconds": elapsed_seconds,
        "epoch_seconds": epoch_seconds,
        "execution": {
            "cumulative": asdict(counters),
            "epoch": asdict(counter_delta),
            "branch_combinations_per_second": counters.branch_combinations / elapsed_seconds if elapsed_seconds else 0.0,
            "injection_combinations_per_second": (
                counters.injection_combinations / elapsed_seconds if elapsed_seconds else 0.0
            ),
            "epoch_branch_combinations_per_second": (
                counter_delta.branch_combinations / epoch_seconds if epoch_seconds else 0.0
            ),
        },
        "quality": {
            "initial_fitness": float(initial_fitness),
            "best_fitness": best_fitness,
            "fitness_improvement": fitness_improvement,
            "relative_fitness_improvement": relative_improvement,
            "best_candidate_metrics": best_metrics,
        },
        "archive": asdict(archive),
        "feedback": extract_feedback_statistics(emitter_state),
    }
    return _to_json_value(record), counters


def export_archive_cells(repertoire: DiscreteMapElitesRepertoire) -> list[dict[str, Any]]:
    """Serialize the best retained candidate from every occupied logical cell.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        Final CPU-resident MAP-Elites archive.

    Returns
    -------
    list[dict[str, Any]]
        One descriptor, fitness, and observed-metric record per occupied cell.
    """
    fitnesses = np.asarray(jax.device_get(repertoire.fitnesses), dtype=float).reshape(-1)
    descriptors = np.asarray(jax.device_get(repertoire.descriptors))
    n_cells = len(fitnesses) // repertoire.cell_depth
    records: list[dict[str, Any]] = []
    for cell_index in range(n_cells):
        slot_indices = cell_index + np.arange(repertoire.cell_depth) * n_cells
        occupied_indices = slot_indices[np.isfinite(fitnesses[slot_indices])]
        if not len(occupied_indices):
            continue
        elite_index = int(occupied_indices[np.argmax(fitnesses[occupied_indices])])
        records.append(
            {
                "cell_index": cell_index,
                "fitness": fitnesses[elite_index],
                "descriptors": descriptors[elite_index],
                "metrics": {
                    metric: np.asarray(jax.device_get(values))[elite_index]
                    for metric, values in repertoire.extra_scores.items()
                },
            }
        )
    return _to_json_value(records)  # type: ignore[return-value]


def make_repertoire_snapshot_record(
    repertoire: DiscreteMapElitesRepertoire,
    emitter_state: EmitterState,
    epoch: int,
    jax_iteration: int,
) -> dict[str, Any]:
    """Create a sparse, JSON-ready logical-cell repertoire snapshot.

    Parameters
    ----------
    repertoire : DiscreteMapElitesRepertoire
        Current CPU-resident MAP-Elites archive.
    emitter_state : EmitterState
        Current CPU-resident emitter state with cumulative selection telemetry.
    epoch : int
        Completed optimizer epoch, or zero for the initialized archive.
    jax_iteration : int
        Cumulative number of JAX optimization iterations.

    Returns
    -------
    dict[str, Any]
        Sparse cell indices with their best fitness and cumulative parent
        selection counts. Missing cell indices imply an empty cell and zero
        selections.

    Raises
    ------
    ValueError
        If the repertoire layout or selection telemetry is inconsistent.
    """
    flat_fitnesses = np.asarray(jax.device_get(repertoire.fitnesses), dtype=float).reshape(-1)
    cell_depth = repertoire.cell_depth
    if cell_depth <= 0 or flat_fitnesses.size % cell_depth != 0:
        raise ValueError("Repertoire size must be divisible by a positive cell depth.")

    n_logical_cells = flat_fitnesses.size // cell_depth
    if math.prod(repertoire.n_cells_per_dim) != n_logical_cells:
        raise ValueError("Repertoire layout does not match its configured descriptor cell shape.")

    raw_selection_counts = getattr(emitter_state, "parent_selection_counts", None)
    if raw_selection_counts is None:
        selection_counts = np.zeros(n_logical_cells, dtype=int)
    else:
        selection_counts = np.asarray(jax.device_get(raw_selection_counts), dtype=int).reshape(-1)
        if selection_counts.size == 0:
            selection_counts = np.zeros(n_logical_cells, dtype=int)
        elif selection_counts.size != n_logical_cells:
            raise ValueError("Parent-selection telemetry does not match the number of logical repertoire cells.")
    if np.any(selection_counts < 0):
        raise ValueError("Parent-selection telemetry must not contain negative counts.")

    fitnesses_by_depth = flat_fitnesses.reshape((cell_depth, n_logical_cells))
    occupied_by_depth = np.isfinite(fitnesses_by_depth)
    elite_fitnesses = np.max(np.where(occupied_by_depth, fitnesses_by_depth, -np.inf), axis=0)
    occupied_cells = np.isfinite(elite_fitnesses)
    retained_cells = occupied_cells | (selection_counts > 0)
    cell_indices = np.flatnonzero(retained_cells)

    return {
        "schema_version": 1,
        "epoch": int(epoch),
        "jax_iteration": int(jax_iteration),
        "cell_indices": cell_indices.tolist(),
        "elite_fitnesses": [
            float(elite_fitnesses[cell_index]) if occupied_cells[cell_index] else None for cell_index in cell_indices
        ],
        "selection_counts": selection_counts[cell_indices].tolist(),
    }


def run_dc_benchmark(  # noqa: PLR0915
    run: RunSpecification,
    prepared_grid: PreparedGrid,
    output_directory: Path,
    ga_parameters: BatchedMEParameters,
    loadflow_parameters: LoadflowSolverParameters,
    show_progress: bool = True,
    descriptor_resolution: DescriptorResolutionConfiguration | None = None,
) -> RunOutcome:
    """Run one mode-and-seed DC optimization while recording every epoch.

    Parameters
    ----------
    run : RunSpecification
        Parent-selection mode and random seed to execute.
    prepared_grid : PreparedGrid
        Reusable preprocessing output for the input grid.
    output_directory : Path
        Directory for trajectory, manifest, and final optimizer artifacts.
    ga_parameters : BatchedMEParameters
        Base GA configuration shared by all benchmark runs.
    loadflow_parameters : LoadflowSolverParameters
        Shared DC solver configuration. Distributed execution is unsupported.
    show_progress : bool, optional
        Whether to render the nested per-run tqdm progress bar.
    descriptor_resolution : DescriptorResolutionConfiguration, optional
        Per-grid automatic descriptor and repertoire-capacity policy recorded
        in the run manifest.

    Returns
    -------
    RunOutcome
        Final optimizer result and artifact locations.

    Raises
    ------
    ValueError
        If distributed execution is enabled.
    Exception
        Any initialization or optimization error after writing a failed manifest.
    """
    if loadflow_parameters.distributed:
        raise ValueError("Parent-selection benchmarks require distributed=False.")

    output_directory.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_directory / "trajectory.jsonl"
    repertoire_trajectory_path = output_directory / "repertoire_trajectory.jsonl"
    manifest_path = output_directory / "run_manifest.json"
    run_ga_parameters = ga_parameters.model_copy(
        update={"parent_selection_mode": run.parent_selection_mode, "random_seed": run.seed, "plot": False}
    )
    optimizer_parameters = DCOptimizerParameters(
        ga_config=run_ga_parameters,
        loadflow_solver_config=loadflow_parameters,
        summary_frequency=run_ga_parameters.iterations_per_epoch,
        check_command_frequency=run_ga_parameters.iterations_per_epoch,
    )
    manifest = _run_manifest(
        run,
        prepared_grid,
        optimizer_parameters,
        status="running",
        descriptor_resolution=descriptor_resolution,
    )
    _write_json(manifest_path, manifest)
    trajectory_path.unlink(missing_ok=True)
    repertoire_trajectory_path.unlink(missing_ok=True)

    initialization_start = time.perf_counter()
    try:
        optimizer_data, _stats, _initial_topology = initialize_optimization(
            params=optimizer_parameters,
            optimization_id=run.identifier,
            static_information_files=(prepared_grid.static_information_path,),
            processed_gridfile_fs=LocalFileSystem(),
        )
        initialization_seconds = time.perf_counter() - initialization_start

        # Compile and synchronize the exact JITted epoch before throughput timing.
        warmup_start = time.perf_counter()
        warmup_data = run_epoch(optimizer_data)
        jax.block_until_ready(warmup_data.jax_data)
        warmup_seconds = time.perf_counter() - warmup_start
        dc_start = time.perf_counter()
        epoch = 0
        previous_counters: EvaluationCounters | None = None

        cpu_repertoire, cpu_emitter_state = _cpu_optimizer_state(optimizer_data)
        initial_record, previous_counters = make_epoch_record(
            repertoire=cpu_repertoire,
            emitter_state=cpu_emitter_state,
            observed_metrics=run_ga_parameters.observed_metrics,
            initial_fitness=optimizer_data.initial_fitness,
            epoch=epoch,
            jax_iteration=_as_int(optimizer_data.jax_data.latest_iteration),
            elapsed_seconds=0.0,
            epoch_seconds=0.0,
            previous_counters=previous_counters,
        )
        _append_jsonl(trajectory_path, initial_record)
        _append_jsonl(
            repertoire_trajectory_path,
            make_repertoire_snapshot_record(
                repertoire=cpu_repertoire,
                emitter_state=cpu_emitter_state,
                epoch=epoch,
                jax_iteration=_as_int(optimizer_data.jax_data.latest_iteration),
            ),
        )

        with tqdm(
            total=float(run_ga_parameters.runtime_seconds),
            desc=f"{run.grid.identifier} {run.mode_label} seed={run.seed}",
            unit="s",
            leave=False,
            disable=not show_progress,
        ) as progress_bar:
            while time.perf_counter() - dc_start < run_ga_parameters.runtime_seconds:
                epoch_start = time.perf_counter()
                optimizer_data = run_epoch(optimizer_data)
                epoch_seconds = time.perf_counter() - epoch_start
                elapsed_seconds = time.perf_counter() - dc_start
                epoch += 1

                cpu_repertoire, cpu_emitter_state = _cpu_optimizer_state(optimizer_data)
                record, previous_counters = make_epoch_record(
                    repertoire=cpu_repertoire,
                    emitter_state=cpu_emitter_state,
                    observed_metrics=run_ga_parameters.observed_metrics,
                    initial_fitness=optimizer_data.initial_fitness,
                    epoch=epoch,
                    jax_iteration=_as_int(optimizer_data.jax_data.latest_iteration),
                    elapsed_seconds=elapsed_seconds,
                    epoch_seconds=epoch_seconds,
                    previous_counters=previous_counters,
                )
                _append_jsonl(trajectory_path, record)
                _append_jsonl(
                    repertoire_trajectory_path,
                    make_repertoire_snapshot_record(
                        repertoire=cpu_repertoire,
                        emitter_state=cpu_emitter_state,
                        epoch=epoch,
                        jax_iteration=_as_int(optimizer_data.jax_data.latest_iteration),
                    ),
                )
                progress_bar.update(max(elapsed_seconds - progress_bar.n, 0.0))
                progress_bar.set_postfix(_progress_postfix(record))

        final_result = _summarize_optimizer_result(optimizer_data, epoch, run_ga_parameters, loadflow_parameters)
        _write_json(output_directory / "res.json", final_result)
        _write_json(output_directory / "archive_cells.json", export_archive_cells(cpu_repertoire))
        phase_seconds = {
            "initialization": initialization_seconds,
            "warmup": warmup_seconds,
            "dc_optimization": time.perf_counter() - dc_start,
        }
        manifest.update(
            {
                "status": "completed",
                "phase_seconds": phase_seconds,
                "epochs_completed": epoch,
                "trajectory_path": str(trajectory_path),
                "repertoire_trajectory_path": str(repertoire_trajectory_path),
                "result_path": str(output_directory / "res.json"),
                "archive_cells_path": str(output_directory / "archive_cells.json"),
            }
        )
        _write_json(manifest_path, manifest)
        return RunOutcome(
            specification=run,
            output_directory=output_directory,
            final_result=final_result,
            phase_seconds=phase_seconds,
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "phase_seconds": {"initialization": time.perf_counter() - initialization_start},
            }
        )
        _write_json(manifest_path, manifest)
        raise


def run_parent_selection_study(  # noqa: PLR0913, PLR0917
    grids: Sequence[GridSpecification],
    modes: Sequence[ParentSelectionMode],
    seeds: Sequence[int],
    output_directory: Path,
    ga_parameters: BatchedMEParameters,
    loadflow_parameters: LoadflowSolverParameters,
    preprocessing_parameters: PreprocessParameters,
    ac_validation: ACValidationConfiguration | None = None,
    show_progress: bool = True,
    stop_on_error: bool = False,
    descriptor_resolution: DescriptorResolutionConfiguration | None = None,
) -> list[RunOutcome]:
    """Execute a complete multi-grid parent-selection benchmark study.

    Parameters
    ----------
    grids : Sequence[GridSpecification]
        Grid collection to preprocess and benchmark.
    modes : Sequence[ParentSelectionMode]
        Parent-selection modes to compare.
    seeds : Sequence[int]
        Shared seed set used for every mode.
    output_directory : Path
        Root directory for prepared grids, run artifacts, and study manifest.
    ga_parameters : BatchedMEParameters
        Base GA configuration common to all mode-and-seed runs.
    loadflow_parameters : LoadflowSolverParameters
        Shared DC solver configuration.
    preprocessing_parameters : PreprocessParameters
        Grid preprocessing options.
    ac_validation : ACValidationConfiguration, optional
        Optional final top-k AC validation settings.
    show_progress : bool, optional
        Whether to show outer study and nested run tqdm bars.
    stop_on_error : bool, optional
        Whether to raise immediately after a failed run instead of continuing.
    descriptor_resolution : DescriptorResolutionConfiguration, optional
        Per-grid automatic descriptor and repertoire-capacity policy.

    Returns
    -------
    list[RunOutcome]
        Completed DC run outcomes. Failed runs remain documented in their manifests.
    """
    validate_study_inputs(grids, modes, seeds, loadflow_parameters)
    ac_validation = ac_validation if ac_validation is not None else ACValidationConfiguration()
    output_directory.mkdir(parents=True, exist_ok=True)
    run_specifications = expand_run_specifications(grids, modes, seeds)
    prepared_grids_directory = output_directory / "prepared_grids"
    runs_directory = output_directory / "runs"
    study_manifest_path = output_directory / "study_manifest.json"
    study_manifest = {
        "schema_version": 1,
        "status": "running",
        "grids": [_to_json_value(asdict(grid)) for grid in grids],
        "modes": list(modes),
        "seeds": list(seeds),
        "n_planned_runs": len(run_specifications),
        "completed_runs": [],
        "failed_runs": [],
    }
    _write_json(study_manifest_path, study_manifest)

    prepared_grids = {
        grid.identifier: prepare_grid(grid, prepared_grids_directory, preprocessing_parameters) for grid in grids
    }
    resolved_ga_parameters = {
        grid_id: resolve_ga_parameters_for_grid(
            ga_parameters=ga_parameters,
            prepared_grid=prepared_grid,
            descriptor_resolution=descriptor_resolution,
        )
        for grid_id, prepared_grid in prepared_grids.items()
    }
    outcomes: list[RunOutcome] = []
    with tqdm(total=len(run_specifications), desc="Parent-selection study", unit="run", disable=not show_progress) as bar:
        for run in run_specifications:
            run_output_directory = runs_directory / run.identifier
            bar.set_postfix(grid=run.grid.identifier, mode=run.mode_label, seed=run.seed, status="running")
            try:
                outcome = run_dc_benchmark(
                    run=run,
                    prepared_grid=prepared_grids[run.grid.identifier],
                    output_directory=run_output_directory,
                    ga_parameters=resolved_ga_parameters[run.grid.identifier],
                    loadflow_parameters=loadflow_parameters,
                    show_progress=show_progress,
                    descriptor_resolution=descriptor_resolution,
                )
                outcomes.append(outcome)
                study_manifest["completed_runs"].append(run.identifier)
                if ac_validation.enabled:
                    ac_succeeded = _run_optional_ac_validation(
                        outcome,
                        prepared_grids[run.grid.identifier],
                        ac_validation,
                    )
                    if not ac_succeeded:
                        study_manifest.setdefault("ac_failed_runs", []).append(run.identifier)
                bar.set_postfix(grid=run.grid.identifier, mode=run.mode_label, seed=run.seed, status="completed")
            except Exception as error:
                logger.error("Parent-selection benchmark run failed", run=run.identifier, error=str(error))
                study_manifest["failed_runs"].append(
                    {"run": run.identifier, "error_type": type(error).__name__, "error": str(error)}
                )
                bar.set_postfix(grid=run.grid.identifier, mode=run.mode_label, seed=run.seed, status="failed")
                if stop_on_error:
                    study_manifest["status"] = "failed"
                    _write_json(study_manifest_path, study_manifest)
                    raise
            finally:
                bar.update(1)
                _write_json(study_manifest_path, study_manifest)

    study_manifest["status"] = "completed" if not study_manifest["failed_runs"] else "completed_with_failures"
    _write_json(study_manifest_path, study_manifest)
    return outcomes


def _cpu_optimizer_state(optimizer_data: OptimizerData) -> tuple[DiscreteMapElitesRepertoire, EmitterState]:
    """Synchronize and return the current single-device repertoire and emitter state."""
    return (
        jax.device_get(optimizer_data.jax_data.repertoire),
        jax.device_get(optimizer_data.jax_data.emitter_state),
    )


def _extract_best_metrics(
    repertoire: DiscreteMapElitesRepertoire,
    observed_metrics: Sequence[str],
    best_index: int | None,
) -> dict[str, float | None]:
    """Extract requested metrics for the current best candidate."""
    if best_index is None:
        return {metric: None for metric in observed_metrics}
    return {
        metric: _as_float(repertoire.extra_scores.get(metric)[best_index])
        if repertoire.extra_scores.get(metric) is not None
        else None
        for metric in observed_metrics
    }


def _counter_delta(current: EvaluationCounters, previous: EvaluationCounters | None) -> EvaluationCounters:
    """Return nonnegative evaluation-counter increments since the prior record."""
    if previous is None:
        return current
    return EvaluationCounters(
        branch_combinations=max(current.branch_combinations - previous.branch_combinations, 0),
        injection_combinations=max(current.injection_combinations - previous.injection_combinations, 0),
        split_grids=max(current.split_grids - previous.split_grids, 0),
    )


def _summarize_optimizer_result(
    optimizer_data: OptimizerData,
    epoch: int,
    ga_parameters: BatchedMEParameters,
    loadflow_parameters: LoadflowSolverParameters,
) -> dict[str, Any]:
    """Create the standard final optimizer result required by AC validation."""
    repertoire, emitter_state = _cpu_optimizer_state(optimizer_data)
    nodal_injection_information = optimizer_data.jax_data.dynamic_informations[0].nodal_injection_information
    result = summarize(
        repertoire=repertoire,
        emitter_state=emitter_state,
        initial_fitness=optimizer_data.initial_fitness,
        initial_metrics=optimizer_data.initial_metrics,
        contingency_ids=optimizer_data.solver_configs[0].contingency_ids,
        grid_model_low_tap=(
            nodal_injection_information.grid_model_low_tap if nodal_injection_information is not None else None
        ),
    )
    result.update(
        {
            "args": {
                "ga_config": ga_parameters.model_dump(mode="json"),
                "lf_config": loadflow_parameters.model_dump(mode="json"),
            },
            "iteration": epoch,
        }
    )
    return _to_json_value(result)


def _run_optional_ac_validation(
    outcome: RunOutcome,
    prepared_grid: PreparedGrid,
    configuration: ACValidationConfiguration,
) -> bool:
    """Run optional AC validation without invalidating a completed DC run.

    Returns
    -------
    bool
        Whether the AC validation completed successfully.
    """
    ac_start = time.perf_counter()
    try:
        topology_paths = perform_ac_analysis(
            data_folder=prepared_grid.data_directory,
            optimisation_run_path=outcome.output_directory,
            ac_validation_cfg={
                "k_best_topos": configuration.k_best_topos,
                "n_processes": configuration.n_processes,
                "critical_voltage_jump_percent": configuration.critical_voltage_jump_percent,
                "critical_va_diff_degree": configuration.critical_va_diff_degree,
            },
            pandapower_runner=prepared_grid.specification.grid_type == "pandapower",
        )
    except Exception as error:
        logger.error("Optional AC validation failed", run=outcome.specification.identifier, error=str(error))
        summary = {
            "enabled": True,
            "status": "failed",
            "duration_seconds": time.perf_counter() - ac_start,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        summary = {
            "enabled": True,
            "status": "completed",
            "duration_seconds": time.perf_counter() - ac_start,
            "unsplit_metrics": _load_json_if_exists(outcome.output_directory / "unsplit_ac_metrics.json"),
            "topologies": [
                {
                    "path": str(path),
                    "metrics": _load_json_if_exists(path / "ac_metrics.json"),
                }
                for path in topology_paths
            ],
        }
    summary_path = outcome.output_directory / "ac_validation_summary.json"
    _write_json(summary_path, summary)
    manifest_path = outcome.output_directory / "run_manifest.json"
    manifest = _load_json_if_exists(manifest_path) or {}
    manifest["ac_validation"] = {
        "status": summary["status"],
        "summary_path": str(summary_path),
        "duration_seconds": summary["duration_seconds"],
    }
    _write_json(manifest_path, manifest)
    return summary["status"] == "completed"


def _run_manifest(
    run: RunSpecification,
    prepared_grid: PreparedGrid,
    parameters: DCOptimizerParameters,
    status: str,
    descriptor_resolution: DescriptorResolutionConfiguration | None = None,
) -> dict[str, Any]:
    """Build the immutable portion of a run manifest."""
    descriptor_definitions = parameters.ga_config.me_descriptors
    n_cells_per_dim = tuple(descriptor.num_cells for descriptor in descriptor_definitions)
    return {
        "schema_version": 2,
        "status": status,
        "run_id": run.identifier,
        "grid": {
            "id": run.grid.identifier,
            "source_path": str(run.grid.source_path),
            "source_sha256": prepared_grid.source_sha256,
            "grid_type": run.grid.grid_type,
            "static_information_path": str(prepared_grid.static_information_path),
        },
        "parent_selection_mode": run.parent_selection_mode,
        "parent_selection_label": run.mode_label,
        "seed": run.seed,
        "parameters": parameters.model_dump(mode="json"),
        "repertoire_layout": {
            "descriptor_names": [descriptor.metric for descriptor in descriptor_definitions],
            "n_cells_per_dim": list(n_cells_per_dim),
            "cell_depth": parameters.ga_config.cell_depth,
            "n_logical_cells": math.prod(n_cells_per_dim),
            "descriptor_resolution": (
                {
                    "auto_metrics": list(descriptor_resolution.auto_metrics),
                    "max_logical_cells": descriptor_resolution.max_logical_cells,
                }
                if descriptor_resolution is not None
                else None
            ),
        },
    }


def _progress_postfix(record: dict[str, Any]) -> dict[str, Any]:
    """Build the compact nested-tqdm postfix for one epoch record."""
    quality = record["quality"]
    archive = record["archive"]
    execution = record["execution"]
    return {
        "cand/s": f"{execution['branch_combinations_per_second']:.3g}",
        "epoch": record["epoch"],
        "iter": record["jax_iteration"],
        "fit": f"{quality['best_fitness']:.3g}",
        "imp": f"{quality['fitness_improvement']:.3g}",
        "qd": f"{archive['qd_score_improvement']:.3g}",
        "cov": f"{archive['cell_coverage']:.1%}",
        "occ": f"{archive['occupied_candidates']}/{archive['n_flat_slots']}",
    }


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON-safe record and flush it for interruption resilience."""
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(_to_json_value(data), allow_nan=False, sort_keys=True))
        file_handle.write("\n")
        file_handle.flush()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON document with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(_to_json_value(data), allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load a JSON mapping when it exists, otherwise return ``None``."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _to_json_value(value: object) -> object:
    """Convert dataclass, JAX, NumPy, and path values to strict JSON values."""
    if value is None or isinstance(value, (str, bool)):
        json_value: object = value
    elif isinstance(value, Path):
        json_value = str(value)
    elif isinstance(value, (int, np.integer)):
        json_value = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        json_value = numeric_value if math.isfinite(numeric_value) else None
    elif isinstance(value, np.ndarray):
        json_value = _to_json_value(value.tolist())
    elif isinstance(value, dict):
        json_value = {str(key): _to_json_value(item) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        json_value = [_to_json_value(item) for item in value]
    elif hasattr(value, "item"):
        json_value = _to_json_value(value.item())
    elif hasattr(value, "tolist"):
        json_value = _to_json_value(value.tolist())
    else:
        raise TypeError(f"Cannot serialize benchmark value of type {type(value).__name__}.")
    return json_value


def _as_int(value: object) -> int:
    """Convert a scalar JAX or NumPy value to a Python integer."""
    return int(np.asarray(jax.device_get(value)).item())


def _as_float(value: object) -> float | None:
    """Convert a scalar JAX or NumPy value to a finite Python float."""
    numeric_value = float(np.asarray(jax.device_get(value)).item())
    return numeric_value if math.isfinite(numeric_value) else None


def _validate_grids(grids: Sequence[GridSpecification]) -> None:
    """Validate unique grid identifiers and source-file existence."""
    grid_ids = [grid.identifier for grid in grids]
    if len(set(grid_ids)) != len(grid_ids):
        raise ValueError("Grid identifiers must be unique.")
    for grid in grids:
        if not grid.source_path.is_file():
            raise FileNotFoundError(f"Configured grid does not exist: {grid.source_path}")


def _validate_modes_and_seeds(modes: Sequence[ParentSelectionMode], seeds: Sequence[int]) -> None:
    """Validate supported modes and a distinct shared seed set."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("Benchmark seeds must be distinct.")
    for mode in modes:
        if mode not in PARENT_SELECTION_LABELS:
            raise ValueError(f"Unsupported parent-selection mode: {mode}")
