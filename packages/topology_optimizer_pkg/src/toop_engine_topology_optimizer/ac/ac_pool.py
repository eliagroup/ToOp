# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Process pools for the AC loadflow stages.

The N-1 translation and metrics around the AC loadflow are GIL-bound Python work, so threads do not scale. Every
worker process loads its own runner once and then scores one topology per task. One executor is created per stage
(worst-k and remaining/full contingencies), as the stages use different runner settings.
"""

import multiprocessing as mp
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime

import structlog
from beartype.typing import Any, Callable, Generator, Optional, Sequence
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_result_helpers_polars import load_loadflow_results_polars, save_loadflow_results_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_topology_optimizer.ac.runner import load_preprocessing_inputs, make_runner
from toop_engine_topology_optimizer.ac.scoring_functions import (
    ACScoringParameters,
    error_result_for_topology,
    score_strategy_full,
    score_strategy_worst_k,
    score_topology_remaining,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import EarlyStoppingStageResult, TopologyScoringResult
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

logger = structlog.get_logger(__name__)


@dataclass
class AcWorkerSpec:
    """Everything a worker process needs to build its runner and score topologies. Must be picklable."""

    grid_file: GridFile
    """The grid file the runner is loaded from"""

    processed_gridfile_fs: AbstractFileSystem
    """The filesystem containing the processed grid files"""

    n_contingency_processes: int
    """Number of processes (OpenLoadFlow threads) each worker's runner uses for the contingencies"""

    batch_size: Optional[int]
    """Maximum number of contingencies per security-analysis run of the runner, None for a single run"""

    result_filter: Optional[LoadflowResultFilter]
    """Policy for dropping result rows without decision value"""

    metrics_unsplit: Metrics
    """The metrics of the unsplit topology"""

    scoring_params: ACScoringParameters
    """The parameters for scoring and acceptance"""

    loadflow_result_fs: AbstractFileSystem
    """The filesystem the unsplit loadflow results are loaded from"""

    initial_loadflow_reference: Optional[StoredLoadflowReference]
    """Reference to the unsplit loadflow results. Only the worst-k stage needs them, so it is None for the others
    to save the memory in every worker."""

    optimization_id: str
    """The optimization the workers store loadflow results for, used in the file names"""


@dataclass
class _WorkerState:
    """What a worker process keeps between tasks."""

    runner: AbstractLoadflowRunner
    loadflow_unsplit: Optional[LoadflowResultsPolars]
    spec: AcWorkerSpec


_worker_state: Optional[_WorkerState] = None


def _init_worker(spec: AcWorkerSpec) -> None:
    """Load the runner and, if requested, the unsplit loadflow results once per worker process."""
    global _worker_state  # noqa: PLW0603
    action_set, nminus1_definition, lf_params = load_preprocessing_inputs(spec.grid_file, spec.processed_gridfile_fs)
    runner = make_runner(
        action_set,
        nminus1_definition,
        spec.grid_file,
        n_processes=spec.n_contingency_processes,
        batch_size=spec.batch_size,
        processed_gridfile_fs=spec.processed_gridfile_fs,
        lf_params=lf_params,
        result_filter=spec.result_filter,
    )
    loadflow_unsplit = (
        None
        if spec.initial_loadflow_reference is None
        else load_loadflow_results_polars(spec.loadflow_result_fs, reference=spec.initial_loadflow_reference)
    )
    _worker_state = _WorkerState(runner=runner, loadflow_unsplit=loadflow_unsplit, spec=spec)


def _store_in_worker(result: TopologyScoringResult) -> TopologyScoringResult:
    """Write the loadflow frames to the result filesystem and return only the reference.

    The frames are hundreds of MB to GB, and unpickling them in the parent is single-threaded and starves the workers.
    """
    assert _worker_state is not None, "Worker was not initialized"
    loadflow = result.loadflow_results
    if loadflow is None:
        return result
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    name = f"{_worker_state.spec.optimization_id}-{loadflow.job_id}-{timestamp}-{os.getpid()}"
    result.loadflow_reference = save_loadflow_results_polars(_worker_state.spec.loadflow_result_fs, name, loadflow)
    result.loadflow_results = None
    return result


def _score_worst_k_in_worker(topology_dump: dict) -> EarlyStoppingStageResult:
    """Score one topology on the worst-k contingencies in a worker process, see `score_strategy_worst_k`."""
    assert _worker_state is not None, "Worker was not initialized"
    assert _worker_state.loadflow_unsplit is not None, "Worker has no unsplit loadflow results"
    result = score_strategy_worst_k(
        ACOptimTopology(**topology_dump),
        _worker_state.runner,
        _worker_state.loadflow_unsplit,
        _worker_state.spec.metrics_unsplit,
        _worker_state.spec.scoring_params,
    )
    return _store_in_worker(result)  # type: ignore[return-value]


def _score_full_in_worker(topology_dump: dict) -> TopologyScoringResult:
    """Score one topology on all contingencies in a worker process, see `score_strategy_full`."""
    assert _worker_state is not None, "Worker was not initialized"
    return _store_in_worker(
        score_strategy_full(
            ACOptimTopology(**topology_dump),
            _worker_state.runner,
            _worker_state.spec.metrics_unsplit,
            _worker_state.spec.scoring_params,
        )
    )


def _score_remaining_in_worker(topology_dump: dict, early_stage_result: EarlyStoppingStageResult) -> TopologyScoringResult:
    """Score one topology on the contingencies left after the worst-k stage, see `score_topology_remaining`."""
    assert _worker_state is not None, "Worker was not initialized"
    spec = _worker_state.spec
    reference = early_stage_result.loadflow_reference
    if early_stage_result.loadflow_results is None and reference is not None:
        early_stage_result = replace(
            early_stage_result,
            loadflow_results=load_loadflow_results_polars(spec.loadflow_result_fs, reference=reference),
            loadflow_reference=None,
        )
    result = _store_in_worker(
        score_topology_remaining(
            ACOptimTopology(**topology_dump),
            _worker_state.runner,
            spec.metrics_unsplit,
            spec.scoring_params,
            early_stage_result,
        )
    )
    # A surviving topology's worst-k frames were only an intermediate; rejected ones are already final results.
    if reference is not None and early_stage_result.rejection_reason is None:
        try:
            spec.loadflow_result_fs.rm(reference.relative_path, recursive=True)
        except Exception:
            logger.warning("Could not remove intermediate worst-k loadflow results", path=reference.relative_path)
    return result


def create_ac_executor(n_workers: int, spec: AcWorkerSpec) -> ProcessPoolExecutor:
    """Create an executor for one AC stage.

    Workers are spawned on the first submission, up to ``n_workers``, and live until the executor is shut down.

    Parameters
    ----------
    n_workers : int
        The maximum number of worker processes. Each one holds its own copy of the grid.
    spec : AcWorkerSpec
        What every worker needs to build its runner.

    Returns
    -------
    ProcessPoolExecutor
        The executor. Use it as a context manager or call ``shutdown`` to stop the workers.
    """
    # spawn: the Graal runtime, JAX and Kafka clients are not fork-safe
    return ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(spec,)
    )


def get_worker_environment(blas_threads: int, polars_threads: int) -> dict[str, str]:
    """Get the thread limits for the worker processes, so that n workers do not oversubscribe the cores.

    Parameters
    ----------
    blas_threads : int
        The number of threads for OpenMP, MKL and OpenBLAS per worker.
    polars_threads : int
        The number of polars threads per worker.

    Returns
    -------
    dict[str, str]
        The environment variables to set while the workers are spawned.
    """
    return {
        "OMP_NUM_THREADS": str(blas_threads),
        "MKL_NUM_THREADS": str(blas_threads),
        "OPENBLAS_NUM_THREADS": str(blas_threads),
        "POLARS_MAX_THREADS": str(polars_threads),
    }


@contextmanager
def _temporary_environ(environment: dict[str, str]) -> Generator[None]:
    """Set environment variables and restore the previous values on exit."""
    saved = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_batch(
    executor: Executor,
    worker_environment: dict[str, str],
    task: Callable[..., Any],
    task_args: Sequence[tuple],
    early_stopping: bool,
) -> list:
    """Submit one task per entry of ``task_args`` and collect the results in order.

    A task that raised is returned as an error rejection. A dead worker raises `BrokenProcessPool`, as the executor can
    not be used afterwards.
    """
    # Workers are spawned inside submit and copy the current environment
    with _temporary_environ(worker_environment):
        futures = [executor.submit(task, *args) for args in task_args]

    results = []
    for future in futures:
        try:
            results.append(future.result())
        except BrokenProcessPool:
            raise
        except Exception as exc:
            logger.exception("AC stage failed", task=task.__name__)
            results.append(error_result_for_topology(str(exc), early_stopping=early_stopping))
    return results


def score_worst_k_with_executor(
    topologies: list[ACOptimTopology],
    executor: Executor,
    worker_environment: dict[str, str],
) -> list[EarlyStoppingStageResult]:
    """Evaluate the worst-k stage for a batch of topologies on the worker processes.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to evaluate.
    executor : Executor
        The executor from `create_ac_executor`, whose workers have the unsplit loadflow results.
    worker_environment : dict[str, str]
        The environment for newly spawned workers, see `get_worker_environment`.

    Returns
    -------
    list[EarlyStoppingStageResult]
        One result per topology in input order. A topology whose evaluation raised is returned as an error rejection.

    Raises
    ------
    BrokenProcessPool
        If a worker died or failed to initialize.
    """
    return _run_batch(
        executor,
        worker_environment,
        _score_worst_k_in_worker,
        [(topology.model_dump(),) for topology in topologies],
        early_stopping=True,
    )


def score_full_with_executor(
    topologies: list[ACOptimTopology],
    executor: Executor,
    worker_environment: dict[str, str],
) -> list[TopologyScoringResult]:
    """Evaluate a batch of topologies on all contingencies on the worker processes.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies to evaluate.
    executor : Executor
        The executor from `create_ac_executor`.
    worker_environment : dict[str, str]
        The environment for newly spawned workers, see `get_worker_environment`.

    Returns
    -------
    list[TopologyScoringResult]
        One result per topology in input order. A topology whose evaluation raised is returned as an error rejection.

    Raises
    ------
    BrokenProcessPool
        If a worker died or failed to initialize.
    """
    return _run_batch(
        executor,
        worker_environment,
        _score_full_in_worker,
        [(topology.model_dump(),) for topology in topologies],
        early_stopping=False,
    )


def score_remaining_with_executor(
    topologies: list[ACOptimTopology],
    early_stage_results: list[EarlyStoppingStageResult],
    executor: Executor,
    worker_environment: dict[str, str],
) -> list[TopologyScoringResult]:
    """Evaluate the contingencies left after the worst-k stage for a batch of topologies on the worker processes.

    Parameters
    ----------
    topologies : list[ACOptimTopology]
        The topologies that survived the worst-k stage.
    early_stage_results : list[EarlyStoppingStageResult]
        The worst-k results of the topologies, in the same order.
    executor : Executor
        The executor from `create_ac_executor`.
    worker_environment : dict[str, str]
        The environment for newly spawned workers, see `get_worker_environment`.

    Returns
    -------
    list[TopologyScoringResult]
        One result per topology in input order. A topology whose evaluation raised is returned as an error rejection.

    Raises
    ------
    ValueError
        If topologies and early stage results differ in length.
    BrokenProcessPool
        If a worker died or failed to initialize.
    """
    if len(topologies) != len(early_stage_results):
        raise ValueError("Topologies and early-stage results must have the same length")
    return _run_batch(
        executor,
        worker_environment,
        _score_remaining_in_worker,
        [(topology.model_dump(), result) for topology, result in zip(topologies, early_stage_results, strict=True)],
        early_stopping=False,
    )
