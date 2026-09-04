# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Spawn-safe process workers for outer AC topology parallelism."""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

import pypowsybl
import structlog
from beartype.typing import Optional
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    load_loadflow_results_polars,
    save_loadflow_results_polars,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_topology_optimizer.ac.runner_factory import make_runner
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile

logger = structlog.get_logger()


@dataclass(frozen=True)
class RunnerSpec:
    """Serializable inputs used to construct one runner in each process worker."""

    action_set: ActionSet
    nminus1_definition: Nminus1Definition
    grid_file: GridFile
    contingency_processes: int
    processed_gridfile_fs_json: str
    loadflow_result_fs_json: str
    loadflow_result_prefix: str
    lf_params: pypowsybl.loadflow.Parameters | dict | None


@dataclass
class WorkerContext:
    """Process-local state initialized for each runner worker."""

    runner: Optional[AbstractLoadflowRunner] = None
    loadflow_result_fs: Optional[AbstractFileSystem] = None
    loadflow_result_prefix: Optional[str] = None


_worker_context = WorkerContext()


def _initialize_runner_worker(spec: RunnerSpec) -> None:
    """Construct the worker-local runner after the process has been spawned."""
    processed_gridfile_fs = AbstractFileSystem.from_json(spec.processed_gridfile_fs_json)
    _worker_context.loadflow_result_fs = AbstractFileSystem.from_json(spec.loadflow_result_fs_json)
    _worker_context.loadflow_result_prefix = spec.loadflow_result_prefix
    _worker_context.runner = make_runner(
        action_set=spec.action_set,
        nminus1_definition=spec.nminus1_definition,
        grid_file=spec.grid_file,
        n_processes=spec.contingency_processes,
        batch_size=None,
        processed_gridfile_fs=processed_gridfile_fs,
        lf_params=spec.lf_params,
    )


def get_worker_runner() -> AbstractLoadflowRunner:
    """Return the runner exclusively owned by the current process worker."""
    if _worker_context.runner is None:
        raise RuntimeError("AC runner process worker was not initialized")
    return _worker_context.runner


def store_worker_loadflow_results(loadflow_results: LoadflowResultsPolars) -> StoredLoadflowReference:
    """Store loadflow results in the worker and return their shared-filesystem reference."""
    if _worker_context.loadflow_result_fs is None or _worker_context.loadflow_result_prefix is None:
        raise RuntimeError("AC runner process worker has no loadflow result filesystem")
    return save_loadflow_results_polars(
        _worker_context.loadflow_result_fs,
        f"{_worker_context.loadflow_result_prefix}-{uuid4()}",
        loadflow_results,
    )


def load_worker_loadflow_results(loadflow_reference: StoredLoadflowReference) -> LoadflowResultsPolars:
    """Load a worker-written result from the shared filesystem."""
    if _worker_context.loadflow_result_fs is None:
        raise RuntimeError("AC runner process worker has no loadflow result filesystem")
    return load_loadflow_results_polars(_worker_context.loadflow_result_fs, loadflow_reference)


def _warm_runner_worker() -> int:
    """Run a minimal task after process initialization has completed."""
    get_worker_runner()
    return os.getpid()


def warm_runner_process_pool(process_pool: ProcessPoolExecutor, runner_processes: int) -> list[int]:
    """Start every runner process and wait until its initializer is ready."""
    warmup_futures = [process_pool.submit(_warm_runner_worker) for _ in range(runner_processes)]
    worker_pids = [future.result() for future in warmup_futures]
    logger.info(
        "Warmed AC runner process pool", requested_workers=runner_processes, initialized_workers=len(set(worker_pids))
    )
    return worker_pids


def create_runner_process_pool(spec: RunnerSpec, runner_processes: int) -> ProcessPoolExecutor:
    """Create a spawn-based pool whose workers own isolated AC runners."""
    return ProcessPoolExecutor(
        max_workers=runner_processes,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_runner_worker,
        initargs=(spec,),
    )
