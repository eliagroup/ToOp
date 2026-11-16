# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock

import pytest
from beartype.typing import Callable
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_topology_optimizer.ac.ac_pool import (
    get_worker_environment,
    score_full_with_executor,
    score_remaining_with_executor,
    score_worst_k_with_executor,
)
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import EarlyStoppingStageResult, TopologyScoringResult
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics


def _topology(action: int) -> ACOptimTopology:
    return ACOptimTopology(
        actions=[action],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=bytes([action]),
        optimization_id="test",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )


def _result(fitness: float) -> EarlyStoppingStageResult:
    return EarlyStoppingStageResult(
        loadflow_results=MagicMock(spec=LoadflowResultsPolars),
        metrics=Metrics(fitness=fitness, extra_scores={}),
        rejection_reason=None,
        cases_subset=["c1"],
    )


def test_get_worker_environment() -> None:
    assert get_worker_environment(blas_threads=1, polars_threads=3) == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "3",
    }


def test_score_worst_k_keeps_order_and_turns_errors_into_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score(topology_dump: dict) -> EarlyStoppingStageResult:
        action = topology_dump["actions"][0]
        if action == 2:
            raise ValueError("boom")
        return _result(float(action))

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.ac_pool._score_worst_k_in_worker", fake_score)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = score_worst_k_with_executor([_topology(1), _topology(2), _topology(3)], executor, {})

    assert results[0].metrics.fitness == 1.0
    assert results[2].metrics.fitness == 3.0
    assert results[1].rejection_reason is not None
    assert results[1].rejection_reason.criterion == "error"
    assert "boom" in results[1].rejection_reason.description


def test_score_worst_k_raises_on_broken_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_score(_topology_dump: dict) -> EarlyStoppingStageResult:
        raise BrokenProcessPool("worker died")

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.ac_pool._score_worst_k_in_worker", broken_score)

    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(BrokenProcessPool):
        score_worst_k_with_executor([_topology(1)], executor, {})


def test_score_worst_k_restores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toop_engine_topology_optimizer.ac.ac_pool._score_worst_k_in_worker", lambda _dump: _result(0.0))
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    seen = []

    class RecordingExecutor(ThreadPoolExecutor):
        def submit(self, fn: Callable, *args: object, **kwargs: object) -> Future:
            seen.append(os.environ.get("POLARS_MAX_THREADS"))
            return super().submit(fn, *args, **kwargs)

    with RecordingExecutor(max_workers=1) as executor:
        score_worst_k_with_executor([_topology(1)], executor, {"POLARS_MAX_THREADS": "2"})

    assert seen == ["2"]
    assert "POLARS_MAX_THREADS" not in os.environ


def test_score_full_keeps_order_and_marks_errors_as_not_early_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score(topology_dump: dict) -> TopologyScoringResult:
        action = topology_dump["actions"][0]
        if action == 2:
            raise ValueError("boom")
        return _result(float(action))

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.ac_pool._score_full_in_worker", fake_score)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = score_full_with_executor([_topology(1), _topology(2), _topology(3)], executor, {})

    assert [results[0].metrics.fitness, results[2].metrics.fitness] == [1.0, 3.0]
    assert results[1].rejection_reason is not None
    assert results[1].rejection_reason.criterion == "error"
    assert not results[1].rejection_reason.early_stopping


def test_score_remaining_passes_early_results_per_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score(topology_dump: dict, early_stage_result: EarlyStoppingStageResult) -> TopologyScoringResult:
        assert early_stage_result.metrics.fitness == float(topology_dump["actions"][0]) * 10
        return _result(float(topology_dump["actions"][0]))

    monkeypatch.setattr("toop_engine_topology_optimizer.ac.ac_pool._score_remaining_in_worker", fake_score)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = score_remaining_with_executor([_topology(1), _topology(2)], [_result(10.0), _result(20.0)], executor, {})

    assert [result.metrics.fitness for result in results] == [1.0, 2.0]


def test_score_remaining_rejects_length_mismatch() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(ValueError, match="same length"):
        score_remaining_with_executor([_topology(1)], [], executor, {})
