# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Benchmark the scaling of the outer AC topology evaluator."""

from time import perf_counter
from unittest.mock import Mock

import pytest
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_topology_optimizer.ac.scoring_functions import ACScoringParameters, score_strategy_full_batch
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics


def _make_topology(index: int) -> ACOptimTopology:
    """Create a minimal topology for a full AC scoring batch."""
    return ACOptimTopology(
        actions=[index],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=f"topology-{index}".encode(),
        optimization_id="ac-parallelization-benchmark",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=["c1"],
    )


def _scoring_parameters() -> ACScoringParameters:
    """Create scoring parameters that do not reject the synthetic results."""
    return ACScoringParameters(
        reject_convergence_threshold=1.0,
        reject_overload_threshold=0.95,
        reject_critical_branch_threshold=1.1,
        reject_voltage_jump_threshold=1.1,
        reject_critical_va_diff_threshold=1.1,
        enable_critical_voltage_rejection=False,
        critical_voltage_jump_percent=5.0,
        critical_va_diff_degree=0.0,
        base_case_id=None,
        early_stop_validation=True,
    )


@pytest.mark.performance
def test_ac_scoring_thread_scaling_for_cpu_bound_runner(
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """Measure whether outer threads speed up CPU-bound runner work.

    The synthetic runner deliberately holds the GIL, modelling Python-side topology
    preparation and result processing around an AC backend invocation. A process-based
    implementation should be benchmarked separately with a real backend runner.
    """

    def cpu_bound_loadflow(*args, **kwargs) -> LoadflowResultsPolars:
        del args, kwargs
        state = 0
        for _ in range(2_000_000):
            state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        assert state >= 0
        return Mock(spec=LoadflowResultsPolars)

    def fake_compute_loadflow_and_metrics(*, runner, topology, **kwargs):
        del kwargs
        return (
            runner.run_ac_loadflow(topology.actions, topology.disconnections, topology.pst_setpoints),
            None,
            Metrics(fitness=float(topology.actions[0]), extra_scores={}),
        )

    monkeypatch.setattr(
        "toop_engine_topology_optimizer.ac.scoring_functions.compute_loadflow_and_metrics",
        fake_compute_loadflow_and_metrics,
    )
    topologies = [_make_topology(index) for index in range(4)]
    scoring_params = _scoring_parameters()

    serial_runner = Mock(spec=AbstractLoadflowRunner)
    serial_runner.run_ac_loadflow.side_effect = cpu_bound_loadflow
    serial_start = perf_counter()
    serial_results = score_strategy_full_batch(
        topologies=topologies,
        runner_groups=[serial_runner],
        metrics_unsplit=Metrics(fitness=0.0, extra_scores={}),
        scoring_params=scoring_params,
    )
    serial_elapsed_seconds = perf_counter() - serial_start

    parallel_runners = [Mock(spec=AbstractLoadflowRunner), Mock(spec=AbstractLoadflowRunner)]
    for runner in parallel_runners:
        runner.run_ac_loadflow.side_effect = cpu_bound_loadflow
    parallel_start = perf_counter()
    parallel_results = score_strategy_full_batch(
        topologies=topologies,
        runner_groups=parallel_runners,
        metrics_unsplit=Metrics(fitness=0.0, extra_scores={}),
        scoring_params=scoring_params,
    )
    parallel_elapsed_seconds = perf_counter() - parallel_start

    record_property("ac_scoring_serial_runner_count", 1)
    record_property("ac_scoring_parallel_runner_count", len(parallel_runners))
    record_property("ac_scoring_serial_elapsed_seconds", serial_elapsed_seconds)
    record_property("ac_scoring_parallel_elapsed_seconds", parallel_elapsed_seconds)
    record_property("ac_scoring_thread_scaling_speedup", serial_elapsed_seconds / parallel_elapsed_seconds)

    assert len(serial_results) == len(topologies)
    assert len(parallel_results) == len(topologies)
