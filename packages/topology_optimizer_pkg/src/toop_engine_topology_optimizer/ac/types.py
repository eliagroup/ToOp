# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared type aliases for the AC optimizer implementation."""

from dataclasses import dataclass

from beartype.typing import Callable, Optional, TypeAlias
from numpy.random import Generator as Rng
from sqlmodel import Session
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, TopologyRejectionReason

ACStrategy: TypeAlias = list[ACOptimTopology]
RunnerGroup: TypeAlias = list[AbstractLoadflowRunner]


@dataclass
class TopologyScoringResult:
    """Scoring output for a single AC strategy."""

    loadflow_results: Optional[LoadflowResultsPolars]
    metrics: Metrics
    rejection_reason: Optional[TopologyRejectionReason]


@dataclass
class EarlyStoppingStageResult(TopologyScoringResult):
    """Intermediate result after the worst-k stage for a single strategy."""

    cases_subset: Optional[list[str]]


@dataclass
class OptimizerData:
    """The epoch-to-epoch storage for the AC optimizer"""

    params: ACOptimizerParameters
    """The parameters this optimizer was initialized with"""

    session: Session
    """A in-memory sqlite session for storing the repertoire"""

    evolution_fn: Callable[[], ACStrategy]
    """The curried evolution function"""

    scoring_fn: Callable[[ACStrategy, Optional[list[EarlyStoppingStageResult]]], list[TopologyScoringResult]]
    """The curried batched scoring function."""

    worst_k_scoring_fn: Callable[[ACStrategy], list[EarlyStoppingStageResult]]
    """The curried batched worst-k scoring function."""

    store_loadflow_fn: Callable[[LoadflowResultsPolars], StoredLoadflowReference]
    """The function to store loadflow results"""

    load_loadflow_fn: Callable[[StoredLoadflowReference], LoadflowResultsPolars]
    """The function to load loadflow results"""

    rng: Rng
    """The random number generator for the optimizer"""

    action_set: ActionSet
    """The action set for the grid file"""

    framework: Framework
    """The framework of the grid file"""

    runners: RunnerGroup
    """The initialized loadflow runners, one for each configured grid file."""

    worst_k_runner_groups: RunnerGroup
    """Dedicated runners for the worst-k stage"""
