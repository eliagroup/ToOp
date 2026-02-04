# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines the schema for the messages sent to the results topic.

The optimizer will push results to a results topic.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field, NonNegativeInt, field_validator
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType


class Metrics(BaseModel):
    """Result message containing the metrics of the optimization."""

    fitness: float
    """The current best fitness value, which is the optimized quantity"""

    extra_scores: dict[MetricType, float]
    """Additional metrics such as max_flow_n_0, etc.
    """

    worst_k_contingency_cases: Optional[list[str]] = None
    """For contingency analysis, a list of strings (length k) represents the case ids
    of the k worst contingencies.
    """


class Topology(BaseModel):
    """A results class encapsulating a single topology for a single timestep in action index format."""

    actions: list[NonNegativeInt]
    """The branch/injection reconfiguration actions as indices into the action set. This can only be parsed
    when combined with the action set. This includes only actions that are actually used, i.e. no int_max entries.
    Actions will be automatically sorted in ascending order, as the order does not matter and we want
    to avoid duplicates."""

    disconnections: list[NonNegativeInt]
    """The applied disconnections in this topology as an index into the disconnectable branches set in the action set.
    This includes only disconnections that are actually used, i.e. no int_max entries.
    Disconnections will be automatically sorted in ascending order, as the order does not matter and we want
    to avoid duplicates."""

    pst_setpoints: list[int]
    """The setpoints for the PSTs if they have been computed. This is an index into the range of pst taps, i.e. the
    smallest tap is 0 and the neutral tap somewhere in the middle of the range. The tap range is defined in the action set.
    The list always has the same length, i.e. the number of controllable PSTs in the system, and each entry corresponds to
    the PST at the same position in the action set.
    TODO currently this is just the angle converted to int, will be changed later to be the actual tap position.
    """

    metrics: Metrics
    """The metrics of this topology and timestep. While the optimizer might optimize for all
    timesteps, these metrics should be broken down on a per-timestep basis if possible as they will
    be summed up before processing."""

    loadflow_results: Optional[StoredLoadflowReference] = None
    """The loadflow results of this topology in this timestep, if they were computed. Mostly for AC
    results, as for DC we don't need to send full loadflow results every time.
    This is a reference to the loadflow results stored on disk as sending them every time would be too large.
    """

    @field_validator("actions")
    @classmethod
    def sort_actions(cls, v: list[NonNegativeInt]) -> list[NonNegativeInt]:
        """Sort the actions in ascending order. This is important for the hashing."""
        return sorted(v)

    @field_validator("disconnections")
    @classmethod
    def sort_disconnections(cls, v: list[NonNegativeInt]) -> list[NonNegativeInt]:
        """Sort the disconnections in ascending order. This is important for the hashing."""
        return sorted(v)


class Strategy(BaseModel):
    """A series of topologies over multiple timesteps. This is basically a list of topologies,

    one for every timestep.
    """

    timesteps: list[Topology]
    """The topologies for every timestep"""


class TopologyPushResult(BaseModel):
    """A message that pushes new topology results. These must include the topo-vects

    and disconnections and may include loadflow results.
    """

    message_type: Literal["topology_push"] = "topology_push"
    """The result type, don't change this"""

    strategies: list[Strategy]
    """The strategies to be pushed to the master. Each strategy contains a list of timestep-
    topologies, one for every timestep that was optimized (i.e. Strategy.timesteps have the
    same length for all strategies)"""

    epoch: Optional[int] = None
    """The epoch of the optimization run. Enables plotting the results over time on backend side."""

    @field_validator("strategies", mode="after")
    @classmethod
    def strategy_same_length(cls, v: list[Strategy]) -> list[Strategy]:
        """Ensure that all strategies have the same number of timesteps."""
        if len(set(len(strategy.timesteps) for strategy in v)) > 1:
            raise ValueError("All strategies must have the same number of timesteps")
        return v


class OptimizationStoppedResult(BaseModel):
    """A message that is sent if the optimization was successfully stopped."""

    message_type: Literal["stopped"] = "stopped"
    """The result type, don't change this"""

    reason: Literal["error", "stopped", "converged", "ac-not-converged", "unknown"] = "unknown"
    """The reason why the optimization was stopped

    Possible values:
    - Error means an unexpected exception was raised/the optimizer crashed
    - Stopped means it was stopped through a command
    - Converged means it stopped after hitting the convergence criterium
    - ac-not-converged means the AC convergence in the base grid was too poor to run an AC optimization, this will
      only be sent by the AC optimizer.
    """

    message: str = ""
    """A message that can be sent with the stop message, i.e. an error message"""

    epoch: Optional[int] = None
    """The epoch of the optimization run when it was stopped."""


class OptimizationStartedResult(BaseModel):
    """Result message for when an optimization run has just started."""

    message_type: Literal["optimization_started"] = "optimization_started"
    """The result type, don't change this"""

    initial_topology: Strategy
    """The initial topology including the starting metrics of the optimization. This is the topology
    without any splits or disconnections"""

    initial_stats: Optional[list[StaticInformationStats]] = None
    """The initial statistics of the optimization run, i.e. the number of disconnections, etc. This is only filled
    by the DC optimizer and will hold one entry for each timestep in the initial topology."""


ResultUnion: TypeAlias = Union[
    TopologyPushResult,
    OptimizationStoppedResult,
    OptimizationStartedResult,
]


class Result(BaseModel):
    """A generic result message sent back from the optimizer to the results topic.

    Results can be read by either the backend or another optimizer in the waterfall.
    """

    result: ResultUnion = Field(discriminator="message_type")
    """The actual result message"""

    optimization_id: str
    """The optimization id as sent in the commands"""

    optimizer_type: OptimizerType
    """On which type of optimizer created this result"""

    instance_id: str = ""
    """The instance id of the optimizer that created this result"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this result message, used to avoid duplicates during processing"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the result was sent"""
