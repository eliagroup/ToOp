# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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

from beartype.typing import Literal, Optional, TypeAlias, Union
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

    pst_setpoints: list[int] | None
    """The setpoints for the PSTs if they have been computed.

    These are the taps as they are stored in the original grid model, i.e. if a PST has a tap range from -20 to 20 then this
    can take all these values. Note that inside the dc optimizer, taps are always starting with 0 and have to be converted
    by adding grid_model_low_tap.

    The list has the length of the number of controllable PSTs in the grid model and the nth entry corresponds to the nth
    controllable PST in the network data.

    If the PST taps were not optimized, then this is None. Empty list is only allowed if there are no controllable PSTs in
    the grid model.
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


RejectionCriterion: TypeAlias = Literal[
    "convergence", "voltage-magnitude", "voltage-angle", "overload-energy", "critical-branch-count", "other"
]


class TopologyRejectionReason(BaseModel):
    """The reason for rejecting a topology including a criterion that was violated and further data."""

    criterion: RejectionCriterion
    """The criterion that was violated for the rejection."""

    description: Optional[str] = None
    """A more detailed description of the rejection, e.g. which lines were overloaded, etc."""

    value_after: float
    """The value of the metric that caused the rejection after applying the strategy.

    Depending on the criterion, this has different meanings:

    - For convergence this is the number of non-converging loadflows.
    - For voltage magnitude this is the maximum voltage violation in p.u.
    - For voltage angle this is the maximum voltage angle violation in degrees.
    - For overload energy this is the overload in MW.
    - For critical branch count this is the number of critical branches, i.e. branches above their operational limit.
    """

    value_before: float
    """The value of the metric that caused the rejection before applying the strategy.

    Meaning of the value is similar to value_after dependent on the criterion.
    """

    threshold: Optional[float] = None
    """The threshold for rejection that was set in the AC configuration, if applicable."""

    early_stopping: bool = False
    """Whether this rejection was part of an early stopping run, i.e. only a subset of the cases were computed to determine
    the rejection. In that case, the value_before and value_after are only based on the subset of cases that were computed.
    """


def get_topology_rejection_message(result: TopologyRejectionReason) -> str:
    """Condense a TopologyRejectionReason into a human-readable message for logging purposes."""
    message_map = {
        "convergence": "Rejecting topology due to insufficient convergence",
        "overload-energy": "Rejecting topology due to overload energy not improving",
        "critical-branch-count": "Rejecting topology due to critical branches increasing too much",
        "voltage-magnitude": "Rejecting topology due to voltage magnitude violation",
        "voltage-angle": "Rejecting topology due to voltage angle violation",
        "other": "Rejecting topology due to other reason",
    }

    base_message = message_map.get(result.criterion, "Rejecting topology due to unknown reason")
    base_message = (
        f"{base_message} (value before: {result.value_before}, "
        f"value after: {result.value_after}, early_stopping: {result.early_stopping})."
    )
    if result.description:
        base_message = f"{base_message} Details: {result.description}"
    return base_message


class TopologyRejectionResult(BaseModel):
    """A rejection of a topology in a later stage.

    For example if the DC optimizer found a topology as sensible but the AC optimizer rejects it, this will result in a
    rejection result. For the moment, this is only used in the backend to gather statistics on rejections, however a future
    version could implement some sort of learning algorithm that uses these rejections to try and extrapolate the reasons.
    """

    message_type: Literal["topology_rejection"] = "topology_rejection"
    """The result type, don't change this"""

    strategy: Strategy
    """The strategy that is rejected. Note that only a single strategy is rejected at a time, if multiple were rejected then
    send multiple messages."""

    epoch: Optional[int] = None
    """The epoch of the optimization run. Enables plotting the results over time on backend side."""

    reason: TopologyRejectionReason
    """The reason for the rejection along with additional data

    This is separated into a different class to make it easier to pass along inside the code, as it turns out often it is
    not required to read the strategy itself.
    """


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

    reason: Literal["error", "stopped", "converged", "ac-not-converged", "dc-not-started", "unknown"] = "unknown"
    """The reason why the optimization was stopped

    Possible values:
    - Error means an unexpected exception was raised/the optimizer crashed
    - Stopped means it was stopped through a command
    - Converged means it stopped after hitting the convergence criterium
    - ac-not-converged means the AC convergence in the base grid was too poor to run an AC optimization, this will
      only be sent by the AC optimizer.
    - dc-not-started means the DC optimization results did not arrive, potentially due to a suspected failure on dc side and
      the optimization was abandoned. This will only be sent by the AC optimizer.
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
    TopologyRejectionResult,
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
