# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Database models for topology and stage evaluation storage.

This is equivalent to the "results" topic from the previous architecture, note that "commands" and "heartbeats" live
in ``command_models.py``.

The lifecycle of a topology begins in a discovery stage. This is usually the DC stage, but other stages could also discover
new topology. When that happens, a ``Topology`` is inserted into the database and never modified again. The
canonical element holds everything that is required to define the topology itself (actions, disconnections, pst setpoints)
but does not yet contain metrics. When an evaluation runs in any stage (including the discovery stage), metrics for this
topology are generated and stored in a StageTopologyEvaluation.

For future multi-timestep support, topologies are furthermore grouped into strategies, but this is not fully supported at the
moment.

A major design decision revolves around the StageTopologyEvaluation lifecycle. The most naive approach would be:
- A stage picks up a topology, evaluates it and then inserts a new StageTopologyEvaluation row
Here we choose a slightly different semantic
- Upon the topology discovery/previous stage evaluation, a StageTopologyEvaluation for the next stage is already inserted
into the database, but with a TRIGGERED flag. When a stage starts evaluation of a topology, it moves it to RUNNING and then
to a terminal state ACCEPTED/WARN/REJECT. Except for the terminal stage (AC), all stages have the responsibility to create
StageTopologyEvaluation objects for the next stage if they deem a topology feasible for evaluation by that stage (e.g. it is
in ACCEPTED or WARN category.

The stage workers are free to use the StageTopologyEvaluation table as an additional synchronization mechanism for in-worker
parallelism, i.e. if two threads within the worker evaluate topologies in parallel they can sync using the table.

Worst-k evaluation requires information from the previous stage - which N-1 cases were the most severe ones. Concretely,
the AC-FAST-FAIL stage requires information from the DC stage. Now, the worst-k cases could be stored either in the
Evaluation entry of the early (DC) or the later (AC-FAST-FAIL) stage. As the worst-k cases are a denser form of metrics, it
semantically seems to be preffered to write them in the Evaluation of the stage that computed them.

In contrast to StageWorkItems, StageTopologyEvaluations do not carry a lease. This makes a recovery of individual failed
worker threads impossible. Instead, we assume that if a stage worker fails then all worker threads from the job will fail.
The cleanup routine in utils.py performs a cleanup of stale StageTopologyEvaluations, everything that is in RUNNING can be
reset to TRIGGERED. This must happen in a transaction together with the reset of the StageWorkItem. If not, there would be a
possible race condition in case of a too short lease time where a worker would still be working, trying to write results.
However, as the workers check the StageWorkItem for a possible termination (cancellation or lease expiry) __before__ writing
results to StageTopologyEvaluation, this is remedied.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, cast
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.database.json_adapter import TypedJson
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import TopologyRejectionReason


ACTION_INDEX_LIST_JSON: TypedJson[list[int]] = TypedJson(list[int])
DISCONNECTION_INDEX_LIST_JSON: TypedJson[list[int]] = TypedJson(list[int])
PST_SETPOINT_LIST_JSON: TypedJson[list[int]] = TypedJson(list[int])
METRIC_SCORES_JSON: TypedJson[dict[MetricType, float]] = TypedJson(dict[MetricType, float])
WORST_CASE_IDS_JSON: TypedJson[list[str]] = TypedJson(list[str])
LOADFLOW_REFERENCE_JSON: TypedJson[StoredLoadflowReference] = TypedJson(StoredLoadflowReference)
REJECTION_REASON_JSON: TypedJson[TopologyRejectionReason] = TypedJson(TopologyRejectionReason)


class StageEvaluationStatus(str, Enum):
    """The lifecycle state of a stage topology evaluation row."""

    # Transient stages
    TRIGGERED = "triggered"
    """The strategy evaluation was triggered, i.e. an evaluation of the strategy in this stage shall happen."""

    RUNNING = "running"
    """The evaluation is happening right now, i.e. a worker is computing the loadflow"""

    # Terminal stages
    ACCEPTED = "accepted"
    """The topology passed the stage, it can be shown to operators or processed further"""

    WARN = "warn"
    """There are some doubts, but no clear rejection reason"""
    
    REJECTED = "rejected"
    """The topology was rejected"""


class Topology(SQLModel, table=True):
    """The immutable raw topology payload for one timestep.
    
    Immutability is not checked on database level, and a cleanup job might still want to delete old records, however
    application logic should not delete any Topology entries.

    Every topology belongs to exactly one strategy and represents one
    concrete timestep within that strategy, even though multi-timestep support is currently not fully developed.


    """


    __tablename__ = cast(Any, "topology")

    __table_args__ = (
        UniqueConstraint("strategy_id", "timestep", name="uq_topology_timestep"),
    )

    id: UUID | None = Field(default=None, primary_key=True)
    """The primary key as a topology UUID. We choose a UUID to persist topology IDs even when the optimizer database is
    wiped so they can be disambiguated in the persisted api-service database."""

    strategy_id: UUID = Field(foreign_key="strategy.id", nullable=False, index=True)
    """The strategy this topology belongs to."""

    strategy: "Strategy" = Relationship(back_populates="topologies")
    """The strategy this topology belongs to."""

    timestep: int = Field(ge=0, nullable=False)
    """The timestep index of this topology within its parent strategy."""

    topology_hash: bytes = Field(nullable=False, index=True)
    """A stable hash over actions, disconnections and PST setpoints.

    This is intentionally not globally unique because identical raw topology
    payloads may occur in different strategies or different optimization jobs.
    """

    actions: list[int] = Field(default_factory=list, sa_type=cast(Any, ACTION_INDEX_LIST_JSON))
    """Substation splits as action set indices applied in this topology."""

    disconnections: list[int] = Field(default_factory=list, sa_type=cast(Any, DISCONNECTION_INDEX_LIST_JSON))
    """Disconnectable-branch indices disconnected in this topology."""

    pst_setpoints: list[int] | None = Field(default=None, sa_type=cast(Any, PST_SETPOINT_LIST_JSON))
    """Absolute PST setpoints for controllable PSTs, if present."""

    unsplit: bool = Field(nullable=False)
    """Whether this topology represents the unsplit baseline configuration."""

    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """When this topology row was first persisted."""

    topology_evaluations: list["StageTopologyEvaluation"] = Relationship(
        back_populates="topology", cascade_delete=True
    )
    """All stage evaluations written against this topology."""


class Strategy(SQLModel, table=True):
    """The ordered identity of a multi-timestep strategy."""

    __tablename__ = cast(Any, "strategy")

    __table_args__ = (
        UniqueConstraint("optimization_job_id", "strategy_hash", name="uq_strategy"),
    )

    id: UUID | None = Field(default=None, primary_key=True)
    """The primary key."""

    optimization_job_id: UUID = Field(foreign_key="optimization_job.id", nullable=False, index=True)
    """The optimization job this strategy belongs to."""

    strategy_hash: bytes = Field(nullable=False, index=True)
    """A stable hash over the ordered sequence of topologies."""

    n_timesteps: int = Field(ge=1, nullable=False)
    """How many timestep members belong to this strategy."""

    unsplit: bool = Field(nullable=False)
    """Whether all timestep members are unsplit baseline topologies."""

    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """When this strategy row was first persisted."""

    topologies: list[Topology] = Relationship(back_populates="strategy", cascade_delete=True)
    """The ordered topology rows that make up this strategy."""


class StageTopologyEvaluation(SQLModel, table=True):
    """A stage-specific evaluation of one topology.

    The evaluation is owned by ``Topology`` rather than by a strategy.
    Since every topology belongs to exactly one strategy, no extra occurrence
    indirection is needed.

    Unlike a purely append-only result table, this row is also the scheduling
    surface for the next stage on topology granularity. It is created in
    ``TRIGGERED``, then moved to ``RUNNING`` by the stage, and finally to one
    of the terminal states ``ACCEPTED``, ``WARN`` or ``REJECTED``.
    """

    __tablename__ = cast(Any, "stage_topology_evaluation")

    __table_args__ = (
        UniqueConstraint("topology_id", "stage", name="uq_stage_topology_evaluation"),
    )

    id: int | None = Field(default=None, primary_key=True)
    """The primary key."""

    optimization_job_id: UUID = Field(foreign_key="optimization_job.id", nullable=False, index=True)
    """The optimization job this topology evaluation belongs to."""

    stage_execution_history_id: int | None = Field(default=None, foreign_key="stage_execution_history.id", nullable=True, index=True)
    """The execution attempt currently responsible for this topology evaluation.

    This is ``None`` while the row is only triggered for future work. Once a
    worker starts evaluation, it can be set to the corresponding execution
    attempt from ``StageExecutionHistory``.
    """

    stage: OptimizerType = Field(nullable=False, index=True)
    """The optimizer stage that shall produce this topology evaluation."""

    topology_id: UUID = Field(foreign_key="topology.id", nullable=False, index=True)
    """The topology that was evaluated."""

    topology: Topology = Relationship(back_populates="topology_evaluations")
    """The topology that was evaluated."""

    status: StageEvaluationStatus = Field(default=StageEvaluationStatus.TRIGGERED, nullable=False, index=True)
    """The current lifecycle state of this stage topology evaluation."""

    epoch: int | None = Field(default=None, nullable=True, index=True)
    """The optimization epoch during which this evaluation was written, if any."""

    iteration: int | None = Field(default=None, nullable=True, index=True)
    """The optimization iteration during which this evaluation was written, if any."""

    rejection_reason: TopologyRejectionReason | None = Field(default=None, sa_type=cast(Any, REJECTION_REASON_JSON))
    """The rejection reason if this stage evaluation rejected the topology."""

    fitness: float | None = Field(default=None, nullable=True)
    """The primary objective value reported for this evaluated topology. This must be set if the Evaluation reached
    ACCEPTED or WARN stage."""

    metrics: dict[MetricType, float] = Field(default_factory=dict, sa_type=cast(Any, METRIC_SCORES_JSON))
    """Additional metric values emitted for this evaluated topology."""

    worst_k_contingency_cases: list[str] | None = Field(default=None, sa_type=cast(Any, WORST_CASE_IDS_JSON))
    """The worst contingency-case identifiers reported in this stage, if available."""

    loadflow_reference: StoredLoadflowReference | None = Field(default=None, sa_type=cast(Any, LOADFLOW_REFERENCE_JSON))
    """Optional reference to stored detailed loadflow results for this topology."""

    last_edited: datetime = Field(default_factory=datetime.now, nullable=False, sa_column_kwargs={"onupdate": datetime.now})
    """When this evaluation row was last modified."""

    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """When this stage topology evaluation row was persisted."""