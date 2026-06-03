# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Database models for the optimizer work queue.

The database replaces Kafka for optimizer-internal coordination inside ToOp. The
goal of this schema is to keep the request payload, the current stage work item
and the historical execution record separate so that each concern can evolve
without overloading a single table with conflicting responsibilities.

The design is intentionally split into three layers:

- ``OptimizationJob`` stores the immutable optimization request. It is the
    durable replacement for the old start command payload and acts as the anchor
    id referenced by all later stages. It is write-once by the API service.
- ``StageWorkItem`` materializes one row per optimizer stage and job. This row
    persists across pending, running, retry and terminal states and is the queue
    surface workers poll with ``SKIP LOCKED``.
- ``StageExecutionHistory`` stores one append-only row per attempt. It is the
    audit log and does not serve a role in work scheduling.

The sequence of actions is designed as follows:

1. A new optimization job is scheduled by the user, so a row in the OptimizationJob table is created. Also, a row for every
stage in the StageWorkItem table is created in TRIGGERED
2. A worker from each stage finds the StageWorkItem and locks it to itself by setting the status to RUNNING and the lease to
a date in the future.
3. During their work, the workers repeatedly update the lease so no other worker picks up the same job. They also check for
cancellations during this update.
3. Workers finish their work and set their rows to COMPLETED.

We distinguish two types of failure: transient and deterministic failures. "FAILED" means we do not know exactly what went
wrong and would like to attempt a restart. "BLOCKED" means we know that a retry will not solve the problem (e.g. non-
converging initial loadflows).

If a worker fails gracefully (can still write the db), it deletes the lease. If it can not write anymore e.g. due to a
segfault, the lease will just expire. Future workers will see the row, which is in RUNNING state without a valid lease
(SKIP LOCKED will only show it once the database session was killed). They will increase the attempt counter and
- if the attempt counter is less than a configured threshold, reaquire the lease on themselves and keep working.
- if the attempt threshold is exceeded, set the stage to FAILED

If a worker took very long during an epoch and exceeded the lease, two things might happen: either the job has not yet been
picked up - in this case it can keep working, or another worker took over the stage and it has to drop all work
and assume the job is no longer allocated to itself.

To cancel a run, the user shall set every stage work item to CANCELLED upon which the workers will exit on their next update.
Note that setting this from None to CANCELLED is the desired flow, but once it was set and has propagated, this should
never be taken back - otherwise the job could end up in a state where some stages saw the cancellation and dropped work while
others still continue working.


"""

from datetime import datetime
from enum import Enum
from typing import Any, cast
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel
from toop_engine_topology_optimizer.database.json_adapter import TypedJson
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


AC_OPTIMIZER_PARAMETERS_JSON: TypedJson[ACOptimizerParameters] = TypedJson(ACOptimizerParameters)
DC_OPTIMIZER_PARAMETERS_JSON: TypedJson[DCOptimizerParameters] = TypedJson(DCOptimizerParameters)
GRID_FILES_JSON: TypedJson[list[GridFile]] = TypedJson(list[GridFile])


class OptimizationJob(SQLModel, table=True):
    """Represents an optimization request written by the API or listener.

    The optimization job purposely does not hold a mutable aggregate status
    field. Doing so would introduce a second writable source of truth next to
    the stage-level state tables and would make it too easy to create
    inconsistencies during retries, cancellation or worker crashes.

    Instead, job status is inferred from ``StageWorkItem`` and
    ``StageExecutionHistory``:

    - an optimization is RUNNING if there is at least one RUNNING
        ``StageWorkItem`` with a valid lease;
    - an optimization is COMPLETED when all required work items reached
        COMPLETED;
    - an optimization is FAILED if at least one required work item is FAILED
        and no further retry is scheduled.

    To compute this status logic, use the infer_status helper in TODO instead.
    """

    __tablename__ = cast(Any, "optimization_job")

    id: UUID = Field(primary_key=True)
    """The optimization id shared across commands, results and the API database."""

    ac_params: ACOptimizerParameters = Field(sa_type=AC_OPTIMIZER_PARAMETERS_JSON)
    """Json-serialized representation of the AC optimizer parameters in this job."""

    dc_params: DCOptimizerParameters = Field(sa_type=DC_OPTIMIZER_PARAMETERS_JSON)
    """Json-serialized representation of the DC optimizer parameters in this job."""

    grid_files: list[GridFile] = Field(sa_type=GRID_FILES_JSON)
    """Json-serialized representation of the grid file under optimization.

    We currently only support single-timestep operation, but for compatibility reasons we will defer the refactor to single
    timesteps for the moment.
    """

    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """When the optimization job was written into the database. This is equal to the trigger time. If there are two
    optimization jobs with equal priority, the one with the earlier created_at date will take preference."""


    priority: int = 0
    """The priority used when workers claim pending jobs in descending order.

    Higher value jobs will be tackled first. ``created_at`` acts as the stable
    tiebreaker for jobs with identical priority so that queue ordering remains
    deterministic.
    """

    stage_work_items: list["StageWorkItem"] = Relationship(back_populates="optimization_job", cascade_delete=True)
    """The persistent per-stage work items for this optimization job.

    These rows exist before a worker picks
    up work and remain after a worker releases it. That is what makes them
    suitable as a ``FOR UPDATE SKIP LOCKED`` polling surface.
    """

    stage_execution_history: list["StageExecutionHistory"] = Relationship(back_populates="optimization_job", cascade_delete=True)
    """The append-only execution history for this optimization job."""


class StageWorkItemStatus(str, Enum):
    """The status of a materialized stage work item."""

    TRIGGERED = "triggered"
    """The stage is eligible and may be claimed by a worker."""

    RUNNING = "running"
    """The stage is currently owned by a worker and protected by a lease."""

    COMPLETED = "completed"
    """The stage finished successfully."""

    FAILED = "failed"
    """The latest attempt failed and but a retry might be feasible."""

    BLOCKED = "blocked"
    """An attempt ran into a hard blocker/deterministic failure which makes execution of the stage impossible. Future
    execution is not desired"""

    CANCELLED = "cancelled"
    """The stage was cancelled by the user. Future execution is not desired."""


class StageWorkItem(SQLModel, table=True):
    """The materialized queue row for one optimizer stage on one job.

    Exactly one row may exist per optimization job and stage. The row persists
    across the full lifecycle of the stage and is therefore the right surface to
    poll with ``SKIP LOCKED``.

    This table answers the question "what can a worker do next?" It stores the
    hot-path mutable state that determines claimability:

    - whether the stage is pending, running or terminal,
    - whether a running attempt has an active lease,
    - and which attempt number is current.

    Detailed per-attempt metadata is deliberately kept in
    ``StageExecutionHistory`` so that the work-item row can stay compact and
    operational.
    """

    __tablename__ = cast(Any, "stage_work_item")

    __table_args__ = (
        UniqueConstraint("optimization_job_id", "stage", name="uq_stage_work_item"),
    )

    id: int | None = Field(default=None, primary_key=True)
    """The primary key."""

    optimization_job_id: UUID = Field(foreign_key="optimization_job.id", nullable=False, index=True)
    """The optimization job id this stage work item belongs to."""

    optimization_job: OptimizationJob = Relationship(back_populates="stage_work_items")
    """The optimization job this stage work item belongs to."""

    stage: OptimizerType = Field(index=True)
    """Which optimizer stage this work item represents.

    The basic setup of the optimization is a staged run where first dc, then maybe dc+, then fast-failing ac and finally ac
    review the same topology.
    """

    status: StageWorkItemStatus = Field(default=StageWorkItemStatus.TRIGGERED, index=True)
    """The current queue-level status of the stage work item.

    Once the row is inserted, this is set to TRIGGERED. When a worker of the stage picks up the work item, it transitiones it
    to RUNNING and when it finishes it moves it to COMPLETED or FAILED. The FAILED transition is only done if the attempts
    exceed the threshold. However when a crash indicates that a retry is unsuccessful, the work item is instead transitioned
    to BLOCKED and thus it is made sure it will never be picked up again.
    """

    current_attempt: int = Field(default=0, ge=0)
    """The attempt number currently associated with this work item.

    ``0`` means no attempt has been started yet. When a worker claims the work
    item, it increments this number and creates the corresponding
    ``StageExecutionHistory`` row.
    """

    last_edited: datetime = Field(default_factory=datetime.now, nullable=False, sa_column_kwargs={"onupdate": datetime.now})
    """The last edit to this row, which is auto-updated by postgres.
    """

    lease_expires_at: datetime | None = Field(default=None, nullable=True, index=True)
    """When another worker may treat the current RUNNING attempt as stale.

    This field turns a claim into a recoverable lease rather than a hard lock.
    It may be ``None`` when the work item is not currently running.
    """

    execution_history: list["StageExecutionHistory"] = Relationship(
        back_populates="stage_work_item", cascade_delete=True
    )
    """The append-only attempt history for this work item."""


class StageExecutionHistory(SQLModel, table=True):
    """An append-only like record of one stage execution attempt.

    This table stores one row per attempt, created when the attempt is claimed and updated when it reaches a terminal
    state or during an epoch update. This table is not used in the scheduling hot loop but serves as an audit trail.
    It is loosely equivalent to the optimization stats heartbeat in the old kafka based architecture.

    A new row is created for every attempt number. The row then moves through its
    lifecycle from ``RUNNING`` to a terminal state such as ``COMPLETED`` or
    ``FAILED``. The work item may move on to another attempt during that
    lifecycle, but the history row remains. The worker furthermore reports some basic KPIs like number of loadflows computed

    If a run crashes, this might end up stale - the finished at might not be filled even though the optimization crashed and
    thus technically finished.
    """

    __tablename__ = cast(Any, "stage_execution_history")

    __table_args__ = (
        UniqueConstraint("stage_work_item_id", "attempt", name="uq_stage_execution_history_attempt"),
    )

    id: int | None = Field(default=None, primary_key=True)
    """The primary key."""

    stage_work_item_id: int = Field(foreign_key="stage_work_item.id", nullable=False, index=True)
    """The stage work item this history entry belongs to."""

    stage_work_item: StageWorkItem = Relationship(back_populates="execution_history")
    """The stage work item this history entry belongs to."""

    optimization_job_id: UUID = Field(foreign_key="optimization_job.id", nullable=False, index=True)
    """The optimization job this history entry belongs to.

    This is redundant with the work-item reference but convenient for job-scoped
    history queries and explicit referential integrity.
    """

    optimization_job: OptimizationJob = Relationship(back_populates="stage_execution_history")
    """The optimization job this history entry belongs to."""

    attempt: int = Field(default=1, ge=1)
    """Which retry this execution corresponds to.

    When the execution is active, this matches the current_attempt field in the 
    work item. However, the current_attempt field is increased while this will not be changed.
    """

    status: StageWorkItemStatus = Field(default=StageWorkItemStatus.RUNNING, index=True)
    """The current status of this execution attempt.

    A history row starts at ``RUNNING`` when the claim is created and is later
    transitioned to a terminal state. Keeping the status in history instead of
    solely in the work-item row allows later debugging even after the work item
    moved on to a subsequent attempt.
    The state of TRIGGERED will never occur in this table, as a not started execution is indicated by a missing row. We still
    use the same type for portability
    """

    worker_id: str = Field(nullable=False)
    """The worker id that owned this execution attempt. The worker id is set in the worker configuration.
    """

    claimed_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """When the worker first claimed this execution attempt. This corresponds to the beginning of the initialization phase"""

    started_at: datetime | None = Field(default=None, nullable=True)
    """When the worker actually began the core optimization, i.e. when the initialization phase finished successfully and
    the first optimization epoch started."""

    finished_at: datetime | None = Field(default=None, nullable=True)
    """When the stage execution finished, successfully or not."""

    error_message: str | None = Field(default=None, nullable=True)
    """The terminal error message for failed executions, if available. This will also be logged through the obs stack but
    for easier debugging it can be stored in the table as well.

    Persisting the message here avoids having to correlate database state with
    transient logs when investigating repeated failures or retry exhaustion.
    """

    loadflows_computed: int = 0
    """The number of loadflows computed in this stage for the KPI collection. Every N-1 case counts as a loadflow"""

    topologies_checked: int = 0
    """The number of topologies that were checked in this stage for the KPI collection."""

    epoch: int = 0
    """The current epoch of the optimization"""

    iteration: int = 0
    """The current iteration if an optimization consists of multiple iterations per epoch. Otherwise this shall be equal
    to the number of epochs"""

class ActiveWorker(SQLModel, table=True):
    """A table for workers to self-report liveliness
    
    In the kafka based architecture, workers would send heartbeats every few seconds to signal to the api service that
    they are alive. This table serves the same purpose, where each worker is tasked with updating its own row periodically"""

    id: str = Field(primary_key=True)
    """The worker id as stored in the startup configuration. This is the same as worker_id in the StageExecutionHistory and
    a join will reveal all executions this worker performed."""

    started_at: datetime = Field(default_factory=datetime.now, nullable=False)
    """The time the worker first started up"""

    last_heartbeat: datetime = Field(default_factory=datetime.now, nullable=False, sa_column_kwargs={"onupdate": datetime.now})
    """The last time the worker reported liveliness."""

    n_gpus: int = 0
    """The number of GPUs visible to this worker"""

    n_cpus: int = 0
    """The number of CPUs visible to this worker"""


