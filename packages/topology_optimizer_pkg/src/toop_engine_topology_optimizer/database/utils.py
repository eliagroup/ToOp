# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Utility functions for the optimizer command database.

These helpers expect a ``Session`` configured with automatic transaction
opening disabled. The utility layer owns transaction boundaries explicitly so
locking and writes happen inside predictable database transactions.
"""

from datetime import datetime, timedelta
from sqlalchemy import and_, or_
from sqlmodel import Session, select
from toop_engine_topology_optimizer.database.command_models import (
	StageExecutionHistory,
	StageWorkItem,
	StageWorkItemStatus,
)
from toop_engine_topology_optimizer.database.command_models import OptimizationJob
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType


def _begin_utility_transaction(session: Session):
	"""Start a utility-managed transaction.

	Database utility functions are expected to receive a session with
	``autobegin=False`` and no currently active transaction. This keeps
	transaction ownership inside the utility layer so row locking semantics stay
	predictable.

	Parameters
	----------
	session : Session
		The SQLModel session on which the utility wants to open a transaction.

	Returns
	-------
	SessionTransaction
		A context manager representing the opened transaction.

	Raises
	------
	RuntimeError
		If the session already has an active transaction.
	"""
	if session.in_transaction():
		raise RuntimeError("Database utility functions expect a session without an active transaction.")
	return session.begin()


def _select_next_stage_work_item(
	session: Session,
	stage: OptimizerType,
	claim_time: datetime,
) -> StageWorkItem | None:
	"""Return the next eligible work item candidate for a stage.

	Parameters
	----------
	session : Session
		The database session used to issue the locking select.
	stage : OptimizerType
		The stage for which to search a claimable work item.
	claim_time : datetime
		The reference time used to determine whether a running lease has expired.

	Returns
	-------
	StageWorkItem | None
		The highest-priority eligible work item for the given stage, or ``None``
		if no claimable row exists.
	"""
	statement = (
		select(StageWorkItem)
		.join(OptimizationJob)
		.where(
			StageWorkItem.stage == stage,
			or_(
				StageWorkItem.status == StageWorkItemStatus.TRIGGERED,
				and_(
					StageWorkItem.status == StageWorkItemStatus.RUNNING,
					StageWorkItem.lease_expires_at.is_not(None),
					StageWorkItem.lease_expires_at < claim_time,
				),
			),
		)
		.order_by(OptimizationJob.priority.desc(), OptimizationJob.created_at.asc())
		.limit(1)
		.with_for_update(skip_locked=True)
	)

	return session.exec(statement).first()


def _retire_exhausted_stage_work_item(session: Session, stage_work_item: StageWorkItem) -> None:
	"""Mark an expired work item as permanently failed after retry exhaustion.

	Parameters
	----------
	session : Session
		The database session used to persist the status update.
	stage_work_item : StageWorkItem
		The work item that exceeded the retry budget and should be retired.
	"""
	stage_work_item.status = StageWorkItemStatus.FAILED
	stage_work_item.lease_expires_at = None
	session.add(stage_work_item)
	session.flush()


def _claim_stage_work_item(
	session: Session,
	stage_work_item: StageWorkItem,
	worker_id: str,
	claim_time: datetime,
	lease_duration: timedelta,
) -> StageWorkItem:
	"""Claim a selected work item and create its execution history row.

	Parameters
	----------
	session : Session
		The database session used to persist the claim and history entry.
	stage_work_item : StageWorkItem
		The work item selected for claiming.
	worker_id : str
		The identifier of the worker claiming the row.
	claim_time : datetime
		The timestamp used for the claim record and lease calculation.
	lease_duration : timedelta
		The duration of the new lease.

	Returns
	-------
	StageWorkItem
		The updated work item after its claim fields were written.
	"""
	stage_work_item.current_attempt += 1
	stage_work_item.status = StageWorkItemStatus.RUNNING
	stage_work_item.lease_expires_at = claim_time + lease_duration

	session.add(
		StageExecutionHistory(
			stage_work_item_id=stage_work_item.id,
			optimization_job_id=stage_work_item.optimization_job_id,
			attempt=stage_work_item.current_attempt,
			status=StageWorkItemStatus.RUNNING,
			worker_id=worker_id,
			claimed_at=claim_time,
		)
	)

	session.flush()
	return stage_work_item


def update_stage_work_item(
	session: Session,
	stage_work_item_id: int,
	worker_id: str,
	status: StageWorkItemStatus = StageWorkItemStatus.RUNNING,
	lease_duration: timedelta | None = None,
	now: datetime | None = None,
	started_at: datetime | None = None,
	error_message: str | None = None,
	loadflows_computed: int | None = None,
	topologies_checked: int | None = None,
	epoch: int | None = None,
	iteration: int | None = None,
) -> StageWorkItem:
	"""Update a claimed work item and its current execution history row.

	This is the worker-side state update entry point used during periodic lease
	refreshes and terminal completion. The function updates both the hot-path
	``StageWorkItem`` row and the matching ``StageExecutionHistory`` record for
	the current attempt.

	If the stage was cancelled by the user before the update arrives, the update
	will not revive it. Instead, the current history row is finalized as
	``CANCELLED`` and the work item remains cancelled so the worker can stop
	cooperatively.

	Parameters
	----------
	session : Session
		The database session used for the update. The session is expected to be
		configured with ``autobegin=False`` because this utility manages its own
		transaction scope.
	stage_work_item_id : int
		The work item primary key to update.
	worker_id : str
		The worker that owns the current attempt. The matching history row must
		belong to this worker.
	status : StageWorkItemStatus, optional
		The new stage status, by default ``RUNNING``.
	lease_duration : timedelta | None, optional
		The new lease duration when the work item remains running. If omitted, the
		existing lease is left unchanged for running updates.
	now : datetime | None, optional
		The reference timestamp used for lease extension and terminal timestamps.
		If omitted, the current local time is used.
	started_at : datetime | None, optional
		The timestamp at which the core optimization started. When provided, it is
		stored on the current history row.
	error_message : str | None, optional
		A terminal error message to persist on the history row.
	loadflows_computed : int | None, optional
		The latest number of computed loadflows to store on the history row.
	topologies_checked : int | None, optional
		The latest number of checked topologies to store on the history row.
	epoch : int | None, optional
		The current optimization epoch to store on the history row.
	iteration : int | None, optional
		The current optimization iteration to store on the history row.

	Returns
	-------
	StageWorkItem
		The updated work item after the state change was persisted.

	Raises
	------
	RuntimeError
		If the work item does not exist, has no active history row for the current
		attempt, or the current attempt belongs to a different worker.
	"""
	update_time = now if now is not None else datetime.now()
	terminal_statuses = {
		StageWorkItemStatus.COMPLETED,
		StageWorkItemStatus.FAILED,
		StageWorkItemStatus.BLOCKED,
		StageWorkItemStatus.CANCELLED,
	}

	with _begin_utility_transaction(session):
		stage_work_item = session.exec(
			select(StageWorkItem).where(StageWorkItem.id == stage_work_item_id).with_for_update()
		).first()
		if stage_work_item is None:
			raise RuntimeError(f"Unknown stage work item {stage_work_item_id}.")

		execution_history = session.exec(
			select(StageExecutionHistory)
			.where(
				StageExecutionHistory.stage_work_item_id == stage_work_item_id,
				StageExecutionHistory.attempt == stage_work_item.current_attempt,
			)
			.with_for_update()
		).first()
		if execution_history is None:
			raise RuntimeError(
				f"Missing execution history for work item {stage_work_item_id} attempt {stage_work_item.current_attempt}."
			)
		if execution_history.worker_id != worker_id:
			raise RuntimeError(
				f"Worker {worker_id} does not own work item {stage_work_item_id} attempt {stage_work_item.current_attempt}."
			)

		if started_at is not None:
			execution_history.started_at = started_at
		if error_message is not None:
			execution_history.error_message = error_message
		if loadflows_computed is not None:
			execution_history.loadflows_computed = loadflows_computed
		if topologies_checked is not None:
			execution_history.topologies_checked = topologies_checked
		if epoch is not None:
			execution_history.epoch = epoch
		if iteration is not None:
			execution_history.iteration = iteration

		if stage_work_item.status == StageWorkItemStatus.CANCELLED:
			execution_history.status = StageWorkItemStatus.CANCELLED
			execution_history.finished_at = update_time
			stage_work_item.lease_expires_at = None
			session.add(execution_history)
			session.add(stage_work_item)
			return stage_work_item

		stage_work_item.status = status
		execution_history.status = status

		if status == StageWorkItemStatus.RUNNING:
			if lease_duration is not None:
				stage_work_item.lease_expires_at = update_time + lease_duration
		else:
			stage_work_item.lease_expires_at = None
			execution_history.finished_at = update_time

		session.add(execution_history)
		session.add(stage_work_item)
		return stage_work_item


def poll_stage_work_item(
	session: Session,
	stage: OptimizerType,
	worker_id: str,
	lease_duration: timedelta,
	max_retries: int | None = None,
	now: datetime | None = None,
) -> StageWorkItem | None:
	"""Claim the next eligible work item for a stage worker.

	A worker may claim either a freshly triggered work item or a running work
	item whose lease has expired. Claiming a work item moves it to ``RUNNING``,
	increments ``current_attempt``, writes the new lease deadline and creates the
	matching ``StageExecutionHistory`` row for the new attempt.

	Pending work is ordered by job priority descending and creation time ascending
	so workers claim the most urgent job first while preserving FIFO behaviour
	within equal-priority jobs.

	Parameters
	----------
	session : Session
		The database session used to lock, update and persist the claimed work
		item and its history row. The session is expected to be configured with
		``autobegin=False`` because this utility manages its own transaction
		scope.
	stage : OptimizerType
		The optimizer stage the worker is responsible for. Only work items for
		this stage are considered claimable.
	worker_id : str
		The identifier of the worker claiming the work item. This is stored in
		the created ``StageExecutionHistory`` row.
	lease_duration : timedelta
		The duration of the worker lease. The claimed work item's
		``lease_expires_at`` is set to ``now + lease_duration``.
	max_retries : int | None, optional
		The maximum number of attempts allowed for a work item. If an expired
		running work item has already reached this attempt count, it is marked as
		``FAILED`` instead of being claimed again. If ``None``, retry exhaustion
		is not enforced by this utility.
	now : datetime | None, optional
		The reference time used for lease expiry checks and new lease creation.
		If omitted, the current local time is used.

	Returns
	-------
	StageWorkItem | None
		The claimed and updated work item if one is available. Returns ``None``
		when no eligible work item exists for the requested stage.
	"""
	claim_time = now if now is not None else datetime.now()

	while True:
		claimed_work_item: StageWorkItem | None = None
		with _begin_utility_transaction(session):
			stage_work_item = _select_next_stage_work_item(session=session, stage=stage, claim_time=claim_time)
			if stage_work_item is None:
				return None

			if max_retries is not None and stage_work_item.current_attempt >= max_retries:
				_retire_exhausted_stage_work_item(session=session, stage_work_item=stage_work_item)
			else:
				claimed_work_item = _claim_stage_work_item(
					session=session,
					stage_work_item=stage_work_item,
					worker_id=worker_id,
					claim_time=claim_time,
					lease_duration=lease_duration,
				)

		if claimed_work_item is None:
			continue

		return claimed_work_item
