# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""User-facing utilities for the optimizer command database.

These helpers expect a ``Session`` configured with automatic transaction
opening disabled. The utility layer owns transaction boundaries explicitly so
locking and writes happen inside predictable database transactions.
"""

from uuid import UUID

from sqlmodel import Session, select
from toop_engine_topology_optimizer.database.command_models import OptimizationJob, StageWorkItem, StageWorkItemStatus
from toop_engine_topology_optimizer.database.utils import _begin_utility_transaction
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


def start_optimization(
	session: Session,
	optimization_id: UUID,
	grid_files: list[GridFile],
	dc_params: DCOptimizerParameters,
	ac_params: ACOptimizerParameters,
	priority: int = 0,
) -> OptimizationJob:
	"""Seed a new optimization job and one work item for every optimizer stage.

	Parameters
	----------
	session : Session
		The database session used for insertion. The session is expected to be
		configured with ``autobegin=False`` because this utility manages its own
		transaction scope.
	optimization_id : UUID
		The persistent optimization identifier shared between the job row and all
		created stage work items.
	grid_files : list[GridFile]
		The grid files to optimize.
	dc_params : DCOptimizerParameters
		The DC optimizer parameters for the job.
	ac_params : ACOptimizerParameters
		The AC optimizer parameters for the job.
	priority : int, optional
		The queue priority of the optimization job, by default ``0``.

	Returns
	-------
	OptimizationJob
		The inserted optimization job instance.
	"""
	optimization_job = OptimizationJob(
		id=optimization_id,
		grid_files=grid_files,
		dc_params=dc_params,
		ac_params=ac_params,
		priority=priority,
	)
	with _begin_utility_transaction(session):
		session.add(optimization_job)
		session.flush()

		for stage in OptimizerType:
			session.add(
				StageWorkItem(
					optimization_job_id=optimization_id,
					optimization_job=optimization_job,
					stage=stage,
					status=StageWorkItemStatus.TRIGGERED,
				)
			)

	return optimization_job


def cancel_optimization(session: Session, optimization_id: UUID) -> list[StageWorkItem]:
	"""Cancel all stage work items for an optimization job.

	Cancellation is modeled on the hot-path stage rows. This utility marks every
	stage work item of the given optimization job as ``CANCELLED`` so workers see
	the cancellation on their next polling or lease-refresh cycle.

	Parameters
	----------
	session : Session
		The database session used to lock and update the stage work items. The
		session is expected to be configured with ``autobegin=False`` because
		this utility manages its own transaction scope.
	optimization_id : UUID
		The optimization job whose stage work items should be cancelled.

	Returns
	-------
	list[StageWorkItem]
		The stage work items that were marked as ``CANCELLED``.
	"""
	with _begin_utility_transaction(session):
		stage_work_items = session.exec(
			select(StageWorkItem).where(StageWorkItem.optimization_job_id == optimization_id).with_for_update()
		).all()

		for stage_work_item in stage_work_items:
			stage_work_item.status = StageWorkItemStatus.CANCELLED
			session.add(stage_work_item)

		return list(stage_work_items)