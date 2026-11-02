# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime, timedelta
from uuid import uuid4

from sqlmodel import Session, select
from toop_engine_topology_optimizer.database.command_models import (
    StageExecutionHistory,
    StageWorkItem,
    StageWorkItemStatus,
)
from toop_engine_topology_optimizer.database.user_utils import cancel_optimization, start_optimization
from toop_engine_topology_optimizer.database.utils import poll_stage_work_item, update_stage_work_item
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


def test_poll_stage_work_item_claims_highest_priority_work_and_creates_history(
    command_database_session: Session,
) -> None:
    low_priority_optimization_id = uuid4()
    high_priority_optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    claim_time = datetime(2026, 6, 2, 12, 0, 0)
    lease_duration = timedelta(minutes=5)

    start_optimization(
        session=command_database_session,
        optimization_id=low_priority_optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=1,
    )
    start_optimization(
        session=command_database_session,
        optimization_id=high_priority_optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=10,
    )

    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    with command_database_session.begin():
        execution_history = command_database_session.exec(select(StageExecutionHistory)).all()

    assert claimed_work_item is not None
    assert claimed_work_item.optimization_job_id == high_priority_optimization_id
    assert claimed_work_item.stage == OptimizerType.DC
    assert claimed_work_item.status == StageWorkItemStatus.RUNNING
    assert claimed_work_item.current_attempt == 1
    assert claimed_work_item.lease_expires_at == claim_time + lease_duration
    assert len(execution_history) == 1
    assert execution_history[0].stage_work_item_id == claimed_work_item.id
    assert execution_history[0].optimization_job_id == high_priority_optimization_id
    assert execution_history[0].attempt == 1
    assert execution_history[0].worker_id == "worker-1"
    assert execution_history[0].status == StageWorkItemStatus.RUNNING
    assert execution_history[0].claimed_at == claim_time

    # A second poll returns the other job
    reclaimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    assert reclaimed_work_item is not None
    assert reclaimed_work_item.optimization_job_id == low_priority_optimization_id

    # A third poll returns None as no more optimization jobs are pending
    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )
    assert claimed_work_item is None


def test_poll_stage_work_item_reclaims_expired_running_work_item(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    expired_claim_time = datetime(2026, 6, 2, 11, 55, 0)
    reclaim_time = datetime(2026, 6, 2, 12, 0, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )

    with command_database_session.begin():
        stage_work_item = command_database_session.exec(
            select(StageWorkItem).where(StageWorkItem.optimization_job_id == optimization_id, StageWorkItem.stage == OptimizerType.DC)
        ).one()
        stage_work_item.status = StageWorkItemStatus.RUNNING
        stage_work_item.current_attempt = 2
        stage_work_item.lease_expires_at = expired_claim_time
        command_database_session.add(
            StageExecutionHistory(
                stage_work_item_id=stage_work_item.id,
                optimization_job_id=optimization_id,
                attempt=2,
                status=StageWorkItemStatus.RUNNING,
                worker_id="worker-0",
                claimed_at=expired_claim_time,
            )
        )

    reclaimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-2",
        lease_duration=lease_duration,
        now=reclaim_time,
    )

    with command_database_session.begin():
        execution_history = command_database_session.exec(
            select(StageExecutionHistory).where(StageExecutionHistory.stage_work_item_id == stage_work_item.id).order_by(StageExecutionHistory.attempt)
        ).all()

    assert reclaimed_work_item is not None
    assert reclaimed_work_item.id == stage_work_item.id
    assert reclaimed_work_item.status == StageWorkItemStatus.RUNNING
    assert reclaimed_work_item.current_attempt == 3
    assert reclaimed_work_item.lease_expires_at == reclaim_time + lease_duration
    assert len(execution_history) == 2
    assert execution_history[-1].attempt == 3
    assert execution_history[-1].worker_id == "worker-2"
    assert execution_history[-1].status == StageWorkItemStatus.RUNNING
    assert execution_history[-1].claimed_at == reclaim_time


def test_poll_stage_work_item_failes_after_many_attempts(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    expired_claim_time = datetime(2026, 6, 2, 11, 55, 0)
    reclaim_time = datetime(2026, 6, 2, 12, 0, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )

    with command_database_session.begin():
        stage_work_item = command_database_session.exec(
            select(StageWorkItem).where(StageWorkItem.optimization_job_id == optimization_id, StageWorkItem.stage == OptimizerType.DC)
        ).one()
        stage_work_item.status = StageWorkItemStatus.RUNNING
        stage_work_item.current_attempt = 3
        stage_work_item.lease_expires_at = expired_claim_time
        command_database_session.add(
            StageExecutionHistory(
                stage_work_item_id=stage_work_item.id,
                optimization_job_id=optimization_id,
                attempt=3,
                status=StageWorkItemStatus.RUNNING,
                worker_id="worker-0",
                claimed_at=expired_claim_time,
            )
        )

    reclaimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-2",
        lease_duration=lease_duration,
        now=reclaim_time,
        max_retries=3
    )

    assert reclaimed_work_item is None
    with command_database_session.begin():
        command_database_session.refresh(stage_work_item)
    assert stage_work_item.status == StageWorkItemStatus.FAILED

def test_poll_stage_work_item_respects_cancellation(command_database_session: Session) -> None:
    low_priority_optimization_id = uuid4()
    high_priority_optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    claim_time = datetime(2026, 6, 2, 12, 0, 0)
    lease_duration = timedelta(minutes=5)

    start_optimization(
        session=command_database_session,
        optimization_id=low_priority_optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=1,
    )
    start_optimization(
        session=command_database_session,
        optimization_id=high_priority_optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=10,
    )
    cancelled_work_items = cancel_optimization(session=command_database_session, optimization_id=high_prio_job.id)

    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    assert len(cancelled_work_items) == len(OptimizerType)
    assert all(stage_work_item.status == StageWorkItemStatus.CANCELLED for stage_work_item in cancelled_work_items)
    assert claimed_work_item is not None
    assert claimed_work_item.optimization_job_id == low_priority_optimization_id


def test_poll_stage_work_item_respects_blocked_state(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    expired_claim_time = datetime(2026, 6, 2, 11, 55, 0)
    reclaim_time = datetime(2026, 6, 2, 12, 0, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )

    with command_database_session.begin():
        stage_work_item = command_database_session.exec(
            select(StageWorkItem).where(StageWorkItem.optimization_job_id == optimization_id, StageWorkItem.stage == OptimizerType.DC)
        ).one()
        stage_work_item.status = StageWorkItemStatus.BLOCKED
        stage_work_item.current_attempt = 1
        stage_work_item.lease_expires_at = expired_claim_time
        command_database_session.add(
            StageExecutionHistory(
                stage_work_item_id=stage_work_item.id,
                optimization_job_id=optimization_id,
                attempt=1,
                status=StageWorkItemStatus.BLOCKED,
                worker_id="worker-0",
                claimed_at=expired_claim_time,
            )
        )

    reclaimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-2",
        lease_duration=lease_duration,
        now=reclaim_time,
    )

    assert reclaimed_work_item is None


def test_update_stage_work_item_refreshes_running_lease_and_history(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    claim_time = datetime(2026, 6, 2, 12, 0, 0)
    update_time = datetime(2026, 6, 2, 12, 5, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )
    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    assert claimed_work_item is not None

    updated_work_item = update_stage_work_item(
        session=command_database_session,
        stage_work_item_id=claimed_work_item.id,
        worker_id="worker-1",
        status=StageWorkItemStatus.RUNNING,
        lease_duration=lease_duration,
        now=update_time,
        started_at=claim_time,
        loadflows_computed=12,
        topologies_checked=4,
        epoch=3,
        iteration=8,
    )

    with command_database_session.begin():
        execution_history = command_database_session.exec(
            select(StageExecutionHistory).where(StageExecutionHistory.stage_work_item_id == claimed_work_item.id)
        ).one()

    assert updated_work_item.status == StageWorkItemStatus.RUNNING
    assert updated_work_item.lease_expires_at == update_time + lease_duration
    assert execution_history.status == StageWorkItemStatus.RUNNING
    assert execution_history.started_at == claim_time
    assert execution_history.loadflows_computed == 12
    assert execution_history.topologies_checked == 4
    assert execution_history.epoch == 3
    assert execution_history.iteration == 8
    assert execution_history.finished_at is None


def test_update_stage_work_item_completes_current_attempt(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    claim_time = datetime(2026, 6, 2, 12, 0, 0)
    completion_time = datetime(2026, 6, 2, 12, 10, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )
    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    assert claimed_work_item is not None

    completed_work_item = update_stage_work_item(
        session=command_database_session,
        stage_work_item_id=claimed_work_item.id,
        worker_id="worker-1",
        status=StageWorkItemStatus.COMPLETED,
        now=completion_time,
    )

    with command_database_session.begin():
        execution_history = command_database_session.exec(
            select(StageExecutionHistory).where(StageExecutionHistory.stage_work_item_id == claimed_work_item.id)
        ).one()

    assert completed_work_item.status == StageWorkItemStatus.COMPLETED
    assert completed_work_item.lease_expires_at is None
    assert execution_history.status == StageWorkItemStatus.COMPLETED
    assert execution_history.finished_at == completion_time


def test_update_stage_work_item_respects_cancellation(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]
    claim_time = datetime(2026, 6, 2, 12, 0, 0)
    cancel_time = datetime(2026, 6, 2, 12, 3, 0)
    update_time = datetime(2026, 6, 2, 12, 4, 0)
    lease_duration = timedelta(minutes=10)

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )
    claimed_work_item = poll_stage_work_item(
        session=command_database_session,
        stage=OptimizerType.DC,
        worker_id="worker-1",
        lease_duration=lease_duration,
        now=claim_time,
    )

    assert claimed_work_item is not None

    with command_database_session.begin():
        stage_work_item = command_database_session.exec(select(StageWorkItem).where(StageWorkItem.id == claimed_work_item.id)).one()
        stage_work_item.status = StageWorkItemStatus.CANCELLED
        stage_work_item.lease_expires_at = cancel_time
        command_database_session.add(stage_work_item)

    cancelled_work_item = update_stage_work_item(
        session=command_database_session,
        stage_work_item_id=claimed_work_item.id,
        worker_id="worker-1",
        status=StageWorkItemStatus.RUNNING,
        lease_duration=lease_duration,
        now=update_time,
    )

    with command_database_session.begin():
        execution_history = command_database_session.exec(
            select(StageExecutionHistory).where(StageExecutionHistory.stage_work_item_id == claimed_work_item.id)
        ).one()

    assert cancelled_work_item.status == StageWorkItemStatus.CANCELLED
    assert cancelled_work_item.lease_expires_at is None
    assert execution_history.status == StageWorkItemStatus.CANCELLED
    assert execution_history.finished_at == update_time