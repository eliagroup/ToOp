# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from uuid import uuid4

from sqlmodel import Session, select
from toop_engine_topology_optimizer.database.command_models import OptimizationJob, StageWorkItem, StageWorkItemStatus
from toop_engine_topology_optimizer.database.user_utils import cancel_optimization, start_optimization
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


def test_start_optimization_inserts_job_and_one_work_item_per_stage(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )

    with command_database_session.begin():
        optimization_job = command_database_session.exec(select(OptimizationJob)).one()
        stage_work_items = command_database_session.exec(select(StageWorkItem)).all()

    assert optimization_job.id == optimization_id
    assert optimization_job.priority == 5
    assert optimization_job.grid_files == grid_files
    assert optimization_job.dc_params == DCOptimizerParameters()
    assert optimization_job.ac_params == ACOptimizerParameters()
    assert len(stage_work_items) == len(OptimizerType)
    assert {stage_work_item.stage for stage_work_item in stage_work_items} == set(OptimizerType)
    assert all(stage_work_item.optimization_job_id == optimization_id for stage_work_item in stage_work_items)
    assert all(stage_work_item.status == StageWorkItemStatus.TRIGGERED for stage_work_item in stage_work_items)
    assert all(stage_work_item.current_attempt == 0 for stage_work_item in stage_work_items)


def test_cancel_optimization_marks_all_stage_work_items_cancelled(command_database_session: Session) -> None:
    optimization_id = uuid4()
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]

    start_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
        grid_files=grid_files,
        dc_params=DCOptimizerParameters(),
        ac_params=ACOptimizerParameters(),
        priority=5,
    )

    cancelled_work_items = cancel_optimization(
        session=command_database_session,
        optimization_id=optimization_id,
    )

    with command_database_session.begin():
        persisted_work_items = command_database_session.exec(
            select(StageWorkItem).where(StageWorkItem.optimization_job_id == optimization_id)
        ).all()

    assert len(cancelled_work_items) == len(OptimizerType)
    assert {stage_work_item.id for stage_work_item in cancelled_work_items} == {
        stage_work_item.id for stage_work_item in persisted_work_items
    }
    assert all(stage_work_item.status == StageWorkItemStatus.CANCELLED for stage_work_item in cancelled_work_items)
    assert all(stage_work_item.status == StageWorkItemStatus.CANCELLED for stage_work_item in persisted_work_items)