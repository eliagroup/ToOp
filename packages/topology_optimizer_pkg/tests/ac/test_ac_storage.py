# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine, select
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, create_session, scrub_db
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType


@pytest.fixture
def db_session() -> Session:
    session = create_session()
    try:
        yield session
    finally:
        session.close()


def _insert_topology(
    session: Session,
    optimizer_type: OptimizerType,
    created_at: datetime,
) -> None:
    topology = ACOptimTopology(
        actions=[],
        disconnections=[],
        pst_setpoints={},
        unsplit=True,
        timestep=0,
        strategy_hash=uuid4().bytes,
        optimization_id="opt-id",
        optimizer_type=optimizer_type,
        fitness=0.0,
        metrics={},
        worst_k_contingency_cases=[],
        created_at=created_at,
    )
    session.add(topology)
    session.commit()


def test_scrub_db_removes_topologies_older_than_cutoff(
    db_session: Session,
) -> None:
    optimizer_type = next(iter(OptimizerType))
    max_age_seconds = 3600
    current_time = datetime.now()
    old_created_at = current_time - timedelta(seconds=max_age_seconds + 1000.0)
    fresh_created_at = current_time - timedelta(seconds=max_age_seconds - 1000.0)

    _insert_topology(db_session, optimizer_type, old_created_at)
    _insert_topology(db_session, optimizer_type, fresh_created_at)

    scrub_db(db_session, max_age_seconds=max_age_seconds)

    remaining = db_session.exec(select(ACOptimTopology)).all()
    assert len(remaining) == 1
    assert remaining[0].created_at == fresh_created_at


def test_ac_topology():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine, tables=[ACOptimTopology.__table__])
    with Session(engine) as session:
        topo = ACOptimTopology(
            actions=[1, 2, 3],
            disconnections=[0, 1],
            pst_setpoints=[0, 0, 0, 0],
            unsplit=False,
            timestep=0,
            strategy_hash=bytes.fromhex("deadbeef"),
            optimization_id="test",
            optimizer_type=OptimizerType.DC,
            fitness=0.5,
            metrics={"overload_energy_n_1": 123.4},
        )
        session.add(topo)
        session.commit()

        topo = session.exec(select(ACOptimTopology)).one()
        assert topo.actions == [1, 2, 3]
        assert topo.disconnections == [0, 1]
        assert topo.pst_setpoints == [0, 0, 0, 0]
        assert topo.timestep == 0
        assert topo.optimization_id == "test"
        assert topo.optimizer_type == OptimizerType.DC
        assert topo.fitness == 0.5
        assert topo.metrics == {"overload_energy_n_1": 123.4}
