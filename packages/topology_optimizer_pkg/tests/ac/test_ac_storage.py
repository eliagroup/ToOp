# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from sqlmodel import Session, SQLModel, create_engine, select
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType


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
