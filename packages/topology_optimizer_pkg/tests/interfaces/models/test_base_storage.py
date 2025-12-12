import pytest
import sqlalchemy
from sqlmodel import Session, select
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, convert_message_topo_to_db_topo
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, Strategy, Topology
from toop_engine_topology_optimizer.interfaces.models.base_storage import (
    BaseDBTopology,
    convert_db_topo_to_message_topo,
    hash_strategy,
    hash_topologies,
    metrics_dataframe,
)


def test_storage(session: Session) -> None:
    topo = ACOptimTopology(
        actions=[5, 4, 3],
        disconnections=[0, 1],
        pst_setpoints=[0, 0, 0],
        unsplit=False,
        timestep=0,
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        strategy_hash=bytes.fromhex("deadbeef"),
        fitness=0.5,
        metrics={"overload_energy_n_1": 123.4},
    )

    # Insert the topology into the database

    session.add(topo)
    session.commit()
    session.refresh(topo)

    # Query the database
    topo = session.exec(select(ACOptimTopology)).one()
    assert topo.actions == [5, 4, 3]
    assert topo.disconnections == [0, 1]
    assert topo.pst_setpoints == [0, 0, 0]
    assert topo.timestep == 0
    assert topo.optimization_id == "test"
    assert topo.optimizer_type == OptimizerType.DC
    assert topo.fitness == 0.5
    assert topo.metrics == {"overload_energy_n_1": 123.4}
    assert topo.strategy_hash_str == "deadbeef"
    assert topo.created_at is not None
    assert topo.id is not None
    assert topo.stored_loadflow_reference is None
    assert topo.unsplit is False

    # Topologies should have a unique constraint
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.add(
            ACOptimTopology(
                actions=[5, 4, 3],
                disconnections=[0, 1],
                pst_setpoints=[0, 0, 0],
                unsplit=False,
                timestep=0,
                optimization_id="test",
                optimizer_type=OptimizerType.DC,
                strategy_hash=bytes.fromhex("deadbeef"),
                fitness=0.8,
                metrics={"overload_energy_n_1": 125.6},
            )
        )
        session.commit()


def test_metrics_dataframe(dc_repertoire: list[ACOptimTopology]) -> None:
    # Convert the topologies to a DataFrame
    df = metrics_dataframe(dc_repertoire)

    assert len(df) == len(dc_repertoire)
    assert df["fitness"].dtype == float
    assert df["overload_energy_n_1"].dtype == float
    assert set(df.index) == set(topo.id for topo in dc_repertoire)


def test_hashing() -> None:
    # Create some example topologies
    strategy1 = Strategy(
        timesteps=[
            Topology(
                actions=[2, 3, 4],
                disconnections=[0, 1],
                pst_setpoints=[0, 0, 0],
                metrics=Metrics(
                    fitness=0.5,
                    extra_scores={"overload_energy_n_1": 123.4},
                ),
            ),
            Topology(
                actions=[2, 3, 4],
                disconnections=[1],
                pst_setpoints=[0, 0, 0],
                metrics=Metrics(
                    fitness=0.6,
                    extra_scores={"overload_energy_n_1": 124.4},
                ),
            ),
        ]
    )

    strategy2 = Strategy(
        timesteps=[
            Topology(
                actions=[
                    3,
                    4,
                ],
                disconnections=[0, 1],
                pst_setpoints=[0, 0, 0],
                metrics=Metrics(
                    fitness=0.5,
                    extra_scores={"overload_energy_n_1": 123.4},
                ),
            ),
            Topology(
                actions=[3, 4],
                disconnections=[1],
                pst_setpoints=[
                    0,
                    0,
                    0,
                ],
                metrics=Metrics(
                    fitness=0.6,
                    extra_scores={"overload_energy_n_1": 124.4},
                ),
            ),
        ]
    )

    # Hash the strategies
    hash1 = hash_strategy(strategy1)
    hash2 = hash_strategy(strategy2)
    assert hash1 != hash2

    # Convert them to db topologies and re-hash them
    db_topos = convert_message_topo_to_db_topo(
        message_strategies=[strategy1, strategy2],
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
    )

    assert len(db_topos) == 4

    db_strategy1 = [topo for topo in db_topos if topo.strategy_hash == hash1]
    db_strategy2 = [topo for topo in db_topos if topo.strategy_hash == hash2]

    assert len(db_strategy1) == 2
    assert len(db_strategy2) == 2

    assert hash_topologies(db_strategy1) == hash1
    assert hash_topologies(db_strategy2) == hash2


def test_convert_db_topo_to_message_topo() -> None:
    # Create some example BaseDBTopology objects
    db_topo1 = BaseDBTopology(
        id=1,
        actions=[2, 3, 4],
        disconnections=[0, 1],
        pst_setpoints=[0, 0, 0],
        unsplit=False,
        timestep=0,
        strategy_hash=bytes.fromhex("cafebabe"),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.5,
        metrics={"overload_energy_n_1": 123.4},
    ).set_loadflow_reference(
        StoredLoadflowReference(
            relative_path="this/does/not/exist",
        )
    )

    db_topo2 = BaseDBTopology(
        id=2,
        actions=[5, 4, 3],
        disconnections=[1, 0],
        pst_setpoints=[0, 0, 0],
        unsplit=False,
        timestep=1,
        strategy_hash=bytes.fromhex("cafebabe"),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.6,
        metrics={"overload_energy_n_1": 124.4},
    ).set_loadflow_reference(
        StoredLoadflowReference(
            relative_path="this/does/not/exist",
        )
    )

    db_topo3 = BaseDBTopology(
        id=3,
        actions=[],
        disconnections=[0, 1],
        pst_setpoints=[0, 0, 0],
        unsplit=False,
        timestep=0,
        strategy_hash=bytes.fromhex("deadbeef"),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.5,
        metrics={"overload_energy_n_1": 123.4},
    )

    db_topo4 = BaseDBTopology(
        id=4,
        actions=[3, 4],
        disconnections=[1, 0],
        pst_setpoints=[0, 0, 0],
        unsplit=False,
        timestep=1,
        strategy_hash=bytes.fromhex("deadbeef"),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.6,
        metrics={"overload_energy_n_1": 124.4},
    )

    # Convert the db topologies to message topologies
    strategies = convert_db_topo_to_message_topo([db_topo1, db_topo2, db_topo3, db_topo4])

    assert len(strategies) == 2

    strategy1 = strategies[0]
    assert len(strategy1.timesteps) == 2
    assert strategy1.timesteps[0].actions == [2, 3, 4]
    assert strategy1.timesteps[0].disconnections == [0, 1]
    assert strategy1.timesteps[0].pst_setpoints == [0, 0, 0]
    assert strategy1.timesteps[0].metrics.fitness == 0.5
    assert strategy1.timesteps[0].metrics.extra_scores == {"overload_energy_n_1": 123.4}
    assert strategy1.timesteps[0].loadflow_results.relative_path == "this/does/not/exist"

    assert strategy1.timesteps[1].actions == [3, 4, 5]
    assert strategy1.timesteps[1].disconnections == [0, 1]
    assert strategy1.timesteps[1].pst_setpoints == [0, 0, 0]
    assert strategy1.timesteps[1].metrics.fitness == 0.6
    assert strategy1.timesteps[1].metrics.extra_scores == {"overload_energy_n_1": 124.4}
    assert strategy1.timesteps[1].loadflow_results.relative_path == "this/does/not/exist"

    strategy2 = strategies[1]
    assert len(strategy2.timesteps) == 2
    assert strategy2.timesteps[0].actions == []
    assert strategy2.timesteps[0].disconnections == [0, 1]
    assert strategy2.timesteps[0].pst_setpoints == [0, 0, 0]
    assert strategy2.timesteps[0].metrics.fitness == 0.5
    assert strategy2.timesteps[0].metrics.extra_scores == {"overload_energy_n_1": 123.4}
    assert strategy2.timesteps[0].loadflow_results is None

    assert strategy2.timesteps[1].actions == [3, 4]
    assert strategy2.timesteps[1].disconnections == [0, 1]
    assert strategy2.timesteps[1].pst_setpoints == [0, 0, 0]
    assert strategy2.timesteps[1].metrics.fitness == 0.6
    assert strategy2.timesteps[1].metrics.extra_scores == {"overload_energy_n_1": 124.4}
    assert strategy2.timesteps[1].loadflow_results is None
