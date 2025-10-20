from unittest.mock import Mock

import pytest
from sqlmodel import Session, select
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import (
    Metrics,
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    Strategy,
    TopologyPushResult,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Topology as MessageTopology


@pytest.fixture
def result() -> Result:
    return Result(
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        result=TopologyPushResult(
            strategies=[
                Strategy(
                    timesteps=[
                        MessageTopology(
                            actions=[1, 2, 555],
                            disconnections=[0, 1],
                            pst_setpoints=[1, 2, 3, 4],
                            metrics=Metrics(
                                fitness=0.5,
                                extra_scores={"overload_energy_n_1": 123.4},
                            ),
                        ),
                        MessageTopology(
                            actions=[1, 2, 3],
                            disconnections=[0, 1],
                            pst_setpoints=[1, 2, 3, 4],
                            metrics=Metrics(
                                fitness=0.5,
                                extra_scores={"overload_energy_n_1": 123.4},
                            ),
                        ),
                    ],
                )
            ]
        ),
    )


def test_poll_results_topic(result: Result, session: Session) -> None:
    # Add a test topology to possibly confuse the processor
    topo = ACOptimTopology(
        actions=[5, 6, 7],
        disconnections=[0, 1],
        pst_setpoints=[0, 0, 0, 0],
        unsplit=False,
        timestep=0,
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        strategy_hash=bytes.fromhex("deadbeef"),
        fitness=0.5,
        metrics={"overload_energy_n_1": 123.4},
    )
    session.add(topo)
    session.commit()

    consumer = Mock(spec=LongRunningKafkaConsumer)
    message = Mock()
    message.value.return_value = result.model_dump_json().encode()
    consumer.consume.return_value = [message]
    processed = poll_results_topic(db=session, consumer=consumer, first_poll=True)
    assert len(processed) == 2

    topologies = session.exec(select(ACOptimTopology)).all()
    assert len(topologies) == 3
    assert topologies[1].actions == [1, 2, 555]
    assert topologies[1].disconnections == [0, 1]
    assert topologies[1].pst_setpoints == [1, 2, 3, 4]
    assert topologies[1].fitness == 0.5
    assert topologies[1].metrics == {"overload_energy_n_1": 123.4}
    assert topologies[1].timestep == 0

    assert topologies[2].actions == [1, 2, 3]
    assert topologies[2].disconnections == [0, 1]
    assert topologies[2].pst_setpoints == [1, 2, 3, 4]
    assert topologies[2].fitness == 0.5
    assert topologies[2].metrics == {"overload_energy_n_1": 123.4}
    assert topologies[2].timestep == 1

    assert topologies[1].strategy_hash == topologies[2].strategy_hash

    # Running it a second time should not add any new topologies
    processed = poll_results_topic(db=session, consumer=consumer, first_poll=True)
    assert len(processed) == 0
    topologies = session.exec(select(ACOptimTopology)).all()
    assert len(topologies) == 3


def test_poll_results_topic_optimization_started_result(result: Result, session: Session) -> None:
    # Should handle OptimizationStartedResult messages
    result.result = OptimizationStartedResult(
        initial_topology=result.result.strategies[0], initial_stats=[StaticInformationStats()]
    )

    consumer = Mock(spec=LongRunningKafkaConsumer)
    message = Mock()
    message.value.return_value = result.model_dump_json().encode()
    consumer.consume.return_value = [message]

    processed = poll_results_topic(db=session, consumer=consumer, first_poll=True)
    assert len(processed) == 2

    topologies = session.exec(select(ACOptimTopology)).all()
    assert len(topologies) == 2
    assert topologies[0].actions == [1, 2, 555]
    assert topologies[0].disconnections == [0, 1]
    assert topologies[0].pst_setpoints == [1, 2, 3, 4]
    assert topologies[0].fitness == 0.5
    assert topologies[0].metrics == {"overload_energy_n_1": 123.4}
    assert topologies[0].timestep == 0

    assert topologies[1].actions == [1, 2, 3]
    assert topologies[1].disconnections == [0, 1]
    assert topologies[1].pst_setpoints == [1, 2, 3, 4]
    assert topologies[1].fitness == 0.5
    assert topologies[1].metrics == {"overload_energy_n_1": 123.4}
    assert topologies[1].timestep == 1


def test_poll_results_topic_invalid_result_type(result: Result, session: Session) -> None:
    # It should skip OptimizationStoppedResult messages
    result.result = OptimizationStoppedResult(reason="error")

    consumer = Mock(spec=LongRunningKafkaConsumer)
    message = Mock()
    message.value.return_value = result.model_dump_json().encode()
    consumer.consume.return_value = [message]

    processed = poll_results_topic(db=session, consumer=consumer, first_poll=True)
    assert len(processed) == 0
    topologies = session.exec(select(ACOptimTopology)).all()
    assert len(topologies) == 0
