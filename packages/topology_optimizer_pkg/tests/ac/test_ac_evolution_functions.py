# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import copy

import numpy as np
from jaxtyping import Int
from sqlmodel import Session, select
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.evolution_functions import (
    default_scorer,
    evolution_try,
    pull,
    select_repertoire,
)
from toop_engine_topology_optimizer.ac.select_strategy import select_strategy
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import BaseDBTopology, hash_topo_data


def test_pull(dc_repertoire: list[BaseDBTopology]) -> None:
    strategy = select_strategy(np.random.default_rng(0), dc_repertoire, dc_repertoire, default_scorer)
    pulled = pull(strategy)
    for new, old in zip(pulled, strategy):
        assert isinstance(new, ACOptimTopology)
        assert new.actions == old.actions
        assert new.disconnections == old.disconnections
        assert new.pst_setpoints == old.pst_setpoints
        assert new.strategy_hash == pulled[0].strategy_hash
        assert new.strategy_hash == old.strategy_hash
        assert new.optimizer_type == OptimizerType.AC

    assert pull([]) == []


def test_pull_with_worst_k_contingencies(
    unsplit_ac_dc_repertoire: tuple[list[ACOptimTopology], Session],
    n_minus1_definitions_case_57: list[Nminus1Definition],
):
    repo, session = unsplit_ac_dc_repertoire

    strategy = select_strategy(np.random.default_rng(0), repo, repo, default_scorer)
    # Create a copy of the strategy to avoid mutating the original during pull
    strategy_copy = [copy.deepcopy(t) for t in strategy]

    del n_minus1_definitions_case_57
    pulled = pull(strategy_copy, session=session)

    # Find the unsplit AC topology in the pulled repertoire
    unsplit_topos = [t for t in repo if getattr(t, "unsplit", False)]
    assert len(unsplit_topos) == 1
    unsplit_topo = unsplit_topos[0]
    assert unsplit_topo.unsplit is True

    # assert that the top_k_overloads_n_1 of the pulled topologies is the same as the unsplit topology
    for new, old in zip(pulled, strategy):
        assert isinstance(new, ACOptimTopology)
        assert new.actions == old.actions
        assert new.disconnections == old.disconnections
        assert new.pst_setpoints == old.pst_setpoints
        assert new.strategy_hash == old.strategy_hash
        assert new.optimizer_type == OptimizerType.AC

        # Check that the worst k contingencies are included in the metrics
        assert "top_k_overloads_n_1" in new.metrics
        assert unsplit_topo.metrics["top_k_overloads_n_1"] == new.metrics["top_k_overloads_n_1"]

        # Assert that the case_ids of the new topology is the union of the case_ids of the unsplit AC topology and the old topology
        unsplit_case_ids = set(unsplit_topo.worst_k_contingency_cases)

        # These are extracted from the n-1 definitions
        old_case_ids = set(old.worst_k_contingency_cases)
        new_case_ids = set(new.worst_k_contingency_cases)
        assert sorted(new_case_ids) == sorted(unsplit_case_ids.union(old_case_ids))


def test_select_repertoire(dc_repertoire: list[ACOptimTopology], session: Session) -> None:
    # Copy the first strategy to the AC database
    strategy = dc_repertoire[0].strategy_hash
    copy = pull([t for t in dc_repertoire if t.strategy_hash == strategy])
    for topo in copy:
        session.add(topo)
    session.commit()

    pulled = select_repertoire("test", [OptimizerType.DC], [], session)
    assert len(pulled) == len(dc_repertoire)
    assert set(p.id for p in pulled) == set(p.id for p in dc_repertoire)

    pulled = select_repertoire("test", [], [], session)
    assert len(pulled) == 0

    pulled = select_repertoire("nottest", [OptimizerType.DC], [], session)
    assert len(pulled) == 0


def test_select_repertoire_without_parent_on(session: Session) -> None:
    evaluated_dc_topology = ACOptimTopology(
        actions=[1],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=hash_topo_data([([1], [], None)]),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.5,
    )
    unevaluated_dc_topology = ACOptimTopology(
        actions=[2],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=hash_topo_data([([2], [], None)]),
        optimization_id="test",
        optimizer_type=OptimizerType.DC,
        fitness=0.5,
    )
    other_run_dc_topology = ACOptimTopology(
        actions=[3],
        disconnections=[],
        pst_setpoints=None,
        unsplit=False,
        timestep=0,
        strategy_hash=hash_topo_data([([3], [], None)]),
        optimization_id="nottest",
        optimizer_type=OptimizerType.DC,
        fitness=0.5,
    )
    session.add_all([evaluated_dc_topology, unevaluated_dc_topology, other_run_dc_topology])
    session.commit()
    session.refresh(evaluated_dc_topology)

    session.add(
        ACOptimTopology(
            actions=evaluated_dc_topology.actions,
            disconnections=evaluated_dc_topology.disconnections,
            pst_setpoints=evaluated_dc_topology.pst_setpoints,
            unsplit=False,
            timestep=0,
            strategy_hash=evaluated_dc_topology.strategy_hash,
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.5,
            parent_id=evaluated_dc_topology.id,
        )
    )
    session.commit()
    assert len(session.exec(select(ACOptimTopology)).all()) == 4

    filtered = select_repertoire(
        optimization_id="test",
        optimizer_type=[OptimizerType.DC],
        without_parent_on=[OptimizerType.AC],
        session=session,
    )
    assert len(filtered) == 1
    assert filtered[0].strategy_hash == unevaluated_dc_topology.strategy_hash
    assert filtered[0].parent_id is None

    unfiltered = select_repertoire(
        optimization_id="test",
        optimizer_type=[OptimizerType.DC],
        without_parent_on=[],
        session=session,
    )
    assert len(unfiltered) == 2


def test_evolution_try_returns_pulled_topology_batch(
    session: Session,
    dc_repertoire: list[ACOptimTopology],
    dc_repertoire_elements_per_sub: tuple[Int[np.ndarray, " n_relevant_subs"], Int[np.ndarray, " n_relevant_subs"]],
) -> None:
    del dc_repertoire_elements_per_sub
    res = evolution_try(
        rng=np.random.default_rng(0),
        session=session,
        optimization_id="test",
        batch_size=1,
    )

    assert len(res) == 1
    assert res[0].optimizer_type == OptimizerType.AC
    assert res[0].parent_id is not None
    parent = session.get(ACOptimTopology, res[0].parent_id)
    assert parent is not None
    assert res[0].actions == parent.actions
    assert res[0].disconnections == parent.disconnections
    assert res[0].pst_setpoints == parent.pst_setpoints
    assert res[0].strategy_hash == parent.strategy_hash
    assert res[0].optimization_id == parent.optimization_id
    assert res[0].timestep == parent.timestep


def test_evolution_try_pull(session: Session, dc_repertoire: list[ACOptimTopology]) -> None:
    res = evolution_try(
        rng=np.random.default_rng(0),
        session=session,
        optimization_id="test",
        batch_size=1,
    )

    assert len(res) == 1
    assert res[0].optimizer_type == OptimizerType.AC
    assert res[0].parent_id is not None
    parent = session.get(ACOptimTopology, res[0].parent_id)
    assert parent is not None
    assert res[0].actions == parent.actions
    assert res[0].disconnections == parent.disconnections
    assert res[0].pst_setpoints == parent.pst_setpoints
    assert res[0].strategy_hash == parent.strategy_hash


def test_evolution_try_returns_worst_k_sized_batch_of_unevaluated_topologies(session: Session) -> None:
    source_topologies: list[ACOptimTopology] = []
    for index, fitness in enumerate([-1.0, -2.0, -3.0]):
        topology = ACOptimTopology(
            actions=[index + 1],
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([([index + 1], [], None)]),
            optimization_id="evolution-batch",
            optimizer_type=OptimizerType.DC,
            fitness=fitness,
            metrics={"overload_energy_n_1": abs(fitness)},
            worst_k_contingency_cases=[f"c{index}"],
        )
        session.add(topology)
        source_topologies.append(topology)
    session.commit()
    for topology in source_topologies:
        session.refresh(topology)

    already_evaluated = ACOptimTopology(
        actions=source_topologies[0].actions,
        disconnections=source_topologies[0].disconnections,
        pst_setpoints=source_topologies[0].pst_setpoints,
        unsplit=False,
        timestep=0,
        strategy_hash=source_topologies[0].strategy_hash,
        optimization_id="evolution-batch",
        optimizer_type=OptimizerType.AC,
        fitness=0.0,
        parent_id=source_topologies[0].id,
        metrics={"fitness_dc": source_topologies[0].fitness},
        worst_k_contingency_cases=source_topologies[0].worst_k_contingency_cases,
    )
    session.add(already_evaluated)
    session.commit()

    result = evolution_try(
        rng=np.random.default_rng(0),
        session=session,
        optimization_id="evolution-batch",
        batch_size=2,
    )

    assert len(result) == 2
    assert all(topology.optimizer_type == OptimizerType.AC for topology in result)
    assert {topology.parent_id for topology in result} == {source_topologies[1].id, source_topologies[2].id}
