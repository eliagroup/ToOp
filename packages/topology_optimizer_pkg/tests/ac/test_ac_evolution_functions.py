# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import copy

import numpy as np
import pytest
from jaxtyping import Int
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.evolution_functions import (
    close_coupler,
    default_scorer,
    evolution,
    evolution_try,
    pull,
    reconnect,
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

    # Copy the same n-1 definition for all the topos of the strategy
    pulled = pull(strategy_copy, session=session, n_minus1_definitions=n_minus1_definitions_case_57 * len(strategy))

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


def test_reconnect(dc_repertoire: list[BaseDBTopology]) -> None:
    strategy = select_strategy(np.random.default_rng(0), dc_repertoire, dc_repertoire, default_scorer)
    reconnected = reconnect(np.random.default_rng(0), strategy)

    assert len(reconnected) == len(strategy)
    has_reconnected = False
    for new, old in zip(reconnected, strategy):
        assert isinstance(new, ACOptimTopology)
        assert new.actions == old.actions
        assert new.pst_setpoints == old.pst_setpoints
        assert new.strategy_hash == reconnected[0].strategy_hash
        assert new.strategy_hash != old.strategy_hash
        assert new.optimizer_type == OptimizerType.AC
        assert new.unsplit is False

        len_before = len(old.disconnections)
        len_after = len(new.disconnections)

        assert len_after == len_before or len_after == len_before - 1
        if len_after == len_before - 1:
            has_reconnected = True
    assert has_reconnected

    assert reconnect(np.random.default_rng(0), []) == []


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
    # Create a repertoire with one topology that has a parent on DC and one that doesn't have a parent
    repertoire = [
        ACOptimTopology(
            actions=[1],
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([([1], [], None)]),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.5,
            parent=ACOptimTopology(
                actions=[1],
                disconnections=[],
                pst_setpoints=None,
                unsplit=False,
                timestep=0,
                strategy_hash=hash_topo_data([([1], [], None)]),
                optimization_id="test",
                optimizer_type=OptimizerType.DC,
                fitness=0.5,
            ),
        ),
        ACOptimTopology(
            actions=[2],
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([([2], [], None)]),
            optimization_id="test",
            optimizer_type=OptimizerType.DC,
            fitness=0.5,
            parent=None,
        ),
        ACOptimTopology(
            actions=[2],
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([([2], [], None)]),
            optimization_id="nottest",
            optimizer_type=OptimizerType.DC,
            fitness=0.5,
            parent=None,
        ),
    ]
    session.add_all(repertoire)
    session.commit()
    assert len(session.exec(select(ACOptimTopology)).all()) == 4

    filtered = select_repertoire(
        optimization_id="test",
        optimizer_type=[OptimizerType.DC],
        without_parent_on=[OptimizerType.AC],
        session=session,
    )
    assert len(filtered) == 1
    assert filtered[0].strategy_hash == hash_topo_data([([2], [], None)])
    assert filtered[0].parent is None

    unfiltered = select_repertoire(
        optimization_id="test",
        optimizer_type=[OptimizerType.DC],
        without_parent_on=[],
        session=session,
    )
    assert len(unfiltered) == 2


def test_close_coupler(
    dc_repertoire: list[ACOptimTopology],
) -> None:
    strategy = select_strategy(np.random.default_rng(0), dc_repertoire, dc_repertoire, default_scorer)

    closed = close_coupler(
        rng=np.random.default_rng(0),
        selected_strategy=strategy,
    )

    assert len(closed) == len(strategy)

    has_closed = False
    for new, old in zip(closed, strategy):
        assert isinstance(new, ACOptimTopology)
        assert new.disconnections == old.disconnections
        assert new.pst_setpoints == old.pst_setpoints
        assert new.strategy_hash == closed[0].strategy_hash
        assert new.strategy_hash != old.strategy_hash
        assert new.optimizer_type == OptimizerType.AC
        assert new.unsplit is False

        assert len(old.actions) >= len(new.actions)

        # At least one coupler anywhere must have been closed
        if len(old.actions) > len(new.actions):
            has_closed = True
    assert has_closed

    assert close_coupler(np.random.default_rng(0), []) == []


def test_close_coupler_no_coupler(
    dc_repertoire: list[ACOptimTopology],
) -> None:
    strategy = select_strategy(np.random.default_rng(0), dc_repertoire, dc_repertoire, default_scorer)
    # If a strategy completely without splits is fed in, no coupler can be closed
    no_coupler_open = [
        ACOptimTopology(
            **topo.model_dump(exclude=["actions"]),
            actions=[],
        )
        for topo in strategy
    ]

    assert (
        close_coupler(
            rng=np.random.default_rng(0),
            selected_strategy=no_coupler_open,
        )
        == []
    )


def test_evolution_try_close_coupler(
    session: Session,
    dc_repertoire: list[ACOptimTopology],
    dc_repertoire_elements_per_sub: tuple[Int[np.ndarray, " n_relevant_subs"], Int[np.ndarray, " n_relevant_subs"]],
) -> None:
    branches_per_sub, injections_per_sub = dc_repertoire_elements_per_sub

    rng = np.random.default_rng(0)

    # For mocking deterministic random, we do a rng.choice to obtain the same random state
    # for select_strategy and close coupler
    # Thus, the results should match exactly
    rng.choice(["pull", "reconnect", "close_coupler"], p=[0, 0, 1])
    strategy = select_strategy(rng, dc_repertoire, dc_repertoire, default_scorer)
    # Repeat elements per sub to match strategy timestep dimension
    branches_per_sub = np.repeat(branches_per_sub[None], len(strategy), axis=0)
    injections_per_sub = np.repeat(injections_per_sub[None], len(strategy), axis=0)

    reference = close_coupler(
        rng,
        selected_strategy=strategy,
    )

    rng = np.random.default_rng(0)
    res = evolution_try(
        rng=rng,
        session=session,
        optimization_id="test",
        close_coupler_prob=1.0,
        pull_prob=0.0,
        reconnect_prob=0.0,
    )

    assert len(res) == len(reference)
    for r, ref in zip(res, reference):
        assert r.actions == ref.actions
        assert r.disconnections == ref.disconnections
        assert r.pst_setpoints == ref.pst_setpoints
        assert r.strategy_hash == ref.strategy_hash
        assert r.optimizer_type == ref.optimizer_type
        assert r.optimization_id == ref.optimization_id
        assert r.timestep == ref.timestep
        assert r.unsplit == ref.unsplit
        assert r.fitness == ref.fitness
        assert r.metrics == ref.metrics


def test_evolution_try_reconnect(session: Session, dc_repertoire: list[ACOptimTopology]) -> None:
    rng = np.random.default_rng(0)

    # For mocking deterministic random, we do a rng.choice to obtain the same random state
    # for select_strategy and close coupler
    # Thus, the results should match exactly
    rng.choice(["pull", "reconnect", "close_coupler"], p=[0, 1, 0])
    strategy = select_strategy(rng, dc_repertoire, dc_repertoire, default_scorer)
    reference = reconnect(
        rng,
        selected_strategy=strategy,
    )

    rng = np.random.default_rng(0)
    res = evolution_try(
        rng=rng,
        session=session,
        optimization_id="test",
        close_coupler_prob=0.0,
        pull_prob=0.0,
        reconnect_prob=1.0,
    )

    assert len(res) == len(reference)
    for r, ref in zip(res, reference):
        assert r.actions == ref.actions
        assert r.disconnections == ref.disconnections
        assert r.pst_setpoints == ref.pst_setpoints
        assert r.strategy_hash == ref.strategy_hash
        assert r.optimizer_type == ref.optimizer_type
        assert r.optimization_id == ref.optimization_id
        assert r.timestep == ref.timestep
        assert r.unsplit == ref.unsplit
        assert r.fitness == ref.fitness
        assert r.metrics == ref.metrics


def test_evolution_try_pull(session: Session, dc_repertoire: list[ACOptimTopology]) -> None:
    rng = np.random.default_rng(0)

    # For mocking deterministic random, we do a rng.choice to obtain the same random state
    # for select_strategy and close coupler
    # Thus, the results should match exactly
    rng.choice(["pull", "reconnect", "close_coupler"], p=[1, 0, 0])
    strategy = select_strategy(rng, dc_repertoire, dc_repertoire, default_scorer)
    reference = pull(
        selected_strategy=strategy,
    )

    rng = np.random.default_rng(0)
    res = evolution_try(
        rng=rng,
        session=session,
        optimization_id="test",
        close_coupler_prob=0.0,
        pull_prob=1.0,
        reconnect_prob=0.0,
    )

    assert len(res) == len(reference)
    for r, ref in zip(res, reference):
        assert r.actions == ref.actions
        assert r.disconnections == ref.disconnections
        assert r.pst_setpoints == ref.pst_setpoints
        assert r.strategy_hash == ref.strategy_hash
        assert r.optimizer_type == ref.optimizer_type
        assert r.optimization_id == ref.optimization_id
        assert r.timestep == ref.timestep
        assert r.unsplit == ref.unsplit
        assert r.fitness == ref.fitness
        assert r.metrics == ref.metrics


def test_evolution_try_accidental_duplicate(session: Session) -> None:
    # There is an edge case in which a mutate function could yield a duplicate that is already
    # in the database. This should be handled gracefully.
    # We can fake this edge case by inserting two strategies, one with exactly one coupler open
    # and the unsplit strategy. When close_coupler is called, the unsplit strategy will be returned
    # and the duplicate should be handled gracefully.

    repo = [
        ACOptimTopology(
            actions=[2],
            disconnections=[],
            pst_setpoints=None,
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([([2], [], None)]),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.5,
            metrics={"overload_energy_n_1": 123.4},
        ),
        ACOptimTopology(
            actions=[],
            disconnections=[],
            pst_setpoints=None,
            unsplit=True,
            timestep=0,
            strategy_hash=hash_topo_data([([], [], None)]),
            optimization_id="test",
            optimizer_type=OptimizerType.AC,
            fitness=0.5,
            metrics={"overload_energy_n_1": 123.4},
        ),
    ]
    for topo in repo:
        session.add(topo)
        session.commit()
        session.refresh(topo)

    closed = close_coupler(np.random.default_rng(42), repo[0:1])

    assert len(closed) == 1
    closed = closed[0]
    assert not len(closed.actions)
    assert closed.strategy_hash == repo[1].strategy_hash
    assert closed.unsplit is True

    with pytest.raises(IntegrityError):
        session.add(closed)
        session.commit()
    session.rollback()

    res = evolution_try(
        rng=np.random.default_rng(0),
        session=session,
        optimization_id="test",
        close_coupler_prob=1.0,
        pull_prob=0.0,
        reconnect_prob=0.0,
    )

    assert res == []

    res = evolution(
        rng=np.random.default_rng(0),
        session=session,
        optimization_id="test",
        close_coupler_prob=1.0,
        pull_prob=0.0,
        reconnect_prob=0.0,
        max_retries=10,
    )

    assert res == []
