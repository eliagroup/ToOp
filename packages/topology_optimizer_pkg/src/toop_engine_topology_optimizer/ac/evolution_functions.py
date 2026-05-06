# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Implements an adjusted AC evolution

Instead of the original GA evolution which only knows mutate and crossover, we introduce the
following operations:
-  The pull operator will take a promising topology from the DC repertoire and re-evaluate
it on AC. The notion of promising is defined through a interest-scoring function which tries to
balance the explore/exploit trade-off.
"""

import pandas as pd
import structlog
from beartype.typing import Collection, Optional
from numpy.random import Generator as Rng
from sqlalchemy import exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased
from sqlmodel import Session, select
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.select_strategy import select_strategy
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.ac.types import ACStrategy
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy, OptimizerType

logger = structlog.get_logger(__name__)


def select_repertoire(
    optimization_id: str, optimizer_type: list[OptimizerType], without_parent_on: list[OptimizerType], session: Session
) -> list[ACOptimTopology]:
    """Select the topologies that are suitable for mutation and crossover

    In this case, all topologies that satisfy the filter criteria are suitable, however a check after
    the mutate is necessary to ensure that the topology is not already in the database.

    The unsplit strategy is never selected for mutation or crossover.

    Parameters
    ----------
    optimization_id : str
        The optimization ID to filter for
    optimizer_type : list[OptimizerType]
        The optimizer types to filter for (whitelist)
    without_parent_on : list[OptimizerType]
        If the topology has a parent on any of these types, the topology will be filtered out (blacklist). A parent means the
        topology has already been evaluated on that optimizer type.

    session : Session
        The database session to use

    Returns
    -------
    list[ACOptimTopology]
        The topologies that are suitable for mutation and crossover
    """
    # Query to select topologies suitable for mutation and crossover
    mutate_query = select(ACOptimTopology).where(
        ACOptimTopology.optimization_id == optimization_id,
        ACOptimTopology.optimizer_type.in_(optimizer_type),
        ACOptimTopology.unsplit == False,  # noqa: E712
    )

    # Filter out topologies whose parent has the specified optimizer types
    # (i.e., topologies that already spawned children on those optimizer types)
    if without_parent_on:
        child = aliased(ACOptimTopology)
        mutate_query = mutate_query.where(
            ~exists(
                select(1)
                .select_from(child)
                .where(
                    child.parent_id == ACOptimTopology.id,
                    child.optimizer_type.in_(without_parent_on),
                    child.optimization_id == optimization_id,
                )
            )
        )

    # Execute the query and get the results
    suitable_topologies = session.exec(mutate_query).all()

    return suitable_topologies


def get_unsplit_ac_topology(
    optimization_id: str,
    session: Session,
) -> Optional[ACOptimTopology]:
    """Get the unsplit AC topology for the given optimization ID

    Parameters
    ----------
    optimization_id : str
        The optimization ID to filter for
    session : Session
        The database session to use

    Returns
    -------
    ACOptimTopology
        The unsplit AC topology for the given optimization ID, or None if not found
    """
    query = select(ACOptimTopology).where(
        ACOptimTopology.optimization_id == optimization_id,
        ACOptimTopology.unsplit == True,  # noqa: E712
        ACOptimTopology.optimizer_type == OptimizerType.AC,
    )
    return session.exec(query).first()


def default_scorer(metrics: pd.DataFrame) -> pd.Series:
    """Score topologies so that lower-fitness candidates receive higher sampling weight.

    Parameters
    ----------
    metrics : pd.DataFrame
        The metrics DataFrame to score

    Returns
    -------
    pd.Series
        The fitness scores
    """
    return metrics["fitness"].max() - metrics["fitness"]


def get_contingency_indices_from_ids(case_ids: Collection[str], n_minus1_definition: Nminus1Definition) -> list[int]:
    """Map contingency ids to their indices in the N-1 definition.

    This is a helper method used in update_initial_metrics_with_worst_k_contingencies
    method.

    Parameters
    ----------
    case_ids : Sequence[str]
        A list of contingency ids for a specific topology.
    n_minus1_definition : Nminus1Definition
        The N-1 definition containing the contingencies.

    Returns
    -------
    list[int]
        A list of indices of the contingencies in the N-1 definition. If a contingency id is not found, it
        will be skipped.
    """
    id_to_index = {cont.id: idx for idx, cont in enumerate(n_minus1_definition.contingencies)}
    case_indices = [id_to_index[case_id] for case_id in case_ids if case_id in id_to_index]
    return case_indices


INF_FITNESS = 9999999.0


def pull(
    selected_strategy: ACStrategy,
    session: Session = None,
) -> ACStrategy:
    """Pull a promising topology from the DC repertoire to AC

    This only copies the topology without any changes other than setting the optimizer type to AC.
    This function takes a list of selected DC topologies and creates corresponding AC topologies by copying
    relevant attributes, setting the optimizer type to AC, and merging contingency case indices from both
    the DC and unsplit AC topologies.

    Parameters
    ----------
    selected_strategy : ACStrategy
        The selected strategy to pull
    session : Session, optional
        The database session to use, by default None. The session object is used to fetch the unsplit AC topology
        which is then used to add the critical contingency cases to the pulled strategy. These critical cases
        can then be used for early stopping of AC N-1 contingency analysis.

    Returns
    -------
    ACStrategy
        The a copy of the input topologies with the optimizer type set to AC
    """
    if not selected_strategy:
        return []

    optimization_id = selected_strategy[0].optimization_id

    # Unsplit AC topology will be used to merge critical contingency cases. This topology is pushed in the repertoire
    # in the initialization phase of the AC optimizer.
    unsplit_ac_topo = get_unsplit_ac_topology(optimization_id=optimization_id, session=session) if session else None
    worst_k_cont_ids_unsplit = set(unsplit_ac_topo.worst_k_contingency_cases) if unsplit_ac_topo else set()
    pulled_strategy = []
    for topo in selected_strategy:
        # Merge case_ids from DC and unsplit AC
        merged_ids = sorted(set(topo.worst_k_contingency_cases) | worst_k_cont_ids_unsplit)

        data = topo.model_dump(
            include=[
                "actions",
                "disconnections",
                "pst_setpoints",
                "unsplit",
                "timestep",
                "optimization_id",
                "strategy_hash",
            ]
        )
        metrics = topo.metrics
        metrics["fitness_dc"] = topo.fitness

        new_topo = ACOptimTopology(
            **data,
            optimizer_type=OptimizerType.AC,
            fitness=-INF_FITNESS,
            parent_id=topo.id,
            metrics=metrics,
            worst_k_contingency_cases=merged_ids,
        )
        new_topo.metrics["top_k_overloads_n_1"] = (
            unsplit_ac_topo.metrics.get("top_k_overloads_n_1", None) if unsplit_ac_topo else None
        )
        pulled_strategy.append(new_topo)

    return pulled_strategy


def evolution(
    rng: Rng,
    session: Session,
    optimization_id: str,
    max_retries: int,
    batch_size: int,
    filter_strategy: Optional[FilterStrategy] = None,
) -> list[ACOptimTopology]:
    """Perform the AC evolution.

    Parameters
    ----------
    rng : Rng
        The random number generator to use
    session : Session
        The database session to use, will write the new topologies to the database
    optimization_id : str
        The optimization ID to filter for
    max_retries : int
        The maximum number of retries to perform if a strategy is already in the database
    batch_size : int
        Number of unevaluated topologies to sample and convert to AC in one try.
    filter_strategy : Optional[FilterStrategy]
        The filter strategy to use for the optimization,
        used to filter out strategies that are too far away from the original topology.

    Returns
    -------
    ACStrategy
        The strategy that was created during the evolution or an empty list if something went
        wrong at all retries
    """
    for _try in range(max_retries):
        new_topo_batch = evolution_try(
            rng=rng,
            session=session,
            optimization_id=optimization_id,
            batch_size=batch_size,
            filter_strategy=filter_strategy,
        )
        if len(new_topo_batch):
            return new_topo_batch
    return []


def evolution_try(
    rng: Rng,
    session: Session,
    optimization_id: str,
    batch_size: int,
    filter_strategy: Optional[FilterStrategy] = None,
) -> list[ACOptimTopology]:
    """Perform a single try of the AC evolution.

    Parameters
    ----------
    rng : Rng
        The random number generator to use
    session : Session
        The database session to use, will write the new topologies to the database
    optimization_id : str
        The optimization ID to filter for
    batch_size : int
        Number of unevaluated topologies to sample and convert to AC.
    filter_strategy : Optional[FilterStrategy]
        The filter strategy to use for the optimization,
        used to filter out strategies that are too far away from the original topology.

    Returns
    -------
    list[ACOptimTopology]
        The list of topologies that were created during the evolution or an empty list if something went
        wrong during the try
    """
    selected_topologies = select_strategy(
        rng=rng,
        repertoire=select_repertoire(
            optimization_id=optimization_id,
            optimizer_type=[OptimizerType.DC, OptimizerType.AC],
            without_parent_on=[],
            session=session,
        ),
        candidates=select_repertoire(
            optimization_id=optimization_id,
            optimizer_type=[OptimizerType.DC],
            without_parent_on=[OptimizerType.AC],
            session=session,
        ),
        interest_scorer=default_scorer,
        batch_size=batch_size,
        filter_strategy=filter_strategy,
    )
    new_strategy = pull(selected_strategy=selected_topologies, session=session)
    # Something went wrong during the mutation
    if new_strategy == []:
        return []

    # Insert the new strategies into the database
    for topo in new_strategy:
        session.add(topo)

    try:
        session.commit()
        for topo in new_strategy:
            session.refresh(topo)
    except IntegrityError:
        session.rollback()
        return []
    return new_strategy
