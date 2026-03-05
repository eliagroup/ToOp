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
- The reconnect operator will take a promising topology from the DC repertoire and reconnect a
single branch in all timesteps. The idea is that the DC part might have disconnected too many
branches, so we try to simplify the topology by reconnecting a single branch.
- The close_coupler operator will take a promising topology from the DC repertoire and close a
coupler in all timesteps. The idea is that the DC part might have too many open couplers, so we
try to simplify the topology by closing a single coupler.
"""

import logbook
import pandas as pd
from beartype.typing import Optional
from numpy.random import Generator as Rng
from sqlalchemy import exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased
from sqlmodel import Session, select
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_topology_optimizer.ac.select_strategy import select_strategy
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import (
    hash_topologies,
    is_unsplit_topologies,
)

logger = logbook.Logger(__name__)


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
    # (i.e., topologies that were created from parents of those types)
    if without_parent_on:
        parent = aliased(ACOptimTopology)
        mutate_query = mutate_query.where(
            ~exists(
                select(1)
                .select_from(parent)
                .where(
                    ACOptimTopology.parent_id == parent.id,
                    parent.optimizer_type.in_(without_parent_on),
                    parent.optimization_id == optimization_id,
                )
            )
        )

    # Execute the query and get the results
    suitable_topologies = session.exec(mutate_query).all()

    return suitable_topologies


def get_unsplit_ac_topology(
    optimization_id: str,
    session: Session,
) -> ACOptimTopology:
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
    """Score the topologies based on their fitness only (greedy selection)

    Parameters
    ----------
    metrics : pd.DataFrame
        The metrics DataFrame to score

    Returns
    -------
    pd.Series
        The fitness scores
    """
    return metrics["fitness"] + metrics["fitness"].min()


def get_contingency_indices_from_ids(
    case_ids_all_t: list[list[str]], n_minus1_definitions: list[Nminus1Definition]
) -> list[list[int]]:
    """Map contingency ids to their indices in the N-1 definition for each timestep.

    This is a helper method used in update_initial_metrics_with_worst_k_contingencies
    method.

    Parameters
    ----------
    case_ids_all_t : list[list[str]]
        A list of lists, where each inner list contains the contingency ids for a specific timestep.
    n_minus1_definitions : list[Nminus1Definition]
        A list of N-1 definitions, one for each timestep, containing the contingencies.

    Returns
    -------
    list[list[int]]
        A list of lists, where each inner list contains the indices of the contingencies in the
        N-1 definition for the corresponding timestep. If a contingency id is not found, it
        will be skipped.
    """
    case_indices_all_t = []
    for case_ids, n_minus1_def in zip(case_ids_all_t, n_minus1_definitions, strict=True):
        id_to_index = {cont.id: idx for idx, cont in enumerate(n_minus1_def.contingencies)}
        case_indices = [id_to_index[case_id] for case_id in case_ids if case_id in id_to_index]
        case_indices_all_t.append(case_indices)
    return case_indices_all_t


def pull(
    selected_strategy: list[ACOptimTopology],
    session: Session = None,
    n_minus1_definitions: Optional[list[Nminus1Definition]] = None,
) -> list[ACOptimTopology]:
    """Pull a promising topology from the DC repertoire to AC

    This only copies the topology without any changes other than setting the optimizer type to AC.
    This function takes a list of selected DC topologies and creates corresponding AC topologies by copying
    relevant attributes, setting the optimizer type to AC, and merging contingency case indices from both
    the DC and unsplit AC topologies.

    Parameters
    ----------
    selected_strategy : list[ACOptimTopology]
        The selected strategy to pull
    session : Session, optional
        The database session to use, by default None. The session object is used to fetch the unsplit AC topology
        which is then used to add the critical contingency cases to the pulled strategy. These critical cases
        can then be used for early stopping of AC N-1 contingency analysis.
    n_minus1_definitions : Optional[list[Nminus1Definition]]
        The N-1 definitions to use for the pulled strategy. If not provided, the pulled strategy will not
        include any N-1 contingency cases while calculating the top critical contingencies for early stopping.

    Returns
    -------
    list[ACOptimTopology]
        The a copy of the input topologies with the optimizer type set to AC
    """
    if not selected_strategy:
        return []

    optimization_id = selected_strategy[0].optimization_id

    # Unsplit AC topology will be used to merge critical contingency cases. This topology is pushed in the repertoire
    # in the initialization phase of the AC optimizer.
    unsplit_ac_topo = get_unsplit_ac_topology(optimization_id=optimization_id, session=session) if session else None
    worst_k_cont_ids_unsplit = set(unsplit_ac_topo.worst_k_contingency_cases) if unsplit_ac_topo else set()
    if n_minus1_definitions is None:
        logger.warning(
            "N-1 definition is not provided to pull method."
            " Early stopping in AC validation will not consider worst_k DC indices"
        )
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
            fitness=-9999999,
            parent_id=topo.id,
            metrics=metrics,
            worst_k_contingency_cases=merged_ids,
        )
        new_topo.metrics["top_k_overloads_n_1"] = (
            unsplit_ac_topo.metrics.get("top_k_overloads_n_1", None) if unsplit_ac_topo else None
        )
        pulled_strategy.append(new_topo)

    return pulled_strategy


def reconnect(rng: Rng, selected_strategy: list[ACOptimTopology]) -> list[ACOptimTopology]:
    """Reconnect a disconnected branch

    The idea behind this mutation operation is that the DC part might have disconnected too many
    branches, so we try to simplify the topology by reconnecting a single branch in all timesteps

    This might accidentally create the unsplit topology and does not check for that case. A check
    for this happens upon insertion into the database where the unique constraint will prevent
    duplicates.

    Parameters
    ----------
    rng : Rng
        The random number generator to use
    selected_strategy : list[ACOptimTopology]
        The selected DC or AC strategy to reconnect

    Returns
    -------
    list[ACOptimTopology]
        The a copy of the input topologies with the optimizer type set to AC and a branch reconnected
        or the empty list if no disconnections were present in the input topologies
    """
    branch_set = set([disc for topo in selected_strategy for disc in topo.disconnections])
    if len(branch_set) == 0:
        return []

    chosen = rng.choice(list(branch_set))
    topos = [
        ACOptimTopology(
            **topo.model_dump(
                include=[
                    "actions",
                    "pst_setpoints",
                    "timestep",
                    "optimization_id",
                    "strategy_hash",
                ]
            ),
            optimizer_type=OptimizerType.AC,
            fitness=-9999999,
            disconnections=[disc for disc in topo.disconnections if disc != chosen],
            unsplit=False,
            parent_id=topo.id,
        )
        for topo in selected_strategy
    ]

    updated_hash = hash_topologies(topos)
    unsplit = is_unsplit_topologies(topos)
    for topo in topos:
        topo.strategy_hash = updated_hash
        topo.unsplit = unsplit

    return topos


def close_coupler(
    rng: Rng,
    selected_strategy: list[ACOptimTopology],
) -> list[ACOptimTopology]:
    """Close a coupler in the selected strategy.

    A station that has an open coupler in any of the timesteps is selected and the coupler is closed
    for all timesteps in the topology.

    This might accidentally create the unsplit topology and does not prohibit that case. A check
    for this happens upon insertion into the database where the unique constraint will prevent
    duplicates. The unsplit flag will be set correctly though, so in case the unsplit topology was
    not yet in the database for some reason, this will add it.

    Parameters
    ----------
    rng : Rng
        The random number generator to use
    selected_strategy : list[ACOptimTopology]
        The selected DC or AC strategy to close a coupler. The number of timesteps must be equal to
        the number of timesteps in the branches_per_sub and injections_per_sub arrays. If the empty
        list is passed, the function will return an empty list.

    Returns
    -------
    list[ACOptimTopology]
        The a copy of the input topologies with the optimizer type set to AC and a coupler closed
        or the empty list if no open couplers were present in the input topologies
    """
    if selected_strategy == []:
        return []

    if not any(len(topo.actions) for topo in selected_strategy):
        return []

    # Select an action to delete (will close the coupler) and delete it in all timesteps
    selected_action = rng.choice(([action for topo in selected_strategy for action in topo.actions]))

    new_actions = [[action for action in topo.actions if action != selected_action] for topo in selected_strategy]

    # Copy the input strategy and replace the action in all timesteps
    topos = [
        ACOptimTopology(
            **topo.model_dump(
                include=[
                    "disconnections",
                    "pst_setpoints",
                    "timestep",
                    "optimization_id",
                    "strategy_hash",  # Will be updated below
                ]
            ),
            optimizer_type=OptimizerType.AC,
            fitness=-9999999,
            actions=new_actions_timestep,
            unsplit=False,  # Will be updated below
            parent_id=topo.id,
        )
        for topo, new_actions_timestep in zip(selected_strategy, new_actions, strict=True)
    ]

    updated_hash = hash_topologies(topos)
    unsplit = is_unsplit_topologies(topos)
    for topo in topos:
        topo.strategy_hash = updated_hash
        topo.unsplit = unsplit
    return topos


def evolution(
    rng: Rng,
    session: Session,
    optimization_id: str,
    close_coupler_prob: float,
    reconnect_prob: float,
    pull_prob: float,
    max_retries: int,
    n_minus1_definitions: Optional[list[Nminus1Definition]] = None,
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
    close_coupler_prob : float
        The probability of closing a coupler
    reconnect_prob : float
        The probability of reconnecting a branch
    pull_prob : float
        The probability of pulling a strategy
    max_retries : int
        The maximum number of retries to perform if a strategy is already in the database
    n_minus1_definitions : Optional[list[Nminus1Definition]]
        A list of N-1 definitions, one for each timestep, containing the contingencies.
    filter_strategy : Optional[FilterStrategy]
        The filter strategy to use for the optimization,
        used to filter out strategies that are too far away from the original topology.

    Returns
    -------
    list[ACOptimTopology]
        The strategy that was created during the evolution or an empty list if something went
        wrong at all retries
    """
    for _try in range(max_retries):
        new_strategy = evolution_try(
            rng=rng,
            session=session,
            optimization_id=optimization_id,
            close_coupler_prob=close_coupler_prob,
            reconnect_prob=reconnect_prob,
            pull_prob=pull_prob,
            n_minus1_definition=n_minus1_definitions,
            filter_strategy=filter_strategy,
        )
        if len(new_strategy):
            return new_strategy
    return []


def evolution_try(
    rng: Rng,
    session: Session,
    optimization_id: str,
    close_coupler_prob: float,
    reconnect_prob: float,
    pull_prob: float,
    n_minus1_definition: Optional[list[Nminus1Definition]] = None,
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
    close_coupler_prob : float
        The probability of closing a coupler
    reconnect_prob : float
        The probability of reconnecting a branch
    pull_prob : float
        The probability of pulling a strategy
    n_minus1_definition : Optional[list[Nminus1Definition]]
        A list of N-1 definitions, one for each timestep, containing the contingencies.
    filter_strategy : Optional[FilterStrategy]
        The filter strategy to use for the optimization,
        used to filter out strategies that are too far away from the original topology.

    Returns
    -------
    list[ACOptimTopology]
        The strategy that was created during the evolution or an empty list if something went
        wrong.
    """
    action_choice = rng.choice(["pull", "reconnect", "close_coupler"], p=[pull_prob, reconnect_prob, close_coupler_prob])
    if action_choice == "pull":
        old_strategy = select_strategy(
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
            filter_strategy=filter_strategy,
        )
        new_strategy = pull(selected_strategy=old_strategy, session=session, n_minus1_definitions=n_minus1_definition)
    elif action_choice == "reconnect":
        old_strategy = select_strategy(
            rng=rng,
            repertoire=select_repertoire(
                optimization_id=optimization_id,
                optimizer_type=[OptimizerType.DC, OptimizerType.AC],
                without_parent_on=[],
                session=session,
            ),
            candidates=select_repertoire(
                optimization_id=optimization_id,
                optimizer_type=[OptimizerType.DC, OptimizerType.AC],
                without_parent_on=[OptimizerType.AC],
                session=session,
            ),
            interest_scorer=default_scorer,
            filter_strategy=None,
        )

        new_strategy = reconnect(
            rng=rng,
            selected_strategy=old_strategy,
        )
    elif action_choice == "close_coupler":
        old_strategy = select_strategy(
            rng=rng,
            repertoire=select_repertoire(
                optimization_id=optimization_id,
                optimizer_type=[OptimizerType.DC, OptimizerType.AC],
                without_parent_on=[],
                session=session,
            ),
            candidates=select_repertoire(
                optimization_id=optimization_id,
                optimizer_type=[OptimizerType.DC, OptimizerType.AC],
                without_parent_on=[OptimizerType.AC],
                session=session,
            ),
            interest_scorer=default_scorer,
            filter_strategy=None,
        )
        new_strategy = close_coupler(
            rng=rng,
            selected_strategy=old_strategy,
        )
    else:
        raise RuntimeError("np.random.choice returned an unexpected value")

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
