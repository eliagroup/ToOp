# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The database models to store topologies in the AC optimizer"""

from typing import Optional

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Strategy, Topology
from toop_engine_topology_optimizer.interfaces.models.base_storage import BaseDBTopology, hash_strategy, is_unsplit_strategy


class ACOptimTopology(BaseDBTopology, table=True):
    """Inherits from the base topology to make a database table for AC optimizer topologies

    This can include both AC and DC topologies, with the specific needs of the AC optimizer
    """

    __tablename__ = "ac_optim_topology"

    parent_id: Optional[int] = Field(foreign_key="ac_optim_topology.id", nullable=True, default=None)
    """The mutation parent id, i.e. the topology that this topology was mutated from. This is mostly for
    debugging purposes to see where a topology came from and understand the mutation process."""

    parent: Optional["ACOptimTopology"] = Relationship()
    """The mutation parent"""

    acceptance: Optional[bool] = Field(
        default=None,
    )
    """Whether the strategy was accepted or not."""


def convert_single_topology(
    topology: Topology,
    optimization_id: str,
    optimizer_type: OptimizerType,
    timestep: int,
    strategy_hash: bytes,
    unsplit: bool,
) -> ACOptimTopology:
    """Convert a single Topology to a ACOptimTopology

    Parameters
    ----------
    topology : Topology
        The topology to convert
    optimization_id : str
        The optimization ID to assign to the db topology
    optimizer_type : OptimizerType
        The optimizer type to assign to the db topology
    timestep : int
        The timestep of the topology
    strategy_hash : bytes
        The hash of the strategy computed through hash_strategy
    unsplit : bool
        Whether the strategy is unsplit

    Returns
    -------
    ACOptimTopology
        The converted topology
    """
    return ACOptimTopology(
        actions=topology.actions,
        disconnections=topology.disconnections,
        pst_setpoints=topology.pst_setpoints,
        unsplit=unsplit,
        timestep=timestep,
        strategy_hash=strategy_hash,
        optimization_id=optimization_id,
        optimizer_type=optimizer_type,
        fitness=topology.metrics.fitness,
        metrics=topology.metrics.extra_scores,
        worst_k_contingency_cases=topology.metrics.worst_k_contingency_cases,
    ).set_loadflow_reference(topology.loadflow_results)


def convert_message_topo_to_db_topo(
    message_strategies: list[Strategy], optimization_id: str, optimizer_type: OptimizerType
) -> list[ACOptimTopology]:
    """Convert a TopologyPushResult to a list of ACOptimTopology

    Parameters
    ----------
    message_strategies : list[Strategy]
        The strategies to convert, usually from a TopologyPushResult or OptimizationStartedResult. Strategies
        are flattened into the list of topologies that are being returned.
    optimization_id : str
        The optimization ID to assign to the topologies. This was sent through with the parent
        result message and is not part of TopologyPushResult
    optimizer_type : OptimizerType
        The optimizer type to assign. This was sent through with the parent result message and is not
        part of TopologyPushResult

    Returns
    -------
    list[ACOptimTopology]
        A list of converted topologies where for each topology and for each timestep a new
        ACOptimTopology instance is created.
    """
    converted = []
    for strategy in message_strategies:
        # Hash the strategy to get a unique global identifier
        strategy_hash = hash_strategy(strategy)
        unsplit = is_unsplit_strategy(strategy)
        for time_id, timestep_topo in enumerate(strategy.timesteps):
            converted.append(
                convert_single_topology(
                    topology=timestep_topo,
                    optimization_id=optimization_id,
                    optimizer_type=optimizer_type,
                    timestep=time_id,
                    strategy_hash=strategy_hash,
                    unsplit=unsplit,
                )
            )
    return converted


def create_session() -> Session:
    """Create an in-memory SQLite session with the ACOptimTopology table created.

    Returns
    -------
    Session
        The created session with the in-memory SQLite database
    """
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine, tables=[ACOptimTopology.__table__])
    session = Session(engine)
    return session
