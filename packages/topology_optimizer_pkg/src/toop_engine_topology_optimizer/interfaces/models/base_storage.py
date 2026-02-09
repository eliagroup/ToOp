# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines a base SQLModel class for storing topologies from a result listener.

This is not a Table model yet, as an implementation might want to store extra data along with
the topology
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
from beartype.typing import Optional
from sqlalchemy import UniqueConstraint
from sqlmodel import JSON, Field, SQLModel
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics, Strategy, Topology


class BaseDBTopology(SQLModel):
    """Base class for storing a single-timestep topology in a database.

    Each single-timestep topology belongs to different entities:
        - A strategy, which is a set of topologies over time (time increases by one for every
            topology in the strategy)
        - An optimization/optimizer_type pair, which will produce many strategies. At the moment,
        only DC and AC optimizer types are implemented.
    """

    id: int = Field(default=None, primary_key=True)
    """The table primary key"""

    actions: list[int] = Field(sa_type=JSON)
    """The branch/injection reconfiguration actions as indices into the action set."""

    disconnections: list[int] = Field(sa_type=JSON)
    """A list of disconnections, indexing into the disconnectable branches set in the action set."""

    pst_setpoints: list[int] = Field(sa_type=JSON)
    """The setpoints for the PSTs if they have been computed. This is an index into the range of pst taps, i.e. the
    smallest tap is 0 and the neutral tap somewhere in the middle of the range. The tap range is defined in the action set.
    The list always has the same length, i.e. the number of controllable PSTs in the system, and each entry corresponds to
    the PST at the same position in the action set."""

    unsplit: bool
    """Whether all topologies in the strategy including this one have no branch assignments,
    disconnections or injections."""

    timestep: int
    """The timestep of this topology, starting at 0"""

    strategy_hash: bytes
    """The hash of the strategy - this hashes actions, disconnections and pst_setpoints
    for all timesteps in the strategy, making it possible to form a unique constraint on the strategy.
    This value will be set to the same for all topologies in the same strategy, furthermore making
    it possible to group timesteps."""

    @property
    def strategy_hash_str(self) -> str:
        """The strategy hash as a string, for human readability"""
        return self.strategy_hash.hex()

    optimization_id: str
    """The optimization ID this topology belongs to"""

    optimizer_type: OptimizerType
    """Which optimizer created this topology"""

    fitness: float
    """The fitness of this topology"""

    metrics: dict[MetricType, float] = Field(default_factory=lambda: {}, sa_type=JSON)
    """The metrics of this topology"""

    worst_k_contingency_cases: Optional[list[str]] = Field(default_factory=lambda: [], sa_type=JSON)
    """
    The worst k contingency case IDs for the topology.
    """
    created_at: datetime = Field(default=datetime.now(), nullable=False)
    """The time the topology was recorded in the database"""

    stored_loadflow_reference: Optional[str] = None
    """The file reference for the loadflow results of this topology/strategy, if they were computed. Multiple topologies
    belonging to the same strategy will have the same serialized loadflow results object as there is a timestep notion in
    the loadflow results. To obtain the correct loadflow results, use the timestep attribute.
    This is stored as a json serialized StoredLoadflowReference object"""

    __table_args__ = (
        UniqueConstraint(
            "optimization_id",
            "optimizer_type",
            "strategy_hash",
            "timestep",
            name="topo_unique",
        ),
    )

    def get_loadflow_reference(self) -> Optional[StoredLoadflowReference]:
        """Get the loadflow reference as a StoredLoadflowReference object

        Returns
        -------
        Optional[StoredLoadflowReference]
            The loadflow reference, or None if it is not set
        """
        if self.stored_loadflow_reference is None:
            return None
        return StoredLoadflowReference.model_validate_json(self.stored_loadflow_reference)

    def set_loadflow_reference(self, loadflow_reference: Optional[StoredLoadflowReference]) -> BaseDBTopology:
        """Set the loadflow reference from a StoredLoadflowReference object

        Parameters
        ----------
        loadflow_reference : Optional[StoredLoadflowReference]
            The loadflow reference to set, or None to unset it
        """
        if loadflow_reference is None:
            self.stored_loadflow_reference = None
        else:
            self.stored_loadflow_reference = loadflow_reference.model_dump_json()
        return self


def is_unsplit(data: list[tuple[list[int], list[int]]]) -> bool:
    """Check if a strategy is completely unsplit

    Parameters
    ----------
    data : list[tuple[list[int], list[int]]]
        A list of actions and disconnections for all topologies in a strategy

    Returns
    -------
    bool
        True if all topologies in the strategy are unsplit, False otherwise
    """
    return all((not len(actions) and not len(disc)) for actions, disc in data)


def is_unsplit_strategy(strategy: Strategy) -> bool:
    """Check if a strategy is completely unsplit

    Parameters
    ----------
    strategy : Strategy
        The strategy to check

    Returns
    -------
    bool
        True if all topologies in the strategy are unsplit, False otherwise
    """
    return is_unsplit([(topo.actions, topo.disconnections) for topo in strategy.timesteps])


def is_unsplit_topologies(topologies: list[BaseDBTopology]) -> bool:
    """Check if a list of topologies are completely unsplit

    Parameters
    ----------
    topologies : list[BaseDBTopology]
        The topologies to check

    Returns
    -------
    bool
        True if all topologies in the list are unsplit, False otherwise
    """
    return is_unsplit([(topo.actions, topo.disconnections) for topo in topologies])


def hash_topo_data(data: list[tuple[list[int], list[int], list[int]]]) -> bytes:
    """Hash a list of actions, disconnections and pst taps to a bytes digest directly

    Sorts the actions and disconnections to avoid duplicates in the hash.

    Parameters
    ----------
    data : list[tuple[list[int], list[int], list[int]]]
        A list of actions, disconnections and pst setpoints for all topologies in a strategy

    Returns
    -------
    bytes
        The hash of the data
    """
    hasher = hashlib.sha256(usedforsecurity=False)
    for action, disc, pst in data:
        action_sorted = sorted(action)
        disc_sorted = sorted(disc)
        string_repr = "*" + str(action_sorted) + "/" + str(disc_sorted) + "/" + str(pst)
        hasher.update(string_repr.encode("utf-8"))
    return hasher.digest()


def hash_topologies(topologies: list[BaseDBTopology]) -> bytes:
    """Re-computes the strategy hash from a list of topologies

    This needs to be used if any of these fields of the topology have changed:
        - actions
        - disconnections
        - pst_setpoints
        - timestep order

    Parameters
    ----------
    topologies : list[BaseDBTopology]
        The topologies to hash

    Returns
    -------
    bytes
        The hash of the topologies
    """
    # First order the topologies by timestep
    topologies = sorted(topologies, key=lambda topo: topo.timestep)
    return hash_topo_data([(topo.actions, topo.disconnections, topo.pst_setpoints) for topo in topologies])


def hash_strategy(strategy: Strategy) -> bytes:
    """Hash a strategy to bytes digest

    This will hash the actions, disconnections and pst_setpoints of all topologies
    in the strategy to a bytes array digest, suitable for unique constraint checking.

    Parameters
    ----------
    strategy : Strategy
        The strategy to hash

    Returns
    -------
    str
        The hash of the strategy
    """
    return hash_topo_data([(topo.actions, topo.disconnections, topo.pst_setpoints) for topo in strategy.timesteps])


def convert_db_topo_to_message_topo(topologies: list[BaseDBTopology]) -> list[Strategy]:
    """Convert a list of BaseDBTopology to a list of Strategy objects to be sent via kafka

    This will group the strategies in the input by strategy_hash

    Parameters
    ----------
    topologies : list[BaseDBTopology]
        The topologies to convert, can belong to multiple strategies

    Returns
    -------
    list[Strategy]
        The converted strategies in message format
    """
    if not topologies:
        return []

    # First sort the topologies by strategy hash and timestep
    topologies = sorted(topologies, key=lambda topo: (topo.strategy_hash, topo.timestep))

    # Walk through the topologies and whenever the strategy hash changes, create a new strategy
    # We collect topologies in the cache until we encounter a new strategy hash. Because we sorted before
    # this means we will never encounter that hash again.
    strategies = []
    topology_cache = []
    last_hash = topologies[0].strategy_hash
    for topo in topologies:
        if topo.strategy_hash != last_hash:
            strategies.append(Strategy(timesteps=topology_cache))
            topology_cache = []
            last_hash = topo.strategy_hash

        case_ids = topo.worst_k_contingency_cases
        topology_cache.append(
            Topology(
                actions=topo.actions,
                disconnections=topo.disconnections,
                pst_setpoints=topo.pst_setpoints,
                metrics=Metrics(fitness=topo.fitness, extra_scores=topo.metrics, worst_k_contingency_cases=case_ids),
                loadflow_results=topo.get_loadflow_reference(),
            )
        )
    strategies.append(Strategy(timesteps=topology_cache))
    return strategies


def metrics_dataframe(topologies: list[BaseDBTopology]) -> pd.DataFrame:
    """Convert a list of topologies to a pandas DataFrame, keeping only the metrics information

    Parameters
    ----------
    topologies : list[BaseDBTopology]
        The topologies to convert

    Returns
    -------
    pd.DataFrame
        The converted DataFrame. Will have the topology ID as the index and all present metrics
        including fitness as columns. Furthermore, the strategy hash and optimizer types will be included as a column.
    """
    if not topologies:
        return pd.DataFrame(columns=["fitness", "strategy_hash", "optimizer_type"], index=["id"])

    metrics_df = pd.DataFrame.from_records(
        (
            {
                "id": topo.id,
                "strategy_hash": topo.strategy_hash,
                "optimizer_type": topo.optimizer_type.value,
                "fitness": topo.fitness,
                **topo.metrics,
            }
            for topo in topologies
        ),
        index="id",
    )

    # fill fitness_dc for dc topologies to avoid NaN values
    cond_dc_topos = metrics_df["optimizer_type"] == OptimizerType.DC.value
    # if a new AC topology is pulled, it needs to save the fitness of the DC value
    # so that it can be used for the discriminator -> it does not get evaluated twice
    metrics_df.loc[cond_dc_topos, "fitness_dc"] = metrics_df[cond_dc_topos]["fitness"]

    return metrics_df
