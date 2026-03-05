# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A listener that listens for new topologies on a kafka result stream and saves them to db

This is intended for usage both in the AC optimizer and the backend, as they both listen for
topologies on the result kafka stream. The backend has some further logic as it watches all
optimizations and might change the state of the optimization to "running" if it receives a start
optimization result and "stopped" if a stop optimization result is received.

One of the requirements is that the AC listener will only want to listen to a single optimization_id and
not to messages by itself. Hence a filtering mechanic is added.
"""

import logbook
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, convert_message_topo_to_db_topo
from toop_engine_topology_optimizer.interfaces.messages.results import (
    OptimizationStartedResult,
    OptimizationStoppedResult,
    Result,
    TopologyPushResult,
)

logger = logbook.Logger(__name__)


def poll_results_topic(
    db: Session,
    consumer: LongRunningKafkaConsumer,
    first_poll: bool = True,
) -> tuple[list[ACOptimTopology], list[str]]:
    """Poll the results topic for new topologies to store in the DB

    We store topologies from all optimization jobs, as it could be that we will later optimize the same job in this worker.

    Parameters
    ----------
    db : Session
        The database session to use for saving the topologies
    consumer : LongRunningKafkaConsumer
        The kafka consumer to poll messages from. It should already be subscribed to the result
        topic.
    first_poll : bool, optional
        If True, we assume the optimizatin has just started and we can afford to wait for the first DC results for
        a longer time (30 seconds). If False, we assume the optimization is already running and we don't want to block
        the consumer for too long (100 ms), by default True

    Returns
    -------
    list[ACOptimTopology]
        A list of topologies that were added to the database
    list[str]
        A list of optimization IDs for which a stop optimization result was received.
    """
    added_topos = []
    finished_optimizations = []
    messages = consumer.consume(timeout=30.0 if first_poll else 0.1, num_messages=10000)
    for message in messages:
        result = Result.model_validate_json(deserialize_message(message.value()))

        strategies = None
        if isinstance(result.result, TopologyPushResult):
            strategies = result.result.strategies
        elif isinstance(result.result, OptimizationStartedResult):
            strategies = [result.result.initial_topology]
        elif isinstance(result.result, OptimizationStoppedResult):
            finished_optimizations.append(result.optimization_id)
            continue
        else:
            continue

        topologies = convert_message_topo_to_db_topo(strategies, result.optimization_id, result.optimizer_type)

        # Push the topologies to the database, ignoring duplicates
        # Duplicates will trigger an IntegrityError
        for topo in topologies:
            try:
                db.add(topo)
                db.commit()
                added_topos.append(topo)
            except IntegrityError:
                db.rollback()
                pass

    return added_topos, finished_optimizations
