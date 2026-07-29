# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared idle loop for DC worker processes."""

from datetime import datetime, timedelta

import structlog
from beartype.typing import Callable
from toop_engine_contingency_analysis.ac_loadflow_service.kafka_client import LongRunningKafkaConsumer
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, ShutdownCommand, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.heartbeats import HeartbeatUnion, IdleHeartbeat
from toop_engine_topology_optimizer.interfaces.messages.results import OptimizationStoppedResult, ResultUnion

logger = structlog.get_logger(__name__)


def idle_loop(
    consumer: LongRunningKafkaConsumer,
    send_heartbeat_fn: Callable[[HeartbeatUnion], None],
    send_result_fn: Callable[[ResultUnion, str], None],
    heartbeat_interval_ms: int,
    max_command_age_hours: float,
) -> StartOptimizationCommand:
    """Run idle loop of the worker.

    This will be running when the worker is currently not optimizing.
    This will wait until a StartOptimizationCommand is received and return it. In case a
    ShutdownCommand is received, the worker will exit with the exit code provided in the command.

    Parameters
    ----------
    consumer : LongRunningKafkaConsumer
        The initialized Kafka consumer to listen for commands on.
    send_heartbeat_fn : Callable[[HeartbeatUnion], None]
        A function to call when there were no messages received for a while.
    send_result_fn : Callable[[ResultUnion, str], None]
        A function to call to send results back to the results topic, used to send a message in case a command is too old.
    heartbeat_interval_ms : int
        The time to wait for a new command in milliseconds. If no command has been received, a
        heartbeat will be sent and then the receiver will wait for commands again.
    max_command_age_hours : float
        The maximum age of a command in hours.
        If a command is received that is older than this, the command will be ignored
        and a message will be sent to the results topic.

    Returns
    -------
    StartOptimizationCommand
        The start optimization command to start the optimization run with.

    Raises
    ------
    SystemExit
        If a ShutdownCommand is received.
    """
    send_heartbeat_fn(IdleHeartbeat())
    logger.info("Entering idle loop")
    while True:
        message = consumer.poll(timeout=heartbeat_interval_ms / 1000)

        if not message:
            send_heartbeat_fn(IdleHeartbeat())
            continue

        command = Command.model_validate_json(deserialize_message(message.value()))

        if isinstance(command.command, ShutdownCommand):
            logger.info("Shutting down due to ShutdownCommand")
            consumer.commit()
            consumer.consumer.close()
            raise SystemExit(command.command.exit_code)
        if isinstance(command.command, StartOptimizationCommand):
            time_of_command = datetime.fromisoformat(command.timestamp)
            if time_of_command < datetime.now() - timedelta(hours=max_command_age_hours):
                logger.warning(
                    f"Received command with timestamp from the past (timestamp: {time_of_command}, "
                    f"now: {datetime.now()}), skipping command"
                )
                send_result_fn(
                    OptimizationStoppedResult(
                        reason="command-too-old", message=f"Received outdated command: {command}. Skipping.."
                    ),
                    command.command.optimization_id,
                )
                consumer.commit()
                continue
            with structlog.contextvars.bound_contextvars(
                optimization_id=command.command.optimization_id,
            ):
                return command.command

        logger.warning("Received unknown command, dropping!", command=command, payload=message.value())
        consumer.commit()
