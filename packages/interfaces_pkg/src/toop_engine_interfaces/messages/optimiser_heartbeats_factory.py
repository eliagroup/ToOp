"""
Factory functions for creating protobuf messages defined in `messages.optimiser.heartbeat.proto`.

These messages are used by optimizer instances to periodically communicate
their status, progress, and logs to a backend or monitoring service.

Includes:
---------
- **OptimizationStartedHeartbeat**: Sent when an optimization begins.
- **OptimizationStatsHeartbeat**: Reports progress statistics during optimization.
- **IdleHeartbeat**: Indicates the optimizer is idle.
- **LogMessage**: A log or error message emitted by the optimizer.
- **HeartbeatUnion**: A discriminated union wrapping one of the above messages.
- **Heartbeat**: The top-level envelope for any heartbeat message.

Each factory validates fields according to protobuf semantics and
provides sensible defaults where possible.
"""

import uuid
from datetime import datetime
from typing import Literal, Union

from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    OptimizerType,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    Heartbeat as PbHeartbeat,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    HeartbeatUnion as PbHeartbeatUnion,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    IdleHeartbeat as PbIdleHeartbeat,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    LogMessage as PbLogMessage,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    OptimizationStartedHeartbeat as PbOptimizationStartedHeartbeat,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_heartbeats_pb2 import (
    OptimizationStatsHeartbeat as PbOptimizationStatsHeartbeat,
)


def create_optimization_started_heartbeat(
    optimization_id: str,
    message_type: Literal["optimization_started"] = "optimization_started",
) -> PbOptimizationStartedHeartbeat:
    """
    Create an OptimizationStartedHeartbeat message.

    Parameters
    ----------
    message_type : Literal["optimization_started"]
        The message type (always "optimization_started").
    optimization_id : str
        The optimization ID associated with this heartbeat.

    Returns
    -------
    PbOptimizationStartedHeartbeat
        A protobuf instance of `OptimizationStartedHeartbeat`.

    Raises
    ------
    ValueError
        If message_type or optimization_id is empty.
    """
    if not message_type:
        raise ValueError("message_type must be provided.")
    if not optimization_id:
        raise ValueError("optimization_id must be provided.")

    return PbOptimizationStartedHeartbeat(
        message_type=message_type,
        optimization_id=optimization_id,
    )


def create_optimization_stats_heartbeat(
    optimization_id: str,
    wall_time: float,
    iteration: int,
    num_branch_topologies_tried: int,
    num_injection_topologies_tried: int,
    message_type: Literal["optimization_stats"] = "optimization_stats",
) -> PbOptimizationStatsHeartbeat:
    """
    Create an OptimizationStatsHeartbeat message.

    Parameters
    ----------
    message_type : Literal["optimization_stats"]
        The message type (always "optimization_stats").
    optimization_id : str
        The optimization ID.
    wall_time : float
        Number of seconds since optimization start.
    iteration : int
        Current iteration number.
    num_branch_topologies_tried : int
        Number of branch topologies tried.
    num_injection_topologies_tried : int
        Number of injection topologies tried.

    Returns
    -------
    PbOptimizationStatsHeartbeat
        A protobuf `OptimizationStatsHeartbeat` instance.

    Raises
    ------
    ValueError
        If any numeric argument is negative or identifiers missing.
    """
    if not message_type:
        raise ValueError("message_type must be non-empty.")
    if not optimization_id:
        raise ValueError("optimization_id must be non-empty.")
    if wall_time < 0 or iteration < 0:
        raise ValueError("wall_time and iteration must be non-negative.")
    if num_branch_topologies_tried < 0 or num_injection_topologies_tried < 0:
        raise ValueError("Topology counts must be non-negative.")

    return PbOptimizationStatsHeartbeat(
        message_type=message_type,
        optimization_id=optimization_id,
        wall_time=wall_time,
        iteration=iteration,
        num_branch_topologies_tried=num_branch_topologies_tried,
        num_injection_topologies_tried=num_injection_topologies_tried,
    )


def create_idle_heartbeat(message_type: Literal["idle"] = "idle") -> PbIdleHeartbeat:
    """
    Create an IdleHeartbeat message.

    Parameters
    ----------
    message_type : str, optional
        Message type identifier (default: "idle").

    Returns
    -------
    PbIdleHeartbeat
        A protobuf `IdleHeartbeat` instance.

    Raises
    ------
    ValueError
        If message_type is empty.
    """
    return PbIdleHeartbeat(message_type=message_type)


def create_log_message(
    optimization_id: str,
    message: str,
    error: bool = False,
    message_type: Literal["log"] = "log",
) -> PbLogMessage:
    """
    Create a LogMessage for the optimizer.

    Parameters
    ----------
    message_type : Literal["log"]
        Message type (always "log").
    optimization_id : str
        Optimization ID the log belongs to.
    message : str
        Log message content.
    error : bool, optional
        Whether this log represents an error (default: False).

    Returns
    -------
    PbLogMessage
        A protobuf `LogMessage` instance.

    Raises
    ------
    ValueError
        If message_type, optimization_id, or message are empty.
    """
    return PbLogMessage(
        message_type=message_type,
        optimization_id=optimization_id,
        message=message,
        error=error,
    )


def create_heartbeat_union(
    message: Union[PbIdleHeartbeat, PbLogMessage, PbOptimizationStatsHeartbeat, PbOptimizationStartedHeartbeat],
) -> PbHeartbeatUnion:
    """
    Create a HeartbeatUnion message wrapping a specific heartbeat type.

    Parameters
    ----------
    message : one of
        PbIdleHeartbeat, PbLogMessage, PbOptimizationStatsHeartbeat,
        PbOptimizationStartedHeartbeat

    Returns
    -------
    PbHeartbeatUnion
        A protobuf `HeartbeatUnion` instance.

    Raises
    ------
    ValueError
        If message is of an unsupported type.
    """
    if isinstance(message, PbIdleHeartbeat):
        return PbHeartbeatUnion(idle=message)
    if isinstance(message, PbLogMessage):
        return PbHeartbeatUnion(log=message)
    if isinstance(message, PbOptimizationStatsHeartbeat):
        return PbHeartbeatUnion(optimization_stats=message)
    if isinstance(message, PbOptimizationStartedHeartbeat):
        return PbHeartbeatUnion(optimization_started=message)

    raise ValueError(f"Unsupported message type for HeartbeatUnion: {type(message).__name__}")


def create_heartbeat(
    message: PbHeartbeatUnion,
    optimizer_type: OptimizerType,
    instance_id: str,
    timestamp: str = str(datetime.now()),
    uuid: str = str(uuid.uuid4()),
) -> PbHeartbeat:
    """
    Create a Heartbeat message (the full envelope).

    Parameters
    ----------
    message : PbHeartbeatUnion
        The heartbeat payload.
    optimizer_type : OptimizerType
        Which optimizer sent this heartbeat.
    instance_id : str
        Unique optimizer instance identifier.
    timestamp : str
        When the heartbeat was sent (ISO 8601).
    uuid : str
        Unique heartbeat identifier for deduplication.

    Returns
    -------
    PbHeartbeat
        A protobuf `Heartbeat` instance.

    Raises
    ------
    ValueError
        If any identifier or message is missing.
    """
    if not isinstance(message, PbHeartbeatUnion):
        raise ValueError("message must be a PbHeartbeatUnion instance.")
    if not all([optimizer_type, instance_id, timestamp, uuid]):
        raise ValueError("All identifiers (optimizer_type, instance_id, timestamp, uuid) must be non-empty.")

    return PbHeartbeat(
        message=message,
        optimizer_type=optimizer_type.value,
        instance_id=instance_id,
        timestamp=timestamp,
        uuid=uuid,
    )
