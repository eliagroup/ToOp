# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains the heartbeat messages that are sent by the worker."""

import uuid
from datetime import datetime

from beartype.typing import Literal, TypeAlias, Union
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt
from toop_engine_topology_optimizer.interfaces.messages.commons import OptimizerType


class OptimizationStartedHeartbeat(BaseModel):
    """A message that is sent by the optimizer when it starts."""

    message_type: Literal["optimization_started"] = "optimization_started"
    """The message type, don't change this"""

    optimization_id: str
    """The ID of the optimization"""


class OptimizationStatsHeartbeat(BaseModel):
    """Optimization statistics to track the progress of the optimization."""

    message_type: Literal["optimization_stats"] = "optimization_stats"
    """The message type, don't change this"""

    optimization_id: str
    """The ID of the optimization"""

    wall_time: NonNegativeFloat
    """The number of seconds since the start of the optimization"""

    iteration: NonNegativeInt
    """The current iteration number"""

    num_branch_topologies_tried: NonNegativeInt
    """The number of branch topologies tried so far"""

    num_injection_topologies_tried: NonNegativeInt
    """The number of injection topologies tried so far"""


class IdleHeartbeat(BaseModel):
    """A heartbeat that is sent by the optimizer every now and then."""

    message_type: Literal["idle"] = "idle"
    """The message type, don't change this"""


class LogMessage(BaseModel):
    """A log message that is sent by the optimizer to the master."""

    message_type: Literal["log"] = "log"
    """The message type, don't change this"""

    optimization_id: str
    """The ID of the optimization this message belongs to"""

    message: str
    """The message to log"""

    error: bool = False
    """Whether this is an error message"""


HeartbeatUnion: TypeAlias = Union[IdleHeartbeat, LogMessage, OptimizationStatsHeartbeat, OptimizationStartedHeartbeat]


class Heartbeat(BaseModel):
    """A parent heartbeat message"""

    message: HeartbeatUnion = Field(discriminator="message_type")
    """The actual heartbeat message"""

    optimizer_type: OptimizerType
    """Which optimizer has sent the heartbeat"""

    instance_id: str = ""
    """The instance ID of the optimizer"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the heartbeat was sent"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this heartbeat message, used to avoid duplicates during processing"""
