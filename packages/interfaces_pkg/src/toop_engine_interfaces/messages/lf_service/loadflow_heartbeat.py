"""Loadflow Heartbeat Commands for the kafka worker."""

import uuid
from datetime import datetime

from beartype.typing import Optional
from pydantic import BaseModel, Field


class LoadflowStatusInfo(BaseModel):
    """A status info to inform about an ongoint Loadflow solving action."""

    loadflow_id: str
    """The id of the loadflow solving job."""

    runtime: float
    """The amount of time since the start of the optimization."""

    message: Optional[str] = ""
    """An optional message"""


class LoadflowHeartbeat(BaseModel):
    """A message class for heartbeats from the loadflow worker.

    When idle, this just sends a hello, and when solving it also conveys the current status of the Loadflow Analysis
    """

    idle: bool
    """Whether the worker is idle"""

    status_info: Optional[LoadflowStatusInfo]
    """If not idle, a status update"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the heartbeat was sent"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this heartbeat message, used to avoid duplicates during processing"""
