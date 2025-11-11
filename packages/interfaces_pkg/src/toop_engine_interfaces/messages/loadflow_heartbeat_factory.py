"""Factory methods for LoadflowStatusInfo and LoadflowHeartbeat."""

import uuid
from datetime import datetime

from beartype.typing import Optional
from toop_engine_interfaces.messages.lf_service.loadflow_heartbeat import LoadflowHeartbeat, LoadflowStatusInfo


def create_loadflow_status_info(loadflow_id: str, runtime: float, message: Optional[str] = "") -> LoadflowStatusInfo:
    """
    Create a LoadflowStatusInfo instance.

    Parameters
    ----------
    loadflow_id : str
        The id of the loadflow solving job.
    runtime : float
        The amount of time since the start of the optimization.
    message : Optional[str], optional
        An optional message (default is "")

    Returns
    -------
    LoadflowStatusInfo
        The created LoadflowStatusInfo object.
    """
    return LoadflowStatusInfo(loadflow_id=loadflow_id, runtime=runtime, message=message)


def create_loadflow_heartbeat(
    idle: bool,
    status_info: Optional[LoadflowStatusInfo] = None,
    timestamp: Optional[str] = None,
    uuid_str: Optional[str] = None,
) -> LoadflowHeartbeat:
    """
    Create a LoadflowHeartbeat instance.

    Parameters
    ----------
    idle : bool
        Whether the worker is idle.
    status_info : Optional[LoadflowStatusInfo], optional
        If not idle, a status update (default is None).
    timestamp : Optional[str], optional
        When the heartbeat was sent (default is current time).
    uuid_str : Optional[str], optional
        A unique identifier for this heartbeat message (default is generated UUID).

    Returns
    -------
    LoadflowHeartbeat
        The created LoadflowHeartbeat object.
    """
    if timestamp is None:
        timestamp = str(datetime.now())
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())
    return LoadflowHeartbeat(idle=idle, status_info=status_info, timestamp=timestamp, uuid=uuid_str)
