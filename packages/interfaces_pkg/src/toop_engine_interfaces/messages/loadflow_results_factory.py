"""Factory methods for Loadflow results messages."""

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import NonNegativeFloat
from toop_engine_interfaces.messages.protobuf_schema.lf_service.loadflow_results_pb2 import (
    ErrorResult,
    LoadflowBaseResult,
    LoadflowStartedResult,
    LoadflowStreamResult,
    LoadflowSuccessResult,
)
from toop_engine_interfaces.messages.protobuf_schema.lf_service.stored_loadflow_reference_pb2 import StoredLoadflowReference


def create_loadflow_stream_result(
    loadflow_reference: StoredLoadflowReference,
    solved_timesteps: List[int],
    remaining_timesteps: List[int],
    result_type: Literal["loadflow_stream"] = "loadflow_stream",
) -> LoadflowStreamResult:
    """
    Create a LoadflowStreamResult instance.

    Parameters
    ----------
    loadflow_reference : StoredLoadflowReference
        The reference to the stored loadflow result.
    solved_timesteps : list of int
        The list of solved timesteps.
    remaining_timesteps : list of int
        The list of remaining timesteps.
    result_type : Literal["loadflow_stream"], optional
        The discriminator for the Result Union (default is "loadflow_stream").

    Returns
    -------
    LoadflowStreamResult
        The created LoadflowStreamResult object.
    """
    return LoadflowStreamResult(
        loadflow_reference=loadflow_reference,
        solved_timesteps=solved_timesteps,
        remaining_timesteps=remaining_timesteps,
        result_type=result_type,
    )


def create_loadflow_success_result(
    loadflow_reference: StoredLoadflowReference, result_type: Literal["loadflow_success"] = "loadflow_success"
) -> LoadflowSuccessResult:
    """
    Create a LoadflowSuccessResult instance.

    Parameters
    ----------
    loadflow_reference : StoredLoadflowReference
        The reference to the stored loadflow result.
    result_type : Literal["loadflow_success"], optional
        The discriminator for the Result Union (default is "loadflow_success").

    Returns
    -------
    LoadflowSuccessResult
        The created LoadflowSuccessResult object.
    """
    return LoadflowSuccessResult(
        loadflow_reference=loadflow_reference,
        result_type=result_type,
    )


def create_loadflow_started_result() -> LoadflowStartedResult:
    """
    Create a LoadflowStartedResult instance.

    Returns
    -------
    LoadflowStartedResult
        The created LoadflowStartedResult object.
    """
    return LoadflowStartedResult(result_type="loadflow_started")


def create_error_result(
    error: str,
) -> ErrorResult:
    """
    Create an ErrorResult instance.

    Parameters
    ----------
    error : str
        The error message.

    Returns
    -------
    ErrorResult
        The created ErrorResult object.
    """
    return ErrorResult(error=error, result_type="error")


def create_loadflow_base_result(
    loadflow_id: str,
    job_id: str,
    runtime: NonNegativeFloat,
    result: ErrorResult | LoadflowSuccessResult | LoadflowStreamResult | LoadflowStartedResult,
    instance_id: str = "",
    uuid_str: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> LoadflowBaseResult:
    """
    Create a LoadflowBaseResult instance.

    Parameters
    ----------
    loadflow_id : str
        The loadflow_id sent in the loadflow_command.
    job_id : str
        The job_id sent in the loadflow_command.
    runtime : NonNegativeFloat
        The runtime in seconds.
    result : ErrorResult or LoadflowSuccessResult or LoadflowStreamResult or LoadflowStartedResult
        The actual result data.
    instance_id : str, optional
        The instance id of the importer worker (default is "").
    uuid_str : str, optional
        Unique identifier for the result message (default is generated).
    timestamp : str, optional
        Timestamp when the result was sent (default is generated).

    Returns
    -------
    LoadflowBaseResult
        The created LoadflowBaseResult object.
    """
    if runtime < 0:
        raise ValueError("Runtime must be a non-negative float.")
    res = LoadflowBaseResult(
        loadflow_id=loadflow_id,
        job_id=job_id,
        instance_id=instance_id,
        runtime=runtime,
        uuid=uuid_str or str(uuid.uuid4()),
        timestamp=timestamp or str(datetime.now()),
    )
    if isinstance(result, ErrorResult):
        res.error_result.CopyFrom(result)
    elif isinstance(result, LoadflowSuccessResult):
        res.success_result.CopyFrom(result)
    elif isinstance(result, LoadflowStreamResult):
        res.stream_result.CopyFrom(result)
    elif isinstance(result, LoadflowStartedResult):
        res.started_result.CopyFrom(result)
    else:
        raise ValueError("Invalid result type provided.")
    return res
