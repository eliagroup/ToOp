# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Loadflow results messages for the loadflow service."""

import uuid
from datetime import datetime

from beartype.typing import Literal, Union
from pydantic import BaseModel, Field, NonNegativeFloat
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference


class LoadflowStreamResult(BaseModel):
    """Results of a loadflow solving job, including the timesteps as they are solved."""

    loadflow_reference: StoredLoadflowReference
    """The reference to the stored loadflow result"""

    solved_timesteps: list[int]
    """The list of solved timesteps"""

    remainging_timesteps: list[int]
    """The list of remaining timesteps. If there are none, send a LoadflowSuccessResult instead"""

    result_type: Literal["loadflow_stream"] = "loadflow_stream"
    """The discriminator for the Result Union"""


class LoadflowSuccessResult(BaseModel):
    """Results of a loadflow solving run, including the"""

    loadflow_reference: StoredLoadflowReference
    """The reference to the stored loadflow result"""

    result_type: Literal["loadflow_success"] = "loadflow_success"
    """The discriminator for the Result Union"""


class LoadflowStartedResult(BaseModel):
    """A message that is sent when the preprocessing process has started"""

    result_type: Literal["loadflow_started"] = "loadflow_started"
    """The discriminator for the Result Union"""


class ErrorResult(BaseModel):
    """A message that is sent if an error occurred"""

    error: str
    """The error message"""

    result_type: Literal["error"] = "error"
    """The discriminator for the Result Union"""


class LoadflowBaseResult(BaseModel):
    """A generic class for result, holding either a successful or an unsuccessful result"""

    loadflow_id: str
    """The loadflow_id that was sent in the loadflow_command, used to identify the result"""

    job_id: str
    """The job_id that was sent in the loadflow_command, used to identify the result"""

    instance_id: str = ""
    """The instance id of the importer worker that created this result"""

    runtime: NonNegativeFloat
    """The runtime in seconds that the preprocessing took until the result"""

    result: Union[ErrorResult, LoadflowSuccessResult, LoadflowStreamResult, LoadflowStartedResult] = Field(
        discriminator="result_type"
    )
    """The actual result data in a discriminated union"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this result message, used to avoid duplicates during processing"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the result was sent"""
