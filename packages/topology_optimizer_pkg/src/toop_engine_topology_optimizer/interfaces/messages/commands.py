# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines the interaction between the optimizer and a backend.

The backend will send commands in the form of messages to the optimizer, which will trigger
a certain behaviour. The Optimizer will respond with results, for this see results.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, Field
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


def validate_first_gridfile_uncoupled(val: list[GridFile]) -> list[GridFile]:
    """Check that the first gridfile is uncoupled"""
    if val:
        if val[0].coupling != "none":
            raise ValueError("The first gridfile must be uncoupled")
    return val


class StartOptimizationCommand(BaseModel):
    """Command with parameters for starting an optimization run."""

    message_type: Literal["start_optimization"] = "start_optimization"
    """The command type for deserialization, don't change this"""

    dc_params: DCOptimizerParameters = DCOptimizerParameters()
    """The parameters for the DC optimizer"""

    ac_params: ACOptimizerParameters = ACOptimizerParameters()
    """The parameters for the AC optimizer"""

    grid_files: Annotated[list[GridFile], AfterValidator(validate_first_gridfile_uncoupled)]
    """The grid files to load, where each gridfile represents one timestep. The grid files also
    include coupling information for the timesteps."""

    optimization_id: str
    """The id of the optimization run, used to identify the optimization run in the results. Should
    stay the same for the whole optimization run and should be equal to the kafka event key"""


class ShutdownCommand(BaseModel):
    """Command to shutdown the worker."""

    message_type: Literal["shutdown"] = "shutdown"
    """The command type for deserialization, don't change this"""

    exit_code: int = 0
    """The exit code to exit with"""


class Command(BaseModel):
    """Base class for all commands to the optimizer."""

    command: Union[
        StartOptimizationCommand,
        ShutdownCommand,
    ] = Field(discriminator="message_type")
    """The actual command"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for the command message, used to avoid duplicate processing on
    optimizer side"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the command was sent"""
