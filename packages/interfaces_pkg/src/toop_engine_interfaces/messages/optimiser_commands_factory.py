"""Module: optimiser_commands_factory

This module provides factory functions for creating protobuf command messages used in the optimizer engine.
It includes utilities to construct StartOptimizationCommand and ShutdownCommand messages, as well as a generic
Command wrapper for sending commands with metadata such as UUID and timestamp.

Functions
---------
- create_start_optimization_command(grid_files, optimization_id, dc_params=None, ac_params=None)
    Creates a StartOptimizationCommand message with provided grid files and optimizer parameters.

- create_shutdown_command(exit_code=0)
    Creates a ShutdownCommand message to terminate the worker with a specified exit code.

- create_command(command, uuid_=None, timestamp=None)
    Wraps a StartOptimizationCommand or ShutdownCommand in a Command message, adding UUID and timestamp metadata.
"""

import uuid
from datetime import datetime
from typing import List, Union

from toop_engine_interfaces.messages.optimiser_ac_params_factory import create_ac_optimizer_parameters
from toop_engine_interfaces.messages.optimiser_dc_params_factory import create_dc_optimizer_parameters
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    GridFile as PbGridFile,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_optimizer_params_pb2 import (
    ACOptimizerParameters as PbACOptimizerParameters,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.dc_optimizer_params_pb2 import (
    DCOptimizerParameters as PbDCOptimizerParameters,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_commands_pb2 import (
    Command as PbCommand,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_commands_pb2 import (
    ShutdownCommand as PbShutdownCommand,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.optimiser_commands_pb2 import (
    StartOptimizationCommand as PbStartOptimizationCommand,
)


def create_start_optimization_command(
    grid_files: List[PbGridFile],
    optimization_id: str,
    dc_params: PbDCOptimizerParameters = None,
    ac_params: PbACOptimizerParameters = None,
) -> PbStartOptimizationCommand:
    """
    Create a StartOptimizationCommand message.

    Parameters
    ----------
    dc_params : PbDCOptimizerParameters
        The parameters for the DC optimizer.
    ac_params : PbACOptimizerParameters
        The parameters for the AC optimizer.
    grid_files : list of PbGridFile
        The grid files to load, where each GridFile represents one timestep.
        The first grid file must be uncoupled (`coupling == "none"`).
    optimization_id : str
        The ID of the optimization run (should remain constant for the entire run).

    Returns
    -------
    PbStartOptimizationCommand
        A protobuf `StartOptimizationCommand` instance.

    Raises
    ------
    ValueError
        If no grid files are provided, or if the first grid file is not uncoupled.
    """
    if dc_params is None:
        dc_params = create_dc_optimizer_parameters()
    if ac_params is None:
        ac_params = create_ac_optimizer_parameters()
    if not grid_files:
        raise ValueError("At least one GridFile must be provided.")
    if grid_files[0].coupling != "none":
        raise ValueError(f"The first GridFile must be uncoupled (coupling == 'none'), got '{grid_files[0].coupling}'.")
    if not optimization_id:
        raise ValueError("optimization_id must be a non-empty string.")

    return PbStartOptimizationCommand(
        message_type="start_optimization",
        dc_params=dc_params,
        ac_params=ac_params,
        grid_files=grid_files,
        optimization_id=optimization_id,
    )


def create_shutdown_command(exit_code: int = 0) -> PbShutdownCommand:
    """
    Create a ShutdownCommand message.

    Parameters
    ----------
    exit_code : int, optional
        The exit code to terminate the worker with. Default is 0.

    Returns
    -------
    PbShutdownCommand
        A protobuf `ShutdownCommand` instance.

    Raises
    ------
    ValueError
        If `exit_code` is negative.
    """
    if exit_code < 0:
        raise ValueError("exit_code must be a non-negative integer.")

    return PbShutdownCommand(
        message_type="shutdown",
        exit_code=exit_code,
    )


def create_command(
    command: Union[PbStartOptimizationCommand, PbShutdownCommand],
    uuid_: str | None = None,
    timestamp: str | None = None,
) -> PbCommand:
    """
    Create a Command wrapper message containing either a StartOptimizationCommand or a ShutdownCommand.

    Parameters
    ----------
    command : Union[PbStartOptimizationCommand, PbShutdownCommand]
        The actual command to wrap.
    uuid_ : str, optional
        A unique identifier for this command message.
        If not provided, a UUID4 string will be generated automatically.
    timestamp : str, optional
        When the command was sent (ISO 8601 string).
        If not provided, the current datetime will be used.

    Returns
    -------
    PbCommand
        A protobuf `Command` instance wrapping the provided command.

    Raises
    ------
    TypeError
        If the command type is unsupported.
    """
    uuid_ = uuid_ or str(uuid.uuid4())
    timestamp = timestamp or datetime.now().isoformat()

    if isinstance(command, PbStartOptimizationCommand):
        return PbCommand(
            start_optimization=command,
            uuid=uuid_,
            timestamp=timestamp,
        )
    if isinstance(command, PbShutdownCommand):
        return PbCommand(
            shutdown=command,
            uuid=uuid_,
            timestamp=timestamp,
        )
    raise TypeError(
        f"Unsupported command type '{type(command).__name__}'. Expected StartOptimizationCommand or ShutdownCommand."
    )
