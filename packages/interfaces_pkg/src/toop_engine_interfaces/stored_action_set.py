# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds a format for storing the action set for later use in postprocessing.

This is different from the jax-internal action set as defined in jax/types.py where only jax-relevant
information is stored, but is instead aimed at use in postprocessing and visualization. Instead of just
storing the electrical switching state, this bases on the asset topology to store physical switchings
to make a translation to .dgs or other formats easier.

One of the decisions to take was was whether to use a single action set for all timesteps or a different
one for each timestep. As the jax part currently also only supports one action set for all timesteps, we
decide to mirror this for the time being, i.e. we do not store strategies but topologies in the action set.

Furthermore, it should also be possible to use a global action set if necessary. Meaning, by default
 an action is substation-local, but it it can span multiple substations as well. Using a format that
is suitable for both options is desirable for easier collaboration.

Another question was whether to store the switching distance and busbar information in the action set, but the
switching distance can be trivially recomputed by using the station_diff between the starting topology and the
station in the action set. BB outage information can also be retrieved from the asset topology.

There is a slim hope of storing the action set independent of the grid state but based on the master grid, however
right now there is a fundamental way that 'binds' an action set to the specific grid it has been computed on: During
the enumerations, all electrical actions are enumerated and then physical realizations are found for it based on
heuristics. These heuristics take the grid state into account, so it could be that an electrical action can not be
realized the same way if maintenances are active. Hence, for the moment, it is no problem to tie the initial
topology into the action set.
"""

from pathlib import Path

import numpy as np
from beartype.typing import Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.nminus1_definition import GridElement


class PSTRange(GridElement):
    """Phase shifting transformers can be set within the scope of non-costly optimization.

    A PST has a list of taps, each with an angle shift.
    """

    shift_steps: list[float]
    """The list of possible tap positions, characterized by their angle shift for a given tap position. While there are
    secondary effects (losses, voltage shifts) when changing the tap, these are modelled in the grid and not in the
    action set."""


class HVDCRange(GridElement):
    """High voltage direct current lines can be set within the scope of non-costly optimization.

    An HVDC has a minimum and maximum power setpoint
    """

    min_power: float
    """The lowest power setpoint the HVDC supports"""

    max_power: float
    """The highest power setpoint the HVDC supports"""


class ActionSet(BaseModel):
    """A collection of actions in the form of asset topology stations.

    We make a convention that the beginning of the action set always includes substation
    local actions and the global actions are at the end.
    """

    starting_topology: Topology
    """How the grid looked like when the action set was first generated."""

    connectable_branches: list[GridElement]
    """A list of assets that can be connected as a remedial action."""

    disconnectable_branches: list[GridElement]
    """A list of assets that can be disconnected as a remedial action. Currently the DC solver supports only branches."""

    pst_ranges: list[PSTRange]
    """A list of phase shifting transformers that can be set as a remedial action."""

    hvdc_ranges: list[HVDCRange]
    """A list of high voltage direct current lines that can be set as a remedial action. This is currently not implemented
    yet in the solver."""

    local_actions: list[Station]
    """A list of split/reconfiguration actions that affect exactly one substation."""

    global_actions: list[list[Station]]
    """A list of split/reconfiguration actions that affect multiple substations. Each action contains a list of affected
    stations."""


def load_action_set_fs(filesystem: AbstractFileSystem, file_path: Union[str, Path]) -> ActionSet:
    """Load an action set from a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to load the action set.
    file_path : Union[str, Path]
        The path to the file containing the action set in json format.

    Returns
    -------
    ActionSet
        The action set loaded from the file.
    """
    with filesystem.open(str(file_path), "r") as f:
        return ActionSet.model_validate_json(f.read())


def load_action_set(file_path: Union[str, Path]) -> ActionSet:
    """Load an action set from a file.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the file containing the action set in json format.

    Returns
    -------
    ActionSet
        The action set loaded from the file.
    """
    return load_action_set_fs(LocalFileSystem(), file_path)


def save_action_set(file_path: Union[str, Path], action_set: ActionSet) -> None:
    """Save an action set to a file.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the file to save the action set to in json format.
    action_set : ActionSet
        The action set to save.

    """
    save_pydantic_model_fs(filesystem=LocalFileSystem(), file_path=file_path, pydantic_model=action_set)


def random_actions(action_set: ActionSet, rng: np.random.Generator, n_split_subs: int) -> list[int]:
    """Generate a random topology from the action set.

    Makes sure to sample each substation at most once.

    Parameters
    ----------
    action_set : ActionSet
        The action set to generate the random topology from.
    rng : np.random.Generator
        The random number generator to use.
    n_split_subs : int
        The number of substations to split. If this is more than total number of substations, all substations are split.
        (i.e. will be clipped to the number of substations)

    Returns
    -------
    list[int]
        A list of indices of the action set with substations to split.
    """
    # First sample the substations to split
    substations = list(set(station.grid_model_id for station in action_set.local_actions))
    sub_choice = rng.choice(substations, size=min(n_split_subs, len(substations)), replace=False).tolist()

    # Then sample an action for each substation
    actions = []
    for grid_model_id in sub_choice:
        applicable_indices = [
            i for i, station in enumerate(action_set.local_actions) if station.grid_model_id == grid_model_id
        ]
        actions.append(rng.choice(applicable_indices).item())
    return actions
