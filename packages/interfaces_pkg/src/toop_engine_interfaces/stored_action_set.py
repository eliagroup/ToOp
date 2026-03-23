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

import io
import itertools
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from beartype.typing import Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from jaxtyping import Bool
from pydantic import BaseModel, model_validator
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.nminus1_definition import GridElement


class PSTRange(GridElement):
    """Phase shifting transformers can be set within the scope of non-costly optimization.

    A PST has a list of taps, each with an angle shift.
    """

    starting_tap: int
    """The tap the PST was set to before optimization. To filter out actions that do not change anything in the
    UI, this is required."""

    low_tap: int
    """The lowest tap the PST supports"""

    high_tap: int
    """The highest tap the PST supports"""


class HVDCRange(GridElement):
    """High voltage direct current lines can be set within the scope of non-costly optimization.

    An HVDC has a minimum and maximum power setpoint
    """

    min_power: float
    """The lowest power setpoint the HVDC supports"""

    max_power: float
    """The highest power setpoint the HVDC supports"""


class ActionSet(BaseModel):
    """A collection of actions available to the optimizer in readable form.

    All actions are also stored directly in jax, but without IDs, names or other useful information to
    introspect them.
    """

    starting_topology: Topology
    """How the grid looked like when the action set was first generated. This does not include any
    asset topology simplifications but is just a copy of the importing result"""

    simplified_starting_topology: Topology
    """The starting topology in a preprocessed form. This does include simplifications made and is the basis for
    the local actions."""

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
    """A list of split/reconfiguration actions that affect exactly one substation. These are must be ordered by station,
    i.e. actions affecting the same station are next to each other. The grid_model_id of
    the station should be used to determine which substation it affects."""

    @model_validator(mode="after")
    def _validate_local_actions_grouped(self) -> "ActionSet":
        """Validate local actions are grouped by station."""
        validate_actions_grouped(self.local_actions)
        return self


@dataclass
class StationDiffArray:
    """A difference between copies of a station in the local action set and the starting topology.

    So that the action set does not have to store copies of the full station with all associated information, we only store
    the changes in the station that are typical for the actions in the action set, i.e. the switching table and coupler
    states. Furthermore, we store them in array form for the entire action set, so that we can potentially store them in
    parquet format.

    A full action set consists of station diffs for every switchable station in the grid.
    """

    grid_model_id: str
    """The grid model id of the station."""

    coupler_open: Bool[np.ndarray, " n_actions n_couplers"]
    """The state of the "open" field for every coupler in the station. The array dimension n_couplers is equivalent to
    station.couplers in length and order and the entries correspond to open (True) and closed (False). The n_actions
    dimension provides an entry per action in the action set."""

    switching_table: Bool[np.ndarray, " n_actions n_busbars n_assets"]
    """The switching table of the station. The array dimensions n_busbars and n_assets are equivalent to the
    switching table in the station, """


def validate_actions_grouped(actions: list[Station]) -> None:
    """Validate that actions are grouped by station grid model id.

    Parameters
    ----------
    actions : list[Station]
        Action stations to validate.

    Raises
    ------
    ValueError
        If a station grid model id appears in multiple non-contiguous groups.
    """
    seen_grid_model_ids: set[str] = set()
    last_grid_model_id: str | None = None
    for action in actions:
        grid_model_id = action.grid_model_id
        if grid_model_id != last_grid_model_id:
            if grid_model_id in seen_grid_model_ids:
                raise ValueError(
                    f"Actions are not grouped by station. Grid model id {grid_model_id} appears in multiple groups."
                )
            seen_grid_model_ids.add(grid_model_id)
            last_grid_model_id = grid_model_id


def _validate_station_diff_hypothesis(starting_station: Station, action: Station) -> None:
    """Validate that only coupler open states and switching table values differ.

    Parameters
    ----------
    starting_station : Station
        The reference station from the starting topology.
    action : Station
        The action station to validate.

    Raises
    ------
    ValueError
        If any field differs besides coupler open states and switching table values.
    """
    if action.grid_model_id != starting_station.grid_model_id:
        raise ValueError(
            f"Action station grid_model_id {action.grid_model_id} does not match starting station "
            f"{starting_station.grid_model_id}."
        )

    def normalize_station(station: Station) -> dict[str, object]:
        station_data = station.model_dump(mode="json")
        station_data.pop("asset_switching_table", None)
        for coupler in station_data.get("couplers", []):
            if isinstance(coupler, dict):
                coupler.pop("open", None)
        return station_data

    if normalize_station(action) != normalize_station(starting_station):
        raise ValueError(
            f"Action station {action.grid_model_id} changed fields other than coupler open states and asset switching table."
        )


def store_station_diff_fs(
    filesystem: AbstractFileSystem, station_diffs: list[StationDiffArray], diff_file_path: str | Path
) -> None:
    """Store a station diff to a hdf5 file, using a different group for every station

    Use load_station_diff_fs to load it again

    Parameters
    ----------
    filesystem : AbstractFileSystem
        A filesystem to store the station diffs in.
    station_diffs : list[StationDiffArray]
        A list of station diffs to store.
    diff_file_path : str | Path
        The file path to store the station diffs in.
    """
    filesystem.makedirs(Path(diff_file_path).parent.as_posix(), exist_ok=True)

    bytes_io = io.BytesIO()
    with h5py.File(bytes_io, mode="w") as file:
        for station_diff in station_diffs:
            group = file.create_group(station_diff.grid_model_id)
            group.create_dataset("coupler_open", data=station_diff.coupler_open)
            group.create_dataset("switching_table", data=station_diff.switching_table)
    bytes_io.seek(0)
    with filesystem.open(str(diff_file_path), "wb") as file:
        file.write(bytes_io.getbuffer())


def _load_station_diff_io(binaryio: io.IOBase) -> list[StationDiffArray]:
    """Load station diffs from a hdf5 file, using a different group for every station

    Use store_station_diff_io to store it.

    Parameters
    ----------
    binaryio : io.BufferedIOBase
        A binary IO to load the station diffs from.

    Returns
    -------
    list[StationDiffArray]
        A list of station diffs loaded from the file.
    """
    station_diffs = []
    with h5py.File(binaryio, mode="r") as file:
        for grid_model_id in file.keys():
            group = file[grid_model_id]
            coupler_open = group["coupler_open"][:]
            switching_table = group["switching_table"][:]
            station_diff = StationDiffArray(
                grid_model_id=grid_model_id, coupler_open=coupler_open, switching_table=switching_table
            )
            station_diffs.append(station_diff)
    return station_diffs


def load_station_diff_fs(filesystem: AbstractFileSystem, diff_file_path: str | Path) -> list[StationDiffArray]:
    """Load station diffs from a hdf5 file, using a different group for every station

    Use store_station_diff_io to store it.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        A filesystem to load the station diffs from.
    diff_file_path : str | Path
        The file path to load the station diffs from.

    Returns
    -------
    list[StationDiffArray]
        A list of station diffs loaded from the file.
    """
    with filesystem.open(str(diff_file_path), "rb") as file:
        file_bytes = file.read()
    buffer = io.BytesIO(file_bytes)
    return _load_station_diff_io(buffer)


def expand_single_station_diff_to_actions(starting_station: Station, station_diff: StationDiffArray) -> list[Station]:
    """Expand densely stored station diffs to a list of stations with the same format as in the action set.

    This only expands a single station diff, so it should be called once per station in the action set.

    Parameters
    ----------
    starting_station : Station
        The station as it looks in the starting topology. All fields from the station will be copied except for the
        coupler states and switching table, which will be overwritten by the station diff.
    station_diff : StationDiffArray
        The station diff to expand.

    Returns
    -------
    list[Station]
        A list of stations, each corresponding to an action in the station diffs action dimension.
    """
    actions = []
    for i in range(station_diff.coupler_open.shape[0]):
        coupler_array = station_diff.coupler_open[i]
        couplers = [
            coupler.model_copy(update={"open": bool(coupler_open)})
            for coupler, coupler_open in zip(starting_station.couplers, coupler_array, strict=True)
        ]
        switching_table = station_diff.switching_table[i]

        action = starting_station.model_copy(
            update={
                "couplers": couplers,
                "asset_switching_table": switching_table,
            },
        )
        actions.append(action)
    return actions


def expand_station_diffs(starting_topology: Topology, station_diffs: list[StationDiffArray]) -> list[Station]:
    """Expand densely stored station diffs to a list of stations with the same format as in the action set.

    This expands a list of station diffs, so it can be called once per action set.

    Parameters
    ----------
    starting_topology : Topology
        The topology as it looks in the starting topology. The station diffs will be matched to the stations in the topology
        based on their grid_model_id and all fields from the station will be copied except for the coupler states and
        switching table, which will be overwritten by the station diff.
    station_diffs : list[StationDiffArray]
        The station diffs to expand.

    Returns
    -------
    list[Station]
        A list of stations, each corresponding to an action in the station diffs action dimension.
    """
    grid_model_id_to_station = {station.grid_model_id: station for station in starting_topology.stations}
    actions = []
    for station_diff in station_diffs:
        starting_station = grid_model_id_to_station[station_diff.grid_model_id]
        actions.extend(expand_single_station_diff_to_actions(starting_station, station_diff))
    return actions


def compress_actions_to_station_diffs(
    starting_topology: Topology, actions: list[Station], validate_diff_hypothesis: bool = False
) -> list[StationDiffArray]:
    """Compress a list of stations with the same format as in the action set to densely stored station diffs.

    This compresses a list of stations, so it can be called once per action set.
    Note that this assumes
    - The list of actions is grouped by station, i.e. all actions for the same station are next to each other in the list.
    - The change between actions for the same station only regards the coupler states and switching table. If
      validate_diff_hypothesis is True, then this will be checked and it will raise a Value Error


    Parameters
    ----------
    starting_topology : Topology
        The topology as it looks in the starting topology. The stations will be matched to the stations in the topology
        based on their grid_model_id and the coupler states and switching table will be compared to the ones in the topology
        to create the station diffs.
        Note that this should be the simplified starting topology if simplifications have been applies, as they will also be
        present in all stations in the action set.
    actions : list[Station]
        A list of stations, each corresponding to an action in the station diffs action dimension.
    validate_diff_hypothesis : bool
        Whether to validate the hypothesis that the change between actions for the same station only regards the coupler
        states and switching table. If True, this will check the actions and raise a Value Error if this is not the case.
        Note that this will make the compression significantly slower, so it should only be used for debugging purposes.

    Returns
    -------
    list[StationDiffArray]
        The station diffs corresponding to the actions.

    Raises
    ------
    ValueError
        If the actions are not grouped by station
    ValueError
        If validate_diff_hypothesis is True and the change between actions for the same station regards fields other than the
        coupler states and switching table.
    """
    grid_model_id_to_station = {station.grid_model_id: station for station in starting_topology.stations}
    station_diffs = {}
    for grid_model_id, group in itertools.groupby(actions, key=lambda action: action.grid_model_id):
        if grid_model_id not in grid_model_id_to_station:
            raise ValueError(f"Action station grid_model_id {grid_model_id} not found in starting topology.")
        starting_station = grid_model_id_to_station[grid_model_id]

        coupler_open = []
        switching_table = []
        for action in group:
            assert len(action.couplers) == len(starting_station.couplers), (
                "Number of couplers in action station does not match starting station."
            )
            assert action.asset_switching_table.shape == starting_station.asset_switching_table.shape, (
                "Switching table shape in action station does not match starting station."
            )
            if validate_diff_hypothesis:
                _validate_station_diff_hypothesis(starting_station=starting_station, action=action)
            coupler_open.append([coupler.open for coupler in action.couplers])
            switching_table.append(action.asset_switching_table)
        coupler_open_array = np.array(coupler_open).astype(bool)
        switching_table_array = np.array(switching_table).astype(bool)
        station_diff = StationDiffArray(
            grid_model_id=grid_model_id, coupler_open=coupler_open_array, switching_table=switching_table_array
        )
        if station_diff.grid_model_id in station_diffs:
            raise ValueError(f"Duplicate station diff for grid_model_id {grid_model_id}, actions were not in order.")
        station_diffs[grid_model_id] = station_diff
    return list(station_diffs.values())


def load_action_set_fs(
    filesystem: AbstractFileSystem, json_file_path: Union[str, Path], diff_file_path: Union[str, Path] | None
) -> ActionSet:
    """Load an action set from a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to load the action set.
    json_file_path : Union[str, Path]
        The path to the JSON file containing the action set without local actions.
    diff_file_path : Union[str, Path] | None
        The path to the HDF5 file containing the station diffs to expand to local actions. If this is none, the
        local_actions field will not be filled and be the empty list.

    Returns
    -------
    ActionSet
        The action set loaded from the file.
    """
    with filesystem.open(str(json_file_path), "r") as f:
        action_set = ActionSet.model_validate_json(f.read())
    if diff_file_path is not None:
        station_diffs = load_station_diff_fs(filesystem, diff_file_path)
        local_actions = expand_station_diffs(
            starting_topology=action_set.simplified_starting_topology, station_diffs=station_diffs
        )
        action_set = action_set.model_copy(update={"local_actions": local_actions})
    return action_set


def load_action_set(json_file_path: Union[str, Path], diff_file_path: Union[str, Path] | None) -> ActionSet:
    """Load an action set from a file.

    Parameters
    ----------
    json_file_path : Union[str, Path]
        The path to the JSON file containing the action set without local actions.
    diff_file_path : Union[str, Path] | None
        The path to the HDF5 file containing the station diffs to expand to local actions. If this is none, the
        local_actions field will not be filled and be the empty list.

    Returns
    -------
    ActionSet
        The action set loaded from the file.
    """
    return load_action_set_fs(LocalFileSystem(), json_file_path=json_file_path, diff_file_path=diff_file_path)


def save_action_set_fs(
    filesystem: AbstractFileSystem,
    json_file_path: Union[str, Path],
    diff_file_path: Union[str, Path],
    action_set: ActionSet,
    validate_diff_hypothesis: bool = False,
) -> None:
    """Save an action set to a file system.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        The file system to use to save the action set.
    json_file_path : Union[str, Path]
        The path to the JSON file to save the pydantic payload.
    diff_file_path : Union[str, Path]
        The path to the HDF5 file to save the station diffs.
    action_set : ActionSet
        The action set to save.
    validate_diff_hypothesis : bool
        Whether to validate that local action changes only affect coupler open states and switching tables.
        This is intended for debugging and can make saving slower.
    """
    station_diffs = compress_actions_to_station_diffs(
        # Station diffs are computed against the simplified topology used to generate local actions.
        starting_topology=action_set.simplified_starting_topology,
        actions=action_set.local_actions,
        validate_diff_hypothesis=validate_diff_hypothesis,
    )

    # local_actions are persisted in the HDF5 file as compressed station diffs.
    action_set_without_local_actions = action_set.model_copy(update={"local_actions": []})
    save_pydantic_model_fs(
        filesystem=filesystem, file_path=str(json_file_path), pydantic_model=action_set_without_local_actions
    )
    store_station_diff_fs(filesystem, station_diffs, diff_file_path)


def save_action_set(
    json_file_path: Union[str, Path],
    diff_file_path: Union[str, Path],
    action_set: ActionSet,
    validate_diff_hypothesis: bool = False,
) -> None:
    """Save an action set to a file.

    Parameters
    ----------
    json_file_path : Union[str, Path]
        The path to the JSON file to save the pydantic payload.
    diff_file_path : Union[str, Path]
        The path to the HDF5 file to save the station diffs.
    action_set : ActionSet
        The action set to save.
    validate_diff_hypothesis : bool
        Whether to validate that local action changes only affect coupler open states and switching tables.
        This is intended for debugging and can make saving slower.

    """
    save_action_set_fs(
        filesystem=LocalFileSystem(),
        json_file_path=json_file_path,
        diff_file_path=diff_file_path,
        action_set=action_set,
        validate_diff_hypothesis=validate_diff_hypothesis,
    )


def random_actions(action_set: ActionSet, rng: np.random.Generator, n_split_subs: int) -> list[int]:
    """Sample a random topology from the action set.

    Makes sure to sample each substation at most once.

    Parameters
    ----------
    action_set : ActionSet
        The action set to sample the random topology from.
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
    substations.sort()  # Sort to make sure the order is deterministic for the same random seed
    sub_choice = rng.choice(substations, size=min(n_split_subs, len(substations)), replace=False).tolist()

    # Then sample an action for each substation
    actions = []
    for grid_model_id in sub_choice:
        applicable_indices = [
            i for i, station in enumerate(action_set.local_actions) if station.grid_model_id == grid_model_id
        ]
        actions.append(rng.choice(applicable_indices).item())
    return actions
