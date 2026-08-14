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
import json
from pathlib import Path

import h5py
import numpy as np
from beartype.typing import Union
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict, model_validator
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeBusGroup
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import SimplifiedBusGroup
from toop_engine_interfaces.nminus1_definition import GridElement

STATION_DIFF_ORDER_ATTR = "station_order"


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

    pst_group: str | None = None
    """The optimization group of the PST.

    When omitted in serialized action sets, this defaults to the PST id for backward compatibility.
    """

    @model_validator(mode="after")
    def _default_pst_group(self) -> "PSTRange":
        """Default missing group ids to the PST id for backward compatibility."""
        if self.pst_group is None:
            self.pst_group = str(self.id)
        return self


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

    model_config = ConfigDict(extra="forbid")

    starting_stations: list[RuntimeBusGroup]
    """Runtime-aware station snapshots for the starting grid state.

    When present, these are the first-class station references for consumers that need realized
    station payloads. We store runtime stations here instead of master data because postprocessing,
    switch-distance reconstruction, and diff expansion need the as-is switching state and current
    station-local asset ordering.
    """

    simplified_starting_stations: list[SimplifiedBusGroup]
    """Runtime-aware station snapshots for the simplified starting grid state.

    These station snapshots define the station and asset ordering contract for ``local_actions``.
    They are still runtime snapshots, but projected to the reduced DC-solver asset view rather than
    the full physical station view.
    """

    connectable_branches: list[GridElement]
    """A list of assets that can be connected as a remedial action."""

    disconnectable_branches: list[GridElement]
    """A list of assets that can be disconnected as a remedial action. Currently the DC solver supports only branches."""

    pst_ranges: list[PSTRange]
    """A list of phase shifting transformers that can be set as a remedial action."""

    hvdc_ranges: list[HVDCRange]
    """A list of high voltage direct current lines that can be set as a remedial action. This is currently not implemented
    yet in the solver."""

    local_actions: list[SimplifiedBusGroup]
    """A list of split/reconfiguration actions that affect exactly one electrical bus. These are must be ordered by station,
    i.e. actions affecting the same station are next to each other. The grid_model_id of
    the station should be used to determine which substation it affects. Within a station, asset
    ordering matches the corresponding station in ``simplified_starting_stations``."""

    @model_validator(mode="after")
    def _validate_action_grouping(self) -> "ActionSet":
        """Validate reference-station uniqueness and local action grouping."""
        _validate_unique_reference_station_ids(self.starting_stations)
        _validate_unique_reference_station_ids(self.simplified_starting_stations)
        validate_actions_grouped(self.local_actions)
        return self

    def get_starting_stations(self) -> list[RuntimeBusGroup]:
        """Return normalized runtime-aware station snapshots for the starting topology."""
        return self.starting_stations

    def get_simplified_starting_stations(self) -> list[SimplifiedBusGroup]:
        """Return normalized runtime-aware station snapshots for the simplified starting topology."""
        return self.simplified_starting_stations


class StationDiffArray(BaseModel):
    """A difference between copies of a station in the local action set and the starting topology.

    So that the action set does not have to store copies of the full station with all associated information, we only store
    the changes in the station that are typical for the actions in the action set, i.e. the switching table and coupler
    states. Furthermore, we store them in array form for the entire action set, so that we can potentially store them in
    parquet format.

    A full action set consists of station diffs for every switchable station in the grid.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_model_id: str
    """The grid model id of the station."""

    coupler_open: Bool[np.ndarray, " _n_actions _n_couplers"]
    """The state of the "open" field for every coupler in the station. The array dimension n_couplers is equivalent to
    station.couplers in length and order and the entries correspond to open (True) and closed (False). The n_actions
    dimension provides an entry per action in the action set."""

    branch_switching_table: Bool[np.ndarray, " _n_actions _n_busbars _n_branch_assets"]
    """Branch switching tables for the station actions.

    The busbar and branch-asset dimensions match ``station.branch_switching_table``.
    """

    injection_switching_table: Bool[np.ndarray, " _n_actions _n_busbars _n_injection_assets"]
    """Injection switching tables for the station actions.

    The busbar and injection-asset dimensions match ``station.injection_switching_table``.
    """

    @model_validator(mode="after")
    def _validate_station_diff_arrays(self) -> "StationDiffArray":
        """Validate stored station diff array shapes.

        Different stations can legitimately have different action counts, so the relevant invariant is
        local to each station diff: coupler_open, branch_switching_table, and injection_switching_table
        must agree on their first dimension per station. The two switching tables must also agree on
        their busbar dimension. However, the beartype checker invokes the checks in such a way that a
        global instantiation of dimension values was happening, raising. Hence, we check the shapes
        manually here.
        """
        if self.coupler_open.ndim != 2:
            raise ValueError("coupler_open must be a 2D array of shape (n_actions, n_couplers)")
        if self.branch_switching_table.ndim != 3:
            raise ValueError("branch_switching_table must be a 3D array of shape (n_actions, n_busbars, n_branch_assets)")
        if self.injection_switching_table.ndim != 3:
            raise ValueError(
                "injection_switching_table must be a 3D array of shape (n_actions, n_busbars, n_injection_assets)"
            )
        if self.coupler_open.shape[0] != self.branch_switching_table.shape[0]:
            raise ValueError(
                "coupler_open and branch_switching_table must have the same n_actions dimension, got "
                f"{self.coupler_open.shape[0]} and {self.branch_switching_table.shape[0]}"
            )
        if self.coupler_open.shape[0] != self.injection_switching_table.shape[0]:
            raise ValueError(
                "coupler_open and injection_switching_table must have the same n_actions dimension, got "
                f"{self.coupler_open.shape[0]} and {self.injection_switching_table.shape[0]}"
            )
        if self.branch_switching_table.shape[1] != self.injection_switching_table.shape[1]:
            raise ValueError(
                "branch_switching_table and injection_switching_table must have the same n_busbars dimension, got "
                f"{self.branch_switching_table.shape[1]} and {self.injection_switching_table.shape[1]}"
            )
        return self


def validate_actions_grouped(actions: list[SimplifiedBusGroup]) -> None:
    """Validate that actions are grouped by station grid model id.

    Parameters
    ----------
    actions : list[SimplifiedBusGroup]
        Action stations to validate.

    Raises
    ------
    ValueError
        If a station grid model id appears in multiple non-contiguous groups.
    """
    seen_grid_model_ids: set[str] = set()
    last_grid_model_id: str | None = None
    for action in actions:
        grid_model_id = action.bus_group_id
        if grid_model_id != last_grid_model_id:
            if grid_model_id in seen_grid_model_ids:
                raise ValueError(
                    f"Actions are not grouped by station. Grid model id {grid_model_id} appears in multiple groups."
                )
            seen_grid_model_ids.add(grid_model_id)
            last_grid_model_id = grid_model_id


def _validate_unique_reference_station_ids(reference_stations: list[RuntimeBusGroup] | list[SimplifiedBusGroup]) -> None:
    """Validate that reference stations are unique by ``bus_group_id``."""
    seen_station_ids: set[str] = set()
    for station in reference_stations:
        if station.bus_group_id in seen_station_ids:
            raise ValueError(f"Reference stations must be unique by station id, got duplicate {station.bus_group_id}.")
        seen_station_ids.add(station.bus_group_id)


def _validate_station_diff_hypothesis(starting_busgroups: SimplifiedBusGroup, action: SimplifiedBusGroup) -> None:
    """Validate that only coupler open states and switching table values differ.

    Parameters
    ----------
    starting_busgroups : SimplifiedBusGroup
        The reference station from the starting topology.
    action : SimplifiedBusGroup
        The action station to validate.

    Raises
    ------
    ValueError
        If any field differs besides coupler open states and switching table values.
    """
    if action.bus_group_id != starting_busgroups.bus_group_id:
        raise ValueError(
            f"Action station id {action.bus_group_id} does not match starting station {starting_busgroups.bus_group_id}."
        )

    def normalize_station(station: SimplifiedBusGroup) -> dict[str, object]:
        station_data = station.model_dump(mode="json")
        station_data.pop("branch_switching_table", None)
        station_data.pop("injection_switching_table", None)
        for coupler in station_data.get("couplers", []):
            if isinstance(coupler, dict):
                coupler.pop("open", None)
        return station_data

    if normalize_station(action) != normalize_station(starting_busgroups):
        raise ValueError(
            f"Action station {action.bus_group_id} changed fields other than coupler open states and switching tables."
        )


def _construct_action_from_station_diff(
    starting_busgroup: SimplifiedBusGroup,
    couplers: list,
    branch_switching_table: np.ndarray,
    injection_switching_table: np.ndarray,
) -> SimplifiedBusGroup:
    """Construct one action station from a validated reference station and diff payload.

    The reference station contributes all static metadata and runtime payloads. Only coupler
    open states and switching tables are replaced from the diff representation.
    """
    return SimplifiedBusGroup.model_construct(
        bus_group_id=starting_busgroup.bus_group_id,
        voltage_level_id=starting_busgroup.voltage_level_id,
        name=starting_busgroup.name,
        station_type=starting_busgroup.station_type,
        region=starting_busgroup.region,
        voltage_level=starting_busgroup.voltage_level,
        busbars=starting_busgroup.busbars,
        bus_branch_bus_ids=starting_busgroup.bus_branch_bus_ids,
        couplers=couplers,
        branch_connections=starting_busgroup.branch_connections,
        injection_connections=starting_busgroup.injection_connections,
        branch_switching_table=branch_switching_table,
        injection_switching_table=injection_switching_table,
        branch_connectivity=starting_busgroup.branch_connectivity,
        injection_connectivity=starting_busgroup.injection_connectivity,
        model_log=starting_busgroup.model_log,
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
        file.attrs[STATION_DIFF_ORDER_ATTR] = np.array(
            [station_diff.grid_model_id for station_diff in station_diffs],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        for station_diff in station_diffs:
            group = file.create_group(station_diff.grid_model_id)
            group.create_dataset("coupler_open", data=station_diff.coupler_open)
            group.create_dataset("branch_switching_table", data=station_diff.branch_switching_table)
            group.create_dataset("injection_switching_table", data=station_diff.injection_switching_table)
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
        if STATION_DIFF_ORDER_ATTR in file.attrs:
            station_order = [
                grid_model_id.decode("utf-8") if isinstance(grid_model_id, bytes) else str(grid_model_id)
                for grid_model_id in file.attrs[STATION_DIFF_ORDER_ATTR]
            ]
        else:
            station_order = list(file.keys())

        for grid_model_id in station_order:
            group = file[grid_model_id]
            coupler_open = group["coupler_open"][:]
            branch_switching_table = group["branch_switching_table"][:]
            injection_switching_table = group["injection_switching_table"][:]
            station_diff = StationDiffArray(
                grid_model_id=grid_model_id,
                coupler_open=coupler_open,
                branch_switching_table=branch_switching_table,
                injection_switching_table=injection_switching_table,
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


def expand_single_station_diff_to_actions(
    starting_busgroup: SimplifiedBusGroup, station_diff: StationDiffArray
) -> list[SimplifiedBusGroup]:
    """Expand densely stored station diffs to a list of stations with the same format as in the action set.

    This only expands a single station diff, so it should be called once per station in the action set.

    Parameters
    ----------
    starting_busgroup : SimplifiedBusGroup
        The station as it looks in the starting topology. All fields from the busgroup will be copied except for the
        coupler states and switching tables, which will be overwritten by the station diff.
    station_diff : StationDiffArray
        The station diff to expand.

    Returns
    -------
    list[SimplifiedBusGroup]
        A list of stations, each corresponding to an action in the station diffs action dimension.
    """
    actions = []
    coupler_state_cache: dict[tuple[bool, ...], list] = {}
    for i in range(station_diff.coupler_open.shape[0]):
        coupler_state_key = tuple(bool(coupler_open) for coupler_open in station_diff.coupler_open[i])
        couplers = coupler_state_cache.get(coupler_state_key)
        if couplers is None:
            couplers = [
                coupler.model_copy(update={"open": coupler_open}, deep=False)
                for coupler, coupler_open in zip(starting_busgroup.couplers, coupler_state_key, strict=True)
            ]
            coupler_state_cache[coupler_state_key] = couplers

        branch_switching_table = station_diff.branch_switching_table[i]
        injection_switching_table = station_diff.injection_switching_table[i]

        action = _construct_action_from_station_diff(
            starting_busgroup=starting_busgroup,
            couplers=couplers,
            branch_switching_table=branch_switching_table,
            injection_switching_table=injection_switching_table,
        )
        actions.append(action)
    return actions


def expand_station_diffs_from_starting_stations(
    starting_stations: list[SimplifiedBusGroup],
    station_diffs: list[StationDiffArray],
) -> list[SimplifiedBusGroup]:
    """Expand densely stored station diffs from reference runtime stations."""
    grid_model_id_to_station = {station.bus_group_id: station for station in starting_stations}
    actions = []
    for station_diff in station_diffs:
        starting_station = grid_model_id_to_station[station_diff.grid_model_id]
        actions.extend(expand_single_station_diff_to_actions(starting_station, station_diff))
    return actions


def compress_actions_to_station_diffs_from_starting_stations(
    starting_stations: list[SimplifiedBusGroup],
    actions: list[SimplifiedBusGroup],
    validate_diff_hypothesis: bool = False,
) -> list[StationDiffArray]:
    """Compress action stations to station diffs using reference runtime stations.

    This is the inverse of ``expand_station_diffs_from_starting_stations`` and keeps only the
    state that actually varies across local actions: coupler openness and the two switching tables.
    """
    grid_model_id_to_station = {station.bus_group_id: station for station in starting_stations}
    station_diffs = {}
    for grid_model_id, group in itertools.groupby(actions, key=lambda action: action.bus_group_id):
        if grid_model_id not in grid_model_id_to_station:
            raise ValueError(f"Action station id {grid_model_id} not found in starting topology.")
        starting_station = grid_model_id_to_station[grid_model_id]

        coupler_open = []
        branch_switching_tables = []
        injection_switching_tables = []
        for action in group:
            assert len(action.couplers) == len(starting_station.couplers), (
                "Number of couplers in action station does not match starting station."
            )
            assert action.branch_switching_table.shape == starting_station.branch_switching_table.shape, (
                "Branch switching table shape in action station does not match starting station."
            )
            assert action.injection_switching_table.shape == starting_station.injection_switching_table.shape, (
                "Injection switching table shape in action station does not match starting station."
            )
            if validate_diff_hypothesis:
                _validate_station_diff_hypothesis(starting_busgroups=starting_station, action=action)
            coupler_open.append([coupler.open for coupler in action.couplers])
            branch_switching_tables.append(action.branch_switching_table)
            injection_switching_tables.append(action.injection_switching_table)
        coupler_open_array = np.array(coupler_open).astype(bool)
        branch_switching_table_array = np.array(branch_switching_tables).astype(bool)
        injection_switching_table_array = np.array(injection_switching_tables).astype(bool)
        station_diff = StationDiffArray(
            grid_model_id=grid_model_id,
            coupler_open=coupler_open_array,
            branch_switching_table=branch_switching_table_array,
            injection_switching_table=injection_switching_table_array,
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
        The action set loaded from the file. When ``diff_file_path`` is given, ``local_actions``
        are reconstructed from the stored station diffs and ``simplified_starting_stations``.
    """
    with filesystem.open(str(json_file_path), "r") as f:
        payload = json.loads(f.read())
    action_set = ActionSet.model_validate(payload)
    if diff_file_path is not None:
        station_diffs = load_station_diff_fs(filesystem, diff_file_path)
        local_actions = expand_station_diffs_from_starting_stations(
            starting_stations=action_set.get_simplified_starting_stations(),
            station_diffs=station_diffs,
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
    revalidate_action_set: bool = True,
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
    revalidate_action_set : bool
        Whether to round-trip the action set through Pydantic validation before saving.
        Disable this in hot paths when the caller already constructed a validated ``ActionSet``.

    Notes
    -----
    The JSON payload stores only the reference stations and scalar metadata. ``local_actions`` are
    serialized separately as dense station diffs in HDF5 to avoid repeating unchanged station payloads.
    """
    if revalidate_action_set:
        action_set = ActionSet.model_validate(action_set.model_dump(mode="python", round_trip=True))
    station_diffs = compress_actions_to_station_diffs_from_starting_stations(
        starting_stations=action_set.get_simplified_starting_stations(),
        actions=action_set.local_actions,
        validate_diff_hypothesis=validate_diff_hypothesis,
    )

    # local_actions are persisted in the HDF5 file as compressed station diffs.
    action_set_without_local_actions = action_set.model_copy(update={"local_actions": []})
    with filesystem.open(str(json_file_path), "w") as f:
        f.write(action_set_without_local_actions.model_dump_json(indent=2, exclude_none=True))
    store_station_diff_fs(filesystem, station_diffs, diff_file_path)


def save_action_set(
    json_file_path: Union[str, Path],
    diff_file_path: Union[str, Path],
    action_set: ActionSet,
    validate_diff_hypothesis: bool = False,
    revalidate_action_set: bool = True,
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
    revalidate_action_set : bool
        Whether to round-trip the action set through Pydantic validation before saving.

    """
    save_action_set_fs(
        filesystem=LocalFileSystem(),
        json_file_path=json_file_path,
        diff_file_path=diff_file_path,
        action_set=action_set,
        validate_diff_hypothesis=validate_diff_hypothesis,
        revalidate_action_set=revalidate_action_set,
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
    substations = list(set(station.bus_group_id for station in action_set.local_actions))
    substations.sort()  # Sort to make sure the order is deterministic for the same random seed
    sub_choice = rng.choice(substations, size=min(n_split_subs, len(substations)), replace=False).tolist()

    # Then sample an action for each substation
    actions = []
    for grid_model_id in sub_choice:
        applicable_indices = [
            i for i, station in enumerate(action_set.local_actions) if station.bus_group_id == grid_model_id
        ]
        actions.append(rng.choice(applicable_indices).item())
    return actions
