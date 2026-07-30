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
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedStation
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

    starting_stations: list[MaterializedStation] | None = None
    """Runtime-aware station snapshots for the starting grid state.

    When present, these are the first-class station references for consumers that need realized
    station payloads.
    """

    simplified_starting_stations: list[MaterializedStation] | None = None
    """Runtime-aware station snapshots for the simplified starting grid state.

    These station snapshots define the station and asset ordering contract for ``local_actions``.
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

    local_actions: list[MaterializedStation]
    """A list of split/reconfiguration actions that affect exactly one substation. These are must be ordered by station,
    i.e. actions affecting the same station are next to each other. The grid_model_id of
    the station should be used to determine which substation it affects. Within a station, asset
    ordering matches the corresponding station in ``simplified_starting_stations``."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_station_references(cls, data: object) -> object:
        """Validate and normalize runtime station references."""
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload["starting_stations"] = _coerce_reference_stations(payload.get("starting_stations"))
        payload["simplified_starting_stations"] = _coerce_reference_stations(payload.get("simplified_starting_stations"))
        if payload.get("starting_stations") is not None:
            _validate_unique_reference_stations(payload["starting_stations"])
        if payload.get("simplified_starting_stations") is not None:
            _validate_unique_reference_stations(payload["simplified_starting_stations"])
        return payload

    @model_validator(mode="after")
    def _validate_action_grouping(self) -> "ActionSet":
        """Validate local action grouping after reference normalization."""
        validate_actions_grouped(self.local_actions)
        return self

    def get_starting_stations(self) -> list[MaterializedStation]:
        """Return normalized runtime-aware station snapshots for the starting topology."""
        return _require_reference_stations(self.starting_stations, context="starting topology")

    def get_simplified_starting_stations(self) -> list[MaterializedStation]:
        """Return normalized runtime-aware station snapshots for the simplified starting topology."""
        return _require_reference_stations(self.simplified_starting_stations, context="simplified starting topology")


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


def validate_actions_grouped(actions: list[MaterializedStation]) -> None:
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
        grid_model_id = action.bus_group_id
        if grid_model_id != last_grid_model_id:
            if grid_model_id in seen_grid_model_ids:
                raise ValueError(
                    f"Actions are not grouped by station. Grid model id {grid_model_id} appears in multiple groups."
                )
            seen_grid_model_ids.add(grid_model_id)
            last_grid_model_id = grid_model_id


def _require_reference_stations(
    reference_stations: list[MaterializedStation] | None,
    *,
    context: str,
) -> list[MaterializedStation]:
    """Require explicit runtime reference stations.

    Parameters
    ----------
    reference_stations : list[MaterializedStation] | None
        Runtime-aware station snapshots.
    context : str
        Human-readable description of the caller context.

    Returns
    -------
    list[MaterializedStation]
        Validated reference stations.

    Raises
    ------
    ValueError
        If explicit reference stations are missing or contain duplicates.
    """
    if reference_stations is None:
        raise ValueError(f"ActionSet requires explicit reference stations for {context}.")

    _validate_unique_reference_stations(reference_stations)
    return reference_stations


def _validate_unique_reference_stations(reference_stations: list[MaterializedStation]) -> None:
    """Validate that reference stations are unique by station id."""
    seen_station_ids: set[str] = set()
    for station in reference_stations:
        if station.bus_group_id in seen_station_ids:
            raise ValueError(f"Reference stations must be unique by station id, got duplicate {station.bus_group_id}.")
        seen_station_ids.add(station.bus_group_id)


def _coerce_reference_stations(reference_stations: object) -> list[MaterializedStation] | None:
    """Coerce reference stations to validated materialized-station models when present."""
    if reference_stations is None:
        return None
    return [
        station if isinstance(station, MaterializedStation) else MaterializedStation.model_validate(station)
        for station in reference_stations
    ]


def _validate_station_diff_hypothesis(starting_station: MaterializedStation, action: MaterializedStation) -> None:
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
    if action.bus_group_id != starting_station.bus_group_id:
        raise ValueError(
            f"Action station id {action.bus_group_id} does not match starting station {starting_station.bus_group_id}."
        )

    def normalize_station(station: MaterializedStation) -> dict[str, object]:
        station_data = station.model_dump(mode="json")
        station_data.pop("branch_switching_table", None)
        station_data.pop("injection_switching_table", None)
        for coupler in station_data.get("couplers", []):
            if isinstance(coupler, dict):
                coupler.pop("open", None)
        return station_data

    if normalize_station(action) != normalize_station(starting_station):
        raise ValueError(
            f"Action station {action.bus_group_id} changed fields other than coupler open states and switching tables."
        )


def _construct_action_from_station_diff(
    starting_station: MaterializedStation,
    couplers: list,
    branch_switching_table: np.ndarray,
    injection_switching_table: np.ndarray,
) -> MaterializedStation:
    """Construct an expanded action station from a validated reference station and diff payload."""
    return MaterializedStation.model_construct(
        bus_group_id=starting_station.bus_group_id,
        voltage_level_id=starting_station.voltage_level_id,
        name=starting_station.name,
        station_type=starting_station.station_type,
        region=starting_station.region,
        voltage_level=starting_station.voltage_level,
        busbars=starting_station.busbars,
        bus_branch_bus_ids=starting_station.bus_branch_bus_ids,
        couplers=couplers,
        branch_connections=starting_station.branch_connections,
        injection_connections=starting_station.injection_connections,
        branch_switching_table=branch_switching_table,
        injection_switching_table=injection_switching_table,
        branch_connectivity=starting_station.branch_connectivity,
        injection_connectivity=starting_station.injection_connectivity,
        model_log=starting_station.model_log,
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
    starting_station: MaterializedStation, station_diff: StationDiffArray
) -> list[MaterializedStation]:
    """Expand densely stored station diffs to a list of stations with the same format as in the action set.

    This only expands a single station diff, so it should be called once per station in the action set.

    Parameters
    ----------
    starting_station : Station
        The station as it looks in the starting topology. All fields from the station will be copied except for the
        coupler states and switching tables, which will be overwritten by the station diff.
    station_diff : StationDiffArray
        The station diff to expand.

    Returns
    -------
    list[Station]
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
                for coupler, coupler_open in zip(starting_station.couplers, coupler_state_key, strict=True)
            ]
            coupler_state_cache[coupler_state_key] = couplers

        branch_switching_table = station_diff.branch_switching_table[i]
        injection_switching_table = station_diff.injection_switching_table[i]

        action = _construct_action_from_station_diff(
            starting_station=starting_station,
            couplers=couplers,
            branch_switching_table=branch_switching_table,
            injection_switching_table=injection_switching_table,
        )
        actions.append(action)
    return actions


def expand_station_diffs_from_starting_stations(
    starting_stations: list[MaterializedStation],
    station_diffs: list[StationDiffArray],
) -> list[MaterializedStation]:
    """Expand densely stored station diffs from reference materialized stations."""
    grid_model_id_to_station = {station.bus_group_id: station for station in starting_stations}
    actions = []
    for station_diff in station_diffs:
        starting_station = grid_model_id_to_station[station_diff.grid_model_id]
        actions.extend(expand_single_station_diff_to_actions(starting_station, station_diff))
    return actions


def compress_actions_to_station_diffs_from_starting_stations(
    starting_stations: list[MaterializedStation],
    actions: list[MaterializedStation],
    validate_diff_hypothesis: bool = False,
) -> list[StationDiffArray]:
    """Compress action stations to station diffs using reference materialized stations."""
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
                _validate_station_diff_hypothesis(starting_station=starting_station, action=action)
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
        The action set loaded from the file.
    """
    with filesystem.open(str(json_file_path), "r") as f:
        payload = json.loads(f.read())
    action_set = ActionSet.model_validate(_drop_legacy_reference_master_data_fields(payload))
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


def _drop_legacy_reference_master_data_fields(payload: object) -> object:
    """Drop legacy master-data references from serialized ActionSet payloads."""
    if not isinstance(payload, dict):
        return payload
    sanitized_payload = dict(payload)
    sanitized_payload.pop("starting_master_data", None)
    sanitized_payload.pop("simplified_starting_master_data", None)
    return sanitized_payload


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
