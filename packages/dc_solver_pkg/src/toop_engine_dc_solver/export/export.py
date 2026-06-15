# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""High-level node-breaker update export entrypoints.

The main function to use is ``get_node_breaker_updates_from_action_set``, which translates directly from the stored
action set representation to the low-level node-breaker update format. For more fine-grained control, the underlying
functions can be used to translate from target topologies or lists of changed stations and disconnections.
"""

import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import cast
from toop_engine_dc_solver.export.disconnection_switch_updates import get_changing_switches_from_disconnections
from toop_engine_dc_solver.export.station_switch_updates import (
    get_changing_switches_from_changed_stations,
)
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.node_breaker_update import (
    NodeBreakerUpdate,
    PSTUpdateSchema,
    SwitchUpdateSchema,
)
from toop_engine_interfaces.stored_action_set import ActionSet

logger = structlog.get_logger(__name__)


def _get_changed_stations_from_action_indices(
    action_set: ActionSet,
    actions: list[int],
) -> list[Station]:
    """Resolve action indices to concrete changed stations.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing local actions.
    actions : list[int]
        Requested indices into ``action_set.local_actions``.

    Returns
    -------
    list[Station]
        Concrete changed stations corresponding to ``actions``.

    Raises
    ------
    ValueError
        If any action index is negative or beyond the available range.
    """
    changed_stations: list[Station] = []
    for action_index in actions:
        if action_index < 0 or action_index >= len(action_set.local_actions):
            raise ValueError(f"Action index {action_index} is out of bounds for the action set")
        changed_stations.append(action_set.local_actions[action_index])
    return changed_stations


def _get_disconnections_from_indices(
    action_set: ActionSet,
    disconnections: list[int] | None,
) -> list[GridElement]:
    """Resolve disconnection indices to concrete grid elements.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing disconnectable branches.
    disconnections : list[int] | None
        Requested indices into ``action_set.disconnectable_branches``.

    Returns
    -------
    list[GridElement]
        Concrete disconnected branches corresponding to ``disconnections``.

    Raises
    ------
    ValueError
        If any disconnection index is negative or beyond the available range.
    """
    disconnected_branches: list[GridElement] = []
    for disconnection_index in [] if disconnections is None else disconnections:
        if disconnection_index < 0 or disconnection_index >= len(action_set.disconnectable_branches):
            raise ValueError(f"Disconnection index {disconnection_index} is out of bounds for the action set")
        disconnected_branches.append(action_set.disconnectable_branches[disconnection_index])
    return disconnected_branches


@pa.check_types
def get_pst_updates_from_setpoints(
    action_set: ActionSet,
    pst_setpoints: list[int] | None,
) -> pat.DataFrame[PSTUpdateSchema]:
    """Get PST updates from absolute setpoints in stored ActionSet order.

    This is the PST-only export entrypoint for consumers that already know the requested
    absolute taps and only need the raw node-breaker PST update table.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing controllable PST ranges.
    pst_setpoints : list[int] | None
        Absolute PST tap positions in ``action_set.pst_ranges`` order, matching the
        convention accepted by the postprocess runner. If ``None`` or empty, an empty
        PST update table is returned.

    Returns
    -------
    pat.DataFrame[PSTUpdateSchema]
        PST updates corresponding to ``pst_setpoints``.

    Raises
    ------
    ValueError
        If the number of PST setpoints does not match the number of controllable PSTs.
    """
    if pst_setpoints is None or len(pst_setpoints) == 0:
        return cast(pat.DataFrame[PSTUpdateSchema], get_empty_dataframe_from_model(PSTUpdateSchema))

    if len(pst_setpoints) != len(action_set.pst_ranges):
        raise ValueError(
            "Number of PST setpoints must match number of controllable PSTs, "
            f"got {pst_setpoints} setpoints for {[pst.id for pst in action_set.pst_ranges]} PSTs"
        )

    for pst_range, pst_setpoint in zip(action_set.pst_ranges, pst_setpoints, strict=True):
        if pst_setpoint < pst_range.low_tap or pst_setpoint > pst_range.high_tap:
            raise ValueError(
                "PST setpoint out of tap range: "
                f"{pst_setpoint} for {pst_range.id}, expected between {pst_range.low_tap} and {pst_range.high_tap}"
            )

    pst_updates = pd.DataFrame(
        {
            "grid_model_id": [pst.id for pst in action_set.pst_ranges],
            "tap": list(pst_setpoints),
        }
    ).astype({"grid_model_id": str, "tap": int})
    return cast(pat.DataFrame[PSTUpdateSchema], pst_updates)


@pa.check_types
def get_changing_switches_from_actions(
    changed_stations: list[Station],
    simplified_starting_topology: Topology,
    disconnections: list[GridElement] | None = None,
    full_starting_topology: Topology | None = None,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates for changed stations and explicit disconnections.

    This is the composed export entrypoint for switch updates derived from the simplified action
    representation. Changed stations contribute action-driven switch updates, while explicit
    disconnections are translated into breaker-open updates when they can be represented through
    switchable stations in the provided topology.

    Parameters
    ----------
    changed_stations : list[Station]
        Stations describing the target state for switchable substations.
    simplified_starting_topology : Topology
        Simplified starting topology used as reference for station ordering and switch layout.
        This should be action_set.simplified_starting_topology which has the same amount of
        assets and stations as the changed stations from the action set.
    disconnections : list[GridElement] | None, optional
        Explicit branch disconnections requested for the target state.
    full_starting_topology : Topology | None, optional
        Full starting topology with all assets and stations detectable by the importing routine.
        This is used to map out disconnections as disconnections can not be performed if the branch
        is not in the simplified topology. Note that even with the full starting topology, disconnections
        might be missed.
        Use action_set.starting_topology for the unfiltered version.
        If none, this falls back to use the simplified starting topology.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing both station actions and representable disconnections.
    """
    if full_starting_topology is None:
        full_starting_topology = simplified_starting_topology
    action_switch_updates = get_changing_switches_from_changed_stations(
        changed_stations=changed_stations,
        starting_topology=simplified_starting_topology,
    )
    if disconnections and len(disconnections) > 0:
        disconnection_switch_updates = get_changing_switches_from_disconnections(
            starting_topology=full_starting_topology,
            disconnections=disconnections,
        )
    else:
        disconnection_switch_updates = get_empty_dataframe_from_model(SwitchUpdateSchema)

    overlapping_switch_ids = sorted(
        set(action_switch_updates["grid_model_id"]).intersection(disconnection_switch_updates["grid_model_id"])
    )
    if overlapping_switch_ids:
        logger.warning(
            "Action and disconnection switch updates overlap. Disconnection updates take precedence.",
            overlapping_switch_ids=overlapping_switch_ids,
        )
        action_switch_updates = action_switch_updates.loc[
            ~action_switch_updates["grid_model_id"].isin(overlapping_switch_ids)
        ]

    combined_switch_updates = pd.concat(
        [action_switch_updates, disconnection_switch_updates],
        ignore_index=True,
    )
    if combined_switch_updates.empty:
        combined_switch_updates = get_empty_dataframe_from_model(SwitchUpdateSchema)
    combined_switch_updates = combined_switch_updates.astype({"grid_model_id": str, "open": bool})
    return cast(pat.DataFrame[SwitchUpdateSchema], combined_switch_updates)


@pa.check_types
def get_node_breaker_updates_from_action_set(
    action_set: ActionSet,
    actions: list[int],
    disconnections: list[int] | None = None,
    pst_setpoints: list[int] | None = None,
) -> NodeBreakerUpdate:
    """Get raw node-breaker updates from stored ActionSet indices.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing the simplified starting topology, local actions,
        disconnectable branches, and controllable PST ranges.
    actions : list[int]
        Indices into ``action_set.local_actions`` describing the selected station actions.
    disconnections : list[int] | None, optional
        Indices into ``action_set.disconnectable_branches`` describing explicit branch
        disconnections.
    pst_setpoints : list[int] | None, optional
        Absolute PST tap positions in ``action_set.pst_ranges`` order, matching the
        convention accepted by the postprocess runner.

    Returns
    -------
    NodeBreakerUpdate
        Low-level switch and PST updates representing the requested actions.

    Raises
    ------
    ValueError
        If any action or disconnection index is out of bounds, or the PST setpoint count
        does not match the controllable PST count.
    """
    changed_stations = _get_changed_stations_from_action_indices(action_set=action_set, actions=actions)
    disconnected_branches = _get_disconnections_from_indices(action_set=action_set, disconnections=disconnections)
    switch_updates = get_changing_switches_from_actions(
        changed_stations=changed_stations,
        simplified_starting_topology=action_set.simplified_starting_topology,
        disconnections=disconnected_branches,
        full_starting_topology=action_set.starting_topology,
    )
    pst_updates = get_pst_updates_from_setpoints(action_set=action_set, pst_setpoints=pst_setpoints)
    return NodeBreakerUpdate(
        switch_updates=switch_updates,
        pst_updates=pst_updates,
    )
