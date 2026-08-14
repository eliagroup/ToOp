# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""High-level switch update export entrypoints.

The main function to use is ``get_changing_switches_from_action_set``, which translates directly from the stored action set
representation to the switch update schema format. For more fine-grained control, the underlying functions can be used to
translate from target topologies or lists of changed stations and disconnections.
"""

import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from toop_engine_dc_solver.export.disconnection_switch_updates import (
    get_changing_switches_from_disconnections,
)
from toop_engine_dc_solver.export.station_switch_updates import (
    get_changing_switches_from_changed_bus_groups,
)
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeBusGroup
from toop_engine_interfaces.asset_topology.simplified_runtime_topology import SimplifiedBusGroup
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_interfaces.switch_update_schema import SwitchUpdateSchema

logger = structlog.get_logger(__name__)


def _get_changed_bus_groups_from_action_indices(
    action_set: ActionSet,
    actions: list[int],
) -> list[SimplifiedBusGroup]:
    """Resolve action indices to concrete changed bus groups.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing local actions.
    actions : list[int]
        Requested indices into ``action_set.local_actions``.

    Returns
    -------
    list[SimplifiedBusGroup]
        Concrete changed bus groups corresponding to ``actions``.

    Raises
    ------
    ValueError
        If any action index is negative or beyond the available range.
    """
    changed_bus_groups: list[SimplifiedBusGroup] = []
    for action_index in actions:
        if action_index < 0 or action_index >= len(action_set.local_actions):
            raise ValueError(f"Action index {action_index} is out of bounds for the action set")
        changed_bus_groups.append(action_set.local_actions[action_index])
    return changed_bus_groups


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
def get_changing_switches_from_actions(
    changed_bus_groups: list[SimplifiedBusGroup],
    simplified_starting_bus_groups: list[SimplifiedBusGroup],
    disconnections: list[GridElement] | None = None,
    full_starting_bus_groups: list[RuntimeBusGroup] | None = None,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates for changed bus groups and explicit disconnections.

    This is the composed export entrypoint for switch updates derived from the simplified action
    representation. Changed bus groups contribute action-driven switch updates, while explicit
    disconnections are translated into breaker-open updates when they can be represented through
    switchable bus groups in the provided topology.

    Parameters
    ----------
    changed_bus_groups : list[SimplifiedBusGroup]
        Bus groups describing the target state for switchable substations.
    simplified_starting_bus_groups : list[SimplifiedBusGroup]
        Simplified starting bus-group snapshots used as reference for ordering and switch layout.
        This should be action_set.get_simplified_starting_bus_groups() which has the same amount of
        assets and stations as the changed stations from the action set.
    disconnections : list[GridElement] | None, optional
        Explicit branch disconnections requested for the target state.
    full_starting_bus_groups : list[RuntimeBusGroup] | None, optional
        Full starting bus-group snapshots with all assets detectable by the importing routine.
        This is used to map out disconnections as disconnections can not be performed if the branch
        is not in the simplified topology. Note that even with the full starting stations, disconnections
        might be missed.
        Use ``action_set.starting_bus_groups`` for the unfiltered version.
        If none, this falls back to the simplified starting bus groups.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing both bus-group actions and representable disconnections.
    """
    if full_starting_bus_groups is None:
        full_starting_bus_groups = simplified_starting_bus_groups
    action_switch_updates = get_changing_switches_from_changed_bus_groups(
        changed_bus_groups=changed_bus_groups,
        starting_bus_groups=simplified_starting_bus_groups,
    )
    if disconnections and len(disconnections) > 0:
        disconnection_switch_updates = get_changing_switches_from_disconnections(
            starting_bus_groups=full_starting_bus_groups,
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
    return combined_switch_updates


@pa.check_types
def get_changing_switches_from_action_set(
    action_set: ActionSet,
    actions: list[int],
    disconnections: list[int] | None = None,
) -> pat.DataFrame[SwitchUpdateSchema]:
    """Get switch updates from stored ActionSet indices.

    Parameters
    ----------
    action_set : ActionSet
        Stored action set containing the simplified starting topology, local actions, and
        disconnectable branches.
    actions : list[int]
        Indices into ``action_set.local_actions`` describing the selected station actions.
    disconnections : list[int] | None, optional
        Indices into ``action_set.disconnectable_branches`` describing explicit branch
        disconnections.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing the requested indexed actions and disconnections.

    Raises
    ------
    ValueError
        If any action or disconnection index is out of bounds.
    """
    changed_bus_groups = _get_changed_bus_groups_from_action_indices(action_set=action_set, actions=actions)
    disconnected_branches = _get_disconnections_from_indices(action_set=action_set, disconnections=disconnections)
    return get_changing_switches_from_actions(
        changed_bus_groups=changed_bus_groups,
        simplified_starting_bus_groups=action_set.simplified_starting_bus_groups,
        disconnections=disconnected_branches,
        full_starting_bus_groups=action_set.starting_bus_groups,
    )
