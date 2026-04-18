# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""High-level switch update export entrypoints."""

import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import cast
from toop_engine_dc_solver.export.asset_topology_to_dgs import SwitchUpdateSchema
from toop_engine_dc_solver.export.disconnection_switch_updates import get_changing_switches_from_disconnections
from toop_engine_dc_solver.export.station_switch_updates import get_changing_switches_from_changed_stations
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.nminus1_definition import GridElement

logger = structlog.get_logger(__name__)


@pa.check_types
def get_changing_switches_from_actions(
    changed_stations: list[Station],
    starting_topology: Topology,
    disconnections: list[GridElement] | None = None,
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
    starting_topology : Topology
        Simplified starting topology used as reference for station ordering and switch layout.
    disconnections : list[GridElement] | None, optional
        Explicit branch disconnections requested for the target state.

    Returns
    -------
    pat.DataFrame[SwitchUpdateSchema]
        Switch update rows representing both station actions and representable disconnections.
    """
    action_switch_updates = get_changing_switches_from_changed_stations(
        changed_stations=changed_stations,
        starting_topology=starting_topology,
    )
    disconnection_switch_updates = get_changing_switches_from_disconnections(
        starting_topology=starting_topology,
        disconnections=disconnections,
    )

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
