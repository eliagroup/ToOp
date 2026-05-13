# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Create user-facing cascade event records from outage groups."""

import hashlib

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeEvent, CascadeReasonType
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id


def _hash_outage_group_element_names(
    net: pp.pandapowerNet,
    outage_group: list[tuple[int, str]],
) -> str:
    """Build a stable id from all element names in one outage group.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing element names.
    outage_group : list[tuple[int, str]]
        Elements in the outage group.

    Returns
    -------
    str
        SHA-256 hash of the sorted element names in the outage group.
    """
    element_names = sorted(str(net[el_type].loc[idx]["name"]) for idx, el_type in outage_group)
    return hashlib.sha256("\n".join(element_names).encode("utf-8")).hexdigest()


def get_outage_group_distance_protection_log_info(
    net: pp.pandapowerNet,
    outage_groups: dict[int, list[tuple[int, str]]],
    cascade_log_elements: list[str],
    step_no: int,
    tripped_switches_df: pd.DataFrame,
) -> list[CascadeEvent]:
    """Create event log entries for distance-protection outages.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing element names and external ids.
    outage_groups : dict[int, list[tuple[int, str]]]
        Elements affected by each tripped switch.
    cascade_log_elements : list[str]
        Element types that should appear in the event log.
    step_no : int
        Cascade step number.
    tripped_switches_df : pd.DataFrame
        Switch rows that caused the outage groups.

    Returns
    -------
    list[CascadeEvent]
        List of event log entries for the selected element types.
    """
    events = []
    for ind, out_gr in outage_groups.items():
        element = tripped_switches_df[tripped_switches_df.switch_id == ind].iloc[0]
        severity = "DANGER" if bool(element.danger_inside) else "WARNING"
        outage_group_id = _hash_outage_group_element_names(net, out_gr)

        for idx, el_type in out_gr:
            if el_type in cascade_log_elements and net[el_type].loc[idx].get("in_service", True):
                mrid = net[el_type].loc[idx]["origin_id"]
                name = net[el_type].loc[idx]["name"]
                events.append(
                    CascadeEvent(
                        element_mrid=mrid,
                        element_id=get_globally_unique_id(idx, el_type),
                        element_name=name,
                        cascade_number=step_no,
                        cascade_reason=CascadeReasonType.CASCADE_REASON_DISTANCE,
                        outage_group_id=outage_group_id,
                        distance_protection_severity=severity,
                    )
                )

    return events


def get_outage_group_current_violation_log_info(
    net: pp.pandapowerNet,
    outage_groups: dict[int, list[tuple[int, str]]],
    cascade_log_elements: list[str],
    step_no: int,
    current_overloaded_df: pd.DataFrame,
) -> list[CascadeEvent]:
    """Create event log entries for current-overload outages.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing element names and external ids.
    outage_groups : dict[int, list[tuple[int, str]]]
        Elements affected by each overloaded branch.
    cascade_log_elements : list[str]
        Element types that should appear in the event log.
    step_no : int
        Cascade step number.
    current_overloaded_df : pd.DataFrame
        Current-overload trigger rows with element ids and loading values.

    Returns
    -------
    list[CascadeEvent]
        List of event log entries for the selected element types.
    """
    events = []
    loading_by_element = current_overloaded_df.set_index("element")["loading"].to_dict()
    for element_mrid, out_gr in outage_groups.items():
        outage_group_id = _hash_outage_group_element_names(net, out_gr)
        loading = loading_by_element.get(element_mrid)
        for idx, el_type in out_gr:
            if el_type in cascade_log_elements and net[el_type].loc[idx]["in_service"]:
                mrid = net[el_type].loc[idx]["origin_id"]
                name = net[el_type].loc[idx]["name"]
                events.append(
                    CascadeEvent(
                        element_mrid=mrid,
                        element_id=get_globally_unique_id(idx, el_type),
                        element_name=name,
                        cascade_number=step_no,
                        cascade_reason=CascadeReasonType.CASCADE_REASON_CURRENT,
                        loading=loading,
                        outage_group_id=outage_group_id,
                    )
                )

    return events
