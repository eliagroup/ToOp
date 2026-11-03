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
import pandera.typing as pat
from toop_engine_contingency_analysis.pandapower.cascade.configuration import DistanceProtectionSeverity
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeEvent, CascadeReasonType
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_results import BranchResultSchema, SwitchResultsSchema


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
    outage_groups: dict[str, list[tuple[int, str]]],
    cascade_log_elements: list[str],
    step_no: int,
    tripped_relays_df: pat.DataFrame[SwitchResultsSchema],
) -> list[CascadeEvent]:
    """Create event log entries for distance-protection outages.

    An element taken out by several relays in the same step is logged once, credited to the
    relay that reached the innermost zone.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing element names and external ids.
    outage_groups : dict[str, list[tuple[int, str]]]
        Elements taken out by each tripped relay, keyed by relay id.
    cascade_log_elements : list[str]
        Element types that should appear in the event log.
    step_no : int
        Cascade step number.
    tripped_relays_df : pat.DataFrame[SwitchResultsSchema]
        Relay rows that caused the outage groups.

    Returns
    -------
    list[CascadeEvent]
        List of event log entries for the selected element types.
    """
    # A breaker solved with an impedance has one result row per terminal, and a relay on several
    # breakers has a row per breaker, so a relay can show up more than once; it is reported once,
    # from the row that reached the innermost zone.
    relays = (
        tripped_relays_df.sort_values(["danger_inside", "alarm_inside", "warning_inside"], ascending=False, kind="stable")
        .drop_duplicates(subset="relay_id")
        .set_index("relay_id")
    )
    severities = {
        relay_id: DistanceProtectionSeverity.innermost(
            danger_inside=bool(relays.at[relay_id, "danger_inside"]),
            alarm_inside=bool(relays.at[relay_id, "alarm_inside"]),
            warning_inside=bool(relays.at[relay_id, "warning_inside"]),
        )
        for relay_id in outage_groups
    }

    events = []
    logged: set[str] = set()
    for relay_id in sorted(outage_groups, key=lambda relay_id: DistanceProtectionSeverity.rank(severities[relay_id])):
        out_gr = outage_groups[relay_id]
        relay = relays.loc[relay_id]
        relay_name = None if pd.isna(relay.relay_name) else str(relay.relay_name)
        outage_group_id = _hash_outage_group_element_names(net, out_gr)

        for idx, el_type in out_gr:
            element_id = get_globally_unique_id(idx, el_type)
            if element_id in logged:
                continue
            if el_type in cascade_log_elements and net[el_type].loc[idx].get("in_service", True):
                mrid = net[el_type].loc[idx]["origin_id"]
                name = net[el_type].loc[idx]["name"]
                events.append(
                    CascadeEvent(
                        element_mrid=mrid,
                        element_id=element_id,
                        element_name=name,
                        cascade_number=step_no,
                        cascade_reason=CascadeReasonType.CASCADE_REASON_DISTANCE,
                        r_ohm=float(relay.r_ohm),
                        x_ohm=float(relay.x_ohm),
                        outage_group_id=outage_group_id,
                        distance_protection_severity=severities[relay_id],
                        relay_id=str(relay_id),
                        relay_name=relay_name,
                    )
                )
                logged.add(element_id)

    return events


def get_outage_group_current_violation_log_info(
    net: pp.pandapowerNet,
    outage_groups: dict[str, list[tuple[int, str]]],
    cascade_log_elements: list[str],
    step_no: int,
    current_overloaded_df: pat.DataFrame[BranchResultSchema],
) -> list[CascadeEvent]:
    """Create event log entries for current-overload outages.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing element names and external ids.
    outage_groups : dict[str, list[tuple[int, str]]]
        Elements affected by each overloaded branch.
    cascade_log_elements : list[str]
        Element types that should appear in the event log.
    step_no : int
        Cascade step number.
    current_overloaded_df : pat.DataFrame[BranchResultSchema]
        Current-overload trigger rows with element ids and loading values.

    Returns
    -------
    list[CascadeEvent]
        List of event log entries for the selected element types.
    """
    events = []
    loading_by_element = current_overloaded_df.set_index("element")["loading"].to_dict()
    visited_elements = set()
    for element_mrid, out_gr in outage_groups.items():
        outage_group_id = _hash_outage_group_element_names(net, out_gr)
        loading = loading_by_element.get(element_mrid)
        for idx, el_type in out_gr:
            if el_type in cascade_log_elements and net[el_type].loc[idx]["in_service"]:
                mrid = net[el_type].loc[idx]["origin_id"]
                name = net[el_type].loc[idx]["name"]
                if mrid not in visited_elements:
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
                    visited_elements.add(mrid)

    return events
