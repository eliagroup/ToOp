# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Screen the base case for violations that make contingency cascades meaningless.

When the N-0 network is already overloaded, or a relay impedance already sits inside a
distance protection zone, every contingency would start its cascade from an already-broken
state and practically all of them would report one. This module detects that situation once,
before any contingency is computed, so the caller can report the base-case violation and skip
cascade simulation for the whole run.

Base-case violations reuse the ordinary cascade reasons. What marks them out is
``cascade_number == BASECASE_CASCADE_NUMBER`` (0) under the ``BASECASE`` contingency: simulated
cascade events always start at step 1, so step 0 belongs to the base case alone.
"""

import logging

import pandapower as pp
import pandas as pd
import pandera.typing as pat
import polars as pl
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection.context import build_cascade_context
from toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection import (
    evaluate_distance_protection_triggers,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection.overload import (
    evaluate_overload_triggers,
    prepare_branch_results_for_overload,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection.switch_preparation import (
    prepare_switch_results_for_protection,
)
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeEvent, CascadeReasonType
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PandapowerMonitoredElementSchema
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR, get_globally_unique_id
from toop_engine_interfaces.nminus1_definition import SwitchMonitoringScope

_logger = logging.getLogger(__name__)

#: Cascade step number used for base-case violations. They are found before the first cascade
#: step runs, so they sit at step 0 while simulated events start at 1.
BASECASE_CASCADE_NUMBER = 0


def _resolve_branch_element(net: pp.pandapowerNet, element_id: str) -> tuple[str | None, str | None]:
    """Look up the origin id and name of a branch from its globally unique element id.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network holding the branch tables.
    element_id : str
        Globally unique element id, formatted as ``"<table id><separator><table>"``.

    Returns
    -------
    tuple[str | None, str | None]
        Pair ``(origin_id, name)``, both None when the id cannot be resolved.
    """
    table_id, _, table = str(element_id).rpartition(SEPARATOR)
    if not table or table not in net or table_id == "":
        return None, None

    try:
        row = net[table].loc[int(table_id)]
    except (KeyError, TypeError, ValueError):
        return None, None

    return row.get("origin_id"), row.get("name")


def _basecase_overload_events(net: pp.pandapowerNet, overloaded: pd.DataFrame) -> list[CascadeEvent]:
    """Turn overloaded base-case branch rows into cascade events.

    Branch results carry one row per side, so the rows are collapsed to the most heavily
    loaded side of each element.
    """
    if overloaded.empty:
        return []

    worst_per_element = overloaded.loc[overloaded.groupby("element")["loading"].idxmax()]

    events = []
    for element_id, loading in zip(worst_per_element["element"], worst_per_element["loading"], strict=True):
        mrid, name = _resolve_branch_element(net, element_id)
        events.append(
            CascadeEvent(
                element_mrid=mrid,
                element_id=str(element_id),
                element_name=name,
                cascade_number=BASECASE_CASCADE_NUMBER,
                cascade_reason=CascadeReasonType.CASCADE_REASON_CURRENT,
                loading=float(loading),
            )
        )
    return events


def _basecase_distance_protection_events(tripped: pd.DataFrame) -> list[CascadeEvent]:
    """Turn base-case relay trips into cascade events.

    No outage group is computed here: the base case is reported as-is rather than simulated,
    so each event describes the breaker whose relay measurement sits inside a protection zone.
    """
    if tripped.empty:
        return []

    return [
        CascadeEvent(
            element_mrid=row.origin_id,
            element_id=get_globally_unique_id(row.switch_id, "switch"),
            element_name=row.switch_name,
            cascade_number=BASECASE_CASCADE_NUMBER,
            cascade_reason=CascadeReasonType.CASCADE_REASON_DISTANCE,
            r_ohm=float(row.r_ohm),
            x_ohm=float(row.x_ohm),
            distance_protection_severity="DANGER" if bool(row.danger_inside) else "WARNING",
        )
        for row in tripped.itertuples()
    ]


def screen_basecase_for_violations(
    net: pp.pandapowerNet,
    branch_results: pl.DataFrame,
    switch_results: pl.DataFrame,
    cascade_configuration: CascadeConfig,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    all_cb_couplers: set[str],
) -> list[CascadeEvent]:
    """Report base-case violations that make contingency cascade results meaningless.

    Uses the same overload and distance-protection detection as the cascade simulator, so the
    base-case thresholds and the base-case distance protection factor apply exactly as they
    would inside a cascade step. Both the warning and the danger zone count as a violation:
    either means the relay is already reading into its protection characteristic in N-0.

    Events carry the ordinary cascade reasons; ``cascade_number`` is 0 to separate them from
    the simulated cascade steps, which start at 1.

    Parameters
    ----------
    net : pp.pandapowerNet
        Solved base-case network, with ``res_*_polars`` already cached.
    branch_results : pl.DataFrame
        Base-case branch result table as a flat polars frame.
    switch_results : pl.DataFrame
        Base-case switch result table as a flat polars frame.
    cascade_configuration : CascadeConfig
        Cascade settings holding the base-case thresholds and protection factors.
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        Elements to monitor. Only PROTECTION-scoped switches can trip, so the relay check is
        limited to them, exactly as in the cascade simulator.
    all_cb_couplers : set[str]
        Base-case busbar-coupler origin ids from ``prepare_cascade_run_constants``.

    Returns
    -------
    list[CascadeEvent]
        Base-case violation events, empty when the base case is clean. A non-empty list means
        the caller should skip cascade simulation for every contingency.
    """
    # Mirrors CascadeSimulator.simulate: the detection helpers work on pandas frames indexed
    # by the result keys, while the pipeline hands over flat polars frames.
    branch_results_df = branch_results.to_pandas().set_index(["timestep", "contingency", "element", "side"])
    switch_results_df = switch_results.to_pandas().set_index(["timestep", "contingency", "element"])

    overloaded = evaluate_overload_triggers(
        current_res=prepare_branch_results_for_overload(branch_results_df),
        cascade_configuration=cascade_configuration,
    )
    overload_events = _basecase_overload_events(net, overloaded)
    relay_events: list[CascadeEvent] = []

    monitored_breakers = monitored_elements[
        monitored_elements["monitoring_scope"].apply(lambda s: s is not None and SwitchMonitoringScope.PROTECTION in s)
    ]
    switch_results_df = switch_results_df[switch_results_df.index.get_level_values("element").isin(monitored_breakers.index)]

    if not switch_results_df.empty and "sw_characteristics" in net:
        cascade_context = build_cascade_context(net, all_cb_couplers)
        switch_prepared = prepare_switch_results_for_protection(net, switch_results_df, cascade_context=cascade_context)
        tripped = evaluate_distance_protection_triggers(switch_prepared, cascade_configuration)
        relay_events = _basecase_distance_protection_events(tripped)

    if overload_events or relay_events:
        _logger.warning(
            "cascading: base case already violates (%s overloaded elements, %s relay trips); "
            "skipping cascade simulation for every contingency",
            len(overload_events),
            len(relay_events),
        )

    return [*overload_events, *relay_events]
