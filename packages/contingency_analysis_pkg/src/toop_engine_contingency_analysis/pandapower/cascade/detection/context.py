# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build shared cascade detection context from a pandapower network."""

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection import describe_relay_zones
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeContext
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import get_busbars_couplers


def get_switch_characteristics(net: pp.pandapowerNet, closed_status: bool | None = None) -> pd.DataFrame:
    """Build the relay information table used by cascade checks.

    This combines pandapower switch rows with their protection settings, such as
    relay side, protected element type, the per-relay factor overrides and the
    distance protection shape. The ``angle``/``poly`` columns of
    ``net.sw_characteristics`` are expected to be already prepared (radians +
    polygon) by :func:`prepare_cascade_run_constants`; this function only reads
    them.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network that contains switches and switch characteristics.
    closed_status : bool or None
        Optional switch state filter. Use True for closed switches, False for
        open switches, or None for all switches.

    Returns
    -------
    pd.DataFrame
        DataFrame with switch metadata and relay protection characteristics.
    """
    filtered_switches = net.switch[net.switch.closed == closed_status] if closed_status is not None else net.switch

    return filtered_switches[["bus", "element", "origin_id"]].merge(
        net.sw_characteristics[
            [
                "breaker_uuid",
                "poly",
                "relay_side",
                "protection_side",
                # The factor each relay resolves to, per zone and case. prepare_cascade_run_constants
                # folded protection_element and the custom_* overrides into these once per job, so
                # neither is needed here. The names must match what resolve_effective_factors writes.
                "effective_alarm_basecase",
                "effective_alarm_contingency",
                "effective_warning_basecase",
                "effective_warning_contingency",
            ]
        ],
        left_on="origin_id",
        right_on="breaker_uuid",
        how="inner",
    )


def prepare_cascade_run_constants(net: pp.pandapowerNet, cascade_configuration: CascadeConfig) -> set[str]:
    """Compute per-run cascade constants once, on the base-case network.

    Two things are prepared here so they are not redone for every outage:

    1. ``net.sw_characteristics`` is replaced by a prepared copy from
       :func:`~toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection.describe_relay_zones`
       — ``angle`` in radians, the derived ``poly`` polygon, and the factor every relay
       resolves to for both zones and both cases — so the per-outage
       :func:`build_cascade_context` becomes a pure read.

       A relay's factor depends only on the relay and the configuration, never on which
       contingency is running, so the element-type lookup and the override fallback happen
       once here instead of twice per outage. The polygon conversion is idempotent, ``poly``
       acting as the sentinel, while the factors are rewritten on every call so a second call
       with a different configuration cannot leave stale ones behind.
    2. The busbar-coupler classification is computed for **all** ``CB`` switches on
       the base-case (all-closed) topology. Per outage, :func:`build_cascade_context`
       just intersects this set with the currently closed switches.

    Parameters
    ----------
    net : pp.pandapowerNet
        Base-case pandapower network (switch topology applied). Its ``sw_characteristics``
        table is replaced with the prepared one.
    cascade_configuration : CascadeConfig
        Cascade settings the factors are resolved from.

    Returns
    -------
    set[str]
        Origin ids of every CB switch that couples two busbars on the base-case
        topology. Empty when the network has no switch characteristics.
    """
    if "sw_characteristics" not in net:
        return set()

    # The prepared copy carries every original column plus poly and the factor columns, so it
    # can simply replace the table the per-outage merge reads from.
    net["sw_characteristics"] = describe_relay_zones(net.sw_characteristics, cascade_configuration)

    cb_origin_ids = net.switch.loc[net.switch.type == "CB", "origin_id"].tolist()
    return set(get_busbars_couplers(net, cb_origin_ids))


def build_cascade_context(net: pp.pandapowerNet, all_cb_couplers: set[str]) -> CascadeContext:
    """Prepare reusable cascade data for one network state.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to analyze.
    all_cb_couplers : set[str]
        Base-case busbar-coupler origin ids from :func:`prepare_cascade_run_constants`.
        Filtered here to the switches that are currently closed.

    Returns
    -------
    CascadeContext
        CascadeContext containing closed-switch relay data and busbar coupler ids.
    """
    switch_characteristics = get_switch_characteristics(net, closed_status=True)
    bus_couplers_mrids = all_cb_couplers & set(switch_characteristics["breaker_uuid"])
    return CascadeContext(
        switch_characteristics=switch_characteristics,
        bus_couplers_mrids=bus_couplers_mrids,
    )
