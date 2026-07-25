# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build shared cascade detection context from a pandapower network."""

import numpy as np
import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection import _build_poly
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeContext
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import get_busbars_couplers


def get_switch_characteristics(net: pp.pandapowerNet, closed_status: bool | None = None) -> pd.DataFrame:
    """Build the relay information table used by cascade checks.

    This combines pandapower switch rows with their protection settings, such as
    relay side and distance protection shape. The ``angle``/``poly`` columns of
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
            ["breaker_uuid", "poly", "relay_side", "protection_side", "custom_warning_distance_protection"]
        ],
        left_on="origin_id",
        right_on="breaker_uuid",
        how="inner",
    )


def prepare_cascade_run_constants(net: pp.pandapowerNet) -> set[str]:
    """Compute per-run cascade constants once, on the base-case network.

    Two things are prepared here so they are not redone for every outage:

    1. ``net.sw_characteristics`` is converted in place — ``angle`` to radians and
       the derived ``poly`` polygon — so the (per-outage) :func:`build_cascade_context`
       becomes a pure read. The conversion is idempotent: ``poly`` acts as a sentinel,
       so calling this twice on the same net (e.g. a reused network) is safe.
    2. The busbar-coupler classification is computed for **all** ``CB`` switches on
       the base-case (all-closed) topology. Per outage, :func:`build_cascade_context`
       just intersects this set with the currently closed switches.

    Parameters
    ----------
    net : pp.pandapowerNet
        Base-case pandapower network (switch topology applied). Mutated in place.

    Returns
    -------
    set[str]
        Origin ids of every CB switch that couples two busbars on the base-case
        topology. Empty when the network has no switch characteristics.
    """
    if "sw_characteristics" not in net:
        return set()

    # Create ``poly`` even when the table is empty so the per-outage merge in
    # ``get_switch_characteristics`` still finds the column (mirrors the previous
    # unconditional conversion). ``poly`` doubles as the idempotency sentinel.
    if "poly" not in net.sw_characteristics.columns:
        net.sw_characteristics["angle"] = np.radians(net.sw_characteristics["angle"])
        net.sw_characteristics["poly"] = net.sw_characteristics.apply(_build_poly, axis=1)

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
