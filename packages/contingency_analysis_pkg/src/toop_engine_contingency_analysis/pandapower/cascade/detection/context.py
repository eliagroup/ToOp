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
    relay side and distance protection shape.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network that contains switches and switch characteristics.
    closed_status : bool
        Optional switch state filter. Use True for closed switches, False for
        open switches, or None for all switches.

    Returns
    -------
    pd.DataFrame
        DataFrame with switch metadata and relay protection characteristics.
    """
    net.sw_characteristics["angle"] = np.radians(net.sw_characteristics["angle"])
    net.sw_characteristics["poly"] = net.sw_characteristics.apply(_build_poly, axis=1)

    filtered_switches = net.switch[net.switch.closed == closed_status] if closed_status is not None else net.switch

    return filtered_switches[["bus", "element", "origin_id"]].merge(
        net.sw_characteristics[["breaker_uuid", "poly", "relay_side", "custom_warning_distance_protection"]],
        left_on="origin_id",
        right_on="breaker_uuid",
        how="inner",
    )


def build_cascade_context(net: pp.pandapowerNet) -> CascadeContext:
    """Prepare reusable cascade data for one network state.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to analyze.

    Returns
    -------
    CascadeContext
        CascadeContext containing closed-switch relay data and busbar coupler ids.
    """
    switch_characteristics = get_switch_characteristics(net, closed_status=True)
    bus_couplers_mrids = set(get_busbars_couplers(net, switch_characteristics.breaker_uuid.to_list()))
    return CascadeContext(
        switch_characteristics=switch_characteristics,
        bus_couplers_mrids=bus_couplers_mrids,
    )
