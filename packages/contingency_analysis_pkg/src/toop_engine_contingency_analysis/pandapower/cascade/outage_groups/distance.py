# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Build cascade outage groups caused by distance-protection trips."""

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups.topology import (
    compute_affected_nodes,
    get_elements,
    get_outage_group_for_elements,
)


def compute_switches_outage_group(
    net: pp.pandapowerNet,
    overloaded_switches_df: pd.DataFrame,
    *,
    bus_couplers_mrids: set[str],
) -> dict[int, list[tuple[int, str]]]:
    """Build outage groups caused by distance-protection switch trips.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network to inspect.
    overloaded_switches_df : pd.DataFrame
        Switches selected by the distance protection check.
    bus_couplers_mrids : set[str]
        External ids of busbar coupler switches.

    Returns
    -------
    dict[int, list[tuple[int, str]]]
        Mapping from switch id to the elements that should be outaged.
    """
    if overloaded_switches_df is None or overloaded_switches_df.empty:
        return {}

    el_list = overloaded_switches_df[~overloaded_switches_df["origin_id"].isin(bus_couplers_mrids)]
    bus_couplers_df = overloaded_switches_df[overloaded_switches_df["origin_id"].isin(bus_couplers_mrids)]

    bus_couplers_groups: dict[int, list[tuple[int, str]]] = {
        row.switch_id: [(row.switch_id, "switch")] for row in bus_couplers_df.itertuples()
    }

    if el_list.empty:
        grouped: dict[int, list[tuple[int, str]]] = {}
    else:
        affected_nodes = compute_affected_nodes(net, el_list)
        contingency_elements = {sw_idx: get_elements(net, buses) for sw_idx, buses in affected_nodes.items()}
        grouped = get_outage_group_for_elements(net, contingency_elements)

    grouped.update(bus_couplers_groups)
    return grouped
