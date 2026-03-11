# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""
Utilities for identifying circuit breakers in a pandapower network that should be

observed during contingency analysis.
"""

from typing import Optional

import pandapower as pp
import pandas as pd
from toop_engine_grid_helpers.pandapower.outage_group import build_connected_components_for_contingency_analysis


def get_observed_circuit_breakers(
    net: pp.pandapowerNet,
    exclude_lower_vn_kv: Optional[float] = None,
    exclude_higher_vn_kv: Optional[float] = None,
    region: Optional[str] = None,
) -> set[int]:
    """
    Return circuit breaker indices connected to buses in connected components used for contingency analysis.

    Buses can optionally be filtered by voltage range (`exclude_lower_vn_kv`,
    `exclude_higher_vn_kv`) and geographical region. If a connected component
    contains at least one bus that is not excluded by the filters, all circuit breakers
    connected to buses in that component are collected.

    Switch indices are returned as a set to ensure uniqueness.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network containing bus and switch tables.

    exclude_lower_vn_kv : float, optional
        Exclude buses with nominal voltage less than or equal to this value.

    exclude_higher_vn_kv : float, optional
        Exclude buses with nominal voltage greater than or equal to this value.

    region : str, optional
        Filter buses by `GeographicalRegion_name`.

    Returns
    -------
    Set[int]
        A set of unique switch indices connected to buses in the filtered
        connected components.
    """
    connected_components = build_connected_components_for_contingency_analysis(net)
    switch_indices = set()
    circuit_breakers = net.switch[(net.switch.type == "CB") & (net.switch.et == "b")]
    for cc in connected_components:
        buses = [int(el.split("&&", 1)[1]) for el in cc if el.split("&&", 1)[0] == "b"]

        if not buses:
            continue

        bus_df = net.bus.loc[buses]

        in_scope = pd.Series(True, index=bus_df.index)

        if exclude_lower_vn_kv is not None:
            in_scope &= bus_df["vn_kv"] > exclude_lower_vn_kv

        if exclude_higher_vn_kv is not None:
            in_scope &= bus_df["vn_kv"] < exclude_higher_vn_kv

        if region is not None:
            in_scope &= bus_df["GeographicalRegion_name"] == region

        # If no bus in the component passes filters, skip this component
        if not in_scope.any():
            continue

        res_cb = circuit_breakers[circuit_breakers["bus"].isin(buses) | circuit_breakers["element"].isin(buses)]

        switch_indices.update(res_cb.index)

    return switch_indices
