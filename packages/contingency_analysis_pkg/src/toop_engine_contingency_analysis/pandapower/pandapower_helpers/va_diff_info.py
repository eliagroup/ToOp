# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for identifying voltage-angle-difference monitoring points

by mapping contingency branch buses to their nearest circuit breakers.
"""

from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PandapowerContingency, VADiffInfo

BUS_COLUMN_MAP = {
    "line": ["from_bus", "to_bus"],
    "trafo": ["hv_bus", "lv_bus"],
    "trafo3w": ["hv_bus", "lv_bus"],
}


def get_va_diff_info(
    contingency: PandapowerContingency,
    net: pandapowerNet,
    node_to_switch_map: dict[int, dict[str, str]],
) -> list[VADiffInfo]:
    """Add information about which switches to monitor for voltage angle difference to the contingency.

    This function modifies the contingency in place.

    Parameters
    ----------
    contingency : PandapowerContingency
        The contingency to add the information to.
    net : pandapowerNet
        The pandapower network to use for the translation. This is used to get buses etc.
    node_to_switch_map : dict[int, list[int]]
        A mapping from nodes at branches and their closest Circuit breaker switches.
    """
    va_diff_info: list[VADiffInfo] = []
    for element in contingency.elements:
        if element.table not in BUS_COLUMN_MAP:
            continue
        from_bus_id, to_bus_id = net[element.table].loc[element.table_id, BUS_COLUMN_MAP[element.table]]
        switches_from = node_to_switch_map.get(from_bus_id, {})
        switches_to = node_to_switch_map.get(to_bus_id, {})
        if not {**switches_from, **switches_to}:
            continue
        va_diff_info.append(
            VADiffInfo(
                from_bus=from_bus_id, to_bus=to_bus_id, power_switches_from=switches_from, power_switches_to=switches_to
            )
        )
    return va_diff_info
