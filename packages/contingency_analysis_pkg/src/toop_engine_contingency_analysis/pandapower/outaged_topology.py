# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutations on a pandapower net for outage topology (circuit breakers, element in_service)."""

import pandapower as pp
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import PandapowerElements


def open_outaged_circuit_breakers(net: pp.pandapowerNet, outaged_elements: list[PandapowerElements]) -> list[int]:
    """
    Isolate outaged buses by opening boundary circuit breakers (CBs).

    The function identifies all buses marked as outaged and finds circuit breakers
    (`type == "CB"`) that form the electrical boundary between the outaged area and
    the rest of the network. These breakers are detected as switches connected to
    outaged buses (via either the `bus` or `element` column) that are currently closed.

    All such breakers are opened (`closed = False`) to electrically isolate the
    outaged portion of the network from the healthy grid.

    Args:
        net: pandapower network object.
        outaged_elements: List of outage descriptors. Only elements with
            `table == "bus"` are considered. Bus indices are parsed from `unique_id`.

    Returns
    -------
        List of switch indices that were opened to isolate the outaged area.

    Notes
    -----
        - Only closed circuit breakers (`type == "CB"`) are affected.
        - The function assumes that opening all CBs connected to outaged buses
          effectively isolates the outage region (i.e., CBs represent boundary points).
    """
    outaged_bus_ids = [int(element.unique_id.split("%%", 1)[0]) for element in outaged_elements if element.table == "bus"]
    if not outaged_bus_ids:
        return []

    cb_mask = (
        (net.switch["type"] == "CB")
        & net.switch["closed"]
        & (net.switch["bus"].isin(outaged_bus_ids) | net.switch["element"].isin(outaged_bus_ids))
    )

    affected_switches = net.switch.index[cb_mask].tolist()
    net.switch.loc[affected_switches, "closed"] = False
    return affected_switches


def set_outaged_elements_out_of_service(net: pp.pandapowerNet, outaged_elements: list[PandapowerElements]) -> list[bool]:
    """Set the outaged elements in the network to out of service.

    Returns info if the elements were in service before being set out of service.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to set the elements out of service in
    outaged_elements : list[PandapowerElements]
        The elements to set out of service

    Returns
    -------
    list[bool]
        A list indicating whether each element was in service before being set out of service
    """
    were_in_service = []
    if len(outaged_elements) == 0:
        # This is the base case. Append a dummy True so it does not raise due to no elements being outaged
        were_in_service.append(True)
    else:
        for element in outaged_elements:
            was_in_service = net[element.table].loc[element.table_id, "in_service"]
            were_in_service.append(bool(was_in_service))
            net[element.table].loc[element.table_id, "in_service"] = False
    return were_in_service


def restore_outaged_circuit_breakers(net: pp.pandapowerNet, opened_cb_indices: list[int]) -> None:
    """
    Restore previously opened circuit breakers (CBs) to service.

    This function closes the switches (circuit breakers) that were previously
    opened to isolate an outage area.

    Args:
        net: pandapower network object.
        opened_cb_indices: Indices of switches returned by :func:`open_outaged_circuit_breakers`
            to be closed again.

    Notes
    -----
        - Non-existent indices are ignored.
        - Only switches currently open (`closed == False`) are modified.
    """
    net.switch.loc[opened_cb_indices, "closed"] = True


def restore_elements_to_service(
    net: pp.pandapowerNet, outaged_elements: list[PandapowerElements], were_in_service: list[bool]
) -> None:
    """Restore the outaged elements to their original in_service status.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to restore the elements in
    outaged_elements : list[PandapowerElements]
        The elements that were outaged
    were_in_service : list[bool]
        A list indicating whether each element was in service before being set out of service
    """
    for i, element in enumerate(outaged_elements):
        if were_in_service[i]:
            net[element.table].loc[int(element.table_id), "in_service"] = True
