# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Outage load-flow execution (plain AC/DC or SpPS) on a pandapower net."""

import pandapower as pp
from beartype.typing import Any, Literal
from toop_engine_contingency_analysis.pandapower.outaged_topology import (
    open_outaged_circuit_breakers,
    set_outaged_elements_out_of_service,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import PandapowerElements
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import SingleOutageSppsContext
from toop_engine_contingency_analysis.pandapower.spps import SppsPowerFlowError, SppsResult, run_spps
from toop_engine_contingency_analysis.pandapower.spps.engine import _run_power_flow
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_results import ConvergenceStatus


def run_outage_power_flow(
    net: pp.pandapowerNet,
    spps: SingleOutageSppsContext,
    method: Literal["ac", "dc"],
    outaged_elements: list[PandapowerElements],
    *,
    runpp_kwargs: dict[str, Any] | None = None,
) -> tuple[ConvergenceStatus, SppsResult | None]:
    """Execute load flow for the current outaged *net* (mutated in place).

    Applies :func:`~toop_engine_contingency_analysis.pandapower.outaged_topology.set_outaged_elements_out_of_service`
    to *outaged_elements*, then runs plain AC/DC or SpPS per *spps* and *method*.
    Returns ``were_in_service`` for
    :func:`~toop_engine_contingency_analysis.pandapower.outaged_topology.restore_elements_to_service`.

    Additional pandapower arguments go through *runpp_kwargs*.
    """
    open_outaged_circuit_breakers(net, outaged_elements)

    were_in_service = set_outaged_elements_out_of_service(net, outaged_elements)

    if not any(were_in_service):
        return ConvergenceStatus.NO_CALCULATION, None

    merged_runpp = runpp_kwargs or {}

    try:
        if spps.conditions.empty:
            _run_power_flow(
                net=net,
                method=method,
                runpp_kwargs=merged_runpp,
            )
            return ConvergenceStatus.CONVERGED, None

        spps_result = run_spps(
            net=net,
            conditions=spps.conditions,
            actions=spps.actions,
            method=method,
            failed_elements={get_globally_unique_id(element.table_id, element.table) for element in outaged_elements},
            runpp_kwargs=merged_runpp,
            max_iterations=spps.rules_max_iterations,
            on_power_flow_error=spps.on_power_flow_error,
        )

        if spps_result.power_flow_failed or spps_result.max_iterations_reached:
            return ConvergenceStatus.FAILED, spps_result

        return ConvergenceStatus.CONVERGED, spps_result

    except (pp.LoadflowNotConverged, SppsPowerFlowError):
        return ConvergenceStatus.FAILED, None
