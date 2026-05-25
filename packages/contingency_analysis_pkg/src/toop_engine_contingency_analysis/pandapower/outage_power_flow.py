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
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    SingleOutageSppsContext,
    SlackAllocationConfig,
)
from toop_engine_contingency_analysis.pandapower.spps import SppsPowerFlowError, SppsResult, run_spps
from toop_engine_contingency_analysis.pandapower.spps.engine import _run_power_flow
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.loadflow_results import ConvergenceStatus


def run_outage_power_flow(
    net: pp.pandapowerNet,
    spps: SingleOutageSppsContext,
    method: Literal["ac", "dc"],
    outaged_elements: list[PandapowerElements],
    *,
    runpp_kwargs: dict[str, Any] | None = None,
    slack_allocation_config: SlackAllocationConfig | None = None,
) -> tuple[ConvergenceStatus, SppsResult | None]:
    """Execute load flow for the current outaged *net* (mutated in place).

    Applies outage topology changes, optionally assigns slack buses per island,
    then runs a plain AC/DC power flow or the SpPS rule engine per *spps*.

    When *slack_allocation_config* is provided:

    * **Initial PF** — a fresh NetworkX graph is built from *net* after the
      outage is applied and :func:`assign_slack_per_island` is called so every
      electrical island has a valid slack bus before the first solve.
    * **SpPS in-loop PFs** — the config is forwarded to :func:`run_spps` so
      that slack buses are reassigned after each batch of rule actions (switch
      openings, setpoint changes) before the subsequent power flow.

    Additional pandapower arguments go through *runpp_kwargs*.
    """
    open_outaged_circuit_breakers(net, outaged_elements)

    were_in_service = set_outaged_elements_out_of_service(net, outaged_elements)

    if not any(were_in_service):
        return ConvergenceStatus.NO_CALCULATION, None

    if slack_allocation_config is not None:
        assign_slack_per_island(
            net=net,
            min_island_size=slack_allocation_config.min_island_size,
        )

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
            slack_allocation_config=slack_allocation_config,
        )

        if spps_result.power_flow_failed or spps_result.max_iterations_reached:
            return ConvergenceStatus.FAILED, spps_result

        return ConvergenceStatus.CONVERGED, spps_result

    except (pp.LoadflowNotConverged, SppsPowerFlowError):
        return ConvergenceStatus.FAILED, None
