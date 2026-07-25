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
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.polars_results import (
    cache_res_tables_as_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    SingleOutageSppsContext,
    SlackAllocationConfig,
)
from toop_engine_contingency_analysis.pandapower.spps import SppsPowerFlowError, SppsResult, run_spps
from toop_engine_contingency_analysis.pandapower.spps.engine import _run_power_flow
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.loadflow_results import ConvergenceStatus


def _with_warm_start(net: pp.pandapowerNet, method: str, runpp_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add ``init="results"`` for AC runs when the net carries usable base-case voltages.

    The outage net is a deep copy of the *solved* base case, so its ``res_bus`` voltages
    are a far better Newton-Raphson start than the default DC initialization (measured:
    6 iterations from DC init vs 0-4 warm-started). The caller's explicit ``init`` always
    wins, and DC runs are left untouched (``rundcpp`` takes no init).
    """
    if method != "ac" or "init" in runpp_kwargs:
        return runpp_kwargs
    res_bus = net.res_bus
    if len(res_bus) != len(net.bus) or not res_bus["vm_pu"].notna().any():
        return runpp_kwargs
    return {**runpp_kwargs, "init": "results"}


def run_outage_power_flow(
    net: pp.pandapowerNet,
    spps: SingleOutageSppsContext,
    method: Literal["ac", "dc"],
    outaged_elements: list[PandapowerElements],
    basecase_net: pp.pandapowerNet,
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

    *basecase_net* (deep-copy of the network after the base-case load flow) is
    forwarded to :func:`run_spps`, which uses it to evaluate condition rows
    whose ``condition_mode`` is ``"BC"`` against base-case results.

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
            try:
                _run_power_flow(
                    net=net,
                    method=method,
                    runpp_kwargs=_with_warm_start(net, method, merged_runpp),
                )
            except pp.LoadflowNotConverged:
                # The warm start changes the solver trajectory; a contingency that would
                # have converged from the cold (DC) start must not be reported as failed
                # because of it. Retry once with the caller's original settings.
                _run_power_flow(
                    net=net,
                    method=method,
                    runpp_kwargs=merged_runpp,
                )
            cache_res_tables_as_polars(net)
            return ConvergenceStatus.CONVERGED, None

        spps_result = run_spps(
            net=net,
            conditions=spps.conditions,
            actions=spps.actions,
            failed_elements={get_globally_unique_id(element.table_id, element.table) for element in outaged_elements},
            basecase_net=basecase_net,
            method=method,
            runpp_kwargs=merged_runpp,
            max_iterations=spps.rules_max_iterations,
            on_power_flow_error=spps.on_power_flow_error,
            slack_allocation_config=slack_allocation_config,
        )

        if spps_result.power_flow_failed or spps_result.max_iterations_reached:
            return ConvergenceStatus.FAILED, spps_result

        # Snapshot the freshly solved res_* tables so result extraction can read polars.
        cache_res_tables_as_polars(net)
        return ConvergenceStatus.CONVERGED, spps_result

    except (pp.LoadflowNotConverged, SppsPowerFlowError):
        return ConvergenceStatus.FAILED, None
