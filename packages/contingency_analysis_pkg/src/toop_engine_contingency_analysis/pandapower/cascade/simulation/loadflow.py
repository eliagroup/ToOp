# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run cascade-internal load flows and collect the needed result tables."""

from typing import Any

import pandapower as pp
import pandapower.topology as top
import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Literal
from toop_engine_contingency_analysis.pandapower.cascade.models import (
    CascadeSppsBranchSwitchResults,
)
from toop_engine_contingency_analysis.pandapower.outage_power_flow import run_outage_power_flow
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    get_branch_results,
    get_node_result_df,
    get_switch_results,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import (
    SwitchElementMappingSchema,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerMonitoredElementSchema,
    SingleOutageSppsContext,
)
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import ConvergenceStatus, SwitchResultsSchema


def cascade_monitored_breakers_dataframe(
    net: pp.pandapowerNet,
    breaker_origin_ids: pd.Series | list[str] | set[str],
) -> pat.DataFrame[PandapowerMonitoredElementSchema]:
    """Create the switch monitor table needed inside cascade simulation.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network containing switches.
    breaker_origin_ids : pd.Series | list[str] | set[str]
        External breaker ids that should be monitored.

    Returns
    -------
    pat.DataFrame[PandapowerMonitoredElementSchema]
        Monitored-element table containing the selected circuit breakers.
    """
    empty_df = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    index: list[str] = []
    rows: list[dict[str, Any]] = []
    switch_df = net.switch[net.switch.origin_id.isin(breaker_origin_ids)]
    for idx, row in switch_df.iterrows():
        sid = int(idx)
        index.append(get_globally_unique_id(sid, "switch"))
        rows.append(
            {
                "table": "switch",
                "table_id": sid,
                "kind": "switch",
                "name": str(row["name"]) if pd.notna(row.get("name")) else "",
            }
        )

    if not rows:
        return empty_df
    return pd.concat([empty_df, pd.DataFrame(rows, index=index)])


@pa.check_types
def run_spps_with_branch_switch_results(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    spps: SingleOutageSppsContext,
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    timestep: int,
    basecase_voltage: pat.Series[float],
    method: Literal["ac", "dc"] = "ac",
    runpp_kwargs: dict[str, Any] | None = None,
    min_island_size: int = 11,
) -> CascadeSppsBranchSwitchResults:
    """Run one load flow step and collect results needed by cascade logic.

    This assigns slack buses, runs the outage power flow, and then extracts
    branch, node, and switch results if the load flow converged.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network for this cascade step.
    contingency : PandapowerContingency
        Elements that are unavailable in this step.
    spps : SingleOutageSppsContext
        Special protection scheme settings for the load flow.
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping used to calculate switch results.
    timestep : int
        Timestep label to write into result tables.
    basecase_voltage : pat.Series[float]
        Base voltage values used by node result calculations.
    method : Literal["ac", "dc"]
        Load-flow method, either ac or dc.
    runpp_kwargs : dict[str, Any] | None
        Extra arguments forwarded to pandapower.
    min_island_size : int
        Smallest island size that can receive a slack bus.

    Returns
    -------
    CascadeSppsBranchSwitchResults
        Object with convergence status and result tables, or empty result fields
        when the step failed.
    """
    slack_net_graph = top.create_nxgraph(net)
    bus_lookup = create_bus_lookup_simple(net)[0]
    failed_element_uids = {get_globally_unique_id(el.table_id, el.table) for el in contingency.elements}
    assign_slack_per_island(
        net=net,
        net_graph=slack_net_graph,
        bus_lookup=bus_lookup,
        elements_ids=list(failed_element_uids),
        min_island_size=min_island_size,
    )

    convergence_status, spps_result = run_outage_power_flow(
        net,
        spps,
        method,
        contingency.elements,
        runpp_kwargs=runpp_kwargs,
    )

    if convergence_status != ConvergenceStatus.CONVERGED:
        return CascadeSppsBranchSwitchResults(
            convergence_status=convergence_status,
            spps_result=spps_result,
            branch_results=None,
            node_results=None,
            switch_results=None,
        )

    try:
        branch_results = get_branch_results(net, contingency, timestep)
        node_results = get_node_result_df(net, contingency, timestep, basecase_voltage)
        switch_results: pat.DataFrame[SwitchResultsSchema] = get_switch_results(
            net,
            contingency,
            timestep,
            branch_results,
            node_results,
            switch_element_mapping,
        )
    except Exception:
        return CascadeSppsBranchSwitchResults(
            convergence_status=ConvergenceStatus.FAILED,
            spps_result=spps_result,
            branch_results=None,
            node_results=None,
            switch_results=None,
        )

    return CascadeSppsBranchSwitchResults(
        convergence_status=convergence_status,
        spps_result=spps_result,
        branch_results=branch_results,
        node_results=node_results,
        switch_results=switch_results,
    )
