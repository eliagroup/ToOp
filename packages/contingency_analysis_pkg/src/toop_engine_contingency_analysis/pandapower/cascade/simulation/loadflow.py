# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run cascade-internal load flows and collect the needed result tables."""

from typing import Any

import pandapower as pp
import pandera as pa
import pandera.typing as pat
from beartype.typing import Literal
from toop_engine_contingency_analysis.pandapower.cascade.models import (
    CascadeSppsBranchSwitchResults,
)
from toop_engine_contingency_analysis.pandapower.outage_power_flow import run_outage_power_flow
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    get_branch_results_polars,
    get_node_results_polars,
    get_switch_results,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.result_constants import (
    ResultConstants,
    cache_res_tables_as_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import (
    SwitchElementMappingSchema,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerMonitoredElementSchema,
    SingleOutageSppsContext,
    SlackAllocationConfig,
)
from toop_engine_interfaces.loadflow_results import ConvergenceStatus, SwitchResultsSchema


@pa.check_types
def run_spps_with_branch_switch_results(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    spps: SingleOutageSppsContext,
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    timestep: int,
    basecase_net: pp.pandapowerNet,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    method: Literal["ac", "dc"] = "ac",
    runpp_kwargs: dict[str, Any] | None = None,
    min_island_size: int = 11,
) -> CascadeSppsBranchSwitchResults:
    """Run one load flow step and collect results needed by cascade logic.

    Builds a :class:`SlackAllocationConfig` from *min_island_size* and
    delegates to :func:`run_outage_power_flow`, which owns all slack-bus
    assignment (initial PF and SpPS in-loop reassignment).  Extracts branch,
    node, and switch results when the load flow converged.

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
    basecase_net : pp.pandapowerNet
        Deep-copy of the network whose ``res_bus.vm_pu`` column holds the
        reference (pre-step) voltages used by node result calculations.
    method : Literal["ac", "dc"]
        Load-flow method, either ac or dc.
    runpp_kwargs : dict[str, Any] | None
        Extra arguments forwarded to pandapower.
    min_island_size : int
        Smallest island size that can receive a slack bus; passed through to
        :class:`SlackAllocationConfig`.
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        Branch and node results are filtered to this set after
        the load flow. Only monitored branches and nodes are then used for
        subsequent cascade trigger detection.

    Returns
    -------
    CascadeSppsBranchSwitchResults
        Object with convergence status and result tables, or empty result fields
        when the step failed.
    """
    slack_allocation_config = SlackAllocationConfig(
        min_island_size=min_island_size,
    )

    convergence_status, spps_result = run_outage_power_flow(
        net,
        spps,
        method,
        contingency.elements,
        runpp_kwargs=runpp_kwargs,
        slack_allocation_config=slack_allocation_config,
        basecase_net=basecase_net,
    )

    if convergence_status != ConvergenceStatus.CONVERGED:
        return CascadeSppsBranchSwitchResults(
            convergence_status=convergence_status,
            spps_result=spps_result,
            branch_results=None,
            node_results=None,
            switch_results=None,
        )

    # Branch/node/switch results are produced on polars. Snapshot the res_* tables and build
    # the per-outage constants the polars extractors need, then convert the (small) outputs
    # back to the indexed pandas frames the cascade detection internals still expect.
    cache_res_tables_as_polars(net)
    constants = ResultConstants.from_network(net, basecase_net, monitored_elements, switch_element_mapping)

    branch_results_pl = get_branch_results_polars(net, contingency, timestep, constants)
    node_results_pl = get_node_results_polars(net, contingency, timestep, constants)
    switch_results_pl = get_switch_results(
        net,
        contingency,
        timestep,
        branch_results_pl,
        node_results_pl,
        constants.switch_element_mapping_pl,
    )

    branch_results = branch_results_pl.to_pandas().set_index(["timestep", "contingency", "element", "side"])
    node_results = node_results_pl.to_pandas().set_index(["timestep", "contingency", "element"])
    switch_results: pat.DataFrame[SwitchResultsSchema] = switch_results_pl.to_pandas().set_index(
        ["timestep", "contingency", "element"]
    )

    monitored_index = monitored_elements.index
    branch_results = branch_results[branch_results.index.isin(monitored_index, level="element")]
    node_results = node_results[node_results.index.isin(monitored_index, level="element")]

    return CascadeSppsBranchSwitchResults(
        convergence_status=convergence_status,
        spps_result=spps_result,
        branch_results=branch_results,
        node_results=node_results,
        switch_results=switch_results,
    )
