# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower bus (node) simulation results per contingency."""

import pandera as pa
import pandera.typing as pat
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    NodeResultSchema,
)


@pa.check_types
def get_node_result_df(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
) -> pat.DataFrame[NodeResultSchema]:
    """Get the node results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the node results for
    contingency: PandapowerContingency
        The contingency to compute the node results for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
        The list of monitored elements including buses
    timestep : int
        The timestep of the results

    Returns
    -------
    pat.DataFrame[NodeResultSchema]
        The node results for the given network and contingency
    """
    monitored_buses = monitored_elements.query("kind == 'bus'")
    if monitored_buses.empty:
        # If no buses are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(NodeResultSchema)
    table_ids = monitored_buses.table_id.to_list()
    unique_ids = monitored_buses.index.to_list()
    node_results_df = net.res_bus.reindex(table_ids)
    node_results_df = node_results_df.assign(timestep=timestep, contingency=contingency.unique_id, element=unique_ids)
    node_results_df.set_index(["timestep", "contingency", "element"], inplace=True)
    max_allowed_deviation = 0.2  # 20% voltage deviation is considered acceptable
    node_results_df["vm_loading"] = (node_results_df["vm_pu"] - 1) / max_allowed_deviation
    node_results_df.rename(columns={"vm_pu": "vm", "va_degree": "va", "p_mw": "p", "q_mvar": "q"}, inplace=True)
    voltage_levels = net.bus.reindex(table_ids)["vn_kv"].values
    node_results_df["vm"] *= voltage_levels
    # fill missing columns with NaN
    node_results_df["element_name"] = ""
    node_results_df["contingency_name"] = ""
    return node_results_df
