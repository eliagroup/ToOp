# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


"""Utilities for extracting pandapower bus (node) simulation results per contingency."""

import numpy as np
import pandera as pa
import pandera.typing as pat
from pandapower import pandapowerNet
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id_from_index


@pa.check_types
def get_node_result_df(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    basecase_voltage: pat.Series[float],
) -> pat.DataFrame:
    """Get the node results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the node results for
    contingency: PandapowerContingency
        The contingency to compute the node results for
    timestep : int
        The timestep of the results
    basecase_voltage: pat.DataFrame[float]
        The basecase voltage results

    Returns
    -------
    pat.DataFrame[NodeResultSchema]
        The node results for the given network and contingency
    """
    # Add logic for 5% ΔV voltage limit
    net.res_bus["vm_basecase_deviation"] = (
        abs(net.res_bus["vm_pu"] - basecase_voltage) / basecase_voltage.replace(0, np.nan)
    ) * 100
    node_results_df = net.res_bus
    unique_ids = get_globally_unique_id_from_index(node_results_df.index, element_type="bus")
    node_results_df = node_results_df.assign(timestep=timestep, contingency=contingency.unique_id, element=unique_ids)

    max_allowed_deviation = 0.2  # 20% voltage deviation is considered acceptable
    node_results_df["vm_loading"] = (node_results_df["vm_pu"] - 1) / max_allowed_deviation
    node_results_df.rename(columns={"vm_pu": "vm", "va_degree": "va", "p_mw": "p", "q_mvar": "q"}, inplace=True)
    voltage_levels = net.bus["vn_kv"].values
    node_results_df["vm"] *= voltage_levels
    # fill missing columns with NaN
    node_results_df["element_name"] = ""
    node_results_df["contingency_name"] = ""
    return node_results_df
