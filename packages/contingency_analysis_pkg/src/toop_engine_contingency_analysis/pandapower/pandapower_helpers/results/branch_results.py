# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Utilities for extracting and formatting pandapower branch result metrics per contingency."""

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
from pandapower import pandapowerNet
from pandapower.toolbox import res_power_columns
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
)


@pa.check_types
def get_branch_results(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
) -> pat.DataFrame[BranchResultSchema]:
    """Get the branch results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the branch results for
    contingency: PandapowerContingency
        The contingency to compute the branch results for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema]
        The list of monitored elements including branches
    timestep : int
        The timestep of the results

    Returns
    -------
    pat.DataFrame[BranchResultSchema]
        The branch results for the given network and contingency
    """
    monitored_branches = monitored_elements.query("kind == 'branch'")
    if monitored_branches.empty:
        # If no elements are monitored, return an empty dataframe
        return get_empty_dataframe_from_model(BranchResultSchema)
    max_amount_of_sides = 3
    branch_element_list = []

    for branch_type in monitored_branches.table.unique():
        branch_type_df = monitored_elements.loc[monitored_elements.table == branch_type]
        table_ids = branch_type_df.table_id.values
        unique_ids = branch_type_df.index.values
        for side in range(max_amount_of_sides):
            try:
                columns = res_power_columns(branch_type, side=side)
                columns.append(
                    columns[0].replace("p_", "i_").replace("_mw", "_ka")
                )  # hacky way to include the current aswell
                columns.append("loading_percent")
            except KeyError:
                # This means all sides were considered
                break
            common_columns = net[f"res_{branch_type}"].columns.intersection(columns)
            branch_df = net[f"res_{branch_type}"].loc[table_ids, common_columns]
            branch_df = branch_df.assign(
                timestep=timestep, contingency=contingency.unique_id, side=side + 1, element=unique_ids
            )
            branch_df.set_index(["timestep", "contingency", "element", "side"], inplace=True)
            branch_df.rename(columns=dict(zip(columns, ["p", "q", "i", "loading"], strict=True)), inplace=True)
            # Fix kA -> A and % -> 1 scale only if present
            if "i" in branch_df.columns:
                branch_df["i"] *= 1000
            if "loading" in branch_df.columns:
                branch_df["loading"] /= 100
            branch_df.loc[branch_df.i.isna(), "p"] = np.nan
            branch_df.loc[branch_df.i.isna(), "q"] = np.nan
            branch_element_list.append(branch_df)
    branch_element_df = pd.concat(branch_element_list)
    # fill missing columns with NaN
    branch_element_df["element_name"] = ""
    branch_element_df["contingency_name"] = ""
    return branch_element_df
