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
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.branch_res_power_columns import (
    branch_res_power_columns,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id_from_index
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
)


def _get_imax_for_trafos(net_table: pd.DataFrame, sn_col: str, vn_col: str, i_limit_col: str) -> np.ndarray:
    """Vectorized imax assignment for a given trafo table and ids.

    Returns the maximum of the rated current calculated via rated power and rated voltage, and the rated current
    taken from the CurrentLimit value.
    """
    if i_limit_col in net_table.columns:
        i_limit = net_table[i_limit_col].fillna(0).to_numpy() / 1000  # already in A
    else:
        i_limit = np.zeros(len(net_table))

    i_rated = net_table[sn_col].to_numpy() / (np.sqrt(3) * net_table[vn_col].to_numpy())  # convert kA to A
    imax = np.maximum(i_rated, i_limit)

    return imax


@pa.check_types
def get_branch_results(
    net: pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
) -> pat.DataFrame[BranchResultSchema]:
    """Get the branch results for the given network and contingency

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the branch results for
    contingency: PandapowerContingency
        The contingency to compute the branch results for
    timestep : int
        The timestep of the results

    Returns
    -------
    pat.DataFrame[BranchResultSchema]
        The branch results for the given network and contingency
    """
    max_amount_of_sides = 3
    branch_element_list = []

    net.res_line["i_max"] = net.line["max_i_ka"].to_numpy()
    net.res_line["loading_percent_from"] = net.res_line["i_from_ka"] / net.res_line["i_max"]
    net.res_line["loading_percent_to"] = net.res_line["i_to_ka"] / net.res_line["i_max"]

    net.res_trafo["i_hv_max"] = _get_imax_for_trafos(
        net_table=net.trafo,
        sn_col="sn_mva",
        vn_col="vn_hv_kv",
        i_limit_col="CurrentLimit.value_hv",
    )
    net.res_trafo["loading_percent_hv"] = net.res_trafo["i_hv_ka"] / net.res_trafo["i_hv_max"]

    net.res_trafo["i_lv_max"] = _get_imax_for_trafos(
        net_table=net.trafo,
        sn_col="sn_mva",
        vn_col="vn_lv_kv",
        i_limit_col="CurrentLimit.value_lv",
    )
    net.res_trafo["loading_percent_lv"] = net.res_trafo["i_lv_ka"] / net.res_trafo["i_lv_max"]

    net.res_trafo3w["i_hv_max"] = _get_imax_for_trafos(
        net_table=net.trafo3w,
        sn_col="sn_hv_mva",
        vn_col="vn_hv_kv",
        i_limit_col="CurrentLimit.value_hv",
    )
    net.res_trafo3w["loading_percent_hv"] = net.res_trafo3w["i_hv_ka"] / net.res_trafo3w["i_hv_max"]

    net.res_trafo3w["i_mv_max"] = _get_imax_for_trafos(
        net_table=net.trafo3w,
        sn_col="sn_mv_mva",
        vn_col="vn_mv_kv",
        i_limit_col="CurrentLimit.value_mv",
    )
    net.res_trafo3w["loading_percent_mv"] = net.res_trafo3w["i_mv_ka"] / net.res_trafo3w["i_mv_max"]

    net.res_trafo3w["i_lv_max"] = _get_imax_for_trafos(
        net_table=net.trafo3w,
        sn_col="sn_lv_mva",
        vn_col="vn_lv_kv",
        i_limit_col="CurrentLimit.value_lv",
    )
    net.res_trafo3w["loading_percent_lv"] = net.res_trafo3w["i_lv_ka"] / net.res_trafo3w["i_lv_max"]

    for branch_type in ("line", "trafo", "trafo3w", "impedance"):
        for side in range(max_amount_of_sides):
            try:
                columns = branch_res_power_columns(branch_type, side=side)
            except IndexError:
                # This means all sides were considered
                break
            common_columns = net[f"res_{branch_type}"].columns.intersection(columns)
            branch_df = net[f"res_{branch_type}"][common_columns]

            unique_ids = get_globally_unique_id_from_index(branch_df.index, element_type=branch_type)
            branch_df = branch_df.assign(
                timestep=timestep, contingency=contingency.unique_id, side=side + 1, element=unique_ids
            )
            branch_df.set_index(["timestep", "contingency", "element", "side"], inplace=True)
            branch_df.rename(
                columns=dict(zip(columns, ["p", "q", "i", "loading"], strict=False)),
                inplace=True,
            )
            # Fix kA -> A and % -> 1 scale only if present
            if "i" in branch_df.columns:
                branch_df["i"] *= 1000

            branch_df.loc[branch_df.i.isna(), "p"] = np.nan
            branch_df.loc[branch_df.i.isna(), "q"] = np.nan
            branch_element_list.append(branch_df)
    branch_element_df = pd.concat(branch_element_list)
    # fill missing columns with NaN
    branch_element_df["element_name"] = ""
    branch_element_df["contingency_name"] = ""
    return branch_element_df
