# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Bundles some helper functions around powsybl.

The helper functions are independent of the backend but work
for general powsybl net (and frankly, these should be implemented in pypowsybl itself)
"""

import math
from copy import deepcopy

import pandas as pd
import pandera as pa
import pandera.typing as pat
from beartype.typing import Optional
from pandera import DataFrameModel, Field
from pandera.typing import Index, Series
from pypowsybl.network import Network
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model


class BranchModel(DataFrameModel):
    """Schema for the branch data required by the backend."""

    id: Index[str]
    x: Series[float] = Field(nullable=True, description="Reactance in pu")
    r: Series[float] = Field(nullable=True, description="Resistance in pu")
    name: Series[str]
    rho: Series[float] = Field(nullable=True, description="Ratio of the rated voltages of the transformer")
    alpha: Series[float] = Field(nullable=True, description="Phase shift angle in degrees")
    has_pst_tap: Series[bool] = Field(
        nullable=True, default=False, description="Whether the transformer has a phase tap changer"
    )
    for_reward: Series[bool] = Field(
        nullable=True, default=False, description="Whether the branch is used for reward calculation"
    )
    for_nminus1: Series[bool] = Field(
        nullable=True, default=False, description="Whether the branch is used for N-1 calculations"
    )
    overload_weight: Series[float] = Field(nullable=True, default=1.0, description="Multiplier for overload calculations")
    p_max_mw: Series[float] = Field(nullable=True, description="Maximum active power in MW (taken from 'permanent_limit')")
    p_max_mw_n_1: Series[float] = Field(
        nullable=True, description="Maximum active power in MW for N-1 cases (taken from 'N-1')"
    )
    disconnectable: Series[bool] = Field(nullable=True, default=False, description="Whether the branch can be disconnected")
    pst_controllable: Series[bool] = Field(
        nullable=True, default=False, description="Whether the branch can be controlled by a phase tap changer"
    )
    n0_n1_max_diff_factor: Series[float] = Field(
        nullable=True, default=-1.0, description="Maximum difference factor between N-0 and N-1 limits"
    )

    class Config:
        """Configuration for the BranchModel."""

        add_missing_columns = True


def get_p_max(net: Network, fillna: float = 99999.0) -> pd.DataFrame:
    """Get the maximum active power of each branch in the network

    This probes permanent_limit to get the p_max_mw and N-1 to get p_max_mw_n_1

    Parameters
    ----------
    net : Network
        The powsybl network
    fillna : float, optional
        The value to fill NaNs with, by default 99999 MW

    Returns
    -------
    pd.DataFrame
        A dataframe with the index corresponding to all branches and two columns:
        - p_max_mw: The maximum active power in MW (taken from "permanent_limit")
        - p_max_mw_n_1: The maximum active power in MW for N-1 cases (taken from "N-1")
        These values will be the same if no N-1 limits are defined and if also no
        permanent_limit is defined, the value will be fillna
    """
    branches = net.get_branches(attributes=["voltage_level1_id", "voltage_level2_id"])
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])
    branches["from_voltage"] = voltage_levels.loc[branches["voltage_level1_id"].values, "nominal_v"].values
    branches["to_voltage"] = voltage_levels.loc[branches["voltage_level2_id"].values, "nominal_v"].values

    cur_limits = net.get_operational_limits().reset_index()[["element_id", "name", "value"]]
    merged_branches = branches.merge(cur_limits, how="left", left_index=True, right_on="element_id")
    merged_branches["p_limit"] = merged_branches["value"] * merged_branches["to_voltage"] * 1e-3 * math.sqrt(3)
    # For each limit type and branch, get the max limit
    grouped_limits = merged_branches.groupby(["name", "element_id"]).p_limit.max().reset_index(0)
    # Get permanent n0-limit and whitelisted n1-limit
    branches["permanent_limit"] = grouped_limits[grouped_limits["name"] == "permanent_limit"]["p_limit"]
    branches["permanent_limit"] = branches["permanent_limit"].fillna(fillna)
    branches["n1_limit"] = grouped_limits[grouped_limits["name"] == "N-1"]["p_limit"]
    branches["n1_limit"] = branches["n1_limit"].fillna(branches["permanent_limit"])

    # Get artificial border limits (dso trafos or tso lines). nan if not set
    branches["loadflow_based_n0"] = grouped_limits[grouped_limits["name"] == "loadflow_based_n0"]["p_limit"]
    branches["loadflow_based_n1"] = grouped_limits[grouped_limits["name"] == "loadflow_based_n1"]["p_limit"]
    # Where artificial border limits are present, use them
    branches["p_max_mw"] = branches["loadflow_based_n0"].fillna(branches["permanent_limit"])

    branches["p_max_mw_n_1"] = branches["loadflow_based_n1"].fillna(branches["n1_limit"])
    return branches[["p_max_mw", "p_max_mw_n_1"]]


def get_network_as_pu(net: Network) -> Network:
    """Get the network in per unit (pu) representation.

    This is useful for loadflow calculations, as it allows to work with normalized values.

    Parameters
    ----------
    net : Network
        The powsybl network

    Returns
    -------
    Network
        The powsybl network in per unit representation
    """
    net_pu = deepcopy(net)
    net_pu.per_unit = True
    return net_pu


@pa.check_types
def get_trafos(net: Network, net_pu: Optional[Network] = None) -> pat.DataFrame[BranchModel]:
    """Get the transformers with tap changer data

    This function merges the transformer data with the tap changers, which are returned by
    individual functions.

    Transformer model:
    https://powsybl.readthedocs.io/projects/powsybl-core/en/stable/grid_model/network_subnetwork.html
    #two-winding-transformer

    Parameters
    ----------
    net : Network
        The powsybl network
    net_pu : Optional[Network], optional
        The powsybl network in per unit representation, by default None.

    Returns
    -------
    pat.DataFrame[BranchModel]
    """
    trafos_nopu = net.get_2_windings_transformers(all_attributes=True)

    if trafos_nopu.empty:
        return get_empty_dataframe_from_model(BranchModel)
    if net_pu is None:
        net_pu = get_network_as_pu(net)
    trafos_pu = net_pu.get_2_windings_transformers(all_attributes=True)

    phase_tap_changers = pd.merge(
        left=net.get_phase_tap_changers(),
        right=net.get_phase_tap_changer_steps(),
        left_on=["id", "tap"],
        right_on=["id", "position"],
        how="left",
    )
    phase_tap_changers.columns = [f"{col}_ptap" for col in phase_tap_changers.columns]

    trafos = net.get_2_windings_transformers(all_attributes=True)
    trafos_pu = net_pu.get_2_windings_transformers(all_attributes=True)
    trafos["x"] = trafos_pu["x_at_current_tap"]
    trafos["r"] = trafos_pu["r_at_current_tap"]
    trafos["rho"] = trafos_pu["rho"]
    trafos["x"] = trafos["x"] / trafos["rho"]
    trafos = pd.merge(
        left=trafos,
        right=phase_tap_changers,
        left_index=True,
        right_index=True,
        how="left",
    )

    if net._source_format == "UCTE":
        trafos["name"] = trafos.index + " " + trafos.name + " " + trafos.elementName
    elif net._source_format == "CGMES":
        trafos["name"] = trafos.name
    else:
        trafos["name"] = (
            trafos["bus_breaker_bus1_id"]
            + " ## "
            + trafos["bus_breaker_bus2_id"]
            + " ## "
            + (trafos["elementName"] if "elementName" in trafos.keys() else trafos["name"])
        )

    trafos["has_pst_tap"] = ~trafos["low_tap_ptap"].isna() & (trafos["low_tap_ptap"] != trafos["high_tap_ptap"])

    return trafos[["x", "r", "rho", "alpha", "name", "has_pst_tap"]]


@pa.check_types
def get_tie_lines(net: Network, net_pu: Optional[Network] = None) -> pat.DataFrame[BranchModel]:
    """Get the tie lines in the network.

    This function merges the tie lines with the dangling lines, which are returned by
    individual functions.

    Parameters
    ----------
    net : Network
        The powsybl network
    net_pu : Optional[Network], optional
        The powsybl network in per unit representation, by default None.
        If None, the network will be converted to per unit representation.

    Returns
    -------
    pat.DataFrame[BranchModel]
    """
    tie_lines_nopu = net.get_tie_lines(attributes=["dangling_line1_id", "dangling_line2_id", "pairing_key"])
    if tie_lines_nopu.empty:
        return get_empty_dataframe_from_model(BranchModel)
    if net_pu is None:
        net_pu = get_network_as_pu(net)

    dangling_lines_nopu = net.get_dangling_lines(all_attributes=True)
    dangling_lines_pu = net_pu.get_dangling_lines()
    dangling_lines = pd.merge(
        left=dangling_lines_pu,
        right=dangling_lines_nopu.add_suffix("_nopu"),
        left_index=True,
        right_index=True,
        how="left",
    )

    tie_lines = pd.merge(
        left=tie_lines_nopu,
        right=dangling_lines.add_suffix("_d1"),
        left_on="dangling_line1_id",
        right_on="id",
        how="left",
    )
    tie_lines = pd.merge(
        left=tie_lines,
        right=dangling_lines.add_suffix("_d2"),
        left_on="dangling_line2_id",
        right_on="id",
        how="left",
    )
    tie_lines.index = tie_lines_nopu.index
    tie_lines["x"] = tie_lines["x_d1"] + tie_lines["x_d2"]
    tie_lines["r"] = tie_lines["r_d1"] + tie_lines["r_d2"]
    if net._source_format == "UCTE":
        tie_lines["name"] = tie_lines.index
    elif net._source_format == "CGMES":
        tie_lines["name"] = tie_lines["name_d1"] + " ## " + tie_lines["name_d2"]
    else:
        tie_lines["name"] = (
            tie_lines["bus_breaker_bus_id_nopu_d1"]
            + " ## "
            + tie_lines["pairing_key"]
            + " ## "
            + tie_lines["bus_breaker_bus_id_nopu_d2"]
            + " ## "
            + (tie_lines["elementName_nopu_d1"] if "elementName_nopu_d1" in tie_lines.keys() else tie_lines["name_d1"])
            + " ## "
            + (tie_lines["elementName_nopu_d2"] if "elementName_nopu_d2" in tie_lines.keys() else tie_lines["name_d2"])
        )

    return tie_lines[["x", "r", "name"]]


@pa.check_types
def get_lines(net: Network, net_pu: Optional[Network] = None) -> pat.DataFrame[BranchModel]:
    """Get the relevant line data for the backend.

    Parameters
    ----------
    net : Network
        The powsybl network
    net_pu : Optional[Network], optional
        The powsybl network in per unit representation, by default None.

    Returns
    -------
    pat.DataFrame[BranchModel]
    """
    lines_nopu = net.get_lines(all_attributes=True)
    if lines_nopu.empty:
        return get_empty_dataframe_from_model(BranchModel)
    if net_pu is None:
        net_pu = get_network_as_pu(net)
    lines_pu = net_pu.get_lines()

    lines = pd.merge(
        left=lines_pu,
        right=lines_nopu.add_suffix("_nopu"),
        left_index=True,
        right_index=True,
        how="left",
    )
    if net._source_format == "UCTE":
        lines["name"] = lines.index + lines.name + lines.elementName_nopu
    elif net._source_format == "CGMES":
        lines["name"] = lines.name
    else:
        lines["name"] = (
            lines["bus_breaker_bus1_id_nopu"]
            + " ## "
            + lines["bus_breaker_bus2_id_nopu"]
            + " ## "
            + (lines["elementName_nopu"] if "elementName_nopu" in lines.keys() else lines["name"])
        )
    return lines[["x", "r", "name"]]
