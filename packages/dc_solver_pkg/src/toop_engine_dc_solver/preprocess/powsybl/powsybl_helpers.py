# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import structlog
from beartype.typing import Literal, Optional
from pandera import Field
from pandera.typing import Index, Series
from pypowsybl.network import Network
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model

logger = structlog.get_logger(__name__)


BRANCH_MODEL_DEFAULTS = {
    "has_pst_tap": False,
    "has_pst_linear_tap": False,
    "for_reward": False,
    "for_nminus1": False,
    "overload_weight": 1.0,
    "disconnectable": False,
    "pst_controllable": False,
    "n0_n1_max_diff_factor": -1.0,
}


class BranchModel(pa.DataFrameModel):
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
    for_reward: Series[bool] = Field(nullable=True, description="Whether the branch is used for reward calculation")
    for_nminus1: Series[bool] = Field(nullable=True, description="Whether the branch is used for N-1 calculations")
    overload_weight: Series[float] = Field(nullable=True, description="Multiplier for overload calculations")
    p_max_mw: Series[float] = Field(nullable=True, description="Maximum active power in MW (taken from 'permanent_limit')")
    p_max_mw_n_1: Series[float] = Field(
        nullable=True, description="Maximum active power in MW for N-1 cases (taken from 'N-1')"
    )
    disconnectable: Series[bool] = Field(nullable=True, default=False, description="Whether the branch can be disconnected")
    pst_linear: Series[bool] = Field(
        nullable=True, default=False, description="Whether the branch can be controlled by a linear phase tap changer"
    )
    pst_group: Series[int] = Field(
        nullable=True, default=-1, description="Parallel-PST group label (-1 = not a controllable PST / not grouped)"
    )
    n0_n1_max_diff_factor: Series[float] = Field(
        nullable=True, description="Maximum difference factor between N-0 and N-1 limits"
    )


def add_missing_branch_model_columns(branches: pd.DataFrame) -> pat.DataFrame[BranchModel]:
    """Add any missing BranchModel columns using schema defaults or null values.

    This function ensures that the input dataframe has all the columns defined in the BranchModel schema.
    If any columns are missing, they will be added with default values from the schema or NaN if no default is defined.
    This is used instead of settings add_missing_col on the pandera model, since the latter will not work,
    if pandera validation is disabled (like it is in production)

    Parameters
    ----------
    branches: pd.DataFrame
        The input dataframe containing branch data, which may be missing some columns defined in the BranchModel schema.

    Returns
    -------
    pat.DataFrame[BranchModel]
        A dataframe that conforms to the BranchModel schema,
        with all missing columns added and filled with default values or NaN.
    """
    branch_template = get_empty_dataframe_from_model(BranchModel).reindex(branches.index)
    normalized_branches = branches.copy()

    for column_name, (_, field) in BranchModel.__fields__.items():
        if column_name == branch_template.index.name or column_name in normalized_branches.columns:
            continue

        default_value = BRANCH_MODEL_DEFAULTS.get(column_name, field.default)
        if default_value is None or default_value is ...:
            default_value = np.nan
        branch_template[column_name] = default_value

    missing_columns = [
        column_name for column_name in branch_template.columns if column_name not in normalized_branches.columns
    ]
    normalized_branches = pd.concat([normalized_branches, branch_template[missing_columns]], axis=1)
    return normalized_branches[branch_template.columns]


def get_cgmes_ids(merged_net: Network) -> list[str]:
    """Get the CGMES IDs from the merged network.

    This function looks for the sub-network with CGMES format and returns the IDs of the elements in that sub-network.

    Parameters
    ----------
    merged_net: Network
        The merged powsybl network containing sub-networks of different formats.

    Returns
    -------
    list[str]
        The list of CGMES IDs from the CGMES sub-network.
    """
    cgmes_nets = []
    for sub_net_id in merged_net.get_sub_networks().index.values:
        sub_net = merged_net.get_sub_network(sub_net_id)
        if sub_net._source_format == "CGMES":
            cgmes_nets.append(sub_net)
    if len(cgmes_nets) == 0:
        raise ValueError("No CGMES sub-network found in the merged network.")
    if len(cgmes_nets) > 1:
        logger.warning(
            f"Multiple CGMES sub-networks found in the merged network. Returning IDs from the first one. "
            f"Sub-network IDs: {[net.get_id() for net in cgmes_nets]}"
        )
    return cgmes_nets[0].get_identifiables().index.tolist()


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

    cur_limits = net.get_operational_limits().reset_index()[["element_id", "name", "side", "value"]]
    merged_branches = branches.merge(cur_limits, how="left", left_index=True, right_on="element_id")
    merged_branches["limit_voltage"] = np.select(
        [merged_branches["side"] == "ONE", merged_branches["side"] == "TWO"],
        [merged_branches["from_voltage"], merged_branches["to_voltage"]],
        default=np.maximum(merged_branches["from_voltage"], merged_branches["to_voltage"]),
    )
    merged_branches["p_limit"] = merged_branches["value"] * merged_branches["limit_voltage"] * 1e-3 * math.sqrt(3)
    # For each limit type and branch, get the max limit
    grouped_limits = merged_branches.groupby(["name", "element_id"]).p_limit.min().reset_index(0)
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

    trafos = net.get_2_windings_transformers(all_attributes=True)
    trafos_pu = net_pu.get_2_windings_transformers(all_attributes=True)
    trafos["x"] = trafos_pu["x_at_current_tap"]
    trafos["r"] = trafos_pu["r_at_current_tap"]
    trafos["rho"] = trafos_pu["rho"]
    trafos["x"] = trafos["x"] / trafos["rho"]

    if net._source_format == "UCTE":
        trafos["name"] = trafos.index + ": " + trafos.elementName
    elif net._source_format == "CGMES":
        trafos["name"] = trafos.name
    elif net._source_format == "hybrid":
        cgmes_ids = get_cgmes_ids(net)
        is_cgmes = trafos.index.isin(cgmes_ids)
        ucte_name = trafos.index + ": " + trafos.elementName
        cgmes_name = trafos.name
        trafos["name"] = np.where(is_cgmes, cgmes_name, ucte_name)

    else:
        trafos["name"] = (
            trafos["bus_breaker_bus1_id"]
            + " ## "
            + trafos["bus_breaker_bus2_id"]
            + " ## "
            + (trafos["elementName"] if "elementName" in trafos.keys() else trafos["name"])
        )
    linear_psts = get_linear_pst(net, mode="dc")
    trafos["pst_linear"] = False
    trafos["has_pst_tap"] = False
    trafos.loc[linear_psts.index, "pst_linear"] = linear_psts.values
    trafos.loc[linear_psts.index, "has_pst_tap"] = True
    trafos["pst_group"] = _build_pst_group_labels(trafos, net)
    return add_missing_branch_model_columns(
        trafos[["x", "r", "rho", "alpha", "name", "pst_linear", "has_pst_tap", "pst_group"]]
    )


def _build_pst_group_labels(trafos: pd.DataFrame, net: Network) -> np.ndarray:
    """Build per-transformer parallel PST group labels from the current network model.

    Parameters
    ----------
    trafos : pd.DataFrame
        Transformer dataframe indexed by transformer id. Must include the PST metadata
        columns produced in get_trafos.
    net : Network
        Powsybl network providing phase-tap-changer and voltage-level metadata.

    Returns
    -------
    np.ndarray
        Integer array aligned with trafos rows. Parallel PSTs share the same non-negative
        group label, while non-PST transformers keep the sentinel value -1.
    """
    pst_group_labels = np.full(len(trafos), -1, dtype=int)
    step_tables, buckets = _identify_pst_buckets(trafos=trafos, net=net)
    if not buckets:
        return pst_group_labels

    next_label = 0
    for positions in buckets.values():
        representatives: list[tuple[int, np.ndarray]] = []
        for position in positions:
            table = step_tables[str(trafos.index[position])]
            matching_label = next(
                (label for label, representative in representatives if np.allclose(representative, table)),
                None,
            )
            if matching_label is None:
                matching_label = next_label
                representatives.append((matching_label, table))
                next_label += 1
            pst_group_labels[position] = matching_label

    return pst_group_labels


def _identify_pst_buckets(
    trafos: pd.DataFrame, net: Network
) -> tuple[dict[str, np.ndarray], dict[tuple[object, ...], list[int]]]:
    """Collect PST comparison buckets keyed by structural transformer properties.

    Parameters
    ----------
    trafos : pd.DataFrame
        Transformer dataframe indexed by transformer id. Must contain has_pst_tap,
        bus1_id, bus2_id and voltage_level1_id.
    net : Network
        Powsybl network providing phase-tap-changer and voltage-level metadata.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[tuple[object, ...], list[int]]]
        Two items:
        1. Mapping from PST id to its numeric phase-tap-changer step table.
        2. Mapping from bucket keys to row positions in trafos. Buckets group PSTs that
           already match on unordered bus pair, nominal voltage, tap range, number of
           steps and linearity, before the full step-table comparison.
    """
    tap_changers = net.get_phase_tap_changers()
    pst_positions = np.flatnonzero(trafos["has_pst_tap"].to_numpy(dtype=bool))
    if pst_positions.size == 0:
        return {}, defaultdict(list)

    pst_ids = trafos.index[pst_positions]
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])
    nominal_v = trafos.loc[pst_ids, "voltage_level1_id"].map(voltage_levels["nominal_v"]).to_numpy()
    steps = net.get_phase_tap_changer_steps(attributes=["alpha", "rho", "x", "r", "g", "b"])

    step_tables: dict[str, np.ndarray] = {}
    buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for local_idx, (position, pst_id) in enumerate(zip(pst_positions, pst_ids, strict=True)):
        pst_id_str = str(pst_id)
        low_tap = int(tap_changers.at[pst_id_str, "low_tap"])
        high_tap = int(tap_changers.at[pst_id_str, "high_tap"])
        bus_pair = frozenset({trafos.at[pst_id_str, "bus1_id"], trafos.at[pst_id_str, "bus2_id"]})
        step_table = steps.loc[pst_id_str].sort_index()
        step_tables[pst_id_str] = step_table.to_numpy(dtype=float)
        bucket_key = (
            bus_pair,
            round(float(nominal_v[local_idx])),
            low_tap,
            high_tap,
            step_table.shape[0],
            _is_linear_pst_step_table(step_table),
        )
        buckets[bucket_key].append(int(position))

    return step_tables, buckets


def _is_linear_pst_step_table(step_table: pd.DataFrame) -> bool:
    """Return whether a phase-tap-changer step table is linear.

    Parameters
    ----------
    step_table : pd.DataFrame
        Phase-tap-changer step table for a single PST.

    Returns
    -------
    bool
        True if rho, x, r, g and b stay constant across all tap positions, otherwise False.
    """
    for column in ["rho", "x", "r", "g", "b"]:
        if column in step_table.columns and not np.allclose(step_table[column], step_table[column].iloc[0]):
            return False
    return True


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
    tie_lines_nopu = net.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id", "pairing_key"])
    if tie_lines_nopu.empty:
        return get_empty_dataframe_from_model(BranchModel)
    if net_pu is None:
        net_pu = get_network_as_pu(net)

    dangling_lines_nopu = net.get_boundary_lines(all_attributes=True)
    dangling_lines_pu = net_pu.get_boundary_lines()
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
        left_on="boundary_line1_id",
        right_on="id",
        how="left",
    )
    tie_lines = pd.merge(
        left=tie_lines,
        right=dangling_lines.add_suffix("_d2"),
        left_on="boundary_line2_id",
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
    elif net._source_format == "hybrid":
        cgmes_ids = get_cgmes_ids(net)
        is_cgmes = tie_lines.boundary_line1_id.isin(cgmes_ids) | tie_lines.boundary_line2_id.isin(cgmes_ids)
        ucte_name = tie_lines.index
        cgmes_name = tie_lines["name_d1"] + " ## " + tie_lines["name_d2"]
        tie_lines["name"] = np.where(is_cgmes, cgmes_name, ucte_name)
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

    return add_missing_branch_model_columns(tie_lines[["x", "r", "name"]])


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
    elif net._source_format == "hybrid":
        cgmes_ids = get_cgmes_ids(net)
        is_cgmes = lines.index.isin(cgmes_ids)
        ucte_name = lines.index + ": " + lines.elementName_nopu
        cgmes_name = lines.name
        lines["name"] = np.where(is_cgmes, cgmes_name, ucte_name)
    else:
        lines["name"] = (
            lines["bus_breaker_bus1_id_nopu"]
            + " ## "
            + lines["bus_breaker_bus2_id_nopu"]
            + " ## "
            + (lines["elementName_nopu"] if "elementName_nopu" in lines.keys() else lines["name"])
        )
    return add_missing_branch_model_columns(lines[["x", "r", "name"]])


def get_linear_pst(net: Network, mode: Literal["ac", "dc"], tol: float = 1e-9) -> pd.Series:
    """Check if a given branch has a linear phase shift transformer (PST) tap changer."""
    tap_steps = net.get_phase_tap_changer_steps()
    if mode == "dc":
        linear_cols = ["x"]
    elif mode == "ac":
        linear_cols = ["r", "x", "g", "b"]
    else:
        raise ValueError(f"Invalid mode {mode}. Must be 'ac' or 'dc'.")

    pst_ids = tap_steps.index.get_level_values("id").unique()
    trafo_linear_pst = pd.Series(True, index=pst_ids)

    for pst_id in pst_ids:
        pst_info = tap_steps.loc[pst_id]
        for col in linear_cols:
            pst_info_col = pst_info[col].values
            if not np.allclose(pst_info_col, pst_info_col[0], atol=tol):
                trafo_linear_pst[pst_id] = False
                break

    return trafo_linear_pst
