# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to create artificical operational limits based on the loadflow result for the PowSyBl backend.

File: loadflow_based_current_limits.py
Author:  Leonard Hilfrich
Created: 2024-12-19
"""

import numpy as np
import pandas as pd
import polars as pl
from beartype.typing import Literal, Optional, Union
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import sort_powsybl_element_frame_by_id
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    DoubleLimits,
    LimitAdjustmentParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.network_masks import NetworkMasks
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.types import LimitCase, OperationalLimit

LOADFLOW_BASED_LIMITS = (OperationalLimit.LOADFLOW_BASED_N_0, OperationalLimit.LOADFLOW_BASED_N_1)
ARTIFICIAL_LIMITS = (OperationalLimit.ARTIFICIAL_N_0, OperationalLimit.ARTIFICIAL_N_1)
ARTIFICIAL_DC_LIMITS = (OperationalLimit.ARTIFICIAL_DC_N_0, OperationalLimit.ARTIFICIAL_DC_N_1)


def create_current_limits_df(
    new_limit_series: pd.Series,
    element_type: Literal["LINE", "BOUNDARY_LINE", "TWO_WINDINGS_TRANSFORMER"],
    side: BranchSide,
    limit: OperationalLimit,
    group_names: pd.Series,
) -> pd.DataFrame:
    """Create a dataframe matching the operational_limits format from pyposwybl.

    If the new limit is np.nan because their is no loadflow it is dropped

    Parameters
    ----------
    new_limit_series: pd.Series
        The new limits for the elements including the element ids as indices and the new limits as values
    element_type: Literal["LINE", "BOUNDARY_LINE", "TWO_WINDINGS_TRANSFORMER"]
        The type of the element. For Tielines always use the corresponding Boundary Lines
    side: Literal["ONE", "TWO", "NONE"]
        The side of the limit on the element
    limit: OperationalLimit
        The new limit, which defines the name and the acceptable duration
    group_names: pd.Series
        The group names of the limits. Required to match permanent limit

    Returns
    -------
    pd.DataFrame
        The dataframe containing the limits. Can be used together with network.create_operational_limits()
    """
    n_elements = len(new_limit_series)
    new_limits = pd.DataFrame(
        index=new_limit_series.index,
        data={
            "element_type": np.full(n_elements, element_type),
            "side": np.full(n_elements, side.name),
            "name": np.full(n_elements, limit.value),
            "type": np.full(n_elements, "CURRENT"),
            "value": new_limit_series.values,
            "acceptable_duration": np.full(n_elements, limit.acceptable_duration),
            "group_name": group_names.values,
        },
    )
    new_limits.index.name = "element_id"
    new_limits.set_index(["side", "type", "acceptable_duration", "group_name"], append=True, inplace=True)
    return new_limits.dropna()


def get_branches_including_limits_and_dangling_lines(
    branches_df: pd.DataFrame,
    operational_limits: pd.DataFrame,
    tie_lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add the old limits and the dangling lines to the branches dataframe.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns: type, i1, i2
    operational_limits: pd.DataFrame
        The operational limits dataframe with the index "element_id" and the columns:name, value
    tie_lines_df: pd.DataFrame
        The tie lines dataframe with the columns: boundary_line1_id, boundary_line2_id

    Returns
    -------
    pd.DataFrame
        The branches dataframe with the columns:
        type, update_i, i1, i2, old_limit_n_0, old_limit_n_1, boundary_line1_id, boundary_line2_id
    """
    op_lims = operational_limits.reset_index()
    for side in [BranchSide.ONE, BranchSide.TWO]:
        side_limits = op_lims[op_lims.side == side.name]
        n_0_limits = side_limits[side_limits.name.isin([OperationalLimit.PERMANENT, OperationalLimit.LOADFLOW_BASED_N_0])]
        n_0_limits = n_0_limits.assign(_priority=n_0_limits.name.eq(OperationalLimit.LOADFLOW_BASED_N_0))
        branches_df[[f"n_0_i{side.value}_max", f"n_0_group_name_{side.value}"]] = (
            n_0_limits.sort_values("_priority").groupby("element_id")[["value", "group_name"]].last()
        )
        n_1_limits = side_limits[
            side_limits.name.isin([OperationalLimit.DACF_PERMANENT_N_1, OperationalLimit.LOADFLOW_BASED_N_1])
        ]
        n_1_limits = n_1_limits.assign(_priority=n_1_limits.name.eq(OperationalLimit.LOADFLOW_BASED_N_1))
        branches_df[[f"n_1_i{side.value}_max", f"n_1_group_name_{side.value}"]] = (
            n_1_limits.sort_values("_priority").groupby("element_id")[["value", "group_name"]].last()
        )
        branches_df[f"n_1_i{side.value}_max"] = branches_df[f"n_1_i{side.value}_max"].fillna(
            branches_df[f"n_0_i{side.value}_max"]
        )
        branches_df[f"n_1_group_name_{side.value}"] = branches_df[f"n_1_group_name_{side.value}"].fillna(
            branches_df[f"n_0_group_name_{side.value}"]
        )

    branches_df[["boundary_line1_id", "boundary_line2_id"]] = tie_lines_df[["boundary_line1_id", "boundary_line2_id"]]
    return branches_df


def compute_artificial_limits(
    flow: pd.Series,
    limit: pd.Series,
    optimized: pd.Series,
    non_worsening: pd.Series,
    double_limits: DoubleLimits,
) -> pd.Series:
    """Compute the optimization limit of every branch and side from its worst flow.

    - Branches that are optimized and non-worsening ignore the lower and upper limit.
      They get the physical limit if overloaded, otherwise their flow plus ``non_worsening_margin`` (capped at
      the physical limit), at least ``minimum_lower_physical_non_worsening_limit`` times the physical limit.
    - Optimized branches above the upper limit get the upper limit.
    - Non-worsening branches get their flow plus ``non_worsening_margin``, at least
      ``minimum_lower_physical_non_worsening_limit`` times the physical limit.
    - Other optimized branches get the larger of the lower limit and their flow.

    Parameters
    ----------
    flow: pd.Series
        The worst absolute flow of the case.
    limit: pd.Series
        The physical limit of the case, in the unit of the flow.
    optimized: pd.Series
        Whether the branch is optimized.
    non_worsening: pd.Series
        Whether the branch is non-worsening.
    double_limits: DoubleLimits
        The lower and upper limit and the minimum non-worsening limit, relative to the physical limit.

    Returns
    -------
    pd.Series
        The optimization limits. NaN for branches that are neither optimized nor non-worsening and for branches
        without a flow or a physical limit.
    """
    upper_limit = limit * double_limits.upper
    non_worsening_limit = (flow * (1 + double_limits.non_worsening_margin)).clip(
        lower=limit * double_limits.minimum_lower_physical_non_worsening_limit
    )
    optimized_limit = flow.clip(lower=limit * double_limits.lower)
    optimized_limit = optimized_limit.where(flow <= upper_limit, upper_limit)
    both_limit = non_worsening_limit.clip(upper=limit).where(flow <= limit, limit)

    limits = non_worsening_limit.where(non_worsening)
    limits = limits.mask(optimized, optimized_limit)
    limits = limits.mask(optimized & non_worsening, both_limit)

    return limits.where(flow.notna() & limit.notna())


def get_new_limits_for_branch(
    loadflow_current: pd.Series, old_limit: pd.Series, factor: float, min_increase: float
) -> pd.Series:
    """Calculate new limits for a branch based on the loadflow current and old limit.

    The new limit is calculated by multiplying the loadflow current with a factor.
    The new limit is clipped to be at least the loadflow current + a percentage of the old limit
    and at most the old limit.

    Parameters
    ----------
    loadflow_current: pd.Series
        The current load on the branch from the loadflow.
    old_limit: pd.Series
        The old limit of the branch.
    factor: float
        The factor to multiply the loadflow current with to get the new limit.
    min_increase: float
        The minimum increase of the old limit that the new limit should have.
        This is a percentage of the old limit.

    Returns
    -------
    pd.Series
        The new limit for the branch.
    """
    # The lower limit is defined by the current load + an percentage of the maximum load
    lower_limit = loadflow_current + old_limit * min_increase
    # The lower limit cant be higher than the upper limit
    lower_limit = np.minimum(old_limit, lower_limit)
    new_limit = loadflow_current * factor
    return new_limit.clip(lower_limit, old_limit)


def get_loadflow_based_line_limits(
    lines_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    limit: OperationalLimit,
) -> list[pd.DataFrame]:
    """Get new limits for lines based on the current flow.

    Parameters
    ----------
    lines_df: pd.DataFrame
        The lines dataframe with the columns: update_i, old_limit_n_0, old_limit_n_1
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    limit: OperationalLimit
        The limit to create, LOADFLOW_BASED_N_0 or LOADFLOW_BASED_N_1

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits with the new limits for the lines.
        The new limits are called "loadflow_based_n_0" and "loadflow_based_n_1"
    """
    if lines_df.empty:
        return []
    update_i = lines_df[["i1", "i2"]].max(axis=1)
    case = limit.case
    old_limit = lines_df[[f"{case}_i1_max", f"{case}_i2_max"]].min(axis=1)
    border_line_limits = []
    sides: tuple[BranchSide, ...] = (BranchSide.ONE, BranchSide.TWO)
    for side in sides:
        old_limit = lines_df[f"{case}_i{side.value}_max"]
        factor, min_increase = limit_parameters.get_parameters_for_case(case)
        new_limit = get_new_limits_for_branch(update_i, old_limit, factor, min_increase)
        group_names = lines_df[f"{case}_group_name_{side.value}"]
        border_line_limits.append(
            create_current_limits_df(
                new_limit[~old_limit.isna()],
                element_type="LINE",
                side=side,
                limit=limit,
                group_names=group_names[~old_limit.isna()],
            )
        )
    return border_line_limits


def get_loadflow_based_tie_line_limits(
    tie_lines_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    limit: OperationalLimit,
) -> list[pd.DataFrame]:
    """Get new limits for tie lines based on the current flow.

    Parameters
    ----------
    tie_lines_df: pd.DataFrame
        The tie lines dataframe with the columns:
        update_i, old_limit_n_0, old_limit_n_1, boundary_line1_id, boundary_line2_id
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    limit: OperationalLimit
        The limit to create, LOADFLOW_BASED_N_0 or LOADFLOW_BASED_N_1

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits with the new limits for the tie lines.
        The new limits are called "loadflow_based_n_0" and "loadflow_based_n_1"
    """
    if tie_lines_df.empty:
        return []
    border_dangling_limits = []
    case = limit.case
    tie_lines_df = tie_lines_df.copy()
    tie_lines_df["update_i"] = tie_lines_df[["i1", "i2"]].max(axis=1)
    for side_value, dangling_line_col in zip([1, 2], ["boundary_line1_id", "boundary_line2_id"], strict=True):
        dangling_df = tie_lines_df.set_index(dangling_line_col)
        old_limit = dangling_df[[f"{case}_i1_max", f"{case}_i2_max"]].min(axis=1)

        new_limit = get_new_limits_for_branch(
            dangling_df["update_i"], old_limit, *limit_parameters.get_parameters_for_case(case)
        )
        group_names = dangling_df[f"{case}_group_name_{side_value}"]
        dangling_limit_df = create_current_limits_df(
            new_limit[~old_limit.isna()],
            element_type="BOUNDARY_LINE",
            side=BranchSide.NONE,
            limit=limit,
            group_names=group_names[~old_limit.isna()],
        )
        border_dangling_limits.append(dangling_limit_df)
    return border_dangling_limits


def get_loadflow_based_trafo_limits(
    trafos_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    limit: OperationalLimit,
) -> list[pd.DataFrame]:
    """Get new limits for trafos based on the current flow.

    Parameters
    ----------
    trafos_df: pd.DataFrame
        The trafos dataframe with the columns: update_i, old_limit_n_0, old_limit_n_1
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    limit: OperationalLimit
        The limit to create, LOADFLOW_BASED_N_0 or LOADFLOW_BASED_N_1

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits
        with the new limits for the trafos that are bordering the viewed area.
        The new limits are called "loadflow_based_n_0" and "loadflow_based_n_1"
    """
    if trafos_df.empty:
        return []

    border_trafo_limits = []
    case = limit.case
    sides: tuple[BranchSide, ...] = (BranchSide.ONE, BranchSide.TWO)
    for side in sides:
        update_i = trafos_df[f"i{side.value}"]
        old_limit = trafos_df[f"{case}_i{side.value}_max"]
        factor, min_increase = limit_parameters.get_parameters_for_case(case)
        new_limit = get_new_limits_for_branch(update_i, old_limit, factor, min_increase)
        group_names = trafos_df[f"{case}_group_name_{side.value}"]
        side_limits = create_current_limits_df(
            new_limit[~old_limit.isna()],
            element_type="TWO_WINDINGS_TRANSFORMER",
            side=side,
            limit=limit,
            group_names=group_names[~old_limit.isna()],
        )
        border_trafo_limits.append(side_limits)

    return border_trafo_limits


def get_all_border_line_limits(
    branches_df: pd.DataFrame,
    tso_border_factors: LimitAdjustmentParameters,
    line_tso_border: np.ndarray,
    tie_line_tso_border: np.ndarray,
) -> list[pd.DataFrame]:
    """Get a list of dataframes with the new limits for the lines and tie lines that are leaving the viewed area.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with
            the current i_update,
            the current limits "old_limit_n_0" and "old_limit_n_1" and
            the dangling_lines "boundary_line1_id", "boundary_line2_id"
    tso_border_factors: LimitAdjustmentParameters
        The parameters (factor and min) for the calculation of the new limits
    line_tso_border: np.ndarray
        A boolean mask for the lines that are bordering the specified area
    tie_line_tso_border: np.ndarray
        A boolean mask for the tie lines that are bordering the specified area

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits
        with the new limits for the lines and tie lines (as dangling lines)
        that are bordering the DSO area.
        The new limits are called "loadflow_based_n_0" and "loadflow_based_n_0"
    """
    lines_df = branches_df[branches_df.type == "LINE"]
    tie_lines_df = branches_df[branches_df.type == "TIE_LINE"]
    limits = []
    for limit in LOADFLOW_BASED_LIMITS:
        limits += get_loadflow_based_line_limits(lines_df[line_tso_border], tso_border_factors, limit)
        limits += get_loadflow_based_tie_line_limits(tie_lines_df[tie_line_tso_border], tso_border_factors, limit)
    return limits


def get_all_dso_trafo_limits(
    branches_df: pd.DataFrame, dso_trafo_factors: LimitAdjustmentParameters, trafo_dso_border: np.ndarray
) -> list[pd.DataFrame]:
    """Get a list of dataframes with the new limits for the trafos that are bordering the DSO area.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the current i2 and the current limits "old_limit_n_0" and "old_limit_n_1"
    dso_trafo_factors: LimitAdjustmentParameters
        The parameters (factor and min) for the calculation of the new limits
    trafo_dso_border: np.ndarray
        A boolean mask for the trafos that are bordering the specified area

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits
        with the new limits for the trafos that are bordering the DSO area.
        The new limits are called "loadflow_based_n_0" and "loadflow_based_n_0"
    """
    trafo_df = sort_powsybl_element_frame_by_id(branches_df[branches_df.type == "TWO_WINDINGS_TRANSFORMER"])
    limits = []
    for limit in LOADFLOW_BASED_LIMITS:
        limits += get_loadflow_based_trafo_limits(trafo_df[trafo_dso_border], dso_trafo_factors, limit)
    return limits


def create_new_border_limits(
    network: Network,
    network_masks: NetworkMasks,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
) -> pd.DataFrame:
    """Create the new border limits for the network.

    Based on the parameters in the ImporterParameters and the identified masks

    Parameters
    ----------
    network: Network
        The network to create the limits for. The loadflow calculation needs to have happened
    network_masks: NetworkMasks
        The network masks for the network containing the trafo_dso_border-, line_tso_border- and tie_line_tso_border-mask
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The parameters for the creation of the limits

    Returns
    -------
    pd.DataFrame
        The new limits for the network including the already existing ones
    """
    existing_limits = network.get_operational_limits()
    branches_df = get_branches_including_limits_and_dangling_lines(
        network.get_branches(attributes=["type", "i1", "i2"]),
        existing_limits,
        network.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id"]),
    )
    # Exclude tie lines, since they cant be directly updated
    old_limits = [existing_limits[existing_limits.element_type != "TIE_LINE"]]
    new_limits = []
    if importer_parameters.area_settings.border_line_factors:
        new_limits += get_all_border_line_limits(
            branches_df,
            importer_parameters.area_settings.border_line_factors,
            network_masks.line_tso_border,
            network_masks.tie_line_tso_border,
        )
    if importer_parameters.area_settings.dso_trafo_factors:
        new_limits += get_all_dso_trafo_limits(
            branches_df, importer_parameters.area_settings.dso_trafo_factors, network_masks.trafo_dso_border
        )
    updated_border_limits_df = pd.concat(old_limits + new_limits)
    # drop element_type column -> deprecated
    updated_border_limits_df.drop(columns=["element_type"], inplace=True)
    updated_border_limits_df = updated_border_limits_df[
        updated_border_limits_df.index.get_level_values("element_id").isin(
            existing_limits.index.get_level_values("element_id")
        )
    ]
    network.create_operational_limits(updated_border_limits_df.reset_index("acceptable_duration"))
    return updated_border_limits_df


def _get_worst_flows(branches_df: pd.DataFrame, worst_case: pd.DataFrame, side: BranchSide, case: LimitCase) -> pd.Series:
    """Get the worst absolute current of every branch on one side for one case.

    Branches without results keep the current of the loadflow.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns i1 and i2.
    worst_case: pd.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1.
    side: BranchSide
        The side of the branches.
    case: LimitCase
        The case being looked at (N-0 or N-1).

    Returns
    -------
    pd.Series
        The worst current per branch of the dataframe.
    """
    worst = worst_case[worst_case["side"] == side.value].set_index("element").reindex(branches_df.index)
    n_0_flow = worst["n0"].fillna(branches_df[f"i{side.value}"].abs())
    if case == "n_0":
        return n_0_flow
    return worst["n1"].fillna(n_0_flow)


def get_artificial_limits_for_case(
    branches_df: pd.DataFrame,
    worst_case: pd.DataFrame,
    double_limits: DoubleLimits,
    limit: OperationalLimit,
) -> list[pd.DataFrame]:
    """Get the optimization limits of the monitored branches for one limit.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns type, i1, i2, optimized, non_worsening, the limits and group names
        of the case and the boundary lines of the tie lines.
    worst_case: pd.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1.
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    limit: OperationalLimit
        The limit to create, e.g. ARTIFICIAL_N_0 or ARTIFICIAL_DC_N_1. It defines the case, name and acceptable duration.

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits.
    """
    case = limit.case
    sides = (BranchSide.ONE, BranchSide.TWO)
    flows = {side: _get_worst_flows(branches_df, worst_case, side, case) for side in sides}

    artificial_limits = []
    for element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
        elements_df = branches_df[branches_df["type"] == element_type]
        if elements_df.empty:
            continue
        for side in sides:
            new_limit = compute_artificial_limits(
                flow=flows[side][elements_df.index],
                limit=elements_df[f"{case}_i{side.value}_max"],
                optimized=elements_df["optimized"],
                non_worsening=elements_df["non_worsening"],
                double_limits=double_limits,
            )
            artificial_limits.append(
                create_current_limits_df(
                    new_limit,
                    element_type=element_type,
                    side=side,
                    limit=limit,
                    group_names=elements_df[f"{case}_group_name_{side.value}"],
                )
            )

    tie_lines_df = branches_df[branches_df["type"] == "TIE_LINE"]
    if not tie_lines_df.empty:
        tie_lines_df = tie_lines_df.assign(
            new_limit=compute_artificial_limits(
                flow=pd.concat(flows, axis=1).max(axis=1)[tie_lines_df.index],
                limit=tie_lines_df[[f"{case}_i1_max", f"{case}_i2_max"]].min(axis=1),
                optimized=tie_lines_df["optimized"],
                non_worsening=tie_lines_df["non_worsening"],
                double_limits=double_limits,
            )
        )
        for side_value, dangling_line_col in zip([1, 2], ["boundary_line1_id", "boundary_line2_id"], strict=True):
            dangling_df = tie_lines_df.set_index(dangling_line_col)
            artificial_limits.append(
                create_current_limits_df(
                    dangling_df["new_limit"],
                    element_type="BOUNDARY_LINE",
                    side=BranchSide.NONE,
                    limit=limit,
                    group_names=dangling_df[f"{case}_group_name_{side_value}"],
                )
            )
    return artificial_limits


def add_default_permanent_limits(network: Network, monitored_branch_ids: list[str], default_permanent: float) -> None:
    """Add a permanent limit to the monitored branches that have None on a branch SIDE.

    The limits of tie lines are set on their boundary lines.

    Parameters
    ----------
    network: Network
        The network to add the limits to. It is modified in place.
    monitored_branch_ids: list[str]
        The ids of the monitored branches.
    default_permanent: float
        The permanent limit in A of the added limits.
    """
    existing_limits = network.get_operational_limits().reset_index()
    has_permanent = set(
        existing_limits.loc[existing_limits["name"] == OperationalLimit.PERMANENT, ["element_id", "side"]].itertuples(
            index=False
        )
    )
    branches = network.get_branches(attributes=["type"])
    branches = branches[branches.index.isin(monitored_branch_ids)]
    tie_lines = network.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id"])
    missing = []
    for element_id, element_type in branches["type"].items():
        for side in (BranchSide.ONE, BranchSide.TWO):
            if (element_id, side.name) in has_permanent:
                continue
            if element_type == "TIE_LINE":
                missing.append((tie_lines.at[element_id, f"boundary_line{side.value}_id"], BranchSide.NONE.name))
            elif element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
                missing.append((element_id, side.name))
    if not missing:
        return
    # The dataframe form silently creates nothing for elements without any limit, the keyword form does not
    network.create_loading_limits(
        element_id=[element_id for element_id, _ in missing],
        side=[side for _, side in missing],
        name=[OperationalLimit.DEFAULT_PERMANENT.value] * len(missing),
        type=["CURRENT"] * len(missing),
        value=[default_permanent] * len(missing),
        acceptable_duration=[-1] * len(missing),
    )


def shift_occupied_durations(new_limits_df: pd.DataFrame, occupied_limits_df: pd.DataFrame) -> pd.DataFrame:
    """Raise the acceptable duration of new limits until it is free on their element, side, type and group.

    Powsybl keys temporary limits by these and the duration, not by name, so a taken duration would overwrite a limit.

    Parameters
    ----------
    new_limits_df: pd.DataFrame
        The new limits, indexed by element_id, side, type, acceptable_duration and group_name.
    occupied_limits_df: pd.DataFrame
        The limits that stay in the network, same index.

    Returns
    -------
    pd.DataFrame
        The new limits with free durations.
    """
    key_levels = ["element_id", "side", "type", "group_name"]
    new_df = new_limits_df.reset_index()
    taken = set(occupied_limits_df.reset_index()[[*key_levels, "acceptable_duration"]].itertuples(index=False, name=None))
    durations = []
    for row in new_df[[*key_levels, "acceptable_duration"]].itertuples(index=False, name=None):
        *key, duration = row
        while (*key, duration) in taken:
            duration += 1
        taken.add((*key, duration))
        durations.append(duration)
    new_df["acceptable_duration"] = durations
    return new_df.set_index(new_limits_df.index.names)


def create_artificial_limits(
    network: Network,
    nminus1_definition: Nminus1Definition,
    worst_case_currents: pl.DataFrame,
    double_limits: DoubleLimits,
    limits: tuple[OperationalLimit, ...] = ARTIFICIAL_LIMITS,
    default_permanent: Optional[float] = None,
) -> pd.DataFrame:
    """Create the optimization limits of the monitored branches in the network.

    The limits replace the limits of the same element, side and name, see compute_artificial_limits.

    Parameters
    ----------
    network: Network
        The network to create the limits for. The loadflow calculation needs to have happened
    nminus1_definition: Nminus1Definition
        The N-1 definition with the optimized and non-worsening monitored branches
    worst_case_currents: pl.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1,
        see extract_worst_case_branch_results_polars
    double_limits: DoubleLimits
        The lower and upper limit, relative to the physical limit.
    limits: tuple[OperationalLimit, ...]
        The limits to create, ARTIFICIAL_LIMITS by default. Use ARTIFICIAL_DC_LIMITS for DC results.
    default_permanent: Optional[float]
        If given, monitored branches without a permanent limit get this one first, so they also get an artificial limit.

    Returns
    -------
    pd.DataFrame
        The new limits for the network including the already existing ones
    """
    monitored_branches = [element for element in nminus1_definition.monitored_elements if element.kind == "branch"]
    if default_permanent is not None:
        add_default_permanent_limits(network, [element.id for element in monitored_branches], default_permanent)
    existing_limits = network.get_operational_limits()
    branches_df = get_branches_including_limits_and_dangling_lines(
        network.get_branches(attributes=["type", "i1", "i2"]),
        existing_limits,
        network.get_tie_lines(attributes=["boundary_line1_id", "boundary_line2_id"]),
    )
    monitored_elements_df = pd.DataFrame(
        {
            "optimized": [element.optimized for element in monitored_branches],
            "non_worsening": [element.non_worsening for element in monitored_branches],
        },
        index=[element.id for element in monitored_branches],
        dtype=bool,
    )
    branches_df[["optimized", "non_worsening"]] = monitored_elements_df.reindex(branches_df.index, fill_value=False)
    worst_case = pd.DataFrame(worst_case_currents.to_dict(as_series=False))

    new_limits = []
    for limit in limits:
        new_limits += get_artificial_limits_for_case(branches_df, worst_case, double_limits, limit)
    if not new_limits:
        return existing_limits
    new_limits_df = pd.concat(new_limits)

    # Exclude tie lines, since they cant be directly updated
    old_limits_df = existing_limits[existing_limits.element_type != "TIE_LINE"]
    key_columns = ["element_id", "side", "name"]
    replaced = pd.MultiIndex.from_frame(old_limits_df.reset_index()[key_columns]).isin(
        pd.MultiIndex.from_frame(new_limits_df.reset_index()[key_columns])
    )
    kept_limits_df = old_limits_df[~replaced]
    updated_limits_df = pd.concat([kept_limits_df, shift_occupied_durations(new_limits_df, kept_limits_df)])
    updated_limits_df = updated_limits_df.drop(columns=["element_type"])
    updated_limits_df = updated_limits_df[
        updated_limits_df.index.get_level_values("element_id").isin(existing_limits.index.get_level_values("element_id"))
    ]
    network.create_operational_limits(updated_limits_df.reset_index("acceptable_duration"))
    return updated_limits_df
