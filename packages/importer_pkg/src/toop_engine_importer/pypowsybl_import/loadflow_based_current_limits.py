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
from beartype.typing import Literal, Union
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import sort_powsybl_element_frame_by_id
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    DoubleLimitsSetpoint,
    LimitAdjustmentParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.network_masks import NetworkMasks
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

LimitCase = Literal["n0", "n1"]


def create_current_limits_df(
    new_limit_series: pd.Series,
    element_type: Literal["LINE", "BOUNDARY_LINE", "TWO_WINDINGS_TRANSFORMER"],
    side: BranchSide,
    limit_name: str,
    acceptable_duration: int,
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
    limit_name: str
        The name of the new limit
    acceptable_duration: int
        The length of time this limit holds
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
            "name": np.full(n_elements, limit_name),
            "type": np.full(n_elements, "CURRENT"),
            "value": new_limit_series.values,
            "acceptable_duration": np.full(n_elements, acceptable_duration),
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
        type, update_i, i1, i2, old_limit_n0, old_limit_n1, boundary_line1_id, boundary_line2_id
    """
    op_lims = operational_limits.reset_index()
    for side in [BranchSide.ONE, BranchSide.TWO]:
        branches_df[[f"n0_i{side.value}_max", f"n0_group_name_{side.value}"]] = (
            op_lims[(op_lims.name == "permanent_limit") & (op_lims.side == side.name)]
            .groupby("element_id")[["value", "group_name"]]
            .max()
        )
        branches_df[[f"n1_i{side.value}_max", f"n1_group_name_{side.value}"]] = (
            op_lims[(op_lims.name == "N-1") & (op_lims.side == side.name)]
            .groupby("element_id")[["value", "group_name"]]
            .max()
        )
        branches_df[f"n1_i{side.value}_max"] = branches_df[f"n1_i{side.value}_max"].fillna(
            branches_df[f"n0_i{side.value}_max"]
        )
        branches_df[f"n1_group_name_{side.value}"] = branches_df[f"n1_group_name_{side.value}"].fillna(
            branches_df[f"n0_group_name_{side.value}"]
        )

    branches_df[["boundary_line1_id", "boundary_line2_id"]] = tie_lines_df[["boundary_line1_id", "boundary_line2_id"]]
    return branches_df


def compute_optimization_limits(
    flow: pd.Series,
    limit: pd.Series,
    optimized: pd.Series,
    non_worsening: pd.Series,
    double_limits: DoubleLimitsSetpoint,
) -> pd.Series:
    """Compute the optimization limit of every branch and side from its worst flow.

    - Optimized branches above the upper limit get the upper limit.
    - Otherwise non-worsening branches get their flow.
    - Otherwise optimized branches get the larger of the lower limit and their flow.

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
    double_limits: DoubleLimitsSetpoint
        The lower and upper limit, relative to the physical limit.

    Returns
    -------
    pd.Series
        The optimization limits. NaN for branches that are neither optimized nor non-worsening and for branches
        without a flow or a physical limit.
    """
    upper_limit = limit * double_limits.upper
    above_upper_limit = optimized & (flow > upper_limit)
    not_above = np.where(non_worsening, flow, np.maximum(limit * double_limits.lower, flow))
    limits = pd.Series(np.where(above_upper_limit, upper_limit, not_above), index=flow.index)
    return limits.where((optimized | non_worsening) & flow.notna() & limit.notna())


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
    case: LimitCase,
) -> list[pd.DataFrame]:
    """Get new limits for lines based on the current flow.

    Parameters
    ----------
    lines_df: pd.DataFrame
        The lines dataframe with the columns: update_i, old_limit_n0, old_limit_n1
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    case: Literal["n0", "n1"]
        The case being looked at (N-0 or N-1)

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits with the new limits for the lines.
        The new limits are called "loadflow_based_n0" and "loadflow_based_n1"
    """
    if lines_df.empty:
        return []
    update_i = lines_df[["i1", "i2"]].max(axis=1)
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
                limit_name=f"loadflow_based_{case}",
                acceptable_duration=100 if case == "n0" else 200,
                group_names=group_names[~old_limit.isna()],
            )
        )
    return border_line_limits


def get_loadflow_based_tie_line_limits(
    tie_lines_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    case: LimitCase,
) -> list[pd.DataFrame]:
    """Get new limits for tie lines based on the current flow.

    Parameters
    ----------
    tie_lines_df: pd.DataFrame
        The tie lines dataframe with the columns:
        update_i, old_limit_n0, old_limit_n1, boundary_line1_id, boundary_line2_id
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    case: Case
        The case being looked at (N-0 or N-1)

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits with the new limits for the tie lines.
        The new limits are called "loadflow_based_n0" and "loadflow_based_n1"
    """
    if tie_lines_df.empty:
        return []
    border_dangling_limits = []
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
            limit_name=f"loadflow_based_{case}",
            acceptable_duration=100 if case == "n0" else 200,
            group_names=group_names[~old_limit.isna()],
        )
        border_dangling_limits.append(dangling_limit_df)
    return border_dangling_limits


def get_loadflow_based_trafo_limits(
    trafos_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    case: LimitCase,
) -> list[pd.DataFrame]:
    """Get new limits for trafos based on the current flow.

    Parameters
    ----------
    trafos_df: pd.DataFrame
        The trafos dataframe with the columns: update_i, old_limit_n0, old_limit_n1
    limit_parameters: LimitAdjustmentParameters
        The parameters for the calculation of the new limits
    case: Case
        The case being looked at (N-0 or N-1)

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits
        with the new limits for the trafos that are bordering the viewed area.
        The new limits are called "loadflow_based_n0" and "loadflow_based_n1"
    """
    if trafos_df.empty:
        return []

    border_trafo_limits = []
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
            limit_name=f"loadflow_based_{case}",
            acceptable_duration=100 if case == "n0" else 200,
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
            the current limits "old_limit_n0" and "old_limit_n1" and
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
        The new limits are called "loadflow_based_n0" and "loadflow_based_n0"
    """
    lines_df = branches_df[branches_df.type == "LINE"]
    tie_lines_df = branches_df[branches_df.type == "TIE_LINE"]
    limits = []
    cases: tuple[LimitCase, ...] = ("n0", "n1")
    for case in cases:
        limits += get_loadflow_based_line_limits(lines_df[line_tso_border], tso_border_factors, case)
        limits += get_loadflow_based_tie_line_limits(tie_lines_df[tie_line_tso_border], tso_border_factors, case)
    return limits


def get_all_dso_trafo_limits(
    branches_df: pd.DataFrame, dso_trafo_factors: LimitAdjustmentParameters, trafo_dso_border: np.ndarray
) -> list[pd.DataFrame]:
    """Get a list of dataframes with the new limits for the trafos that are bordering the DSO area.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the current i2 and the current limits "old_limit_n0" and "old_limit_n1"
    dso_trafo_factors: LimitAdjustmentParameters
        The parameters (factor and min) for the calculation of the new limits
    trafo_dso_border: np.ndarray
        A boolean mask for the trafos that are bordering the specified area

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits
        with the new limits for the trafos that are bordering the DSO area.
        The new limits are called "loadflow_based_n0" and "loadflow_based_n0"
    """
    trafo_df = sort_powsybl_element_frame_by_id(branches_df[branches_df.type == "TWO_WINDINGS_TRANSFORMER"])
    limits = []
    cases: tuple[LimitCase, ...] = ("n0", "n1")
    for case in cases:
        limits += get_loadflow_based_trafo_limits(trafo_df[trafo_dso_border], dso_trafo_factors, case)
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


def get_optimization_limits_for_case(
    branches_df: pd.DataFrame,
    worst_case: pd.DataFrame,
    double_limits: DoubleLimitsSetpoint,
    limit_case: LimitCase,
    limit_name: str,
) -> list[pd.DataFrame]:
    """Get the optimization limits of the monitored branches for one case.

    Parameters
    ----------
    branches_df: pd.DataFrame
        The branches dataframe with the columns type, i1, i2, optimized, non_worsening, the limits and group names
        of the case and the boundary lines of the tie lines.
    worst_case: pd.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1.
    double_limits: DoubleLimitsSetpoint
        The lower and upper limit, relative to the physical limit.
    limit_case: LimitCase
        The case being looked at (N-0 or N-1)
    limit_name: str
        The name of the limits, extended by the case.

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes in the required format for create_operational_limits.
        The new limits are called "{limit_name}_n0" and "{limit_name}_n1"
    """
    sides = (BranchSide.ONE, BranchSide.TWO)
    flows = {}
    for side in sides:
        worst = worst_case[worst_case["side"] == side.value].set_index("element")
        # Branches without results keep the current of the loadflow
        n_0 = worst["n0"].reindex(branches_df.index).fillna(branches_df[f"i{side.value}"].abs())
        flows[side] = n_0 if limit_case == "n0" else worst["n1"].reindex(branches_df.index).fillna(n_0)

    limit_name = f"{limit_name}_{limit_case}"
    acceptable_duration = 100 if limit_case == "n0" else 200
    optimization_limits = []
    for element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
        elements_df = branches_df[branches_df["type"] == element_type]
        if elements_df.empty:
            continue
        for side in sides:
            new_limit = compute_optimization_limits(
                flow=flows[side][elements_df.index],
                limit=elements_df[f"{limit_case}_i{side.value}_max"],
                optimized=elements_df["optimized"],
                non_worsening=elements_df["non_worsening"],
                double_limits=double_limits,
            )
            optimization_limits.append(
                create_current_limits_df(
                    new_limit,
                    element_type=element_type,
                    side=side,
                    limit_name=limit_name,
                    acceptable_duration=acceptable_duration,
                    group_names=elements_df[f"{limit_case}_group_name_{side.value}"],
                )
            )

    tie_lines_df = branches_df[branches_df["type"] == "TIE_LINE"]
    if not tie_lines_df.empty:
        tie_lines_df = tie_lines_df.assign(
            new_limit=compute_optimization_limits(
                flow=pd.concat(flows, axis=1).max(axis=1)[tie_lines_df.index],
                limit=tie_lines_df[[f"{limit_case}_i1_max", f"{limit_case}_i2_max"]].min(axis=1),
                optimized=tie_lines_df["optimized"],
                non_worsening=tie_lines_df["non_worsening"],
                double_limits=double_limits,
            )
        )
        for side_value, dangling_line_col in zip([1, 2], ["boundary_line1_id", "boundary_line2_id"], strict=True):
            dangling_df = tie_lines_df.set_index(dangling_line_col)
            optimization_limits.append(
                create_current_limits_df(
                    dangling_df["new_limit"],
                    element_type="BOUNDARY_LINE",
                    side=BranchSide.NONE,
                    limit_name=limit_name,
                    acceptable_duration=acceptable_duration,
                    group_names=dangling_df[f"{limit_case}_group_name_{side_value}"],
                )
            )
    return optimization_limits


def create_optimization_limits(
    network: Network,
    nminus1_definition: Nminus1Definition,
    worst_case_currents: pl.DataFrame,
    double_limits: DoubleLimitsSetpoint,
    limit_name: str = "optimization_limit",
) -> pd.DataFrame:
    """Create the optimization limits of the monitored branches in the network.

    The limits replace the limits of the same element, side and name, see compute_optimization_limits.

    Parameters
    ----------
    network: Network
        The network to create the limits for. The loadflow calculation needs to have happened
    nminus1_definition: Nminus1Definition
        The N-1 definition with the optimized and non-worsening monitored branches
    worst_case_currents: pl.DataFrame
        The worst absolute currents with the columns element, side, n0 and n1,
        see extract_worst_case_branch_results_polars
    double_limits: DoubleLimitsSetpoint
        The lower and upper limit, relative to the physical limit.
    limit_name: str
        The name of the limits, extended by the case. The new limits are called "optimization_limit_n0"
        and "optimization_limit_n1" by default.

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
    monitored_branches = [element for element in nminus1_definition.monitored_elements if element.kind == "branch"]
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
    limit_cases: tuple[LimitCase, ...] = ("n0", "n1")
    for limit_case in limit_cases:
        new_limits += get_optimization_limits_for_case(branches_df, worst_case, double_limits, limit_case, limit_name)
    if not new_limits:
        return existing_limits
    new_limits_df = pd.concat(new_limits)

    # Exclude tie lines, since they cant be directly updated
    old_limits_df = existing_limits[existing_limits.element_type != "TIE_LINE"]
    key_columns = ["element_id", "side", "name"]
    replaced = pd.MultiIndex.from_frame(old_limits_df.reset_index()[key_columns]).isin(
        pd.MultiIndex.from_frame(new_limits_df.reset_index()[key_columns])
    )
    updated_limits_df = pd.concat([old_limits_df[~replaced], new_limits_df])
    updated_limits_df = updated_limits_df.drop(columns=["element_type"])
    updated_limits_df = updated_limits_df[
        updated_limits_df.index.get_level_values("element_id").isin(existing_limits.index.get_level_values("element_id"))
    ]
    network.create_operational_limits(updated_limits_df.reset_index("acceptable_duration"))
    return updated_limits_df
