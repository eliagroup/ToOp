"""Module contains functions to create artificical operational limits based on the loadflow result for the PowSyBl backend.

File: loadflow_based_current_limits.py
Author:  Leonard Hilfrich
Created: 2024-12-19
"""

from typing import Literal, Union

import numpy as np
import pandas as pd
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.powsybl_masks import NetworkMasks
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    LimitAdjustmentParameters,
    UcteImporterParameters,
)

Case = Literal["n0", "n1"]


def create_current_limits_df(
    new_limit_series: pd.Series,
    element_type: Literal["LINE", "DANGLING_LINE", "TWO_WINDINGS_TRANSFORMER"],
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
    element_type: Literal["LINE", "DANGLING_LINE", "TWO_WINDINGS_TRANSFORMER"]
        The type of the element. For Tielines always use the corresponding Dangling Lines
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
            "side": np.full(n_elements, side),
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
        The tie lines dataframe with the columns: dangling_line1_id, dangling_line2_id

    Returns
    -------
    pd.DataFrame
        The branches dataframe with the columns:
        type, update_i, i1, i2, old_limit_n0, old_limit_n1, dangling_line1_id, dangling_line2_id
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

    branches_df[["dangling_line1_id", "dangling_line2_id"]] = tie_lines_df[["dangling_line1_id", "dangling_line2_id"]]
    return branches_df


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
    case: Case,
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
                side=side.name,
                limit_name=f"loadflow_based_{case}",
                acceptable_duration=100 if case == "n0" else 200,
                group_names=group_names[~old_limit.isna()],
            )
        )
    return border_line_limits


def get_loadflow_based_tie_line_limits(
    tie_lines_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    case: Case,
) -> list[pd.DataFrame]:
    """Get new limits for tie lines based on the current flow.

    Parameters
    ----------
    tie_lines_df: pd.DataFrame
        The tie lines dataframe with the columns:
        update_i, old_limit_n0, old_limit_n1, dangling_line1_id, dangling_line2_id
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
    tie_lines_df["update_i"] = tie_lines_df[["i1", "i2"]].max(axis=1)
    for side_value, dangling_line_col in zip([1, 2], ["dangling_line1_id", "dangling_line2_id"], strict=True):
        dangling_df = tie_lines_df.set_index(dangling_line_col)
        old_limit = dangling_df[[f"{case}_i1_max", f"{case}_i2_max"]].min(axis=1)

        new_limit = get_new_limits_for_branch(
            dangling_df["update_i"], old_limit, *limit_parameters.get_parameters_for_case(case)
        )
        group_names = dangling_df[f"{case}_group_name_{side_value}"]
        dangling_limit_df = create_current_limits_df(
            new_limit[~old_limit.isna()],
            element_type="DANGLING_LINE",
            side="NONE",
            limit_name=f"loadflow_based_{case}",
            acceptable_duration=100 if case == "n0" else 200,
            group_names=group_names[~old_limit.isna()],
        )
        border_dangling_limits.append(dangling_limit_df)
    return border_dangling_limits


def get_loadflow_based_trafo_limits(
    trafos_df: pd.DataFrame,
    limit_parameters: LimitAdjustmentParameters,
    case: Case,
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
            side=side.name,
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
            the dangling_lines "dangling_line1_id", "dangling_line2_id"
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
    cases: tuple[Case, ...] = ("n0", "n1")
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
    trafo_df = branches_df[branches_df.type == "TWO_WINDINGS_TRANSFORMER"]
    limits = []
    cases: tuple[Case, ...] = ("n0", "n1")
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
        network.get_tie_lines(attributes=["dangling_line1_id", "dangling_line2_id"]),
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
