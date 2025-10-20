"""Module contains network analysis functions.

File: network_analysis.py
Author: Nico Westerbeck, Benjamin Petrick
Created: 2024-Q1
"""

from typing import Literal, Optional, Union

import logbook
import numpy as np
import pandas as pd
import pypowsybl
from pypowsybl.network.impl.network import Network
from pypowsybl.security import SecurityAnalysis
from toop_engine_importer.pypowsybl_import import dacf_whitelists, powsybl_masks
from toop_engine_importer.pypowsybl_import.dacf_whitelists import (
    apply_white_list_to_operational_limits,
)
from toop_engine_importer.pypowsybl_import.data_classes import (
    PowsyblSecurityAnalysisParam,
    PreProcessingStatistics,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    UcteImporterParameters,
)

logger = logbook.Logger(__name__)


def create_default_security_analysis_param(
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    ac_run: bool = False,
    current_limit_factor: float = 1.0,
    blacklisted_ids: list[str] | None = None,
) -> PowsyblSecurityAnalysisParam:
    """Create a default PowsyblSecurityAnalysisParam object for the given network.

    Parameters
    ----------
    network: Network
        The network to create the default PowsyblSecurityAnalysisParam object for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters to create the masks with
    ac_run: bool
        If True, the security analysis will be run as AC run, otherwise as DC run.
    current_limit_factor: float
        The factor to adjust the current limit for the security analysis.
        All branches with load > current_limit_factor will be returned
    blacklisted_ids: list[str] | None
        The list of blacklisted ids that should not be part of the reward

    Returns
    -------
    PowsyblSecurityAnalysisParam
        The default PowsyblSecurityAnalysisParam object.
            single_element_contingencies_ids keys:
            "dangling", "generator", "line", "switch", "tie", "transformer", "load"

    """
    masks = powsybl_masks.make_masks(
        network=network,
        importer_parameters=importer_parameters,
        blacklisted_ids=blacklisted_ids,
    )
    default_contigencies = {}
    default_contigencies["dangling"] = (
        network.get_dangling_lines(attributes=[]).index[masks.dangling_line_for_nminus1].to_list()
    )
    default_contigencies["generator"] = network.get_generators(attributes=[]).index[masks.generator_for_nminus1].to_list()
    line_ids = network.get_lines(attributes=[]).index[masks.line_for_nminus1].to_list()

    trafo_ids = network.get_2_windings_transformers(attributes=[]).index[masks.trafo_for_nminus1].to_list()
    tie_ids = network.get_tie_lines(attributes=[]).index[masks.tie_line_for_nminus1].to_list()
    default_contigencies["line"] = line_ids
    default_contigencies["tie"] = tie_ids
    default_contigencies["transformer"] = trafo_ids
    default_contigencies["load"] = []
    default_contigencies["switch"] = []

    bus_mask = powsybl_masks.get_mask_for_area_codes(network.get_voltage_levels(), ["D"], "substation_id", "substation_id")
    monitored_buses = network.get_voltage_levels(attributes=[])[bus_mask].index.tolist()
    monitored_branches = []
    monitored_branches += line_ids
    monitored_branches += trafo_ids
    monitored_branches += tie_ids

    return PowsyblSecurityAnalysisParam(
        single_element_contingencies_ids=default_contigencies,
        current_limit_factor=current_limit_factor,
        monitored_branches=monitored_branches,
        monitored_buses=monitored_buses,
        ac_run=ac_run,
    )


def convert_low_impedance_lines(net: Network, voltage_level_prefix: str, x_threshold_line: float = 0.05) -> pd.DataFrame:
    """Convert all lines in the same voltage level with very low impedance to breakers.

    Parameters
    ----------
    net: Network
        The network to modify. Note: This function modifies the network in place.
    voltage_level_prefix: str
        The prefix of the voltage level to consider.
    x_threshold_line: float
        The threshold for x, everything below will be converted.

    Returns
    -------
    low_impedance_lines: pd.DataFrame
        The lines that were converted to breakers.

    """
    lines = net.get_lines(all_attributes=True)
    low_impedance_lines = lines[
        (lines["voltage_level1_id"] == lines["voltage_level2_id"])
        & (lines["x"] <= x_threshold_line)
        & (lines["voltage_level1_id"].str.startswith(voltage_level_prefix))
        & (lines["connected1"] & lines["connected2"])
    ]
    net.remove_elements(low_impedance_lines.index)
    low_impedance_lines = low_impedance_lines[
        [
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "elementName",
            "voltage_level1_id",
        ]
    ].rename(
        columns={
            "bus_breaker_bus1_id": "bus1_id",
            "bus_breaker_bus2_id": "bus2_id",
            "elementName": "name",
            "voltage_level1_id": "voltage_level_id",
        }
    )
    low_impedance_lines["kind"] = "BREAKER"
    low_impedance_lines["open"] = False
    low_impedance_lines["retained"] = True
    net.create_switches(low_impedance_lines)
    return low_impedance_lines


def remove_branches_across_switch(net: Network) -> pd.DataFrame:
    """Remove all branches that span across a closed switch, i.e. have the same from+to bus.

    Parameters
    ----------
    net: Network
        The network to modify. Note: This function modifies the network in place.

    Returns
    -------
    to_remove: pd.DataFrame
        The branches that were removed.

    """
    # remove branches that span across a closed switch
    # the bus1_id == bus2_id, in case the branch is a closed switch
    # in case no switch is between the buses, bus1_id != bus2_id
    # -> a line between two buses is not removed, if there is no switch between them
    to_remove = net.get_branches()[
        (net.get_branches()["bus1_id"] == net.get_branches()["bus2_id"])
        & (net.get_branches()["connected1"] & net.get_branches()["connected2"])
    ]
    net.remove_elements(to_remove.index)
    return to_remove


def get_voltage_level_for_df(  # noqa: C901
    net: Network,
    kind: Literal["line", "trafo", "switch", "bus", "generator", "load", "dangling", "tie"],
) -> pd.DataFrame:
    """Get a DataFrame with the voltage levels for the given kind.

    Parameters
    ----------
    net: Network
        The network to get the voltage levels from.
    kind: Literal["line", "trafo", "switch", "bus", "generator", "load", "dangling", "tie"]
        The kind of element to get the voltage levels for.

    Returns
    -------
    merged_df: pd.DataFrame
        The DataFrame with the voltage levels for the given kind.

    Raises
    ------
    ValueError
        If the kind is not valid. Must be either "line", "trafo", "switch", "bus", "generator", "load", "dangling", "tie".

    """
    left_on = "voltage_level_id"
    if kind == "line":
        left_df = net.get_lines(all_attributes=True)
        left_on = "voltage_level1_id"
    elif kind == "trafo":
        left_df = net.get_2_windings_transformers(all_attributes=True)
        left_on = "voltage_level1_id"
    elif kind == "switch":
        left_df = net.get_switches(all_attributes=True)
    elif kind == "bus":
        left_df = net.get_buses(all_attributes=True)
    elif kind == "generator":
        left_df = net.get_generators(all_attributes=True)
    elif kind == "load":
        left_df = net.get_loads(all_attributes=True)
    elif kind == "dangling":
        left_df = net.get_dangling_lines(all_attributes=True)
    elif kind == "tie":
        left_df = net.get_dangling_lines(all_attributes=True)
    else:
        raise ValueError(
            "Invalid kind. Must be one of: ['line', 'trafo', 'switch', 'bus', 'generator', 'load', 'dangling', 'tie']."
            + f" Got: '{kind}'."
        )

    merged_df = pd.merge(
        left=left_df,
        right=net.get_voltage_levels(),
        left_on=left_on,
        right_on="id",
        how="left",
        suffixes=("", "_1"),
    )
    if kind == "trafo":
        left_on = "voltage_level2_id"
        merged_df = pd.merge(
            left=merged_df,
            right=net.get_voltage_levels(),
            left_on=left_on,
            right_on="id",
            how="left",
            suffixes=("", "_2"),
        )
        merged_df["nominal_v_max"] = np.maximum(merged_df["nominal_v"], merged_df["nominal_v_2"])
    elif kind == "tie":
        merged_df.index = left_df.index
        left_on = "dangling_line1_id"
        left_df = net.get_tie_lines(all_attributes=True)
        merged_df = pd.merge(
            left=left_df,
            right=merged_df,
            left_on=left_on,
            right_on="id",
            how="left",
            suffixes=("", "_2"),
        )
        merged_df.drop(
            columns=[
                "name_2",
                "fictitious_2",
                "geographicalName_2",
                "pairing_key_2",
                "ucte_xnode_code_2",
                "status_XNode_2",
                "elementName_2",
            ],
            inplace=True,
        )
    merged_df.index = left_df.index
    return merged_df


def run_n1_analysis(
    network: Network,
    security_analysis_param: PowsyblSecurityAnalysisParam,
) -> pypowsybl.security.SecurityAnalysis:
    """Run security analysis for the network.

    Parameters
    ----------
    network: Network
        The network to run the security analysis on.
    security_analysis_param: PowsyblSecurityAnalysisParam
        The parameters for the security analysis.

    Returns
    -------
    n1_res: pypowsybl.security.SecurityAnalysisResult
        The result of the security analysis.

    """
    set_new_operational_limit(network=network, factor=security_analysis_param.current_limit_factor)

    # run security analysis
    analysis = pypowsybl.security.create_analysis()

    for element in security_analysis_param.single_element_contingencies_ids.values():
        analysis.add_single_element_contingencies(element)

    analysis.add_monitored_elements(branch_ids=security_analysis_param.monitored_branches)
    analysis.add_monitored_elements(voltage_level_ids=security_analysis_param.monitored_buses)

    # run security analysis
    if security_analysis_param.ac_run:
        n1_res = analysis.run_ac(network)
    else:
        n1_res = analysis.run_dc(network)

    set_new_operational_limit(network=network, factor=1 / security_analysis_param.current_limit_factor)

    return n1_res


def convert_tie_to_dangling(violation_df: pd.DataFrame) -> pd.DataFrame:
    """Convert all tie lines to dangling lines by splitting them into two lines.

    Note: This function expects the UCTE format for the tie lines.
    e.g. "ABCDEF11 XAB_CD21 1 + XAB_CD21 AA_BBB11 1"

    Parameters
    ----------
    violation_df : pd.DataFrame
        The violation dataframe

    Returns
    -------
    pd.DataFrame
        The violation dataframe with all tie lines converted to dangling lines
    """
    ucte_name_len_tie_line = 41
    condition_tie = violation_df.index.get_level_values(1).str.len() == ucte_name_len_tie_line
    violation_df_side_one = violation_df[condition_tie & (violation_df["side"] == "ONE")].copy()
    # "ABCDEF11 XAB_CD21 1 + XAB_CD21 AA_BBB11 1" -> "ABCDEF11 XAB_CD21 1"
    violation_df_side_one["index"] = violation_df_side_one.index.get_level_values(1).str[0:19]
    index_arrays = [
        violation_df_side_one.index.get_level_values(0),
        violation_df_side_one["index"],
    ]
    multi_index = pd.MultiIndex.from_arrays(index_arrays, names=("contingency_id", "subject_id"))
    violation_df_side_one.set_index(multi_index, inplace=True)

    condition_tie = violation_df.index.get_level_values(1).str.len() == ucte_name_len_tie_line
    violation_df_side_two = violation_df[condition_tie & (violation_df["side"] == "TWO")].copy()
    # "ABCDEF11 XAB_CD21 1 + XAB_CD21 AA_BBB11 1" -> "XAB_CD21 AA_BBB11 1"
    violation_df_side_two["index"] = violation_df_side_two.index.get_level_values(1).str[22:]
    index_arrays = [
        violation_df_side_two.index.get_level_values(0),
        violation_df_side_two["index"],
    ]
    multi_index = pd.MultiIndex.from_arrays(index_arrays, names=("contingency_id", "subject_id"))
    violation_df_side_two.set_index(multi_index, inplace=True)

    violation_df = pd.concat([violation_df, violation_df_side_one, violation_df_side_two]).drop(columns=["index"])
    return violation_df


def get_branches_with_dangling_lines(network: Network) -> pd.DataFrame:
    """Get a DataFrame with all branches from network.get_branches(all_attributes=True) and adds dangling lines to it.

    Parameters
    ----------
    network : Network
        The network object containing the network data

    Returns
    -------
    branches: pd.DataFrame
        A dataframe with all branches and dangling lines
        type = DANGLING_LINE
    """
    branches = network.get_branches(all_attributes=True)
    dangling = network.get_dangling_lines(all_attributes=True)
    dangling["type"] = "DANGLING_LINE"
    dangling["voltage_level1_id"] = dangling["voltage_level_id"]
    dangling["voltage_level2_id"] = dangling["voltage_level_id"]
    dangling["bus1_id"] = dangling["bus_id"]
    dangling["bus2_id"] = dangling["bus_id"]
    dangling["connected1"] = dangling["connected"]
    dangling["connected2"] = dangling["connected"]
    dangling["bus_breaker_bus1_id"] = dangling.index.str[0:8]
    dangling["bus_breaker_bus2_id"] = dangling.index.str[9:17]
    dangling = dangling[
        [
            "type",
            "voltage_level1_id",
            "bus_breaker_bus1_id",
            "bus1_id",
            "connected1",
            "voltage_level2_id",
            "bus_breaker_bus2_id",
            "bus2_id",
            "connected2",
        ]
    ]
    branches = pd.concat([branches, dangling])
    return branches


def add_element_name_to_branches_df(network: Network, branches_df: pd.DataFrame) -> pd.DataFrame:
    """Add the elementName to the branches_df.

    Merging elementName with the line, trafo, tie and dangling DataFrame from the network.

    Parameters
    ----------
    network : Network
        The network object containing the network data
    branches_df : pd.DataFrame
        The dataframe with the branches

    Returns
    -------
    pd.DataFrame
        The dataframe with the elementName added
    """
    tie_lines = network.get_tie_lines(all_attributes=True)
    if "elementName_1" in tie_lines.columns and "elementName_2" in tie_lines.columns:
        tie_lines["elementName"] = tie_lines["elementName_1"] + " + " + tie_lines["elementName_2"]
    else:
        tie_lines["elementName"] = ""

    trafo = network.get_2_windings_transformers(all_attributes=True)
    if "elementName" not in trafo.columns:
        trafo["elementName"] = ""

    lines = network.get_lines(all_attributes=True)
    if "elementName" not in lines.columns:
        lines["elementName"] = ""
    dangling = network.get_dangling_lines(all_attributes=True)
    if "elementName" not in dangling.columns:
        dangling["elementName"] = ""

    concat_branches = pd.concat(
        [
            trafo["elementName"],
            lines["elementName"],
            tie_lines["elementName"],
            dangling["elementName"],
        ]
    )
    branches_df = pd.merge(branches_df, concat_branches, left_on="id", right_index=True, how="left")
    return branches_df


def merge_voltage_levels_to_branches_df(network: Network, branches_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the voltage levels to the branches_df by merging it with the voltage_levels DataFrame from the network.

    Parameters
    ----------
    network : Network
        The network object containing the network data
    branches_df : pd.DataFrame
        The dataframe with the branches

    Returns
    -------
    pd.DataFrame
        The dataframe with the voltage levels added
    """
    # get nominal voltage
    voltage_levels = network.get_voltage_levels()[["nominal_v"]]
    voltage_levels.rename(columns={"nominal_v": "nominal_v1"}, inplace=True)
    branches_df = pd.merge(
        branches_df,
        voltage_levels,
        how="left",
        left_on=["voltage_level1_id"],
        right_index=True,
    )
    # get norminal voltage bus 2 -> relevant for transformer
    voltage_levels.rename(columns={"nominal_v1": "nominal_v2"}, inplace=True)
    branches_df = pd.merge(
        branches_df,
        voltage_levels,
        how="left",
        left_on=["voltage_level2_id"],
        right_index=True,
    )
    return branches_df


def merge_vmag_and_vangle_to_branches_df(
    security_analysis_results: SecurityAnalysis, violation_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge the v_mag and v_angle to the branches_df.

    Merging the bus_results DataFrame with the security_analysis_results.

    Parameters
    ----------
    security_analysis_results : SecurityAnalysis
        The security analysis results
    violation_df : pd.DataFrame
        The dataframe with the branches
        expected columns: 'contingency_id', 'voltage_level1_id', 'bus_breaker_bus1_id',
                          'voltage_level2_id', 'bus_breaker_bus2_id'

    Returns
    -------
    pd.DataFrame
        The dataframe with the v_mag and v_angle added

    """
    # merge with buses from network -> get bus data v_mag, v_angle for each bus
    moitored_buses_res = security_analysis_results.bus_results
    moitored_buses_res.rename(columns={"v_mag": "v_mag1", "v_angle": "v_angle1"}, inplace=True)
    # reset index, so subject_id will not be lost
    violation_df.reset_index(inplace=True, level=1)
    violation_df = pd.merge(
        violation_df,
        moitored_buses_res,
        how="left",
        left_on=["contingency_id", "voltage_level1_id", "bus_breaker_bus1_id"],
        right_on=["contingency_id", "voltage_level_id", "bus_id"],
    )
    # get voltage for bus 2
    moitored_buses_res.rename(columns={"v_mag1": "v_mag2", "v_angle1": "v_angle2"}, inplace=True)
    violation_df = pd.merge(
        violation_df,
        moitored_buses_res,
        how="left",
        left_on=["contingency_id", "voltage_level2_id", "bus_breaker_bus2_id"],
        right_on=["contingency_id", "voltage_level_id", "bus_id"],
    )
    index_arrays = [violation_df.index.get_level_values(0), violation_df["subject_id"]]
    multi_index = pd.MultiIndex.from_arrays(index_arrays, names=("contingency_id", "subject_id"))
    violation_df.set_index(multi_index, inplace=True)
    return violation_df


def get_all_data_from_violation_df_in_one_dataframe(
    network: Network,
    security_analysis_results: SecurityAnalysis,
    all_attributes: bool = False,
    current_limit_factor: float = 1.0,
    add_dangling: bool = False,
) -> pd.DataFrame:
    """Get a violation_df with all bus and branch information.

    It adds to violation the following data:
    - merge with SecurityAnalysis.branch_results
    -> get power flows for each branch e.g. p, q, i
    - merge with branches from network
    -> get branch data e.g. voltage levels, bus_breaker_bus1_id, elementName
    - merge with SecurityAnalysis.bus_results
    -> get bus data v_mag, v_angle for each bus
    - merge with voltage levels from network
    -> get nominal voltage for each voltage level
    -> transformer will have both voltage levels included
    - optionally convert tie lines to dangling lines and include them in all merged data
    - calculate pmax -> use nominal_v1, as this is by definition the max value on the higher voltage level
    - calculate I percent

    Parameters
    ----------
    network : Network
        The network object containing the network data
    security_analysis_results : SecurityAnalysis
        The results of the security analysis
        Note: monitored branches and busses are used -> make sure all elements in the violation are monitored
    all_attributes : bool, optional
        If True all attributes of the branches are included, by default False
        Reduced Attributes are:
                ['imax', 'i_violation', 'side', 'p1',
            'q1', 'i1', 'p2', 'q2', 'i2', 'type', 'voltage_level1_id', 'bus_breaker_bus1_id',
            'voltage_level2_id', 'bus_breaker_bus2_id', 'elementName',
            'v_mag1', 'v_angle1', 'v_mag2', 'v_angle2', 'nominal_v1', 'nominal_v2',
            'pmax', 'I%']
        All Attributes are:
                ['subject_name', 'limit_type', 'limit_name', 'imax',
            'acceptable_duration', 'limit_reduction', 'i_violation', 'side', 'p1',
            'q1', 'i1', 'p2', 'q2', 'i2', 'flow_transfer', 'type',
            'voltage_level1_id', 'bus_breaker_bus1_id', 'bus1_id', 'connected1',
            'voltage_level2_id', 'bus_breaker_bus2_id', 'bus2_id', 'connected2',
            'elementName', 'v_mag1', 'v_angle1', 'v_mag2', 'v_angle2', 'nominal_v1',
            'nominal_v2', 'pmax', 'I%']
    current_limit_factor : float, optional
        The current limit factor is used to adjust the limit for all violations, by dividing by this factor
    add_dangling : bool, optional
        If True, tie lines are converted to dangling lines, by splitting them into two lines, by default False

    Returns
    -------
    pd.DataFrame
        A dataframe containing all relevant data for each violation


    """
    branch_res = security_analysis_results.branch_results.droplevel("operator_strategy_id")
    branch_res.index.names = ["contingency_id", "subject_id"]
    violations_df = security_analysis_results.limit_violations
    violations_df.rename(columns={"limit": "imax", "value": "i_violation"}, inplace=True)
    violations_df["imax"] = violations_df["imax"].div(current_limit_factor)
    if add_dangling:
        violations_df = convert_tie_to_dangling(violations_df)
    violations_df = drop_one_side_from_violation_df(violations_df, column_name="i_violation")

    # merge violations with branch results -> get power flows for each branch e.g. p, q, i
    df_merged = pd.merge(violations_df, branch_res, left_index=True, right_index=True, how="left")

    # merge with branches from network -> get branch data e.g. voltage levels, bus_breaker_bus1_id, elementName
    # get dangling, elementName, voltage levels
    network_branch = get_branches_with_dangling_lines(network=network)
    network_branch = add_element_name_to_branches_df(network=network, branches_df=network_branch)
    network_branch = merge_voltage_levels_to_branches_df(network=network, branches_df=network_branch)

    # get generators
    generators_df = network.get_generators(all_attributes=True).rename({"bus_breaker_bus_id": "bus_breaker_bus1_id"}, axis=1)
    generators_df["bus_breaker_bus1_id"] = generators_df["bus_breaker_bus1_id"].str.pad(8, side="right", fillchar=" ")
    generators_df["type"] = "GENERATOR"

    # Add element info to the critical branches
    df_merged = pd.merge(
        left=df_merged,
        right=network_branch,
        left_on="subject_id",
        right_index=True,
        how="left",
        suffixes=("", "_1"),
    )
    # Add element info to the critical outages
    df_merged = pd.merge(
        df_merged,
        pd.concat([network_branch, generators_df])[["bus_breaker_bus1_id", "bus_breaker_bus2_id", "elementName", "type"]],
        left_on="contingency_id",
        right_index=True,
        how="left",
        suffixes=("", "_contingency"),
    )

    # Add voltage infos
    df_merged = merge_vmag_and_vangle_to_branches_df(
        security_analysis_results=security_analysis_results, violation_df=df_merged
    )

    # imax is based on the voltage level 2 or busbreaker1_id
    df_merged["v_mag_calc"] = np.fmax(df_merged["v_mag1"], df_merged["v_mag2"])
    df_merged["nominal_v_calc"] = np.fmax(df_merged["nominal_v1"], df_merged["nominal_v2"])
    condition_trafo = df_merged["type"] == "TWO_WINDINGS_TRANSFORMER"
    df_merged.loc[condition_trafo, "v_mag_calc"] = df_merged.loc[condition_trafo, "v_mag2"]
    df_merged.loc[condition_trafo, "nominal_v_calc"] = df_merged.loc[condition_trafo, "nominal_v2"]
    df_merged["smax"] = df_merged["imax"] * df_merged["v_mag_calc"] * np.sqrt(3) / 1000
    df_merged["smax_nominal"] = df_merged["imax"] * df_merged["nominal_v_calc"] * np.sqrt(3) / 1000

    df_merged["smax"] = df_merged["smax"].round(0)
    df_merged["smax_nominal"] = df_merged["smax_nominal"].round(0)
    df_merged["I%"] = df_merged["i_violation"] / df_merged["imax"] * 100

    if not all_attributes:
        df_merged = df_merged[
            [
                "imax",
                "i_violation",
                "side",
                "p1",
                "q1",
                "i1",
                "p2",
                "q2",
                "i2",
                "type",
                "voltage_level1_id",
                "bus_breaker_bus1_id",
                "voltage_level2_id",
                "bus_breaker_bus2_id",
                "elementName",
                "bus_breaker_bus1_id_contingency",
                "bus_breaker_bus2_id_contingency",
                "elementName_contingency",
                "type_contingency",
                "v_mag1",
                "v_angle1",
                "v_mag2",
                "v_angle2",
                "nominal_v1",
                "nominal_v2",
                "smax",
                "smax_nominal",
                "I%",
            ]
        ]
    else:
        drop_col = ["p1_1", "q1_1", "i1_1", "p2_1", "q2_1", "i2_1"]
        df_merged.drop(columns=drop_col, inplace=True)

    df_merged.sort_index(inplace=True)
    return df_merged


def drop_one_side_from_violation_df(violation_df: pd.DataFrame, column_name: str = "i_violation") -> pd.DataFrame:
    """Drop the side ONE or TWO with the lower value, default is current, from the violation dataframe.

    Parameters
    ----------
    violation_df : pd.DataFrame
        The violation dataframe
    column_name : str, optional
        The column name to drop the side from, by default "i_violation"

    Returns
    -------
    pd.DataFrame
        The violation dataframe with the side dropped

    """
    violation_df = violation_df.sort_values(column_name, ascending=False)
    violation_df = violation_df[~violation_df.index.duplicated(keep="first")]
    violation_df = violation_df.sort_index()
    return violation_df


def calc_total_overload(
    violation_df: pd.DataFrame,
    column_case_name_of_n1: str = "subject_id",
    overload_column: str = "overload",
) -> float:
    """Calculate the total overload from a dataframe.

    Parameters
    ----------
    violation_df : pd.DataFrame
        The N0 violations
    column_case_name_of_n1 : str, optional
        The column name of the case name of the n1 violations. Default is "subject_id".
    overload_column : str, optional
        The column name of the overload. Default is "overload".

    Returns
    -------
    overload : float
        The total overload

    """
    overload = violation_df.groupby(column_case_name_of_n1).agg({overload_column: "max"}).sum().values[0]
    return overload


def get_voltage_angle(monitored_busses_df: pd.DataFrame) -> dict:
    """Get the voltage angle between two monitored busses.

    Parameters
    ----------
    monitored_busses_df : pd.DataFrame
        The monitored busses with the voltage angle

    Returns
    -------
    statistics : dict
        The statistics of the voltage angle between the monitored busses

    """
    v1 = 0
    v2 = 0
    diff = 0
    statistics = {}  # type: dict[str, dict]
    monitored_busses = monitored_busses_df.index.get_level_values("voltage_level_id").drop_duplicates()
    # get station key
    for key in monitored_busses:
        statistics[key] = {}
        diff = 0
        bus1 = ""
        bus2 = ""
        # find all voltage angles for the station (key)
        for index, row in monitored_busses_df[
            monitored_busses_df.index.get_level_values("voltage_level_id") == key
        ].iterrows():
            # define bus1 and bus2
            if bus1 == "":
                bus1 = index[-1]
            elif bus2 == "":
                bus2 = index[-1]

            # this setup of iteration catches if the df has more than 2 busses
            if index[-1] == bus1:
                v1 = row["v_angle"]
            elif index[-1] == bus2:
                v2 = row["v_angle"]

                # check if the difference is bigger than the current difference
                if abs(v1 - v2) > diff:
                    diff = abs(v1 - v2)
                    statistics[key][bus1] = v1
                    statistics[key][bus2] = v2
                    statistics[key]["diff"] = round(diff, 1)
            else:
                raise ValueError(f"Bus not found: {index[-1]}, expected: {bus1} or {bus2}")
    return statistics


def set_new_operational_limit(
    network: Network,
    value: Optional[float] = None,
    id_list: Optional[list[str]] = None,
    factor: float = 1.0,
) -> None:
    """Multiply the operational limits by a factor.

    Parameters
    ----------
    network : pypowsybl.network.impl.network.Network
        The network object. Note: This function modifies the network in place.
    factor : float, optional
        The factor to multiply the operational limits with
        Note: This factor is always applied to the value of the operational limit
    value : float, optional
        The value to set the operational limit to
    id_list : list[str], optional
        The list of ids to apply the factor to
    """
    op_lim = network.get_operational_limits(
        attributes=["element_type", "side", "name", "type", "value", "acceptable_duration", "fictitious"]
    )
    if id_list is not None:
        op_lim = op_lim[op_lim.index.isin(id_list)]
    # remove tie lines, as they can't be set
    op_lim = op_lim[op_lim["element_type"] != "TIE_LINE"]
    if value is not None:
        op_lim["value"] = value
    op_lim["value"] = op_lim["value"].mul(factor).round(2)
    # drop element_type column -> deprecated
    op_lim.drop(columns=["element_type"], inplace=True)
    network.create_operational_limits(op_lim)


def get_branches_df_with_element_name(network: Network) -> pd.DataFrame:
    """Get the branches with the element name.

    Parameters
    ----------
    network : Network
        The network object

    Returns
    -------
    pd.DataFrame
        The branches with the element name

    """
    branches = network.get_branches(all_attributes=True)
    lines = network.get_lines(all_attributes=True)["elementName"]
    trafos = network.get_2_windings_transformers(all_attributes=True)["elementName"]
    tie_lines = network.get_tie_lines(all_attributes=True)[["elementName_1", "elementName_2", "pairing_key"]]
    tie_lines["elementName"] = tie_lines["elementName_1"] + " + " + tie_lines["elementName_2"]
    tie_lines = tie_lines[["elementName", "pairing_key"]]

    branches = branches.merge(lines, how="left", on="id", suffixes=("", "_1"))
    branches = branches.merge(trafos, how="left", on="id", suffixes=("", "_2"))
    branches = branches.merge(tie_lines, how="left", on="id", suffixes=("", "_3"))
    branches["elementName"] = branches["elementName"].combine_first(branches["elementName_2"])
    branches["elementName"] = branches["elementName"].combine_first(branches["elementName_3"])
    branches = branches.drop(columns=["elementName_2", "elementName_3"])
    return branches


def apply_cb_lists(
    network: Network,
    statistics: PreProcessingStatistics,
    ucte_importer_parameters: UcteImporterParameters,
) -> PreProcessingStatistics:
    """Run the black or white list to the powsybl network.

    Parameters
    ----------
    network : Network
        The network to modify. Note: The network is modified in place.
    statistics : ProcessingStatistics
        The statistics to fill with the id lists of the black and white list
        Note: The statistics are modified in place.
    ucte_importer_parameters : UcteImporterParameters
        Parameters that are required to import the data from a UCTE file. This will utilize
        powsybl and the powsybl backend to the loadflow solver

    Returns
    -------
    statistics: ProcessingStatistics
        The statistics with the id lists of the black and white list

    """
    branches_with_elementname = get_branches_df_with_element_name(network)
    branches_with_elementname["pairing_key"] = branches_with_elementname["pairing_key"].str[0:7]
    # get only the rows needed
    branches_with_elementname = branches_with_elementname[
        powsybl_masks.get_mask_for_area_codes(branches_with_elementname, ["D"], "voltage_level1_id", "voltage_level2_id")
    ]
    op_lim = network.get_operational_limits(attributes=[]).index.to_list()
    branches_with_elementname = branches_with_elementname[branches_with_elementname.index.isin(op_lim)]

    if ucte_importer_parameters.white_list_file is not None:
        white_list_df = pd.read_csv(ucte_importer_parameters.white_list_file, delimiter=";").fillna("")
        dacf_whitelists.assign_element_id_to_cb_df(cb_df=white_list_df, branches_with_elementname=branches_with_elementname)
        statistics.import_result.n_white_list = len(white_list_df)
        white_list_df = white_list_df[white_list_df["element_id"].notnull()]
        apply_white_list_to_operational_limits(network, white_list_df)
        statistics.id_lists["white_list"] = white_list_df["element_id"].to_list()
        statistics.import_result.n_white_list_applied = len(white_list_df["element_id"])
    else:
        statistics.id_lists["white_list"] = []
    if ucte_importer_parameters.black_list_file is not None:
        black_list_df = pd.read_csv(ucte_importer_parameters.black_list_file, delimiter=";").fillna("")
        dacf_whitelists.assign_element_id_to_cb_df(cb_df=black_list_df, branches_with_elementname=branches_with_elementname)
        statistics.import_result.n_black_list = len(black_list_df)
        black_list_df = black_list_df[black_list_df["element_id"].notnull()]
        statistics.id_lists["black_list"] = black_list_df["element_id"].to_list()
        statistics.import_result.n_black_list_applied = len(black_list_df["element_id"])
    else:
        statistics.id_lists["black_list"] = []
    return statistics


def remove_branches_with_same_bus(network: Network) -> None:
    """Remove branches that have the same bus.

    Note: This function should move into the Backend at some point.

    Parameters
    ----------
    network : Network
        The network to modify.
        Note: This function modifies the network in place.
    """
    branches = network.get_branches()
    cond_same_bus_id = branches["bus1_id"] == branches["bus2_id"]
    cond_has_bus_id = branches["bus1_id"]
    same_bus_branches = branches[cond_same_bus_id & cond_has_bus_id].index

    network.remove_elements(list(same_bus_branches))

    logger.warning(
        f"Removed {len(same_bus_branches)} branches with the same bus id. Please check the network for inconsistencies."
    )
