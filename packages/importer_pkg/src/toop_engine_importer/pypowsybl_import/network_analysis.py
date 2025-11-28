"""Module contains network analysis functions.

File: network_analysis.py
Author: Nico Westerbeck, Benjamin Petrick
Created: 2024-Q1
"""

import logbook
import pandas as pd
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import import dacf_whitelists, powsybl_masks
from toop_engine_importer.pypowsybl_import.dacf_whitelists import (
    apply_white_list_to_operational_limits,
)
from toop_engine_importer.pypowsybl_import.data_classes import (
    PreProcessingStatistics,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    UcteImporterParameters,
)

logger = logbook.Logger(__name__)


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
    op_lim = network.get_operational_limits(attributes=[]).index.get_level_values("element_id").to_list()
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

    if len(same_bus_branches) > 0:
        logger.warning(
            f"Removed {len(same_bus_branches)} branches with the same bus id. Please check the network for inconsistencies."
            f" Removed branches: {same_bus_branches.tolist()}"
        )
