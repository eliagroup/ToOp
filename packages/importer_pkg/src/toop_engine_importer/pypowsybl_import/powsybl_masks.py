# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to create masks for the PowSyBl backend.

File: powsybl_masks.py
Author:  Benjamin Petrick
Created: 2024-08-13
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Union

import logbook
import numpy as np
import pandas as pd
import pypowsybl
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    DISTRIBUTED_SLACK,
    SINGLE_SLACK,
)
from toop_engine_importer.contingency_from_power_factory.contingency_from_file import (
    get_contingencies_from_file,
    match_contingencies,
    match_contingencies_with_suffix,
)
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import get_region_for_df
from toop_engine_importer.pypowsybl_import.cgmes.powsybl_masks_cgmes import get_switchable_buses_cgmes
from toop_engine_importer.pypowsybl_import.contingency_from_file.contingency_file_models import ContingencyImportSchema
from toop_engine_importer.pypowsybl_import.contingency_from_file.helper_functions import get_all_element_names
from toop_engine_importer.pypowsybl_import.ucte.powsybl_masks_ucte import get_switchable_buses_ucte
from toop_engine_interfaces.filesystem_helper import save_numpy_filesystem
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    UcteImporterParameters,
)

logger = logbook.Logger(__name__)


@dataclass(frozen=True)
class NetworkMasks:
    """Class to hold the network masks.

    See class PowsyblBackend(BackendInterface) in DCLoadflowsolver for more information.
    """

    relevant_subs: np.ndarray
    """relevant_subs.npy (a boolean mask of relevant nodes)."""

    line_for_nminus1: np.ndarray
    """line_for_nminus1.npy (a boolean mask of lines that are relevant for n-1)."""

    line_for_reward: np.ndarray
    """line_for_reward.npy (a boolean mask of lines that are relevant for the reward)."""

    line_overload_weight: np.ndarray
    """line_overload_weight.npy (a float mask of weights for the overload)."""

    line_disconnectable: np.ndarray
    """line_disconnectable.npy (a boolean mask of lines that can be disconnected)"""

    line_blacklisted: np.ndarray
    """line_blacklisted.npy (a boolean mask of lines that are blacklisted)
    Currently only used during importing and not part of the PowsyblBackend"""

    line_tso_border: np.ndarray
    """line_tso_border.npy (a boolean mask of lines that are leading to TSOs outside the reward area)
    Currently only used during importing and not part of the PowsyblBackend"""

    trafo_for_nminus1: np.ndarray
    """trafo_for_nminus1.npy (a boolean mask of transformers that are relevant for n-1)."""

    trafo_for_reward: np.ndarray
    """trafo_for_reward.npy (a boolean mask of transformers that are relevant for the reward)."""

    trafo_overload_weight: np.ndarray
    """trafo_overload_weight.npy (a float mask of weights for the overload)."""

    trafo_disconnectable: np.ndarray
    """trafo_disconnectable.npy (a boolean mask of transformers that can be disconnected)"""

    trafo_blacklisted: np.ndarray
    """trafo_blacklisted.npy (a boolean mask of transformers that are blacklisted)
    Currently only used during importing and not part of the PowsyblBackend"""

    trafo_n0_n1_max_diff_factor: np.ndarray
    """trafo_n0_n1_max_diff_factor.npy (if a trafo shall be limited in its N-0 to N-1 difference and
      by how much)"""

    trafo_dso_border: np.ndarray
    """trafo_dso_border.npy (a boolean mask of transformers that border the DSO control area)
    Currently only used during importing and not part of the PowsyblBackend"""

    trafo_pst_controllable: np.ndarray
    """Trafos which are a PST and can be controlled"""

    tie_line_for_reward: np.ndarray
    """tie_line_for_reward.npy (a boolean mask of tie lines that are relevant for the reward)."""

    tie_line_for_nminus1: np.ndarray
    """tie_line_for_nminus1.npy (a boolean mask of tie lines that are relevant for n-1)."""

    tie_line_overload_weight: np.ndarray
    """tie_line_overload_weight.npy (a float mask of weights for the overload)."""

    tie_line_disconnectable: np.ndarray
    """tie_line_disconnectable.npy (a boolean mask of tie lines that can be disconnected)"""

    tie_line_tso_border: np.ndarray
    """tie_line_tso_border.npy (a boolean mask of tielines that are leading to TSOs outside the reward area)
    Currently only used during importing and not part of the PowsyblBackend"""

    dangling_line_for_nminus1: np.ndarray
    """tie_line_disconnectable.npy (a boolean mask of tie lines that are relevant for n-1)."""

    generator_for_nminus1: np.ndarray
    """generator_for_nminus1.npy (a boolean mask of generators that are relevant for n-1)."""

    load_for_nminus1: np.ndarray
    """generator_for_nminus1.npy (a boolean mask of loads that are relevant for n-1)."""

    switch_for_nminus1: np.ndarray
    """switches_nminus1.npy (a boolean mask of switches that are relevant for n-1)."""

    switch_for_reward: np.ndarray
    """switches_reward.npy (a boolean mask of switches that are relevant for the reward)."""

    busbar_for_nminus1: np.ndarray
    """busbar_for_nminus1.npy (a boolean mask of busbars/busbar_sections that are relevant for n-1)."""


def create_default_network_masks(network: Network) -> NetworkMasks:
    """Create a default NetworkMasks object with all masks set to False.

    Parameters
    ----------
    network: Network
        The powsybl network to create the masks for.

    Returns
    -------
    network_masks: NetworkMasks
        The default NetworkMasks object.

    """
    # Only loading the index is much faster, if we only care for the size
    bus_df = network.get_buses(attributes=[])
    lines_df = network.get_lines(attributes=[])
    trafo_df = network.get_2_windings_transformers(attributes=[])
    tie_df = network.get_tie_lines(attributes=[])
    dangling_df = network.get_dangling_lines(attributes=[])
    generator_df = network.get_generators(attributes=[])
    load_df = network.get_loads(attributes=[])
    switches_df = network.get_switches(attributes=[])
    busbar_df = network.get_busbar_sections(attributes=[])

    return NetworkMasks(
        relevant_subs=np.zeros(len(bus_df), dtype=bool),
        line_for_nminus1=np.zeros(len(lines_df), dtype=bool),
        line_for_reward=np.zeros(len(lines_df), dtype=bool),
        line_overload_weight=np.ones(len(lines_df), dtype=float),
        line_disconnectable=np.zeros(len(lines_df), dtype=bool),
        line_blacklisted=np.zeros(len(lines_df), dtype=bool),
        line_tso_border=np.zeros(len(lines_df), dtype=bool),
        trafo_for_nminus1=np.zeros(len(trafo_df), dtype=bool),
        trafo_for_reward=np.zeros(len(trafo_df), dtype=bool),
        trafo_overload_weight=np.ones(len(trafo_df), dtype=float),
        trafo_disconnectable=np.zeros(len(trafo_df), dtype=bool),
        trafo_blacklisted=np.zeros(len(trafo_df), dtype=bool),
        trafo_n0_n1_max_diff_factor=np.ones(len(trafo_df), dtype=float) * -1,
        trafo_dso_border=np.zeros(len(trafo_df), dtype=bool),
        trafo_pst_controllable=np.zeros(len(trafo_df), dtype=bool),
        tie_line_for_reward=np.zeros(len(tie_df), dtype=bool),
        tie_line_for_nminus1=np.zeros(len(tie_df), dtype=bool),
        tie_line_overload_weight=np.ones(len(tie_df), dtype=float),
        tie_line_disconnectable=np.zeros(len(tie_df), dtype=bool),
        tie_line_tso_border=np.zeros(len(tie_df), dtype=bool),
        dangling_line_for_nminus1=np.zeros(len(dangling_df), dtype=bool),
        generator_for_nminus1=np.zeros(len(generator_df), dtype=bool),
        load_for_nminus1=np.zeros(len(load_df), dtype=bool),
        switch_for_nminus1=np.zeros(len(switches_df), dtype=bool),
        switch_for_reward=np.zeros(len(switches_df), dtype=bool),
        busbar_for_nminus1=np.zeros(len(busbar_df), dtype=bool),
    )


def get_mask_for_area_codes(element_df: pd.DataFrame, area_codes: list[str], *columns: str) -> np.ndarray:
    """Return the mask for the given area codes.

    Parameters
    ----------
    element_df: pd.DataFrame
        The DataFrame to get the mask from. Must contain the column "column"."
    area_codes: list[str]
        The area codes to consider. e.g. ["D2", "D4", "D7", "D8"] for Germany.
    columns: str
        The columns to check for the area codes. If you check multiple columns their results are "or"ed

    Returns
    -------
    mask: np.ndarray
        The mask for the given area codes.
    """
    if element_df.empty:
        return np.array([], dtype=bool)
    area_mask = np.zeros(len(element_df), dtype=bool)
    for column in columns:
        area_mask |= element_df[column].str.startswith(tuple(area_codes))
    return area_mask.values


def validate_network_masks(network_masks: NetworkMasks, default_mask: NetworkMasks) -> bool:
    """Validate if the network masks are created correctly.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to validate.
    default_mask: NetworkMasks
        The default network masks to validate against.

    Returns
    -------
    bool
        True if the network masks are created correctly, False otherwise.

    """
    if not isinstance(network_masks, NetworkMasks):
        logger.warning("network_masks are not of type NetworkMasks.")
        return False
    for mask_key, mask in asdict(network_masks).items():
        if not isinstance(mask, np.ndarray):
            logger.warning(f"Mask {mask_key} is not a numpy array.")
            return False
        if not mask.shape == asdict(default_mask)[mask_key].shape:
            logger.warning(
                f"Shape of mask {mask_key} is not correct. got: "
                + f"{mask.shape}, expected: {asdict(default_mask)[mask_key].shape}"
            )
            return False
        if mask.dtype != asdict(default_mask)[mask_key].dtype:
            logger.warning(
                f"Dtype of mask {mask_key} is not correct. got: "
                + f"{mask.dtype}, expected: {asdict(default_mask)[mask_key].dtype}"
            )
            return False
    return True


def get_voltage_from_voltage_level_id(network: Network, voltage_level_ids: pd.Series) -> np.ndarray:
    """Map voltage level id to actual physical voltage level in kV.

    Parameters
    ----------
    network: Network
        The powsybl network that contains the voltage levels
    voltage_level_ids: pd.Series
        The series of ids to map

    Returns
    -------
    np.ndarray:
        The voltage levels corresponding to the ids
    """
    voltage_levels = network.get_voltage_levels(attributes=["nominal_v"])
    return voltage_level_ids.map(voltage_levels["nominal_v"]).values


def get_border_line_mask(
    lines_df: pd.DataFrame,
    side_1_in_area: np.ndarray,
    side_2_in_area: np.ndarray,
    hv_line_mask: np.ndarray,
    area_codes: list[str],
) -> np.ndarray:
    """Filter border lines in UCTE.

    Usually these are modeled as tie-lines,
    but for the specific case of germany, the lines between the 4 german TSOs and some borders to Denmark and Luxembourg
    are modeled as lines with a border-bus inbetween.
    e.g.

    Actual TSO 1 Bus -> 1st part of actual line -> artificical border bus -> 2nd part of actual line -> Actual TSO 2 Bus

    This function is used to identify these lines and to decide how to handle them.

    Parameters
    ----------
    lines_df: pd.DataFrame
        The lines in the network including the voltage_level_id columns
    side_1_in_area: np.ndarray
        Boolean array of length n_lines that depicts if the side 1 is inside the border
    side_2_in_area: np.ndarray
        Boolean array of length n_lines that depicts if the side 2 is inside the border
    hv_line_mask: np.ndarray
        Boolean array of length n_lines that depicts if the line is high voltage
    area_codes: list[str]
        A list of area codes that are considered as part of the network

    Returns
    -------
    np.ndarray
        A boolean array over all lines, that depicts outgoing border lines
    np.ndarray
        A boolean array over all lines, that depicts internal border lines
    """
    potential_border_lines = (side_1_in_area != side_2_in_area) & hv_line_mask
    border_voltage_levels = np.concatenate(
        [
            lines_df[potential_border_lines & side_1_in_area].voltage_level2_id.values,
            lines_df[potential_border_lines & side_2_in_area].voltage_level1_id.values,
        ]
    )
    outer_border_line_idx = []
    inner_border_line_idx = []
    for border_voltage_level in border_voltage_levels:
        # Get the other side of all lines that are connected to the border voltage level
        lines_in_area = pd.concat(
            [
                lines_df[lines_df.voltage_level1_id == border_voltage_level].voltage_level2_id,
                lines_df[lines_df.voltage_level2_id == border_voltage_level].voltage_level1_id,
            ]
        )
        n_areas = len(lines_in_area.unique())
        min_areas_to_be_border = 2
        if n_areas >= min_areas_to_be_border:
            # If the line connects less than 2 areas, it is not a border line. This can be the case
            # e.g. for DC lines that are modeled as gens
            in_area = lines_in_area.str.startswith(tuple(area_codes))
            if all(in_area):
                inner_border_line_idx.extend(lines_in_area.index[in_area])
                # These are lines that connect different TSO-areas inside the same optimized region.
                # They could also be relevant to track. TODO Check what to do with them
            else:
                outer_border_line_idx.extend(lines_in_area.index[in_area])

    return lines_df.index.isin(outer_border_line_idx), lines_df.index.isin(inner_border_line_idx)


def update_line_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    blacklisted_ids: list[str],
) -> NetworkMasks:
    """Update the line masks in NetworkMasks for the given network state and import parameters.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters of the current import, containing nminus1_area, control_area,
        cutoffvoltage and border_line_weight
    blacklisted_ids: list[str]
        The ids of the branches that are blacklisted.

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the line masks.
    """
    lines_df = network.get_lines(attributes=["voltage_level1_id", "voltage_level2_id"])

    # Identify relevant parameters for the trafo masks
    hv_line_mask = (
        get_voltage_from_voltage_level_id(network, lines_df["voltage_level1_id"])
        >= importer_parameters.area_settings.cutoff_voltage
    )

    lines_with_limits = get_element_has_limits_mask(network, lines_df)

    if importer_parameters.data_type == "ucte":
        region_colums = ["voltage_level1_id", "voltage_level2_id"]
    elif importer_parameters.data_type == "cgmes":
        lines_df = get_region_for_df(df=lines_df, network=network)
        region_colums = ["region_1", "region_2"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")

    side_1_in_n1_area = get_mask_for_area_codes(lines_df, importer_parameters.area_settings.nminus1_area, region_colums[0])
    side_2_in_n1_area = get_mask_for_area_codes(lines_df, importer_parameters.area_settings.nminus1_area, region_colums[1])
    side_1_in_view_area = get_mask_for_area_codes(lines_df, importer_parameters.area_settings.view_area, region_colums[0])
    side_2_in_view_area = get_mask_for_area_codes(lines_df, importer_parameters.area_settings.view_area, region_colums[1])

    # Create N-1 and Reward masks based on nminus1_area
    nminus1_area_mask = side_1_in_n1_area | side_2_in_n1_area
    view_area_mask = side_1_in_view_area | side_2_in_view_area
    outage_mask = nminus1_area_mask & hv_line_mask
    reward_mask = view_area_mask & lines_with_limits & hv_line_mask

    # Create disconnectable mask based on control_area
    control_area_mask = get_mask_for_area_codes(
        lines_df, importer_parameters.area_settings.control_area, region_colums[0], region_colums[1]
    )
    disconnectable_mask = control_area_mask & hv_line_mask

    blacklisted_lines = lines_df.index.isin(blacklisted_ids)

    # Identify border lines inside the observed area and outside of it
    external_border_mask, _ = get_border_line_mask(
        lines_df,
        side_1_in_n1_area,
        side_2_in_n1_area,
        hv_line_mask,
        area_codes=importer_parameters.area_settings.nminus1_area,
    )
    line_overload_weight = np.where(
        external_border_mask,
        network_masks.line_overload_weight * importer_parameters.area_settings.border_line_weight,
        network_masks.line_overload_weight,
    )

    return replace(
        network_masks,
        line_for_nminus1=outage_mask,
        line_for_reward=reward_mask,
        line_blacklisted=blacklisted_lines,
        line_disconnectable=disconnectable_mask,
        line_tso_border=external_border_mask,
        line_overload_weight=line_overload_weight,
    )


def get_element_has_limits_mask(network: Network, element_df: pd.DataFrame) -> np.ndarray:
    """Return a mask for the elements that have operational limits.

    Parameters
    ----------
    network: Network
        The network to get the limits from.
    element_df: pd.DataFrame
        The DataFrame with the elements.

    Returns
    -------
    np.ndarray
        The mask for the elements that have operational limits.
    """
    limits = network.get_operational_limits(attributes=[]).index.get_level_values("element_id")
    return element_df.index.isin(limits)


def update_trafo_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    blacklisted_ids: list[str],
) -> NetworkMasks:
    """Update the trafo masks in the NetworkMasks for the given network state and import parameters.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
        The import parameters including nminus1_area, cutoff_voltage and optionally dso_trafo_factors and dso_trafo_weight
    blacklisted_ids: list[int]
        The ids of the branches that are blacklisted. DSO borders are excluded from the blacklist
        because we need to consider the effect of our switching actions on the DSOs.

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the trafo masks.
    """
    trafos_df = network.get_2_windings_transformers(
        attributes=["bus1_id", "bus2_id", "voltage_level1_id", "voltage_level2_id"]
    )

    # Identify relevant parameters for the trafo masks
    side_one_in_hv = (
        get_voltage_from_voltage_level_id(network, trafos_df["voltage_level1_id"])
        >= importer_parameters.area_settings.cutoff_voltage
    )
    side_two_in_hv = (
        get_voltage_from_voltage_level_id(network, trafos_df["voltage_level2_id"])
        >= importer_parameters.area_settings.cutoff_voltage
    )

    # If any side is in HV we consider it a TSO trafo
    hv_trafos = side_one_in_hv | side_two_in_hv

    trafos_with_limits = get_element_has_limits_mask(network, trafos_df)

    # Create N-1 and Reward masks based on nminus1_area
    if importer_parameters.data_type == "ucte":
        region_colums = ["voltage_level1_id", "voltage_level2_id"]
    elif importer_parameters.data_type == "cgmes":
        trafos_df = get_region_for_df(df=trafos_df, network=network)
        region_colums = ["region_1", "region_2"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")
    nminus1_area_mask = get_mask_for_area_codes(
        trafos_df, importer_parameters.area_settings.nminus1_area, region_colums[0], region_colums[1]
    )
    controllable_mask = get_mask_for_area_codes(
        trafos_df, importer_parameters.area_settings.control_area, region_colums[0], region_colums[1]
    )
    view_area_mask = get_mask_for_area_codes(
        trafos_df, importer_parameters.area_settings.view_area, region_colums[0], region_colums[1]
    )
    disconnectable_mask = controllable_mask & hv_trafos
    pst_controllable_mask = controllable_mask & hv_trafos
    outage_mask = nminus1_area_mask & hv_trafos
    reward_mask = view_area_mask & trafos_with_limits & hv_trafos

    # If only one side is in HV its a trafo from TSO to DSO
    trafo_dso_border = (side_one_in_hv != side_two_in_hv) & nminus1_area_mask

    trafo_overload_weight = np.where(
        trafo_dso_border,
        network_masks.trafo_overload_weight * importer_parameters.area_settings.dso_trafo_weight,
        network_masks.trafo_overload_weight,
    )
    # Create blacklisted mask based on blacklisted_ids
    # Exlude DSO border trafos from the blacklist because we need to consider the effect of our switching actions on the DSOs
    blacklisted_trafos = trafos_df.index.isin(blacklisted_ids)
    backlist_excl_dso = blacklisted_trafos & ~trafo_dso_border

    return replace(
        network_masks,
        trafo_for_nminus1=outage_mask,
        trafo_for_reward=reward_mask,
        trafo_blacklisted=backlist_excl_dso,
        trafo_dso_border=trafo_dso_border,
        trafo_overload_weight=trafo_overload_weight,
        trafo_disconnectable=disconnectable_mask,
        trafo_pst_controllable=pst_controllable_mask,
    )


def update_bus_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    filesystem: AbstractFileSystem,
) -> NetworkMasks:
    """Update the bus masks for the network.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters including control_area and cutoff_voltage
    filesystem: AbstractFileSystem
        The filesystem to read the ignore list from.

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the bus masks.
    """
    buses = network.get_buses()
    if importer_parameters.data_type == "ucte":
        relevant_subs = buses.index.isin(
            get_switchable_buses_ucte(
                network,
                importer_parameters.area_settings.control_area,
                importer_parameters.area_settings.cutoff_voltage,
                importer_parameters.select_by_voltage_level_id_list,
            )
        )
        substation_ids = buses["voltage_level_id"].values
    elif importer_parameters.data_type == "cgmes":
        relevant_subs = buses.index.isin(
            get_switchable_buses_cgmes(
                net=network,
                area_codes=importer_parameters.area_settings.control_area,
                cutoff_voltage=importer_parameters.area_settings.cutoff_voltage,
                select_by_voltage_level_id_list=importer_parameters.select_by_voltage_level_id_list,
                relevant_station_rules=importer_parameters.relevant_station_rules,
            )
        )
        substation_ids = buses["name"].str[:-2].values
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")

    # apply ignore list
    if importer_parameters.ignore_list_file is not None:
        with filesystem.open(str(importer_parameters.ignore_list_file), "r") as file:
            ignore_df = pd.read_csv(file, sep=";")
        ignore_subs = ignore_df["grid_model_id"].to_list()
        # str[:-2], because the bus names have a suffix e.g. "_0" or "_1" added to the station name
        relevant_subs = np.logical_and(
            relevant_subs,
            ~np.isin(substation_ids, ignore_subs),
        )

    return replace(
        network_masks,
        relevant_subs=relevant_subs,
    )


def update_tie_and_dangling_line_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
) -> NetworkMasks:
    """Update the dangling line and tie line masks in the NetworkMasks for the given network and import parameters.

    Dangling lines that are part of a tie line built a branch
    Dangling lines that are not part of a tie line are considered injections

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters including nminus1_area and cutoff_voltage and border_line_weight

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the tie line and dangling line masks.
    """
    # Get relevant data from tie lines
    tie_line_df = network.get_tie_lines(attributes=[])
    tie_lines_with_limits = get_element_has_limits_mask(network, tie_line_df)

    # Get relevant data from dangling lines.
    dangling_lines_df = network.get_dangling_lines(attributes=["voltage_level_id", "tie_line_id"])
    hv_dangling_mask = (
        get_voltage_from_voltage_level_id(network, dangling_lines_df["voltage_level_id"])
        >= importer_parameters.area_settings.cutoff_voltage
    )
    if importer_parameters.data_type == "ucte":
        region_colums = ["voltage_level_id"]
    elif importer_parameters.data_type == "cgmes":
        dangling_lines_df = get_region_for_df(df=dangling_lines_df, network=network)
        region_colums = ["region"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")
    nminus1_area_dangling_mask = get_mask_for_area_codes(
        dangling_lines_df, importer_parameters.area_settings.nminus1_area, region_colums[0]
    )

    # This includes dangling lines that can be considered branches (part of tielines)
    # and dangling lines that can be considered injections
    dangling_line_for_nminus1 = hv_dangling_mask & nminus1_area_dangling_mask

    # If a dangling line is part of the selected n-1 area, its correspondet tie line should be part aswell
    tie_line_for_nminus1 = tie_line_df.index.isin(dangling_lines_df[dangling_line_for_nminus1].tie_line_id.values)
    # If dangling lines are part of a tieline, they can be part of the reward
    tie_line_for_reward = tie_line_for_nminus1 & tie_lines_with_limits

    # All tie lines are by definition border lines
    tie_line_tso_border = tie_line_for_nminus1.copy()
    tie_line_overload_weight = np.where(
        tie_line_tso_border,
        network_masks.tie_line_overload_weight * importer_parameters.area_settings.border_line_weight,
        network_masks.tie_line_overload_weight,
    )

    return replace(
        network_masks,
        tie_line_for_nminus1=tie_line_for_nminus1,
        tie_line_for_reward=tie_line_for_reward,
        tie_line_tso_border=tie_line_tso_border,
        tie_line_overload_weight=tie_line_overload_weight,
        dangling_line_for_nminus1=dangling_line_for_nminus1,
    )


def update_load_and_generation_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
) -> NetworkMasks:
    """Update the load and generation masks.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters including nminus1_area and cutoff_voltage

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the generator and load masks.
    """
    generators_df = network.get_generators(attributes=["voltage_level_id"])
    load_df = network.get_loads(attributes=["voltage_level_id"])

    if importer_parameters.data_type == "ucte":
        region_colums = ["voltage_level_id"]
    elif importer_parameters.data_type == "cgmes":
        generators_df = get_region_for_df(df=generators_df, network=network)
        load_df = get_region_for_df(df=load_df, network=network)
        region_colums = ["region"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")

    # Update Generator masks
    generator_hv_mask = (
        get_voltage_from_voltage_level_id(network, generators_df.voltage_level_id)
        >= importer_parameters.area_settings.cutoff_voltage
    )
    generator_nminus1_mask = generator_hv_mask & get_mask_for_area_codes(
        generators_df, importer_parameters.area_settings.nminus1_area, region_colums[0]
    )

    # Update Load masks
    load_hv_mask = (
        get_voltage_from_voltage_level_id(network, load_df.voltage_level_id)
        >= importer_parameters.area_settings.cutoff_voltage
    )
    load_nminus1_mask = load_hv_mask & get_mask_for_area_codes(
        load_df, importer_parameters.area_settings.nminus1_area, region_colums[0]
    )
    return replace(
        network_masks,
        generator_for_nminus1=generator_nminus1_mask,
        load_for_nminus1=load_nminus1_mask,
    )


def update_switch_masks(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
) -> NetworkMasks:
    """Update the switch masks.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network mask object to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters including nminus1_area and cutoff_voltage

    Returns
    -------
    network_masks: NetworkMasks
        The updated network mask object including the switch masks.
    """
    # Load relevant data
    switch_df = network.get_switches(attributes=["voltage_level_id"])
    if importer_parameters.data_type == "ucte":
        region_colums = ["voltage_level_id"]
    elif importer_parameters.data_type == "cgmes":
        switch_df = get_region_for_df(df=switch_df, network=network)
        region_colums = ["region"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")
    switch_hv_mask = (
        get_voltage_from_voltage_level_id(network, switch_df["voltage_level_id"])
        >= importer_parameters.area_settings.cutoff_voltage
    )
    switch_with_limits = get_element_has_limits_mask(network, switch_df)
    nminus1_area_mask = get_mask_for_area_codes(switch_df, importer_parameters.area_settings.nminus1_area, region_colums[0])

    # Set reward and outage mask
    outage_mask = nminus1_area_mask & switch_hv_mask
    reward_mask = outage_mask & switch_with_limits
    return replace(
        network_masks,
        switch_for_nminus1=outage_mask,
        switch_for_reward=reward_mask,
    )


def make_masks(
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    filesystem: AbstractFileSystem = None,
    blacklisted_ids: list[str] | None = None,
) -> NetworkMasks:
    """Create all masks for the network, depending on the import parameters.

    Parameters
    ----------
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters including control_area, nminus1_area, cutoff_voltage
        Optional: border_line_factors, border_line_weight, dso_trafo_factors, dso_trafo_weight
    filesystem: AbstractFileSystem
        The filesystem to use for loading the contingency lists from. If not provided, the local filesystem is used.
    blacklisted_ids: list[str] | None
        The ids of the branche that are blacklisted.

    Returns
    -------
    network_masks: NetworkMasks
        The masks for the network.
    """
    if filesystem is None:
        filesystem = LocalFileSystem()
    if blacklisted_ids is None:
        blacklisted_ids = []
    default_masks = create_default_network_masks(network)

    network_masks = update_line_masks(
        default_masks,
        network,
        importer_parameters,
        blacklisted_ids,
    )
    network_masks = update_trafo_masks(
        network_masks,
        network,
        importer_parameters,
        blacklisted_ids,
    )
    network_masks = update_tie_and_dangling_line_masks(network_masks, network, importer_parameters)
    network_masks = update_load_and_generation_masks(network_masks, network, importer_parameters)
    network_masks = update_switch_masks(network_masks, network, importer_parameters)
    network_masks = update_bus_masks(network_masks, network, importer_parameters, filesystem=filesystem)
    network_masks = update_reward_masks_to_include_border_branches(network_masks, importer_parameters)
    network_masks = remove_slack_from_relevant_subs(network, network_masks, distributed_slack=True)

    if importer_parameters.contingency_list_file is not None:
        if importer_parameters.schema_format == "ContingencyImportSchemaPowerFactory":
            network_masks = update_masks_from_power_factory_contingency_list_file(
                network_masks, network, importer_parameters, filesystem=filesystem
            )
        elif importer_parameters.schema_format == "ContingencyImportSchema":
            network_masks = update_masks_from_contingency_list_file(
                network_masks, network, importer_parameters, filesystem=filesystem
            )
        else:
            logger.warning(f"Contingency list processing for {importer_parameters.ingress_id} is not implemented yet.")
    if not validate_network_masks(network_masks, default_masks):
        raise RuntimeError("Network masks are not created correctly.")

    return network_masks


def update_reward_masks_to_include_border_branches(
    network_masks: NetworkMasks, importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
) -> NetworkMasks:
    """Update the reward masks to include the border lines and tie lines.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to update
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The parameters to use for the update

    Returns
    -------
    NetworkMasks
        The updated network masks including the borders in the reward masks
    """
    if importer_parameters.area_settings.border_line_factors:
        network_masks = replace(
            network_masks,
            line_for_reward=network_masks.line_for_reward | network_masks.line_tso_border,
            tie_line_for_reward=network_masks.tie_line_for_reward | network_masks.tie_line_tso_border,
        )

    if importer_parameters.area_settings.dso_trafo_factors:
        network_masks = replace(
            network_masks,
            trafo_for_reward=network_masks.trafo_for_reward | network_masks.trafo_dso_border,
        )
    return network_masks


def save_masks_to_files(network_masks: NetworkMasks, data_folder: Path) -> None:
    """Save the network masks to files.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to save.
    data_folder: Path
        The folder to save the masks to.
    """
    save_masks_to_filesystem(network_masks, data_folder, filesystem=LocalFileSystem())


def save_masks_to_filesystem(network_masks: NetworkMasks, data_folder: Path, filesystem: AbstractFileSystem) -> None:
    """Save the network masks to a filesystem.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to save.
    data_folder: Path
        The folder to save the masks to.
    filesystem: AbstractFileSystem
        The filesystem to save the masks to.
    """
    masks_folder = data_folder / PREPROCESSING_PATHS["masks_path"]
    filesystem.makedirs(str(masks_folder), exist_ok=True)
    for mask_key, mask in asdict(network_masks).items():
        save_numpy_filesystem(filesystem=filesystem, file_path=masks_folder / NETWORK_MASK_NAMES[mask_key], numpy_array=mask)


def remove_slack_from_relevant_subs(
    network: Network, network_masks: NetworkMasks, distributed_slack: bool = True
) -> NetworkMasks:
    """Remove the slack bus from the relevant_subs mask.

    Parameters
    ----------
    network: Network
        The powsybl network.
    network_masks: NetworkMasks
        The network masks to update.
    distributed_slack: bool
        If the slack bus is distributed

    Returns
    -------
    network_masks: NetworkMasks
        The updated network masks without the slack bus in the relevant_subs mask.
    """
    dc_results = pypowsybl.loadflow.run_dc(network, DISTRIBUTED_SLACK if distributed_slack else SINGLE_SLACK)
    slack_id = dc_results[0].reference_bus_id

    return replace(network_masks, relevant_subs=network_masks.relevant_subs & ~network.get_buses().index.isin([slack_id]))


def update_masks_from_power_factory_contingency_list_file(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    filesystem: AbstractFileSystem,
    process_multi_outages: bool = False,
) -> NetworkMasks:
    """Update the network masks from the power factory contingency list file.

    Loads the contingency list file and updates the network masks accordingly.
    This replaces the default masks with the ones from the contingency list file.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters of the current import, containing nminus1_area, control_area,
        cutoffvoltage, border_line_weight and the contingency list file
    filesystem: AbstractFileSystem
        The filesystem to use for loading the contingency list file.
    process_multi_outages: bool
        If True, the contingency list file is processed as a multi-outage file.
        If False, the contingency list file is processed as a single-outage file.
        Default is False.
        The selection of single-outage will add each unique grid_model_id to the masks.

    Returns
    -------
    network_masks: NetworkMasks
        The updated network masks object including the contingency list file masks.

    Raises
    ------
    NotImplementedError
        If process_multi_outages is True, this function is not implemented yet.
    """
    contingency_list = get_contingencies_from_file(
        n1_file=importer_parameters.contingency_list_file, delimiter=";", filesystem=filesystem
    )
    all_element_names = get_all_element_names(net=network)
    processed_n1_definition = match_contingencies(
        n1_definition=contingency_list, all_element_names=all_element_names, match_by_name=True
    )
    three_winding_trafo_suffix = ["-Leg1", "-Leg2", "-Leg3"]
    processed_n1_definition = match_contingencies_with_suffix(
        processed_n1_definition=processed_n1_definition,
        all_element_names=all_element_names,
        grid_model_suffix=three_winding_trafo_suffix,
    )
    grid_model_ids = processed_n1_definition["grid_model_id"].unique()

    generators_df = network.get_generators(attributes=["voltage_level_id"])
    load_df = network.get_loads(attributes=["voltage_level_id"])

    if importer_parameters.data_type == "ucte":
        __annotations__region_colums = ["voltage_level_id"]
    elif importer_parameters.data_type == "cgmes":
        generators_df = get_region_for_df(df=generators_df, network=network)
        load_df = get_region_for_df(df=load_df, network=network)
        _region_colums = ["region"]
    else:
        raise ValueError(f"Data type {importer_parameters.data_type} is not supported.")

    # Update Generator masks
    generator_hv_mask = (
        get_voltage_from_voltage_level_id(network, generators_df.voltage_level_id)
        >= importer_parameters.area_settings.cutoff_voltage
    )
    generator_nminus1_mask = generator_hv_mask & network.get_generators().index.isin(grid_model_ids)

    # Update Load masks
    load_hv_mask = (
        get_voltage_from_voltage_level_id(network, load_df.voltage_level_id)
        >= importer_parameters.area_settings.cutoff_voltage
    )
    load_nminus1_mask = load_hv_mask & network.get_loads().index.isin(grid_model_ids)

    busbar_df = network.get_busbar_sections(attributes=["voltage_level_id"])
    busbar_for_nminus1 = (
        get_voltage_from_voltage_level_id(network, busbar_df.voltage_level_id)
        >= importer_parameters.area_settings.cutoff_voltage
    ) & busbar_df.index.isin(grid_model_ids)

    if not process_multi_outages:
        grid_model_ids = processed_n1_definition["grid_model_id"].unique()
        network_masks = replace(
            network_masks,
            line_for_nminus1=network.get_lines().index.isin(grid_model_ids),
            trafo_for_nminus1=network.get_2_windings_transformers().index.isin(grid_model_ids),
            generator_for_nminus1=generator_nminus1_mask,
            load_for_nminus1=load_nminus1_mask,
            switch_for_nminus1=network.get_switches().index.isin(grid_model_ids),
            dangling_line_for_nminus1=network.get_dangling_lines().index.isin(grid_model_ids),
            busbar_for_nminus1=busbar_for_nminus1,
        )
    else:
        raise NotImplementedError("Multi-outages are not supported yet.")
    return network_masks


def update_masks_from_contingency_list_file(
    network_masks: NetworkMasks,
    network: Network,
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters],
    filesystem: AbstractFileSystem,
    process_multi_outages: bool = False,
) -> NetworkMasks:
    """Update the network masks from the contingency list file.

    Loads the contingency list file and updates the network masks accordingly.
    Currenty only contains branches (monitored and contingency).
    This replaces the default masks with the ones from the contingency list file.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to update.
    network: Network
        The network to get the masks for.
    importer_parameters: Union[UcteImporterParameters, CgmesImporterParameters]
        The import parameters of the current import, containing nminus1_area, control_area,
        cutoffvoltage, border_line_weight and the contingency list file
    filesystem: AbstractFileSystem
        The filesystem to use for loading the contingency list file.
    process_multi_outages: bool
        If True, the contingency list file is processed as a multi-outage file.
        If False, the contingency list file is processed as a single-outage file.
        Default is False.

    Returns
    -------
    network_masks: NetworkMasks
        The updated network masks object including the contingency list file masks.
    """
    trafo3ws = network.get_3_windings_transformers()
    assert trafo3ws.empty, "3-winding transformers should have been converted to 2w-trafos."

    with filesystem.open(str(importer_parameters.contingency_list_file), mode="r") as f:
        contingency_analysis_df: ContingencyImportSchema = pd.read_csv(f, index_col=0, header=0)

    monitored_ids = contingency_analysis_df.query("observe_std").index.to_list()
    contingency_ids = contingency_analysis_df.query("contingency_case").index.to_list()

    lines = network.get_lines(attributes=[])
    line_for_nminus1 = lines.index.isin(contingency_ids)
    line_for_reward = lines.index.isin(monitored_ids)

    trafos = network.get_2_windings_transformers(attributes=[])
    # Replace the appendage of the 3w->2w conversion to get the original trafo ids
    trafo_orig_ids = trafos.index.str.replace("-Leg[123]$", "", regex=True)
    trafo_for_nminus1 = trafo_orig_ids.isin(contingency_ids)
    trafo_for_reward = trafo_orig_ids.isin(monitored_ids)

    dangling_lines = network.get_dangling_lines(attributes=["tie_line_id"])
    dangling_for_nminus1 = dangling_lines.index.isin(contingency_ids)
    dangling_for_reward = dangling_lines.index.isin(monitored_ids)

    tie_lines = network.get_tie_lines(attributes=[])
    tie_lines_for_nminus1 = tie_lines.index.isin(dangling_lines[dangling_for_nminus1].tie_line_id.values)
    tie_lines_for_reward = tie_lines.index.isin(dangling_lines[dangling_for_reward].tie_line_id.values)

    if not process_multi_outages:
        network_masks = replace(
            network_masks,
            line_for_nminus1=line_for_nminus1,
            line_for_reward=line_for_reward,
            trafo_for_nminus1=trafo_for_nminus1,
            trafo_for_reward=trafo_for_reward,
            dangling_line_for_nminus1=dangling_for_nminus1,
            tie_line_for_nminus1=tie_lines_for_nminus1,
            tie_line_for_reward=tie_lines_for_reward,
        )
    else:
        raise NotImplementedError("Multi-outages are not supported yet.")
    return network_masks
