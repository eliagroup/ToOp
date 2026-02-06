# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module contains functions to create masks for the pandapower backend.

File: powsybl_masks.py
Author:  Benjamin Petrick
Created: 2024-10-02
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import logbook
import numpy as np
import pandapower as pp
from jaxtyping import Array, Bool, Int
from pandas import Index
from toop_engine_importer.pandapower_import.pandapower_toolset_node_breaker import (
    get_coupler_types_of_substation,
    get_type_b_nodes,
)
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)

logger = logbook.Logger(__name__)


@dataclass
class NetworkMasks:
    """Class to hold the network masks.

    See class Pandapower(BackendInterface) in DCLoadflowsolver for more information.
    """

    relevant_subs: np.ndarray
    """relevant_subs.npy (a boolean mask of relevant nodes)"""

    line_for_nminus1: np.ndarray
    """line_for_nminus1.npy (a boolean mask of lines that are relevant for n-1)"""

    line_for_reward: np.ndarray
    """line_for_reward.npy (a boolean mask of lines that are relevant for the reward)"""

    line_overload_weight: np.ndarray
    """line_overload_weight.npy (a float mask of weights for the overload)"""

    line_disconnectable: np.ndarray
    """line_disconnectable.npy (a boolean mask of lines that can be disconnected)"""

    trafo_for_nminus1: np.ndarray
    """trafo_for_nminus1.npy (a boolean mask of transformers that are relevant for n-1)"""

    trafo_for_reward: np.ndarray
    """trafo_for_reward.npy (a boolean mask of transformers that are relevant for the reward)"""

    trafo_overload_weight: np.ndarray
    """trafo_overload_weight.npy (a float mask of weights for the overload)"""

    trafo_disconnectable: np.ndarray
    """trafo_disconnectable.npy (a boolean mask of transformers that can be disconnected)"""

    trafo3w_for_nminus1: np.ndarray
    """trafo3w_for_nminus1.npy (a boolean mask of three winding transformers that are relevant for n-1)"""

    trafo3w_for_reward: np.ndarray
    """trafo3w_for_reward.npy (a boolean mask of three winding transformers that are relevant for the reward)"""

    trafo3w_overload_weight: np.ndarray
    """trafo3w_overload_weight.npy (a float mask of three winding transformers weights for the overload)"""

    trafo3w_disconnectable: np.ndarray
    """trafo3w_disconnectable.npy (a boolean mask of three winding transformers that can be disconnected)"""

    generator_for_nminus1: np.ndarray
    """generator_for_nminus1.npy (a boolean mask of generators that are relevant for n-1)"""

    sgen_for_nminus1: np.ndarray
    """sgen_for_nminus1.npy (a boolean mask of static generators that are relevant for n-1)"""

    load_for_nminus1: np.ndarray
    """generator_for_nminus1.npy (a boolean mask of loads that are relevant for n-1)"""


# TODO: refactor input parameters to a config for kafka
# ruff: noqa: PLR0913
def make_pp_masks(
    network: pp.pandapowerNet,
    region: str = "",
    voltage_level: float = 150,
    min_power: float = 100.0,
    trafo_weight: float = 1.2,
    cross_border_weight: float = 1.2,
    min_branches_per_station: int = 4,
    foreign_id_column: str = "equipment",
    exclude_stations: Optional[list[str]] = None,
    substation_column: str = "substat",
    min_busbars_per_substation: int = 2,
    min_busbar_coupler_per_station: int = 1,
) -> NetworkMasks:
    """Create the network masks for the pandapower network.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network.
    region: str
        The region of the network.
    voltage_level: float
        The voltage level of the network to be considered.
    min_power: float
        The minimum power of generators, loads and static generators to be considered,
        for the n-1 analysis and reassignable.
    trafo_weight: float
        The weight of the transformers for the overload.
    cross_border_weight: float
        The weight of the cross border lines for the overload.
    min_branches_per_station: int
        The minimum number of busbars per station to be relevant.
    foreign_id_column: str
        The column containing the foreign id.
    exclude_stations: Optional[list[str]]
        The substations to exclude.
    substation_column: str
        The column containing the substation.
    min_busbars_per_substation: int
        The minimum number of busbars per substation to be relevant.
        Note: you need a substation column in the bus dataframe.
        Note: if you merged the busbars, you need to set this to 1.
    min_busbar_coupler_per_station: int
        The minimum number of busbar coupler per station to be relevant.
        Note: you need a substation column in the bus dataframe.
        Note: if you merged the busbars, you need to set this to 0.


    Returns
    -------
    NetworkMasks
        The network masks.
    """
    region_busses = network.bus.zone == region
    hv_grid_busses = network.bus.vn_kv >= voltage_level
    region_hv_bus_indices = network.bus.index[region_busses & hv_grid_busses]

    line_for_reward = (
        # Only high voltage lines
        (network.line.from_bus.isin(region_hv_bus_indices)) | (network.line.to_bus.isin(region_hv_bus_indices))
    )

    if foreign_id_column in network.line.columns:
        line_for_reward &= (
            # Only lines with FID
            network.line[foreign_id_column].notna()
        )
    line_for_nminus1 = line_for_reward
    if substation_column in network.bus.columns:
        line_for_reward &= (
            # Only lines that connect different substations
            network.bus.loc[network.line.from_bus.values].substat.values
            != network.bus.loc[network.line.to_bus.values].substat.values
        )

    trafo_for_reward = network.trafo.hv_bus.isin(
        # Only trafo with high voltage the specified region
        region_hv_bus_indices
    ) | network.trafo.lv_bus.isin(region_hv_bus_indices)
    if foreign_id_column in network.trafo.columns:
        trafo_for_reward &= (
            # Only trafo with FID
            network.trafo[foreign_id_column].notna()
        )
    trafo_for_nminus1 = trafo_for_reward
    trafo3w_for_reward = (
        network.trafo3w.hv_bus.isin(region_hv_bus_indices)
        | network.trafo3w.mv_bus.isin(region_hv_bus_indices)
        | network.trafo3w.lv_bus.isin(region_hv_bus_indices)
    )
    if foreign_id_column in network.trafo3w.columns:
        trafo3w_for_reward &= (
            # Only trafo3w with FID
            network.trafo3w[foreign_id_column].notna()
        )
    trafo3w_for_nminus1 = trafo3w_for_reward
    load_for_nminus1 = network.load.bus.isin(region_hv_bus_indices) & (network.load.p_mw >= min_power)
    if foreign_id_column in network.load.columns:
        load_for_nminus1 &= (
            # Only loads with FID
            network.load[foreign_id_column].notna()
        )

    gen_for_nminus1 = network.gen.bus.isin(region_hv_bus_indices) & (network.gen.p_mw >= min_power)
    if foreign_id_column in network.gen.columns:
        gen_for_nminus1 &= (
            # Only generators with FID
            network.gen[foreign_id_column].notna()
        )

    sgen_for_nminus1 = network.sgen.bus.isin(region_hv_bus_indices) & (network.sgen.p_mw >= min_power)
    if foreign_id_column in network.sgen.columns:
        sgen_for_nminus1 &= (
            # Only static generators with FID
            network.sgen[foreign_id_column].notna()
        )

    dso_trafos = network.trafo.hv_bus.isin(region_hv_bus_indices) & ~network.trafo.lv_bus.isin(region_hv_bus_indices)
    if foreign_id_column in network.trafo.columns:
        dso_trafos &= (
            # Only trafo with FID
            network.trafo[foreign_id_column].notna()
        )

    trafo_overload_weight = np.ones(len(network.trafo))
    trafo_overload_weight[dso_trafos] *= trafo_weight
    dso_trafo3ws = network.trafo3w.hv_bus.isin(region_hv_bus_indices) & (
        ~network.trafo3w.lv_bus.isin(region_hv_bus_indices) | ~network.trafo3w.lv_bus.isin(region_hv_bus_indices)
    )
    if foreign_id_column in network.trafo3w.columns:
        dso_trafo3ws &= (
            # Only trafo3w with FID
            network.trafo3w[foreign_id_column].notna()
        )

    trafo3w_overload_weight = np.ones(len(network.trafo3w))
    trafo3w_overload_weight[dso_trafo3ws] *= trafo_weight
    cross_border_lines = (
        network.line.from_bus.isin(region_hv_bus_indices) & ~network.line.to_bus.isin(region_hv_bus_indices)
    ) | (network.line.to_bus.isin(region_hv_bus_indices) & ~network.line.from_bus.isin(region_hv_bus_indices))
    line_overload_weight = np.ones(len(network.line))
    line_overload_weight[cross_border_lines] *= cross_border_weight

    relevant_subs = get_relevant_subs(
        network=network,
        region=region,
        voltage_level=voltage_level,
        min_branches_per_station=min_branches_per_station,
        exclude_stations=exclude_stations,
        substation_column=substation_column,
        min_busbars_per_substation=min_busbars_per_substation,
        min_busbar_coupler_per_station=min_busbar_coupler_per_station,
    )

    masks = NetworkMasks(
        relevant_subs=relevant_subs,
        line_for_nminus1=line_for_nminus1.values,
        line_for_reward=line_for_reward.values,
        line_overload_weight=line_overload_weight,
        line_disconnectable=line_for_reward.values,
        trafo_for_nminus1=trafo_for_nminus1.values,
        trafo_for_reward=trafo_for_reward.values,
        trafo_overload_weight=trafo_overload_weight,
        trafo_disconnectable=trafo_for_reward.values,
        trafo3w_for_nminus1=trafo3w_for_nminus1.values,
        trafo3w_for_reward=trafo3w_for_reward.values,
        trafo3w_overload_weight=trafo3w_overload_weight,
        trafo3w_disconnectable=trafo3w_for_reward.values,
        generator_for_nminus1=gen_for_nminus1.values,
        sgen_for_nminus1=sgen_for_nminus1.values,
        load_for_nminus1=load_for_nminus1.values,
    )
    return masks


# TODO: create kafka config for function
def get_relevant_subs(
    network: pp.pandapowerNet,
    region: str = "",
    voltage_level: float = 150,
    min_branches_per_station: int = 4,
    exclude_stations: Optional[list[str]] = None,
    substation_column: str = "substat",
    min_busbars_per_substation: int = 2,
    min_busbar_coupler_per_station: int = 1,
) -> NetworkMasks:
    """Create the network masks for the pandapower network.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network.
    region: str
        The region of the network.
    voltage_level: float
        The voltage level of the network to be considered.
    min_branches_per_station: int
        The minimum number of busbars per station to be relevant.
    exclude_stations: Optional[list[str]]
        The substations to exclude.
    substation_column: str
        The column containing the substation.
    min_busbars_per_substation: int
        The minimum number of busbars per substation to be relevant.
        Note: you need a substation column in the bus dataframe.
        Note: if you merged the busbars, you need to set this to 1.
        Note: set to 0 and ignore busbars and branches
    min_busbar_coupler_per_station: int
        The minimum number of busbar coupler per station to be relevant.
        Note: you need a substation column in the bus dataframe.
        Note: if you merged the busbars, you need to set this to 0.

    Returns
    -------
    np.ndarray
        The relevant substations.
    """
    region_busses = network.bus.zone == region
    hv_grid_busses = network.bus.vn_kv >= voltage_level

    if substation_column in network.bus.columns:
        if min_busbars_per_substation > 0:
            mask_multiple_busbars = mask_min_busbar_per_station(network, min_busbars_per_substation, substation_column)
        else:
            mask_multiple_busbars = np.ones(len(network.bus), dtype=bool)

        if min_branches_per_station > 0:
            mask_min_branches = mask_min_branches_per_station(network, substation_column, min_branches_per_station)
        else:
            mask_min_branches = np.ones(len(network.bus), dtype=bool)

        if min_busbar_coupler_per_station > 0:
            mask_multiple_busbar_coupler = mask_min_busbar_coupler(
                network, min_busbar_coupler_per_station, substation_column
            )
        else:
            mask_multiple_busbar_coupler = np.ones(len(network.bus), dtype=bool)
    else:
        mask_multiple_busbars = np.ones(len(network.bus), dtype=bool)
        mask_min_branches = count_branches_at_buses(network, network.bus.index) >= min_branches_per_station
        mask_multiple_busbar_coupler = np.ones(len(network.bus), dtype=bool)

    if exclude_stations is not None and substation_column in network.bus.columns:
        exclude_mask = ~network.bus[substation_column].isin(exclude_stations)
    else:
        exclude_mask = np.ones(len(network.bus), dtype=bool)

    filtered_busbars = network.bus[
        hv_grid_busses
        & region_busses
        & exclude_mask
        & mask_multiple_busbars
        & mask_multiple_busbar_coupler
        & mask_min_branches
    ]
    relevant_subs = np.isin(network.bus.index, filtered_busbars.index)

    return relevant_subs


def mask_min_branches_per_station(
    network: pp.pandapowerNet,
    substation_column: str,
    min_branches_per_station: int = 4,
    exclude_stations: Optional[list[str]] = None,
) -> Bool[Array, " n_buses"]:
    """Get the mask for the substations with a minimum number of branches.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network.
    substation_column: str
        The column containing the substation.
    min_branches_per_station: int
        The minimum number of branches per station to be relevant.
    exclude_stations: Optional[list[str]]
        The substations to exclude.
        Default: substation with no name (empty string: "").

    Returns
    -------
    Bool[Array, " n_buses"]
        The number of branches connected to the substations.
    """
    if exclude_stations is None:
        # exclues stubstations with no name (empty string)
        exclude_stations = [""]
    station_names = network.bus[substation_column].unique()
    station_names = [station for station in station_names if station not in exclude_stations]
    n_buses = len(network.bus)
    mask_branches = np.zeros(n_buses, dtype=bool)
    for station_name in station_names:
        len_assets = count_assets_in_substation(
            network=network,
            substation_column=substation_column,
            station_name=station_name,
        )
        if len_assets >= min_branches_per_station:
            mask_branches = mask_branches | (network.bus[substation_column] == station_name).values
    return mask_branches


# TODO: create config
def count_assets_in_substation(
    network: pp.pandapowerNet,
    substation_column: str,
    station_name: str,
    include_branches: bool = True,
    include_gen: bool = False,
    include_load: bool = False,
    include_impedance: bool = False,
) -> int:
    """Count the number of assets in a substation.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network.
    substation_column: str
        The column containing the substation.
    station_name: str
        The name of the station.
    include_branches: bool
        Include branches.
    include_gen: bool
        Include generators and static generators.
    include_load: bool
        Include loads.
    include_impedance: bool
        Include impedances.

    Returns
    -------
    int
        The number of assets in the substation.
    """
    station = network.bus[network.bus[substation_column] == station_name]
    branches = pp.toolbox.get_connected_elements_dict(network, station.index, include_empty_lists=True)
    return count_assets(
        branches,
        include_branches=include_branches,
        include_gen=include_gen,
        include_load=include_load,
        include_impedance=include_impedance,
    )


def count_assets(
    branches: dict[str, list[str]],
    include_branches: bool = True,
    include_gen: bool = False,
    include_load: bool = False,
    include_impedance: bool = False,
) -> int:
    """Count the number of assets from a pandapower.toolbox.get_connected_elements_dict.

    Parameters
    ----------
    branches: dict[str, list[str]]
        The branches to count the assets from.
    include_branches: bool
        Include branches.
    include_gen: bool
        Include generators and static generators.
    include_load: bool
        Include loads.
    include_impedance: bool
        Include impedances.

    Returns
    -------
    int
        The number of assets in the dictionary.
    """
    len_assets = 0
    if include_branches:
        len_assets += len(branches["line"])
        len_assets += len(branches["trafo"])
        len_assets += len(branches["trafo3w"])
    if include_gen:
        len_assets += len(branches["gen"])
        len_assets += len(branches["sgen"])
    if include_load:
        len_assets += len(branches["load"])
        len_assets += len(branches["asymmetric_load"])
    if include_impedance:
        len_assets += len(branches["impedance"])
    return len_assets


def count_branches_at_buses(network: pp.pandapowerNet, buses: Index) -> Int[Array, " n_buses"]:
    """Count the number of branches connected to the buses.

    Parameters
    ----------
    network: pp.pandapowerNet
        The pandapower network.
    buses: Index
        The buses to count the branches at.

    Returns
    -------
    int[Array, " n_buses"]
        The number of branches connected to the buses.
    """
    buses = np.array(buses)
    n_buses = len(buses)
    count = np.zeros(n_buses, dtype=int)
    for elem, column in pp.element_bus_tuples(bus_elements=False, branch_elements=True, res_elements=False):
        comp_matrix = network[elem][column].values[None, :] == buses[:, None]
        count += np.sum(comp_matrix, axis=1)
    return count


def mask_min_busbar_per_station(
    net: pp.pandapowerNet,
    min_branches_per_station: int = 2,
    substation_column: str = "substat",
) -> Bool[Array, " n_buses"]:
    """Count the number of busbars per station.

    This function counts the number of busbars per station and
    returns a boolean mask of the substations with a minimum number of busbars.
    A busbar is a bus with a type of "b" in the substation column.
    Note: this only makes sense if the busbars have not been merged
    e.g. with pandapower_toolset.fuse_closed_switches_fast.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network.
    min_branches_per_station: int
        The minimum number of busbars per station to be relevant.
    substation_column: str
        The column containing the substation.

    Returns
    -------
    Bool[Array, " n_buses"]
        A boolean mask of the substations with a minimum number of busbars.
    """
    bus_type_b = get_type_b_nodes(net)
    station_names = bus_type_b[substation_column].unique()
    n_buses = len(net.bus)
    mask_multiple_busbars = np.zeros(n_buses, dtype=bool)
    for station_name in station_names:
        number_of_busbars = count_busbars_at_station(net, station_name, substation_column)
        if number_of_busbars >= min_branches_per_station:
            mask_multiple_busbars = mask_multiple_busbars | (net.bus[substation_column] == station_name).values
    return mask_multiple_busbars


def count_busbars_at_station(
    net: pp.pandapowerNet,
    station_name: str,
    substation_column: str = "substat",
) -> int:
    """Count the number of busbars at a station.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network.
    station_name: str
        The name of the station.
    substation_column: str
        The column containing the substation.

    Returns
    -------
    int
        The number of busbars at the station.
    """
    bus_type_b = get_type_b_nodes(net)
    station = bus_type_b[bus_type_b[substation_column] == station_name]
    return len(station)


def mask_min_busbar_coupler(
    net: pp.pandapowerNet,
    min_busbar_coupler_per_station: int = 1,
    substation_column: str = "substat",
) -> Bool[Array, " n_buses"]:
    """Get the mask for the substations with a minimum number of busbar coupler.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network.
        min_busbar_coupler_per_station: int
        The minimum number of busbar coupler per station to be relevant.
    min_busbar_coupler_per_station: int
        The minimum number of busbar coupler per station to be relevant.
    substation_column: str
        The column containing the substation.

    Returns
    -------
        Bool[Array, " n_buses"]
        A boolean mask of the substations with a minimum number of busbar coupler.
    """
    bus_type_b = get_type_b_nodes(net)
    station_names = bus_type_b[substation_column].unique()
    n_buses = len(net.bus)
    mask_multiple_busbar_coupler = np.zeros(n_buses, dtype=bool)
    for station_name in station_names:
        number_of_couplers = count_busbar_coupler_at_station(net, station_name, substation_column)
        if number_of_couplers >= min_busbar_coupler_per_station:
            mask_multiple_busbar_coupler = mask_multiple_busbar_coupler | (net.bus[substation_column] == station_name).values
    return mask_multiple_busbar_coupler


def count_busbar_coupler_at_station(
    net: pp.pandapowerNet,
    station_name: str,
    substation_column: str = "substat",
) -> int:
    """Count the number of busbar coupler at a station.

    Parameters
    ----------
    net: pp.pandapowerNet
        The pandapower network.
    station_name: str
        The name of the station.
    substation_column: str
        The column containing the substation.

    Returns
    -------
    int
        The number of busbar coupler at the station.
    """
    bus_type_b = get_type_b_nodes(net)
    station = bus_type_b[bus_type_b[substation_column] == station_name]
    busbars = get_coupler_types_of_substation(net, station.index, True)
    return len(busbars["busbar_coupler_bus_ids"])


def create_default_network_masks(network: pp.pandapowerNet) -> NetworkMasks:
    """Create a default NetworkMasks object with all masks set to False.

    Parameters
    ----------
    network: pp.pandapowerNet
        The powsybl network to create the masks for.

    Returns
    -------
    network_masks: NetworkMasks
        The default NetworkMasks object.

    """
    return NetworkMasks(
        relevant_subs=np.zeros(len(network.bus), dtype=bool),
        line_for_nminus1=np.zeros(len(network.line), dtype=bool),
        line_for_reward=np.zeros(len(network.line), dtype=bool),
        line_overload_weight=np.zeros(len(network.line), dtype=float),
        line_disconnectable=np.zeros(len(network.line), dtype=bool),
        trafo_for_nminus1=np.zeros(len(network.trafo), dtype=bool),
        trafo_for_reward=np.zeros(len(network.trafo), dtype=bool),
        trafo_overload_weight=np.zeros(len(network.trafo), dtype=float),
        trafo_disconnectable=np.zeros(len(network.trafo), dtype=bool),
        trafo3w_for_nminus1=np.zeros(len(network.trafo3w), dtype=bool),
        trafo3w_for_reward=np.zeros(len(network.trafo3w), dtype=bool),
        trafo3w_overload_weight=np.zeros(len(network.trafo3w), dtype=float),
        trafo3w_disconnectable=np.zeros(len(network.trafo3w), dtype=bool),
        generator_for_nminus1=np.zeros(len(network.gen), dtype=bool),
        sgen_for_nminus1=np.zeros(len(network.sgen), dtype=bool),
        load_for_nminus1=np.zeros(len(network.load), dtype=bool),
    )


def validate_network_masks(net: pp.pandapowerNet, network_masks: NetworkMasks) -> bool:
    """Test if the network masks are created correctly.

    Parameters
    ----------
    net: pp.pandapowerNet
        The network to test the masks on.
    network_masks: NetworkMasks
        The network masks to test.

    Returns
    -------
    bool
        True if the network masks are created correctly, False otherwise.

    """
    if not isinstance(network_masks, NetworkMasks):
        logger.warning("network_masks are not of type NetworkMasks.")
        return False
    default_mask = create_default_network_masks(net)
    for mask_key, mask in asdict(network_masks).items():
        if not isinstance(mask, np.ndarray):
            logger.warning(f"Mask {mask_key} is not a numpy array.")
            return False
        if not mask.shape == asdict(default_mask)[mask_key].shape:
            logger.warning(
                f"Shape of mask {mask_key} is not correct. got: {mask.shape}, "
                + f"expected: {asdict(default_mask)[mask_key].shape}"
            )
            return False
        if mask.dtype != asdict(default_mask)[mask_key].dtype:
            logger.warning(
                f"Dtype of mask {mask_key} is not correct. got: {mask.dtype}, "
                + f"expected: {asdict(default_mask)[mask_key].dtype}"
            )
            return False
    return True


def save_masks_to_files(network_masks: NetworkMasks, data_folder: Path) -> None:
    """Save the network masks to files.

    Parameters
    ----------
    network_masks: NetworkMasks
        The network masks to save.
    data_folder: Path
        The folder to save the masks to.

    """
    masks_folder = data_folder / PREPROCESSING_PATHS["masks_path"]
    masks_folder.mkdir(exist_ok=True, parents=True)
    for mask_key, mask in asdict(network_masks).items():
        np.save(masks_folder / NETWORK_MASK_NAMES[mask_key], mask)


def save_preprocessing(data_folder: Path, network: pp.pandapowerNet, network_masks: NetworkMasks) -> None:
    """Save the preprocessed network and the preprocessed data to the data_folder structure.

    Parameters
    ----------
    data_folder: Path
        The root folder of the processed timestep.
    network: pp.pandapowerNet
        The pandapower network.
    network_masks: NetworkMasks
        NetworkMasks containing the masks for the network and the method to get as dict.

    Raises
    ------
    RuntimeError
        If the network masks are not created correctly.

    """
    if not validate_network_masks(network, network_masks):
        raise RuntimeError("Network masks are not created correctly.")
    grid_folder = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_folder.parent.mkdir(exist_ok=True, parents=True)
    pp.to_json(network, grid_folder)

    save_masks_to_files(network_masks, data_folder)
