# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides example grids for testing the dc_solver package."""

# ruff/sonar: noqa: PLR0915, S3776

import bz2
import datetime
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pypowsybl
from beartype.typing import Literal, Optional
from fsspec.implementations.dirfs import DirFileSystem
from networkx.algorithms.community import kernighan_lin_bisection
from toop_engine_dc_solver.preprocess import NetworkData
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_grid_helpers.asset_topology_helpers import (
    save_asset_topology_stations,
    save_master_asset_topology,
)
from toop_engine_grid_helpers.pandapower.example_grids import (
    pandapower_case30_with_psts_and_weak_branches,
    pandapower_extended_case57,
    pandapower_extended_oberrhein,
    pandapower_non_converging_case57,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_busbar_b_in_ieee,
    create_busbar_outage_always_articulation_grid,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
    extract_station_info_powsybl,
    parallel_pst_example,
    powsybl_case30_with_psts,
    powsybl_case1354,
    powsybl_case9241,
    powsybl_extended_case57,
    three_node_pst_example,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import (
    load_lf_params_from_fs,
    save_lf_params_to_fs,
    sort_powsybl_element_frame_by_id,
)
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_importer.pypowsybl_import.powsybl_masks import make_masks, save_masks_to_filesystem
from toop_engine_interfaces.asset_topology.asset_topology import BusGroupAssetConnection, MasterAssetTopology, MasterBusGroup
from toop_engine_interfaces.asset_topology.assets import (
    AssetBay,
    BranchAsset,
    BusbarCoupler,
    CouplerBay,
    InjectionAsset,
    build_asset_bay_id,
)
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
    RuntimeInjectionAsset,
)
from toop_engine_interfaces.asset_topology.runtime_topology import (
    RuntimeAssetConnection,
    RuntimeAssetTopology,
    RuntimeBusGroup,
)
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    LimitAdjustmentParameters,
    PreprocessParameters,
)


def compress_bz2(source_file: str) -> None:
    """Compress a file with bz2 and remove the original file

    Parameters
    ----------
    source_file : str
        The file to compress
    """
    dest_path = source_file + ".bz2"
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"{source_file} cannot be compressed because it does not exist.")
    with open(source_file, "rb") as source, bz2.BZ2File(dest_path, "wb") as dest:
        dest.writelines(source)
    os.remove(source_file)


def save_timestep_data(
    timestep_nets: list[pp.pandapowerNet],
    folder: str | Path,
    filename_without_ext: str,
    element_type: Literal["gen", "sgen", "load", "dcline"],
    attribute: Literal["p_mw", "q_mvar", "vm_pu"],
    save_grid2op_compatible: bool = False,
) -> None:
    """Saves the prod_p values of a list of pandapower networks to a csv file

    Parameters
    ----------
    timestep_nets : list[pp.pandapowerNet]
        The list of pandapower networks to extract the data from
    folder : str | Path
        The folder to save the data to, should be the grid path / chronics / xxxx where xxxx is the
        timestep number with 4 digits
    filename_without_ext : str
        The filename without extension to save the data to, should be like load_p, gen_p, ...
    element_type : Literal["gen", "sgen", "load"]
        The type of element to extract the data from
    attribute : Literal["p_mw", "q_mvar", "vm_pu"]
        The attribute to extract from the element
    save_grid2op_compatible : bool
        Whether to also save grid2op compatible csv.bz2 chronics
    """
    base_net = timestep_nets[0]
    timestep_nets = timestep_nets[1:]

    # Export the wished attribute
    values = base_net[element_type][["name", attribute]].transpose()
    # Make the first row the header row
    values.columns = values.iloc[0]
    values.drop("name", inplace=True)

    # Append the value for the timestep
    for timestep_net in timestep_nets:
        values.loc[len(values)] = timestep_net[element_type][attribute].values

    if attribute == "vm_pu":
        # The grid2op format wishes for voltage in MW not in pu
        values = base_net.bus.loc[base_net[element_type].bus]["vn_kv"].values * values

    os.makedirs(folder, exist_ok=True)
    # This was the format that grid2op used to save the data
    if save_grid2op_compatible:
        values.to_csv(os.path.join(folder, f"{filename_without_ext}.csv"), index=False, sep=";")
        # Compress with bz2
        compress_bz2(os.path.join(folder, f"{filename_without_ext}.csv"))
    np.save(os.path.join(folder, f"{filename_without_ext}.npy"), values.values.astype(float))


@dataclass
class PandapowerCounters:
    """To generate valid ids for pandapower, we need to count the number of buses and switches

    The asset topology will pretend there is a bus B, but in the IEEE grids there is usually only one bus per station and
    no switches. Hence, the switches and buses B will be created when applying the grid using apply_asset_topo. For this
    however, the IDs need to be valid. And to generate valid IDs, we have to keep count of the number of buses and
    switches in the grid
    """

    highest_switch_id: int
    """The highest index in net.switch.index"""

    highest_bus_id: int
    """The highest index in net.bus.index"""


def random_station_info_backend(
    backend: BackendInterface, node_idx: Integral, pp_counters: Optional[PandapowerCounters]
) -> tuple[RuntimeBusGroup, Optional[PandapowerCounters]]:
    """Generate a random station for any backend

    This will create a Station object with 2 busbars, 1 coupler and a random assignment of assets
    to the busbars

    Parameters
    ----------
    backend : BackendInterface
        The backend to generate the topology for
    node_idx : Integral
        The bus to generate the station for, indexing into all nodes of the backend
    pp_counters : Optional[PandapowerCounters]
        The pandapower counters to generate valid IDs for the pandapower backend. If given, it will generate the bus B ids
        and switch ids based on the highest bus in the counters and increase the counter

    Returns
    -------
    Station
        The generated station
    Optional[PandapowerCounters]
        The updated pandapower counters, if given
    """
    switchable_assets: list[tuple[BranchAsset | InjectionAsset, str | None]] = []
    for branch_id, branch_type, branch_name, branch_node in zip(
        backend.get_branch_ids(),
        backend.get_branch_types(),
        backend.get_branch_names(),
        backend.get_from_nodes(),
        strict=True,
    ):
        if branch_node == node_idx:
            switchable_assets.append(
                (
                    RuntimeBranchAsset(
                        grid_model_id=branch_id,
                        asset_type=branch_type,
                        name=branch_name,
                        in_service=True,
                    ),
                    "from",
                )
            )

    for branch_id, branch_type, branch_name, branch_node in zip(
        backend.get_branch_ids(),
        backend.get_branch_types(),
        backend.get_branch_names(),
        backend.get_to_nodes(),
        strict=True,
    ):
        if branch_node == node_idx:
            switchable_assets.append(
                (
                    RuntimeBranchAsset(
                        grid_model_id=branch_id,
                        asset_type=branch_type,
                        name=branch_name,
                        in_service=True,
                    ),
                    "to",
                )
            )

    for injection_id, injection_type, injection_name, injection_node in zip(
        backend.get_injection_ids(),
        backend.get_injection_types(),
        backend.get_injection_names(),
        backend.get_injection_nodes(),
        strict=True,
    ):
        if injection_node == node_idx:
            switchable_assets.append(
                (
                    RuntimeInjectionAsset(
                        grid_model_id=injection_id,
                        asset_type=injection_type,
                        name=injection_name,
                        in_service=True,
                    ),
                    None,
                )
            )

    branch_assets = [asset for asset, _ in switchable_assets if isinstance(asset, BranchAsset)]
    injection_assets = [asset for asset, _ in switchable_assets if isinstance(asset, InjectionAsset)]
    branch_terminals = [branch_end for asset, branch_end in switchable_assets if isinstance(asset, BranchAsset)]

    branch_switching_table = np.zeros((2, len(branch_assets)), dtype=bool)
    branch_is_on_a = np.random.rand(len(branch_assets)) > 0.5
    branch_switching_table[0, branch_is_on_a] = True
    branch_switching_table[1, ~branch_is_on_a] = True

    injection_switching_table = np.zeros((2, len(injection_assets)), dtype=bool)
    injection_is_on_a = np.random.rand(len(injection_assets)) > 0.5
    injection_switching_table[0, injection_is_on_a] = True
    injection_switching_table[1, ~injection_is_on_a] = True

    global_id = backend.get_node_ids()[node_idx]
    if pp_counters is not None:
        bus_a_id = global_id
        bus_b_id = f"{pp_counters.highest_bus_id + 1}{SEPARATOR}bus"
        switch_id = f"{pp_counters.highest_switch_id + 1}{SEPARATOR}switch"

        pp_counters = replace(
            pp_counters,
            highest_switch_id=pp_counters.highest_switch_id + 1,
            highest_bus_id=pp_counters.highest_bus_id + 1,
        )
    else:
        bus_a_id = global_id + "_a"
        bus_b_id = global_id + "_b"
        switch_id = global_id + "_coupler"

    def build_direct_asset_bay(asset_grid_model_id: str, busbar_grid_model_id: str) -> AssetBay:
        asset_bay_id = build_asset_bay_id(global_id, asset_grid_model_id)
        return AssetBay(
            asset_bay_id=asset_bay_id,
            asset_disconnector_grid_model_id=None,
            dv_switch_grid_model_id=f"{asset_bay_id}::dv",
            busbar_disconnector_grid_model_id={
                busbar_grid_model_id: f"{asset_bay_id}::busbar_disconnector::{busbar_grid_model_id}"
            },
        )

    branch_connections = [
        RuntimeAssetConnection(
            asset=asset,
            branch_end=branch_end,
            asset_bay=build_direct_asset_bay(asset.grid_model_id, bus_a_id),
        )
        for asset, branch_end in zip(branch_assets, branch_terminals, strict=True)
    ]
    injection_connections = [
        RuntimeAssetConnection(
            asset=asset,
            asset_bay=build_direct_asset_bay(asset.grid_model_id, bus_a_id),
        )
        for asset in injection_assets
    ]

    return RuntimeBusGroup(
        bus_group_id=global_id,
        busbars=[
            RuntimeBusbar(
                grid_model_id=bus_a_id,
                name=backend.get_node_names()[node_idx],
                int_id=0,
                in_service=True,
            ),
            RuntimeBusbar(
                grid_model_id=bus_b_id,
                name=backend.get_node_names()[node_idx],
                int_id=1,
                in_service=True,
            ),
        ],
        couplers=[
            RuntimeBusbarCoupler(
                grid_model_id=switch_id,
                busbar_from_id=0,
                busbar_to_id=1,
                open=False,
                in_service=True,
            ),
        ],
        branch_connections=branch_connections,
        injection_connections=injection_connections,
        branch_switching_table=branch_switching_table,
        injection_switching_table=injection_switching_table,
        branch_connectivity=np.ones_like(branch_switching_table, dtype=bool),
        injection_connectivity=np.ones_like(injection_switching_table, dtype=bool),
    ), pp_counters


def _build_random_master_asset_topology(stations: list[RuntimeBusGroup]) -> MasterAssetTopology:
    """Build canonical master data for the random example topology."""
    master_stations: list[MasterBusGroup] = []
    branch_assets_by_id: dict[str, BranchAsset] = {}
    injection_assets_by_id: dict[str, InjectionAsset] = {}
    asset_bays_by_id: dict[str, AssetBay] = {}
    for station in stations:
        station_branch_connections = _copy_station_asset_connections(
            asset_connections=station.branch_connections,
            expected_type=BranchAsset,
            assets_by_id=branch_assets_by_id,
            asset_bays_by_id=asset_bays_by_id,
        )
        station_injection_connections = _copy_station_asset_connections(
            asset_connections=station.injection_connections,
            expected_type=InjectionAsset,
            assets_by_id=injection_assets_by_id,
            asset_bays_by_id=asset_bays_by_id,
        )

        is_bus_branch_model = all(
            asset_connection.asset_bay_id is None
            for asset_connection in [*station_branch_connections, *station_injection_connections]
        )
        branch_connectivity = _build_station_connectivity(
            switching_table=np.asarray(station.branch_switching_table, dtype=bool),
            connectivity=station.branch_connectivity,
            station_connections=station_branch_connections,
            is_bus_branch_model=is_bus_branch_model,
        )
        injection_connectivity = _build_station_connectivity(
            switching_table=np.asarray(station.injection_switching_table, dtype=bool),
            connectivity=station.injection_connectivity,
            station_connections=station_injection_connections,
            is_bus_branch_model=is_bus_branch_model,
        )
        busbar_grid_model_id_by_int_id = {busbar.int_id: busbar.grid_model_id for busbar in station.busbars}
        canonical_couplers = []
        for coupler in station.couplers:
            coupler_bay = coupler.coupler_bay.model_copy(deep=True) if coupler.coupler_bay is not None else None
            if coupler_bay is None:
                coupler_bay = CouplerBay(
                    coupler_breaker_ids=[coupler.grid_model_id],
                    coupler_disconnector_ids=[],
                    from_busbar_ids=[busbar_grid_model_id_by_int_id[coupler.busbar_from_id]],
                    to_busbar_ids=[busbar_grid_model_id_by_int_id[coupler.busbar_to_id]],
                    from_busbar_disconnector_ids={},
                    to_busbar_disconnector_ids={},
                )
            canonical_couplers.append(
                BusbarCoupler(
                    grid_model_id=coupler.grid_model_id,
                    coupler_type=coupler.coupler_type,
                    name=coupler.name,
                    asset_bay=coupler.asset_bay.model_copy(deep=True) if coupler.asset_bay is not None else None,
                    coupler_bay=coupler_bay,
                )
            )

        master_stations.append(
            MasterBusGroup(
                bus_group_id=station.bus_group_id,
                voltage_level_id=station.voltage_level_id,
                name=station.name,
                station_type=station.station_type,
                region=station.region,
                voltage_level=station.voltage_level,
                busbars=[busbar.model_copy(update={"in_service": True}, deep=True) for busbar in station.busbars],
                couplers=canonical_couplers,
                branch_connections=station_branch_connections,
                injection_connections=station_injection_connections,
                branch_connectivity=branch_connectivity,
                injection_connectivity=injection_connectivity,
            )
        )

    return MasterAssetTopology(
        topology_id="random_topology",
        stations=master_stations,
        branch_assets=list(branch_assets_by_id.values()),
        injection_assets=list(injection_assets_by_id.values()),
        asset_bays=list(asset_bays_by_id.values()),
    )


def _copy_station_asset_connections(
    asset_connections: list[RuntimeAssetConnection],
    expected_type: type[BranchAsset] | type[InjectionAsset],
    assets_by_id: dict[str, BranchAsset] | dict[str, InjectionAsset],
    asset_bays_by_id: dict[str, AssetBay],
) -> list[BusGroupAssetConnection]:
    """Copy runtime station connections into canonical station references."""
    station_connections: list[BusGroupAssetConnection] = []
    for asset_connection in asset_connections:
        asset = asset_connection.asset.model_copy(update={"in_service": True}, deep=True)
        assert isinstance(asset, expected_type)
        assets_by_id[asset.grid_model_id] = asset

        asset_bay_id = asset_connection.asset_bay.asset_bay_id if asset_connection.asset_bay is not None else None
        if asset_connection.asset_bay is not None and asset_bay_id is not None:
            asset_bays_by_id[asset_bay_id] = asset_connection.asset_bay.model_copy(deep=True)

        station_connections.append(
            BusGroupAssetConnection(
                asset_id=asset.grid_model_id,
                branch_end=asset_connection.branch_end,
                asset_bay_id=asset_bay_id,
            )
        )

    return station_connections


def _build_station_connectivity(
    switching_table: np.ndarray,
    connectivity: Optional[np.ndarray],
    station_connections: list[BusGroupAssetConnection],
    is_bus_branch_model: bool,
) -> np.ndarray:
    """Build canonical connectivity while preserving single-bus assignments."""
    if is_bus_branch_model:
        return np.ones_like(switching_table, dtype=bool)

    normalized_connectivity = np.array(
        connectivity if connectivity is not None else switching_table,
        dtype=bool,
        copy=True,
    )
    for asset_index, asset_connection in enumerate(station_connections):
        if asset_connection.asset_bay_id is None and switching_table[:, asset_index].sum() == 1:
            normalized_connectivity[:, asset_index] = switching_table[:, asset_index]
    return normalized_connectivity


def random_topology_info_backend(
    backend: BackendInterface, pp_counters: Optional[PandapowerCounters]
) -> list[RuntimeBusGroup]:
    """Generate random runtime stations for any backend.

    This will create an AssetTopology with a station created for each relevant node in the network

    Parameters
    ----------
    backend : BackendInterface
        The backend to generate the topology for
    pp_counters : PandapowerCounters
        The pandapower counters to generate valid IDs for the pandapower backend

    Returns
    -------
    list[RuntimeBusGroup]
        Ordered runtime station snapshots.
    """
    relevant_nodes = np.flatnonzero(backend.get_relevant_node_mask())
    stations = []
    seen_bus_group_ids: set[str] = set()
    for node_idx in relevant_nodes:
        new_station, pp_counters = random_station_info_backend(backend, node_idx, pp_counters)
        if new_station.bus_group_id in seen_bus_group_ids:
            continue
        seen_bus_group_ids.add(new_station.bus_group_id)
        stations.append(new_station)
    return stations


def random_topology_info(folder: Path, pandapower: bool = True) -> None:
    """Generate a random asset topology and save it to the folder

    Parameters
    ----------
    folder : Path
        The grid folder, it will load the folder with the backend and later save the asset topology
        here
    pandapower : bool
        Whether to use the pandapower backend (true) or the powsybl backend (false)
    """
    filesystem_dir = DirFileSystem(folder)
    if pandapower:
        backend = PandaPowerBackend(filesystem_dir)
        pp_counters = PandapowerCounters(
            highest_switch_id=int(backend.net.switch.index.max()) if len(backend.net.switch) else 0,
            highest_bus_id=int(backend.net.bus.index.max()),
        )
    else:
        backend = PowsyblBackend(filesystem_dir)
        pp_counters = None
    stations = random_topology_info_backend(backend, pp_counters)
    master_data = _build_random_master_asset_topology(stations)

    destination = folder / PREPROCESSING_PATHS["asset_topology_runtime_file_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_asset_topology_stations(
        filename=destination,
        stations=RuntimeAssetTopology(stations=stations),
    )
    save_master_asset_topology(
        filename=folder / PREPROCESSING_PATHS["asset_topology_master_data_file_path"],
        master_data=master_data,
    )


# ruff/sonar: noqa: PLR0915, S3776
def oberrhein_data(folder: Path) -> None:
    """Build an example grid file which resembles the grid2op format but has more elements for testing"""
    net = pandapower_extended_oberrhein()
    os.makedirs(folder, exist_ok=True)
    pp.rundcpp(net)
    n_2_safe_line = np.argwhere(net.line.name == "n_2_safe_line").item()

    output_path_grid = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    output_path_grid.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, output_path_grid)

    # Use all lines for n-1
    line_for_nminus1 = np.random.rand(len(net.line)) > 0.8

    line_for_nminus1[n_2_safe_line] = True
    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_for_nminus1)
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_for_reward"],
        np.ones(len(net.line), dtype=bool),
    )

    line_disconnectable = np.random.rand(len(net.line)) > 0.8
    line_disconnectable[n_2_safe_line] = True
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_disconnectable"],
        line_disconnectable,
    )

    # Choose all nodes as relevant nodes, they will be filtered to only contain the nodes with
    # enough branches in the preprocessing
    relevant_node_mask = np.ones(len(net.bus), dtype=bool)
    relevant_node_mask[net.ext_grid.bus.values[0]] = False
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], relevant_node_mask)

    cross_coupler_limits = np.abs(np.random.randn(len(net.bus))) * 100
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["cross_coupler_limits"],
        cross_coupler_limits,
    )

    np.save(
        output_path_masks / NETWORK_MASK_NAMES["trafo3w_for_reward"],
        np.ones(len(net.trafo3w), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["trafo3w_for_nminus1"],
        np.ones(len(net.trafo3w), dtype=bool),
    )

    np.save(
        output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"],
        np.ones(len(net.gen), dtype=bool),
    )
    sgen_for_nminus1 = np.random.rand(len(net.sgen)) > 0.5
    np.save(output_path_masks / NETWORK_MASK_NAMES["sgen_for_nminus1"], sgen_for_nminus1)

    start_datetime_info_file_path = folder / PREPROCESSING_PATHS["start_datetime_info_file_path"]
    start_datetime_info_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(start_datetime_info_file_path, "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    # Generate chronics
    # 2 chronics of 7 timesteps each are enough
    timestep_nets = [net]
    for _ in range(13):
        # Slightly change the loads and consumptions for each timestep (+- 1%)
        net_copy = deepcopy(net)
        net_copy.load["p_mw"] += np.random.randn(len(net_copy.load)) * 0.01
        net_copy.gen["p_mw"] += np.random.randn(len(net_copy.gen)) * 0.01
        net_copy.sgen["p_mw"] += np.random.randn(len(net_copy.sgen)) * 0.01
        net_copy.dcline["p_mw"] += np.random.randn(len(net_copy.dcline)) * 0.01
        # Check convergence
        pp.rundcpp(net_copy)
        timestep_nets.append(net_copy)

    timestep_data = (
        ("load", "p_mw"),
        ("gen", "p_mw"),
        ("sgen", "p_mw"),
        ("dcline", "p_mw"),
    )
    chronics_folder = folder / PREPROCESSING_PATHS["chronics_path"]
    chronics_folder.mkdir(parents=True, exist_ok=True)
    timestep_slices = [slice(0, 7), slice(7, 14)]
    for element_type, attribute in timestep_data:
        for i, ts_slice in enumerate(timestep_slices):
            save_timestep_data(
                timestep_nets[ts_slice],
                chronics_folder / f"000{i}",
                f"{element_type}_{attribute[0]}",
                element_type,
                attribute,
            )

    np.random.seed(0)
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case57_data_pandapower(folder: Path) -> None:
    """A case57 variant that looks the same as the powsybl example network"""
    net = pandapower_extended_case57()
    os.makedirs(folder, exist_ok=True)
    pp.rundcpp(net)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)
    # Masks are just all elements
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_for_nminus1"],
        np.ones(len(net.line), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_for_reward"],
        np.ones(len(net.line), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_disconnectable"],
        np.ones(len(net.line), dtype=bool),
    )

    relevant_nodes = np.ones(len(net.bus), dtype=bool)
    # Disable the slack
    relevant_nodes[0] = False
    np.save(masks_path / NETWORK_MASK_NAMES["relevant_subs"], relevant_nodes)

    np.save(
        masks_path / NETWORK_MASK_NAMES["trafo_for_reward"],
        np.ones(len(net.trafo), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"],
        np.ones(len(net.trafo), dtype=bool),
    )

    np.random.seed(0)
    cross_coupler_limits = np.abs(np.random.randn(len(net.bus))) * 100
    np.save(masks_path / NETWORK_MASK_NAMES["cross_coupler_limits"], cross_coupler_limits)

    start_datetime_info_file_path = folder / PREPROCESSING_PATHS["start_datetime_info_file_path"]
    start_datetime_info_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(start_datetime_info_file_path, "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    np.random.seed(0)
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case57_data_powsybl(folder: Path) -> None:
    """Create a powsybl test grid with a PST and some operational limits"""
    net = powsybl_extended_case57()
    create_busbar_b_in_ieee(net)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )

    output_path_grid = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    output_path_grid.parent.mkdir(parents=True, exist_ok=True)
    net.save(output_path_grid)
    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"],
        np.ones(len(net.get_lines()), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_for_reward"],
        np.ones(len(net.get_lines()), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_disconnectable"],
        np.ones(len(net.get_lines()), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"],
        np.ones(len(net.get_2_windings_transformers()), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"],
        np.ones(len(net.get_2_windings_transformers()), dtype=bool),
    )
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["line_overload_weight"],
        np.full(len(net.get_lines()), 2.0),
    )
    relevant_nodes = np.ones(len(net.get_buses()), dtype=bool)
    # Disable the slack
    relevant_nodes[0] = False
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], relevant_nodes)
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"],
        np.ones(len(net.get_generators())),
    )

    np.random.seed(0)
    cross_coupler_limits = np.abs(np.random.randn(len(net.get_buses()))) * 100
    np.save(
        output_path_masks / NETWORK_MASK_NAMES["cross_coupler_limits"],
        cross_coupler_limits,
    )

    start_datetime_info_file_path = folder / PREPROCESSING_PATHS["start_datetime_info_file_path"]
    start_datetime_info_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(start_datetime_info_file_path, "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def case57_data_powsybl_xiidm(folder: Path) -> None:
    """Load a powsybl xiidm grid and save importing masks to the given folder path."""
    net = pypowsybl.network.load(folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    lf_result, *_ = pypowsybl.loadflow.run_ac(net, CGMES_DISTRIBUTED_SLACK)
    dir_system = DirFileSystem(folder)
    save_lf_params_to_fs(CGMES_DISTRIBUTED_SLACK, dir_system, Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))

    network_masks = make_masks(
        network=net,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=CgmesImporterParameters(
            area_settings=AreaSettings(control_area=[""], view_area=[""], nminus1_area=[""], cutoff_voltage=1),
            data_folder=folder,
            grid_model_file=Path(folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"],
        ),
    )
    save_masks_to_filesystem(network_masks, Path("."), dir_system)
    extract_station_info_powsybl(net, folder)


def case57_non_converging(folder: Path) -> None:
    """A case57 variant that does not converge in AC but does converge in DC"""
    net = pandapower_non_converging_case57()
    os.makedirs(folder, exist_ok=True)
    pp.rundcpp(net)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)
    # Masks are just all elements
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_for_nminus1"],
        np.ones(len(net.line), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_for_reward"],
        np.ones(len(net.line), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["line_disconnectable"],
        np.ones(len(net.line), dtype=bool),
    )

    relevant_nodes = np.ones(len(net.bus), dtype=bool)
    # Disable the slack
    relevant_nodes[0] = False
    np.save(masks_path / NETWORK_MASK_NAMES["relevant_subs"], relevant_nodes)

    np.save(
        masks_path / NETWORK_MASK_NAMES["trafo_for_reward"],
        np.ones(len(net.trafo), dtype=bool),
    )
    np.save(
        masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"],
        np.ones(len(net.trafo), dtype=bool),
    )
    np.random.seed(0)
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case300_pandapower(folder: Path) -> None:
    """A 300 bus case for mini benchmarks"""
    net = pp.networks.case300()
    os.makedirs(folder, exist_ok=True)
    pp.rundcpp(net)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.zeros(len(net.bus), dtype=bool)
    rel_sub_mask[0:50] = True
    np.save(masks_path / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.ones(len(net.line), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafo_mask = np.ones(len(net.trafo), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)

    gen_mask = np.ones(len(net.gen), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    np.random.seed(0)
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case300_powsybl(folder: Path, first_fifty_stations: bool = True) -> None:
    """The case300 network with a powsybl grid"""
    net = pypowsybl.network.create_ieee300()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)
    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.ones(len(net.get_buses()), dtype=bool)
    if first_fifty_stations:
        rel_sub_mask[50:] = False
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafo_mask = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)

    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


# ruff: noqa: PLR0915
# sonar: noqa: S3776
def case9241_pandapower(data_folder: Path) -> None:
    """Create a case9241 example scenario

    This is based on the case9241pegase grid from pandapower, but with some modifications:
     - The loads, gens and sgens are scaled down by 0.7
     - For the timesteps, the loads, gens and sgens are modified by a random walk with a small sigma
     - The grid is partitioned into 4 regions of roughly equal size
     - There are line and trafo masks for each region

    Parameters
    ----------
    data_folder : Path
        The folder to save the data to
    """
    np.random.seed(0)
    net = pp.networks.case9241pegase()
    pp.runpp(net)
    data_folder = Path(data_folder)
    os.makedirs(data_folder, exist_ok=True)

    # Add PST tap changers to the trafos that don't transform voltage
    (tap_min, tap_max, tap_step_percent, tap_step_degree, tap_neutral, tap_pos) = (-30, 30, pd.NA, 2, 0, 0)

    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_min"] = tap_min
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_max"] = tap_max
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_step_degree"] = tap_step_degree
    # Setting tap_step_degree is required to compute voltage angles but setting
    # both tap_step_percent and degree is disallowed, so setting to NA
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_step_percent"] = tap_step_percent
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_changer_type"] = "Ideal"
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_neutral"] = tap_neutral
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_pos"] = tap_pos

    # Reduce the injections to lower the overloads to a more realistic level
    net.load["p_mw"] *= 0.7
    net.gen["p_mw"] *= 0.7
    net.sgen["p_mw"] *= 0.7

    # Generate some timeseries as a random walk
    sigma1 = 0.01
    sigma2 = 0.01
    load_derivative = net.load.p_mw * np.random.randn(len(net.load)) * sigma1
    gen_derivative = net.gen.p_mw * np.random.randn(len(net.gen)) * sigma1
    sgen_derivative = net.sgen.p_mw * np.random.randn(len(net.sgen)) * sigma1
    timestep_nets = [net]
    for _ in range(23):
        net_copy = deepcopy(net)
        net_copy.load["p_mw"] += load_derivative + net.load.p_mw * np.random.randn(len(net.load)) * sigma2
        net_copy.gen["p_mw"] += gen_derivative + net.gen.p_mw * np.random.randn(len(net.gen)) * sigma2
        net_copy.sgen["p_mw"] += sgen_derivative + net.sgen.p_mw * np.random.randn(len(net.sgen)) * sigma2
        pp.runpp(net_copy)
        timestep_nets.append(net_copy)

    chronics_folder = data_folder / PREPROCESSING_PATHS["chronics_path"]
    chronics_folder.mkdir(parents=True, exist_ok=True)
    save_timestep_data(timestep_nets, chronics_folder / "0000", "load_p", "load", "p_mw")
    save_timestep_data(timestep_nets, chronics_folder / "0000", "gen_p", "gen", "p_mw")
    save_timestep_data(timestep_nets, chronics_folder / "0000", "sgen_p", "sgen", "p_mw")

    grid_file_path = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = data_folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)

    line_for_nminus1 = np.ones(len(net.line), dtype=bool)
    trafo_for_nminus1 = np.ones(len(net.trafo), dtype=bool)

    # Don't use bridges for the N-1 analysis
    graph = pp.topology.create_nxgraph(net, multi=True)
    line_for_nminus1, trafo_for_nminus1 = update_masks_for_bridges(line_for_nminus1, trafo_for_nminus1, graph)

    # Partition the grid into 4 regions of roughly equal size
    part1, part2 = kernighan_lin_bisection(graph, seed=np.random.randint(2**32))
    part11, part12 = kernighan_lin_bisection(graph.subgraph(part1), seed=np.random.randint(2**32))
    part21, part22 = kernighan_lin_bisection(graph.subgraph(part2), seed=np.random.randint(2**32))
    regions = [part11, part12, part21, part22]
    region_masks, relevant_sub_indices = generate_region_masks(net, line_for_nminus1, trafo_for_nminus1, graph, regions)

    all_relevant_sub_indices = np.concatenate(relevant_sub_indices)
    all_relevant_subs = np.zeros(len(net.bus), dtype=bool)
    all_relevant_subs[all_relevant_sub_indices] = True

    region_masks.update(
        {
            "line_for_nminus1": line_for_nminus1,
            "line_for_reward": np.ones(len(net.line), dtype=bool),
            "trafo_for_nminus1": trafo_for_nminus1,
            "trafo_for_reward": np.ones(len(net.trafo), dtype=bool),
            "relevant_subs": all_relevant_subs,
            "trafo_controllable": np.ones(len(net.trafo), dtype=bool),
        }
    )

    for key, mask in region_masks.items():
        np.save(masks_path / f"{key}.npy", mask)

    start_datetime_info_file_path = data_folder / PREPROCESSING_PATHS["start_datetime_info_file_path"]
    start_datetime_info_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(start_datetime_info_file_path, "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    np.random.seed(0)
    random_topology_info(data_folder)
    save_lf_params_to_fs({}, DirFileSystem(data_folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def generate_region_masks(
    net: pp.pandapowerNet, line_for_nminus1: np.ndarray, trafo_for_nminus1: np.ndarray, graph: nx.Graph, regions: list[set]
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Generate region-specific masks for lines, transformers, and relevant substations.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object.
    line_for_nminus1 : np.ndarray
        The line mask for n-1 analysis.
    trafo_for_nminus1 : np.ndarray
        The transformer mask for n-1 analysis.
    graph : nx.Graph
        The networkx graph of the network.
    regions : list[set]
        List of regions, each represented as a set of node indices.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[np.ndarray]]
        A tuple containing:
        - A dictionary with region-specific masks for lines, transformers, and relevant substations.
        - A list of arrays containing the indices of relevant substations for each region.
    """
    region_masks = {}
    relevant_sub_indices = []

    for region_id, region in enumerate(regions):
        local_line = np.zeros_like(line_for_nminus1)
        local_trafo = np.zeros_like(trafo_for_nminus1)

        for edge in graph.edges(region):
            edge_data = graph.get_edge_data(*edge)
            for table, index in edge_data.keys():
                if table == "line":
                    local_line[int(index)] = True
                elif table == "trafo":
                    local_trafo[int(index)] = True
                else:
                    raise RuntimeError(f"Unknown table {table}")

        local_subs = [node for node, degree in graph.degree(region) if degree >= 4]
        local_relevant_sub_indices = np.random.choice(local_subs, 100, replace=False)
        local_relevant_subs = np.zeros(len(net.bus), dtype=bool)
        local_relevant_subs[local_relevant_sub_indices] = True

        relevant_sub_indices.append(local_relevant_sub_indices)

        region_masks.update(
            {
                f"line_for_nminus1_{region_id}": np.logical_and(local_line, line_for_nminus1),
                f"line_for_reward_{region_id}": local_line,
                f"trafo_for_nminus1_{region_id}": np.logical_and(local_trafo, trafo_for_nminus1),
                f"trafo_for_reward_{region_id}": local_trafo,
                f"relevant_subs_{region_id}": local_relevant_subs,
            }
        )

    return region_masks, relevant_sub_indices


def update_masks_for_bridges(
    line_for_nminus1: np.ndarray, trafo_for_nminus1: np.ndarray, graph: nx.Graph
) -> tuple[np.ndarray, np.ndarray]:
    """Update the n-1 masks to exclude bridges in the network.

    Parameters
    ----------
    line_for_nminus1 : np.ndarray
        The line mask for n-1
    trafo_for_nminus1 : np.ndarray
        The trafo mask for n-1
    graph : nx.Graph
        The networkx graph of the network
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The updated line and trafo masks for n-1
    """
    bridges = list(nx.bridges(graph))
    for bridge in bridges:
        edge_data = graph.get_edge_data(*bridge)
        for table, index in edge_data.keys():
            if table == "line":
                line_for_nminus1[int(index)] = False
            elif table == "trafo":
                trafo_for_nminus1[int(index)] = False
            else:
                raise RuntimeError(f"Unknown table {table}")
    return line_for_nminus1, trafo_for_nminus1


def case9241_powsybl(folder: Path) -> None:
    """Create a case9241 example scenario for powsybl, loading the converted matpower file"""
    net = powsybl_case9241()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)
    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    # Pick 400 relevant nodes from all nodes that have at least 5 branches
    # (We could restrict to 4 but we have so many nodes that it doesn't matter)
    branch_count = net.get_branches()["bus1_id"].value_counts() + net.get_branches()["bus2_id"].value_counts()
    branch_count = branch_count[branch_count >= 5]
    relevant_node_ids = np.random.choice(branch_count.index, 400, replace=False)

    # Make sure to include the most connected node because we're mean.
    most_connected_node = branch_count.idxmax()
    if most_connected_node not in relevant_node_ids:
        relevant_node_ids[0] = most_connected_node

    relevant_node_mask = net.get_buses().index.isin(relevant_node_ids)

    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], relevant_node_mask)

    all_lines = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], all_lines)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], all_lines)

    all_trafos = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], all_trafos)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], all_trafos)

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def case1354_powsybl(folder: Path, n_stations: int = 1354) -> None:
    """Create a powsybl case1354 scenario.

    Parameters
    ----------
    folder : Path
        Target folder that receives the generated grid, masks, and topology data.
    n_stations : int, default=1354
        Number of initial stations to keep relevant before excluding the slack station.
    """
    net = powsybl_case1354()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)
    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.ones(len(net.get_buses()), dtype=bool)
    if n_stations:
        assert n_stations < len(rel_sub_mask), "n_stations must be less than the total number of buses"
        assert n_stations > 0, "n_stations must be greater than 0"
        rel_sub_mask[n_stations:] = False

    # Exclude the slack bus from the relevant substations
    rel_sub_mask[net.get_buses().index.get_loc("sub_639_0")] = False
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafo_mask = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)

    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def case14_pandapower(folder: Path) -> None:
    """A 14 bus case for basic tests"""
    net = pp.networks.case14()
    pp.rundcpp(net)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)

    relevant_node_mask = np.array(
        [
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    np.save(masks_path / NETWORK_MASK_NAMES["relevant_subs"], relevant_node_mask)

    line_mask = np.ones(len(net.line), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    # One trafo is a stub
    trafo_for_nminus1 = np.array([True, True, True, False, True])
    trafo_for_reward = np.ones(len(net.trafo), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_for_nminus1)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_for_reward)
    random_topology_info(folder)
    np.save(masks_path / NETWORK_MASK_NAMES["generator_for_nminus1"], np.ones(len(net.gen), dtype=bool))
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case30_with_psts_pandapower(folder: Path) -> None:
    """Create the pandapower case30 PST example scenario.

    Parameters
    ----------
    folder : Path
        Target folder that receives the generated grid, masks, and topology data.
    """
    net = pandapower_case30_with_psts_and_weak_branches()

    pp.runpp(net)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid_file_path)

    masks_path = folder / PREPROCESSING_PATHS["masks_path"]
    masks_path.mkdir(parents=True, exist_ok=True)

    relevant_node_mask = np.ones(len(net.bus), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["relevant_subs"], relevant_node_mask)

    line_mask = np.ones(len(net.line), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_reward"], line_mask)

    trafo_mask = np.ones(len(net.trafo), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_controllable"], trafo_mask)
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))


def case30_with_psts_powsybl(folder: Path) -> None:
    """Create the powsybl case30 PST example scenario.

    Parameters
    ----------
    folder : Path
        Target folder that receives the generated grid, masks, and topology data.
    """
    net = powsybl_case30_with_psts()
    create_busbar_b_in_ieee(net)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.ones(len(net.get_buses()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafos = sort_powsybl_element_frame_by_id(net.get_2_windings_transformers())
    pst_ids = net.get_phase_tap_changers().index
    trafo_mask = np.ones(len(trafos), dtype=bool)
    trafo_has_pst_tap = trafos.index.isin(pst_ids)
    trafo_mask_groups = np.full(len(trafos), -1, dtype=int)
    trafo_mask_groups[trafo_has_pst_tap] = np.arange(np.sum(trafo_has_pst_tap))
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_controllable"], trafo_has_pst_tap)

    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def node_breaker_folder_powsybl(folder: Path) -> None:
    """Copy over all data from the data folder"""
    net = basic_node_breaker_network_powsybl()
    file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    net.save(file_path)
    importer_parameters = CgmesImporterParameters(
        grid_model_file=file_path,
        data_folder=folder,
        area_settings=AreaSettings(
            cutoff_voltage=1,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
            dso_trafo_factors=LimitAdjustmentParameters(),
            dso_trafo_weight=1.0,
            border_line_factors=LimitAdjustmentParameters(),
            border_line_weight=1.0,
        ),
    )
    _ = preprocessing.convert_file(importer_parameters=importer_parameters)
    dir_fs = DirFileSystem(folder)
    lf_params = load_lf_params_from_fs(dir_fs, Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    load_grid(
        data_folder_dirfs=dir_fs,
        pandapower=False,
        parameters=PreprocessParameters(
            action_set_filter_bsdf_lodf=True,
            preprocess_bb_outages=True,
        ),
        status_update_fn=None,
        lf_params=lf_params,
    )
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def three_node_pst_example_folder_powsybl(folder: Path) -> None:
    """Create a simple 3 node example to test PST optimization"""
    net = three_node_pst_example()

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)
    rel_sub_mask = np.zeros(len(net.get_buses()), dtype=bool)
    rel_sub_mask[1:3] = True
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)
    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)
    trafo_mask = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    trafo_has_pst_tap = np.array([True, True], dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_controllable"], trafo_has_pst_tap)

    extract_station_info_powsybl(net, folder)
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(folder), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )


def complex_grid_battery_hvdc_svc_3w_trafo_data_folder(folder: Path, linear_pst: np.ndarray | None = None) -> NetworkData:
    """Create a preprocessed folder for create_complex_grid_battery_hvdc_svc_3w_trafo().

    Runs the importer and preprocessing.

    Parameter:
    folder: Path
        The root folder where the data is saved to.
    linear_pst: np.ndarray | None
        The linear PST coefficients to use during the creation of the grid.
    Returns:
    NetworkData
        The network data after preprocessing, which can be used for testing the consistency of the preprocessing step
    """
    # Connect the out of service line to bring the second PST operational
    net = create_complex_grid_battery_hvdc_svc_3w_trafo(linear_pst=linear_pst, connect_line_out_of_service=True)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    importer_parameters = CgmesImporterParameters(
        grid_model_file=folder / PREPROCESSING_PATHS["grid_file_path_powsybl"],
        data_folder=folder,
        area_settings=AreaSettings(
            cutoff_voltage=110.0,
            control_area=["BE"],
            view_area=["BE"],
            nminus1_area=["BE"],
            dso_trafo_factors=None,  # We deactivate this so the limits are the same in runner and solver
            dso_trafo_weight=1.0,
            border_line_factors=None,
            border_line_weight=1.0,
        ),
    )

    _ = preprocessing.convert_file(importer_parameters=importer_parameters)

    _info, _static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(folder)),
        pandapower=False,
    )
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK, DirFileSystem(str(folder)), Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    )

    return network_data


def busbar_outage_always_articulation_data_folder(folder: Path) -> NetworkData:
    """Create a preprocessed folder for the always-articulation busbar outage regression grid."""
    net = create_busbar_outage_always_articulation_grid()
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.zeros(len(net.get_buses()), dtype=bool)
    rel_sub_mask[net.get_buses().index.get_loc("VL2_0")] = True
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.ones(len(net.get_lines()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    extract_station_info_powsybl(net, folder)

    _info, _static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(folder)),
        pandapower=False,
    )
    save_lf_params_to_fs(
        CGMES_DISTRIBUTED_SLACK,
        DirFileSystem(str(folder)),
        Path(PREPROCESSING_PATHS["loadflow_parameters_file_path"]),
    )

    return network_data


def parallel_pst_data_folder(folder: Path) -> NetworkData:
    """Create a preprocessed folder for parallel_pst_example().

    Runs the importer and preprocessing.

    Parameter:
    folder: Path
        The root folder where the data is saved to.
    Returns:
    NetworkData
        The network data after preprocessing, which can be used for testing the consistency of the preprocessing step
    """
    net = parallel_pst_example()
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)

    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    importer_parameters = CgmesImporterParameters(
        grid_model_file=folder / PREPROCESSING_PATHS["grid_file_path_powsybl"],
        data_folder=folder,
        area_settings=AreaSettings(
            cutoff_voltage=1,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
            dso_trafo_factors=LimitAdjustmentParameters(),
            dso_trafo_weight=1.0,
            border_line_factors=LimitAdjustmentParameters(),
            border_line_weight=1.0,
        ),
    )

    _ = preprocessing.convert_file(importer_parameters=importer_parameters)

    preprocessing_parameters = PreprocessParameters(action_set_clip=2**4, preprocess_bb_outages=False)
    _info, _static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(folder),
        pandapower=False,
        status_update_fn=None,
        parameters=preprocessing_parameters,
    )

    return network_data
