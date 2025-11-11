"""Provides example grids for testing the dc_solver package."""

# ruff: noqa: PLR0915
import bz2
import datetime
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import pypowsybl
from beartype.typing import Literal, Optional
from networkx.algorithms.community import kernighan_lin_bisection
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_grid_helpers.pandapower.example_grids import (
    pandapower_case30_with_psts_and_weak_branches,
    pandapower_extended_case57,
    pandapower_extended_oberrhein,
    pandapower_non_converging_case57,
    pandapower_texas,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_grid_helpers.powsybl.example_grids import (
    create_busbar_b_in_ieee,
    extract_station_info_powsybl,
    powsybl_case30_with_psts,
    powsybl_case9241,
    powsybl_extended_case57,
    powsybl_texas,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    Station,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
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
    folder: str,
    filename_without_ext: str,
    element_type: Literal["gen", "sgen", "load"],
    attribute: Literal["p_mw", "q_mvar", "vm_pu"],
    save_grid2op_compatible: bool = False,
) -> None:
    """Saves the prod_p values of a list of pandapower networks to a csv file

    Parameters
    ----------
    timestep_nets : list[pp.pandapowerNet]
        The list of pandapower networks to extract the data from
    folder : str
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
    backend: BackendInterface, node_idx: int, pp_counters: Optional[PandapowerCounters]
) -> tuple[Station, Optional[PandapowerCounters]]:
    """Generate a random station for any backend

    This will create a Station object with 2 busbars, 1 coupler and a random assignment of assets
    to the busbars

    Parameters
    ----------
    backend : BackendInterface
        The backend to generate the topology for
    node_idx : int
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
    switchable_assets = []
    for branch_id, branch_type, branch_name, branch_node in zip(
        backend.get_branch_ids(),
        backend.get_branch_types(),
        backend.get_branch_names(),
        backend.get_from_nodes(),
        strict=True,
    ):
        if branch_node == node_idx:
            switchable_assets.append(
                SwitchableAsset(
                    grid_model_id=branch_id,
                    type=branch_type,
                    name=branch_name,
                    in_service=True,
                    branch_end="from",
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
                SwitchableAsset(
                    grid_model_id=branch_id,
                    type=branch_type,
                    name=branch_name,
                    in_service=True,
                    branch_end="to",
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
                SwitchableAsset(
                    grid_model_id=injection_id,
                    type=injection_type,
                    name=injection_name,
                    in_service=True,
                )
            )

    asset_switching_table = np.zeros((2, len(switchable_assets)), dtype=bool)
    is_on_a = np.random.rand(len(switchable_assets)) > 0.5
    asset_switching_table[0, is_on_a] = True
    asset_switching_table[1, ~is_on_a] = True

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

    return Station(
        grid_model_id=global_id,
        busbars=[
            Busbar(
                grid_model_id=bus_a_id,
                name=backend.get_node_names()[node_idx],
                int_id=0,
            ),
            Busbar(
                grid_model_id=bus_b_id,
                name=backend.get_node_names()[node_idx],
                int_id=1,
            ),
        ],
        couplers=[
            BusbarCoupler(
                grid_model_id=switch_id,
                busbar_from_id=0,
                busbar_to_id=1,
                open=False,
            ),
        ],
        assets=switchable_assets,
        asset_switching_table=asset_switching_table,
        asset_connectivity=np.ones_like(asset_switching_table, dtype=bool),
    ), pp_counters


def random_topology_info_backend(backend: BackendInterface, pp_counters: Optional[PandapowerCounters]) -> Topology:
    """Generate a random topology for any backend

    This will create an AssetTopology with a station created for each relevant node in the network

    Parameters
    ----------
    backend : BackendInterface
        The backend to generate the topology for
    pp_counters : PandapowerCounters
        The pandapower counters to generate valid IDs for the pandapower backend

    Returns
    -------
    Topology
        The generated topology
    """
    relevant_nodes = np.flatnonzero(backend.get_relevant_node_mask())
    stations = []
    for node_idx in relevant_nodes:
        new_station, pp_counters = random_station_info_backend(backend, node_idx, pp_counters)
        stations.append(new_station)

    return Topology(
        stations=stations,
        topology_id="random_topology",
        timestamp=datetime.datetime.now(),
    )


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
    if pandapower:
        backend = PandaPowerBackend(folder)
        pp_counters = PandapowerCounters(
            highest_switch_id=int(backend.net.switch.index.max()) if len(backend.net.switch) else 0,
            highest_bus_id=int(backend.net.bus.index.max()),
        )
    else:
        backend = PowsyblBackend(folder)
        pp_counters = None
    topo_info = random_topology_info_backend(backend, pp_counters)

    destination = folder / PREPROCESSING_PATHS["asset_topology_file_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        f.write(topo_info.model_dump_json(indent=2))


# ruff: noqa: PLR0915
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

    logs_path = folder / PREPROCESSING_PATHS["logs_path"]
    logs_path.mkdir(parents=True, exist_ok=True)
    with open(logs_path / "start_datetime.info", "w", encoding="utf-8") as f:
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

    logs_path = folder / PREPROCESSING_PATHS["logs_path"]
    logs_path.mkdir(parents=True, exist_ok=True)
    with open(logs_path / "start_datetime.info", "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    np.random.seed(0)
    random_topology_info(folder)


def case57_data_powsybl(folder: Path) -> None:
    """Create a powsybl test grid with a PST and some operational limits"""
    net = powsybl_extended_case57()
    create_busbar_b_in_ieee(net)
    pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK)

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

    output_path_logs = folder / PREPROCESSING_PATHS["logs_path"]
    output_path_logs.mkdir(parents=True, exist_ok=True)
    with open(output_path_logs / "start_datetime.info", "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

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


def texas_grid_pandapower(folder: Path) -> None:
    """An artificial texas grid with 2000 buses.

    Obtain the grid file from the ACTIVSg2000 website and remove generator costs to make it
    importable in pandapower.
    """
    net = pandapower_texas()
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

    line_mask = np.zeros(len(net.line), dtype=bool)
    line_mask[0:500] = 1
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafo_mask = np.zeros(len(net.trafo), dtype=bool)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)

    np.random.seed(0)
    random_topology_info(folder)


def texas_grid_powsybl(folder: Path) -> None:
    """An artificical texas grid with 2000 buses.

    Obtain the grid file from the ACTIVSg2000 website and remove generator costs to make it
    importable in pandapower.
    """
    net = powsybl_texas()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)
    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.zeros(len(net.get_buses()), dtype=bool)
    rel_sub_mask[0:50] = True
    np.save(output_path_masks / NETWORK_MASK_NAMES["relevant_subs"], rel_sub_mask)

    line_mask = np.zeros(len(net.get_lines()), dtype=bool)
    line_mask[0:500] = 1
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_reward"], line_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["line_for_nminus1"], line_mask)

    trafo_mask = np.zeros(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)

    extract_station_info_powsybl(net, folder)


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


def case300_powsybl(folder: Path) -> None:
    """The case300 network with a powsybl grid"""
    net = pypowsybl.network.create_ieee300()
    create_busbar_b_in_ieee(net)
    os.makedirs(folder, exist_ok=True)
    grid_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_path)

    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    rel_sub_mask = np.zeros(len(net.get_buses()), dtype=bool)
    rel_sub_mask[0:50] = True
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


def case9241_pandapower(data_folder: Path) -> None:  # noqa: PLR0912, C901
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
    net.trafo.loc[net.trafo.vn_lv_kv == net.trafo.vn_hv_kv, "tap_changer_type"] = True
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

    # Partition the grid into 4 regions of roughly equal size
    part1, part2 = kernighan_lin_bisection(graph, seed=np.random.randint(2**32))
    part11, part12 = kernighan_lin_bisection(graph.subgraph(part1), seed=np.random.randint(2**32))
    part21, part22 = kernighan_lin_bisection(graph.subgraph(part2), seed=np.random.randint(2**32))

    regions = [part11, part12, part21, part22]

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
            "trafo_pst_controllable": np.ones(len(net.trafo), dtype=bool),
        }
    )

    for key, mask in region_masks.items():
        np.save(masks_path / f"{key}.npy", mask)

    logs_path = data_folder / PREPROCESSING_PATHS["logs_path"]
    logs_path.mkdir(parents=True, exist_ok=True)
    with open(logs_path / "start_datetime.info", "w", encoding="utf-8") as f:
        f.write(str(datetime.datetime.now()))

    np.random.seed(0)
    random_topology_info(data_folder)


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


def case30_with_psts(folder: Path) -> None:
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
    np.save(masks_path / NETWORK_MASK_NAMES["trafo_pst_controllable"], trafo_mask)
    random_topology_info(folder)


def case30_with_psts_powsybl(folder: Path) -> None:
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

    trafo_mask = np.ones(len(net.get_2_windings_transformers()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_reward"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_for_nminus1"], trafo_mask)
    np.save(output_path_masks / NETWORK_MASK_NAMES["trafo_pst_controllable"], trafo_mask)

    gen_mask = np.ones(len(net.get_generators()), dtype=bool)
    np.save(output_path_masks / NETWORK_MASK_NAMES["generator_for_nminus1"], gen_mask)

    extract_station_info_powsybl(net, folder)


def node_breaker_folder_powsybl(folder: Path) -> None:
    """Copy over all data from the data folder"""
    source = Path(__file__).parent.parent.parent / "tests" / "files" / "test_grid_node_breaker"
    shutil.copytree(source, folder, dirs_exist_ok=True)
