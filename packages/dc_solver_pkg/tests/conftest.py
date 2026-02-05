# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import bz2
import gc
import json
import logging
import os
import shutil
import time
import uuid
from copy import deepcopy
from pathlib import Path

import chex
import docker
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandapower as pp
import pandera
import pypowsybl
import pytest
import ray
import yaml
from beartype.typing import Generator, List, Literal
from docker import DockerClient
from docker.models.containers import Container
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from pypowsybl.network import Network
from toop_engine_dc_solver.example_classes import (
    get_basic_node_breaker_topology,
)
from toop_engine_dc_solver.example_grids import (
    case14_pandapower,
    case30_with_psts,
    case57_data_powsybl,
    node_breaker_folder_powsybl,
    oberrhein_data,
)
from toop_engine_dc_solver.jax.injections import (
    convert_action_index_to_numpy,
    random_injection,
)
from toop_engine_dc_solver.jax.inputs import save_static_information
from toop_engine_dc_solver.jax.inspector import is_valid_batch
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    convert_branch_topo_vect,
    get_random_topology_results,
    product_action_set,
    random_topology,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.postprocess.write_aux_data import write_aux_data
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax, load_grid
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_n_minus_2_safe_branches,
)
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_network_data_from_interface,
    load_network_data,
    save_network_data,
)
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_dc_solver.preprocess.preprocess import (
    add_nodal_injections_to_network_data,
    compute_bridging_branches,
    compute_psdf_if_not_given,
    compute_ptdf_if_not_given,
    preprocess,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import table_id
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    case14_matching_asset_topo_powsybl,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.asset_topology import (
    Busbar,
    BusbarCoupler,
    Station,
    SwitchableAsset,
    Topology,
)
from toop_engine_interfaces.asset_topology_helpers import load_asset_topology
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    PreprocessParameters,
    UcteImporterParameters,
)

chex.set_n_cpu_devices(2)
jax.config.update("jax_enable_x64", True)
## Set up loggers
# JAX
jax.config.update("jax_logging_level", "WARNING")

# NUMBA
logging.getLogger("numba").setLevel(logging.WARNING)

# pandera
config = pandera.config.PanderaConfig(
    validation_enabled=True, validation_depth=pandera.config.ValidationDepth.SCHEMA_AND_DATA
)
pandera.config.reset_config_context(config)


@pytest.fixture(autouse=True)
def shutdown_ray():
    yield
    # If we're running on CI, we regularly run out of memory
    # Hence, we cleanup ray after each test where ray was used
    if "CLEANUP_RAY_AFTER_TESTS" in os.environ and ray.is_initialized():
        ray.shutdown()
        gc.collect()


@pytest.fixture(scope="session")
def case14_data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("case14")
    case14_pandapower(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def case14_topologies() -> np.ndarray:
    file_path = Path(__file__).parent.parent / "data" / "14_bus" / "batch_0_0.hdf5"

    with h5py.File(file_path, "r") as f:
        assert "topo_sel_sorted" in f
        topo_sel_sorted = np.array(f["topo_sel_sorted"][:], dtype=bool)
    return topo_sel_sorted


@pytest.fixture(scope="session")
def case14_network_data(case14_data_folder: Path) -> NetworkData:
    fs_dir = DirFileSystem(str(case14_data_folder))
    backend = PandaPowerBackend(fs_dir)
    network_data = preprocess(backend)

    return network_data


@pytest.fixture(scope="session")
def _jax_inputs(
    case14_network_data: NetworkData,
) -> tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation]:
    static_information = convert_to_jax(
        case14_network_data,
        number_most_affected=20,
        number_max_out_in_most_affected=5,
        number_most_affected_n_0=20,
        batch_size_bsdf=4,
        batch_size_injection=16,
        buffer_size_injection=128,
        enable_bb_outage=False,
        enable_nodal_inj_optim=False,
        precision_percent=0.1,
    )
    action_set = static_information.dynamic_information.action_set

    # Enumerate all possible actions exhaustively on case14
    computations = product_action_set([0, 1, 2, 3, 4], action_set, limit_n_subs=3)
    is_valid = is_valid_batch(
        computations.topologies,
        computations.sub_ids,
        static_information,
    )
    computations = computations[is_valid]

    candidates = random_injection(
        rng_key=jax.random.PRNGKey(0),
        n_generators_per_sub=static_information.dynamic_information.generators_per_sub,
        n_inj_per_topology=5,
        for_topology=computations,
    )

    disconnectable_branches = find_n_minus_2_safe_branches(
        from_node=case14_network_data.from_nodes,
        to_node=case14_network_data.to_nodes,
        number_of_branches=case14_network_data.ptdf.shape[0],
        number_of_nodes=case14_network_data.ptdf.shape[1],
    )
    disconnectable_branches = jnp.flatnonzero(disconnectable_branches)

    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            disconnectable_branches=disconnectable_branches,
        ),
    )
    return computations, candidates, static_information


@pytest.fixture(scope="function")
def jax_inputs(
    _jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation]:
    return deepcopy(_jax_inputs)


@pytest.fixture
def static_information_with_multi_outages(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> StaticInformation:
    static_information = replace(
        jax_inputs[2],
        dynamic_information=replace(
            jax_inputs[2].dynamic_information,
            multi_outage_branches=[
                jnp.array([[0, 8, 12], [0, 4, 8], [0, 4, 8]], dtype=int),
                jnp.array([[0, 8], [12, 8], [12, 0]], dtype=int),
            ],
            multi_outage_nodes=[
                jnp.array([[-1], [-1], [2]], dtype=int),
                jnp.zeros((3, 0), dtype=int),
            ],
        ),
    )
    return static_information


def compress_bz2(source_file: str) -> None:
    """Compress a file with bz2 and remove the original file

    Parameters
    ----------
        source_file (str): The file to compress
    """
    dest_path = source_file + ".bz2"
    with open(source_file, "rb") as source, bz2.BZ2File(dest_path, "wb") as dest:
        dest.writelines(source)
    os.remove(source_file)


def save_timestep_data(
    timestep_nets: List[pp.pandapowerNet],
    folder: str,
    filename_without_ext: str,
    element_type: Literal["gen", "sgen", "load"],
    attribute: Literal["p_mw", "q_mvar", "vm_pu"],
) -> None:
    """Saves the prod_p values of a list of pandapower networks to a csv file

    Parameters
    ----------
        timestep_nets (List[pp.pandapowerNet]): The pandapower networks to save
        folder (str): Where to save to
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
    values.to_csv(os.path.join(folder, f"{filename_without_ext}.csv"), index=False, sep=";")

    # Compress with bz2
    compress_bz2(os.path.join(folder, f"{filename_without_ext}.csv"))
    np.save(os.path.join(folder, f"{filename_without_ext}.npy"), values.values.astype(float))


@pytest.fixture(scope="session")
def oberrhein_data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("grid_oberrhein_data")
    oberrhein_data(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def data_folder(oberrhein_data_folder: Path) -> Path:
    return oberrhein_data_folder


@pytest.fixture(scope="session")
def network_data(data_folder: Path) -> NetworkData:
    fs_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(fs_dir)
    network_data = extract_network_data_from_interface(backend)
    return network_data


@pytest.fixture(scope="session")
def network_data_with_ptdf(network_data: NetworkData) -> NetworkData:
    network_data = compute_ptdf_if_not_given(network_data)
    return network_data


@pytest.fixture(scope="session")
def network_data_filled(network_data_with_ptdf: NetworkData) -> NetworkData:
    network_data = compute_psdf_if_not_given(network_data_with_ptdf)
    network_data = add_nodal_injections_to_network_data(network_data)
    network_data = compute_bridging_branches(network_data)
    return network_data


@pytest.fixture(scope="session")
def network_data_preprocessed(data_folder: Path, oberrhein_outage_station_busbars_map: dict) -> NetworkData:
    class TestBackend(PandaPowerBackend):
        def get_busbar_outage_map(self):
            return oberrhein_outage_station_busbars_map

    fs_dir = DirFileSystem(str(data_folder))
    backend = TestBackend(fs_dir)
    network_data = preprocess(backend, parameters=PreprocessParameters(enable_bb_outage=True))
    return network_data


@pytest.fixture(scope="session")
def preprocessed_data_folder(data_folder: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("result")
    tmp_grid_file_path_pandapower = tmp_path / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    tmp_grid_file_path_pandapower.parent.mkdir(parents=True, exist_ok=True)
    temp_network_data_file_path = tmp_path / PREPROCESSING_PATHS["network_data_file_path"]
    temp_network_data_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_static_information_file_path = tmp_path / PREPROCESSING_PATHS["static_information_file_path"]
    temp_static_information_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy over the grid file
    shutil.copy(
        data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"],
        tmp_grid_file_path_pandapower,
    )

    # Extract data from the backend, run preprocessing
    fs_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(fs_dir)
    network_data = preprocess(backend)
    save_network_data(temp_network_data_file_path, network_data)
    static_information = convert_to_jax(network_data, enable_bb_outage=False)
    save_static_information(temp_static_information_file_path, static_information)
    write_aux_data(data_folder=data_folder, network_data=network_data)

    # Generate random "optimization results"
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=10),
    )
    best_actions = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=10,
        unsplit_prob=0,
        topo_vect_format=False,
    )
    best_branch = convert_action_set_index_to_topo(
        topologies=best_actions,
        action_set=static_information.dynamic_information.action_set,
    )
    best_branch = convert_branch_topo_vect(
        best_branch.topologies,
        best_branch.sub_ids,
        static_information.solver_config.branches_per_sub,
    )
    best_injections = convert_action_index_to_numpy(
        action_index=best_actions.action,
        action_set=static_information.dynamic_information.action_set,
        n_generators_per_sub=static_information.dynamic_information.generators_per_sub,
    )
    best_disconnections = jax.random.choice(
        jax.random.PRNGKey(0),
        len(static_information.dynamic_information.disconnectable_branches),
        shape=(10, 1),
    )

    # Then save in a json file similar to the optimizer output
    best = [
        {
            "branch": b.tolist(),
            "injection": i.tolist(),
            "disconnection": d.tolist(),
            "actions": [int(x) for x in a if x < static_information.n_actions],
            "metrics": {"n_failures": 0},
        }
        for a, b, i, d in zip(best_actions.action, best_branch, best_injections, best_disconnections)
    ]

    post_process_file_path = (
        tmp_path / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    post_process_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(post_process_file_path, "w") as f:
        json.dump({"best_topos": best, "initial_metrics": {"n_failures": 0}}, f)

    return tmp_path


@pytest.fixture(scope="session")
def loaded_net(data_folder: Path) -> pp.pandapowerNet:
    grid_file_path = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    pp.rundcpp(net)
    return net


@pytest.fixture(scope="session")
def benchmark_config_file(
    tmp_path_factory: pytest.TempPathFactory,
    _jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> Path:
    orig_path = Path(__file__).parent / "files" / "jax" / "benchmarks" / "test.yaml"
    tmp_path = tmp_path_factory.mktemp("benchmarks")

    save_static_information(tmp_path / "static_information.hdf5", _jax_inputs[2])

    with open(orig_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Replace the path to the static information with the case14 file
    for i in range(len(config["benchmarks"])):
        config["benchmarks"][i]["static_information_path"] = str(tmp_path / "static_information.hdf5")

    with open(tmp_path / "test.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return tmp_path / "test.yaml"


@pytest.fixture()
def benchmark_config(benchmark_config_file: Path) -> dict:
    with open(benchmark_config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="session")
def data_folder_with_more_branches(
    data_folder: Path,
    network_data_preprocessed: NetworkData,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    temp_dir = tmp_path_factory.mktemp("more_branches")
    # Copy everything over
    shutil.copytree(data_folder, temp_dir, dirs_exist_ok=True)
    grid_file_path = temp_dir / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    mask_path = temp_dir / PREPROCESSING_PATHS["masks_path"]

    # Redo the same network data object but with one more branch that ends in a relevant substation
    # Find a suitable branch
    nd = network_data_preprocessed
    net = pp.from_json(grid_file_path)
    relevant_nodes = [table_id(nd.node_ids[i]) for i in np.flatnonzero(nd.relevant_node_mask)]

    def _try_add_to_mask(filename: str, index: int, value: bool) -> None:
        try:
            data = np.load(mask_path / filename)
            np.save(mask_path / filename, np.insert(data, index, value, axis=0))
        except FileNotFoundError:
            pass

    np.random.seed(42)
    for node in relevant_nodes:
        other_buses = net.bus.index
        other_buses = other_buses[~other_buses.isin(relevant_nodes)]
        line_id = pp.create_line(
            net=net,
            from_bus=node,
            to_bus=np.random.choice(other_buses),
            std_type="NA2XS2Y 1x95 RM/25 12/20 kV",
            length_km=1,
        )
        line_index = np.flatnonzero(net.line.index == line_id)[0]

        _try_add_to_mask(NETWORK_MASK_NAMES["line_for_reward"], line_index, True)
        _try_add_to_mask(NETWORK_MASK_NAMES["line_for_nminus1"], line_index, True)
        _try_add_to_mask(NETWORK_MASK_NAMES["line_disconnectable"], line_index, True)

    pp.to_json(net, grid_file_path)

    return temp_dir


@pytest.fixture(scope="session")
def powsybl_case57_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_dir = tmp_path_factory.mktemp("powsybl_case57")
    case57_data_powsybl(temp_dir)
    return temp_dir


@pytest.fixture(scope="session")
def powsybl_data_folder(powsybl_case57_folder: Path) -> Path:
    return powsybl_case57_folder


@pytest.fixture(scope="session")
def loaded_powsybl_net(powsybl_data_folder: Path) -> pypowsybl.network.Network:
    grid_file_path = powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    net = pypowsybl.network.load(grid_file_path)
    pypowsybl.loadflow.run_ac(net)
    return net


@pytest.fixture(scope="session")
def preprocessed_powsybl_data_folder(powsybl_data_folder: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("powsybl_result")
    tmp_grid_file_path = tmp_path / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    tmp_grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_network_data_file_path = tmp_path / PREPROCESSING_PATHS["network_data_file_path"]
    temp_network_data_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_static_information_file_path = tmp_path / PREPROCESSING_PATHS["static_information_file_path"]
    temp_static_information_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy over the grid file
    shutil.copy(
        powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"],
        tmp_grid_file_path,
    )

    # Extract data from the backend, run preprocessing
    fs_dir = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(fs_dir)
    network_data = preprocess(backend)
    save_network_data(temp_network_data_file_path, network_data)
    static_information = convert_to_jax(network_data, enable_bb_outage=False)
    save_static_information(temp_static_information_file_path, static_information)
    write_aux_data(data_folder=tmp_path, network_data=network_data)

    # Generate random "optimization results"
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=10, limit_n_subs=3),
    )
    best = get_random_topology_results(static_information, random_seed=50)
    post_process_file_path = (
        tmp_path / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    post_process_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(post_process_file_path, "w") as f:
        json.dump(
            {"best_topos": best, "initial_metrics": {"n_failures": 0, "fitness": 234}},
            f,
        )

    return tmp_path


@pytest.fixture(scope="session")
def oberrhein_outage_station_busbars_map(oberrhein_data_folder: Path) -> dict:
    stations_desired = ["71%%bus", "98%%bus", "130%%bus", "8%%bus", "58%%bus", "157%%bus", "165%%bus"]

    asset_topo = load_asset_topology(oberrhein_data_folder / PREPROCESSING_PATHS["asset_topology_file_path"])
    retval = {}
    for station in asset_topo.stations:
        if station.grid_model_id in stations_desired:
            # Get the busbar IDs for the station
            busbars = [bb.grid_model_id for bb in station.busbars]
            # Create a mapping of the station to its busbars
            retval[station.grid_model_id] = busbars

    # 71%%bus, 157%%bus, "165%%bus" are relevant subs
    return retval


@pytest.fixture(scope="session")
def case30_data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("case30_with_psts")
    case30_with_psts(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def basic_node_breaker_grid_v1() -> Network:
    return basic_node_breaker_network_powsybl()


@pytest.fixture(scope="session")
def node_breaker_grid_preprocessed_data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("node_breaker_grid_preprocessed")
    node_breaker_folder_powsybl(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    stats, static_information, _ = load_grid(filesystem_dir)
    assert stats.n_relevant_subs > 0

    best_actions = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=1,
        batch_size=10,
        unsplit_prob=0,
        topo_vect_format=False,
    )
    best_disconnections = jax.random.choice(
        jax.random.PRNGKey(0),
        len(static_information.dynamic_information.disconnectable_branches),
        shape=(10, 1),
    )

    # Then save in a json file similar to the optimizer output
    best = [
        {
            "disconnection": d.tolist(),
            "actions": [int(x) for x in a if x < static_information.n_actions],
            "metrics": {"n_failures": 0},
        }
        for a, d in zip(best_actions.action, best_disconnections)
    ]
    post_process_file_path = (
        tmp_path / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    post_process_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(post_process_file_path, "w") as f:
        json.dump(
            {"best_topos": best, "initial_metrics": {"n_failures": 0, "fitness": 234}},
            f,
        )

    return tmp_path


@pytest.fixture(scope="session")
def test_grid_folder_path() -> Path:
    return Path(__file__).parent / "files" / "test_grid_node_breaker"


@pytest.fixture(scope="session")
def network_data_test_grid(test_grid_folder_path: Path, outage_map_test_grid: dict) -> NetworkData:
    class TestBackend(PowsyblBackend):
        def get_busbar_outage_map(self):
            return outage_map_test_grid

    fs_dir = DirFileSystem(str(test_grid_folder_path))
    backend = TestBackend(fs_dir, distributed_slack=False)
    network_data = preprocess(backend, parameters=PreprocessParameters(enable_bb_outage=True))
    return network_data


@pytest.fixture(scope="session")
def outage_map_test_grid():
    return {
        "VL2_0": ["BBS2_1", "BBS2_2", "BBS2_3"],
        "VL3_0": ["BBS3_1", "BBS3_2"],
    }


@pytest.fixture(scope="session")
def jax_inputs_test_grid(
    network_data_test_grid: NetworkData,
) -> tuple[ActionIndexComputations, StaticInformation]:
    static_information = convert_to_jax(network_data_test_grid, enable_bb_outage=True)

    topo_indices: ActionIndexComputations = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=None,
        batch_size=1,
        topo_vect_format=False,
    )
    return topo_indices, static_information


@pytest.fixture(scope="session")
def jax_inputs_oberrhein(network_data_preprocessed: NetworkData) -> tuple[ActionIndexComputations, StaticInformation]:
    static_information = convert_to_jax(
        network_data_preprocessed,
        enable_bb_outage=True,
    )
    topo_indices: ActionIndexComputations = random_topology(
        jax.random.PRNGKey(42),
        branch_action_set=static_information.dynamic_information.action_set,
        limit_n_subs=2,
        batch_size=static_information.solver_config.batch_size_bsdf,
        topo_vect_format=False,
    )
    return topo_indices, static_information


@pytest.fixture(scope="session")
def ucte_file() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_ucte_powsybl_example.uct")
    return ucte_file


@pytest.fixture(scope="session")
def create_ucte_data_path(ucte_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("ucte_grid")
    create_ucte_data_folder(tmp_path, ucte_file=ucte_file)
    return tmp_path


@pytest.fixture(scope="function")
def basic_ucte_data_folder(create_ucte_data_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    powsybl_data_folder = create_ucte_data_path
    tmp_path = tmp_path_factory.mktemp("ucte_grid", numbered=True)

    # Copy over the grid file
    shutil.copytree(
        powsybl_data_folder,
        tmp_path,
        dirs_exist_ok=True,
    )

    return tmp_path


@pytest.fixture(scope="session")
def case14_data_with_asset_topo_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture to create a temporary folder for the case14 test."""
    tmp_path = tmp_path_factory.mktemp("case14")
    case14_matching_asset_topo_powsybl(tmp_path)
    return tmp_path


@pytest.fixture
def case14_data_with_asset_topo(case14_data_with_asset_topo_path: Path) -> tuple[Path, Topology]:
    """Fixture to create a temporary folder for the case14 test."""
    with open(case14_data_with_asset_topo_path / PREPROCESSING_PATHS["asset_topology_file_path"], "r") as f:
        asset_topology = Topology.model_validate_json(f.read())
    return case14_data_with_asset_topo_path, asset_topology


@pytest.fixture(scope="session")
def basic_node_breaker_topology() -> Topology:
    """Fixture to create a realized topology with a node breaker topology.
    Based on example_grid.basic_node_breaker_network_powsybl().
    """
    return get_basic_node_breaker_topology()


@pytest.fixture(scope="session")
def mock_station() -> Station:
    asset1 = SwitchableAsset(grid_model_id="branch_01", in_service=True, branch_end="from", type="line")
    asset2 = SwitchableAsset(grid_model_id="branch_02", in_service=True, branch_end="to", type="line")
    asset3 = SwitchableAsset(grid_model_id="branch_03", in_service=True, branch_end="from", type="line")
    asset4 = SwitchableAsset(grid_model_id="branch_04", in_service=True, branch_end="to", type="line")

    # Create mock Busbar objects
    busbar_0 = Busbar(grid_model_id="busbar_0", int_id=1)
    busbar_1 = Busbar(grid_model_id="busbar_1", int_id=2)
    busbar_2 = Busbar(grid_model_id="busbar_2", int_id=3)
    busbar_3 = Busbar(grid_model_id="busbar_3", int_id=4)
    busbar_4 = Busbar(grid_model_id="busbar_4", int_id=5)

    # 3
    # |
    # 1-2-3-4-5

    # Create a mock Station object
    station = Station(
        grid_model_id="station_1",
        busbars=[busbar_0, busbar_1, busbar_2, busbar_3, busbar_4],
        couplers=[
            BusbarCoupler(
                grid_model_id="VL4_BREAKER",
                type="busbar_coupler",
                name="VL4_BREAKER",
                busbar_from_id=1,
                busbar_to_id=2,
                open=False,
                in_service=True,
            ),
            BusbarCoupler(
                grid_model_id="VL5_BREAKER",
                type="busbar_coupler",
                name="VL5_BREAKER",
                busbar_from_id=2,
                busbar_to_id=3,
                open=False,
                in_service=True,
            ),
            BusbarCoupler(
                grid_model_id="VL6_BREAKER",
                type="busbar_coupler",
                name="VL6_BREAKER",
                busbar_from_id=3,
                busbar_to_id=4,
                open=False,
                in_service=True,
            ),
            BusbarCoupler(
                grid_model_id="VL7_BREAKER",
                type="busbar_coupler",
                name="VL7_BREAKER",
                busbar_from_id=4,
                busbar_to_id=5,
                open=False,
                in_service=True,
            ),
            BusbarCoupler(
                grid_model_id="VL9_BREAKER",
                type="busbar_coupler",
                name="VL9_BREAKER",
                busbar_from_id=1,
                busbar_to_id=3,
                open=False,
                in_service=True,
            ),
        ],
        assets=[asset1, asset2, asset3, asset4],
        asset_switching_table=np.array(
            [
                [True, False, True, False],  # Busbar 0
                [False, True, False, False],  # Busbar 1
                [False, False, True, True],  # Busbar 2
                [False, False, False, True],  # Busbar 3
                [False, False, False, True],  # Busbar 4
            ],
            dtype=bool,
        ),
    )
    return station


@pytest.fixture(scope="session")
def docker_client() -> DockerClient:
    return docker.from_env()


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_connection_str(kafka_container: Container) -> str:
    for _ in range(100):
        kafka_container.reload()
        if kafka_container.status == "running":
            return f"localhost:{kafka_container.ports['9092/tcp'][0]['HostPort']}"

        time.sleep(0.1)
    raise RuntimeError("Could not get Kafka port")


def make_topic(kafka_container: Container, topic: str) -> None:
    # Remove existing topic if it exists (due to previous tests)
    exit_code, output = kafka_container.exec_run(
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic {topic} --if-exists"
    )
    assert exit_code == 0, output.decode()

    # Wait for the topic to be deleted (max 3 seconds)
    for _ in range(30):
        exit_code, output = kafka_container.exec_run(
            f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic {topic}"
        )
        if exit_code != 0:
            break
        time.sleep(0.1)

    # Create new topic
    exit_code, output = kafka_container.exec_run(
        f"/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic {topic} --partitions 2 --replication-factor 1"
    )
    assert exit_code == 0, output.decode()


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_command_topic(kafka_container: Container) -> str:
    topic = f"command_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_results_topic(kafka_container: Container) -> str:
    topic = f"results_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            "kafka",
            marks=pytest.mark.xdist_group("kafka"),
        ),
    ],
)
def kafka_heartbeat_topic(kafka_container: Container) -> str:
    topic = f"heartbeat_topic_{uuid.uuid4().hex[:8]}"
    make_topic(kafka_container, topic)
    return topic


def kill_all_containers_with_name(docker_client: DockerClient, target_name: str) -> None:
    """Kill all docker containers with the given name.

    There might be left-over containers from previous tests in case the test crashed and didn't clean up properly."""
    # Get all containers
    containers: list[Container] = docker_client.containers.list()

    containers_to_kill = []
    for container in containers:
        if container.name == target_name:
            containers_to_kill.append(container.id)

    for container_id in set(containers_to_kill):
        container = docker_client.containers.get(container_id)
        container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def kafka_container(docker_client: DockerClient) -> Generator[Container, None, None]:
    # Kill all containers that expose port 9092
    kill_all_containers_with_name(docker_client, "test_kafka")

    container = docker_client.containers.run(
        "apache/kafka",
        detach=True,
        name="test_kafka",
        auto_remove=True,
        ports={"9092/tcp": 9092},
        environment={
            "KAFKA_NODE_ID": "1",
            "KAFKA_PROCESS_ROLES": "broker,controller",
            "KAFKA_LISTENERS": "PLAINTEXT://:9092,CONTROLLER://:9093",
            "KAFKA_ADVERTISED_LISTENERS": "PLAINTEXT://localhost:9092",
            "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
            "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@localhost:9093",
            "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
            "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS": "0",
            "KAFKA_NUM_PARTITIONS": "1",
            "KAFKA_AUTO_CREATE_TOPICS_ENABLE": "false",
            "KAFKA_DELETE_TOPIC_ENABLE": "true",
            "KAFKA_LOG4J_LOGGERS": "kafka.controller=DEBUG",
        },
    )
    for log_line in container.logs(stream=True):
        if "Kafka Server started" in log_line.decode():
            break
    yield container
    container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def kafka_connection_str(kafka_container: Container) -> str:
    for _ in range(100):
        kafka_container.reload()
        if kafka_container.status == "running":
            return f"localhost:{kafka_container.ports['9092/tcp'][0]['HostPort']}"

        time.sleep(0.1)
    raise RuntimeError("Could not get Kafka port")


@pytest.fixture(scope="module")
def overlapping_branch_data(
    preprocessed_powsybl_data_folder: Path,
) -> tuple[NetworkData, StaticInformation, list[dict]]:
    """
    Fixture to load the network data for testing non-overlapping branch masks.
    """
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    outage_mask = network_data.outaged_branch_mask
    # Make sure all branch masks are identical
    updated_outage_mask = outage_mask
    updated_monitored_branch_mask = outage_mask
    updated_disconnection_mask = outage_mask
    assert np.sum(updated_outage_mask) > 10, "There should be at least 10 branches set to true"
    assert np.array_equal(updated_outage_mask, updated_disconnection_mask), "The branch masks should be equal"
    assert np.array_equal(updated_outage_mask, updated_monitored_branch_mask), "The branch masks should be equal"
    network_data = replace(
        network_data,
        disconnectable_branch_mask=updated_disconnection_mask,
        outaged_branch_mask=updated_outage_mask,
        monitored_branch_mask=updated_monitored_branch_mask,
    )
    static_information = convert_to_jax(network_data, enable_bb_outage=False, batch_size_bsdf=10, limit_n_subs=2)
    # Generate random "optimization results"
    best_actions = get_random_topology_results(static_information)
    check_branches_match_between_network_data_and_static_info(network_data, static_information)
    return network_data, static_information, best_actions


def check_branches_match_between_network_data_and_static_info(network_data, static_information):
    branch_names = np.array(network_data.branch_names)
    assert all(
        branch_names[network_data.disconnectable_branch_mask]
        == branch_names[static_information.dynamic_information.disconnectable_branches]
    ), "Mismatch between static info and network data for disconnectable branches"
    assert all(
        branch_names[network_data.outaged_branch_mask]
        == branch_names[static_information.dynamic_information.branches_to_fail]
    ), "Mismatch between static info and network data for outaged branches"
    assert all(
        branch_names[network_data.monitored_branch_mask]
        == branch_names[static_information.dynamic_information.branches_monitored]
    ), "Mismatch between static info and network data for monitored branches"


@pytest.fixture(scope="module")
def non_overlapping_branch_data(
    preprocessed_powsybl_data_folder: Path,
) -> tuple[NetworkData, StaticInformation, list[dict]]:
    """
    Fixture to load the network data for testing non-overlapping branch masks.
    """
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    disconnection_mask = network_data.disconnectable_branch_mask
    outage_mask = network_data.outaged_branch_mask
    monitored_branch_mask = network_data.monitored_branch_mask

    # Make sure there is no overlap between the branch masks
    # 1) Exclude the disconnections from the outage mask
    updated_outage_mask = outage_mask & ~disconnection_mask
    outaged_indizes = np.where(updated_outage_mask)[0]
    updated_outage_mask[outaged_indizes[10:]] = False
    # 2) Exclude the outage branches from the monitored branches
    updated_monitored_branch_mask = monitored_branch_mask & ~updated_outage_mask & ~disconnection_mask
    assert np.any(disconnection_mask), "There should be at least one disconnectable branch"
    assert sum(updated_outage_mask) == 10, "There should be at least one outaged branch"
    assert sum(updated_monitored_branch_mask) >= 20, "There should be at least 20 monitored branch"

    assert not np.any(disconnection_mask & updated_outage_mask), "Disconnection and outage branches should not overlap"
    assert not np.any(updated_outage_mask & updated_monitored_branch_mask), (
        "Outage and monitored branches should not overlap"
    )
    assert not np.any(disconnection_mask & updated_monitored_branch_mask), (
        "Disconnection and monitored branches should not overlap"
    )

    network_data = replace(
        network_data,
        disconnectable_branch_mask=disconnection_mask,
        outaged_branch_mask=updated_outage_mask,
        monitored_branch_mask=updated_monitored_branch_mask,
    )
    static_information = convert_to_jax(network_data, enable_bb_outage=False, batch_size_bsdf=10, limit_n_subs=2)
    # Generate random "optimization results"
    best_actions = get_random_topology_results(static_information)
    check_branches_match_between_network_data_and_static_info(network_data, static_information)
    return network_data, static_information, best_actions


@pytest.fixture(scope="module")
def overlapping_monitored_and_disconnected_branch_data(
    preprocessed_powsybl_data_folder: Path,
) -> tuple[NetworkData, StaticInformation, list[dict]]:
    """
    Fixture to load the network data for testing partially overlapping branch masks.
    """
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    disconnection_mask = network_data.disconnectable_branch_mask
    outage_mask = network_data.outaged_branch_mask
    monitored_branch_mask = network_data.monitored_branch_mask

    # Make sure there is no overlap between the branch masks
    # 1) Exclude the disconnections from the outage mask

    updated_outage_mask = outage_mask & ~disconnection_mask
    outaged_indizes = np.where(updated_outage_mask)[0]
    updated_outage_mask[outaged_indizes[10:]] = False
    # 2) Exclude the outage branches from the monitored branches
    # Make sure that the disconnection branches are included in the monitored branches
    updated_monitored_branch_mask = (monitored_branch_mask & ~updated_outage_mask) | disconnection_mask
    assert np.any(disconnection_mask), "There should be at least one disconnectable branch"
    assert sum(updated_outage_mask) == 10, "There should be at least one outaged branch"
    assert sum(updated_monitored_branch_mask) >= 20, "There should be at least 20 monitored branch"

    assert not np.any(disconnection_mask & updated_outage_mask), "Disconnection and outage branches should not overlap"
    assert not np.any(updated_outage_mask & updated_monitored_branch_mask), (
        "Outage and monitored branches should not overlap"
    )
    assert np.any(disconnection_mask & updated_monitored_branch_mask), "Disconnection and monitored branches should overlap"

    network_data = replace(
        network_data,
        disconnectable_branch_mask=disconnection_mask,
        outaged_branch_mask=updated_outage_mask,
        monitored_branch_mask=updated_monitored_branch_mask,
    )
    static_information = convert_to_jax(network_data, enable_bb_outage=False, batch_size_bsdf=10, limit_n_subs=2)
    # Generate random "optimization results"
    best_actions = get_random_topology_results(static_information)
    check_branches_match_between_network_data_and_static_info(network_data, static_information)
    return network_data, static_information, best_actions


@pytest.fixture(scope="session")
def create_complex_grid_battery_hvdc_svc_3w_trafo_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("complex_grid")
    create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def complex_grid_battery_hvdc_svc_3w_trafo_data_folder(
    create_complex_grid_battery_hvdc_svc_3w_trafo_data_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    powsybl_data_folder = create_complex_grid_battery_hvdc_svc_3w_trafo_data_path
    tmp_path = tmp_path_factory.mktemp("complex_grid", numbered=True)

    # Copy over the grid file
    shutil.copytree(
        powsybl_data_folder,
        tmp_path,
        dirs_exist_ok=True,
    )

    return tmp_path


@pytest.fixture
def default_nodal_inj_start_options(static_information: StaticInformation):
    """Create default nodal injection start options for tests.

    This provides a simple starting point for tests that need NodalInjStartOptions
    but don't care about the specific optimization configuration.
    """
    from toop_engine_dc_solver.jax.types import NodalInjOptimResults, NodalInjStartOptions

    n_pst = static_information.dynamic_information.n_controllable_pst
    n_timesteps = static_information.dynamic_information.nodal_injections.shape[0]
    batch_size = 1

    return NodalInjStartOptions(
        previous_results=NodalInjOptimResults(pst_taps=jnp.zeros((batch_size, n_timesteps, n_pst), dtype=jnp.float32)),
        precision_percent=jnp.array(1.0),
    )


def create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder(folder: Path) -> None:
    """Create a preprocessed folder for create_complex_grid_battery_hvdc_svc_3w_trafo().

    Runs the importer and preprocessing.

    Parameter:
    folder: Path
        The root folder where the data is saved to.
    """
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK)

    output_path_grid = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    output_path_grid.parent.mkdir(parents=True, exist_ok=True)
    net.save(output_path_grid)
    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    importer_parameters = CgmesImporterParameters(
        grid_model_file=output_path_grid,
        data_folder=folder,
        area_settings=AreaSettings(
            cutoff_voltage=1,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
            cross_border_limits_n0=None,
            cross_border_limits_n1=None,
        ),
    )
    preprocessing_parameters = PreprocessParameters(action_set_clip=2**4, enable_bb_outage=False, bb_outage_as_nminus1=False)

    _import_result = preprocessing.convert_file(importer_parameters=importer_parameters)
    filesystem_dir = DirFileSystem(str(folder))
    _info, _static_information, _ = load_grid(
        data_folder_dirfs=filesystem_dir,
        pandapower=False,
        status_update_fn=None,
        parameters=preprocessing_parameters,
    )


def create_ucte_data_folder(folder: Path, ucte_file: Path) -> None:
    """Create a preprocessed folder for an ucte file.

    Runs the importer and preprocessing.

    Parameter:
    folder: Path
        The root folder where the data is saved to.
    ucte_file: Path
        The path to the UCTE file to load.
    """
    net = pypowsybl.network.load(ucte_file)
    pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK)

    output_path_grid = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    output_path_grid.parent.mkdir(parents=True, exist_ok=True)
    net.save(output_path_grid)
    output_path_masks = folder / PREPROCESSING_PATHS["masks_path"]
    output_path_masks.mkdir(parents=True, exist_ok=True)

    importer_parameters = UcteImporterParameters(
        grid_model_file=output_path_grid,
        data_folder=folder,
        area_settings=AreaSettings(
            cutoff_voltage=1,
            control_area=[""],
            view_area=[""],
            nminus1_area=[""],
            cross_border_limits_n0=None,
            cross_border_limits_n1=None,
        ),
    )
    preprocessing_parameters = PreprocessParameters(action_set_clip=2**4, enable_bb_outage=False, bb_outage_as_nminus1=False)

    _import_result = preprocessing.convert_file(importer_parameters=importer_parameters)

    filesystem_dir = DirFileSystem(str(folder))
    _info, _static_information, _ = load_grid(
        data_folder_dirfs=filesystem_dir,
        pandapower=False,
        status_update_fn=None,
        parameters=preprocessing_parameters,
    )
