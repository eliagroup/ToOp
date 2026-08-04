# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
import shutil
from pathlib import Path

import pandapower
import pypowsybl
import pytest
from fsspec.implementations.local import LocalFileSystem
from pandapower import pp_dir
from pandapower.converter import to_mpc
from toop_engine_grid_helpers.powsybl.example_grids import case14_matching_asset_topo_powsybl
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import materialize_runtime_bus_groups_from_network_state
from toop_engine_interfaces.asset_topology.asset_topology import MasterAssetTopology
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeBusGroup
from toop_engine_interfaces.filesystem_helper import load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


@pytest.fixture(scope="session")
def ucte_file() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_ucte_powsybl_example.uct")
    return ucte_file


@pytest.fixture(scope="session")
def ieee14_mat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("case14_mat")
    net = pandapower.networks.case14()
    pandapower.rundcpp(net)
    to_mpc(net, tmp_path / "case14_matpower.mat")
    return tmp_path / "case14_matpower.mat"


@pytest.fixture(scope="session")
def eurostag_tutorial_example1_cgmes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("eurostag_tutorial_example1_cgmes")
    net = pypowsybl.network.create_eurostag_tutorial_example1_network()
    net.save(tmp_path / "eurostag_tutorial_example1.zip", format="CGMES")
    return tmp_path / "eurostag_tutorial_example1.zip"


@pytest.fixture(scope="session")
def basic_node_breaker_grid_xiidm() -> Path:
    """Fixture get saved example_grid.basic_node_breaker_network_powsybl()"""
    return Path(__file__).parent / "files" / "basic_node_breaker.xiidm"


@pytest.fixture(scope="session")
def ieee14_json() -> Path:
    pp_dir_path = Path(pp_dir)
    ieee14_json_path = pp_dir_path / "networks" / "power_system_test_case_jsons" / "case14.json"
    return ieee14_json_path


@pytest.fixture(scope="session")
def _case14_data_with_asset_topo_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture to create a temporary folder for the case14 test."""
    tmp_path = tmp_path_factory.mktemp("case14")
    case14_matching_asset_topo_powsybl(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def case14_data_with_asset_topo_path(_case14_data_with_asset_topo_path: Path, tmp_path: Path) -> Path:
    """Per-test copy of case14 test directory."""
    shutil.copytree(_case14_data_with_asset_topo_path, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def case14_data_with_asset_topo(
    case14_data_with_asset_topo_path: Path,
) -> tuple[Path, MasterAssetTopology, list[RuntimeBusGroup]]:
    """Fixture to create a temporary folder for the case14 test."""
    master_data = load_pydantic_model_fs(
        filesystem=LocalFileSystem(),
        file_path=case14_data_with_asset_topo_path / PREPROCESSING_PATHS["asset_topology_master_data_file_path"],
        model_class=MasterAssetTopology,
    )
    network = pypowsybl.network.load(case14_data_with_asset_topo_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    stations = materialize_runtime_bus_groups_from_network_state(network=network, master_data=master_data)
    return case14_data_with_asset_topo_path, master_data, stations
