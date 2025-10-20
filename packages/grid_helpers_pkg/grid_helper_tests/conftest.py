import os
from pathlib import Path

import pytest
from toop_engine_grid_helpers.powsybl.example_grids import case14_matching_asset_topo_powsybl
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


@pytest.fixture(scope="session")
def ucte_file() -> Path:
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_ucte_powsybl_example.uct")
    return ucte_file


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
