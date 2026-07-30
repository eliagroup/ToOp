# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pandapower
import pypowsybl
import pytest
from pandapower import pp_dir
from pandapower.converter import to_mpc
from toop_engine_grid_helpers.powsybl.example_grids import (
    case14_matching_asset_topo_powsybl,
)
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS

TESTS_ROOT = Path(__file__).parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from powsybl.network_graph.network_graph_test_helper import (
    SecurityAnalysisTestContext,
    _build_security_analysis_test_context,
)


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
def case14_data_with_asset_topo(case14_data_with_asset_topo_path: Path) -> tuple[Path, Topology]:
    """Fixture to create a temporary folder for the case14 test."""
    with open(case14_data_with_asset_topo_path / PREPROCESSING_PATHS["asset_topology_file_path"], "r") as f:
        asset_topology = Topology.model_validate_json(f.read())
    return case14_data_with_asset_topo_path, asset_topology


@pytest.fixture(scope="module")
def _security_analysis_test_context_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, SecurityAnalysisTestContext]:
    """Build the default DC security-analysis context once per module and persist the network for reloading."""
    net, context = _build_security_analysis_test_context(add_3_windings_transformer_outage=True, run_ac=False)
    tmp_path = tmp_path_factory.mktemp("security_analysis_test_context")
    net_path = tmp_path / "security_analysis_test_context.xiidm"
    net.save(net_path)
    return net_path, context


@pytest.fixture(scope="function")
def security_analysis_test_context(
    _security_analysis_test_context_template: tuple[Path, SecurityAnalysisTestContext],
) -> tuple[pypowsybl.network.Network, SecurityAnalysisTestContext]:
    """Reload the cached default DC network for each test while reusing the prepared context data."""
    net_path, context = _security_analysis_test_context_template
    return pypowsybl.network.load(net_path), context.model_copy(deep=False)


@pytest.fixture(scope="function")
def security_analysis_test_context_factory() -> Callable[..., tuple[pypowsybl.network.Network, SecurityAnalysisTestContext]]:
    """Provide an on-demand builder for non-default security-analysis test contexts."""

    def _factory(
        add_3_windings_transformer_outage: bool = True, run_ac: bool = False
    ) -> tuple[pypowsybl.network.Network, SecurityAnalysisTestContext]:
        return _build_security_analysis_test_context(
            add_3_windings_transformer_outage=add_3_windings_transformer_outage,
            run_ac=run_ac,
        )

    return _factory
