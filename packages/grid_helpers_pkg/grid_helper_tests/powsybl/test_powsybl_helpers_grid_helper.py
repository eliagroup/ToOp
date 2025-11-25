import fsspec
import pypowsybl
from toop_engine_grid_helpers.powsybl.powsybl_helpers import (
    load_powsybl_from_fs,
)


def test_load_powsybl_from_fs_mat(ieee14_mat):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, ieee14_mat)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_uct(ucte_file):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, ucte_file)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_cgmes(eurostag_tutorial_example1_cgmes):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, eurostag_tutorial_example1_cgmes)
    assert isinstance(pp_net, pypowsybl.network.Network)


def test_load_powsybl_from_fs_xiidm(basic_node_breaker_grid_xiidm):
    file_system = fsspec.filesystem("file")

    pp_net = load_powsybl_from_fs(file_system, basic_node_breaker_grid_xiidm)
    assert isinstance(pp_net, pypowsybl.network.Network)
