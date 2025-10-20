import pypowsybl
import pytest
from pydantic_core._pydantic_core import ValidationError
from toop_engine_importer.pypowsybl_import.merge_ucte_cgmes.merge_ucte_cgmes import (
    DanglingLineSchema,
    TieLineSchema,
    UcteCgmesMerge,
    get_ucte_border_tie_lines,
    remove_area_from_ucte,
    remove_station,
)


def test_remove_station_node_breaker(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    for station in net.get_substations().index:
        remove_station(net, station)

    assert len(net.get_substations()) == 0


def test_schmea(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2

    TieLineSchema.validate(net.get_tie_lines())
    DanglingLineSchema.validate(net.get_dangling_lines())


def test_ucte_cgmes_merge_class(basic_node_breaker_network_powsyblV2):
    net = basic_node_breaker_network_powsyblV2
    UcteCgmesMerge(
        ucte_area_name="test",
        country_name=None,
        ucte_border_lines=net.get_tie_lines(),
        cgmes_dangling_lines=net.get_dangling_lines(),
        removed_tie_lines=[],
        removed_dangling_lines=[],
    )

    UcteCgmesMerge(
        ucte_area_name=None,
        country_name="test",
        ucte_border_lines=net.get_tie_lines(),
        cgmes_dangling_lines=net.get_dangling_lines(),
        removed_tie_lines=[],
        removed_dangling_lines=[],
    )
    with pytest.raises(ValidationError):
        UcteCgmesMerge(
            ucte_area_name=None,
            country_name=None,
            ucte_border_lines=net.get_tie_lines(),
            cgmes_dangling_lines=net.get_dangling_lines(),
            removed_tie_lines=[],
            removed_dangling_lines=[],
        )


def test_get_ucte_border_tie_lines(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    tie_lines = get_ucte_border_tie_lines(net_ucte=network, net_cgmes=network)
    assert len(tie_lines) == 2
    list(tie_lines.index) == ["XB__F_21 B_SU1_21 1 + XB__F_21 D8SU1_21 1", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"]


def testremove_area_from_ucte(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    vl = network.get_voltage_levels()
    assert "D8SU1_1" in vl.index
    assert "D8SU1_2" in vl.index

    merge_info = UcteCgmesMerge(
        ucte_area_name="D8",
        country_name=None,
        ucte_border_lines=network.get_tie_lines(),
        cgmes_dangling_lines=network.get_dangling_lines(),
        removed_tie_lines=[],
        removed_dangling_lines=[],
    )
    remove_area_from_ucte(net_ucte=network, ucte_cgmes_merge_info=merge_info)
    vl = network.get_voltage_levels()
    assert "D8SU1_1" not in vl.index
    assert "D8SU1_2" not in vl.index
