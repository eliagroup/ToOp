from copy import deepcopy

from toop_engine_importer.pypowsybl_import.cgmes.powsybl_masks_cgmes import get_switchable_buses_cgmes


def test_get_switchable_buses_cgmes(basic_node_breaker_network_powsybl):
    net = deepcopy(basic_node_breaker_network_powsybl)
    res = get_switchable_buses_cgmes(net, area_codes=["BE"])
    assert res == ["VL1_0", "VL2_0", "VL3_0"]
    net.update_switches(id="VL2_BREAKER", open=True)
    net.update_switches(id="VL2_BREAKER#0", open=True)
    res = get_switchable_buses_cgmes(net, area_codes=["BE"])
    assert res == ["VL1_0", "VL3_0"]
