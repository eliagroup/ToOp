# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

from toop_engine_importer.pypowsybl_import.cgmes.powsybl_masks_cgmes import get_switchable_buses_cgmes


def test_get_switchable_buses_cgmes(basic_node_breaker_network_powsybl_network_graph):
    net = deepcopy(basic_node_breaker_network_powsybl_network_graph)
    res = get_switchable_buses_cgmes(net, area_codes=["BE"])
    expected = ["VL1_0", "VL2_0", "VL3_0"]
    assert res == expected

    select_by_voltage_level_id_list = ["VL1", "VL2", "VL3"]
    res = get_switchable_buses_cgmes(
        net, area_codes=["DE"], cutoff_voltage=1000, select_by_voltage_level_id_list=select_by_voltage_level_id_list
    )
    assert res == expected

    select_by_voltage_level_id_list = ["VL1", "VL2"]
    res = get_switchable_buses_cgmes(
        net, area_codes=["DE"], cutoff_voltage=1000, select_by_voltage_level_id_list=select_by_voltage_level_id_list
    )
    assert res == expected[:-1]

    net.update_switches(id="VL2_BREAKER", open=True)
    net.update_switches(id="VL2_BREAKER#0", open=True)
    res = get_switchable_buses_cgmes(net, area_codes=["BE"])
    assert res == ["VL1_0", "VL3_0"]
