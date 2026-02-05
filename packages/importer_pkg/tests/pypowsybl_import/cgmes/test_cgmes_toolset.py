# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_importer.pypowsybl_import.cgmes.cgmes_toolset import (
    get_busbar_sections_with_in_service,
    get_region_for_df,
    get_voltage_level_with_region,
)


def test_get_voltage_level_with_region(basic_node_breaker_network_powsybl_network_graph):
    net = basic_node_breaker_network_powsybl_network_graph
    res = get_voltage_level_with_region(net).columns
    assert len(res) == 6
    for col in ["name", "substation_id", "nominal_v", "high_voltage_limit", "low_voltage_limit", "region"]:
        assert col in res

    res = get_voltage_level_with_region(net, all_attributes=True).columns
    assert len(res) >= 8  # in case of new attributes added in pypowsybl
    for col in [
        "name",
        "substation_id",
        "nominal_v",
        "high_voltage_limit",
        "low_voltage_limit",
        "fictitious",
        "topology_kind",
        "region",
    ]:
        assert col in res

    attributes = ["name", "substation_id"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 3
    for col in attributes + ["region"]:
        assert col in res

    attributes = ["name", "substation_id", "region"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 3
    for col in attributes:
        assert col in res

    attributes = ["region"]
    res = get_voltage_level_with_region(net, attributes=attributes).columns
    assert len(res) == 1
    for col in attributes:
        assert col in res

    with pytest.raises(ValueError):
        get_voltage_level_with_region(net, attributes=attributes, all_attributes=True)


def test_get_region_for_df(basic_node_breaker_network_powsybl_network_graph):
    net = basic_node_breaker_network_powsybl_network_graph
    sw = net.get_switches()
    res = get_region_for_df(network=net, df=sw)
    assert "region" in res.columns
    assert res["region"].isna().sum() == 0
    br = net.get_branches()
    res = get_region_for_df(network=net, df=br)
    assert "region_1" in res.columns
    assert "region_2" in res.columns
    assert res["region_1"].isna().sum() == 0
    assert res["region_2"].isna().sum() == 0


def test_get_busbar_sections_with_in_service(basic_node_breaker_network_powsybl_network_graph):
    net = basic_node_breaker_network_powsybl_network_graph
    res = get_busbar_sections_with_in_service(net)
    assert "in_service" in res.columns, "in_service column is missing"
    assert [*list(net.get_busbar_sections().columns), "in_service"] == list(res.columns), (
        "net.get_busbar_sections().columns does not match res.columns"
    )

    res = get_busbar_sections_with_in_service(net, attributes=["name", "in_service"])
    assert list(res.columns) == ["name", "in_service"], "res.columns does not match expected columns"

    res = get_busbar_sections_with_in_service(net, attributes=["name"])
    assert list(res.columns) == ["name"], "res.columns does not match expected columns"

    res = get_busbar_sections_with_in_service(net, all_attributes=True)
    res_org_all_attributes = net.get_busbar_sections(all_attributes=True)
    assert [*list(res_org_all_attributes.columns), "in_service"] == list(res.columns), (
        "net.get_busbar_sections().columns does not match res.columns"
    )
    assert res[list(res_org_all_attributes.columns)].equals(res_org_all_attributes), (
        "res[res_org_all_attributes.columns] does not match res_org_all_attributes"
    )

    net.update_switches(id="VL2_BREAKER#0", open=True)
    res = get_busbar_sections_with_in_service(net)
    assert not res.loc["BBS2_3", "in_service"], "in_service column is not correct"
