# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower import (
    PandapowerContingency,
    PandapowerElements,
    PandapowerMonitoredElementSchema,
    get_va_diff_results,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import VADiffInfo
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model


def create_test_net_for_va_diff_with_trafo():
    net = pp.create_empty_network()

    # --- Base network ---
    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    pp.create_gen(net, bus=b1, p_mw=0.0, vm_pu=1.0, slack=True, name="slack_gen")

    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1, std_type="NAYY 4x50 SE", name="line_1")

    # HV buses
    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")
    pp.create_switch(net, bus=b2, element=b3, et="b", closed=False, type="CB", name="switch_1")

    b4 = pp.create_bus(net, vn_kv=110, name="bus_4")
    pp.create_switch(net, bus=b2, element=b4, et="b", closed=True, type="CB", name="switch_2")

    # --- Trafo3w #1 ---
    lv1 = pp.create_bus(net, vn_kv=10, name="lv_bus_1")
    mv1 = pp.create_bus(net, vn_kv=20, name="mv_bus_1")
    pp.create_transformer3w(net, hv_bus=b3, mv_bus=mv1, lv_bus=lv1, std_type="63/25/38 MVA 110/20/10 kV", name="trafo_3w_1")
    b5 = pp.create_bus(net, vn_kv=10, name="bus_5")
    b6 = pp.create_bus(net, vn_kv=20, name="bus_6")
    pp.create_switch(net, bus=lv1, element=b5, et="b", closed=False, type="CB", name="switch_3")
    pp.create_switch(net, bus=mv1, element=b6, et="b", closed=False, type="CB", name="switch_4")

    pp.create_sgen(net, bus=b5, p_mw=5.0, q_mvar=0.0, name="sgen_1")
    pp.create_sgen(net, bus=b6, p_mw=5.0, q_mvar=0.0, name="sgen_2")
    # --- Trafo3w #2 ---
    lv2 = pp.create_bus(net, vn_kv=10, name="lv_bus_2")
    mv2 = pp.create_bus(net, vn_kv=20, name="mv_bus_2")
    pp.create_transformer3w(net, hv_bus=b4, mv_bus=mv2, lv_bus=lv2, std_type="63/25/38 MVA 110/20/10 kV", name="trafo_3w_2")

    b7 = pp.create_bus(net, vn_kv=10, name="bus_5")
    b8 = pp.create_bus(net, vn_kv=20, name="bus_6")
    pp.create_switch(net, bus=lv2, element=b7, et="b", closed=True, type="CB", name="switch_5")
    pp.create_switch(net, bus=mv2, element=b8, et="b", closed=True, type="CB", name="switch_6")

    pp.create_sgen(net, bus=b7, p_mw=5.0, q_mvar=0.0, name="sgen_1")
    pp.create_sgen(net, bus=b8, p_mw=5.0, q_mvar=0.0, name="sgen_2")

    pp.create_switch(net, bus=b5, element=b7, et="b", closed=True, type="CB", name="switch_7")
    pp.create_switch(net, bus=b6, element=b8, et="b", closed=True, type="CB", name="switch_8")
    return net


def create_test_net_for_va_diff_with_multiple_els():
    net = pp.create_empty_network()

    # --- Base network ---
    b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
    pp.create_gen(net, bus=b1, p_mw=0.0, vm_pu=1.0, slack=True, name="slack_gen")

    b2 = pp.create_bus(net, vn_kv=110, name="bus_2")
    pp.create_switch(net, bus=b1, element=b2, et="b", closed=False, type="CB", name="switch_1")

    b3 = pp.create_bus(net, vn_kv=110, name="bus_3")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1, std_type="NAYY 4x50 SE", name="line_1")

    b4 = pp.create_bus(net, vn_kv=110, name="bus_4")

    pp.create_impedance(net, from_bus=b3, to_bus=b4, rft_pu=0.01, xft_pu=0.05, sn_mva=100, name="impedance_b3_b4")

    b5 = pp.create_bus(net, vn_kv=110, name="bus_5")

    pp.create_switch(net, bus=b4, element=b5, et="b", closed=False, type="CB", name="switch_2")

    b6 = pp.create_bus(net, vn_kv=110, name="bus_6")
    pp.create_line(net, from_bus=b3, to_bus=b6, length_km=1, std_type="NAYY 4x50 SE", name="line_1")
    b7 = pp.create_bus(net, vn_kv=110, name="bus_7")
    pp.create_switch(net, bus=b6, element=b7, et="b", closed=False, type="CB", name="switch_3")

    b8 = pp.create_bus(net, vn_kv=110, name="bus_8")

    pp.create_transformer(net, hv_bus=b3, lv_bus=b8, std_type="25 MVA 110/10 kV", name="trafo_b3_b8")

    b9 = pp.create_bus(net, vn_kv=110, name="bus_9")
    pp.create_switch(net, bus=b8, element=b9, et="b", closed=False, type="CB", name="switch_8")
    return net


def test_va_diff_out_group_trafo():
    net = create_test_net_for_va_diff_with_trafo()
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    for row in net.switch.itertuples():
        monitored_elements.loc[get_globally_unique_id(int(row.Index), "switch"), ["table", "table_id", "kind", "name"]] = (
            "switch",
            row.Index,
            "switch",
            row.name,
        )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    pp.runpp(net)

    # mock VA values at candidates switches
    net.res_bus.loc[net.bus.name == "bus_2", "va_degree"] = 2
    net.res_bus.loc[net.bus.name == "bus_5", "va_degree"] = 4
    net.res_bus.loc[net.bus.name == "bus_6", "va_degree"] = 6

    va_diff_df = get_va_diff_results(net, timestep, monitored_elements, contingency)
    va_diff_df.reset_index(inplace=True)
    # we have 3 open switches
    # Result for switch_1 with index 0 should be max(va difference between bus_2 and bus_5 and bus_6) =
    # = max(abs(bus_2_va - bus_5_va), abs(bus_2_va - bus_6_va)) = max(abs(2-4), abs(2-6)) = 4
    assert va_diff_df.loc[va_diff_df.element == "0%%switch"].va_diff.item() == 4

    # Result for switch_2 with index 2 should be max(va difference between bus_2 and bus_5) =
    # = max(abs(bus_2_va - bus_5_va)) = max(abs(2-4)) = 2
    assert va_diff_df.loc[va_diff_df.element == "2%%switch"].va_diff.item() == 2

    # Result for switch_3 with index 3 should be max(va difference between bus_2 and bus_6) =
    # = max(abs(bus_2_va - bus_6_va)) = max(abs(2-6)) = 4
    assert va_diff_df.loc[va_diff_df.element == "3%%switch"].va_diff.item() == 4


def test_va_diff_out_group_multiple_els():
    net = create_test_net_for_va_diff_with_multiple_els()
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    for row in net.switch.itertuples():
        monitored_elements.loc[get_globally_unique_id(int(row.Index), "switch"), ["table", "table_id", "kind", "name"]] = (
            "switch",
            row.Index,
            "switch",
            row.name,
        )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    pp.runpp(net)

    # mock VA values at candidates switches
    net.res_bus.loc[net.bus.name == "bus_1", "va_degree"] = 2
    net.res_bus.loc[net.bus.name == "bus_5", "va_degree"] = 5
    net.res_bus.loc[net.bus.name == "bus_7", "va_degree"] = 10
    net.res_bus.loc[net.bus.name == "bus_9", "va_degree"] = 14

    va_diff_df = get_va_diff_results(net, timestep, monitored_elements, contingency)
    va_diff_df.reset_index(inplace=True)
    assert va_diff_df.loc[va_diff_df.element == "0%%switch"].va_diff.item() == 12
    assert va_diff_df.loc[va_diff_df.element == "1%%switch"].va_diff.item() == 9
    assert va_diff_df.loc[va_diff_df.element == "2%%switch"].va_diff.item() == 8
    assert va_diff_df.loc[va_diff_df.element == "3%%switch"].va_diff.item() == 12


def test_get_va_diff_results(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    outaged_line_id = 1
    va_diff_info = VADiffInfo(
        from_bus=pandapower_net.line.loc[outaged_line_id, "from_bus"],
        to_bus=pandapower_net.line.loc[outaged_line_id, "to_bus"],
        power_switches_from={"PW_SWITCH_ID1": "PW_SWITCH_NAME1"},
        power_switches_to={"PW_SWITCH_ID2": "PW_SWITCH_NAME2"},
    )
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(
                unique_id=get_globally_unique_id(outaged_line_id, "line"), table_id=outaged_line_id, table="line", name=""
            )
        ],
        va_diff_info=[va_diff_info],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0", type="CB")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[outaged_line_id, "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + list(
        va_diff_info.power_switches_from.keys()
    ) + list(va_diff_info.power_switches_to.keys()), "Element IDs should match monitored elements + the outaged line"
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree,
        -1
        * (outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree),
    ], "VA differences should match the outage net"

    # Test what happens if there is only one switch
    contingency.va_diff_info[0].power_switches_to = {}
    contingency.va_diff_info[0].power_switches_from = {"PW_SWITCH_ID1": "PW_SWITCH_NAME1"}
    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + ["PW_SWITCH_ID1"], (
        "Element IDs should match monitored elements. No line switches since there arent any"
    )
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree,
    ], "VA differences should match the outage net"

    # Test what happens if there are no switches
    contingency.va_diff_info = []
    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist(), (
        "Element IDs should match monitored elements. No line switches since there arent any"
    )
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
    ], "VA differences should match the outage net"


def test_get_va_diff_results_multioutage(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    va_diff_info_1 = VADiffInfo(
        from_bus=pandapower_net.line.loc[0, "from_bus"],
        to_bus=pandapower_net.line.loc[0, "to_bus"],
        power_switches_from={"PW_SWITCH_ID1": "PW_SWITCH_NAME1"},
        power_switches_to={"PW_SWITCH_ID2": "PW_SWITCH_NAME2"},
    )
    va_diff_info_2 = VADiffInfo(
        from_bus=pandapower_net.line.loc[1, "from_bus"],
        to_bus=pandapower_net.line.loc[1, "to_bus"],
        power_switches_from={"PW_SWITCH_ID3": "PW_SWITCH_NAME3"},
        power_switches_to={"PW_SWITCH_ID4": "PW_SWITCH_NAME4"},
    )

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="line", name="", type="line")
            for id in lines.index[:2]
        ],
        va_diff_info=[va_diff_info_1, va_diff_info_2],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0", type="CB")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + [
        "PW_SWITCH_ID1",
        "PW_SWITCH_ID2",
        "PW_SWITCH_ID3",
        "PW_SWITCH_ID4",
    ], "Element IDs should match monitored elements + the outaged line"
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.iloc[0].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[0].to_bus].va_degree,
        -1
        * (
            outage_net.res_bus.loc[lines.iloc[0].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[0].to_bus].va_degree
        ),
        outage_net.res_bus.loc[lines.iloc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[1].to_bus].va_degree,
        -1
        * (
            outage_net.res_bus.loc[lines.iloc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[1].to_bus].va_degree
        ),
    ], "VA differences should match the outage net"


def test_get_va_diff_results_basecase(pandapower_net: pp.pandapowerNet):
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 0
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0", type="CB")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )

    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    pp.runpp(pandapower_net)

    va_diff_df = get_va_diff_results(pandapower_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert va_diff_df.va_diff.tolist() == [
        pandapower_net.res_bus.loc[0].va_degree - pandapower_net.res_bus.loc[1].va_degree
    ], "VA differences should match the outage net"


def test_get_va_diff_results_no_elements(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:1], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert va_diff_df.empty, "The result should be empty if no monitored elements are provided"


def test_get_va_diff_results_outage_element_trafo3w(pandapower_net: pp.pandapowerNet):
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "trafo3w"), table_id=id, table="trafo3w", name="")
            for id in range(2)
        ],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    va_diff_df = get_va_diff_results(pandapower_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert va_diff_df.va_diff.isna().all(), "For trafo3w outages, the VA differences should be NaN for now"
