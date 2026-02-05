# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import logbook
import pandapower as pp
import pandas as pd
import pytest
from toop_engine_grid_helpers.pandapower.example_grids import pandapower_extended_oberrhein
from toop_engine_grid_helpers.pandapower.pandapower_import_helpers import (
    create_virtual_slack,
    drop_elements_connected_to_one_bus,
    drop_unsupplied_buses,
    fuse_closed_switches_fast,
    remove_out_of_service,
    replace_zero_branches,
    select_connected_subnet,
)


def test_fuse_closed_switches_fast() -> None:
    net = pp.networks.example_multivoltage()
    pp.rundcpp(net)
    res_before = net.res_line.copy()
    relevant_switches = net.switch[net.switch.et == "b"]
    len_switches = len(net.switch) - net.switch[(net.switch.et == "b")].shape[0]
    assert len(relevant_switches) > 0, "No closed switches found in the network -> test is useless"
    fuse_closed_switches_fast(net)
    relevant_switches = net.switch[net.switch.et == "b"]
    assert relevant_switches.closed.sum() == 0
    pp.rundcpp(net)
    res_after = net.res_line
    assert res_before.equals(res_after)
    assert len_switches == net.switch.shape[0]

    # test with open switches
    net = pp.networks.example_multivoltage()
    net.switch.loc[14, "closed"] = False
    pp.rundcpp(net)
    res_before = net.res_line.copy()
    relevant_switches = net.switch[net.switch.et == "b"]
    len_switches = len(net.switch) - net.switch[(net.switch.et == "b") & net.switch.closed].shape[0]
    assert len(relevant_switches) > 0, "No closed switches found in the network -> test is useless"
    fuse_closed_switches_fast(net)
    relevant_switches = net.switch[net.switch.et == "b"]
    assert relevant_switches.closed.sum() == 0
    pp.rundcpp(net)
    res_after = net.res_line
    assert res_before.equals(res_after)
    assert len_switches == net.switch.shape[0]

    # test with specific switch_ids
    net = pp.networks.example_multivoltage()
    fuse_closed_switches_fast(net, switch_ids=[0, 1, 2, 3])
    assert not net.switch.index.isin([0, 1, 2]).all(), "Closed switches have not been removed"
    assert 3 in net.switch.index, "Open switch has been removed, but should not have been"


def test_select_connected_subnet() -> None:
    loaded_net = pp.networks.mv_oberrhein()
    net = select_connected_subnet(loaded_net)
    assert len(net.ext_grid) + net.gen.slack.sum() >= 1
    assert len(net.bus) > 0
    assert len(net.bus) <= len(loaded_net.bus)
    net = pp.networks.case9()
    net.ext_grid
    pp.drop_elements(net, element_index=0, element_type="ext_grid")

    with pytest.raises(ValueError):
        select_connected_subnet(net)


def test_replace_zero_branches() -> None:
    loaded_net = pp.networks.mv_oberrhein()
    pp.rundcpp(loaded_net)
    line_res = deepcopy(loaded_net.res_line)
    replace_zero_branches(loaded_net)
    pp.rundcpp(loaded_net)
    line_res_merged = pd.merge(
        left=loaded_net.res_line,
        right=line_res,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_new", "_old"),
    )

    assert (line_res_merged["p_from_mw_old"] - line_res_merged["p_from_mw_new"]).abs().max() < 1


def test_drop_unsupplied_buses() -> None:
    loaded_net_pp_small = pp.networks.example_multivoltage()
    loaded_net_pp_small.bus.loc[5, "in_service"] = False
    unsupplied_buses = pp.topology.unsupplied_buses(loaded_net_pp_small)
    drop_unsupplied_buses(loaded_net_pp_small)
    assert len(unsupplied_buses) > 0
    assert len(pp.topology.unsupplied_buses(loaded_net_pp_small)) == 0
    assert not loaded_net_pp_small.bus.loc[5, "in_service"]
    assert all(bus not in loaded_net_pp_small.bus.index for bus in unsupplied_buses)


def test_create_virtual_slack() -> None:
    loaded_net_pp_small = pp.networks.example_multivoltage()
    net = deepcopy(loaded_net_pp_small)
    create_virtual_slack(net)
    for key in net.keys():
        if isinstance(net[key], pd.DataFrame):
            assert key in loaded_net_pp_small
            assert loaded_net_pp_small[key].equals(net[key])

    gen_slack = net.gen.iloc[0].to_dict()
    gen_slack["slack"] = True
    pp.drop_elements(net=net, element_type="gen", element_index=0)
    pp.create_gen(net, **gen_slack)
    pp.create_gen(net, **gen_slack)

    create_virtual_slack(net)
    assert "virtual_slack" in net.bus.name.values
    assert net.gen.slack.sum() == 0


def test_remove_out_of_service() -> None:
    net = pandapower_extended_oberrhein()
    if "in_service" not in net.switch.columns:
        net.switch["in_service"] = True
    remove_out_of_service(net)
    assert net.switch.in_service.sum() == len(net.switch)
    assert net.line.in_service.sum() == len(net.line)
    assert net.trafo.in_service.sum() == len(net.trafo)
    assert net.bus.in_service.sum() == len(net.bus)
    assert net.gen.in_service.sum() == len(net.gen)
    assert net.load.in_service.sum() == len(net.load)
    assert net.shunt.in_service.sum() == len(net.shunt)


def test_drop_elements_connected_to_one_bus():
    net = pp.networks.example_multivoltage()
    net.switch.loc[0, "bus"] = 0
    net.switch.loc[0, "element"] = 0
    net.line.loc[0, "from_bus"] = 0
    net.line.loc[0, "to_bus"] = 0
    drop_elements_connected_to_one_bus(net)
    assert 0 not in net.switch.index.values
    assert 0 not in net.line.index.values
    net.trafo.loc[0, "hv_bus"] = 0
    net.trafo.loc[0, "lv_bus"] = 0
    with pytest.raises(AssertionError) as e_info:
        drop_elements_connected_to_one_bus(net)
    assert "Two winding transformer with same hv and lv bus found in" in e_info.value.args[0]
    net.trafo.loc[0, "hv_bus"] = 1

    net.trafo3w.loc[0, "hv_bus"] = 1
    net.trafo3w.loc[0, "mv_bus"] = 1
    net.trafo3w.loc[0, "lv_bus"] = 1
    with pytest.raises(AssertionError) as e_info:
        drop_elements_connected_to_one_bus(net)
    assert "Three winding transformer with same hv == lv or hv == mv bus found in" in e_info.value.args[0]

    net.trafo3w.loc[0, "hv_bus"] = 0
    with logbook.handlers.TestHandler() as caplog:
        drop_elements_connected_to_one_bus(net)
        assert "Three winding transformer with same mv and lv bus found in" in "".join(caplog.formatted_records)


def test_drop_elements_connected_to_one_bus_attr_not_exist():
    net = pp.networks.example_multivoltage()
    with pytest.raises(AttributeError):
        drop_elements_connected_to_one_bus(net, branch_types=["NOT_EXISTING"])
