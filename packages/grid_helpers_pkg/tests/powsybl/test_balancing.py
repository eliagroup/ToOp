# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import pandas as pd
import pypowsybl
import pytest
from beartype.typing import Literal
from toop_engine_grid_helpers.powsybl.balancing import balance_grid
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    get_cgmes_distributed_slack_parameters,
    get_single_slack_parameters,
)

INJECTION_COLUMNS = ["p", "q", "target_p", "target_q"]


@pytest.fixture
def complex_node_breaker_net() -> pypowsybl.network.Network:
    return create_complex_grid_battery_hvdc_svc_3w_trafo()


def _run(net: pypowsybl.network.Network, method: Literal["ac", "dc"], lf_params: pypowsybl.loadflow.Parameters) -> list:
    run = pypowsybl.loadflow.run_ac if method == "ac" else pypowsybl.loadflow.run_dc
    return run(net, lf_params)


def _injections(net: pypowsybl.network.Network) -> pd.DataFrame:
    return pd.concat([net.get_generators(attributes=INJECTION_COLUMNS), net.get_batteries(attributes=INJECTION_COLUMNS)])


@pytest.mark.parametrize("method", ["ac", "dc"])
def test_balanced_grid_keeps_the_flows_of_the_distributed_slack_without_one(
    complex_node_breaker_net: pypowsybl.network.Network, method: Literal["ac", "dc"]
) -> None:
    """After balancing, a loadflow without a distributed slack gives the flows of the distributed slack."""
    net = complex_node_breaker_net
    _run(net, method, get_cgmes_distributed_slack_parameters())
    distributed = net.get_branches(attributes=["p1", "i1"])
    injections = _injections(net)
    assert not (injections["target_p"] == -injections["p"]).all(), "the slack has to be distributed to test anything"

    assert balance_grid(net, method, get_cgmes_distributed_slack_parameters())

    balanced = _injections(net)
    has_result = balanced["p"].notna()
    assert balanced.loc[has_result, "target_p"].tolist() == pytest.approx((-balanced.loc[has_result, "p"]).tolist())
    assert balanced.loc[~has_result, "target_p"].tolist() == injections.loc[~has_result, "target_p"].tolist(), (
        "injections without a result keep their targets"
    )
    if method == "ac":
        has_q = balanced["q"].notna()
        assert balanced.loc[has_q, "target_q"].tolist() == pytest.approx((-balanced.loc[has_q, "q"]).tolist())
    else:
        assert balanced["target_q"].tolist() == pytest.approx(injections["target_q"].tolist(), nan_ok=True), (
            "DC has no reactive power to balance"
        )

    no_distributed_slack = deepcopy(get_cgmes_distributed_slack_parameters())
    no_distributed_slack.distributed_slack = False
    result = _run(net, method, no_distributed_slack)
    assert abs(result[0].slack_bus_results[0].active_power_mismatch) < 1.0, "the slack bus has (almost) nothing left"
    flows = net.get_branches(attributes=["p1", "i1"])
    connected = distributed["p1"].notna()
    assert flows.loc[connected, "p1"].tolist() == pytest.approx(distributed.loc[connected, "p1"].tolist(), abs=0.5)


def test_ac_and_dc_balance_differently(complex_node_breaker_net: pypowsybl.network.Network) -> None:
    """An AC balance covers the losses as well, so the generators produce more than after a DC balance."""
    dc_net = deepcopy(complex_node_breaker_net)
    assert balance_grid(complex_node_breaker_net, "ac", get_cgmes_distributed_slack_parameters())
    assert balance_grid(dc_net, "dc", get_cgmes_distributed_slack_parameters())
    ac_generation = _injections(complex_node_breaker_net)["target_p"].sum()
    dc_generation = _injections(dc_net)["target_p"].sum()
    assert ac_generation > dc_generation + 1.0, "the AC losses are generated as well"


def test_without_distributed_slack_the_grid_is_left_alone(complex_node_breaker_net: pypowsybl.network.Network) -> None:
    before = _injections(complex_node_breaker_net)
    assert not balance_grid(complex_node_breaker_net, "ac", get_single_slack_parameters())
    pd.testing.assert_frame_equal(_injections(complex_node_breaker_net), before)


def test_a_failed_loadflow_leaves_the_targets_alone(complex_node_breaker_net: pypowsybl.network.Network) -> None:
    """A loadflow that does not converge still writes intermediate flows, which must not become the targets."""
    targets_before = _injections(complex_node_breaker_net)[["target_p", "target_q"]]
    one_iteration = deepcopy(get_cgmes_distributed_slack_parameters())
    one_iteration.provider_parameters = {**one_iteration.provider_parameters, "maxNewtonRaphsonIterations": "1"}

    assert not balance_grid(complex_node_breaker_net, "ac", one_iteration)

    injections = _injections(complex_node_breaker_net)
    assert injections["p"].notna().any(), "powsybl wrote the intermediate flows"
    pd.testing.assert_frame_equal(injections[["target_p", "target_q"]], targets_before)
