# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Controller results produced by the real DiscreteTapControl of the MCCS pandapower fork.

Stock pandapower records no controller rows, so the module is skipped there. The grid is one
110/20 kV transformer feeding a load over two parallel cables, with the tap controller regulating the
load bus; each contingency trips one cable.
"""

import importlib
from copy import deepcopy

import pandapower as pp
import pandas as pd
import pandera as pa
import polars as pl
import pytest
from pandapower.control import DiscreteTapControl
from toop_engine_contingency_analysis.pandapower import outage_power_flow
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pandapower.controller_data import read_controller_data
from toop_engine_contingency_analysis.pandapower.outage_power_flow import run_outage_power_flow
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    PandapowerElements,
    SingleOutageSppsContext,
    SppsActionsPandapowerSchema,
    SppsConditionsPandapowerSchema,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import ControllerResultSchemaPolars, LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
)

_tap_module = importlib.import_module("pandapower.control.controller.trafo.DiscreteTapControl")
pytestmark = pytest.mark.skipif(
    not hasattr(_tap_module, "ensure_controller_data"),
    reason="needs the MCCS pandapower fork, whose DiscreteTapControl records controller rows",
)

BASECASE = "BASECASE"
LOAD_BUS = 2
# The base case sits inside this band at tap 0 (0.954 p.u.); tripping a cable drops the load bus to
# 0.924 p.u., two tap steps below it.
VM_LOWER_PU = 0.95
VM_UPPER_PU = 0.97


def _net() -> pp.pandapowerNet:
    """110/20 kV transformer with a tap controller on the load bus, feeding the load over two cables.

    Returns
    -------
    pp.pandapowerNet
        The net with one DiscreteTapControl on trafo 0.
    """
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.0)
    lv = pp.create_bus(net, vn_kv=20.0)
    far = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, hv)
    pp.create_transformer(net, hv, lv, std_type="25 MVA 110/20 kV")
    for _ in range(2):
        pp.create_line(net, lv, far, length_km=10.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
    pp.create_load(net, far, p_mw=12.0, q_mvar=4.0)
    # Slack allocation reads this CGMES column; the grid has no (s)gens to carry it.
    net.gen["referencePriority"] = pd.Series(dtype=float)
    net.sgen["referencePriority"] = pd.Series(dtype=float)
    DiscreteTapControl(net, element_index=0, vm_lower_pu=VM_LOWER_PU, vm_upper_pu=VM_UPPER_PU, controlled_bus=far)
    return net


def _nminus1_definition(net: pp.pandapowerNet) -> Nminus1Definition:
    """The basecase and one contingency per cable, all buses monitored.

    Returns
    -------
    Nminus1Definition
        Monitored buses, the basecase and one contingency per line.
    """
    return Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=get_globally_unique_id(index, "bus"), name=str(index), kind="bus", type="bus")
            for index in net.bus.index
        ],
        contingencies=[Contingency(id=BASECASE, elements=[])]
        + [
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(index), kind="branch", type="line")
                ],
            )
            for index in net.line.index
        ],
    )


def _run(net: pp.pandapowerNet, runpp_kwargs: dict) -> LoadflowResultsPolars:
    nminus1_definition = _nminus1_definition(net)
    cfg = ContingencyAnalysisConfig(method="ac", polars=True, runpp_kwargs=runpp_kwargs)
    results = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=3,
        cfg=cfg,
    )
    return results


def _empty_spps() -> SingleOutageSppsContext:
    conditions = pa.typing.DataFrame[SppsConditionsPandapowerSchema](
        get_empty_dataframe_from_model(SppsConditionsPandapowerSchema)
    )
    actions = pa.typing.DataFrame[SppsActionsPandapowerSchema](get_empty_dataframe_from_model(SppsActionsPandapowerSchema))
    spps = SingleOutageSppsContext(conditions=conditions, actions=actions)
    return spps


def _line_outage(index: int) -> list[PandapowerElements]:
    unique_id = get_globally_unique_id(index, "line")
    elements = [PandapowerElements(unique_id=unique_id, table="line", table_id=index)]
    return elements


def test_converged_outages_return_step_and_final_rows() -> None:
    """A controller that settles reports its steps and one final row per contingency, valid against the schema."""
    net = _net()

    results = _run(net, runpp_kwargs={"run_control": True})

    rows = ControllerResultSchemaPolars.validate(results.controller_results).collect()
    outage_rows = rows.filter(pl.col("contingency") != BASECASE)
    assert set(outage_rows["contingency"].to_list()) == {str(index) for index in net.line.index}
    final_rows = outage_rows.filter(pl.col("row_type") == "final")
    assert final_rows.group_by("contingency").len()["len"].to_list() == [1] * final_rows["contingency"].n_unique()
    assert final_rows["in_band"].all()
    assert set(rows["controlled_bus"].to_list()) == {LOAD_BUS}


def test_outages_whose_controller_does_not_converge_are_failed_and_keep_their_steps() -> None:
    """ControllerNotConverged marks the outage failed instead of aborting the run, and its step rows survive.

    The outage needs two tap moves and ``max_iter`` allows one, so run_control raises before
    finalize_control - the only place the fork rebuilds ``net.controller_data``.
    """
    net = _net()

    results = _run(net, runpp_kwargs={"run_control": True, "max_iter": 1})

    converged = results.converged.collect().filter(pl.col("contingency") != BASECASE)
    assert set(converged["status"].to_list()) == {ConvergenceStatus.FAILED.value}
    rows = results.controller_results.collect().filter(pl.col("contingency") != BASECASE)
    assert set(rows["contingency"].to_list()) == {str(index) for index in net.line.index}
    assert set(rows["row_type"].to_list()) == {"step"}


def test_cold_start_retry_drops_the_rows_of_the_abandoned_warm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warm-start attempt records its steps, then fails; only the retry's rows remain."""
    net = _net()
    pp.runpp(net)
    real_run_power_flow = outage_power_flow._run_power_flow
    calls = []

    def warm_start_fails_after_control(**kwargs: object) -> None:
        real_run_power_flow(**kwargs)
        calls.append(kwargs)
        if len(calls) == 1:
            # Undo the taps the attempt moved, so the retry has to take the same steps again.
            net.trafo["tap_pos"] = 0
            raise pp.LoadflowNotConverged("warm start diverged")

    monkeypatch.setattr(outage_power_flow, "_run_power_flow", warm_start_fails_after_control)

    status, _ = run_outage_power_flow(
        net=net,
        spps=_empty_spps(),
        method="ac",
        outaged_elements=_line_outage(0),
        basecase_net=deepcopy(net),
        runpp_kwargs={"run_control": True},
    )

    assert status == ConvergenceStatus.CONVERGED
    assert len(calls) == 2
    rows = read_controller_data(net)
    assert (rows["row_type"] == "final").sum() == 1
    step_numbers = rows.loc[rows["row_type"] == "step", "control_step"].to_list()
    assert step_numbers == list(range(1, len(step_numbers) + 1)), "one step sequence, not two"
