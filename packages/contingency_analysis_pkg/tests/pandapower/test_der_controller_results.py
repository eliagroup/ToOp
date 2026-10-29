# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Q(U) controller results produced by the real DERController of the MCCS pandapower fork.

Stock pandapower records no Q(U) controller rows, so the module is skipped there. The grid is one
plant (a 100 MVA sgen behind a 220/30 kV transformer) connected to the external grid over two
parallel lines; each contingency trips one line. The grid holds 1.07 p.u. (235 kV), above the
droop's 216..232 kV deadband, so the controller absorbs Q in every outage.
"""

import importlib
from copy import deepcopy

import pandapower as pp
import pandas as pd
import pandera as pa
import polars as pl
import pytest
from toop_engine_contingency_analysis.pandapower import outage_power_flow
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pandapower.controller_data import read_der_controller_data
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
from toop_engine_interfaces.loadflow_results import ConvergenceStatus, LoadflowResults
from toop_engine_interfaces.loadflow_results_polars import DerControllerResultSchemaPolars, LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
)

# Looked up through the modules, not imported by name: stock pandapower has no QModelTwoSlopeDroop, and a
# failing import would stop collection instead of skipping the module.
_der_package = importlib.import_module("pandapower.control.controller.DERController")
_der_module = importlib.import_module("pandapower.control.controller.DERController.der_control")
pytestmark = pytest.mark.skipif(
    not hasattr(_der_module, "der_controller_data"),
    reason="needs the MCCS pandapower fork, whose DERController records Q(U) controller rows",
)

BASECASE = "BASECASE"
POC_BUS = 1
PLANT_SGEN = 0


def _net(record_data: bool = True) -> pp.pandapowerNet:
    """One plant with a Q(U) droop controller measuring at its point of connection.

    Returns
    -------
    pp.pandapowerNet
        The net with one DERController on sgen 0.
    """
    net = pp.create_empty_network()
    grid = pp.create_bus(net, vn_kv=220.0)
    poc = pp.create_bus(net, vn_kv=220.0)
    mv = pp.create_bus(net, vn_kv=30.0)
    pp.create_ext_grid(net, grid, vm_pu=1.07)
    for _ in range(2):
        pp.create_line_from_parameters(
            net, grid, poc, length_km=60.0, r_ohm_per_km=0.03, x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=1.0
        )
    pp.create_transformer_from_parameters(
        net,
        poc,
        mv,
        sn_mva=200.0,
        vn_hv_kv=220.0,
        vn_lv_kv=30.0,
        vkr_percent=0.3,
        vk_percent=12.0,
        pfe_kw=0.0,
        i0_percent=0.0,
    )
    pp.create_sgen(net, mv, p_mw=80.0, q_mvar=0.0, sn_mva=100.0)
    # Slack allocation reads this CGMES column.
    net.gen["referencePriority"] = pd.Series(dtype=float)
    net.sgen["referencePriority"] = float("nan")
    q_model = _der_package.QModelTwoSlopeDroop(
        vm_min_kv=216.0,
        vm_max_kv=232.0,
        droop_capacitive_u_kv_per_q_pu=7.1,
        droop_inductive_u_kv_per_q_pu=3.6,
        q_set_pu=0.0,
        q_min_pu=-1.44,
        q_max_pu=1.155,
    )
    _der_package.DERController(net, [PLANT_SGEN], q_model=q_model, voltage_bus_index=poc, record_data=record_data)
    return net


def _nminus1_definition(net: pp.pandapowerNet) -> Nminus1Definition:
    """The basecase and one contingency per line, all buses monitored.

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


def _run(net: pp.pandapowerNet, runpp_kwargs: dict, polars: bool = True) -> LoadflowResultsPolars | LoadflowResults:
    nminus1_definition = _nminus1_definition(net)
    cfg = ContingencyAnalysisConfig(method="ac", polars=polars, runpp_kwargs=runpp_kwargs)
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
    """Every outage reports its own steps and one final row, valid against the schema.

    One final row per contingency also shows the base case's rows were not carried into the outages.
    """
    net = _net()

    results = _run(net, runpp_kwargs={"run_control": True})

    rows = DerControllerResultSchemaPolars.validate(results.der_controller_results).collect()
    outage_rows = rows.filter(pl.col("contingency") != BASECASE)
    assert set(outage_rows["contingency"].to_list()) == {str(index) for index in net.line.index}
    assert set(outage_rows["timestep"].to_list()) == {3}
    final_rows = outage_rows.filter(pl.col("row_type") == "final")
    assert final_rows.group_by("contingency").len()["len"].to_list() == [1] * final_rows["contingency"].n_unique()
    assert set(rows["measurement_bus"].to_list()) == {POC_BUS}
    assert set(rows["element_index"].to_list()) == {PLANT_SGEN}
    # Above the deadband the plant absorbs Q.
    assert (final_rows["q_mvar"] < 0).all()
    assert "contingency_name" not in rows.columns
    # No element is held, so held_reason is None in every row; it still has to be stored as String
    assert rows.schema["held_reason"] == pl.String


def test_der_controller_results_survive_the_pandas_conversion() -> None:
    """polars=False returns the rows indexed by timestep and contingency, validated by DerControllerResultSchema."""
    net = _net()

    results = _run(net, runpp_kwargs={"run_control": True}, polars=False)

    assert isinstance(results.der_controller_results, pd.DataFrame)
    assert results.der_controller_results.index.names == ["timestep", "contingency"]
    contingencies = set(results.der_controller_results.index.get_level_values("contingency"))
    assert contingencies == {str(index) for index in net.line.index} | {BASECASE}


def test_controllers_without_record_data_give_no_der_controller_results() -> None:
    """DERControllers record only with record_data=True; otherwise the field stays None."""
    net = _net(record_data=False)

    results = _run(net, runpp_kwargs={"run_control": True})

    assert results.der_controller_results is None


def test_outages_whose_controller_does_not_converge_are_failed_and_keep_their_steps() -> None:
    """ControllerNotConverged marks the outage failed, and the steps the controller took are still reported."""
    net = _net()

    results = _run(net, runpp_kwargs={"run_control": True, "max_iter": 1})

    converged = results.converged.collect().filter(pl.col("contingency") != BASECASE)
    assert set(converged["status"].to_list()) == {ConvergenceStatus.FAILED.value}
    rows = results.der_controller_results.collect().filter(pl.col("contingency") != BASECASE)
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
            # Undo the setpoints the attempt wrote, so the retry has to take the same steps again.
            net.sgen["q_mvar"] = 0.0
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
    rows = read_der_controller_data(net)
    assert (rows["row_type"] == "final").sum() == 1
    step_numbers = rows.loc[rows["row_type"] == "step", "control_step"].to_list()
    assert step_numbers == list(range(1, len(step_numbers) + 1)), "one step sequence, not two"
