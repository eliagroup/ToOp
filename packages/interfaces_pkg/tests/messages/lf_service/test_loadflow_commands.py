# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_interfaces.asset_topology import Strategy
from toop_engine_interfaces.messages.lf_service import loadflow_commands as lf_cmds
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition


def test_worst_contingency_branch_filter():
    f = lf_cmds.WorstContingencyBranchFilter(return_basecase=True)
    assert f.return_basecase is True
    assert f.filter_type == "worst_contingency"


def test_voltage_band_filter():
    f = lf_cmds.VoltageBandFilter(return_basecase=True, v_min=0.95, v_max=1.05)
    assert f.return_basecase is True
    assert f.v_min == 0.95
    assert f.v_max == 1.05
    assert f.filter_type == "voltage_band"


def test_percent_cutoff_branch_filter():
    f = lf_cmds.PercentCutoffBranchFilter(loading_threshold=80.0)
    assert f.loading_threshold == 80.0
    assert f.filter_type == "percent_cutoff"


def test_job_and_subclasses():
    job = lf_cmds.Job(id="job1")
    assert job.id == "job1"
    assert job.job_type == "bare"

    strategy = Strategy(strategy_id="test", timesteps=[], strategy_type="test_strategy")
    job2 = lf_cmds.JobWithSwitchingStrategy(id="job2", strategy=strategy)
    assert job2.strategy == strategy
    assert job2.job_type == "strategy"

    job3 = lf_cmds.JobWithCGMESChanges(id="job3", tp_files=["a.tp"], ssh_files=["a.ssh"])
    assert job3.tp_files == ["a.tp"]
    assert job3.ssh_files == ["a.ssh"]
    assert job3.job_type == "cgmes_changes"

    addition = lf_cmds.InjectionAddition(node=GridElement(kind="bus", id="n1", type="bus"), p_mw=10.0, q_mw=5.0)
    job4 = lf_cmds.JobWithInjectionAdditions(id="job4", additions=[addition])
    assert job4.additions[0] == addition
    assert job4.job_type == "injection_additions"


def test_grid_classes():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    cgmes = lf_cmds.CGMESGrid(n_1_definition=n1def, grid_files=["a.tp"])
    assert cgmes.grid_type == "cgmes"
    assert cgmes.grid_files == ["a.tp"]

    ucte = lf_cmds.UCTEGrid(n_1_definition=n1def, grid_files=["a.ucte"])
    assert ucte.grid_type == "ucte"
    assert ucte.grid_files == ["a.ucte"]

    powsybl = lf_cmds.PowsyblGrid(n_1_definition=n1def, grid_files=["a.xiidm"])
    assert powsybl.grid_type == "powsybl"
    assert powsybl.grid_files == ["a.xiidm"]

    pandapower = lf_cmds.PandapowerGrid(n_1_definition=n1def, grid_files=["a.json"])
    assert pandapower.grid_type == "pandapower"
    assert pandapower.grid_files == ["a.json"]


def test_start_calculation_command():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    grid = lf_cmds.PowsyblGrid(n_1_definition=n1def, grid_files=["a.xiidm"])
    job = lf_cmds.Job(id="job1")
    cmd = lf_cmds.StartCalculationCommand(loadflow_id="lf1", grid_data=grid, method="ac", jobs=[job])
    assert cmd.loadflow_id == "lf1"
    assert cmd.grid_data == grid
    assert cmd.method == "ac"
    assert cmd.jobs == [job]


def test_shutdown_command():
    cmd = lf_cmds.ShutdownCommand(exit_code=1)
    assert cmd.exit_code == 1


def test_loadflow_service_command():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    grid = lf_cmds.PowsyblGrid(n_1_definition=n1def, grid_files=["a.xiidm"])
    job = lf_cmds.Job(id="job1")
    start_cmd = lf_cmds.StartCalculationCommand(loadflow_id="lf1", grid_data=grid, method="ac", jobs=[job])
    cmd = lf_cmds.LoadflowServiceCommand(command=start_cmd)
    assert cmd.command == start_cmd
    assert isinstance(cmd.timestamp, str)
    assert isinstance(cmd.uuid, str)
