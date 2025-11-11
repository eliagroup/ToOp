from toop_engine_interfaces.asset_topology import Strategy
from toop_engine_interfaces.messages.loadflow_commands_factory import (
    create_cgmes_grid,
    create_injection_addition,
    create_job,
    create_job_with_cgmes_changes,
    create_job_with_injection_additions,
    create_job_with_switching_strategy,
    create_loadflow_service_command,
    create_pandapower_grid,
    create_percent_cutoff_branch_filter,
    create_powsybl_grid,
    create_shutdown_command,
    create_start_calculation_command,
    create_ucte_grid,
    create_voltage_band_filter,
    create_worst_contingency_branch_filter,
)
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition


def test_worst_contingency_branch_filter():
    f = create_worst_contingency_branch_filter(return_basecase=True)
    assert f.return_basecase is True
    assert f.filter_type == "worst_contingency"


def test_voltage_band_filter():
    f = create_voltage_band_filter(return_basecase=True, v_min=0.95, v_max=1.05)
    assert f.return_basecase is True
    assert f.v_min == 0.95
    assert f.v_max == 1.05
    assert f.filter_type == "voltage_band"


def test_percent_cutoff_branch_filter():
    f = create_percent_cutoff_branch_filter(loading_threshold=80.0)
    assert f.loading_threshold == 80.0
    assert f.filter_type == "percent_cutoff"


def test_job_and_subclasses():
    job = create_job(id="job1")
    assert job.id == "job1"
    assert job.job_type == "bare"

    strategy = Strategy(strategy_id="test", timesteps=[], strategy_type="test_strategy")
    job2 = create_job_with_switching_strategy(id="job2", strategy=strategy)
    assert Strategy.model_validate_json(job2.strategy) == strategy
    assert job2.base.job_type == "strategy"

    job3 = create_job_with_cgmes_changes(id="job3", tp_files=["a.tp"], ssh_files=["a.ssh"])
    assert job3.tp_files == ["a.tp"]
    assert job3.ssh_files == ["a.ssh"]
    assert job3.base.job_type == "cgmes_changes"

    addition = create_injection_addition(node=GridElement(kind="bus", id="n1", type="bus"), p_mw=10.0, q_mw=5.0)
    job4 = create_job_with_injection_additions(id="job4", additions=[addition])
    assert job4.additions[0] == addition
    assert job4.base.job_type == "injection_additions"


def test_grid_classes():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    cgmes = create_cgmes_grid(n_1_definition=n1def, grid_files=["a.tp"])
    assert cgmes.grid_type == "cgmes"
    assert cgmes.grid_files == ["a.tp"]

    ucte = create_ucte_grid(n_1_definition=n1def, grid_files=["a.ucte"])
    assert ucte.grid_type == "ucte"
    assert ucte.grid_files == ["a.ucte"]

    powsybl = create_powsybl_grid(n_1_definition=n1def, grid_files=["a.xiidm"])
    assert powsybl.grid_type == "powsybl"
    assert powsybl.grid_files == ["a.xiidm"]

    pandapower = create_pandapower_grid(n_1_definition=n1def, grid_files=["a.json"])
    assert pandapower.grid_type == "pandapower"
    assert pandapower.grid_files == ["a.json"]


def test_start_calculation_command():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    grid = create_powsybl_grid(n_1_definition=n1def, grid_files=["a.xiidm"])
    job = create_job(id="job1")
    cmd = create_start_calculation_command(loadflow_id="lf1", grid_data=grid, method="ac", jobs=[job])
    assert cmd.loadflow_id == "lf1"
    assert cmd.powsybl_grid == grid
    assert cmd.method == "ac"
    assert cmd.jobs[0] == job


def test_shutdown_command():
    cmd = create_shutdown_command(exit_code=1)
    assert cmd.exit_code == 1


def test_loadflow_service_command():
    n1def = Nminus1Definition(contingencies=[], monitored_elements=[])
    grid = create_powsybl_grid(n_1_definition=n1def, grid_files=["a.xiidm"])
    job = create_job(id="job1")
    start_cmd = create_start_calculation_command(loadflow_id="lf1", grid_data=grid, method="ac", jobs=[job])
    cmd = create_loadflow_service_command(command=start_cmd)
    assert cmd.start_calculation == start_cmd
    assert isinstance(cmd.timestamp, str)
    assert isinstance(cmd.uuid, str)
