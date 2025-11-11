"""
Factory methods for creating objects from loadflow_commands.py.

Each function creates and returns an instance of the corresponding class.
"""

import uuid
from datetime import datetime
from typing import List, Literal, Optional, TypeAlias, Union

import google.protobuf.message
from toop_engine_interfaces.asset_topology import Strategy
from toop_engine_interfaces.messages.protobuf_schema.lf_service.loadflow_commands_pb2 import (
    CGMESGrid,
    InjectionAddition,
    Job,
    JobWithCGMESChanges,
    JobWithInjectionAdditions,
    JobWithSwitchingStrategy,
    LoadflowServiceCommand,
    PandapowerGrid,
    PercentCutoffBranchFilter,
    PowsyblGrid,
    ShutdownCommand,
    StartCalculationCommand,
    UCTEGrid,
    VoltageBandFilter,
    WorstContingencyBranchFilter,
)
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition

JobType: TypeAlias = Literal["bare", "strategy", "cgmes_changes", "injection_additions"]
MethodType: TypeAlias = Literal["ac", "dc"]
GridType: TypeAlias = Literal["powsybl", "pandapower", "ucte", "cgmes"]
FilterType: TypeAlias = Literal["worst_contingency", "percent_cutoff", "voltage_band"]


def create_worst_contingency_branch_filter(
    return_basecase: bool = False, filter_type: Literal["worst_contingency"] = "worst_contingency"
) -> WorstContingencyBranchFilter:
    """
    Create a WorstContingencyBranchFilter object.

    Parameters
    ----------
    return_basecase : bool, optional
        Whether to return the basecase still.
    filter_type: Literal["worst_contingency"] = "worst_contingency"
        An identifier for the discriminated union. Default is "worst_contingency".

    Returns
    -------
    WorstContingencyBranchFilter
        The filter object.
    """
    return WorstContingencyBranchFilter(return_basecase=return_basecase, filter_type=filter_type)


def create_voltage_band_filter(
    v_min: float, v_max: float, return_basecase: bool = False, filter_type: Literal["voltage_band"] = "voltage_band"
) -> VoltageBandFilter:
    """
    Create a VoltageBandFilter object.

    Parameters
    ----------
    v_min : float
        Minimum voltage in p.u.
    v_max : float
        Maximum voltage in p.u.
    return_basecase : bool, optional
        Whether to always return the basecase.

    filter_type: Literal["voltage_band"] = "voltage_band"
        An identifier for the discriminated union. Default is "voltage_band".

    Returns
    -------
    VoltageBandFilter
        The filter object.
    """
    return VoltageBandFilter(v_min=v_min, v_max=v_max, return_basecase=return_basecase, filter_type=filter_type)


def create_percent_cutoff_branch_filter(
    loading_threshold: float, filter_type: Literal["percent_cutoff"] = "percent_cutoff"
) -> PercentCutoffBranchFilter:
    """
    Create a PercentCutoffBranchFilter object.

    Parameters
    ----------
    loading_threshold : float
        Loading threshold in percent.
    filter_type: Literal["percent_cutoff"] = "percent_cutoff"
        An identifier for the discriminated union. Default is "percent_cutoff".

    Returns
    -------
    PercentCutoffBranchFilter
        The filter object.
    """
    return PercentCutoffBranchFilter(loading_threshold=loading_threshold, filter_type=filter_type)


def _set_branch_filter(
    msg: google.protobuf.message.Message,
    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]],
) -> None:
    """
    Set branch_filter on job.

    Parameters
    ----------
    msg : protobuf message
        The job object to modify.
    branch_filter : WorstContingencyBranchFilter or PercentCutoffBranchFilter, optional
        Branch filter to set.
    """
    if branch_filter is not None:
        if isinstance(branch_filter, WorstContingencyBranchFilter):
            msg.worst_contingency.CopyFrom(branch_filter)
        elif isinstance(branch_filter, PercentCutoffBranchFilter):
            msg.percent_cutoff.CopyFrom(branch_filter)


def _set_node_filter(msg: google.protobuf.message.Message, node_filter: Optional[VoltageBandFilter]) -> None:
    """
    Set node_filter on job.

    Parameters
    ----------
    msg : protobuf message
        The job object to modify.
    node_filter : VoltageBandFilter, optional
        Node filter to set.
    """
    if node_filter is not None and hasattr(msg, "node_filter"):
        msg.node_filter.CopyFrom(node_filter)


def _set_timestep_subselection(msg: google.protobuf.message.Message, timestep_subselection: Optional[List[int]]) -> None:
    """
    Set timestep_subselection on job.

    Parameters
    ----------
    msg : protobuf message.Message
        The job object to modify.
    timestep_subselection : list of int, optional
        Timesteps to set.
    """
    if timestep_subselection is not None:
        msg.timestep_subselection.extend(timestep_subselection)


def create_job(
    id: str,
    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]] = None,
    node_filter: Optional[VoltageBandFilter] = None,
    timestep_subselection: Optional[List[int]] = None,
    job_type: JobType = "bare",
) -> Job:
    """
    Create a Job object.

    Parameters
    ----------
    id : str
        Unique job identifier.
    branch_filter : WorstContingencyBranchFilter or PercentCutoffBranchFilter, optional
        Branch filter.
    node_filter : VoltageBandFilter, optional
        Node filter.
    timestep_subselection : list of int, optional
        Timesteps to compute.
    job_type: JobType, optional
        The type of job.

    Returns
    -------
    Job
        The job object.
    """
    job = Job(
        id=id,
    )
    _set_branch_filter(job, branch_filter)
    _set_node_filter(job, node_filter)
    _set_timestep_subselection(job, timestep_subselection)
    job.job_type = job_type
    return job


def create_job_with_switching_strategy(
    id: str,
    strategy: Strategy,
    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]] = None,
    node_filter: Optional[VoltageBandFilter] = None,
    timestep_subselection: Optional[List[int]] = None,
    job_type: Literal["strategy"] = "strategy",
) -> JobWithSwitchingStrategy:
    """
    Create a JobWithSwitchingStrategy object.

    Parameters
    ----------
    id : str
        Unique job identifier.
    strategy : Strategy
        Switching strategy.
    branch_filter : WorstContingencyBranchFilter or PercentCutoffBranchFilter, optional
        Branch filter.
    node_filter : VoltageBandFilter, optional
        Node filter.
    timestep_subselection : list of int, optional
        Timesteps to compute.
    job_type: Literal["strategy"] = "strategy",
        The type of job.

    Returns
    -------
    JobWithSwitchingStrategy
        The job object.
    """
    base_job = create_job(
        id=id,
        branch_filter=branch_filter,
        node_filter=node_filter,
        timestep_subselection=timestep_subselection,
        job_type=job_type,
    )
    job_with_strategy = JobWithSwitchingStrategy(
        base=base_job,
        strategy=strategy.model_dump_json(),
    )
    return job_with_strategy


def create_job_with_cgmes_changes(
    id: str,
    tp_files: List[str],
    ssh_files: List[str],
    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]] = None,
    node_filter: Optional[VoltageBandFilter] = None,
    timestep_subselection: Optional[List[int]] = None,
    job_type: Literal["cgmes_changes"] = "cgmes_changes",
) -> JobWithCGMESChanges:
    """
    Create a JobWithCGMESChanges object.

    Parameters
    ----------
    id : str
        Unique job identifier.
    tp_files : list of str
        Topology change files.
    ssh_files : list of str
        State/injection change files.
    branch_filter : WorstContingencyBranchFilter or PercentCutoffBranchFilter, optional
        Branch filter.
    node_filter : VoltageBandFilter, optional
        Node filter.
    timestep_subselection : list of int, optional
        Timesteps to compute.
    job_type: Literal["cgmes_changes"] = "cgmes_changes",
        The type of job.

    Returns
    -------
    JobWithCGMESChanges
        The job object.
    """
    base_job = create_job(
        id=id,
        branch_filter=branch_filter,
        node_filter=node_filter,
        timestep_subselection=timestep_subselection,
        job_type=job_type,
    )
    job_with_cgmes_changes = JobWithCGMESChanges(
        base=base_job,
        tp_files=tp_files,
        ssh_files=ssh_files,
    )
    return job_with_cgmes_changes


def create_injection_addition(
    node: GridElement,
    p_mw: float,
    q_mw: float,
    timestep_subselection: Optional[List[int]] = None,
) -> InjectionAddition:
    """
    Create an InjectionAddition object.

    Parameters
    ----------
    node : GridElement
        Node to add injection.
    p_mw : float
        Active power in MW.
    q_mw : float
        Reactive power in MW.
    timestep_subselection : list of int, optional
        Timesteps for addition.

    Returns
    -------
    InjectionAddition
        The injection addition object.
    """
    inj_add = InjectionAddition(
        node=node.model_dump_json(),
        p_mw=p_mw,
        q_mw=q_mw,
    )
    _set_timestep_subselection(inj_add, timestep_subselection)
    return inj_add


def create_job_with_injection_additions(
    id: str,
    additions: List[InjectionAddition],
    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]] = None,
    node_filter: Optional[VoltageBandFilter] = None,
    timestep_subselection: Optional[List[int]] = None,
    job_type: Literal["injection_additions"] = "injection_additions",
) -> JobWithInjectionAdditions:
    """
    Create a JobWithInjectionAdditions object.

    Parameters
    ----------
    id : str
        Unique job identifier.
    additions : list of InjectionAddition
        Injections to add.
    branch_filter : WorstContingencyBranchFilter or PercentCutoffBranchFilter, optional
        Branch filter.
    node_filter : VoltageBandFilter, optional
        Node filter.
    timestep_subselection : list of int, optional
        Timesteps to compute.
    job_type: Literal["injection_additions"] = "injection_additions",
        The type of job.

    Returns
    -------
    JobWithInjectionAdditions
        The job object.
    """
    base_job = create_job(
        id=id,
        branch_filter=branch_filter,
        node_filter=node_filter,
        timestep_subselection=timestep_subselection,
        job_type=job_type,
    )

    job_with_injection_additions = JobWithInjectionAdditions(
        base=base_job,
        additions=additions,
    )
    return job_with_injection_additions


def _save_n_1_definition_to_grid(
    grid: google.protobuf.message.Message,
    n_1_definition: Nminus1Definition,
    grid_type: GridType,
) -> None:
    """
    Save N-1 definition to grid object.

    Parameters
    ----------
    grid : protobuf message
        The grid object to modify.
    n_1_definition : Nminus1Definition
        N-1 cases definition.
    grid_type: GridType
        The type of grid.
    """
    grid.grid_type = grid_type
    if n_1_definition is not None:
        grid.n_1_definition = n_1_definition.model_dump_json()


def create_cgmes_grid(
    grid_files: List[str],
    n_1_definition: Optional[Nminus1Definition] = None,
    grid_type: Literal["cgmes"] = "cgmes",
) -> CGMESGrid:
    """
    Create a CGMESGrid object.

    Parameters
    ----------
    grid_files : list of str
        Paths to CGMES grid files.
    n_1_definition : Nminus1Definition, optional
        N-1 cases definition.
    grid_type: GridType = "cgmes",
        The type of grid.

    Returns
    -------
    CGMESGrid
        The grid object.
    """
    grid = CGMESGrid(grid_files=grid_files)
    _save_n_1_definition_to_grid(grid, n_1_definition, grid_type)
    return grid


def create_ucte_grid(
    grid_files: List[str],
    n_1_definition: Optional[Nminus1Definition] = None,
    grid_type: Literal["ucte"] = "ucte",
) -> UCTEGrid:
    """
    Create a UCTEGrid object.

    Parameters
    ----------
    grid_files : list of str
        Paths to UCTE grid files.
    n_1_definition : Nminus1Definition, optional
        N-1 cases definition.
    grid_type: GridType = "ucte",
        The type of grid.

    Returns
    -------
    UCTEGrid
        The grid object.
    """
    grid = UCTEGrid(grid_files=grid_files)
    _save_n_1_definition_to_grid(grid, n_1_definition, grid_type)
    return grid


def create_powsybl_grid(
    grid_files: List[str],
    n_1_definition: Optional[Nminus1Definition] = None,
    grid_type: Literal["powsybl"] = "powsybl",
) -> PowsyblGrid:
    """
    Create a PowsyblGrid object.

    Parameters
    ----------
    grid_files : list of str
        Paths to xiidm files.
    n_1_definition : Nminus1Definition, optional
        N-1 cases definition.
    grid_type: GridType = "powsybl",
        The type of grid.

    Returns
    -------
    PowsyblGrid
        The grid object.
    """
    grid = PowsyblGrid(grid_files=grid_files)
    _save_n_1_definition_to_grid(grid, n_1_definition, grid_type)
    return grid


def create_pandapower_grid(
    grid_files: List[str],
    n_1_definition: Optional[Nminus1Definition] = None,
    grid_type: Literal["pandapower"] = "pandapower",
) -> PandapowerGrid:
    """
    Create a PandapowerGrid object.

    Parameters
    ----------
    grid_files : list of str
        Paths to pandapower files.
    n_1_definition : Nminus1Definition, optional
        N-1 cases definition.
    grid_type: GridType = "pandapower",
        The type of grid.

    Returns
    -------
    PandapowerGrid
        The grid object.
    """
    grid = PandapowerGrid(grid_files=grid_files)
    _save_n_1_definition_to_grid(grid, n_1_definition, grid_type)
    return grid


def create_start_calculation_command(
    loadflow_id: str,
    grid_data: Union[PowsyblGrid, PandapowerGrid, UCTEGrid, CGMESGrid],
    method: MethodType,
    jobs: List[Job],
) -> StartCalculationCommand:
    """
    Create a StartCalculationCommand object.

    Parameters
    ----------
    loadflow_id : str
        Unique loadflow run identifier.
    grid_data : PowsyblGrid or PandapowerGrid or UCTEGrid or CGMESGrid
        Grid data.
    method : MethodType
        Loadflow method ("ac" or "dc").
    jobs : list of Job
        Jobs to execute.

    Returns
    -------
    StartCalculationCommand
        The command object.
    """
    com = StartCalculationCommand(
        loadflow_id=loadflow_id,
        method=method,
        jobs=jobs,
    )
    if isinstance(grid_data, PowsyblGrid):
        com.powsybl_grid.CopyFrom(grid_data)
    elif isinstance(grid_data, PandapowerGrid):
        com.pandapower_grid.CopyFrom(grid_data)
    elif isinstance(grid_data, UCTEGrid):
        com.ucte_grid.CopyFrom(grid_data)
    elif isinstance(grid_data, CGMESGrid):
        com.cgmes_grid.CopyFrom(grid_data)
    return com


def create_shutdown_command(exit_code: Optional[int] = 0) -> ShutdownCommand:
    """
    Create a ShutdownCommand object.

    Parameters
    ----------
    exit_code : int, optional
        Exit code to return.

    Returns
    -------
    ShutdownCommand
        The command object.
    """
    return ShutdownCommand(exit_code=exit_code)


def create_loadflow_service_command(
    command: Union[StartCalculationCommand, ShutdownCommand],
    timestamp: Optional[str] = None,
    uuid_str: Optional[str] = None,
) -> LoadflowServiceCommand:
    """
    Create a LoadflowServiceCommand object.

    Parameters
    ----------
    command : StartCalculationCommand or ShutdownCommand
        The actual command.
    timestamp : str, optional
        Timestamp of the command.
    uuid_str : str, optional
        Unique identifier for the command.

    Returns
    -------
    LoadflowServiceCommand
        The service command object.
    """
    if timestamp is None:
        timestamp = str(datetime.now())
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())
    com = LoadflowServiceCommand(timestamp=timestamp, uuid=uuid_str)
    if isinstance(command, StartCalculationCommand):
        com.start_calculation.CopyFrom(command)
    elif isinstance(command, ShutdownCommand):
        com.shutdown.CopyFrom(command)
    return com
