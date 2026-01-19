# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Describes the interfaces for a loadflow service providing N-1 computations in AC or DC to customers.

The communication follows a 2 step pattern:

1. Grid load - Load a grid file into the engine and potentially perform some preprocessing. The engine should return a grid
    reference upon this call, which is used in the job to reference the grid. Multiple grids can be loaded at the same time,
    it is the responsibility of the engine to perform memory management (i.e. swap out a grid to disk if memory is full).
2. Execute jobs - Run a loadflow job on the engine. The job references the grid that was loaded in step 2. A call to this can
    contain multiple jobs, allowing the engine to parallelize over jobs in addition to parallelizing over timesteps/outages.

The LoadflowEngine protocol describes this two step process in detail.

"""

import uuid
from abc import ABC
from datetime import datetime

from beartype.typing import Literal, Optional, Protocol, Union
from pydantic import BaseModel, Field
from toop_engine_interfaces.asset_topology import Strategy
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition


class BaseFilter(BaseModel, ABC):
    """A base class for filters.

    The idea behind filters is to implement logics that can not be implemented with the monitored elements directly but that
    are dependent on the loadflow results

    This is not to be used directly, but to be subclassed.
    """


class WorstContingencyBranchFilter(BaseFilter):
    """If this filter is applied, it will reduce the branch results to only the worst N-1 case/side per branch and timestep.

    Worst is determined with respect to the loading value. Branches that don't have a rating will never be returned.

    If there are multiple worst N-1 cases/sides that produce a tie, one will be chosen at random.
    This filtering happens on a per-timestep basis, meaning for every timestep there should be as many results as monitored
    branches, but they can refer to different N-1 cases.
    """

    return_basecase: bool = False
    """Whether to return the basecase still. If this is set to True, the basecase will always be returned even if it is not
    the worst case."""

    filter_type: Literal["worst_contingency"] = "worst_contingency"
    """An identifier for the discriminated union"""


class VoltageBandFilter(BaseFilter):
    """If this filter is applied, it will reduce the node results to only the results that are outside of a specified band.

    The band is defined by a minimum and maximum p.u. value for all nodes
    """

    return_basecase: bool = False
    """Whether to return the basecase at all times. If this is set to True, the basecase will always be returned even if it
    is inside the band and hence should be filtered out."""

    v_min: float
    """The minimum voltage in p.u. - values below this will be returned."""

    v_max: float
    """The maximum voltage in p.u. - values above this will be returned"""

    filter_type: Literal["voltage_band"] = "voltage_band"
    """An identifier for the discriminated union"""


class PercentCutoffBranchFilter(BaseFilter):
    """Filter, if applied returns only branch results that are above a loading threshold.

    Elements for which no loading could be computed (e.g. due to missing ratings) are never returned.
    This filtering happens on a per-timestep basis, i.e. if a branch/contingency is above the threshold in one timestep, it
    will be returned in exactly that timestep.
    """

    loading_threshold: float
    """The loading threshold in percent. Only branches with a loading above this threshold are returned."""

    filter_type: Literal["percent_cutoff"] = "percent_cutoff"
    """An identifier for the discriminated union"""


class Job(BaseModel):
    """A job constitutes a single workload and will produce a LoadflowResults object.

    There are different types of jobs based on the workload, the simple being a base job with no changes to the base
    grid.

    """

    id: str
    """A unique identifier for the job. This is used to reference the job in the results."""

    branch_filter: Optional[Union[WorstContingencyBranchFilter, PercentCutoffBranchFilter]] = Field(
        default=None, discriminator="filter_type"
    )
    """Filters for the branch results table. Exactly one filter can be active per table and job"""

    node_filter: Optional[VoltageBandFilter] = Field(default=None, discriminator="filter_type")
    """Filters for the node results table. Exactly one filter can be active per table and job"""

    job_type: Literal["bare"] = "bare"
    """An identifier for the discriminated union"""

    timestep_subselection: Optional[list[int]] = None
    """If this is set, only the timesteps in this list are computed. If this is not set, all timesteps are computed.
    Timesteps are referenced by their index in the grid file, starting at 0."""


class JobWithSwitchingStrategy(Job):
    """A job that includes a switching strategy.

    This strategy shall be applied before the loadflow computation and might alter the topology of the grid.
    """

    strategy: Strategy
    """The switching strategy that is to be applied before the loadflow computation"""

    job_type: Literal["strategy"] = "strategy"
    """An identifier for the discriminated union"""


class JobWithCGMESChanges(Job):
    """A job that includes changes in CGMES format. This is only applicable if the grid is a CGMES grid"""

    tp_files: list[str]
    """The file including the topology changes that shall be applied.

    There must be as many entries as timesteps in the grid, but the same file can be referenced multiple times.
    """

    ssh_files: list[str]
    """The file including the state/injection changes that shall be applied.

    There must be as many entries as timesteps in the grid, but the same file can be referenced multiple times.
    """

    job_type: Literal["cgmes_changes"] = "cgmes_changes"
    """An identifier for the discriminated union"""


class InjectionAddition(BaseModel):
    """A single addition of an injection at a node.

    This feature only support PQ nodes, if attempted to apply to a branch, pv node or slack node, the engine should ignore
    this addition and log a warning.

    Positive values shall have the same effect as sgens, i.e. power is produced, while negative values will have the same
    effect as loads, i.e. power is consumed.
    """

    node: GridElement
    """The node to which the injection is added"""

    p_mw: float
    """The active power in MW that is added to the node"""

    q_mw: float
    """The reactive power that is added to the node"""

    timestep_subselection: Optional[list[int]] = None
    """If this is given, the addition only happens in the timesteps that are in this list. If this is not given, the addition
    happens in all timesteps."""


class JobWithInjectionAdditions(Job):
    """Adds a constant injection to a node in the grid.

    This feature assumes all injections are added to PQ nodes - otherwise they will be ignored.

    Positive values shall have the same effect as sgens, i.e. power is produced, while negative values will have the same
    effect as loads, i.e. power is consumed.
    """

    additions: list[InjectionAddition]
    """The injections that are added to the grid"""

    job_type: Literal["injection_additions"] = "injection_additions"
    """An identifier for the discriminated union"""


class BaseGrid(BaseModel, ABC):
    """A base class for grid files. This is not to be used directly, but to be subclassed"""

    n_1_definition: Optional[Nminus1Definition] = None
    """The N-1 cases that are to be computed. If this is provided, this shall overwrite the N-1 cases that are defined in
    the grid files if the format supports such definition. If this is not provided, the N-1 cases that are defined in the
    grid files shall be used. If neither is provided, the engine should throw an error."""

    grid_type: Literal["cgmes", "ucte", "pandapower", "powsybl"]
    """An identifier for the discriminated union, to be set by the subclasses"""


class CGMESGrid(BaseGrid):
    """A CGMES grid file does not need to store much additional information"""

    grid_files: list[str]
    """A list of paths to grid files. This can include multiple .tp and .ssh files which are to be interpreted as multiple
    timesteps. If a .tp and a .ssh file have the same filename or the same timestep metadata inside the file, they correspond
    to the same timestep. Timesteps should be sorted by the timestep information inside the CGMES files."""

    grid_type: Literal["cgmes"] = "cgmes"
    """An identifier for the discriminated union"""


class UCTEGrid(BaseGrid):
    """A list of UCTE files that are to be loaded into the engine"""

    grid_files: list[str]
    """A list of paths to grid files. This can include multiple .ucte files which are to be interpreted as multiple
    timesteps. Timesteps should be interpreted in the order of this list"""

    grid_type: Literal["ucte"] = "ucte"
    """An identifier for the discriminated union"""


class PowsyblGrid(BaseGrid):
    """A list of powsybl xiidm files that are to be loaded into the engine"""

    grid_files: list[str]
    """A list of xiidm files that represent the timesteps. Timesteps should be interpreted in the order of this list"""

    grid_type: Literal["powsybl"] = "powsybl"


class PandapowerGrid(BaseGrid):
    """A list of pandapower json files that are to be loaded into the engine"""

    grid_files: list[str]
    """A list of pandapower files that represent the timesteps. Timesteps should be interpreted in the order of this list"""

    grid_type: Literal["pandapower"] = "pandapower"


class StartCalculationCommand(BaseModel):
    """A command to run a list of jobs on the engine.

    This can involve multiple N-1 computations with different changes to
    the base grid, but a job must share the same grid file.
    """

    loadflow_id: str
    """A unique identifier for the loadflow run. This is used to identify the result"""

    grid_data: Union[PowsyblGrid, PandapowerGrid, UCTEGrid, CGMESGrid] = Field(discriminator="grid_type")
    """The string that was returned by load_grid, identifying the grid file that this job collection shall run on"""

    method: Literal["ac", "dc"]
    """The method that is to be used for the loadflow computations. This can be either AC or DC. This must be the same
    for all jobs in the list"""

    jobs: list[Job]
    """The jobs to be executed"""


class LoadflowEngine(Protocol):
    """A protocol for a loadflow engine.

    Roughly, an engine shall be able to load grids and execute jobs on them. There is some memory management that the engine
    needs to perform internally, i.e. it could happen that two users want to use the same engine in parallel. In that case,
    the engine should swap out grids to disk if memory is full.
    """

    def run_job(self, job: StartCalculationCommand) -> list[LoadflowResults]:
        """Run a job on the engine.

        This can involve multiple N-1 computations with different changes to the base grid,
        identified through multiple jobs in the BatchJob. The engine should return the results of the jobs as in-memory
        dataframes.
        """


class ShutdownCommand(BaseModel):
    """A command to shut down the preprocessing worker"""

    exit_code: Optional[int] = 0
    """The exit code to return"""


class LoadflowServiceCommand(BaseModel):
    """A wrapper to aid deserialization"""

    command: Union[StartCalculationCommand, ShutdownCommand]
    """The actual command posted"""

    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
    """When the command was sent"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for this command message, used to avoid duplicates during processing"""
