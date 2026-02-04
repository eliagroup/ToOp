# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Defines the functional interfaces for different procedures during the postprocessing phase.

This includes
-> Loading the grid data
-> Extracting the network data information
-> Applying the topology
-> Running the DC loadflow
-> Running the AC loadflow
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, TypeAlias, Union

import pandera.typing as pat
from fsspec import AbstractFileSystem
from toop_engine_dc_solver.export.asset_topology_to_dgs import SwitchUpdateSchema
from toop_engine_interfaces.asset_topology import RealizedTopology
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet

AdditionalActionInfo: TypeAlias = Union[pat.DataFrame[SwitchUpdateSchema], RealizedTopology]


class AbstractLoadflowRunner(ABC):
    """A object interface for a loadflow runner, encapsulating a grid modelling software.

    A loadflow runner goes through the following state transitions:

    - unloaded: object just created, no grid data in the Loadflow runner
    - base grid loaded: Only the unsplit base grid is loaded, but no topology is applied
    - network data extracted: The network data is extracted from the grid
    - topology applied: The topology is applied to the grid, the runner is ready to perform
        loadflows
    -> AC + DC loadflows can be run

    """

    @abstractmethod
    def load_base_grid_fs(self, filesystem: AbstractFileSystem, grid_path: Path) -> None:
        """Load the base grid into the loadflow runner, loading from a file system.

        Parameters
        ----------
        filesystem : AbstractFileSystem
            The file system to use to load the grid.
        grid_path : Path
            The path to the grid file
        """

    @abstractmethod
    def load_base_grid(self, grid_path: Path) -> None:
        """Load the base grid into the loadflow runner.

        Parameters
        ----------
        grid_path : Path
            The path to the grid file
        """

    @abstractmethod
    def store_nminus1_definition(self, nminus1_definition: Nminus1Definition) -> None:
        """Store the N-1 definition in the loadflow runner.

        This needs to reference the grid data.
        """

    @abstractmethod
    def get_nminus1_definition(self) -> Nminus1Definition:
        """Get the N-1 definition from the loadflow runner.

        Returns
        -------
        Nminus1Definition
            The N-1 definition stored in the loadflow runner.
        """

    @abstractmethod
    def store_action_set(self, action_set: ActionSet) -> None:
        """Store the action set in the loadflow runner.

        This needs to reference the grid data.
        """

    @abstractmethod
    def run_dc_n_0(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run only the N-0 analysis, no N-1

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with exactly one case, the BASECASE only. The job_id will be "" and the timesteps will be
            in arange(n_timesteps).
        """

    @abstractmethod
    def run_dc_loadflow(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run the DC loadflow on the grid.

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with a full N-1 analysis. The job_id will be "" and the timestep will be in
            arange(n_timesteps).
        """

    @abstractmethod
    def run_ac_loadflow(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run the AC loadflow on the grid.

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with a full N-1 analysis. The job_id will be "" and the timestep will be in
            arange(n_timesteps).
        """

    @abstractmethod
    def get_last_action_info(self) -> Optional[AdditionalActionInfo]:
        """Return additional information about the last applied action.

        Each call to run_... that takes actions shall update an internal storage of additional information
        which can be returned by a consumer of the interface, e.g. for returning the switching distance of
        the actions.

        If no previous action was applied, this method should return None.
        """

    def copy(self) -> "AbstractLoadflowRunner":
        """Create a deep copy of the loadflow runner

        if this is not overridden, the default implementation will use the copy.deepcopy function

        Returns
        -------
        AbstractLoadflowRunner
            A deep copy of the loadflow runner
        """
        return deepcopy(self)
