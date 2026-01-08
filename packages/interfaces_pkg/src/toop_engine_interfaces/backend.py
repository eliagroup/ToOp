# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The abstract interface definition for accessing data from pandapower/powerfactory/..."""

from abc import ABC, abstractmethod

import numpy as np
from beartype.typing import Optional, Sequence, Union
from jaxtyping import Bool, Float, Int
from toop_engine_interfaces.asset_topology import Topology


class BackendInterface(ABC):
    """Interface for the backend.

    The task of this interface is to provide routines for accessing data from the grid
    modelling software (pandapower/powerfactory/...)

    Specifically not task of this interface is to perform any validations or processing of the data

    This assume a node-branch model, hence busbars would be nodes and lines, trafos, etc would be
    branches. Injections inject onto a node and represent both generators, loads, sgens, ...
    """

    def get_ptdf(self) -> Optional[Float[np.ndarray, " n_branch n_node"]]:
        """Get the PTDF matrix, if it was computed already

        For the relevant substations it is important that only node A is given
        as a column in the reference topology. This is to ensure node A and B
        are treated properly by the algorithm.

        If None is returned, the PTDF matrix will be computed by the solver based on
        from_node, to_node and susceptance.

        Returns
        -------
        Float[np.ndarray, " n_branch n_node"]
            The unextended PTDF matrix, not including second nodes for the relevant
            substations, and not including the PSDF.
        """
        return None

    def get_psdf(self) -> Optional[Float[np.ndarray, " n_branch n_phaseshifters"]]:
        """Get the PSDF matrix, if it was computed already

        If None is returned, the PSDF matrix will be computed by the solver based on
        shift_angle and susceptance.

        This refers to the already reduced PSDF matrix, i.e. without elements that will
        never have a shift angle. See get_phase_shifters for more information.

        Returns
        -------
        Float[np.ndarray, " n_branch n_phaseshifters"]
            The PSDF matrix, not including the PTDF.
        """
        return None

    @abstractmethod
    def get_slack(self) -> int:
        """Get the index of the slack node

        Note that the solver does not support distributed slack nodes, if you have a
        distributed slack, replace all but one slack node by their injections or create
        a virtual slack node that is connected with same-impendance lines to the other
        slack nodes.

        Returns
        -------
        int
            The index of the slack node
        """

    def get_ac_dc_mismatch(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the AC-DC mismatch for each branch

        This is the difference between the AC and DC flow on each branch, i.e. the
        difference between the AC and DC loadflow results.

        This is used in the solver to adjust the DC flow to match the AC flow in the N-0 case.
        If all zeros are returned, the solver will return pure DC flows.

        Positive values mean the AC flow is higher than the DC flow, negative values mean the
        AC flow is lower than the DC flow.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The AC-DC mismatch for each branch and per timestep
        """
        return np.zeros_like(self.get_max_mw_flows(), dtype=float)

    @abstractmethod
    def get_max_mw_flows(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the maximum flow per branch

        The timestep dimension is added to represent temperature-dependent capacity
        limits. If the capacity limits are not temperature-dependent, the same value
        should be returned for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The maximum flow per branch and per timestep
        """

    def get_max_mw_flows_n_1(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get a varying max flow for N-1 if there is a difference or NaN.

        In some circumstances, a higher N-1 load is allowed than N-0 as N-1 leaves some time to
        address an overload in practice - a line won't melt right away if it's overloaded for a
        few seconds until the operators can react.

        If not overloaded, returns all NaNs and hence the values from get_max_mw_flows will be
        used.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The maximum flow per branch and per timestep if overridden, else NaN
        """
        return np.full_like(self.get_max_mw_flows(), np.nan)

    def get_overload_weights(self) -> Float[np.ndarray, " n_branch"]:
        """Get a factor that the overloads are multiplied with for each branch

        This can be used to penalize overloads on certain branches more than on others.

        If this funcion is not overloaded, returns all ones.

        Returns
        -------
        Float[np.ndarray, " n_branch"]
            The overload weights for each branch
        """
        return np.ones(self.get_max_mw_flows().shape[-1])

    def get_n0_n1_max_diff_factors(self) -> Float[np.ndarray, " n_branch"]:
        """Get limits for the relative difference between N-0 and N-1 flows.

        This is an array of factors to the base case flows. Negative factors or NaN values mean the
        branch will be ignored and always have a penalty of 0.
        For example if a branch has a 20 MW diff between N-0 and N-1 in the base case (in the
        unsplit configuration) and the factor is 2, then the maximum allowed diff for the
        n0_n1_delta penalty would be 40 MW. If a negative factor is used, this branch has no
        N-0 to N-1 maximum delta and will always incur a penalty of 0. See
        dc_solver.jax.aggregate_results.compute_n0_n1_max_diff for how these factors are used

        If this function is not overloaded, returns all minus ones (i.e. no branch has a limit).
        """
        return -np.ones(self.get_max_mw_flows().shape[-1])

    def get_cross_coupler_limits(self) -> Float[np.ndarray, " n_bus"]:
        """Get the cross-coupler limits for each relevant substation.

        Returns over all buses to match conventions and if relevant substations are modified
        independently of the cross-coupler limits.

        The limits are a P[MW] Value for each coupler.
        """
        return np.zeros(self.get_relevant_node_mask().shape, dtype=float)

    @abstractmethod
    def get_susceptances(self) -> Float[np.ndarray, " n_branch"]:
        """Get the susceptances of the branches

        Returns
        -------
        Float[np.ndarray, " n_branch"]
            The susceptances of the branches
        """

    @abstractmethod
    def get_from_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get the from nodes of the branches

        Returns
        -------
        Int[np.ndarray, " n_branch"]
            The from nodes of the branches
        """

    @abstractmethod
    def get_to_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get the to nodes of the branches

        Returns
        -------
        Int[np.ndarray, " n_branch"]
            The to nodes of the branches
        """

    def get_controllable_pst_node_mask(self) -> Bool[np.ndarray, " n_node"]:
        """Get the mask of controllable phase shifters over nodes

        True means a node is (bogus node and) a controllable phase shifter, i.e. is connected to a branch
        that is a controllable phase shifter. False means it normal node.

        Returns
        -------
        Bool[np.ndarray, " n_node"]
            The mask of controllable phase shifters over nodes
        """
        # TODO: Implement in backends
        return np.zeros([], dtype=bool)

    @abstractmethod
    def get_shift_angles(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the shift angles of the branches in degree

        The timestep dimension is added to represent time-varying phase shift angles.
        If the phase shift angles are not time-varying, the same value should be returned
        for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The shift angles of the branches
        """

    @abstractmethod
    def get_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the mask of phase shifters

        True means a branch is a phase shifter, i.e. can have shift_degree != 0
        False means it is not a phase shifter. Note that the controllable phase shifters are a subset of this, i.e. not every
        phase shifter is controllable.

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of phase shifters
        """

    def get_controllable_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Which of the phase shifters are controllable

        This must be a subset of get_phase_shift_mask()

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of controllable phase shifters
        """
        return np.zeros_like(self.get_phase_shift_mask())

    def get_phase_shift_taps_and_angles(
        self,
    ) -> tuple[list[Float[np.ndarray, " n_tap_positions"]], list[Float[np.ndarray, " n_tap_positions"]]]:
        """Get a list of tap positions and corresponding angles for each pst.

        The outer list has as many entries as there are controllable PSTs (see
        controllable_phase_shift_mask). The inner np array has as many entries as there are taps for the given PST with each
        value representing the angle shift for the given tap position. The taps are ordered smallest to largest angle shift.
        Each controllable PST must have at least one tap position.
        """
        # Get the viable shift from the zeroth timestep as a viable default value if the user hasn't overloaded the function
        viable_shifts = self.get_shift_angles()[0, self.get_controllable_phase_shift_mask()]
        tap_positions = [0] * len(viable_shifts)
        return tap_positions, [np.array([shift]) for shift in viable_shifts]

    @abstractmethod
    def get_relevant_node_mask(self) -> Bool[np.ndarray, " n_node"]:
        """Get true if a node is part of the relevant nodes

        This refers to the node A (the node that is present in the un-extended PTDF) of the
        relevant substations. The relevant nodes are those that can be split later on
        in the solver.

        Returns
        -------
        Bool[np.ndarray, " n_node"]
            The mask over nodes, indicating if they are relevant (splittable)
        """

    @abstractmethod
    def get_monitored_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the mask of monitored branches for the reward calculation

        True means a branch is monitored, False means it is not monitored

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of monitored branches
        """

    @abstractmethod
    def get_branches_in_maintenance(
        self,
    ) -> Bool[np.ndarray, " n_timestep n_branch"]:
        """Get the mask of branches in maintenance

        True means a branch is in maintenance, False means it is not in maintenance

        The timestep dimension is added to represent time-varying maintenance schedules.
        If the maintenance schedules are not time-varying, the same value should be returned
        for all timesteps.

        Returns
        -------
        Bool[np.ndarray, " n_timestep n_branch"]
            The mask of branches in maintenance
        """

    @abstractmethod
    def get_disconnectable_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the mask of disconnectable branches

        True means a branch is disconnectable as a remedial action, False means it must stay online

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of disconnectable branches
        """

    @abstractmethod
    def get_outaged_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the mask of outaged branches for the N-1 computation

        True means a branch is outaged, False means it is not outaged

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of outaged branches
        """

    @abstractmethod
    def get_outaged_injection_mask(self) -> Bool[np.ndarray, " n_injection"]:
        """Get the mask of outaged injections for the N-1 computation

        True means an injection is outaged, False means it is not outaged

        Returns
        -------
        Bool[np.ndarray, " n_injection"]
            The mask of outaged injections
        """

    @abstractmethod
    def get_multi_outage_branches(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_branch"]:
        """Get the mask of outaged branches for potential multi-outages

        True means a branch is outaged, False means it is not outaged.

        get_multi_outage_branches, get_multi_outage_nodes and get_multi_outage_names have to return
        the same first dimension, i.e. the same number of multi-outages.

        Returns
        -------
        Bool[np.ndarray, " n_multi_outages n_branch"]
            The mask of outaged branches for every multi-outage
        """

    @abstractmethod
    def get_multi_outage_nodes(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_node"]:
        """Get the mask of outaged nodes for potential multi-outages

        True means a node is outaged, False means it is not outaged.

        get_multi_outage_branches, get_multi_outage_nodes and get_multi_outage_names have to return
        the same first dimension, i.e. the same number of multi-outages.

        Returns
        -------
        Bool[np.ndarray, " n_multi_outages n_node"]
            The mask of outaged nodes for every multi-outage
        """

    @abstractmethod
    def get_injection_nodes(self) -> Int[np.ndarray, " n_injection"]:
        """Get the node index of the injections

        Returns
        -------
        Int[np.ndarray, " n_injection"]
            The node index that the injection injects onto
        """

    @abstractmethod
    def get_mw_injections(self) -> Float[np.ndarray, " n_timestep n_injection"]:
        """Get the MW injections of the injections

        The timestep dimension is added to represent time-varying injections.
        If the injections are not time-varying, the same value should be returned
        for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_injection"]
            The MW injections of the injections
        """

    @abstractmethod
    def get_base_mva(self) -> float:
        """Get the baseMVA of the grid

        Returns
        -------
        float
            The base MVA of the grid
        """

    def get_asset_topology(self) -> Optional[Topology]:
        """Get the asset topology of the grid.

        If given, the asset topology for the grid can be returned, describing more
        information about the physical layout of the stations

        Returns
        -------
        Optional[Topology]
            The asset topology of the grid
        """
        return None

    ################################
    # Reporting functions
    @abstractmethod
    def get_node_ids(
        self,
    ) -> Union[Sequence[str], Sequence[int]]:
        """Get the ids of the nodes as a Sequence of length N_node

        Returns
        -------
        Union[Sequence[str], Sequence[int]]
            The ids of the nodes
        """

    @abstractmethod
    def get_branch_ids(
        self,
    ) -> Union[Sequence[str], Sequence[int]]:
        """Get the ids of the branches as a Sequence of length N_branch

        Returns
        -------
        Union[Sequence[str], Sequence[int]]
            The ids of the branches
        """

    @abstractmethod
    def get_injection_ids(
        self,
    ) -> Union[Sequence[str], Sequence[int]]:
        """Get the ids of the injections as a Sequence of length N_injection

        Returns
        -------
        Union[Sequence[str], Sequence[int]]
            The ids of the injections
        """

    @abstractmethod
    def get_multi_outage_ids(self) -> Union[Sequence[str], Sequence[int]]:
        """Get the ids of the multi-outages as a Sequence of length N_multi_outages

        Returns
        -------
        Union[Sequence[str], Sequence[int]]
            The ids of the multi-outages
        """

    @abstractmethod
    def get_node_names(self) -> Sequence[str]:
        """Get the names of the nodes as a Sequence of length N_node

        Returns
        -------
        Sequence[str]
            The names of the nodes
        """

    @abstractmethod
    def get_branch_names(self) -> Sequence[str]:
        """Get the names of the branches as a Sequence of length N_branch

        Returns
        -------
        Sequence[str]
            The names of the branches
        """

    @abstractmethod
    def get_injection_names(self) -> Sequence[str]:
        """Get the names of the injections

        Returns
        -------
        Sequence[str]
            The names of the injections
        """

    @abstractmethod
    def get_multi_outage_names(self) -> Sequence[str]:
        """Get the names of the multi-outages as a Sequence of length N_multi_outages

        If more than one element are involved in a multi-outage you can return a concatenated name

        Returns
        -------
        Sequence[str]
            The names of the multi-outages
        """

    @abstractmethod
    def get_branch_types(self) -> Sequence[str]:
        """Get the type of the branches

        Returns
        -------
        Sequence[str]
            The type of the branches
        """

    @abstractmethod
    def get_node_types(self) -> Sequence[str]:
        """Get the type of the nodes

        Returns
        -------
        Sequence[str]
            The type of the nodes
        """

    @abstractmethod
    def get_injection_types(self) -> Sequence[str]:
        """Get the type of the injections

        Returns
        -------
        Sequence[str]
            The type of the injections
        """

    @abstractmethod
    def get_multi_outage_types(self) -> Sequence[str]:
        """Get the type of the multi-outages as a Sequence of length N_multi_outages

        Returns
        -------
        Sequence[str]
            The type of the multi-outages
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Can be used to return metadata or additional information about the grid.

        This is not used by the solver but rather to easy postprocessing and validation. You can
        return an empty dict if you don't want to use this field.

        Returns
        -------
        dict
            The metadata of the grid
        """

    def get_busbar_outage_map(
        self,
    ) -> Optional[dict[str, Sequence[str]]]:
        """Get the mapping of stations to busbars for the busbar-outages

        The key of the dict is the station's grid_model_id and the value is a list of grid_mdoel_ids
        of the busbars that have to be outaged. If this method is not overloaded, all the physical
        busbars of the relevant stations will be outaged.

        Returns
        -------
        Optional[dict[str, Sequence[str]]]
            The mapping of busbar-outages to the relevant nodes
        """
        return None
