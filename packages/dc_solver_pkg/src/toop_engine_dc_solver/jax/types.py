# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Holds dataclasses and types used throughout the jax solver.

This is in one central file to simplify import management
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from beartype.typing import Optional, Protocol, Union
from jax import numpy as jnp
from jax_dataclasses import Static, pytree_dataclass
from jaxtyping import Array, Bool, Float, Int, PyTree
from toop_engine_dc_solver.jax.utils import HashableArrayWrapper
from toop_engine_interfaces.types import MetricType


@dataclass
class SolverConfig:
    """Contains the solver configuration, i.e. all the variables that jax shouldn't trace."""

    branches_per_sub: HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
    """The number of powerlines connected to each relevant substation. In the numpy code, this is
    called looper.c_l
    """

    slack: int
    """The index of the slack bus. Note that we don't support distributed slacks"""

    n_stat: int
    """The number of substations in the network before splitting/extending the ptdf. I.e. in
    case 14 there are n_stat=14 substations but because 5 substations can be split, there need to be
    19 busbars."""

    rel_stat_map: HashableArrayWrapper[Int[Array, " n_sub_relevant"]]
    """A mapping from the relevant substations to the actual substations (busbars) in the network.
    For every relevant substation, you can find out the respective bus among all busbars that
    all of these substations elements are on in the unsplit configuration"""

    number_most_affected_n_0: int
    """The number of worst N-0 flows to track for a topology"""

    number_most_affected: int
    """The number of worst N-1 contingency results to track for a topology."""

    number_max_out_in_most_affected: Optional[int]
    """The number of worst N-1 contingency results to track for a single outage."""

    batch_size_bsdf: int
    """The batch size for the BSDF computation, static flow computation and LODF matrix computation.
    A high bsdf batch size will incur a memory penalty as a high number of ptdf and lodf matrices
    need to be stored"""

    batch_size_injection: int
    """The batch size for the injection computation and contingency analysis"""

    buffer_size_injection: Optional[int]
    """The buffer size for how many injection batches can be stored. The theoretical upper bound is
    batch_size_bsdf * max_injections_per_topology / batch_size_injections, however likely a lower
    value is sufficient. A high value has few runtime implications (only on aggregate) but will
    consume more memory. If not given, will default to the upper bound"""

    limit_n_subs: Optional[int]
    """Limit the number of affected substations. If None, there will always be n_sub_relevant
    computations, even if there is never a topology that splits all substations"""

    aggregation_metric: MetricType
    """The metric to use for selecting the best injection candidate in bruteforce mode. This will be
    passed to aggregate_to_metric to compute a metric, which the bruteforce module tries to
    minimize. You can also pass a custom function if you use run_solver_inj_bruteforce directly."""

    distributed: bool
    """Whether to use distributed computing (i.e. jax.pmap) within run_solver to parallelize the
    computation across multiple devices. If false, it will run only on jax default device, if true
    it will use all available devices. If true, the batch_size_bsdf parameter corresponds to the
    batch-size per device and the effective batch size is multiplied by the number of devices."""

    contingency_ids: list[str]
    """The IDs of the contingencies that have to be calculated in N-1 analysis. The length
    of contingency_ids should match the number of N-1 cases."""

    enable_bb_outages: bool = False
    """
    Whether to enable busbar outages. This will change the way the N-1 flows are computed."""

    bb_outage_as_nminus1: bool = True
    """
    Whether to treat busbar outages as N-1 outages. This means that the busbar outage will be
    conisdered as another N-1 case and the optimiser will try to solve busbar outage problems.

    If False, we calculate the penalty for the busbar outage by comparing the bb_otages before
    and after the splits. In this case, the optimiser will try to ensure that the output
    topology doesn't exacerbate the busbar outage problem."""

    clip_bb_outage_penalty: bool = False
    """
    Whether to clip the lower bound of the busbar outage penalty to 0.
    We set this parameter to False, if we want the optimiser to solve busbar outage problems in the grid. However,
    when we just want to ensure that the busbar outage problems are not exacerbated due to the optimiser, we set
    this to True."""

    def __hash__(self) -> int:
        """Get id as the hash for the static information.

        We expect only one instance of this class to be used at a time so id(self) is a good hash,
        no need to access the content

        Returns
        -------
            int
                The hash of the object
        """
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Test for equality.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the self is other, False otherwise.
        """
        return self is other

    @property
    def n_sub_relevant(self) -> int:
        """The number of relevant substations"""
        return len(self.rel_stat_map)

    @property
    def max_branch_per_sub(self) -> int:
        """The maximum number of branches connected to a substation"""
        return self.branches_per_sub.val.max().item()


@pytree_dataclass
class DynamicInformation:
    """Holds the dynamic information for the solver.

    A dataclass holding all the information that is needed for the solver and that is ok to be
    traced, i.e. there are no data-dependent conditionals within the code. Hence, changing values
    but leaving shape/dtype constant does not require recompilation.
    """

    from_node: Int[Array, " n_branches"]
    """Stores the from node of each branch, i.e. the id of the bus that the from end of the branch
    is connected to. Can hold values between [0, n_bus]"""

    to_node: Int[Array, " n_branches"]
    """Stores the to node of each branch, i.e. the id of the bus that the to end of the branch is
    connected to. Can hold values between [0, n_bus]"""

    ptdf: Float[Array, " n_branches n_bus"]
    """The base ptdf matrix as obtained by the preprocessing script, already extended as to suit
    the bsdf computation. In the numpy code, this is called ptdf_ext."""

    generators_per_sub: Int[Array, " n_sub_relevant"]
    """The number of generators connected to each relevant substation. In the numpy code, this is
    called looper.c_g
    """

    branch_limits: BranchLimits
    """The branch limits for the network. This is a dataclass that holds the different types of
    branch limits. The static_information loading and saving routines take care of BranchLimits
    too"""

    tot_stat: Int[Array, " n_sub_relevant max_branch_per_sub"]
    """The indices of branches that are connected to each relevant substation. This includes
    branches that are leaving and arriving in the station. We pad this with int_max to
    max_branch_per_sub In the numpy code, this is called looper.tot_stat and is not padded, hence of
    varying length.
    """

    from_stat_bool: Bool[Array, " n_sub_relevant max_branch_per_sub"]
    """Whether the branch in tot_stat is leaving the substation. It is padded with False to
    max_branch_per_sub"""

    susceptance: Float[Array, " n_branches"]
    """The susceptance of each branch"""

    relevant_injections: Float[Array, " n_timesteps n_sub_relevant max_inj_per_sub"]
    """The individual injections at the relevant subs, representing individual loads and generators that can be reassigned
    independently. The injections are padded to max_inj_per_sub with zeros, i.e. reassigning out of range will do nothing."""

    nodal_injections: Float[Array, " n_timesteps n_bus"]
    """The default nodal injections in the unsplit configuration. This is the sum of all
    consumptions and productions for each bus. The solver will adjust this based on the injection
    combination. There is a different vector for each timestep."""

    branches_to_fail: Int[Array, " n_failures"]
    """The branches that should be failed in the network for the N-1 analysis. This is a list of
    indices into the branches array"""

    action_set: ActionSet
    """An action set to be used in the solver. This holds the possible configurations for each
    substation. Topology actions that are passed into the solver index into this action set."""

    multi_outage_branches: list[Int[Array, " n_multi_outages n_branches_failed"]]
    """A multi-outage consists of a set of branches that are failed simultaneously and a set of
    nodes (multi_outage_nodes) to which all injections are zeroed. The last dimension of each array
    represents the set of branches, the first dimension is a collection of multi-outages. This
    supports padding, hence if you want to group multi-outages with varying numbers of branches, you
    can pad the arrays with invalid branch indices, e.g. int_max. Trade carefully, as this will
    solve a system of linear equations the size of n_branches_failed, irrespective of whether
    padding was applied or not. Hence, if possible, use a new group for each number of branches"""

    multi_outage_nodes: list[Int[Array, " n_multi_outages n_nodes_failed"]]
    """The nodes that correspond to the failures in multi_outage_branches. The length of the list
    and the first dimension of each list entry is the same as multi_outage_branches. The second can
    be different. This supports padding, hence if a different number of nodes needs to be zeroed for
    a branch outage, the array can be padded with invalid node indices, e.g. int_max."""

    disconnectable_branches: Int[Array, " n_disconnectable_branches"]
    """The branches that can be disconnected as a remedial action. This is a list of indices into
    the branches array"""

    nonrel_injection_outage_deltap: Float[Array, " n_timesteps n_nonrel_inj_failures"]
    """For every injection outage at a non-relevant sub, the delta in nodal injection that is to be expected from this
    outage."""

    nonrel_injection_outage_node: Int[Array, " n_nonrel_inj_failures"]
    """On which node the non-relevant injection outage is applied for. This should index into n_nodes"""

    relevant_injection_outage_sub: Int[Array, " n_rel_inj_failures"]
    """If a relevant injection is part of an injection outage, this is the index into the relevant
    substation that this injection is at"""

    relevant_injection_outage_idx: Int[Array, " n_rel_inj_failures"]
    """If a relevant injection is part of an injection outage, this is the index into the injections at that relevant
    substation. Meaning relevant_injections[relevant_injection_outage_sub, relevant_injection_outage_idx] is the
    injection that is failing.."""

    unsplit_flow: Float[Array, " n_timesteps n_branches"]
    """The flow in the network before any bus splits. This can either be the DC loadflow, the AC loadflow or
    any mixture of the two."""

    branches_monitored: Int[Array, " n_branches_monitored"]
    """The branches that we want to get loadflow results for. In the numpy code this is called
    sel_mon"""

    n2_baseline_analysis: Optional[N2BaselineAnalysis]
    """If provided, the results of the N-2 baseline analysis to compare against. If this is given,
    this automatically enables the N-2 analysis feature in the solver, meaning a N-2 analysis will
    be run on all split substations branches."""

    non_rel_bb_outage_data: Optional[NonRelBBOutageData]
    """
    The dataclass NonRelBBOutageData contains the data for the non-relevant busbar outages.
    Note: The data for relevant busbar outages is stored in the action set.
    """

    bb_outage_baseline_analysis: Optional[BBOutageBaselineAnalysis]
    """The results of the busbar outage analysis for unsplit grid is stored in this dataclass.
    This is calculated if a comparision of bb_outage analysis has to be made between the unsplit
    and split grid."""

    controllable_pst_indices: Int[Array, " n_controllable_pst"]
    """An index over controllable PSTs indexing into all nodes. The injections of these nodes are
    actually shift angles and can be varied between shift_min and shift_max."""

    shift_degree_min: Float[Array, " n_controllable_pst"]
    """The minimum shift angle for each controllable PST"""

    shift_degree_max: Float[Array, " n_controllable_pst"]
    """The maximum shift angle for each controllable PST"""

    pst_n_taps: Int[Array, " n_controllable_pst"]
    """The number of discrete taps for each controllable PST"""

    pst_tap_values: Float[Array, " n_controllable_pst max_n_tap_positions"]
    """Discrete individual taps of controllable PSTs. The array is zero-padded to the maximum number of
    pst_n_taps."""

    @property
    def n_timesteps(self) -> int:
        """The number of timesteps in the data"""
        return self.nodal_injections.shape[0]

    @property
    def n_branches(self) -> int:
        """The number of branches in the network"""
        return self.ptdf.shape[0]

    @property
    def n_disconnectable_branches(self) -> int:
        """The number of disconnectable branches in the branch set"""
        return self.disconnectable_branches.shape[0]

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the network"""
        return self.ptdf.shape[1]

    @property
    def n_sub_relevant(self) -> int:
        """The number of relevant substations"""
        return len(self.generators_per_sub)

    @property
    def n_outages(self) -> int:
        """The number of (simple) N-1 outages in the network"""
        return len(self.branches_to_fail)

    @property
    def n_multi_outages(self) -> int:
        """The number of multi-outages"""
        return sum(len(x) for x in self.multi_outage_branches)

    @property
    def n_nonrel_inj_failures(self) -> int:
        """The number of non-relevant injection outages"""
        return len(self.nonrel_injection_outage_node)

    @property
    def n_rel_inj_failures(self) -> int:
        """The number of relevant injection outages"""
        return len(self.relevant_injection_outage_sub)

    @property
    def n_inj_failures(self) -> int:
        """The number of injection outages"""
        return self.n_nonrel_inj_failures + self.n_rel_inj_failures

    @property
    def n_bb_outages(self) -> int:
        """The number of busbar outages"""
        n_bb_outages = 0
        if self.non_rel_bb_outage_data is not None:
            n_bb_outages += self.non_rel_bb_outage_data.nodal_indices.shape[0]

        if self.action_set.rel_bb_outage_data is not None:
            max_bbs_per_sub = self.action_set.rel_bb_outage_data.nodal_indices.shape[1]
            max_n_rel_bbs = self.n_sub_relevant * max_bbs_per_sub
            n_bb_outages += max_n_rel_bbs
        return n_bb_outages

    @property
    def n_nminus1_cases(self) -> int:
        """The number of N-1 cases to consider

        bb_outage_baseline_analysis is None if solver_config.bb_outage_as_nminus1 is True.
        Therefore, we count the actual number of busbar outages in this case. However, if
        bb_outage_baseline_analysis is not None, we don't count the busbar outages as N-1 cases.
        """
        n_bb_outages = self.n_bb_outages if self.bb_outage_baseline_analysis is None else 0
        return self.n_outages + self.n_multi_outages + self.n_inj_failures + n_bb_outages

    @property
    def n_branches_monitored(self) -> int:
        """The number of monitored branches"""
        return len(self.branches_monitored)

    @property
    def n_controllable_pst(self) -> int:
        """The number of controllable PSTs"""
        return len(self.controllable_pst_indices)

    @property
    def n_actions(self) -> int:
        """Number of actions in the action set"""
        return len(self.action_set)

    @property
    def max_branch_per_sub(self) -> int:
        """Maximum branch degree of any relevant substation"""
        return self.action_set.branch_actions.shape[1]

    @property
    def max_inj_per_sub(self) -> int:
        """Maximum injection degree of any relevant substation"""
        return self.action_set.inj_actions.shape[1]

    @property
    def max_n_tap_positions(self) -> int:
        """Maximum number of discrete tap positions of any controllable PST"""
        return self.pst_tap_values.shape[1]


@pytree_dataclass
class StaticInformation:
    """Holds the static information for the solver.

    The static information comprises a set of data that is not changing during the computation
    (SolverConfig) and a set of data that can, but probably won't change during the computation
    (DynamicInformation).
    """

    dynamic_information: DynamicInformation
    """The dynamic information that is used for the computation"""

    solver_config: Static[SolverConfig]
    """The static configuration that is used for the computation"""

    def __hash__(self) -> int:
        """Get the id as the hash for the static information.

        We expect only one instance of this class to be used at a time so id(self) is a good hash,
        no need to access the content.
        """
        return id(self)

    def __eq__(self, other: object) -> Bool:
        """Test for equality.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the self is other, False otherwise.
        """
        return self is other

    @property
    def n_timesteps(self) -> int:
        """The number of timesteps in the data"""
        return self.dynamic_information.n_timesteps

    @property
    def n_branches(self) -> int:
        """The number of branches in the network"""
        return self.dynamic_information.n_branches

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the network"""
        return self.dynamic_information.n_nodes

    @property
    def n_branches_monitored(self) -> int:
        """The number of monitored branches"""
        return self.dynamic_information.n_branches_monitored

    @property
    def n_sub_relevant(self) -> int:
        """The number of relevant substations"""
        return self.solver_config.n_sub_relevant

    @property
    def n_outages(self) -> int:
        """The number of (simple) N-1 outages in the network"""
        return self.dynamic_information.n_outages

    @property
    def n_multi_outages(self) -> int:
        """The number of multi-outages"""
        return self.dynamic_information.n_multi_outages

    @property
    def n_inj_failures(self) -> int:
        """The number of injection outages"""
        return self.dynamic_information.n_inj_failures

    @property
    def n_nminus1_cases(self) -> int:
        """The number of N-1 cases to consider"""
        return self.n_outages + self.n_multi_outages + self.n_inj_failures

    @property
    def n_actions(self) -> int:
        """Number of actions in the action set"""
        return self.dynamic_information.n_actions


@pytree_dataclass
class BranchLimits:
    """A dataclass holding the different branch limits.

    As there are many slightly different types of limits, we introduce a dataclass to encapsulate them.
    """

    max_mw_flow: Float[Array, " n_branches_monitored"]
    """The maximum flow in MW for each branch as stored in the specs of the line"""

    max_mw_flow_n_1: Optional[Float[Array, " n_branches_monitored"]] = None
    """Optionally, a different flow capacity in the N-1 case. If this is not None, it will override
    max_mw_flow for N-1 computations. Otherwise, max_mw_flow will be used for both N-1 and N-0."""

    overload_weight: Optional[Float[Array, " n_branches_monitored"]] = None
    """Optionally, a different weight for each branch in the overload energy computation. If this is
    not None, it will multiply the overload energy by the weight for each branch. Otherwise, a
    constant weight of 1 will be used."""

    max_mw_flow_limited: Optional[Float[Array, " n_branches_monitored"]] = None
    """Optionally, a lower flow capacity to artificially constrain branches below their physical
    limits. This is useful to avoid bringing branches too close to critical and can be computed
    through aggregate_results.apply_double_limit"""

    max_mw_flow_n_1_limited: Optional[Float[Array, " n_branches_monitored"]] = None
    """Optionally, a lower flow capacity in the N-1 case to artificially constrain branches below
    their physical limits. This is useful to avoid bringing branches too close to critical and can
    be computed through aggregate_results.apply_double_limit"""

    n0_n1_max_diff: Optional[Float[Array, " n_branches_monitored"]] = None
    """Optionally, a maximum difference between the N-0 and N-1 flows in MW. 0 means the N-1 flows
    shall be exactly the N-0 flows or lower, any value larger than 0 means that the relative
    difference shall not exceed this value - 20 means that the N-1 flows can be at most 20 MW higher
    than the N-0 flows. By applying this limit to transformers, the optimizer is penalized for
    creating scenarios that pump a lot of energy into the distribution grid upon simple failures."""

    coupler_limits: Optional[Float[Array, " n_subs_relevant"]] = None
    """Optionally, a limit on the flow on the couplers for each relevant substation."""


@pytree_dataclass
class ActionSet:
    """Holds the information for an ActionSet.

    Instead of allowing all reassignments, it is required to constrain the optimizer
    to choose from a set of actions per substation that have been pre-computed in the preprocessing
    phase. This dataclass holds the information for these actions. Each action is a substation-local assignment of all
    branches and injections, completely defining the electrical switching configuration of the substation. The physical
    configuration and the switching actions to reach it are not stored in the jax part as they are not needed at runtime.
    """

    branch_actions: Bool[Array, " n_actions max_branch_per_sub"]
    """A padded out boolean array with the branch assignment for each action in the action set. Each action is a combination
    of a branch and injection assignment. As each action corresponds to a single substation, this holds up to
    max_branch_per_sub booleans and is padded with False if the sub has fewer branches. This holds a True if a branch is on
    bus B and False if it is on bus A. The actions for all substations are concatenated to a large list of n_inj_action
    assignments."""

    inj_actions: Bool[Array, " n_actions max_inj_per_sub"]
    """A padded out boolean array with the injection assignment for each action in the action set. This holds
    a True if an injection is on bus B and False if it is on bus A. The actions for all substations are
    concatenated to a large list of n_inj_action assignments. If a sub has fewer than max_inj_per_sub
    injections, the remaining entries are padded with False.
    Potentially this can be overwritten by an injection combination passed in to the inj bruteforce mode.
    """

    n_actions_per_sub: Int[Array, " n_sub_relevant"]
    """The number of branch actions for each substation. The actions in branch_actions are
    concatenated for all substations, meaning n_actions == sum(n_actions_per_sub)"""

    substation_correspondence: Int[Array, " n_actions"]
    """Holds for each action which substation it corresponds to. This improves search
    performance in the array, though retrieval via n_actions_per_sub is also possible as the
    actions are guaranteed to be in order of substation"""

    unsplit_action_mask: Bool[Array, " n_actions"]
    """A mask of unsplit actions, i.e. columns in the actions array that are all False.
    This is used to avoid sampling the unsplit action in the random_topology generator. This is True where
    the unsplit action is stored."""

    reassignment_distance: Int[Array, " n_actions"]
    """The number of reassignments that were necessary to get from the starting topology in that station to the topology
    described by the action in the action set."""

    rel_bb_outage_data: Optional[RelBBOutageData] = None
    """
    Busbar outage data corresponding to each action in the action set. Each action in the action set determine
    which branches and injections are on which busbar. This in turn determines which assets are outaged if
    a busbar is outaged. This is optional because we may chose to skip calculating busbar outages in some cases.
    """

    def __eq__(self, other: object) -> bool:
        """Equality is defined by array_equals checks

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, ActionSet):
            return False
        return (
            jnp.array_equal(self.branch_actions, other.branch_actions)
            and jnp.array_equal(self.n_actions_per_sub, other.n_actions_per_sub)
            and jnp.array_equal(self.substation_correspondence, other.substation_correspondence)
            and jnp.array_equal(self.unsplit_action_mask, other.unsplit_action_mask)
            and jnp.array_equal(self.reassignment_distance, other.reassignment_distance)
            and jnp.array_equal(self.inj_actions, other.inj_actions)
            and self.rel_bb_outage_data == other.rel_bb_outage_data
        )

    def __getitem__(self, index: Union[Int[Array, " n_indices"], Bool[Array, " n_actions"]]) -> ActionSet:
        """Index a branch action set.

        Parameters
        ----------
        index : Union[Int[Array, " n_indices"], Bool[Array, " n_actions"]]
            The indices to index the branch action set with. Supported are boolean indices and
            integer indices.

        Returns
        -------
        BranchActionSet
            The indexed branch action set.
        """
        assert index.ndim == 1, "Index must be 1D"
        sub_correspondence = self.substation_correspondence[index]
        n_actions_per_sub = jnp.bincount(
            sub_correspondence, minlength=self.n_actions_per_sub.shape[0], length=self.n_actions_per_sub.shape[0]
        )

        return ActionSet(
            branch_actions=self.branch_actions[index],
            substation_correspondence=sub_correspondence,
            n_actions_per_sub=n_actions_per_sub,
            unsplit_action_mask=self.unsplit_action_mask[index],
            reassignment_distance=self.reassignment_distance[index],
            inj_actions=self.inj_actions[index],
            rel_bb_outage_data=self.rel_bb_outage_data[index] if self.rel_bb_outage_data is not None else None,
        )

    def __len__(self) -> int:
        """Get the number of actions"""
        return self.branch_actions.shape[0]


@pytree_dataclass
class ActionIndexComputations:
    """Holds branch+injection topology computations in the form of indices into the action set.

    Each action consist of a set of splits, where each split is determined by an index into the ActionSet.
    This is the main format of input for branch topologies in the solver, TopoVectBranchComputations are no longer supported.
    For the injection bruteforce mode, the injection topology stored in the action set will be overwritten by the injection
    combinations that were passed in.
    It does not contain a sub_id because the sub_id can be determined based on the action - each action in the action set
    belongs to exactly one substation, so if we have an action, we can determine the substation by looking up
    substation_correspondence
    """

    action: Int[Array, " ... n_topologies n_splits"]
    """An index into the action set, determining the reconfiguration and the substation that was split. Each split
    must refer to a different substation, otherwise the action is invalid. If an action wants to represent less than
    n_split splits, the remaining slots are to be filled with invalid indices, e.g. int_max. Consequently, passing only
    int_max will result in the unsplit configuration."""

    pad_mask: Bool[Array, "... n_topologies"]
    """In case the computations are padded to a common batch size, mark which computations are
    valid (True) and which are just padding (False)."""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> ActionIndexComputations:
        """Access the first batch dimension of the topology computations"""
        return ActionIndexComputations(
            action=self.action[key],
            pad_mask=self.pad_mask[key],
        )

    def __len__(self) -> int:
        """Get the number of topologies"""
        return self.action.shape[0]


@pytree_dataclass
class TopoVectBranchComputations:
    """Stores branch topology computations in topo-vect form.

    This is a padded topo vect with an entry per split, showing the busbar assignment
    for each branch in the station. The station is identified via the sub_ids array.
    """

    topologies: Bool[Array, " ... n_topologies n_splits max_branch_per_sub"]
    """An array of topologies that we want to have computed. The topologies are stored as topo-vects
    for the substation that's being reconfigured, where each entry is a boolean value indicating
    whether the line is on bus 0 (False) or on bus 1 (True). Note that loads and generators are not
    included here, as these are implicitely optimized by the nodal injections module. These
    topo-vects always have length max_branch_per_sub, if there are fewer elements in the substation,
    the rest is ignored.

    Each last-dimension slice in this array is a topo-vect for the current substation, but there
    might be up to n_splits substations affected by a single topology configuration and
    there are n_topologies that should be computed, hence we have 2 batch dimensions.

    n_splits might be less than what's statically relevant, as some substations might never
    have seen a topology change.
    """

    sub_ids: Int[Array, " ... n_topologies n_splits"]
    """The substation id that is affected by the currently computed bus split. This indexes into
    relevant substations only, as the others can't be split.
    """

    pad_mask: Bool[Array, "... n_topologies"]
    """In case the computations are padded to a common batch size, mark which computations are
    valid (True) and which are just padding (False)."""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> TopoVectBranchComputations:
        """Access the first batch dimension of the topology computations"""
        return TopoVectBranchComputations(
            topologies=self.topologies[key],
            sub_ids=self.sub_ids[key],
            pad_mask=self.pad_mask[key],
        )

    def __len__(self) -> int:
        """Get the number of topologies"""
        return self.topologies.shape[0]


@pytree_dataclass
class InjectionComputations:
    """Injection combinations can be either used from the action set or overwritten with custom injection combinations.

    If this is passed, that means the the injection actions from the set shall be overwritten.
    """

    corresponding_topology: Int[Array, " ... n_injection_combinations"]
    """An index into either ActionIndexComputations, telling for which branch topology this injection combination
    is computed. The injection action associated with the action set will be overwritten by the injection combination
    in this topo-vect. Entries are sorted by corresponding_topology to make
    retrieval more performant. n_injection_combinations can be more than the number of branch actions, meaning there can be
    one or more injection combinations per branch action. If a branch action is never mentioned, this entails undefined
    behaviour for that branch action. The number of injection combinations per branch action is not fixed and can vary."""

    injection_topology: Bool[Array, " ... n_injection_combinations n_splits max_inj_per_sub"]
    """The injection topology for each substation. This is a topo vect which holds whether an
    injection is on busbar A (False) or busbar B (True). The last dimension is padded to max_inj_per_sub.
    It holds an entry for every substation that is split in the corresponding branch topology, as injection
    reassignments on unsplit stations do not make sense. The substations ids are the same as in the
    corresponding branch topology."""

    pad_mask: Bool[Array, "... n_injection_combinations"]
    """In case the computations are padded to a common batch size, mark which computations are
    valid (True) and which are just padding (False)."""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> InjectionComputations:
        """Access the first batch dimension of the injection computations"""
        return InjectionComputations(
            corresponding_topology=self.corresponding_topology[key],
            injection_topology=self.injection_topology[key],
            pad_mask=self.pad_mask[key],
        )


@pytree_dataclass
class SparseNMinus0:
    """A dataclass for tracking maximum N-0 results in a sparse fashion."""

    pf_n_0_max: Float[Array, " ... n_timesteps number_most_affected"]
    """The maximum rho values encountered"""

    hist_mon: Int[Array, " ... n_timesteps number_most_affected"]
    """The index of the overloaded branch, indexing into only monitored lines"""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> SparseNMinus0:
        """Access the first batch dimension of the topology computations"""
        return SparseNMinus0(
            pf_n_0_max=self.pf_n_0_max[key],
            hist_mon=self.hist_mon[key],
        )


@pytree_dataclass
class SparseNMinus1:
    """A dataclass for tracking maximum N-1 results in a sparse fashion.

    Very similar to a COO sparse matrix, this tracks the values (pf_n_1_max) and the row and column
    indices into the N-1 matrix, where the row (failure) index is hist_out and the column
    (monitored branch) index is hist_mon.
    """

    # TODO maybe just use jax.experimental.sparse.BCOO?

    pf_n_1_max: Float[Array, " ... n_timesteps number_most_affected"]
    """The maximum rho values encountered"""

    hist_mon: Int[Array, " ... n_timesteps number_most_affected"]
    """The index of the overloaded branch"""

    hist_out: Int[Array, " ... n_timesteps number_most_affected"]
    """The index of the outaged branch that caused the overload, i.e. the N-1 case"""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> SparseNMinus1:
        """Access the first batch dimension of the topology computations"""
        return SparseNMinus1(
            pf_n_1_max=self.pf_n_1_max[key],
            hist_mon=self.hist_mon[key],
            hist_out=self.hist_out[key],
        )


@pytree_dataclass
class BSDFResults:
    """A dataclass encapsulating the data that is updated within the bsdf computation.

    This holds only the changing parts, information from the StaticInformation dataclass and the to-
    be computed topology is still needed to run the computation.
    """

    ptdf: Float[Array, " ... n_branches n_bus"]
    """The updated ptdf matrix with the bus splits applied.
    """

    from_node: Int[Array, " ... n_branches"]
    """The updated from/ nodes of the branches. After a bus split, branches that are on bus 1 will
    have their from node updated to the new bus id.
    """

    to_node: Int[Array, " ... n_branches"]
    """Similar to from_node but for the to_end"""

    success: Bool[Array, " ..."]
    """A boolean indicating whether the bsdf computation was successful. This is false if the
    computation failed due to a zero denominator, which can be caused by a split in the network
    that was missed during pre-processing"""

    bsdf: Float[Array, " ... n_splits n_branches"]
    """The BSDF vectors for each split, used in the cross-coupler flow updates."""


@pytree_dataclass
class DisconnectionResults:
    """A dataclass encapsulating the results of the disconnection computations.

    This holds only the changing parts, information from the StaticInformation dataclass and the to-
    be computed topology is still needed to run the computation.
    """

    ptdf: Float[Array, " ... n_branches n_bus"]
    """The updated ptdf matrix with the disconnections applied."""

    from_node: Int[Array, " ... n_branches"]
    """The updated from/ nodes of the branches. After a disconnection, branches that are disconnected
    will have their from node updated to the new bus id."""

    to_node: Int[Array, " ... n_branches"]
    """Similar to from_node but for the to_end"""

    success: Bool[Array, " ..."]
    """A boolean indicating whether the disconnection computation was successful."""

    modf: MODFMatrix
    """The MODF matrices for the disconnections to apply the loadflow update"""


@pytree_dataclass
class SparseSolverOutput:
    """Store the results of the loadflow computations on a per-topology basis."""

    n_0_results: SparseNMinus0
    n_1_results: SparseNMinus1
    best_inj_combi: Int[Array, " ... n_sub_relevant"]
    success: Bool[Array, " ..."]


@pytree_dataclass
class MODFMatrix:
    """A MODF matrix for a single batch of multi-outages."""

    modf: Float[Array, " ... n_branches n_outaged_branches"]
    """The impact of the multi-outages on all branches in the network."""

    branch_indices: Int[Array, " ... n_outaged_branches"]
    """Which branches were outaged in the multi-outage"""

    def __getitem__(self, key: Union[slice, int, jnp.ndarray]) -> MODFMatrix:
        """Get a slice of the MODF matrix.

        Parameters
        ----------
        key : Union[slice, int, jnp.ndarray]
            The key to index the MODF matrix with. This can be a slice, an integer or a boolean
            array.

        Returns
        -------
        MODFMatrix
            The indexed MODF matrix.
        """
        return MODFMatrix(
            modf=self.modf[key],
            branch_indices=self.branch_indices[key],
        )


@pytree_dataclass
class TopologyResults:
    """Stores the results of the BSDF, LODF and static flow computations.

    All computations that happen over topology batches.
    """

    ptdf: Float[Array, " ... n_branches n_bus"]
    """The ptdf matrices after applying every topology"""

    from_node: Int[Array, " ... n_branches"]
    """The from nodes after applying every topology"""

    to_node: Int[Array, " ... n_branches"]
    """The to nodes after applying every topology"""

    lodf: Float[Array, " ... n_failures n_branches_monitored"]
    """The LODF matrices for every topology and every failure"""

    success: Bool[Array, " ..."]
    """Whether all computations for a topology were successful (i.e. all bsdfs, all outages, all
    LODFs and all MODFs."""

    outage_modf: list[MODFMatrix]
    """The MODF matrices for the multi-outages"""

    failure_cases_to_zero: Optional[Bool[Array, " ... n_failures"]]
    """If disconnections are applied, potentially some failure cases need to be zeroed out after the N-1
    computation."""

    bsdf: Float[Array, " ... n_splits n_branches"]
    """If cross-busbar flows are to be computed, this holds the BSDF vectors required to do that"""

    disconnection_modf: Optional[MODFMatrix]
    """If disconnections are applied, this holds the MODF matrices for the disconnections."""


@pytree_dataclass
class N2BaselineAnalysis:
    """The output of the N-2 baseline analysis, used to compare the split n-2 analysis against."""

    l1_branches: Int[Array, " n_l1_outages"]
    """All branches that have been analysed as L1 outages"""

    tot_stat_blacklisted: Int[Array, " n_sub_relevant max_branch_per_sub"]
    """Branches that could not be outaged because they split the grid already in the unsplit
    analysis don't need to be considered in the split analysis. This will include all stub lines to
    relevant substations. These L1 cases are blacklisted. This is a copy of
    dynamic_information.tot_stat where all the blacklisted cases are set to int_max."""

    n_2_overloads: Float[Array, " n_l1_outages"]
    """The overload energy for each L1 outage"""

    n_2_success_count: Int[Array, " n_l1_outages"]
    """How many N-2 cases were successfully computed for each L1 outage"""

    more_splits_penalty: Float[Array, ""]
    """If a topology causes more non-converging DC loadflows due to splits in the grid than the
    baseline, a penalty is added to the metrics. I.e. if success_count is lower in the split
    analysis than in the unsplit, this split penalty is added for every point difference in the
    success_counts."""

    max_mw_flow: Float[Array, " n_branches_monitored"]
    """The branch limits used to compute the N-2 overload energy. This is likely a copy of
    branch_limits.max_mw_flow, however it is less bug-prone to replicate it so the unsplit and split
    analysis will always use the same limits."""

    overload_weight: Optional[Float[Array, " n_branches_monitored"]]
    """The overload weights used to compute the N-2 overload energy. This is likely a copy of
    branch_limits.overload_weight, however it is less bug-prone to replicate it so the unsplit and
    split analysis will always use the same weights."""


@pytree_dataclass
class SolverLoadflowResults:
    """The loadflow results without any preprocessing in matrix form.

    This is the primary input for the aggregate_output and aggregate_metrics functions,
    which are supposed to do some first processing of the results to reduce storage requirements.
    """

    n_0_matrix: Float[Array, " ... n_timesteps n_branches_monitored"]
    """The N-0 p values for all monitored branches"""

    n_1_matrix: Float[Array, " ... n_timesteps n_failures n_branches_monitored"]
    """The N-1 p values for all monitored branches where the failures are ordered by normal N-1
    single branch contingencies, then multi-outages, then injection outages, then busbar outages."""

    cross_coupler_flows: Float[Array, " ... n_splits n_timesteps"]
    """The cross-coupler flows for each coupler/split and timestep in MW."""

    branch_action_index: Int[Array, " ... n_splits"]
    """The index into the branch action set, which determined the branch topology in binary format. Corresponds
    to the input passed into the solver."""

    branch_topology: Bool[Array, " ... n_splits max_branch_per_sub"]
    """The branch topology that was evaluated for these loadflow results. Corresponds to the input
    passed into the solver."""

    sub_ids: Int[Array, " ... n_splits"]
    """The substation ids that are affected by splits. Will be padded with int_max if the number of
    actual splits is less than the maximum number of splits."""

    injection_topology: Bool[Array, " ... n_splits max_inj_per_sub"]
    """The injection combination that was evaluated for these loadflow results. Returns the array
    over the split substations."""

    n_2_penalty: Optional[Float[Array, " ... "]]
    """The penalty from n_2_analysis, if an N-2 analysis was performed. This is a single scalar per
    topology, as aggregation happens inside the N-2 routine to save memory"""

    disconnections: Optional[Int[Array, " ... n_disconnections"]]
    """If disconnections are active, this passes in the disconnected branches. Required to compute the
    actual number of disconnections as some of the slots might be filled with int_max."""

    bb_outage_penalty: Optional[Float[Array, " ... "]] = None
    """The final computed penalty from busbar outages. This is a single scalar per topology, as aggregation
    happens inside the busbar outage routine to save memory. If no busbar outages were computed
    this will be None."""

    bb_outage_splits: Optional[Int[Array, " ... "]] = None
    """The number of cases where grid splits happen due to busbar outage."""

    bb_outage_overload: Optional[Float[Array, " ... "]] = None
    """The overload energy caused due to busbar outages"""

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> SolverLoadflowResults:
        """Access the first batch dimension of the loadflow matrices"""
        assert self.n_0_matrix.ndim >= 3, "Only works if a batch dimension is present"
        return SolverLoadflowResults(
            n_0_matrix=self.n_0_matrix[key],
            n_1_matrix=self.n_1_matrix[key],
            cross_coupler_flows=self.cross_coupler_flows[key],
            branch_action_index=self.branch_action_index[key],
            branch_topology=self.branch_topology[key],
            sub_ids=self.sub_ids[key],
            injection_topology=self.injection_topology[key],
            n_2_penalty=(self.n_2_penalty[key] if self.n_2_penalty is not None else None),
            bb_outage_penalty=(self.bb_outage_penalty[key] if self.bb_outage_penalty is not None else None),
            bb_outage_splits=(self.bb_outage_splits[key] if self.bb_outage_splits is not None else None),
            bb_outage_overload=(self.bb_outage_overload[key] if self.bb_outage_overload is not None else None),
            disconnections=(self.disconnections[key] if self.disconnections is not None else None),
        )


class AggregateMetricProtocol(Protocol):
    """A protocol for the aggregate metric function.

    Is used to compute the metric from the results of the solver.
    This is used in the bruteforce module to select the best injection combination
    """

    def __call__(
        self,
        loadflows: SolverLoadflowResults,
        output_metrics: Optional[PyTree],
    ) -> Float[Array, " "]:
        """Compute the aggregate metric from the N-0 and N-1 results.

        The output metrics are an optional input that will be used if the solver doesn't run in
        metrics-first mode, so you can use the output of the AggregateOutputProtocol here.

        Note that this does not take a batch dimension and is expected to be vmappable across the
        batch dimension.

        Parameters
        ----------
        loadflows: LoadflowMatrices
            The loadflow results from the solver, for a single topology and injection combination
            (i.e. without any leading batch dimensions, the ... in the LoadflowMatrices don't exist
        output_metrics: Optional[PyTree]
            The output from AggregateOutputFn, if the solver ran in injection bruteforce mode
            with metrics-first=False. Otherwise None.

        Returns
        -------
        Float[Array, " "]
            The aggregate metric
        """

    def __hash__(self) -> int:
        """Get the hash of the function.

        A good choice is to return the hash of the static_information object, as the function
        should be static and not change during the computation.

        This is needed because jax recompiles based on the hash and equality check of the inputs, so
        if the hash/__eq__ result changes, jax will recompile even if it is the same function.
        """

    def __eq__(self, other: object) -> bool:
        """Check if the functions are equal.

        This is needed because jax recompiles based on the hash and equality check of the inputs, so
        if the hash/__eq__ result changes, jax will recompile even if it is the same function.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        return hash(self) == hash(other)


class AggregateOutputProtocol(Protocol):
    """A protocol for the aggregate output function.

    Is used to compute the output data from the results of the solver.
    Output data can be anything the user desires, e.g. sparsified result matrices only
    holding the worst N-0/N-1 cases. The primary goal of the output function is
    to lower the storage requirement for the loadflow results, as a full N-1 matrix for every
    topology is too much to store long-term.
    """

    def __call__(
        self,
        loadflows: SolverLoadflowResults,
    ) -> PyTree:
        """Compute the output quantities of interest from the N-0 and N-1 results.

        Note that this does not take a batch dimension and is expected to be vmappable across the
        batch dimension.

        Parameters
        ----------
        loadflows: LoadflowMatrices
            The loadflow results from the solver, for a single topology (i.e. without any leading
            batch dimensions, the ... in the LoadflowMatrices don't exist here)

        Returns
        -------
        PyTree
            The output in whatever format you desire, however still needs to be jax-compatible.
        """

    def __hash__(self) -> int:
        """Get the hash of the function

        This is needed because jax recompiles based on the hash and equality check of the inputs, so
        if the hash/__eq__ result changes, jax will recompile even if it is the same function.
        """
        raise NotImplementedError("A hash function must be implemented for the aggregate output function.")

    def __eq__(self, other: object) -> bool:
        """Check if the functions are equal.

        This is needed because jax recompiles based on the hash and equality check of the inputs, so
        if the hash/__eq__ result changes, jax will recompile even if it is the same function.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        return hash(self) == hash(other)


@pytree_dataclass
class RelBBOutageData:
    """Holds the relevant busbar outage data."""

    branch_outage_set: Int[Array, " n_actions n_max_bb_to_outage_per_sub max_branches_per_sub"]
    """
    This corresponds to the branches that are outaged in the busbar-outage cases, represented as integers.
    These integers are essentially branch indices that index into branch_ids of NetworkData.
    n_actions = sum(num_combis_per_sub)
    If a sub has fewer than max_branch_per_sub branches, the remaining entries are padded with False.
    """
    deltap_set: Float[Array, " n_actions n_max_bb_to_outage_per_sub n_timesteps"]
    """
    The injection outages for each branch_action. The delta_p vector would depend on each action in the action_set.
    The branch actions branch actions determine the placement of stub branches (if any) and the placement of
    stub branches influence the delta_p vector. Additionally, injection_action determine the placement of the
    injections on different busbars.
    """
    nodal_indices: Int[Array, " n_actions n_max_bb_to_outage_per_sub"]
    """
    The nodal index of the outaged busbar for each combination of branch and injection action
    """
    articulation_node_mask: Bool[Array, " n_actions n_max_bb_to_outage_per_sub"]
    """
    A mask that indicates if a particular busbar can be outaged or not. The idea is to not outage any busbar
    that may result in a split in the station as this might lead to splitting the station twice. A True means
    that the busbar is an articulation node and can't be outaged.
    For ex, if bus_a: 1 - 2 - 3 ; bus_b: 4
    Here busbar 2 is an articulation node as if it is outaged, bus_a will be split into 1 and 3.
    """

    def __eq__(self, other: object) -> bool:
        """Equality is defined by array_equals checks

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, RelBBOutageData):
            return False
        return (
            jnp.array_equal(self.branch_outage_set, other.branch_outage_set)
            and jnp.array_equal(self.deltap_set, other.deltap_set)
            and jnp.array_equal(self.nodal_indices, other.nodal_indices)
            and jnp.array_equal(self.articulation_node_mask, other.articulation_node_mask)
        )

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> RelBBOutageData:
        """Access the first batch dimension of the RelBBOutageData"""
        return RelBBOutageData(
            branch_outage_set=self.branch_outage_set[key],
            deltap_set=self.deltap_set[key],
            nodal_indices=self.nodal_indices[key],
            articulation_node_mask=self.articulation_node_mask[key],
        )


@pytree_dataclass
class NonRelBBOutageData:
    """Holds the non-relevant busbar outage data."""

    branch_outages: Int[Array, " n_non_rel_bb_outages max_n_branches_failed"]
    """The indices of the branches that are outaged in the busbar-outage cases, represented as integers.
    Note that, this is a padded array with int_max.
    """

    nodal_indices: Int[Array, " n_non_rel_bb_outages"]
    """The nodal_index of the busbars to be outaged. The length of the list
    is the same as branch_outages. """

    deltap: Float[Array, " n_non_rel_bb_outages n_timesteps"]
    """For every busbar outage, the delta in power that has to be subtracted from
    the nodal injection."""


def int_max() -> int:
    """Get the int max depending whether jax runs in 64 or 32 bit mode"""
    return int(jnp.iinfo(jnp.int64).max) if jax.config.read("jax_enable_x64") else int(jnp.iinfo(jnp.int32).max)


@pytree_dataclass
class BBOutageBaselineAnalysis:
    """The output of the busbar outage analysis for unsplit grid.

    This is used as a baseline to compare against the split busbar outage analysis. This baseline data is used when we
    calculate the difference between the split and unsplit busbar outage analysis.
    """

    overload: Float[Array, " "]
    """ The worst overload energy due to busbar_outages."""

    success_count: Int[Array, " "]
    """How many loadflow computations for the busbar outages were successful.
    """

    more_splits_penalty: Float[Array, " "]
    """If a topology causes more non-converging DC loadflows due to splits in the grid than the
    baseline, a penalty is added to the metrics. I.e. if success_count is lower in the split
    analysis than in the unsplit, this split penalty is added for every point difference in the
    success_counts."""

    max_mw_flow: Float[Array, " n_branches_monitored"]
    """The branch limits used to compute the bb_outage overload energy. This is likely a copy of
    branch_limits.max_mw_flow, however it is less bug-prone to replicate it so the unsplit and split
    analysis will always use the same limits."""

    overload_weight: Optional[Float[Array, " n_branches_monitored"]]
    """The overload weights used to compute the bb_outage overload energy. This is likely a copy of
    branch_limits.overload_weight, however it is less bug-prone to replicate it so the unsplit and
    split analysis will always use the same weights."""
