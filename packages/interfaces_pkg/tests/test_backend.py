# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from beartype.typing import Sequence, Union
from jaxtyping import Bool, Float, Int
from toop_engine_interfaces.backend import BackendInterface


class TestBackend(BackendInterface):
    """An AI generated implentation so we can test the functions that infer from others"""

    def get_slack(self) -> int:
        return 0

    def get_max_mw_flows(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        return np.array([[100, 200], [100, 200]])

    def get_susceptances(self) -> Float[np.ndarray, " n_branch"]:
        return np.array([0.1, 0.2])

    def get_from_nodes(self) -> Int[np.ndarray, " n_branch"]:
        return np.array([0, 1])

    def get_to_nodes(self) -> Int[np.ndarray, " n_branch"]:
        return np.array([1, 2])

    def get_shift_angles(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        return np.array([[0, 0], [0, 0]])

    def get_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        return np.array([True, False])

    def get_relevant_node_mask(self) -> Bool[np.ndarray, " n_node"]:
        return np.array([True, False, True])

    def get_monitored_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        return np.array([True, True])

    def get_branches_in_maintenance(self) -> Bool[np.ndarray, " n_timestep n_branch"]:
        return np.array([[False, False], [False, False]])

    def get_disconnectable_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        return np.array([True, False])

    def get_outaged_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        return np.array([False, True])

    def get_outaged_injection_mask(self) -> Bool[np.ndarray, " n_injection"]:
        return np.array([False, True])

    def get_multi_outage_branches(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_branch"]:
        return np.array([[False, True], [True, False]])

    def get_multi_outage_nodes(self) -> Bool[np.ndarray, " n_multi_outages n_node"]:
        return np.array([[True, False, True], [False, True, False]])

    def get_injection_nodes(self) -> Int[np.ndarray, " n_injection"]:
        return np.array([0, 1])

    def get_mw_injections(self) -> Float[np.ndarray, " n_timestep n_injection"]:
        return np.array([[50, 60], [50, 60]])

    def get_base_mva(self) -> float:
        return 100.0

    def get_node_ids(self) -> Union[Sequence[str], Sequence[int]]:
        return ["node1", "node2", "node3"]

    def get_branch_ids(self) -> Union[Sequence[str], Sequence[int]]:
        return ["branch1", "branch2"]

    def get_injection_ids(self) -> Union[Sequence[str], Sequence[int]]:
        return ["inj1", "inj2"]

    def get_multi_outage_ids(self) -> Union[Sequence[str], Sequence[int]]:
        return ["outage1", "outage2"]

    def get_node_names(self) -> Sequence[str]:
        return ["Node 1", "Node 2", "Node 3"]

    def get_branch_names(self) -> Sequence[str]:
        return ["Branch 1", "Branch 2"]

    def get_injection_names(self) -> Sequence[str]:
        return ["Injection 1", "Injection 2"]

    def get_multi_outage_names(self) -> Sequence[str]:
        return ["Multi Outage 1", "Multi Outage 2"]

    def get_branch_types(self) -> Sequence[str]:
        return ["Type 1", "Type 2"]

    def get_node_types(self) -> Sequence[str]:
        return ["Type A", "Type B", "Type C"]

    def get_injection_types(self) -> Sequence[str]:
        return ["Type X", "Type Y"]

    def get_multi_outage_types(self) -> Sequence[str]:
        return ["Type M1", "Type M2"]

    def get_metadata(self) -> dict:
        return {"key": "value"}


def test_backend():
    backend = TestBackend()

    assert backend.get_ptdf() is None
    assert backend.get_psdf() is None

    n_branch = backend.get_max_mw_flows().shape[1]
    n_bus = backend.get_relevant_node_mask().shape[0]
    assert backend.get_max_mw_flows_n_1().shape == backend.get_max_mw_flows().shape
    assert backend.get_overload_weights().shape == (n_branch,)
    assert backend.get_n0_n1_max_diff_factors().shape == (n_branch,)
    assert backend.get_cross_coupler_limits().shape == (n_bus,)
    assert backend.get_slack() == 0
    assert backend.get_max_mw_flows().shape == (2, n_branch)
    assert backend.get_susceptances().shape == (n_branch,)
    assert backend.get_from_nodes().shape == (n_branch,)
    assert backend.get_to_nodes().shape == (n_branch,)
    assert backend.get_shift_angles().shape == (2, n_branch)
    assert backend.get_phase_shift_mask().shape == (n_branch,)
    assert backend.get_relevant_node_mask().shape == (n_bus,)
    assert backend.get_monitored_branch_mask().shape == (n_branch,)
    assert backend.get_branches_in_maintenance().shape == (2, n_branch)
    assert backend.get_disconnectable_branch_mask().shape == (n_branch,)
    assert backend.get_outaged_branch_mask().shape == (n_branch,)
    assert backend.get_outaged_injection_mask().shape == (2,)
    assert backend.get_multi_outage_branches().shape == (2, n_branch)
    assert backend.get_multi_outage_nodes().shape == (2, n_bus)
    assert backend.get_injection_nodes().shape == (2,)
    assert backend.get_mw_injections().shape == (2, 2)
    assert backend.get_base_mva() == 100.0
    assert backend.get_node_ids() == ["node1", "node2", "node3"]
    assert backend.get_branch_ids() == ["branch1", "branch2"]
    assert backend.get_injection_ids() == ["inj1", "inj2"]
    assert backend.get_multi_outage_ids() == ["outage1", "outage2"]
    assert backend.get_node_names() == ["Node 1", "Node 2", "Node 3"]
    assert backend.get_branch_names() == ["Branch 1", "Branch 2"]
    assert backend.get_injection_names() == ["Injection 1", "Injection 2"]
    assert backend.get_multi_outage_names() == ["Multi Outage 1", "Multi Outage 2"]
    assert backend.get_branch_types() == ["Type 1", "Type 2"]
    assert backend.get_node_types() == ["Type A", "Type B", "Type C"]
    assert backend.get_injection_types() == ["Type X", "Type Y"]
    assert backend.get_multi_outage_types() == ["Type M1", "Type M2"]
