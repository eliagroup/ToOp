# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for configuring result filtering on the AC optimizer.

The AC metrics are computed from the same results the filter shrinks, so the policy and the metric thresholds have to
agree. These tests pin that agreement, and that the policy reaches the runners that produce the per-candidate results.
"""

import pytest
from pydantic import ValidationError
from toop_engine_dc_solver.postprocess.postprocess_pandapower import PandapowerRunner
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters


def _policy(**node_kwargs) -> LoadflowResultFilter:
    """Build a policy with a branch threshold and the given node settings.

    Parameters
    ----------
    **node_kwargs
        Settings for the node sub-filter.

    Returns
    -------
    LoadflowResultFilter
        The assembled policy.
    """
    return LoadflowResultFilter(
        branch_filters=BranchLoadflowResultFilter(loading_above=0.7),
        node_filters=NodeLoadflowResultFilter(**node_kwargs) if node_kwargs else NodeLoadflowResultFilter(),
    )


def test_accepted_ga_configurations():
    """The default is inert, and a policy that leaves the metrics their rows is accepted."""
    assert not ACGAParameters().result_filter.is_active(), "filtering must be off unless someone asks for it"

    matching = ACGAParameters(result_filter=_policy(vm_loading_above=0.7, vm_basecase_deviation_above=5.0))
    assert matching.result_filter.is_active()
    assert matching.result_filter.node_filters.vm_basecase_deviation_above <= matching.critical_voltage_jump_percent

    # Raising the metric threshold is what makes a looser jump threshold legitimate.
    assert (
        ACGAParameters(
            critical_voltage_jump_percent=10.0,
            result_filter=_policy(vm_loading_above=0.7, vm_basecase_deviation_above=9.0),
        ).result_filter.node_filters.vm_basecase_deviation_above
        == 9.0
    )

    # A branch-only policy has nothing to agree with on the node side.
    assert ACGAParameters(
        result_filter=LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=1.0))
    ).result_filter.is_active()


def test_rejected_ga_configurations():
    """Policies that would starve the metrics computed from the same results are refused."""
    with pytest.raises(ValidationError, match="vm_basecase_deviation_above"):
        # Filtering nodes on voltage alone drops a bus that jumps far enough to be critical while staying in band,
        # so voltage_jump_count_n_1 would silently under-report.
        ACGAParameters(result_filter=_policy(vm_loading_above=0.7))

    with pytest.raises(ValidationError, match="vm_basecase_deviation_above"):
        # A jump threshold looser than the metric's own threshold has the same effect.
        ACGAParameters(result_filter=_policy(vm_loading_above=0.7, vm_basecase_deviation_above=9.0))

    with pytest.raises(ValidationError, match="loading_above"):
        # The overload metrics count branches above 1.0, which a higher threshold would drop first.
        ACGAParameters(result_filter=LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=1.5)))


@pytest.mark.parametrize("runner_class", [PandapowerRunner, PowsyblRunner])
def test_runners_carry_the_policy(runner_class):
    """The runners hold the policy for every loadflow they run, and default to keeping every row."""
    policy = _policy(vm_loading_above=0.7, vm_basecase_deviation_above=5.0)

    assert runner_class(result_filter=policy).result_filter == policy, "the runner must keep the policy it was given"
    assert not runner_class().result_filter.is_active(), "a runner built without a policy must keep every row"
