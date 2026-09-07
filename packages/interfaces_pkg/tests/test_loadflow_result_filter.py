# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for the loadflow result filter policy model.

One test for the combinations the model accepts and one for the combinations it rejects, with a message per assert naming
the rule it protects.
"""

import pytest
from pydantic import ValidationError
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)


def test_accepted_policies():
    """Defaults, activity reporting and serialization of the combinations the model allows."""
    inert = LoadflowResultFilter()
    assert not inert.is_active(), "an unconfigured policy must never drop a row"
    assert not inert.branch_filters.is_active(), "an unconfigured branch filter must never drop a row"
    assert not inert.node_filters.is_active(), "an unconfigured node filter must never drop a row"
    assert inert.branch_filters.retain_basecase, "N-0 branch rows are exempt unless someone opts out"
    assert inert.node_filters.retain_basecase, "N-0 node rows are exempt unless someone opts out"

    branch_only = LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=0.7))
    node_only = LoadflowResultFilter(node_filters=NodeLoadflowResultFilter(vm_loading_above=0.7))
    assert branch_only.is_active(), "one configured table is enough to make the filtering pass worthwhile"
    assert node_only.is_active(), "one configured table is enough to make the filtering pass worthwhile"

    # Setting a threshold is what makes lifting the basecase exemption meaningful, so this must construct.
    lifted = BranchLoadflowResultFilter(loading_above=0.7, retain_basecase=False)
    assert lifted.is_active() and not lifted.retain_basecase

    configured = LoadflowResultFilter(
        branch_filters=lifted,
        node_filters=NodeLoadflowResultFilter(vm_loading_above=0.7, vm_basecase_deviation_above=5.0),
    )
    assert LoadflowResultFilter.model_validate_json(configured.model_dump_json()) == configured, (
        "the policy is stored alongside the results, so it has to serialize losslessly"
    )


def test_rejected_policies():
    """Combinations whose effect would not match their reading are refused at construction."""
    rejected = [
        (
            BranchLoadflowResultFilter,
            {"retain_basecase": False},
            "with no threshold nothing can filter basecase branch rows, so lifting the exemption is a no-op",
        ),
        (
            NodeLoadflowResultFilter,
            {"retain_basecase": False},
            "with no threshold nothing can filter basecase node rows, so lifting the exemption is a no-op",
        ),
        (
            NodeLoadflowResultFilter,
            {"vm_basecase_deviation_above": 5.0},
            "the jump filter is an exemption on top of vm_loading_above, not a filter of its own",
        ),
        (
            BranchLoadflowResultFilter,
            {"loading_above": -0.1},
            "thresholds are compared against absolute values, so a negative one has no meaning",
        ),
        (
            NodeLoadflowResultFilter,
            {"vm_loading_above": -0.1},
            "thresholds are compared against absolute values, so a negative one has no meaning",
        ),
        (
            NodeLoadflowResultFilter,
            {"vm_loading_above": 0.7, "vm_basecase_deviation_above": -1.0},
            "a negative voltage jump threshold has no meaning either",
        ),
    ]

    for model, kwargs, reason in rejected:
        try:
            model(**kwargs)
        except ValidationError:
            continue
        pytest.fail(f"{model.__name__}(**{kwargs}) should be rejected: {reason}")
