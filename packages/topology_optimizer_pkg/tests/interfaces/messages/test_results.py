# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_topology_optimizer.interfaces.messages.results import (
    TopologyRejectionReason,
    get_topology_rejection_message,
)


@pytest.mark.parametrize(
    "criterion, value_before, value_after, early_stopping, description, expected_start",
    [
        ("convergence", 2.0, 3.0, False, None, "Rejecting topology due to insufficient convergence"),
        ("overload-energy", 10.0, 8.0, True, None, "Rejecting topology due to overload energy not improving"),
        ("critical-branch-count", 1.0, 5.0, False, None, "Rejecting topology due to critical branches increasing too much"),
        ("voltage-magnitude", 0.95, 1.05, True, None, "Rejecting topology due to voltage magnitude violation"),
        ("voltage-angle", 10.0, 15.0, False, None, "Rejecting topology due to voltage angle violation"),
        ("other", 0.0, 0.0, False, None, "Rejecting topology due to other reason"),
    ],
)
def test_get_topology_rejection_message_basic(
    criterion, value_before, value_after, early_stopping, description, expected_start
):
    reason = TopologyRejectionReason(
        criterion=criterion,
        value_before=value_before,
        value_after=value_after,
        early_stopping=early_stopping,
    )
    msg = get_topology_rejection_message(reason)
    assert msg.startswith(expected_start)
    assert f"value before: {value_before}" in msg
    assert f"value after: {value_after}" in msg
    assert f"early_stopping: {early_stopping}" in msg
    if description is None:
        assert "Details:" not in msg
    else:
        assert f"Details: {description}" in msg
