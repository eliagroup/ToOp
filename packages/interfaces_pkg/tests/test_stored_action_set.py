# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from toop_engine_interfaces.stored_action_set import ActionSet, random_actions


class DummyStation:
    def __init__(self, grid_model_id):
        self.grid_model_id = str(grid_model_id)


@pytest.fixture
def action_set_multiple_subs() -> ActionSet:
    # 3 substations, each with 2 actions
    local_actions = [
        DummyStation(1),
        DummyStation(1),
        DummyStation(2),
        DummyStation(2),
        DummyStation(3),
        DummyStation(3),
    ]
    return ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=local_actions,
        global_actions=[],
    )


def test_random_actions_no_duplicates(action_set_multiple_subs: ActionSet):
    rng = np.random.default_rng(123)
    n_split_subs = 3
    result = random_actions(action_set_multiple_subs, rng, n_split_subs)
    assert len(result) == n_split_subs
    # Each index should correspond to a different substation
    chosen_subs = [action_set_multiple_subs.local_actions[i].grid_model_id for i in result]
    assert len(set(chosen_subs)) == len(chosen_subs)


def test_random_actions_clips_to_available_subs(action_set_multiple_subs: ActionSet):
    rng = np.random.default_rng(7)
    n_split_subs = 10  # more than available substations
    result = random_actions(action_set_multiple_subs, rng, n_split_subs)
    assert len(result) == 3  # only 3 substations available


def test_random_actions_empty_local_actions():
    rng = np.random.default_rng(0)
    action_set = ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=[],
        global_actions=[],
    )
    result = random_actions(action_set, rng, 2)
    assert result == []


def test_random_actions_single_substation():
    rng = np.random.default_rng(0)
    local_actions = [DummyStation(42), DummyStation(42)]
    action_set = ActionSet.model_construct(
        starting_topology=None,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=[],
        hvdc_ranges=[],
        local_actions=local_actions,
        global_actions=[],
    )
    result = random_actions(action_set, rng, 1)
    assert len(result) == 1
    # The only possible indices are 0 or 1
    assert result[0] in [0, 1]
