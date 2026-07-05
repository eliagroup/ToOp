# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
import pytest
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_topology_optimizer.dc_bruteforce.generator import (
    WorksetEntry,
    count_workset_size,
    iter_workset,
    take_workset_chunk,
)


def test_iter_workset_from_groups_exhaustive_count() -> None:
    split_action_groups = ((0, 1), (2, 3, 4), (5,))
    action_start_indices = (0, 2, 5)
    n_actions_per_sub = (2, 3, 1)

    workset = list(
        iter_workset(
            action_start_indices=action_start_indices,
            n_actions_per_sub=n_actions_per_sub,
            max_num_splits=2,
            n_disconnectable_branches=2,
            max_num_disconnections=1,
        )
    )

    expected_count = count_workset_size(
        split_action_groups=split_action_groups,
        max_num_splits=2,
        n_disconnectable_branches=2,
        max_num_disconnections=1,
    )
    assert len(workset) == expected_count
    assert workset[0] == WorksetEntry(action_indices=(), disconnections=())
    assert WorksetEntry(action_indices=(0, 2), disconnections=(1,)) in workset
    assert all(len(entry.action_indices) <= 2 for entry in workset)


def test_take_workset_chunk_advances_iterator() -> None:
    workset = iter_workset(
        action_start_indices=(0, 2),
        n_actions_per_sub=(2, 2),
        max_num_splits=2,
        n_disconnectable_branches=1,
        max_num_disconnections=0,
    )

    first_chunk = take_workset_chunk(workset, 3)
    second_chunk = take_workset_chunk(workset, 3)

    assert first_chunk == [
        WorksetEntry(action_indices=(), disconnections=()),
        WorksetEntry(action_indices=(0,), disconnections=()),
        WorksetEntry(action_indices=(1,), disconnections=()),
    ]
    assert second_chunk == [
        WorksetEntry(action_indices=(2,), disconnections=()),
        WorksetEntry(action_indices=(3,), disconnections=()),
        WorksetEntry(action_indices=(0, 2), disconnections=()),
    ]


def test_iter_workset_includes_all_single_split_actions(_grid_folder: Path) -> None:
    static_information = load_static_information(
        _grid_folder / "complex_grid" / PREPROCESSING_PATHS["static_information_file_path"]
    )
    action_set = static_information.dynamic_information.action_set

    chunk = take_workset_chunk(
        iter_workset(
            action_start_indices=action_set.action_start_indices.tolist(),
            n_actions_per_sub=action_set.n_actions_per_sub.tolist(),
            max_num_splits=1,
            n_disconnectable_branches=0,
            max_num_disconnections=0,
        ),
        len(action_set) + 1,
    )

    assert chunk[0] == WorksetEntry(action_indices=(), disconnections=())
    generated_single_split_actions = {entry.action_indices[0] for entry in chunk[1:]}
    assert generated_single_split_actions == set(range(len(action_set)))


def test_iter_workset_uses_action_set_boundaries(_grid_folder: Path) -> None:
    static_information = load_static_information(
        _grid_folder / "complex_grid" / PREPROCESSING_PATHS["static_information_file_path"]
    )
    action_set = static_information.dynamic_information.action_set

    chunk = take_workset_chunk(
        iter_workset(
            action_start_indices=action_set.action_start_indices.tolist(),
            n_actions_per_sub=action_set.n_actions_per_sub.tolist(),
            max_num_splits=2,
            n_disconnectable_branches=int(static_information.dynamic_information.n_disconnectable_branches),
            max_num_disconnections=1,
        ),
        128,
    )

    substation_correspondence = np.asarray(action_set.substation_correspondence).astype(int).tolist()
    for entry in chunk:
        chosen_substations = {substation_correspondence[action_index] for action_index in entry.action_indices}
        assert len(chosen_substations) == len(entry.action_indices)


def test_take_workset_chunk_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        take_workset_chunk(iter(()), 0)
