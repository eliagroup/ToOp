# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import load_action_set


def _group_local_actions_by_station(local_actions: list) -> list[tuple[str, int]]:
    groups: list[tuple[str, int]] = []
    start = 0
    while start < len(local_actions):
        grid_model_id = local_actions[start].grid_model_id
        end = start + 1
        while end < len(local_actions) and local_actions[end].grid_model_id == grid_model_id:
            end += 1
        groups.append((grid_model_id, end - start))
        start = end
    return groups


def test_loaded_action_set_stays_aligned_with_jax_action_boundaries(_grid_folder: Path) -> None:
    complex_grid_folder = _grid_folder / "complex_grid"
    static_information = load_static_information(complex_grid_folder / PREPROCESSING_PATHS["static_information_file_path"])
    action_set = load_action_set(
        complex_grid_folder / PREPROCESSING_PATHS["action_set_file_path"],
        complex_grid_folder / PREPROCESSING_PATHS["action_set_diff_path"],
    )

    loaded_groups = _group_local_actions_by_station(action_set.local_actions)
    jax_groups: list[tuple[str, int]] = []
    starts = [int(value) for value in static_information.dynamic_information.action_set.action_start_indices]
    counts = [int(value) for value in static_information.dynamic_information.action_set.n_actions_per_sub]

    for start, count in zip(starts, counts, strict=True):
        station_ids = {action_set.local_actions[index].grid_model_id for index in range(start, start + count)}
        assert len(station_ids) == 1, "A JAX action-substation block must map to exactly one stored station id."
        jax_groups.append((next(iter(station_ids)), count))

    assert loaded_groups == jax_groups
