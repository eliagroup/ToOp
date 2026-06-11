# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from datetime import datetime

import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.parallel_pst_groups import load_or_create_parallel_pst_group_mask
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import ActionSet, PSTRange


def _make_action_set(*pst_ranges: PSTRange) -> ActionSet:
    topology = Topology.model_construct(
        topology_id="topology",
        grid_model_file=None,
        name=None,
        stations=[],
        asset_setpoints=None,
        timestamp=datetime.now(),
        metrics=None,
    )
    return ActionSet.model_construct(
        starting_topology=topology,
        simplified_starting_topology=topology,
        connectable_branches=[],
        disconnectable_branches=[],
        pst_ranges=list(pst_ranges),
        hvdc_ranges=[],
        local_actions=[],
    )


def test_load_parallel_pst_group_mask_defaults_to_identity_when_action_set_missing(tmp_path) -> None:
    filesystem = DirFileSystem(str(tmp_path))

    group_mask = load_or_create_parallel_pst_group_mask(filesystem=filesystem, pst_ids=["PST1", "PST2"])

    assert np.array_equal(group_mask, np.eye(2, dtype=bool))


def test_load_parallel_pst_group_mask_reads_groups_from_action_set(tmp_path) -> None:
    filesystem = DirFileSystem(str(tmp_path))
    action_set = _make_action_set(
        PSTRange(
            id="PST1",
            name="PST1",
            type="TWO_WINDINGS_TRANSFORMER",
            kind="branch",
            starting_tap=0,
            low_tap=-30,
            high_tap=31,
            pst_group="group_a",
        ),
        PSTRange(
            id="PST2",
            name="PST2",
            type="TWO_WINDINGS_TRANSFORMER",
            kind="branch",
            starting_tap=0,
            low_tap=-30,
            high_tap=31,
            pst_group="group_a",
        ),
        PSTRange(
            id="PST3",
            name="PST3",
            type="TWO_WINDINGS_TRANSFORMER",
            kind="branch",
            starting_tap=0,
            low_tap=-20,
            high_tap=21,
            pst_group="group_b",
        ),
    )
    with filesystem.open(PREPROCESSING_PATHS["action_set_file_path"], "w", encoding="utf-8") as file:
        file.write(action_set.model_dump_json())

    group_mask = load_or_create_parallel_pst_group_mask(filesystem=filesystem, pst_ids=["PST1", "PST2", "PST3"])

    expected = np.array([[True, True, False], [False, False, True]], dtype=bool)
    assert np.array_equal(group_mask, expected)


def test_load_parallel_pst_group_mask_defaults_missing_pst_entries(tmp_path) -> None:
    filesystem = DirFileSystem(str(tmp_path))
    action_set = _make_action_set(
        PSTRange(
            id="PST1",
            name="PST1",
            type="TWO_WINDINGS_TRANSFORMER",
            kind="branch",
            starting_tap=0,
            low_tap=-30,
            high_tap=31,
            pst_group="group_a",
        )
    )
    with filesystem.open(PREPROCESSING_PATHS["action_set_file_path"], "w", encoding="utf-8") as file:
        file.write(action_set.model_dump_json())

    group_mask = load_or_create_parallel_pst_group_mask(filesystem=filesystem, pst_ids=["PST1", "PST2"])

    expected = np.array([[True, False], [False, True]], dtype=bool)
    assert np.array_equal(group_mask, expected)


def test_load_parallel_pst_group_mask_defaults_and_fills_missing_pst_entries(tmp_path) -> None:
    filesystem = DirFileSystem(str(tmp_path))
    action_set = _make_action_set(
        PSTRange(
            id="PST1",
            name="PST1",
            type="TWO_WINDINGS_TRANSFORMER",
            kind="branch",
            starting_tap=0,
            low_tap=-30,
            high_tap=31,
        )
    )
    with filesystem.open(PREPROCESSING_PATHS["action_set_file_path"], "w", encoding="utf-8") as file:
        file.write(action_set.model_dump_json())

    group_mask = load_or_create_parallel_pst_group_mask(filesystem=filesystem, pst_ids=["PST1", "PST2"])

    expected = np.array([[True, False], [False, True]], dtype=bool)
    assert np.array_equal(group_mask, expected)
