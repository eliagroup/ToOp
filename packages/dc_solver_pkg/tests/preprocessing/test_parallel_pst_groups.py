# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from toop_engine_dc_solver.preprocess.parallel_pst_groups import build_parallel_pst_group_mask


def test_build_parallel_pst_group_mask_distinct_labels_yield_identity() -> None:
    mask, group_ids = build_parallel_pst_group_mask(
        group_labels=np.array([0, 1, 2]),
        pst_ids=["PST1", "PST2", "PST3"],
    )

    assert np.array_equal(mask, np.eye(3, dtype=bool))
    assert group_ids == ["PST1", "PST2", "PST3"]


def test_build_parallel_pst_group_mask_shared_label_groups_members() -> None:
    mask, group_ids = build_parallel_pst_group_mask(
        group_labels=np.array([0, 0, 1]),
        pst_ids=["PST1", "PST2", "PST3"],
    )

    # PST1 and PST2 share group 0 (row 0, named after the first member); PST3 is its own group.
    assert np.array_equal(mask, np.array([[True, True, False], [False, False, True]], dtype=bool))
    assert group_ids == ["PST1", "PST3"]
    # Each PST belongs to exactly one group.
    assert np.array_equal(mask.sum(axis=0), np.ones(3, dtype=int))


def test_build_parallel_pst_group_mask_sentinel_labels_form_singletons() -> None:
    mask, group_ids = build_parallel_pst_group_mask(
        group_labels=np.array([-1, -1, 0]),
        pst_ids=["PST1", "PST2", "PST3"],
    )

    # The -1 sentinel never merges PSTs: each gets its own row.
    assert np.array_equal(mask, np.eye(3, dtype=bool))
    assert group_ids == ["PST1", "PST2", "PST3"]


def test_build_parallel_pst_group_mask_empty() -> None:
    mask, group_ids = build_parallel_pst_group_mask(group_labels=np.array([], dtype=int), pst_ids=[])

    assert mask.shape == (0, 0)
    assert group_ids == []


def test_build_parallel_pst_group_mask_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="entries but"):
        build_parallel_pst_group_mask(group_labels=np.array([0, 1]), pst_ids=["PST1"])
