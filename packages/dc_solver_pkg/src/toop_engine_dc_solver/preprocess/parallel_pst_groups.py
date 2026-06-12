# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for building parallel PST group masks from per-PST group labels.

The group membership itself is identified during importing (see ``pst_group_masks`` in
``powsybl_masks.py``); this module only reshapes the per-PST integer labels into the boolean group
mask consumed by the preprocessing and optimization stages.
"""

import numpy as np
from beartype.typing import Sequence
from jaxtyping import Bool, Int


def build_parallel_pst_group_mask(
    group_labels: Int[np.ndarray, " n_controllable_pst"],
    pst_ids: Sequence[str | int],
) -> tuple[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"], list[str]]:
    """Build the parallel PST group mask and group ids from per-PST group labels.

    PSTs that share a (non-negative) label belong to the same parallel group. The sentinel ``-1``
    marks an ungrouped PST, which is given its own singleton group, so the default with no parallel
    PSTs reduces to one group per PST.

    Parameters
    ----------
    group_labels : Int[np.ndarray, " n_controllable_pst"]
        Integer group label for each controllable PST, aligned with ``pst_ids``.
    pst_ids : Sequence[str | int]
        Ordered controllable PST ids the output mask aligns with.

    Returns
    -------
    tuple[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"], list[str]]
        A boolean group mask with one row per distinct group (in first-seen order), where each
        column has exactly one ``True``, and a list of group identifiers (the first member's PST id
        per row).
    """
    pst_id_list = [str(pst_id) for pst_id in pst_ids]
    n_psts = len(pst_id_list)
    if group_labels.shape[0] != n_psts:
        raise ValueError(f"group_labels has {group_labels.shape[0]} entries but {n_psts} controllable PST ids were given.")
    if n_psts == 0:
        return np.zeros((0, 0), dtype=bool), []

    rows = np.zeros(n_psts, dtype=int)
    row_by_label: dict[int, int] = {}
    group_ids: list[str] = []
    next_row = 0
    for pst_index, label in enumerate(int(label) for label in group_labels):
        if label < 0:
            # Ungrouped sentinel: each such PST forms its own singleton group.
            rows[pst_index] = next_row
        elif label in row_by_label:
            rows[pst_index] = row_by_label[label]
            continue
        else:
            row_by_label[label] = next_row
            rows[pst_index] = next_row
        group_ids.append(pst_id_list[pst_index])
        next_row += 1

    parallel_pst_group_mask = np.zeros((next_row, n_psts), dtype=bool)
    parallel_pst_group_mask[rows, np.arange(n_psts)] = True
    return parallel_pst_group_mask, group_ids
