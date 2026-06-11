# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for loading and defaulting parallel PST group definitions."""

import numpy as np
import structlog
from beartype.typing import Sequence
from fsspec import AbstractFileSystem
from jaxtyping import Bool
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import load_action_set_fs

logger = structlog.get_logger(__name__)


def load_or_create_parallel_pst_groups(
    filesystem: AbstractFileSystem,
    pst_ids: Sequence[str | int],
) -> tuple[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"], list[str]]:
    """Load the parallel PST grouping and preserve the configured group identifiers.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        Filesystem rooted at the preprocessing directory.
    pst_ids : Sequence[str | int]
        Ordered controllable PST ids the output mask should align with.

    Returns
    -------
    tuple[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"],
          list[str]]
        A boolean group mask aligned with ``pst_ids``. Each row corresponds to a group.
        A list of group identifiers corresponding to the rows of the mask.
    """
    pst_id_list = [str(pst_id) for pst_id in pst_ids]
    if not pst_id_list:
        return np.zeros((0, 0), dtype=bool), []

    file_path = PREPROCESSING_PATHS["action_set_file_path"]
    if not filesystem.exists(file_path):
        logger.warning(
            "action_set.json is missing. Using a default PST group mapping with one group per PST.",
            file_path=file_path,
        )
        return np.eye(len(pst_id_list), dtype=bool), pst_id_list.copy()

    action_set = load_action_set_fs(filesystem=filesystem, json_file_path=file_path, diff_file_path=None)
    pst_to_group = _build_pst_group_mapping(
        pst_groups=[(str(pst.id), pst.pst_group or str(pst.id)) for pst in action_set.pst_ranges],
        file_path=file_path,
    )

    missing_pst_ids = [pst_id for pst_id in pst_id_list if pst_id not in pst_to_group]
    if missing_pst_ids:
        logger.warning(
            "action_set.json does not define groups for all controllable PST ids. "
            "Falling back to one group per missing PST.",
            file_path=file_path,
            pst_ids=missing_pst_ids,
        )
        for pst_id in missing_pst_ids:
            pst_to_group[pst_id] = pst_id

    unknown_pst_ids = [pst_id for pst_id in pst_to_group if pst_id not in set(pst_id_list)]
    if unknown_pst_ids:
        logger.warning(
            "action_set.json contains PST ids that are not part of the controllable PST set. Ignoring them.",
            file_path=file_path,
            pst_ids=unknown_pst_ids,
        )

    ordered_group_names: list[str] = []
    group_index_by_name: dict[str, int] = {}
    for pst_id in pst_id_list:
        group_name = pst_to_group[pst_id]
        if group_name not in group_index_by_name:
            group_index_by_name[group_name] = len(ordered_group_names)
            ordered_group_names.append(group_name)

    parallel_pst_group_mask = np.zeros((len(ordered_group_names), len(pst_id_list)), dtype=bool)
    for pst_index, pst_id in enumerate(pst_id_list):
        parallel_pst_group_mask[group_index_by_name[pst_to_group[pst_id]], pst_index] = True

    return parallel_pst_group_mask, ordered_group_names


def load_or_create_parallel_pst_group_mask(
    filesystem: AbstractFileSystem,
    pst_ids: Sequence[str | int],
) -> Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"]:
    """Load the parallel PST grouping from action_set.json or generate a default one.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        Filesystem rooted at the preprocessing directory.
    pst_ids : Sequence[str | int]
        Ordered controllable PST ids the output mask should align with.

    Returns
    -------
    Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"]
        A boolean group mask aligned with ``pst_ids``. Each column belongs to exactly one group.
    """
    parallel_pst_group_mask, _ = load_or_create_parallel_pst_groups(filesystem=filesystem, pst_ids=pst_ids)
    return parallel_pst_group_mask


def _build_pst_group_mapping(pst_groups: list[tuple[str, str]], file_path: str) -> dict[str, str]:
    """Build a mapping from PST id to group name and check for duplicates.

    Parameters
    ----------
    pst_groups : list[tuple[str, str]]
        List of (pst_id, group_name) tuples parsed from the action set.
    file_path : str
        Path to the ActionSet (action_set.json) file, used for error messages.

    Returns
    -------
    dict[str, str]
        Mapping from PST id to group name.

    Raises
    ------
    ValueError
        If a duplicate PST id is found.

    """
    pst_to_group: dict[str, str] = {}
    for pst_id, group_name in pst_groups:
        if pst_id in pst_to_group:
            raise ValueError(f"Duplicate PST id '{pst_id}' found in {file_path}.")
        pst_to_group[pst_id] = group_name
    return pst_to_group
