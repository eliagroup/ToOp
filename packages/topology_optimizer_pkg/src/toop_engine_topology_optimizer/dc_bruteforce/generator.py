# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Lazy workset generation for the DC bruteforce optimizer."""

from dataclasses import dataclass
from itertools import combinations, islice, product
from math import comb

from beartype.typing import Iterator, Sequence


@dataclass(frozen=True)
class WorksetEntry:
    """One exhaustively enumerated topology candidate.

    Empty action/disconnection slots are not represented as the empty action but just by a shorter tuple as that is easier
    to enumerate on using itertools.combinations.
    """

    action_indices: tuple[int, ...]
    """Split-action indices into the action set. Each index belongs to a different substation.
    """

    disconnections: tuple[int, ...]
    """Disconnection indices into ``disconnectable_branches``."""


def count_workset_size(
    split_action_groups: Sequence[Sequence[int]],
    max_num_splits: int,
    n_disconnectable_branches: int,
    max_num_disconnections: int,
) -> int:
    """Count the total number of bruteforce candidates.

    Parameters
    ----------
    split_action_groups : Sequence[Sequence[int]]
        Split-action indices grouped by substation.
    max_num_splits : int
        Maximum number of simultaneously split substations.
    n_disconnectable_branches : int
        Number of disconnectable branches.
    max_num_disconnections : int
        Maximum number of simultaneous disconnections.

    Returns
    -------
    int
        The exact total number of topology candidates the workset iterator will emit.
    """
    limited_num_splits = min(max_num_splits, len(split_action_groups))
    limited_num_disconnections = min(max_num_disconnections, n_disconnectable_branches)
    disconnection_factor = sum(comb(n_disconnectable_branches, n_disc) for n_disc in range(limited_num_disconnections + 1))

    total_split_combinations = 0
    for n_splits in range(limited_num_splits + 1):
        for chosen_groups in combinations(split_action_groups, n_splits):
            split_combination_count = 1
            for action_group in chosen_groups:
                split_combination_count *= len(action_group)
            total_split_combinations += split_combination_count

    return total_split_combinations * disconnection_factor


def iter_workset(
    action_start_indices: Sequence[int],
    n_actions_per_sub: Sequence[int],
    max_num_splits: int,
    n_disconnectable_branches: int,
    max_num_disconnections: int,
) -> Iterator[WorksetEntry]:
    """Yield topology candidates lazily from contiguous per-substation action blocks.

    Parameters
    ----------
    action_start_indices : Sequence[int]
        Start index of each substation's contiguous action block in the flattened action set.
    n_actions_per_sub : Sequence[int]
        Number of actions available for each substation. The bruteforce path assumes every relevant
        substation has at least one action.
    max_num_splits : int
        Maximum number of simultaneously split substations.
    n_disconnectable_branches : int
        Number of disconnectable branches.
    max_num_disconnections : int
        Maximum number of simultaneous disconnections.

    Yields
    ------
    WorksetEntry
        The next lazily generated topology candidate.
    """
    split_action_groups = tuple(
        tuple(range(int(start), int(start) + int(count)))
        for start, count in zip(action_start_indices, n_actions_per_sub, strict=True)
    )
    limited_num_splits = min(max_num_splits, len(split_action_groups))
    limited_num_disconnections = min(max_num_disconnections, n_disconnectable_branches)
    disconnectable_indices = tuple(range(n_disconnectable_branches))

    for n_splits in range(limited_num_splits + 1):
        for chosen_groups in combinations(split_action_groups, n_splits):
            for chosen_actions in product(*chosen_groups) if chosen_groups else [()]:
                action_indices = tuple(int(action_index) for action_index in chosen_actions)
                for n_disconnections in range(limited_num_disconnections + 1):
                    for chosen_disconnections in combinations(disconnectable_indices, n_disconnections):
                        yield WorksetEntry(
                            action_indices=action_indices,
                            disconnections=tuple(int(disconnection) for disconnection in chosen_disconnections),
                        )


def take_workset_chunk(workset: Iterator[WorksetEntry], chunk_size: int) -> list[WorksetEntry]:
    """Take the next chunk from a lazy workset iterator.

    Parameters
    ----------
    workset : Iterator[WorksetEntry]
        The lazy workset iterator.
    chunk_size : int
        Maximum number of entries to retrieve.

    Returns
    -------
    list[WorksetEntry]
        The next chunk from the iterator.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return list(islice(workset, chunk_size))
