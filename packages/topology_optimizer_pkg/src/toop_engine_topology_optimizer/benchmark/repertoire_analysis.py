# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Transform multidimensional repertoire snapshots into 2D analyzer views."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from beartype.typing import Literal, Sequence

RepertoireMetric = Literal["fitness", "selection_count"]


def descriptor_pairs(selected_descriptor_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return ordered, unordered descriptor pairs for an analyzer selection.

    Parameters
    ----------
    selected_descriptor_names : Sequence[str]
        Descriptor names in their configured repertoire order.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Every pair in configuration order.

    Raises
    ------
    ValueError
        If fewer than two, more than four, or duplicate descriptors are selected.
    """
    descriptor_names = tuple(selected_descriptor_names)
    if not 2 <= len(descriptor_names) <= 4:
        raise ValueError("Select between two and four descriptors for pairwise repertoire analysis.")
    if len(set(descriptor_names)) != len(descriptor_names):
        raise ValueError("Selected descriptor names must be unique.")
    return tuple(combinations(descriptor_names, 2))


def project_repertoire_snapshot(
    cell_indices: np.ndarray,
    metric_values: np.ndarray,
    n_cells_per_dim: tuple[int, ...],
    descriptor_names: Sequence[str],
    descriptor_pair: tuple[str, str],
    metric: RepertoireMetric,
) -> np.ndarray:
    """Project one multidimensional sparse repertoire snapshot onto two descriptors.

    The first descriptor of ``descriptor_pair`` is placed on the vertical axis,
    and the second descriptor is placed on the horizontal axis. Fitness retains
    the best finite elite across hidden dimensions. Selection counts sum across
    hidden dimensions, with omitted sparse snapshot cells treated as zero.

    Parameters
    ----------
    cell_indices : np.ndarray
        Sparse logical-cell indices from one epoch snapshot.
    metric_values : np.ndarray
        Fitnesses or cumulative selection counts aligned with ``cell_indices``.
    n_cells_per_dim : tuple[int, ...]
        Number of logical cells in every configured descriptor dimension.
    descriptor_names : Sequence[str]
        Descriptor names aligned with ``n_cells_per_dim``.
    descriptor_pair : tuple[str, str]
        Vertical and horizontal descriptors to retain in the projection.
    metric : RepertoireMetric
        ``"fitness"`` or ``"selection_count"`` aggregation semantics.

    Returns
    -------
    np.ndarray
        A matrix with the first descriptor on rows and the second on columns.
        Empty fitness cells are ``NaN``; empty selection cells are zero.

    Raises
    ------
    ValueError
        If snapshot data, descriptor metadata, or metric values are invalid.
    """
    descriptor_names = tuple(descriptor_names)
    _validate_descriptor_metadata(n_cells_per_dim, descriptor_names, descriptor_pair)

    indices = np.asarray(cell_indices, dtype=int).reshape(-1)
    values = np.asarray(metric_values, dtype=float).reshape(-1)
    n_logical_cells = int(np.prod(n_cells_per_dim))
    if indices.shape != values.shape:
        raise ValueError("Snapshot cell_indices and metric values must have matching shapes.")
    if np.any(indices < 0) or np.any(indices >= n_logical_cells):
        raise ValueError("Snapshot cell indices must lie inside the repertoire layout.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Snapshot cell indices must be unique.")

    vertical_index = descriptor_names.index(descriptor_pair[0])
    horizontal_index = descriptor_names.index(descriptor_pair[1])
    output_shape = (n_cells_per_dim[vertical_index], n_cells_per_dim[horizontal_index])
    if metric == "fitness":
        projected_values = np.full(output_shape, -np.inf, dtype=float)
        valid_values = np.isfinite(values)
        if not valid_values.any():
            return np.full(output_shape, np.nan, dtype=float)
        coordinates = np.asarray(np.unravel_index(indices[valid_values], n_cells_per_dim), dtype=int)
        flat_indices = coordinates[vertical_index] * output_shape[1] + coordinates[horizontal_index]
        np.maximum.at(projected_values.reshape(-1), flat_indices, values[valid_values])
        projected_values[~np.isfinite(projected_values)] = np.nan
        return projected_values

    if metric == "selection_count":
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Selection counts must be finite and nonnegative.")
        projected_values = np.zeros(output_shape, dtype=float)
        if not indices.size:
            return projected_values
        coordinates = np.asarray(np.unravel_index(indices, n_cells_per_dim), dtype=int)
        flat_indices = coordinates[vertical_index] * output_shape[1] + coordinates[horizontal_index]
        np.add.at(projected_values.reshape(-1), flat_indices, values)
        return projected_values

    raise ValueError(f"Unsupported repertoire metric: {metric!r}.")


def aggregate_seed_projections(
    projected_values: Sequence[np.ndarray],
    metric: RepertoireMetric,
) -> np.ndarray:
    """Aggregate same-layout projected snapshots from the seeds available at one epoch.

    Parameters
    ----------
    projected_values : Sequence[np.ndarray]
        One 2D projection per seed that reached the selected epoch.
    metric : RepertoireMetric
        Aggregation semantics for fitness or selection counts.

    Returns
    -------
    np.ndarray
        Per-cell mean across included seeds. Fitness ignores unavailable cells;
        selection counts include true zero values.

    Raises
    ------
    ValueError
        If no projections are supplied or their shapes differ.
    """
    matrices = tuple(np.asarray(values, dtype=float) for values in projected_values)
    if not matrices:
        raise ValueError("At least one seed projection is required for aggregation.")
    if any(values.shape != matrices[0].shape for values in matrices[1:]):
        raise ValueError("All seed projections must have the same shape.")

    stacked_values = np.stack(matrices)
    if metric == "selection_count":
        return stacked_values.mean(axis=0)
    if metric == "fitness":
        finite_values = np.isfinite(stacked_values)
        totals = np.where(finite_values, stacked_values, 0.0).sum(axis=0)
        counts = finite_values.sum(axis=0)
        return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    raise ValueError(f"Unsupported repertoire metric: {metric!r}.")


def _validate_descriptor_metadata(
    n_cells_per_dim: tuple[int, ...],
    descriptor_names: tuple[str, ...],
    descriptor_pair: tuple[str, str],
) -> None:
    """Validate descriptor metadata shared by snapshot projections."""
    if len(n_cells_per_dim) != len(descriptor_names):
        raise ValueError("Descriptor names and repertoire cell dimensions must have matching lengths.")
    if not n_cells_per_dim or any(num_cells < 1 for num_cells in n_cells_per_dim):
        raise ValueError("Repertoire cell dimensions must be nonempty positive integers.")
    if len(set(descriptor_names)) != len(descriptor_names):
        raise ValueError("Repertoire descriptor names must be unique.")
    if descriptor_pair[0] == descriptor_pair[1]:
        raise ValueError("A 2D repertoire projection requires two distinct descriptors.")
    unknown_descriptors = sorted(set(descriptor_pair) - set(descriptor_names))
    if unknown_descriptors:
        raise ValueError(f"Descriptor pair contains names not present in the repertoire: {unknown_descriptors}.")
