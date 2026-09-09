# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for parent-selection repertoire analyzer transformations."""

import numpy as np
import pytest
from toop_engine_topology_optimizer.benchmark.repertoire_analysis import (
    aggregate_seed_projections,
    descriptor_pairs,
    project_repertoire_snapshot,
)


@pytest.mark.parametrize(
    ("descriptors", "expected_pairs"),
    [
        (("a", "b"), (("a", "b"),)),
        (("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))),
        (
            ("a", "b", "c", "d"),
            (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
        ),
    ],
)
def test_descriptor_pairs_returns_every_pair_in_descriptor_order(
    descriptors: tuple[str, ...],
    expected_pairs: tuple[tuple[str, str], ...],
) -> None:
    assert descriptor_pairs(descriptors) == expected_pairs


def test_project_repertoire_snapshot_projects_fitness_and_selection_counts() -> None:
    cell_indices = np.array([0, 1, 11])
    dimensions = (2, 3, 2)
    descriptor_names = ("a", "b", "c")

    projected_fitness = project_repertoire_snapshot(
        cell_indices=cell_indices,
        metric_values=np.array([2.0, 5.0, 7.0]),
        n_cells_per_dim=dimensions,
        descriptor_names=descriptor_names,
        descriptor_pair=("a", "b"),
        metric="fitness",
    )
    projected_counts = project_repertoire_snapshot(
        cell_indices=cell_indices,
        metric_values=np.array([3.0, 4.0, 5.0]),
        n_cells_per_dim=dimensions,
        descriptor_names=descriptor_names,
        descriptor_pair=("a", "b"),
        metric="selection_count",
    )

    assert np.allclose(projected_fitness, np.array([[5.0, np.nan, np.nan], [np.nan, np.nan, 7.0]]), equal_nan=True)
    assert np.array_equal(projected_counts, np.array([[7.0, 0.0, 0.0], [0.0, 0.0, 5.0]]))


def test_project_repertoire_snapshot_supports_every_pair_of_four_descriptors() -> None:
    descriptor_names = ("a", "b", "c", "d")
    n_cells_per_dim = (2, 3, 2, 2)
    coordinates = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [1, 2, 0, 1]])
    cell_indices = np.ravel_multi_index(coordinates.T, n_cells_per_dim)

    projections = {
        pair: project_repertoire_snapshot(
            cell_indices=cell_indices,
            metric_values=np.array([3.0, 7.0, 5.0]),
            n_cells_per_dim=n_cells_per_dim,
            descriptor_names=descriptor_names,
            descriptor_pair=pair,
            metric="fitness",
        )
        for pair in descriptor_pairs(descriptor_names)
    }

    expected_pairs = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
    assert tuple(projections) == expected_pairs
    assert all(
        projection.shape == tuple(n_cells_per_dim[descriptor_names.index(name)] for name in pair)
        for pair, projection in projections.items()
    )
    assert projections[("a", "b")][0, 1] == 7.0


def test_aggregate_seed_projections_averages_counts_and_available_fitness() -> None:
    first_projection = np.array([[1.0, np.nan], [3.0, 5.0]])
    second_projection = np.array([[2.0, 4.0], [np.nan, 7.0]])

    fitness = aggregate_seed_projections((first_projection, second_projection), metric="fitness")
    selection_counts = aggregate_seed_projections(
        (np.array([[1.0, 0.0], [3.0, 5.0]]), np.array([[2.0, 4.0], [0.0, 7.0]])),
        metric="selection_count",
    )

    assert np.allclose(fitness, np.array([[1.5, 4.0], [3.0, 6.0]]), equal_nan=True)
    assert np.array_equal(selection_counts, np.array([[1.5, 2.0], [1.5, 6.0]]))


def test_descriptor_pairs_rejects_an_invalid_selection_size() -> None:
    with pytest.raises(ValueError, match="between two and four"):
        descriptor_pairs(("a",))
