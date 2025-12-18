# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os

import jax.numpy as jnp
from toop_engine_topology_optimizer.dc.repertoire.plotting import plot_repertoire


def test_plot_repertoire_1D(tmp_path: str):
    fitnesses = jnp.array([1, 2, 3, -jnp.inf])
    iteration = 1
    folder = str(tmp_path)
    n_cells_per_dim = (4,)
    descriptor_metrics = ("switching_distance",)
    plot = plot_repertoire(fitnesses, iteration, folder, n_cells_per_dim, descriptor_metrics, True)
    assert plot.gca().get_xlabel() == "switching_distance"
    assert os.path.exists(os.path.join(folder, "plots", f"repertoire_{iteration}.png")), "Figure does not exist."


def test_plot_repertoire_2D(tmp_path: str):
    fitnesses = jnp.array([1, 2, 3, 4, 5, 6, 7, -jnp.inf])
    iteration = None  # Might as well test the default value
    n_cells_per_dim = (
        2,
        4,
    )
    folder = str(tmp_path)
    descriptor_metrics = ("switching_distance", "num_splits_grid")
    ax = plot_repertoire(fitnesses, iteration, folder, n_cells_per_dim, descriptor_metrics, True)
    assert ax.get_ylabel() == "switching_distance"  # y label is the first
    assert ax.get_xlabel() == "num_splits_grid"
    ax.get_figure().clear()
    assert os.path.exists(os.path.join(folder, "plots", "repertoire.png")), "Figure does not exist."


def test_plot_repertoire_bad_dimensions(tmp_path: str):
    folder = str(tmp_path)
    fitnesses = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    n_cells_per_dim = (
        2,
        4,
        8,
    )
    descriptor_metrics = ("switching_distance", "num_splits_grid", "num_disconnections")
    try:
        plot_repertoire(fitnesses, None, folder, n_cells_per_dim, descriptor_metrics, True)
    except ValueError as e:
        assert str(e) == "Only 1D and 2D plots are supported"
    else:
        assert False, "ValueError not raised"
