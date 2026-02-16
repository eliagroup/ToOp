# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Plotting functions for the Map-Elites algorithm."""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from beartype.typing import Optional
from jax import numpy as jnp
from matplotlib.axes import Axes
from qdax.custom_types import Fitness


def plot_repertoire_1d(
    fitnesses: Fitness,
    n_cells_per_dim: tuple[int, ...],
    descriptor_metrics: tuple[str, ...],
) -> plt.Figure:
    """Plot a bar chart of the repertoire's fitnesses.

    Parameters
    ----------
    fitnesses : Fitness
        The fitnesses of the repertoire
    n_cells_per_dim : tuple[int, ...]
        The number of cells per dimension
    descriptor_metrics : tuple[str, ...]
        The descriptor metrics

    Returns
    -------
    plt.Figure
        The plot
    """
    plt.bar(
        x=range(n_cells_per_dim[0]),
        height=fitnesses,
    )
    plt.xlabel(descriptor_metrics[0])
    plt.ylabel("Fitness")
    return plt


def plot_repertoire_2d(
    fitnesses: Fitness,
    n_cells_per_dim: tuple[int, ...],
    descriptor_metrics: tuple[str, ...],
) -> Axes:
    """Plot a heatmap of the repertoire's fitnesses.

    Parameters
    ----------
    fitnesses : Fitness
        The fitnesses of the repertoire
    n_cells_per_dim : tuple[int, ...]
        The number of cells per dimension
    descriptor_metrics : tuple[str, ...]
        The descriptor metrics

    Returns
    -------
    Axes
        The plot
    """
    reshaped_fitnesses = fitnesses.reshape(n_cells_per_dim)
    ax = sns.heatmap(
        reshaped_fitnesses,
        cbar_kws={"label": "Fitness"},
        linewidths=0.01,
        linecolor=(0.3, 0.3, 0.3, 0.3),
    )  # vmin=-15000, vmax=-5000
    ax.invert_yaxis()
    ax.set_xlabel(descriptor_metrics[1])  # first axis is vertical for some reason
    ax.set_ylabel(descriptor_metrics[0])
    return ax


def plot_repertoire(
    fitnesses: Fitness,
    iteration: Optional[int],
    folder: Optional[str],
    n_cells_per_dim: tuple[int, ...],
    descriptor_metrics: tuple[str, ...],
    save_plot: bool,
) -> Axes | plt.Figure:
    """Plot the repertoire (1D or 2D) and saves the figure.

    Parameters
    ----------
    fitnesses : Fitness
        The fitnesses of the repertoire
    iteration : Optional[int]
        The current iteration number
    folder : str
        The folder to save the plot. Will create a "plots" folder inside.
    n_cells_per_dim : tuple[int, ...]
        The number of cells per dimension
    descriptor_metrics : tuple[str, ...]
        The descriptor metrics
    save_plot : bool
        Whether to save the plot

    Returns
    -------
    Axes | plt.Figure
        The plot. Axes for heatmap, plt.Figure for barchart.
    """
    plt.clf()  # clear or it plots on top of itself

    # Prepare fitnesses : convert neginf to minimum fitness * weight
    weight = 1.05  # Any number greater than 1, can fine-tune to your liking
    minimum_fitness = jnp.min(fitnesses, where=jnp.isfinite(fitnesses), initial=0)
    fitnesses = jnp.nan_to_num(fitnesses, neginf=weight * minimum_fitness)

    # plot in N dimensions
    if len(n_cells_per_dim) == 1:  # can replace with plotting_fn = plot_repertoire_1d
        plot = plot_repertoire_1d(fitnesses, n_cells_per_dim, descriptor_metrics)
    elif len(n_cells_per_dim) == 2:
        plot = plot_repertoire_2d(fitnesses, n_cells_per_dim, descriptor_metrics)
    else:
        raise ValueError(
            "Only 1D and 2D plots are supported"
        )  # TODO : 3D heatmaps https://www.geeksforgeeks.org/3d-heatmap-in-python/

    # add iteration to the title
    if iteration is not None:
        plt.title(f"Map-Elites repertoire iteration {iteration}")
    else:
        plt.title("Map-Elites repertoire")

    # save the figure
    if save_plot:
        folder = os.path.join(folder, "plots")
        os.makedirs(folder, exist_ok=True)
        filename = "repertoire.png" if iteration is None else f"repertoire_{iteration}.png"
        plt.savefig(os.path.join(folder, filename))
    return plot
