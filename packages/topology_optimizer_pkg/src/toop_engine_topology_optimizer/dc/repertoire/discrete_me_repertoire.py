"""Contains the DiscreteMapElitesRepertoire class and utility functions.

This file contains util functions and a class to define
a repertoire, used to store individuals in the MAP-Elites
algorithm as well as several variants. Adapted from QDax
(https://github.com/adaptive-intelligent-robotics/QDax)
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax_dataclasses import Static, pytree_dataclass
from jaxtyping import Array, Float, Int, PyTree, Shaped
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


@partial(jax.jit, static_argnames=("n_cells_per_dim",))
def get_cells_indices(
    descriptors: Int[Array, " batch_size *dim"],
    n_cells_per_dim: tuple[int, ...],
) -> Int[Array, " batch_size"]:
    """Compute the index for many descriptors at once.

    Args:
        descriptors: an array that contains the descriptors of the
            solutions. Its shape is (batch_size, num_descriptors)
        n_cells_per_dim: the number of cells per dimension

    Returns
    -------
        The index of the cell in which each descriptor belongs.
    """
    return jax.vmap(partial(get_cell_index, n_cells_per_dim=n_cells_per_dim))(descriptors)


@partial(jax.jit, static_argnames=("n_cells_per_dim",))
def get_cell_index(
    descriptor: Int[Array, " n_dims"],
    n_cells_per_dim: tuple[int, ...],
) -> Int[Array, " "]:
    """Compute the index of the cell in which each descriptor belongs.

    The index is effectively like the reshape operation, spreading multiple dimensions to a flat
    single dimension.

    Args:
        descriptor: an array that contains the descriptors. There is one descriptor per dimension
        n_cells_per_dim: the number of cells per dimension

    Returns
    -------
        The index of the cell in which each descriptor belongs.
    """
    assert len(descriptor) == len(n_cells_per_dim)
    assert all(c > 0 for c in n_cells_per_dim)

    return jnp.ravel_multi_index(
        multi_index=descriptor,
        dims=n_cells_per_dim,
        mode="clip",
    )  # jittable thanks to clip mode


@pytree_dataclass
class DiscreteMapElitesRepertoire:
    """A class to store the MAP-Elites repertoire."""

    genotypes: PyTree[Shaped[Array, " repertoire_size *feature_dims"]]
    """The genotypes in the repertoire.

    The PyTree can be a simple Jax array or a more complex nested structure
    such as to represent parameters of neural network in Flax."""

    fitnesses: Float[Array, " repertoire_size"]
    """The fitness of solutions in each cell of the repertoire."""

    descriptors: Int[Array, " repertoire_size n_dims"]
    """The descriptors of solutions in each cell of the repertoire."""

    extra_scores: Optional[PyTree[Float[Array, " repertoire_size *extra_dims"]]]
    """The extra scores of solutions in each cell of the repertoire. Usually the metrics"""

    n_cells_per_dim: Static[tuple[int, ...]]
    """The number of cells per dimension."""

    cell_depth: Static[int] = 1
    """Each cell contains cell_depth unique individuals"""

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self: DiscreteMapElitesRepertoire, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample elements in the repertoire.

        Parameters
        ----------
        random_key: jax PRNG key
            the random key to be used for sampling
        num_samples: int
            the number of samples to be drawn from the repertoire

        Returns
        -------
        samples: Genotype
            the sampled genotypes
        random_key: jax PRNG key
            the updated random key
        """
        repertoire_empty = self.fitnesses == -jnp.inf
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        return samples, random_key

    def __getitem__(self, index: Union[int, slice, jnp.ndarray]) -> DiscreteMapElitesRepertoire:
        """Get a slice of the repertoire.

        Parameters
        ----------
        index: int, slice or jnp.ndarray
            the index of the elements to be selected

        Returns
        -------
            a new repertoire with the selected elements
        """
        return DiscreteMapElitesRepertoire(
            genotypes=jax.tree_util.tree_map(lambda x: x[index], self.genotypes),
            fitnesses=self.fitnesses[index],
            descriptors=self.descriptors[index],
            extra_scores=jax.tree_util.tree_map(lambda x: x[index], self.extra_scores),
            n_cells_per_dim=self.n_cells_per_dim,
            cell_depth=self.cell_depth,
        )


@jax.jit
def add_to_repertoire(
    repertoire: DiscreteMapElitesRepertoire,
    batch_of_genotypes: PyTree[Shaped[Array, " batch_size *feature_dims"]],
    batch_of_descriptors: Int[Array, " batch_size n_dims"],
    batch_of_fitnesses: Float[Array, " batch_size"],
    batch_of_extra_scores: Optional[PyTree[Shaped[Array, " batch_size *extra_dims"]]] = None,
) -> DiscreteMapElitesRepertoire:
    """Add a batch of elements to the repertoire.

    Args:
        repertoire: the MAP-Elites repertoire to which the elements will be added.
        batch_of_genotypes: a batch of genotypes to be added to the repertoire.
            Similarly to the repertoire.genotypes argument, this is a PyTree in which
            the leaves have a shape (batch_size, num_features)
        batch_of_descriptors: an array that contains the descriptors of the
            aforementioned genotypes. Its shape is (batch_size, num_descriptors). This will
            determine the cell in which the genotype will be stored, all descriptors are clipped
            to be within the bounds of n_cells_per_dim.
        batch_of_fitnesses: an array that contains the fitnesses of the
            aforementioned genotypes. Its shape is (batch_size,)
        batch_of_extra_scores: tree that contains the extra_scores of
            aforementioned genotypes.

    Returns
    -------
        The updated MAP-Elites repertoire.
    """
    if repertoire.cell_depth > 1:
        return add_to_repertoire_with_cell_depth(
            repertoire,
            batch_of_genotypes,
            batch_of_descriptors,
            batch_of_fitnesses,
            batch_of_extra_scores,
        )
    return add_to_repertoire_without_cell_depth(
        repertoire,
        batch_of_genotypes,
        batch_of_descriptors,
        batch_of_fitnesses,
        batch_of_extra_scores,
    )


@jax.jit
def add_to_repertoire_without_cell_depth(
    repertoire: DiscreteMapElitesRepertoire,
    batch_of_genotypes: PyTree[Shaped[Array, " batch_size *feature_dims"]],
    batch_of_descriptors: Int[Array, " batch_size n_dims"],
    batch_of_fitnesses: Float[Array, " batch_size"],
    batch_of_extra_scores: Optional[PyTree[Shaped[Array, " batch_size *extra_dims"]]] = None,
) -> DiscreteMapElitesRepertoire:
    """Add a batch of elements to the repertoire.

    Parameters
    ----------
    repertoire: DiscreteMapElitesRepertoire
        the MAP-Elites repertoire to which the elements will be added.
    batch_of_genotypes: PyTree[Shaped[Array, " batch_size *feature_dims"]]
        a batch of genotypes to be added to the repertoire.
    batch_of_descriptors: Int[Array, " batch_size n_dims"]
        an array that contains the descriptors of the genotypes.
    batch_of_fitnesses: Float[Array, " batch_size"]
        an array that contains the fitnesses of the genotypes.
    batch_of_extra_scores: Optional[PyTree[Shaped[Array, " batch_size *extra_dims"]]]
        tree that contains the extra_scores of genotypes.

    Returns
    -------
    DiscreteMapElitesRepertoire
        The updated MAP-Elites repertoire.
    """
    batch_of_indices = get_cells_indices(batch_of_descriptors, repertoire.n_cells_per_dim)
    batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
    batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)
    repertoire_size = repertoire.fitnesses.shape[0]

    # get fitness segment max
    # Necessary because there could be multiple scores with the same descriptor
    best_fitnesses = jax.ops.segment_max(
        batch_of_fitnesses,
        batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
        num_segments=repertoire_size,
    )

    cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

    # put dominated fitness to -jnp.inf
    batch_of_fitnesses = jnp.where(batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf)

    # get addition condition
    repertoire_fitnesses = jnp.expand_dims(repertoire.fitnesses, axis=-1)
    current_fitnesses = jnp.take_along_axis(repertoire_fitnesses, batch_of_indices, 0)
    addition_condition = batch_of_fitnesses > current_fitnesses

    # assign fake position when relevant : num_centroids is out of bound
    batch_of_indices = jnp.where(addition_condition, batch_of_indices, repertoire_size).squeeze(axis=-1)

    # create new repertoire
    new_repertoire_genotypes = jax.tree_util.tree_map(
        lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[batch_of_indices].set(
            new_genotypes, mode="drop"
        ),
        repertoire.genotypes,
        batch_of_genotypes,
    )

    # compute new fitness and descriptors
    new_fitnesses = repertoire.fitnesses.at[batch_of_indices].set(batch_of_fitnesses.squeeze(axis=-1), mode="drop")
    new_descriptors = repertoire.descriptors.at[batch_of_indices].set(batch_of_descriptors, mode="drop")
    if batch_of_extra_scores is None:
        new_extra_scores = None
    else:
        new_extra_scores = jax.tree_util.tree_map(
            lambda repertoire_extra_scores, new_extra_scores: repertoire_extra_scores.at[batch_of_indices].set(
                new_extra_scores, mode="drop"
            ),
            repertoire.extra_scores,
            batch_of_extra_scores,
        )

    return DiscreteMapElitesRepertoire(
        genotypes=new_repertoire_genotypes,
        fitnesses=new_fitnesses,
        descriptors=new_descriptors,
        extra_scores=new_extra_scores,
        n_cells_per_dim=repertoire.n_cells_per_dim,
        cell_depth=repertoire.cell_depth,
    )


@jax.jit
def add_to_repertoire_with_cell_depth(
    repertoire: DiscreteMapElitesRepertoire,
    batch_of_genotypes: PyTree[Shaped[Array, " batch_size *feature_dims"]],
    batch_of_descriptors: Int[Array, " batch_size n_dims"],
    batch_of_fitnesses: Float[Array, " batch_size"],
    batch_of_extra_scores: Optional[PyTree[Shaped[Array, " batch_size *extra_dims"]]] = None,
) -> DiscreteMapElitesRepertoire:
    """Add a batch of elements to a repertoire with depth.

    Assumes the repertoire puts elements on a same depth level next to another (layer1 layer1 layer2 layer2 ...)
    Manipulates abstract indices instead of moving genotypes directly.

    Parameters
    ----------
    repertoire: DiscreteMapElitesRepertoire
        the MAP-Elites repertoire to which the elements will be added.
    batch_of_genotypes: PyTree[Shaped[Array, " batch_size *feature_dims"]]
        a batch of genotypes to be added to the repertoire.
    batch_of_descriptors: Int[Array, " batch_size n_dims"]
        an array that contains the descriptors of the genotypes.
    batch_of_fitnesses: Float[Array, " batch_size"]
        an array that contains the fitnesses of the genotypes.
    batch_of_extra_scores: Optional[PyTree[Shaped[Array, " batch_size *extra_dims"]]]
        tree that contains the extra_scores of genotypes.

    Returns
    -------
    DiscreteMapElitesRepertoire
        The updated MAP-Elites repertoire.
    """
    cell_depth = repertoire.cell_depth
    num_cells = np.prod(np.array(repertoire.n_cells_per_dim)).item()
    repertoire_size = cell_depth * num_cells
    batch_size = batch_of_fitnesses.shape[0]

    # Work on indices in order to abstract the genotype
    abstract_current_genotypes = jnp.arange(repertoire_size)
    abstract_new_genotypes = jnp.arange(start=repertoire_size, stop=repertoire_size + batch_size)

    """
    Part 1 : Sort
    """

    # rearrange repertoire fitnesses so cell fitnesses are in the same dimension
    current_fitnesses_per_cell: Float[Array, "num_cells cell_depth"] = jnp.reshape(
        repertoire.fitnesses,
        (-1, num_cells),  # one dimension per cell
    ).T  # transpose to have cells in the first dimension

    # calculate new indices for each layer
    new_indices: Int[Array, "batch_size"] = get_cells_indices(
        batch_of_descriptors, repertoire.n_cells_per_dim
    )  # example : [0, 0, 0, 1]

    # repeat to get :
    # [[0, 0, 0, 1],
    #  [0, 0, 0, 1],
    #  [0, 0, 0, 1]]
    # shape (num_cells, batch_size)
    repeated_indices: Int[Array, "num_cells batch_size"] = jnp.tile(new_indices, reps=(num_cells, 1))

    # Create an array of shape (num_cells, batch_size) where the nth element contains the number n
    # example : [[0, 0, 0, 0],
    #            [1, 1, 1, 1],
    #            [2, 2, 2, 2]]
    cell_number: Int[Array, "num_cells"] = jnp.arange(num_cells)
    repeated_cell_number: Int[Array, "num_cells batch_size"] = jnp.repeat(
        cell_number.reshape(-1, 1),
        repeats=batch_size,
        axis=-1,
        total_repeat_length=batch_size,  # to make it jittable
    )

    # Repeat new fitnesses
    repeated_new_fitnesses: Float[Array, "num_cells batch_size"] = jnp.tile(batch_of_fitnesses, reps=(num_cells, 1))

    # condition is true if the fitness is at the right index
    # example : [[True , True , True , False],
    #            [False, False, False, True ],
    #            [False, False, False, False]]
    belongs_to_right_cell = repeated_cell_number == repeated_indices

    filtered_new_fitnesses_per_cell: Float[Array, "num_cells batch_size"] = jnp.where(
        belongs_to_right_cell,
        repeated_new_fitnesses,
        -jnp.inf,
    )

    # extend current fitnesses per cell to welcome new ones
    all_fitnesses_per_cell: Float[Array, "num_cells max_size_per_cell"] = jnp.concatenate(
        [current_fitnesses_per_cell, filtered_new_fitnesses_per_cell], axis=1
    )

    sorted_fitness_indices_per_cell: Int[Array, "num_cells max_size_per_cell"] = jnp.argsort(
        all_fitnesses_per_cell, axis=-1, descending=True
    )

    # we only want the first cell_depth elements of each cell
    cropped_best_fitness_indices_per_cell: Int[Array, "num_cells cell_depth"] = sorted_fitness_indices_per_cell[
        :, :cell_depth
    ]

    # adapt indices to select genotypes : must be laid flat with first num_cells belonging to the first depth etc
    # adapted_indices_for_selection = cropped_best_fitness_indices_per_cell.T.reshape(-1)

    # Shape genotypes per-cell to use the indices
    current_genotypes_per_cell: Int[Array, "num_cells cell_depth"] = abstract_current_genotypes.reshape(-1, num_cells).T
    repeated_batch_of_genotypes: Int[Array, "num_cells batch_size"] = (
        jnp.repeat(abstract_new_genotypes, repeats=num_cells).reshape(-1, num_cells).T
    )
    all_genotypes_per_cell: Int[Array, "num_cells max_size_per_cell"] = jnp.concatenate(
        [current_genotypes_per_cell, repeated_batch_of_genotypes],
        axis=1,  # concatenate on cells
    )

    # select the genotypes by order of fitness. If a cell doesn't belong there,
    # it will be placed last (-inf fitness) and thus cropped out
    cropped_selected_genotypes_per_cell: Int[Array, "num_cells cell_depth"] = jnp.take_along_axis(
        all_genotypes_per_cell, cropped_best_fitness_indices_per_cell, axis=1
    )

    # reshape to the original shape to apply on a concat of repertoire and new batch
    final_indices_selection: Int[Array, "repertoire_size"] = cropped_selected_genotypes_per_cell.T.reshape(
        num_cells * cell_depth
    )

    """
    Part 2 : Selection
    """

    # genotypes
    selected_genotypes = jax.tree.map(
        lambda x, y: jnp.concat([x, y])[final_indices_selection],
        repertoire.genotypes,
        batch_of_genotypes,
    )

    # fitness
    selected_fitness = jnp.concat([repertoire.fitnesses, batch_of_fitnesses])[final_indices_selection]

    # descriptors
    selected_descriptors = jnp.concat([repertoire.descriptors, batch_of_descriptors], axis=0)[final_indices_selection]

    # extra scores
    if batch_of_extra_scores is None:
        selected_extra_scores = None
    else:
        selected_extra_scores = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0)[final_indices_selection],
            repertoire.extra_scores,
            batch_of_extra_scores,
        )

    return DiscreteMapElitesRepertoire(
        genotypes=selected_genotypes,
        fitnesses=selected_fitness,
        descriptors=selected_descriptors,
        extra_scores=selected_extra_scores,
        n_cells_per_dim=repertoire.n_cells_per_dim,
        cell_depth=cell_depth,
    )


def init_repertoire(
    genotypes: Genotype,
    fitnesses: Fitness,
    descriptors: Descriptor,
    extra_scores: Optional[ExtraScores],
    n_cells_per_dim: tuple[int, ...],
    cell_depth: Static[int],
) -> DiscreteMapElitesRepertoire:
    """Initialize a Map-Elites repertoire with an initial population of genotypes.

    Requires the definition of centroids that can be computed with any method
    such as CVT or Euclidean mapping.

    Note: this function has been kept outside of the object MapElites, so it can
    be called easily called from other modules.

    Args:
        genotypes: initial genotypes, pytree in which leaves
            have shape (batch_size, num_features)
        fitnesses: fitness of the initial genotypes of shape (batch_size,)
        descriptors: descriptors of the initial genotypes
            of shape (batch_size, num_descriptors)
        extra_scores: the observed load flow metrics
        n_cells_per_dim: the number of cells per dimension
        cell_depth: the number of topologies per cell

    Returns
    -------
        an initialized MAP-Elite repertoire
    """
    # retrieve one genotype from the population
    (first_genotype, first_extra_score) = jax.tree_util.tree_map(lambda x: x[0], (genotypes, extra_scores))

    # create a repertoire with default values
    repertoire = _init_default(
        genotype=first_genotype,
        extra_scores=first_extra_score,
        n_cells_per_dim=n_cells_per_dim,
        cell_depth=cell_depth,
    )

    # add initial population to the repertoire
    new_repertoire = add_to_repertoire(repertoire, genotypes, descriptors, fitnesses, extra_scores)

    return new_repertoire  # type: ignore


def _init_default(
    genotype: Genotype,
    extra_scores: Optional[ExtraScores],
    n_cells_per_dim: tuple[int, ...],
    cell_depth: Static[int],
) -> DiscreteMapElitesRepertoire:
    """Initialize a Map-Elites repertoire.

    Initialization with an initial population of
    genotypes. Requires the definition of centroids that can be computed
    with any method such as CVT or Euclidean mapping.

    Note: this function has been kept outside of the object MapElites, so
    it can be called easily called from other modules.

    Args:
        genotype: the typical genotype that will be stored.
        extra_scores: the observed load flow metrics
        n_cells_per_dim: the number of cells per dimension
        cell_depth: the number of topologies per cell

    Returns
    -------
        A repertoire filled with default values.
    """
    # get repertoire size
    repertoire_size = np.prod(n_cells_per_dim).item() * cell_depth

    # default fitness is -inf
    default_fitnesses = jnp.full(shape=repertoire_size, fill_value=-jnp.inf, dtype=float)

    # default genotypes is copied from the given genotype
    default_genotypes = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], repeats=repertoire_size, axis=0), genotype)

    # default descriptor is all zeros
    default_descriptors = jnp.zeros(shape=(repertoire_size, len(n_cells_per_dim)), dtype=int)

    # default extra scores is all 0
    if extra_scores is None:
        default_extra_scores = None
    else:
        default_extra_scores = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(repertoire_size, *x.shape), dtype=x.dtype),
            extra_scores,
        )

    return DiscreteMapElitesRepertoire(
        genotypes=default_genotypes,
        fitnesses=default_fitnesses,
        descriptors=default_descriptors,
        extra_scores=default_extra_scores,
        n_cells_per_dim=n_cells_per_dim,
        cell_depth=cell_depth,
    )
