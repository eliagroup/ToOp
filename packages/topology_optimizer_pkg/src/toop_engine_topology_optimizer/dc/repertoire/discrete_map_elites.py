"""Core components of the MAP-Elites algorithm.

Adapted from QDax (https://github.com/adaptive-intelligent-robotics/QDax)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from pydantic import PositiveInt
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
    add_to_repertoire,
    init_repertoire,
)

EmitterScores = PyTree


class DiscreteMapElites:
    """Discrete MAP-Elites algorithm."""

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey, PyTree],
            Tuple[Fitness, Descriptor, ExtraScores, EmitterScores, RNGKey],
        ],
        emitter: Emitter,
        metrics_function: Callable[[DiscreteMapElitesRepertoire], Metrics],
        n_cells_per_dim: tuple[int, ...],
        cell_depth: PositiveInt = 1,
        distributed: bool = False,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._distributed = distributed
        self._n_cells_per_dim = n_cells_per_dim
        self._cell_depth = cell_depth

    def init(
        self,
        genotypes: Genotype,
        random_key: RNGKey,
        scoring_data: PyTree,
    ) -> Tuple[DiscreteMapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """Initialize a Map-Elites repertoire with an initial population of genotypes.

        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Before the repertoire is initialised, individuals are gathered from all the
        devices.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            random_key: a random key used for stochastic operations.
            n_cells_per_dim: number of cells per dimension in the repertoire

        Returns
        -------
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        (
            fitnesses,
            descriptors,
            extra_scores,
            emitter_scores,
            random_key,
        ) = self._scoring_function(genotypes, random_key, scoring_data)

        # gather across all devices
        if self._distributed:
            (
                genotypes,
                fitnesses,
                descriptors,
                extra_scores,
            ) = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
                (genotypes, fitnesses, descriptors, extra_scores),
            )

        # init the repertoire
        repertoire = init_repertoire(
            genotypes=genotypes,
            descriptors=descriptors,
            fitnesses=fitnesses,
            extra_scores=extra_scores,
            n_cells_per_dim=self._n_cells_per_dim,
            cell_depth=self._cell_depth,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            init_genotypes=genotypes,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=emitter_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        scoring_data: PyTree,
    ) -> Tuple[DiscreteMapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """Perform one iteration of the MAP-Elites algorithm.

        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Before the repertoire is updated, individuals are gathered from all the
        devices.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns
        -------
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, _extra_info, random_key = self._emitter.emit(repertoire, emitter_state, random_key)
        # scores the offsprings
        (
            fitnesses,
            descriptors,
            extra_scores,
            emitter_info,
            random_key,
        ) = self._scoring_function(genotypes, random_key, scoring_data)

        # gather across all devices
        if self._distributed:
            (
                genotypes,
                fitnesses,
                descriptors,
                extra_scores,
            ) = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
                (genotypes, fitnesses, descriptors, extra_scores),
            )

        # add genotypes in the repertoire
        repertoire = add_to_repertoire(
            repertoire=repertoire,
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=emitter_info,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
