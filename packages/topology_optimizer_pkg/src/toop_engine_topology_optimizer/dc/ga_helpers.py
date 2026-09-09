# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for genetic algorithms."""

import time
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import jax
from average import EWMA
from beartype.typing import Callable, Optional
from flax import struct
from jax import numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.standard_emitters import EmitterState, ExtraScores, MixingEmitter
from qdax.custom_types import Descriptor, Fitness
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire
from toop_engine_topology_optimizer.dc.repertoire.parent_selection import (
    ParentSelector,
    ParentSelectorState,
    UniformParentSelector,
    repertoire_indices_to_cell_indices,
)


def update_parent_selection_telemetry(
    selection_counts: Int[Array, " n_current_cells"],
    success_counts: Int[Array, " n_current_cells"],
    repertoire: Repertoire | DiscreteMapElitesRepertoire,
    extra_scores: ExtraScores,
) -> tuple[Int[Array, " n_cells"], Int[Array, " n_cells"]]:
    """Accumulate policy-neutral parent-selection feedback per logical cell.

    Parameters
    ----------
    selection_counts : Int[Array, " n_cells"]
        Existing number of parent selections attributed to each logical cell.
    success_counts : Int[Array, " n_cells"]
        Existing number of surviving offspring attributed to each logical cell.
    repertoire : Repertoire | DiscreteMapElitesRepertoire
        Current depth-expanded MAP-Elites repertoire.
    extra_scores : ExtraScores
        Parent indices emitted in the prior iteration and the corresponding survival mask.

    Returns
    -------
    tuple[Int[Array, " n_cells"], Int[Array, " n_cells"]]
        Updated cumulative parent-selection and offspring-survival counts.
    """
    cell_depth = repertoire.cell_depth
    num_cells = repertoire.fitnesses.shape[0] // cell_depth
    if selection_counts.shape[0] != num_cells:
        selection_counts = jnp.zeros((num_cells,), dtype=int)
        success_counts = jnp.zeros((num_cells,), dtype=int)

    mutation_parent_indices = extra_scores.get("mutation_parent_indices")
    crossover_parent_indices_a = extra_scores.get("crossover_parent_indices_a")
    crossover_parent_indices_b = extra_scores.get("crossover_parent_indices_b")
    survived_mask = extra_scores.get("survived_mask")
    if survived_mask is None:
        return selection_counts, success_counts

    n_crossover = 0 if crossover_parent_indices_a is None else crossover_parent_indices_a.shape[0]
    n_mutation = 0 if mutation_parent_indices is None else mutation_parent_indices.shape[0]
    crossover_survived = survived_mask[:n_crossover].astype(int)
    mutation_survived = survived_mask[n_crossover : n_crossover + n_mutation].astype(int)

    if crossover_parent_indices_a is not None and crossover_parent_indices_b is not None:
        crossover_parent_cells_a = repertoire_indices_to_cell_indices(crossover_parent_indices_a, num_cells)
        crossover_parent_cells_b = repertoire_indices_to_cell_indices(crossover_parent_indices_b, num_cells)
        selection_counts = selection_counts.at[crossover_parent_cells_a].add(1)
        selection_counts = selection_counts.at[crossover_parent_cells_b].add(1)
        success_counts = success_counts.at[crossover_parent_cells_a].add(crossover_survived)
        success_counts = success_counts.at[crossover_parent_cells_b].add(crossover_survived)

    if mutation_parent_indices is not None:
        mutation_parent_cells = repertoire_indices_to_cell_indices(mutation_parent_indices, num_cells)
        selection_counts = selection_counts.at[mutation_parent_cells].add(1)
        success_counts = success_counts.at[mutation_parent_cells].add(mutation_survived)

    return selection_counts, success_counts


class MixingEmitterState(EmitterState):
    """The state of a MixingEmitter.

    It extends the EmitterState with additional fields to track the number of
    branch and injection combinations and splits.
    """

    total_branch_combis: jnp.ndarray
    total_inj_combis: jnp.ndarray
    total_num_splits: jnp.ndarray
    parent_selector_state: ParentSelectorState
    parent_selection_counts: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), dtype=int))
    parent_success_counts: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), dtype=int))


class TrackingMixingEmitter(MixingEmitter):
    """A MixingEmitter that tracks the number of branch and injection combinations and splits."""

    def __init__(
        self,
        mutation_fn: Callable[[Genotype, PRNGKeyArray], tuple[Genotype, PRNGKeyArray]],
        variation_fn: Callable[[Genotype, Genotype, PRNGKeyArray], tuple[Genotype, PRNGKeyArray]],
        variation_percentage: float,
        batch_size: int,
        parent_selector: ParentSelector | None = None,
    ) -> None:
        super().__init__(mutation_fn, variation_fn, variation_percentage, batch_size)
        self._parent_selector = parent_selector if parent_selector is not None else UniformParentSelector()

    def init(
        self,
        random_key: PRNGKeyArray,
        init_genotypes: Optional[Genotype],  # noqa: ARG002
    ) -> tuple[EmitterState, PRNGKeyArray]:
        """Overwrite the Emitter.init function to seed an EmitterState."""
        return MixingEmitterState(
            total_branch_combis=jnp.array(0, dtype=int),
            total_inj_combis=jnp.array(0, dtype=int),
            total_num_splits=jnp.array(0, dtype=int),
            parent_selector_state=self._parent_selector.init_state(),
            parent_selection_counts=jnp.zeros((0,), dtype=int),
            parent_success_counts=jnp.zeros((0,), dtype=int),
        ), random_key

    def state_update(
        self,
        emitter_state: Optional[EmitterState],
        repertoire: Optional[Repertoire | DiscreteMapElitesRepertoire],
        genotypes: Optional[Genotype],  # noqa: ARG002
        fitnesses: Optional[Fitness],  # noqa: ARG002
        descriptors: Optional[Descriptor],  # noqa: ARG002
        extra_scores: ExtraScores,
    ) -> EmitterState:
        """Overwrite the state update to store information for the running means."""
        assert emitter_state is not None
        assert repertoire is not None
        assert extra_scores is not None
        parent_selection_counts, parent_success_counts = update_parent_selection_telemetry(
            selection_counts=emitter_state.parent_selection_counts,
            success_counts=emitter_state.parent_success_counts,
            repertoire=repertoire,
            extra_scores=extra_scores,
        )
        return MixingEmitterState(
            total_branch_combis=emitter_state.total_branch_combis
            + extra_scores.get("n_branch_combis", jnp.array(0, dtype=int)).astype(int),
            total_inj_combis=emitter_state.total_inj_combis
            + extra_scores.get("n_inj_combis", jnp.array(0, dtype=int)).astype(int),
            total_num_splits=emitter_state.total_num_splits
            + extra_scores.get("n_split_grids", jnp.array(0, dtype=int)).astype(int),
            parent_selector_state=self._parent_selector.state_from_telemetry(
                parent_selector_state=emitter_state.parent_selector_state,
                selection_counts=parent_selection_counts,
                success_counts=parent_success_counts,
            ),
            parent_selection_counts=parent_selection_counts,
            parent_success_counts=parent_success_counts,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, ExtraScores, PRNGKeyArray]:
        """Emitter that performs both mutation and variation.

        Two batches of ``variation_percentage * batch_size`` genotypes are
        selected in the repertoire, copied, and crossed over to obtain new
        offspring. One batch of ``(1.0 - variation_percentage) * batch_size``
        genotypes is selected in the repertoire, copied, and mutated.

        This override preserves the upstream QDax emitter behavior while
        routing parent selection through the configured parent selector.

        Parameters
        ----------
        repertoire : DiscreteMapElitesRepertoire
            The MAP-Elites repertoire to select parents from.
        emitter_state : Optional[EmitterState]
            Emitter state containing the parent-selector state.
        random_key : PRNGKeyArray
            A JAX PRNG random key.

        Returns
        -------
        tuple[Genotype, ExtraScores, PRNGKeyArray]
            A batch of offspring, emitter extra scores, and a new JAX PRNG key.
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation
        parent_selector_state = (
            emitter_state.parent_selector_state if emitter_state is not None else self._parent_selector.init_state()
        )
        occupied_mask = repertoire.fitnesses != -jnp.inf

        emitter_extra_scores: ExtraScores = {}
        total_parent_samples = 2 * n_variation + n_mutation
        parent_indices, random_key = self._parent_selector.select_for_emission(
            occupied_mask=occupied_mask,
            parent_selector_state=parent_selector_state,
            random_key=random_key,
            num_samples=total_parent_samples,
            fitnesses=repertoire.fitnesses,
        )

        if n_variation > 0:
            parent_indices_a = parent_indices[:n_variation]
            x1 = repertoire.get_genotypes(parent_indices_a)
            parent_indices_b = parent_indices[n_variation : 2 * n_variation]
            x2 = repertoire.get_genotypes(parent_indices_b)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)
            emitter_extra_scores["crossover_parent_indices_a"] = parent_indices_a
            emitter_extra_scores["crossover_parent_indices_b"] = parent_indices_b

        if n_mutation > 0:
            mutation_parent_indices = parent_indices[2 * n_variation :]
            x1 = repertoire.get_genotypes(mutation_parent_indices)
            x_mutation, random_key = self._mutation_fn(x1, random_key)
            emitter_extra_scores["mutation_parent_indices"] = mutation_parent_indices

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, emitter_extra_scores, random_key


@dataclass
class RunningMeans:
    """A dataclass to hold the running means estimations for the TQDM progress bar."""

    start_time: float
    last_time: float
    time_step: float
    total_branch_combis: int
    total_inj_combis: int
    br_per_sec: EWMA
    inj_per_sec: EWMA
    split_per_iter: EWMA
    last_emitter_state: Optional[EmitterState]
    n_outages: int
    n_devices: int


def init_running_means(n_outages: int, n_devices: int) -> RunningMeans:
    """Initialize an empty RunningMeans object.

    Parameters
    ----------
    n_outages : int
        The number of outages
    n_devices : int
        The number of devices

    Returns
    -------
    RunningMeans
        The initialized running means
    """
    now = time.time()
    return RunningMeans(
        start_time=now,
        last_time=now,
        time_step=0.0,
        total_branch_combis=0,
        total_inj_combis=0,
        br_per_sec=EWMA(),
        inj_per_sec=EWMA(),
        split_per_iter=EWMA(),
        last_emitter_state=None,
        n_outages=n_outages,
        n_devices=n_devices,
    )


def update_running_means(running_means: RunningMeans, emitter_state: EmitterState) -> RunningMeans:
    """Aggregate the emitter state statistics into the running means.

    Parameters
    ----------
    running_means : RunningMeans
        The running means to be updated
    emitter_state : EmitterState
        The emitter state of the current iteration

    Returns
    -------
    RunningMeans
        The updated running means
    """
    now = time.time()
    running_means = deepcopy(running_means)
    emitter_state = jax.tree_util.tree_map(lambda x: x * running_means.n_devices, emitter_state)
    last_emitter_state = (
        running_means.last_emitter_state
        if running_means.last_emitter_state is not None
        else MixingEmitterState(
            total_branch_combis=jnp.array(0, dtype=int),
            total_inj_combis=jnp.array(0, dtype=int),
            total_num_splits=jnp.array(0, dtype=int),
            parent_selector_state=emitter_state.parent_selector_state,
            parent_selection_counts=emitter_state.parent_selection_counts,
            parent_success_counts=emitter_state.parent_success_counts,
        )
    )

    branch_diff = emitter_state.total_branch_combis.item() - last_emitter_state.total_branch_combis.item()
    inj_diff = emitter_state.total_inj_combis.item() - last_emitter_state.total_inj_combis.item()
    split_diff = emitter_state.total_num_splits.item() - last_emitter_state.total_num_splits.item()

    # Clip diffs due to sporadic integer overflows
    branch_diff = max(branch_diff, 0)
    inj_diff = max(inj_diff, 0)
    split_diff = max(split_diff, 0)

    running_means.total_branch_combis += branch_diff
    running_means.total_inj_combis += inj_diff

    running_means.time_step = now - running_means.last_time

    running_means.br_per_sec.update(branch_diff / running_means.time_step)
    running_means.inj_per_sec.update(inj_diff / running_means.time_step)
    running_means.split_per_iter.update(split_diff)

    running_means.last_time = now
    running_means.last_emitter_state = emitter_state

    return running_means
