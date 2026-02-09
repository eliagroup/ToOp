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

import jax
from average import EWMA
from beartype.typing import Optional
from jax import numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.standard_emitters import EmitterState, ExtraScores, MixingEmitter
from qdax.custom_types import Descriptor, Fitness, Genotype


class TrackingMixingEmitter(MixingEmitter):
    """A MixingEmitter that tracks the number of branch and injection combinations and splits."""

    def init(
        self,
        random_key: jax.random.PRNGKey,
        init_genotypes: Optional[Genotype],  # noqa: ARG002
    ) -> tuple[EmitterState, jax.random.PRNGKey]:
        """Overwrite the Emitter.init function to seed an EmitterState."""
        return {
            "total_branch_combis": jnp.array(0, dtype=int),
            "total_inj_combis": jnp.array(0, dtype=int),
            "total_num_splits": jnp.array(0, dtype=int),
        }, random_key

    def state_update(
        self,
        emitter_state: Optional[EmitterState],
        repertoire: Optional[Repertoire],  # noqa: ARG002
        genotypes: Optional[Genotype],  # noqa: ARG002
        fitnesses: Optional[Fitness],  # noqa: ARG002
        descriptors: Optional[Descriptor],  # noqa: ARG002
        extra_scores: Optional[ExtraScores],
    ) -> EmitterState:
        """Overwrite the state update to store information for the running means."""
        assert emitter_state is not None
        assert extra_scores is not None
        return {
            "total_branch_combis": emitter_state["total_branch_combis"] + extra_scores["n_branch_combis"].astype(int),
            "total_inj_combis": emitter_state["total_inj_combis"] + extra_scores["n_inj_combis"].astype(int),
            "total_num_splits": emitter_state["total_num_splits"] + extra_scores["n_split_grids"].astype(int),
        }


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
        else {
            "total_branch_combis": jnp.array(0, dtype=int),
            "total_inj_combis": jnp.array(0, dtype=int),
            "total_num_splits": jnp.array(0, dtype=int),
        }
    )

    branch_diff = emitter_state["total_branch_combis"].item() - last_emitter_state["total_branch_combis"].item()
    inj_diff = emitter_state["total_inj_combis"].item() - last_emitter_state["total_inj_combis"].item()
    split_diff = emitter_state["total_num_splits"].item() - last_emitter_state["total_num_splits"].item()

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


def make_description_string(running_means: RunningMeans, runtime_seconds: float) -> str:
    """Create a string representation of the running means for the command line tqdm output.

    Parameters
    ----------
    running_means : RunningMeans
        The running means to be displayed
    runtime_seconds : float
        The total runtime of the optimization in seconds

    Returns
    -------
    str
        The string representation of the running means
    """
    now = time.time()

    loadflows = running_means.total_inj_combis * running_means.n_outages
    return (
        f"br/s: {running_means.br_per_sec.get():.2f}, "
        + f"inj/s: {running_means.inj_per_sec.get():.2f}, "
        + f"lfs: {loadflows:.2e}, "
        + f"time: {(now - running_means.start_time):.0f}/{runtime_seconds:.0f}s"
    )
