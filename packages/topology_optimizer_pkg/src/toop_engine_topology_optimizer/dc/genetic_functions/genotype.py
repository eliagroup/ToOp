# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Dataclass for the genotype used in the genetic algorithm."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Int
from toop_engine_dc_solver.jax.types import NodalInjOptimResults, int_max


class Genotype(eqx.Module):
    """A single genome in the repertoire representing a topology."""

    action_index: Int[Array, " *batch_size max_num_splits"]
    """An action index into the action set"""

    disconnections: Int[Array, " *batch_size max_num_disconnections"]
    """The disconnections to apply, padded with int_max for disconnection slots that are unused.
    These are indices into the disconnectable branches set."""

    nodal_injections_optimized: Optional[NodalInjOptimResults]
    """The results of the nodal injection optimization, if any was performed."""


def deduplicate_genotypes(
    genotypes: Genotype,
    desired_size: Optional[int] = None,
) -> tuple[Genotype, Int[Array, " n_unique"]]:
    """Deduplicate the genotypes in the repertoire.

    This version is jittable because we set the size

    Parameters
    ----------
    genotypes : Genotype
        The genotypes to deduplicate. These should be sorted by action id already.
    desired_size : Optional[int]
        How many unique values you are expecting. If not given, this is not jittable

    Returns
    -------
    Genotype
        The deduplicated genotypes
    Int[Array, " n_unique"]
        The indices of the unique genotypes
    """
    # Use the action indices and the disconnections for the uniqueness check.
    genotype_parts = [
        genotypes.action_index,
        genotypes.disconnections,
    ]
    # Include nodal_injections_optimized (PST taps) in deduplication when present
    if genotypes.nodal_injections_optimized is not None:
        # Flatten the nodal injection optimization results into the comparison
        # Shape: (batch_size, n_timesteps, n_controllable_pst) -> (batch_size, n_timesteps * n_controllable_pst)
        pst_taps_flat = genotypes.nodal_injections_optimized.pst_tap_idx.reshape(
            genotypes.nodal_injections_optimized.pst_tap_idx.shape[0], -1
        )
        genotype_parts.append(pst_taps_flat)

    genotype_flat = jnp.concatenate(genotype_parts, axis=1)

    _, indices = jnp.unique(
        genotype_flat,
        axis=0,
        return_index=True,
        size=desired_size,
        # fill_value takes the minimum flattened topology by default
        # it also corresponds to the first index in the list
    )
    unique_genotypes = jax.tree_util.tree_map(lambda x: x[indices], genotypes)
    return unique_genotypes, indices


def fix_dtypes(genotypes: Genotype) -> Genotype:
    """Fix the dtypes of the genotypes to their native type.

    For some reason, qdax aggressively converts everything to float

    Parameters
    ----------
    genotypes : Genotype
        The genotypes to fix

    Returns
    -------
    Genotype
        The genotypes with fixed dtypes
    """
    return Genotype(
        action_index=genotypes.action_index.astype(int),
        disconnections=genotypes.disconnections.astype(int),
        nodal_injections_optimized=genotypes.nodal_injections_optimized,
    )


def empty_repertoire(
    batch_size: int,
    max_num_splits: int,
    max_num_disconnections: int,
    n_timesteps: int,
    starting_taps: Optional[Int[Array, " num_psts"]] = None,
) -> Genotype:
    """Create an initial genotype repertoire with all zeros for all entries and int_max for all subs.

    Parameters
    ----------
    batch_size : int
        The batch size
    max_num_splits : int
        The maximum number of splits per topology
    max_num_disconnections : int
        The maximum number of disconnections as topological measures per topology
    n_timesteps : int
        The number of timesteps in the optimization horizon, used for the nodal injection optimization results
    starting_taps : Optional[Int[Array, " num_psts"]]
        The starting taps for the psts. If None, no nodal inj optimization will be enabled and nodal_injections_optimized
        will be set to None. If provided, nodal_injections_optimized will be initialized with these starting taps.

    Returns
    -------
    Genotype
        The initial genotype
    """
    if starting_taps is not None:
        nodal_injections_optimized = NodalInjOptimResults(
            pst_tap_idx=jnp.tile(starting_taps[None, None, :], (batch_size, n_timesteps, 1))
        )
    else:
        nodal_injections_optimized = None

    return Genotype(
        action_index=jnp.full((batch_size, max_num_splits), int_max(), dtype=int),
        disconnections=jnp.full((batch_size, max_num_disconnections), int_max(), dtype=int),
        nodal_injections_optimized=nodal_injections_optimized,
    )
