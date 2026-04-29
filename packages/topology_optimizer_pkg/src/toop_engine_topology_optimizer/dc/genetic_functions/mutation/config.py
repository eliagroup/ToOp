# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Mutation configuration classes for the genetic algorithm."""

import equinox as eqx
from beartype.typing import Optional
from jaxtyping import Array, Int


class SubstationMutationConfig(eqx.Module):
    """Configuration for the substation mutation operation.

    The sum of all probabilities should be <= 1.0, the remaining probability will be used for no mutation.
    If all substations are already split, the add_split_prob will be ignored and the probabilities will be
    renormalized to sum to 1.
    """

    n_subs_mutated_lambda: float = eqx.field(static=True)

    add_split_prob: float = eqx.field(static=True)
    """The probability to split an additional substation. In case all substations are already split, this
    probability is ignored."""

    change_split_prob: float = eqx.field(static=True)
    """The probability to change an already split substation to a different split in the same or another substation."""

    remove_split_prob: float = eqx.field(static=True)
    """The probability to reset a substation to the unsplit state."""

    n_rel_subs: int = eqx.field(static=True)
    """The number of substations in the topology, used to determine the valid range of substation ids."""


class DisconnectionMutationConfig(eqx.Module):
    """Configuration for the disconnection mutation operation.

    Holds the random parameters used during the mutation of the disconnections,
    which are applied after the substation mutation.
    """

    add_disconnection_prob: float = eqx.field(static=True)
    """The probability to disconnect an additional branch."""

    change_disconnection_prob: float = eqx.field(static=True)
    """The probability to change a disconnected branch to another one."""

    remove_disconnection_prob: float = eqx.field(static=True)
    """The probability to remove a reconnect a disconnected branch."""

    n_disconnectable_branches: int = eqx.field(static=True)
    """The number of disconnectable branches in the topology, used to determine the valid range of branch ids."""


class NodalInjectionMutationConfig(eqx.Module):
    """Configuration for the nodal injection mutation operation.

    Holds the random parameters used during the mutation of the nodal injection optimization results,
    which are applied after the substation mutation.
    """

    pst_mutation_sigma: float = eqx.field(static=True)
    """The sigma to use for the normal distribution to sample the PST tap mutation from."""

    pst_mutation_probability: float = eqx.field(static=True)
    """The probability for an individual PST to be selected for mutation."""

    pst_n_taps: Int[Array, " num_psts"]
    """The number of taps for each PST, used to determine the valid range of tap positions for mutation."""


class MutationConfig(eqx.Module):
    """Configuration for the mutation operation."""

    mutation_repetition: int = eqx.field(static=True)
    """More chance to get unique mutations by mutating mutation_repetition copies of the repertoire.
    The repertoire is repeated x times before mutation and deduplicated after mutation."""

    random_topo_prob: float = eqx.field(static=True)
    """The probability to apply a completely random topology, instead of mutating.
    This is added to increase the exploration capabilities of the mutation operator,
    but should be used with care as it can easily lead to a decrease in performance
    if the random topologies are of low quality.

    This does not include PSTs for now.
    """

    substation_mutation_config: SubstationMutationConfig = eqx.field(static=True)
    """The configuration for the substation mutation operation."""

    disconnection_mutation_config: DisconnectionMutationConfig = eqx.field(static=True)
    """The configuration for the disconnection mutation operation."""

    nodal_injection_mutation_config: Optional[NodalInjectionMutationConfig]
    """The configuration for the nodal injection mutation operation."""
