# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides common routines for switching distance calculations."""

import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, ArrayLike, Bool, Int


def per_station_switching_distance(
    target_configuration: Bool[Array, " n_assets"],
    current_coupler_state: Bool[Array, " n_couplers"],
    separation_set: Bool[Array, " n_configurations 2 n_assets"],
    coupler_states: Bool[Array, " n_configurations n_couplers"],
    ignore_assets: Optional[Bool[Array, " n_assets"]] = None,
) -> tuple[Int[Array, " "], Bool[Array, ""], Int[Array, " "], Int[Array, " "]]:
    """Compute the switching distance between a current configuration and a target configuration.

    The current configuration is represented through the current coupler state and all the possible
    two-way splits stored in the configuration table along with the corresponding coupler states.

    This method compares bus A and bus B separately, i.e. if a branch is both on bus A and bus B or
    on neither it will be correctly accounted for. If you want a more imprecise but faster method,
    use station_switching_distance_fast.

    It will return the integer reassignment and coupler distances, if you want to recreate the cost
    used internally, use: reconfiguration_cost_factor * reassignment + coupler.

    Parameters
    ----------
    target_configuration : Bool[Array, " n_assets"]
        The electrical target configuration to be realized. True means that the asset is connected
        to busbar A, False means that the asset is connected to busbar B.
    current_coupler_state : Bool[Array, " n_couplers"]
        The current coupler state of the station.
    separation_set : Bool[Array, " n_configurations 2 n_assets"]
        The separation set of the station. Each row corresponds to a possible two-way split of
        the station.
    coupler_states : Bool[Array, " n_configurations n_couplers"]
        The coupler states for each configuration in the configuration table, i.e. how this two-way
        split could be realized with busbar breaker switchings.
    ignore_assets : Optional[Bool[Array, " n_assets"]], optional
        A mask to ignore certain assets in the comparison, by default None. If a boolean is True at
        index i, the asset at index i will always produce a hamming distance of 0.

    Returns
    -------
    Int[Array, " "]
        The index of the best configuration in the configuration table.
    Bool[Array, ""]
        Whether the target configuration should be inverted, i.e. True should map to busbar B and
        False should map to busbar A.
    Int[Array, " "]
        The reassignment distance, i.e. the amount of busbar reassignments that have to be made.
    Int[Array, " "]
        The coupler distance, i.e. the amount of coupler switchings that have to be made.
    """
    if ignore_assets is None:
        ignore_assets = jnp.zeros(target_configuration.shape, dtype=bool)

    # Compute the coupler hamming distance, i.e. the amount of coupler states that have to be
    # changed.
    coupler_distance = hamming_distance(coupler_states, current_coupler_state)

    busbars_a = separation_set[:, 0, :]
    busbars_b = separation_set[:, 1, :]

    # Compute the reassignment distance, i.e. how many busbar reassignments have to be made.
    # This needs to happen for the normal and inverted target configuration as either could be better
    reassignment_distance = hamming_distance(busbars_a, ~target_configuration, ignore_assets) + hamming_distance(
        busbars_b, target_configuration, ignore_assets
    )
    reassignment_distance_inv = hamming_distance(busbars_a, target_configuration, ignore_assets) + hamming_distance(
        busbars_b, ~target_configuration, ignore_assets
    )

    # If the inverted target configuration is better, choose that
    invert = reassignment_distance_inv.min() < reassignment_distance.min()
    target_configuration = jnp.where(invert, ~target_configuration, target_configuration)
    reassignment_distance = jnp.where(invert, reassignment_distance_inv, reassignment_distance)

    best_configuration = jnp.argmin(reassignment_distance)
    return (
        best_configuration,
        invert,
        reassignment_distance[best_configuration],
        coupler_distance[best_configuration],
    )


def hamming_distance(
    starting_configuration: Bool[Array, " n_configurations n_assets"],
    target_configuration: Bool[Array, " n_assets"],
    ignore_assets: Optional[Bool[Array, " n_assets"]] = None,
) -> Int[Array, " n_configurations"]:
    """Compute the hamming distance between a target configuration and a table of configurations.

    Parameters
    ----------
    starting_configuration : Bool[Array, " n_configurations n_assets"]
        The table of configurations to compare against.
    target_configuration : Bool[Array, " n_assets"]
        The target configuration to compare against.
    ignore_assets : Optional[Bool[Array, " n_assets"]], optional
        A mask to ignore certain assets in the comparison, by default None. If a boolean is True at
        index i, the asset at index i will always produce a hamming distance of 0.

    Returns
    -------
    Int[Array, " n_configurations"]
        The hamming distance between the target configuration and each configuration in the table.
    """
    assert starting_configuration.shape[-1] == target_configuration.shape[-1]
    diff = jnp.logical_xor(starting_configuration, target_configuration)
    if ignore_assets is not None:
        assert ignore_assets.shape == target_configuration.shape
        diff = diff & ~ignore_assets
    return jnp.sum(diff, axis=-1)


def min_hamming_distance_matrix(
    permutations: Bool[ArrayLike, " n_permutations n_assets"],
    possible_starting_configs: Bool[ArrayLike, " n_possible_starting_configs n_assets"],
) -> Int[Array, " n_permutations"]:
    """Compute the hamming distance matrix between two sets of configurations.

    Parameters
    ----------
    permutations : Bool[Array, " n_permutations n_assets"]
        The first table of configurations to compare.
    possible_starting_configs : Bool[Array, " n_possible_starting_configs n_assets"]
        The second table of configurations to compare.

    Returns
    -------
    The minimum hamming distance between each configuration in permutations
    and any configuration in possible_starting_configs.
    """
    dist_matrix = jnp.sum(jnp.logical_xor(permutations[:, None, :], possible_starting_configs[None, :, :]), axis=2)
    min_changes = jnp.min(dist_matrix, axis=1)
    return min_changes
