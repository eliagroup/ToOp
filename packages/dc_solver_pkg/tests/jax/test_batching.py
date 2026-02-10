# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import math

import pytest
from jax import numpy as jnp
from toop_engine_dc_solver.jax.batching import (
    batch_injections,
    batch_topologies,
    count_injection_combinations_from_corresponding_topology,
    get_injections_for_topo_range,
    greedy_buffer_size_selection,
    greedy_n_subs_selection,
    pad_topologies,
    slice_injections,
    slice_nodal_inj_start_options,
    slice_topologies,
    split_injections,
    upper_bound_buffer_size_injections,
)
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    NodalInjOptimResults,
    NodalInjStartOptions,
    StaticInformation,
    TopoVectBranchComputations,
)


def test_batch_topologies(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, candidates, static_information = jax_inputs

    n_computations = computations.topologies.shape[0]
    batched_computations = batch_topologies(computations, 16)

    assert batched_computations.topologies.shape == (n_computations // 16 + 1, 16, 5, 5)
    assert batched_computations.sub_ids.shape == (n_computations // 16 + 1, 16, 5)
    assert batched_computations.pad_mask.shape == (n_computations // 16 + 1, 16)

    assert batched_computations.pad_mask[-1, -1].item() is False
    assert batched_computations.pad_mask[0, 0].item() is True

    assert jnp.sum(computations.topologies) == jnp.sum(batched_computations.topologies)

    for i in range(batched_computations.topologies.shape[0]):
        sliced_computations = slice_topologies(computations, i, 16)

        assert jnp.array_equal(batched_computations.topologies[i], sliced_computations.topologies)
        assert jnp.array_equal(batched_computations.sub_ids[i], sliced_computations.sub_ids)
        assert jnp.array_equal(batched_computations.pad_mask[i], sliced_computations.pad_mask)


def test_get_injections_for_topo_range(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, candidates, _ = jax_inputs

    candidate_range = get_injections_for_topo_range(
        all_injections=candidates,
        topo_index=1,
        batch_size_bsdf=16,
        batch_size_injection=32,
        buffer_size_injection=99,
        return_relative_index=False,
    )

    acceptable_range = jnp.concatenate(
        [
            jnp.arange(16, 32),
            jnp.array([jnp.iinfo(candidates.corresponding_topology.dtype).max]),
        ]
    )
    assert jnp.all(jnp.isin(candidate_range.corresponding_topology, acceptable_range))
    assert candidate_range.injection_topology.shape == (
        99,
        32,
        candidates.injection_topology.shape[1],
        candidates.injection_topology.shape[2],
    )
    assert candidate_range.pad_mask.shape == (99, 32)
    assert candidate_range.corresponding_topology.shape == (99, 32)


def test_batch_injections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    topologies, candidates, static_information = jax_inputs

    batch_size_bsdf = 16
    batch_size_injections = 32

    batched_topologies = batch_topologies(topologies, batch_size_bsdf)

    # We assume max(n_injections_per_topology) to be the injections for the fully split topology
    max_injections_per_topology = 2 ** jnp.sum(static_information.dynamic_information.generators_per_sub)
    buffer_size_injection = math.ceil(max_injections_per_topology * batch_size_bsdf / batch_size_injections)
    n_splits = candidates.injection_topology.shape[1]
    max_inj_per_sub = candidates.injection_topology.shape[2]

    batched_injections = batch_injections(
        all_injections=candidates,
        batched_topologies=batched_topologies,
        batch_size_injection=batch_size_injections,
        buffer_size_injection=buffer_size_injection,
    )

    assert batched_injections.injection_topology.shape == (
        batched_topologies.pad_mask.shape[0],
        buffer_size_injection,
        batch_size_injections,
        n_splits,
        max_inj_per_sub,
    )
    assert batched_injections.pad_mask.shape == (
        batched_topologies.pad_mask.shape[0],
        buffer_size_injection,
        batch_size_injections,
    )
    assert batched_injections.corresponding_topology.shape == (
        batched_topologies.pad_mask.shape[0],
        buffer_size_injection,
        batch_size_injections,
    )

    assert jnp.sum(batched_injections.injection_topology) == jnp.sum(candidates.injection_topology)

    for topo_batch_id in range(batched_topologies.pad_mask.shape[0]):
        buffer_ended = False
        for buffer_id in range(buffer_size_injection):
            if buffer_ended:
                assert jnp.all(batched_injections.injection_topology[topo_batch_id, buffer_id] == 0)
                assert jnp.all(batched_injections.pad_mask[topo_batch_id, buffer_id] == 0)
                assert jnp.all(
                    batched_injections.corresponding_topology[topo_batch_id, buffer_id]
                    == jnp.iinfo(batched_injections.corresponding_topology.dtype).max
                )
            if batched_injections.pad_mask[topo_batch_id, buffer_id, 0].item() is False:
                buffer_ended = True

    with pytest.raises(Exception):
        batch_injections(
            all_injections=candidates,
            batched_topologies=batched_topologies,
            batch_size_injection=batch_size_injections,
            buffer_size_injection=2,
        )


def test_slice_nodal_inj_start_options() -> None:
    """Test slicing of NodalInjStartOptions preserves scalar precision_percent and slices pst_taps correctly."""
    batch_size_bsdf = 16
    n_batches = 3
    total_size = batch_size_bsdf * n_batches
    n_timesteps = 24
    n_pst = 5

    # Create test data with distinct values per batch
    pst_taps_data = jnp.arange(total_size * n_timesteps * n_pst, dtype=jnp.float32).reshape(total_size, n_timesteps, n_pst)
    precision_scalar = jnp.array(0.5)

    options = NodalInjStartOptions(
        previous_results=NodalInjOptimResults(pst_tap_idx=pst_taps_data),
        precision_percent=precision_scalar,
    )

    # Test slicing at index 0
    sliced_0 = slice_nodal_inj_start_options(options, 0, batch_size_bsdf)
    assert sliced_0.previous_results.pst_tap_idx.shape == (batch_size_bsdf, n_timesteps, n_pst)
    assert sliced_0.precision_percent.ndim == 0
    assert sliced_0.precision_percent == precision_scalar
    assert jnp.array_equal(
        sliced_0.previous_results.pst_tap_idx,
        pst_taps_data[:batch_size_bsdf],
    )

    # Test slicing at index 1
    sliced_1 = slice_nodal_inj_start_options(options, 1, batch_size_bsdf)
    assert sliced_1.previous_results.pst_tap_idx.shape == (batch_size_bsdf, n_timesteps, n_pst)
    assert sliced_1.precision_percent.ndim == 0
    assert sliced_1.precision_percent == precision_scalar
    assert jnp.array_equal(
        sliced_1.previous_results.pst_tap_idx,
        pst_taps_data[batch_size_bsdf : 2 * batch_size_bsdf],
    )

    # Test slicing at index 2 (last batch)
    sliced_2 = slice_nodal_inj_start_options(options, 2, batch_size_bsdf)
    assert sliced_2.previous_results.pst_tap_idx.shape == (batch_size_bsdf, n_timesteps, n_pst)
    assert sliced_2.precision_percent.ndim == 0
    assert sliced_2.precision_percent == precision_scalar
    assert jnp.array_equal(
        sliced_2.previous_results.pst_tap_idx,
        pst_taps_data[2 * batch_size_bsdf :],
    )

    # Test out-of-bounds slicing (index 10) - should fill with nan
    sliced_oob = slice_nodal_inj_start_options(options, 10, batch_size_bsdf)
    assert sliced_oob.previous_results.pst_tap_idx.shape == (batch_size_bsdf, n_timesteps, n_pst)
    assert sliced_oob.precision_percent.ndim == 0
    assert sliced_oob.precision_percent == precision_scalar
    assert jnp.all(jnp.isnan(sliced_oob.previous_results.pst_tap_idx))


def test_slice_nodal_inj_start_options_reconstruction() -> None:
    """Test that slicing NodalInjStartOptions in a loop reconstructs the original data."""
    batch_size_bsdf = 16
    n_batches = 3
    total_size = batch_size_bsdf * n_batches
    n_timesteps = 24
    n_pst = 5

    # Create test data
    pst_taps_data = jnp.arange(total_size * n_timesteps * n_pst, dtype=jnp.float32).reshape(total_size, n_timesteps, n_pst)
    precision_scalar = jnp.array(0.75)

    options = NodalInjStartOptions(
        previous_results=NodalInjOptimResults(pst_tap_idx=pst_taps_data),
        precision_percent=precision_scalar,
    )

    # Slice each batch and verify reconstruction
    for i in range(n_batches):
        sliced = slice_nodal_inj_start_options(options, i, batch_size_bsdf)

        # Verify scalar precision is preserved
        assert sliced.precision_percent.ndim == 0
        assert sliced.precision_percent == precision_scalar

        # Verify pst_taps data matches original at correct indices
        start_idx = i * batch_size_bsdf
        end_idx = (i + 1) * batch_size_bsdf
        assert jnp.array_equal(
            sliced.previous_results.pst_tap_idx,
            pst_taps_data[start_idx:end_idx],
        )


def test_slice_injections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, candidates, _ = jax_inputs

    batch_size_injections = 32
    n_splits = candidates.injection_topology.shape[1]
    max_inj_per_sub = candidates.injection_topology.shape[2]
    inj = slice_injections(candidates, 0, batch_size_injections)

    assert inj.injection_topology.shape == (32, n_splits, max_inj_per_sub)
    assert inj.pad_mask.shape == (32,)
    assert inj.corresponding_topology.shape == (32,)


def test_split_injections(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    _, candidates, _ = jax_inputs

    batch_size_bsdf = 16
    batch_size_injections = 32
    n_split_subs = candidates.injection_topology.shape[1]
    max_inj_per_sub = candidates.injection_topology.shape[2]
    buffer_size_injections = greedy_buffer_size_selection(
        count_injection_combinations_from_corresponding_topology(
            candidates.corresponding_topology,
            batch_size_bsdf,
        ),
        batch_size_injections,
    )
    n_splits = 2
    n_topologies = candidates.corresponding_topology.max().item() + 1

    n_topos_per_split = n_topologies // n_splits
    assert n_topologies % n_splits == 0
    packet_size_injection = batch_size_injections * buffer_size_injections * math.ceil(n_topos_per_split / batch_size_bsdf)

    split_inj = split_injections(
        candidates,
        n_splits=n_splits,
        packet_size_injection=packet_size_injection,
        n_topos_per_split=n_topos_per_split,
    )

    assert split_inj.injection_topology.shape == (n_splits, packet_size_injection, n_split_subs, max_inj_per_sub)
    assert split_inj.pad_mask.shape == (n_splits, packet_size_injection)
    assert split_inj.corresponding_topology.shape == (n_splits, packet_size_injection)

    for i in range(n_splits):
        split_range = jnp.arange(i * n_topos_per_split, (i + 1) * n_topos_per_split)
        assert jnp.sum(split_inj.pad_mask[i]) == jnp.sum(jnp.isin(candidates.corresponding_topology, split_range))


def test_count_injection_combinations_from_corresponding_topology(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, candidates, static_information = jax_inputs

    n_topologies = computations.topologies.shape[0]
    n_combis = count_injection_combinations_from_corresponding_topology(
        candidates.corresponding_topology,
        static_information.solver_config.batch_size_bsdf,
        n_topologies,
    )

    assert n_combis.shape == (math.ceil(n_topologies / static_information.solver_config.batch_size_bsdf),)

    for i in range(math.ceil(n_topologies / static_information.solver_config.batch_size_bsdf)):
        topo_indices = jnp.arange(
            i * static_information.solver_config.batch_size_bsdf,
            (i + 1) * static_information.solver_config.batch_size_bsdf,
        )
        n_matches = jnp.sum(jnp.isin(candidates.corresponding_topology, topo_indices))
        assert n_combis[i] == n_matches


def test_greedy_buffer_size_selection(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, candidates, static_information = jax_inputs

    candidate_size_greedy = greedy_buffer_size_selection(
        count_injection_combinations_from_corresponding_topology(
            candidates.corresponding_topology,
            static_information.solver_config.batch_size_bsdf,
        ),
        static_information.solver_config.batch_size_injection,
    )

    buffer_size_upper_bound = upper_bound_buffer_size_injections(
        2**static_information.dynamic_information.generators_per_sub,
        static_information.solver_config.batch_size_bsdf,
        static_information.solver_config.batch_size_injection,
    )

    limited_upper_bound = upper_bound_buffer_size_injections(
        2**static_information.dynamic_information.generators_per_sub,
        static_information.solver_config.batch_size_bsdf,
        static_information.solver_config.batch_size_injection,
        limit_n_subs=3,
    )

    assert limited_upper_bound < buffer_size_upper_bound
    assert candidate_size_greedy < buffer_size_upper_bound


def test_greedy_n_subs_selection(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, _candidates, _static_information = jax_inputs

    n_subs_greedy = greedy_n_subs_selection(computations)

    assert n_subs_greedy == 3


def test_pad_topologies(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    computations, _candidates, _static_information = jax_inputs

    cur_size = computations.topologies.shape[0]

    padded_computations = pad_topologies(computations, cur_size + 16)

    assert padded_computations.topologies.shape == (cur_size + 16, 5, 5)
    assert padded_computations.sub_ids.shape == (cur_size + 16, 5)
    assert padded_computations.pad_mask.shape == (cur_size + 16,)
    assert not jnp.any(padded_computations.pad_mask[-16:])
    assert jnp.array_equal(padded_computations.topologies[:cur_size], computations.topologies)
    assert jnp.array_equal(padded_computations.sub_ids[:cur_size], computations.sub_ids)
    assert jnp.array_equal(padded_computations.pad_mask[:cur_size], computations.pad_mask)
