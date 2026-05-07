# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


from jax import numpy as jnp
from toop_engine_dc_solver.jax.batching import batch_topologies, slice_nodal_inj_start_options, slice_topologies
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    NodalInjOptimResults,
    NodalInjStartOptions,
    StaticInformation,
    TopoVectBranchComputations,
    int_max,
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


def test_slice_nodal_inj_start_options() -> None:
    """Test slicing of NodalInjStartOptions preserves scalar precision_percent and slices pst_taps correctly."""
    batch_size_bsdf = 16
    n_batches = 3
    total_size = batch_size_bsdf * n_batches
    n_timesteps = 24
    n_pst = 5

    # Create test data with distinct values per batch
    pst_taps_data = jnp.arange(total_size * n_timesteps * n_pst, dtype=int).reshape(total_size, n_timesteps, n_pst)
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
    assert jnp.all(sliced_oob.previous_results.pst_tap_idx == int_max())


def test_slice_nodal_inj_start_options_reconstruction() -> None:
    """Test that slicing NodalInjStartOptions in a loop reconstructs the original data."""
    batch_size_bsdf = 16
    n_batches = 3
    total_size = batch_size_bsdf * n_batches
    n_timesteps = 24
    n_pst = 5

    # Create test data
    pst_taps_data = jnp.arange(total_size * n_timesteps * n_pst, dtype=int).reshape(total_size, n_timesteps, n_pst)
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
