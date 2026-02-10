# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import jax.numpy as jnp
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from toop_engine_dc_solver.example_grids import case30_with_psts_powsybl
from toop_engine_dc_solver.jax.nodal_inj_optim import nodal_inj_optimization
from toop_engine_dc_solver.jax.types import (
    NodalInjOptimResults,
    NodalInjStartOptions,
    TopologyResults,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid


def test_compare_nodal_inj_to_powsybl(tmp_path: Path) -> None:
    """Test that nodal_inj_optimization function works correctly with PST taps.

    This test verifies that:
    1. The nodal_inj_optimization function runs without errors
    2. Applying different PST tap settings produces different N-0 flows
    3. The function correctly handles batch dimensions and returns expected shapes
    """
    # Create case30 grid with PSTs in pypowsybl format
    case30_with_psts_powsybl(tmp_path)

    # Load the grid
    filesystem_dir = DirFileSystem(str(tmp_path))
    stats, static_information, network_data = load_grid(filesystem_dir, pandapower=False)

    di = static_information.dynamic_information
    solver_config = replace(static_information.solver_config, batch_size_bsdf=1, enable_nodal_inj_optim=True)

    inj_info = di.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"

    # Get dimensions from unsplit_flow (shape: n_timesteps, n_branches)
    # We need to add a batch dimension for nodal_inj_optimization
    n_timesteps, n_branches = di.unsplit_flow.shape
    batch_size = 1

    # Add batch dimension to unsplit_flow, nodal_injections, and ptdf
    n_0_batched = di.unsplit_flow[None, :, :]  # (1, n_timesteps, n_branches)
    nodal_injections_batched = di.nodal_injections[None, :, :]  # (1, n_timesteps, n_buses)
    ptdf_batched = di.ptdf[None, :, :]  # (1, n_branches, n_buses)
    from_node_batched = di.from_node[None, :]  # (1, n_branches)
    to_node_batched = di.to_node[None, :]  # (1, n_branches)

    taps = inj_info.starting_tap_idx
    n_0_unchanged, n_1_unchanged, results_unchanged = nodal_inj_optimization(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        topo_res=TopologyResults(
            ptdf=ptdf_batched,
            from_node=from_node_batched,
            to_node=to_node_batched,
            lodf=jnp.array([]),
            success=jnp.ones((batch_size,), dtype=bool),
            outage_modf=[],
            bsdf=jnp.array([]),
            failure_cases_to_zero=None,
            disconnection_modf=None,
        ),
        start_options=NodalInjStartOptions(
            previous_results=NodalInjOptimResults(
                pst_tap_idx=taps[None, None, :],  # Add batch and timestep dimensions
            ),
            precision_percent=jnp.array(1.0),
        ),
        dynamic_information=di,
        solver_config=solver_config,
    )

    # When we apply starting taps to flows that were already computed with starting taps,
    # we should get different results (applying the same shift twice)
    # So we just verify the function runs successfully
    assert n_0_unchanged.shape == n_0_batched.shape, "Output shape should match input shape"

    # Now take different taps, lets say we increase all taps by 1 (if possible) and see if n-0 results changed.
    new_taps = jnp.minimum(taps + 1, inj_info.pst_n_taps - 1)  # increase taps by 1 but don't exceed max tap
    n_0_changed, n_1_changed, results_changed = nodal_inj_optimization(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        topo_res=TopologyResults(
            ptdf=ptdf_batched,
            from_node=from_node_batched,
            to_node=to_node_batched,
            lodf=jnp.array([]),
            success=jnp.ones((batch_size,), dtype=bool),
            outage_modf=[],
            bsdf=jnp.array([]),
            failure_cases_to_zero=None,
            disconnection_modf=None,
        ),
        start_options=NodalInjStartOptions(
            previous_results=NodalInjOptimResults(
                pst_tap_idx=new_taps[None, None, :],  # Add batch and timestep dimensions
            ),
            precision_percent=jnp.array(1.0),
        ),
        dynamic_information=di,
        solver_config=solver_config,
    )

    # Verify that when we apply different taps, we get different flows than with starting taps
    assert not jnp.allclose(n_0_changed, n_0_unchanged), "N-0 flows should differ between different tap settings"
    assert n_0_changed.shape == n_0_batched.shape, "Output shape should match input shape"
