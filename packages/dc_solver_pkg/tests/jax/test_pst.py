# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import (
    case30_with_psts_powsybl,
    complex_grid_battery_hvdc_svc_3w_trafo_data_folder,
    parallel_pst_data_folder,
)
from toop_engine_dc_solver.jax.pst import (
    prepare_pst_tap_state,
    update_n0_for_pst_taps,
    write_pst_taps_to_nodal_injections,
)
from toop_engine_dc_solver.jax.types import NodalInjOptimResults, NodalInjStartOptions, TopologyResults
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid


def _build_topology_results(
    ptdf_batched: jnp.ndarray,
    from_node_batched: jnp.ndarray,
    to_node_batched: jnp.ndarray,
    n_branches: int,
) -> TopologyResults:
    """Build a minimal topology result fixture for PST helper tests."""
    return TopologyResults(
        ptdf=ptdf_batched,
        from_node=from_node_batched,
        to_node=to_node_batched,
        lodf=jnp.array([[]]),
        success=jnp.ones((1,), dtype=bool),
        outage_modf=[],
        bsdf=jnp.zeros((1, n_branches)),
        failure_cases_to_zero=None,
        disconnection_modf=None,
    )


def _build_start_options(pst_tap_idx: jnp.ndarray) -> NodalInjStartOptions:
    """Create PST start options with explicit batch and timestep dimensions."""
    return NodalInjStartOptions(
        previous_results=NodalInjOptimResults(pst_tap_idx=pst_tap_idx[None, None, :]),
        precision_percent=jnp.array(1.0),
    )


def _build_start_options_2d(pst_tap_idx: jnp.ndarray) -> NodalInjStartOptions:
    """Create PST start options with timestep and PST dimensions only."""
    return NodalInjStartOptions(
        previous_results=NodalInjOptimResults(pst_tap_idx=pst_tap_idx[None, :]),
        precision_percent=jnp.array(1.0),
    )


def test_pst_helpers_preserve_starting_case_and_change_flows(tmp_path: Path) -> None:
    """PST helpers should preserve the base case at starting taps and change flows for new taps."""
    case30_with_psts_powsybl(tmp_path)

    filesystem_dir = DirFileSystem(str(tmp_path))
    _, static_information, _ = load_grid(filesystem_dir, pandapower=False)

    di = static_information.dynamic_information
    inj_info = di.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"

    n_timesteps, n_branches = di.unsplit_flow.shape
    assert n_timesteps > 0

    n_0_batched = di.unsplit_flow[None, :, :]
    nodal_injections_batched = di.nodal_injections[None, :, :]
    ptdf_batched = di.ptdf[None, :, :]
    from_node_batched = di.from_node[None, :]
    to_node_batched = di.to_node[None, :]
    topo_res = _build_topology_results(ptdf_batched, from_node_batched, to_node_batched, n_branches)

    starting_taps = inj_info.starting_tap_idx
    pst_tap_indices, pst_tap_susceptance_values, optimized_results = prepare_pst_tap_state(
        start_options=_build_start_options(starting_taps),
        nodal_inj_info=inj_info,
    )

    assert pst_tap_indices is not None
    assert pst_tap_susceptance_values is not None
    assert optimized_results is not None
    assert pst_tap_indices.shape == (1, 1, starting_taps.shape[0])
    assert pst_tap_susceptance_values.shape == (1, starting_taps.shape[0])
    assert jnp.array_equal(optimized_results.pst_tap_idx, pst_tap_indices)

    nodal_injections_at_start = write_pst_taps_to_nodal_injections(
        nodal_injections=nodal_injections_batched,
        pst_tap_indices=pst_tap_indices,
        nodal_inj_info=inj_info,
    )
    n_0_at_start = update_n0_for_pst_taps(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        updated_nodal_injections=nodal_injections_at_start,
        pst_tap_indices=pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=inj_info,
    )

    assert jnp.allclose(nodal_injections_at_start, nodal_injections_batched)
    assert jnp.allclose(n_0_at_start, n_0_batched)

    changed_taps = jnp.minimum(starting_taps + 1, inj_info.pst_n_taps - 1)
    changed_pst_tap_indices, _changed_susceptance_values, _changed_results = prepare_pst_tap_state(
        start_options=_build_start_options(changed_taps),
        nodal_inj_info=inj_info,
    )

    assert changed_pst_tap_indices is not None
    nodal_injections_changed = write_pst_taps_to_nodal_injections(
        nodal_injections=nodal_injections_batched,
        pst_tap_indices=changed_pst_tap_indices,
        nodal_inj_info=inj_info,
    )
    n_0_changed = update_n0_for_pst_taps(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        updated_nodal_injections=nodal_injections_changed,
        pst_tap_indices=changed_pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=inj_info,
    )

    assert not jnp.allclose(nodal_injections_changed, nodal_injections_batched)
    assert not jnp.allclose(n_0_changed, n_0_batched)
    assert n_0_changed.shape == n_0_batched.shape


def test_prepare_pst_tap_state_promotes_2d_indices(tmp_path: Path) -> None:
    """2D PST tap inputs should be promoted to a single-batch 3D tensor."""
    case30_with_psts_powsybl(tmp_path)

    filesystem_dir = DirFileSystem(str(tmp_path))
    _, static_information, _ = load_grid(filesystem_dir, pandapower=False)

    inj_info = static_information.dynamic_information.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"

    pst_tap_indices, pst_tap_susceptance_values, optimized_results = prepare_pst_tap_state(
        start_options=_build_start_options_2d(inj_info.starting_tap_idx),
        nodal_inj_info=inj_info,
    )

    assert pst_tap_indices is not None
    assert pst_tap_susceptance_values is not None
    assert optimized_results is not None
    assert pst_tap_indices.shape == (1, 1, inj_info.starting_tap_idx.shape[0])
    assert jnp.array_equal(pst_tap_indices[0, 0], inj_info.starting_tap_idx)
    assert jnp.array_equal(optimized_results.pst_tap_idx, pst_tap_indices)


def test_prepare_pst_tap_state_rejects_zero_timestep_batches(tmp_path: Path) -> None:
    """PST tap inputs without a timestep dimension should fail fast."""
    case30_with_psts_powsybl(tmp_path)

    filesystem_dir = DirFileSystem(str(tmp_path))
    _, static_information, _ = load_grid(filesystem_dir, pandapower=False)

    inj_info = static_information.dynamic_information.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"

    start_options = NodalInjStartOptions(
        previous_results=NodalInjOptimResults(
            pst_tap_idx=jnp.empty((1, 0, inj_info.starting_tap_idx.shape[0]), dtype=inj_info.starting_tap_idx.dtype)
        ),
        precision_percent=jnp.array(1.0),
    )

    with pytest.raises(ValueError, match="must include at least one timestep"):
        prepare_pst_tap_state(
            start_options=start_options,
            nodal_inj_info=inj_info,
        )


def test_pst_helpers_handle_parallel_psts(tmp_path: Path) -> None:
    """PST helpers should handle grids with multiple controllable PSTs in parallel groups."""
    parallel_pst_data_folder(tmp_path)

    filesystem_dir = DirFileSystem(str(tmp_path))
    _, static_information, _ = load_grid(filesystem_dir, pandapower=False)

    di = static_information.dynamic_information
    inj_info = di.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"
    assert inj_info.starting_tap_idx.shape[0] == 3, "Test grid should have 3 controllable PSTs"
    assert inj_info.parallel_pst_group_mask is not None, "Test grid should have parallel PST group mask"
    assert inj_info.parallel_pst_group_mask.shape == (2, 3), "PST group mask should have shape (2, 3)"

    _n_timesteps, n_branches = di.unsplit_flow.shape
    n_0_batched = di.unsplit_flow[None, :, :]
    nodal_injections_batched = di.nodal_injections[None, :, :]
    ptdf_batched = di.ptdf[None, :, :]
    from_node_batched = di.from_node[None, :]
    to_node_batched = di.to_node[None, :]
    topo_res = _build_topology_results(ptdf_batched, from_node_batched, to_node_batched, n_branches)

    changed_taps = jnp.minimum(inj_info.starting_tap_idx + 1, inj_info.pst_n_taps - 1)
    pst_tap_indices, pst_tap_susceptance_values, optimized_results = prepare_pst_tap_state(
        start_options=_build_start_options(changed_taps),
        nodal_inj_info=inj_info,
    )

    assert pst_tap_indices is not None
    assert pst_tap_susceptance_values is not None
    assert optimized_results is not None
    assert pst_tap_indices.shape == (1, 1, 3)
    assert pst_tap_susceptance_values.shape == (1, 3)

    nodal_injections_changed = write_pst_taps_to_nodal_injections(
        nodal_injections=nodal_injections_batched,
        pst_tap_indices=pst_tap_indices,
        nodal_inj_info=inj_info,
    )
    n_0_changed = update_n0_for_pst_taps(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        updated_nodal_injections=nodal_injections_changed,
        pst_tap_indices=pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=inj_info,
    )

    assert not jnp.allclose(nodal_injections_changed, nodal_injections_batched)
    assert not jnp.allclose(n_0_changed, n_0_batched)
    assert n_0_changed.shape == n_0_batched.shape


def test_pst_helpers_handle_nonlinear_psts(tmp_path: Path) -> None:
    """PST helpers should exercise the nonlinear PTDF branch on nonlinear PST grids."""
    _ = complex_grid_battery_hvdc_svc_3w_trafo_data_folder(tmp_path, linear_pst=np.array([False, False]))

    filesystem_dir = DirFileSystem(str(tmp_path))
    _, static_information, _ = load_grid(filesystem_dir, pandapower=False)

    di = static_information.dynamic_information
    inj_info = di.nodal_injection_information
    assert inj_info is not None, "Grid should have PSTs"
    valid_mask = jnp.arange(inj_info.pst_tap_values.shape[1])[None, :] < inj_info.pst_n_taps[:, None]
    first_susceptance = inj_info.pst_tap_susceptance_values[:, :1]
    nonlinear_mask = jnp.any(
        valid_mask & (inj_info.pst_tap_susceptance_values != first_susceptance),
        axis=1,
    )
    assert bool(jnp.any(nonlinear_mask)), "Grid should contain nonlinear PSTs"

    _n_timesteps, n_branches = di.unsplit_flow.shape
    n_0_batched = di.unsplit_flow[None, :, :]
    nodal_injections_batched = di.nodal_injections[None, :, :]
    ptdf_batched = di.ptdf[None, :, :]
    from_node_batched = di.from_node[None, :]
    to_node_batched = di.to_node[None, :]
    topo_res = _build_topology_results(ptdf_batched, from_node_batched, to_node_batched, n_branches)

    starting_taps = inj_info.starting_tap_idx
    starting_pst_tap_indices, starting_susceptance_values, _starting_results = prepare_pst_tap_state(
        start_options=_build_start_options(starting_taps),
        nodal_inj_info=inj_info,
    )
    assert starting_pst_tap_indices is not None
    assert starting_susceptance_values is not None

    nodal_injections_at_start = write_pst_taps_to_nodal_injections(
        nodal_injections=nodal_injections_batched,
        pst_tap_indices=starting_pst_tap_indices,
        nodal_inj_info=inj_info,
    )
    n_0_at_start = update_n0_for_pst_taps(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        updated_nodal_injections=nodal_injections_at_start,
        pst_tap_indices=starting_pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=inj_info,
    )

    assert jnp.allclose(nodal_injections_at_start, nodal_injections_batched)
    assert jnp.allclose(n_0_at_start, n_0_batched)

    low_taps = jnp.zeros_like(inj_info.starting_tap_idx)
    high_taps = inj_info.pst_n_taps - 1
    changed_taps = jnp.where(jnp.arange(inj_info.pst_n_taps.shape[0]) % 2 == 0, low_taps, high_taps)

    changed_pst_tap_indices, changed_susceptance_values, _changed_results = prepare_pst_tap_state(
        start_options=_build_start_options(changed_taps),
        nodal_inj_info=inj_info,
    )
    assert changed_pst_tap_indices is not None
    assert changed_susceptance_values is not None
    assert not jnp.allclose(changed_susceptance_values, starting_susceptance_values)

    nodal_injections_changed = write_pst_taps_to_nodal_injections(
        nodal_injections=nodal_injections_batched,
        pst_tap_indices=changed_pst_tap_indices,
        nodal_inj_info=inj_info,
    )
    n_0_changed = update_n0_for_pst_taps(
        n_0=n_0_batched,
        nodal_injections=nodal_injections_batched,
        updated_nodal_injections=nodal_injections_changed,
        pst_tap_indices=changed_pst_tap_indices,
        topo_res=topo_res,
        nodal_inj_info=inj_info,
    )

    assert not jnp.allclose(nodal_injections_changed, nodal_injections_batched)
    assert not jnp.allclose(n_0_changed, n_0_batched)
    assert n_0_changed.shape == n_0_batched.shape
