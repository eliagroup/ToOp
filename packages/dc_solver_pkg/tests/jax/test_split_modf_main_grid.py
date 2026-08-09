# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for main-grid-only split MODF against OpenLoadFlow DC."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pypowsybl
import pytest
from tests.network_data_pickle import load_network_data
from toop_engine_dc_solver.jax.multi_outages import compute_multi_outage
from toop_engine_dc_solver.jax.split_modf_main_grid import (
    compute_split_modf_main_grid,
    compute_split_modf_main_grid_from_mask,
    main_grid_component_labels,
    main_grid_reachable_mask,
    main_grid_reachable_mask_parallel,
    main_grid_reachable_masks_parallel,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.helpers.find_bridges import find_bridges
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, load_lf_params
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition

TOL = 1e-8
DATA_FOLDER_FIXTURES = [
    pytest.param("complex_grid_battery_hvdc_svc_3w_trafo_be_nl_de_data_folder", id="complex-grid"),
    pytest.param("node_breaker_grid_preprocessed_data_folder", id="basic-node-breaker"),
]


def test_main_grid_reachable_mask_basic_node_breaker_grid(basic_node_breaker_grid_v1: pypowsybl.network.Network) -> None:
    """Identify the mainland after opening the fixture's bridge line L9."""
    lines = basic_node_breaker_grid_v1.get_lines(attributes=["bus1_id", "bus2_id"])
    bus_ids = basic_node_breaker_grid_v1.get_buses().index
    from_node = jnp.asarray(bus_ids.get_indexer(lines.bus1_id), dtype=jnp.int32)
    to_node = jnp.asarray(bus_ids.get_indexer(lines.bus2_id), dtype=jnp.int32)

    reference_mask = main_grid_reachable_mask(
        from_node=from_node,
        to_node=to_node,
        outages=jnp.array([lines.index.get_loc("L9")], dtype=jnp.int32),
        slack_bus=jnp.array(bus_ids.get_loc("VL1_0"), dtype=jnp.int32),
        n_bus=len(bus_ids),
    )
    parallel_mask = main_grid_reachable_mask_parallel(
        from_node=from_node,
        to_node=to_node,
        outages=jnp.array([lines.index.get_loc("L9")], dtype=jnp.int32),
        slack_bus=jnp.array(bus_ids.get_loc("VL1_0"), dtype=jnp.int32),
        n_bus=len(bus_ids),
    )

    np.testing.assert_array_equal(np.asarray(reference_mask), [True, True, True, True, False])
    np.testing.assert_array_equal(parallel_mask, reference_mask)


def test_parallel_component_labels_match_reachability_on_large_deep_grid() -> None:
    """Match the reference on a 6,000-node, 13,000-branch deep grid."""
    n_bus = 6_000
    n_branch = 13_000
    first_detached_bus = 5_500

    # Start with a full chain and add short-range chords without crossing the
    # bridge into the final 500-node component. The graph remains deliberately
    # deep while matching the expected production-scale dimensions.
    chain_from = np.arange(n_bus - 1, dtype=np.int32)
    chain_to = chain_from + 1
    main_chord_from = np.arange(first_detached_bus - 2, dtype=np.int32)
    stub_chord_from = np.arange(first_detached_bus, n_bus - 2, dtype=np.int32)
    remaining_chord_from = np.arange(1_005, dtype=np.int32)
    from_node = np.concatenate((chain_from, main_chord_from, stub_chord_from, remaining_chord_from))
    to_node = np.concatenate((chain_to, main_chord_from + 2, stub_chord_from + 2, remaining_chord_from + 3))
    assert from_node.shape == (n_branch,)
    assert to_node.shape == (n_branch,)

    bridge_index = first_detached_bus - 1
    outages = jnp.array([bridge_index, -1, n_branch], dtype=jnp.int32)
    from_node_jax = jnp.asarray(from_node)
    to_node_jax = jnp.asarray(to_node)
    slack_bus = jnp.array(0, dtype=jnp.int32)

    reference_mask = main_grid_reachable_mask(
        from_node=from_node_jax,
        to_node=to_node_jax,
        outages=outages,
        slack_bus=slack_bus,
        n_bus=n_bus,
    )
    parallel_mask = main_grid_reachable_mask_parallel(
        from_node=from_node_jax,
        to_node=to_node_jax,
        outages=outages,
        slack_bus=slack_bus,
        n_bus=n_bus,
    )
    labels, rounds = main_grid_component_labels(
        from_node=from_node_jax,
        to_node=to_node_jax,
        outages=outages,
        n_bus=n_bus,
    )

    expected_mask = np.arange(n_bus) < first_detached_bus
    np.testing.assert_array_equal(reference_mask, expected_mask)
    np.testing.assert_array_equal(parallel_mask, reference_mask)
    assert int(labels[0]) != int(labels[first_detached_bus])
    assert int(rounds) < 32


def test_parallel_component_labels_match_reachability_for_varied_outages() -> None:
    """Match the reference for padded, repeated, and topology-invalid edges."""
    rng = np.random.default_rng(42)
    n_bus = 128
    chain_from = np.arange(n_bus - 1, dtype=np.int32)
    chain_to = chain_from + 1
    from_node = np.concatenate((chain_from, rng.integers(0, n_bus, size=256, dtype=np.int32)))
    to_node = np.concatenate((chain_to, rng.integers(0, n_bus, size=256, dtype=np.int32)))
    from_node[-1] = -1
    to_node[-2] = n_bus

    outage_sets = np.array(
        [
            [31, -1, len(from_node), 31],
            [0, 1, 2, 3],
            [100, 180, 250, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    slack_buses = np.array([0, 63, 127, 17], dtype=np.int32)
    from_node_jax = jnp.asarray(from_node)
    to_node_jax = jnp.asarray(to_node)

    for outages, slack_bus in zip(outage_sets, slack_buses, strict=True):
        reference_mask = main_grid_reachable_mask(
            from_node=from_node_jax,
            to_node=to_node_jax,
            outages=jnp.asarray(outages),
            slack_bus=jnp.asarray(slack_bus),
            n_bus=n_bus,
        )
        parallel_mask = main_grid_reachable_mask_parallel(
            from_node=from_node_jax,
            to_node=to_node_jax,
            outages=jnp.asarray(outages),
            slack_bus=jnp.asarray(slack_bus),
            n_bus=n_bus,
        )
        labels, _rounds = main_grid_component_labels(
            from_node=from_node_jax,
            to_node=to_node_jax,
            outages=jnp.asarray(outages),
            n_bus=n_bus,
        )

        np.testing.assert_array_equal(parallel_mask, reference_mask)
        np.testing.assert_array_equal(parallel_mask, labels == labels[slack_bus])


def test_split_modf_batch_composes_with_production_topology_vmap() -> None:
    """Keep topology, timestep, contingency, and monitored-branch axes aligned."""
    ptdf_single = jnp.array(
        [
            [0.0, -1.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    from_node_single = jnp.array([0, 1], dtype=jnp.int32)
    to_node_single = jnp.array([1, 2], dtype=jnp.int32)
    nodal_injections = jnp.array(
        [
            [[3.0, -1.0, -2.0]],
            [[6.0, -2.0, -4.0]],
        ]
    )
    ptdf = jnp.broadcast_to(ptdf_single, (2, *ptdf_single.shape))
    from_node = jnp.broadcast_to(from_node_single, (2, from_node_single.size))
    to_node = jnp.broadcast_to(to_node_single, (2, to_node_single.size))
    n_0_flow = jnp.einsum("bij,btj->bti", ptdf, nodal_injections)
    outages = jnp.array([[[0], [1]], [[1], [0]]], dtype=jnp.int32)
    branches_monitored = jnp.arange(2, dtype=jnp.int32)

    flows, main_bus, main_monitored, main_imbalance, rank_loss = jax.vmap(
        compute_split_modf_main_grid,
        in_axes=(0, 0, 0, 0, 0, 0, None, None),
    )(
        ptdf,
        from_node,
        to_node,
        nodal_injections,
        n_0_flow,
        outages,
        branches_monitored,
        jnp.array(0, dtype=jnp.int32),
    )

    assert flows.shape == (2, 1, 2, 2)
    assert main_bus.shape == (2, 2, 3)
    assert main_monitored.shape == (2, 2, 2)
    assert main_imbalance.shape == (2, 1, 2)
    assert rank_loss.shape == (2, 2)
    np.testing.assert_array_equal(
        main_bus,
        [
            [[True, False, False], [True, True, False]],
            [[True, True, False], [True, False, False]],
        ],
    )
    assert np.all(np.isnan(np.asarray(flows)[~np.asarray(main_monitored)[:, None, :, :]]))


def _load_split_modf_test_grid(
    data_folder: Path,
) -> tuple[PowsyblRunner, NetworkData, np.ndarray, np.ndarray, np.ndarray]:
    """Load topology-complete data and separate physical injections from auxiliary PST coordinates."""
    network_data = load_network_data(data_folder / "network_data.pkl")
    assert network_data.ptdf is not None
    assert network_data.nodal_injection is not None
    assert network_data.bridging_branch_mask is not None
    runner = PowsyblRunner(lf_params=load_lf_params(data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    runner.load_base_grid(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))

    nodal_injections = network_data.nodal_injection[0].copy()
    physical_nodal_injections = nodal_injections.copy()
    physical_nodal_injections[np.asarray(network_data.node_types) == "PSTNode"] = 0.0

    recomputed_bridging_branch_mask = find_bridges(
        network_data.from_nodes,
        network_data.to_nodes,
        len(network_data.branch_ids),
        len(network_data.node_ids),
    )
    np.testing.assert_array_equal(recomputed_bridging_branch_mask, network_data.bridging_branch_mask)
    return (
        runner,
        network_data,
        nodal_injections,
        physical_nodal_injections,
        network_data.bridging_branch_mask,
    )


def _build_branch_outage_definition(
    network_data: NetworkData,
    outage_sets: list[np.ndarray],
) -> Nminus1Definition:
    """Build a Powsybl N-1 definition for the requested branch outage sets."""
    monitored_elements = [
        MonitoredElement(id=branch_id, name=branch_name, type=branch_type, kind="branch")
        for branch_id, branch_name, branch_type in zip(
            network_data.branch_ids,
            network_data.branch_names,
            network_data.branch_types,
            strict=True,
        )
    ]
    contingencies = [Contingency(id="BASECASE", elements=[])]
    for outage_set_index, outage_indices in enumerate(outage_sets):
        contingencies.append(
            Contingency(
                id=f"outage-{outage_set_index}",
                elements=[
                    GridElement(
                        id=network_data.branch_ids[branch_index],
                        name=network_data.branch_names[branch_index],
                        type=network_data.branch_types[branch_index],
                        kind="branch",
                    )
                    for branch_index in outage_indices
                ],
            )
        )
    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        id_type="powsybl",
    )


def _run_dc_loadflows(
    runner: PowsyblRunner,
    nminus1_definition: Nminus1Definition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DC N-0/N-1 loadflows and return ToOp-sign-convention matrices."""
    runner.store_nminus1_definition(nminus1_definition)
    results = runner.run_dc_loadflow([], [])
    n_0_flow, n_1_flow, success = extract_solver_matrices_polars(results, nminus1_definition, 0)
    return -n_0_flow, -n_1_flow, success


def _get_increasing_bridge_multioutages(
    bridging_branch_mask: np.ndarray,
    configured_branch_outage_mask: np.ndarray,
) -> list[np.ndarray]:
    """Build increasing N-k outage sets containing each bridge and configured ordinary outages.

    Each bridge is placed first and then configured ordinary outages are
    appended in index order. The N-1 bridge case is covered separately, so
    these cases start at two outaged branches.
    """
    configured_branch_outages = np.flatnonzero(configured_branch_outage_mask)
    multioutages = []
    for bridge_index in np.flatnonzero(bridging_branch_mask):
        outage_order = np.concatenate(([bridge_index], configured_branch_outages))
        multioutages.extend(outage_order[:n_outages] for n_outages in range(2, outage_order.size + 1))
    return multioutages


@pytest.mark.parametrize("data_folder_fixture", DATA_FOLDER_FIXTURES)
def test_split_modf_n_0_matches_dc(
    data_folder_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Match the production PTDF N-0 flow against the persisted DC setup."""
    data_folder = request.getfixturevalue(data_folder_fixture)
    runner, network_data, nodal_injections, _physical_nodal_injections, _bridging_branch_mask = _load_split_modf_test_grid(
        data_folder
    )
    assert network_data.ptdf is not None
    nminus1_definition = _build_branch_outage_definition(network_data, [])
    reference_n_0_flow, _reference_n_1_flow, _success = _run_dc_loadflows(runner, nminus1_definition)

    np.testing.assert_allclose(network_data.ptdf @ nodal_injections, reference_n_0_flow, atol=TOL)


@pytest.mark.parametrize("data_folder_fixture", DATA_FOLDER_FIXTURES)
def test_normal_modf_matches_dc_for_configured_branch_outages(
    data_folder_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Match the existing MODF path against DC for configured branch outages.

    Shows that the existing MODF path is correct for ordinary N-1 outages.
    """
    data_folder = request.getfixturevalue(data_folder_fixture)
    runner, network_data, nodal_injections, _physical_nodal_injections, _bridging_branch_mask = _load_split_modf_test_grid(
        data_folder
    )
    assert network_data.ptdf is not None
    branches_monitored = jnp.arange(len(network_data.branch_ids), dtype=jnp.int32)

    configured_branch_outages = np.flatnonzero(network_data.outaged_branch_mask)
    assert configured_branch_outages.size > 0
    nminus1_definition = _build_branch_outage_definition(
        network_data,
        [np.array([outage_index]) for outage_index in configured_branch_outages],
    )
    reference_n_0_flow, reference_n_1_flow, success = _run_dc_loadflows(runner, nminus1_definition)
    assert np.all(success)

    n_0_flow = network_data.ptdf @ nodal_injections
    np.testing.assert_allclose(n_0_flow, reference_n_0_flow, atol=TOL)
    for reference_index, outage_index in enumerate(configured_branch_outages):
        modf_flow, modf_success = compute_multi_outage(
            ptdf=jnp.asarray(network_data.ptdf),
            from_node=jnp.asarray(network_data.from_nodes),
            to_node=jnp.asarray(network_data.to_nodes),
            n_0_flow=jnp.asarray(n_0_flow[None, :]),
            multi_outages=jnp.asarray([outage_index], dtype=jnp.int32),
            branches_monitored=branches_monitored,
        )

        assert bool(modf_success)
        reference_flow = reference_n_1_flow[reference_index].copy()
        reference_flow[outage_index] = 0.0
        np.testing.assert_allclose(np.asarray(modf_flow[0]), reference_flow, atol=TOL)


@pytest.mark.parametrize("data_folder_fixture", DATA_FOLDER_FIXTURES)
def test_split_modf_matches_dc_for_configured_non_bridging_branches(
    data_folder_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Match DC flow for every ordinary N-1 branch outage in the test grid."""
    data_folder = request.getfixturevalue(data_folder_fixture)
    runner, network_data, _nodal_injections, physical_nodal_injections, bridging_branch_mask = _load_split_modf_test_grid(
        data_folder
    )
    assert network_data.ptdf is not None
    branches_monitored = jnp.arange(len(network_data.branch_ids), dtype=jnp.int32)
    physical_node_mask = np.zeros(len(network_data.node_ids), dtype=bool)
    physical_node_mask[network_data.from_nodes] = True
    physical_node_mask[network_data.to_nodes] = True

    non_bridging_branches = np.flatnonzero(network_data.outaged_branch_mask)
    assert non_bridging_branches.size > 0
    assert not np.any(bridging_branch_mask[non_bridging_branches])
    nminus1_definition = _build_branch_outage_definition(
        network_data,
        [np.array([outage_index]) for outage_index in non_bridging_branches],
    )
    n_0_flow, reference_n_1_flow, success = _run_dc_loadflows(runner, nminus1_definition)
    assert np.all(success)
    outages = jnp.asarray(non_bridging_branches[:, None], dtype=jnp.int32)
    split_flow, main_bus, main_monitored, _main_imbalance, rank_loss = compute_split_modf_main_grid(
        ptdf=jnp.asarray(network_data.ptdf),
        from_node=jnp.asarray(network_data.from_nodes),
        to_node=jnp.asarray(network_data.to_nodes),
        nodal_injections=jnp.asarray(physical_nodal_injections[None, :]),
        n_0_flow=jnp.asarray(n_0_flow[None, :]),
        outages=outages,
        branches_monitored=branches_monitored,
        slack_bus=jnp.asarray(network_data.slack, dtype=jnp.int32),
    )

    reference_flow = reference_n_1_flow.copy()
    reference_flow[np.arange(non_bridging_branches.size), non_bridging_branches] = 0.0
    np.testing.assert_allclose(np.asarray(split_flow[0]), reference_flow, atol=TOL)
    assert np.all(np.asarray(main_bus)[:, physical_node_mask])
    np.testing.assert_array_equal(np.asarray(main_monitored), np.ones(main_monitored.shape, dtype=bool))
    np.testing.assert_array_equal(np.asarray(rank_loss), np.zeros(non_bridging_branches.size, dtype=int))


@pytest.mark.parametrize("data_folder_fixture", DATA_FOLDER_FIXTURES)
def test_split_modf_matches_main_grid_dc_for_all_bridging_branches(
    data_folder_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Match retained-main-grid DC flow using precomputed masks for bridge outages."""
    data_folder = request.getfixturevalue(data_folder_fixture)
    runner, network_data, _nodal_injections, physical_nodal_injections, bridging_branch_mask = _load_split_modf_test_grid(
        data_folder
    )
    assert network_data.ptdf is not None
    branches_monitored = jnp.arange(len(network_data.branch_ids), dtype=jnp.int32)

    bridging_branches = np.flatnonzero(bridging_branch_mask)
    assert bridging_branches.size > 0
    nminus1_definition = _build_branch_outage_definition(
        network_data,
        [np.array([outage_index]) for outage_index in bridging_branches],
    )
    n_0_flow, reference_n_1_flow, success = _run_dc_loadflows(runner, nminus1_definition)
    assert np.all(success)
    assert len(success) > 0
    outages = jnp.asarray(bridging_branches[:, None], dtype=jnp.int32)

    # Known bridge and 3-winding-transformer contingencies can store this
    # result during preprocessing and skip connectivity search at runtime.
    main_bus = main_grid_reachable_masks_parallel(
        from_node=jnp.asarray(network_data.from_nodes),
        to_node=jnp.asarray(network_data.to_nodes),
        outages=outages,
        slack_bus=jnp.asarray(network_data.slack, dtype=jnp.int32),
        n_bus=network_data.ptdf.shape[1],
    )
    split_flow, main_monitored, _main_imbalance, rank_loss = compute_split_modf_main_grid_from_mask(
        ptdf=jnp.asarray(network_data.ptdf),
        from_node=jnp.asarray(network_data.from_nodes),
        to_node=jnp.asarray(network_data.to_nodes),
        nodal_injections=jnp.asarray(physical_nodal_injections[None, :]),
        n_0_flow=jnp.asarray(n_0_flow[None, :]),
        outages=outages,
        branches_monitored=branches_monitored,
        main_bus=main_bus,
        slack_bus=jnp.asarray(network_data.slack, dtype=jnp.int32),
    )

    split_flow = np.asarray(split_flow[0])
    main_monitored = np.asarray(main_monitored)
    assert not np.any(np.all(np.asarray(main_bus), axis=1))
    np.testing.assert_allclose(split_flow[main_monitored], reference_n_1_flow[main_monitored], atol=TOL)
    assert np.all(np.isnan(split_flow[~main_monitored]))
    assert np.all(np.asarray(rank_loss) > 0)


@pytest.mark.parametrize("data_folder_fixture", DATA_FOLDER_FIXTURES)
def test_split_modf_matches_main_grid_dc_for_increasing_bridge_multioutages(
    data_folder_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Match DC main-grid flow for increasing bridge plus configured ordinary outages."""
    data_folder = request.getfixturevalue(data_folder_fixture)
    runner, network_data, _nodal_injections, physical_nodal_injections, bridging_branch_mask = _load_split_modf_test_grid(
        data_folder
    )
    assert network_data.ptdf is not None
    branches_monitored = jnp.arange(len(network_data.branch_ids), dtype=jnp.int32)
    multioutages = _get_increasing_bridge_multioutages(
        bridging_branch_mask,
        network_data.outaged_branch_mask,
    )

    assert len(multioutages) == int(bridging_branch_mask.sum()) * int(network_data.outaged_branch_mask.sum())
    nminus1_definition = _build_branch_outage_definition(network_data, multioutages)
    n_0_flow, reference_n_1_flow, success = _run_dc_loadflows(runner, nminus1_definition)
    assert np.all(success)
    assert sum(success) > 3
    outage_widths = sorted({len(outages) for outages in multioutages})
    for outage_width in outage_widths:
        outage_indices = [index for index, outages in enumerate(multioutages) if len(outages) == outage_width]
        outages = jnp.asarray(np.stack([multioutages[index] for index in outage_indices]), dtype=jnp.int32)
        split_flow, main_bus, main_monitored, _main_imbalance, rank_loss = compute_split_modf_main_grid(
            ptdf=jnp.asarray(network_data.ptdf),
            from_node=jnp.asarray(network_data.from_nodes),
            to_node=jnp.asarray(network_data.to_nodes),
            nodal_injections=jnp.asarray(physical_nodal_injections[None, :]),
            n_0_flow=jnp.asarray(n_0_flow[None, :]),
            outages=outages,
            branches_monitored=branches_monitored,
            slack_bus=jnp.asarray(network_data.slack, dtype=jnp.int32),
        )

        reference_flow = reference_n_1_flow[outage_indices]
        split_flow = np.asarray(split_flow[0])
        main_monitored = np.asarray(main_monitored)
        assert not np.any(np.all(np.asarray(main_bus), axis=1))
        np.testing.assert_allclose(split_flow[main_monitored], reference_flow[main_monitored], atol=TOL)
        assert np.all(np.isnan(split_flow[~main_monitored]))
        assert np.all(np.asarray(rank_loss) > 0)
