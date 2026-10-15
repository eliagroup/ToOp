# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from math import sqrt

import jax
import jax.numpy as jnp
import pytest
from toop_engine_topology_optimizer.dc.ga_helpers import (
    MixingEmitterState,
    TrackingMixingEmitter,
    init_running_means,
    update_parent_selection_telemetry,
    update_running_means,
)
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import empty_repertoire
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire
from toop_engine_topology_optimizer.dc.repertoire.parent_selection import (
    BatchedUCBParentSelector,
    ExploitationParentSelector,
    ExplorationParentSelector,
    GreedyParentSelector,
    ParentSelectorState,
    SnapshotUCBParentSelector,
    UCBParentSelector,
    UCBParentSelectorState,
    UniformCellParentSelector,
    UniformParentSelector,
    build_parent_selector,
)


def test_update_running_means_handles_initial_parent_selector_state(monkeypatch: pytest.MonkeyPatch) -> None:
    running_means = init_running_means(n_outages=1, n_devices=1)
    running_means.last_time = 1.0
    monkeypatch.setattr("toop_engine_topology_optimizer.dc.ga_helpers.time.time", lambda: 2.0)

    emitter_state = MixingEmitterState(
        total_branch_combis=jnp.array(1, dtype=int),
        total_inj_combis=jnp.array(2, dtype=int),
        total_num_splits=jnp.array(3, dtype=int),
        parent_selector_state=ParentSelectorState(),
    )

    updated = update_running_means(running_means, emitter_state)

    assert updated.total_branch_combis == 1
    assert updated.total_inj_combis == 2
    assert updated.last_emitter_state.total_num_splits == 3
    assert isinstance(updated.last_emitter_state.parent_selector_state, ParentSelectorState)


def test_tracking_mixing_emitter_exposes_parent_indices() -> None:
    batch_size = 4
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, -jnp.inf, 2.0, 3.0]),
        descriptors=jnp.zeros((4, 1), dtype=int),
        extra_scores={},
        n_cells_per_dim=(4,),
        cell_depth=1,
    )
    emitter = TrackingMixingEmitter(
        lambda topologies, key: (topologies, key),
        lambda topo_a, _topo_b, key: (topo_a, key),
        0.5,
        batch_size=batch_size,
    )

    emitter_state, _ = emitter.init(jax.random.PRNGKey(1), None)
    assert isinstance(emitter_state.parent_selector_state, ParentSelectorState)

    _, emitter_extra_scores, _ = emitter.emit(repertoire, emitter_state, jax.random.PRNGKey(0))

    occupied_indices = jnp.array([0, 2, 3], dtype=int)
    assert set(emitter_extra_scores) == {
        "crossover_parent_indices_a",
        "crossover_parent_indices_b",
        "mutation_parent_indices",
    }
    assert jnp.all(jnp.isin(emitter_extra_scores["crossover_parent_indices_a"], occupied_indices))
    assert jnp.all(jnp.isin(emitter_extra_scores["crossover_parent_indices_b"], occupied_indices))
    assert jnp.all(jnp.isin(emitter_extra_scores["mutation_parent_indices"], occupied_indices))


def test_tracking_mixing_emitter_passes_fitnesses_to_greedy_selector() -> None:
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, -jnp.inf, 2.0, 3.0]),
        descriptors=jnp.zeros((4, 1), dtype=int),
        extra_scores={},
        n_cells_per_dim=(4,),
        cell_depth=1,
    )
    emitter = TrackingMixingEmitter(
        lambda topologies, key: (topologies, key),
        lambda topo_a, _topo_b, key: (topo_a, key),
        0.5,
        batch_size=4,
        parent_selector=GreedyParentSelector(),
    )

    emitter_state, _ = emitter.init(jax.random.PRNGKey(1), None)
    _, emitter_extra_scores, _ = emitter.emit(repertoire, emitter_state, jax.random.PRNGKey(0))

    for parent_indices in emitter_extra_scores.values():
        assert jnp.all(parent_indices == 3)


def test_parent_selection_telemetry_accounts_for_logical_cells() -> None:
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, 2.0, 3.0, 4.0]),
        descriptors=jnp.zeros((4, 1), dtype=int),
        extra_scores={},
        n_cells_per_dim=(2,),
        cell_depth=2,
    )

    selection_counts, success_counts = update_parent_selection_telemetry(
        selection_counts=jnp.zeros((0,), dtype=int),
        success_counts=jnp.zeros((0,), dtype=int),
        repertoire=repertoire,
        extra_scores={
            "crossover_parent_indices_a": jnp.array([0, 1], dtype=int),
            "crossover_parent_indices_b": jnp.array([2, 1], dtype=int),
            "mutation_parent_indices": jnp.array([3], dtype=int),
            "survived_mask": jnp.array([True, False, True]),
        },
    )

    assert jnp.array_equal(selection_counts, jnp.array([2, 3], dtype=int))
    assert jnp.array_equal(success_counts, jnp.array([2, 1], dtype=int))


def test_build_parent_selector_supports_all_modes() -> None:
    for mode in (
        "uniform",
        "uniform_cell",
        "ucb",
        "ucb_batched",
        "ucb_snapshot",
        "exploitation",
        "exploration",
        "greedy",
    ):
        selector = build_parent_selector(
            parent_selection_mode=mode,
            ucb_exploration_constant=1.0,
            cell_depth=2,
            ucb_selection_block_size=3,
        )
        assert selector.mode == mode


def test_snapshot_ucb_parent_selector_prioritizes_unvisited_cells() -> None:
    selector = SnapshotUCBParentSelector(ucb_exploration_constant=1.0, cell_depth=1, temperature=0.1)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([4, 0, 0], dtype=int),
        success_counts=jnp.array([4, 0, 0], dtype=int),
        total_selections=jnp.array(4, dtype=int),
    )

    selected_indices, updated_state, _ = selector.select(
        occupied_mask=jnp.array([True, True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(9),
        num_samples=5,
    )

    assert set(selected_indices[:2].tolist()) == {1, 2}
    assert jnp.all(jnp.isin(selected_indices, jnp.array([0, 1, 2], dtype=int)))
    assert updated_state.total_selections == 9


def test_ucb_state_reuses_cumulative_parent_telemetry() -> None:
    selector = UCBParentSelector(ucb_exploration_constant=1.0, cell_depth=1)
    state = selector.state_from_telemetry(
        parent_selector_state=selector.init_state(),
        selection_counts=jnp.array([2, 3], dtype=int),
        success_counts=jnp.array([1, 2], dtype=int),
    )

    assert isinstance(state, UCBParentSelectorState)
    assert jnp.array_equal(state.selection_counts, jnp.array([2, 3], dtype=int))
    assert jnp.array_equal(state.success_counts, jnp.array([1, 2], dtype=int))
    assert state.total_selections == 5


def test_ucb_parent_selector_prioritizes_unvisited_occupied_cells() -> None:
    selector = UCBParentSelector(ucb_exploration_constant=1.0, cell_depth=1)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([2, 0, 1], dtype=int),
        success_counts=jnp.array([1, 0, 1], dtype=int),
        total_selections=jnp.array(3, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, False]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(0),
        num_samples=4,
    )

    assert selected_indices[0] == 1
    assert jnp.all(jnp.isin(selected_indices, jnp.array([0, 1], dtype=int)))


def test_ucb_parent_selector_prefers_higher_success_rate_with_equal_visits() -> None:
    selector = UCBParentSelector(ucb_exploration_constant=0.5, cell_depth=1)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([3, 3], dtype=int),
        success_counts=jnp.array([3, 0], dtype=int),
        total_selections=jnp.array(6, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(1),
        num_samples=3,
    )

    assert jnp.array_equal(selected_indices, jnp.zeros((3,), dtype=int))


def test_exploitation_parent_selector_prefers_highest_success_rate_at_cell_depth() -> None:
    selector = ExploitationParentSelector(cell_depth=2)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([3, 3], dtype=int),
        success_counts=jnp.array([3, 0], dtype=int),
        total_selections=jnp.array(6, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(4),
        num_samples=20,
    )

    assert jnp.all(jnp.mod(selected_indices, 2) == 0)
    assert jnp.all(jnp.isin(jnp.array([0, 2]), selected_indices))


def test_exploration_parent_selector_prefers_least_selected_cell_at_cell_depth() -> None:
    selector = ExplorationParentSelector(cell_depth=2)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([100, 0], dtype=int),
        success_counts=jnp.array([0, 0], dtype=int),
        total_selections=jnp.array(100, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(5),
        num_samples=20,
    )

    assert jnp.all(jnp.mod(selected_indices, 2) == 1)
    assert jnp.all(jnp.isin(jnp.array([1, 3]), selected_indices))


def test_uniform_parent_selector_samples_candidates_uniformly() -> None:
    selector = UniformParentSelector()

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, True, False, True, False]),
        parent_selector_state=ParentSelectorState(),
        random_key=jax.random.PRNGKey(8),
        num_samples=1000,
    )

    probabilities = jnp.bincount(selected_indices, length=6) / selected_indices.shape[0]
    assert jnp.all((probabilities[jnp.array([0, 1, 2, 4])] > 0.2) & (probabilities[jnp.array([0, 1, 2, 4])] < 0.3))


def test_uniform_cell_parent_selector_is_not_weighted_by_cell_depth() -> None:
    selector = UniformCellParentSelector(cell_depth=3)

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, True, False, True, False]),
        parent_selector_state=ParentSelectorState(),
        random_key=jax.random.PRNGKey(6),
        num_samples=1000,
    )

    selected_cell_zero_ratio = jnp.mean((jnp.mod(selected_indices, 2) == 0).astype(float))
    assert 0.4 < selected_cell_zero_ratio < 0.6
    probabilities = jnp.bincount(selected_indices, length=6) / selected_indices.shape[0]
    assert jnp.max(probabilities[jnp.array([0, 2, 4])]) - jnp.min(probabilities[jnp.array([0, 2, 4])]) < 0.07


def test_greedy_parent_selector_chooses_highest_fitness() -> None:
    selector = GreedyParentSelector()

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, False, True]),
        parent_selector_state=ParentSelectorState(),
        random_key=jax.random.PRNGKey(7),
        num_samples=3,
        fitnesses=jnp.array([2.0, -jnp.inf, 5.0]),
    )

    assert jnp.array_equal(selected_indices, jnp.full((3,), 2, dtype=int))


def test_ucb_parent_selector_samples_from_selected_cell_depth() -> None:
    selector = UCBParentSelector(ucb_exploration_constant=1.0, cell_depth=2)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([1, 0], dtype=int),
        success_counts=jnp.array([0, 0], dtype=int),
        total_selections=jnp.array(1, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, False, False, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(2),
        num_samples=3,
    )

    assert selected_indices[0] == 3
    assert jnp.all(jnp.isin(selected_indices, jnp.array([0, 3], dtype=int)))


def test_ucb_parent_selector_consumes_unvisited_cells_within_batch() -> None:
    selector = UCBParentSelector(ucb_exploration_constant=1.0, cell_depth=1)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([0, 0, 0], dtype=int),
        success_counts=jnp.array([0, 0, 0], dtype=int),
        total_selections=jnp.array(0, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(3),
        num_samples=3,
    )

    assert jnp.array_equal(jnp.sort(selected_indices), jnp.array([0, 1, 2], dtype=int))


def test_batched_ucb_parent_selector_consumes_unvisited_cells_within_block() -> None:
    selector = BatchedUCBParentSelector(ucb_exploration_constant=1.0, cell_depth=1, selection_block_size=8)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([0, 0, 0], dtype=int),
        success_counts=jnp.array([0, 0, 0], dtype=int),
        total_selections=jnp.array(0, dtype=int),
    )

    selected_indices, updated_state, _ = selector.select(
        occupied_mask=jnp.array([True, True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(3),
        num_samples=3,
    )

    assert jnp.array_equal(jnp.sort(selected_indices), jnp.array([0, 1, 2], dtype=int))
    assert isinstance(updated_state, UCBParentSelectorState)
    assert jnp.array_equal(updated_state.selection_counts, jnp.ones((3,), dtype=int))


def test_batched_ucb_parent_selector_block_size_one_matches_exact_ucb() -> None:
    exact_selector = UCBParentSelector(ucb_exploration_constant=1 / sqrt(2), cell_depth=1)
    batched_selector = BatchedUCBParentSelector(
        ucb_exploration_constant=1 / sqrt(2),
        cell_depth=1,
        selection_block_size=1,
    )
    state = UCBParentSelectorState(
        selection_counts=jnp.array([4, 5], dtype=int),
        success_counts=jnp.array([3, 4], dtype=int),
        total_selections=jnp.array(6, dtype=int),
    )
    selection_kwargs = {
        "occupied_mask": jnp.array([True, True]),
        "parent_selector_state": state,
        "random_key": jax.random.PRNGKey(0),
        "num_samples": 1,
    }

    exact_indices, exact_state, _ = exact_selector.select(**selection_kwargs)
    batched_indices, batched_state, _ = batched_selector.select(**selection_kwargs)

    assert jnp.array_equal(exact_indices, jnp.array([1], dtype=int))
    assert jnp.array_equal(batched_indices, exact_indices)
    assert isinstance(exact_state, UCBParentSelectorState)
    assert isinstance(batched_state, UCBParentSelectorState)
    assert jnp.array_equal(batched_state.selection_counts, exact_state.selection_counts)
    assert jnp.array_equal(batched_state.success_counts, exact_state.success_counts)
    assert batched_state.total_selections == exact_state.total_selections


def test_batched_ucb_parent_selector_prefers_high_success_rate_cells() -> None:
    selector = BatchedUCBParentSelector(ucb_exploration_constant=0.5, cell_depth=1, selection_block_size=4)
    state = UCBParentSelectorState(
        selection_counts=jnp.array([3, 3], dtype=int),
        success_counts=jnp.array([3, 0], dtype=int),
        total_selections=jnp.array(6, dtype=int),
    )

    selected_indices, _, _ = selector.select(
        occupied_mask=jnp.array([True, True]),
        parent_selector_state=state,
        random_key=jax.random.PRNGKey(1),
        num_samples=4,
    )

    assert jnp.array_equal(selected_indices, jnp.zeros((4,), dtype=int))
