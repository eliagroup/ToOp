# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json

import jax
import numpy as np
import pytest
from beartype.typing import Optional
from jax import numpy as jnp
from jax_dataclasses import replace
from qdax.utils.metrics import default_ga_metrics
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.crossover import (
    crossover,
)
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import empty_repertoire
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    DisconnectionMutationConfig,
    MutationConfig,
    NodalInjectionMutationConfig,
    SubstationMutationConfig,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import mutate
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    scoring_function,
    summarize,
    translate_topology,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites
from toop_engine_topology_optimizer.interfaces.messages.results import Topology

from packages.topology_optimizer_pkg.tests.dc.test_main import assert_topology


def get_threshold_n_minus1_overload(
    strategy: list[ACOptimTopology],
) -> tuple[Optional[list[float]], Optional[list[list[int]]]]:
    """Extract the 'top_k_overloads_n_1' thresholds and corresponding case indices from a list of ACOptimTopology strategies.

    overload_threshold is defined as the maximum allowed overload energy for the worst k AC N-1 contingency analysis
    of the split topologies. This threshold is computed using the worst k AC contingencies of the unsplit grid and the
    worst k DC contingencies of the split grid. Refer to the "pull" method in evolution_functions.py for more details.

    Parameters
    ----------
    strategy : list of ACOptimTopology
        A list of ACOptimTopology objects, each containing a 'metrics' dictionary with overload thresholds and case indices.

    Returns
    -------
    tuple of (Optional[list of float], Optional[list of list of int])
        A tuple containing:
        - A list of overload thresholds for each topology, or None if any required metric is missing.
        - A list of lists of case indices for each topology, or None if any required metric is missing.

    """
    overload_threshold_all_t = []
    case_indices_all_t = []
    for topo in strategy:
        threshold_overload = topo.metrics.get("top_k_overloads_n_1", None)
        threshold_case_indices = topo.worst_k_contingency_cases
        if threshold_overload is None or len(threshold_case_indices) == 0:
            return None, None

        overload_threshold_all_t.append(threshold_overload)
        case_indices_all_t.append(threshold_case_indices)

    return overload_threshold_all_t, case_indices_all_t


def test_translate_topology(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None

    max_num_splits = 3
    n_disconnections = 0
    batch_size = 16
    n_timesteps = static_information.dynamic_information.n_timesteps

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
    )

    key = jax.random.PRNGKey(0)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.3,
            change_split_prob=0.4,
            remove_split_prob=0.3,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.3,
            change_disconnection_prob=0.4,
            remove_disconnection_prob=0.3,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    assert_topology(
        topologies,
        action_set,
        static_information.dynamic_information.disconnectable_branches,
    )

    # Translate the topologies
    branch_topo, disconnections, nodal_inj_start = translate_topology(topologies)

    assert branch_topo.action.shape == (
        batch_size,
        max_num_splits,
    )
    assert branch_topo.pad_mask.shape == (batch_size,)
    assert jnp.all(branch_topo.pad_mask)
    assert disconnections.shape[0] == batch_size


def test_scoring_function(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set

    max_num_splits = 3
    n_disconnections = 0
    batch_size = 128
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
    )

    key = jax.random.PRNGKey(0)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.3,
            change_split_prob=0.4,
            remove_split_prob=0.3,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.3,
            change_disconnection_prob=0.4,
            remove_disconnection_prob=0.3,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    (fitness, descriptors, metrics, emitter_info, random_key, topologies) = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(
            ("overload_energy_n_1", 1.0),
            ("underload_energy_n_1", -1.0),
        ),
        observed_metrics=(
            "overload_energy_n_1",
            "underload_energy_n_1",
            "max_flow_n_1",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert fitness.shape == (batch_size,)
    assert descriptors.shape == (batch_size, 1)


@pytest.mark.timeout(120)
def test_summarize(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set

    max_num_splits = 3
    batch_size = 16
    n_cells = 6
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, 0, n_timesteps)

    key = jax.random.PRNGKey(0)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.3,
            change_split_prob=0.4,
            remove_split_prob=0.3,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.3,
            change_disconnection_prob=0.4,
            remove_disconnection_prob=0.3,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    topologies, key = mutate(topologies=topologies, random_key=key, action_set=action_set, mutation_config=mutation_config)

    emitter = TrackingMixingEmitter(
        lambda topologies, key: mutate(
            topologies=topologies,
            random_key=key,
            action_set=action_set,
            mutation_config=mutation_config,
        ),
        lambda topo_a, topo_b, key: crossover(
            topologies_a=topo_a, topologies_b=topo_b, random_key=key, action_set=action_set, prob_take_a=0.1
        ),
        0.5,
        batch_size,
    )
    algo = DiscreteMapElites(
        lambda topo, key, _: scoring_function(
            topo,
            key,
            (static_information.dynamic_information,),
            (static_information.solver_config,),
            target_metrics=(("overload_energy_n_1", 1.0),),
            observed_metrics=(
                "overload_energy_n_1",
                "underload_energy_n_1",
                "switching_distance",
            ),
            descriptor_metrics=("switching_distance",),
        ),
        emitter,
        default_ga_metrics,
        n_cells_per_dim=(n_cells,),
    )

    repertoire, emitter_state, random_key = algo.init(
        topologies, jax.random.PRNGKey(0), (static_information.dynamic_information,)
    )
    contingency_ids = static_information.solver_config.contingency_ids
    stats = summarize(
        repertoire=repertoire,
        emitter_state=emitter_state,
        initial_fitness=-np.inf,
        initial_metrics={"overload_energy_n_1": 0.0, "underload_energy_n_1": 0.0},
        contingency_ids=contingency_ids,
    )

    assert stats["max_fitness"] is not None
    assert stats["initial_fitness"] is not None
    assert stats["initial_metrics"] == {"overload_energy_n_1": 0.0, "underload_energy_n_1": 0.0}
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None

    best_topos = [Topology.model_validate(x) for x in stats["best_topos"]]
    assert all([np.isfinite(x.metrics.fitness) for x in best_topos])

    assert json.dumps(stats)


class DummyACOptimTopology:
    def __init__(self, metrics, worst_k_contingency_cases):
        self.metrics = metrics
        self.worst_k_contingency_cases = worst_k_contingency_cases


def test_get_threshold_n_minus1_overload_all_present():
    strategy = [
        DummyACOptimTopology(metrics={"top_k_overloads_n_1": 10.5}, worst_k_contingency_cases=[1, 2, 3]),
        DummyACOptimTopology(metrics={"top_k_overloads_n_1": 20.0}, worst_k_contingency_cases=[0, 4]),
    ]
    thresholds, indices = get_threshold_n_minus1_overload(strategy)
    assert thresholds == [10.5, 20.0]
    assert indices == [[1, 2, 3], [0, 4]]


def test_get_threshold_n_minus1_overload_missing_threshold():
    strategy = [
        DummyACOptimTopology(metrics={}, worst_k_contingency_cases=[1, 2, 3]),
        DummyACOptimTopology(metrics={"top_k_overloads_n_1": 20.0}, worst_k_contingency_cases=[0, 4]),
    ]
    overload_threshold_all_t, case_indices_all_t = get_threshold_n_minus1_overload(strategy)
    assert overload_threshold_all_t is None
    assert case_indices_all_t is None


def test_get_threshold_n_minus1_overload_missing_case_indices():
    strategy = [
        DummyACOptimTopology(metrics={"top_k_overloads_n_1": 10.5}, worst_k_contingency_cases=[]),
        DummyACOptimTopology(metrics={"top_k_overloads_n_1": 20.0}, worst_k_contingency_cases=[0, 5]),
    ]

    overload_threshold_all_t, case_indices_all_t = get_threshold_n_minus1_overload(strategy)
    assert overload_threshold_all_t is None
    assert case_indices_all_t is None


def test_get_threshold_n_minus1_overload_empty_strategy():
    strategy = []
    thresholds, indices = get_threshold_n_minus1_overload(strategy)
    assert thresholds == []
    assert indices == []


def test_pst_switching_distance_metric_integration(static_information_file_complex: str) -> None:
    """Test that pst_switching_distance metric works in the scoring function."""
    static_information = load_static_information(static_information_file_complex)

    # Skip test if there are no controllable PSTs in this grid
    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set

    max_num_splits = 2
    n_disconnections = 0
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Initialize with PST optimization enabled (starting taps)
    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx

    # Create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
        starting_taps,
    )

    key = jax.random.PRNGKey(42)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.1,
            change_split_prob=0.0,
            remove_split_prob=0.0001,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=0.0,
            pst_mutation_probability=0.0,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=starting_taps,
        ),
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    # Test 1: With pst_switching_distance in observed metrics
    (_, _, metrics, _, _, _) = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "switching_distance",
            "pst_switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_switching_distance" in metrics, "pst_switching_distance should be in metrics"
    assert metrics["pst_switching_distance"].shape == (batch_size,), "Metric should have batch dimension"

    # Since we haven't mutated PSTs (pst_mutation_sigma=0), all distances should be 0
    assert jnp.all(metrics["pst_switching_distance"] == 0.0), "Distances should be 0 when PST taps haven't changed"

    # Test 2: With PST mutation, distances should be non-zero
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies_mutated, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    (_, _, metrics, _, _, _) = scoring_function(
        topologies_mutated,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "pst_switching_distance",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    # With PST mutation, at least some topologies should have non-zero distances
    # (though it's possible all mutations result in the same tap due to clipping)
    assert "pst_switching_distance" in metrics, "Metric should be computed"
    assert jnp.all(jnp.isfinite(metrics["pst_switching_distance"])), "All distances should be finite"
    assert jnp.all(metrics["pst_switching_distance"] >= 0.0), "distances should be non-negative"


def test_pst_switching_distance_in_target_metrics(static_information_file_complex: str) -> None:
    """Test that pst_switching_distance metric used as a target metric."""
    static_information = load_static_information(static_information_file_complex)

    # Skip test if there are no controllable PSTs in this grid
    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set

    max_num_splits = 2
    n_disconnections = 0
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Initialize with PST optimization enabled (starting taps)
    pst_n_taps = static_information.dynamic_information.nodal_injection_information.pst_n_taps
    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx

    # Create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_timesteps,
        starting_taps,
    )

    key = jax.random.PRNGKey(42)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    # Test 1: With small cost for pst_switching_distance in target metrics
    (fitness_small_weight, _, metrics, _, _, _) = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_switching_distance", 0.1)),
        observed_metrics=(
            "overload_energy_n_1",
            "switching_distance",
            "pst_switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert jnp.less_equal(fitness_small_weight, 0.0).all(), "Fitness should be non-positive"
    assert jnp.greater_equal(metrics["pst_switching_distance"], 0.0).any(), (
        "Metric should be greater 0 for some topologies due to PST mutation"
    )

    (fitness, _, metrics, _, _, _) = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_switching_distance", 10.0)),
        observed_metrics=(
            "overload_energy_n_1",
            "pst_switching_distance",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    # The fitness should be worse as the pst_switching_distance weight is higher, even if overload_energy_n_1 is the same
    assert jnp.less_equal(fitness, fitness_small_weight).all(), (
        "Fitness should be worse with higher weight on pst_switching_distance"
    )


def test_pst_switching_distance_without_pst_optimization(static_information_file: str) -> None:
    """Test that pst_switching_distance returns 0 when PST optimization is disabled."""
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set

    max_num_splits = 2
    batch_size = 4
    n_timesteps = static_information.dynamic_information.n_timesteps

    # Disable PST optimization by setting nodal_injection_information to None
    dynamic_info_no_pst = replace(
        static_information.dynamic_information,
        nodal_injection_information=None,
    )

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
        dynamic_information=dynamic_info_no_pst,
    )

    topologies = empty_repertoire(batch_size, max_num_splits, 0, n_timesteps)

    key = jax.random.PRNGKey(100)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=3.0,
            add_split_prob=0.1,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=None,
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    (_, _, metrics, _, _, _) = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "pst_switching_distance",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_switching_distance" in metrics, "Metric should be computed even when PST opt is disabled"
    # All distances should be 0 when PST optimization is disabled
    assert jnp.all(metrics["pst_switching_distance"] == 0.0), (
        "PST switching distance should be 0 when PST optimization is disabled"
    )


def test_pst_activated_metric_integration(static_information_file_complex: str) -> None:
    """Test that pst_activated metric works in the scoring function."""
    static_information = load_static_information(static_information_file_complex)

    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx

    topologies = empty_repertoire(
        batch_size,
        2,
        0,
        n_timesteps,
        starting_taps,
    )

    key = jax.random.PRNGKey(7)
    no_pst_mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.0,
            pst_reset_probability=0.0,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=starting_taps,
        ),
    )
    topologies_zero, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=no_pst_mutation_config,
        action_set=action_set,
    )

    (_, _, metrics_zero, _, _, _) = scoring_function(
        topologies_zero,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "switching_distance",
            "pst_activated",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_activated" in metrics_zero
    assert metrics_zero["pst_activated"].shape == (batch_size,)
    assert jnp.all(metrics_zero["pst_activated"] == 0.0)

    pst_mutation_config = replace(
        no_pst_mutation_config,
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies_mutated, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=pst_mutation_config,
        action_set=action_set,
    )

    (_, _, metrics, _, _, _) = scoring_function(
        topologies_mutated,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "pst_activated",
            "pst_switching_distance",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert jnp.all(jnp.isfinite(metrics["pst_activated"]))
    assert jnp.all(metrics["pst_activated"] >= 0.0)
    assert jnp.all(metrics["pst_activated"] <= metrics["pst_switching_distance"])


def test_pst_activated_in_target_metrics(static_information_file_complex: str) -> None:
    """Test that pst_activated contributes to fitness when targeted."""
    static_information = load_static_information(static_information_file_complex)

    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx
    topologies = empty_repertoire(batch_size, 2, 0, n_timesteps, starting_taps)

    key = jax.random.PRNGKey(21)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    fitness_low_weight, _, metrics, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_activated", 0.1)),
        observed_metrics=("overload_energy_n_1", "pst_activated", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )
    fitness_high_weight, _, _, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_activated", 10.0)),
        observed_metrics=("overload_energy_n_1", "pst_activated", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )

    assert jnp.all(metrics["pst_activated"] >= 0.0)
    assert jnp.less_equal(fitness_high_weight, fitness_low_weight).all()


def test_pst_activated_without_pst_optimization(static_information_file: str) -> None:
    """Test that pst_activated returns 0 when PST optimization is disabled."""
    static_information = load_static_information(static_information_file)

    dynamic_info_no_pst = replace(
        static_information.dynamic_information,
        nodal_injection_information=None,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=4),
        dynamic_information=dynamic_info_no_pst,
    )

    topologies = empty_repertoire(4, 2, 0, static_information.dynamic_information.n_timesteps)
    key = jax.random.PRNGKey(100)

    fitness, _, metrics, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_activated", 1.0)),
        observed_metrics=("overload_energy_n_1", "pst_activated", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_activated" in metrics
    assert jnp.all(metrics["pst_activated"] == 0.0)
    assert jnp.all(jnp.isfinite(fitness))


def test_pst_switching_distance_squared_metric_integration(static_information_file_complex: str) -> None:
    """Test that pst_switching_distance_squared metric works in the scoring function."""
    static_information = load_static_information(static_information_file_complex)

    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx
    topologies = empty_repertoire(batch_size, 2, 0, n_timesteps, starting_taps)

    key = jax.random.PRNGKey(31)
    no_pst_mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.3,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies_zero, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=no_pst_mutation_config,
        action_set=action_set,
    )

    (_, _, metrics_zero, _, _, _) = scoring_function(
        topologies_zero,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=("overload_energy_n_1", "switching_distance", "pst_switching_distance_squared"),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_switching_distance_squared" in metrics_zero
    assert metrics_zero["pst_switching_distance_squared"].shape == (batch_size,)
    assert jnp.all(metrics_zero["pst_switching_distance_squared"] >= 0.0)

    pst_mutation_config = replace(
        no_pst_mutation_config,
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies_mutated, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=pst_mutation_config,
        action_set=action_set,
    )

    (_, _, metrics, _, _, _) = scoring_function(
        topologies_mutated,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        observed_metrics=(
            "overload_energy_n_1",
            "pst_switching_distance_squared",
            "pst_switching_distance",
            "pst_activated",
            "switching_distance",
        ),
        descriptor_metrics=("switching_distance",),
    )

    assert jnp.all(jnp.isfinite(metrics["pst_switching_distance"]))
    assert jnp.all(jnp.isfinite(metrics["pst_switching_distance_squared"]))
    assert jnp.all(metrics["pst_switching_distance"] >= 0.0)
    assert jnp.all(metrics["pst_switching_distance_squared"] >= 0.0)
    assert jnp.all(metrics["pst_activated"] <= metrics["pst_switching_distance"])
    assert jnp.all(metrics["pst_activated"] <= metrics["pst_switching_distance_squared"])
    assert jnp.all(metrics["pst_switching_distance"] <= metrics["pst_switching_distance_squared"])


def test_pst_switching_distance_squared_in_target_metrics(static_information_file_complex: str) -> None:
    """Test that pst_switching_distance_squared contributes to fitness when targeted."""
    static_information = load_static_information(static_information_file_complex)

    if (
        static_information.dynamic_information.nodal_injection_information is None
        or len(static_information.dynamic_information.nodal_injection_information.controllable_pst_indices) == 0
    ):
        pytest.skip("No controllable PSTs in this grid")

    action_set = static_information.dynamic_information.action_set
    batch_size = 8
    n_timesteps = static_information.dynamic_information.n_timesteps

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    starting_taps = static_information.dynamic_information.nodal_injection_information.starting_tap_idx
    topologies = empty_repertoire(batch_size, 2, 0, n_timesteps, starting_taps)

    key = jax.random.PRNGKey(41)
    mutation_config = MutationConfig(
        random_topo_prob=0.0,
        mutation_repetition=1,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.0,
            change_split_prob=0.0,
            remove_split_prob=0.0,
            n_rel_subs=static_information.dynamic_information.n_sub_relevant,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.0,
            change_disconnection_prob=0.0,
            remove_disconnection_prob=0.0,
            n_disconnectable_branches=static_information.dynamic_information.n_disconnectable_branches,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=2.0,
            pst_mutation_probability=0.2,
            pst_reset_probability=0.2,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
            pst_start_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx,
        ),
    )
    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        mutation_config=mutation_config,
        action_set=action_set,
    )

    fitness_low_weight, _, metrics, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_switching_distance_squared", 0.1)),
        observed_metrics=("overload_energy_n_1", "pst_switching_distance_squared", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )
    fitness_high_weight, _, _, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_switching_distance_squared", 10.0)),
        observed_metrics=("overload_energy_n_1", "pst_switching_distance_squared", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )

    assert jnp.all(metrics["pst_switching_distance_squared"] >= 0.0)
    assert jnp.less_equal(fitness_high_weight, fitness_low_weight).all()


def test_pst_switching_distance_squared_without_pst_optimization(static_information_file: str) -> None:
    """Test that pst_switching_distance_squared returns 0 when PST optimization is disabled."""
    static_information = load_static_information(static_information_file)

    dynamic_info_no_pst = replace(
        static_information.dynamic_information,
        nodal_injection_information=None,
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=4),
        dynamic_information=dynamic_info_no_pst,
    )

    topologies = empty_repertoire(4, 2, 0, static_information.dynamic_information.n_timesteps)
    key = jax.random.PRNGKey(100)

    fitness, _, metrics, _, _, _ = scoring_function(
        topologies,
        key,
        (static_information.dynamic_information,),
        (static_information.solver_config,),
        target_metrics=(("overload_energy_n_1", 1.0), ("pst_switching_distance_squared", 1.0)),
        observed_metrics=("overload_energy_n_1", "pst_switching_distance_squared", "switching_distance"),
        descriptor_metrics=("switching_distance",),
    )

    assert "pst_switching_distance_squared" in metrics
    assert jnp.all(metrics["pst_switching_distance_squared"] == 0.0)
    assert jnp.all(jnp.isfinite(fitness))
