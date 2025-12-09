import json

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax_dataclasses import replace
from optimizer_tests.dc.test_main import assert_topology
from qdax.utils.metrics import default_ga_metrics
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_topology_optimizer.ac.scoring_functions import get_threshold_n_minus1_overload
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    crossover,
    empty_repertoire,
    mutate,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    scoring_function,
    summarize,
    translate_topology,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites
from toop_engine_topology_optimizer.interfaces.messages.results import Topology


def test_translate_topology(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)

    n_disconnectable_branches = len(static_information.dynamic_information.disconnectable_branches)

    action_set = static_information.dynamic_information.action_set
    assert action_set is not None

    max_num_splits = 3
    n_disconnections = 0
    n_psts = 3
    batch_size = 16

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_psts,
    )

    key = jax.random.PRNGKey(0)

    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        substation_split_prob=0.2,
        substation_unsplit_prob=0.00001,
        action_set=action_set,
        n_disconnectable_branches=n_disconnectable_branches,
        n_subs_mutated_lambda=5.0,
        disconnect_prob=0.5,
        reconnect_prob=0.5,
        mutation_repetition=1,
    )

    assert_topology(
        topologies,
        action_set,
        static_information.dynamic_information.disconnectable_branches,
    )

    # Translate the topologies
    branch_topo, disconnections = translate_topology(topologies)

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
    n_disconnectable_branches = len(static_information.dynamic_information.disconnectable_branches)

    max_num_splits = 3
    n_disconnections = 0
    n_psts = 0
    batch_size = 128

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Randomly create some topologies
    topologies = empty_repertoire(
        batch_size,
        max_num_splits,
        n_disconnections,
        n_psts,
    )

    key = jax.random.PRNGKey(0)

    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        substation_split_prob=0.2,
        substation_unsplit_prob=0.00001,
        action_set=action_set,
        n_disconnectable_branches=n_disconnectable_branches,
        n_subs_mutated_lambda=20.0,
        disconnect_prob=0.5,
        reconnect_prob=0.5,
        mutation_repetition=1,
    )

    (
        fitness,
        descriptors,
        metrics,
        emitter_info,
        random_key,
    ) = scoring_function(
        topologies,
        key,
        [static_information.dynamic_information],
        [static_information.solver_config],
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
    disconnectable_branches = static_information.dynamic_information.disconnectable_branches
    n_disconnectable_branches = len(disconnectable_branches)

    max_num_splits = 3
    batch_size = 16
    n_cells = 6

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    # Randomly create some topologies
    topologies = empty_repertoire(batch_size, max_num_splits, 0, 0)

    key = jax.random.PRNGKey(0)

    topologies, key = mutate(
        topologies=topologies,
        random_key=key,
        substation_split_prob=0.2,
        substation_unsplit_prob=0.00001,
        action_set=action_set,
        n_disconnectable_branches=n_disconnectable_branches,
        n_subs_mutated_lambda=20.0,
        disconnect_prob=0.5,
        reconnect_prob=0.5,
        mutation_repetition=1,
    )

    emitter = TrackingMixingEmitter(
        lambda topologies, key: mutate(
            topologies=topologies,
            random_key=key,
            substation_split_prob=0.2,
            substation_unsplit_prob=0.0001,
            action_set=action_set,
            n_disconnectable_branches=n_disconnectable_branches,
            n_subs_mutated_lambda=20.0,
            disconnect_prob=0.5,
            reconnect_prob=0.5,
            mutation_repetition=1,
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
            [static_information.dynamic_information],
            [static_information.solver_config],
            target_metrics=(("overload_energy_n_1", 1.0),),
            observed_metrics=[
                "overload_energy_n_1",
                "underload_energy_n_1",
                "switching_distance",
            ],
            descriptor_metrics=("switching_distance",),
        ),
        emitter,
        default_ga_metrics,
        n_cells_per_dim=(n_cells,),
    )

    repertoire, emitter_state, random_key = algo.init(
        topologies, jax.random.PRNGKey(0), [static_information.dynamic_information]
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
