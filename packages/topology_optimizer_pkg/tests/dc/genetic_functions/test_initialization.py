# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (
    get_repertoire_metrics,
    initialize_genetic_algorithm,
    update_max_mw_flows_according_to_double_limits,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import (
    DiscreteMapElitesRepertoire,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef


def test_update_max_mw_flows_according_to_double_limits(
    static_information_file: str,
) -> None:
    static_information = load_static_information(static_information_file)
    updated_dynamic_informations = update_max_mw_flows_according_to_double_limits(
        [static_information.dynamic_information],
        [static_information.solver_config],
        0.9,
        1.0,
    )
    new_limits = updated_dynamic_informations[0].branch_limits
    old_limits = static_information.dynamic_information.branch_limits
    assert new_limits.max_mw_flow_limited.shape == old_limits.max_mw_flow.shape
    assert old_limits.max_mw_flow.sum() >= new_limits.max_mw_flow_limited.sum()

    static_information_with_n_1 = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            branch_limits=replace(
                static_information.dynamic_information.branch_limits,
                max_mw_flow_n_1=static_information.dynamic_information.branch_limits.max_mw_flow * 2,
            ),
        ),
    )
    updated_dynamic_informations_with_n_1 = update_max_mw_flows_according_to_double_limits(
        [static_information_with_n_1.dynamic_information],
        [static_information_with_n_1.solver_config],
        0.9,
        1.0,
    )
    new_limits = updated_dynamic_informations_with_n_1[0].branch_limits
    old_limits = static_information_with_n_1.dynamic_information.branch_limits

    assert new_limits.max_mw_flow_limited.shape == old_limits.max_mw_flow.shape
    assert new_limits.max_mw_flow_n_1_limited.shape == old_limits.max_mw_flow_n_1.shape
    assert old_limits.max_mw_flow_n_1.sum() >= new_limits.max_mw_flow_n_1_limited.sum()


def test_initialize_genetic_algorithm(
    static_information_file: str,
) -> None:
    static_information = load_static_information(static_information_file)

    (algo, jax_data) = initialize_genetic_algorithm(
        batch_size=1,
        max_num_splits=2,
        max_num_disconnections=2,
        static_informations=[static_information],
        target_metrics=[("overload_energy_n_1", 1.0)],
        substation_split_prob=0.1,
        substation_unsplit_prob=0.1,
        action_set=static_information.dynamic_information.action_set,
        n_subs_mutated_lambda=1.0,
        disconnect_prob=0.1,
        reconnect_prob=0.1,
        proportion_crossover=0.5,
        crossover_mutation_ratio=0.5,
        random_seed=42,
        observed_metrics=("overload_energy_n_1", "split_subs"),
        me_descriptors=(DescriptorDef(metric="split_subs", num_cells=10),),
        distributed=False,
        devices=None,
    )
    assert jax_data.repertoire.fitnesses.shape[0] == 10


def test_distributed_initialize(static_information_file) -> None:
    devices = jax.devices()

    static_information = load_static_information(static_information_file)

    (algo, jax_data) = initialize_genetic_algorithm(
        batch_size=10,
        max_num_splits=2,
        max_num_disconnections=2,
        static_informations=[static_information],
        target_metrics=[("overload_energy_n_1", 1.0)],
        substation_split_prob=0.1,
        substation_unsplit_prob=0.1,
        action_set=static_information.dynamic_information.action_set,
        n_subs_mutated_lambda=1.0,
        disconnect_prob=0.1,
        reconnect_prob=0.1,
        proportion_crossover=0.5,
        crossover_mutation_ratio=0.5,
        random_seed=42,
        observed_metrics=("overload_energy_n_1", "split_subs"),
        me_descriptors=(DescriptorDef(metric="split_subs", num_cells=5),),
        distributed=True,
        devices=devices,
    )

    def assert_node(x):
        assert x.shape[0] == 2, f"Expected 2 to be the first dimension, got {x.shape}"
        assert len(x.global_shards) == len(devices)

    jax.tree_util.tree_map(assert_node, jax_data)


def test_get_repertoire_metrics():
    fitnesses = jnp.array([1, 2, 3, 4, 5, 6, 7, -jnp.inf])
    metrics = {
        "test_metric": jnp.array([9, 10, 11, 12, 13, 14, 15, 16]),
        "test_metric2": jnp.array([17, 18, 19, 20, 21, 22, 23, 24]),
    }
    descriptors = jnp.array([25, 26, 27, 28, 29, 30, 31, 32])

    test_repertoire = DiscreteMapElitesRepertoire(
        genotypes=jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
        fitnesses=fitnesses,
        descriptors=descriptors,
        extra_scores=metrics,
        n_cells_per_dim=(8,),
        cell_depth=1,
    )

    fitness_best, metrics_best = get_repertoire_metrics(test_repertoire, ["test_metric"])
    assert fitness_best == jnp.array(7)
    assert metrics_best["test_metric"] == jnp.array(15)
    assert "test_metric2" not in metrics_best.keys()
    fitness_again, metrics_again = get_repertoire_metrics(test_repertoire, ["test_metric"])
    assert jnp.all(fitness_best == fitness_again)
    assert jnp.all(metrics_best["test_metric"] == metrics_again["test_metric"])
