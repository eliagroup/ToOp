# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import pytest
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import BBOutageBaselineAnalysis
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import (
    get_repertoire_metrics,
    initialize_genetic_algorithm,
    update_max_mw_flows_according_to_double_limits,
    update_static_information,
    verify_static_information,
)
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    DisconnectionMutationConfig,
    MutationConfig,
    SubstationMutationConfig,
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
        (static_information.dynamic_information,),
        (static_information.solver_config,),
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
        (static_information_with_n_1.dynamic_information,),
        (static_information_with_n_1.solver_config,),
        0.9,
        1.0,
    )
    new_limits = updated_dynamic_informations_with_n_1[0].branch_limits
    old_limits = static_information_with_n_1.dynamic_information.branch_limits

    assert new_limits.max_mw_flow_limited.shape == old_limits.max_mw_flow.shape
    assert new_limits.max_mw_flow_n_1_limited.shape == old_limits.max_mw_flow_n_1.shape
    assert old_limits.max_mw_flow_n_1.sum() >= new_limits.max_mw_flow_n_1_limited.sum()


def test_update_static_information_overrides_busbar_penalty(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)
    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            clip_bb_outage_penalty=True,
        ),
        dynamic_information=replace(
            static_information.dynamic_information,
            bb_outage_baseline_analysis=BBOutageBaselineAnalysis(
                overload=jnp.array(1.0),
                success_count=jnp.array(2),
                more_splits_penalty=jnp.array(50.0),
                overload_weight=static_information.dynamic_information.branch_limits.overload_weight,
                max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow,
            ),
        ),
    )

    updated = update_static_information(
        (static_information,),
        batch_size=3,
        enable_nodal_inj_optim=False,
        enable_bb_outage=True,
        bb_outage_as_nminus1=False,
        clip_bb_outage_penalty=False,
        bb_outage_more_islands_penalty=125.0,
    )[0]

    assert updated.solver_config.batch_size_bsdf == 3
    assert updated.solver_config.batch_size_injection == 3
    assert updated.solver_config.enable_bb_outages
    assert updated.solver_config.bb_outage_as_nminus1 is False
    assert updated.solver_config.clip_bb_outage_penalty is False
    assert updated.dynamic_information.bb_outage_baseline_analysis is not None
    assert updated.dynamic_information.bb_outage_baseline_analysis.more_splits_penalty == 125.0


def test_update_static_information_removes_busbar_data_when_disabled(static_information_file: str) -> None:
    static_information = load_static_information(static_information_file)
    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            bb_outage_baseline_analysis=BBOutageBaselineAnalysis(
                overload=jnp.array(1.0),
                success_count=jnp.array(2),
                more_splits_penalty=jnp.array(50.0),
                overload_weight=static_information.dynamic_information.branch_limits.overload_weight,
                max_mw_flow=static_information.dynamic_information.branch_limits.max_mw_flow,
            ),
        ),
    )

    updated = update_static_information(
        (static_information,),
        batch_size=3,
        enable_nodal_inj_optim=False,
        enable_bb_outage=False,
        bb_outage_as_nminus1=False,
        clip_bb_outage_penalty=False,
        bb_outage_more_islands_penalty=125.0,
    )[0]

    assert updated.solver_config.enable_bb_outages is False
    assert updated.dynamic_information.bb_outage_baseline_analysis is None
    assert updated.dynamic_information.non_rel_bb_outage_data is None
    assert updated.dynamic_information.action_set.rel_bb_outage_data is None


def test_initialize_genetic_algorithm(
    static_information_file: str,
) -> None:
    static_information = load_static_information(static_information_file)
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
    (algo, jax_data) = initialize_genetic_algorithm(
        batch_size=1,
        max_num_splits=2,
        max_num_disconnections=2,
        static_informations=tuple([static_information]),
        target_metrics=(("overload_energy_n_1", 1.0),),
        action_set=static_information.dynamic_information.action_set,
        mutation_config=mutation_config,
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
    (algo, jax_data) = initialize_genetic_algorithm(
        batch_size=10,
        max_num_splits=2,
        max_num_disconnections=2,
        static_informations=tuple([static_information]),
        target_metrics=tuple([("overload_energy_n_1", 1.0)]),
        action_set=static_information.dynamic_information.action_set,
        mutation_config=mutation_config,
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
        return x

    jax.tree_util.tree_map(assert_node, jax_data)


def test_get_repertoire_metrics():
    fitnesses = jnp.array([1, 2, 3, 4, 5, 6, 7, -jnp.inf])
    metrics = {
        "overload_energy_n_1": jnp.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=float),
        "overload_energy_n_0": jnp.array([17, 18, 19, 20, 21, 22, 23, 24], dtype=float),
    }
    descriptors = jnp.array([[25], [26], [27], [28], [29], [30], [31], [32]])
    genotypes = Genotype(  # 4 topologies
        action_index=jnp.array([[0], [1], [2], [0]]),
        disconnections=jnp.array([[0], [1], [2], [0]]),
        nodal_injections_optimized=None,
    )
    test_repertoire = DiscreteMapElitesRepertoire(
        genotypes=genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        extra_scores=metrics,
        n_cells_per_dim=(8,),
        cell_depth=1,
    )

    fitness_best, metrics_best = get_repertoire_metrics(test_repertoire, ("overload_energy_n_1",))
    assert fitness_best == jnp.array(7)
    assert metrics_best["overload_energy_n_1"] == jnp.array(15)
    assert "overload_energy_n_0" not in metrics_best.keys()
    fitness_again, metrics_again = get_repertoire_metrics(test_repertoire, ("overload_energy_n_1",))
    assert jnp.all(fitness_best == fitness_again)
    assert jnp.all(metrics_best["overload_energy_n_1"] == metrics_again["overload_energy_n_1"])

    fitness_again, two_metrics = get_repertoire_metrics(test_repertoire, ("overload_energy_n_1", "overload_energy_n_0"))
    assert jnp.all(fitness_best == fitness_again)
    assert jnp.all(metrics_best["overload_energy_n_1"] == two_metrics["overload_energy_n_1"])
    assert "overload_energy_n_0" in two_metrics.keys()


def test_verify_static_information(static_information_file) -> None:
    static_information = load_static_information(static_information_file)

    # This should not raise an error
    verify_static_information([static_information], max_num_disconnections=1, enable_nodal_inj_optim=False)
    # Should raise because there are no PSTs in this grid but nodal injection optimization is enabled
    with pytest.raises(AssertionError):
        verify_static_information([static_information], max_num_disconnections=5, enable_nodal_inj_optim=True)
