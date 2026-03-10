# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int, Shaped
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import JaxOptimizerData, initialize_genetic_algorithm
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.config import (
    DisconnectionMutationConfig,
    MutationConfig,
    NodalInjectionMutationConfig,
    SubstationMutationConfig,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites
from toop_engine_topology_optimizer.dc.worker.optimizer import run_single_device_epoch
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef


def test_jax_no_tracing_flag() -> None:
    """No Re-compilation test for JAX functions

    Simple example to check that simple functions do not recompile, when called with traced objects
    of the correct shape/dtype, and that they do recompile when called with non-traced objects
    or traced objects of the wrong shape/dtype."""
    jax.clear_caches()

    def increment(x: Shaped[Array, " len_array"]) -> Shaped[Array, " len_array"]:
        return x + 1

    def change_type(x: Int[Array, " len_array"]) -> Float[Array, " len_array"]:
        return x.astype(jnp.float32)

    def increment_by_more(x: Int[Array, " len_array"]) -> Shaped[Array, " len_array"]:
        return x + 2

    increment_jit = jax.jit(increment)
    change_type_jit = jax.jit(change_type)
    increment_by_more_jit = jax.jit(increment_by_more)

    result = jnp.array(0, dtype=jnp.int32)
    other_values_result = jnp.array(1, dtype=jnp.int32)
    other_shape_result = jnp.array([1], dtype=jnp.int32)
    other_dtypes_result = jnp.array([0], dtype=jnp.float32)
    # Compile once for the traced shape/dtype, and once for each non-traced argument variation.
    new_traced_result = increment_jit(result)

    ## Call with different shapes, but allow retracing
    increment_jit(result)
    increment_jit(new_traced_result)
    increment_jit(other_dtypes_result)
    increment_jit(other_values_result)
    increment_jit(other_shape_result)

    jax.clear_caches()
    new_traced_result = increment_jit(result)
    changed_type_trace_result = change_type_jit(result)
    more_increment_trace_result = increment_by_more_jit(result)
    with jax.no_tracing():
        increment_jit(result)  # Should not re-trace or raise.
        increment_jit(new_traced_result)  # Should not re-trace or raise.
        increment_jit(other_values_result)  # Should not re-trace or raise.
        increment_by_more_jit(more_increment_trace_result)  # Should not re-trace or raise.
        with pytest.raises(RuntimeError, match="re-tracing function increment"):
            increment_jit(changed_type_trace_result)  # Traced array from other function with different dtype should raise.
        with pytest.raises(RuntimeError, match="re-tracing function increment"):
            increment_jit(other_dtypes_result)  # Non-traced array with other dtype should raise
        with pytest.raises(RuntimeError, match="re-tracing function increment"):
            increment_jit(other_shape_result)


def _initialize_small_optimizer(static_information_file: str, batch_size: int) -> tuple[DiscreteMapElites, JaxOptimizerData]:
    static_information = load_static_information(static_information_file)
    mutation_config = MutationConfig(
        mutation_repetition=1,
        random_topo_prob=0.0,
        substation_mutation_config=SubstationMutationConfig(
            n_subs_mutated_lambda=1.0,
            add_split_prob=0.1,
            change_split_prob=0.1,
            remove_split_prob=0.1,
            n_rel_subs=20,
        ),
        disconnection_mutation_config=DisconnectionMutationConfig(
            add_disconnection_prob=0.1,
            change_disconnection_prob=0.1,
            remove_disconnection_prob=0.1,
            n_disconnectable_branches=5,
        ),
        nodal_injection_mutation_config=NodalInjectionMutationConfig(
            pst_mutation_sigma=0.0,
            pst_n_taps=static_information.dynamic_information.nodal_injection_information.pst_n_taps,
        )
        if static_information.dynamic_information.nodal_injection_information is not None
        else None,
    )
    algo, jax_data = initialize_genetic_algorithm(
        batch_size=batch_size,
        max_num_splits=2,
        max_num_disconnections=2,
        static_informations=(static_information,),
        target_metrics=(("overload_energy_n_1", 1.0),),
        action_set=static_information.dynamic_information.action_set,
        proportion_crossover=0.5,
        crossover_mutation_ratio=0.5,
        random_seed=42,
        observed_metrics=("overload_energy_n_1", "split_subs"),
        me_descriptors=(DescriptorDef(metric="split_subs", num_cells=8),),
        distributed=False,
        mutation_config=mutation_config,
    )
    return algo, jax_data


def test_dc_optimization_does_not_retrace(
    static_information_file: str,
) -> None:
    """Test that disables jax tracing temporarily to check that during optimization
    no re-compilations occur
    """
    jax.clear_caches()
    algo, jax_data = _initialize_small_optimizer(static_information_file, batch_size=2)

    with jax.no_tracing():
        with pytest.raises(RuntimeError, match="re-tracing function"):
            run_single_device_epoch(jax_data, 1, algo.update)
    jax.clear_caches()
    jax_data_traced = run_single_device_epoch(
        jax_data, 1, algo.update
    )  # Warm up to ensure any lazy compilation paths are triggered before measurement.
    assert jax.tree_util.tree_structure(jax_data) == jax.tree_util.tree_structure(jax_data_traced), (
        "Trees have different shapes"
    )

    with jax.no_tracing():
        jax_data_traced_1 = run_single_device_epoch(jax_data_traced, 1, algo.update)  # Should not re-trace or raise.
        jax_data_traced_2 = run_single_device_epoch(jax_data_traced_1, 1, algo.update)  # Should not re-trace or raise.
        with pytest.raises(RuntimeError, match="re-tracing function"):
            # Changing static arg leads to recompilation
            run_single_device_epoch(jax_data_traced_2, 2, algo.update)  # Should raise due to non-traced input.
        with pytest.raises(RuntimeError, match="re-tracing function"):
            # untraced input of jax data leads to recompilation
            _, jax_data_increased_batch = _initialize_small_optimizer(static_information_file, batch_size=4)

            run_single_device_epoch(jax_data_increased_batch, 1, algo.update)  # Should raise due to non-traced input.
    # Check sensibility of the results
    # This leads to retracing, so outside of the no_tracing context
    # First run
    assert jax.tree_util.tree_structure(jax_data_traced_1) == jax.tree_util.tree_structure(jax_data_traced), (
        "Trees have different shapes"
    )
    equal_leaves = jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), jax_data, jax_data_traced)
    assert not all(jax.tree_util.tree_leaves(equal_leaves)), "The data should have changed after one iteration"

    # Second run
    assert jax.tree_util.tree_structure(jax_data_traced_2) == jax.tree_util.tree_structure(jax_data_traced), (
        "Trees have different shapes"
    )
    equal_leaves_2 = jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), jax_data_traced_2, jax_data_traced)
    assert not all(jax.tree_util.tree_leaves(equal_leaves_2)), "The data should not have changed after the second iteration"
