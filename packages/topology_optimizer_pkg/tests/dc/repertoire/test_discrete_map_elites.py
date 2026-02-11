# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from pypowsybl.network import Network
from qdax.utils.metrics import default_ga_metrics
from toop_engine_dc_solver.example_grids import three_node_pst_example_folder_powsybl
from toop_engine_dc_solver.jax.aggregate_results import get_overload_energy_n_1_matrix
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.inputs import load_static_information, validate_static_information
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import NodalInjOptimResults, NodalInjStartOptions, StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    crossover,
    empty_repertoire,
    mutate,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
    convert_to_topologies,
    scoring_function,
)
from toop_engine_topology_optimizer.dc.repertoire.discrete_map_elites import DiscreteMapElites


@pytest.mark.parametrize("cell_depth", [1, 2])
def test_discrete_mapelites(static_information_file: str, cell_depth: int) -> None:
    static_information = load_static_information(static_information_file)

    action_set = static_information.dynamic_information.action_set
    disconnectable_branches = static_information.dynamic_information.disconnectable_branches

    max_num_splits = 3
    batch_size = 4

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=batch_size),
    )

    me = DiscreteMapElites(
        scoring_function=partial(
            scoring_function,
            solver_configs=[static_information.solver_config],
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
        ),
        emitter=TrackingMixingEmitter(
            lambda topologies, key: mutate(
                topologies,
                key,
                substation_split_prob=0.2,
                substation_unsplit_prob=0.0001,
                action_set=action_set,
                n_disconnectable_branches=len(disconnectable_branches),
                n_subs_mutated_lambda=5.0,
                disconnect_prob=0.5,
                reconnect_prob=0.5,
                pst_n_taps=jnp.array([], dtype=int),
                mutation_repetition=1,
            ),
            lambda topo_a, topo_b, key: crossover(topo_a, topo_b, key, action_set=action_set, prob_take_a=0.1),
            0.5,
            batch_size,
        ),
        metrics_function=default_ga_metrics,
        n_cells_per_dim=(20,),
        cell_depth=cell_depth,
    )

    n_timesteps = static_information.dynamic_information.n_timesteps
    empty_genotypes = empty_repertoire(batch_size, max_num_splits, 0, n_timesteps)

    repertoire, emitter_state, rng_key = me.init(
        genotypes=empty_genotypes,
        random_key=jax.random.PRNGKey(0),
        scoring_data=[static_information.dynamic_information],
    )

    assert repertoire.fitnesses.shape == (20 * cell_depth,)

    repertoire, emitter_state, _, rng_key = me.update(
        repertoire=repertoire,
        emitter_state=emitter_state,
        random_key=rng_key,
        scoring_data=[static_information.dynamic_information],
    )

    assert repertoire.fitnesses.shape == (20 * cell_depth,)


@pytest.fixture
def create_3_node_pst_example_grid(
    tmp_path_factory,
) -> tuple[StaticInformationStats, StaticInformation, NetworkData, Network]:
    tmp_path = tmp_path_factory.mktemp("three_node_pst_example_grid")

    three_node_pst_example_folder_powsybl(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    stats, static_information, network_data = load_grid(filesystem_dir, pandapower=False)
    net = pypowsybl.network.load(tmp_path / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    return stats, static_information, network_data, net


def test_manual_pst_optimization(
    create_3_node_pst_example_grid: tuple[StaticInformationStats, StaticInformation, NetworkData, Network],
) -> None:
    stats, static_information, network_data, net = create_3_node_pst_example_grid
    validate_static_information(static_information)
    di = static_information.dynamic_information
    solver_config = replace(static_information.solver_config, batch_size_bsdf=1)

    inj_info = di.nodal_injection_information
    assert jnp.array_equal(
        inj_info.pst_tap_values[jnp.arange(len(inj_info.starting_tap_idx)), inj_info.starting_tap_idx], jnp.array([0.0, 0.0])
    )

    # Default taps should lead to overload, optimization should fix it
    # First run the solver without any taps
    res, success = compute_symmetric_batch(
        topology_batch=default_topology(solver_config=solver_config),
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=None,
        dynamic_information=di,
        solver_config=solver_config,
    )
    assert jnp.all(success)

    overload = get_overload_energy_n_1_matrix(n_1_matrix=res.n_1_matrix, max_mw_flow=di.branch_limits.max_mw_flow)
    assert overload > 0

    # Check if NodalInj with starting taps gives the same result as skipping nodal inj optimization altogether.
    res2, success = compute_symmetric_batch(
        topology_batch=default_topology(solver_config=solver_config),
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=NodalInjStartOptions(
            previous_results=NodalInjOptimResults(
                pst_tap_idx=di.nodal_injection_information.starting_tap_idx[None, None, :]
            ),
            precision_percent=jnp.array(0.0),
        ),
        dynamic_information=di,
        solver_config=solver_config,
    )
    assert jnp.all(success)
    assert jnp.allclose(res.n_0_matrix, res2.n_0_matrix)
    assert jnp.allclose(res.n_1_matrix, res2.n_1_matrix)

    # PST tap of -12 (tap index 18) should match PowerSybl reference
    # With corrected sign, this tap eliminates overload
    solution = np.array([18, 18])
    res, success = compute_symmetric_batch(
        topology_batch=default_topology(solver_config=solver_config),
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=NodalInjStartOptions(
            previous_results=NodalInjOptimResults(pst_tap_idx=jnp.array([solution.tolist()])),
            precision_percent=jnp.array([0.0]),
        ),
        dynamic_information=di,
        solver_config=solver_config,
    )
    assert jnp.all(success)

    overload = get_overload_energy_n_1_matrix(n_1_matrix=res.n_1_matrix, max_mw_flow=di.branch_limits.max_mw_flow)

    assert overload == 0

    # Cross check with the pypowsybl load flow
    # First verify the unsplit flow matches
    pypowsybl.loadflow.run_dc(net)
    unsplit_n_0 = net.get_branches().loc[network_data.branch_ids]["p1"].values
    assert jnp.allclose(unsplit_n_0, -di.unsplit_flow[0], atol=1e-2)

    # Now update PST taps in pypowsybl and verify that the flow matches the optimized flow
    pst_indices = ["pst_LINE_BC_1", "pst_LINE_BC_2"]
    low_tap = net.get_phase_tap_changers().loc[pst_indices]["low_tap"]

    # Note that powsybl taps do not start at zero but at low_tap
    net.update_phase_tap_changers(id=pst_indices, tap=(low_tap + solution).tolist())
    pypowsybl.loadflow.run_dc(net)
    n_0_ref = net.get_branches().loc[network_data.branch_ids]["p1"].values
    assert jnp.allclose(n_0_ref, -res.n_0_matrix[0, 0], atol=1e-2)


def test_pst_optimization(
    create_3_node_pst_example_grid: tuple[StaticInformationStats, StaticInformation, NetworkData, Network],
) -> None:
    stats, static_information, network_data, net = create_3_node_pst_example_grid
    di = static_information.dynamic_information
    solver_config = replace(static_information.solver_config, batch_size_bsdf=1, enable_nodal_inj_optim=True)

    me = DiscreteMapElites(
        scoring_function=partial(
            scoring_function,
            solver_configs=[solver_config],
            target_metrics=(("overload_energy_n_1", 1.0),),
            observed_metrics=("overload_energy_n_1", "split_subs"),
            descriptor_metrics=("split_subs",),
            n_worst_contingencies=1,
        ),
        emitter=TrackingMixingEmitter(
            lambda topologies, key: mutate(
                topologies,
                key,
                substation_split_prob=0.0,
                substation_unsplit_prob=0.0,
                action_set=di.action_set,
                n_disconnectable_branches=0,
                n_subs_mutated_lambda=0.0,
                disconnect_prob=0.0,
                reconnect_prob=0.0,
                pst_n_taps=di.nodal_injection_information.pst_n_taps,
                mutation_repetition=1,
            ),
            lambda topo_a, topo_b, key: crossover(topo_a, topo_b, key, action_set=di.action_set, prob_take_a=0.5),
            0.5,
            batch_size=1,
        ),
        metrics_function=default_ga_metrics,
        n_cells_per_dim=(20,),
        cell_depth=1,
    )
    rng_key = jax.random.PRNGKey(0)
    repertoire, emitter_state, rng_key = me.init(
        genotypes=empty_repertoire(
            batch_size=1,
            max_num_splits=1,
            max_num_disconnections=0,
            n_timesteps=1,
            starting_taps=di.nodal_injection_information.starting_tap_idx,
        ),
        random_key=rng_key,
        scoring_data=[di],
    )

    assert repertoire.genotypes.nodal_injections_optimized is not None

    for _ in range(100):
        repertoire, emitter_state, _, rng_key = me.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            random_key=rng_key,
            scoring_data=[di],
        )

    assert repertoire.genotypes.nodal_injections_optimized is not None
    best_fitness = jnp.argmax(repertoire.fitnesses)
    best_taps = repertoire.genotypes.nodal_injections_optimized[best_fitness]
    assert not jnp.array_equal(best_taps.pst_tap_idx[0], di.nodal_injection_information.starting_tap_idx)
    assert jnp.isclose(repertoire.fitnesses[best_fitness], 0)
    # With corrected sign, optimal tap should be lower than starting tap
    assert jnp.all(best_taps.pst_tap_idx < di.nodal_injection_information.starting_tap_idx)

    # Check if convert_to_topologies would send out the PST taps
    conv_topos = convert_to_topologies(
        repertoire,
        contingency_ids=network_data.contingency_ids,
        grid_model_low_tap=di.nodal_injection_information.grid_model_low_tap,
    )
    assert len(conv_topos)
    assert conv_topos[0].pst_setpoints is not None
    assert len(conv_topos[0].pst_setpoints) == di.n_controllable_pst
    assert conv_topos[0].pst_setpoints == list(
        repertoire.genotypes.nodal_injections_optimized[0].pst_tap_idx[0] + di.nodal_injection_information.grid_model_low_tap
    )
