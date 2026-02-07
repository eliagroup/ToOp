# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from qdax.utils.metrics import default_ga_metrics
from toop_engine_dc_solver.example_grids import three_node_pst_example_folder_powsybl
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    crossover,
    empty_repertoire,
    mutate,
)
from toop_engine_topology_optimizer.dc.genetic_functions.scoring_functions import (
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
    empty_genotypes = empty_repertoire(batch_size, max_num_splits, 0, 0, n_timesteps)

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
def create_3_node_pst_example_grid(tmp_path_factory) -> tuple[StaticInformationStats, StaticInformation, NetworkData]:
    tmp_path = tmp_path_factory.mktemp("three_node_pst_example_grid")

    three_node_pst_example_folder_powsybl(tmp_path)
    filesystem_dir = DirFileSystem(str(tmp_path))
    stats, static_information, network_data = load_grid(filesystem_dir, pandapower=False)
    return stats, static_information, network_data


def test_pst_fixture(create_3_node_pst_example_grid: tuple[StaticInformationStats, StaticInformation, NetworkData]) -> None:
    stats, static_information, network_data = create_3_node_pst_example_grid

    inj_info = static_information.dynamic_information.nodal_injection_information
    assert jnp.array_equal(
        inj_info.pst_tap_values[jnp.arange(len(inj_info.starting_tap)), inj_info.starting_tap], jnp.array([0.0, 0.0])
    )


def test_pst_optimization(
    create_3_node_pst_example_grid: tuple[StaticInformationStats, StaticInformation, NetworkData],
) -> None:
    stats, static_information, network_data = create_3_node_pst_example_grid

    assert static_information
