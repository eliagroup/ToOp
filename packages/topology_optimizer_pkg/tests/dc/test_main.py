# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
import os

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from jax import numpy as jnp
from toop_engine_dc_solver.jax.topology_computations import extract_sub_ids
from toop_engine_dc_solver.jax.types import ActionSet, int_max
from toop_engine_topology_optimizer.dc.genetic_functions.evolution_functions import (
    Genotype,
)
from toop_engine_topology_optimizer.dc.main import CLIArgs, main
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DescriptorDef,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Topology


def assert_topology(
    topologies: Genotype,
    action_set: ActionSet,
    disconnectable_branches: jnp.ndarray,
) -> None:
    sub_ids = extract_sub_ids(topologies.action_index, action_set)
    batch_size, max_num_splits = sub_ids.shape
    n_branch_actions = len(action_set)
    n_subs_rel = action_set.n_actions_per_sub.shape[0]

    for i in range(batch_size):
        for j in range(max_num_splits):
            if sub_ids[i, j] == int_max():
                assert topologies.action_index[i, j] == int_max()
            else:
                assert topologies.action_index[i, j] < n_branch_actions
                assert topologies.action_index[i, j] >= 0
                assert sub_ids[i, j] < n_subs_rel

        # Every substation appears at most once
        unique_val, unique_count = jnp.unique(sub_ids[i], return_counts=True)
        assert jnp.all((unique_val == int_max()) | (unique_count == 1))

        # Every outage appears at most once
        unique_val, unique_count = jnp.unique(topologies.disconnections[i], return_counts=True)
        assert jnp.all((unique_val == int_max()) | (unique_count == 1))

        # Every outage is in disconnectable_branches
        assert jnp.all(
            jnp.isin(topologies.disconnections[i], jnp.arange(len(disconnectable_branches)))
            | (topologies.disconnections[i] == int_max())
        )


@pytest.mark.timeout(120)
def test_main(tmp_path: str, static_information_file: str) -> None:
    args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            iterations_per_epoch=2,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
    )

    stats_file = os.path.join(tmp_path, "res_dir", "res.json")
    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["max_fitness"] is not None
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None


@pytest.mark.timeout(120)
def test_main_dist(tmp_path: str, static_information_file: str) -> None:
    args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            iterations_per_epoch=2,
        ),
        lf_config=LoadflowSolverParameters(
            distributed=True,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
    )

    stats_file = os.path.join(tmp_path, "res_dir", "res.json")
    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["max_fitness"] is not None
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None


@pytest.mark.timeout(120)
def test_main_double_limits(tmp_path: str, static_information_file: str) -> None:
    args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            iterations_per_epoch=2,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        double_limits=(0.9, 0.95),
    )

    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)

    # Make sure there are no errors
    invalid_args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            iterations_per_epoch=2,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        double_limits=(1.0, 0.95),
    )
    with pytest.raises(ValueError):
        main(invalid_args, processed_gridfile_fs=processed_gridfile_fs)


@pytest.mark.timeout(120)
def test_main_weight_zero(tmp_path: str, static_information_file: str) -> None:
    args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            target_metrics=(("overload_energy_n_1", 0.0),),
            iterations_per_epoch=2,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        double_limits=(0.9, 0.95),
    )
    stats_file = os.path.join(tmp_path, "res_dir", "res.json")
    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["initial_fitness"] is not None
    assert stats["initial_fitness"] == 0.0


@pytest.mark.timeout(120)
def test_main_multi_objective(tmp_path: str, static_information_file: str) -> None:
    weight_a = 1.234
    weight_b = 4.321
    args = CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            target_metrics=(
                ("overload_energy_n_1", weight_a),
                ("underload_energy_n_1", weight_b),
            ),
            iterations_per_epoch=2,
        ),
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        double_limits=(0.9, 0.95),
    )

    stats_file = os.path.join(tmp_path, "res_dir", "res.json")

    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["max_fitness"] is not None
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None
    for topo in stats["best_topos"]:
        topo = Topology.model_validate(topo)
        assert np.isclose(
            topo.metrics.extra_scores["overload_energy_n_1"] * weight_a
            + topo.metrics.extra_scores["underload_energy_n_1"] * weight_b,
            -topo.metrics.fitness,
        )


@pytest.mark.timeout(120)
def test_main_mapelites(tmp_path: str, static_information_file: str) -> None:
    batch_size = 16
    args = CLIArgs(
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        ga_config=BatchedMEParameters(
            observed_metrics=(
                "overload_energy_n_1",
                "switching_distance",
            ),
            target_metrics=(("overload_energy_n_1", 1.0),),
            iterations_per_epoch=10,
            runtime_seconds=10,
            # MapElites specifics
            me_descriptors=(DescriptorDef(metric="switching_distance", num_cells=40),),
            plot=False,
        ),
        lf_config=LoadflowSolverParameters(batch_size=batch_size),
    )

    stats_file = os.path.join(tmp_path, "res_dir", "res.json")

    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["max_fitness"] is not None
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None
    assert len(stats["best_topos"])
    assert stats["initial_metrics"] is not None
    assert all(metric in stats["initial_metrics"].keys() for metric in args.ga_config.observed_metrics)

    for topo in stats["best_topos"]:
        topo = Topology.model_validate(topo)
        assert np.isfinite(topo.metrics.fitness)
        assert all(metric in topo.metrics.extra_scores.keys() for metric in args.ga_config.observed_metrics)


def test_main_mapelites_2D(tmp_path: str, static_information_file: str) -> None:
    batch_size = 16
    args = CLIArgs(
        stats_dir=os.path.join(tmp_path, "res_dir"),
        fixed_files=(str(static_information_file), str(static_information_file)),
        ga_config=BatchedMEParameters(
            observed_metrics=(
                "overload_energy_n_1",
                "switching_distance",
                "split_subs",
            ),
            target_metrics=(("overload_energy_n_1", 1.0),),
            iterations_per_epoch=10,
            runtime_seconds=10,
            # MapElites specifics
            me_descriptors=(
                DescriptorDef(metric="split_subs", num_cells=5),
                DescriptorDef(metric="switching_distance", num_cells=40),
            ),
            plot=False,
            cell_depth=4,
        ),
        lf_config=LoadflowSolverParameters(batch_size=batch_size),
    )

    stats_file = os.path.join(tmp_path, "res_dir", "res.json")

    processed_gridfile_fs = LocalFileSystem()
    main(args, processed_gridfile_fs=processed_gridfile_fs)
    assert os.path.exists(stats_file)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert stats["max_fitness"] is not None
    assert np.isfinite(stats["max_fitness"])
    assert stats["best_topos"] is not None
    assert len(stats["best_topos"])
    assert stats["initial_metrics"] is not None
    assert all(metric in stats["initial_metrics"].keys() for metric in args.ga_config.observed_metrics)

    for topo in stats["best_topos"]:
        topo = Topology.model_validate(topo)
        assert np.isfinite(topo.metrics.fitness)
