# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests for parent-selection benchmark bookkeeping."""

import json
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from toop_engine_topology_optimizer.benchmark.parent_selection_study import (
    ACValidationConfiguration,
    DescriptorResolutionConfiguration,
    EvaluationCounters,
    GridSpecification,
    PreparedGrid,
    RunOutcome,
    RunSpecification,
    _run_manifest,
    _run_optional_ac_validation,
    compute_archive_statistics,
    derive_pst_activated_num_cells,
    derive_pst_switching_distance_num_cells,
    expand_run_specifications,
    export_archive_cells,
    extract_feedback_statistics,
    make_epoch_record,
    make_repertoire_snapshot_record,
    resolve_ga_parameters_for_grid,
    run_parent_selection_study,
    validate_study_inputs,
)
from toop_engine_topology_optimizer.dc.ga_helpers import MixingEmitterState
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import empty_repertoire
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire
from toop_engine_topology_optimizer.dc.repertoire.parent_selection import ParentSelectorState
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)


def test_compute_archive_statistics_tracks_depth_coverage_and_qd_scores() -> None:
    archive = compute_archive_statistics(
        fitnesses=np.array([-np.inf, 2.0, 3.0, -np.inf, 4.0, 5.0]),
        cell_depth=3,
        initial_fitness=2.0,
    )

    assert archive.n_logical_cells == 2
    assert archive.n_flat_slots == 6
    assert archive.occupied_cells == 2
    assert archive.occupied_candidates == 4
    assert archive.cell_coverage == 1.0
    assert archive.candidate_coverage == pytest.approx(4 / 6)
    assert archive.occupied_candidates_per_cell_mean == 2.0
    assert archive.occupied_candidates_per_cell_min == 2
    assert archive.occupied_candidates_per_cell_max == 2
    assert archive.qd_score_raw == 9.0
    assert archive.qd_score_raw_per_cell == 4.5
    assert archive.qd_score_improvement == 5.0
    assert archive.qd_score_improvement_per_cell == 2.5


def test_compute_archive_statistics_handles_empty_repertoire() -> None:
    archive = compute_archive_statistics(
        fitnesses=np.full((4,), -np.inf),
        cell_depth=2,
        initial_fitness=-5.0,
    )

    assert archive.occupied_cells == 0
    assert archive.occupied_candidates == 0
    assert archive.cell_coverage == 0.0
    assert archive.candidate_coverage == 0.0
    assert archive.occupied_candidates_per_cell_mean is None
    assert archive.qd_score_raw == 0.0
    assert archive.qd_score_improvement == 0.0


def test_derive_pst_switching_distance_num_cells_uses_start_taps_and_timesteps() -> None:
    assert (
        derive_pst_switching_distance_num_cells(
            pst_n_taps=np.array([5, 3]),
            starting_tap_idx=np.array([2, 0]),
            n_timesteps=2,
        )
        == 9
    )


def test_derive_pst_switching_distance_num_cells_rejects_immobile_psts() -> None:
    with pytest.raises(ValueError, match="can change tap position"):
        derive_pst_switching_distance_num_cells(
            pst_n_taps=np.array([1, 1]),
            starting_tap_idx=np.array([0, 0]),
            n_timesteps=1,
        )


def test_derive_pst_activated_num_cells_uses_movable_psts_and_timesteps() -> None:
    assert (
        derive_pst_activated_num_cells(
            pst_n_taps=np.array([5, 3, 1]),
            starting_tap_idx=np.array([2, 0, 0]),
            n_timesteps=2,
        )
        == 5
    )


def test_run_parent_selection_study_keeps_existing_optional_argument_order() -> None:
    parameter_names = tuple(signature(run_parent_selection_study).parameters)

    assert parameter_names[-4:] == (
        "ac_validation",
        "show_progress",
        "stop_on_error",
        "descriptor_resolution",
    )


def test_run_manifest_records_descriptor_resolution(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.xiidm"
    grid_path.write_text("grid")
    grid = GridSpecification(identifier="pst-grid", source_path=grid_path)
    prepared_grid = PreparedGrid(
        specification=grid,
        data_directory=tmp_path,
        grid_path=grid_path,
        static_information_path=tmp_path / "static_information.hdf5",
        source_sha256="test",
    )
    parameters = DCOptimizerParameters(
        ga_config=BatchedMEParameters(me_descriptors=(DescriptorDef(metric="split_subs", num_cells=5),)),
        loadflow_solver_config=LoadflowSolverParameters(),
    )

    manifest = _run_manifest(
        run=RunSpecification(grid=grid, parent_selection_mode="uniform", seed=1),
        prepared_grid=prepared_grid,
        parameters=parameters,
        status="running",
        descriptor_resolution=DescriptorResolutionConfiguration(
            auto_metrics=("pst_switching_distance",),
            max_logical_cells=100_000,
        ),
    )

    assert manifest["repertoire_layout"]["descriptor_resolution"] == {
        "auto_metrics": ["pst_switching_distance"],
        "max_logical_cells": 100_000,
    }


def test_resolve_ga_parameters_for_grid_adds_derived_pst_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    grid_path = tmp_path / "grid.xiidm"
    grid_path.write_text("grid")
    prepared_grid = PreparedGrid(
        specification=GridSpecification(identifier="pst-grid", source_path=grid_path),
        data_directory=tmp_path,
        grid_path=grid_path,
        static_information_path=tmp_path / "static_information.hdf5",
        source_sha256="test",
    )
    static_information = SimpleNamespace(
        dynamic_information=SimpleNamespace(
            nodal_injection_information=SimpleNamespace(
                pst_n_taps=jnp.array([5, 3]),
                starting_tap_idx=jnp.array([2, 0]),
            ),
            n_timesteps=2,
        )
    )
    monkeypatch.setattr(
        "toop_engine_topology_optimizer.benchmark.parent_selection_study.load_static_information",
        lambda _path: static_information,
    )
    parameters = BatchedMEParameters(
        enable_nodal_inj_optim=True,
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=5),
            DescriptorDef(metric="switching_distance", num_cells=20),
        ),
    )

    resolved = resolve_ga_parameters_for_grid(
        ga_parameters=parameters,
        prepared_grid=prepared_grid,
        descriptor_resolution=DescriptorResolutionConfiguration(
            auto_metrics=("pst_switching_distance", "pst_activated"),
            max_logical_cells=10_000,
        ),
    )

    assert [(descriptor.metric, descriptor.num_cells) for descriptor in resolved.me_descriptors] == [
        ("split_subs", 5),
        ("switching_distance", 20),
        ("pst_switching_distance", 9),
        ("pst_activated", 5),
    ]
    assert "pst_switching_distance" in resolved.observed_metrics
    assert "pst_activated" in resolved.observed_metrics


def test_resolve_ga_parameters_for_grid_rejects_excessive_repertoire_size(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.xiidm"
    grid_path.write_text("grid")
    prepared_grid = PreparedGrid(
        specification=GridSpecification(identifier="large-grid", source_path=grid_path),
        data_directory=tmp_path,
        grid_path=grid_path,
        static_information_path=tmp_path / "static_information.hdf5",
        source_sha256="test",
    )

    with pytest.raises(ValueError, match="max_logical_cells=99"):
        resolve_ga_parameters_for_grid(
            ga_parameters=BatchedMEParameters(me_descriptors=(DescriptorDef(metric="split_subs", num_cells=100),)),
            prepared_grid=prepared_grid,
            descriptor_resolution=DescriptorResolutionConfiguration(max_logical_cells=99),
        )


def test_export_archive_cells_keeps_the_best_candidate_per_logical_cell() -> None:
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, 3.0, 2.0, -jnp.inf]),
        descriptors=jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]]),
        extra_scores={"overload_energy_n_1": jnp.array([5.0, 2.0, 3.0, 0.0])},
        n_cells_per_dim=(2,),
        cell_depth=2,
    )

    assert export_archive_cells(repertoire) == [
        {
            "cell_index": 0,
            "fitness": 2.0,
            "descriptors": [0, 0],
            "metrics": {"overload_energy_n_1": 3.0},
        },
        {
            "cell_index": 1,
            "fitness": 3.0,
            "descriptors": [1, 1],
            "metrics": {"overload_energy_n_1": 2.0},
        },
    ]


def test_make_repertoire_snapshot_record_keeps_elites_and_cumulative_selection_counts() -> None:
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, -jnp.inf, 4.0, -jnp.inf]),
        descriptors=jnp.zeros((4, 1), dtype=int),
        extra_scores={},
        n_cells_per_dim=(2,),
        cell_depth=2,
    )
    emitter_state = MixingEmitterState(
        total_branch_combis=jnp.array(0, dtype=int),
        total_inj_combis=jnp.array(0, dtype=int),
        total_num_splits=jnp.array(0, dtype=int),
        parent_selector_state=ParentSelectorState(),
        parent_selection_counts=jnp.array([0, 3], dtype=int),
    )

    snapshot = make_repertoire_snapshot_record(
        repertoire=repertoire,
        emitter_state=emitter_state,
        epoch=2,
        jax_iteration=16,
    )

    assert snapshot == {
        "schema_version": 1,
        "epoch": 2,
        "jax_iteration": 16,
        "cell_indices": [0, 1],
        "elite_fitnesses": [4.0, None],
        "selection_counts": [0, 3],
    }
    json.dumps(snapshot, allow_nan=False)


def test_make_epoch_record_captures_archive_quality_and_execution_metrics() -> None:
    repertoire = DiscreteMapElitesRepertoire(
        genotypes=empty_repertoire(batch_size=4, max_num_splits=1, max_num_disconnections=0, n_timesteps=1),
        fitnesses=jnp.array([1.0, -jnp.inf, 4.0, 3.0]),
        descriptors=jnp.zeros((4, 1), dtype=int),
        extra_scores={"overload_energy_n_1": jnp.array([5.0, 0.0, 2.0, 3.0])},
        n_cells_per_dim=(2,),
        cell_depth=2,
    )
    emitter_state = MixingEmitterState(
        total_branch_combis=jnp.array(10, dtype=int),
        total_inj_combis=jnp.array(20, dtype=int),
        total_num_splits=jnp.array(3, dtype=int),
        parent_selector_state=ParentSelectorState(),
    )

    record, counters = make_epoch_record(
        repertoire=repertoire,
        emitter_state=emitter_state,
        observed_metrics=("overload_energy_n_1",),
        initial_fitness=1.0,
        epoch=2,
        jax_iteration=100,
        elapsed_seconds=2.0,
        epoch_seconds=1.0,
        previous_counters=EvaluationCounters(branch_combinations=4, injection_combinations=10, split_grids=1),
    )

    assert counters == EvaluationCounters(branch_combinations=10, injection_combinations=20, split_grids=3)
    assert record["quality"]["best_fitness"] == 4.0
    assert record["quality"]["fitness_improvement"] == 3.0
    assert record["quality"]["best_candidate_metrics"] == {"overload_energy_n_1": 2.0}
    assert record["archive"]["qd_score_raw"] == 7.0
    assert record["archive"]["qd_score_improvement"] == 5.0
    assert record["archive"]["cell_coverage"] == 1.0
    assert record["execution"]["epoch"]["branch_combinations"] == 6
    assert record["execution"]["branch_combinations_per_second"] == 5.0
    assert record["feedback"] == {
        "n_logical_cells": 0,
        "total_parent_selections": 0,
        "total_attributed_survivals": 0,
        "selected_cells": 0,
        "selected_cell_fraction": 0.0,
        "unvisited_cells": 0,
        "attributed_survival_rate": None,
        "success_rate_mean": None,
        "success_rate_min": None,
        "success_rate_max": None,
        "selection_concentration": 0.0,
        "effective_selected_cells": None,
    }


def test_extract_feedback_statistics_reports_universal_telemetry() -> None:
    emitter_state = MixingEmitterState(
        total_branch_combis=jnp.array(0, dtype=int),
        total_inj_combis=jnp.array(0, dtype=int),
        total_num_splits=jnp.array(0, dtype=int),
        parent_selector_state=ParentSelectorState(),
        parent_selection_counts=jnp.array([2, 3], dtype=int),
        parent_success_counts=jnp.array([2, 1], dtype=int),
    )

    statistics = extract_feedback_statistics(emitter_state)

    assert statistics == {
        "n_logical_cells": 2,
        "total_parent_selections": 5,
        "total_attributed_survivals": 3,
        "selected_cells": 2,
        "selected_cell_fraction": 1.0,
        "unvisited_cells": 0,
        "attributed_survival_rate": 0.6,
        "success_rate_mean": pytest.approx(2 / 3),
        "success_rate_min": pytest.approx(1 / 3),
        "success_rate_max": 1.0,
        "selection_concentration": pytest.approx(13 / 25),
        "effective_selected_cells": pytest.approx(25 / 13),
    }


def test_optional_ac_failure_preserves_dc_run_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.xiidm"
    grid_path.write_text("grid")
    grid = GridSpecification(identifier="small", source_path=grid_path)
    run = RunSpecification(grid=grid, parent_selection_mode="uniform", seed=1)
    output_directory = tmp_path / "run"
    output_directory.mkdir()
    (output_directory / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
    outcome = RunOutcome(
        specification=run,
        output_directory=output_directory,
        final_result={},
        phase_seconds={},
    )
    prepared_grid = PreparedGrid(
        specification=grid,
        data_directory=tmp_path,
        grid_path=grid_path,
        static_information_path=tmp_path / "static_information.hdf5",
        source_sha256="test",
    )

    def raise_ac_error(**_kwargs: object) -> list[Path]:
        raise RuntimeError("AC unavailable")

    monkeypatch.setattr(
        "toop_engine_topology_optimizer.benchmark.parent_selection_study.perform_ac_analysis",
        raise_ac_error,
    )

    succeeded = _run_optional_ac_validation(outcome, prepared_grid, ACValidationConfiguration(enabled=True))

    summary = json.loads((output_directory / "ac_validation_summary.json").read_text())
    manifest = json.loads((output_directory / "run_manifest.json").read_text())
    assert succeeded is False
    assert summary["status"] == "failed"
    assert summary["error"] == "AC unavailable"
    assert manifest["status"] == "completed"
    assert manifest["ac_validation"]["status"] == "failed"


def test_expand_run_specifications_uses_every_mode_and_shared_seed(tmp_path: Path) -> None:
    grid_file = tmp_path / "grid.xiidm"
    grid_file.write_text("grid")
    grid = GridSpecification(identifier="small", source_path=grid_file)

    runs = expand_run_specifications(
        grids=(grid,),
        modes=("uniform", "ucb", "ucb_batched"),
        seeds=(3, 7),
    )

    assert [(run.parent_selection_mode, run.seed) for run in runs] == [
        ("uniform", 3),
        ("uniform", 7),
        ("ucb", 3),
        ("ucb", 7),
        ("ucb_batched", 3),
        ("ucb_batched", 7),
    ]
    assert [run.identifier for run in runs] == [
        "small/UNIi/seed_3",
        "small/UNIi/seed_7",
        "small/UCBc/seed_3",
        "small/UCBc/seed_7",
        "small/UCBb/seed_3",
        "small/UCBb/seed_7",
    ]


def test_validate_study_inputs_rejects_distributed_execution(tmp_path: Path) -> None:
    grid_file = tmp_path / "grid.xiidm"
    grid_file.write_text("grid")

    with pytest.raises(ValueError, match="distributed=False"):
        validate_study_inputs(
            grids=(GridSpecification(identifier="small", source_path=grid_file),),
            modes=("uniform",),
            seeds=(1,),
            loadflow_parameters=LoadflowSolverParameters(distributed=True),
        )


def test_jsonl_records_are_strict_json(tmp_path: Path) -> None:
    output_file = tmp_path / "record.json"
    output_file.write_text(json.dumps({"value": 1.0}, allow_nan=False))

    assert json.loads(output_file.read_text()) == {"value": 1.0}
