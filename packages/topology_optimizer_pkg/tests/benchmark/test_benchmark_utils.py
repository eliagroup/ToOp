# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
import multiprocessing as mp
from pathlib import Path

import pytest
from omegaconf import DictConfig
from toop_engine_grid_helpers.powsybl.example_grids import basic_node_breaker_network_powsybl
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    CgmesImporterParameters,
    PreprocessParameters,
    UcteImporterParameters,
)
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    PipelineConfig,
    get_paths,
    prepare_importer_parameters,
    run_pipeline,
    run_preprocessing,
    run_task_process,
    set_environment_variables,
)
from toop_engine_topology_optimizer.dc.genetic_functions.initialization import update_static_information


def test_run_task_process_no_conn(dc_config):
    # Change grid file to complex grid

    #  Set the env variables
    set_environment_variables(dc_config)
    # Run the task
    res = run_task_process(dc_config)
    assert res is not None
    assert res["max_fitness"] > res["initial_fitness"], (
        "Initial fitness is greater than max fitness. Optimisation didn't work well"
    )
    # Assert the folder got created and is not empty
    res_path = Path(dc_config["output_json"]).parent
    assert len(list(res_path.iterdir())) > 0


def test_run_task_process_with_conn(dc_config):
    # Set the env variables
    set_environment_variables(dc_config)
    # Create a pipe
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    # Run the task
    res = run_task_process(dc_config, conn=child_conn)
    # Read from the parent connection
    res_from_conn = parent_conn.recv()
    assert res is None, "When using a connection, the return value should be None"
    assert res_from_conn["max_fitness"] > res_from_conn["initial_fitness"], (
        "Initial fitness is greater than max fitness. Optimisation didn't work well"
    )
    # Assert the folder got created and is not empty
    res_path = Path(dc_config["output_json"]).parent
    assert len(list(res_path.iterdir())) > 0


def test_get_paths_file_does_not_exist(pipeline_and_configs):
    import copy

    pipeline_params, _, _ = pipeline_and_configs

    pipeline_params_ = copy.deepcopy(PipelineConfig(**pipeline_params))
    pipeline_params_.file_name = "non_existent_file.xiidm"
    with pytest.raises(FileNotFoundError):
        get_paths(pipeline_params_)


def test_prepare_importer_parameters(pipeline_and_configs):
    pipeline_params, _, _ = pipeline_and_configs
    pipeline_params = PipelineConfig(**pipeline_params)
    _, file_path, data_folder, _ = get_paths(pipeline_params)  # to create the paths

    importer_params = prepare_importer_parameters(file_path, data_folder)
    assert importer_params.area_settings.cutoff_voltage == 10
    assert isinstance(importer_params, CgmesImporterParameters), (
        "Importer parameters should be of type CgmesImporterParameters"
    )

    # UCTE
    file_path = file_path.with_suffix(".uct")
    importer_params = prepare_importer_parameters(file_path, data_folder)
    assert isinstance(importer_params, UcteImporterParameters), (
        "Importer parameters should be of type UcteImporterParameters"
    )


def test_run_task_process_invalid_ga_config():
    """Test that run_task_process handles invalid GA configuration parameters correctly."""

    # Create a config with invalid ga_config parameters
    invalid_config = DictConfig(
        {
            "task_name": "test_invalid_config",
            "ga_config": {
                "runtime_seconds": "invalid_string",  # Should be int/float
                "invalid_param": "value",
            },
            "lf_config": {"distributed": False},
            "tensorboard_dir": "/tmp/test",
            "stats_dir": "/tmp/test",
            "omp_num_threads": 1,
            "num_cuda_devices": 1,
            "xla_force_host_platform_device_count": None,
        }
    )

    # Test without connection - should raise exception
    with pytest.raises((TypeError, ValueError)):
        run_task_process(invalid_config)


def test_run_task_process_invalid_lf_config():
    """Test that run_task_process handles invalid loadflow configuration parameters correctly."""

    # Create a config with invalid lf_config parameters
    invalid_config = DictConfig(
        {
            "task_name": "test_invalid_lf_config",
            "ga_config": {
                "runtime_seconds": 10,
                "me_descriptors": [{"metric": "split_subs", "num_cells": 5}],
                "observed_metrics": ["overload_energy_n_1"],
            },
            "lf_config": {
                "distributed": "invalid_boolean",  # Should be bool
                "invalid_lf_param": 123,
            },
            "tensorboard_dir": "/tmp/test",
            "stats_dir": "/tmp/test",
            "omp_num_threads": 1,
            "num_cuda_devices": 1,
            "xla_force_host_platform_device_count": None,
        }
    )

    # Test without connection - should raise exception
    with pytest.raises((TypeError, ValueError)):
        run_task_process(invalid_config)


def test_run_task_process_invalid_config_with_connection():
    """Test that run_task_process handles invalid configuration with connection correctly."""
    import multiprocessing as mp

    # Create a config with invalid parameters
    invalid_config = DictConfig(
        {
            "task_name": "test_invalid_with_conn",
            "ga_config": {
                "runtime_seconds": "not_a_number",  # Invalid type
            },
            "lf_config": {"distributed": False},
            "tensorboard_dir": "/tmp/test",
            "stats_dir": "/tmp/test",
            "omp_num_threads": 1,
            "num_cuda_devices": 1,
            "xla_force_host_platform_device_count": None,
        }
    )

    # Create a pipe for testing with connection
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    # Run the task with connection - should return None and send error through connection
    result = run_task_process(invalid_config, conn=child_conn)

    # Should return None when using connection
    assert result is None

    # Should receive error message through connection
    error_result = parent_conn.recv()
    assert "error" in error_result
    assert "Invalid configuration parameters" in error_result["error"]


def test_run_pipeline(pipeline_and_configs, preprocessing_parameters):
    def _get_serialized_topology_fitness(topology: dict) -> float:
        return topology["metrics"]["fitness"]

    pipeline_cfg, dc_cfg, ac_cfg = pipeline_and_configs
    ac_cfg.k_best_topos = 10  # To cover warning branch
    pipeline_cfg = PipelineConfig(**pipeline_cfg)
    preprocessing_parameters = PreprocessParameters(**preprocessing_parameters)
    _, file_path, data_folder, _ = get_paths(pipeline_cfg)

    importer_parameters = prepare_importer_parameters(file_path, data_folder)

    topo_paths = run_pipeline(
        pipeline_cfg=pipeline_cfg,
        importer_parameters=importer_parameters,
        preprocessing_parameters=preprocessing_parameters,
        dc_optim_config=dc_cfg,
        ac_validation_cfg=ac_cfg,
        run_preprocessing_stage=True,
        run_optimization_stage=True,
        run_ac_validation_stage=True,
    )
    # We test quality of the output in the benchmark
    assert len(topo_paths) > 0, "No topologies were generated by the pipeline."
    unsplit_metrics_path = topo_paths[0].parent / "unsplit_ac_metrics.json"
    assert unsplit_metrics_path.exists(), "Missing unsplit AC metrics export in the optimisation run directory."
    res = json.loads((topo_paths[0].parent / "res.json").read_text(encoding="utf-8"))
    expected_topologies = sorted(res["best_topos"], key=_get_serialized_topology_fitness, reverse=True)[: len(topo_paths)]

    for topo_path, expected_topology in zip(topo_paths, expected_topologies, strict=True):
        orao_summary_path = topo_path / "orao_summary.json"
        ac_metrics_path = topo_path / "ac_metrics.json"
        assert orao_summary_path.exists(), f"Missing ORAO summary export for topology output {topo_path}"
        assert ac_metrics_path.exists(), f"Missing AC metrics export for topology output {topo_path}"

        orao_summary = json.loads(orao_summary_path.read_text())
        ac_metrics = json.loads(ac_metrics_path.read_text(encoding="utf-8"))
        assert "forced-actions" in orao_summary
        assert "preventive-actions-list" in orao_summary["forced-actions"]
        assert _get_serialized_topology_fitness(ac_metrics["dc_info"]) == _get_serialized_topology_fitness(expected_topology)
        assert ac_metrics["dc_info"]["actions"] == (expected_topology.get("actions") or [])
        assert ac_metrics["dc_info"]["disconnections"] == (expected_topology.get("disconnections") or [])


def test_run_task_process_with_imported_busbar_outages(tmp_path: Path) -> None:
    input_grid = tmp_path / "grid.xiidm"
    net = basic_node_breaker_network_powsybl()
    # create a busbar outage related overload for L2 when the second busbar in VL2 is outaged
    open_switches = ["load1_DISCONNECTOR_18_0", "L71_DISCONNECTOR_10_1"]
    close_switches = ["load1_DISCONNECTOR_18_1", "L71_DISCONNECTOR_10_0"]
    for switch in close_switches:
        net.close_switch(switch)
    for switch in open_switches:
        net.open_switch(switch)

    net.save(input_grid)

    data_folder = tmp_path / input_grid.stem
    importer_parameters = prepare_importer_parameters(input_grid, data_folder)
    preprocessing_parameters = PreprocessParameters(action_set_clip=2**10, preprocess_bb_outages=True)

    _, static_information = run_preprocessing(
        importer_parameters=importer_parameters,
        data_folder=data_folder,
        preprocessing_parameters=preprocessing_parameters,
        is_pandapower_net=False,
    )

    assert (
        static_information.dynamic_information.action_set.rel_bb_outage_data is not None
        or static_information.dynamic_information.non_rel_bb_outage_data is not None
        or static_information.dynamic_information.bb_outage_baseline_analysis is not None
    )
    n_worst_contingencies = min(20, int(static_information.dynamic_information.n_nminus1_cases))
    assert n_worst_contingencies > 0

    dc_config = DictConfig(
        {
            "task_name": "test_busbar_outage_optimizer",
            "fixed_files": [str(data_folder / "static_information.hdf5")],
            "double_precision": None,
            "tensorboard_dir": str(tmp_path / "results" / "{task_name}"),
            "stats_dir": str(tmp_path / "results" / "{task_name}"),
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "stdout": None,
            "double_limits": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "output_json": str(tmp_path / "results" / "output.json"),
            "lf_config": {"distributed": False},
            "ga_config": {
                "runtime_seconds": 5,
                "enable_bb_outage": True,
                "bb_outage_as_nminus1": True,
                "n_worst_contingencies": n_worst_contingencies,
                "target_metrics": [["overload_energy_n_1", 1.0]],
                "me_descriptors": [{"metric": "split_subs", "num_cells": 5}],
                "observed_metrics": ["overload_energy_n_1", "split_subs"],
            },
        }
    )

    updated_static_information = update_static_information(
        static_informations=(static_information,),
        batch_size=8,
        enable_nodal_inj_optim=False,
        enable_parallel_pst_group_optim=False,
        enable_bb_outage=True,
        bb_outage_as_nminus1=True,
        clip_bb_outage_penalty=False,
        bb_outage_more_islands_penalty=0.0,
    )[0]
    assert updated_static_information.solver_config.enable_bb_outages
    # assert updated_static_information.solver_config.bb_outage_as_nminus1

    set_environment_variables(dc_config)
    result = run_task_process(dc_config)

    assert result is not None
    assert "overload_energy_n_1" in result["initial_metrics"]
    assert result["initial_metrics"]["overload_energy_n_1"] > 0
    assert result["max_fitness"] > result["initial_fitness"]
    assert len(result["best_topos"]) > 0
    best_topology = max(result["best_topos"], key=lambda topo: topo["metrics"]["fitness"])
    assert best_topology["metrics"]["fitness"] > result["initial_fitness"]
    assert (
        best_topology["metrics"]["extra_scores"]["overload_energy_n_1"] <= result["initial_metrics"]["overload_energy_n_1"]
    )
    result_dir = Path(dc_config["output_json"]).parent
    assert result_dir.exists()
    assert len(list(result_dir.iterdir())) > 0


def test_run_pipeline_no_optimization_stage(pipeline_and_configs, preprocessing_parameters):
    pipeline_cfg, dc_cfg, ac_cfg = pipeline_and_configs
    pipeline_cfg = PipelineConfig(**pipeline_cfg)
    preprocessing_parameters = PreprocessParameters(**preprocessing_parameters)
    _, file_path, data_folder, _ = get_paths(pipeline_cfg)

    importer_parameters = prepare_importer_parameters(file_path, data_folder)

    # Should raise an error, because so optimisation dir was provided
    with pytest.raises(ValueError):
        run_pipeline(
            pipeline_cfg=pipeline_cfg,
            importer_parameters=importer_parameters,
            preprocessing_parameters=preprocessing_parameters,
            dc_optim_config=dc_cfg,
            ac_validation_cfg=ac_cfg,
            run_preprocessing_stage=True,
            run_optimization_stage=False,
            run_ac_validation_stage=True,
            optimisation_run_dir=None,
        )


def test_run_task_process_invalid_file_path():
    """Test that run_task_process handles invalid file paths correctly (should trigger OSError)."""

    # Create a config with invalid file path that should trigger OSError
    invalid_config = DictConfig(
        {
            "task_name": "test_invalid_file_path",
            "fixed_files": ["/nonexistent/path/static_info.hdf5"],  # Non-existent file path
            "double_precision": None,
            "tensorboard_dir": "/tmp/test",
            "stats_dir": "/tmp/test",
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "double_limits": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "output_json": "/tmp/test/output.json",
            "ga_config": {
                "runtime_seconds": 10,
                "me_descriptors": [{"metric": "split_subs", "num_cells": 5}],
                "observed_metrics": ["overload_energy_n_1"],
            },
            "lf_config": {"distributed": False},
        }
    )

    # Test without connection - should raise OSError or RuntimeError
    with pytest.raises((OSError, RuntimeError, ValueError)):
        run_task_process(invalid_config)


def test_run_task_process_invalid_file_path_with_connection():
    """Test that run_task_process handles invalid file paths with connection correctly."""
    import multiprocessing as mp

    # Create a config with invalid file path
    invalid_config = DictConfig(
        {
            "task_name": "test_invalid_file_path_conn",
            "fixed_files": ["/nonexistent/path/static_info.hdf5"],  # Non-existent file path
            "double_precision": None,
            "tensorboard_dir": "/tmp/test",
            "stats_dir": "/tmp/test",
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "double_limits": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "output_json": "/tmp/test/output.json",
            "ga_config": {
                "runtime_seconds": 10,
                "me_descriptors": [{"metric": "split_subs", "num_cells": 5}],
                "observed_metrics": ["overload_energy_n_1"],
            },
            "lf_config": {"distributed": False},
        }
    )

    # Create a pipe for testing with connection
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    # Run the task with connection - should return None and send error through connection
    result = run_task_process(invalid_config, conn=child_conn)

    # Should return None when using connection
    assert result is None

    # Should receive error message through connection
    error_result = parent_conn.recv()
    assert "error" in error_result
    assert "Optimization failed" in error_result["error"]
