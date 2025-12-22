# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import multiprocessing as mp
from pathlib import Path

import pytest
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
    run_task_process,
    set_environment_variables,
)


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

    pipeline_params_ = copy.deepcopy(pipeline_params)
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
    from omegaconf import DictConfig

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
    from omegaconf import DictConfig

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

    from omegaconf import DictConfig

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
    from omegaconf import DictConfig

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

    from omegaconf import DictConfig

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
