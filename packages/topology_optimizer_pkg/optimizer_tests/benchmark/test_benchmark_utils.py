import multiprocessing as mp
from pathlib import Path

import pytest
from toop_engine_interfaces.messages.preprocess.preprocess_commands import CgmesImporterParameters, UcteImporterParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    get_paths,
    prepare_importer_parameters,
    run_task_process,
    set_environment_variables,
)


def test_run_task_process_no_conn(cfg):
    #  Set the env variables
    set_environment_variables(cfg)
    # Run the task
    res = run_task_process(cfg)
    assert res is not None
    assert res["max_fitness"] > res["initial_fitness"], (
        "Initial fitness is greater than max fitness. Optimisation didn't work well"
    )
    # Assert the folder got created and is not empty
    res_path = Path(cfg["output_json"]).parent
    assert len(list(res_path.iterdir())) > 0


def test_run_task_process_with_conn(cfg):
    # Set the env variables
    set_environment_variables(cfg)
    # Create a pipe
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    # Run the task
    res = run_task_process(cfg, conn=child_conn)
    # Read from the parent connection
    res_from_conn = parent_conn.recv()
    assert res is None, "When using a connection, the return value should be None"
    assert res_from_conn["max_fitness"] > res_from_conn["initial_fitness"], (
        "Initial fitness is greater than max fitness. Optimisation didn't work well"
    )
    # Assert the folder got created and is not empty
    res_path = Path(cfg["output_json"]).parent
    assert len(list(res_path.iterdir())) > 0


def test_get_paths_file_does_not_exist(pipeline_cfg):
    import copy

    pipeline_cfg_ = copy.deepcopy(pipeline_cfg)
    pipeline_cfg_.file_name = "non_existent_file.xiidm"
    with pytest.raises(FileNotFoundError):
        get_paths(pipeline_cfg_)


def test_prepare_importer_parameters(pipeline_cfg):
    _, file_path, data_folder, _ = get_paths(pipeline_cfg)  # to create the paths

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
