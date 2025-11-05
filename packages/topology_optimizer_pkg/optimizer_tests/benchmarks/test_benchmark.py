from pathlib import Path

from toop_engine_topology_optimizer.benchmark.benchmark_utils import run_task_process, set_environment_variables


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
