# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
from pathlib import Path

from toop_engine_topology_optimizer.benchmark.benchmark import run_task_process


def test_run_task_process_no_conn(cfg):
    #  Set the env variables
    os.environ["OMP_NUM_THREADS"] = str(cfg["omp_num_threads"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(0, cfg["num_cuda_devices"])])
    if cfg["xla_force_host_platform_device_count"] is not None:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cfg['xla_force_host_platform_device_count']}"

    # Run the task
    res = run_task_process(cfg)
    assert res is not None
    assert res["max_fitness"] > res["initial_fitness"], (
        "Initial fitness is greater than max fitness. Optimisation didn't work well"
    )
    # Assert the folder got created and is not empty
    res_path = Path(cfg["output_json"]).parent
    assert len(list(res_path.iterdir())) > 0
