# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from toop_engine_dc_solver.jax.benchmarks.bench_postprocessing import (
    Args,
    main,
    run_benchmark,
    setup_benchmark,
)
from toop_engine_dc_solver.preprocess.network_data import load_network_data
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


@pytest.mark.parametrize("method", ["dc", "ac"])
def test_benchmark(preprocessed_data_folder: Path, method: str) -> None:
    runner, topologies = setup_benchmark(
        grid_path=preprocessed_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"],
        network_data_path=preprocessed_data_folder / PREPROCESSING_PATHS["network_data_file_path"],
        n_topologies=4,
        n_substations_split=2,
        n_processes_per_topology=2,
        batch_size_per_topology=None,
        framework="pandapower",
    )

    n_loadflows, n_success, time = run_benchmark(runner, topologies, method=method)

    network_data = load_network_data(preprocessed_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    assert n_loadflows == 4 * (
        +network_data.outaged_branch_mask.sum()
        + network_data.outaged_injection_mask.sum()
        + len(network_data.multi_outage_ids)
    ), "Number of Loadflows does not match the expected number of outages. Basecase is not included here!"
    assert n_success >= n_loadflows * 0.9
    assert time > 0


def test_main(preprocessed_data_folder: Path) -> None:
    with TemporaryDirectory() as res_folder:
        res_file = Path(res_folder) / "results.json"
        args = Args(
            network_data_file=str(preprocessed_data_folder / PREPROCESSING_PATHS["network_data_file_path"]),
            grid_file=str(preprocessed_data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]),
            framework="pandapower",
            n_topologies=4,
            n_substations_split=2,
            n_processes_per_topology=2,
            batch_size_per_topology=None,
            seed=0,
            method="dc",
            result_file=str(res_file),
        )
        main(args)

        assert res_file.exists()

        with open(args.result_file) as f:
            data = json.load(f)
            assert "n_loadflows" in data
            assert "n_success" in data
            assert "time" in data
            assert "timestamp" in data
            assert "args" in data
            assert data["n_loadflows"] > 0
            assert data["n_success"] > 0
            assert data["time"] > 0
            assert data["args"] == args.model_dump()
