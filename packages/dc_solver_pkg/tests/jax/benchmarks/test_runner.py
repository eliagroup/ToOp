# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from pathlib import Path
from unittest import mock

import pytest
from toop_engine_dc_solver.jax.benchmarks.runner import main, run_benchmark_process


@pytest.mark.timeout(100)
def test_runner(benchmark_config_file: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    output = tmp_path_factory.mktemp("test_runner") / "output.json"
    main(["--yaml_config", str(benchmark_config_file), "--output_json", str(output)])
    assert output.exists()
    with open(output, "r", encoding="utf-8") as f:
        results = json.load(f)
    assert isinstance(results, list)
    assert len(results) == 3


def test_run_benchmark_process(benchmark_config: dict):
    mock_conn = mock.Mock()
    run_benchmark_process(mock_conn, benchmark_config["benchmarks"][0])
    assert mock_conn.send.called
