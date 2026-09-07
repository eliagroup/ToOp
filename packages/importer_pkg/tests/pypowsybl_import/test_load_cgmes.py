# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs


def _load_cgmes(
    grid_file: Path,
    *,
    with_subnetworks: bool,
) -> tuple[Network, float]:
    parameters = {
        "iidm.import.cgmes.post-processors": "cgmesGLImport",
        "iidm.import.cgmes.cgm-with-subnetworks": "true" if with_subnetworks else "false",
    }

    start = time.perf_counter()
    network = load_powsybl_from_fs(
        filesystem=LocalFileSystem(),
        file_path=grid_file,
        parameters=parameters,
    )
    elapsed_seconds = time.perf_counter() - start
    return network, elapsed_seconds


def _assert_core_views_equal(left: Network, right: Network) -> None:
    assert left.get_buses().sort_index().equals(right.get_buses().sort_index())
    assert left.get_branches().sort_index().equals(right.get_branches().sort_index())
    assert left.get_injections().sort_index().equals(right.get_injections().sort_index())


def test_cgmes_full_subnetwork_toggle_preserves_core_network(
    test_pypowsybl_cgmes_with_3w_trafo: Path,
) -> None:
    baseline_net, _ = _load_cgmes(test_pypowsybl_cgmes_with_3w_trafo, with_subnetworks=True)
    optimized_net, _ = _load_cgmes(test_pypowsybl_cgmes_with_3w_trafo, with_subnetworks=False)

    _assert_core_views_equal(baseline_net, optimized_net)


@pytest.mark.performance
@pytest.mark.timeout(300)
def test_cgmes_full_import_without_subnetworks_is_faster(
    test_pypowsybl_cgmes_with_3w_trafo: Path,
    record_property,
) -> None:
    # Warm up JVM / importer internals once so the comparison is less dominated by first-use cost.
    _load_cgmes(test_pypowsybl_cgmes_with_3w_trafo, with_subnetworks=True)
    total_baseline_seconds = 0
    total_optimized_seconds = 0
    for i in range(100):
        _, optimized_seconds = _load_cgmes(test_pypowsybl_cgmes_with_3w_trafo, with_subnetworks=False)
        _, baseline_seconds = _load_cgmes(test_pypowsybl_cgmes_with_3w_trafo, with_subnetworks=True)

        total_baseline_seconds += baseline_seconds
        total_optimized_seconds += optimized_seconds

    assert total_optimized_seconds < total_baseline_seconds
