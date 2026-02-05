# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower as pp
import pandera as pa
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import run_contingency_analysis_pandapower
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_result_helpers import (
    extract_branch_results,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def test_run_ac_contingency_analysis_pandapower(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)
    with pa.config.config_context(validation_enabled=True, validation_depth=pa.config.ValidationDepth.SCHEMA_AND_DATA):
        lf_result_sequential_polars = get_ac_loadflow_results(
            pandapower_net, nminus1_definition, job_id="test_job", n_processes=1
        )
    with pa.config.config_context(validation_enabled=False):
        lf_result_sequential_polars_no_val = get_ac_loadflow_results(
            pandapower_net, nminus1_definition, job_id="test_job", n_processes=1
        )
    assert lf_result_sequential_polars is not None
    assert lf_result_sequential_polars_no_val == lf_result_sequential_polars


@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(300)
@pytest.mark.skip(reason="Does not work on CI")
def test_run_ac_contingency_analysis_pandapower_mt(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)

    lf_result_parallel_polars = get_ac_loadflow_results(pandapower_net, nminus1_definition, job_id="test_job", n_processes=2)
    assert lf_result_parallel_polars is not None


def test_extract_branch_results_pandapower_disconnected():
    net = pp.networks.case14()
    net.line.loc[0, "in_service"] = False  # Disconnect the first line
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )

    res = run_contingency_analysis_pandapower(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    contingencies = [contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies


def test_extract_branch_results_pandapower_disconnected():
    net = pp.networks.case14()
    net.line.loc[0, "in_service"] = False  # Disconnect the first line
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )

    res = run_contingency_analysis_pandapower(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies
