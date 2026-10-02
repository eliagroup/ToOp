# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""AC validation of the multi-outages the DC solver keeps from the complex N-1 definition.

The definition is extended with ``C_DOUBLE_LINE``, an imported two-line group, next to the synthesised
trafo3w group ``C_3W``. DC and AC loadflow values are deliberately not compared (DC ignores losses and
voltage); instead AC must cover the DC scope and agree with a brute-force AC reference.
"""

import copy
from pathlib import Path

import numpy as np
import polars as pl
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from tests.complex_grid_import import import_complex_grid
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, extract_nminus1_definition
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_nminus1_definition
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    load_nminus1_definition,
)

# Both lines are already non-bridge single outages, so grouping them cannot island the grid and DC
# keeps the group as a genuine two-branch multi-outage, opening the breakers of the single cases.
IMPORTED_GROUP_BRANCH_IDS = ["L_DE_BE_1", "L_NL_1_2"]
IMPORTED_GROUP = Contingency(
    id="C_DOUBLE_LINE",
    name="Simultaneous outage of DE-BE interconnector 1 and NL corridor line 1-2",
    elements=[GridElement(id=line, name=line, type="LINE", kind="branch") for line in IMPORTED_GROUP_BRANCH_IDS]
    + [
        GridElement(id=breaker, name=breaker, type="SWITCH", kind="switch")
        for line in IMPORTED_GROUP_BRANCH_IDS
        for breaker in (f"{line}1_BREAKER", f"{line}2_BREAKER")
    ],
)
SINGLE_OUTAGE_IDS = ["C_L8_WITH_LINE_OUT_OF_SERVICE", "C_L_DE_BE_1", "C_L_NL_1_2"]
MULTI_OUTAGE_IDS = ["C_3W", IMPORTED_GROUP.id]
# Grouped cases DC drops (islanding, unsupported) but AC still attempts.
DROPPED_GROUP_IDS = ["C_NL_3W_1", "C_HVDC_LCC"]


@pytest.fixture(scope="module")
def multi_outage_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("ac_dc_multi_outage")
    nminus1_definition = create_complex_grid_nminus1_definition()
    nminus1_definition.contingencies.append(IMPORTED_GROUP)
    import_complex_grid(folder, nminus1_definition)
    return folder


@pytest.fixture(scope="module")
def multi_outage_network_data(multi_outage_folder: Path) -> NetworkData:
    _stats, _static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(multi_outage_folder)), pandapower=False
    )
    return network_data


def test_imported_group_survives_dc_as_genuine_multi_outage(multi_outage_network_data: NetworkData) -> None:
    """The imported two-line group is kept whole - two branches, no leg spared - unlike a trafo3w."""
    network_data = multi_outage_network_data
    multi_outage_ids = list(network_data.multi_outage_ids)
    assert IMPORTED_GROUP.id in multi_outage_ids
    group_row = multi_outage_ids.index(IMPORTED_GROUP.id)

    group_mask = network_data.multi_outage_branch_mask[group_row]
    assert {network_data.branch_ids[i] for i in np.flatnonzero(group_mask)} == set(IMPORTED_GROUP_BRANCH_IDS)

    # An imported group is computed as declared or not at all; the drop rule already ran the bridge
    # and cut-set tests on the unreduced graph, so surviving intact is the feasibility proof.
    assert network_data.multi_outage_spared_branch_mask is not None
    assert not network_data.multi_outage_spared_branch_mask[group_row].any()

    # The synthesised trafo3w group is repaired by sparing exactly one leg.
    trafo3w_row = multi_outage_ids.index("C_3W")
    assert network_data.multi_outage_types[trafo3w_row] == "trafo3w"
    assert int(network_data.multi_outage_spared_branch_mask[trafo3w_row].sum()) == 1


def test_dc_analysis_scope_is_expected_subset_of_ac_scope(multi_outage_folder: Path) -> None:
    """DC computes a subset of the canonical cases; AC still attempts the ones DC drops."""
    canonical = load_nminus1_definition(multi_outage_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    dc_definition = load_nminus1_definition(multi_outage_folder / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])

    dc_ids = [contingency.id for contingency in dc_definition.contingencies]
    assert sorted(dc_ids) == sorted(["BASECASE", *SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])

    net = pypowsybl.network.load(str(multi_outage_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]))
    ac_results = get_ac_loadflow_results(
        net=net, n_minus_1_definition=canonical, timestep=0, lf_params=CGMES_DISTRIBUTED_SLACK
    )
    ac_ids = set(ac_results.converged.filter(pl.col("timestep") == 0).collect()["contingency"].to_list())

    assert {contingency.id for contingency in dc_definition.contingencies if not contingency.is_basecase()} <= ac_ids
    assert {*MULTI_OUTAGE_IDS, *DROPPED_GROUP_IDS} <= ac_ids


def test_ac_security_analysis_matches_brute_force_for_surviving_multi_outages(
    multi_outage_folder: Path, multi_outage_network_data: NetworkData
) -> None:
    """Each converging surviving multi-outage's AC flows match a one-off AC loadflow of the same outage."""
    dc_definition = extract_nminus1_definition(multi_outage_network_data)
    runner = PowsyblRunner(lf_params=CGMES_DISTRIBUTED_SLACK)
    runner.load_base_grid(multi_outage_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(multi_outage_network_data))
    runner.store_nminus1_definition(dc_definition)
    _n_0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=runner.run_ac_loadflow([], []), nminus1_definition=dc_definition, timestep=0
    )

    monitored_branch_ids = [element.id for element in dc_definition.monitored_elements if element.kind == "branch"]
    contingency_order = [contingency.id for contingency in dc_definition.contingencies if not contingency.is_basecase()]

    # Pin the slack to the base-case reference bus so the one-off loadflows match the security analysis.
    base_result, *_ = pypowsybl.loadflow.run_ac(runner.net, CGMES_DISTRIBUTED_SLACK)
    lf_params = copy.deepcopy(CGMES_DISTRIBUTED_SLACK)
    lf_params.read_slack_bus = False
    lf_params.provider_parameters["slackBusSelectionMode"] = "NAME"
    lf_params.provider_parameters["slackBusesIds"] = base_result.reference_bus_id
    lf_params.provider_parameters["alwaysUpdateNetwork"] = "true"

    surviving_multi_outages = [contingency for contingency in dc_definition.contingencies if len(contingency.elements) > 1]
    assert {contingency.id for contingency in surviving_multi_outages} == set(MULTI_OUTAGE_IDS)

    compared_ids = []
    for contingency in surviving_multi_outages:
        row = contingency_order.index(contingency.id)
        # Groups that island part of the grid (e.g. a trafo3w star node) do not converge by nature.
        if not success[row]:
            continue
        outage_net = copy.deepcopy(runner.net)
        outaged_branch_ids = [element.id for element in contingency.elements if element.kind == "branch"]
        for branch_id in outaged_branch_ids:
            outage_net.disconnect(branch_id)
        result, *_ = pypowsybl.loadflow.run_ac(outage_net, lf_params)
        if result.status != pypowsybl.loadflow.ComponentStatus.CONVERGED:
            continue

        reference_flows = outage_net.get_branches(attributes=["p1"]).loc[monitored_branch_ids, "p1"].fillna(0.0).to_numpy()
        # Outaged monitored branches read NaN and are left out of the comparison.
        keep = ~np.isin(monitored_branch_ids, outaged_branch_ids)
        assert n_1[row].shape == reference_flows.shape
        np.testing.assert_allclose(np.abs(n_1[row][keep]), np.abs(reference_flows[keep]), atol=1e-2)
        compared_ids.append(contingency.id)

    # The two-line group must actually be validated; the trafo3w is skipped by nature.
    assert IMPORTED_GROUP.id in compared_ids
