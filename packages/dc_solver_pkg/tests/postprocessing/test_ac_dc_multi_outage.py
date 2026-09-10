# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""AC validation of the complex list's multi-outages, and its N-1 scope against the DC solver.

The complex contingency list pairs each faulted component with the switches that isolate it, so a
line becomes a single branch outage in DC while a three-winding transformer becomes a genuine
multi-outage. This module extends that list with one imported multi-branch business group -
``C_DOUBLE_LINE``, two lines that are each already a non-bridge single outage - so an imported
(non-synthesised) multi-outage also survives DC unchanged.

Two things are checked, per the design decision that DC and AC loadflow *values* are not compared
(DC ignores losses and voltage):

- the DC-computed N-1 scope is the expected subset of what AC runs on the canonical definition, and
  the grouped multi-outages AC drops for DC (islanding, unsupported) are still attempted by AC;
- the Powsybl AC security analysis of each surviving multi-outage agrees with a brute-force AC
  reference on the monitored branches (AC-vs-AC self-consistency), skipping any case that does not
  converge by nature.
"""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    extract_action_set,
    extract_nminus1_definition,
)
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, load_nminus1_definition

BASE_CONTINGENCY_LIST_FILE = Path(__file__).parents[4] / "data/complex_grid/contingency_list_complex.json"

# One imported multi-branch business group. Both lines are already non-bridge single outages of this
# grid (C_L_DE_BE_1, C_L_NL_1_2 survive DC), so grouping the two distant lines is guaranteed not to
# island and DC keeps it as a genuine two-branch multi-outage. Its breakers are the ones the two
# single cases already open.
IMPORTED_GROUP_ID = "C_DOUBLE_LINE"
IMPORTED_GROUP_BRANCH_IDS = ["L_DE_BE_1", "L_NL_1_2"]
IMPORTED_GROUP_ENTRY = {
    "Name": IMPORTED_GROUP_ID,
    "FaultCase": "Simultaneous outage of DE-BE interconnector 1 and NL corridor line 1-2",
    "InterruptedComponents": [
        {"Name": "L_DE_BE_1", "RdfId": "_L_DE_BE_1"},
        {"Name": "L_NL_1_2", "RdfId": "_L_NL_1_2"},
    ],
    "OpenedSwitches": [
        {"Name": "L_DE_BE_11_BREAKER", "RdfId": "_L_DE_BE_11_BREAKER"},
        {"Name": "L_DE_BE_12_BREAKER", "RdfId": "_L_DE_BE_12_BREAKER"},
        {"Name": "L_NL_1_21_BREAKER", "RdfId": "_L_NL_1_21_BREAKER"},
        {"Name": "L_NL_1_22_BREAKER", "RdfId": "_L_NL_1_22_BREAKER"},
    ],
    "ClosedSwitches": [],
    "OutOfService": 0,
}

# What the DC projection keeps, mirroring test_complex_contingency_end_to_end plus the imported group.
SINGLE_OUTAGE_IDS = ["C_L8_WITH_LINE_OUT_OF_SERVICE", "C_L_DE_BE_1", "C_L_NL_1_2"]
MULTI_OUTAGE_IDS = ["C_3W", IMPORTED_GROUP_ID]
# Grouped cases AC still attempts even though DC cannot represent them.
DROPPED_GROUP_IDS = ["C_NL_3W_1", "C_HVDC_LCC"]


@pytest.fixture(scope="module")
def _multi_outage_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Import the complex list - extended with the imported group - and run DC preprocessing."""
    folder = tmp_path_factory.mktemp("ac_dc_multi_outage")

    extended_list = json.loads(BASE_CONTINGENCY_LIST_FILE.read_text())
    extended_list[IMPORTED_GROUP_ID] = IMPORTED_GROUP_ENTRY
    contingency_list_file = folder / "contingency_list_complex_with_group.json"
    contingency_list_file.write_text(json.dumps(extended_list, indent=1))

    net = create_complex_grid_battery_hvdc_svc_3w_trafo(connect_line_out_of_service=True)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    preprocessing.convert_file(
        importer_parameters=CgmesImporterParameters(
            grid_model_file=grid_file_path,
            data_folder=folder,
            contingency_list_file=contingency_list_file,
            schema_format="ContingencyImportSchemaComplex",
            fail_on_non_convergence=False,
            area_settings=AreaSettings(
                cutoff_voltage=1.0,
                control_area=["BE", "NL"],
                view_area=["BE", "NL"],
                nminus1_area=["BE", "NL"],
                dso_trafo_factors=None,
                dso_trafo_weight=1.0,
                border_line_factors=None,
                border_line_weight=1.0,
            ),
        )
    )
    load_grid(data_folder_dirfs=DirFileSystem(str(folder)), pandapower=False)
    return folder


@pytest.fixture(scope="function")
def multi_outage_folder(_multi_outage_folder: Path, tmp_path: Path) -> Path:
    shutil.copytree(_multi_outage_folder, tmp_path, dirs_exist_ok=True)
    return tmp_path


def _canonical(folder: Path) -> Nminus1Definition:
    return load_nminus1_definition(folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])


def _dc_definition(folder: Path) -> Nminus1Definition:
    return load_nminus1_definition(folder / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])


def _network_data(folder: Path) -> NetworkData:
    _stats, _static_information, network_data = load_grid(data_folder_dirfs=DirFileSystem(str(folder)), pandapower=False)
    return network_data


def _multi_outage_row(network_data: NetworkData, multi_outage_id: str) -> int:
    return list(network_data.multi_outage_ids).index(multi_outage_id)


def test_imported_group_survives_dc_as_genuine_multi_outage(multi_outage_folder: Path) -> None:
    """The imported two-line group is kept whole - two branches, no leg spared - unlike a trafo3w."""
    network_data = _network_data(multi_outage_folder)

    assert IMPORTED_GROUP_ID in list(network_data.multi_outage_ids)
    group_row = _multi_outage_row(network_data, IMPORTED_GROUP_ID)
    group_mask = network_data.multi_outage_branch_mask[group_row]

    group_branch_ids = {network_data.branch_ids[i] for i in np.flatnonzero(group_mask)}
    assert group_branch_ids == set(IMPORTED_GROUP_BRANCH_IDS)

    # An imported group is computed as declared or not at all: nothing is spared. Surviving DC with
    # both branches intact is the feasibility proof - the drop rule already ran the bridge and
    # cut-set tests on the unreduced graph during preprocessing.
    assert network_data.multi_outage_spared_branch_mask is not None
    assert not network_data.multi_outage_spared_branch_mask[group_row].any()

    # The trafo3w remains a synthesised group that is repaired by sparing exactly one leg.
    trafo3w_row = _multi_outage_row(network_data, "C_3W")
    assert network_data.multi_outage_types[trafo3w_row] == "trafo3w"
    assert int(network_data.multi_outage_spared_branch_mask[trafo3w_row].sum()) == 1


def test_dc_analysis_scope_is_expected_subset_of_ac_scope(multi_outage_folder: Path) -> None:
    """DC computes a subset of the canonical cases; AC still attempts the ones DC drops."""
    canonical = _canonical(multi_outage_folder)
    dc_definition = _dc_definition(multi_outage_folder)

    dc_ids = [contingency.id for contingency in dc_definition.contingencies]
    assert sorted(dc_ids) == sorted(["BASECASE", *SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])

    net = pypowsybl.network.load(str(multi_outage_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]))
    ac_results = get_ac_loadflow_results(
        net=net, n_minus_1_definition=canonical, timestep=0, lf_params=CGMES_DISTRIBUTED_SLACK
    )
    ac_converged = ac_results.converged.filter(pl.col("timestep") == 0).select("contingency").unique().collect()
    ac_ids = set(ac_converged["contingency"].to_list())

    # Everything DC computes is also run by AC.
    dc_non_basecase = {contingency.id for contingency in dc_definition.contingencies if not contingency.is_basecase()}
    assert dc_non_basecase <= ac_ids

    # AC runs the surviving multi-outages...
    for multi_outage_id in MULTI_OUTAGE_IDS:
        assert multi_outage_id in ac_ids
    # ...and also the grouped cases DC had to drop.
    for dropped_id in DROPPED_GROUP_IDS:
        assert dropped_id in ac_ids


def test_ac_security_analysis_matches_brute_force_for_surviving_multi_outages(multi_outage_folder: Path) -> None:
    """Each surviving multi-outage's AC flows match a one-off AC loadflow of the same outage.

    Non-converging cases are skipped explicitly: some grouped outages do not converge by nature and
    carry no comparable flow.
    """
    network_data = _network_data(multi_outage_folder)
    dc_definition = extract_nminus1_definition(network_data)

    runner = PowsyblRunner(lf_params=CGMES_DISTRIBUTED_SLACK)
    runner.load_base_grid(multi_outage_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))
    runner.store_nminus1_definition(dc_definition)

    results = runner.run_ac_loadflow([], [])
    _n_0, n_1, _success = extract_solver_matrices_polars(
        loadflow_results=results, nminus1_definition=dc_definition, timestep=0
    )

    monitored_branch_ids = [element.id for element in dc_definition.monitored_elements if element.kind == "branch"]
    contingency_order = [contingency.id for contingency in dc_definition.contingencies if not contingency.is_basecase()]

    net = runner.net
    base_result, *_ = pypowsybl.loadflow.run_ac(net, CGMES_DISTRIBUTED_SLACK)
    lf_params = copy.deepcopy(CGMES_DISTRIBUTED_SLACK)
    # Pin the slack to the base-case reference bus so the one-off loadflows match the security analysis.
    lf_params.read_slack_bus = False
    lf_params.provider_parameters["slackBusSelectionMode"] = "NAME"
    lf_params.provider_parameters["slackBusesIds"] = base_result.reference_bus_id
    lf_params.provider_parameters["alwaysUpdateNetwork"] = "true"

    surviving_multi_outages = [contingency for contingency in dc_definition.contingencies if len(contingency.elements) > 1]
    assert {contingency.id for contingency in surviving_multi_outages} == set(MULTI_OUTAGE_IDS)

    compared_ids = []
    for contingency in surviving_multi_outages:
        row = contingency_order.index(contingency.id)
        # A group that islands part of the grid (e.g. a trafo3w star node) does not converge as a
        # plain AC outage; the security analysis reports it as such. Skip those explicitly.
        if not _success[row]:
            continue

        outage_net = copy.deepcopy(net)
        outaged_branch_ids = [element.id for element in contingency.elements if element.kind == "branch"]
        for branch_id in outaged_branch_ids:
            outage_net.disconnect(branch_id)
        result, *_ = pypowsybl.loadflow.run_ac(outage_net, lf_params)
        if result.status != pypowsybl.loadflow.ComponentStatus.CONVERGED:
            continue

        branches = outage_net.get_branches(attributes=["p1"])
        reference_flows = branches.loc[monitored_branch_ids, "p1"].fillna(0.0).to_numpy()
        # A monitored branch that is itself outaged reads NaN/absent - drop it from the comparison.
        keep = np.array([branch_id not in set(outaged_branch_ids) for branch_id in monitored_branch_ids], dtype=bool)

        assert n_1[row].shape == reference_flows.shape
        np.testing.assert_allclose(np.abs(n_1[row][keep]), np.abs(reference_flows[keep]), atol=1e-2)
        compared_ids.append(contingency.id)

    # The imported two-line group converges as a plain AC outage and must actually be validated;
    # the trafo3w islands its star node and is skipped by nature (see the success guard above).
    assert IMPORTED_GROUP_ID in compared_ids
