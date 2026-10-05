# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""End-to-end handoff of an input N-1 definition into the DC solver.

    create_complex_grid_nminus1_definition() -> convert_file -> nminus1_definition.json
      -> dc_nminus1_definition.json -> PowsyblBackend -> NetworkData / StaticInformation

The source cases pair a component with its isolating switches, which DC cannot represent:

===============================  ====================================  ==========================
source case                      source elements                       DC projection
===============================  ====================================  ==========================
C_L_DE_BE_1                      line + 2 breakers                     single branch outage
C_L_NL_1_2                       line + 2 breakers                     single branch outage
C_L8_WITH_LINE_OUT_OF_SERVICE    line + 2 breakers                     single branch outage
C_3W                             3W trafo + 6 switches                 3-branch multi-outage
C_NL_3W_1                        3W trafo + 4 switches                 dropped (islanding)
C_HVDC_LCC                       HVDC + 2 breakers                     dropped (unsupported)
C_MV_COUPLER                     coupler breaker only                  dropped (switch-only)
===============================  ====================================  ==========================
"""

import shutil
from pathlib import Path

import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import complex_grid_with_nminus1_definition_data_folder
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_busbar_outage_ids
from toop_engine_dc_solver.preprocess.preprocess import PreprocessParameters
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_nminus1_definition
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition

SOURCE_CONTINGENCY_IDS = [
    "BASECASE",
    "C_L_DE_BE_1",
    "C_L_NL_1_2",
    "C_L8_WITH_LINE_OUT_OF_SERVICE",
    "C_3W",
    "C_NL_3W_1",
    "C_HVDC_LCC",
    "C_MV_COUPLER",
]
SINGLE_OUTAGE_IDS = ["C_L8_WITH_LINE_OUT_OF_SERVICE", "C_L_DE_BE_1", "C_L_NL_1_2"]
MULTI_OUTAGE_IDS = ["C_3W"]


@pytest.fixture(scope="module")
def imported_complex_grid(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, StaticInformation, NetworkData]:
    """The data folder, static information and network data of the complex grid imported with its N-1 definition."""
    folder = tmp_path_factory.mktemp("complex_contingency_end_to_end")
    return folder, *complex_grid_with_nminus1_definition_data_folder(folder, create_complex_grid_nminus1_definition())


def test_grid_validated_definition_keeps_source_cases_grouping_and_spps(
    imported_complex_grid: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """The importer's grid-validated definition keeps every source case, its grouping and its SPPS rules."""
    definition = load_nminus1_definition(imported_complex_grid[0] / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    by_id = {contingency.id: contingency for contingency in definition.contingencies}

    assert list(by_id) == SOURCE_CONTINGENCY_IDS
    assert [element.id for element in by_id["C_L8_WITH_LINE_OUT_OF_SERVICE"].elements] == [
        "L8",
        "L81_BREAKER",
        "L82_BREAKER",
    ]
    assert [element.id for element in by_id["C_3W"].elements[:3]] == ["3W-Leg1", "3W-Leg2", "3W-Leg3"]
    assert definition.spps_rules is not None
    assert [rule.scheme_name for rule in definition.spps_rules] == ["C_L_DE_BE_1", "C_L8_WITH_LINE_OUT_OF_SERVICE", "C_3W"]


def test_dc_definition_holds_only_what_dc_computes(
    imported_complex_grid: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """The DC definition is a projection: cases DC cannot compute are absent, ids and no SPPS rules are kept."""
    dc_definition = load_nminus1_definition(
        imported_complex_grid[0] / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"]
    )

    assert dc_definition.base_case is not None
    assert sorted(c.id for c in dc_definition.contingencies) == sorted(["BASECASE", *SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])
    assert dc_definition.id_type == "powsybl"
    assert dc_definition.spps_rules is None


def test_network_data_classifies_single_and_multi_outages(
    imported_complex_grid: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """Isolating switches collapse into single outages and a 3W transformer is a multi-outage of its three legs.

    Solver results are joined positionally, so the solver's contingency order must match the network data exactly.
    """
    _, static_information, network_data = imported_complex_grid

    single_outage_ids = [
        network_data.contingency_id_by_element_id.get(branch_id, branch_id)
        for branch_id, outaged in zip(network_data.branch_ids, network_data.outaged_branch_mask, strict=True)
        if outaged
    ]
    assert sorted(single_outage_ids) == sorted(SINGLE_OUTAGE_IDS)

    assert list(network_data.multi_outage_ids) == MULTI_OUTAGE_IDS
    multi_outage_branch_ids = {
        multi_outage_id: [
            branch_id for branch_id, outaged in zip(network_data.branch_ids, branch_mask, strict=True) if outaged
        ]
        for multi_outage_id, branch_mask in zip(
            network_data.multi_outage_ids, network_data.multi_outage_branch_mask, strict=True
        )
    }
    assert multi_outage_branch_ids == {"C_3W": ["3W-Leg1", "3W-Leg2", "3W-Leg3"]}

    assert list(static_information.solver_config.contingency_ids) == network_data.contingency_ids
    assert sorted(network_data.contingency_ids) == sorted([*SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])


def test_definition_without_busbar_cases_outages_no_busbars(
    imported_complex_grid: tuple[Path, StaticInformation, NetworkData], tmp_path: Path
) -> None:
    """An input N-1 definition without busbar contingencies yields no DC busbar outages, even with them enabled."""
    data_folder, _static_information, _network_data = imported_complex_grid
    shutil.copytree(data_folder, tmp_path, dirs_exist_ok=True)
    _stats, _static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(tmp_path)),
        pandapower=False,
        parameters=PreprocessParameters(preprocess_bb_outages=True),
    )
    assert extract_busbar_outage_ids(network_data) == []
    dc_definition = load_nminus1_definition(tmp_path / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])
    assert not [c.id for c in dc_definition.contingencies if any(e.kind == "bus" for e in c.elements)]
