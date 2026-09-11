# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""End-to-end handoff of an imported complex contingency list into the DC solver.

Covers the chain the importer handoff calls out as untested:

    contingency_list_complex.json -> convert_file -> nminus1_definition.json
      -> dc_nminus1_definition.json -> PowsyblBackend -> NetworkData / StaticInformation

The fixture list is chosen so every projection outcome appears exactly once, because the source
cases pair a component with the switches that isolate it and DC cannot represent switches:

===============================  ====================================  ==========================
source case                      source elements                       DC projection
===============================  ====================================  ==========================
C_L_DE_BE_1                      line + 2 breakers                     single branch outage
C_L_NL_1_2                       line + 2 breakers                     single branch outage
C_L8_WITH_LINE_OUT_OF_SERVICE    line + 2 breakers                     single branch outage
C_3W                             3W trafo + 6 switches                 3-branch multi-outage
C_NL_3W_1                        3W trafo + 4 switches                 dropped (islanding)
C_HVDC_LCC                       HVDC + 2 breakers                     dropped (unsupported)
C_MV_COUPLER                     coupler breaker only                  dropped (nothing to outage)
===============================  ====================================  ==========================
"""

import shutil
from pathlib import Path

import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, load_nminus1_definition

CONTINGENCY_LIST_FILE = Path(__file__).parents[4] / "data/complex_grid/contingency_list_complex.json"

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
DROPPED_IDS = ["C_NL_3W_1", "C_HVDC_LCC", "C_MV_COUPLER"]


@pytest.fixture(scope="module")
def _imported_complex_contingency_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Import the complex contingency list against the complex grid and run DC preprocessing."""
    folder = tmp_path_factory.mktemp("complex_contingency_end_to_end")
    net = create_complex_grid_battery_hvdc_svc_3w_trafo(connect_line_out_of_service=True)
    pypowsybl.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    grid_file_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(grid_file_path)

    preprocessing.convert_file(
        importer_parameters=CgmesImporterParameters(
            grid_model_file=grid_file_path,
            data_folder=folder,
            contingency_list_file=CONTINGENCY_LIST_FILE,
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
def imported_complex_contingency_folder(_imported_complex_contingency_folder: Path, tmp_path: Path) -> Path:
    shutil.copytree(_imported_complex_contingency_folder, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture(scope="module")
def dc_runtime(_imported_complex_contingency_folder: Path) -> tuple[StaticInformation, NetworkData]:
    _stats, static_information, network_data = load_grid(
        data_folder_dirfs=DirFileSystem(str(_imported_complex_contingency_folder)), pandapower=False
    )
    return static_information, network_data


def _canonical(folder: Path) -> Nminus1Definition:
    return load_nminus1_definition(folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])


def _dc(folder: Path) -> Nminus1Definition:
    return load_nminus1_definition(folder / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])


def test_canonical_definition_keeps_every_source_case(imported_complex_contingency_folder: Path) -> None:
    """The canonical definition is the importer's artifact and keeps the full source list."""
    canonical = _canonical(imported_complex_contingency_folder)

    assert [contingency.id for contingency in canonical.contingencies] == SOURCE_CONTINGENCY_IDS
    assert canonical.source_schema == "complex"


def test_canonical_definition_keeps_grouped_membership_and_spps_rules(
    imported_complex_contingency_folder: Path,
) -> None:
    """Grouping and SPPS survive import even where DC later discards them."""
    canonical = _canonical(imported_complex_contingency_folder)
    by_id = {contingency.id: contingency for contingency in canonical.contingencies}

    assert [element.id for element in by_id["C_L8_WITH_LINE_OUT_OF_SERVICE"].elements] == [
        "L8",
        "L81_BREAKER",
        "L82_BREAKER",
    ]
    assert [element.id for element in by_id["C_3W"].elements[:3]] == ["3W-Leg1", "3W-Leg2", "3W-Leg3"]
    assert canonical.spps_rules is not None
    assert [rule.scheme_name for rule in canonical.spps_rules] == [
        "C_L_DE_BE_1",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W",
    ]


def test_dc_definition_holds_only_what_dc_computes(imported_complex_contingency_folder: Path) -> None:
    """The DC definition is a projection: cases DC cannot compute are absent, provenance is kept."""
    dc_definition = _dc(imported_complex_contingency_folder)
    contingency_ids = [contingency.id for contingency in dc_definition.contingencies]

    assert dc_definition.base_case is not None
    assert sorted(contingency_ids) == sorted(["BASECASE", *SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])
    for dropped_id in DROPPED_IDS:
        assert dropped_id not in contingency_ids
    assert dc_definition.source_schema == "complex"
    assert dc_definition.id_type == "powsybl"


def test_dc_definition_carries_no_spps_rules(imported_complex_contingency_folder: Path) -> None:
    """DC neither executes nor stores SPPS rules, while the canonical definition keeps them."""
    assert _dc(imported_complex_contingency_folder).spps_rules is None
    assert _canonical(imported_complex_contingency_folder).spps_rules is not None


def test_isolating_switches_collapse_to_single_branch_outages(
    dc_runtime: tuple[StaticInformation, NetworkData],
) -> None:
    """A component plus its isolators is the single outage of that component, not a multi-outage."""
    _static_information, network_data = dc_runtime

    outaged_contingency_ids = [
        network_data.contingency_id_by_element_id.get(branch_id, branch_id)
        for branch_id, outaged in zip(network_data.branch_ids, network_data.outaged_branch_mask, strict=True)
        if outaged
    ]

    assert sorted(outaged_contingency_ids) == sorted(SINGLE_OUTAGE_IDS)
    # They must not also appear as MODF cases, which would double-count them.
    assert not set(SINGLE_OUTAGE_IDS) & set(network_data.multi_outage_ids)


def test_three_winding_transformer_is_a_genuine_multi_outage(
    dc_runtime: tuple[StaticInformation, NetworkData],
) -> None:
    """A 3W transformer expands to three legs, which is a real multi-outage for MODF."""
    _static_information, network_data = dc_runtime

    assert list(network_data.multi_outage_ids) == MULTI_OUTAGE_IDS
    outaged_branches = {
        multi_outage_id: [
            branch_id for branch_id, outaged in zip(network_data.branch_ids, branch_mask, strict=True) if outaged
        ]
        for multi_outage_id, branch_mask in zip(
            network_data.multi_outage_ids, network_data.multi_outage_branch_mask, strict=True
        )
    }
    assert outaged_branches == {"C_3W": ["3W-Leg1", "3W-Leg2", "3W-Leg3"]}


def test_no_multi_outage_row_is_empty(dc_runtime: tuple[StaticInformation, NetworkData]) -> None:
    """An emptied multi-outage would be a no-op contingency masquerading as a real one."""
    _static_information, network_data = dc_runtime

    assert network_data.multi_outage_branch_mask.shape[0] == len(network_data.multi_outage_ids)
    assert network_data.multi_outage_branch_mask.any(axis=1).all()


def test_runtime_contingency_order_is_consistent(dc_runtime: tuple[StaticInformation, NetworkData]) -> None:
    """Solver results are joined positionally, so both id lists must agree exactly."""
    static_information, network_data = dc_runtime

    assert list(static_information.solver_config.contingency_ids) == network_data.contingency_ids
    assert sorted(network_data.contingency_ids) == sorted([*SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])
