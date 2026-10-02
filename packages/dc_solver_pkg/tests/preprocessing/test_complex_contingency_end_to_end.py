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
C_MV_COUPLER                     coupler breaker only                  dropped (nothing to outage)
===============================  ====================================  ==========================
"""

from pathlib import Path

import pytest
from tests.complex_grid_import import import_complex_grid
from toop_engine_dc_solver.jax.types import StaticInformation
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_nminus1_definition
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import Nminus1Definition, load_nminus1_definition

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
def dc_runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, StaticInformation, NetworkData]:
    folder = tmp_path_factory.mktemp("complex_contingency_end_to_end")
    return folder, *import_complex_grid(folder, create_complex_grid_nminus1_definition())


def _canonical(folder: Path) -> Nminus1Definition:
    return load_nminus1_definition(folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])


def test_canonical_definition_keeps_every_source_case(dc_runtime: tuple[Path, StaticInformation, NetworkData]) -> None:
    """The canonical definition is the importer's artifact and keeps the full source list."""
    assert [contingency.id for contingency in _canonical(dc_runtime[0]).contingencies] == SOURCE_CONTINGENCY_IDS


def test_canonical_definition_keeps_grouped_membership_and_spps_rules(
    dc_runtime: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """Grouping and SPPS survive import even where DC later discards them."""
    canonical = _canonical(dc_runtime[0])
    by_id = {contingency.id: contingency for contingency in canonical.contingencies}

    assert [element.id for element in by_id["C_L8_WITH_LINE_OUT_OF_SERVICE"].elements] == [
        "L8",
        "L81_BREAKER",
        "L82_BREAKER",
    ]
    assert [element.id for element in by_id["C_3W"].elements[:3]] == ["3W-Leg1", "3W-Leg2", "3W-Leg3"]
    assert canonical.spps_rules is not None
    assert [rule.scheme_name for rule in canonical.spps_rules] == ["C_L_DE_BE_1", "C_L8_WITH_LINE_OUT_OF_SERVICE", "C_3W"]


def test_dc_definition_holds_only_what_dc_computes(dc_runtime: tuple[Path, StaticInformation, NetworkData]) -> None:
    """The DC definition is a projection: cases DC cannot compute are absent, ids and no SPPS rules are kept."""
    dc_definition = load_nminus1_definition(dc_runtime[0] / PREPROCESSING_PATHS["dc_nminus1_definition_file_path"])

    assert dc_definition.base_case is not None
    assert sorted(c.id for c in dc_definition.contingencies) == sorted(["BASECASE", *SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])
    assert dc_definition.id_type == "powsybl"
    assert dc_definition.spps_rules is None


def test_isolating_switches_collapse_to_single_branch_outages(
    dc_runtime: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """A component plus its isolators is the single outage of that component, not a multi-outage."""
    _, _, network_data = dc_runtime

    outaged_contingency_ids = [
        network_data.contingency_id_by_element_id.get(branch_id, branch_id)
        for branch_id, outaged in zip(network_data.branch_ids, network_data.outaged_branch_mask, strict=True)
        if outaged
    ]

    assert sorted(outaged_contingency_ids) == sorted(SINGLE_OUTAGE_IDS)
    # They must not also appear as MODF cases, which would double-count them.
    assert not set(SINGLE_OUTAGE_IDS) & set(network_data.multi_outage_ids)


def test_three_winding_transformer_is_a_genuine_multi_outage(
    dc_runtime: tuple[Path, StaticInformation, NetworkData],
) -> None:
    """A 3W transformer expands to three legs, which is a real multi-outage for MODF."""
    _, _, network_data = dc_runtime

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


def test_runtime_contingency_order_is_consistent(dc_runtime: tuple[Path, StaticInformation, NetworkData]) -> None:
    """Solver results are joined positionally, so both id lists must agree exactly."""
    _, static_information, network_data = dc_runtime

    assert list(static_information.solver_config.contingency_ids) == network_data.contingency_ids
    assert sorted(network_data.contingency_ids) == sorted([*SINGLE_OUTAGE_IDS, *MULTI_OUTAGE_IDS])
