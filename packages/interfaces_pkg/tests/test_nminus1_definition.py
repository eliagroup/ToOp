# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    Nminus1Definition,
    load_nminus1_definition,
    save_nminus1_definition,
)


@pytest.fixture
def example_nminus1_definition():
    # Create a simple Nminus1Definition with a base case and one contingency
    contingencies = [
        Contingency(id="BASECASE", name="base_case", elements=[]),
        Contingency(id="branch1", elements=[GridElement(id="branch1", type="line", kind="branch")]),
        Contingency(id="branch2", elements=[GridElement(id="branch2", type="line", kind="branch")]),
        Contingency(
            id="multi_outage",
            elements=[
                GridElement(id="branch1", type="line", kind="branch"),
                GridElement(id="branch2", type="line", kind="branch"),
            ],
        ),
    ]

    monitored_elements = [
        GridElement(id="branch1", type="line", kind="branch"),
        GridElement(id="branch2", type="line", kind="branch"),
        GridElement(id="bus1", type="bus", kind="bus"),
    ]

    return Nminus1Definition(
        contingencies=contingencies,
        monitored_elements=monitored_elements,
    )


def test_nminus1_definition(example_nminus1_definition: Nminus1Definition):
    # Test basic properties of the Nminus1Definition
    assert len(example_nminus1_definition.contingencies) == 4, "Should have 4 contingencies"
    assert example_nminus1_definition.base_case is not None, "Should have a base case contingency"
    assert example_nminus1_definition.base_case.is_basecase(), "Base case should be identified correctly"
    assert example_nminus1_definition.base_case.id == "BASECASE", "Base case id should match"

    # Test contingency identification
    for contingency in example_nminus1_definition.contingencies:
        if contingency.is_single_outage():
            assert len(contingency.elements) == 1, "Single outage should have exactly one element"
        elif contingency.is_multi_outage():
            assert len(contingency.elements) > 1, "Multi outage should have more than one element"


def test_load_save_nminus1_definition(
    example_nminus1_definition: Nminus1Definition, tmp_path_factory: pytest.TempPathFactory
):
    with tmp_path_factory.mktemp("nminus1") as temp_dir:
        # Save the Nminus1Definition to a file
        file_path = temp_dir / "nminus1_definition.json"
        save_nminus1_definition(file_path, example_nminus1_definition)

        copy = load_nminus1_definition(file_path)
        assert copy == example_nminus1_definition, "Loaded Nminus1Definition does not match"


def test_contingency_methods():
    basecase_contingency = Contingency(id="basecase", elements=[])
    assert basecase_contingency.is_basecase(), "Basecase contingency should be identified as basecase"
    assert not basecase_contingency.is_single_outage(), "Basecase contingency should not be a single outage"
    assert not basecase_contingency.is_multi_outage(), "Basecase contingency should not be a multi-outage"

    single_contingency = Contingency(id="single_outage", elements=[GridElement(id="line1", type="line", kind="branch")])
    assert not single_contingency.is_basecase(), "Single outage contingency should not be identified as basecase"
    assert single_contingency.is_single_outage(), "Single outage contingency should be identified as single outage"
    assert not single_contingency.is_multi_outage(), "Single outage contingency should not be a multi-outage"
    multi_contingency = Contingency(
        id="multi_outage",
        elements=[
            GridElement(id="line1", type="line", kind="branch"),
            GridElement(id="line2", type="line", kind="branch"),
        ],
    )
    assert not multi_contingency.is_basecase(), "Multi outage contingency should not be identified as basecase"
    assert not multi_contingency.is_single_outage(), "Multi outage contingency should not be a single outage"
    assert multi_contingency.is_multi_outage(), "Multi outage contingency should be identified as multi-outage"


def test_slice_n_minus_1_definition(example_nminus1_definition: Nminus1Definition) -> None:
    # Test the extraction of the N-1 definition
    n_minus_1_definition = example_nminus1_definition
    n_minus_1_definition_slice = n_minus_1_definition[1]
    assert len(n_minus_1_definition_slice.contingencies) == 1, "Only one contingency should be selected"
    assert n_minus_1_definition_slice.contingencies[0].id == n_minus_1_definition.contingencies[1].id, (
        "Since the second contingency is selected, it should match the original definition"
    )
    assert len(n_minus_1_definition_slice.monitored_elements) == len(n_minus_1_definition.monitored_elements), (
        "All monitored elements should be included in the slice"
    )

    n_minus_1_definition_slice = n_minus_1_definition[0:2]
    assert len(n_minus_1_definition_slice.contingencies) == 2, "Two contingencies should be selected"
    assert n_minus_1_definition_slice.contingencies[0].id == n_minus_1_definition.contingencies[0].id, (
        "First contingency should match the original definition"
    )
    assert n_minus_1_definition_slice.contingencies[1].id == n_minus_1_definition.contingencies[1].id, (
        "Second contingency should match the original definition"
    )
    assert len(n_minus_1_definition_slice.monitored_elements) == len(n_minus_1_definition.monitored_elements), (
        "All monitored elements should be included in the slice"
    )

    pick_by_id = n_minus_1_definition.contingencies[1].id
    n_minus_1_definition_slice = n_minus_1_definition[pick_by_id]
    assert len(n_minus_1_definition_slice.contingencies) == 1, "Only one contingency should be selected by id"
    assert n_minus_1_definition_slice.contingencies[0].id == pick_by_id, "Selected contingency should match the id"
    assert len(n_minus_1_definition_slice.monitored_elements) == len(n_minus_1_definition.monitored_elements), (
        "All monitored elements should be included in the slice"
    )
