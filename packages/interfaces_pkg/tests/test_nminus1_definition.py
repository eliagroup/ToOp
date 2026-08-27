# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SppsRule,
    copy_without_spps_rules,
    load_nminus1_definition,
    save_nminus1_definition,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionType,
    SppsMeasureType,
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
        MonitoredElement(id="branch1", type="line", kind="branch"),
        MonitoredElement(id="branch2", type="line", kind="branch"),
        MonitoredElement(id="bus1", type="bus", kind="bus"),
    ]

    return Nminus1Definition(
        contingencies=contingencies,
        monitored_elements=monitored_elements,
    )


@pytest.fixture
def example_nminus1_definition_spps():
    # Create an Nminus1Definition that contains a multi-outage contingency, and safety protection schemes
    contingencies = [
        Contingency(id="BASECASE", name="base_case", elements=[]),
        Contingency(id="branch1", elements=[GridElement(id="branch1", type="line", kind="branch")]),
        Contingency(
            id="multi_outage",
            elements=[
                GridElement(id="branch1", type="line", kind="branch"),
                GridElement(id="branch2", type="line", kind="branch"),
            ],
        ),
        Contingency(
            id="multi_outage_with_switch",
            elements=[
                GridElement(id="branch1", type="line", kind="branch"),
                GridElement(id="switch1", type="switch", kind="switch"),
            ],
        ),
    ]

    monitored_elements = [
        MonitoredElement(id="branch1", type="line", kind="branch"),
        MonitoredElement(id="branch2", type="line", kind="branch"),
        MonitoredElement(id="bus1", type="bus", kind="bus"),
    ]

    condition1 = Condition(
        condition_type=SppsConditionType.STATE,
        condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
        condition_element_unique_id="branch1",
    )
    condition2 = Condition(
        condition_type=SppsConditionType.STATE,
        condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
        condition_element_unique_id="branch2",
    )
    action1 = Action(
        measure_element_unique_id="switch1",
        measure_type=SppsMeasureType.SWITCHING_STATE,
        measure_value="closed",
    )
    action2 = Action(
        measure_element_unique_id="switch2",
        measure_type=SppsMeasureType.SWITCHING_STATE,
        measure_value="closed",
    )

    spps_rules = [
        SppsRule(scheme_name="branch1", condition_logic=SppsConditionLogic.ALL, conditions=[condition1], actions=[action2]),
        SppsRule(
            scheme_name="multi_outage_with_switch",
            condition_logic=SppsConditionLogic.ALL,
            conditions=[condition2],
            actions=[action1],
        ),
    ]

    return Nminus1Definition(
        contingencies=contingencies,
        monitored_elements=monitored_elements,
        spps_rules=spps_rules,
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


def test_nminus1_definition_spps(example_nminus1_definition_spps: Nminus1Definition):
    assert len(example_nminus1_definition_spps.contingencies) == 4, "Should have 4 contingencies"
    assert example_nminus1_definition_spps.base_case is not None, "Should have a base case contingency"
    assert example_nminus1_definition_spps.base_case.is_basecase(), "Base case should be identified correctly"
    assert example_nminus1_definition_spps.base_case.id == "BASECASE", "Base case id should match"

    example_nminus1_definition_spps_rules = example_nminus1_definition_spps.spps_rules
    assert len(example_nminus1_definition_spps_rules) == 2, "Should have 2 SPPS rules"


def test_load_save_nminus1_definition(
    example_nminus1_definition: Nminus1Definition, tmp_path_factory: pytest.TempPathFactory
):
    with tmp_path_factory.mktemp("nminus1") as temp_dir:
        # Save the Nminus1Definition to a file
        file_path = temp_dir / "nminus1_definition.json"
        save_nminus1_definition(file_path, example_nminus1_definition)

        copy = load_nminus1_definition(file_path)
        assert copy == example_nminus1_definition, "Loaded Nminus1Definition does not match"


def test_nminus1_definition_rejects_unknown_spps_scheme(example_nminus1_definition_spps: Nminus1Definition) -> None:
    spps_rules = example_nminus1_definition_spps.spps_rules
    assert spps_rules is not None
    with pytest.raises(ValueError, match="missing"):
        Nminus1Definition(
            monitored_elements=example_nminus1_definition_spps.monitored_elements,
            contingencies=example_nminus1_definition_spps.contingencies,
            spps_rules=[spps_rules[0].model_copy(update={"scheme_name": "missing"}), spps_rules[1]],
        )


def test_nminus1_definition_rejects_duplicate_contingency_ids_with_spps(
    example_nminus1_definition_spps: Nminus1Definition,
) -> None:
    spps_rules = example_nminus1_definition_spps.spps_rules
    assert spps_rules is not None

    with pytest.raises(ValueError, match="branch1"):
        Nminus1Definition(
            monitored_elements=example_nminus1_definition_spps.monitored_elements,
            contingencies=example_nminus1_definition_spps.contingencies + [example_nminus1_definition_spps.contingencies[1]],
            spps_rules=spps_rules,
        )


def test_copy_without_spps_rules_preserves_definition_fields(example_nminus1_definition_spps: Nminus1Definition) -> None:
    copy = copy_without_spps_rules(example_nminus1_definition_spps)

    assert type(copy) is type(example_nminus1_definition_spps)
    assert copy.id_type == example_nminus1_definition_spps.id_type
    assert copy.monitored_elements == example_nminus1_definition_spps.monitored_elements
    assert copy.contingencies == example_nminus1_definition_spps.contingencies
    assert copy.spps_rules is None
    assert copy.contingencies is not example_nminus1_definition_spps.contingencies


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
