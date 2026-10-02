# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import numpy as np
import pytest
from toop_engine_interfaces.asset_topology.asset_topology import MasterBusGroup
from toop_engine_interfaces.asset_topology.assets import Busbar, BusbarCoupler
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SppsRule,
    copy_without_spps_rules,
    get_monitored_station_elements,
    load_nminus1_definition,
    save_nminus1_definition,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
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
def example_nminus1_definition_spps(example_nminus1_definition: Nminus1Definition) -> Nminus1Definition:
    # Replace the branch2 outage by a multi-outage with a switch, and close a switch when a branch is de-energized
    def switch_closing_rule(scheme_name: str, condition_id: str, action_id: str) -> SppsRule:
        return SppsRule(
            scheme_name=scheme_name,
            conditions=[
                Condition(
                    condition_type=SppsConditionType.STATE,
                    condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
                    condition_element_unique_id=condition_id,
                )
            ],
            actions=[
                Action(
                    measure_element_unique_id=action_id, measure_type=SppsMeasureType.SWITCHING_STATE, measure_value="closed"
                )
            ],
        )

    basecase, branch1, _branch2, multi_outage = example_nminus1_definition.contingencies
    switch_outage = Contingency(
        id="multi_outage_with_switch", elements=[*branch1.elements, GridElement(id="switch1", type="switch", kind="switch")]
    )
    return Nminus1Definition(
        contingencies=[basecase, branch1, multi_outage, switch_outage],
        monitored_elements=example_nminus1_definition.monitored_elements,
        spps_rules=[
            switch_closing_rule("branch1", condition_id="branch1", action_id="switch2"),
            switch_closing_rule("multi_outage_with_switch", condition_id="branch2", action_id="switch1"),
        ],
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
    assert example_nminus1_definition_spps.base_case.id == "BASECASE", "Base case id should match"
    assert len(example_nminus1_definition_spps.spps_rules) == 2, "Should have 2 SPPS rules"


def test_load_save_nminus1_definition(
    example_nminus1_definition: Nminus1Definition, tmp_path_factory: pytest.TempPathFactory
):
    temp_dir = tmp_path_factory.mktemp("nminus1")
    # Save the Nminus1Definition to a file
    file_path = temp_dir / "nminus1_definition.json"
    save_nminus1_definition(file_path, example_nminus1_definition)

    copy = load_nminus1_definition(file_path)
    assert copy == example_nminus1_definition, "Loaded Nminus1Definition does not match"


@pytest.mark.parametrize(
    ("make_inconsistent", "invalid_scheme_name"),
    [
        (lambda dump: dump["spps_rules"][0].update(scheme_name="missing"), "missing"),
        (lambda dump: dump["contingencies"].append(dump["contingencies"][1]), "branch1"),
    ],
    ids=["unknown_scheme_name", "duplicate_contingency_id"],
)
def test_nminus1_definition_rejects_inconsistent_spps_rules(
    example_nminus1_definition_spps: Nminus1Definition, make_inconsistent, invalid_scheme_name: str
) -> None:
    dump = example_nminus1_definition_spps.model_dump()
    make_inconsistent(dump)
    with pytest.raises(ValueError, match=invalid_scheme_name):
        Nminus1Definition.model_validate(dump)


def test_copy_without_spps_rules_preserves_definition_fields(example_nminus1_definition_spps: Nminus1Definition) -> None:
    copy = copy_without_spps_rules(example_nminus1_definition_spps)

    assert copy == example_nminus1_definition_spps.model_copy(update={"spps_rules": None})
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


def test_get_monitored_station_elements_uses_powsybl_types() -> None:
    """Node-breaker busbars are busbar sections, other busbars bus-breaker buses, and couplers switches."""
    station = MasterBusGroup(
        bus_group_id="station",
        busbars=[
            Busbar(int_id=0, grid_model_id="BBS1", busbar_type="busbar", name="busbar section"),
            Busbar(int_id=1, grid_model_id="BUS2", busbar_type=None),
        ],
        couplers=[BusbarCoupler(grid_model_id="COUPLER", coupler_type="BREAKER")],
        branch_connectivity=np.zeros((2, 0), dtype=bool),
        injection_connectivity=np.zeros((2, 0), dtype=bool),
    )

    monitored = get_monitored_station_elements([station])

    assert [(element.id, element.name, element.type, element.kind) for element in monitored] == [
        ("BBS1", "busbar section", "BUSBAR_SECTION", "bus"),
        ("BUS2", "", "BUS_BREAKER_BUS", "bus"),
        ("COUPLER", "", "SWITCH", "switch"),
    ]
