# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
import pytest
import structlog.testing
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file import (
    convert_three_winding_transformers_in_nminus1_definition,
    get_nminus1_definition_element_ids,
)
from toop_engine_importer.pypowsybl_import.network_reduction import get_voltage_level_ids_of_elements
from toop_engine_interfaces.messages.preprocess.preprocess_commands import Trafo3wLegMonitoring
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SppsRule,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionType,
    SppsMeasureType,
    SppsSwitchActionTarget,
)

TRAFO3W = GridElement(id="3W", name="3W 380/110/63", type="THREE_WINDINGS_TRANSFORMER", kind="branch")
MONITORED_TRAFO3W = MonitoredElement(**TRAFO3W.model_dump())
LEG_IDS = ["3W-Leg1", "3W-Leg2", "3W-Leg3"]
LINE_L1 = GridElement(id="L1", type="LINE", kind="branch")
LINE_L8 = GridElement(id="L8", type="LINE", kind="branch")
BASECASE = Contingency(id="BASECASE", name="BASECASE", elements=[])
MONITORING_FLAGS = {"optimized", "non_worsening", "displayed"}


def _spps_rule_closing_switches_when_de_energized(
    scheme_name: str, de_energized_ids: list[str], closed_switch_ids: list[str], logic: SppsConditionLogic
) -> SppsRule:
    return SppsRule(
        scheme_name=scheme_name,
        condition_logic=logic,
        conditions=[
            Condition(
                condition_type=SppsConditionType.STATE,
                condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
                condition_element_unique_id=element_id,
            )
            for element_id in de_energized_ids
        ],
        actions=[
            Action(
                measure_element_unique_id=switch_id,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
            for switch_id in closed_switch_ids
        ],
    )


@pytest.fixture
def network_with_leg2_and_leg3_limits(complex_grid_network: Network) -> Network:
    """The complex grid with a permanent limit on legs 2 and 3 of ``3W``, so Leg2 is its lead leg."""
    complex_grid_network.create_operational_limits(
        element_id=["3W-Leg3", "3W-Leg2"],
        element_type=["TWO_WINDINGS_TRANSFORMER"] * 2,
        side=["ONE"] * 2,
        name=["permanent_limit"] * 2,
        type=["CURRENT"] * 2,
        value=[1000.0] * 2,
        acceptable_duration=[-1] * 2,
    )
    return complex_grid_network


def test_three_winding_transformer_elements_become_legs(complex_grid_network: Network) -> None:
    """A 3W outage or monitored 3W is replaced in place by its three legs, carrying the leg names and element class."""
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_3W", elements=[LINE_L8, TRAFO3W, LINE_L1])],
        monitored_elements=[MONITORED_TRAFO3W, MonitoredElement(**LINE_L8.model_dump())],
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)

    elements = converted.contingencies[1].elements
    assert [element.id for element in elements] == ["L8", *LEG_IDS, "L1"]
    assert [element.name for element in elements[1:4]] == [f"3W 380/110/63-Leg{leg}" for leg in (1, 2, 3)]
    assert {(element.type, element.kind) for element in elements[1:4]} == {("TWO_WINDINGS_TRANSFORMER", "branch")}
    assert converted.contingencies[0] == BASECASE
    assert [element.id for element in converted.monitored_elements] == [*LEG_IDS, "L8"]
    assert all(isinstance(element, MonitoredElement) for element in converted.monitored_elements)


@pytest.mark.parametrize("logic", [SppsConditionLogic.ALL, SppsConditionLogic.ANY])
def test_spps_three_winding_transformer_is_expanded_regardless_of_logic(
    complex_grid_network: Network, logic: SppsConditionLogic
) -> None:
    """3W conditions and actions are copied once per leg, whatever the condition logic."""
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_3W", elements=[TRAFO3W])],
        monitored_elements=[],
        spps_rules=[
            _spps_rule_closing_switches_when_de_energized(
                "C_3W", de_energized_ids=["L8", "3W"], closed_switch_ids=["3W", "BREAKER_3W_HV"], logic=logic
            )
        ],
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)

    assert converted.spps_rules is not None
    [rule] = converted.spps_rules
    assert rule.condition_logic == logic
    assert [condition.condition_element_unique_id for condition in rule.conditions] == ["L8", *LEG_IDS]
    assert {condition.condition_check_type for condition in rule.conditions} == {SppsConditionCheckType.DE_ENERGIZED}
    assert [action.measure_element_unique_id for action in rule.actions] == [*LEG_IDS, "BREAKER_3W_HV"]


def test_conversion_is_idempotent_and_does_not_duplicate_legs(complex_grid_network: Network) -> None:
    """Converting twice is a no-op, and legs that are already listed are not added twice."""
    leg1 = GridElement(id="3W-Leg1", type="TWO_WINDINGS_TRANSFORMER", kind="branch")
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_3W", elements=[leg1, TRAFO3W])], monitored_elements=[]
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)
    converted_twice = convert_three_winding_transformers_in_nminus1_definition(converted, complex_grid_network)

    assert [element.id for element in converted.contingencies[1].elements] == LEG_IDS
    assert converted_twice == converted


def test_three_winding_transformer_without_legs_is_kept_and_warned(complex_grid_network_unconverted: Network) -> None:
    """On a grid without converted legs, the 3W reference is left for the final grid check to handle."""
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_3W", elements=[TRAFO3W])], monitored_elements=[]
    )

    with structlog.testing.capture_logs() as cap_logs:
        converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network_unconverted)

    assert converted == definition
    assert [entry["event"] for entry in cap_logs] == ["three_winding_transformer_legs_missing"]


def test_definition_without_three_winding_transformers_is_unchanged(complex_grid_network: Network) -> None:
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_L8", elements=[LINE_L8])],
        monitored_elements=[MonitoredElement(**LINE_L1.model_dump())],
        spps_rules=[
            _spps_rule_closing_switches_when_de_energized("C_L8", ["L8"], ["BREAKER_3W_HV"], SppsConditionLogic.ALL)
        ],
    )

    assert convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network) == definition


@pytest.mark.parametrize(
    ("leg_monitoring", "trafo3w_flags", "expected_leg_flags"),
    [
        pytest.param(
            Trafo3wLegMonitoring.ALL_LEGS,
            {"optimized": False, "non_worsening": True},
            {leg_id: {"optimized": False, "non_worsening": True, "displayed": True} for leg_id in LEG_IDS},
            id="all_legs-every_leg_inherits",
        ),
        pytest.param(
            Trafo3wLegMonitoring.SINGLE_LEG,
            {"optimized": True, "non_worsening": False},
            {
                "3W-Leg1": {"optimized": False, "non_worsening": True, "displayed": False},
                "3W-Leg2": {"optimized": True, "non_worsening": False, "displayed": True},
                "3W-Leg3": {"optimized": False, "non_worsening": True, "displayed": False},
            },
            id="single_leg-optimized_keeps_other_legs_non_worsening",
        ),
        pytest.param(
            Trafo3wLegMonitoring.SINGLE_LEG,
            {"optimized": False, "non_worsening": True},
            {
                "3W-Leg1": {"optimized": False, "non_worsening": False, "displayed": False},
                "3W-Leg2": {"optimized": False, "non_worsening": True, "displayed": True},
                "3W-Leg3": {"optimized": False, "non_worsening": False, "displayed": False},
            },
            id="single_leg-non_worsening_unflags_other_legs",
        ),
        pytest.param(
            Trafo3wLegMonitoring.SINGLE_LEG,
            {"optimized": True, "non_worsening": True},
            {
                "3W-Leg1": {"optimized": False, "non_worsening": False, "displayed": False},
                "3W-Leg2": {"optimized": True, "non_worsening": True, "displayed": True},
                "3W-Leg3": {"optimized": False, "non_worsening": False, "displayed": False},
            },
            id="single_leg-optimized_and_non_worsening_unflags_other_legs",
        ),
        pytest.param(
            Trafo3wLegMonitoring.SINGLE_LEG,
            {"optimized": False, "non_worsening": False},
            {
                "3W-Leg1": {"optimized": False, "non_worsening": False, "displayed": False},
                "3W-Leg2": {"optimized": False, "non_worsening": False, "displayed": True},
                "3W-Leg3": {"optimized": False, "non_worsening": False, "displayed": False},
            },
            id="single_leg-unflagged_hides_other_legs",
        ),
        pytest.param(
            Trafo3wLegMonitoring.SINGLE_LEG,
            {"optimized": True, "non_worsening": False, "displayed": False},
            {
                "3W-Leg1": {"optimized": False, "non_worsening": True, "displayed": False},
                "3W-Leg2": {"optimized": True, "non_worsening": False, "displayed": False},
                "3W-Leg3": {"optimized": False, "non_worsening": True, "displayed": False},
            },
            id="single_leg-hidden_trafo3w_stays_hidden",
        ),
        pytest.param(
            Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS,
            {"optimized": True, "non_worsening": False},
            {"3W-Leg2": {"optimized": True, "non_worsening": False, "displayed": True}},
            id="remove_non_lead_legs-optimized",
        ),
        pytest.param(
            Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS,
            {"optimized": False, "non_worsening": True},
            {"3W-Leg2": {"optimized": False, "non_worsening": True, "displayed": True}},
            id="remove_non_lead_legs-non_worsening",
        ),
    ],
)
def test_monitored_trafo3w_legs_follow_leg_monitoring_mode(
    network_with_leg2_and_leg3_limits: Network,
    leg_monitoring: Trafo3wLegMonitoring,
    trafo3w_flags: dict[str, bool],
    expected_leg_flags: dict[str, dict[str, bool]],
) -> None:
    """The lead leg (the first with a limit, Leg2) inherits the 3W, the mode decides what happens to the other legs."""
    definition = Nminus1Definition(
        contingencies=[BASECASE], monitored_elements=[MonitoredElement(**TRAFO3W.model_dump(), **trafo3w_flags)]
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, network_with_leg2_and_leg3_limits, leg_monitoring=leg_monitoring
    )

    leg_flags = {element.id: element.model_dump(include=MONITORING_FLAGS) for element in converted.monitored_elements}
    assert leg_flags == expected_leg_flags


def test_single_leg_monitoring_keeps_leg_names_and_weight(network_with_leg2_and_leg3_limits: Network) -> None:
    definition = Nminus1Definition(
        contingencies=[BASECASE], monitored_elements=[MONITORED_TRAFO3W.model_copy(update={"weight": 2.0})]
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, network_with_leg2_and_leg3_limits, leg_monitoring=Trafo3wLegMonitoring.SINGLE_LEG
    )

    assert [element.name for element in converted.monitored_elements] == [f"3W 380/110/63-Leg{leg}" for leg in (1, 2, 3)]
    assert {element.weight for element in converted.monitored_elements} == {2.0}


@pytest.mark.parametrize(
    ("leg_monitoring", "expected_leg_ids"),
    [(Trafo3wLegMonitoring.SINGLE_LEG, LEG_IDS), (Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS, ["3W-Leg1"])],
)
def test_lead_leg_falls_back_to_leg1_without_leg_limits(
    complex_grid_network: Network, leg_monitoring: Trafo3wLegMonitoring, expected_leg_ids: list[str]
) -> None:
    """Without any leg limit, Leg1 inherits the optimized 3W and a warning is logged."""
    definition = Nminus1Definition(contingencies=[BASECASE], monitored_elements=[MONITORED_TRAFO3W])

    with structlog.testing.capture_logs() as cap_logs:
        converted = convert_three_winding_transformers_in_nminus1_definition(
            definition, complex_grid_network, leg_monitoring=leg_monitoring
        )

    assert [element.id for element in converted.monitored_elements] == expected_leg_ids
    assert [element.id for element in converted.monitored_elements if element.optimized] == ["3W-Leg1"]
    assert [entry["event"] for entry in cap_logs] == ["three_winding_transformer_legs_without_limit"]


def test_lead_leg_is_the_first_remaining_leg_after_reduction(complex_grid_network: Network) -> None:
    """A leg removed from the grid is skipped, so the first remaining leg is the fallback."""
    complex_grid_network.remove_elements(["3W-Leg1"])
    definition = Nminus1Definition(contingencies=[BASECASE], monitored_elements=[MONITORED_TRAFO3W])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, leg_monitoring=Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS
    )

    assert [element.id for element in converted.monitored_elements] == ["3W-Leg2"]


@pytest.mark.parametrize(
    ("leg_monitoring", "expected_leg_ids"),
    [(Trafo3wLegMonitoring.SINGLE_LEG, LEG_IDS), (Trafo3wLegMonitoring.REMOVE_NON_LEAD_LEGS, ["3W-Leg1", "3W-Leg2"])],
)
def test_explicitly_listed_leg_keeps_its_own_flags(
    network_with_leg2_and_leg3_limits: Network, leg_monitoring: Trafo3wLegMonitoring, expected_leg_ids: list[str]
) -> None:
    """A leg that the definition lists before its 3W keeps its flags, even if it is not the lead leg."""
    leg1 = MonitoredElement(id="3W-Leg1", type="TWO_WINDINGS_TRANSFORMER", kind="branch", optimized=True)
    definition = Nminus1Definition(contingencies=[BASECASE], monitored_elements=[leg1, MONITORED_TRAFO3W])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, network_with_leg2_and_leg3_limits, leg_monitoring=leg_monitoring
    )

    assert [element.id for element in converted.monitored_elements] == expected_leg_ids
    assert converted.monitored_elements[0] == leg1


@pytest.mark.parametrize("leg_monitoring", list(Trafo3wLegMonitoring))
def test_leg_monitoring_does_not_change_contingencies_or_spps(
    network_with_leg2_and_leg3_limits: Network, leg_monitoring: Trafo3wLegMonitoring
) -> None:
    """Outaging a 3W still outages all of its legs, and SPPS rules still cover all legs."""
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_3W", elements=[TRAFO3W])],
        monitored_elements=[MONITORED_TRAFO3W],
        spps_rules=[
            _spps_rule_closing_switches_when_de_energized("C_3W", ["3W"], ["BREAKER_3W_HV"], SppsConditionLogic.ALL)
        ],
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, network_with_leg2_and_leg3_limits, leg_monitoring=leg_monitoring
    )

    assert [element.id for element in converted.contingencies[1].elements] == LEG_IDS
    assert converted.spps_rules is not None
    assert [condition.condition_element_unique_id for condition in converted.spps_rules[0].conditions] == LEG_IDS


@pytest.mark.parametrize("leg_monitoring", list(Trafo3wLegMonitoring))
def test_leg_monitoring_is_idempotent(
    network_with_leg2_and_leg3_limits: Network, leg_monitoring: Trafo3wLegMonitoring
) -> None:
    definition = Nminus1Definition(contingencies=[BASECASE], monitored_elements=[MONITORED_TRAFO3W])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, network_with_leg2_and_leg3_limits, leg_monitoring=leg_monitoring
    )
    converted_twice = convert_three_winding_transformers_in_nminus1_definition(
        converted, network_with_leg2_and_leg3_limits, leg_monitoring=leg_monitoring
    )

    assert converted_twice == converted


def test_get_nminus1_definition_element_ids() -> None:
    definition = Nminus1Definition(
        contingencies=[BASECASE, Contingency(id="C_L8", elements=[LINE_L8])],
        monitored_elements=[MonitoredElement(**LINE_L1.model_dump())],
        spps_rules=[_spps_rule_closing_switches_when_de_energized("C_L8", ["COND"], ["ACT"], SppsConditionLogic.ALL)],
    )

    assert get_nminus1_definition_element_ids(definition) == {"L8", "L1", "COND", "ACT"}


def test_get_voltage_level_ids_of_elements(complex_grid_network: Network) -> None:
    """Branches contribute both sides, legs include the star voltage level, HVDC lines their converter stations."""
    branches = complex_grid_network.get_branches(attributes=["voltage_level1_id", "voltage_level2_id"])
    switches = complex_grid_network.get_switches(attributes=["voltage_level_id"])
    injections = complex_grid_network.get_injections(attributes=["voltage_level_id"])
    hvdc = complex_grid_network.get_hvdc_lines(attributes=["converter_station1_id", "converter_station2_id"]).loc["HVDC_LCC"]
    a_voltage_level = complex_grid_network.get_voltage_levels().index[0]

    voltage_level_ids = get_voltage_level_ids_of_elements(
        complex_grid_network, ["L8", "3W-Leg1", "BREAKER_3W_HV", "HVDC_LCC", a_voltage_level, "UNKNOWN"]
    )

    expected = {
        *branches.loc["L8", ["voltage_level1_id", "voltage_level2_id"]],
        *branches.loc["3W-Leg1", ["voltage_level1_id", "voltage_level2_id"]],
        switches.loc["BREAKER_3W_HV", "voltage_level_id"],
        injections.loc[hvdc.converter_station1_id, "voltage_level_id"],
        injections.loc[hvdc.converter_station2_id, "voltage_level_id"],
        a_voltage_level,
    }
    assert "3W-Star-VL" in expected
    assert voltage_level_ids == sorted(expected)
