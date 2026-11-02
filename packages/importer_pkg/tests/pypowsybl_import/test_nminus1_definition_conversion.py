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
LEG_IDS = ["3W-Leg1", "3W-Leg2", "3W-Leg3"]
BASECASE = Contingency(id="BASECASE", name="BASECASE", elements=[])

HIDDEN_NON_WORSENING = (False, True, False)
UNFLAGGED = (False, False, False)


def _line(element_id: str) -> GridElement:
    return GridElement(id=element_id, type="LINE", kind="branch")


def _definition(
    contingencies: list[Contingency],
    monitored_elements: list[MonitoredElement] | None = None,
    spps_rules: list[SppsRule] | None = None,
) -> Nminus1Definition:
    return Nminus1Definition(
        contingencies=[BASECASE, *contingencies],
        monitored_elements=monitored_elements or [],
        spps_rules=spps_rules,
        id_type="powsybl",
    )


def _rule(scheme_name: str, condition_ids: list[str], action_ids: list[str], logic: SppsConditionLogic) -> SppsRule:
    return SppsRule(
        scheme_name=scheme_name,
        condition_logic=logic,
        conditions=[
            Condition(
                condition_type=SppsConditionType.STATE,
                condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
                condition_element_unique_id=condition_id,
            )
            for condition_id in condition_ids
        ],
        actions=[
            Action(
                measure_element_unique_id=action_id,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
            for action_id in action_ids
        ],
    )


def test_three_winding_transformer_elements_become_legs(complex_grid_network: Network) -> None:
    """A 3W outage or monitored 3W is replaced in place by its three legs, carrying the leg names and element class."""
    definition = _definition(
        [Contingency(id="C_3W", elements=[_line("L8"), TRAFO3W, _line("L1")])],
        monitored_elements=[MonitoredElement(**TRAFO3W.model_dump()), MonitoredElement(**_line("L8").model_dump())],
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
    definition = _definition(
        [Contingency(id="C_3W", elements=[TRAFO3W])],
        spps_rules=[_rule("C_3W", condition_ids=["L8", "3W"], action_ids=["3W", "BREAKER_3W_HV"], logic=logic)],
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
    definition = _definition([Contingency(id="C_3W", elements=[leg1, TRAFO3W])])

    converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)
    converted_twice = convert_three_winding_transformers_in_nminus1_definition(converted, complex_grid_network)

    assert [element.id for element in converted.contingencies[1].elements] == LEG_IDS
    assert converted_twice == converted


def test_three_winding_transformer_without_legs_is_kept_and_warned(complex_grid_network_unconverted: Network) -> None:
    """On a grid without converted legs, the 3W reference is left for the final grid check to handle."""
    definition = _definition([Contingency(id="C_3W", elements=[TRAFO3W])])

    with structlog.testing.capture_logs() as cap_logs:
        converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network_unconverted)

    assert converted == definition
    assert [entry["event"] for entry in cap_logs] == ["three_winding_transformer_legs_missing"]


def test_definition_without_three_winding_transformers_is_unchanged(complex_grid_network: Network) -> None:
    definition = _definition(
        [Contingency(id="C_L8", elements=[_line("L8")])],
        monitored_elements=[MonitoredElement(id="L1", type="LINE", kind="branch")],
        spps_rules=[_rule("C_L8", ["L8"], ["BREAKER_3W_HV"], SppsConditionLogic.ALL)],
    )

    assert convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network) == definition


def _add_leg_limits(network: Network, leg_ids: list[str]) -> None:
    network.create_operational_limits(
        element_id=leg_ids,
        element_type=["TWO_WINDINGS_TRANSFORMER"] * len(leg_ids),
        side=["ONE"] * len(leg_ids),
        name=["permanent_limit"] * len(leg_ids),
        type=["CURRENT"] * len(leg_ids),
        value=[1000.0] * len(leg_ids),
        acceptable_duration=[-1] * len(leg_ids),
    )


def _monitoring_flags(definition: Nminus1Definition) -> dict[str, tuple[bool, bool, bool]]:
    """(optimized, non_worsening, displayed) per monitored element id."""
    return {
        element.id: (element.optimized, element.non_worsening, element.displayed)
        for element in definition.monitored_elements
    }


def _monitored_trafo3w(**flags: bool) -> MonitoredElement:
    return MonitoredElement(**TRAFO3W.model_dump(), **flags)


def test_single_leg_monitoring_optimized_trafo3w_is_inherited_by_first_leg_with_limit(
    complex_grid_network: Network,
) -> None:
    """Only the first leg with a limit is optimized and displayed, the others are hidden and non-worsening."""
    _add_leg_limits(complex_grid_network, ["3W-Leg3", "3W-Leg2"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=True).model_copy(update={"weight": 2.0})])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": HIDDEN_NON_WORSENING,
        "3W-Leg2": (True, False, True),
        "3W-Leg3": HIDDEN_NON_WORSENING,
    }
    assert [element.name for element in converted.monitored_elements] == [f"3W 380/110/63-Leg{leg}" for leg in (1, 2, 3)]
    assert {element.weight for element in converted.monitored_elements} == {2.0}


@pytest.mark.parametrize("optimized", [False, True])
def test_single_leg_monitoring_non_worsening_trafo3w_is_only_kept_on_first_leg_with_limit(
    complex_grid_network: Network, optimized: bool
) -> None:
    """A non-worsening 3W, optimized or not, is inherited by the first leg with a limit; the others get no flags."""
    _add_leg_limits(complex_grid_network, ["3W-Leg1"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=optimized, non_worsening=True)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": (optimized, True, True),
        "3W-Leg2": UNFLAGGED,
        "3W-Leg3": UNFLAGGED,
    }


def test_single_leg_monitoring_plain_monitored_trafo3w_only_hides_other_legs(complex_grid_network: Network) -> None:
    """A 3W that is neither optimized nor non-worsening stays so on all legs; the legs without priority are hidden."""
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=False, non_worsening=False)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": (False, False, False),
        "3W-Leg2": (False, False, True),
        "3W-Leg3": (False, False, False),
    }


def test_single_leg_monitoring_keeps_hidden_trafo3w_hidden(complex_grid_network: Network) -> None:
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=True, displayed=False)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": HIDDEN_NON_WORSENING,
        "3W-Leg2": (True, False, False),
        "3W-Leg3": HIDDEN_NON_WORSENING,
    }


def test_single_leg_monitoring_without_leg_limits_falls_back_to_first_leg(complex_grid_network: Network) -> None:
    """Without any leg limit, the first leg inherits the 3W and a warning is logged."""
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=True)])

    with structlog.testing.capture_logs() as cap_logs:
        converted = convert_three_winding_transformers_in_nminus1_definition(
            definition, complex_grid_network, single_leg_monitoring=True
        )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": (True, False, True),
        "3W-Leg2": HIDDEN_NON_WORSENING,
        "3W-Leg3": HIDDEN_NON_WORSENING,
    }
    assert [entry["event"] for entry in cap_logs] == ["three_winding_transformer_legs_without_limit"]


def test_single_leg_monitoring_uses_remaining_legs_after_reduction(complex_grid_network: Network) -> None:
    """A leg removed from the grid is skipped, the first remaining leg is the fallback."""
    complex_grid_network.remove_elements(["3W-Leg1"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=True)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {"3W-Leg2": (True, False, True), "3W-Leg3": HIDDEN_NON_WORSENING}


def test_single_leg_monitoring_keeps_explicitly_listed_leg(complex_grid_network: Network) -> None:
    """A leg that the definition lists itself, before its 3W, keeps its own flags."""
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    leg1 = MonitoredElement(id="3W-Leg1", type="TWO_WINDINGS_TRANSFORMER", kind="branch", optimized=True)
    definition = _definition([], monitored_elements=[leg1, _monitored_trafo3w(optimized=True)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert _monitoring_flags(converted) == {
        "3W-Leg1": (True, False, True),
        "3W-Leg2": (True, False, True),
        "3W-Leg3": HIDDEN_NON_WORSENING,
    }


def test_single_leg_monitoring_does_not_change_contingencies_or_spps(complex_grid_network: Network) -> None:
    """Outaging a 3W still outages all of its legs, and SPPS rules still cover all legs."""
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    definition = _definition(
        [Contingency(id="C_3W", elements=[TRAFO3W])],
        monitored_elements=[_monitored_trafo3w(optimized=True)],
        spps_rules=[_rule("C_3W", condition_ids=["3W"], action_ids=["3W"], logic=SppsConditionLogic.ALL)],
    )

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )
    converted_all_legs = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)

    assert converted.contingencies == converted_all_legs.contingencies
    assert converted.spps_rules == converted_all_legs.spps_rules
    assert [element.id for element in converted.contingencies[1].elements] == LEG_IDS


def test_single_leg_monitoring_is_idempotent(complex_grid_network: Network) -> None:
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=True)])

    converted = convert_three_winding_transformers_in_nminus1_definition(
        definition, complex_grid_network, single_leg_monitoring=True
    )

    assert (
        convert_three_winding_transformers_in_nminus1_definition(converted, complex_grid_network, single_leg_monitoring=True)
        == converted
    )


def test_without_single_leg_monitoring_all_legs_inherit_the_trafo3w(complex_grid_network: Network) -> None:
    """The default: every leg copies the 3W's monitoring flags, whichever legs have limits."""
    _add_leg_limits(complex_grid_network, ["3W-Leg2"])
    definition = _definition([], monitored_elements=[_monitored_trafo3w(optimized=False, non_worsening=True)])

    converted = convert_three_winding_transformers_in_nminus1_definition(definition, complex_grid_network)

    assert _monitoring_flags(converted) == dict.fromkeys(LEG_IDS, (False, True, True))


def test_get_nminus1_definition_element_ids() -> None:
    definition = _definition(
        [Contingency(id="C_L8", elements=[_line("L8")])],
        monitored_elements=[MonitoredElement(id="L1", type="LINE", kind="branch")],
        spps_rules=[_rule("C_L8", ["COND"], ["ACT"], SppsConditionLogic.ALL)],
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
