# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
from pathlib import Path

import structlog.testing
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file import (
    filter_nminus1_definition_to_network,
    load_nminus1_definition_for_network,
)
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SppsRule,
    SwitchMonitoringScope,
    load_nminus1_definition,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionType,
    SppsMeasureType,
    SppsSwitchActionTarget,
)


def _line_element(element_id: str) -> GridElement:
    return GridElement(id=element_id, type="LINE", kind="branch")


def _definition_with_basecase(
    contingencies: list[Contingency],
    monitored_elements: list[MonitoredElement] | None = None,
    spps_rules: list[SppsRule] | None = None,
) -> Nminus1Definition:
    """A definition with a base case in front of the given contingencies."""
    return Nminus1Definition(
        contingencies=[Contingency(id="BASECASE", name="BASECASE", elements=[]), *contingencies],
        monitored_elements=monitored_elements or [],
        spps_rules=spps_rules,
        id_type="powsybl",
    )


def _switch_closing_spps_rule(
    scheme_name: str, condition_id: str, action_id: str = "LINE_out_of_service_BREAKER1"
) -> SppsRule:
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
                measure_element_unique_id=action_id,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
        ],
    )


def test_input_nminus1_definition_example_is_grid_valid(
    complex_grid_network_unconverted: Network, input_nminus1_definition_file: Path
) -> None:
    """The input N-1 definition example passes grid validation unchanged and without warnings."""
    with structlog.testing.capture_logs() as cap_logs:
        definition = load_nminus1_definition_for_network(
            network=complex_grid_network_unconverted,
            file_path=input_nminus1_definition_file,
            filesystem=LocalFileSystem(),
        )

    assert cap_logs == []
    assert definition == load_nminus1_definition(input_nminus1_definition_file)
    assert definition.spps_rules


def test_three_winding_transformer_requires_unconverted_grid(
    complex_grid_network_unconverted: Network, complex_grid_network: Network
) -> None:
    """3W transformers keep their original id and are only found before the 3W to 2W conversion."""
    trafo3w = GridElement(id="3W", type="THREE_WINDINGS_TRANSFORMER", kind="branch")
    definition = _definition_with_basecase(
        [Contingency(id="C_3W", elements=[trafo3w])], monitored_elements=[MonitoredElement(**trafo3w.model_dump())]
    )

    assert filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted) == definition

    with structlog.testing.capture_logs() as cap_logs:
        converted = filter_nminus1_definition_to_network(definition, complex_grid_network)
    assert [contingency.id for contingency in converted.contingencies] == ["BASECASE"]
    assert converted.monitored_elements == []
    assert [entry["event"] for entry in cap_logs].count("unknown_nminus1_element_dropped") == 2


def test_unknown_elements_are_dropped_from_contingencies(complex_grid_network_unconverted: Network) -> None:
    """Unknown elements are removed from multi-outages; emptied contingencies and their SPPS rules are dropped."""
    definition = _definition_with_basecase(
        [
            Contingency(id="multi", elements=[_line_element("L8"), _line_element("MISSING"), _line_element("L1")]),
            Contingency(id="gone", elements=[_line_element("MISSING_1"), _line_element("MISSING_2")]),
        ],
        spps_rules=[_switch_closing_spps_rule("gone", condition_id="L8")],
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert [contingency.id for contingency in filtered.contingencies] == ["BASECASE", "multi"]
    assert [element.id for element in filtered.contingencies[1].elements] == ["L8", "L1"]
    assert filtered.spps_rules is None
    assert [(entry["event"], entry.get("element_id"), entry.get("reason")) for entry in cap_logs] == [
        ("unknown_nminus1_element_dropped", "MISSING", None),
        ("unknown_nminus1_element_dropped", "MISSING_1", None),
        ("unknown_nminus1_element_dropped", "MISSING_2", None),
        ("empty_nminus1_contingency_dropped", None, None),
        ("spps_rule_dropped", None, "contingency_dropped"),
    ]


def test_type_and_kind_mismatch_is_corrected(complex_grid_network_unconverted: Network) -> None:
    """Elements keep their id but take type and kind from the grid."""
    definition = _definition_with_basecase(
        [
            Contingency(
                id="mismatch",
                elements=[
                    GridElement(id="L8", name="line L8", type="SWITCH", kind="switch"),
                    GridElement(id="L81_BREAKER", type=None, kind="switch"),
                ],
            )
        ]
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    line, switch = filtered.contingencies[1].elements
    assert (line.id, line.name, line.type, line.kind) == ("L8", "line L8", "LINE", "branch")
    assert (switch.type, switch.kind) == ("SWITCH", "switch")
    assert [(entry["element_id"], entry["declared_type"], entry["actual_type"]) for entry in cap_logs] == [
        ("L8", "SWITCH", "LINE"),
        ("L81_BREAKER", None, "SWITCH"),
    ]


def test_monitored_elements_are_filtered_and_keep_monitoring_scope(complex_grid_network_unconverted: Network) -> None:
    """Unknown monitored elements are dropped; surviving switches keep their monitoring scope."""
    scope = frozenset({SwitchMonitoringScope.FLOW})
    definition = _definition_with_basecase(
        [],
        monitored_elements=[
            MonitoredElement(id="L81_BREAKER", type="SWITCH", kind="switch", monitoring_scope=scope),
            MonitoredElement(id="MISSING", type="LINE", kind="branch"),
        ],
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    [monitored] = filtered.monitored_elements
    assert isinstance(monitored, MonitoredElement)
    assert (monitored.id, monitored.monitoring_scope) == ("L81_BREAKER", scope)
    assert [(entry["event"], entry["context"]) for entry in cap_logs] == [
        ("unknown_nminus1_element_dropped", "monitored_element")
    ]


def test_spps_rule_with_unknown_elements_is_dropped(complex_grid_network_unconverted: Network) -> None:
    """A rule is dropped as a whole if a condition or an action references an unknown element."""
    definition = _definition_with_basecase(
        [
            Contingency(id=scheme_name, elements=[_line_element("L8")])
            for scheme_name in ("valid", "bad_condition", "bad_action")
        ],
        spps_rules=[
            _switch_closing_spps_rule("valid", condition_id="L8"),
            _switch_closing_spps_rule("bad_condition", condition_id="MISSING_CONDITION"),
            _switch_closing_spps_rule("bad_action", condition_id="L8", action_id="MISSING_ACTION"),
        ],
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert [rule.scheme_name for rule in filtered.spps_rules] == ["valid"]
    assert [entry["missing_ids"] for entry in cap_logs] == [["MISSING_CONDITION"], ["MISSING_ACTION"]]
