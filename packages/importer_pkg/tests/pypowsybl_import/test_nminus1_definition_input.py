# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

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

INPUT_NMINUS1_DEFINITION_FILE = Path(__file__).parents[4] / "data/complex_grid/nminus1_definition_complex.json"
BASECASE = Contingency(id="BASECASE", name="BASECASE", elements=[])


def _line(element_id: str) -> GridElement:
    return GridElement(id=element_id, type="LINE", kind="branch")


def _switch(element_id: str) -> GridElement:
    return GridElement(id=element_id, type="SWITCH", kind="switch")


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


def _closing_rule(scheme_name: str, condition_id: str, action_id: str) -> SppsRule:
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


def _events(cap_logs: list[MutableMapping[str, Any]]) -> list[str]:
    return [entry["event"] for entry in cap_logs]


def test_input_nminus1_definition_example_is_grid_valid(complex_grid_network_unconverted: Network) -> None:
    """The committed input N-1 definition example passes grid validation unchanged and without warnings."""
    with structlog.testing.capture_logs() as cap_logs:
        definition = load_nminus1_definition_for_network(
            network=complex_grid_network_unconverted,
            file_path=INPUT_NMINUS1_DEFINITION_FILE,
            filesystem=LocalFileSystem(),
        )

    assert cap_logs == []
    assert definition == load_nminus1_definition(INPUT_NMINUS1_DEFINITION_FILE)
    assert [contingency.id for contingency in definition.contingencies] == [
        "BASECASE",
        "C_L_DE_BE_1",
        "C_L_NL_1_2",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W",
        "C_NL_3W_1",
        "C_HVDC_LCC",
        "C_MV_COUPLER",
    ]
    assert definition.spps_rules is not None
    assert [rule.scheme_name for rule in definition.spps_rules] == [
        "C_L_DE_BE_1",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W",
    ]


def test_three_winding_transformer_requires_unconverted_grid(
    complex_grid_network_unconverted: Network, complex_grid_network: Network
) -> None:
    """3W transformers keep their original id and are only found before the 3W to 2W conversion."""
    trafo3w = GridElement(id="3W", type="THREE_WINDINGS_TRANSFORMER", kind="branch")
    definition = _definition(
        [Contingency(id="C_3W", elements=[trafo3w])],
        monitored_elements=[MonitoredElement(**trafo3w.model_dump())],
    )

    unconverted = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)
    assert unconverted == definition

    with structlog.testing.capture_logs() as cap_logs:
        converted = filter_nminus1_definition_to_network(definition, complex_grid_network)
    assert [contingency.id for contingency in converted.contingencies] == ["BASECASE"]
    assert converted.monitored_elements == []
    assert _events(cap_logs).count("unknown_nminus1_element_dropped") == 2


def test_unknown_element_is_dropped_from_multi_outage(complex_grid_network_unconverted: Network) -> None:
    """Unknown elements are removed while the rest of the multi-outage is kept."""
    definition = _definition([Contingency(id="multi", elements=[_line("L8"), _line("MISSING"), _switch("L81_BREAKER")])])

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert [element.id for element in filtered.contingencies[1].elements] == ["L8", "L81_BREAKER"]
    [warning] = cap_logs
    assert warning["event"] == "unknown_nminus1_element_dropped"
    assert warning["element_id"] == "MISSING"
    assert warning["contingency_id"] == "multi"


def test_emptied_contingency_is_dropped_and_basecase_kept(complex_grid_network_unconverted: Network) -> None:
    """A contingency whose elements are all unknown is dropped; the empty base case is not."""
    definition = _definition(
        [
            Contingency(id="gone", elements=[_line("MISSING_1"), _line("MISSING_2")]),
            Contingency(id="kept", elements=[_line("L8")]),
        ]
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert [contingency.id for contingency in filtered.contingencies] == ["BASECASE", "kept"]
    assert "empty_nminus1_contingency_dropped" in _events(cap_logs)


def test_type_and_kind_mismatch_is_corrected(complex_grid_network_unconverted: Network) -> None:
    """Elements keep their id but take type and kind from the grid."""
    definition = _definition(
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
    corrections = [entry for entry in cap_logs if entry["event"] == "nminus1_element_type_corrected"]
    assert [(entry["element_id"], entry["declared_type"], entry["actual_type"]) for entry in corrections] == [
        ("L8", "SWITCH", "LINE"),
        ("L81_BREAKER", None, "SWITCH"),
    ]


def test_monitored_elements_are_filtered_and_keep_monitoring_scope(complex_grid_network_unconverted: Network) -> None:
    """Unknown monitored elements are dropped; surviving switches keep their monitoring scope."""
    scope = frozenset({SwitchMonitoringScope.FLOW})
    definition = _definition(
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
    definition = _definition(
        [
            Contingency(id="valid", elements=[_line("L8")]),
            Contingency(id="bad_condition", elements=[_line("L8")]),
            Contingency(id="bad_action", elements=[_line("L8")]),
        ],
        spps_rules=[
            _closing_rule("valid", condition_id="L8", action_id="LINE_out_of_service_BREAKER1"),
            _closing_rule("bad_condition", condition_id="MISSING_CONDITION", action_id="LINE_out_of_service_BREAKER1"),
            _closing_rule("bad_action", condition_id="L8", action_id="MISSING_ACTION"),
        ],
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert filtered.spps_rules is not None
    assert [rule.scheme_name for rule in filtered.spps_rules] == ["valid"]
    assert [entry["missing_ids"] for entry in cap_logs if entry["event"] == "spps_rule_dropped"] == [
        ["MISSING_CONDITION"],
        ["MISSING_ACTION"],
    ]


def test_spps_rule_of_dropped_contingency_is_dropped(complex_grid_network_unconverted: Network) -> None:
    """Rules follow their contingency, keeping the SPPS integrity validator satisfied."""
    definition = _definition(
        [Contingency(id="gone", elements=[_line("MISSING")])],
        spps_rules=[_closing_rule("gone", condition_id="L8", action_id="LINE_out_of_service_BREAKER1")],
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert filtered.spps_rules is None
    assert [entry.get("reason") for entry in cap_logs if entry["event"] == "spps_rule_dropped"] == ["contingency_dropped"]


def test_hvdc_and_tie_line_elements_are_kept(complex_grid_network_unconverted: Network) -> None:
    """HVDC lines and tie lines are part of the grid inventory."""
    tie_line_id = complex_grid_network_unconverted.get_tie_lines().index[0]
    definition = _definition(
        [
            Contingency(id="hvdc", elements=[GridElement(id="HVDC_LCC", type="HVDC_LINE", kind="branch")]),
            Contingency(id="tie", elements=[GridElement(id=tie_line_id, type="TIE_LINE", kind="branch")]),
        ]
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert filtered == definition
    assert cap_logs == []
