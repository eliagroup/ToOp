# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
from pathlib import Path

import pypowsybl
import pytest
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
    save_nminus1_definition,
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
    trafo3w = GridElement(id="3W", name="3W 380/110/63", type="THREE_WINDINGS_TRANSFORMER", kind="branch")
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


@pytest.mark.parametrize(
    ("grid_file_fixture", "line_id", "line_name", "switch_id", "switch_name"),
    [
        ("ucte_file", "D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 1: Test Line", "D8SU1_12 D8SU1_11 1", "Test Coupler"),
        (
            "test_pypowsybl_cgmes_with_3w_trafo",
            "ffbabc27-1ccd-4fdc-b037-e341706c8d29",
            "BE-Line_6",
            "6b564930-b5e2-49d3-9d06-e1de28d6fd65",
            "BE_Breaker_2",
        ),
    ],
    ids=["ucte", "cgmes"],
)
def test_element_names_are_taken_from_the_grid(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    grid_file_fixture: str,
    line_id: str,
    line_name: str,
    switch_id: str,
    switch_name: str,
) -> None:
    """Element names and single-outage contingency names follow the grid, named as by the DC solver preprocessing."""
    definition = _definition_with_basecase(
        [
            Contingency(
                id="C_LINE",
                name="stale_contingency:line",
                elements=[GridElement(id=line_id, name="stale name", type="LINE", kind="branch")],
            )
        ],
        monitored_elements=[
            MonitoredElement(id=line_id, name="stale name", type="LINE", kind="branch"),
            MonitoredElement(id=switch_id, type="SWITCH", kind="switch"),
        ],
    )
    file_path = tmp_path / "nminus1_definition.json"
    save_nminus1_definition(file_path, definition)

    with structlog.testing.capture_logs() as cap_logs:
        loaded = load_nminus1_definition_for_network(
            network=pypowsybl.network.load(request.getfixturevalue(grid_file_fixture)),
            file_path=file_path,
            filesystem=LocalFileSystem(),
        )

    assert [element.name for element in loaded.contingencies[1].elements] == [line_name]
    assert loaded.contingencies[1].name == line_name
    assert [element.name for element in loaded.monitored_elements] == [line_name, switch_name]
    assert [(entry["event"], entry["context"], entry["declared_name"], entry["actual_name"]) for entry in cap_logs] == [
        ("nminus1_element_name_corrected", "contingency", "stale name", line_name),
        ("nminus1_contingency_name_corrected", "contingency", "stale_contingency:line", line_name),
        ("nminus1_element_name_corrected", "monitored_element", "stale name", line_name),
        ("nminus1_element_name_corrected", "monitored_element", "", switch_name),
    ]


def test_element_name_is_kept_if_the_grid_has_none(complex_grid_network_unconverted: Network) -> None:
    """Elements without a name in the grid keep the name of the input N-1 definition."""
    definition = _definition_with_basecase(
        [], monitored_elements=[MonitoredElement(id="L8", name="line L8", type="LINE", kind="branch")]
    )

    with structlog.testing.capture_logs() as cap_logs:
        filtered = filter_nminus1_definition_to_network(definition, complex_grid_network_unconverted)

    assert filtered == definition
    assert cap_logs == []


def test_ucte_names_match_the_dc_solver(ucte_file: Path) -> None:
    """UCTE transformers are named ``<id>: <elementName>``, tie lines by their id; multi-outages keep their name."""
    trafo_id, tie_line_id = "B_SU1_11 B_SU1_21 1", "XB__F_21 B_SU1_21 1 + XB__F_21 D8SU1_21 1"
    trafo = GridElement(id=trafo_id, type="TWO_WINDINGS_TRANSFORMER", kind="branch")
    tie_line = GridElement(id=tie_line_id, type="TIE_LINE", kind="branch")
    definition = _definition_with_basecase(
        [
            Contingency(id="C_TRAFO", name="B_SU1_1_B_SU1_2:trf", elements=[trafo]),
            Contingency(id="C_MULTI", name="trafo and tie line", elements=[trafo, tie_line]),
        ]
    )

    filtered = filter_nminus1_definition_to_network(definition, pypowsybl.network.load(ucte_file))

    assert [
        (contingency.name, [element.name for element in contingency.elements]) for contingency in filtered.contingencies[1:]
    ] == [
        ("B_SU1_11 B_SU1_21 1: Test 2WT 2", ["B_SU1_11 B_SU1_21 1: Test 2WT 2"]),
        ("trafo and tie line", ["B_SU1_11 B_SU1_21 1: Test 2WT 2", tie_line_id]),
    ]


CGMES_LINE, CGMES_TRAFO3W = "ffbabc27-1ccd-4fdc-b037-e341706c8d29", "84ed55f4-61f5-4d9d-8755-bba7b877a246"
UCTE_LINE, UCTE_OTHER_LINE, UCTE_LOAD = "D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2", "B_SU1_11_load"


def test_contingency_with_three_winding_transformer_keeps_its_name(test_pypowsybl_cgmes_with_3w_trafo: Path) -> None:
    """A 3W transformer becomes three DC branches, so a contingency with one is a multi-outage and keeps its name."""
    elements = [
        GridElement(id=CGMES_LINE, type="LINE", kind="branch"),
        GridElement(id=CGMES_TRAFO3W, type="THREE_WINDINGS_TRANSFORMER", kind="branch"),
    ]
    definition = _definition_with_basecase([Contingency(id="C", name="input name", elements=elements)])

    filtered = filter_nminus1_definition_to_network(definition, pypowsybl.network.load(test_pypowsybl_cgmes_with_3w_trafo))

    assert filtered.contingencies[1].name == "input name"


@pytest.mark.parametrize(
    ("elements", "expected_name"),
    [
        pytest.param(
            [
                GridElement(id=UCTE_LINE, type="LINE", kind="branch"),
                GridElement(id=UCTE_LOAD, type="LOAD", kind="injection"),
            ],
            "input name",
            id="disconnected_line_and_unnamed_load_is_a_load_outage",
        ),
        pytest.param(
            [
                GridElement(id=UCTE_LINE, type="LINE", kind="branch"),
                GridElement(id=UCTE_OTHER_LINE, type="LINE", kind="branch"),
            ],
            "D8SU1_12 D7SU2_11 2: Test Line 2",
            id="disconnected_and_connected_line_is_a_single_outage",
        ),
    ],
)
def test_disconnected_branches_do_not_count_for_the_single_outage_name(
    ucte_file: Path, elements: list[GridElement], expected_name: str
) -> None:
    """As in the DC solver, a branch that is not connected at both sides is not outaged as a DC branch."""
    network = pypowsybl.network.load(ucte_file)
    network.update_lines(id=UCTE_LINE, connected1=False, connected2=False)
    definition = _definition_with_basecase([Contingency(id="C", name="input name", elements=elements)])

    filtered = filter_nminus1_definition_to_network(definition, network)

    assert filtered.contingencies[1].name == expected_name


def _legs(trafo3w_id: str, legs: str = "123") -> list[GridElement]:
    return [GridElement(id=f"{trafo3w_id}-Leg{leg}", type="TWO_WINDINGS_TRANSFORMER", kind="branch") for leg in legs]


@pytest.mark.parametrize(
    ("contingency", "expected_name"),
    [
        pytest.param(Contingency(id="C", name="C", elements=_legs(CGMES_TRAFO3W)), "BE-TR3_1", id="three_legs_named_by_id"),
        pytest.param(
            Contingency(id="C", name="curated", elements=_legs(CGMES_TRAFO3W)), "curated", id="three_legs_curated_name"
        ),
        pytest.param(Contingency(id="C", name="C", elements=_legs(CGMES_TRAFO3W, "12")), "C", id="two_legs_are_no_trafo3w"),
        pytest.param(
            Contingency(
                id="C", name="C", elements=[*_legs(CGMES_TRAFO3W), GridElement(id=CGMES_LINE, type="LINE", kind="branch")]
            ),
            "C",
            id="three_legs_and_a_line",
        ),
    ],
)
def test_contingency_of_three_legs_named_by_id_takes_the_trafo3w_name(
    test_pypowsybl_cgmes_with_3w_trafo: Path, contingency: Contingency, expected_name: str
) -> None:
    """Only a contingency named by its id that outages exactly the three legs of one 3W takes the 3W name."""
    network = pypowsybl.network.load(test_pypowsybl_cgmes_with_3w_trafo)
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(network)

    filtered = filter_nminus1_definition_to_network(_definition_with_basecase([contingency]), network)

    assert filtered.contingencies[1].name == expected_name


@pytest.mark.parametrize(
    ("contingency_name", "expected_name"), [("C_L8", "line L8"), ("curated", "curated")], ids=["named_by_id", "curated"]
)
def test_single_outage_named_by_id_takes_the_element_name_without_grid_name(
    complex_grid_network_unconverted: Network, contingency_name: str, expected_name: str
) -> None:
    """L8 has no name in the grid, so only a contingency named by its id takes the element name of the definition."""
    contingency = Contingency(
        id="C_L8", name=contingency_name, elements=[GridElement(id="L8", name="line L8", type="LINE", kind="branch")]
    )

    filtered = filter_nminus1_definition_to_network(
        _definition_with_basecase([contingency]), complex_grid_network_unconverted
    )

    assert filtered.contingencies[1].name == expected_name
