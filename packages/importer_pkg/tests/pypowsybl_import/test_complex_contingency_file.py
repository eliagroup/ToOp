import json
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network.impl.network import Network
from toop_engine_importer.pypowsybl_import.contingency_from_file.complex_contingency_file import (
    load_nminus1_definition_from_file,
)


def test_load_complex_contingency_file(complex_grid_network: Network, tmp_path: Path) -> None:
    """Load one grouped contingency and its closed-switch SPPS action."""
    lines = complex_grid_network.get_lines(attributes=["name"])
    switches = complex_grid_network.get_switches(attributes=["name"])
    first_line_id, second_line_id = lines.index[:2]
    switch_id = switches.index[0]

    contingency_file = tmp_path / "contingencies.json"
    contingency_file.write_text(
        json.dumps(
            {
                "150.0": {
                    "Name": "150.0",
                    "FaultCase": "Fault Case Name 1",
                    "InterruptedComponents": [
                        {"Name": lines.loc[first_line_id, "name"], "RdfId": f"_{first_line_id}"},
                        {"Name": lines.loc[second_line_id, "name"], "RdfId": f"_{second_line_id}"},
                    ],
                    "OpenedSwitches": [],
                    "ClosedSwitches": [
                        {"Name": switches.loc[switch_id, "name"], "RdfId": f"_{switch_id}"},
                    ],
                    "OutOfService": 0,
                }
            }
        )
    )

    definition = load_nminus1_definition_from_file(
        network=complex_grid_network,
        file_path=contingency_file,
        filesystem=LocalFileSystem(),
        monitored_elements=[],
    )

    assert [contingency.id for contingency in definition.contingencies] == ["BASECASE", "150.0"]
    contingency = definition.contingencies[1]
    assert contingency.name == "Fault Case Name 1"
    assert [element.id for element in contingency.elements] == [first_line_id, second_line_id]
    assert contingency.is_multi_outage()

    assert definition.spps_rules is not None
    assert len(definition.spps_rules) == 1
    rule = definition.spps_rules[0]
    assert rule.scheme_name == "150.0"
    assert [condition.condition_element_unique_id for condition in rule.conditions] == [first_line_id, second_line_id]
    assert rule.actions[0].measure_element_unique_id == switch_id
    assert rule.actions[0].measure_value == "closed"


def test_load_meaningful_complex_contingency_file(complex_grid_network: Network) -> None:
    """Load the committed complex-grid contingency list and its L8 transfer SPPS."""
    contingency_file = Path(__file__).parents[4] / "data/complex_grid/contingency_list_complex_meaningful.json"

    definition = load_nminus1_definition_from_file(
        network=complex_grid_network,
        file_path=contingency_file,
        filesystem=LocalFileSystem(),
        monitored_elements=[],
    )

    assert [contingency.id for contingency in definition.contingencies] == [
        "BASECASE",
        "C_L_DE_BE_1",
        "C_L_NL_1_2",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W_COMPLETE",
        "C_NL_3W_1_COMPLETE",
        "C_HVDC_LCC",
        "C_MV_COUPLER",
    ]

    l8_contingency = next(
        contingency for contingency in definition.contingencies if contingency.id == "C_L8_WITH_LINE_OUT_OF_SERVICE"
    )
    assert [element.id for element in l8_contingency.elements] == ["L8", "L81_BREAKER", "L82_BREAKER"]

    three_winding_contingency = next(
        contingency for contingency in definition.contingencies if contingency.id == "C_3W_COMPLETE"
    )
    assert [element.id for element in three_winding_contingency.elements[:3]] == [
        "3W-Leg1",
        "3W-Leg2",
        "3W-Leg3",
    ]

    nl_three_winding_contingency = next(
        contingency for contingency in definition.contingencies if contingency.id == "C_NL_3W_1_COMPLETE"
    )
    assert [element.id for element in nl_three_winding_contingency.elements[:3]] == [
        "NL_3W_1-Leg1",
        "NL_3W_1-Leg2",
        "NL_3W_1-Leg3",
    ]

    hvdc_contingency = next(contingency for contingency in definition.contingencies if contingency.id == "C_HVDC_LCC")
    assert [element.id for element in hvdc_contingency.elements] == [
        "HVDC_LCC",
        "LCC1_BREAKER",
        "LCC2_BREAKER",
    ]

    assert definition.spps_rules is not None
    l8_rule = next(rule for rule in definition.spps_rules if rule.scheme_name == "C_L8_WITH_LINE_OUT_OF_SERVICE")
    assert [condition.condition_element_unique_id for condition in l8_rule.conditions] == [
        "L8",
        "L81_BREAKER",
        "L82_BREAKER",
    ]
    assert [condition.condition_limit_value for condition in l8_rule.conditions] == [None, "open", "open"]
    assert [action.measure_element_unique_id for action in l8_rule.actions] == [
        "LINE_out_of_service_BREAKER1",
        "LINE_out_of_service_BREAKER2",
    ]
    assert [action.measure_value for action in l8_rule.actions] == ["closed", "closed"]


def test_duplicate_complex_contingency_id_keeps_first_case_and_warns(
    complex_grid_network: Network, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    line_id = complex_grid_network.get_lines().index[0]
    line_name = complex_grid_network.get_lines().loc[line_id, "name"]
    case = {
        "Name": "DUPLICATE",
        "FaultCase": "first",
        "InterruptedComponents": [{"Name": line_name, "RdfId": line_id}],
        "OpenedSwitches": [],
        "ClosedSwitches": [],
        "OutOfService": 0,
    }
    duplicate = {**case, "FaultCase": "second"}
    contingency_file = tmp_path / "duplicates.json"
    contingency_file.write_text(json.dumps({"first": case, "second": duplicate}))

    definition = load_nminus1_definition_from_file(
        network=complex_grid_network,
        file_path=contingency_file,
        filesystem=LocalFileSystem(),
        monitored_elements=[],
    )

    assert [contingency.name for contingency in definition.contingencies] == ["BASECASE", "first"]
    assert "duplicate_contingency_id" in caplog.text


def test_empty_complex_contingency_is_rejected(complex_grid_network: Network, tmp_path: Path) -> None:
    contingency_file = tmp_path / "empty.json"
    contingency_file.write_text(
        json.dumps(
            {
                "empty": {
                    "Name": "EMPTY",
                    "FaultCase": "empty case",
                    "InterruptedComponents": [],
                    "OpenedSwitches": [],
                    "ClosedSwitches": [],
                    "OutOfService": 0,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="no outage elements"):
        load_nminus1_definition_from_file(
            network=complex_grid_network,
            file_path=contingency_file,
            filesystem=LocalFileSystem(),
            monitored_elements=[],
        )
