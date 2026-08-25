import json
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network import Network
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
