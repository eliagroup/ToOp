# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import pandas as pd
import pypowsybl
from pydantic import BaseModel, ConfigDict
from pypowsybl.security import Parameters as SecurityParameters
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.electrical_circuit_groups import (
    identify_circuit_groups,
)
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo


class SecurityAnalysisTestContext(BaseModel):
    """Prepared inputs and reference results for circuit-group security-analysis tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    identified_circuit_groups: object
    monitored_branches: list[str]
    outage_busbar_sections: list[str]
    result_powsybl: object


def _3_windings_transformer_outage_mapping(net: pypowsybl.network.Network) -> dict[str, list[str]]:
    """Map busbar-section outages to converted three-winding transformer legs."""
    t3 = (
        net.get_3_windings_transformers(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "bus_breaker_bus3_id"])
        .reset_index()
        .rename(columns={"id": "3_windings_transformer_id"})
    )

    def convert_to_powsybl_3w_convention(transformer_id: str) -> list[str]:
        return [f"{transformer_id}-Leg1", f"{transformer_id}-Leg2", f"{transformer_id}-Leg3"]

    busbar = (
        net.get_busbar_sections(attributes=["bus_breaker_bus_id"]).reset_index().rename(columns={"id": "busbar_section_id"})
    )
    busbar = busbar.merge(
        t3[["bus_breaker_bus1_id", "3_windings_transformer_id"]],
        left_on="bus_breaker_bus_id",
        right_on="bus_breaker_bus1_id",
        how="left",
    )
    busbar = busbar.merge(
        t3[["bus_breaker_bus2_id", "3_windings_transformer_id"]],
        left_on="bus_breaker_bus_id",
        right_on="bus_breaker_bus2_id",
        how="left",
        suffixes=("_bus1", "_bus2"),
    )
    busbar = busbar.merge(
        t3[["bus_breaker_bus3_id", "3_windings_transformer_id"]],
        left_on="bus_breaker_bus_id",
        right_on="bus_breaker_bus3_id",
        how="left",
    ).rename(columns={"3_windings_transformer_id": "3_windings_transformer_id_bus3"})

    three_windings_transformer_outage_map: dict[str, list[str]] = {}
    for _, row in busbar.iterrows():
        outage_list: list[str] = []
        if pd.notna(row["3_windings_transformer_id_bus1"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus1"]))
        if pd.notna(row["3_windings_transformer_id_bus2"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus2"]))
        if pd.notna(row["3_windings_transformer_id_bus3"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus3"]))
        if outage_list:
            three_windings_transformer_outage_map[row["busbar_section_id"]] = outage_list

    return three_windings_transformer_outage_map


def _build_security_analysis_test_context(
    add_3_windings_transformer_outage: bool = True, run_ac: bool = False
) -> tuple[pypowsybl.network.Network, SecurityAnalysisTestContext]:
    """Build the shared network and reference security-analysis results for circuit-group tests."""
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    three_windings_transformer_outage_map = _3_windings_transformer_outage_mapping(net)
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)
    pypowsybl.loadflow.run_ac(net)

    replace_switches = [
        "L81_BREAKER",
        "L91_BREAKER",
        "2W_MV_HV_12_BREAKER",
        "2W_MV_HV_22_BREAKER",
    ]
    switches = net.get_switches(attributes=["kind", "voltage_level_id", "node1", "node2"]).loc[replace_switches]
    net.remove_elements(replace_switches)
    switches["kind"] = "DISCONNECTOR"
    net.create_switches(switches)

    identified_circuit_groups = identify_circuit_groups(net)
    monitored_branches = net.get_branches().index.to_list()
    outage_busbar_sections = net.get_busbar_sections().index.to_list()

    sa_parameter_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "true"})
    security_analysis = pypowsybl.security.create_analysis()
    for busbar_section_id in outage_busbar_sections:
        if add_3_windings_transformer_outage and (busbar_section_id in three_windings_transformer_outage_map):
            outage_elements = list(three_windings_transformer_outage_map[busbar_section_id])
            outage_elements.append(busbar_section_id)
            security_analysis.add_multiple_elements_contingency(outage_elements, busbar_section_id)
        else:
            security_analysis.add_multiple_elements_contingency([busbar_section_id], busbar_section_id)
    security_analysis.add_monitored_elements(branch_ids=monitored_branches)
    if run_ac:
        result_powsybl = security_analysis.run_ac(net, parameters=sa_parameter_propagation)
    else:
        result_powsybl = security_analysis.run_dc(net, parameters=sa_parameter_propagation)

    return net, SecurityAnalysisTestContext(
        identified_circuit_groups=identified_circuit_groups,
        monitored_branches=monitored_branches,
        outage_busbar_sections=outage_busbar_sections,
        result_powsybl=result_powsybl,
    )
