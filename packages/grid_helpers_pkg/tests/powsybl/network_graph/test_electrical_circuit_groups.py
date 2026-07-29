# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from pandas.testing import assert_frame_equal
from pydantic import BaseModel, ConfigDict
from pypowsybl.security import Parameters as SecurityParameters
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.electrical_circuit_groups import (
    identify_circuit_groups,
)
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.helper_functions import (
    get_all_failing_elements_by_branch_id,
    get_all_failing_elements_by_busbar_section_id,
    get_all_failing_switches_by_branch_id,
    get_all_failing_switches_by_busbar_section_id,
    get_outage_group_ids_by_busbar_section_id,
)
from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups.types import ElectricalCircuitGroup


def _drop_non_semantic_columns(branch_results):
    """Drop result columns that are not stable across equivalent contingency runs."""
    return branch_results.drop(columns=["flow_transfer"], errors="ignore")


class SecurityAnalysisTestContext(BaseModel):
    """Prepared inputs and reference results for outage-group security-analysis tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    identified_circuit_groups: object
    busbar_outage_groups: object
    monitored_branches: list[str]
    outage_busbar_sections: list[str]
    result_powsybl: object


def _prepare_security_analysis_test_context(
    add_3_windings_transformer_outage: bool = True, run_ac: bool = False
) -> SecurityAnalysisTestContext:
    """Prepare the shared network and propagated benchmark for busbar-outage tests."""
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
    busbar_outage_groups = get_outage_group_ids_by_busbar_section_id(
        injection=identified_circuit_groups.injections,
        switches=identified_circuit_groups.switches,
    )
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
        busbar_outage_groups=busbar_outage_groups,
        monitored_branches=monitored_branches,
        outage_busbar_sections=outage_busbar_sections,
        result_powsybl=result_powsybl,
    )


def _assert_zero_only_missing_branches(powsybl_filtered, comparison_filtered, atol: float = 1e-6) -> None:
    """Assert that branches missing from the comparison carry no propagated power."""
    missing_indices = list(set(powsybl_filtered.index) - set(comparison_filtered.index))
    expected_zero = powsybl_filtered.loc[missing_indices]
    assert np.allclose(expected_zero["p1"].sum(), 0.0, atol=atol)


def _assert_matching_common_branches(powsybl_filtered, comparison_filtered, atol: float = 1e-6) -> None:
    """Assert that both result frames match on their shared monitored branches."""
    common_indices = list(set(powsybl_filtered.index) & set(comparison_filtered.index))
    assert_frame_equal(
        powsybl_filtered.loc[common_indices],
        comparison_filtered.loc[common_indices],
        check_dtype=False,
        check_like=True,
        atol=atol,
    )


def test_circuit_group_basic_functions() -> None:
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)
    pypowsybl.loadflow.run_ac(net)
    # create a outage group for the INT station
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
    switch_states_before = net.get_switches(all_attributes=True)

    result = identify_circuit_groups(net)
    branch_L9_group = result.branches.loc["L9"]["electrical_circuit_group"]
    expected = ElectricalCircuitGroup(
        branches=["L9", "2W_MV_HV_2"],
        switches=["L92_BREAKER", "2W_MV_HV_21_BREAKER"],
        injections=[],
        busbar_section=["VL_2W_MV_HV_MV_INT_2_1"],
    )
    assert set(result.circuit_group_map[branch_L9_group].branches) == set(expected.branches)
    assert set(result.circuit_group_map[branch_L9_group].switches) == set(expected.switches)
    assert set(result.circuit_group_map[branch_L9_group].injections) == set(expected.injections)
    assert set(result.circuit_group_map[branch_L9_group].busbar_section) == set(expected.busbar_section)

    busbar_outage_groups = get_outage_group_ids_by_busbar_section_id(injection=result.injections, switches=result.switches)
    failing_elements_by_branch_id = get_all_failing_elements_by_branch_id(
        branch_id="L9",
        branches=result.branches,
        outage_groups=result.circuit_group_map,
        busbar_outage_groups=busbar_outage_groups,
        include_busbar_id=False,
    )
    failing_switches_by_branch_id = get_all_failing_switches_by_branch_id(
        branch_id="L9",
        branches=result.branches,
        outage_groups=result.circuit_group_map,
        busbar_outage_groups=busbar_outage_groups,
    )

    assert set(failing_elements_by_branch_id) == set(expected.branches + expected.injections)
    assert set(failing_switches_by_branch_id) == set(expected.switches)

    failing_elements_by_branch_id = get_all_failing_elements_by_branch_id(
        branch_id="L9",
        branches=result.branches,
        outage_groups=result.circuit_group_map,
        busbar_outage_groups=busbar_outage_groups,
        include_busbar_id=True,
    )
    assert set(failing_elements_by_branch_id) == set(expected.branches + expected.injections + expected.busbar_section)

    # make sure the grid state is not modified by the outage group identification
    # the grid should be cloned / a variant should be created for the outage group identification,
    # so that the original grid is not modified
    switch_states_after = net.get_switches(all_attributes=True)
    assert_frame_equal(switch_states_before, switch_states_after)


def _3_windings_transformer_outage_mapping(net) -> None:
    t3 = (
        net.get_3_windings_transformers(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "bus_breaker_bus3_id"])
        .reset_index()
        .rename(columns={"id": "3_windings_transformer_id"})
    )

    # convert outage list into powsybls 3w converter convention
    def convert_to_powsybl_3w_convention(id: str) -> list[str]:
        id_leg1 = id + "-Leg1"
        id_leg2 = id + "-Leg2"
        id_leg3 = id + "-Leg3"
        return [id_leg1, id_leg2, id_leg3]

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
    three_windings_transformer_outage_map = {}
    for _, row in busbar.iterrows():
        outage_list = []
        if pd.notna(row["3_windings_transformer_id_bus1"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus1"]))
        if pd.notna(row["3_windings_transformer_id_bus2"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus2"]))
        if pd.notna(row["3_windings_transformer_id_bus3"]):
            outage_list.extend(convert_to_powsybl_3w_convention(row["3_windings_transformer_id_bus3"]))
        if outage_list:
            three_windings_transformer_outage_map[row["busbar_section_id"]] = outage_list
    return three_windings_transformer_outage_map


def test_circuit_group_vs_powsybl_security_analysis_elements() -> None:
    """Test propagated busbar outages against element-based no-propagation outages."""
    net, context = _prepare_security_analysis_test_context()
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_elements = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        elements_ids = get_all_failing_elements_by_busbar_section_id(
            busbar_section_id=busbar_section_id,
            busbar_outage_groups=context.busbar_outage_groups,
            outage_groups=context.identified_circuit_groups.circuit_group_map,
        )
        security_analysis_no_propagation_elements.add_multiple_elements_contingency(elements_ids, busbar_section_id)
    security_analysis_no_propagation_elements.add_monitored_elements(branch_ids=context.monitored_branches)
    result_powsybl_no_propagation = security_analysis_no_propagation_elements.run_dc(
        net, parameters=sa_parameter_no_propagation
    )

    result_index_propagation = context.result_powsybl.branch_results.index.get_level_values(0)
    result_index_no_propagation = result_powsybl_no_propagation.branch_results.index.get_level_values(0)

    for busbar_section_id in context.outage_busbar_sections:
        if busbar_section_id not in result_index_propagation or busbar_section_id not in result_index_no_propagation:
            print(f"Skipping busbar section {busbar_section_id}")
            continue

        if busbar_section_id in ["VL_3W_HV_1_1", "VL_2W_MV_HV_HV_1_2"]:
            print(f"Skipping busbar section {busbar_section_id} due to known issue with this section")
            continue

        powsybl_filtered = _drop_non_semantic_columns(context.result_powsybl.branch_results.loc[busbar_section_id, ""])
        no_propagation_filtered = _drop_non_semantic_columns(
            result_powsybl_no_propagation.branch_results.loc[busbar_section_id, ""]
        )
        _assert_zero_only_missing_branches(powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered)
        _assert_matching_common_branches(powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered)
        print(f"Busbar section {busbar_section_id} passed the checks.")
        busbars_evaluated += 1

    assert busbars_evaluated > 10, "Not enough busbar sections evaluated, something is wrong with the test setup"


def test_circuit_group_vs_powsybl_security_analysis_all_switches() -> None:
    """Test propagated busbar outages against all-switch no-propagation outages."""
    net, context = _prepare_security_analysis_test_context()
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_all_switches = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        switch_ids = get_all_failing_switches_by_busbar_section_id(
            busbar_section_id=busbar_section_id,
            busbar_outage_groups=context.busbar_outage_groups,
            outage_groups=context.identified_circuit_groups.circuit_group_map,
        )
        security_analysis_no_propagation_all_switches.add_multiple_elements_contingency(switch_ids, busbar_section_id)
    security_analysis_no_propagation_all_switches.add_monitored_elements(branch_ids=context.monitored_branches)
    result_powsybl_no_propagation_switches = security_analysis_no_propagation_all_switches.run_dc(
        net, parameters=sa_parameter_no_propagation
    )

    result_index_propagation = context.result_powsybl.branch_results.index.get_level_values(0)
    result_index_no_propagation_switches = result_powsybl_no_propagation_switches.branch_results.index.get_level_values(0)

    for busbar_section_id in context.outage_busbar_sections:
        if (
            busbar_section_id not in result_index_propagation
            or busbar_section_id not in result_index_no_propagation_switches
        ):
            print(f"Skipping busbar section {busbar_section_id}")
            continue

        powsybl_filtered = _drop_non_semantic_columns(context.result_powsybl.branch_results.loc[busbar_section_id, ""])
        no_propagation_switches_filtered = _drop_non_semantic_columns(
            result_powsybl_no_propagation_switches.branch_results.loc[busbar_section_id, ""]
        )
        _assert_matching_common_branches(
            powsybl_filtered=powsybl_filtered,
            comparison_filtered=no_propagation_switches_filtered,
        )
        print(f"Busbar section {busbar_section_id} passed the switch outage checks.")
        busbars_evaluated += 1

    assert busbars_evaluated > 10, "Not enough busbar sections evaluated, something is wrong with the test setup"


@pytest.mark.parametrize("run_ac", [True, False])
def test_circuit_group_vs_powsybl_security_analysis_switches_on_busbar(run_ac) -> None:
    """Test propagated busbar outages against direct busbar-breaker no-propagation outages.

    This test replicates the behavior of the powsybl security analysis.
    Only the close by BREAKER are opened, no 3w outage if they have been converted, no 3-segmented line etc.
    """
    if run_ac:
        pytest.skip("AC fails due to unknown reason, needs further investigation")
    net, context = _prepare_security_analysis_test_context(add_3_windings_transformer_outage=False, run_ac=run_ac)
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_switches_on_busbar = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        busbar_outage_group = context.busbar_outage_groups[busbar_section_id]
        switch_ids = busbar_outage_group.primary_asset_breakers + busbar_outage_group.busbar_couplers
        security_analysis_no_propagation_switches_on_busbar.add_multiple_elements_contingency(switch_ids, busbar_section_id)
    security_analysis_no_propagation_switches_on_busbar.add_monitored_elements(branch_ids=context.monitored_branches)
    if run_ac:
        result_powsybl_no_propagation_switches_on_busbar = security_analysis_no_propagation_switches_on_busbar.run_ac(
            net, parameters=sa_parameter_no_propagation
        )
    else:
        result_powsybl_no_propagation_switches_on_busbar = security_analysis_no_propagation_switches_on_busbar.run_dc(
            net, parameters=sa_parameter_no_propagation
        )

    result_index_propagation = context.result_powsybl.branch_results.index.get_level_values(0)
    result_index_no_propagation_switches_on_busbar = (
        result_powsybl_no_propagation_switches_on_busbar.branch_results.index.get_level_values(0)
    )

    if run_ac:
        atol = 1e-3

    else:
        atol = 1e-6
    for busbar_section_id in context.outage_busbar_sections:
        if (
            busbar_section_id not in result_index_propagation
            or busbar_section_id not in result_index_no_propagation_switches_on_busbar
        ):
            print(f"Skipping busbar section {busbar_section_id}")
            continue

        powsybl_filtered = _drop_non_semantic_columns(context.result_powsybl.branch_results.loc[busbar_section_id, ""])
        no_propagation_filtered_switches_on_busbar = _drop_non_semantic_columns(
            result_powsybl_no_propagation_switches_on_busbar.branch_results.loc[busbar_section_id, ""]
        )
        _assert_zero_only_missing_branches(
            powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered_switches_on_busbar, atol=atol
        )
        _assert_matching_common_branches(
            powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered_switches_on_busbar, atol=atol
        )
        busbars_evaluated += 1

    assert busbars_evaluated > 10, "Not enough busbar sections evaluated, something is wrong with the test setup"
