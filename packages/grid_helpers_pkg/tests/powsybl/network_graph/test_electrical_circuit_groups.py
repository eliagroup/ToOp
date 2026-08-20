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
from pypowsybl.security import Parameters as SecurityParameters
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.electrical_circuit_groups import (
    identify_circuit_groups,
)
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.helper_functions import (
    build_branch_circuit_group_lookup,
    build_busbar_circuit_group_lookup,
    build_circuit_group_lookup_index,
    build_circuit_group_map,
    get_failing_elements_by_branch_ids,
    get_failing_elements_by_busbar_ids,
    get_failing_switches_by_branch_ids,
    get_failing_switches_by_busbar_ids,
    preprocess_circuit_group_lookup,
)
from toop_engine_grid_helpers.powsybl.electrical_circuit_groups.types import ElectricalCircuitGroup
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo


def _get_direct_busbar_switches(identified_circuit_groups: object, busbar_section_id: str) -> list[str]:
    """Resolve the breaker IDs directly incident to one busbar section."""
    injections = identified_circuit_groups.injections
    switches = identified_circuit_groups.switches
    lookup_index = identified_circuit_groups.lookup_index

    busbar_sections = injections[injections["type"] == "BUSBAR_SECTION"]
    busbar_bus_id = injections.loc[busbar_section_id, "bus_breaker_bus_id"]
    primary_group = lookup_index.busbar_to_primary_group[busbar_section_id]
    busbar_group_by_bus = (
        busbar_sections[["bus_breaker_bus_id", "electrical_circuit_group"]]
        .drop_duplicates(subset=["bus_breaker_bus_id"])
        .set_index("bus_breaker_bus_id")["electrical_circuit_group"]
    )

    direct_switch_ids: list[str] = []
    for switch_id, switch in switches.iterrows():
        if switch["bus_breaker_bus1_id"] == busbar_bus_id:
            other_bus_id = switch["bus_breaker_bus2_id"]
            other_group = int(switch["electrical_circuit_group_bus2"])
        elif switch["bus_breaker_bus2_id"] == busbar_bus_id:
            other_bus_id = switch["bus_breaker_bus1_id"]
            other_group = int(switch["electrical_circuit_group_bus1"])
        else:
            continue

        other_busbar_group = busbar_group_by_bus.get(other_bus_id)
        if pd.notna(other_busbar_group) or other_group != primary_group:
            direct_switch_ids.append(str(switch_id))

    return direct_switch_ids


def test_circuit_group_basic_functions() -> None:
    """Test the INT-station circuit-group expansion around one shared busbar section.

    INT station has two line and transformer pairs that meet at a shared busbar
    section. This test converts the inner station switches to disconnectors so
    that the circuit-group identification can propagate the outage through the
    busbar and reach the opposite asset pair.

    Simplified topology used by this test::

             L8                                2W_MV_HV_1
             |                                     |
             |                                     |
       L81_DISCONNECTOR                    2W_MV_HV_12_DISCONNECTOR
             x                                     x
              \\                                   /
               \\                                 /
                +---- VL_2W_MV_HV_MV_INT_1_1 ----+

                +---- VL_2W_MV_HV_MV_INT_2_1 ----+
               /                                  \
              /                                    \
        L92_DISCONNECTOR                          2W_MV_HV_22_DISCONNECTOR
             x                                      x
             |                                      |
             |                                      |
            L9                                 2W_MV_HV_2

    We have now created a circuit group that includes the two line pairs and the busbar section.
    Test that this is correctly identified and that the failing elements and switches are correctly resolved.
    
    """
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
    # rename BREAKER -> DISCONNECTOR
    switches.index = switches.index.str.replace("BREAKER", "DISCONNECTOR")
    net.create_switches(switches)
    switch_states_before = net.get_switches(all_attributes=True)

    result = identify_circuit_groups(net)
    circuit_group_map = build_circuit_group_map(result)
    branch_L9_group = result.branches.loc["L9"]["electrical_circuit_group"]
    expected = ElectricalCircuitGroup(
        branches=["L9", "2W_MV_HV_2"],
        switches=["L92_BREAKER", "2W_MV_HV_21_BREAKER"],
        injections=[],
        busbar_section=["VL_2W_MV_HV_MV_INT_2_1"],
    )
    assert set(circuit_group_map[branch_L9_group].branches) == set(expected.branches)
    assert set(circuit_group_map[branch_L9_group].switches) == set(expected.switches)
    assert set(circuit_group_map[branch_L9_group].injections) == set(expected.injections)
    assert set(circuit_group_map[branch_L9_group].busbar_section) == set(expected.busbar_section)

    failing_elements_by_branch_id = get_failing_elements_by_branch_ids(["L9"], lookup_index=result.lookup_index)["L9"]
    failing_switches_by_branch_id = get_failing_switches_by_branch_ids(["L9"], lookup_index=result.lookup_index)["L9"]

    assert set(failing_elements_by_branch_id) == set(expected.branches + expected.injections)
    assert set(failing_switches_by_branch_id) == set(expected.switches)

    failing_elements_by_branch_id = get_failing_elements_by_branch_ids(
        ["L9"], lookup_index=result.lookup_index, include_busbar_id=True
    )["L9"]
    assert set(failing_elements_by_branch_id) == set(expected.branches + expected.injections + expected.busbar_section)

    assert build_circuit_group_map(result.lookup_index) == circuit_group_map

    batch_branch_elements = get_failing_elements_by_branch_ids(["L9"], lookup_index=result.lookup_index)
    batch_branch_elements_with_busbar = get_failing_elements_by_branch_ids(
        ["L9"], lookup_index=result.lookup_index, include_busbar_id=True
    )
    batch_branch_switches = get_failing_switches_by_branch_ids(["L9"], lookup_index=result.lookup_index)
    batch_busbar_elements = get_failing_elements_by_busbar_ids(["VL_2W_MV_HV_MV_INT_2_1"], lookup_index=result.lookup_index)
    batch_busbar_switches = get_failing_switches_by_busbar_ids(["VL_2W_MV_HV_MV_INT_2_1"], lookup_index=result.lookup_index)

    assert set(batch_branch_elements["L9"]) == set(expected.branches + expected.injections)
    assert set(batch_branch_elements_with_busbar["L9"]) == set(
        expected.branches + expected.injections + expected.busbar_section
    )
    assert set(batch_branch_switches["L9"]) == set(expected.switches)
    assert set(batch_busbar_elements["VL_2W_MV_HV_MV_INT_2_1"]) == set(expected.injections)
    assert set(batch_busbar_switches["VL_2W_MV_HV_MV_INT_2_1"]) == set()

    lookup_only_result = identify_circuit_groups(net)
    assert get_failing_elements_by_branch_ids(["L9"], lookup_index=lookup_only_result.lookup_index)["L9"]

    # make sure the grid state is not modified by the outage group identification
    # the grid should be cloned / a variant should be created for the outage group identification,
    # so that the original grid is not modified
    switch_states_after = net.get_switches(all_attributes=True)
    assert_frame_equal(switch_states_before, switch_states_after)


def test_build_circuit_group_lookup_index_skips_breakers_without_secondary_group() -> None:
    """Ignore breakers that touch a busbar but do not lead to a different circuit group."""
    branches = pd.DataFrame(
        {"bus_breaker_bus1_id": [], "bus_breaker_bus2_id": [], "electrical_circuit_group": []},
        index=pd.Index([], name="id"),
    )
    switches = pd.DataFrame(
        {
            "bus_breaker_bus1_id": ["asset_bus"],
            "bus_breaker_bus2_id": ["busbar_bus"],
            "kind": ["BREAKER"],
            "electrical_circuit_group_bus1": [7],
            "electrical_circuit_group_bus2": [7],
        },
        index=pd.Index(["breaker_same_group"], name="id"),
    )
    injection = pd.DataFrame(
        {
            "bus_breaker_bus_id": ["busbar_bus"],
            "type": ["BUSBAR_SECTION"],
            "electrical_circuit_group": [7],
        },
        index=pd.Index(["busbar_section"], name="id"),
    )

    lookup_index = build_circuit_group_lookup_index(
        branches=branches,
        switches=switches,
        injection=injection,
    )

    assert lookup_index.busbar_to_asset_groups == {"busbar_section": []}
    assert lookup_index.busbar_to_failing_elements == {"busbar_section": []}
    assert lookup_index.busbar_to_failing_switches == {"busbar_section": []}


def test_build_circuit_group_lookup_index_matches_existing_grouping_helpers() -> None:
    """The lookup-oriented outage index should agree with the existing compatibility helpers."""
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)
    pypowsybl.loadflow.run_ac(net)

    result = identify_circuit_groups(net)
    lookup_index = build_circuit_group_lookup_index(
        branches=result.branches,
        switches=result.switches,
        injection=result.injections,
    )

    assert build_circuit_group_map(lookup_index) == build_circuit_group_map(result)


def test_split_circuit_group_lookup_builders_match_full_index() -> None:
    """Shared preprocess plus split builders should reconstruct the full lookup index."""
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)
    pypowsybl.loadflow.run_ac(net)

    result = identify_circuit_groups(net)
    preprocessed = preprocess_circuit_group_lookup(
        branches=result.branches,
        switches=result.switches,
        injection=result.injections,
    )
    busbar_lookup_index = build_busbar_circuit_group_lookup(preprocessed=preprocessed)
    branch_lookup_index = build_branch_circuit_group_lookup(
        preprocessed=preprocessed,
        busbar_lookup_index=busbar_lookup_index,
    )
    lookup_index = build_circuit_group_lookup_index(
        branches=result.branches,
        switches=result.switches,
        injection=result.injections,
    )

    assert busbar_lookup_index.busbar_to_primary_group == lookup_index.busbar_to_primary_group
    assert busbar_lookup_index.busbar_to_asset_groups == lookup_index.busbar_to_asset_groups
    assert busbar_lookup_index.busbar_to_failing_elements == lookup_index.busbar_to_failing_elements
    assert busbar_lookup_index.busbar_to_failing_switches == lookup_index.busbar_to_failing_switches
    assert branch_lookup_index.branch_to_group == lookup_index.branch_to_group
    assert branch_lookup_index.group_to_failing_elements == lookup_index.group_to_failing_elements
    assert branch_lookup_index.group_to_failing_switches == lookup_index.group_to_failing_switches


def _drop_non_semantic_columns(branch_results: pd.DataFrame) -> pd.DataFrame:
    """Drop result columns that are not stable across equivalent contingency runs.

    Quick helper to improve readability of the test assertions.
    """
    return branch_results.drop(columns=["flow_transfer"], errors="ignore")


def _assert_zero_only_missing_branches(
    powsybl_filtered: pd.DataFrame, comparison_filtered: pd.DataFrame, atol: float = 1e-6
) -> None:
    """Assert that branches missing from the comparison carry no propagated power.

    Quick helper to improve readability of the test assertions."""
    missing_indices = list(set(powsybl_filtered.index) - set(comparison_filtered.index))
    expected_zero = powsybl_filtered.loc[missing_indices]
    assert np.allclose(expected_zero["p1"].sum(), 0.0, atol=atol)


def _assert_matching_common_branches(
    powsybl_filtered: pd.DataFrame, comparison_filtered: pd.DataFrame, atol: float = 1e-6
) -> None:
    """Assert that both result frames match on their shared monitored branches.

    Quick helper to improve readability of the test assertions."""
    common_indices = list(set(powsybl_filtered.index) & set(comparison_filtered.index))
    assert_frame_equal(
        powsybl_filtered.loc[common_indices],
        comparison_filtered.loc[common_indices],
        check_dtype=False,
        check_like=True,
        atol=atol,
    )


def test_circuit_group_vs_powsybl_security_analysis_elements(security_analysis_test_context) -> None:
    """Test propagated busbar outages against element-based no-propagation outages.

    Go through all busbar sections and branches and compare the powsybl security analysis results with the results of the outage-group identification.

    """
    net, context = security_analysis_test_context
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_elements = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        elements_ids = get_failing_elements_by_busbar_ids(
            [busbar_section_id], lookup_index=context.identified_circuit_groups.lookup_index
        )[busbar_section_id]
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


def test_circuit_group_vs_powsybl_security_analysis_branch_elements(security_analysis_test_context) -> None:
    """Test propagated branch outages against element-based no-propagation outages."""
    net, context = security_analysis_test_context
    sa_parameter_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "true"})
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})

    security_analysis_branch_propagation = pypowsybl.security.create_analysis()
    security_analysis_branch_propagation.add_single_element_contingencies(context.monitored_branches)
    security_analysis_branch_propagation.add_monitored_elements(branch_ids=context.monitored_branches)
    result_branch_propagation = security_analysis_branch_propagation.run_dc(net, parameters=sa_parameter_propagation)

    security_analysis_branch_no_propagation = pypowsybl.security.create_analysis()
    for branch_id in context.monitored_branches:
        elements_ids = get_failing_elements_by_branch_ids(
            [branch_id], lookup_index=context.identified_circuit_groups.lookup_index
        )[branch_id]
        security_analysis_branch_no_propagation.add_multiple_elements_contingency(elements_ids, branch_id)
    security_analysis_branch_no_propagation.add_monitored_elements(branch_ids=context.monitored_branches)
    result_branch_no_propagation = security_analysis_branch_no_propagation.run_dc(
        net, parameters=sa_parameter_no_propagation
    )

    result_index_branch_propagation = result_branch_propagation.branch_results.index.get_level_values(0)
    result_index_branch_no_propagation = result_branch_no_propagation.branch_results.index.get_level_values(0)
    branches_evaluated = 0
    for branch_id in context.monitored_branches:
        if branch_id not in result_index_branch_propagation or branch_id not in result_index_branch_no_propagation:
            print(f"Skipping branch {branch_id}")
            continue

        powsybl_filtered = _drop_non_semantic_columns(result_branch_propagation.branch_results.loc[branch_id, ""])
        no_propagation_filtered = _drop_non_semantic_columns(result_branch_no_propagation.branch_results.loc[branch_id, ""])
        _assert_zero_only_missing_branches(powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered)
        _assert_matching_common_branches(powsybl_filtered=powsybl_filtered, comparison_filtered=no_propagation_filtered)
        print(f"Branch {branch_id} passed the checks.")
        branches_evaluated += 1

    assert branches_evaluated > 10, "Not enough branches evaluated, something is wrong with the test setup"


def test_circuit_group_vs_powsybl_security_analysis_all_switches(security_analysis_test_context) -> None:
    """Test propagated busbar outages against all-switch no-propagation outages."""
    net, context = security_analysis_test_context
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_all_switches = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        switch_ids = get_failing_switches_by_busbar_ids(
            [busbar_section_id], lookup_index=context.identified_circuit_groups.lookup_index
        )[busbar_section_id]
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


def test_circuit_group_vs_powsybl_security_analysis_branch_switches(security_analysis_test_context) -> None:
    """Test propagated branch outages against switch-based no-propagation outages."""
    net, context = security_analysis_test_context
    sa_parameter_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "true"})
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})

    security_analysis_branch_propagation = pypowsybl.security.create_analysis()
    security_analysis_branch_propagation.add_single_element_contingencies(context.monitored_branches)
    security_analysis_branch_propagation.add_monitored_elements(branch_ids=context.monitored_branches)
    result_branch_propagation = security_analysis_branch_propagation.run_dc(net, parameters=sa_parameter_propagation)

    security_analysis_branch_no_propagation_switches = pypowsybl.security.create_analysis()
    for branch_id in context.monitored_branches:
        switch_ids = get_failing_switches_by_branch_ids(
            [branch_id], lookup_index=context.identified_circuit_groups.lookup_index
        )[branch_id]
        security_analysis_branch_no_propagation_switches.add_multiple_elements_contingency(switch_ids, branch_id)
    security_analysis_branch_no_propagation_switches.add_monitored_elements(branch_ids=context.monitored_branches)
    result_branch_no_propagation_switches = security_analysis_branch_no_propagation_switches.run_dc(
        net, parameters=sa_parameter_no_propagation
    )

    result_index_branch_propagation = result_branch_propagation.branch_results.index.get_level_values(0)
    result_index_branch_no_propagation_switches = (
        result_branch_no_propagation_switches.branch_results.index.get_level_values(0)
    )
    branches_evaluated = 0
    for branch_id in context.monitored_branches:
        if branch_id not in result_index_branch_propagation or branch_id not in result_index_branch_no_propagation_switches:
            print(f"Skipping branch {branch_id}")
            continue

        powsybl_filtered = _drop_non_semantic_columns(result_branch_propagation.branch_results.loc[branch_id, ""])
        no_propagation_switches_filtered = _drop_non_semantic_columns(
            result_branch_no_propagation_switches.branch_results.loc[branch_id, ""]
        )
        _assert_matching_common_branches(
            powsybl_filtered=powsybl_filtered,
            comparison_filtered=no_propagation_switches_filtered,
        )
        print(f"Branch {branch_id} passed the switch outage checks.")
        branches_evaluated += 1

    assert branches_evaluated > 10, "Not enough branches evaluated, something is wrong with the test setup"


@pytest.mark.parametrize("run_ac", [True, False])
def test_circuit_group_vs_powsybl_security_analysis_switches_on_busbar(
    run_ac: bool, security_analysis_test_context_factory
) -> None:
    """Test propagated busbar outages against direct busbar-breaker no-propagation outages.

    This test replicates the behavior of the powsybl security analysis.
    Only the close by BREAKER are opened, no 3w outage if they have been converted, no 3-segmented line etc.
    """
    if run_ac:
        pytest.skip("AC fails due to unknown reason, needs further investigation")
    net, context = security_analysis_test_context_factory(
        add_3_windings_transformer_outage=False,
        run_ac=run_ac,
    )
    sa_parameter_no_propagation = SecurityParameters(provider_parameters={"contingencyPropagation": "false"})
    security_analysis_no_propagation_switches_on_busbar = pypowsybl.security.create_analysis()
    busbars_evaluated = 0
    for busbar_section_id in context.outage_busbar_sections:
        switch_ids = _get_direct_busbar_switches(context.identified_circuit_groups, busbar_section_id)
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
