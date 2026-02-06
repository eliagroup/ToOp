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
from toop_engine_contingency_analysis.pypowsybl import (
    POWSYBL_CONVERGENCE_MAP,
    PowsyblContingency,
    PowsyblMonitoredElements,
    PowsyblNMinus1Definition,
    add_name_column,
    get_blank_va_diff,
    get_blank_va_diff_with_buses,
    get_branch_results,
    get_convergence_result_df,
    get_node_results,
    get_va_diff_results,
    prepare_branch_limits,
    set_target_values_to_lf_values_incl_distributed_slack,
    translate_contingency_to_powsybl,
    translate_monitored_elements_to_powsybl,
    translate_nminus1_for_powsybl,
    update_basename,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import BranchResultSchema, NodeResultSchema, VADiffResultSchema
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, LoadflowParameters, Nminus1Definition


def test_powsybl_n_1_definition_slice():
    monitored_elements = PowsyblMonitoredElements(
        branches=["branch1", "branch2"],
        trafo3w=["trafo3w1"],
        buses=["bus1", "bus2"],
        voltage_levels=["voltage_level1", "voltage_level2"],
        switches=["switch1", "switch2"],
    )
    n_1_def = PowsyblNMinus1Definition(
        contingencies=[
            PowsyblContingency(id="cont_1", name="name1", elements=["line_1", "line_2"]),
            PowsyblContingency(id="cont_2", name="name1", elements=["line_2"]),
            PowsyblContingency(id="cont_3", name="name2", elements=[]),
        ],
        monitored_elements=monitored_elements,
        branch_limits=pd.DataFrame(),
        bus_map=pd.DataFrame(columns=["bus_breaker_bus_id"]),
        element_name_mapping={},
        contingency_name_mapping={},
        voltage_levels=pd.DataFrame(),
        blank_va_diff=pd.DataFrame(),
    )

    slice_1 = n_1_def[1:]
    assert len(slice_1.contingencies) == 2
    slice1_ids = [slice_.id for slice_ in slice_1.contingencies]
    assert "cont_2" in slice1_ids
    assert "cont_3" in slice1_ids
    assert "cont_1" not in slice1_ids

    slice_2 = n_1_def["cont_1"]
    assert len(slice_2.contingencies) == 1
    slice2_ids = [slice_.id for slice_ in slice_2.contingencies]
    assert "cont_1" in slice2_ids
    assert "cont_2" not in slice2_ids
    assert "cont_3" not in slice2_ids

    slice_3 = n_1_def[1]
    assert len(slice_3.contingencies) == 1
    slice_3_ids = [slice_.id for slice_ in slice_3.contingencies]
    assert "cont_2" in slice_3_ids
    assert "cont_1" not in slice_3_ids
    assert "cont_3" not in slice_3_ids

    for slice in [slice_1, slice_2, slice_3]:
        assert slice.monitored_elements == n_1_def.monitored_elements
        assert slice.branch_limits.equals(n_1_def.branch_limits)
        assert slice.blank_va_diff.equals(n_1_def.blank_va_diff)
        assert slice.bus_map.equals(n_1_def.bus_map)


def test_translate_contingency_to_powsybl():
    contingencies = [
        Contingency(id=f"cont_{i}", elements=[GridElement(id=f"element_{i}", type="line", kind="branch")]) for i in range(3)
    ]

    basecase_contingency = [Contingency(id="basecase", elements=[])]
    multi_contingencies = [
        Contingency(
            id=f"multi_cont_{i}",
            elements=[
                GridElement(id=f"element_{i}_1", type="line", kind="branch"),
                GridElement(id=f"element_{i}_2", type="line", kind="branch"),
            ],
        )
        for i in range(2)
    ]
    multi_element_ids = [f"element_{i}_1" for i in range(2)] + [f"element_{i}_2" for i in range(2)]
    single_element_ids = [f"element_{i}" for i in range(3)]

    # check single contingencies
    identifiables = pd.DataFrame(index=single_element_ids + multi_element_ids).index
    pow_contingencies, missing_contingencies = translate_contingency_to_powsybl(contingencies, identifiables=identifiables)
    assert len(pow_contingencies) == 3
    assert len(missing_contingencies) == 0
    ids = {pow_contingency.id for pow_contingency in pow_contingencies}
    single_ids = {contingency.id for contingency in contingencies}
    assert ids == single_ids, "All ids should be in the translated contingencies"
    # with missing elements
    identifiables = pd.DataFrame(index=multi_element_ids).index
    pow_contingencies, missing_contingencies = translate_contingency_to_powsybl(contingencies, identifiables=identifiables)
    assert len(pow_contingencies) == 0
    assert missing_contingencies == contingencies

    # multi
    identifiables = pd.DataFrame(index=multi_element_ids).index
    pow_contingencies, missing_contingencies = translate_contingency_to_powsybl(
        multi_contingencies, identifiables=identifiables
    )
    assert len(pow_contingencies) == 2
    assert len(missing_contingencies) == 0
    ids = {pow_contingency.id for pow_contingency in pow_contingencies}
    multi_ids = {contingency.id for contingency in multi_contingencies}
    assert ids == multi_ids, "All ids should be in the translated contingencies"
    # with missing elements
    identifiables = pd.DataFrame(index=single_element_ids).index
    pow_contingencies, missing_contingencies = translate_contingency_to_powsybl(
        multi_contingencies, identifiables=identifiables
    )
    assert len(pow_contingencies) == 0
    assert missing_contingencies == multi_contingencies

    # Check basecase
    identifiables = pd.DataFrame(index=[]).index
    pow_basecase, missing_basecase = translate_contingency_to_powsybl(basecase_contingency, identifiables=identifiables)
    assert len(pow_basecase) == 1
    assert pow_basecase[0].id == basecase_contingency[0].id, "Id should stay the same"


def test_translate_monitored_elements_to_powsybl(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    branches = powsybl_bus_breaker_net.get_branches()
    switches = powsybl_bus_breaker_net.get_switches()
    buses = powsybl_bus_breaker_net.get_bus_breaker_view_buses()

    branch_elements = [GridElement(id=id, type=row.type, kind="branch") for id, row in branches.iterrows()]
    switch_elements = [GridElement(id=id, type=row.kind, kind="switch") for id, row in switches.iterrows()]
    bus_elements = [GridElement(id=id, type="bus", kind="bus") for id, _row in buses.iterrows()]
    contingencies = [Contingency(id=elem.id, name=elem.id, elements=[elem]) for elem in branch_elements]

    # Test branches
    nminus1_definition = Nminus1Definition(
        contingencies=contingencies, monitored_elements=branch_elements, id_type="powsybl"
    )
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 0, (
        "Since all branch elements are in the branches df, there should be no missing elements"
    )
    assert branches.index.tolist() == monitored_elements["branches"] + monitored_elements["trafo3w"], (
        "Branches translation did not match"
    )
    assert len(monitored_elements["switches"]) == 0, "There should be no monitored switches if no switches are passed"
    assert len(monitored_elements["buses"]) == 0, "There should be no monitored buses if no buses are passed"
    expected_voltage_levels = branches.voltage_level1_id.to_list() + branches.voltage_level2_id.to_list()
    assert all(np.isin(np.array(monitored_elements["voltage_levels"]), expected_voltage_levels)), (
        "Monitored voltage levels should match bus indices"
    )
    assert len(element_map) == len(branch_elements), "Element map should contain all branch elements"
    # Test switches
    nminus1_definition = Nminus1Definition(
        contingencies=contingencies, monitored_elements=switch_elements, id_type="powsybl"
    )
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 0, (
        "Since all switch elements are in the switches df, there should be no missing elements"
    )
    assert switches.index.tolist() == monitored_elements["switches"], "Switch translation did not match"
    assert len(monitored_elements["branches"]) == 0, "There should be no monitored branches if no switches are passed"
    assert len(monitored_elements["trafo3w"]) == 0, "There should be no monitored trafo3w if no branches are passed"
    assert len(monitored_elements["buses"]) == 0, "There should be no monitored buses if no buses are passed"
    expected_voltage_levels = switches.voltage_level_id.to_list()
    assert all(np.isin(np.array(monitored_elements["voltage_levels"]), expected_voltage_levels)), (
        "Monitored voltage levels should match bus indices"
    )
    assert len(element_map) == len(switch_elements), "Element map should contain all branch elements"
    # Test buses
    nminus1_definition = Nminus1Definition(contingencies=contingencies, monitored_elements=bus_elements, id_type="powsybl")
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 0, "Since all bus elements are in the buses df, there should be no missing elements"
    assert buses.index.tolist() == monitored_elements["buses"], "Node translation did not match"
    assert len(monitored_elements["branches"]) == 0, "There should be no monitored branches if no switches are passed"
    assert len(monitored_elements["trafo3w"]) == 0, "There should be no monitored trafo3w if no branches are passed"
    expected_voltage_levels = buses.voltage_level_id.to_list()
    assert all(np.isin(np.array(monitored_elements["voltage_levels"]), expected_voltage_levels)), (
        "Monitored voltage levels should match bus indices"
    )
    assert len(element_map) == len(bus_elements), "Element map should contain all bus elements"
    # Test all together
    nminus1_definition = Nminus1Definition(
        contingencies=contingencies, monitored_elements=branch_elements + switch_elements + bus_elements, id_type="powsybl"
    )
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 0, "Since all elements are in the df, there should be no missing elements"
    assert branches.index.tolist() == monitored_elements["branches"] + monitored_elements["trafo3w"], (
        "Branches translation did not match"
    )
    assert switches.index.tolist() == monitored_elements["switches"], "Switch translation did not match"
    assert buses.index.tolist() == monitored_elements["buses"], "Node translation did not match"
    assert len(element_map) == len(branch_elements) + len(switch_elements) + len(bus_elements), (
        "Element map should contain all elements"
    )

    expected_voltage_levels = (
        branches.voltage_level1_id.to_list()
        + branches.voltage_level2_id.to_list()
        + switches.voltage_level_id.to_list()
        + buses.voltage_level_id.to_list()
    )
    assert all(np.isin(np.array(monitored_elements["voltage_levels"]), expected_voltage_levels)), (
        "Monitored voltage levels should match bus indices"
    )

    # Test empty input
    nminus1_definition = Nminus1Definition(contingencies=[], monitored_elements=[], id_type="powsybl")
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 0, "There should be no missing elements if no elements are passed"
    assert len(monitored_elements["branches"]) == 0, "There should be no monitored branches if no elements are passed"
    assert len(monitored_elements["trafo3w"]) == 0, "There should be no monitored trafo3w if no elements are passed"
    assert len(monitored_elements["switches"]) == 0, "There should be no monitored switches if no elements are passed"
    assert len(monitored_elements["buses"]) == 0, "There should be no monitored buses if no elements are passed"
    assert len(monitored_elements["voltage_levels"]) == 0, (
        "There should be no monitored voltage levels if no elements are passed"
    )
    assert len(element_map) == 0, "Element map should be empty if no elements are passed"
    # Test non existing elements
    nminus1_definition = Nminus1Definition(
        contingencies=[],
        monitored_elements=[GridElement(id="I do not exist", type="branch", kind="branch")],
        id_type="powsybl",
    )
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    assert len(missing_elements) == 1, "There should be one missing element"
    assert nminus1_definition.monitored_elements == missing_elements, "The missing element should be the one that was passed"
    assert len(monitored_elements["branches"]) == 0, "There should be no monitored branches if no elements are passed"
    # Test only contingencies. The branch voltage levels should still show up fo the va_diff
    nminus1_definition = Nminus1Definition(contingencies=contingencies, monitored_elements=[], id_type="powsybl")
    monitored_elements, element_map, missing_elements = translate_monitored_elements_to_powsybl(
        nminus1_definition, branches, buses, switches
    )
    expected_voltage_levels = branches.voltage_level1_id.to_list() + branches.voltage_level2_id.to_list()
    assert all(np.isin(np.array(monitored_elements["voltage_levels"]), expected_voltage_levels)), (
        "Monitored voltage levels should match bus indices"
    )


def test_prepare_branch_limits(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    branch_limits = powsybl_bus_breaker_net.get_operational_limits().query("type == 'CURRENT'")
    branch_elements = powsybl_bus_breaker_net.get_branches([]).index.tolist()

    translated_branch_limits = prepare_branch_limits(
        branch_limits, chosen_limit="permanent_limit", monitored_branches=branch_elements
    )

    assert translated_branch_limits.index.names == ["element_id", "side"], "Index names should be ['element_id', 'side']"
    assert np.all(translated_branch_limits.index.get_level_values("side").isin([1, 2, 3])), "Side should be 1, 2, or 3"
    assert np.all(translated_branch_limits.index.get_level_values("element_id").isin(branch_elements)), (
        "There should be no branch_ids that are not monitored"
    )
    translated_branch_limits_N_1 = prepare_branch_limits(
        branch_limits, chosen_limit="N-1", monitored_branches=branch_elements
    )
    assert np.any(translated_branch_limits_N_1.values != translated_branch_limits.values), (
        "The N-1 limits should be different from the permanent limits"
    )


def test_get_blank_va_diff():
    all_outages = ["outage_1", "outage_2", "outage_3"]
    single_branch_outages = {"outage_1": "branch_1"}
    monitored_switches = ["switch_1", "switch_2"]

    blank_va_diff = get_blank_va_diff(all_outages, single_branch_outages, monitored_switches)
    # For all outages, we check all monitored switches
    # In Addition a blank row for the base case with every switch is added
    # For every single branch outage we check the branch itself
    expected_length = (len(all_outages) + 1) * len(monitored_switches) + len(single_branch_outages)
    assert len(blank_va_diff) == expected_length

    for outage in [*all_outages, ""]:
        outage_rows = blank_va_diff.loc[blank_va_diff.index.get_level_values("contingency") == outage]
        elements = outage_rows.index.get_level_values("element").to_list()
        assert all(switch in elements for switch in monitored_switches), "All switches should show up as elements"
        if outage in single_branch_outages:
            assert len(outage_rows) == len(monitored_switches) + 1, (
                f"Outage {outage} should have n_monitored_switches + 1 rows"
            )
            assert single_branch_outages[outage] in elements, (
                f"Outage {outage} should have the branch {single_branch_outages[outage]} in the elements"
            )
        else:
            assert len(outage_rows) == len(monitored_switches), f"Outage {outage} should have n_monitored_switches rows"


def test_get_blank_va_diff_with_buses(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    branches = powsybl_bus_breaker_net.get_branches(all_attributes=True)
    switches = powsybl_bus_breaker_net.get_switches(all_attributes=True)
    switches.open = True
    switches.retained = True
    all_outage_ids = branches.index[:5].tolist()
    contingencies = []
    for i in range(5):
        if i < 3:
            contingencies.append(
                PowsyblContingency(
                    id=f"outage_{i}",
                    elements=[all_outage_ids[i]],
                )
            )
        else:
            contingencies.append(
                PowsyblContingency(
                    id=f"multi_outage_{i}",
                    elements=[all_outage_ids[0], all_outage_ids[i]],
                )
            )
    single_branch_outages = {f"outage_{i}": all_outage_ids[i] for i in range(3)}
    monitored_switches = switches.index[:2].tolist()

    blank_va_diff_with_buses = get_blank_va_diff_with_buses(branches, switches, contingencies, monitored_switches)

    for switch in monitored_switches:
        switch_rows = blank_va_diff_with_buses.loc[blank_va_diff_with_buses.index.get_level_values("element") == switch]
        assert switch_rows.bus_breaker_bus1_id.unique().size == 1, (
            f"Switch {switch} should only have one bus associated with it"
        )
        assert switch_rows.bus_breaker_bus2_id.unique().size == 1, (
            f"Switch {switch} should only have one bus associated with it"
        )
        assert switch_rows.bus_breaker_bus1_id.unique()[0] == switches.loc[switch].bus_breaker_bus1_id, (
            f"Switch {switch} should have the correct bus from the switches df"
        )
        assert switch_rows.bus_breaker_bus2_id.unique()[0] == switches.loc[switch].bus_breaker_bus2_id, (
            f"Switch {switch} should have the correct bus from the switches df"
        )

    for contingency, outaged_element in single_branch_outages.items():
        outage_rows = blank_va_diff_with_buses.loc[
            (blank_va_diff_with_buses.index.get_level_values("element") == outaged_element)
        ]
        assert len(outage_rows) == 1, "The branches should only be checked for their specific outage"
        assert outage_rows.index.get_level_values("contingency")[0] == contingency, (
            f"Outage {contingency} should have the correct contingency"
        )
        assert outage_rows.bus_breaker_bus1_id.unique().size == 1, (
            f"Outage {contingency} should only have one bus associated with it"
        )
        assert outage_rows.bus_breaker_bus2_id.unique().size == 1, (
            f"Outage {contingency} should only have one bus associated with it"
        )
        assert outage_rows.bus_breaker_bus1_id.unique()[0] == branches.loc[outaged_element].bus_breaker_bus1_id, (
            f"Outage {contingency} should have the correct bus from the branches df"
        )
        assert outage_rows.bus_breaker_bus2_id.unique()[0] == branches.loc[outaged_element].bus_breaker_bus2_id, (
            f"Outage {contingency} should have the correct bus from the branches df"
        )


def test_get_va_diff_results():
    blank_va_diff_with_buses = pd.DataFrame()
    blank_va_diff_with_buses["contingency"] = ["contingency_1", "contingency_1", "contingency_2", "contingency_2"]
    blank_va_diff_with_buses["element"] = ["element_1", "element_2", "element_1", "element_2"]
    blank_va_diff_with_buses["bus_breaker_bus1_id"] = ["bus_1", "bus_2", "bus_2", "bus_2"]
    blank_va_diff_with_buses["bus_breaker_bus2_id"] = ["bus_2", "bus_1", "bus_1", "bus_1"]
    blank_va_diff_with_buses.set_index(["contingency", "element"], inplace=True)

    bus_results = pd.DataFrame()
    bus_results["contingency_id"] = ["contingency_1", "contingency_1", "contingency_2", "contingency_2"]
    bus_results["operator_strategy_id"] = ""
    bus_results["voltage_level_id"] = ["placeholder"] * 4
    bus_results["bus_id"] = ["bus_1", "bus_2", "bus_1", "bus_2"]
    bus_results["v_mag"] = [10.0] * 4
    bus_results["v_angle"] = [180.0, 0, 10, np.nan]
    bus_results.set_index(["contingency_id", "operator_strategy_id", "voltage_level_id", "bus_id"], inplace=True)
    bus_map = pd.DataFrame(
        index=["bus_1", "bus_2"],
        data={"bus_breaker_bus_id": ["bus_1", "bus_2"]},
    )
    bus_map.index.name = "id"
    outages = ["contingency_1", "contingency_2"]
    timestep = 0
    va_results = get_va_diff_results(
        bus_results,
        outages,
        va_diff_with_buses=blank_va_diff_with_buses,
        bus_map=bus_map,
        timestep=timestep,
    )
    assert len(va_results) == 4, "As all contingencies are considered, there should be 4 rows"
    assert va_results.index.names == ["timestep", "contingency", "element"]
    assert all(va_results.index.get_level_values("timestep") == timestep), "All rows should have the same timestep"
    assert va_results.loc[(0, "contingency_1", "element_1"), "va_diff"] == 180, (
        "The va_diff for contingency_1 and element_1 should be 180"
    )
    assert va_results.loc[(0, "contingency_1", "element_2"), "va_diff"] == -180, (
        "The va_diff for contingency_1 and element_2 should be -180"
    )
    assert np.isnan(va_results.loc[(0, "contingency_2", "element_1"), "va_diff"]), (
        "The va_diff for contingency_2 and element_1 should be Nan"
    )
    assert np.isnan(va_results.loc[(0, "contingency_2", "element_2"), "va_diff"]), (
        "The va_diff for contingency_2 and element_2 should be NaN"
    )

    outages = ["contingency_1"]
    va_results = get_va_diff_results(
        bus_results,
        outages,
        va_diff_with_buses=blank_va_diff_with_buses,
        bus_map=bus_map,
        timestep=timestep,
    )
    assert len(va_results) == 2, "As only the first contingency is considered, there should be 2 rows"


def test_translate_nminus1_for_powsybl(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    buses = powsybl_bus_breaker_net.get_bus_breaker_view_buses().iloc[:6]
    branches = powsybl_bus_breaker_net.get_branches(all_attributes=True).iloc[:6]
    switches = powsybl_bus_breaker_net.get_switches(all_attributes=True).iloc[:2]
    multi_contingencies = [
        Contingency(
            id="multi_cont_1", elements=[GridElement(id=id, kind="branch", type="line") for id in branches.index[:2]]
        ),
    ]
    basecase = [Contingency(id="BASECASE", elements=[])]
    single_contingencies = [
        Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
        for index, row in branches.iterrows()
    ]

    monitored_branches = [
        GridElement(id=index, name=row.name, kind="branch", type=row.type) for index, row in branches.iterrows()
    ]
    monitored_buses = [GridElement(id=index, name=row.name, kind="bus", type="bus") for index, row in buses.iterrows()]
    monitored_switches = [
        GridElement(id=index, name=row.name, kind="switch", type=row.kind) for index, row in switches.iterrows()
    ]

    nminus1_def = Nminus1Definition(
        monitored_elements=monitored_branches + monitored_buses + monitored_switches,
        contingencies=single_contingencies + multi_contingencies + basecase,
        loadflow_parameters=LoadflowParameters(distributed_slack=True),
        id_type="powsybl",
    )
    translated_nminus1 = translate_nminus1_for_powsybl(nminus1_def, powsybl_bus_breaker_net)
    assert isinstance(translated_nminus1, PowsyblNMinus1Definition), (
        "The translated N-1 definition should be of type PowsyblNMinus1Definition"
    )
    assert len(translated_nminus1.contingencies) == 8, (
        "There should be 8 outages in the translated N-1 definition (6 branches + 1 multi outage + base case)"
    )
    assert "BASECASE" in [contingency.id for contingency in translated_nminus1.contingencies], (
        "The base case should be included in the outages"
    )
    assert len(translated_nminus1.monitored_elements["branches"]) == 6, (
        "There should be 6 monitored branches in the translated N-1 definition"
    )
    assert len(translated_nminus1.monitored_elements["trafo3w"]) == 0, (
        "There should be no monitored trafo3w in the translated N-1 definition"
    )
    assert len(translated_nminus1.monitored_elements["switches"]) == 2, (
        "There should be 2 monitored switches in the translated N-1 definition"
    )
    assert len(translated_nminus1.monitored_elements["buses"]) == 6, (
        "There should be 6 monitored buses in the translated N-1 definition"
    )
    expected_voltage_levels = set(
        branches.voltage_level1_id.to_list()
        + branches.voltage_level2_id.to_list()
        + switches.voltage_level_id.to_list()
        + buses.voltage_level_id.to_list()
    )
    assert len(translated_nminus1.monitored_elements["voltage_levels"]) == len(expected_voltage_levels), (
        "The number of monitored voltage levels should match the expected number"
    )
    assert all(np.isin(translated_nminus1.monitored_elements["voltage_levels"], list(expected_voltage_levels))), (
        "All monitored voltage levels should be in the expected voltage levels"
    )
    assert all(np.isin(translated_nminus1.monitored_elements["branches"], branches.index.tolist())), (
        "All monitored branches should be in the translated N-1 definition"
    )
    assert all(np.isin(translated_nminus1.monitored_elements["switches"], switches.index.tolist())), (
        "All monitored switches should be in the translated N-1 definition"
    )
    assert all(np.isin(translated_nminus1.monitored_elements["buses"], buses.index.tolist())), (
        "All monitored buses should be in the translated N-1 definition"
    )

    assert not translated_nminus1.branch_limits.empty, "Branch limits should not be empty in the translated N-1 definition"
    assert not translated_nminus1.blank_va_diff.empty, "Blank VA diff should not be empty in the translated N-1 definition"
    assert not translated_nminus1.bus_map.empty, "There should be a busbar mapping in the translated N-1 definition"
    assert translated_nminus1.distributed_slack is True, (
        "The distributed slack should be set to True in the translated N-1 definition"
    )

    nminus1_def.loadflow_parameters.distributed_slack = False
    translated_nminus1 = translate_nminus1_for_powsybl(nminus1_def, powsybl_bus_breaker_net)
    assert translated_nminus1.distributed_slack is False, (
        "The distributed slack should be set to False in the translated N-1 definition"
    )

    nminus1_def.id_type = "unique_pandapower"
    with pytest.raises(ValueError):
        translate_nminus1_for_powsybl(nminus1_def, powsybl_bus_breaker_net)


def test_get_branch_results():
    ca_branch_results = pd.DataFrame(
        {
            "branch_id": ["branch_1", "branch_2", "branch_1", "branch_2"],
            "contingency_id": ["cont_1", "cont_1", "cont_2", "cont_2"],
            "operator_strategy_id": ["placeholder"] * 4,
            "p1": [100.0, 200.0, 0.0, np.nan],
            "q1": [50.0, 100.0, 0.0, np.nan],
            "i1": [10.0, 20.0, 0.0, np.nan],
            "p2": [90.0, 190.0, 0.0, np.nan],
            "q2": [40.0, 90.0, 0.0, np.nan],
            "i2": [5.0, 10.0, 0.0, np.nan],
            "flow_transfer": [np.nan] * 4,
        }
    )
    ca_branch_results.set_index(["contingency_id", "operator_strategy_id", "branch_id"], inplace=True)

    three_winding_results = pd.DataFrame(
        {
            "transformer_id": ["trafo_1", "trafo_2", "trafo_1", "trafo_2"],
            "contingency_id": ["cont_1", "cont_1", "cont_2", "cont_2"],
            "p1": [100.0, 200.0, 0.0, np.nan],
            "q1": [50.0, 100.0, 0.0, np.nan],
            "i1": [10.0, 20.0, 0.0, np.nan],
            "p2": [90.0, 190.0, 0.0, np.nan],
            "q2": [40.0, 90.0, 0.0, np.nan],
            "i2": [5.0, 10.0, 0.0, np.nan],
            "p3": [90.0, 190.0, 0.0, np.nan],
            "q3": [40.0, 90.0, 0.0, np.nan],
            "i3": [5.0, 10.0, 0.0, np.nan],
            "flow_transfer": [np.nan] * 4,
        }
    )
    three_winding_results.set_index(["contingency_id", "transformer_id"], inplace=True)
    monitored_branches = ["branch_1", "branch_2"]
    monitored_trafo3w = ["trafo_1", "trafo_2"]
    failed_outages = ["cont_3"]  # This wont show up in the results
    timestep = 0
    branch_limits = pd.DataFrame(
        {
            "branch_id": ["branch_1", "branch_1", "trafo_1"],
            "side": [1, 2, 1],
            "value": [100, 100, 100],
            "acceptable_duration": [-1, -1, -1],
        }
    )
    branch_limits.set_index(["branch_id", "side"], inplace=True)
    branch_results = get_branch_results(
        ca_branch_results,
        three_winding_results,
        monitored_branches,
        monitored_trafo3w,
        failed_outages,
        timestep,
        branch_limits,
    )

    n_contingencies = 3  # 1 failed + 2 successful contingencies
    n_branches = 2  # 2 branches
    n_trafo3w = 2  # 2 three winding transformers
    n_expected_rows = n_contingencies * (n_branches * 2 + n_trafo3w * 3)

    assert len(branch_results) == n_expected_rows, (
        "The number of rows in the branch results should match the expected number"
    )
    assert branch_results.index.names == ["timestep", "contingency", "element", "side"]
    assert all(branch_results.index.get_level_values("timestep") == timestep), "All rows should have the same timestep"
    assert all(branch_results.index.get_level_values("contingency").isin(["cont_1", "cont_2", "cont_3"])), (
        "All contingencies should be present in the results"
    )
    assert all(branch_results.index.get_level_values("element").isin(monitored_branches + monitored_trafo3w)), (
        "All elements should be present in the results"
    )

    # check that the values for the branches are correctly translated
    for (contingency, _op_strat, element), branch in ca_branch_results.iterrows():
        for side in [1, 2]:
            for value in ["p", "q", "i"]:
                original_value = branch[f"{value}{side}"]
                result_value = branch_results.loc[(timestep, contingency, element, side), value]
                assert result_value == original_value or (np.isnan(result_value) and np.isnan(original_value)), (
                    f"Power flow {value}{side} for {element} should match"
                )
            expected_loading = (
                branch[f"i{side}"] / branch_limits.loc[(element, side), "value"]
                if (element, side) in branch_limits.index
                else np.nan
            )
            result_loading = branch_results.loc[(timestep, contingency, element, side), "loading"]
            assert result_loading == expected_loading or (np.isnan(result_loading) and np.isnan(expected_loading)), (
                f"Loading for {element} should match"
            )
    # Check that the values for the three-winding transformers are correctly translated
    for (contingency, element), trafo in three_winding_results.iterrows():
        for side in [1, 2, 3]:
            for value in ["p", "q", "i"]:
                original_value = trafo[f"{value}{side}"]
                result_value = branch_results.loc[(timestep, contingency, element, side), value]
                assert result_value == original_value or (np.isnan(result_value) and np.isnan(original_value)), (
                    f"Power flow {value}{side} for {element} should match"
                )
            expected_loading = (
                trafo[f"i{side}"] / branch_limits.loc[(element, side), "value"]
                if (element, side) in branch_limits.index
                else np.nan
            )
            result_loading = branch_results.loc[(timestep, contingency, element, side), "loading"]
            assert result_loading == expected_loading or (np.isnan(result_loading) and np.isnan(expected_loading)), (
                f"Loading for {element} should match"
            )
    # Check that all rows for the failed outages are NaN
    assert np.all(
        branch_results.loc[branch_results.index.get_level_values("contingency") == failed_outages[0]][
            ["p", "q", "i", "loading"]
        ].isna()
    ), "All rows for failed outages should be NaN"


def test_get_convergence_df():
    net = pypowsybl.network.create_ieee14()

    analysis = pypowsybl.security.create_analysis()
    analysis.add_monitored_elements(
        branch_ids=net.get_branches().index.tolist(),
    )
    for id in net.get_branches().index:
        analysis.add_single_element_contingency(id, contingency_id=id)
    res = analysis.run_ac(net)
    timestep = 0
    basecase_name = "BASECASE"
    outages = net.get_branches().index.tolist()
    convergence_df, failed_outages = get_convergence_result_df(
        res.post_contingency_results,
        res.pre_contingency_result,
        outages=outages,
        basecase_name=basecase_name,
        timestep=timestep,
    )

    assert len(failed_outages) == 0, "In the test case all should converge"
    assert len(convergence_df) == len(outages) + 1, (
        "The convergence dataframe should have one row for each outage and one for the base case"
    )
    for outage, result in res.post_contingency_results.items():
        assert convergence_df.loc[(timestep, outage), "status"] == POWSYBL_CONVERGENCE_MAP[result.status.value], (
            f"Convergence for outage {outage} should match the result"
        )
    # Check basecase
    assert (
        convergence_df.loc[(timestep, basecase_name), "status"]
        == POWSYBL_CONVERGENCE_MAP[res.pre_contingency_result.status.value]
    ), "Basecase convergence should match the pre-contingency result"
    assert np.all(convergence_df.index.get_level_values("timestep") == timestep), "All rows should have the same timestep"
    basecase_name = None
    convergence_df_without_basecase, failed_outages = get_convergence_result_df(
        res.post_contingency_results,
        res.pre_contingency_result,
        outages=net.get_branches().index.tolist(),
        basecase_name=basecase_name,
        timestep=timestep,
    )
    assert len(convergence_df_without_basecase) == len(outages), (
        "The convergence dataframe should have one row for each outage and no base case"
    )


def test_get_node_results_dc():
    bus_results = pd.DataFrame()
    bus_results["contingency_id"] = ["contingency_1", "contingency_1", "contingency_2", "contingency_2"]
    bus_results["operator_strategy_id"] = ""
    bus_results["voltage_level_id"] = ["VL_1"] * 4
    bus_results["bus_id"] = ["bus_1", "bus_2", "bus_1", "bus_2"]
    bus_results["v_mag"] = [10.0, 10.0, 10.0, np.nan]
    bus_results["v_angle"] = [180.0, 0, 10, np.nan]
    bus_results.set_index(["contingency_id", "operator_strategy_id", "voltage_level_id", "bus_id"], inplace=True)

    voltage_levels = pd.DataFrame(
        {
            "id": ["VL_1", "VL_2"],
            "nominal_v": [10.0, 20.0],
            "high_voltage_limit": [11.0, 22.0],
            "low_voltage_limit": [9.0, 18.0],
        }
    )
    voltage_levels.set_index("id", inplace=True)

    monitored_buses = ["bus_1", "bus_2"]
    busmap = pd.DataFrame(index=monitored_buses, data={"bus_breaker_bus_id": monitored_buses})
    failed_outages = ["contingency_3"]  # This wont show up in the results
    timestep = 0
    method = "dc"
    node_results = get_node_results(bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method)
    assert len(node_results) == 6, "There should be 6 rows in the node results (2 for each contingency)"
    assert all(node_results.loc[node_results.index.get_level_values("timestep") == timestep]), (
        "All rows should have the same timestep"
    )
    assert all(node_results.loc[node_results.index.get_level_values("element") == failed_outages[0]].isna()), (
        "All rows for failed outages should be NaN"
    )
    for (contingency, _op_strat, _voltage_level, bus_id), row in bus_results.iterrows():
        if bus_id in monitored_buses:
            vm_result = node_results.loc[(timestep, contingency, bus_id), "vm"]
            orig_vm = row["v_mag"]
            voltage = voltage_levels.loc["VL_1", "nominal_v"]
            assert vm_result == voltage or (np.isnan(vm_result) and np.isnan(orig_vm)), (
                f"Voltage magnitude for {bus_id} in {contingency} in {method} should match"
            )
            vm_loading = node_results.loc[(timestep, contingency, bus_id), "vm_loading"]
            assert vm_loading == 0.0 or (np.isnan(vm_loading) and np.isnan(orig_vm)), (
                f"Voltage magnitude loading for {bus_id} in {contingency} in {method} should be 0"
            )
            va_result = node_results.loc[(timestep, contingency, bus_id), "va"]
            orig_va = row["v_angle"]
            assert va_result == orig_va or (np.isnan(va_result) and np.isnan(orig_va)), (
                f"Voltage angle for {bus_id} in {contingency} in {method} should match"
            )
        else:
            assert bus_id not in node_results.index.get_level_values("element"), (
                f"Bus {bus_id} should not be in the node results if it is not monitored"
            )
    monitored_buses = ["bus_1"]
    node_results = get_node_results(bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method)
    assert len(node_results) == 3, (
        "There should be 3 rows in the node results (1 for each contingency), since only one bus is monitored"
    )


def test_get_node_results_ac():
    bus_results = pd.DataFrame()
    bus_results["contingency_id"] = ["contingency_1", "contingency_1", "contingency_2", "contingency_2"]
    bus_results["operator_strategy_id"] = ""
    bus_results["voltage_level_id"] = ["VL_1"] * 4
    bus_results["bus_id"] = ["bus_1", "bus_2", "bus_1", "bus_2"]
    bus_results["v_mag"] = [10.0, 11.0, 9.0, np.nan]
    bus_results["v_angle"] = [180.0, 0, 10, np.nan]
    bus_results.set_index(["contingency_id", "operator_strategy_id", "voltage_level_id", "bus_id"], inplace=True)

    voltage_levels = pd.DataFrame(
        {
            "id": ["VL_1", "VL_2"],
            "nominal_v": [10.0, 20.0],
            "high_voltage_limit": [11.0, 22.0],
            "low_voltage_limit": [9.0, 18.0],
        }
    )
    voltage_levels.set_index("id", inplace=True)

    monitored_buses = ["bus_1", "bus_2"]
    failed_outages = ["contingency_3"]  # This wont show up in the results
    timestep = 0
    method = "ac"
    busmap = pd.DataFrame(index=monitored_buses, data={"bus_breaker_bus_id": monitored_buses})
    node_results = get_node_results(bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method)
    assert len(node_results) == 6, "There should be 6 rows in the node results (2 for each contingency)"
    assert all(node_results.loc[node_results.index.get_level_values("timestep") == timestep]), (
        "All rows should have the same timestep"
    )
    assert all(node_results.loc[node_results.index.get_level_values("element") == failed_outages[0]].isna()), (
        "All rows for failed outages should be NaN"
    )
    for (contingency, _op_strat, _voltage_level, bus_id), row in bus_results.iterrows():
        if bus_id in monitored_buses:
            # Test voltage magnitude
            vm_result = node_results.loc[(timestep, contingency, bus_id), "vm"]
            orig_vm = row["v_mag"]
            assert vm_result == orig_vm or (np.isnan(vm_result) and np.isnan(orig_vm)), (
                f"Voltage magnitude for {bus_id} in {contingency} in {method} should match"
            )

            # Test voltage magnitude loading
            vm_loading = node_results.loc[(timestep, contingency, bus_id), "vm_loading"]
            nominal_v = voltage_levels.loc["VL_1", "nominal_v"]
            if np.isnan(vm_loading):
                assert np.isnan(orig_vm), "Loading should only by NaN if the original voltage is NaN"
            elif vm_loading > nominal_v:
                voltage_max = voltage_levels.loc["VL_1", "high_voltage_limit"]
                assert vm_loading == (vm_result - nominal_v) / (voltage_max - nominal_v), (
                    f"Voltage loading for {bus_id} in {contingency} in {method} should match"
                )
            elif vm_loading == nominal_v:
                assert vm_loading == 0.0, "Loading should be 0 if the voltage is equal to the nominal voltage"
            else:
                voltage_min = voltage_levels.loc["VL_1", "low_voltage_limit"]
                assert vm_loading == (vm_result - nominal_v) / (nominal_v - voltage_min), (
                    f"Voltage loading for {bus_id} in {contingency} in {method} should match"
                )

            # Test angle
            va_result = node_results.loc[(timestep, contingency, bus_id), "va"]
            orig_va = row["v_angle"]
            assert va_result == orig_va or (np.isnan(va_result) and np.isnan(orig_va)), (
                f"Voltage angle for {bus_id} in {contingency} in {method} should match"
            )
        else:
            assert bus_id not in node_results.index.get_level_values("element"), (
                f"Bus {bus_id} should not be in the node results if it is not monitored"
            )
    monitored_buses = ["bus_1"]
    node_results = get_node_results(bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method)
    assert len(node_results) == 3, (
        "There should be 3 rows in the node results (1 for each contingency), since only one bus is monitored"
    )


def test_update_basename_with_new_name():
    # Test with a valid base case name
    base_case_name = "BASECASE"
    timestep = 0
    contingency = ""
    element = "test_element"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df.loc[(timestep, contingency, element), "vm"] = 2.0
    updated_df = update_basename(node_df, base_case_name)
    assert node_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"
    assert updated_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"

    branch_df = get_empty_dataframe_from_model(BranchResultSchema)
    branch_df.loc[(timestep, contingency, element, 1), "p"] = 2.0
    updated_df = update_basename(branch_df, base_case_name)
    assert branch_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"
    assert updated_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"

    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    va_diff_df.loc[(timestep, contingency, element), "va_diff"] = 5.0
    updated_df = update_basename(va_diff_df, base_case_name)
    assert va_diff_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"
    assert updated_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df.loc[(timestep, contingency, element), "vm"] = 2.0
    node_df.loc[(timestep, contingency, element + "_2"), "vm"] = 2.0
    node_df.loc[(timestep, "OTHER_CONTINGENCY", element), "vm"] = 2.0

    updated_df = update_basename(node_df, base_case_name)
    assert node_df.index.get_level_values("contingency")[0] == "BASECASE", "The contingency should be updated to BASECASE"
    assert node_df.index.get_level_values("contingency")[1] == "BASECASE", "The contingency should be updated to BASECASE"
    assert node_df.index.get_level_values("contingency")[2] == "OTHER_CONTINGENCY", (
        "The non-basecase contingency should not be updated to BASECASE"
    )

    empty_df = get_empty_dataframe_from_model(NodeResultSchema)
    updated_empty_df = update_basename(empty_df, base_case_name)
    assert empty_df.empty, "The empty dataframe should remain empty"
    assert updated_empty_df.empty, "The updated empty dataframe should remain empty"


def test_update_basename_drops():
    base_case_name = None
    timestep = 0
    contingency = ""
    element = "test_element"
    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df.loc[(timestep, contingency, element), "vm"] = 2.0
    updated_df = update_basename(node_df, base_case_name)
    assert len(updated_df) == 0, "The dataframe should be empty when base_case_name is None"
    assert len(node_df) == 0, "The original dataframe should also be empty when base_case_name is None"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df.loc[(timestep, contingency, element), "vm"] = 2.0
    node_df.loc[(timestep, contingency, element + "_2"), "vm"] = 2.0
    node_df.loc[(timestep, "OTHER_CONTINGENCY", element), "vm"] = 2.0

    updated_df = update_basename(node_df, base_case_name)

    assert len(updated_df) == 1, "Only the non-basecase contingency should remain"
    assert len(node_df) == 1, "Only the non-basecase contingency should remain in the original dataframe"

    empty_df = get_empty_dataframe_from_model(NodeResultSchema)
    updated_empty_df = update_basename(empty_df, base_case_name)
    assert empty_df.empty, "The empty dataframe should remain empty"
    assert updated_empty_df.empty, "The updated empty dataframe should remain empty"


def test_translate_element_names():
    timestep = 0
    contingency = "test_contingency"
    element_id = "to_be_translated"
    element_name = "translated_element"
    element_mapping = {element_id: element_name, "another_element": "another_translated_element"}

    node_df = get_empty_dataframe_from_model(NodeResultSchema)

    # Test with empty
    updated_df = add_name_column(node_df, element_mapping, index_level="element")
    assert updated_df.empty, "The dataframe should remain empty when no elements are present"

    node_df.loc[(timestep, contingency, element_id), "vm"] = 2.0

    updated_df = add_name_column(node_df, element_mapping, index_level="element")
    assert updated_df.index.get_level_values("element")[0] == element_id, "The index should still be as before"
    assert node_df.index.get_level_values("element")[0] == element_id, "The original index should still be as before"
    assert updated_df.loc[(timestep, contingency, element_id), "element_name"] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert node_df.loc[(timestep, contingency, element_id), "element_name"] == element_name, (
        "The original element_name column should contain the translated name"
    )

    # Adding another column without entry
    missing_element_id = "missing"
    node_df.loc[(timestep, contingency, missing_element_id), "vm"] = 2.0
    updated_df = add_name_column(node_df, element_mapping, index_level="element")
    assert updated_df.loc[(timestep, contingency, missing_element_id), "element_name"] == "", (
        "The map does not contain the key, so the name should be empty ('')"
    )
    assert node_df.loc[(timestep, contingency, missing_element_id), "element_name"] == "", (
        "The map does not contain the key, so the name should be empty ('')"
    )

    branch_df = get_empty_dataframe_from_model(BranchResultSchema)
    branch_df.loc[(timestep, contingency, element_id, 1), "p"] = 2.0
    updated_branch_df = add_name_column(branch_df, element_mapping, index_level="element")
    assert updated_branch_df.index.get_level_values("element")[0] == element_id, "The index should still be as before"
    assert branch_df.index.get_level_values("element")[0] == element_id, "The original index should still be as before"
    assert updated_branch_df.loc[(timestep, contingency, element_id, 1), "element_name"] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert branch_df.loc[(timestep, contingency, element_id, 1), "element_name"] == element_name, (
        "The original element_name column should contain the translated name"
    )

    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    va_diff_df.loc[(timestep, contingency, element_id), "va_diff"] = 5.0
    updated_va_diff_df = add_name_column(va_diff_df, element_mapping, index_level="element")
    assert updated_va_diff_df.index.get_level_values("element")[0] == element_id, "The index should still be as before"
    assert va_diff_df.index.get_level_values("element")[0] == element_id, "The original index should still be as before"
    assert updated_va_diff_df.loc[(timestep, contingency, element_id), "element_name"] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert va_diff_df.loc[(timestep, contingency, element_id), "element_name"] == element_name, (
        "The original element_name column should contain the translated name"
    )


def test_set_target_values_to_lf_values_incl_distributed_slack_dc(
    powsybl_bus_breaker_net: pypowsybl.network.Network,
) -> None:
    pypowsybl.loadflow.run_dc(powsybl_bus_breaker_net, DISTRIBUTED_SLACK)
    generators_before = powsybl_bus_breaker_net.get_generators()
    assert not np.all(generators_before.p == generators_before.target_p), (
        "Make sure the initial target values are different from the loadflow values. Otherwise this test is useless."
    )

    powsybl_bus_breaker_net = set_target_values_to_lf_values_incl_distributed_slack(powsybl_bus_breaker_net, "dc")
    lf_result = pypowsybl.loadflow.run_dc(powsybl_bus_breaker_net, DISTRIBUTED_SLACK)
    assert lf_result[0].status == pypowsybl.loadflow.ComponentStatus.CONVERGED, (
        "Loadflow did not converge after setting the target values."
    )
    generators_after = powsybl_bus_breaker_net.get_generators()
    p_not_nan = ~generators_after.p.isna()
    assert np.allclose(generators_after.loc[p_not_nan].target_p, -generators_after.loc[p_not_nan].p), (
        "Target p was not set to the loadflow p values."
    )
    assert np.allclose(generators_before[["p", "q"]], generators_after[["p", "q"]], equal_nan=True), (
        "The loadflow p values changed after resetting the target values."
    )


def test_set_target_values_to_lf_values_incl_distributed_slack_ac(
    powsybl_bus_breaker_net: pypowsybl.network.Network,
) -> None:
    pypowsybl.loadflow.run_ac(powsybl_bus_breaker_net, DISTRIBUTED_SLACK)
    generators_before = powsybl_bus_breaker_net.get_generators(all_attributes=True)
    assert not np.all(generators_before.p == generators_before.target_p), (
        "Make sure the initial p-target values are different from the loadflow values. Otherwise this test is useless."
    )
    assert not np.all(generators_before.q == generators_before.target_q), (
        "Make sure the initial q-target values are different from the loadflow values. Otherwise this test is useless."
    )

    powsybl_bus_breaker_net = set_target_values_to_lf_values_incl_distributed_slack(powsybl_bus_breaker_net, "ac")
    lf_result = pypowsybl.loadflow.run_ac(powsybl_bus_breaker_net, DISTRIBUTED_SLACK)
    assert lf_result[0].status == pypowsybl.loadflow.ComponentStatus.CONVERGED, (
        "Loadflow did not converge after setting the target values."
    )

    generators_after = powsybl_bus_breaker_net.get_generators(all_attributes=True)
    p_not_nan = ~generators_after.p.isna()
    assert np.allclose(generators_after.loc[p_not_nan].target_p, -generators_after.loc[p_not_nan].p), (
        "Target p was not set to the loadflow p values."
    )
    q_not_nan = ~generators_after.q.isna()
    assert np.allclose(generators_after.loc[q_not_nan].q, -generators_after.loc[q_not_nan].target_q), (
        "Target q was not set to the loadflow p values."
    )
    assert np.allclose(generators_before[["p", "q"]], generators_after[["p", "q"]], equal_nan=True), (
        "The loadflow values changed after resetting the target values."
    )
