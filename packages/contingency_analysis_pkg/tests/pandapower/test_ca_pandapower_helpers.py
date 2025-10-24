from copy import deepcopy

import numpy as np
import pandapower as pp
import pandas as pd
import pandera.typing as pat
import pytest
from toop_engine_contingency_analysis.pandapower import (
    PandapowerContingency,
    PandapowerElements,
    PandapowerMonitoredElementSchema,
    PandapowerNMinus1Definition,
    extract_contingencies_with_cgmes_id,
    extract_monitored_elements_with_cgmes_id,
    get_branch_results,
    get_convergence_df,
    get_failed_va_diff_results,
    get_node_result_df,
    get_regulating_element_results,
    get_va_diff_results,
    translate_contingencies,
    translate_monitored_elements,
    translate_nminus1_for_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import VADiffInfo, match_node_to_next_switch_type
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import BranchSide
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


@pytest.fixture
def n_1_definition_unique_pp_id(pandapower_net: pp.pandapowerNet) -> Nminus1Definition:
    contingencies = [
        Contingency(
            id=get_globally_unique_id(id, "line"),
            name=f"contingency_{id}",
            elements=[GridElement(id=get_globally_unique_id(id, "line"), type="line", name=f"line_{id}", kind="branch")],
        )
        for id in pandapower_net.line.index
    ]
    monitored_elements = [
        GridElement(id=get_globally_unique_id(id, "line"), type="line", name="id", kind="branch")
        for id in pandapower_net.line.index
    ]
    nminus1_definition = Nminus1Definition(
        contingencies=contingencies, monitored_elements=monitored_elements, id_type="unique_pandapower"
    )
    return nminus1_definition


def test_panda_power_n_minus_1_definition():
    contingencies = [
        PandapowerContingency(
            unique_id="contingency_1",
            name="contingency_1",
            elements=[PandapowerElements(unique_id="element_1", table_id=1, table="line", name="", type="line")],
        ),
        PandapowerContingency(
            unique_id="basecase",
            name="basecase",
            elements=[],
        ),
        PandapowerContingency(
            unique_id="contingency_2",
            name="contingency_2",
            elements=[PandapowerElements(unique_id="element_2", table_id=2, table="line", name="", type="transformer")],
        ),
    ]
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    missing_elements = [GridElement(id="not_existing_id", type="line", name="not_existing_name", kind="branch")]
    missing_contingencies = [Contingency(id="contingency_2", name="contingency_2", elements=missing_elements)]

    nminus1_definition = PandapowerNMinus1Definition(
        contingencies=contingencies,
        monitored_elements=monitored_elements,
        missing_elements=missing_elements,
        missing_contingencies=missing_contingencies,
    )

    # get by string
    sliced_nminus1 = nminus1_definition["basecase"]
    assert sliced_nminus1.contingencies[0].unique_id == "basecase", "The id of the contingency should match the key"
    assert len(sliced_nminus1.contingencies) == 1, "Only one contingency should be returned"
    assert sliced_nminus1.monitored_elements.equals(nminus1_definition.monitored_elements), (
        "The monitored elements should match the original definition"
    )
    assert sliced_nminus1.missing_elements == nminus1_definition.missing_elements, (
        "The missing elements should match the original definition"
    )
    assert sliced_nminus1.missing_contingencies == nminus1_definition.missing_contingencies, (
        "The missing contingencies should match the original definition"
    )

    # by index
    sliced_nminus1 = nminus1_definition[0]
    assert sliced_nminus1.contingencies[0].unique_id == "contingency_1", "The id of the contingency should match the key"
    assert len(sliced_nminus1.contingencies) == 1, "Only one contingency should be returned"
    assert sliced_nminus1.monitored_elements.equals(nminus1_definition.monitored_elements), (
        "The monitored elements should match the original definition"
    )
    assert sliced_nminus1.missing_elements == nminus1_definition.missing_elements, (
        "The missing elements should match the original definition"
    )
    assert sliced_nminus1.missing_contingencies == nminus1_definition.missing_contingencies, (
        "The missing contingencies should match the original definition"
    )

    # by slice
    sliced_nminus1 = nminus1_definition[1:]
    assert len(sliced_nminus1.contingencies) == 2, "Two contingencies should be returned"
    assert sliced_nminus1.contingencies[0].unique_id == "basecase", "The id of the first contingency should match the key"
    assert sliced_nminus1.contingencies[1].unique_id == "contingency_2", (
        "The id of the second contingency should match the key"
    )
    assert sliced_nminus1.monitored_elements.equals(nminus1_definition.monitored_elements), (
        "The monitored elements should match the original definition"
    )
    assert sliced_nminus1.missing_elements == nminus1_definition.missing_elements, (
        "The missing elements should match the original definition"
    )
    assert sliced_nminus1.missing_contingencies == nminus1_definition.missing_contingencies, (
        "The missing contingencies should match the original definition"
    )

    with pytest.raises(KeyError):
        _ = nminus1_definition["non_existing_key"], "Accessing a non-existing key should raise a KeyError"


def test_translate_monitored_elements(pandapower_net: pp.pandapowerNet):
    monitored_elements = [
        GridElement(id=get_globally_unique_id(id, "line"), type="line", name="id", kind="branch")
        for id in pandapower_net.line.index
    ]
    non_existent_lines = [
        GridElement(id=f"non_existent_line_{i}", type="line", name=f"non_existent_line_{i}", kind="branch") for i in range(5)
    ]

    translated_elements, missing_elements, _duplicated_ids = translate_monitored_elements(
        pandapower_net, monitored_elements + non_existent_lines
    )
    assert len(translated_elements) == len(monitored_elements), (
        "The number of translated elements should match the number of monitored elements"
    )
    assert missing_elements == non_existent_lines, "The missing elements should match the non-existent lines"
    assert translated_elements.index.to_list() == [elem.id for elem in monitored_elements], (
        "The translated elements should match the monitored elements"
    )
    assert all(translated_elements.table == "line"), "All translated elements should be of type 'line'"
    assert translated_elements.table_id.to_list() == pandapower_net.line.index.to_list(), (
        "Table IDs should match the line index"
    )


def test_translate_contingencies(pandapower_net: pp.pandapowerNet, n_1_definition_unique_pp_id: Nminus1Definition):
    contingencies = n_1_definition_unique_pp_id.contingencies
    non_existent_lines = [
        Contingency(
            id=f"non_existent_line_{i}",
            name=f"non_existent_line_{i}",
            elements=[GridElement(id=f"non_existent_line_{i}", type="line", name=f"non_existent_line_{i}", kind="branch")],
        )
        for i in range(5)
    ]

    translated_contingencies, missing_elements, duplicated_ids = translate_contingencies(
        pandapower_net, contingencies + non_existent_lines
    )
    assert len(translated_contingencies) == len(contingencies), (
        "The number of translated contingencies should match the number of input contingencies"
    )
    assert missing_elements == non_existent_lines, "The missing elements should match the non-existent lines"
    original_contingency_ids = [contingency.id for contingency in contingencies]
    translated_ids = [contingency.unique_id for contingency in translated_contingencies]
    assert original_contingency_ids == translated_ids, "The translated contingency IDs should match the original IDs"
    original_element_ids = [element.id for contingency in contingencies for element in contingency.elements]
    translated_element_ids = [
        element.unique_id for contingency in translated_contingencies for element in contingency.elements
    ]
    assert original_element_ids == translated_element_ids, "The translated element IDs should match the original element IDs"


def test_translate_nminus1_for_pandapower(pandapower_net: pp.pandapowerNet, n_1_definition_unique_pp_id: Nminus1Definition):
    contingencies = n_1_definition_unique_pp_id.contingencies
    monitored_elements = n_1_definition_unique_pp_id.monitored_elements

    translated_nminus1 = translate_nminus1_for_pandapower(n_1_definition_unique_pp_id, pandapower_net)
    assert isinstance(translated_nminus1, PandapowerNMinus1Definition), (
        "The translated definition should be of type PandapowerNMinus1Definition"
    )
    assert len(translated_nminus1.contingencies) == len(contingencies), (
        "The number of translated contingencies should match the number of input contingencies"
    )
    assert len(translated_nminus1.monitored_elements) == len(monitored_elements), (
        "The number of translated monitored elements should match the number of input monitored elements"
    )

    nminus1_definition = Nminus1Definition(contingencies=[], monitored_elements=[], id_type="unique_pandapower")
    translated_nminus1 = translate_nminus1_for_pandapower(nminus1_definition, pandapower_net)
    assert isinstance(translated_nminus1, PandapowerNMinus1Definition), (
        "The translated definition should be of type PandapowerNMinus1Definition"
    )
    assert len(translated_nminus1.contingencies) == 0, "No contingencies should be translated"
    assert len(translated_nminus1.monitored_elements) == 0, "No monitored elements should be translated"


def test_get_branch_results(pandapower_net: pp.pandapowerNet, n_1_definition_unique_pp_id: Nminus1Definition):
    translated_nminus1 = translate_nminus1_for_pandapower(n_1_definition_unique_pp_id, pandapower_net)
    contingency = translated_nminus1.contingencies[0]
    monitored_elements = translated_nminus1.monitored_elements

    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[outage_net.line.index[:1], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    branch_results = get_branch_results(outage_net, contingency, monitored_elements, timestep=timestep)
    assert isinstance(branch_results, pd.DataFrame), "The result should be a DataFrame"
    assert all(branch_results.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(branch_results.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert (
        branch_results.loc[:, :, :, BranchSide.ONE.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert (
        branch_results.loc[:, :, :, BranchSide.TWO.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].p.values == outage_net.res_line.p_from_mw.values), (
        "Active power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].p.values == outage_net.res_line.p_to_mw.values), (
        "Active power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].q.values == outage_net.res_line.q_from_mvar.values), (
        "Reactive power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].q.values == outage_net.res_line.q_to_mvar.values), (
        "Reactive power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].i.values == outage_net.res_line.i_from_ka.values * 1000), (
        "Current from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].i.values == outage_net.res_line.i_to_ka.values * 1000), (
        "Current to side should match the outage net"
    )
    assert all(
        branch_results.loc[:, :, :, BranchSide.ONE.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading from side should match the outage net"
    assert all(
        branch_results.loc[:, :, :, BranchSide.TWO.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading to side should match the outage net"


def test_get_branch_results_basecase(pandapower_net: pp.pandapowerNet, n_1_definition_unique_pp_id: Nminus1Definition):
    translated_nminus1 = translate_nminus1_for_pandapower(n_1_definition_unique_pp_id, pandapower_net)
    contingency = PandapowerContingency(unique_id="BASECASE", name="BASECASE", elements=[])
    monitored_elements = translated_nminus1.monitored_elements
    timestep = 1
    outage_net = deepcopy(pandapower_net)
    pp.runpp(outage_net)
    branch_results = get_branch_results(outage_net, contingency, monitored_elements, timestep=timestep)
    assert isinstance(branch_results, pd.DataFrame), "The result should be a DataFrame"
    assert all(branch_results.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(branch_results.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert (
        branch_results.loc[:, :, :, BranchSide.ONE.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert (
        branch_results.loc[:, :, :, BranchSide.TWO.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].p.values == outage_net.res_line.p_from_mw.values), (
        "Active power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].p.values == outage_net.res_line.p_to_mw.values), (
        "Active power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].q.values == outage_net.res_line.q_from_mvar.values), (
        "Reactive power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].q.values == outage_net.res_line.q_to_mvar.values), (
        "Reactive power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].i.values == outage_net.res_line.i_from_ka.values * 1000), (
        "Current from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].i.values == outage_net.res_line.i_to_ka.values * 1000), (
        "Current to side should match the outage net"
    )
    assert all(
        branch_results.loc[:, :, :, BranchSide.ONE.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading from side should match the outage net"
    assert all(
        branch_results.loc[:, :, :, BranchSide.TWO.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading to side should match the outage net"


def test_get_branch_results_multi_outage(pandapower_net: pp.pandapowerNet, n_1_definition_unique_pp_id: Nminus1Definition):
    translated_nminus1 = translate_nminus1_for_pandapower(n_1_definition_unique_pp_id, pandapower_net)
    monitored_elements = translated_nminus1.monitored_elements

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="line", name="", type="line")
            for id in pandapower_net.line.index[:2]
        ],
    )
    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[pandapower_net.line.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    branch_results = get_branch_results(outage_net, contingency, monitored_elements, timestep=timestep)

    assert isinstance(branch_results, pd.DataFrame), "The result should be a DataFrame"
    assert all(branch_results.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(branch_results.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert (
        branch_results.loc[:, :, :, BranchSide.ONE.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert (
        branch_results.loc[:, :, :, BranchSide.TWO.value].index.get_level_values("element").tolist()
        == monitored_elements.index.tolist()
    ), "Element IDs should match monitored elements"
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].p.values == outage_net.res_line.p_from_mw.values), (
        "Active power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].p.values == outage_net.res_line.p_to_mw.values), (
        "Active power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].q.values == outage_net.res_line.q_from_mvar.values), (
        "Reactive power from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].q.values == outage_net.res_line.q_to_mvar.values), (
        "Reactive power to side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.ONE.value].i.values == outage_net.res_line.i_from_ka.values * 1000), (
        "Current from side should match the outage net"
    )
    assert all(branch_results.loc[:, :, :, BranchSide.TWO.value].i.values == outage_net.res_line.i_to_ka.values * 1000), (
        "Current to side should match the outage net"
    )
    assert all(
        branch_results.loc[:, :, :, BranchSide.ONE.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading from side should match the outage net"
    assert all(
        branch_results.loc[:, :, :, BranchSide.TWO.value].loading.values == outage_net.res_line.loading_percent.values / 100
    ), "Loading to side should match the outage net"


def test_get_branch_results_no_monitored(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="line", name="", type="line")
            for id in lines.index[:2]
        ],
    )
    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    branch_results = get_branch_results(outage_net, contingency, monitored_elements, timestep=timestep)
    assert branch_results.empty, "The result should be empty if no monitored elements are provided"


@pytest.fixture
def monitored_buses(pandapower_net: pp.pandapowerNet) -> pat.DataFrame[PandapowerMonitoredElementSchema]:
    buses = pandapower_net.bus
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    for node_id, row in buses.iterrows():
        monitored_elements.loc[get_globally_unique_id(node_id, "bus"), ["table", "table_id", "kind", "name"]] = (
            "bus",
            node_id,
            "bus",
            row.name,
        )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)
    return monitored_elements


def test_get_node_results(
    pandapower_net: pp.pandapowerNet, monitored_buses: pat.DataFrame[PandapowerMonitoredElementSchema]
):
    lines = pandapower_net.line

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="bus", name="", type="line")
            for id in lines.index[:1]
        ],
    )
    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:1], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    node_result_df = get_node_result_df(outage_net, contingency, monitored_buses, timestep=timestep)
    assert isinstance(node_result_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(node_result_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(node_result_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert node_result_df.index.get_level_values("element").tolist() == monitored_buses.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert node_result_df.vm.tolist() == (outage_net.res_bus.vm_pu * outage_net.bus.vn_kv).tolist(), (
        "Voltage magnitudes should match the outage net"
    )
    assert node_result_df.va.tolist() == outage_net.res_bus.va_degree.tolist(), "Voltage angles should match the outage net"


def test_get_node_results_basecase(
    pandapower_net: pp.pandapowerNet, monitored_buses: pat.DataFrame[PandapowerMonitoredElementSchema]
):
    contingency = PandapowerContingency(
        unique_id="base",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 1

    outage_net = deepcopy(pandapower_net)
    pp.runpp(outage_net)
    node_result_df = get_node_result_df(outage_net, contingency, monitored_buses, timestep=timestep)
    assert isinstance(node_result_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(node_result_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(node_result_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert node_result_df.index.get_level_values("element").tolist() == monitored_buses.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert node_result_df.vm.tolist() == (outage_net.res_bus.vm_pu * outage_net.bus.vn_kv).tolist(), (
        "Voltage magnitudes should match the outage net"
    )
    assert node_result_df.va.tolist() == outage_net.res_bus.va_degree.tolist(), "Voltage angles should match the outage net"


def test_get_node_results_multi_outage(
    pandapower_net: pp.pandapowerNet, monitored_buses: pat.DataFrame[PandapowerMonitoredElementSchema]
):
    buses = pandapower_net.bus
    lines = pandapower_net.line

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="bus", name="", type="line")
            for id in lines.index[:2]
        ],
    )
    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    node_result_df = get_node_result_df(outage_net, contingency, monitored_buses, timestep=timestep)
    assert isinstance(node_result_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(node_result_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(node_result_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert node_result_df.index.get_level_values("element").tolist() == monitored_buses.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert node_result_df.vm.tolist() == (outage_net.res_bus.vm_pu * outage_net.bus.vn_kv).tolist(), (
        "Voltage magnitudes should match the outage net"
    )
    assert node_result_df.va.tolist() == outage_net.res_bus.va_degree.tolist(), "Voltage angles should match the outage net"


def test_get_node_results_no_monitored(pandapower_net: pp.pandapowerNet):
    buses = pandapower_net.bus
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="bus", name="", type="line")
            for id in buses.index[:2]
        ],
    )
    timestep = 0

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[buses.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)
    node_result_df = get_node_result_df(outage_net, contingency, monitored_elements, timestep=timestep)
    assert node_result_df.empty, "The result should be empty if no monitored elements are provided"


def test_get_convergence_df():
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 0
    convergence_df = get_convergence_df(timestep, contingency, status="CONVERGED")
    assert isinstance(convergence_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(convergence_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(convergence_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert all(convergence_df.status == "CONVERGED"), "Status should be 'CONVERGED'"

    timestep = 1
    convergence_df = get_convergence_df(timestep, contingency, status="NO_CALCULATION")
    assert isinstance(convergence_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(convergence_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(convergence_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert all(convergence_df.status == "NO_CALCULATION"), "Status should be 'NO_CALCULATION'"


def test_get_failed_va_diff_results(pandapower_net: pp.pandapowerNet):
    contingency = PandapowerContingency(unique_id="contingency_1", name="contingency_1_name", elements=[], va_diff_info=[])
    timestep = 0
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    for i in range(3):
        monitored_elements.loc[get_globally_unique_id(i, "switch"), ["table", "table_id", "kind", "name"]] = (
            "switch",
            i,
            "switch",
            f"Switch {i}",
        )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    failed_va_diff_df = get_failed_va_diff_results(timestep, monitored_elements, contingency)
    assert isinstance(failed_va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(failed_va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(failed_va_diff_df.index.get_level_values("contingency") == contingency.unique_id), (
        "Contingency ID should match"
    )
    assert failed_va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert failed_va_diff_df.va_diff.isna().all(), "All VA differences should be NaN for failed results"

    no_monitored_switch_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    failed_va_diff_df_no_monitored = get_failed_va_diff_results(timestep, no_monitored_switch_elements, contingency)
    assert failed_va_diff_df_no_monitored.empty, "The result should be empty if no monitored elements are provided"
    va_diff_info = VADiffInfo(
        from_bus=pandapower_net.line.loc[1, "from_bus"],
        to_bus=pandapower_net.line.loc[1, "to_bus"],
        power_switches_from={"PW_SWITCH_ID1": "PW_SWITCH_NAME1"},
        power_switches_to={"PW_SWITCH_ID2": "PW_SWITCH_NAME2"},
    )
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[PandapowerElements(unique_id=get_globally_unique_id(1, "line"), table_id=1, table="line", name="")],
        va_diff_info=[va_diff_info],
    )
    failed_va_diff = get_failed_va_diff_results(timestep, no_monitored_switch_elements, contingency)
    assert failed_va_diff.index.get_level_values("element").tolist() == ["PW_SWITCH_ID1", "PW_SWITCH_ID2"]

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[PandapowerElements(unique_id=get_globally_unique_id(1, "trafo3w"), table_id=1, table="trafo3w", name="")],
    )
    failed_va_diff = get_failed_va_diff_results(timestep, no_monitored_switch_elements, contingency)
    assert failed_va_diff_df.va_diff.isna().all(), "Trafo3w outage elements should be nan for now"


def test_get_va_diff_results(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    outaged_line_id = 1
    va_diff_info = VADiffInfo(
        from_bus=pandapower_net.line.loc[outaged_line_id, "from_bus"],
        to_bus=pandapower_net.line.loc[outaged_line_id, "to_bus"],
        power_switches_from={"PW_SWITCH_ID1": "PW_SWITCH_NAME1"},
        power_switches_to={"PW_SWITCH_ID2": "PW_SWITCH_NAME2"},
    )
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(
                unique_id=get_globally_unique_id(outaged_line_id, "line"), table_id=outaged_line_id, table="line", name=""
            )
        ],
        va_diff_info=[va_diff_info],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[outaged_line_id, "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + list(
        va_diff_info.power_switches_from.keys()
    ) + list(va_diff_info.power_switches_to.keys()), "Element IDs should match monitored elements + the outaged line"
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree,
        -1
        * (outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree),
    ], "VA differences should match the outage net"

    # Test what happens if there is only one switch
    contingency.va_diff_info[0].power_switches_to = {}
    contingency.va_diff_info[0].power_switches_from = {"PW_SWITCH_ID1": "PW_SWITCH_NAME1"}
    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + ["PW_SWITCH_ID1"], (
        "Element IDs should match monitored elements. No line switches since there arent any"
    )
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.loc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.loc[1].to_bus].va_degree,
    ], "VA differences should match the outage net"

    # Test what happens if there are no switches
    contingency.va_diff_info = []
    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist(), (
        "Element IDs should match monitored elements. No line switches since there arent any"
    )
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
    ], "VA differences should match the outage net"


def test_get_va_diff_results_multioutage(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    va_diff_info_1 = VADiffInfo(
        from_bus=pandapower_net.line.loc[0, "from_bus"],
        to_bus=pandapower_net.line.loc[0, "to_bus"],
        power_switches_from={"PW_SWITCH_ID1": "PW_SWITCH_NAME1"},
        power_switches_to={"PW_SWITCH_ID2": "PW_SWITCH_NAME2"},
    )
    va_diff_info_2 = VADiffInfo(
        from_bus=pandapower_net.line.loc[1, "from_bus"],
        to_bus=pandapower_net.line.loc[1, "to_bus"],
        power_switches_from={"PW_SWITCH_ID3": "PW_SWITCH_NAME3"},
        power_switches_to={"PW_SWITCH_ID4": "PW_SWITCH_NAME4"},
    )

    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "line"), table_id=id, table="line", name="", type="line")
            for id in lines.index[:2]
        ],
        va_diff_info=[va_diff_info_1, va_diff_info_2],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:2], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist() + [
        "PW_SWITCH_ID1",
        "PW_SWITCH_ID2",
        "PW_SWITCH_ID3",
        "PW_SWITCH_ID4",
    ], "Element IDs should match monitored elements + the outaged line"
    assert va_diff_df.va_diff.tolist() == [
        outage_net.res_bus.loc[0].va_degree - outage_net.res_bus.loc[1].va_degree,
        outage_net.res_bus.loc[lines.iloc[0].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[0].to_bus].va_degree,
        -1
        * (
            outage_net.res_bus.loc[lines.iloc[0].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[0].to_bus].va_degree
        ),
        outage_net.res_bus.loc[lines.iloc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[1].to_bus].va_degree,
        -1
        * (
            outage_net.res_bus.loc[lines.iloc[1].from_bus].va_degree - outage_net.res_bus.loc[lines.iloc[1].to_bus].va_degree
        ),
    ], "VA differences should match the outage net"


def test_get_va_diff_results_basecase(pandapower_net: pp.pandapowerNet):
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 0
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    # create a switch
    switch_id = pp.create_switch(net=pandapower_net, bus=0, element=1, et="b", closed=False, name="Switch 0")

    monitored_elements.loc[get_globally_unique_id(int(switch_id), "switch"), ["table", "table_id", "kind", "name"]] = (
        "switch",
        switch_id,
        "switch",
        f"Switch {switch_id}",
    )

    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    monitored_elements.name = monitored_elements.name.astype(str)

    pp.runpp(pandapower_net)

    va_diff_df = get_va_diff_results(pandapower_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert all(va_diff_df.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(va_diff_df.index.get_level_values("contingency") == contingency.unique_id), "Contingency ID should match"
    assert va_diff_df.index.get_level_values("element").tolist() == monitored_elements.index.tolist(), (
        "Element IDs should match monitored elements"
    )
    assert va_diff_df.va_diff.tolist() == [
        pandapower_net.res_bus.loc[0].va_degree - pandapower_net.res_bus.loc[1].va_degree
    ], "VA differences should match the outage net"


def test_get_va_diff_results_no_elements(pandapower_net: pp.pandapowerNet):
    lines = pandapower_net.line
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    outage_net = deepcopy(pandapower_net)
    outage_net.line.loc[lines.index[:1], "in_service"] = False  # Simulate an outage for the branch
    pp.runpp(outage_net)

    va_diff_df = get_va_diff_results(outage_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert va_diff_df.empty, "The result should be empty if no monitored elements are provided"


def test_get_va_diff_results_outage_element_trafo3w(pandapower_net: pp.pandapowerNet):
    contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "trafo3w"), table_id=id, table="trafo3w", name="")
            for id in range(2)
        ],
    )
    timestep = 1
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)

    va_diff_df = get_va_diff_results(pandapower_net, timestep, monitored_elements, contingency)
    assert isinstance(va_diff_df, pd.DataFrame), "The result should be a DataFrame"
    assert va_diff_df.va_diff.isna().all(), "For trafo3w outages, the VA differences should be NaN for now"


def test_get_regulating_element_results():
    monitored_elements = get_empty_dataframe_from_model(PandapowerMonitoredElementSchema)
    for i in range(3):
        monitored_elements.loc[get_globally_unique_id(i, "regulating_element"), ["table", "table_id", "kind", "name"]] = (
            "line",
            i,
            "branch",
            f"Regulating Element {i}",
        )
    monitored_elements.table_id = monitored_elements.table_id.astype(int)
    single_contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "trafo3w"), table_id=id, table="trafo3w", name="")
            for id in range(1)
        ],
    )
    multi_contingency = PandapowerContingency(
        unique_id="contingency_1",
        name="contingency_1_name",
        elements=[
            PandapowerElements(unique_id=get_globally_unique_id(id, "trafo3w"), table_id=id, table="trafo3w", name="")
            for id in range(2)
        ],
    )
    base_case = PandapowerContingency(unique_id="contingency_1", name="contingency_1_name", elements=[])

    timestep = 0

    regulating_element_results = get_regulating_element_results(
        timestep,
        get_empty_dataframe_from_model(PandapowerMonitoredElementSchema),
        base_case,
    )
    assert regulating_element_results.empty, "The result should be empty if no monitored elements are provided"

    regulating_element_results = get_regulating_element_results(
        timestep,
        monitored_elements,
        base_case,
    )
    assert all(regulating_element_results.index.get_level_values("timestep") == timestep), f"Timestep should be {timestep}"
    assert all(regulating_element_results.index.get_level_values("contingency") == base_case.unique_id), (
        "Contingency ID should match"
    )
    assert all(regulating_element_results.index.get_level_values("element").isin(monitored_elements.index)), (
        "Element IDs should match monitored elements"
    )
    assert all(regulating_element_results.value.abs() == 9999.0), (
        "Value should be fake values (9999.0) for regulating elements as it is not implemented yet"
    )

    regulating_element_results = get_regulating_element_results(
        timestep,
        monitored_elements,
        single_contingency,
    )
    assert regulating_element_results.empty, "The result should be empty for single contingency as it is not implemented yet"

    regulating_element_results = get_regulating_element_results(
        timestep,
        monitored_elements,
        multi_contingency,
    )
    assert regulating_element_results.empty, "The result should be empty for multi contingency as it is not implemented yet"


def test_extract_contingencies_with_cgmes_id(pandapower_net: pp.pandapowerNet):
    contingencies = [
        Contingency(id="contingency_1", name="Name of Contingency 1", elements=[]),
        Contingency(
            id="contingency_2",
            name="Name of Contingency 2",
            elements=[GridElement(id="line_1", name="Line 1", type="line", kind="branch")],
        ),
    ]
    translated_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_cgmes_id(
        pandapower_net, contingencies
    )
    assert "origin_id" not in pandapower_net.line.columns, (
        "The 'origin_id' column should not be present in the net for this test"
    )
    assert missing_contingencies == contingencies[1:], (
        "Since there is no origin_id in the net, all contingencies should be missing"
    )
    assert len(translated_contingencies) == 1, (
        "No contingencies should be translated since there are no origin_ids in the net"
    )
    assert len(translated_contingencies[0].elements) == 0, "The translated contingency should be the one without elements"
    assert len(duplicated_ids) == 0, "There should be no duplicated IDs in this case"
    net = deepcopy(pandapower_net)
    net.line.loc[net.line.index[0], "origin_id"] = "line_1"
    translated_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_cgmes_id(net, contingencies)
    assert len(translated_contingencies) == 2, (
        "Two contingencies should be translated since one has an origin_id and the other is the basecase"
    )
    assert len(missing_contingencies) == 0, "No contingencies should be missing"
    assert translated_contingencies[0].unique_id == "contingency_1"
    assert translated_contingencies[0].name == "Name of Contingency 1", "The name of the first contingency should match"
    assert translated_contingencies[0].elements == [], "The first contingency should have no elements"
    assert translated_contingencies[1].unique_id == "contingency_2"
    assert translated_contingencies[1].name == "Name of Contingency 2", "The name of the second contingency should match"
    assert len(translated_contingencies[1].elements) == 1, "The second contingency should have one element"
    assert translated_contingencies[1].elements[0].unique_id == "line_1", (
        "The element ID should match the origin_id in the net"
    )
    assert translated_contingencies[1].elements[0].table == "line", "The element should be of type 'line'"
    assert translated_contingencies[1].elements[0].table_id == 0, "The element should have the correct table_id"
    assert len(duplicated_ids) == 0, "There should be no duplicated IDs in this case"
    net = deepcopy(pandapower_net)
    net.line.loc[net.line.index[0], "origin_id"] = "line_1"
    net.line.loc[net.line.index[1], "origin_id"] = "line_1"
    translated_contingencies, missing_contingencies, duplicated_ids = extract_contingencies_with_cgmes_id(net, contingencies)
    assert len(duplicated_ids) == 1, "There should be one duplicated ID since two lines have the same origin_id"
    assert duplicated_ids[0] == "line_1", "The duplicated ID should be 'line_1'"
    assert len(translated_contingencies) == 2, (
        "Two contingencies should be translated since one has an origin_id and the other is the basecase"
    )
    assert len(missing_contingencies) == 0, "No contingencies should be missing"


def test_extract_monitored_elements_with_cgmes_id(pandapower_net: pp.pandapowerNet):
    monitored_elements = [
        GridElement(id="line_1", name="Line 1", type="line", kind="branch"),
        GridElement(id="trafo_1", name="Trafo 1", type="trafo", kind="branch"),
        GridElement(id="bus_1", name="Bus 1", type="bus", kind="bus"),
    ]

    translated_monitored_elements, missing_elements, duplicated_ids = extract_monitored_elements_with_cgmes_id(
        pandapower_net, monitored_elements
    )
    assert "origin_id" not in pandapower_net.line.columns, (
        "The 'origin_id' column should not be present in the net for this test"
    )
    assert translated_monitored_elements.empty, (
        "No monitored elements should be extracted if there are no origin_ids in the net"
    )
    assert len(missing_elements) == 3, "All monitored elements should be missing since there are no origin_ids in the net"
    assert missing_elements == monitored_elements, "The missing elements should match the original monitored elements"
    assert len(duplicated_ids) == 0, "There should be no duplicated IDs in this case"

    # Now let's add an origin_id to one of the lines
    net = deepcopy(pandapower_net)
    net.line.loc[net.line.index[0], "origin_id"] = "line_1"
    net.trafo.loc[net.trafo.index[1], "origin_id"] = "trafo_1"
    net.bus.loc[net.bus.index[0], "origin_id"] = "bus_1"

    translated_monitored_elements, missing_elements, duplicated_ids = extract_monitored_elements_with_cgmes_id(
        net, monitored_elements
    )
    assert len(translated_monitored_elements) == 3, (
        "One monitored element should be extracted since one line has an origin_id"
    )
    assert translated_monitored_elements.iloc[0]["name"] == "Line 1", (
        "The name of the monitored element should match the line name"
    )
    assert translated_monitored_elements.iloc[0].table == "line", "The monitored element should be of type 'line'"
    assert translated_monitored_elements.iloc[0].table_id == 0, "The monitored element should have the correct table_id"

    assert translated_monitored_elements.iloc[1]["name"] == "Trafo 1", (
        "The name of the monitored element should match the trafo name"
    )
    assert translated_monitored_elements.iloc[1].table == "trafo", "The monitored element should be of type 'trafo'"
    assert translated_monitored_elements.iloc[1].table_id == 1, "The monitored element should have the correct table_id"

    assert translated_monitored_elements.iloc[2]["name"] == "Bus 1", (
        "The name of the monitored element should match the bus name"
    )
    assert translated_monitored_elements.iloc[2].table == "bus", "The monitored element should be of type 'bus'"
    assert translated_monitored_elements.iloc[2].table_id == 0, "The monitored element should have the correct table_id"

    # Now let's add an duplicated origin_id to two of the lines
    net = deepcopy(pandapower_net)
    net.line.loc[net.line.index[0], "origin_id"] = "line_1"
    net.line.loc[net.line.index[1], "origin_id"] = "line_1"

    translated_monitored_elements, missing_elements, duplicated_ids = extract_monitored_elements_with_cgmes_id(
        net, monitored_elements
    )
    assert len(duplicated_ids) == 1, "There should be one duplicated ID since two lines have the same origin_id"
    assert duplicated_ids[0] == "line_1", "The duplicated ID should be 'line_1'"
    assert len(missing_elements) == 2, "Two elements should be missing since only the line has on origin_id"
    assert missing_elements == [
        monitored_elements[1],
        monitored_elements[2],
    ], "The missing elements should be the trafo and bus"
    assert len(translated_monitored_elements) == 1, (
        "One monitored element should be extracted since one line has an origin_id"
    )
    assert translated_monitored_elements.iloc[0]["name"] == "Line 1", (
        "The name of the monitored element should match the line name"
    )
    assert translated_monitored_elements.iloc[0].table == "line", "The monitored element should be of type 'line'"
    assert translated_monitored_elements.iloc[0].table_id == 0, "The monitored element should have the correct table_id"


def test_match_node_direct_cb_match():
    data = {
        "bus": [0],
        "element": [1],
        "type": ["CB"],
        "name": ["sw0"],
        "closed": [True],
        "origin_id": ["cgmes0"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0, 1, 2])
    actual_buses = np.array([3])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="unique_pandapower", max_jumps=2
    )
    assert set(matched["original_node"]) == {0, 1}
    assert set(matched["unique_id"]) == {get_globally_unique_id(0, "switch")}

    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=2
    )
    assert set(matched["original_node"]) == {0, 1}
    assert set(matched["unique_id"]) == {"cgmes0"}


def test_no_match_if_busbar_in_between():
    data = {
        "bus": [0, 1],
        "element": [1, 2],
        "type": ["SR", "CB"],
        "name": ["sw0", "sw1"],
        "closed": [True, True],
        "origin_id": ["cgmes0", "cgmes1"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0, 2])
    actual_buses = np.array([1])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=2
    )
    assert set(matched["original_node"]) == {2}
    assert set(matched["unique_id"]) == {"cgmes1"}


def test_two_cbs_after_sr():
    data = {
        "bus": [0, 1, 1],
        "element": [1, 2, 3],
        "type": ["SR", "CB", "CB"],
        "name": ["sw0", "sw1", "sw2"],
        "closed": [True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0])
    actual_buses = np.array([3])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=3
    )
    assert set(matched["original_node"]) == {0}
    assert set(matched["unique_id"]) == {"cgmes1", "cgmes2"}


def test_two_jumps_to_cb():
    data = {
        "bus": [0, 1, 1],
        "element": [1, 2, 3],
        "type": ["SR", "SR", "CB"],
        "name": ["sw0", "sw1", "sw2"],
        "closed": [True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0])
    actual_buses = np.array([3])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=3
    )
    assert set(matched["original_node"]) == {0}
    assert set(matched["unique_id"]) == {"cgmes2"}


def test_no_match_if_too_many_jumps():
    data = {
        "bus": [0, 1, 2, 3],
        "element": [1, 2, 3, 4],
        "type": ["SR", "SR", "SR", "CB"],
        "name": ["sw0", "sw1", "sw2", "sw3"],
        "closed": [True, True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2", "cgmes3"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0])
    actual_buses = np.array([4])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=2
    )
    assert matched.empty


def test_two_matches_different_jumps():
    data = {
        "bus": [0, 1, 1, 3],
        "element": [1, 2, 3, 4],
        "type": ["SR", "CB", "SR", "CB"],
        "name": ["sw0", "sw1", "sw2", "sw3"],
        "closed": [True, True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2", "cgmes3"],
    }
    switches_df = pd.DataFrame(data)
    node_ids = np.array([0])
    actual_buses = np.array([4])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=3
    )
    assert set(matched["original_node"]) == {0}
    assert set(matched["unique_id"]) == {"cgmes1", "cgmes3"}


def test_matches_with_indizes_missing_cgmes_id():
    data = {
        "bus": [0, 1, 1, 3],
        "element": [1, 2, 3, 4],
        "type": ["SR", "CB", "SR", "CB"],
        "name": ["sw0", "sw1", "sw2", "sw3"],
        "closed": [True, True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2", "cgmes3"],
    }
    switches_df = pd.DataFrame(data, index=[0, 1, 2, 4])  # Note missing index 3
    node_ids = np.array([0])
    actual_buses = np.array([4])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="cgmes", max_jumps=3
    )
    assert set(matched["original_node"]) == {0}
    assert set(matched["unique_id"]) == {"cgmes1", "cgmes3"}


def test_matches_with_indizes_missing_unique_pandapower_id():
    data = {
        "bus": [0, 1, 1, 3],
        "element": [1, 2, 3, 4],
        "type": ["SR", "CB", "SR", "CB"],
        "name": ["sw0", "sw1", "sw2", "sw3"],
        "closed": [True, True, True, True],
        "origin_id": ["cgmes0", "cgmes1", "cgmes2", "cgmes3"],
    }
    switches_df = pd.DataFrame(data, index=[0, 1, 2, 4])  # Note missing index 3
    node_ids = np.array([0])
    actual_buses = np.array([4])
    # Should match CB switches for nodes 0 and 1, not for 2 (DS type)
    matched = match_node_to_next_switch_type(
        node_ids, switches_df, actual_buses, switch_type="CB", id_type="unique_pandapower", max_jumps=3
    )
    assert set(matched["original_node"]) == {0}
    assert set(matched["unique_id"]) == {"1%%switch", "4%%switch"}
