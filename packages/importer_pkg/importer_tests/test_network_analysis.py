from pathlib import Path

import logbook
import numpy as np
import pandas as pd
import pypowsybl
from toop_engine_importer.pypowsybl_import import network_analysis
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    UcteImportResult,
)


def test_get_all_data_from_violation_df_in_one_dataframe(ucte_file):
    expexted = {
        "imax": {
            ("", "B_SU2_11 B_SU1_11 1"): 100.0,
            ("", "D8SU1_12 D7SU2_11 1"): 100.0,
            ("", "D8SU1_12 D7SU2_11 2"): 100.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 100.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 100.0,
        },
        "i_violation": {
            ("", "B_SU2_11 B_SU1_11 1"): 242.2,
            ("", "D8SU1_12 D7SU2_11 1"): 320.2,
            ("", "D8SU1_12 D7SU2_11 2"): 282.7,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 605.1,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 603.0,
        },
        "side": {
            ("", "B_SU2_11 B_SU1_11 1"): "TWO",
            ("", "D8SU1_12 D7SU2_11 1"): "TWO",
            ("", "D8SU1_12 D7SU2_11 2"): "TWO",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "TWO",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "TWO",
        },
        "p1": {
            ("", "B_SU2_11 B_SU1_11 1"): -150.0,
            ("", "D8SU1_12 D7SU2_11 1"): -213.5,
            ("", "D8SU1_12 D7SU2_11 2"): -187.2,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): -400.6,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): -400.6,
        },
        "q1": {
            ("", "B_SU2_11 B_SU1_11 1"): 75.2,
            ("", "D8SU1_12 D7SU2_11 1"): 58.1,
            ("", "D8SU1_12 D7SU2_11 2"): 55.4,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 121.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 116.1,
        },
        "i1": {
            ("", "B_SU2_11 B_SU1_11 1"): 242.2,
            ("", "D8SU1_12 D7SU2_11 1"): 319.4,
            ("", "D8SU1_12 D7SU2_11 2"): 281.8,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 604.2,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 602.1,
        },
        "p2": {
            ("", "B_SU2_11 B_SU1_11 1"): 150.0,
            ("", "D8SU1_12 D7SU2_11 1"): 213.6,
            ("", "D8SU1_12 D7SU2_11 2"): 187.3,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 401.4,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 401.2,
        },
        "q2": {
            ("", "B_SU2_11 B_SU1_11 1"): -75.2,
            ("", "D8SU1_12 D7SU2_11 1"): -59.7,
            ("", "D8SU1_12 D7SU2_11 2"): -57.1,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): -121.1,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): -116.3,
        },
        "i2": {
            ("", "B_SU2_11 B_SU1_11 1"): 242.2,
            ("", "D8SU1_12 D7SU2_11 1"): 320.2,
            ("", "D8SU1_12 D7SU2_11 2"): 282.7,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 605.1,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 603.0,
        },
        "type": {
            ("", "B_SU2_11 B_SU1_11 1"): "LINE",
            ("", "D8SU1_12 D7SU2_11 1"): "LINE",
            ("", "D8SU1_12 D7SU2_11 2"): "LINE",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "LINE",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "LINE",
        },
        "voltage_level1_id": {
            ("", "B_SU2_11 B_SU1_11 1"): "B_SU2_1",
            ("", "D8SU1_12 D7SU2_11 1"): "D8SU1_1",
            ("", "D8SU1_12 D7SU2_11 2"): "D8SU1_1",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D8SU1_1",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D8SU1_1",
        },
        "bus_breaker_bus1_id": {
            ("", "B_SU2_11 B_SU1_11 1"): "B_SU2_11",
            ("", "D8SU1_12 D7SU2_11 1"): "D8SU1_12",
            ("", "D8SU1_12 D7SU2_11 2"): "D8SU1_12",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D8SU1_12",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D8SU1_12",
        },
        "voltage_level2_id": {
            ("", "B_SU2_11 B_SU1_11 1"): "B_SU1_1",
            ("", "D8SU1_12 D7SU2_11 1"): "D7SU2_1",
            ("", "D8SU1_12 D7SU2_11 2"): "D7SU2_1",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D7SU2_1",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D7SU2_1",
        },
        "bus_breaker_bus2_id": {
            ("", "B_SU2_11 B_SU1_11 1"): "B_SU1_11",
            ("", "D8SU1_12 D7SU2_11 1"): "D7SU2_11",
            ("", "D8SU1_12 D7SU2_11 2"): "D7SU2_11",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D7SU2_11",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D7SU2_11",
        },
        "elementName": {
            ("", "B_SU2_11 B_SU1_11 1"): "Test Line 4",
            ("", "D8SU1_12 D7SU2_11 1"): "Test Line",
            ("", "D8SU1_12 D7SU2_11 2"): "Test Line 2",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "Test Line 2",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "Test Line",
        },
        "v_mag1": {
            ("", "B_SU2_11 B_SU1_11 1"): 400.0,
            ("", "D8SU1_12 D7SU2_11 1"): 400.0,
            ("", "D8SU1_12 D7SU2_11 2"): 400.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 399.9,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 399.9,
        },
        "v_angle1": {
            ("", "B_SU2_11 B_SU1_11 1"): -0.5,
            ("", "D8SU1_12 D7SU2_11 1"): -0.1,
            ("", "D8SU1_12 D7SU2_11 2"): -0.1,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): -0.3,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): -0.3,
        },
        "v_mag2": {
            ("", "B_SU2_11 B_SU1_11 1"): 400.0,
            ("", "D8SU1_12 D7SU2_11 1"): 400.0,
            ("", "D8SU1_12 D7SU2_11 2"): 400.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 400.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 400.0,
        },
        "v_angle2": {
            ("", "B_SU2_11 B_SU1_11 1"): -0.5,
            ("", "D8SU1_12 D7SU2_11 1"): 0.0,
            ("", "D8SU1_12 D7SU2_11 2"): 0.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 0.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 0.0,
        },
        "nominal_v1": {
            ("", "B_SU2_11 B_SU1_11 1"): 380.0,
            ("", "D8SU1_12 D7SU2_11 1"): 380.0,
            ("", "D8SU1_12 D7SU2_11 2"): 380.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 380.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 380.0,
        },
        "nominal_v2": {
            ("", "B_SU2_11 B_SU1_11 1"): 380.0,
            ("", "D8SU1_12 D7SU2_11 1"): 380.0,
            ("", "D8SU1_12 D7SU2_11 2"): 380.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 380.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 380.0,
        },
        "smax": {
            ("", "B_SU2_11 B_SU1_11 1"): 69.0,
            ("", "D8SU1_12 D7SU2_11 1"): 69.0,
            ("", "D8SU1_12 D7SU2_11 2"): 69.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 69.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 69.0,
        },
        "smax_nominal": {
            ("", "B_SU2_11 B_SU1_11 1"): 66.0,
            ("", "D8SU1_12 D7SU2_11 1"): 66.0,
            ("", "D8SU1_12 D7SU2_11 2"): 66.0,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 66.0,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 66.0,
        },
        "I%": {
            ("", "B_SU2_11 B_SU1_11 1"): 242.2,
            ("", "D8SU1_12 D7SU2_11 1"): 320.2,
            ("", "D8SU1_12 D7SU2_11 2"): 282.7,
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): 605.1,
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): 603.0,
        },
        "bus_breaker_bus1_id_contingency": {
            ("", "B_SU2_11 B_SU1_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 2"): "nan",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D8SU1_12",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D8SU1_12",
        },
        "bus_breaker_bus2_id_contingency": {
            ("", "B_SU2_11 B_SU1_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 2"): "nan",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "D7SU2_11",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "D7SU2_11",
        },
        "elementName_contingency": {
            ("", "B_SU2_11 B_SU1_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 2"): "nan",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "Test Line",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "Test Line 2",
        },
        "type_contingency": {
            ("", "B_SU2_11 B_SU1_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 1"): "nan",
            ("", "D8SU1_12 D7SU2_11 2"): "nan",
            ("D8SU1_12 D7SU2_11 1", "D8SU1_12 D7SU2_11 2"): "LINE",
            ("D8SU1_12 D7SU2_11 2", "D8SU1_12 D7SU2_11 1"): "LINE",
        },
    }

    network = pypowsybl.network.load(ucte_file)
    analysis = pypowsybl.security.create_analysis()
    analysis.add_single_element_contingencies(network.get_lines().index)
    analysis.add_single_element_contingencies(network.get_2_windings_transformers().index)
    analysis.add_monitored_elements(branch_ids=network.get_lines().index)
    analysis.add_monitored_elements(voltage_level_ids=network.get_voltage_levels().index)

    analysis_results = analysis.run_ac(network)
    result = (
        network_analysis.get_all_data_from_violation_df_in_one_dataframe(network, analysis_results)
        .round(1)
        .fillna("nan")
        .to_dict()
    )
    for key, value in result.items():
        assert value == expexted[key], f"key: '{key}', value: '{value}' does not match expected value: '{expexted[key]}'"
    assert result == expexted


def test_convert_low_impedance_lines(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D8SU1_11 2",
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]
    network_analysis.convert_low_impedance_lines(network, "D8")
    network.get_lines()
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]
    assert "D8SU1_12 D8SU1_11 2" in network.get_switches().index.to_list()
    network_analysis.convert_low_impedance_lines(network, "D2")
    network.get_lines()
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]


def test_remove_branches_across_switch(ucte_file):
    # test 1
    network = pypowsybl.network.load(ucte_file)
    initial_branch_count = len(network.get_branches())
    # check starting point
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D8SU1_11 2",
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]

    network_analysis.remove_branches_across_switch(network)
    final_branch_count = len(network.get_branches())

    assert final_branch_count + 2 == initial_branch_count
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "B_SU2_11 B_SU1_11 1",
    ]

    # test 2
    network = pypowsybl.network.load(ucte_file)
    network.remove_elements("D8SU1_12 D8SU1_11 1")
    network_analysis.remove_branches_across_switch(network)
    final_branch_count = len(network.get_branches())
    assert final_branch_count + 1 == initial_branch_count
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D8SU1_11 2",
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "B_SU2_11 B_SU1_11 1",
    ]


def test_create_default_security_analysis_param(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)
    security_analysis_param = network_analysis.create_default_security_analysis_param(network, ucte_importer_parameters)

    security_analysis_param_expected = network_analysis.PowsyblSecurityAnalysisParam(
        single_element_contingencies_ids={
            "dangling": [
                "XB__F_11 D8SU1_11 1",
                "XB__F_21 D8SU1_21 1",
                "XG__F_21 D8SU1_21 1",
            ],
            "generator": ["D7SU2_11_generator"],
            "line": [
                "D8SU1_12 D7SU2_11 1",
                "D8SU1_12 D7SU2_11 2",
                "D8SU1_12 D8SU1_11 2",
            ],
            "tie": [
                "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
                "XB__F_21 B_SU1_21 1 + XB__F_21 D8SU1_21 1",
            ],
            "transformer": ["D8SU1_11 D8SU1_21 1"],
            "load": [],
            "switch": [],
        },
        current_limit_factor=1.0,
        monitored_branches=[
            "D8SU1_12 D7SU2_11 1",
            "D8SU1_12 D7SU2_11 2",
            "D8SU1_11 D8SU1_21 1",
            "D8SU1_12 D8SU1_11 2",
            "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
            "XB__F_21 B_SU1_21 1 + XB__F_21 D8SU1_21 1",
        ],
        monitored_buses=["D8SU1_1", "D8SU1_2", "D2SU1_3", "D2SU2_3", "D7SU2_1"],
        ac_run=False,
    )
    assert security_analysis_param_expected.current_limit_factor == security_analysis_param.current_limit_factor
    assert security_analysis_param_expected.ac_run == security_analysis_param.ac_run
    security_analysis_param_expected.monitored_branches.sort()
    security_analysis_param.monitored_branches.sort()
    for el1, el2 in zip(
        security_analysis_param_expected.monitored_branches,
        security_analysis_param.monitored_branches,
    ):
        assert el1 == el2, f"el1: {el1}, el2: {el2}"
    security_analysis_param_expected.monitored_buses.sort()
    security_analysis_param.monitored_buses.sort()
    for el1, el2 in zip(
        security_analysis_param_expected.monitored_buses,
        security_analysis_param.monitored_buses,
    ):
        assert el1 == el2, f"el1: {el1}, el2: {el2}"
    for (
        key,
        value,
    ) in security_analysis_param_expected.single_element_contingencies_ids.items():
        value.sort()
        security_analysis_param.single_element_contingencies_ids[key].sort()
        for el1, el2 in zip(value, security_analysis_param.single_element_contingencies_ids[key]):
            assert el1 == el2, f"el1: {el1}, el2: {el2}"


def test_run_N1_analysis(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)
    security_analysis_param = network_analysis.create_default_security_analysis_param(network, ucte_importer_parameters)
    security_analysis_results = network_analysis.run_n1_analysis(network, security_analysis_param)
    assert isinstance(security_analysis_results, pypowsybl.security.SecurityAnalysisResult)
    assert all(security_analysis_results.branch_results["q1"].isna())
    security_analysis_param.ac_run = True
    security_analysis_results = network_analysis.run_n1_analysis(network, security_analysis_param)
    assert isinstance(security_analysis_results, pypowsybl.security.SecurityAnalysisResult)
    assert not any(security_analysis_results.branch_results["q1"].isna())


def test_get_violation_df(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)
    security_analysis_param = network_analysis.create_default_security_analysis_param(network, ucte_importer_parameters)
    security_analysis_results = network_analysis.run_n1_analysis(network, security_analysis_param)
    violation_df = network_analysis.get_all_data_from_violation_df_in_one_dataframe(network, security_analysis_results)
    expected_col = [
        "imax",
        "i_violation",
        "side",
        "p1",
        "q1",
        "i1",
        "p2",
        "q2",
        "i2",
        "type",
        "voltage_level1_id",
        "bus_breaker_bus1_id",
        "voltage_level2_id",
        "bus_breaker_bus2_id",
        "elementName",
        "bus_breaker_bus1_id_contingency",
        "bus_breaker_bus2_id_contingency",
        "elementName_contingency",
        "type_contingency",
        "v_mag1",
        "v_angle1",
        "v_mag2",
        "v_angle2",
        "nominal_v1",
        "nominal_v2",
        "smax",
        "smax_nominal",
        "I%",
    ]
    assert len(violation_df) == 5
    assert violation_df.columns.tolist() == expected_col
    violation_df = network_analysis.get_all_data_from_violation_df_in_one_dataframe(
        network, security_analysis_results, all_attributes=True
    )
    assert len(violation_df.columns) == 45


def test_drop_one_side_from_violation_df():
    violation_df = pd.DataFrame(
        {
            "line_id": ["line1", "line1", "line3"],
            "i_violation": [100, 200, 300],
            "side": ["TWO", "ONE", "ONE"],
        }
    )
    violation_df.set_index(["line_id"], inplace=True)
    column_name = "i_violation"
    new_df = network_analysis.drop_one_side_from_violation_df(violation_df, column_name)
    assert isinstance(new_df, pd.DataFrame)
    assert len(new_df) == 2
    assert violation_df[violation_df["side"] == "ONE"].equals(new_df)


def test_calc_total_overload():
    data_dict = {
        "contingency_id": {
            0: "",
            1: "",
            2: "",
            3: "D8SU1_12 D7SU2_11 1",
            4: "D8SU1_12 D7SU2_11 2",
        },
        "subject_id": {
            0: "B_SU2_11 B_SU1_11 1",
            1: "D8SU1_12 D7SU2_11 1",
            2: "D8SU1_12 D7SU2_11 2",
            3: "D8SU1_12 D7SU2_11 2",
            4: "D8SU1_12 D7SU2_11 1",
        },
        "S_overload": {0: 12.0, 1: 146.0, 2: 122.0, 3: 334.0, 4: 334.0},
    }

    violation_df = pd.DataFrame.from_dict(data_dict)
    overload = network_analysis.calc_total_overload(
        violation_df=violation_df,
        column_case_name_of_n1="subject_id",
        overload_column="S_overload",
    )
    assert isinstance(overload, float)
    assert overload == 680.0


def test_get_voltage_angle():
    data_dict = {
        "contingency_id": {0: "", 1: "", 2: "", 4: "con_id", 5: "con_id", 6: "con_id"},
        "voltage_level_id": {
            0: "D8SU1_1",
            1: "D8SU1_1",
            2: "D8SU1_2",
            4: "D8SU1_1",
            5: "D8SU1_1",
            6: "D8SU1_2",
        },
        "bus_id": {
            0: "D8SU1_12",
            1: "D8SU1_11",
            2: "D8SU1_21",
            4: "D8SU1_12",
            5: "D8SU1_11",
            6: "D8SU1_21",
        },
        "v_angle": {
            0: 1.0,
            1: 2.0,
            2: 3.0,
            4: 50.0,
            5: 60.0,
            6: 70.0,
        },
    }

    expected = {
        "D8SU1_1": {
            "D8SU1_12": np.float64(50.0),
            "D8SU1_11": np.float64(60.0),
            "diff": np.float64(10.0),
        },
        "D8SU1_2": {},
    }

    bus_res = pd.DataFrame(data_dict)
    bus_res.set_index(["contingency_id", "voltage_level_id", "bus_id"], inplace=True)
    results = network_analysis.get_voltage_angle(bus_res)
    assert results == expected
    results = network_analysis.get_voltage_angle(bus_res[bus_res.index.get_level_values("contingency_id") == ""])
    expected = {
        "D8SU1_1": {
            "D8SU1_12": np.float64(1.0),
            "D8SU1_11": np.float64(2.0),
            "diff": np.float64(1.0),
        },
        "D8SU1_2": {},
    }
    assert results == expected


def test_set_new_operational_limit(ucte_file):
    # test 1 - factor = 1.0
    network = pypowsybl.network.load(ucte_file)
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.0)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == original_limit

    # test 2 - factor = 0.7
    network_analysis.set_new_operational_limit(network=network, factor=0.7)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == [round(0.7 * limit, 2) for limit in original_limit]

    # test 3 - factor = 1.2
    network = pypowsybl.network.load(ucte_file)
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.2)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == [round(1.2 * limit, 2) for limit in original_limit]

    # test 4 - factor = 1.0, value = 100
    network = pypowsybl.network.load(ucte_file)
    original_limit = network.get_operational_limits()["value"].to_list()
    factor = 1.0
    value = 123.0
    network_analysis.set_new_operational_limit(network=network, factor=factor, value=value)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == [value * factor] * len(original_limit)

    # test 5 - factor = 1.1, value = 100
    network = pypowsybl.network.load(ucte_file)
    original_limit = network.get_operational_limits()["value"].to_list()
    factor = 1.1
    network_analysis.set_new_operational_limit(network=network, factor=factor, value=value)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == [value * factor] * len(original_limit)

    # test 5 - factor = 1.1, value = 100, id_list length = 1 but id does not exist
    network = pypowsybl.network.load(ucte_file)
    id_list = ["NOT_EXISTING"]
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.1, value=value, id_list=id_list)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == original_limit

    # test 6 - factor = 1.1, value = 100, id_list length = 1
    network = pypowsybl.network.load(ucte_file)
    id_list = ["D8SU1_12 D8SU1_11 2"]
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.1, value=value, id_list=id_list)
    new_limit = network.get_operational_limits()["value"].to_list()
    expected = original_limit.copy()
    expected[0] = value * factor
    expected[1] = value * factor
    assert new_limit == expected

    # test 7 - factor = 1.1, value = 100, id_list length = 2
    network = pypowsybl.network.load(ucte_file)
    id_list = ["D8SU1_12 D8SU1_11 2", "D8SU1_12 D7SU2_11 2"]
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.1, value=value, id_list=id_list)
    new_limit = network.get_operational_limits()["value"].to_list()
    expected = original_limit.copy()
    expected[0] = value * factor
    expected[1] = value * factor
    expected[4] = value * factor
    expected[5] = value * factor
    assert new_limit == expected

    # test 7 - factor = 1.1, value = 100, id_list TIE line -> no changes
    network = pypowsybl.network.load(ucte_file)
    id_list = ["XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"]
    original_limit = network.get_operational_limits()["value"].to_list()
    network_analysis.set_new_operational_limit(network=network, factor=1.1, value=value, id_list=id_list)
    new_limit = network.get_operational_limits()["value"].to_list()
    assert new_limit == original_limit


def test_get_branches_df_with_element_name(ucte_file):
    network = pypowsybl.network.load(ucte_file)

    result = network_analysis.get_branches_df_with_element_name(network)
    assert result.columns.to_list() == network.get_branches(all_attributes=True).columns.to_list() + [
        "elementName",
        "pairing_key",
    ]
    assert result.drop(columns=["elementName", "pairing_key"]).equals(network.get_branches(all_attributes=True))
    result_line_names = result[result["type"] == "LINE"]["elementName"].to_list()
    line_name = network.get_lines(all_attributes=True)["elementName"].to_list()
    assert result_line_names == line_name
    result_trafo_names = result[result["type"] == "TWO_WINDINGS_TRANSFORMER"]["elementName"].to_list()
    trafo_name = network.get_2_windings_transformers(all_attributes=True)["elementName"].to_list()
    assert result_trafo_names == trafo_name
    result_tie_names = result[result["type"] == "TIE_LINE"]["pairing_key"].to_list()
    tie_name = network.get_tie_lines(all_attributes=True)["pairing_key"].to_list()
    assert result_tie_names == tie_name
    result_tie_names = result[result["type"] == "TIE_LINE"]["elementName"].to_list()
    tie_name = (
        network.get_tie_lines(all_attributes=True)["elementName_1"]
        + " + "
        + network.get_tie_lines(all_attributes=True)["elementName_2"]
    ).to_list()
    assert result_tie_names == tie_name


def test_apply_CB_lists(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)
    # Create a sample DataFrame
    white_list_df = pd.DataFrame(
        {
            "Anfangsknoten": ["D8SU1_1"],
            "Endknoten": ["D8SU1_1"],
            "Elementname": ["Test C. Line"],
            "Auslastungsgrenze_n_0": [110],
            "Auslastungsgrenze_n_1": [190],
        }
    )

    black_list_df = pd.DataFrame(
        {
            "Anfangsknoten": ["D7SU2_1"],
            "Endknoten": ["D8SU1_1"],
            "Elementname": ["Test Line 2"],
        }
    )
    white_list_df.to_csv(ucte_importer_parameters.data_folder / "CB_White-Liste.csv", index=False, sep=";")
    black_list_df.to_csv(ucte_importer_parameters.data_folder / "CB_Black-Liste.csv", index=False, sep=";")

    # test 1 - apply white and black list
    import_result = UcteImportResult(
        data_folder=ucte_importer_parameters.data_folder,
    )

    ucte_importer_parameters.white_list_file = ucte_importer_parameters.data_folder / "CB_White-Liste.csv"
    ucte_importer_parameters.black_list_file = ucte_importer_parameters.data_folder / "CB_Black-Liste.csv"
    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=import_result,
        border_current={},
        network_changes={},
        import_parameter=ucte_importer_parameters,
    )
    network_analysis.apply_cb_lists(
        network=network,
        statistics=statistics,
        ucte_importer_parameters=ucte_importer_parameters,
    )
    assert statistics.id_lists["white_list"] == ["D8SU1_12 D8SU1_11 2"]
    assert statistics.id_lists["black_list"] == ["D8SU1_12 D7SU2_11 2"]
    assert statistics.import_result.n_white_list == 1
    assert statistics.import_result.n_black_list == 1
    assert statistics.import_result.n_black_list_applied == 1
    assert statistics.import_result.n_white_list_applied == 1

    # test 2 - apply black list only
    import_result = UcteImportResult(
        data_folder=ucte_importer_parameters.data_folder,
    )
    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=import_result,
        border_current={},
        network_changes={},
        import_parameter=ucte_importer_parameters,
    )
    ucte_importer_parameters.white_list_file = None
    network_analysis.apply_cb_lists(
        network=network,
        statistics=statistics,
        ucte_importer_parameters=ucte_importer_parameters,
    )
    assert statistics.id_lists["black_list"] == ["D8SU1_12 D7SU2_11 2"]
    assert "white_list" in statistics.id_lists
    assert statistics.id_lists["white_list"] == []
    assert statistics.import_result.n_white_list == 0
    assert statistics.import_result.n_black_list == 1
    assert statistics.import_result.n_black_list_applied == 1
    assert statistics.import_result.n_white_list_applied == 0

    # test 3 - apply no list
    import_result = UcteImportResult(
        data_folder=Path(""),
    )
    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=import_result,
        border_current={},
        network_changes={},
        import_parameter=ucte_importer_parameters,
    )
    ucte_importer_parameters.black_list_file = None
    network_analysis.apply_cb_lists(
        network=network,
        statistics=statistics,
        ucte_importer_parameters=ucte_importer_parameters,
    )
    assert "white_list" in statistics.id_lists
    assert "black_list" in statistics.id_lists
    assert statistics.id_lists["white_list"] == []
    assert statistics.id_lists["black_list"] == []
    assert statistics.import_result.n_white_list == 0
    assert statistics.import_result.n_black_list == 0
    assert statistics.import_result.n_black_list_applied == 0
    assert statistics.import_result.n_white_list_applied == 0


def test_convert_tie_to_dangling():
    expected_df = pd.DataFrame.from_dict(
        {
            "subject_name": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): "",
                ("", "XB__F_11 B_SU1_11 1"): "",
                ("", "XB__F_11 D8SU1_11 1"): "",
            },
            "limit_type": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): "CURRENT",
                ("", "XB__F_11 B_SU1_11 1"): "CURRENT",
                ("", "XB__F_11 D8SU1_11 1"): "CURRENT",
            },
            "limit_name": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): "permanent",
                ("", "XB__F_11 B_SU1_11 1"): "permanent",
                ("", "XB__F_11 D8SU1_11 1"): "permanent",
            },
            "imax": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): 120.0,
                ("", "XB__F_11 B_SU1_11 1"): 100.0,
                ("", "XB__F_11 D8SU1_11 1"): 120.0,
            },
            "acceptable_duration": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): 123,
                ("", "XB__F_11 B_SU1_11 1"): 123,
                ("", "XB__F_11 D8SU1_11 1"): 123,
            },
            "limit_reduction": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): 1.0,
                ("", "XB__F_11 B_SU1_11 1"): 1.0,
                ("", "XB__F_11 D8SU1_11 1"): 1.0,
            },
            "i_violation": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): 364,
                ("", "XB__F_11 B_SU1_11 1"): 362,
                ("", "XB__F_11 D8SU1_11 1"): 364,
            },
            "side": {
                ("", "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1"): "TWO",
                ("", "XB__F_11 B_SU1_11 1"): "ONE",
                ("", "XB__F_11 D8SU1_11 1"): "TWO",
            },
        }
    )

    result = network_analysis.convert_tie_to_dangling(violation_df=expected_df.iloc[0:2])
    assert result.equals(expected_df)


def test_get_branches_with_dangling_lines(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    branches = network_analysis.get_branches_with_dangling_lines(network)
    lines = network.get_lines()
    trafo = network.get_2_windings_transformers()
    tie = network.get_tie_lines()
    dangling = network.get_dangling_lines()
    assert len(branches) == len(lines) + len(trafo) + len(tie) + len(dangling)
    assert all(branches["type"].isin(["LINE", "TWO_WINDINGS_TRANSFORMER", "TIE_LINE", "DANGLING_LINE"]))
    assert lines.index.isin(branches.index).all()
    assert trafo.index.isin(branches.index).all()
    assert tie.index.isin(branches.index).all()
    assert dangling.index.isin(branches.index).all()


def test_add_element_name_to_branches_df(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    branches = network.get_lines()
    branches = network_analysis.add_element_name_to_branches_df(network=network, branches_df=branches)
    assert "elementName" in branches.columns
    assert branches["elementName"].equals(network.get_lines(all_attributes=True)["elementName"])
    branches = network.get_2_windings_transformers()
    branches = network_analysis.add_element_name_to_branches_df(network=network, branches_df=branches)
    assert "elementName" in branches.columns
    assert branches["elementName"].equals(network.get_2_windings_transformers(all_attributes=True)["elementName"])
    branches = network.get_tie_lines()
    branches = network_analysis.add_element_name_to_branches_df(network=network, branches_df=branches)
    assert "elementName" in branches.columns
    assert branches["elementName"].equals(
        network.get_tie_lines(all_attributes=True)["elementName_1"]
        + " + "
        + network.get_tie_lines(all_attributes=True)["elementName_2"]
    )
    branches = network.get_dangling_lines()
    branches = network_analysis.add_element_name_to_branches_df(network=network, branches_df=branches)
    assert "elementName" in branches.columns


def test_merge_voltage_levels_to_branches_df(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    lines = network.get_lines()
    lines = network_analysis.merge_voltage_levels_to_branches_df(network=network, branches_df=lines)
    assert "nominal_v1" in lines.columns
    assert "nominal_v2" in lines.columns


def test_remove_branches_with_same_bus(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    network.create_lines(
        id="TO_BE_REMOVED",
        voltage_level1_id="D8SU1_1",
        bus1_id="D8SU1_12",
        voltage_level2_id="D8SU1_1",
        bus2_id="D8SU1_12",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=0.5,
        x=10,
    )
    branches = network.get_branches()
    with logbook.handlers.TestHandler() as caplog:
        network_analysis.remove_branches_with_same_bus(network=network)
        assert caplog.has_warnings
        assert "branches with the same bus id" in "".join(caplog.formatted_records)

    removed_branches = network.get_branches()
    assert len(removed_branches) == len(branches) - 3
    assert "D2SU1_31 D2SU1_31 2" not in removed_branches.index
    assert "TO_BE_REMOVED" not in removed_branches.index
    assert "D8SU1_12 D8SU1_11 2" not in removed_branches.index
    assert "D2SU1_31 D2SU1_31 2" in branches.index
    assert "TO_BE_REMOVED" in branches.index
    assert "D8SU1_12 D8SU1_11 2" in branches.index
