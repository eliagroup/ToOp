# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import logbook
import pandas as pd
import pytest
from toop_engine_importer.exporter import uct_exporter
from toop_engine_importer.ucte_toolset.ucte_io import parse_ucte


def get_test_data_json():
    """Return test data for the tests"""
    return {
        "id": "D8ABC_1_0",
        "type": "BUS",
        "name": "",
        "relevant_node_index": 8416,
        "branch_assignments": [
            {
                "id": "D8ABC_11 D8EFG_11 1",
                "type": "LINE",
                "name": "D8ABC_11 ## D8EFG_11 ## 123",
                "index": 1048,
                "on_bus_b": False,
                "from_end": True,
                "other_end": {"id": "D8EFG_1_0", "type": "BUS", "name": ""},
            },
            {
                "id": "D8ABC_11 D8EFG_11 1",
                "type": "LINE",
                "name": "D8ABC_11 ## D8EFG_11 ## 123",
                "index": 1048,
                "on_bus_b": True,
                "from_end": True,
                "other_end": {"id": "D8EFG_1_0", "type": "BUS", "name": ""},
            },
            {
                "id": "D8ABC_12 D8EFG_11 1",
                "type": "LINE",
                "name": "D8ABC_11 ## D8EFG_11 ## 123",
                "index": 1048,
                "on_bus_b": False,
                "from_end": True,
                "other_end": {"id": "D8EFG_1_0", "type": "BUS", "name": ""},
            },
        ],
    }


def get_test_data_switches():
    """Return test data for the tests
    Note: this is a testing DF with no real world representation!
    Goal was to have a DF with the most use cases.
    """
    dict_data = {
        "from": {
            0: "D8ABC_13",
            1: "D8ABC_11",
            2: "D8ABC_11",
            3: "D8ABC_13",
            4: "D8ABC_13",
        },
        "to": {
            0: "D8EFG_12",
            1: "D8ABC_12",
            2: "D8EFG_13",
            3: "D8ABC_14",
            4: "D8EFG_12",
        },
        "order": {0: "1", 1: "1", 2: "1", 3: "1", 4: "2"},
        "status": {0: "2", 1: "2", 2: "2", 3: "7", 4: "2"},
        "r": {0: "1", 1: "1", 2: "1", 3: "1", 4: "1"},
        "x": {0: "2", 1: "2", 2: "2", 3: "2", 4: "2"},
        "b": {0: "3", 1: "3", 2: "3", 3: "3", 4: "3"},
        "i": {0: "4", 1: "4", 2: "4", 3: "4", 4: "4"},
        "name": {
            0: "            ",
            1: "            ",
            2: "            ",
            3: "            ",
            4: "            ",
        },
    }
    return pd.DataFrame(dict_data)


def test_get_replacement_id():
    """Test the function replacement_id from uct_exporter.py"""
    test_data = get_test_data_json()
    code = test_data["id"][0:7]
    bus1 = test_data["branch_assignments"][0]
    id1 = bus1["id"]
    bus2 = test_data["branch_assignments"][1]
    # id2 = bus2["id"]
    bus3 = test_data["branch_assignments"][2]
    id3 = bus3["id"]
    bus_a = id1.split(" ")[0]
    bus_b = id3.split(" ")[0]
    assert uct_exporter.get_replacement_id(bus1, code, bus_a, bus_b) == id1
    assert uct_exporter.get_replacement_id(bus2, code, bus_a, bus_b) == id3
    assert uct_exporter.get_replacement_id(bus3, code, bus_a, bus_b) == id1

    assert uct_exporter.get_replacement_id(bus1, code, bus_b, bus_a) == id3
    assert uct_exporter.get_replacement_id(bus2, code, bus_b, bus_a) == id1
    assert uct_exporter.get_replacement_id(bus3, code, bus_b, bus_a) == id3

    id4 = id1.split(" ")[1] + id1.split(" ")[0] + id1.split(" ")[2]
    bus1["id"] = id4
    id5 = id3.split(" ")[1] + id3.split(" ")[0] + id3.split(" ")[2]
    bus3["id"] = id5
    assert uct_exporter.get_replacement_id(bus1, code, bus_a, bus_b) == id4
    assert uct_exporter.get_replacement_id(bus3, code, bus_a, bus_b) == id4


def test_find_switches():
    """Test the function find_switches from uct_exporter.py"""
    test_df = get_test_data_switches()
    node_id = "D8ABC_1"  # Note: this is the first 7 characters of the id, not the full id of busbar that has one additional character
    assert uct_exporter.find_switches(test_df, node_id).equals(test_df.iloc[1:2, :])
    assert uct_exporter.find_switches(test_df, "no_id_given_or_id_not_found").empty

    test_df.loc[3, "status"] = "2"
    assert uct_exporter.find_switches(test_df, node_id).equals(pd.concat([test_df.iloc[1:2, :], test_df.iloc[3:4, :]]))


def test_get_unique_busbars():
    res_keys = ["D8ABC_13", "D8EFG_12", "D8ABC_11", "D8ABC_12", "D8EFG_13", "D8ABC_14"]
    test_df = get_test_data_switches()
    result = uct_exporter.group_switches(test_df)
    assert list(result.keys()) == res_keys


def test_group_switches():
    """Test the function group_switches from uct_exporter.py"""
    test_df = get_test_data_switches()
    result = uct_exporter.group_switches(test_df)

    res_keys = ["D8ABC_13", "D8EFG_12", "D8ABC_11", "D8ABC_12", "D8EFG_13", "D8ABC_14"]
    assert len(result) == 6
    assert result[res_keys[0]].equals(pd.concat([test_df.iloc[0:1, :], test_df.iloc[3:5, :]]))  # D8ABC_13
    assert result[res_keys[1]].equals(pd.concat([test_df.iloc[0:1, :], test_df.iloc[4:5, :]]))  # D8EFG_12
    assert result[res_keys[2]].equals(test_df.iloc[1:3, :])  # D8ABC_11
    assert result[res_keys[3]].equals(test_df.iloc[1:2, :])  # D8ABC_12
    assert result[res_keys[4]].equals(test_df.iloc[2:3, :])  # D8EFG_13
    assert result[res_keys[5]].equals(test_df.iloc[3:4, :])  # D8ABC_14


def test_get_bus_a_b():
    """Test the function get_bus_a_b from uct_exporter.py"""
    test_df = get_test_data_switches()
    standart_result = tuple([test_df.iloc[0, 0], test_df.iloc[0, 1]])
    assert uct_exporter.get_bus_a_b(test_df.iloc[0:1, :]) == standart_result
    assert uct_exporter.get_bus_a_b(pd.concat([test_df.iloc[0:1, :], test_df.iloc[4:5, :]])) == standart_result

    # test false input
    with pytest.raises(ValueError):
        uct_exporter.get_bus_a_b(test_df.iloc[0:2, :])


def test_execute_branch_assignment():
    """Test the function execute_branch_assignment from uct_exporter.py"""
    test_data = get_test_data_json()
    test_df_ref = get_test_data_switches()
    test_df = get_test_data_switches()
    id = test_data["branch_assignments"][0]["id"]
    id2 = test_df.iloc[0]["from"] + " " + test_df.iloc[0]["to"] + " " + test_df.iloc[0]["order"]
    replacement_id = test_df.iloc[2]["from"] + " " + test_df.iloc[2]["to"] + " " + "6"

    statistics_all_stations = {"STATION": {"branches": []}}

    # test 1
    assert test_df_ref.equals(test_df)
    replaced = uct_exporter.execute_branch_assignment(test_df, id, replacement_id, statistics_all_stations)
    # id not in df -> should not change anything
    assert not replaced
    assert test_df_ref.equals(test_df)

    # test 2
    replaced = uct_exporter.execute_branch_assignment(test_df, id2, replacement_id, statistics_all_stations)
    assert not test_df_ref.equals(test_df)
    assert replaced
    # check if assignment was correct
    assert test_df.iloc[0, 0] == test_df_ref.iloc[2, 0]  # from
    assert test_df.iloc[0, 1] == test_df_ref.iloc[2, 1]  # to
    assert test_df.iloc[0, 2] == "6"  # order


def test_open_switches():
    """Test the function open_switches from uct_exporter.py"""
    test_df = get_test_data_switches()
    test_df_ref = get_test_data_switches()
    status_ref = test_df_ref["status"].to_list()

    uct_exporter.open_switches(test_df, test_df_ref.iloc[1:2, :])
    status_expected = status_ref
    status_expected[1] = "7"
    assert test_df["status"].to_list() == status_expected
    uct_exporter.open_switches(test_df, test_df_ref.iloc[3:4, :])

    test_df = get_test_data_switches()
    uct_exporter.open_switches(test_df, pd.concat([test_df.iloc[0:1, :], test_df.iloc[4:5, :]]))
    assert test_df["status"].to_list() == ["7", "2", "2", "7", "7"]


def test_apply_branch_assignment():
    test_df = get_test_data_switches()
    test_data = get_test_data_json()
    bus_a = "D8ABC_11"
    bus_b = "D8ABC_13"

    statistics_all_stations = {"STATION": {"branches": []}}
    test_data["branch_assignments"][0]["id"] = "D8ABC_13 D8EFG_12 1"
    test_data["branch_assignments"][1]["id"] = "D8ABC_11 D8EFG_13 1"
    test_data["branch_assignments"][2]["id"] = "D8ABC_13 D8EFG_12 2"

    # test 1
    expected_output = [
        {
            "original_id": "D8ABC_13 D8EFG_12 1",
            "replacement_id": "D8ABC_11 D8EFG_12 1",
            "type": "LINE",
        },
        {
            "original_id": "D8ABC_11 D8EFG_13 1",
            "replacement_id": "D8ABC_13 D8EFG_13 1",
            "type": "LINE",
        },
        {
            "original_id": "D8ABC_13 D8EFG_12 2",
            "replacement_id": "D8ABC_11 D8EFG_12 2",
            "type": "LINE",
        },
    ]
    results = uct_exporter.apply_branch_assignment(
        topology_optimizer_results=test_data,
        lines=test_df,
        trafos=test_df,
        trafo_reg=test_df,
        bus_a=bus_a,
        bus_b=bus_b,
        statistics_all_stations=statistics_all_stations,
    )
    assert results == expected_output

    # test 2
    test_df = get_test_data_switches()
    test_data["branch_assignments"][0]["id"] = "D8ABC_13 D8EFG_12 1"
    test_data["branch_assignments"][1]["id"] = "D8ABC_11 D8ABC_12 1"
    test_data["branch_assignments"][2]["id"] = "D8ABC_13 D8ABC_13 1"
    expected_output = [
        {
            "original_id": "D8ABC_13 D8EFG_12 1",
            "replacement_id": "D8ABC_11 D8EFG_12 1",
            "type": "LINE",
        },
        {
            "original_id": "D8ABC_11 D8ABC_12 1",
            "replacement_id": "D8ABC_13 D8ABC_13 1",
            "type": "LINE",
        },
        {
            "original_id": "D8ABC_13 D8ABC_13 1",
            "replacement_id": "D8ABC_11 D8ABC_11 1",
            "type": "LINE",
        },
    ]
    results = uct_exporter.apply_branch_assignment(
        topology_optimizer_results=test_data,
        lines=test_df,
        trafos=test_df,
        trafo_reg=test_df,
        bus_a=bus_a,
        bus_b=bus_b,
        statistics_all_stations=statistics_all_stations,
    )
    assert results == expected_output

    # test 3
    test_df = get_test_data_switches()
    test_data = get_test_data_json()
    test_data["branch_assignments"][0]["id"] = "D8ABC_13 D8EFG_12 1"
    test_data["branch_assignments"][1]["id"] = "D8ABC_11 D8EFG_13 1"
    test_data["branch_assignments"][2]["id"] = "D8ABC_13 D8EFG_12 2"
    test_data["branch_assignments"][0]["type"] = "TWO_WINDINGS_TRANSFORMER"
    test_data["branch_assignments"][1]["type"] = "TWO_WINDINGS_TRANSFORMER"
    test_data["branch_assignments"][2]["type"] = "TWO_WINDINGS_TRANSFORMER"

    expected_output = [
        {
            "original_id": "D8ABC_13 D8EFG_12 1",
            "replacement_id": "D8ABC_11 D8EFG_12 1",
            "type": "TWO_WINDINGS_TRANSFORMER",
        },
        {
            "original_id": "D8ABC_11 D8EFG_13 1",
            "replacement_id": "D8ABC_13 D8EFG_13 1",
            "type": "TWO_WINDINGS_TRANSFORMER",
        },
        {
            "original_id": "D8ABC_13 D8EFG_12 2",
            "replacement_id": "D8ABC_11 D8EFG_12 2",
            "type": "TWO_WINDINGS_TRANSFORMER",
        },
    ]
    results = uct_exporter.apply_branch_assignment(
        topology_optimizer_results=test_data,
        lines=test_df,
        trafos=test_df,
        trafo_reg=test_df,
        bus_a=bus_a,
        bus_b=bus_b,
        statistics_all_stations=statistics_all_stations,
    )
    assert results == expected_output

    # test 4
    test_df = get_test_data_switches()
    test_data = get_test_data_json()
    with pytest.raises(ValueError):
        results = uct_exporter.apply_branch_assignment(
            topology_optimizer_results=test_data,
            lines=test_df,
            trafos=test_df,
            trafo_reg=test_df,
            bus_a=bus_a,
            bus_b=bus_b,
            statistics_all_stations=statistics_all_stations,
        )

    # test 5
    statistics_all_stations = {"STATION": {"branches": []}}
    test_df = get_test_data_switches()
    test_df.loc[1, "to"] = "XB__F_11"
    test_df.loc[2, "to"] = "XB__F_11"
    test_df.loc[2, "order"] = "2"
    test_data = get_test_data_json()
    test_data["branch_assignments"][0]["id"] = "XB__F_11 B_SU1_11 1 + D8ABC_11 XB__F_11 1"
    test_data["branch_assignments"][1]["id"] = "XB__F_11 B_SU1_11 1 + D8ABC_11 XB__F_11 2"
    test_data["branch_assignments"][2]["id"] = "XB__F_11 B_SU1_11 1 + XB__F_11 D8ABC_12 1"
    test_data["branch_assignments"][0]["type"] = "TIE_LINE"
    test_data["branch_assignments"][1]["type"] = "TIE_LINE"
    test_data["branch_assignments"][2]["type"] = "TIE_LINE"

    expected_output = [
        {
            "original_id": "XB__F_11 B_SU1_11 1 + D8ABC_11 XB__F_11 1",
            "replacement_id": "",
            "type": "TIE_LINE",
        },
        {
            "original_id": "XB__F_11 B_SU1_11 1 + D8ABC_11 XB__F_11 2",
            "replacement_id": "XB__F_11 B_SU1_11 1 + D8ABC_13 XB__F_11 2",
            "type": "TIE_LINE",
        },
        {
            "original_id": "XB__F_11 B_SU1_11 1 + XB__F_11 D8ABC_12 1",
            "replacement_id": "",
            "type": "TIE_LINE",
        },
    ]
    results = uct_exporter.apply_branch_assignment(
        topology_optimizer_results=test_data,
        lines=test_df,
        trafos=test_df,
        trafo_reg=test_df,
        bus_a=bus_a,
        bus_b=bus_b,
        statistics_all_stations=statistics_all_stations,
    )
    assert results == expected_output


def test_get_switch_group_number(ucte_file_exporter_test):
    input_uct = ucte_file_exporter_test
    with open(input_uct, "r") as f:
        ucte_contents = f.read()
    preamble, nodes, lines, trafos, trafo_reg, postamble = parse_ucte(ucte_contents)
    # test 1 - normal cases
    unique_busbars_ref = [
        "D8ABC_11",
        "D8ABC_12",
        "D8ABC_13",
        "D8ASD_12",
        "D8ASD_13",
        "D8ASD_11",
        "D8TZU_13",
        "D8TZU_12",
        "D8TZU_11",
    ]
    result_ref = [
        "D8ABC_13",
        "D8ABC_13",
        "D8ABC_13",
        "D8ASD_13",
        "D8ASD_13",
        "D8ASD_13",
        "D8TZU_12",
        "D8TZU_12",
        "D8TZU_12",
    ]
    res = []
    for busbars in unique_busbars_ref:
        switches = uct_exporter.find_switches(lines, busbars[0:7])
        grouped_switches = uct_exporter.group_switches(switches)

        reassignment_key = uct_exporter.get_switch_group_number(grouped_switches)
        # manuel check:
        # display(grouped_switches) # -> look for least amount of switches
        # print(reassignment_key) # -> reassignment_key should match the least amount of switches
        res.append(reassignment_key)
    assert res == result_ref

    # test 2 - no switches found
    grouped_switches = {}  # no switches found
    with pytest.raises(ValueError):
        reassignment_key = uct_exporter.get_switch_group_number(grouped_switches)

    # test 3 - all busbars are connected to each other
    unique_busbars_ref = [
        "D8QWE_12",
        "D8QWE_11",
        "D8QWE_13",
    ]

    result_ref = [
        "D8QWE_12",
        "D8QWE_12",
        "D8QWE_12",
    ]
    res = []
    for busbars in unique_busbars_ref:
        switches = uct_exporter.find_switches(lines, busbars[0:7])
        grouped_switches = uct_exporter.group_switches(switches)

        with logbook.handlers.TestHandler() as caplog:
            reassignment_key = uct_exporter.get_switch_group_number(grouped_switches)
            assert "has more than 2 busbars" in "".join(caplog.formatted_records)

        # manuel check:
        # display(grouped_switches) # -> look for least amount of switches
        # print(reassignment_key) # -> reassignment_key should match the least amount of switches
        res.append(reassignment_key)

    assert res == result_ref


def test_process_file(ucte_json_exporter_test, ucte_file_exporter_test, output_uct_exporter_ref, tmp_path):
    expected_output = {
        "changed_ids": {
            "D8ABC_1_0": {
                "branches": [
                    {
                        "original_id": "D8ABC_11 D8DEF_12 1",
                        "replacement_id": "",
                        "type": "LINE",
                    },
                    {
                        "original_id": "D8ABC_11 D8ASD_11 1",
                        "replacement_id": "D8ABC_13 D8ASD_11 1",
                        "type": "LINE",
                    },
                ],
                "injections": {},
                "switches": {
                    "bus_a": "D8ABC_11",
                    "bus_b": "D8ABC_13",
                    "from": "D8ABC_11",
                    "to": "D8ABC_13",
                    "order_of_switches": ["2"],
                },
            }
        }
    }

    output_uct = tmp_path / "test_uct_exporter_uct_file_output.uct"

    output = uct_exporter.process_file(
        input_uct=ucte_file_exporter_test, input_json=ucte_json_exporter_test, output_uct=output_uct, topo_id=0
    )
    assert output == expected_output

    with open(output_uct_exporter_ref, "r") as f:
        ucte_contents_ref = f.read()
    with open(output_uct, "r") as f:
        ucte_contents = f.read()
    assert ucte_contents == ucte_contents_ref

    with pytest.raises(NotImplementedError):
        uct_exporter.process_file(
            input_uct=ucte_file_exporter_test,
            input_json=ucte_json_exporter_test,
            output_uct=output_uct,
            topo_id=0,
            reassign_injections=True,
        )
