# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import tempfile
from copy import deepcopy
from pathlib import Path

import logbook
import pandas as pd
import pytest
from toop_engine_importer.exporter.asset_topology_to_ucte import (
    asset_topo_to_uct,
    change_busbar_coupler_state,
    change_trafos_lines_in_ucte,
    disconnect_line_from_ucte,
    get_changes_from_switching_table,
    get_coupler_state_ucte,
    handle_duplicated_grid_ids,
    update_busbar_name,
    update_coupler_state,
)
from toop_engine_importer.exporter.uct_exporter import validate_ucte_changes
from toop_engine_importer.ucte_toolset.ucte_io import make_ucte, parse_ucte
from toop_engine_interfaces.asset_topology import (
    AssetSetpoint,
    Busbar,
    BusbarCoupler,
)


def test_get_coupler_state_ucte():
    # Test one open coupler
    couplers = [
        BusbarCoupler(grid_model_id="coupler1", open=True, busbar_from_id=1, busbar_to_id=2),
    ]
    expected = [
        {"grid_model_id": "coupler1", "coupler_state_ucte": 7},
    ]
    result = get_coupler_state_ucte(couplers)
    assert result == expected

    # Test all closed couplers
    couplers = [
        BusbarCoupler(grid_model_id="coupler1", open=False, busbar_from_id=1, busbar_to_id=2),
        BusbarCoupler(grid_model_id="coupler2", open=False, busbar_from_id=1, busbar_to_id=2),
    ]
    expected = [
        {"grid_model_id": "coupler1", "coupler_state_ucte": 2},
        {"grid_model_id": "coupler2", "coupler_state_ucte": 2},
    ]
    result = get_coupler_state_ucte(couplers)
    assert result == expected

    # Test one open and one closed coupler
    couplers = [
        BusbarCoupler(grid_model_id="coupler1", open=True, busbar_from_id=1, busbar_to_id=2),
        BusbarCoupler(grid_model_id="coupler2", open=False, busbar_from_id=1, busbar_to_id=2),
    ]
    expected = [
        {"grid_model_id": "coupler1", "coupler_state_ucte": 7},
        {"grid_model_id": "coupler2", "coupler_state_ucte": 2},
    ]
    result = get_coupler_state_ucte(couplers)
    assert result == expected
    # Test empty list
    couplers = []
    expected = []
    result = get_coupler_state_ucte(couplers)
    assert result == expected


def test_update_busbars_name():
    # Test case where 'from' is updated
    row = pd.Series(
        {
            "from": "busbar1",
            "to": "busbar2",
            "initial_busbar": "busbar1",
            "final_busbar": "busbar3",
        }
    )
    expected = pd.Series(
        {
            "from": "busbar3",
            "to": "busbar2",
            "initial_busbar": "busbar1",
            "final_busbar": "busbar3",
        }
    )
    result = update_busbar_name(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where 'from' and 'to' are not updated
    row = pd.Series(
        {
            "from": "busbar1",
            "to": "busbar2",
            "initial_busbar": "busbar4",
            "final_busbar": "busbar3",
        }
    )
    expected = pd.Series(
        {
            "from": "busbar1",
            "to": "busbar2",
            "initial_busbar": "busbar4",
            "final_busbar": "busbar3",
        }
    )
    result = update_busbar_name(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where only 'to' is updated
    row = pd.Series(
        {
            "from": "busbar1",
            "to": "busbar2",
            "initial_busbar": "busbar2",
            "final_busbar": "busbar3",
        }
    )
    expected = pd.Series(
        {
            "from": "busbar1",
            "to": "busbar3",
            "initial_busbar": "busbar2",
            "final_busbar": "busbar3",
        }
    )
    result = update_busbar_name(row)
    pd.testing.assert_series_equal(result, expected)


def test_change_trafos_lines_in_ucte():
    # Test case where 'from' and 'to' are updated
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    asset_change_list = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "initial_busbar": "busbar1",
                "final_busbar": "busbar3",
            },
            {
                "grid_model_id": "busbar2 busbar3 1",
                "initial_busbar": "busbar2",
                "final_busbar": "busbar4",
            },
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar3", "busbar4"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    change_trafos_lines_in_ucte(ucte_df, asset_change_list)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where no changes are made
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    asset_change_list = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "initial_busbar": "busbar4",
                "final_busbar": "busbar5",
            },
            {
                "grid_model_id": "busbar2 busbar3 1",
                "initial_busbar": "busbar6",
                "final_busbar": "busbar7",
            },
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    change_trafos_lines_in_ucte(ucte_df, asset_change_list)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where only 'from' is updated
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    asset_change_list = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "initial_busbar": "busbar1",
                "final_busbar": "busbar3",
            }
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar3", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    change_trafos_lines_in_ucte(ucte_df, asset_change_list)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where only 'to' is updated
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    asset_change_list = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "initial_busbar": "busbar2",
                "final_busbar": "busbar4",
            }
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar4", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    change_trafos_lines_in_ucte(ucte_df, asset_change_list)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where a line gets disconnected
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["0", "0"],
        }
    )
    asset_change_list = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "initial_busbar": None,
                "final_busbar": None,
            }
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
            "status": ["8", "0"],
        }
    )
    change_trafos_lines_in_ucte(ucte_df, asset_change_list)
    pd.testing.assert_frame_equal(ucte_df, expected)


def test_get_changes_from_switching_table(ucte_asset_topology, caplog):
    topology_model = deepcopy(ucte_asset_topology)

    # Test case where asset is reassigned
    station = topology_model.stations[0]
    expected = []
    result = get_changes_from_switching_table(station)
    assert result == expected

    topology_model.stations[0].asset_switching_table[0][3] = False
    topology_model.stations[0].asset_switching_table[1][3] = True
    station = topology_model.stations[0]
    expected = [
        {
            "grid_model_id": "D8SU1_11 D8SU1_21 1",
            "initial_busbar": "D8SU1_11",
            "final_busbar": "D8SU1_12",
            "asset_type": "TWO_WINDINGS_TRANSFORMER",
        }
    ]
    result = get_changes_from_switching_table(station)
    assert result == expected

    # Test case where asset is connected to multiple busbars (should raise ValueError)
    topology_model.stations[0].asset_switching_table[0][3] = True
    topology_model.stations[0].asset_switching_table[1][3] = True
    station = topology_model.stations[0]
    with pytest.raises(ValueError):
        get_changes_from_switching_table(station)

    # test case disconnected asset
    topology_model.stations[0].asset_switching_table[0][3] = False
    topology_model.stations[0].asset_switching_table[1][3] = False
    station = topology_model.stations[0]
    expected = [
        {
            "grid_model_id": "D8SU1_11 D8SU1_21 1",
            "initial_busbar": None,
            "final_busbar": None,
            "asset_type": "TWO_WINDINGS_TRANSFORMER",
        }
    ]
    result = get_changes_from_switching_table(station)
    assert result == expected

    # test where an asset is connected to two busbars within a station an is now reassigned
    topology_model.stations[0].asset_switching_table[0][3] = False
    topology_model.stations[0].asset_switching_table[1][3] = True
    topology_model.stations[0].busbars.append(
        Busbar(grid_model_id="D8SU1_13", type=None, name="", int_id=0, in_service=True)
    )
    topology_model.stations[0].assets[3].grid_model_id = "D8SU1_11 D8SU1_13 1"
    station = topology_model.stations[0]
    with pytest.raises(ValueError):
        get_changes_from_switching_table(station)

    # Test case where busbar connection is not found
    topology_model.stations[0].asset_switching_table[0][3] = False
    topology_model.stations[0].asset_switching_table[1][3] = True
    topology_model.stations[0].assets[3].grid_model_id = "NOT_A_VALID_ID"
    station = topology_model.stations[0]
    with pytest.raises(ValueError):
        get_changes_from_switching_table(station)


def test_update_coupler_state():
    # Test case where status is 2 and should be updated
    row = pd.Series({"status": "2", "coupler_state_ucte": "7", "grid_model_id": "line1"})
    expected = pd.Series({"status": "7", "coupler_state_ucte": "7", "grid_model_id": "line1"})
    result = update_coupler_state(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where status is 7 and should be updated
    row = pd.Series({"status": "7", "coupler_state_ucte": "2", "grid_model_id": "line2"})
    expected = pd.Series({"status": "2", "coupler_state_ucte": "2", "grid_model_id": "line2"})
    result = update_coupler_state(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where status is neither 2 nor 7 and should raise a warning
    # 1 -> in_service, 8 -> out_of_service
    # corresponding statulen(caplog.messages) == 1s codes in UCTE for 1 is 9
    # warning gives you a hint that you tried to close a couple, that is not of type coupler
    row = pd.Series({"status": "1", "coupler_state_ucte": "8", "grid_model_id": "line3"})
    expected = pd.Series({"status": "9", "coupler_state_ucte": "8", "grid_model_id": "line3"})
    with logbook.handlers.TestHandler() as caplog:
        result = update_coupler_state(row)
        pd.testing.assert_series_equal(result, expected)

        assert "has a status different from 2 or 7 with status" in caplog.formatted_records[0]
        assert len(caplog.formatted_records) == 1


def test_change_busbar_coupler_state():
    # Test case where 'status' is updated
    lines_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["2", "2"],
            "order": ["1", "1"],
        }
    )
    change_df = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "coupler_state_ucte": "7",
            },
            {
                "grid_model_id": "busbar2 busbar3 1",
                "coupler_state_ucte": "7",
            },
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["7", "7"],
            "order": ["1", "1"],
        }
    )
    change_busbar_coupler_state(lines_df, change_df)
    pd.testing.assert_frame_equal(lines_df, expected)

    # Test case where no changes are made
    lines_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["2", "2"],
            "order": ["1", "1"],
        }
    )
    change_df = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar3 busbar4 1",
                "coupler_state_ucte": "7",
            },
        ]
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["2", "2"],
            "order": ["1", "1"],
        }
    )
    change_busbar_coupler_state(lines_df, change_df)
    pd.testing.assert_frame_equal(lines_df, expected)

    # Test case where 'status' is updated with a warning
    lines_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["1", "1"],
            "order": ["1", "1"],
        }
    )
    change_df = pd.DataFrame(
        [
            {
                "grid_model_id": "busbar1 busbar2 1",
                "coupler_state_ucte": "8",
            },
        ]
    )
    # 1 -> in_service, 8 -> out_of_service
    # corresponding status codes in UCTE for 1 is 9
    # warning gives you a hint that you tried to close a couple, that is not of type coupler
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "status": ["9", "1"],
            "order": ["1", "1"],
        }
    )
    with logbook.handlers.TestHandler() as caplog:
        change_busbar_coupler_state(lines_df, change_df)
        pd.testing.assert_frame_equal(lines_df, expected)
        assert "has a status different from 2 or 7 with status" in caplog.formatted_records[0]
        assert len(caplog.formatted_records) == 1


def test_disconnect_asset_from_ucte():
    # Test case where asset is in service and should be disconnected
    row = pd.Series({"status": "0", "grid_model_id": "asset1"})
    expected = pd.Series({"status": "8", "grid_model_id": "asset1"})
    result = disconnect_line_from_ucte(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where asset is already out of service and should remain unchanged
    row = pd.Series({"status": "7", "grid_model_id": "asset2"})
    expected = pd.Series({"status": "7", "grid_model_id": "asset2"})
    result = disconnect_line_from_ucte(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where asset is in service with a different status code and should be disconnected
    row = pd.Series({"status": "1", "grid_model_id": "asset3"})
    expected = pd.Series({"status": "9", "grid_model_id": "asset3"})
    result = disconnect_line_from_ucte(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where asset is out of service with a different status code and should remain unchanged
    row = pd.Series({"status": "8", "grid_model_id": "asset4"})
    expected = pd.Series({"status": "8", "grid_model_id": "asset4"})
    result = disconnect_line_from_ucte(row)
    pd.testing.assert_series_equal(result, expected)

    # Test case where asset has an unkown status
    row = pd.Series({"status": "10", "grid_model_id": "asset5"})

    with pytest.raises(KeyError):
        disconnect_line_from_ucte(row)


def test_asset_topo_to_uct(ucte_asset_topology, ucte_file):
    topology_model = deepcopy(ucte_asset_topology)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_ucte = tmp / "output.uct"
        # test case where no changes are made -> input and output should be the same
        asset_topo_to_uct(
            asset_topology=topology_model,
            grid_model_file_output=output_ucte,
            grid_model_file_input=ucte_file,
        )
        with open(ucte_file, "r") as f:
            input_uct_contents = f.read()
        with open(output_ucte, "r") as f:
            output_uct_contents = f.read()
        validate_ucte_changes(ucte_contents=input_uct_contents, ucte_contents_out=output_uct_contents)
        assert input_uct_contents == output_uct_contents

        # test case where no input file is given -> use from asset_topology
        asset_topo_to_uct(
            asset_topology=topology_model,
            grid_model_file_output=output_ucte,
        )
        with open(output_ucte, "r") as f:
            output_uct_contents = f.read()
        validate_ucte_changes(ucte_contents=input_uct_contents, ucte_contents_out=output_uct_contents)
        assert input_uct_contents == output_uct_contents

        # test not implemented
        ucte_asset_topology.asset_setpoints = [AssetSetpoint(grid_model_id="D8SU1_11", setpoint=1.0)]
        with pytest.raises(NotImplementedError):
            asset_topo_to_uct(
                asset_topology=ucte_asset_topology,
                grid_model_file_output=output_ucte,
                grid_model_file_input=ucte_file,
            )

        # Test case where asset is reassigned
        # test trafo
        topology_model.stations[0].asset_switching_table[0][3] = False
        topology_model.stations[0].asset_switching_table[1][3] = True
        # test line
        topology_model.stations[0].asset_switching_table[0][4] = True
        topology_model.stations[0].asset_switching_table[1][4] = False
        # test coupler
        topology_model.stations[0].couplers[0].open = True

        with open(ucte_file, "r") as f:
            input_uct_contents = f.read()
        preamble, nodes, lines, trafos, trafo_reg, postamble = parse_ucte(input_uct_contents)
        # test order change of line
        # original grid id: "D2SU1_31 D2SU1_31 2"
        topology_model.stations[0].assets[1].grid_model_id = "D8SU1_12 D7SU1_11 1"
        lines.iloc[4, 2] = "1"
        lines.iloc[4, 0] = "D8SU1_11"
        lines.iloc[4, 1] = "D7SU2_11"
        output_ucte_str = make_ucte(preamble, nodes, lines, trafos, trafo_reg, postamble)
        test_ucte = tmp / "test.uct"
        with open(test_ucte, "w") as f:
            f.write(output_ucte_str)

        # run test
        asset_topo_to_uct(
            asset_topology=topology_model,
            grid_model_file_output=output_ucte,
            grid_model_file_input=test_ucte,
        )

        with open(output_ucte, "r") as f:
            output_uct_contents = f.read()
        with open(test_ucte, "r") as f:
            input_uct_contents = f.read()
        validate_ucte_changes(ucte_contents=input_uct_contents, ucte_contents_out=output_uct_contents)
        _, _, lines, trafos, _, _ = parse_ucte(output_uct_contents)
        _, _, lines_org, trafos_org, _, _ = parse_ucte(input_uct_contents)
        # test coupler
        assert lines_org.iloc[0]["status"] == "2"
        assert lines.iloc[0]["status"] == "7"
        # test line
        assert lines_org.iloc[2]["from"] == "D8SU1_12"
        assert lines.iloc[2]["from"] == "D8SU1_11"
        # test trafo
        assert trafos_org.iloc[2]["from"] == "D8SU1_11"
        assert trafos.iloc[2]["from"] == "D8SU1_12"
        # test order change of line
        assert lines.iloc[4]["from"] == "D8SU1_11"
        assert lines.iloc[4]["to"] == "D7SU2_11"
        assert lines.iloc[4]["order"] == "2"
        assert lines.iloc[2]["to"] == "D7SU2_11"
        assert lines.iloc[2]["order"] == "1"


def test_handle_duplicated_grid_ids():
    # Test case where no duplicates exist
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar2"],
            "to": ["busbar2", "busbar3"],
            "order": ["1", "1"],
        }
    )
    expected = ucte_df.copy()
    handle_duplicated_grid_ids(ucte_df)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where duplicates exist and are resolved
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar1"],
            "to": ["busbar2", "busbar2"],
            "order": ["1", "1"],
        }
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar1"],
            "to": ["busbar2", "busbar2"],
            "order": ["1", "2"],
        }
    )
    handle_duplicated_grid_ids(ucte_df)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where duplicates exist and require multiple iterations to resolve
    ucte_df = pd.DataFrame(
        {
            "from": ["busbar1", "busbar1", "busbar1"],
            "to": ["busbar2", "busbar2", "busbar2"],
            "order": ["1", "1", "1"],
        }
    )
    expected = pd.DataFrame(
        {
            "from": ["busbar1", "busbar1", "busbar1"],
            "to": ["busbar2", "busbar2", "busbar2"],
            "order": ["1", "2", "3"],
        }
    )
    handle_duplicated_grid_ids(ucte_df)
    pd.testing.assert_frame_equal(ucte_df, expected)

    # Test case where duplicates cannot be resolved within 20 iterations
    ucte_df = pd.DataFrame({"from": ["busbar1"] * 22, "to": ["busbar2"] * 22, "order": ["1"] * 22})
    with pytest.raises(
        ValueError,
        match="Duplicated grid_model_ids could not be resolved. More than 20 iterations have been reached.",
    ):
        handle_duplicated_grid_ids(ucte_df)
