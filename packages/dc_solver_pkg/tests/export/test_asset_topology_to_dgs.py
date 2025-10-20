import io
import os
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pandera
import pytest
from toop_engine_dc_solver.export.asset_topology_to_dgs import (
    ForeignIdSchema,
    SwitchUpdateSchema,
    get_asset_bay_grid_model_id_list,
    get_asset_switch_states_from_station,
    get_busbar_lookup,
    get_changing_switches_from_topology,
    get_coupler_states_from_busbar_couplers,
    get_dgs_general_schema,
    get_diff_switch_states,
    get_switch_update_schema_from_topology,
    switch_dgs_schema_to_bytes_io,
    switch_dgs_schema_to_xlsx,
    switch_update_schema_to_dgs,
)
from toop_engine_dc_solver.export.dgs_v7_definitions import (
    DGS_GENERAL_SHEET_CONTENT_FID,
    DGS_GENERAL_SHEET_CONTENT_FID_CIM,
    DgsElmCoupSchema,
)
from toop_engine_interfaces.asset_topology import BusbarCoupler


def test_get_dgs_general_schema():
    general_info = DGS_GENERAL_SHEET_CONTENT_FID
    general_df = get_dgs_general_schema(general_info=general_info, cim=False)

    expected = pd.DataFrame(general_info)
    assert general_df.equals(expected)

    general_info = DGS_GENERAL_SHEET_CONTENT_FID_CIM
    general_df = get_dgs_general_schema(general_info=general_info, cim=False)

    expected_cim = pd.DataFrame(general_info)
    assert general_df.equals(expected_cim)

    general_df = get_dgs_general_schema(cim=False)
    assert general_df.equals(expected)

    general_df = get_dgs_general_schema(cim=True)
    assert general_df.equals(expected_cim)

    general_df = get_dgs_general_schema()
    assert general_df.equals(expected_cim)

    # wrong schema
    general_info = [{"ID": "1", "Descr": "Version", "Val": "7.0"}]
    with pytest.raises(pandera.errors.SchemaError):
        get_dgs_general_schema(general_info=general_info, cim=False)


def test_dgs_list_to_xlsx(tmp_path):
    dgs_df = pd.DataFrame(
        [
            {"FID(a:40)": "Some-unique-id-1", "on_off": 0, "OP": "U"},
            {"FID(a:40)": "Some-unique-id-2", "on_off": 1, "OP": "U"},
        ]
    )
    DgsElmCoupSchema.validate(dgs_df)
    file_name = os.path.join(tmp_path, "test_dgs.xlsx")
    assert not os.path.exists(file_name)
    sheet_name = "ElmCoup"
    general_info = DGS_GENERAL_SHEET_CONTENT_FID
    general_df = get_dgs_general_schema(general_info=general_info, cim=False)
    switch_dgs_schema_to_xlsx(switch_dgs_schema=dgs_df, file_name=file_name, sheet_name=sheet_name, df_general=general_df)

    assert os.path.exists(file_name)

    with pd.ExcelFile(file_name) as xls:
        df_general = pd.read_excel(xls, sheet_name="General")
        df = pd.read_excel(xls, sheet_name=sheet_name)
    df_general = df_general.astype(str)
    assert df_general.equals(general_df)
    assert df.equals(dgs_df)

    # default general info
    file_name = os.path.join(tmp_path, "test2_dgs.xlsx")
    switch_dgs_schema_to_xlsx(switch_dgs_schema=dgs_df, file_name=file_name, sheet_name=sheet_name, df_general=general_df)
    assert os.path.exists(file_name)

    with pd.ExcelFile(file_name) as xls:
        df_general = pd.read_excel(xls, sheet_name="General")
        df = pd.read_excel(xls, sheet_name=sheet_name)
    df_general = df_general.astype(str)
    assert df_general.equals(general_df)
    assert df.equals(dgs_df)


def test_get_coupler_states_from_busbar_couplers():
    # Mocking the BusbarCoupler object
    mock_coupler_1 = MagicMock(spec=BusbarCoupler)
    mock_coupler_1.grid_model_id = "coupler_1"
    mock_coupler_1.open = True
    mock_coupler_1.in_service = True
    mock_coupler_2 = MagicMock(spec=BusbarCoupler)
    mock_coupler_2.grid_model_id = "coupler_2"
    mock_coupler_2.open = False
    mock_coupler_2.in_service = True

    busbar_couplers = [mock_coupler_1, mock_coupler_2]

    result = get_coupler_states_from_busbar_couplers(busbar_couplers)

    expected_result = {"grid_model_id": {0: "coupler_1", 1: "coupler_2"}, "open": {0: True, 1: False}}

    assert result.equals(pd.DataFrame(expected_result))

    mock_coupler_2.in_service = False
    with pytest.raises(ValueError):
        get_coupler_states_from_busbar_couplers(busbar_couplers)


def test_get_asset_switch_states_from_station(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
    expected_reassignment = [
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
        {"grid_model_id": "L52_DISCONNECTOR_5_0", "open": True},
        {"grid_model_id": "L52_DISCONNECTOR_5_1", "open": False},
    ]
    expected_disconnection = [{"grid_model_id": "L82_BREAKER", "open": True}]
    assert switch_disconnection_df.to_dict(orient="records") == expected_disconnection
    assert switch_reassignment_df.to_dict(orient="records") == expected_reassignment

    # test empty disconnection
    station = station.model_copy(
        update={
            "asset_switching_table": np.array([[False, False, False], [True, True, True]]),
        }
    )
    switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
    expected_reassignment = [
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
        {"grid_model_id": "L52_DISCONNECTOR_5_0", "open": True},
        {"grid_model_id": "L52_DISCONNECTOR_5_1", "open": False},
        {
            "grid_model_id": "L82_DISCONNECTOR_7_0",
            "open": True,
        },
        {
            "grid_model_id": "L82_DISCONNECTOR_7_1",
            "open": False,
        },
    ]
    assert switch_disconnection_df.empty
    assert switch_reassignment_df.to_dict(orient="records") == expected_reassignment

    # test empty reassignment
    station = station.model_copy(update={"asset_switching_table": np.array([[False, False, False], [False, False, False]])})
    switch_reassignment_df, switch_disconnection_df = get_asset_switch_states_from_station(station)
    expected_disconnection = [
        {"grid_model_id": "L42_BREAKER", "open": True},
        {"grid_model_id": "L52_BREAKER", "open": True},
        {"grid_model_id": "L82_BREAKER", "open": True},
    ]
    assert switch_disconnection_df.to_dict(orient="records") == expected_disconnection
    assert switch_reassignment_df.empty


def test_get_asset_bay_sr_fid_list(basic_node_breaker_topology):
    station = deepcopy(basic_node_breaker_topology.stations[0])
    asset_bay_sr_fid_list = get_asset_bay_grid_model_id_list(station)
    expected = [
        {"BBS4_1": "L42_DISCONNECTOR_3_0", "BBS4_2": "L42_DISCONNECTOR_3_1"},
        {"BBS4_1": "L52_DISCONNECTOR_5_0", "BBS4_2": "L52_DISCONNECTOR_5_1"},
        {"BBS4_1": "L82_DISCONNECTOR_7_0", "BBS4_2": "L82_DISCONNECTOR_7_1"},
    ]
    assert asset_bay_sr_fid_list == expected

    station.assets[1] = station.assets[1].model_copy(
        update={
            "asset_bay": None,
        }
    )
    asset_bay_sr_fid_list = get_asset_bay_grid_model_id_list(station)
    expected = [
        {"BBS4_1": "L42_DISCONNECTOR_3_0", "BBS4_2": "L42_DISCONNECTOR_3_1"},
        None,
        {"BBS4_1": "L82_DISCONNECTOR_7_0", "BBS4_2": "L82_DISCONNECTOR_7_1"},
    ]
    assert asset_bay_sr_fid_list == expected


def test_get_busbar_lookup(basic_node_breaker_topology):
    station = basic_node_breaker_topology.stations[0]
    busbar_lookup = get_busbar_lookup(station)
    expected = {0: "BBS4_1", 1: "BBS4_2"}
    assert busbar_lookup == expected


def test_get_switch_update_schema_from_topology(basic_node_breaker_topology):
    topology = basic_node_breaker_topology
    switch_update_schema = get_switch_update_schema_from_topology(topology)
    expected = pd.DataFrame(
        [
            {"grid_model_id": "VL4_BREAKER", "open": True},
            {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
            {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
            {"grid_model_id": "L52_DISCONNECTOR_5_0", "open": True},
            {"grid_model_id": "L52_DISCONNECTOR_5_1", "open": False},
            {"grid_model_id": "L82_BREAKER", "open": True},
        ]
    )
    assert switch_update_schema.equals(expected)


def test_get_diff_switch_states(basic_node_breaker_grid_v1, basic_node_breaker_topology):
    net = basic_node_breaker_grid_v1
    topology = basic_node_breaker_topology
    switch_update_schema = get_switch_update_schema_from_topology(topology)
    diff_switch_states = get_diff_switch_states(network=net, switch_df=switch_update_schema)
    SwitchUpdateSchema.validate(diff_switch_states)
    expected = [
        {"grid_model_id": "VL4_BREAKER", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
        {"grid_model_id": "L82_BREAKER", "open": True},
    ]
    assert diff_switch_states.to_dict(orient="records") == expected


def test_get_changing_switches_from_topology(basic_node_breaker_grid_v1, basic_node_breaker_topology):
    net = basic_node_breaker_grid_v1
    topology = basic_node_breaker_topology
    diff_switch_states = get_changing_switches_from_topology(network=net, target_topology=topology)
    SwitchUpdateSchema.validate(diff_switch_states)
    expected = [
        {"grid_model_id": "VL4_BREAKER", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_0", "open": True},
        {"grid_model_id": "L42_DISCONNECTOR_3_1", "open": False},
        {"grid_model_id": "L82_BREAKER", "open": True},
    ]
    assert diff_switch_states.to_dict(orient="records") == expected


def test_switch_update_schema_to_dgs(basic_node_breaker_grid_v1, basic_node_breaker_topology):
    net = basic_node_breaker_grid_v1
    realized_topology = basic_node_breaker_topology
    switch_update_schema = get_switch_update_schema_from_topology(realized_topology)
    foreign_ids = deepcopy(switch_update_schema)
    foreign_ids["foreign_id"] = foreign_ids["grid_model_id"] + "_foreign_id"
    foreign_ids.drop(columns=["open"], inplace=True)
    ForeignIdSchema.validate(foreign_ids)
    dgs_df = switch_update_schema_to_dgs(switch_update_schema, foreign_ids, cim=False)

    # check if dgs switch states are correct
    # dgs: on_off = 0 for open, 1 for closed
    # powsybl: open = True for open, False for closed
    assert dgs_df.loc[0, "on_off"] == 0
    assert switch_update_schema.loc[0, "open"] == True
    assert all(dgs_df["FID(a:40)"] == foreign_ids["foreign_id"])
    # U for Update
    assert list(dgs_df["OP"].unique()) == ["U"]
    assert len(dgs_df) == len(switch_update_schema)

    dgs_df = switch_update_schema_to_dgs(switch_update_schema, foreign_ids, cim=True)
    foreign_ids["foreign_id"] = "_" + foreign_ids["foreign_id"]
    # check if dgs switch states are correct
    # dgs: on_off = 0 for open, 1 for closed
    # powsybl: open = True for open, False for closed
    assert dgs_df.loc[0, "on_off"] == 0
    assert switch_update_schema.loc[0, "open"] == True
    assert all(dgs_df["FID(a:40)"] == foreign_ids["foreign_id"])
    # U for Update
    assert list(dgs_df["OP"].unique()) == ["U"]
    assert len(dgs_df) == len(switch_update_schema)


def test_switch_dgs_schema_to_bytes_io(tmp_path):
    # Prepare a valid DgsElmCoupSchema DataFrame
    dgs_df = pd.DataFrame(
        [
            {"FID(a:40)": "Some-unique-id-1", "on_off": 0, "OP": "U"},
            {"FID(a:40)": "Some-unique-id-2", "on_off": 1, "OP": "U"},
        ]
    )
    DgsElmCoupSchema.validate(dgs_df)
    general_df = get_dgs_general_schema(cim=False)

    # Call the function
    output = switch_dgs_schema_to_bytes_io(dgs_df, general_df, sheet_name="ElmCoup")

    # Check that output is a BytesIO and not empty
    assert isinstance(output, io.BytesIO)
    assert output.getbuffer().nbytes > 0

    # Save to a file and read back with pandas to check content
    file_path = tmp_path / "test_bytes_io.xlsx"
    with open(file_path, "wb") as f:
        f.write(output.getvalue())

    with pd.ExcelFile(file_path) as xls:
        df_general = pd.read_excel(xls, sheet_name="General")
        df = pd.read_excel(xls, sheet_name="ElmCoup")

    # Compare content
    df_general = df_general.astype(str)
    assert df_general.equals(general_df)
    assert df.equals(dgs_df)
