import pandas as pd
import pypowsybl
from toop_engine_importer.pypowsybl_import import dacf_whitelists, network_analysis


def test_get_element_id_for_cb_df(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    # test 1
    branches_with_elementname = network_analysis.get_branches_df_with_element_name(network)
    branches_with_elementname["pairing_key"] = branches_with_elementname["pairing_key"].str[0:7]
    # Create a sample DataFrame
    cb_df = pd.DataFrame(
        {
            "Anfangsknoten": ["D8SU1_1", "CB2", "CB3"],
            "Endknoten": ["D8SU1_1", "CB2", "CB3"],
            "Elementname": ["Test C. Line", "CB2", "CB3"],
            "Auslastungsgrenze_n_0": [1, 2, 3],
            "Auslastungsgrenze_n_1": [1, 2, 3],
        }
    )

    # Call the function
    dacf_whitelists.assign_element_id_to_cb_df(branches_with_elementname=branches_with_elementname, cb_df=cb_df)
    result = cb_df[cb_df["element_id"].notnull()]["element_id"].values
    expected = ["D8SU1_12 D8SU1_11 2"]
    assert len(result) == 1
    assert all(result == expected)

    # test 2
    branches_with_elementname = network_analysis.get_branches_df_with_element_name(network)
    branches_with_elementname["pairing_key"] = branches_with_elementname["pairing_key"].str[0:7]
    # Create a sample DataFrame
    cb_df = pd.DataFrame(
        {
            "Anfangsknoten": [
                "D8SU1_1",
                "XB__F_1",
                "XB__F_1",
                "XB__F_1",
                "D8SU1_1",
                "D7SU2_1",
            ],
            "Endknoten": [
                "D8SU1_1",
                "B_SU1_1",
                "B_SU1_1",
                "B_SU1_1",
                "D7SU2_1",
                "D8SU1_1",
            ],
            "Elementname": [
                "Test C. Line",
                "TL 1/2",
                "TL 1/1",
                "Test C. Line",
                "Test Line 2",
                "Test Line 2",
            ],
            "Auslastungsgrenze_n_0": [1, 2, 3, 4, 5, 6],
            "Auslastungsgrenze_n_1": [1, 2, 3, 4, 5, 6],
        }
    )
    expected = [
        "D8SU1_12 D8SU1_11 2",
        "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
        "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D8SU1_12 D7SU2_11 2",
    ]
    dacf_whitelists.assign_element_id_to_cb_df(branches_with_elementname=branches_with_elementname, cb_df=cb_df)
    result = cb_df[cb_df["element_id"].notnull()]["element_id"].values
    assert len(result) == 5
    assert all(result == expected)


def test_apply_white_list_to_operational_limits(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    white_list_df = pd.DataFrame(
        {
            "Anfangsknoten": {0: "D8SU1_1", 1: "D8SU1_1", 2: "XB__F_1"},
            "Endknoten": {0: "D8SU1_1", 1: "D7SU2_1", 2: "B_SU1_1"},
            "Elementname": {0: "Test C. Line", 1: "Test Line 2", 2: "TL 1/1"},
            "Auslastungsgrenze_n_0": {0: 110, 1: 80, 2: 90},
            "Auslastungsgrenze_n_1": {0: 90, 1: 80, 2: 80},
            "element_id": {
                0: "D8SU1_12 D8SU1_11 2",
                1: "D8SU1_12 D7SU2_11 2",
                2: "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
            },
        }
    )
    expected = {
        "element_id": {
            0: "D8SU1_12 D8SU1_11 2",
            1: "D8SU1_12 D8SU1_11 2",
            2: "D8SU1_12 D8SU1_11 2",
            3: "D8SU1_12 D8SU1_11 2",
            4: "D8SU1_12 D7SU2_11 2",
            5: "D8SU1_12 D7SU2_11 2",
            6: "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
            7: "XB__F_11 B_SU1_11 1 + XB__F_11 D8SU1_11 1",
        },
        "value": {
            0: 5500.0,
            1: 4500.0,
            2: 5500.0,
            3: 4500.0,
            4: 80.0,
            5: 80.0,
            6: 5000.0,
            7: 5000.0,
        },
    }
    op_lim_org = network.get_operational_limits()
    op_lim_org = op_lim_org[op_lim_org.index.isin(white_list_df["element_id"].to_list())]
    assert len(op_lim_org) == 6
    dacf_whitelists.apply_white_list_to_operational_limits(network=network, white_list_df=white_list_df)
    op_lim_new = network.get_operational_limits()
    op_lim_new = op_lim_new[op_lim_new.index.isin(white_list_df["element_id"].to_list())]
    assert len(op_lim_new) == 8
    assert op_lim_new["value"].reset_index().to_dict() == expected
