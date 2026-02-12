# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import logbook
import pandas as pd
import pypowsybl
from fsspec.implementations.local import LocalFileSystem
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
from toop_engine_importer.pypowsybl_import.network_analysis import (
    apply_cb_lists,
    convert_low_impedance_lines,
    remove_branches_across_switch,
    remove_branches_with_same_bus,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    ImportResult,
)


def test_convert_low_impedance_lines(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D8SU1_11 2",
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]
    convert_low_impedance_lines(network, "D8")
    network.get_lines()
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "D2SU1_31 D2SU1_31 2",
        "B_SU2_11 B_SU1_11 1",
    ]
    assert "D8SU1_12 D8SU1_11 2" in network.get_switches().index.to_list()
    convert_low_impedance_lines(network, "D2")
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

    remove_branches_across_switch(network)
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
    remove_branches_across_switch(network)
    final_branch_count = len(network.get_branches())
    assert final_branch_count + 1 == initial_branch_count
    assert network.get_lines().index.to_list() == [
        "D8SU1_12 D8SU1_11 2",
        "D8SU1_12 D7SU2_11 1",
        "D8SU1_12 D7SU2_11 2",
        "B_SU2_11 B_SU1_11 1",
    ]


def test_apply_cb_lists(ucte_file, ucte_importer_parameters):
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
    import_result = ImportResult(
        data_folder=ucte_importer_parameters.data_folder,
    )

    white_list_file = ucte_importer_parameters.data_folder / "CB_White-Liste.csv"
    black_list_file = ucte_importer_parameters.data_folder / "CB_Black-Liste.csv"
    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=import_result,
        border_current={},
        network_changes={},
        import_parameter=ucte_importer_parameters,
    )
    apply_cb_lists(
        network=network,
        statistics=statistics,
        white_list_file=white_list_file,
        black_list_file=black_list_file,
        fs=LocalFileSystem(),
    )
    assert statistics.id_lists["white_list"] == ["D8SU1_12 D8SU1_11 2"]
    assert statistics.id_lists["black_list"] == ["D8SU1_12 D7SU2_11 2"]
    assert statistics.import_result.n_white_list == 1
    assert statistics.import_result.n_black_list == 1
    assert statistics.import_result.n_black_list_applied == 1
    assert statistics.import_result.n_white_list_applied == 1

    # test 2 - apply black list only
    import_result = ImportResult(
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
    apply_cb_lists(
        network=network,
        statistics=statistics,
        white_list_file=None,
        black_list_file=black_list_file,
        fs=LocalFileSystem(),
    )
    assert statistics.id_lists["black_list"] == ["D8SU1_12 D7SU2_11 2"]
    assert "white_list" in statistics.id_lists
    assert statistics.id_lists["white_list"] == []
    assert statistics.import_result.n_white_list == 0
    assert statistics.import_result.n_black_list == 1
    assert statistics.import_result.n_black_list_applied == 1
    assert statistics.import_result.n_white_list_applied == 0

    # test 3 - apply no list
    import_result = ImportResult(
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
    apply_cb_lists(
        network=network,
        statistics=statistics,
        white_list_file=None,
        black_list_file=None,
        fs=LocalFileSystem(),
    )
    assert "white_list" in statistics.id_lists
    assert "black_list" in statistics.id_lists
    assert statistics.id_lists["white_list"] == []
    assert statistics.id_lists["black_list"] == []
    assert statistics.import_result.n_white_list == 0
    assert statistics.import_result.n_black_list == 0
    assert statistics.import_result.n_black_list_applied == 0
    assert statistics.import_result.n_white_list_applied == 0


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
        remove_branches_with_same_bus(network=network)
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
