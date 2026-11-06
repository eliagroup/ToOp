# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
from pathlib import Path
from unittest import mock

import pypowsybl
from toop_engine_grid_helpers.powsybl.branch_names import get_branch_names


def test_ucte_branch_names(ucte_file: Path) -> None:
    """UCTE lines and transformers are named ``<id>: <elementName>``, tie lines by their id."""
    net = pypowsybl.network.load(ucte_file)

    assert get_branch_names(net, "LINE")["D8SU1_12 D7SU2_11 1"] == "D8SU1_12 D7SU2_11 1: Test Line"
    assert get_branch_names(net, "TWO_WINDINGS_TRANSFORMER")["B_SU1_11 B_SU1_21 1"] == "B_SU1_11 B_SU1_21 1: Test 2WT 2"
    tie_line_names = get_branch_names(net, "TIE_LINE")
    assert tie_line_names.to_dict() == {tie_line_id: tie_line_id for tie_line_id in net.get_tie_lines().index}


def test_hybrid_branch_names_follow_the_source_format_of_each_branch(ucte_file: Path) -> None:
    """In a hybrid grid, CGMES branches keep their name and all others are named as in UCTE."""
    net = pypowsybl.network.load(ucte_file)
    net._source_format = "hybrid"
    lines = net.get_lines(attributes=["name"])
    cgmes_line_id, ucte_line_id = "D8SU1_12 D8SU1_11 2", "D8SU1_12 D7SU2_11 1"

    with mock.patch("toop_engine_grid_helpers.powsybl.branch_names.get_cgmes_ids", return_value=[cgmes_line_id]):
        names = get_branch_names(net, "LINE")

    assert names[cgmes_line_id] == lines.loc[cgmes_line_id, "name"]
    assert names[ucte_line_id] == "D8SU1_12 D7SU2_11 1: Test Line"


def test_other_format_branch_names() -> None:
    """Branches of other formats are named after their bus-breaker buses and their name; no tie lines give no names."""
    net = pypowsybl.network.create_eurostag_tutorial_example1_network()
    lines = net.get_lines(all_attributes=True)

    names = get_branch_names(net, "LINE")

    assert names.to_dict() == {
        line_id: f"{line.bus_breaker_bus1_id} ## {line.bus_breaker_bus2_id} ## {line['name']}"
        for line_id, line in lines.iterrows()
    }
    assert get_branch_names(net, "TIE_LINE").empty
