# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandas as pd
import pypowsybl
import pytest
from toop_engine_grid_helpers.powsybl.trafo3w_legs import TRAFO3W_LEG_PATTERN, get_trafo3w_id, get_trafo3w_leg_ids


def test_leg_ids_match_pypowsybl_conversion() -> None:
    """The leg ids are the ids pypowsybl gives the two-winding transformers it creates."""
    net = pypowsybl.network.create_micro_grid_be_network()
    trafo3w_ids = net.get_3_windings_transformers().index.tolist()
    assert trafo3w_ids

    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

    trafo2w_ids = set(net.get_2_windings_transformers().index)
    for trafo3w_id in trafo3w_ids:
        assert set(get_trafo3w_leg_ids(trafo3w_id)) <= trafo2w_ids


@pytest.mark.parametrize(
    ("element_id", "trafo3w_id"),
    [
        ("3W-Leg1", "3W"),
        ("T-Leg-Leg3", "T-Leg"),
        ("3W-Leg4", None),
        ("3WLeg2", None),
        ("3W-Leg2-X", None),
        ("L1", None),
    ],
)
def test_get_trafo3w_id(element_id: str, trafo3w_id: str | None) -> None:
    assert get_trafo3w_id(element_id) == trafo3w_id


def test_leg_ids_round_trip() -> None:
    assert [get_trafo3w_id(leg_id) for leg_id in get_trafo3w_leg_ids("T")] == ["T", "T", "T"]


def test_pattern_with_pandas_str_accessor() -> None:
    ids = pd.Index(["T-Leg1", "T-Leg3", "T", "TLeg2"])
    assert ids.str.contains(TRAFO3W_LEG_PATTERN).tolist() == [True, True, False, False]
    assert ids.str.replace(TRAFO3W_LEG_PATTERN, "", regex=True).tolist() == ["T", "T", "T", "TLeg2"]
