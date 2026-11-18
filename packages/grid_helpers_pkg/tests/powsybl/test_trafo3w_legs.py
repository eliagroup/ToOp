# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pypowsybl
from toop_engine_grid_helpers.powsybl.trafo3w_legs import get_trafo3w_id, get_trafo3w_leg_ids


def test_get_trafo3w_leg_ids() -> None:
    """The leg ids are the ids pypowsybl gives the two-winding transformers it creates."""
    net = pypowsybl.network.create_micro_grid_be_network()
    trafo3w_ids = net.get_3_windings_transformers().index.tolist()
    assert trafo3w_ids

    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

    trafo2w_ids = set(net.get_2_windings_transformers().index)
    for trafo3w_id in trafo3w_ids:
        assert set(get_trafo3w_leg_ids(trafo3w_id)) <= trafo2w_ids


def test_get_trafo3w_id() -> None:
    assert get_trafo3w_id("T-Leg-Leg3") == "T-Leg"
    assert get_trafo3w_id("T-Leg4") is None
    assert get_trafo3w_id("TLeg2") is None
