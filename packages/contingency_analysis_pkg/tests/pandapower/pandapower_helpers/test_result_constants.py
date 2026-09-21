# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower as pp
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from toop_engine_contingency_analysis.pandapower import translate_nminus1_for_pandapower
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.result_constants import ResultConstants
from toop_engine_grid_helpers.pandapower.outage_group import ConnectivityGraphCache
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import SwitchElementMappingSchema
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition


@pytest.fixture
def net() -> pp.pandapowerNet:
    net = pp.networks.case9()
    pp.runpp(net)
    return net


@pytest.fixture
def monitored_elements(net: pp.pandapowerNet) -> pd.DataFrame:
    line_ids = [get_globally_unique_id(int(idx), "line") for idx in net.line.index]
    definition = Nminus1Definition(
        contingencies=[
            Contingency(id=uid, name=uid, elements=[GridElement(id=uid, type="line", name=uid, kind="branch")])
            for uid in line_ids
        ],
        monitored_elements=[MonitoredElement(id=uid, type="line", name=uid, kind="branch") for uid in line_ids],
        id_type="unique_pandapower",
    )
    return translate_nminus1_for_pandapower(definition, net).monitored_elements


def _mapping_pd(net: pp.pandapowerNet) -> pd.DataFrame:
    mapping = get_empty_dataframe_from_model(SwitchElementMappingSchema)
    return pd.concat(
        [
            mapping,
            pd.DataFrame(
                {
                    "switch_id": [0, 1],
                    "element": [get_globally_unique_id(int(net.line.index[0]), "line")] * 2,
                    "side": [0.0, 1.0],
                }
            ).astype(mapping.dtypes.to_dict()),
        ]
    )


def test_from_network_accepts_pandas_and_polars_mapping(net: pp.pandapowerNet, monitored_elements: pd.DataFrame) -> None:
    mapping_pd = _mapping_pd(net)
    mapping_pl = pl.from_pandas(mapping_pd)

    from_pd = ResultConstants.from_network(net, net, monitored_elements, mapping_pd)
    from_pl = ResultConstants.from_network(net, net, monitored_elements, mapping_pl)

    assert_frame_equal(from_pd.switch_element_mapping_pl, mapping_pl)
    # A polars mapping is taken as-is, not copied.
    assert from_pl.switch_element_mapping_pl is mapping_pl


def test_from_network_shares_a_given_graph_cache(net: pp.pandapowerNet, monitored_elements: pd.DataFrame) -> None:
    mapping = _mapping_pd(net)
    cache = ConnectivityGraphCache()
    graph, _ = cache.get(net)

    constants = ResultConstants.from_network(net, net, monitored_elements, mapping, graph_cache=cache)
    assert constants.graph_cache is cache
    assert constants.graph_cache.get(net)[0] is graph

    fresh = ResultConstants.from_network(net, net, monitored_elements, mapping)
    assert fresh.graph_cache is not cache
