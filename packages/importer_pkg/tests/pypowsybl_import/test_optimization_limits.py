# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import polars as pl
import pypowsybl
import pytest
from pypowsybl.network.impl.network import Network
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import.loadflow_based_current_limits import create_optimization_limits
from toop_engine_interfaces.messages.preprocess.preprocess_commands import DoubleLimitsSetpoint
from toop_engine_interfaces.nminus1_definition import Contingency, MonitoredElement, Nminus1Definition


def test_create_optimization_limits(complex_grid_network: Network) -> None:
    network = complex_grid_network
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    limits = network.get_operational_limits().reset_index()
    limits = limits[limits.element_type == "LINE"]
    permanent = limits[limits.name == "permanent_limit"].groupby(["element_id", "side"]).value.max()
    n_1 = limits[limits.name == "N-1"].groupby(["element_id", "side"]).value.max()
    n_1 = n_1.reindex(permanent.index).fillna(permanent)
    line_ids = permanent.unstack()["ONE"].dropna().index[:3]
    optimized_id, non_worsening_id, both_id = line_ids
    flow_fraction = {optimized_id: 0.5, non_worsening_id: 0.3, both_id: 0.5}
    n_1_factor = 1.2

    # The lines of the complex grid only have limits on side one
    rows = [(line_id, "ONE") for line_id in line_ids]
    worst_case = pl.DataFrame(
        {
            "element": [line_id for line_id, _ in rows],
            "side": [1 for _ in rows],
            "n0": [flow_fraction[line_id] * permanent[line_id, side] for line_id, side in rows],
            "n1": [n_1_factor * flow_fraction[line_id] * n_1[line_id, side] for line_id, side in rows],
        }
    )
    nminus1_definition = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=optimized_id, kind="branch", type="LINE", optimized=True, non_worsening=False),
            MonitoredElement(id=non_worsening_id, kind="branch", type="LINE", optimized=False, non_worsening=True),
            MonitoredElement(id=both_id, kind="branch", type="LINE", optimized=True, non_worsening=True),
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    create_optimization_limits(network, nminus1_definition, worst_case, DoubleLimitsSetpoint(lower=0.9))

    new_limits = network.get_operational_limits().reset_index()
    for case, physical, factor in (("n0", permanent, 1.0), ("n1", n_1, n_1_factor)):
        case_limits = new_limits[new_limits.name == f"optimization_limit_{case}"].set_index(["element_id", "side"]).value
        assert set(case_limits.index.get_level_values("element_id")) == set(line_ids)
        for (line_id, side), value in case_limits.items():
            # Optimized-only branches below the lower limit get the lower limit, the others their flow
            fraction = 0.9 if line_id == optimized_id else flow_fraction[line_id] * factor
            assert value == pytest.approx(fraction * physical[line_id, side])
