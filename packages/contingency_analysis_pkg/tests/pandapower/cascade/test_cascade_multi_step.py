# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import unittest

import numpy as np
import pandapower as pp
import pandas as pd
from toop_engine_contingency_analysis.pandapower import run_contingency_analysis_pandapower
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeReasonType
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    CascadeConfig,
    ContingencyAnalysisConfig,
    ParallelConfig,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def build_cascade_test_net():
    net = pp.create_empty_network(sn_mva=100.0)

    for tbl in ("bus", "line", "load", "gen", "sgen", "switch"):
        if "origin_id" not in net[tbl].columns:
            net[tbl]["origin_id"] = None

    # ---- Buses (20 kV) ----
    b0 = pp.create_bus(
        net, vn_kv=20.0, name="GridBus", origin_id="bus:b0", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b01 = pp.create_bus(
        net, vn_kv=20.0, name="B01", origin_id="bus:b01", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b02 = pp.create_bus(
        net, vn_kv=20.0, name="B02", origin_id="bus:b02", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b1 = pp.create_bus(
        net, vn_kv=20.0, name="B1", origin_id="bus:b1", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b2 = pp.create_bus(
        net, vn_kv=20.0, name="B2", origin_id="bus:b2", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b21 = pp.create_bus(
        net, vn_kv=20.0, name="B21", origin_id="bus:b21", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b22 = pp.create_bus(
        net, vn_kv=20.0, name="B22", origin_id="bus:b22", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b3 = pp.create_bus(
        net, vn_kv=20.0, name="B3", origin_id="bus:b3", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b4 = pp.create_bus(
        net, vn_kv=20.0, name="B4", origin_id="bus:b4", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b41 = pp.create_bus(
        net, vn_kv=20.0, name="B41", origin_id="bus:b41", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b42 = pp.create_bus(
        net, vn_kv=20.0, name="B42", origin_id="bus:b42", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )
    b5 = pp.create_bus(
        net, vn_kv=20.0, name="B5 (far)", origin_id="bus:b5", GeographicalRegion_id="123", GeographicalRegion_name="50Hertz"
    )

    # ---- Slack (as GEN) at b5 ----
    g_slack = pp.create_gen(
        net,
        bus=b5,
        p_mw=0.0,
        vm_pu=1.02,
        name="SlackGen@B5",
        origin_id="gen:slack_b5",
    )
    if "slack" not in net.gen.columns:
        net.gen["slack"] = False
    net.gen.at[g_slack, "slack"] = True

    pp.create_gen(
        net,
        bus=b4,
        p_mw=0.0,
        vm_pu=1.02,
        name="Gen@B4",
        origin_id="gen:b4",
    )

    # ---- Generator at b0 (high production) ----
    pp.create_sgen(
        net,
        bus=b0,
        p_mw=24.0,
        q_mvar=2.0,
        name="SGen@B0",
        origin_id="sgen:sgen_b0",
    )

    # ---- Lines ----
    pp.create_line(net, from_bus=b01, to_bus=b02, length_km=0.3, std_type="NAYY 4x50 SE", name="l1", origin_id="line:l1")
    pp.create_line(net, from_bus=b01, to_bus=b02, length_km=0.3, std_type="NAYY 4x50 SE", name="l2", origin_id="line:l2")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x150 SE", name="l3", origin_id="line:l3")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1.2, std_type="NAYY 4x150 SE", name="l5", origin_id="line:l5")
    pp.create_line(net, from_bus=b3, to_bus=b4, length_km=1.0, std_type="NAYY 4x150 SE", name="l6", origin_id="line:l6")
    pp.create_line(net, from_bus=b41, to_bus=b42, length_km=1.5, std_type="NAYY 4x150 SE", name="l7", origin_id="line:l7")
    pp.create_line(net, from_bus=b21, to_bus=b22, length_km=2.5, std_type="NAYY 4x150 SE", name="l4", origin_id="line:l4")

    # ---- Loads ----
    pp.create_load(net, b3, p_mw=28.0, q_mvar=2.5, name="Load_B3", origin_id="load:b3")

    # ---- Switches (bus-bus breakers) ----
    pp.create_switch(net, bus=b0, element=b01, et="b", closed=True, type="CB", name="SW_L01_grid", origin_id="sw:b0_b01")
    pp.create_switch(net, bus=b02, element=b1, et="b", closed=True, type="CB", name="SW_L01_grid", origin_id="sw:b02_b1")
    pp.create_switch(net, bus=b2, element=b21, et="b", closed=True, type="CB", name="SW_L24_B2", origin_id="sw:b2_b21")
    pp.create_switch(net, bus=b22, element=b4, et="b", closed=True, type="CB", name="SW_L24_B4", origin_id="sw:b22_b4")
    pp.create_switch(net, bus=b4, element=b41, et="b", closed=True, type="CB", name="SW_B4_B41", origin_id="sw:b4_b41")
    pp.create_switch(net, bus=b42, element=b5, et="b", closed=True, type="CB", name="SW_B42_B5", origin_id="sw:b42_b5")

    # ---- Switch characteristics ----
    net["sw_characteristics"] = pd.DataFrame(
        index=net.switch.index,
        data={
            "breaker_uuid": list(net.switch.origin_id),
            "r_i": [1.0, 1.0, 14.0, 35.0, 1.0, 1.0],
            "r_v": [1.0, 1.0, 14.0, 35.0, 1.0, 1.0],
            "x_v": [1.0, 1.0, 14.0, 35.0, 1.0, 1.0],
            "angle": [30.0, 30.0, 30.0, 35.0, 30.0, 30.0],
            "relay_side": ["element", "element", "element", "bus", "element", "element"],
            "custom_warning_distance_protection": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
    )

    net.bus["Busbar_id"] = ""
    return net


def _cascade_results_to_events(cascade_results: pd.DataFrame) -> list[dict]:
    """Convert cascade_results DataFrame rows to event dicts matching the original test format."""
    events = []
    for (_, contingency_id, cascade_number, element_mrid), row in cascade_results.iterrows():
        severity = row["distance_protection_severity"]
        events.append(
            {
                "cascade_number": cascade_number,
                "cascade_reason": row["cascade_reason"],
                "contingency_mrid": None if contingency_id == "BASECASE" else contingency_id,
                "contingency_name": row["contingency_name"],
                "distance_protection_severity": None if pd.isna(severity) else severity,
                "element_mrid": element_mrid,
                "element_name": row["element_name"],
            }
        )
    return events


class TestCascades(unittest.TestCase):
    """Unit test class for validating contingency cascade detection in an electrical grid simulation."""

    def test_cascade_with_2_trips(self):
        net = build_cascade_test_net()
        net.line.loc[0, "max_i_ka"] = 0.3
        net.line.loc[1, "max_i_ka"] = 0.3
        net.line.loc[2, "max_i_ka"] = 0.6
        net.line.loc[3, "max_i_ka"] = 0.4
        net.line.loc[4, "max_i_ka"] = 0.6

        cascade_cfg = CascadeConfig(
            depth_limit=3,
            current_loading_threshold=1.5,
            min_island_size=2,
            cascade_log_elements=["line", "switch"],
            basecase_distance_protection_factor=2,
            contingency_distance_protection_factor=2,
        )
        net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
        net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
        net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

        monitored_elements = (
            [GridElement(id=row.global_id, type="line", kind="branch", name=row.name) for row in net.line.itertuples()]
            + [GridElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
            + [GridElement(id=row.global_id, type="switch", kind="switch", name=row.name) for row in net.switch.itertuples()]
        )
        # Use origin_id as contingency id so cascade_results contingency index == origin_id
        contingencies = [
            Contingency(
                id="BASECASE",
                name="BASECASE",
                elements=[],
            ),
            Contingency(
                id="line:l1",
                name="l1",
                elements=[GridElement(id="0%%line", type="line", kind="branch")],
            ),
        ]
        nminus1_def = Nminus1Definition(
            monitored_elements=monitored_elements,
            contingencies=contingencies,
        )
        cfg = ContingencyAnalysisConfig(
            method="ac",
            min_island_size=2,
            cascade=cascade_cfg,
            parallel=ParallelConfig(
                n_processes=1,
                batch_size=None,
            ),
            runpp_kwargs={
                "lightsim2grid": False,
                "enforce_q_lims": True,
            },
        )

        lf_results = run_contingency_analysis_pandapower(
            net=net,
            n_minus_1_definition=nminus1_def,
            job_id="test",
            timestep=0,
            cfg=cfg,
        )

        all_cascade_events = _cascade_results_to_events(lf_results.cascade_results)
        # Filter to the l1 contingency only — BASECASE may also produce cascade events
        cascade_events = [e for e in all_cascade_events if e["contingency_name"] == "l1"]

        expected = [
            {
                "cascade_number": 1,
                "cascade_reason": CascadeReasonType.CASCADE_REASON_CURRENT,
                "contingency_mrid": "line:l1",
                "contingency_name": "l1",
                "distance_protection_severity": None,
                "element_mrid": "line:l2",
                "element_name": "l2",
            },
            {
                "cascade_number": 2,
                "cascade_reason": CascadeReasonType.CASCADE_REASON_DISTANCE,
                "contingency_mrid": "line:l1",
                "contingency_name": "l1",
                "distance_protection_severity": "WARNING",
                "element_mrid": "line:l4",
                "element_name": "l4",
            },
            {
                "cascade_number": 2,
                "cascade_reason": CascadeReasonType.CASCADE_REASON_CURRENT,
                "contingency_mrid": "line:l1",
                "contingency_name": "l1",
                "distance_protection_severity": None,
                "element_mrid": "line:l7",
                "element_name": "l7",
            },
        ]
        assert cascade_events == expected
