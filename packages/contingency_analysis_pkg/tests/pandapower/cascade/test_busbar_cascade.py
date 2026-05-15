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


def create_net():
    net = pp.create_empty_network(sn_mva=100.0)

    # --- Buses ---
    b1 = pp.create_bus(
        net,
        vn_kv=110.0,
        name="b1",
        origin_id="b1_origin_id",
        Busbar_id="b1",
        GeographicalRegion_name="50Hertz",
        GeographicalRegion_id="123",
    )

    b2 = pp.create_bus(
        net, vn_kv=110.0, name="b2", origin_id="b2_origin_id", GeographicalRegion_name="50Hertz", GeographicalRegion_id="123"
    )

    b3 = pp.create_bus(
        net,
        vn_kv=110.0,
        name="b3",
        origin_id="b3_origin_id",
        Busbar_id="b3",
        GeographicalRegion_name="50Hertz",
        GeographicalRegion_id="123",
    )

    b4 = pp.create_bus(
        net, vn_kv=110.0, name="b4", origin_id="b4_origin_id", GeographicalRegion_name="50Hertz", GeographicalRegion_id="123"
    )

    b5 = pp.create_bus(
        net, vn_kv=110.0, name="b5", origin_id="b5_origin_id", GeographicalRegion_name="50Hertz", GeographicalRegion_id="123"
    )

    # --- Switches ---
    pp.create_switch(
        net, bus=b1, element=b2, et="b", closed=True, type="CB", name="sw_b1_b2", origin_id="sw_b1_b2_origin_id"
    )

    pp.create_switch(
        net, bus=b2, element=b3, et="b", closed=True, type="DS", name="sw_b2_b3", origin_id="sw_b2_b3_origin_id"
    )

    # --- Lines ---
    line_params = dict(
        length_km=10.0,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.25,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )

    pp.create_line_from_parameters(
        net, from_bus=b3, to_bus=b4, name="L1_b3_b4", origin_id="L1_b3_b4_origin_id", **line_params
    )

    pp.create_line_from_parameters(
        net, from_bus=b3, to_bus=b5, name="L2_b3_b5", origin_id="L2_b3_b5_origin_id", **line_params
    )

    # --- Slack generator ---
    pp.create_gen(net, bus=b1, p_mw=0.0, vm_pu=1.0, slack=True, name="Slack Generator", origin_id="slack_gen_origin_id")

    # --- Load ---
    pp.create_load(net, bus=b1, p_mw=20.0, q_mvar=5.0, name="Load at b1", origin_id="load_at_b1_origin_id")

    # --- Static generators ---
    pp.create_sgen(net, bus=b5, p_mw=8.0, q_mvar=0.0, name="sgen1", origin_id="sgen1_origin_id")

    pp.create_sgen(net, bus=b4, p_mw=8.0, q_mvar=0.0, name="sgen2", origin_id="sgen2_origin_id")

    # --- Switch characteristics ---
    net["sw_characteristics"] = pd.DataFrame(
        index=net.switch.index,
        data={
            "breaker_uuid": list(net.switch.origin_id),
            "r_i": [30.0, 14.0],
            "r_v": [30.0, 14.0],
            "x_v": [30.0, 14.0],
            "angle": [30.0, 30.0],
            "relay_side": ["element", "element"],
            "custom_warning_distance_protection": [np.nan, np.nan],
        },
    )

    return net


def _cascade_results_to_events(cascade_results: pd.DataFrame) -> list[dict]:
    """Convert cascade_results DataFrame rows to event dicts matching the original test format."""
    events = []
    for (_, contingency_id, cascade_number, element_mrid), row in cascade_results.iterrows():
        events.append(
            {
                "cascade_number": cascade_number,
                "cascade_reason": row["cascade_reason"],
                "contingency_mrid": None if contingency_id == "BASECASE" else contingency_id,
                "contingency_name": row["contingency_name"],
                "distance_protection_severity": row["distance_protection_severity"],
                "element_mrid": element_mrid,
                "element_name": row["element_name"],
            }
        )
    return events


def _build_nminus1_definition(net: pp.pandapowerNet) -> Nminus1Definition:
    monitored_elements = (
        [GridElement(id=row.global_id, type="line", kind="branch", name=row.name) for row in net.line.itertuples()]
        + [GridElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
        + [GridElement(id=row.global_id, type="switch", kind="switch", name=row.name) for row in net.switch.itertuples()]
    )
    # Use origin_id as contingency id so that cascade_results contingency index == origin_id
    contingencies = [Contingency(id="BASECASE", name="BASECASE", elements=[])]
    contingencies += [
        Contingency(
            id=row.origin_id,
            name=row.name,
            elements=[GridElement(id=row.global_id, type="line", kind="branch")],
        )
        for row in net.line.itertuples()
    ]

    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
    )


class TestCascadesBB(unittest.TestCase):
    """Test class for validating cascading failures in an electrical grid simulation."""

    def test_cascade_with_busbar_trip(self):
        net = create_net()
        # Make current larger to get in warning area
        ## 1. Increase the load at b3
        net.load.loc[net.load["origin_id"].eq("load_at_b1"), "p_mw"] = 2000.0
        net.load.loc[net.load["origin_id"].eq("load_at_b1"), "q_mvar"] = 280.0

        ## 2. Reduce/disable the local generator (forces more current from grid)
        net.sgen.loc[net.sgen.index, ["p_mw", "q_mvar"]] = [1000.0, 110.0]

        ## 3. Reduce line impedance (more current for same power transfer)
        net.line["r_ohm_per_km"] *= 0.2
        net.line["x_ohm_per_km"] *= 0.2

        ## 4. Push voltage a bit higher at the slack (can increase power/current depending on conditions)
        net.ext_grid.loc[:, "vm_pu"] = 1.05

        cascade_cfg = CascadeConfig(
            depth_limit=3,
            current_loading_threshold=1.5,
            min_island_size=2,
            cascade_log_elements=["line", "switch"],
            basecase_distance_protection_factor=1.5,
            contingency_distance_protection_factor=1.5,
        )
        net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
        net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
        net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

        nminus1_def = _build_nminus1_definition(net)
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

        cascade_events = _cascade_results_to_events(lf_results.cascade_results)

        assert len(cascade_events) == 11
        distance_prot_events = [
            i for i in cascade_events if i["cascade_reason"] == CascadeReasonType.CASCADE_REASON_DISTANCE
        ]
        assert isinstance(distance_prot_events, list)
        assert len(distance_prot_events) == 7

        expected_keys = {
            "cascade_number",
            "cascade_reason",
            "contingency_mrid",
            "contingency_name",
            "distance_protection_severity",
            "element_mrid",
            "element_name",
        }

        # basic schema + invariants
        for e in distance_prot_events:
            assert set(e.keys()) == expected_keys
            assert e["cascade_number"] == 1
            assert e["cascade_reason"] == CascadeReasonType.CASCADE_REASON_DISTANCE
            assert e["distance_protection_severity"] == "DANGER"
            assert isinstance(e["contingency_name"], str)
            assert isinstance(e["element_mrid"], str)
            assert isinstance(e["element_name"], str)

        # exact relationships check (contingency_name, contingency_mrid, element_name, element_mrid)
        actual_pairs = {
            (
                e["contingency_name"],
                None if pd.isna(e["contingency_mrid"]) else e["contingency_mrid"],
                e["element_name"],
                e["element_mrid"],
            )
            for e in distance_prot_events
        }

        expected_pairs = {
            ("L2_b3_b5", "L2_b3_b5_origin_id", "L1_b3_b4", "L1_b3_b4_origin_id"),
            ("L2_b3_b5", "L2_b3_b5_origin_id", "sw_b1_b2", "sw_b1_b2_origin_id"),
            ("L1_b3_b4", "L1_b3_b4_origin_id", "L2_b3_b5", "L2_b3_b5_origin_id"),
            ("L1_b3_b4", "L1_b3_b4_origin_id", "sw_b1_b2", "sw_b1_b2_origin_id"),
            ("BASECASE", None, "L1_b3_b4", "L1_b3_b4_origin_id"),
            ("BASECASE", None, "L2_b3_b5", "L2_b3_b5_origin_id"),
            ("BASECASE", None, "sw_b1_b2", "sw_b1_b2_origin_id"),
        }

        assert actual_pairs == expected_pairs

        current_violations = [i for i in cascade_events if i["cascade_reason"] == CascadeReasonType.CASCADE_REASON_CURRENT]
        assert len(current_violations) == 4
        for st_ev in current_violations:
            if st_ev["contingency_name"] == "L2_b3_b5":
                assert st_ev["element_mrid"] == "L1_b3_b4_origin_id"
                assert st_ev["element_name"] == "L1_b3_b4"
            elif st_ev["contingency_name"] == "L1_b3_b4":
                assert st_ev["element_mrid"] == "L2_b3_b5_origin_id"
                assert st_ev["element_name"] == "L2_b3_b5"
            elif st_ev["contingency_name"] == "BASECASE":
                assert st_ev["element_mrid"] in {"L1_b3_b4_origin_id", "L2_b3_b5_origin_id"}
                assert st_ev["element_name"] in {"L1_b3_b4", "L2_b3_b5"}

    def test_cascade_duplicated_result_for_current_and_distance(self):
        net = create_net()
        net.bus["Busbar_id"] = ""
        net.sw_characteristics = net.sw_characteristics[net.sw_characteristics.breaker_uuid == "sw_b1_b2_origin_id"]
        # Make current larger to get in warning area
        ## 1. Increase the load at b3
        net.load.loc[net.load["origin_id"].eq("load_at_b1"), "p_mw"] = 2000.0
        net.load.loc[net.load["origin_id"].eq("load_at_b1"), "q_mvar"] = 280.0

        ## 2. Reduce/disable the local generator (forces more current from grid)
        net.sgen.loc[net.sgen.index, ["p_mw", "q_mvar"]] = [1000.0, 110.0]

        ## 3. Reduce line impedance (more current for same power transfer)
        net.line["r_ohm_per_km"] *= 0.2
        net.line["x_ohm_per_km"] *= 0.2

        ## 4. Push voltage a bit higher at the slack (can increase power/current depending on conditions)
        net.ext_grid.loc[:, "vm_pu"] = 1.05

        cascade_cfg = CascadeConfig(
            depth_limit=3,
            current_loading_threshold=1.5,
            min_island_size=2,
            cascade_log_elements=["line", "switch"],
            basecase_distance_protection_factor=1.5,
            contingency_distance_protection_factor=1.5,
        )
        net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
        net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
        net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

        nminus1_def = _build_nminus1_definition(net)
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

        cascade_events = _cascade_results_to_events(lf_results.cascade_results)

        assert len(cascade_events) == 8
        distance_prot_events = [
            i for i in cascade_events if i["cascade_reason"] == CascadeReasonType.CASCADE_REASON_DISTANCE
        ]
        for st_ev in distance_prot_events:
            if st_ev["contingency_name"] == "L2_b3_b5":
                assert st_ev["element_mrid"] == "L1_b3_b4_origin_id"
                assert st_ev["element_name"] == "L1_b3_b4"
            elif st_ev["contingency_name"] == "L1_b3_b4":
                assert st_ev["element_mrid"] == "L2_b3_b5_origin_id"
                assert st_ev["element_name"] == "L2_b3_b5"

        current_violations = [i for i in cascade_events if i["cascade_reason"] == CascadeReasonType.CASCADE_REASON_CURRENT]
        for st_ev in current_violations:
            if st_ev["contingency_name"] == "L2_b3_b5":
                assert st_ev["element_mrid"] == "L1_b3_b4_origin_id"
                assert st_ev["element_name"] == "L1_b3_b4"
            elif st_ev["contingency_name"] == "L1_b3_b4":
                assert st_ev["element_mrid"] == "L2_b3_b5_origin_id"
                assert st_ev["element_name"] == "L2_b3_b5"
            elif st_ev["contingency_name"] == "BASECASE":
                assert st_ev["element_mrid"] in {"L1_b3_b4_origin_id", "L2_b3_b5_origin_id"}
                assert st_ev["element_name"] in {"L1_b3_b4", "L2_b3_b5"}
