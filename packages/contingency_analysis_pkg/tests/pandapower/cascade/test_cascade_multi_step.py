# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import unittest
from copy import deepcopy
from unittest import mock

import numpy as np
import pandapower as pp
import pandas as pd
import pandera as pa
import polars as pl
import pytest
from toop_engine_contingency_analysis.pandapower import run_contingency_analysis_pandapower
from toop_engine_contingency_analysis.pandapower.cascade.detection import prepare_cascade_run_constants
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeReasonType, CascadeTriggers
from toop_engine_contingency_analysis.pandapower.cascade.simulation.simulator import CascadeSimulator
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    CascadeConfig,
    ContingencyAnalysisConfig,
    DistanceProtectionConfig,
    DistanceProtectionFactors,
    OverloadConfig,
    PandapowerContingency,
    ParallelConfig,
    SingleOutageSppsContext,
    SppsActionsPandapowerSchema,
    SppsConditionsPandapowerSchema,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import BranchResultSchema, SwitchResultsSchema
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
    SwitchMonitoringScope,
)


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
            "protection_side": ["element", "element", "element", "bus", "element", "element"],
            "protection_element": ["line", "line", "line", "line", "line", "line"],
            "custom_base_alarm": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "custom_base_warning": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "custom_contingency_alarm": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "custom_contingency_warning": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
    )

    net.bus["Busbar_id"] = ""
    return net


def build_line_and_transformer_net():
    """Radial net whose line (~171 %) and transformer (~166 %) are both overloaded.

    HV slack --trafo--> MV --switch--> MV2 --line--> load. Ratings are picked so that
    both branches sit between 150 % and 180 %, which lets a per-element-type threshold
    pick exactly one of them.
    """
    net = pp.create_empty_network(sn_mva=100.0)
    for tbl in ("bus", "line", "trafo", "load", "gen", "switch"):
        if "origin_id" not in net[tbl].columns:
            net[tbl]["origin_id"] = None

    hv = pp.create_bus(net, vn_kv=110.0, name="HV", origin_id="bus:hv")
    mv = pp.create_bus(net, vn_kv=20.0, name="MV", origin_id="bus:mv")
    mv2 = pp.create_bus(net, vn_kv=20.0, name="MV2", origin_id="bus:mv2")
    load_bus = pp.create_bus(net, vn_kv=20.0, name="LOAD", origin_id="bus:load")

    slack = pp.create_gen(net, bus=hv, p_mw=0.0, vm_pu=1.02, name="Slack", origin_id="gen:slack")
    if "slack" not in net.gen.columns:
        net.gen["slack"] = False
    net.gen.at[slack, "slack"] = True

    pp.create_transformer_from_parameters(
        net,
        hv_bus=hv,
        lv_bus=mv,
        sn_mva=12.5,
        vn_hv_kv=110.0,
        vn_lv_kv=20.0,
        vkr_percent=0.4,
        vk_percent=12.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        name="t1",
        origin_id="trafo:t1",
    )
    pp.create_switch(net, bus=mv, element=mv2, et="b", closed=True, type="CB", name="sw1", origin_id="sw:mv_mv2")
    pp.create_line_from_parameters(
        net,
        from_bus=mv2,
        to_bus=load_bus,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=0.35,
        name="l1",
        origin_id="line:l1",
    )
    pp.create_load(net, load_bus, p_mw=20.0, q_mvar=2.0, name="Load", origin_id="load:load")

    # A relay with a near-zero protection zone: distance protection must never fire here,
    # so every cascade event in these tests comes from the current-overload check.
    net["sw_characteristics"] = pd.DataFrame(
        index=net.switch.index,
        data={
            "breaker_uuid": list(net.switch.origin_id),
            "r_i": [1e-6],
            "r_v": [1e-6],
            "x_v": [1e-6],
            "angle": [30.0],
            "relay_side": ["element"],
            "protection_side": ["element"],
            "protection_element": ["line"],
            "custom_base_alarm": [np.nan],
            "custom_base_warning": [np.nan],
            "custom_contingency_alarm": [np.nan],
            "custom_contingency_warning": [np.nan],
        },
    )
    net.bus["Busbar_id"] = ""
    return net


def _run_basecase_cascade(net, cascade_cfg: CascadeConfig) -> pd.DataFrame:
    """Run a base-case-only contingency analysis and return its cascade results."""
    for table in ("line", "trafo", "bus", "switch"):
        net[table]["global_id"] = net[table].index.map(lambda idx, tbl=table: get_globally_unique_id(idx, tbl))

    monitored_elements = (
        [MonitoredElement(id=row.global_id, type="line", kind="branch", name=row.name) for row in net.line.itertuples()]
        + [MonitoredElement(id=row.global_id, type="trafo", kind="branch", name=row.name) for row in net.trafo.itertuples()]
        + [MonitoredElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
        + [
            MonitoredElement(id=row.global_id, type="switch", kind="switch", name=row.name)
            for row in net.switch.itertuples()
        ]
    )
    nminus1_def = Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=[Contingency(id="BASECASE", name="BASECASE", elements=[])],
    )
    cfg = ContingencyAnalysisConfig(
        method="ac",
        min_island_size=1,
        cascade=cascade_cfg,
        parallel=ParallelConfig(n_processes=1, batch_size=None),
        runpp_kwargs={"lightsim2grid": False, "enforce_q_lims": True},
    )
    lf_results = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test",
        timestep=0,
        cfg=cfg,
    )
    return lf_results.cascade_results


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
            overload=OverloadConfig(current_loading_threshold=1.5),
            min_island_size=2,
            cascade_log_elements=["line", "switch"],
            distance_protection=DistanceProtectionConfig(
                alarm=DistanceProtectionFactors(
                    basecase_line=1.0,
                    basecase_transformer=1.0,
                    basecase_bus_coupler=1.0,
                    contingency_line=1.0,
                    contingency_transformer=1.0,
                    contingency_bus_coupler=1.0,
                ),
                warning=DistanceProtectionFactors(
                    basecase_line=2,
                    basecase_transformer=2,
                    basecase_bus_coupler=2,
                    contingency_line=2,
                    contingency_transformer=2,
                    contingency_bus_coupler=2,
                ),
            ),
            # This fixture deliberately starts from an overloaded base case to force a cascade,
            # which is exactly what the base-case screen short-circuits.
            stop_cascade_on_basecase_violation=False,
        )
        net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
        net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
        net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

        monitored_elements = (
            [MonitoredElement(id=row.global_id, type="line", kind="branch", name=row.name) for row in net.line.itertuples()]
            + [MonitoredElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
            + [
                MonitoredElement(id=row.global_id, type="switch", kind="switch", name=row.name)
                for row in net.switch.itertuples()
            ]
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

    def test_cascade_stops_when_step2_elements_not_monitored(self):
        """Cascade event log stops after step 1 when step-2 triggers are on non-monitored elements.

        Setup
        -----
        Same network and line limits as ``test_cascade_with_2_trips``.  The
        contingency (l1 trip) would normally produce a 2-step cascade:

          step 1 — l2 trips (current overload)
          step 2 — l4 trips (distance protection) + l7 trips (current overload)

        Here l4 and l7 are **excluded** from ``monitored_elements``.

        Why no step-2 events appear
        ---------------------------
        Current overload (l7):
            Branch results fed to ``_detect_triggers_from_results`` are
            pre-filtered to monitored elements.  l7 is absent, so the overload
            detector never sees it as triggered.

        Distance protection (l4 via sw3):
            Switch results are not filtered — sw3 still fires and its outage
            group (which includes l4) is fully applied to the network.
            However, l4 is not monitored so the resulting event is suppressed
            by ``_filter_events_to_monitored`` and does not appear in the log.

        Outages are intentionally not filtered: when a monitored switch trips,
        all elements in its outage group must be taken out of service to keep
        the topology consistent, even if those elements are not monitored.
        """
        net = build_cascade_test_net()
        net.line.loc[0, "max_i_ka"] = 0.3
        net.line.loc[1, "max_i_ka"] = 0.3
        net.line.loc[2, "max_i_ka"] = 0.6
        net.line.loc[3, "max_i_ka"] = 0.4
        net.line.loc[4, "max_i_ka"] = 0.6

        cascade_cfg = CascadeConfig(
            depth_limit=3,
            overload=OverloadConfig(current_loading_threshold=1.5),
            min_island_size=2,
            cascade_log_elements=["line", "switch"],
            distance_protection=DistanceProtectionConfig(
                alarm=DistanceProtectionFactors(
                    basecase_line=1.0,
                    basecase_transformer=1.0,
                    basecase_bus_coupler=1.0,
                    contingency_line=1.0,
                    contingency_transformer=1.0,
                    contingency_bus_coupler=1.0,
                ),
                warning=DistanceProtectionFactors(
                    basecase_line=2,
                    basecase_transformer=2,
                    basecase_bus_coupler=2,
                    contingency_line=2,
                    contingency_transformer=2,
                    contingency_bus_coupler=2,
                ),
            ),
            # This fixture deliberately starts from an overloaded base case to force a cascade,
            # which is exactly what the base-case screen short-circuits.
            stop_cascade_on_basecase_violation=False,
        )
        net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
        net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
        net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

        # l4 (idx 6, origin_id="line:l4") and l7 (idx 5, origin_id="line:l7")
        # are intentionally excluded — they are the step-2 cascade triggers.
        unmonitored_line_names = {"l4", "l7"}
        monitored_elements = (
            [
                MonitoredElement(id=row.global_id, type="line", kind="branch", name=row.name)
                for row in net.line.itertuples()
                if row.name not in unmonitored_line_names
            ]
            + [MonitoredElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
            + [
                MonitoredElement(id=row.global_id, type="switch", kind="switch", name=row.name)
                for row in net.switch.itertuples()
            ]
        )

        contingencies = [
            Contingency(id="BASECASE", name="BASECASE", elements=[]),
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
            parallel=ParallelConfig(n_processes=1, batch_size=None),
            runpp_kwargs={"lightsim2grid": False, "enforce_q_lims": True},
        )

        lf_results = run_contingency_analysis_pandapower(
            net=net,
            n_minus_1_definition=nminus1_def,
            job_id="test",
            timestep=0,
            cfg=cfg,
        )

        all_cascade_events = _cascade_results_to_events(lf_results.cascade_results)
        cascade_events = [e for e in all_cascade_events if e["contingency_name"] == "l1"]

        # Only the step-1 event (l2 current overload) must appear.
        # l4 and l7 are not monitored, so their step-2 events are suppressed.
        assert len(cascade_events) == 1
        assert cascade_events[0] == {
            "cascade_number": 1,
            "cascade_reason": CascadeReasonType.CASCADE_REASON_CURRENT,
            "contingency_mrid": "line:l1",
            "contingency_name": "l1",
            "distance_protection_severity": None,
            "element_mrid": "line:l2",
            "element_name": "l2",
        }


# ---------------------------------------------------------------------------
# CascadeSimulator.simulate — switch_results_df filtered to monitored_breakers
# ---------------------------------------------------------------------------


def test_simulate_switch_results_filtered_to_protection_scope_only() -> None:
    """switch_results_df passed to _detect_triggers_from_results only contains PROTECTION-scoped switches."""
    protection_uid = get_globally_unique_id(0, "switch")
    flow_only_uid = get_globally_unique_id(1, "switch")

    monitored_elements = pd.DataFrame(
        [
            {
                "unique_id": protection_uid,
                "table": "switch",
                "table_id": 0,
                "kind": "switch",
                "name": "",
                "monitoring_scope": frozenset({SwitchMonitoringScope.PROTECTION}),
            },
            {
                "unique_id": flow_only_uid,
                "table": "switch",
                "table_id": 1,
                "kind": "switch",
                "name": "",
                "monitoring_scope": frozenset({SwitchMonitoringScope.FLOW}),
            },
        ]
    ).set_index("unique_id")

    switch_results_df = pd.DataFrame(
        [
            {
                "timestep": 0,
                "contingency": "c1",
                "element": protection_uid,
                "p": 1.0,
                "q": 0.5,
                "vm": 110.0,
                "i": 5.0,
                "s": 1.1,
                "element_name": "",
                "contingency_name": "c1",
                "side": None,
            },
            {
                "timestep": 0,
                "contingency": "c1",
                "element": flow_only_uid,
                "p": 2.0,
                "q": 0.5,
                "vm": 110.0,
                "i": 5.0,
                "s": 2.1,
                "element_name": "",
                "contingency_name": "c1",
                "side": None,
            },
        ]
    ).set_index(["timestep", "contingency", "element"])

    empty_conditions = pa.typing.DataFrame[SppsConditionsPandapowerSchema](
        get_empty_dataframe_from_model(SppsConditionsPandapowerSchema)
    )
    empty_actions = pa.typing.DataFrame[SppsActionsPandapowerSchema](
        get_empty_dataframe_from_model(SppsActionsPandapowerSchema)
    )
    spps = SingleOutageSppsContext(conditions=empty_conditions, actions=empty_actions)
    simulator = CascadeSimulator(
        cfg=CascadeConfig(
            depth_limit=1,
            overload=OverloadConfig(current_loading_threshold=1.0),
            min_island_size=1,
            cascade_log_elements=[],
            distance_protection=DistanceProtectionConfig(
                alarm=DistanceProtectionFactors(
                    basecase_line=1.0,
                    basecase_transformer=1.0,
                    basecase_bus_coupler=1.0,
                    contingency_line=1.0,
                    contingency_transformer=1.0,
                    contingency_bus_coupler=1.0,
                ),
                warning=DistanceProtectionFactors(
                    basecase_line=1.0,
                    basecase_transformer=1.0,
                    basecase_bus_coupler=1.0,
                    contingency_line=1.0,
                    contingency_transformer=1.0,
                    contingency_bus_coupler=1.0,
                ),
            ),
        ),
        spps=spps,
    )

    net = build_cascade_test_net()
    pp.runpp(net, lightsim2grid=False)
    # Prepare the per-run cascade constants (angle/poly and the resolved factors on
    # sw_characteristics), as the production path does before running the simulator.
    prepare_cascade_run_constants(net, simulator._cfg)

    captured: list[pd.DataFrame] = []

    def _fake_detect(net, branch_results, switch_results):
        captured.append(switch_results)
        return CascadeTriggers(
            tripped_switches=get_empty_dataframe_from_model(SwitchResultsSchema),
            current_overloaded_elements=get_empty_dataframe_from_model(BranchResultSchema),
        )

    with mock.patch.object(simulator, "_detect_triggers_from_results", side_effect=_fake_detect):
        simulator.simulate(
            net=net,
            branch_results=pl.from_pandas(get_empty_dataframe_from_model(BranchResultSchema).reset_index()),
            switch_results=pl.from_pandas(switch_results_df.reset_index()),
            initial_contingency=PandapowerContingency(unique_id="c1", name="c1", elements=[]),
            basecase_net=deepcopy(net),
            monitored_elements=monitored_elements,
        )

    assert len(captured) == 1
    received_elements = set(captured[0].index.get_level_values("element"))
    assert protection_uid in received_elements
    assert flow_only_uid not in received_elements


# ---------------------------------------------------------------------------
# Per-element-type and per-case current loading thresholds
# ---------------------------------------------------------------------------


def _current_violation_mrids(cascade_results: pd.DataFrame) -> list[str]:
    """External ids of the elements that tripped on current overload, sorted."""
    overloaded = cascade_results[cascade_results["cascade_reason"] == CascadeReasonType.CASCADE_REASON_CURRENT]
    return sorted(overloaded.index.get_level_values("element_mrid"))


@pytest.mark.parametrize(
    ("line_threshold", "transformer_threshold", "expected_mrids"),
    [
        # Both branches are loaded to ~165 %: the threshold decides which one trips.
        (1.5, 1.8, ["line:l1"]),
        (1.8, 1.5, ["trafo:t1"]),
        (1.5, 1.5, ["line:l1", "trafo:t1"]),
        (1.8, 1.8, []),
    ],
)
def test_cascade_trips_the_element_types_above_their_own_threshold(
    line_threshold: float,
    transformer_threshold: float,
    expected_mrids: list[str],
) -> None:
    net = build_line_and_transformer_net()
    cascade_cfg = CascadeConfig(
        depth_limit=2,
        # Deliberately unreachable: every trip below must come from a type-specific threshold.
        overload=OverloadConfig(
            current_loading_threshold=99.0, basecase_line=line_threshold, basecase_transformer=transformer_threshold
        ),
        min_island_size=1,
        cascade_log_elements=["line", "trafo"],
        distance_protection=DistanceProtectionConfig(
            alarm=DistanceProtectionFactors(
                basecase_line=1.0,
                basecase_transformer=1.0,
                basecase_bus_coupler=1.0,
                contingency_line=1.0,
                contingency_transformer=1.0,
                contingency_bus_coupler=1.0,
            ),
            warning=DistanceProtectionFactors(
                basecase_line=0.01,
                basecase_transformer=0.01,
                basecase_bus_coupler=0.01,
                contingency_line=0.01,
                contingency_transformer=0.01,
                contingency_bus_coupler=0.01,
            ),
        ),
        # The net is deliberately overloaded in the base case to force a cascade, which is
        # exactly what the base-case screen short-circuits.
        stop_cascade_on_basecase_violation=False,
    )

    cascade_results = _run_basecase_cascade(net, cascade_cfg)

    assert _current_violation_mrids(cascade_results) == expected_mrids
    # The relay zone is near zero, so nothing here may be attributed to distance protection.
    reasons = set(cascade_results["cascade_reason"])
    assert CascadeReasonType.CASCADE_REASON_DISTANCE not in reasons


def _run_l1_cascade(cascade_cfg: CascadeConfig) -> list[dict]:
    """Run the multi-step fixture with *cascade_cfg* and return the l1-contingency events."""
    net = build_cascade_test_net()
    net.line.loc[0, "max_i_ka"] = 0.3
    net.line.loc[1, "max_i_ka"] = 0.3
    net.line.loc[2, "max_i_ka"] = 0.6
    net.line.loc[3, "max_i_ka"] = 0.4
    net.line.loc[4, "max_i_ka"] = 0.6

    net.line["global_id"] = net.line.index.map(lambda imp_id: get_globally_unique_id(imp_id, "line"))
    net.bus["global_id"] = net.bus.index.map(lambda imp_id: get_globally_unique_id(imp_id, "bus"))
    net.switch["global_id"] = net.switch.index.map(lambda imp_id: get_globally_unique_id(imp_id, "switch"))

    monitored_elements = (
        [MonitoredElement(id=row.global_id, type="line", kind="branch", name=row.name) for row in net.line.itertuples()]
        + [MonitoredElement(id=row.global_id, type="bus", kind="bus", name=row.name) for row in net.bus.itertuples()]
        + [
            MonitoredElement(id=row.global_id, type="switch", kind="switch", name=row.name)
            for row in net.switch.itertuples()
        ]
    )
    nminus1_def = Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=[
            Contingency(id="BASECASE", name="BASECASE", elements=[]),
            Contingency(id="line:l1", name="l1", elements=[GridElement(id="0%%line", type="line", kind="branch")]),
        ],
    )
    cfg = ContingencyAnalysisConfig(
        method="ac",
        min_island_size=2,
        cascade=cascade_cfg,
        parallel=ParallelConfig(n_processes=1, batch_size=None),
        runpp_kwargs={"lightsim2grid": False, "enforce_q_lims": True},
    )
    lf_results = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test",
        timestep=0,
        cfg=cfg,
    )
    all_events = _cascade_results_to_events(lf_results.cascade_results)
    return [event for event in all_events if event["contingency_name"] == "l1"]


def _l1_cascade_config(**threshold_fields: float) -> CascadeConfig:
    return CascadeConfig(
        depth_limit=3,
        min_island_size=2,
        cascade_log_elements=["line", "switch"],
        distance_protection=DistanceProtectionConfig(
            alarm=DistanceProtectionFactors(
                basecase_line=1.0,
                basecase_transformer=1.0,
                basecase_bus_coupler=1.0,
                contingency_line=1.0,
                contingency_transformer=1.0,
                contingency_bus_coupler=1.0,
            ),
            warning=DistanceProtectionFactors(
                basecase_line=2,
                basecase_transformer=2,
                basecase_bus_coupler=2,
                contingency_line=2,
                contingency_transformer=2,
                contingency_bus_coupler=2,
            ),
        ),
        # The net is deliberately overloaded in the base case to force a cascade, which is
        # exactly what the base-case screen short-circuits.
        stop_cascade_on_basecase_violation=False,
        overload=OverloadConfig(**threshold_fields),
    )


def test_line_threshold_replaces_the_scalar_threshold_for_lines() -> None:
    """A line override reproduces the scalar-threshold cascade, and reaches contingency rows.

    Only the base-case line threshold is set, so the l1 contingency rows exercise the
    contingency -> base-case fallback. The scalar threshold is unreachable, which proves
    the lines are compared against the override rather than against it.
    """
    scalar_events = _run_l1_cascade(_l1_cascade_config(current_loading_threshold=1.5))
    override_events = _run_l1_cascade(_l1_cascade_config(current_loading_threshold=99.0, basecase_line=1.5))

    assert [event["element_mrid"] for event in scalar_events] == ["line:l2", "line:l4", "line:l7"]
    assert override_events == scalar_events


def test_transformer_threshold_does_not_apply_to_lines() -> None:
    events = _run_l1_cascade(_l1_cascade_config(current_loading_threshold=99.0, basecase_transformer=1.5))

    assert events == []


@pytest.mark.parametrize(
    ("basecase_threshold", "contingency_threshold", "expect_events"),
    [
        # The l1 rows are contingency rows, so only the contingency threshold can trip them.
        (99.0, 1.5, True),
        (1.5, 99.0, False),
    ],
)
def test_basecase_and_contingency_thresholds_are_applied_per_case(
    basecase_threshold: float,
    contingency_threshold: float,
    expect_events: bool,
) -> None:
    events = _run_l1_cascade(
        _l1_cascade_config(
            current_loading_threshold=99.0,
            basecase_line=basecase_threshold,
            contingency_line=contingency_threshold,
        )
    )

    assert bool(events) == expect_events
