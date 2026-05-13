# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Coordinate iterative cascade simulation after an initial outage."""

from itertools import chain
from typing import Any

import pandapower as pp
import pandas as pd
import pandera.typing as pat
from beartype.typing import Literal
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection import (
    build_cascade_context,
    evaluate_distance_protection_triggers,
    evaluate_overload_triggers,
    prepare_branch_results_for_overload,
    prepare_switch_results_for_protection,
)
from toop_engine_contingency_analysis.pandapower.cascade.models import (
    CascadeContext,
    CascadeEvent,
    CascadeReasonType,
    CascadeSppsBranchSwitchResults,
    CascadeTriggers,
)
from toop_engine_contingency_analysis.pandapower.cascade.outage_groups import (
    compute_current_overload_outage_group,
    compute_switches_outage_group,
    get_outage_group_current_violation_log_info,
    get_outage_group_distance_protection_log_info,
    pandapower_grid_element_from_network_outage,
)
from toop_engine_contingency_analysis.pandapower.cascade.simulation.loadflow import (
    cascade_monitored_breakers_dataframe,
    run_spps_with_branch_switch_results,
)
from toop_engine_contingency_analysis.pandapower.outaged_topology import open_outaged_circuit_breakers
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    get_switch_mapped_elements,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerElements,
    PandapowerMonitoredElementSchema,
    SingleOutageSppsContext,
)
from toop_engine_interfaces.loadflow_results import ConvergenceStatus


class CascadeSimulator:
    """Run cascading outage simulation after an initial contingency.

    The simulator handles both current overload triggers and distance protection
    triggers. It repeatedly converts those triggers into outages, runs another
    load flow, and stops when there are no new triggers or when the configured
    depth limit is reached.
    """

    def __init__(
        self,
        cfg: CascadeConfig,
        spps: SingleOutageSppsContext,
        *,
        method: Literal["ac", "dc"] = "ac",
        runpp_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Create a cascade simulator.

        Parameters
        ----------
        cfg : CascadeConfig
            Cascade configuration, including thresholds and depth limit.
        spps : SingleOutageSppsContext
            Special protection scheme settings used in inner load flows.
        method : Literal["ac", "dc"]
            Load-flow method, either ac or dc.
        runpp_kwargs : dict[str, Any] | None
            Extra arguments forwarded to pandapower load flow.
        """
        self._cfg: CascadeConfig = cfg
        self._cascade_context: CascadeContext | None = None
        self._spps = spps
        self._lf_method = method
        self._runpp_kwargs = runpp_kwargs

    @property
    def _context(self) -> CascadeContext:
        """Return prepared cascade context for the current simulation.

        Returns
        -------
        CascadeContext
            CascadeContext with relay data and busbar coupler ids.

        Raises
        ------
        RuntimeError
            If simulation has not initialized the context yet.
        """
        if self._cascade_context is None:
            raise RuntimeError("Cascade context has not been initialized.")
        return self._cascade_context

    def simulate(
        self,
        net: pp.pandapowerNet,
        branch_results_df: pd.DataFrame,
        switch_results_df: pd.DataFrame,
    ) -> list[CascadeEvent]:
        """Run the cascade loop starting from initial load-flow results.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network after the initial contingency load flow.
        branch_results_df : pd.DataFrame
            Branch result table from the initial load flow.
        switch_results_df : pd.DataFrame
            Switch result table from the initial load flow.

        Returns
        -------
        list[CascadeEvent]
            Ordered list of cascade events. The list is empty when no cascade
            trigger is found.
        """
        self._cascade_context = build_cascade_context(net)
        triggers = self._detect_triggers_from_results(
            net=net,
            branch_results=branch_results_df,
            switch_results=switch_results_df,
        )

        if triggers.empty:
            return []

        events: list[CascadeEvent] = []
        accumulative_outages_pp: list[PandapowerElements] = []
        monitored_breakers = cascade_monitored_breakers_dataframe(
            net,
            self._context.switch_characteristics.breaker_uuid,
        )

        for step in range(self._cfg.depth_limit):
            step_no = step + 1
            if triggers.empty:
                return events

            step_events, outages = self._collect_step_events_and_outages(
                net=net,
                triggers=triggers,
                step_no=step_no,
            )
            events.extend(step_events)

            accumulative_outages_pp.extend(self._to_pandapower_outage_elements(net, outages))

            contingency = PandapowerContingency(
                unique_id="BASECASE",
                name="BASECASE",
                elements=list(accumulative_outages_pp),
            )
            bundle = self._run_cascade_loadflow_step(
                net=net,
                contingency=contingency,
                monitored_breakers=monitored_breakers,
            )

            if bundle is None or bundle.convergence_status != ConvergenceStatus.CONVERGED:
                events.append(self._failed_loadflow_event(step_no + 1))
                return events

            triggers = self._detect_triggers_from_results(
                net=net,
                branch_results=bundle.branch_results,
                switch_results=bundle.switch_results,
            )

        return self._append_depth_limit_events(
            events=events,
            net=net,
            triggers=triggers,
        )

    def _detect_triggers_from_results(
        self,
        net: pp.pandapowerNet,
        branch_results: pd.DataFrame,
        switch_results: pd.DataFrame,
    ) -> CascadeTriggers:
        """Find distance-protection and current-overload triggers in result tables.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network for the current cascade state.
        branch_results : pd.DataFrame
            Branch result table to check for current overloads.
        switch_results : pd.DataFrame
            Switch result table to check for relay trips.

        Returns
        -------
        CascadeTriggers
            CascadeTriggers object containing all triggers found in the tables.
        """
        switch_prepared = prepare_switch_results_for_protection(
            net,
            switch_results,
            cascade_context=self._context,
        )
        branch_for_overload = prepare_branch_results_for_overload(branch_results)

        return CascadeTriggers(
            tripped_switches=evaluate_distance_protection_triggers(
                switch_prepared,
                self._cfg,
            ),
            current_overloaded_elements=evaluate_overload_triggers(
                current_res=branch_for_overload,
                threshold=self._cfg.current_loading_threshold,
            ),
        )

    def _collect_step_events_and_outages(
        self,
        net: pp.pandapowerNet,
        triggers: CascadeTriggers,
        step_no: int,
    ) -> tuple[list[CascadeEvent], list[tuple[int, str]]]:
        """Convert current triggers into log events and outage tuples.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network for the current cascade state.
        triggers : CascadeTriggers
            Switch-trip and current-overload triggers for this step.
        step_no : int
            Cascade step number.

        Returns
        -------
        tuple[list[CascadeEvent], list[tuple[int, str]]]
            Pair containing event log entries and raw outage tuples.
        """
        events: list[CascadeEvent] = []
        switch_events, switch_groups = self._collect_distance_protection_events(
            net=net,
            tripped_switches=triggers.tripped_switches,
            step_no=step_no,
        )
        current_events, current_groups = self._collect_current_overload_events(
            net=net,
            current_overloaded=triggers.current_overloaded_elements,
            step_no=step_no,
        )

        events.extend(switch_events)
        events.extend(current_events)
        outages = list(chain.from_iterable(switch_groups.values()))
        outages.extend(chain.from_iterable(current_groups.values()))
        return events, outages

    def _collect_distance_protection_events(
        self,
        net: pp.pandapowerNet,
        tripped_switches: pd.DataFrame,
        step_no: int,
    ) -> tuple[list[CascadeEvent], dict[int, list[tuple[int, str]]]]:
        """Handle switch trips caused by distance protection.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network to update.
        tripped_switches : pd.DataFrame
            Switch rows selected by distance protection.
        step_no : int
            Cascade step number.

        Returns
        -------
        tuple[list[CascadeEvent], dict[int, list[tuple[int, str]]]]
            Event log entries and outage groups created by those switch trips.
        """
        if tripped_switches.empty:
            return [], {}

        net.switch.loc[tripped_switches.switch_id, "closed"] = False
        outage_groups = compute_switches_outage_group(
            net,
            overloaded_switches_df=tripped_switches,
            bus_couplers_mrids=self._context.bus_couplers_mrids,
        )
        events = get_outage_group_distance_protection_log_info(
            net=net,
            outage_groups=outage_groups,
            cascade_log_elements=self._cfg.cascade_log_elements,
            step_no=step_no,
            tripped_switches_df=tripped_switches,
        )
        return events, outage_groups

    def _collect_current_overload_events(
        self,
        net: pp.pandapowerNet,
        current_overloaded: pd.DataFrame,
        step_no: int,
    ) -> tuple[list[CascadeEvent], dict[Any, list[tuple[int, str]]]]:
        """Handle branch outages caused by current overload.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network to inspect.
        current_overloaded : pd.DataFrame
            Branch rows above the current loading threshold.
        step_no : int
            Cascade step number.

        Returns
        -------
        tuple[list[CascadeEvent], dict[Any, list[tuple[int, str]]]]
            Event log entries and outage groups created by current overloads.
        """
        if current_overloaded.empty:
            return [], {}

        outage_groups = compute_current_overload_outage_group(net, current_overloaded)
        events = get_outage_group_current_violation_log_info(
            net=net,
            outage_groups=outage_groups,
            cascade_log_elements=self._cfg.cascade_log_elements,
            step_no=step_no,
            current_overloaded_df=current_overloaded,
        )
        return events, outage_groups

    def _to_pandapower_outage_elements(
        self,
        net: pp.pandapowerNet,
        outages: list[tuple[int, str]],
    ) -> list[PandapowerElements]:
        """Convert raw outage tuples to project element objects.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network containing the elements.
        outages : list[tuple[int, str]]
            Element id and element type pairs.

        Returns
        -------
        list[PandapowerElements]
            List of PandapowerElements that can be used in contingency objects.
        """
        elements: list[PandapowerElements] = []
        for table_id, table in outages:
            element = pandapower_grid_element_from_network_outage(net, int(table_id), table)
            if element is not None:
                elements.append(element)
        return elements

    def _run_cascade_loadflow_step(
        self,
        net: pp.pandapowerNet,
        contingency: PandapowerContingency,
        monitored_breakers: pat.DataFrame[PandapowerMonitoredElementSchema],
    ) -> CascadeSppsBranchSwitchResults | None:
        """Run one inner cascade load flow after applying accumulated outages.

        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network for the current cascade state.
        contingency : PandapowerContingency
            Accumulated outages to apply.
        monitored_breakers : pat.DataFrame[PandapowerMonitoredElementSchema]
            Breakers monitored for switch result calculation.

        Returns
        -------
        CascadeSppsBranchSwitchResults | None
            Load-flow result bundle, or None when the step raises an exception.
        """
        switch_element_mapping = get_switch_mapped_elements(
            net=net,
            monitored_elements=monitored_breakers,
            side="bus",
        )
        open_outaged_circuit_breakers(net, contingency.elements)
        try:
            return run_spps_with_branch_switch_results(
                net,
                contingency,
                self._spps,
                switch_element_mapping,
                timestep=1,
                basecase_voltage=net.res_bus.vm_pu.copy(),
                method=self._lf_method,
                runpp_kwargs=self._runpp_kwargs,
                min_island_size=self._cfg.min_island_size,
            )
        except Exception:
            return None

    @staticmethod
    def _failed_loadflow_event(cascade_number: int) -> CascadeEvent:
        """Create the event used when an inner load flow fails.

        Parameters
        ----------
        cascade_number : int
            Cascade step number to record.

        Returns
        -------
        CascadeEvent
            CascadeEvent describing the failed load-flow stop condition.
        """
        return CascadeEvent(
            element_mrid=None,
            element_id=None,
            element_name=None,
            cascade_number=cascade_number,
            cascade_reason=CascadeReasonType.FAILED_LF,
        )

    def _append_depth_limit_events(
        self,
        events: list[CascadeEvent],
        net: pp.pandapowerNet,
        triggers: CascadeTriggers,
    ) -> list[CascadeEvent]:
        """Add final events when the cascade reaches its depth limit.

        Parameters
        ----------
        events : list[CascadeEvent]
            Events collected so far.
        net : pp.pandapowerNet
            Pandapower network at the final cascade state.
        triggers : CascadeTriggers
            Last triggers that could not be processed because of the limit.

        Returns
        -------
        list[CascadeEvent]
            The same event list with depth-limit events appended.
        """
        depth_stop_no = self._cfg.depth_limit + 1
        depth_events, _ = self._collect_step_events_and_outages(
            net=net,
            triggers=triggers,
            step_no=depth_stop_no,
        )
        events.extend(depth_events)
        return events
