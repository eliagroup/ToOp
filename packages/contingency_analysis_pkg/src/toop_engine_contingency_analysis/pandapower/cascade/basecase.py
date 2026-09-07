# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run-level base-case screening: decide once whether cascading is worth simulating at all.

This is the entry point the contingency analysis calls before its outage loop. It extracts the
base-case result tables, hands them to :mod:`~.detection.basecase_screen`, and turns whatever
that finds into cascade result rows.

Screening has to happen here rather than inside the outage loop. The base case is nominally
just another contingency, but relying on it being one does not work:

- outage batches are dispatched to ray workers in parallel, each with its own copy of the
  context, so a worker that finds the violation cannot stop the others;
- nothing pins the base case to the front of the contingency list;
- the caller's ``Nminus1Definition`` need not contain a base-case contingency at all.

The base-case load flow already runs before the loop, so its state is available here regardless.
"""

import logging

import pandapower as pp
import pandas as pd
import pandera as pa
import pandera.typing as pat
import pandera.typing.polars as patpl
import polars as pl
from toop_engine_contingency_analysis.pandapower.cascade.configuration import CascadeConfig
from toop_engine_contingency_analysis.pandapower.cascade.detection.basecase_screen import (
    screen_basecase_for_violations,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection.overload import BASECASE_CONTINGENCY_ID
from toop_engine_contingency_analysis.pandapower.cascade.models import CascadeEvent
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.result_constants import (
    RES_TABLES_FOR_POLARS,
    ResultConstants,
    cache_res_tables_as_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.branch_results import (
    get_branch_results_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.node_results import (
    get_node_results_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import get_switch_results
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import CascadeResultSchema, ConvergenceStatus
from toop_engine_interfaces.loadflow_results_polars import CascadeResultSchemaPolars

_logger = logging.getLogger(__name__)


def screen_basecase_for_cascade(
    net: pp.pandapowerNet,
    *,
    cascade_configuration: CascadeConfig | None,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    switch_element_mapping: pd.DataFrame,
    bus_couplers_mrids: set[str],
    timestep: int,
    basecase_status: ConvergenceStatus,
) -> list[CascadeEvent]:
    """Check the solved base case for violations before any contingency is computed.

    A base case that is already overloaded, or whose relay impedance already sits inside a
    distance protection zone, would make practically every contingency report a cascade that
    started from an already-broken state. Detecting that once here lets the caller report the
    base-case condition and skip cascade simulation for the whole run.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network as left by the base-case load flow.
    cascade_configuration : CascadeConfig or None
        Cascade settings, or None when cascade screening is disabled for the run.
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        Elements to monitor, used to filter both branch results and protection breakers.
    switch_element_mapping : pd.DataFrame
        Switch-to-element mapping for switch result extraction.
    bus_couplers_mrids : set[str]
        Base-case busbar-coupler origin ids from ``prepare_cascade_run_constants``.
    timestep : int
        Timestep the run belongs to.
    basecase_status : ConvergenceStatus
        Whether the base-case load flow converged.

    Returns
    -------
    list[CascadeEvent]
        Base-case violation events. Empty when the base case is clean, when screening is
        disabled, or when the base-case load flow did not converge — in every one of those
        cases the caller should simulate cascades as usual.
    """
    if cascade_configuration is None or not cascade_configuration.stop_cascade_on_basecase_violation:
        return []

    if basecase_status != ConvergenceStatus.CONVERGED:
        _logger.warning("cascading: base-case load flow did not converge; skipping base-case cascade screening")
        return []

    basecase_contingency = PandapowerContingency(
        unique_id=BASECASE_CONTINGENCY_ID,
        name=BASECASE_CONTINGENCY_ID,
        elements=[],
    )

    # The result extractors read the polars snapshots of the res_* tables; the base-case load
    # flow leaves only the pandas tables behind. They are dropped again afterwards: every outage
    # deep-copies the base-case net (and caches its own snapshots after its own load flow), so
    # leaving them attached would pay for a deep copy of every res_* table once per outage.
    added_polars_tables = [f"{table}_polars" for table in RES_TABLES_FOR_POLARS if f"{table}_polars" not in net]
    cache_res_tables_as_polars(net)
    try:
        result_constants = ResultConstants.from_network(
            net,
            net,
            monitored_elements=monitored_elements,
            switch_element_mapping=switch_element_mapping,
        )

        # Only branch and switch results feed cascade detection, so va-diff extraction is skipped.
        full_branch_results = get_branch_results_polars(net, basecase_contingency, timestep, result_constants)
        node_results = get_node_results_polars(net, basecase_contingency, timestep, result_constants)
        switch_results = get_switch_results(
            net,
            basecase_contingency,
            timestep,
            full_branch_results,
            node_results,
            result_constants.switch_element_mapping_pl,
        )
        branch_results = full_branch_results.filter(
            pl.col("element").is_in(result_constants.monitored_element_ids),
        )

        return screen_basecase_for_violations(
            net=net,
            branch_results=branch_results,
            switch_results=switch_results,
            cascade_configuration=cascade_configuration,
            monitored_elements=monitored_elements,
            all_cb_couplers=bus_couplers_mrids,
        )
    finally:
        for table in added_polars_tables:
            net.pop(table, None)


def basecase_violation_warning(basecase_events: list[CascadeEvent]) -> str:
    """Describe a base-case violation for ``LoadflowResults.warnings``."""
    return (
        f"Base case already violates ({len(basecase_events)} element(s)); "
        "cascade simulation was skipped for every contingency. "
        f"See cascade_results rows with contingency '{BASECASE_CONTINGENCY_ID}'."
    )


@pa.check_types
def build_basecase_cascade_results(
    events: list[CascadeEvent],
    timestep: int,
) -> patpl.DataFrame[CascadeResultSchemaPolars]:
    """Turn base-case violation events into the run's whole cascade result table.

    Base-case violations belong to the base case itself rather than to any contingency, so the
    rows are emitted once under the ``BASECASE`` contingency id instead of being copied across
    every contingency the way simulated cascade events are.

    This *replaces* the per-outage cascade results rather than adding to them. Finding a
    base-case violation is what switches cascade simulation off for the run, so every outage
    contributes an empty cascade frame; a base-case report and simulated cascade rows can never
    both exist.
    """
    if not events:
        return pl.from_pandas(get_empty_dataframe_from_model(CascadeResultSchema).reset_index())

    rows = [
        {
            "timestep": timestep,
            "contingency": BASECASE_CONTINGENCY_ID,
            "cascade_number": event.cascade_number,
            "contingency_outage_id": BASECASE_CONTINGENCY_ID,
            "contingency_name": BASECASE_CONTINGENCY_ID,
            "element_outage_group_id": event.outage_group_id,
            "element_mrid": event.element_mrid,
            "element_id": event.element_id,
            "element_name": event.element_name,
            # CascadeReasonType is a str enum; the result column holds the plain string.
            "cascade_reason": getattr(event.cascade_reason, "value", event.cascade_reason),
            "loading": event.loading,
            "r_ohm": event.r_ohm,
            "x_ohm": event.x_ohm,
            "distance_protection_severity": event.distance_protection_severity,
            "activated_schemes_per_iter": event.activated_schemes_per_iter,
        }
        for event in events
    ]

    return pl.DataFrame(rows).with_columns(
        pl.col("loading").cast(pl.Float64, strict=False),
        pl.col("r_ohm").cast(pl.Float64, strict=False),
        pl.col("x_ohm").cast(pl.Float64, strict=False),
        pl.col(
            "element_outage_group_id",
            "element_mrid",
            "element_id",
            "element_name",
            "distance_protection_severity",
            "activated_schemes_per_iter",
        ).cast(pl.String),
    )
