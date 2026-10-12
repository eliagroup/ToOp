# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compute the N-1 AC/DC power flow for the pandapower network."""

import json
import logging
import math
import time
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum

import pandapower as pp
import pandas as pd
import pandera.pandas as pa
import pandera.typing as pat
import pandera.typing.polars as patpl
import polars as pl
import ray
from beartype.typing import Any, Callable, Final, Optional, Union
from opentelemetry.trace import Span, StatusCode
from ray.util.queue import Queue
from toop_engine_contingency_analysis.pandapower.cascade.basecase import (
    basecase_violation_warning,
    build_basecase_cascade_results,
    screen_basecase_for_cascade,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection import (
    prepare_cascade_run_constants,
)
from toop_engine_contingency_analysis.pandapower.cascade.simulation import (
    CascadeSimulator,
)
from toop_engine_contingency_analysis.pandapower.outage_net_copy import (
    copy_net_for_outage,
    freeze_net_columns,
)
from toop_engine_contingency_analysis.pandapower.outage_power_flow import run_outage_power_flow
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerContingencyGroup,
    PandapowerNMinus1Definition,
    SlackAllocationConfig,
    get_convergence_df,
    get_failed_va_diff_results,
    get_regulating_element_results,
    get_switch_results,
    get_va_diff_results,
    translate_nminus1_for_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.contingency_outage_group import (
    get_outage_group_for_contingency,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.result_constants import (
    ELEMENT_NAME_LOOKUP_COLUMN,
    ResultConstants,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.branch_results import (
    get_branch_results_polars,
    get_failed_branch_results_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.node_results import (
    get_failed_node_results_polars,
    get_node_results_polars,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import (
    get_failed_switch_results,
    get_switch_mapped_elements,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    ParallelContingencyAnalysisContext,
    SequentialContingencyAnalysisContext,
    SingleOutageContext,
    SingleOutageSppsContext,
)
from toop_engine_contingency_analysis.pandapower.spps import SppsResult
from toop_engine_contingency_analysis.result_filter import branch_keep_expr, node_keep_expr
from toop_engine_contingency_analysis.tracing import (
    add_event,
    attached_trace_carrier,
    current_outage_span,
    current_trace_carrier,
    flush_tracer_provider,
    loadflow_attrs,
    net_size_attrs,
    outage_span,
    process_uptime_s,
    ray_runtime_attrs,
    set_attrs,
    span,
    step,
    trace_detail,
)
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_result_helpers import (
    convert_polars_loadflow_results_to_pandas,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import concatenate_loadflow_results_polars
from toop_engine_interfaces.loadflow_results import (
    CascadeResultSchema,
    ConnectivityResultSchema,
    ConvergenceStatus,
    LoadflowResults,
    SppsResultsSchema,
)
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    CascadeResultSchemaPolars,
    ConvergedSchemaPolars,
    LoadflowResultsPolars,
    NodeResultSchemaPolars,
    RegulatingElementResultSchemaPolars,
    SppsResultsSchemaPolars,
    SwitchResultsSchemaPolars,
    VADiffResultSchemaPolars,
)
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

logger = logging.getLogger(__name__)

# Sequential report interval and the parallel ray.wait poll timeout.
_PROGRESS_POLL_SECONDS: Final[float] = 1.0


def _report_progress(on_progress: Callable[[int, int], None], done: int, total: int) -> None:
    """Call ``on_progress``. Exceptions are logged and ignored."""
    add_event("progress", **{"toop.progress.done": done, "toop.progress.total": total})
    try:
        on_progress(done, total)
    except Exception:
        logger.warning("Progress callback failed for %d/%d outage groups", done, total, exc_info=True)


class _ParallelProgress:
    """Driver-side progress for the parallel path. No-ops when ``on_progress`` is unset."""

    def __init__(self, on_progress: Optional[Callable[[int, int], None]], total: int) -> None:
        self._on_progress = on_progress
        self._total = total
        self._done = 0
        self._reported = 0
        # num_cpus=0 so the queue actor does not take a worker slot.
        self.queue = Queue(actor_options={"num_cpus": 0}) if on_progress is not None else None

    @property
    def wait_timeout(self) -> Optional[float]:
        return _PROGRESS_POLL_SECONDS if self._on_progress is not None else None

    def poll(self) -> None:
        if self._on_progress is None or self.queue is None:
            return
        try:
            while not self.queue.empty():
                self._done += self.queue.get_nowait()
        except Exception:
            # Stop after the first failure; a dead queue would otherwise log on every poll.
            logger.warning("Progress queue unavailable; giving up on progress reporting", exc_info=True)
            self._on_progress = None
            self.queue = None
            return
        if self._done != self._reported:
            self._reported = self._done
            _report_progress(self._on_progress, self._done, self._total)

    def finish(self) -> None:
        if self._on_progress is not None and self._reported != self._total:
            _report_progress(self._on_progress, self._total, self._total)


def _scrub_enums_for_json(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _scrub_enums_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_enums_for_json(v) for v in obj]
    return obj


def _serialize_cascade_events(events: list[Any]) -> list[Any]:
    """Convert cascade simulator events into JSON-friendly structures."""
    serialized: list[Any] = []
    for ev in events:
        if is_dataclass(ev):
            serialized.append(_scrub_enums_for_json(asdict(ev)))
        elif callable(getattr(ev, "to_dict", None)):
            serialized.append(_scrub_enums_for_json(ev.to_dict()))
        else:
            serialized.append({"repr": repr(ev)})
    return serialized


def filter_to_monitored(results: pl.DataFrame, monitored_element_ids: pl.Series) -> pl.DataFrame:
    """Keep only rows whose ``element`` is monitored."""
    return results.filter(pl.col("element").is_in(monitored_element_ids))


def _apply_contingency(result: pl.DataFrame, contingency: PandapowerContingency) -> pl.DataFrame:
    """Stamp a contingency's id and name onto a flat polars result frame.

    Result frames are computed once for a group's first contingency; this rewrites the
    ``contingency`` / ``contingency_name`` columns so the same rows can represent any
    contingency in the group.
    """
    return result.with_columns(
        pl.lit(contingency.unique_id).alias("contingency"),
        pl.lit(contingency.name).alias("contingency_name"),
    )


@dataclass
class OutageElementResults:
    """Result tables collected for one outage calculation (flat polars frames)."""

    branch_results: patpl.DataFrame[BranchResultSchemaPolars]
    full_branch_results: patpl.DataFrame[BranchResultSchemaPolars]
    node_results: patpl.DataFrame[NodeResultSchemaPolars]
    va_diff_results: patpl.DataFrame[VADiffResultSchemaPolars]
    regulating_element_results: patpl.DataFrame[RegulatingElementResultSchemaPolars]
    switch_results: patpl.DataFrame[SwitchResultsSchemaPolars]


def run_single_outage(
    net: pp.pandapowerNet,
    grouped_contingency: PandapowerContingencyGroup,
    ctx: SingleOutageContext,
    slack_allocation_config: SlackAllocationConfig | None = None,
) -> LoadflowResultsPolars:
    """Compute a single outage for the given network.

    When *slack_allocation_config* is provided it is forwarded to
    :func:`run_outage_power_flow`, which owns all slack-allocation logic
    (initial PF assignment and in-loop SpPS reassignment).
    """
    outaged_elements = grouped_contingency.elements

    status, spps_result = run_outage_power_flow(
        net=net,
        spps=ctx.spps,
        method=ctx.method,
        outaged_elements=outaged_elements,
        runpp_kwargs=ctx.runpp_kwargs,
        slack_allocation_config=slack_allocation_config,
        basecase_net=ctx.basecase_net,
    )
    outage = current_outage_span()
    set_attrs(outage, **{"toop.status": status.value, "toop.spps.used": spps_result is not None})
    if status == ConvergenceStatus.FAILED:
        outage.set_status(StatusCode.ERROR, "load flow failed")
    if spps_result is not None:
        set_attrs(
            outage,
            **{
                "toop.spps.iterations": spps_result.iterations,
                "toop.spps.max_iterations_reached": spps_result.max_iterations_reached,
                "toop.spps.power_flow_failed": spps_result.power_flow_failed,
            },
        )

    spps_results = (
        _build_spps_results(
            spps_result=spps_result,
            contingencies=grouped_contingency.contingencies,
            timestep=ctx.timestep,
        )
        if spps_result is not None
        else pl.from_pandas(get_empty_dataframe_from_model(SppsResultsSchema).reset_index())
    )

    convergence_df = _build_convergence_results(
        grouped_contingency=grouped_contingency,
        timestep=ctx.timestep,
        status=status,
    )

    with step("results"):
        element_results = _collect_element_results(
            net=net,
            grouped_contingency=grouped_contingency,
            ctx=ctx,
            status=status,
        )

    with step("cascade") as cascade_span:
        cascade_results = _collect_cascade_results(
            net=net,
            ctx=ctx,
            grouped_contingency=grouped_contingency,
            status=status,
            branch_results_df=element_results.branch_results,
            switch_results_df=element_results.switch_results,
        )
        set_attrs(
            cascade_span,
            **{
                "toop.cascade.ran": _should_run_cascade(ctx=ctx, status=status),
                # One row per (contingency in the group, event).
                "toop.cascade.n_events": cascade_results.height // max(len(grouped_contingency.contingencies), 1),
                "toop.cascade.depth": cascade_results["cascade_number"].max() if cascade_results.height else 0,
            },
        )

    # Filtering happens last, on the frames that are about to leave this outage. Everything that needs the complete
    # picture has already run: switch results aggregate over branches and nodes that are not themselves monitored, and
    # cascade screening reads the branch results above. It also has to come after _copy_results_for_all_contingencies,
    # which stamps this group's other contingency ids onto the same rows - a basecase exemption applied any earlier
    # would test the id of the group's first contingency instead. Dropping rows here still spares the concatenation,
    # the ray transfer between workers and the stored file.
    branch_results = element_results.branch_results.filter(
        branch_keep_expr(ctx.result_filter.branch_filters, ctx.basecase_contingency_id)
    )
    node_results = element_results.node_results.filter(
        node_keep_expr(ctx.result_filter.node_filters, ctx.basecase_contingency_id)
    )

    # Each outage produces a flat-polars LoadflowResultsPolars; ``model_construct`` skips
    # per-outage validation. LoadflowResultsPolars is built around LazyFrames, so the eager
    # result frames are made lazy here. Everything is concatenated in polars and converted to
    # pandas once, at the end of the run (see run_contingency_analysis_pandapower).
    return LoadflowResultsPolars.model_construct(
        job_id=ctx.job_id,
        branch_results=branch_results.lazy(),
        node_results=node_results.lazy(),
        converged=convergence_df.lazy(),
        regulating_element_results=element_results.regulating_element_results.lazy(),
        va_diff_results=element_results.va_diff_results.lazy(),
        switch_results=element_results.switch_results.lazy(),
        warnings=[],
        spps_results=spps_results.lazy(),
        cascade_results=cascade_results.lazy(),
    )


@pa.check_types
def _build_spps_results(
    spps_result: SppsResult,
    contingencies: list[PandapowerContingency],
    timestep: int,
) -> patpl.DataFrame[SppsResultsSchemaPolars]:
    n = len(contingencies)
    activated = json.dumps(spps_result.activated_schemes_per_iter) if n else None
    return pl.DataFrame(
        {
            "timestep": [timestep] * n,
            "contingency": [c.unique_id for c in contingencies],
            "iterations": [spps_result.iterations] * n,
            "activated_schemes_per_iter": [activated] * n,
            "max_iterations_reached": [spps_result.max_iterations_reached] * n,
            "power_flow_failed": [spps_result.power_flow_failed] * n,
        },
        schema={
            "timestep": pl.Int64,
            "contingency": pl.String,
            "iterations": pl.Int64,
            "activated_schemes_per_iter": pl.String,
            "max_iterations_reached": pl.Boolean,
            "power_flow_failed": pl.Boolean,
        },
    )


@pa.check_types
def _build_convergence_results(
    grouped_contingency: PandapowerContingencyGroup,
    timestep: int,
    status: ConvergenceStatus,
) -> patpl.DataFrame[ConvergedSchemaPolars]:
    # get_convergence_df stays pandas (small, one row per contingency); convert to flat polars.
    frames = [
        pl.from_pandas(get_convergence_df(timestep=timestep, contingency=contingency, status=status.value).reset_index())
        for contingency in grouped_contingency.contingencies
    ]
    return pl.concat(frames)


def _collect_element_results(
    net: pp.pandapowerNet,
    grouped_contingency: PandapowerContingencyGroup,
    ctx: SingleOutageContext,
    status: ConvergenceStatus,
) -> OutageElementResults:
    first_contingency = grouped_contingency.contingencies[0]

    (
        branch_results_df,
        full_branch_results_df,
        node_results_df,
        va_diff_results_df,
        switch_results_df,
    ) = get_element_results_df(
        net,
        first_contingency,
        ctx.timestep,
        status,
        ctx.result_constants,
    )

    with step("results.regulating"):
        regulating_element_results_df = get_regulating_element_results(
            ctx.timestep, ctx.result_constants.monitored_element_ids, first_contingency
        )

    results = OutageElementResults(
        branch_results=_copy_results_for_all_contingencies(
            branch_results_df,
            grouped_contingency,
        ),
        full_branch_results=full_branch_results_df,
        node_results=_copy_results_for_all_contingencies(
            node_results_df,
            grouped_contingency,
        ),
        va_diff_results=_copy_results_for_all_contingencies(
            va_diff_results_df,
            grouped_contingency,
        ),
        regulating_element_results=_copy_results_for_all_contingencies(
            regulating_element_results_df,
            grouped_contingency,
        ),
        switch_results=_copy_results_for_all_contingencies(
            switch_results_df,
            grouped_contingency,
        ),
    )

    _update_result_names(results=results, element_name_frame=ctx.result_constants.element_name_frame)

    return results


def _copy_results_for_all_contingencies(
    result: pl.DataFrame,
    grouped_contingency: PandapowerContingencyGroup,
) -> pl.DataFrame:
    frames = [_apply_contingency(result, contingency) for contingency in grouped_contingency.contingencies]

    if not frames:
        return result.clear()

    return pl.concat(frames)


def _update_result_names(
    results: OutageElementResults,
    element_name_frame: pl.DataFrame,
) -> None:
    # polars frames are immutable, so reassign the filled result back onto the dataclass.
    results.branch_results = update_results_with_names(results.branch_results, element_name_frame)
    results.node_results = update_results_with_names(results.node_results, element_name_frame)
    results.va_diff_results = update_results_with_names(results.va_diff_results, element_name_frame)
    results.regulating_element_results = update_results_with_names(results.regulating_element_results, element_name_frame)
    results.switch_results = update_results_with_names(results.switch_results, element_name_frame)


def _collect_cascade_results(
    net: pp.pandapowerNet,
    ctx: SingleOutageContext,
    grouped_contingency: PandapowerContingencyGroup,
    status: ConvergenceStatus,
    branch_results_df: pl.DataFrame,
    switch_results_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build cascade result rows for :attr:`LoadflowResults.cascade_results`.

    Each row describes one cascade event generated after the initial
    contingency load flow.
    """
    if not _should_run_cascade(
        ctx=ctx,
        status=status,
    ):
        return pl.from_pandas(get_empty_dataframe_from_model(CascadeResultSchema).reset_index())

    simulator = CascadeSimulator(
        ctx.cascade,
        ctx.spps,
        method=ctx.method,
        runpp_kwargs=ctx.runpp_kwargs,
        bus_couplers_mrids=ctx.bus_couplers_mrids,
    )

    cascade_events = simulator.simulate(
        copy_net_for_outage(net),
        branch_results_df,
        switch_results_df,
        initial_contingency=grouped_contingency.contingencies[0],
        basecase_net=ctx.basecase_net,
        monitored_elements=ctx.monitored_elements,
    )

    return _build_cascade_results_df(
        cascade_events=cascade_events,
        contingencies=grouped_contingency.contingencies,
        contingency_outage_id=grouped_contingency.outage_group_id,
        timestep=ctx.timestep,
    )


@pa.check_types
def _build_cascade_results_df(
    cascade_events: list[Any],
    contingencies: list[PandapowerContingency],
    contingency_outage_id: str,
    timestep: int,
) -> patpl.DataFrame[CascadeResultSchemaPolars]:
    if not cascade_events or not contingencies:
        return pl.from_pandas(get_empty_dataframe_from_model(CascadeResultSchema).reset_index())

    rows = []
    for contingency in contingencies:
        for event in cascade_events:
            event_dict = _scrub_enums_for_json(asdict(event)) if is_dataclass(event) else event.to_dict()
            rows.append(
                {
                    "timestep": timestep,
                    "contingency": contingency.unique_id,
                    "cascade_number": event_dict["cascade_number"],
                    "contingency_outage_id": contingency_outage_id,
                    "contingency_name": contingency.name,
                    "element_outage_group_id": event_dict.get("outage_group_id"),
                    "element_mrid": event_dict.get("element_mrid"),
                    "element_id": event_dict.get("element_id"),
                    "element_name": event_dict.get("element_name"),
                    "cascade_reason": event_dict["cascade_reason"],
                    "loading": event_dict.get("loading"),
                    "r_ohm": event_dict.get("r_ohm"),
                    "x_ohm": event_dict.get("x_ohm"),
                    "distance_protection_severity": event_dict.get("distance_protection_severity"),
                    "activated_schemes_per_iter": event_dict.get("activated_schemes_per_iter"),
                }
            )

    # strict=False coerces non-numeric entries to null, matching pd.to_numeric(errors="coerce").
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


def _should_run_cascade(
    ctx: SingleOutageContext,
    status: ConvergenceStatus,
) -> bool:
    return ctx.cascade is not None and status == ConvergenceStatus.CONVERGED


def update_results_with_names(
    df: pl.DataFrame,
    element_name_frame: pl.DataFrame,
) -> pl.DataFrame:
    """
    Enrich results DataFrame with element names (flat polars frame with an ``element`` column).

    This function fills missing values in the `element_name` column using a
    lookup frame mapping element indices to human-readable names.

    Args:
        df: Results DataFrame. Expected to have:
            - a MultiIndex containing level `"element"`
            - a column `"element_name"`
        element_name_frame: Lookup frame from :func:`build_element_name_frame`, mapping the
            `"element"` index level to element name. Built once per job: it holds one row per
            monitored element, and rebuilding the lookup per result frame is what this avoids.

    Returns
    -------
        Updated DataFrame (a new frame; polars frames are immutable).

    Notes
    -----
        - Only missing or empty `element_name` values are filled.
        - If an element is not found in the lookup, the value falls back to an empty string.
        - Row order is preserved: downstream results are aligned to it.
    """
    return (
        # maintain_order="left": the result frames are positionally aligned with the arrays the
        # extractors built them from, so the join must not reshuffle them.
        df.join(element_name_frame, on="element", how="left", maintain_order="left")
        .with_columns(
            pl.when((pl.col("element_name").is_null()) | (pl.col("element_name") == ""))
            # Unmapped elements have no lookup row, hence the null fallback to "".
            .then(pl.col(ELEMENT_NAME_LOOKUP_COLUMN).fill_null(""))
            .otherwise(pl.col("element_name"))
            .alias("element_name")
        )
        .drop(ELEMENT_NAME_LOOKUP_COLUMN)
    )


def get_element_results_df(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    status: ConvergenceStatus,
    result_constants: ResultConstants,
) -> tuple[
    patpl.DataFrame[BranchResultSchemaPolars],
    patpl.DataFrame[BranchResultSchemaPolars],
    patpl.DataFrame[NodeResultSchemaPolars],
    patpl.DataFrame[VADiffResultSchemaPolars],
    patpl.DataFrame[SwitchResultsSchemaPolars],
]:
    """Get the element results dataframes for the given contingency and monitored elements.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to get the results from
    contingency : PandapowerContingency
        The contingency to get the results for
    timestep : int
        The timestep of the results
    status : ConvergenceStatus
        The convergence status of the loadflow computation
    result_constants : ResultConstants
        Outage-invariant inputs for branch/node/switch result extraction (element ids, rated
        currents, voltage levels, base-case voltages, the polars switch mapping and the
        monitored-element projections). Built once per run and required: rebuilding it per
        outage is exactly the cost it exists to avoid.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Filtered branch results, full branch results, node results, voltage-angle
        difference results, and switch results, all as flat polars frames.
    """
    if status == ConvergenceStatus.CONVERGED:
        with step("results.branch"):
            full_branch_results = get_branch_results_polars(net, contingency, timestep, result_constants)
        with step("results.node"):
            node_results = get_node_results_polars(net, contingency, timestep, result_constants)
        with step("results.va_diff"):
            va_diff_results = get_va_diff_results(net, timestep, contingency, result_constants)

        # IMPORTANT:
        # Do NOT filter branch/node results before this step.
        # Switch result calculation depends on connectivity and may require data
        # from non-monitored branches/nodes (e.g. a monitored switch connected to
        # an unmonitored line/trafo). Therefore we pass full result sets here.
        with step("results.switch"):
            switch_results = get_switch_results(
                net,
                contingency,
                timestep,
                full_branch_results,
                node_results,
                result_constants.switch_element_mapping_pl,
            )
        branch_results = filter_to_monitored(full_branch_results, result_constants.monitored_element_ids)
        node_results = filter_to_monitored(node_results, result_constants.monitored_element_ids)

    else:
        # Native-polars all-null result frames, matching the converged builders' layout.
        branch_results = get_failed_branch_results_polars(
            timestep,
            [contingency.unique_id],
            result_constants.monitored_branch_ids,
            result_constants.monitored_trafo3w_ids,
        )
        full_branch_results = branch_results
        node_results = get_failed_node_results_polars(timestep, [contingency.unique_id], result_constants.monitored_bus_ids)
        va_diff_results = get_failed_va_diff_results(timestep, contingency, result_constants)
        switch_results = get_failed_switch_results(timestep, result_constants.switch_element_mapping_pl, contingency)
    return branch_results, full_branch_results, node_results, va_diff_results, switch_results


def _batch_span(
    n_groups: int,
    trace_carrier: Optional[dict[str, str]],
    batch_index: Optional[int],
    submitted_at: Optional[float],
) -> AbstractContextManager[Span]:
    """``toop.ca.batch`` inside a Ray task (continuing the driver's trace), ``toop.ca.sequential`` otherwise."""
    if trace_carrier is None:
        return span("toop.ca.sequential", **{"toop.n_groups": n_groups})
    attrs: dict[str, Any] = {
        "toop.batch_index": batch_index,
        "toop.n_groups": n_groups,
        "toop.worker.process_uptime_s": round(process_uptime_s(), 3),
        **ray_runtime_attrs(),
    }
    if submitted_at is not None:
        # Queueing, worker start-up, imports and argument unpickling all happen before the task
        # body runs, so this is the only place they can be measured from.
        attrs["toop.submit_to_start_ms"] = round((time.time() - submitted_at) * 1000.0, 3)
    return span("toop.ca.batch", **attrs)


def run_contingency_analysis_sequential(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: SequentialContingencyAnalysisContext,
    on_progress: Optional[Callable[[int, int], None]] = None,
    progress_queue: Optional[Queue] = None,
    trace_carrier: Optional[dict[str, str]] = None,
    batch_index: Optional[int] = None,
    submitted_at: Optional[float] = None,
) -> list[LoadflowResultsPolars]:
    """Compute a full N-1 analysis for the given network for a single timestep.

    Iterates over every contingency group, deep-copies the network, and calls
    :func:`run_single_outage`.  ``ctx.slack_allocation_config`` is forwarded to
    each outage call so that :func:`run_outage_power_flow` can handle slack-bus
    assignment (initial PF and SpPS in-loop reassignment) internally.

    ``on_progress(done, total)`` reports completed outage groups at most once per
    :data:`_PROGRESS_POLL_SECONDS`, and always after the last group. When this
    function runs as a Ray task, leave ``on_progress`` unset and pass
    ``progress_queue`` instead; a callable cannot be used from the worker.

    ``trace_carrier`` (from :func:`~toop_engine_contingency_analysis.tracing.current_trace_carrier`)
    makes the task's spans children of the driver's trace; ``batch_index`` and ``submitted_at``
    (``time.time()`` at submission) are recorded on the batch span.
    """
    with attached_trace_carrier(trace_carrier), trace_detail(ctx.tracing.detail):
        total_groups = len(n_minus_1_definition.grouped_contingencies)
        try:
            with _batch_span(total_groups, trace_carrier, batch_index, submitted_at):
                return _run_groups(net, n_minus_1_definition, ctx, on_progress, progress_queue)
        finally:
            if trace_carrier is not None:
                flush_tracer_provider()


def _run_groups(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: SequentialContingencyAnalysisContext,
    on_progress: Optional[Callable[[int, int], None]],
    progress_queue: Optional[Queue],
) -> list[LoadflowResultsPolars]:
    # Freezing the source once is all the barrier needs: every outage copy shares these columns
    # and inherits their read-only flag, while the columns it is allowed to write are reassigned
    # from a deep copy and stay writable. It has to happen here rather than in the caller because
    # read-only-ness does not survive the pickling that ships the net to a ray worker.
    if ctx.freeze_net_columns:
        with span("toop.ca.freeze_net_columns"):
            freeze_net_columns(net)

    results = []
    total_groups = len(n_minus_1_definition.grouped_contingencies)
    unreported_groups = 0
    last_report = time.monotonic()

    with span("toop.ca.result_constants", **{"toop.n_monitored_elements": len(n_minus_1_definition.monitored_elements)}):
        # Element ids, rated currents, base-case voltages, the polars switch mapping and the
        # element-name map are the same for every outage in this run, so resolve them once here.
        result_constants = ResultConstants.from_network(
            net,
            ctx.basecase_net,
            monitored_elements=n_minus_1_definition.monitored_elements,
            switch_element_mapping=ctx.switch_element_mapping,
        )

    single_outage_ctx = SingleOutageContext(
        result_filter=ctx.result_filter,
        basecase_contingency_id=ctx.basecase_contingency_id,
        monitored_elements=n_minus_1_definition.monitored_elements,
        timestep=ctx.timestep,
        job_id=ctx.job_id,
        method=ctx.method,
        runpp_kwargs=ctx.runpp_kwargs,
        basecase_net=ctx.basecase_net,
        switch_element_mapping=ctx.switch_element_mapping,
        result_constants=result_constants,
        spps=SingleOutageSppsContext(
            conditions=ctx.spps_conditions,
            actions=ctx.spps_actions,
            rules_max_iterations=ctx.spps_rules_max_iterations,
            on_power_flow_error=ctx.on_power_flow_error,
        ),
        cascade=ctx.cascade,
        bus_couplers_mrids=ctx.bus_couplers_mrids,
    )

    for done_groups, grouped_contingency in enumerate(n_minus_1_definition.grouped_contingencies, start=1):
        with outage_span(
            **{
                "toop.outage_group_id": grouped_contingency.outage_group_id,
                "toop.n_contingencies": len(grouped_contingency.contingencies),
                "toop.n_outaged_elements": len(grouped_contingency.elements),
            }
        ):
            with step("copy_net"):
                copy_net = copy_net_for_outage(net)

            single_res = run_single_outage(
                net=copy_net,
                grouped_contingency=grouped_contingency,
                ctx=single_outage_ctx,
                slack_allocation_config=ctx.slack_allocation_config,
            )

        results.append(single_res)

        # At most once per second, and always after the last group.
        unreported_groups += 1
        now = time.monotonic()
        if done_groups == total_groups or now - last_report >= _PROGRESS_POLL_SECONDS:
            last_report = now
            if on_progress is not None:
                _report_progress(on_progress, done_groups, total_groups)
            if progress_queue is not None:
                # put_nowait still hits the actor, so use the same cadence as the callback.
                progress_queue.put_nowait(unreported_groups)
            unreported_groups = 0

    return results


@dataclass
class _Dispatch:
    """Driver-side bookkeeping for the in-flight Ray batches."""

    progress: _ParallelProgress
    started: float = field(default_factory=time.monotonic)
    batches: dict[ray.ObjectRef, tuple[int, int]] = field(default_factory=dict)
    result_lists: list[list[LoadflowResultsPolars]] = field(default_factory=list)
    first_result_after_ms: Optional[float] = None

    def collect_one(self, handles: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
        """Wait for one finished batch (or the progress poll timeout) and take its results."""
        finished, handles = ray.wait(handles, num_returns=1, timeout=self.progress.wait_timeout)
        for handle in finished:
            self.result_lists.append(ray.get(handle))
            elapsed_ms = (time.monotonic() - self.started) * 1000.0
            if self.first_result_after_ms is None:
                self.first_result_after_ms = elapsed_ms
            batch_index, n_groups = self.batches.pop(handle)
            add_event(
                "batch_finished",
                **{"toop.batch_index": batch_index, "toop.n_groups": n_groups, "toop.elapsed_ms": round(elapsed_ms, 3)},
            )
        self.progress.poll()
        return handles


def run_contingency_analysis_parallel(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: ParallelContingencyAnalysisContext,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[LoadflowResultsPolars]:
    """Compute the N-1 AC/DC power flow for the network in parallel.

    ``on_progress(done, total)`` is called from this process. Workers put
    finished group counts on a :class:`ray.util.queue.Queue`; the driver
    reads it while waiting.
    """
    n_outages = len(n_minus_1_definition.grouped_contingencies)
    batch_size = ctx.parallel.batch_size

    if batch_size is None:
        batch_size = math.ceil(n_outages / ctx.parallel.n_processes)

    work = []
    for i in range(0, n_outages, batch_size):
        grouped_batch = n_minus_1_definition.grouped_contingencies[i : i + batch_size]
        work.append(
            n_minus_1_definition.model_copy(
                update={
                    "contingencies": [],
                    "grouped_contingencies": grouped_batch,
                }
            )
        )

    with span(
        "toop.ca.parallel",
        **{"toop.n_batches": len(work), "toop.batch_size": batch_size, "toop.n_processes": ctx.parallel.n_processes},
    ) as parallel_span:
        with span("toop.ca.ray_init") as init_span:
            # Without this, Ray starts implicitly inside the first .remote() and the cluster
            # start-up time is invisible.
            was_initialized = ray.is_initialized()
            if not was_initialized:
                ray.init()
            progress = _ParallelProgress(on_progress, n_outages)
            set_attrs(
                init_span,
                **{
                    "ray.was_initialized": was_initialized,
                    "ray.num_cpus": ray.cluster_resources().get("CPU"),
                    "ray.node_id": ray.get_runtime_context().get_node_id(),
                },
            )

        sequential_ctx = SequentialContingencyAnalysisContext(
            result_filter=ctx.result_filter,
            basecase_contingency_id=ctx.basecase_contingency_id,
            job_id=ctx.job_id,
            timestep=ctx.timestep,
            slack_allocation_config=ctx.slack_allocation_config,
            method=ctx.method,
            runpp_kwargs=ctx.runpp_kwargs,
            basecase_net=ctx.basecase_net,
            switch_element_mapping=ctx.switch_element_mapping,
            spps_conditions=ctx.spps_conditions,
            spps_actions=ctx.spps_actions,
            spps_rules_max_iterations=ctx.spps_rules_max_iterations,
            on_power_flow_error=ctx.on_power_flow_error,
            cascade=ctx.cascade,
            bus_couplers_mrids=ctx.bus_couplers_mrids,
            freeze_net_columns=ctx.freeze_net_columns,
            tracing=ctx.tracing,
        )

        with span("toop.ca.put_inputs", **net_size_attrs(net)):
            # Both nets go to the object store once; passing them as plain arguments would
            # serialise them again for every batch (the ctx carries the base-case net).
            net_ref = ray.put(net)
            ctx_ref = ray.put(sequential_ctx)

        _compute_remote = ray.remote(run_contingency_analysis_sequential)
        trace_carrier = current_trace_carrier()
        dispatch = _Dispatch(progress)
        handles: list[ray.ObjectRef] = []

        with span("toop.ca.submit"):
            for batch_index, batch in enumerate(work):
                n_groups = len(batch.grouped_contingencies)
                # Pass the queue, not on_progress: a callable cannot be serialised into the worker.
                handle = _compute_remote.remote(
                    net=net_ref,
                    n_minus_1_definition=batch,
                    ctx=ctx_ref,
                    progress_queue=progress.queue,
                    trace_carrier=trace_carrier,
                    batch_index=batch_index,
                    submitted_at=time.time(),
                )
                handles.append(handle)
                dispatch.batches[handle] = (batch_index, n_groups)
                add_event("batch_submitted", **{"toop.batch_index": batch_index, "toop.n_groups": n_groups})

                # ray.wait may time out with no result; keep waiting until a slot frees.
                while handles and len(handles) >= ctx.parallel.n_processes:
                    handles = dispatch.collect_one(handles)

        with span("toop.ca.wait"):
            # One finished batch at a time so the driver can poll the progress queue. A single
            # ray.get on the remaining handles would block until the whole run finished.
            while handles:
                handles = dispatch.collect_one(handles)

        progress.finish()
        set_attrs(parallel_span, **{"toop.first_result_after_ms": dispatch.first_result_after_ms})

    return [result for result_list in dispatch.result_lists for result in result_list]


def _run_base_case_loadflow(
    net: pp.pandapowerNet,
    slack_allocation_config: SlackAllocationConfig,
    cfg: ContingencyAnalysisConfig,
) -> ConvergenceStatus:
    """Run load flow calculation for the contingency analysis base case.

    1. Assigns slack buses for each electrical island via
       :func:`assign_slack_per_island` (network graph and bus-lookup are
       derived internally from *net*).
    2. Executes a power flow (AC or DC) per *cfg*.

    *base_case* is accepted for API consistency but is not used; the base-case
    network already reflects the desired topology before this call.

    Parameters
    ----------
    net:
        Pandapower network to be modified and solved.
    base_case:
        Unused; kept for API compatibility.
    slack_allocation_config:
        Provides ``min_island_size`` for slack-bus island filtering.
    cfg:
        Global contingency analysis configuration (load-flow method and
        optional runpp arguments).

    Returns
    -------
    ConvergenceStatus
        Whether the base-case load flow converged. A failed base case leaves stale
        ``res_*`` tables behind, so callers must not read results from it.
    """
    with span("toop.ca.basecase_loadflow", **{"toop.method": cfg.method}) as basecase_span:
        with span("toop.ca.slack_allocation", **{"toop.min_island_size": slack_allocation_config.min_island_size}):
            assign_slack_per_island(
                net=net,
                min_island_size=slack_allocation_config.min_island_size,
            )

        try:
            runpp_kwargs = cfg.runpp_kwargs or {}

            if cfg.method == "dc":
                pp.rundcpp(net, **runpp_kwargs)
            else:
                pp.runpp(net, **runpp_kwargs)

        except (pp.LoadflowNotConverged, pp.ControllerNotConverged) as exc:
            logger.warning("Base-case load flow did not converge; continuing with stale res_* tables: %s", exc)
            set_attrs(basecase_span, **{"toop.status": ConvergenceStatus.FAILED.value}, **loadflow_attrs(net))
            return ConvergenceStatus.FAILED

        set_attrs(basecase_span, **{"toop.status": ConvergenceStatus.CONVERGED.value}, **loadflow_attrs(net))
        return ConvergenceStatus.CONVERGED


def build_connectivity_df(groups: list[PandapowerContingencyGroup]) -> pat.DataFrame[ConnectivityResultSchema]:
    """
    Build a connectivity result table mapping contingencies to affected elements.

    This function flattens a list of PandapowerContingencyGroup objects into a
    tabular representation where each row corresponds to a pair
    (contingency, element) along with the associated outage group identifier.

    For each contingency in a group, all elements of that outage group are
    considered affected. This reflects the modeling assumption that outage
    groups represent sets of elements that become unavailable together when
    separated from the grid by circuit breakers.

    Parameters
    ----------
    groups : list[PandapowerContingencyGroup]
        List of contingency groups. Each group contains:
        - multiple contingencies affecting the same connected component(s),
        - a set of elements representing the full outage scope,
        - a unique outage_group_id.

    Returns
    -------
    pat.DataFrame[ConnectivityResultSchema]
        A Pandas DataFrame with:
        - MultiIndex:
            * contingency (str): contingency identifier
            * element (str): element identifier
        - Column:
            * outage_group_id (str): identifier of the outage group

        Each row indicates that a given element is affected by a given
        contingency through their shared outage group.
    """
    records = [(c.unique_id, e.unique_id, g.outage_group_id) for g in groups for c in g.contingencies for e in g.elements]

    return pd.DataFrame(records, columns=["contingency", "element", "outage_group_id"]).set_index(["contingency", "element"])


def run_contingency_analysis_pandapower(
    net: pp.pandapowerNet,
    n_minus_1_definition: Nminus1Definition,
    job_id: str,
    timestep: int,
    cfg: ContingencyAnalysisConfig,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Union[LoadflowResults, LoadflowResultsPolars]:
    """Compute the N-1 AC/DC power flow for the network.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with topology already applied.
    n_minus_1_definition : Nminus1Definition
        N-1 definition containing contingencies and monitored elements.
    job_id : str
        Identifier of the current job.
    timestep : int
        Timestep associated with the computed results.
    cfg : ContingencyAnalysisConfig
        Execution configuration (method, islanding/slack settings, parallelization,
        cascade screening, etc.).
    on_progress : Optional[Callable[[int, int], None]]
        Called as ``on_progress(done, total)`` in outage groups. ``total`` is
        ``len(grouped_contingencies)`` and can be smaller than the contingency
        list when ``cfg.apply_outage_grouping`` is set. First call is ``(0, total)``
        before the first load flow; later calls are about once a second, then
        ``(total, total)`` on success. Runs on this thread between load flows.
        Exceptions are logged and ignored.

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the loadflow computation
    """
    with (
        trace_detail(cfg.tracing.detail),
        span("toop.ca.run", **_run_attrs(n_minus_1_definition, job_id, timestep, cfg)) as run_span,
    ):
        with span("toop.ca.translate_nminus1") as translate_span:
            pp_n1_definition = translate_nminus1_for_pandapower(n_minus_1_definition, net)
            set_attrs(
                translate_span,
                **{
                    "toop.n_missing_elements": len(pp_n1_definition.missing_elements),
                    "toop.n_missing_contingencies": len(pp_n1_definition.missing_contingencies),
                    "toop.n_duplicated_ids": len(pp_n1_definition.duplicated_grid_elements),
                },
            )
        if cfg.apply_outage_grouping:
            with span("toop.ca.outage_grouping", **{"toop.n_contingencies_in": len(pp_n1_definition.contingencies)}) as s:
                pp_n1_definition.grouped_contingencies = get_outage_group_for_contingency(
                    net=net,
                    contingencies=pp_n1_definition.contingencies,
                )
                set_attrs(s, **{"toop.n_groups_out": len(pp_n1_definition.grouped_contingencies)})
        else:
            pp_n1_definition.grouped_contingencies = [
                PandapowerContingencyGroup(contingencies=[cont], elements=cont.elements, outage_group_id=cont.unique_id)
                for cont in pp_n1_definition.contingencies
            ]
        set_attrs(run_span, **{"toop.n_groups": len(pp_n1_definition.grouped_contingencies)})

        if on_progress is not None:
            _report_progress(on_progress, 0, len(pp_n1_definition.grouped_contingencies))

        slack_allocation_config = SlackAllocationConfig(
            min_island_size=cfg.min_island_size,
        )

        # The filtering logic needs base case ids
        basecase = n_minus_1_definition.base_case
        basecase_contingency_id = basecase.id if basecase is not None else None

        basecase_status = _run_base_case_loadflow(
            net=net,
            cfg=cfg,
            slack_allocation_config=slack_allocation_config,
        )

        with span("toop.ca.switch_element_mapping", **{"toop.n_switches": len(net.switch)}) as s:
            switch_element_mapping = get_switch_mapped_elements(
                net=net,
                monitored_elements=pp_n1_definition.monitored_elements,
                side="bus",
            )
            set_attrs(s, **{"toop.n_mapped_rows": len(switch_element_mapping)})

        # Cascade run-invariants: convert sw_characteristics once and precompute the
        # base-case busbar-coupler set, so neither is redone per outage. Skipped when
        # cascade screening is disabled.
        bus_couplers_mrids: set[str] = set()
        if cfg.cascade is not None:
            with span("toop.ca.cascade_run_constants"):
                bus_couplers_mrids = prepare_cascade_run_constants(net, cfg.cascade)

        # A base case that already violates makes every contingency cascade meaningless, so it is
        # reported once here and cascade simulation is switched off for the whole run. The N-1 load
        # flows themselves are unaffected.
        with span("toop.ca.basecase_cascade_screen") as s:
            basecase_cascade_events = screen_basecase_for_cascade(
                net,
                cascade_configuration=cfg.cascade,
                monitored_elements=pp_n1_definition.monitored_elements,
                switch_element_mapping=switch_element_mapping,
                bus_couplers_mrids=bus_couplers_mrids,
                timestep=timestep,
                basecase_status=basecase_status,
            )
            set_attrs(s, **{"toop.n_events": len(basecase_cascade_events)})
        cascade_cfg = None if basecase_cascade_events else cfg.cascade

        with span("toop.ca.copy_basecase_net", **net_size_attrs(net)):
            basecase_net = deepcopy(net)

        if cfg.parallel.n_processes == 1 and cfg.parallel.batch_size is None:
            results = run_contingency_analysis_sequential(
                net=net,
                n_minus_1_definition=pp_n1_definition,
                ctx=SequentialContingencyAnalysisContext(
                    result_filter=cfg.result_filter,
                    basecase_contingency_id=basecase_contingency_id,
                    job_id=job_id,
                    timestep=timestep,
                    slack_allocation_config=slack_allocation_config,
                    method=cfg.method,
                    runpp_kwargs=cfg.runpp_kwargs,
                    basecase_net=basecase_net,
                    switch_element_mapping=switch_element_mapping,
                    spps_conditions=pp_n1_definition.spps_conditions,
                    spps_actions=pp_n1_definition.spps_actions,
                    spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                    on_power_flow_error=cfg.on_power_flow_error,
                    cascade=cascade_cfg,
                    bus_couplers_mrids=bus_couplers_mrids,
                    freeze_net_columns=cfg.freeze_net_columns,
                    tracing=cfg.tracing,
                ),
                on_progress=on_progress,
            )
        else:
            results = run_contingency_analysis_parallel(
                net=net,
                n_minus_1_definition=pp_n1_definition,
                ctx=ParallelContingencyAnalysisContext(
                    result_filter=cfg.result_filter,
                    basecase_contingency_id=basecase_contingency_id,
                    job_id=job_id,
                    timestep=timestep,
                    slack_allocation_config=slack_allocation_config,
                    basecase_net=basecase_net,
                    switch_element_mapping=switch_element_mapping,
                    spps_conditions=pp_n1_definition.spps_conditions,
                    spps_actions=pp_n1_definition.spps_actions,
                    method=cfg.method,
                    runpp_kwargs=cfg.runpp_kwargs,
                    spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                    on_power_flow_error=cfg.on_power_flow_error,
                    parallel=cfg.parallel,
                    cascade=cascade_cfg,
                    bus_couplers_mrids=bus_couplers_mrids,
                    freeze_net_columns=cfg.freeze_net_columns,
                    tracing=cfg.tracing,
                ),
                on_progress=on_progress,
            )
        # Per-outage results are polars; concatenate in polars and convert to pandas once at the
        # very end (only when the caller wants pandas).
        with span("toop.ca.concatenate_results", **{"toop.n_results": len(results)}):
            lf_result = concatenate_loadflow_results_polars(results)

        if basecase_cascade_events:
            # Every outage contributed an empty cascade frame (the screen switched cascading off),
            # so the base-case report is the whole cascade result table.
            lf_result.cascade_results = build_basecase_cascade_results(basecase_cascade_events, timestep).lazy()
            lf_result.warnings.append(basecase_violation_warning(basecase_cascade_events))

        missing_element_warnings = [
            f"Element with id {element.id} not found in the network." for element in pp_n1_definition.missing_elements
        ]
        missing_contingency_warnings = [
            f"Contingency with id {contingency.id} contains elements that are not found in the network."
            for contingency in pp_n1_definition.missing_contingencies
        ]
        duplicated_id_warnings = [
            f"Element with id {element_id} is not unique in the grid."
            for element_id in pp_n1_definition.duplicated_grid_elements
        ]
        lf_result.warnings = [
            *duplicated_id_warnings,
            *missing_element_warnings,
            *missing_contingency_warnings,
            *lf_result.warnings,
        ]
        # Travels with the results so a reader can tell an absent row from a quiet one.
        lf_result.result_filter = cfg.result_filter if cfg.result_filter.is_active() else None

        if cfg.apply_outage_grouping:
            lf_result.connectivity_result = pl.from_pandas(
                build_connectivity_df(pp_n1_definition.grouped_contingencies).reset_index()
            ).lazy()

        if cfg.polars:
            return lf_result
        with span("toop.ca.to_pandas"):
            return convert_polars_loadflow_results_to_pandas(lf_result)


def _run_attrs(
    n_minus_1_definition: Nminus1Definition, job_id: str, timestep: int, cfg: ContingencyAnalysisConfig
) -> dict[str, Any]:
    runpp_kwargs = cfg.runpp_kwargs or {}
    return {
        "toop.job_id": job_id,
        "toop.timestep": timestep,
        "toop.method": cfg.method,
        "toop.n_contingencies": len(n_minus_1_definition.contingencies),
        "toop.n_monitored_elements": len(n_minus_1_definition.monitored_elements),
        "toop.n_spps_rules": len(n_minus_1_definition.spps_rules or []),
        "toop.cascade.enabled": cfg.cascade is not None,
        "toop.parallel.n_processes": cfg.parallel.n_processes,
        "toop.parallel.batch_size": cfg.parallel.batch_size,
        "toop.outage_grouping": cfg.apply_outage_grouping,
        "toop.polars": cfg.polars,
        "toop.tracing.detail": cfg.tracing.detail,
        "toop.runpp.lightsim2grid": runpp_kwargs.get("lightsim2grid"),
        "toop.runpp.run_control": runpp_kwargs.get("run_control"),
        "toop.runpp.enforce_q_lims": runpp_kwargs.get("enforce_q_lims"),
    }
