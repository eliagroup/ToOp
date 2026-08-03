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
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum

import pandapower as pp
import pandas as pd
import pandera.typing as pat
import polars as pl
import ray
from beartype.typing import Any, Union
from toop_engine_contingency_analysis.pandapower.cascade.detection import (
    prepare_cascade_run_constants,
)
from toop_engine_contingency_analysis.pandapower.cascade.simulation import (
    CascadeSimulator,
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
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

logger = logging.getLogger(__name__)


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

    branch_results: pl.DataFrame
    full_branch_results: pl.DataFrame
    node_results: pl.DataFrame
    va_diff_results: pl.DataFrame
    regulating_element_results: pl.DataFrame
    switch_results: pl.DataFrame


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

    element_results = _collect_element_results(
        net=net,
        grouped_contingency=grouped_contingency,
        ctx=ctx,
        status=status,
    )

    cascade_results = _collect_cascade_results(
        net=net,
        ctx=ctx,
        grouped_contingency=grouped_contingency,
        status=status,
        branch_results_df=element_results.branch_results,
        switch_results_df=element_results.switch_results,
    )

    # Each outage produces a flat-polars LoadflowResultsPolars; ``model_construct`` skips
    # per-outage validation. LoadflowResultsPolars is built around LazyFrames, so the eager
    # result frames are made lazy here. Everything is concatenated in polars and converted to
    # pandas once, at the end of the run (see run_contingency_analysis_pandapower).
    return LoadflowResultsPolars.model_construct(
        job_id=ctx.job_id,
        branch_results=element_results.branch_results.lazy(),
        node_results=element_results.node_results.lazy(),
        converged=convergence_df.lazy(),
        regulating_element_results=element_results.regulating_element_results.lazy(),
        va_diff_results=element_results.va_diff_results.lazy(),
        switch_results=element_results.switch_results.lazy(),
        warnings=[],
        spps_results=spps_results.lazy(),
        cascade_results=cascade_results.lazy(),
    )


def _build_spps_results(
    spps_result: SppsResult,
    contingencies: list[PandapowerContingency],
    timestep: int,
) -> pl.DataFrame:
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


def _build_convergence_results(
    grouped_contingency: PandapowerContingencyGroup,
    timestep: int,
    status: ConvergenceStatus,
) -> pl.DataFrame:
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

    _update_result_names(results=results, element_name_map=ctx.result_constants.element_name_map)

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
    element_name_map: dict[str, str],
) -> None:
    # polars frames are immutable, so reassign the filled result back onto the dataclass.
    results.branch_results = update_results_with_names(results.branch_results, element_name_map)
    results.node_results = update_results_with_names(results.node_results, element_name_map)
    results.va_diff_results = update_results_with_names(results.va_diff_results, element_name_map)
    results.regulating_element_results = update_results_with_names(results.regulating_element_results, element_name_map)
    results.switch_results = update_results_with_names(results.switch_results, element_name_map)


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
        deepcopy(net),
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


def _build_cascade_results_df(
    cascade_events: list[Any],
    contingencies: list[PandapowerContingency],
    contingency_outage_id: str,
    timestep: int,
) -> pl.DataFrame:
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
    )


def _should_run_cascade(
    ctx: SingleOutageContext,
    status: ConvergenceStatus,
) -> bool:
    return ctx.cascade is not None and status == ConvergenceStatus.CONVERGED


def update_results_with_names(
    df: pl.DataFrame,
    element_name_map: dict[str, str],
) -> pl.DataFrame:
    """
    Enrich results DataFrame with element names (flat polars frame with an ``element`` column).

    This function fills missing values in the `element_name` column using a
    provided mapping from element indices to human-readable names.

    Args:
        df: Results DataFrame. Expected to have:
            - a MultiIndex containing level `"element"`
            - a column `"element_name"`
        element_name_map: Mapping from element index (as found in the `"element"`
            index level) to element name.

    Returns
    -------
        Updated DataFrame (same object, modified in-place).

    Notes
    -----
        - Only missing or empty `element_name` values are filled.
        - If an element is not found in `element_name_map`, the value falls back to an empty string.
    """
    # Fill only missing/empty names from the map (unmapped elements fall back to "").
    mapped = pl.col("element").replace_strict(element_name_map, default="", return_dtype=pl.String)
    return df.with_columns(
        pl.when((pl.col("element_name").is_null()) | (pl.col("element_name") == ""))
        .then(mapped)
        .otherwise(pl.col("element_name"))
        .alias("element_name")
    )


def get_element_results_df(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    timestep: int,
    status: ConvergenceStatus,
    result_constants: ResultConstants,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
        full_branch_results = get_branch_results_polars(net, contingency, timestep, result_constants)
        node_results = get_node_results_polars(net, contingency, timestep, result_constants)
        va_diff_results = get_va_diff_results(net, timestep, contingency, result_constants)

        # IMPORTANT:
        # Do NOT filter branch/node results before this step.
        # Switch result calculation depends on connectivity and may require data
        # from non-monitored branches/nodes (e.g. a monitored switch connected to
        # an unmonitored line/trafo). Therefore we pass full result sets here.
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


def run_contingency_analysis_sequential(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: SequentialContingencyAnalysisContext,
) -> list[LoadflowResultsPolars]:
    """Compute a full N-1 analysis for the given network for a single timestep.

    Iterates over every contingency group, deep-copies the network, and calls
    :func:`run_single_outage`.  ``ctx.slack_allocation_config`` is forwarded to
    each outage call so that :func:`run_outage_power_flow` can handle slack-bus
    assignment (initial PF and SpPS in-loop reassignment) internally.
    """
    results = []

    single_outage_ctx = SingleOutageContext(
        monitored_elements=n_minus_1_definition.monitored_elements,
        timestep=ctx.timestep,
        job_id=ctx.job_id,
        method=ctx.method,
        runpp_kwargs=ctx.runpp_kwargs,
        basecase_net=ctx.basecase_net,
        switch_element_mapping=ctx.switch_element_mapping,
        # Element ids, rated currents, base-case voltages, the polars switch mapping and the
        # element-name map are the same for every outage in this run, so resolve them once here.
        result_constants=ResultConstants.from_network(
            net,
            ctx.basecase_net,
            monitored_elements=n_minus_1_definition.monitored_elements,
            switch_element_mapping=ctx.switch_element_mapping,
        ),
        spps=SingleOutageSppsContext(
            conditions=ctx.spps_conditions,
            actions=ctx.spps_actions,
            rules_max_iterations=ctx.spps_rules_max_iterations,
            on_power_flow_error=ctx.on_power_flow_error,
        ),
        cascade=ctx.cascade,
        bus_couplers_mrids=ctx.bus_couplers_mrids,
    )

    for grouped_contingency in n_minus_1_definition.grouped_contingencies:
        copy_net = deepcopy(net)

        single_res = run_single_outage(
            net=copy_net,
            grouped_contingency=grouped_contingency,
            ctx=single_outage_ctx,
            slack_allocation_config=ctx.slack_allocation_config,
        )

        results.append(single_res)

    return results


def run_contingency_analysis_parallel(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: ParallelContingencyAnalysisContext,
) -> list[LoadflowResultsPolars]:
    """Compute the N-1 AC/DC power flow for the network in parallel."""
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

    handles = []
    result_lists = []
    _compute_remote = ray.remote(run_contingency_analysis_sequential)

    sequential_ctx = SequentialContingencyAnalysisContext(
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
    )

    for batch in work:
        handles.append(
            _compute_remote.remote(
                net=net,
                n_minus_1_definition=batch,
                ctx=sequential_ctx,
            )
        )

        if len(handles) >= ctx.parallel.n_processes:
            finished, handles = ray.wait(handles, num_returns=1)
            result_lists.extend(ray.get(finished))

    result_lists.extend(ray.get(handles))

    return [result for result_list in result_lists for result in result_list]


def _run_base_case_loadflow(
    net: pp.pandapowerNet,
    slack_allocation_config: SlackAllocationConfig,
    cfg: ContingencyAnalysisConfig,
) -> None:
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
    """
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

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the loadflow computation
    """
    pp_n1_definition = translate_nminus1_for_pandapower(n_minus_1_definition, net)
    if cfg.apply_outage_grouping:
        pp_n1_definition.grouped_contingencies = get_outage_group_for_contingency(
            net=net,
            contingencies=pp_n1_definition.contingencies,
        )
    else:
        pp_n1_definition.grouped_contingencies = [
            PandapowerContingencyGroup(contingencies=[cont], elements=cont.elements, outage_group_id=str(uuid.uuid4()))
            for cont in pp_n1_definition.contingencies
        ]

    slack_allocation_config = SlackAllocationConfig(
        min_island_size=cfg.min_island_size,
    )

    _run_base_case_loadflow(
        net=net,
        cfg=cfg,
        slack_allocation_config=slack_allocation_config,
    )

    switch_element_mapping = get_switch_mapped_elements(
        net=net,
        monitored_elements=pp_n1_definition.monitored_elements,
        side="bus",
    )

    # Cascade run-invariants: convert sw_characteristics once and precompute the
    # base-case busbar-coupler set, so neither is redone per outage. Skipped when
    # cascade screening is disabled.
    bus_couplers_mrids: set[str] = prepare_cascade_run_constants(net) if cfg.cascade is not None else set()

    if cfg.parallel.n_processes == 1 and cfg.parallel.batch_size is None:
        results = run_contingency_analysis_sequential(
            net=net,
            n_minus_1_definition=pp_n1_definition,
            ctx=SequentialContingencyAnalysisContext(
                job_id=job_id,
                timestep=timestep,
                slack_allocation_config=slack_allocation_config,
                method=cfg.method,
                runpp_kwargs=cfg.runpp_kwargs,
                basecase_net=deepcopy(net),
                switch_element_mapping=switch_element_mapping,
                spps_conditions=pp_n1_definition.spps_conditions,
                spps_actions=pp_n1_definition.spps_actions,
                spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                on_power_flow_error=cfg.on_power_flow_error,
                cascade=cfg.cascade,
                bus_couplers_mrids=bus_couplers_mrids,
            ),
        )
    else:
        results = run_contingency_analysis_parallel(
            net=net,
            n_minus_1_definition=pp_n1_definition,
            ctx=ParallelContingencyAnalysisContext(
                job_id=job_id,
                timestep=timestep,
                slack_allocation_config=slack_allocation_config,
                basecase_net=deepcopy(net),
                switch_element_mapping=switch_element_mapping,
                spps_conditions=pp_n1_definition.spps_conditions,
                spps_actions=pp_n1_definition.spps_actions,
                method=cfg.method,
                runpp_kwargs=cfg.runpp_kwargs,
                spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                on_power_flow_error=cfg.on_power_flow_error,
                parallel=cfg.parallel,
                cascade=cfg.cascade,
                bus_couplers_mrids=bus_couplers_mrids,
            ),
        )
    # Per-outage results are polars; concatenate in polars and convert to pandas once at the
    # very end (only when the caller wants pandas).
    lf_result = concatenate_loadflow_results_polars(results)

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

    if cfg.apply_outage_grouping:
        lf_result.connectivity_result = pl.from_pandas(
            build_connectivity_df(pp_n1_definition.grouped_contingencies).reset_index()
        ).lazy()

    if cfg.polars:
        return lf_result
    return convert_polars_loadflow_results_to_pandas(lf_result)
