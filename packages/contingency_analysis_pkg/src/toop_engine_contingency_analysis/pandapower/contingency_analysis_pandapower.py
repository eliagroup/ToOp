# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compute the N-1 AC/DC power flow for the pandapower network."""

import json
import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum

import pandapower as pp
import pandapower.topology as top
import pandas as pd
import pandera as pa
import pandera.typing as pat
import ray
from beartype.typing import Any, Optional, Union
from toop_engine_contingency_analysis.pandapower.cascade.simulation import (
    CascadeSimulator,
)
from toop_engine_contingency_analysis.pandapower.outage_power_flow import run_outage_power_flow
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerContingencyGroup,
    PandapowerMonitoredElementSchema,
    PandapowerNMinus1Definition,
    SlackAllocationConfig,
    get_branch_results,
    get_convergence_df,
    get_failed_va_diff_results,
    get_node_result_df,
    get_regulating_element_results,
    get_switch_results,
    get_va_diff_results,
    translate_nminus1_for_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.contingency_outage_group import (
    get_outage_group_for_contingency,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.results.switch_results import (
    SwitchElementMappingSchema,
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
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_result_helpers import (
    concatenate_loadflow_results,
    convert_pandas_loadflow_results_to_polars,
    get_failed_branch_results,
    get_failed_node_results,
)
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    CascadeResultSchema,
    ConnectivityResultSchema,
    ConvergenceStatus,
    LoadflowResults,
    NodeResultSchema,
    SppsResultsSchema,
    SwitchResultsSchema,
    VADiffResultSchema,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


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


def _apply_contingency_to_index(df: pd.DataFrame, contingency: PandapowerContingency) -> pd.DataFrame:
    """
    Apply a contingency to a DataFrame by updating its MultiIndex level and adding a corresponding column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a MultiIndex that includes a "contingency" level.
    contingency : object
        Contingency object with attributes:
        - unique_id : value to assign to the "contingency" index level
        - name : value to assign to the "contingency_name" column

    Returns
    -------
    pd.DataFrame
        A new DataFrame with updated MultiIndex and added "contingency_name" column.
    """
    df_copy = df.copy()

    idx = df_copy.index.to_frame(index=False)
    idx["contingency"] = contingency.unique_id
    df_copy.index = pd.MultiIndex.from_frame(idx)

    df_copy["contingency_name"] = contingency.name
    return df_copy


@dataclass
class OutageElementResults:
    """Result tables collected for one outage calculation."""

    branch_results: pd.DataFrame
    full_branch_results: pd.DataFrame
    node_results: pd.DataFrame
    va_diff_results: pd.DataFrame
    regulating_element_results: pd.DataFrame
    switch_results: pd.DataFrame


@pa.check_types
def run_single_outage(
    net: pp.pandapowerNet,
    grouped_contingency: PandapowerContingencyGroup,
    ctx: SingleOutageContext,
) -> LoadflowResults:
    """Compute a single outage for the given network."""
    outaged_elements = grouped_contingency.elements

    status, spps_result = run_outage_power_flow(
        net=net,
        spps=ctx.spps,
        method=ctx.method,
        outaged_elements=outaged_elements,
        runpp_kwargs=ctx.runpp_kwargs,
    )

    spps_results = (
        _build_spps_results(
            spps_result=spps_result,
            contingencies=grouped_contingency.contingencies,
            timestep=ctx.timestep,
        )
        if spps_result is not None
        else get_empty_dataframe_from_model(SppsResultsSchema)
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
        branch_results_df=element_results.full_branch_results,
        switch_results_df=element_results.switch_results,
    )

    return LoadflowResults(
        job_id=ctx.job_id,
        branch_results=element_results.branch_results,
        node_results=element_results.node_results,
        converged=convergence_df,
        regulating_element_results=element_results.regulating_element_results,
        va_diff_results=element_results.va_diff_results,
        switch_results=element_results.switch_results,
        warnings=[],
        spps_results=spps_results,
        cascade_results=cascade_results,
    )


def _build_spps_results(
    spps_result: SppsResult,
    contingencies: list[PandapowerContingency],
    timestep: int,
) -> pd.DataFrame:
    if not contingencies:
        return get_empty_dataframe_from_model(SppsResultsSchema)

    rows = [
        {
            "timestep": timestep,
            "contingency": contingency.unique_id,
            "iterations": spps_result.iterations,
            "activated_schemes_per_iter": json.dumps(
                spps_result.activated_schemes_per_iter,
            ),
            "max_iterations_reached": spps_result.max_iterations_reached,
            "power_flow_failed": spps_result.power_flow_failed,
        }
        for contingency in contingencies
    ]

    return pd.DataFrame(rows).set_index(["timestep", "contingency"])


def _build_convergence_results(
    grouped_contingency: PandapowerContingencyGroup,
    timestep: int,
    status: ConvergenceStatus,
) -> pd.DataFrame:
    return pd.concat(
        [
            get_convergence_df(
                timestep=timestep,
                contingency=contingency,
                status=status.value,
            )
            for contingency in grouped_contingency.contingencies
        ],
    )


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
        ctx.monitored_elements,
        ctx.timestep,
        status,
        ctx.basecase_voltage,
        ctx.switch_element_mapping,
    )

    regulating_element_results_df = get_regulating_element_results(
        ctx.timestep,
        ctx.monitored_elements,
        first_contingency,
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

    _update_result_names(
        results=results,
        monitored_elements=ctx.monitored_elements,
    )

    return results


def _copy_results_for_all_contingencies(
    result_df: pd.DataFrame,
    grouped_contingency: PandapowerContingencyGroup,
) -> pd.DataFrame:
    result_dfs = [_apply_contingency_to_index(result_df, contingency) for contingency in grouped_contingency.contingencies]

    if not result_dfs:
        return pd.DataFrame()

    return pd.concat(result_dfs)


def _update_result_names(
    results: OutageElementResults,
    monitored_elements: pd.DataFrame,
) -> None:
    element_name_map = monitored_elements["name"].to_dict()

    update_results_with_names(results.branch_results, element_name_map)
    update_results_with_names(results.node_results, element_name_map)
    update_results_with_names(results.va_diff_results, element_name_map)
    update_results_with_names(results.regulating_element_results, element_name_map)
    update_results_with_names(results.switch_results, element_name_map)


def _collect_cascade_results(
    net: pp.pandapowerNet,
    ctx: SingleOutageContext,
    grouped_contingency: PandapowerContingencyGroup,
    status: ConvergenceStatus,
    branch_results_df: pd.DataFrame,
    switch_results_df: pd.DataFrame,
) -> pat.DataFrame[CascadeResultSchema]:
    """Build cascade result rows for :attr:`LoadflowResults.cascade_results`.

    Each row describes one cascade event generated after the initial
    contingency load flow.
    """
    if not _should_run_cascade(
        ctx=ctx,
        status=status,
    ):
        return get_empty_dataframe_from_model(CascadeResultSchema)

    simulator = CascadeSimulator(
        ctx.cascade,
        ctx.spps,
        method=ctx.method,
        runpp_kwargs=ctx.runpp_kwargs,
    )

    cascade_events = simulator.simulate(
        deepcopy(net),
        branch_results_df,
        switch_results_df,
        initial_contingency=grouped_contingency.contingencies[0],
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
) -> pat.DataFrame[CascadeResultSchema]:
    if not cascade_events or not contingencies:
        return get_empty_dataframe_from_model(CascadeResultSchema)

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

    cascade_results = pd.DataFrame(rows)
    cascade_results["loading"] = pd.to_numeric(cascade_results["loading"], errors="coerce")
    cascade_results["r_ohm"] = pd.to_numeric(cascade_results["r_ohm"], errors="coerce")
    cascade_results["x_ohm"] = pd.to_numeric(cascade_results["x_ohm"], errors="coerce")
    return cascade_results.set_index(["timestep", "contingency", "cascade_number", "element_mrid"])


def _should_run_cascade(
    ctx: SingleOutageContext,
    status: ConvergenceStatus,
) -> bool:
    return ctx.cascade is not None and status == ConvergenceStatus.CONVERGED


def update_results_with_names(
    df: pd.DataFrame,
    element_name_map: dict[str, str],
) -> pd.DataFrame:
    """
    Enrich results DataFrame with element names.

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
        - If an element is not found in `element_name_map`, the value remains NaN.
    """
    no_name_yet = (df["element_name"] == "") | (df["element_name"].isna())
    df.loc[no_name_yet, "element_name"] = df.loc[no_name_yet].index.get_level_values("element").map(element_name_map)
    return df


def get_element_results_df(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
    status: ConvergenceStatus,
    basecase_voltage: pat.Series[float],
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
) -> tuple[
    pat.DataFrame[BranchResultSchema],
    pat.DataFrame[BranchResultSchema],
    pat.DataFrame[NodeResultSchema],
    pat.DataFrame[VADiffResultSchema],
    pat.DataFrame[SwitchResultsSchema],
]:
    """Get the element results dataframes for the given contingency and monitored elements.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to get the results from
    contingency : PandapowerContingency
        The contingency to get the results for
    monitored_elements : pat.DataFrame[PandapowerMonitoredElementSchema]
        The monitored elements to get the results for
    timestep : int
        The timestep of the results
    status : ConvergenceStatus
        The convergence status of the loadflow computation
    basecase_voltage: pat.Series[float]
        The voltage results from the basecase run.
        Contains computed voltages if the basecase converged,
        otherwise a series of NaN values.
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements, used to compute
        switch-level results during each outage.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Filtered branch results, full branch results, node results, voltage-angle
        difference results, and switch results.
    """
    if status == ConvergenceStatus.CONVERGED:
        full_branch_results_df = get_branch_results(net, contingency, timestep)
        node_results_df = get_node_result_df(net, contingency, timestep, basecase_voltage)
        va_diff_results = get_va_diff_results(net, timestep, monitored_elements, contingency)
        # IMPORTANT:
        # Do NOT filter branch/node results before this step.
        # Switch result calculation depends on connectivity and may require data
        # from non-monitored branches/nodes (e.g. a monitored switch connected to
        # an unmonitored line/trafo). Therefore we pass full result sets here.
        sw_results_df = get_switch_results(
            net, contingency, timestep, full_branch_results_df, node_results_df, switch_element_mapping
        )

        branch_results_df = full_branch_results_df[
            full_branch_results_df.index.isin(monitored_elements.index, level="element")
        ]
        node_results_df = node_results_df[node_results_df.index.isin(monitored_elements.index, level="element")]

    else:
        monitored_trafo3w = monitored_elements.query("table == 'trafo3w'").index.to_list()
        monitored_branches = monitored_elements.query("kind == 'branch' & table != 'trafo3w'").index.to_list()
        monitored_buses = monitored_elements.query("kind == 'bus'").index.to_list()
        branch_results_df = get_failed_branch_results(
            timestep, [contingency.unique_id], monitored_branches, monitored_trafo3w
        )
        full_branch_results_df = branch_results_df
        node_results_df = get_failed_node_results(timestep, [contingency.unique_id], monitored_buses)
        va_diff_results = get_failed_va_diff_results(timestep, monitored_elements, contingency)
        sw_results_df = get_failed_switch_results(timestep, switch_element_mapping, contingency)
    return branch_results_df, full_branch_results_df, node_results_df, va_diff_results, sw_results_df


def run_contingency_analysis_sequential(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: SequentialContingencyAnalysisContext,
) -> list[LoadflowResults]:
    """Compute a full N-1 analysis for the given network, but a single timestep."""
    results = []

    single_outage_ctx = SingleOutageContext(
        monitored_elements=n_minus_1_definition.monitored_elements,
        timestep=ctx.timestep,
        job_id=ctx.job_id,
        method=ctx.method,
        runpp_kwargs=ctx.runpp_kwargs,
        basecase_voltage=ctx.basecase_voltage,
        switch_element_mapping=ctx.switch_element_mapping,
        spps=SingleOutageSppsContext(
            conditions=ctx.spps_conditions,
            actions=ctx.spps_actions,
            rules_max_iterations=ctx.spps_rules_max_iterations,
            on_power_flow_error=ctx.on_power_flow_error,
        ),
        cascade=ctx.cascade,
    )

    for grouped_contingency in n_minus_1_definition.grouped_contingencies:
        copy_net = deepcopy(net)
        elements_ids = [element.unique_id for element in grouped_contingency.elements]

        removed_edges = assign_slack_per_island(
            net=copy_net,
            net_graph=ctx.slack_allocation_config.net_graph,
            bus_lookup=ctx.slack_allocation_config.bus_lookup,
            elements_ids=elements_ids,
            min_island_size=ctx.slack_allocation_config.min_island_size,
        )

        single_res = run_single_outage(
            net=copy_net,
            grouped_contingency=grouped_contingency,
            ctx=single_outage_ctx,
        )

        results.append(single_res)
        ctx.slack_allocation_config.net_graph.add_edges_from(removed_edges)

    return results


def run_contingency_analysis_parallel(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    ctx: ParallelContingencyAnalysisContext,
) -> list[LoadflowResults]:
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
        basecase_voltage=ctx.basecase_voltage,
        switch_element_mapping=ctx.switch_element_mapping,
        spps_conditions=ctx.spps_conditions,
        spps_actions=ctx.spps_actions,
        spps_rules_max_iterations=ctx.spps_rules_max_iterations,
        on_power_flow_error=ctx.on_power_flow_error,
        cascade=ctx.cascade,
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
    base_case: Optional[PandapowerContingency],
    slack_allocation_config: SlackAllocationConfig,
    cfg: ContingencyAnalysisConfig,
) -> None:
    """
    Run load flow calculation for the contingency analysis base case.

    This function performs two main steps:

    1. Assign slack buses for each electrical island based on the
       provided slack allocation configuration.
    2. Execute a power flow calculation (AC or DC) depending on the
       contingency analysis configuration.

    Args:
        net:
            Pandapower network to be modified and solved.
        base_case:
            Contingency definition representing the base system state.
            Its elements are used to determine affected islands.
        slack_allocation_config:
            Configuration used for selecting and assigning slack buses
            per electrical island.
        cfg:
            Global contingency analysis configuration containing
            load-flow method and optional runpp arguments.

    Raises
    ------
        RuntimeError:
            If the base case load flow does not converge.
    """
    elements_ids = []
    if base_case is not None:
        elements_ids = [element.unique_id for element in base_case.elements]
    assign_slack_per_island(
        net=net,
        net_graph=slack_allocation_config.net_graph,
        bus_lookup=slack_allocation_config.bus_lookup,
        elements_ids=elements_ids,
        min_island_size=slack_allocation_config.min_island_size,
    )

    try:
        runpp_kwargs = cfg.runpp_kwargs or {}

        if cfg.method == "dc":
            pp.rundcpp(net, **runpp_kwargs)
        else:
            pp.runpp(net, **runpp_kwargs)

    except pp.LoadflowNotConverged:
        pass


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

    net_graph = top.create_nxgraph(net)
    bus_lookup, _ = create_bus_lookup_simple(net)
    slack_allocation_config = SlackAllocationConfig(
        net_graph=net_graph,
        bus_lookup=bus_lookup,
        min_island_size=cfg.min_island_size,
    )

    _run_base_case_loadflow(
        net=net,
        base_case=pp_n1_definition.base_case,
        cfg=cfg,
        slack_allocation_config=slack_allocation_config,
    )

    switch_element_mapping = get_switch_mapped_elements(
        net=net,
        monitored_elements=pp_n1_definition.monitored_elements,
        side="bus",
    )

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
                basecase_voltage=net.res_bus.vm_pu.copy(),
                switch_element_mapping=switch_element_mapping,
                spps_conditions=pp_n1_definition.spps_conditions,
                spps_actions=pp_n1_definition.spps_actions,
                spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                on_power_flow_error=cfg.on_power_flow_error,
                cascade=cfg.cascade,
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
                basecase_voltage=net.res_bus.vm_pu.copy(),
                switch_element_mapping=switch_element_mapping,
                spps_conditions=pp_n1_definition.spps_conditions,
                spps_actions=pp_n1_definition.spps_actions,
                method=cfg.method,
                runpp_kwargs=cfg.runpp_kwargs,
                spps_rules_max_iterations=cfg.spps_rules_max_iterations,
                on_power_flow_error=cfg.on_power_flow_error,
                parallel=cfg.parallel,
                cascade=cfg.cascade,
            ),
        )
    lf_result = concatenate_loadflow_results(results)

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
        lf_result.connectivity_result = build_connectivity_df(pp_n1_definition.grouped_contingencies)

    if not cfg.polars:
        return lf_result
    return convert_pandas_loadflow_results_to_polars(lf_result)
