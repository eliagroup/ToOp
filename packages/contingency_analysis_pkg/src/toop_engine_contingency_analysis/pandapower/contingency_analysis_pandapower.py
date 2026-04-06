# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compute the N-1 AC/DC power flow for the pandapower network."""

import math
import uuid
from copy import deepcopy

import pandapower as pp
import pandapower.topology as top
import pandas as pd
import pandera as pa
import pandera.typing as pat
import ray
from beartype.typing import Literal, Optional, Union
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerContingencyGroup,
    PandapowerElements,
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
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import ContingencyAnalysisConfig
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.loadflow_result_helpers import (
    concatenate_loadflow_results,
    convert_pandas_loadflow_results_to_polars,
    get_failed_branch_results,
    get_failed_node_results,
)
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    ConnectivityResultSchema,
    ConvergenceStatus,
    LoadflowResults,
    NodeResultSchema,
    SwitchResultsSchema,
    VADiffResultSchema,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


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


@pa.check_types
def run_single_outage(
    net: pp.pandapowerNet,
    grouped_contingency: PandapowerContingencyGroup,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
    job_id: str,
    basecase_voltage: pat.Series[float],
    method: Literal["ac", "dc"],
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    runpp_kwargs: Optional[dict] = None,
) -> LoadflowResults:
    """Compute a single outage for the given network

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the outage for
    grouped_contingency: PandapowerContingencyGroup,
        The grouped contingency to compute the outage for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
        The elements to monitor during the outage
    timestep : int
        The timestep of the results
    job_id : str
        The job id of the current job

    basecase_voltage: pat.Series[float]
        The voltage results from the basecase run.
        Contains computed voltages if the basecase converged,
        otherwise a series of NaN values.
    method : Literal["ac", "dc"]
        The method to use for the loadflow. Either "ac" or "dc"
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements, used to compute
        switch-level results during each outage.
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None


    Returns
    -------
    LoadflowResults
        The results of the ContingencyAnalysis computation
    """
    outaged_elements = grouped_contingency.elements

    opened_cb_indices = open_outaged_circuit_breakers(net, outaged_elements)

    were_in_service = set_outaged_elements_out_of_service(net, outaged_elements)
    if not any(were_in_service):
        # If no elements were outaged, this is the base case and we should not run the loadflow
        status = ConvergenceStatus.NO_CALCULATION
    else:
        runpp_kwargs = runpp_kwargs or {}
        try:
            pp.rundcpp(net, **runpp_kwargs) if method == "dc" else pp.runpp(net, **runpp_kwargs)
            status = ConvergenceStatus.CONVERGED
        except pp.LoadflowNotConverged:
            status = ConvergenceStatus.FAILED

    convergence_df = pd.concat(
        [
            get_convergence_df(
                timestep=timestep,
                contingency=contingency,
                status=status.value,
            )
            for contingency in grouped_contingency.contingencies
        ],
    )

    branch_dfs = []
    node_dfs = []
    va_diff_dfs = []
    regulating_elements_dfs = []
    switch_dfs = []
    element_name_map = monitored_elements["name"].to_dict()
    first_contingency = grouped_contingency.contingencies[0]
    branch_results_df, node_results_df, va_diff_results, sw_results_df = get_element_results_df(
        net, first_contingency, monitored_elements, timestep, status, basecase_voltage, switch_element_mapping
    )
    reg_element_result = get_regulating_element_results(timestep, monitored_elements, first_contingency)

    for contingency in grouped_contingency.contingencies:
        branch_dfs.append(_apply_contingency_to_index(branch_results_df, contingency))
        node_dfs.append(_apply_contingency_to_index(node_results_df, contingency))
        va_diff_dfs.append(_apply_contingency_to_index(va_diff_results, contingency))
        regulating_elements_dfs.append(_apply_contingency_to_index(reg_element_result, contingency))
        switch_dfs.append(_apply_contingency_to_index(sw_results_df, contingency))

    branch_results_df = pd.concat(branch_dfs) if branch_dfs else pd.DataFrame()
    node_results_df = pd.concat(node_dfs) if node_dfs else pd.DataFrame()
    va_diff_results = pd.concat(va_diff_dfs) if va_diff_dfs else pd.DataFrame()
    regulating_elements_df = pd.concat(regulating_elements_dfs) if regulating_elements_dfs else pd.DataFrame()
    switch_df = pd.concat(switch_dfs) if switch_dfs else pd.DataFrame()

    update_results_with_names(branch_results_df, element_name_map)
    update_results_with_names(node_results_df, element_name_map)
    update_results_with_names(va_diff_results, element_name_map)
    update_results_with_names(regulating_elements_df, element_name_map)
    update_results_with_names(switch_df, element_name_map)

    restore_outaged_circuit_breakers(net, opened_cb_indices)
    restore_elements_to_service(net, outaged_elements, were_in_service)

    lf_result = LoadflowResults(
        job_id=job_id,
        branch_results=branch_results_df,
        node_results=node_results_df,
        converged=convergence_df,
        regulating_element_results=regulating_elements_df,
        va_diff_results=va_diff_results,
        switch_results=switch_df,
        warnings=[],
    )
    return lf_result


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


def restore_elements_to_service(
    net: pp.pandapowerNet, outaged_elements: list[PandapowerElements], were_in_service: list[bool]
) -> None:
    """Restore the outaged elements to their original in_service status.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to restore the elements in
    outaged_elements : list[PandapowerElements]
        The elements that were outaged
    were_in_service : list[bool]
        A list indicating whether each element was in service before being set out of service
    """
    for i, element in enumerate(outaged_elements):
        if were_in_service[i]:
            net[element.table].loc[int(element.table_id), "in_service"] = True


def restore_outaged_circuit_breakers(net: pp.pandapowerNet, opened_cb_indices: list[int]) -> None:
    """
    Restore previously opened circuit breakers (CBs) to service.

    This function closes the switches (circuit breakers) that were previously
    opened to isolate an outage area.

    Args:
        net: pandapower network object.
        opened_cb_indices: Indices of switches (typically returned by
            `set_outaged_elements_out_of_service`) to be closed.

    Notes
    -----
        - Non-existent indices are ignored.
        - Only switches currently open (`closed == False`) are modified.
    """
    net.switch.loc[opened_cb_indices, "closed"] = True


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
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The branch results dataframe, node results dataframe and va diff results dataframe
    """
    if status == ConvergenceStatus.CONVERGED:
        branch_results_df = get_branch_results(net, contingency, monitored_elements, timestep)
        node_results_df = get_node_result_df(net, contingency, monitored_elements, timestep, basecase_voltage)
        va_diff_results = get_va_diff_results(net, timestep, monitored_elements, contingency)
        sw_results_df = get_switch_results(
            net, contingency, timestep, branch_results_df, node_results_df, switch_element_mapping
        )
    else:
        monitored_trafo3w = monitored_elements.query("table == 'trafo3w'").index.to_list()
        monitored_branches = monitored_elements.query("kind == 'branch' & table != 'trafo3w'").index.to_list()
        monitored_buses = monitored_elements.query("kind == 'bus'").index.to_list()
        branch_results_df = get_failed_branch_results(
            timestep, [contingency.unique_id], monitored_branches, monitored_trafo3w
        )
        node_results_df = get_failed_node_results(timestep, [contingency.unique_id], monitored_buses)
        va_diff_results = get_failed_va_diff_results(timestep, monitored_elements, contingency)
        sw_results_df = get_failed_switch_results(timestep, switch_element_mapping, contingency)
    return branch_results_df, node_results_df, va_diff_results, sw_results_df


def set_outaged_elements_out_of_service(net: pp.pandapowerNet, outaged_elements: list[PandapowerElements]) -> list[bool]:
    """Set the outaged elements in the network to out of service.

    Returns info if the elements were in service before being set out of service.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to set the elements out of service in
    outaged_elements : list[PandapowerElements]
        The elements to set out of service

    Returns
    -------
    list[bool]
        A list indicating whether each element was in service before being set out of service
    """
    were_in_service = []
    if len(outaged_elements) == 0:
        # This is the base case. Append a dummy True so it does not raise due to no elements being outaged
        were_in_service.append(True)
    else:
        for element in outaged_elements:
            was_in_service = net[element.table].loc[element.table_id, "in_service"]
            were_in_service.append(bool(was_in_service))
            net[element.table].loc[element.table_id, "in_service"] = False
    return were_in_service


def open_outaged_circuit_breakers(net: pp.pandapowerNet, outaged_elements: list[PandapowerElements]) -> list[int]:
    """
    Isolate outaged buses by opening boundary circuit breakers (CBs).

    The function identifies all buses marked as outaged and finds circuit breakers
    (`type == "CB"`) that form the electrical boundary between the outaged area and
    the rest of the network. These breakers are detected as switches connected to
    outaged buses (via either the `bus` or `element` column) that are currently closed.

    All such breakers are opened (`closed = False`) to electrically isolate the
    outaged portion of the network from the healthy grid.

    Args:
        net: pandapower network object.
        outaged_elements: List of outage descriptors. Only elements with
            `table == "bus"` are considered. Bus indices are parsed from `unique_id`.

    Returns
    -------
        List of switch indices that were opened to isolate the outaged area.

    Notes
    -----
        - Only closed circuit breakers (`type == "CB"`) are affected.
        - The function assumes that opening all CBs connected to outaged buses
          effectively isolates the outage region (i.e., CBs represent boundary points).
    """
    outaged_bus_ids = [int(element.unique_id.split("%%", 1)[0]) for element in outaged_elements if element.table == "bus"]
    if not outaged_bus_ids:
        return []

    cb_mask = (
        (net.switch["type"] == "CB")
        & net.switch["closed"]
        & (net.switch["bus"].isin(outaged_bus_ids) | net.switch["element"].isin(outaged_bus_ids))
    )

    affected_switches = net.switch.index[cb_mask].tolist()
    net.switch.loc[affected_switches, "closed"] = False
    return affected_switches


def run_contingency_analysis_sequential(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    job_id: str,
    timestep: int,
    basecase_voltage: pat.Series[float],
    slack_allocation_config: SlackAllocationConfig,
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    method: Literal["ac", "dc"] = "dc",
    runpp_kwargs: Optional[dict] = None,
) -> list[LoadflowResults]:
    """Compute a full N-1 analysis for the given network, but a single timestep

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the N-1 analysis for
    n_minus_1_definition: PandapowerNMinus1Definition,
        The N-1 definition to use for the analysis
    job_id : str
        The job id of the current job
    timestep : int
        The timestep of the results
    slack_allocation_config : SlackAllocationConfig
        Precomputed configuration for slack allocation per island.
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements, used to compute
        switch-level results during each outage.
    method : Literal["ac", "dc"], optional
        The method to use for the loadflow, by default "dc"
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None
    basecase_voltage: pat.Series[float]
        The voltage results from the basecase run.
        Contains computed voltages if the basecase converged,
        otherwise a series of NaN values.

    Returns
    -------
    list[LoadflowResults]
        A list of the results per contingency
    """
    results = []

    for grouped_contingency in n_minus_1_definition.grouped_contingencies:
        copy_net = deepcopy(net)
        elements_ids = [element.unique_id for element in grouped_contingency.elements]
        removed_edges = assign_slack_per_island(
            net=copy_net,
            net_graph=slack_allocation_config.net_graph,
            bus_lookup=slack_allocation_config.bus_lookup,
            elements_ids=elements_ids,
            min_island_size=slack_allocation_config.min_island_size,
        )

        single_res = run_single_outage(
            net=copy_net,
            grouped_contingency=grouped_contingency,
            monitored_elements=n_minus_1_definition.monitored_elements,
            timestep=timestep,
            job_id=job_id,
            method=method,
            runpp_kwargs=runpp_kwargs,
            basecase_voltage=basecase_voltage,
            switch_element_mapping=switch_element_mapping,
        )

        results.append(single_res)
        slack_allocation_config.net_graph.add_edges_from(removed_edges)

    return results


def run_contingency_analysis_parallel(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    job_id: str,
    timestep: int,
    slack_allocation_config: SlackAllocationConfig,
    basecase_voltage: pat.Series[float],
    switch_element_mapping: pat.DataFrame[SwitchElementMappingSchema],
    cfg: ContingencyAnalysisConfig,
) -> list[LoadflowResults]:
    """Compute the N-1 AC/DC power flow for the network.

    Parameters
    ----------
    net: pp.pandapowerNet,
        The pandapower network to compute the N-1 power flow for, with the topolgy already applied.
        You can either pass the network directly or a ray.ObjectRef to the network (wrapped in a
        list to avoid dereferencing the object).
    n_minus_1_definition: PandapowerNMinus1Definition,
        The N-1 definition to use for the analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep of the results
    slack_allocation_config : SlackAllocationConfig
        Precomputed configuration for slack allocation per island.
    cfg : ContingencyAnalysisConfig
        Execution configuration (method, islanding/slack settings, parallelization, etc.).
    basecase_voltage: pat.Series[float]
        The basecase voltage results
    switch_element_mapping : pat.DataFrame[SwitchElementMappingSchema]
        Mapping between switches and connected elements, used to compute
        switch-level results during each outage.

    Returns
    -------
    list[LoadflowResults]
        A list of the results per contingency
    """
    n_outages = len(n_minus_1_definition.grouped_contingencies)
    batch_size = cfg.parallel.batch_size
    if batch_size is None:
        batch_size = math.ceil(n_outages / cfg.parallel.n_processes)
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

    # Schedule work until the handle list is too long, then wait for the first result and continue
    handles = []
    result_lists = []
    _compute_remote = ray.remote(run_contingency_analysis_sequential)
    for batch in work:
        handles.append(
            _compute_remote.remote(
                net=net,
                n_minus_1_definition=batch,
                job_id=job_id,
                timestep=timestep,
                slack_allocation_config=slack_allocation_config,
                method=cfg.method,
                runpp_kwargs=cfg.runpp_kwargs,
                basecase_voltage=basecase_voltage,
                switch_element_mapping=switch_element_mapping,
            )
        )
        if len(handles) >= cfg.parallel.n_processes:
            # Wait for the first result and continue
            finished, handles = ray.wait(handles, num_returns=1)
            result_lists.extend(ray.get(finished))
    result_lists.extend(ray.get(handles))

    # Sort the result_lists back into the original order
    del handles
    # Flatten the result_lists
    results = [result for result_list in result_lists for result in result_list]
    return results


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
        Execution configuration (method, islanding/slack settings, parallelization, etc.).

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
            job_id=job_id,
            timestep=timestep,
            slack_allocation_config=slack_allocation_config,
            method=cfg.method,
            runpp_kwargs=cfg.runpp_kwargs,
            basecase_voltage=net.res_bus.vm_pu.copy(),
            switch_element_mapping=switch_element_mapping,
        )
    else:
        results = run_contingency_analysis_parallel(
            net=net,
            n_minus_1_definition=pp_n1_definition,
            job_id=job_id,
            timestep=timestep,
            slack_allocation_config=slack_allocation_config,
            cfg=cfg,
            basecase_voltage=net.res_bus.vm_pu.copy(),
            switch_element_mapping=switch_element_mapping,
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
