"""Compute the N-1 AC/DC power flow for the pandapower network."""

import math
from copy import deepcopy

import pandapower as pp
import pandapower.topology as top
import pandera as pa
import pandera.typing as pat
import ray
from beartype.typing import Literal, Optional, Union
from toop_engine_contingency_analysis.pandapower.pandapower_helpers import (
    PandapowerContingency,
    PandapowerMonitoredElementSchema,
    PandapowerNMinus1Definition,
    SlackAllocationConfig,
    get_branch_results,
    get_convergence_df,
    get_failed_va_diff_results,
    get_node_result_df,
    get_regulating_element_results,
    get_va_diff_results,
    translate_nminus1_for_pandapower,
)
from toop_engine_grid_helpers.pandapower.bus_lookup import create_bus_lookup_simple
from toop_engine_grid_helpers.pandapower.slack_allocation import assign_slack_per_island
from toop_engine_interfaces.loadflow_result_helpers import (
    concatenate_loadflow_results,
    convert_pandas_loadflow_results_to_polars,
    get_failed_branch_results,
    get_failed_node_results,
)
from toop_engine_interfaces.loadflow_results import (
    ConvergenceStatus,
    LoadflowResults,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


@pa.check_types
def run_single_outage(
    net: pp.pandapowerNet,
    contingency: PandapowerContingency,
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
    timestep: int,
    job_id: str,
    method: Literal["ac", "dc"],
    runpp_kwargs: Optional[dict] = None,
) -> LoadflowResults:
    """Compute a single outage for the given network

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the outage for
    contingency: PandapowerContingency,
        The contingency to compute the outage for
    monitored_elements: pat.DataFrame[PandapowerMonitoredElementSchema],
        The elements to monitor during the outage
    timestep : int
        The timestep of the results
    job_id : str
        The job id of the current job
    method : Literal["ac", "dc"]
        The method to use for the loadflow. Either "ac" or "dc"
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None

    Returns
    -------
    LoadflowResults
        The results of the ContingencyAnalysis computation
    """
    were_in_service = []
    outaged_elements = contingency.elements
    if len(outaged_elements) == 0:
        # This is the base case. Append a dummy True so it does not raise due to no elements being outaged
        were_in_service.append(True)
    else:
        for element in outaged_elements:
            was_in_service = net[element.table].loc[element.table_id, "in_service"]
            were_in_service.append(was_in_service)
            net[element.table].loc[element.table_id, "in_service"] = False
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

    convergence_df = get_convergence_df(timestep=timestep, contingency=contingency, status=status.value)
    regulating_elements_df = get_regulating_element_results(timestep, monitored_elements, contingency)

    if status == ConvergenceStatus.CONVERGED:
        branch_results_df = get_branch_results(net, contingency, monitored_elements, timestep)
        node_results_df = get_node_result_df(net, contingency, monitored_elements, timestep)
        va_diff_results = get_va_diff_results(net, timestep, monitored_elements, contingency)
    else:
        monitored_trafo3w = monitored_elements.query("table == 'trafo3w'").index.to_list()
        monitored_branches = monitored_elements.query("kind == 'branch' & table != 'trafo3w'").index.to_list()
        monitored_buses = monitored_elements.query("kind == 'bus'").index.to_list()
        branch_results_df = get_failed_branch_results(
            timestep, [contingency.unique_id], monitored_branches, monitored_trafo3w
        )
        node_results_df = get_failed_node_results(timestep, [contingency.unique_id], monitored_buses)
        va_diff_results = get_failed_va_diff_results(timestep, monitored_elements, contingency)

    for i, element in enumerate(outaged_elements):
        if were_in_service[i]:
            net[element.table].loc[int(element.table_id), "in_service"] = True

    element_name_map = monitored_elements["name"].to_dict()
    for df in [branch_results_df, node_results_df, regulating_elements_df, va_diff_results]:
        no_name_yet = df["element_name"] == ""
        df.loc[no_name_yet, "element_name"] = df.loc[no_name_yet].index.get_level_values("element").map(element_name_map)
        df["contingency_name"] = contingency.name
    lf_result = LoadflowResults(
        job_id=job_id,
        branch_results=branch_results_df,
        node_results=node_results_df,
        converged=convergence_df,
        regulating_element_results=regulating_elements_df,
        va_diff_results=va_diff_results,
        warnings=[],
    )
    return lf_result


def run_contingency_analysis_sequential(
    net: pp.pandapowerNet,
    n_minus_1_definition: PandapowerNMinus1Definition,
    job_id: str,
    timestep: int,
    slack_allocation_config: SlackAllocationConfig,
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
    method : Literal["ac", "dc"], optional
        The method to use for the loadflow, by default "dc"
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None

    Returns
    -------
    list[LoadflowResults]
        A list of the results per contingency
    """
    results = []

    for contingency in n_minus_1_definition.contingencies:
        copy_net = deepcopy(net)
        elements_ids = [element.unique_id for element in contingency.elements]
        removed_edges = assign_slack_per_island(
            net=copy_net,
            net_graph=slack_allocation_config.net_graph,
            bus_lookup=slack_allocation_config.bus_lookup,
            elements_ids=elements_ids,
            min_island_size=slack_allocation_config.min_island_size,
        )

        single_res = run_single_outage(
            net=copy_net,
            contingency=contingency,
            monitored_elements=n_minus_1_definition.monitored_elements,
            timestep=timestep,
            job_id=job_id,
            method=method,
            runpp_kwargs=runpp_kwargs,
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
    method: Literal["ac", "dc"] = "dc",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
    runpp_kwargs: Optional[dict] = None,
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
    method : Literal["ac", "dc"], optional
        The method to use for the loadflow, by default "dc"
    n_processes : int, optional
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel. Paralelization is done by splitting the contingencies into
        chunks and running each chunk in a separate process
    batch_size : Optional[int]
        The size of the batches to use for the parallelization. If None, the batch size is set to the number of
        contingencies divided by the number of processes, rounded up. This is used to avoid too many handles in
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None

    Returns
    -------
    list[LoadflowResults]
        A list of the results per contingency
    """
    n_outages = len(n_minus_1_definition.contingencies)
    if batch_size is None:
        batch_size = math.ceil(n_outages / n_processes)
    work = [n_minus_1_definition[i : i + batch_size] for i in range(0, n_outages, batch_size)]

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
                method=method,
                runpp_kwargs=runpp_kwargs,
            )
        )
        if len(handles) >= n_processes:
            # Wait for the first result and continue
            finished, handles = ray.wait(handles, num_returns=1)
            result_lists.extend(ray.get(finished))
    result_lists.extend(ray.get(handles))

    # Sort the result_lists back into the original order
    del handles
    # Flatten the result_lists
    results = [result for result_list in result_lists for result in result_list]
    return results


def run_contingency_analysis_pandapower(
    net: pp.pandapowerNet,
    n_minus_1_definition: Nminus1Definition,
    job_id: str,
    timestep: int,
    min_island_size: int = 11,
    method: Literal["ac", "dc"] = "ac",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
    runpp_kwargs: Optional[dict] = None,
    polars: bool = False,
) -> Union[LoadflowResults, LoadflowResultsPolars]:
    """Compute the N-1 AC/DC power flow for the network.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to compute the N-1 power flow for, with the topology already applied.
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to use for the analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep of the results
    min_island_size: int
        The minimum island size to consider
    method : Literal["ac", "dc"], optional
        The method to use for the loadflow, by default "ac"
    n_processes : int, optional
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel. Paralelization is done by splitting the contingencies into
        chunks and running each chunk in a separate process
    batch_size : Optional[int]
        The size of the batches to use for the parallelization. If None, the batch size is set to the number of
        contingencies divided by the number of processes, rounded up.
    runpp_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to runpp/rundcpp functions, by default None
    polars: bool, default=False
        Whether to return the results as a LoadflowResultsPolars object. If False, returns a LoadflowResults
        object. Note that this only affects the type of the returned object, the computations are the same.

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the loadflow computation
    """
    pp_n1_definition = translate_nminus1_for_pandapower(n_minus_1_definition, net)
    net_graph = top.create_nxgraph(net)
    bus_lookup, _ = create_bus_lookup_simple(net)
    slack_allocation_config = SlackAllocationConfig(
        net_graph=net_graph,
        bus_lookup=bus_lookup,
        min_island_size=min_island_size,
    )

    if n_processes == 1 and batch_size is None:
        results = run_contingency_analysis_sequential(
            net=net,
            n_minus_1_definition=pp_n1_definition,
            job_id=job_id,
            timestep=timestep,
            slack_allocation_config=slack_allocation_config,
            method=method,
            runpp_kwargs=runpp_kwargs,
        )
    else:
        results = run_contingency_analysis_parallel(
            net=net,
            n_minus_1_definition=pp_n1_definition,
            job_id=job_id,
            timestep=timestep,
            slack_allocation_config=slack_allocation_config,
            method=method,
            n_processes=n_processes,
            batch_size=batch_size,
            runpp_kwargs=runpp_kwargs,
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
    if not polars:
        return lf_result
    return convert_pandas_loadflow_results_to_polars(lf_result)
