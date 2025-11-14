"""Compute the N-1 AC/DC power flow for the network."""

import polars as pl
import pypowsybl
from beartype.typing import Literal, Union
from pypowsybl.network import Network
from pypowsybl.security import SecurityAnalysisResult
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import (
    PowsyblNMinus1Definition,
    add_name_column,
    get_convergence_result_df,
    get_regulating_element_results,
    set_target_values_to_lf_values_incl_distributed_slack,
    translate_nminus1_for_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers_polars import (
    add_name_column_polars,
    get_branch_results_polars,
    get_node_results_polars,
    get_va_diff_results_polars,
    update_basename_polars,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    DISTRIBUTED_SLACK,
    SINGLE_SLACK,
)
from toop_engine_grid_helpers.powsybl.polars.get_dataframe import (
    get_ca_branch_results,
    get_ca_bus_results,
    get_ca_three_windings_transformer_results,
)
from toop_engine_interfaces.loadflow_result_helpers import convert_polars_loadflow_results_to_pandas
from toop_engine_interfaces.loadflow_results import (
    LoadflowResults,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


def run_powsybl_analysis(
    net: Network,
    n_minus_1_definition: PowsyblNMinus1Definition,
    method: Literal["ac", "dc"] = "ac",
    n_processes: int = 1,
) -> tuple[SecurityAnalysisResult, str | None]:
    """Run the powsybl security analysis for the given network and N-1 definition.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    method : Literal["ac", "dc"], optional
        The method to use for the contingency analysis. Either "ac" or "dc", by default "ac"
    n_processes : int, optional
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.

    Returns
    -------
    res: SecurityAnalysisResult
        The security analysis result from powsybl containing the monitored elements and the results of the contingencies.
    basecase_id : str | None
        The name of the basecase contingency, if it is included in the run. Otherwise None.
    """
    # Run the actual loadflow computation
    analysis = pypowsybl.security.create_analysis()
    analysis.add_monitored_elements(
        branch_ids=n_minus_1_definition.monitored_elements["branches"],
        three_windings_transformer_ids=n_minus_1_definition.monitored_elements["trafo3w"],
        voltage_level_ids=n_minus_1_definition.monitored_elements["voltage_levels"],
    )
    basecase_id = None
    for contingency in n_minus_1_definition.contingencies:
        outages = contingency.elements
        if len(outages) > 1:
            analysis.add_multiple_elements_contingency(outages, contingency_id=contingency.id)
        elif len(outages) == 0:
            # If there are no outages, we add the basecase contingency
            basecase_id = contingency.id
        else:
            analysis.add_single_element_contingency(outages[0], contingency_id=contingency.id)
    if method == "ac" and n_minus_1_definition.distributed_slack:
        # If we have distributed slack and AC loadflows, we need to set the slack to a single generator
        # This is done by setting the target values of the generators to the loadflow values
        lf_params = DISTRIBUTED_SLACK

    else:
        # The security analysis in DC should always run with a single slack to avoid changing gen values for each N-1 case
        # This way it matches the current way our N-1 analysis in the GPU-solver is set up
        lf_params = SINGLE_SLACK
    contingency_propagation = "true" if n_minus_1_definition.contingency_propagation else "false"
    security_params = pypowsybl.security.impl.parameters.Parameters(
        load_flow_parameters=lf_params,
        provider_parameters={"threadCount": str(n_processes), "contingencyPropagation": contingency_propagation},
    )

    res = analysis.run_ac(net, security_params) if method == "ac" else analysis.run_dc(net, security_params)
    return res, basecase_id


def run_contingency_analysis_polars(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"] = "dc",
    n_processes: int = 1,
) -> LoadflowResultsPolars:
    """Compute the N-0 + N-1 power flow for the network.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for
    pow_n1_definition : PowsyblNMinus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep to use for the contingency analysis
    method : Literal["ac", "dc"], optional
        Whether to compute the AC or DC power flow, by default "dc"
    n_processes : int, optional
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel
        Paralelization is done by splitting the contingencies into chunks and running each chunk in a separate process
        This is done via the openloadflow native threadCount parameter,
        which is set in the powsybl security analysis parameters.

    Returns
    -------
    LoadflowResultsPolars
        The results of the loadflow computation. Invalid or otherwise failed results will be set to NaN.
    """
    monitored_elements = pow_n1_definition.monitored_elements
    ca_result, basecase_id = run_powsybl_analysis(net, pow_n1_definition, method, n_processes=n_processes)
    bus_results = get_ca_bus_results(ca_result, lazy=True)
    branch_results = get_ca_branch_results(ca_result, lazy=True)
    three_windings_transformer_results = get_ca_three_windings_transformer_results(ca_result, lazy=True)
    post_contingency_results = ca_result.post_contingency_results
    pre_contingency_result = ca_result.pre_contingency_result

    all_outage_ids = [contingency.id for contingency in pow_n1_definition.contingencies if not contingency.is_basecase()]
    convergence_df, failed_outages = get_convergence_result_df(
        post_contingency_results, pre_contingency_result, all_outage_ids, timestep, basecase_id
    )
    add_name_column(convergence_df, pow_n1_definition.contingency_name_mapping, index_level="contingency")
    convergence_df = pl.from_pandas(convergence_df, include_index=True, nan_to_null=False).lazy()

    branch_limit_polars = pl.from_pandas(pow_n1_definition.branch_limits, include_index=True, nan_to_null=False).lazy()

    branch_results_df = get_branch_results_polars(
        branch_results,
        three_windings_transformer_results,
        monitored_elements["branches"],
        monitored_elements["trafo3w"],
        failed_outages,
        timestep,
        branch_limit_polars,
    )
    node_results_df = get_node_results_polars(
        bus_results,
        monitored_elements["buses"],
        pl.from_pandas(pow_n1_definition.bus_map, include_index=True, nan_to_null=False).lazy(),
        pl.from_pandas(pow_n1_definition.voltage_levels, include_index=True, nan_to_null=False).lazy(),
        failed_outages,
        timestep,
        method,
    )
    regulating_elements_df = get_regulating_element_results(
        monitored_elements["buses"], timestep=timestep, basecase_name=basecase_id
    )
    regulating_elements_df = pl.from_pandas(regulating_elements_df, include_index=True, nan_to_null=False).lazy()
    va_diff_results_df = get_va_diff_results_polars(
        bus_results=bus_results,
        outages=all_outage_ids,
        va_diff_with_buses=pl.from_pandas(pow_n1_definition.blank_va_diff, include_index=True, nan_to_null=False).lazy(),
        bus_map=pl.from_pandas(pow_n1_definition.bus_map, include_index=True, nan_to_null=False).lazy(),
        timestep=timestep,
    )
    branch_results_df = update_basename_polars(branch_results_df, basecase_id)
    branch_results_df = add_name_column_polars(
        branch_results_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    branch_results_df = add_name_column_polars(
        branch_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    node_results_df = update_basename_polars(node_results_df, basecase_id)
    node_results_df = add_name_column_polars(node_results_df, pow_n1_definition.element_name_mapping, index_level="element")
    node_results_df = add_name_column_polars(
        node_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    regulating_elements_df = update_basename_polars(regulating_elements_df, basecase_id)
    regulating_elements_df = add_name_column_polars(
        regulating_elements_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    regulating_elements_df = add_name_column_polars(
        regulating_elements_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    va_diff_results_df = update_basename_polars(va_diff_results_df, basecase_id)
    va_diff_results_df = add_name_column_polars(
        va_diff_results_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    va_diff_results_df = add_name_column_polars(
        va_diff_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    lf_results = LoadflowResultsPolars(
        job_id=job_id,
        branch_results=branch_results_df,
        node_results=node_results_df,
        regulating_element_results=regulating_elements_df,
        va_diff_results=va_diff_results_df,
        converged=convergence_df,
        warnings=[],
        additional_information=[],
        lazy=True,
    )
    return lf_results


def run_contingency_analysis_powsybl(
    net: Network,
    n_minus_1_definition: Nminus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"] = "ac",
    n_processes: int = 1,
    polars: bool = False,
) -> Union[LoadflowResults, LoadflowResultsPolars]:
    """Compute the Contingency Analysis for the network.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep to use for the contingency analysis
    method : Literal["ac", "dc"], optional
        The method to use for the contingency analysis. Either "ac" or "dc", by default "dc"
    n_processes : int, optional
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel
        Paralelization is done by splitting the contingencies into chunks and running each chunk in a separate process
    polars: bool
        Whether to use polars for the dataframe operations.

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the loadflow computation.
    """
    if n_minus_1_definition.loadflow_parameters.distributed_slack:
        # We only do this once, before the first batch. So we dont have to redo it every iteration
        net = set_target_values_to_lf_values_incl_distributed_slack(net, method)
    pow_n1_definition = translate_nminus1_for_powsybl(n_minus_1_definition, net)

    lf_result = run_contingency_analysis_polars(
        net=net,
        pow_n1_definition=pow_n1_definition,
        job_id=job_id,
        timestep=timestep,
        method=method,
        n_processes=n_processes,
    )
    if not polars:
        lf_result = convert_polars_loadflow_results_to_pandas(lf_result)
    missing_element_warnings = [
        f"Element with id {element.id} not found in the network." for element in pow_n1_definition.missing_elements
    ]
    missing_contingency_warnings = [
        f"Contingency with id {contingency.id} contains elements that are not found in the network."
        for contingency in pow_n1_definition.missing_contingencies
    ]
    lf_result.warnings = [*missing_element_warnings, *missing_contingency_warnings, *lf_result.warnings]
    return lf_result
