# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Run the powsybl security analysis and read its limit violation table into loadflow results.

Compared to the monitored element runner in ``toop_engine_contingency_analysis.pypowsybl`` this one

- never modifies the network: there is no slack baking step, lost injections are handled by OpenLoadFlow,
- monitors no elements for contingencies: their branch rows are the current violations against a reduced permanent
  limit, so the loading threshold of the result filter is applied inside powsybl,
- monitors everything for the base case, so N-0 stays complete,
- reports voltage and voltage angle violations as node and angle difference results,
- fills the connectivity table from the elements powsybl disconnected,
- keeps the native result handle and reads every table as a polars frame.
"""

from dataclasses import dataclass

import polars as pl
import pypowsybl
from beartype.typing import Literal, Optional, Union
from pypowsybl import _pypowsybl
from pypowsybl._pypowsybl import JavaHandle, PostContingencyResult
from pypowsybl.network import Network
from pypowsybl.report import ReportNode
from pypowsybl.security import ContingencyContextType
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import (
    PowsyblBranchLimitCacheProtocol,
    get_regulating_element_results,
)
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers_polars import (
    add_name_column_polars,
    get_branch_limits_polars,
    get_branch_results_polars,
    get_node_results_polars,
    update_basename_polars,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table.definition import (
    LimitViolationFilter,
    PowsyblLimitTableDefinition,
    translate_nminus1_for_limit_table,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table.extract import (
    apply_limit_violation_filter,
    get_branch_results_from_violations,
    get_connectivity_results,
    get_convergence_results,
    get_node_results_from_violations,
    get_va_diff_results_from_violations,
    normalize_limit_violations,
)
from toop_engine_contingency_analysis.result_filter import apply_result_filter
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK, SINGLE_SLACK, VIOLATION_THRESHOLD
from toop_engine_grid_helpers.powsybl.polars.get_dataframe import (
    get_ca_branch_results,
    get_ca_bus_results,
    get_ca_limit_violations,
    get_ca_three_windings_transformer_results,
)
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_result_helpers import convert_polars_loadflow_results_to_pandas
from toop_engine_interfaces.loadflow_result_helpers_polars import concatenate_loadflow_results_polars
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


@dataclass
class PowsyblLimitTableRun:
    """A finished powsybl security analysis, kept as its native handle.

    The pypowsybl result wrapper builds a pandas multi-index violation table in its constructor. This holds the bare
    handle instead and reads each table as a polars lazy frame on demand.
    """

    handle: JavaHandle
    """The native handle of the security analysis result."""

    basecase_id: Optional[str]
    """The id of the base case contingency of the run, if it has one."""

    def limit_violations(self) -> pl.LazyFrame:
        """Read the limit violation table, see get_ca_limit_violations."""
        return get_ca_limit_violations(self.handle, lazy=True)

    def branch_results(self) -> pl.LazyFrame:
        """Read the branch results of the monitored branches, only the base case is monitored."""
        return get_ca_branch_results(self.handle, lazy=True)

    def three_windings_transformer_results(self) -> pl.LazyFrame:
        """Read the results of the monitored three winding transformers, only the base case is monitored."""
        return get_ca_three_windings_transformer_results(self.handle, lazy=True)

    def bus_results(self) -> pl.LazyFrame:
        """Read the bus results of the monitored voltage levels, only the base case is monitored."""
        return get_ca_bus_results(self.handle, lazy=True)

    def pre_contingency_status(self) -> str:
        """Read the status name of the pre-contingency computation, e.g. "CONVERGED"."""
        return _pypowsybl.get_pre_contingency_result(self.handle).status.name

    def post_contingency_results(self) -> list[PostContingencyResult]:
        """Read the native post contingency results with status and disconnected elements per contingency."""
        return [result for result in _pypowsybl.get_post_contingency_results(self.handle) if result.contingency_id]


# pylint: disable-next=too-many-positional-arguments
def run_powsybl_analysis_limit_table(
    net: Network,
    definition: PowsyblLimitTableDefinition,
    lf_params: pypowsybl.loadflow.Parameters,
    method: Literal["ac", "dc"] = "ac",
    n_processes: int = 1,
    limit_reduction: float = 1.0,
    basecase_full_monitoring: bool = True,
    contingency_propagation: bool = False,
    report_node: Optional[ReportNode] = None,
) -> PowsyblLimitTableRun:
    """Run the powsybl security analysis for the limit table contingency analysis.

    Contingencies are added one to one. Current limits are reduced on the permanent limit only, so every branch
    loaded above the reduction shows up in the violation table with its permanent limit. Temporary limits stay as
    they are. Powsybl's filter for violations that did not increase against the base case is switched off, see
    REPORT_ALL_VIOLATIONS. When the definition has a base case and full monitoring is on, all branches, three winding
    transformers and voltage levels are monitored for the pre-contingency state only.

    Parameters
    ----------
    net : Network
        The network to analyse. It is not modified.
    definition : PowsyblLimitTableDefinition
        The translated N-1 definition.
    lf_params : pypowsybl.loadflow.Parameters
        The loadflow parameters of the analysis. For DC the distributed slack is switched off on a copy.
    method : Literal["ac", "dc"], optional
        Whether to run AC or DC power flows, by default "ac".
    n_processes : int, optional
        The OpenLoadFlow thread count, by default 1.
    limit_reduction : float, optional
        The factor applied to the permanent current limits for violation detection, in (0, 1]. By default the epsilon
        that reports every loaded branch.
    basecase_full_monitoring : bool, optional
        Whether to monitor the whole network for the base case, by default True.
    contingency_propagation : bool, optional
        Whether powsybl should trip the circuit breakers needed to isolate the contingency in node breaker voltage
        levels, by default False.
    report_node : Optional[ReportNode], optional
        A report node to collect the powsybl execution report in, by default None.

    Returns
    -------
    PowsyblLimitTableRun
        The native result handle with polars accessors.

    Raises
    ------
    ValueError
        If the limit reduction is outside (0, 1].
    """
    analysis = pypowsybl.security.create_analysis()

    single_outages = [contingency for contingency in definition.contingencies if len(contingency.elements) == 1]
    single_outage_elements = [contingency.elements[0] for contingency in single_outages]
    if len(set(single_outage_elements)) == len(single_outage_elements):
        contingency_id_by_element = {contingency.elements[0]: contingency.id for contingency in single_outages}
        if single_outage_elements:
            analysis.add_single_element_contingencies(
                single_outage_elements, contingency_id_provider=contingency_id_by_element.__getitem__
            )
    else:
        for contingency in single_outages:
            analysis.add_single_element_contingency(contingency.elements[0], contingency_id=contingency.id)
    for contingency in definition.contingencies:
        if len(contingency.elements) > 1:
            analysis.add_multiple_elements_contingency(contingency.elements, contingency_id=contingency.id)

    basecase_id = definition.basecase_id
    if basecase_id is not None and basecase_full_monitoring:
        analysis.add_monitored_elements(
            contingency_context_type=ContingencyContextType.NONE,
            branch_ids=definition.branch_ids or None,
            three_windings_transformer_ids=definition.trafo3w_ids or None,
            voltage_level_ids=definition.voltage_level_ids or None,
        )
    # only CURRENT limit reduction is supported by powsybl
    analysis.add_limit_reductions(limit_type="CURRENT", permanent=True, temporary=True, value=limit_reduction)

    security_params = pypowsybl.security.Parameters(
        load_flow_parameters=lf_params,
        increased_violations_parameters=VIOLATION_THRESHOLD,
        provider_parameters={
            "threadCount": str(n_processes),
            "contingencyPropagation": str(contingency_propagation).lower(),
        },
    )
    # Mirror SecurityAnalysis.run_ac / run_dc, but keep the native handle instead of the pandas-building wrapper
    c_params = security_params._to_c_parameters()
    c_params.loadflow_parameters.dc = method == "dc"
    handle = _pypowsybl.run_security_analysis(
        analysis._handle, net._handle, c_params, "", None if report_node is None else report_node._report_node
    )
    return PowsyblLimitTableRun(handle=handle, basecase_id=basecase_id)


# pylint: disable-next=too-many-positional-arguments
def _run_limit_table_batch(
    net: Network,
    definition: PowsyblLimitTableDefinition,
    job_id: str,
    timestep: int,
    lf_params: pypowsybl.loadflow.Parameters,
    method: Literal["ac", "dc"],
    n_processes: int,
    limit_reduction: float,
    basecase_full_monitoring: bool,
    contingency_propagation: bool,
) -> LoadflowResultsPolars:
    """Run one batch of contingencies and build its loadflow results."""
    run = run_powsybl_analysis_limit_table(
        net,
        definition,
        lf_params=lf_params,
        method=method,
        n_processes=n_processes,
        limit_reduction=limit_reduction,
        basecase_full_monitoring=basecase_full_monitoring,
        contingency_propagation=contingency_propagation,
    )
    basecase_id = definition.basecase_id
    full_basecase = basecase_id is not None and basecase_full_monitoring

    violations = normalize_limit_violations(run.limit_violations())
    branch_limits = pl.from_pandas(definition.branch_limits, include_index=True, nan_to_null=False).lazy()
    bus_map = pl.from_pandas(definition.bus_map, include_index=True, nan_to_null=False).lazy()
    voltage_levels = pl.from_pandas(definition.voltage_levels, include_index=True, nan_to_null=False).lazy()

    branch_results = get_branch_results_from_violations(violations, branch_limits, timestep, drop_basecase=full_basecase)
    if full_basecase:
        basecase_branch_results = get_branch_results_polars(
            run.branch_results(), run.three_windings_transformer_results(), [], [], [], timestep, branch_limits
        )
        basecase_node_results = get_node_results_polars(
            run.bus_results(),
            definition.bus_map.index.astype(str).tolist(),
            bus_map,
            voltage_levels,
            [],
            timestep,
            method,
        )
        branch_results = pl.concat([basecase_branch_results, branch_results], how="diagonal_relaxed")
    else:
        basecase_node_results = pl.LazyFrame(schema={"contingency": pl.String, "element": pl.String, "vm": pl.Float64})
    node_results = get_node_results_from_violations(
        violations, voltage_levels, bus_map, basecase_node_results, timestep, drop_basecase=full_basecase
    )
    if full_basecase:
        node_results = pl.concat([basecase_node_results, node_results], how="diagonal_relaxed")
    va_diff_results = get_va_diff_results_from_violations(violations, timestep)
    post_contingency_results = run.post_contingency_results()
    converged = get_convergence_results(
        run.pre_contingency_status(), post_contingency_results, definition.outage_ids, timestep, basecase_id
    )
    connectivity_result = get_connectivity_results(post_contingency_results)
    regulating_element_results = pl.from_pandas(
        get_regulating_element_results([], timestep=timestep), include_index=True, nan_to_null=False
    ).lazy()

    def finalize(frame: pl.LazyFrame) -> pl.LazyFrame:
        frame = update_basename_polars(frame, basecase_id)
        frame = add_name_column_polars(frame, definition.element_name_mapping, index_level="element")
        return add_name_column_polars(frame, definition.contingency_name_mapping, index_level="contingency")

    return LoadflowResultsPolars(
        job_id=job_id,
        branch_results=finalize(branch_results),
        branch_limits=get_branch_limits_polars(branch_limits, timestep),
        node_results=finalize(node_results),
        regulating_element_results=finalize(regulating_element_results),
        va_diff_results=finalize(va_diff_results),
        converged=add_name_column_polars(converged, definition.contingency_name_mapping, index_level="contingency"),
        connectivity_result=connectivity_result,
        warnings=[],
    )


def _batch_contingencies(
    definition: PowsyblLimitTableDefinition, batch_size: Optional[int]
) -> list[PowsyblLimitTableDefinition]:
    """Split the definition into sequential batches, the base case riding with the first one."""
    if batch_size is None:
        return [definition]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    basecases = [contingency for contingency in definition.contingencies if contingency.is_basecase()]
    outages = [contingency for contingency in definition.contingencies if not contingency.is_basecase()]
    outage_batches = [outages[start : start + batch_size] for start in range(0, len(outages), batch_size)]
    batches = [[*basecases, *outage_batches[0]], *outage_batches[1:]] if outage_batches else [basecases]
    return [definition.model_copy(update={"contingencies": contingencies}) for contingencies in batches]


# pylint: disable-next=too-many-positional-arguments
def run_contingency_analysis_powsybl_limit_table(  # noqa: PLR0913,PLR0917
    net: Network,
    n_minus_1_definition: Nminus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"] = "ac",
    n_processes: int = 1,
    polars: bool = False,
    lf_params: Optional[pypowsybl.loadflow.Parameters] = None,
    branch_limit_cache: Optional[PowsyblBranchLimitCacheProtocol] = None,
    result_filter: Optional[LoadflowResultFilter] = None,
    batch_size: Optional[int] = None,
    limit_reduction: Optional[float] = None,
    basecase_full_monitoring: bool = True,
    violation_filter: LimitViolationFilter = LimitViolationFilter.ALL,
    contingency_propagation: bool = False,
) -> Union[LoadflowResults, LoadflowResultsPolars]:
    """Compute the contingency analysis for the network from the powsybl limit violation table.

    The network is not modified. Contingency branch rows are the current violations against the reduced permanent
    limit, with the loading relative to the permanent limit. Node rows of contingencies are the voltage violations,
    keyed by voltage level. Angle difference rows are the violations of the voltage angle limits defined on the
    network. The base case is fully monitored unless switched off, then it is read from the violation table like any
    other contingency.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the contingency analysis for. It is not modified.
    n_minus_1_definition : Nminus1Definition
        The N-1 definition. Its contingencies are run one to one, its monitored elements only lend their names.
    job_id : str
        The job id stored in the results.
    timestep : int
        The timestep stored in the results.
    method : Literal["ac", "dc"], optional
        Whether to run AC or DC power flows, by default "ac".
    n_processes : int, optional
        The OpenLoadFlow thread count, by default 1.
    polars : bool, optional
        Whether to return polars results instead of pandas, by default False.
    lf_params : Optional[pypowsybl.loadflow.Parameters], optional
        The loadflow parameters. If None, the CGMES distributed slack parameters are used for AC and the single slack
        parameters for DC.
    branch_limit_cache : Optional[PowsyblBranchLimitCacheProtocol], optional
        Optional cache for the prepared branch limits of all branches. If stale or None it is recomputed.
    result_filter : Optional[LoadflowResultFilter], optional
        Policy for dropping result rows that carry no decision value. Its branch loading threshold doubles as the
        limit reduction unless one is given explicitly, so the threshold is applied inside powsybl.
    batch_size : Optional[int], optional
        Maximum number of outage contingencies per sequential powsybl analysis. If None, all run together.
    limit_reduction : Optional[float], optional
        The factor applied to the permanent current limits for violation detection, in (0, 1]. If None, the loading
        threshold of the result filter is used, or the epsilon that reports every loaded branch.
    basecase_full_monitoring : bool, optional
        Whether the base case monitors the whole network, by default True.
    violation_filter : LimitViolationFilter, optional
        Which contingency rows of the branch and node results to keep, by default all.
    contingency_propagation : bool, optional
        Whether powsybl should trip the circuit breakers needed to isolate the contingency in node breaker voltage
        levels, by default False.

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the contingency analysis.

    Raises
    ------
    ValueError
        If the batch size is not positive or the limit reduction is outside (0, 1].
    """
    if lf_params is None:
        lf_params = CGMES_DISTRIBUTED_SLACK if method == "ac" else SINGLE_SLACK
    if limit_reduction is None:
        limit_reduction = 1.0
    definition = translate_nminus1_for_limit_table(n_minus_1_definition, net, branch_limit_cache=branch_limit_cache)
    basecase_id = definition.basecase_id

    batch_results = [
        _run_limit_table_batch(
            net,
            batch,
            job_id,
            timestep,
            lf_params,
            method,
            n_processes,
            limit_reduction,
            basecase_full_monitoring,
            contingency_propagation,
        )
        for batch in _batch_contingencies(definition, batch_size)
    ]
    if len(batch_results) == 1:
        lf_results = batch_results[0]
    else:
        lf_results = concatenate_loadflow_results_polars(batch_results)
        lf_results.connectivity_result = pl.concat([result.connectivity_result for result in batch_results], how="diagonal")

    lf_results.branch_results = apply_limit_violation_filter(
        lf_results.branch_results, violation_filter, pl.col("loading"), basecase_id
    )
    lf_results.node_results = apply_limit_violation_filter(
        lf_results.node_results, violation_filter, pl.col("vm_loading").abs(), basecase_id
    )
    if result_filter is not None:
        lf_results = apply_result_filter(lf_results, result_filter, basecase_id)
    lf_results.warnings = [
        f"Contingency with id {contingency.id} contains elements that are not found in the network."
        for contingency in definition.missing_contingencies
    ]
    if not polars:
        return convert_polars_loadflow_results_to_pandas(lf_results)
    return lf_results
