# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Get the results of the AC Contingency Analysis for the given network"""

import pypowsybl
from beartype.typing import Optional
from pypowsybl.network import Network as PowsyblNetwork
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl_limit_table.run_analysis import (
    run_contingency_analysis_powsybl_violation_table,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import get_default_security_analysis_parameters
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


def get_current_limit_reduction(result_filter: Optional[LoadflowResultFilter]) -> float:
    """Translate the branch result filter into the current limit reduction of the limit violation table.

    Powsybl reports a branch side as violated if its current is at least limit * reduction, i.e. if its loading is at
    least the reduction. That is the keep condition of BranchLoadflowResultFilter.loading_above, so the threshold is
    used as the reduction unchanged. Powsybl accepts values above 1 as well.

    Parameters
    ----------
    result_filter : Optional[LoadflowResultFilter]
        The result filter policy. Only branch_filters.loading_above is translated: the base case is fully monitored
        anyway, and voltage limits cannot be reduced.

    Returns
    -------
    float
        The current limit reduction, 1.0 if there is no filter or no branch loading threshold, which reports the
        branches above their limits.
    """
    if result_filter is None or result_filter.branch_filters.loading_above is None:
        return 1.0
    return result_filter.branch_filters.loading_above


def get_security_analysis_results(
    net: PowsyblNetwork,
    n_minus_1_definition: Nminus1Definition,
    timestep: int = 0,
    job_id: str = "",
    n_processes: int = 1,
    batch_size: Optional[int] = None,
    security_analysis_params: pypowsybl.security.Parameters | dict | None = None,
    result_filter: Optional[LoadflowResultFilter] = None,
    use_limit_violations: bool = False,
) -> LoadflowResultsPolars:
    """Get the results of the security analysis for the given network

    Parameters
    ----------
    net : pypowsybl.network.Network
        The network to run the contingency analysis on
    n_minus_1_definition: Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    timestep: int, default=0
        The timestep of the results. Used to identify the results in the database
    job_id: str, default=""
        The job id of the current job
    n_processes: int, default=1
        The number of processes to use for the contingency analysis. If 1, the analysis is run sequentially.
        If > 1, the analysis is run in parallel
        Paralelization is done by splitting the contingencies into chunks and running each chunk in a separate process
    batch_size: int, optional
        The maximum number of outage contingencies per batch. Powsybl executes batches sequentially; pandapower uses
        batches for parallelization.
        If None, all are computed at the same time
    security_analysis_params: pypowsybl.security.Parameters or dict, optional
        Security analysis parameters to use for the computation.
        dict for pandapower, pypowsybl.security.Parameters for powsybl. If None, default parameters are used.
    result_filter: LoadflowResultFilter, optional
        Policy for dropping result rows that carry no decision value, applied inside the contingency lib as the results
        are produced. If None, every row is kept.
    use_limit_violations: bool, default=False
        Whether to use the limit violation table instead of monitoring elements.

    Returns
    -------
    LoadflowResultsPolars
        The results of the Contingency analysis
    """
    if security_analysis_params is None:
        security_analysis_params = get_default_security_analysis_parameters()
        loadflow_params = None
    elif isinstance(security_analysis_params, pypowsybl.security.Parameters):
        loadflow_params = security_analysis_params.load_flow_parameters

    # set n_processes
    if security_analysis_params.provider_parameters is not None:
        security_analysis_params.provider_parameters["threadCount"] = str(n_processes)
    else:
        security_analysis_params.provider_parameters = {"threadCount": str(n_processes)}

    if use_limit_violations:
        current_limit_reduction = get_current_limit_reduction(result_filter)
        lf_results = run_contingency_analysis_powsybl_violation_table(
            net=net,
            n_minus_1_definition=n_minus_1_definition,
            job_id=job_id,
            timestep=timestep,
            security_analysis_params=security_analysis_params,
            current_limit_reduction=current_limit_reduction,
            method="ac",
            polars=True,
            batch_size=batch_size,
        )
    else:
        lf_results = run_contingency_analysis_powsybl(
            net,
            n_minus_1_definition,
            job_id,
            timestep,
            n_processes=n_processes,
            method="ac",
            polars=True,
            lf_params=loadflow_params,
            result_filter=result_filter,
            batch_size=batch_size,
        )

    return lf_results
